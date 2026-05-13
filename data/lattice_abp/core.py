"""
Lattice Active Brownian Particle (ABP) — Core Simulation Engine
================================================================
Based on: "Phase separation and large deviations of lattice active matter"
          Whitelam, Klymko, Mandal (2018), J. Chem. Phys. 148, 154902

Implements GPU-accelerated ensemble parallel CTMC simulation on a 2D lattice.
All transition rates are computed via branch-free tensor masking operations.

Tensor State Representation:
    O ∈ {0,1}^{B×L×L}       — occupancy
    E ∈ {0,1,2,3}^{B×L×L}   — orientation (0:up, 1:right, 2:down, 3:left)
"""

import torch
import numpy as np
from typing import Optional, Tuple, Dict


# Direction vectors: index → (dy, dx) for (row, col) convention
# 0: up (row-1), 1: right (col+1), 2: down (row+1), 3: left (col-1)
DIR_VECTORS = torch.tensor([[-1, 0], [0, 1], [1, 0], [0, -1]], dtype=torch.long)


class BoundaryCondition:
    """
    Boundary condition module.

    Provides index wrapping and boundary masks for lattice simulations.
    Supports 'periodic' and 'hard_wall' modes.
    """

    def __init__(self, mode: str = "periodic", L: int = 64):
        self.mode = mode
        self.L = L

    def wrap(self, coords: torch.Tensor) -> torch.Tensor:
        """Wrap coordinates according to boundary condition.

        Args:
            coords: Integer tensor of lattice coordinates.

        Returns:
            Wrapped coordinates (in-place safe).
        """
        if self.mode == "periodic":
            return coords % self.L
        else:  # hard_wall — clamp but mark out-of-bounds later
            return coords.clamp(0, self.L - 1)

    def valid_mask(self, coords: torch.Tensor, original: torch.Tensor) -> torch.Tensor:
        """Return boolean mask: True where the move is valid.

        For periodic BC, all moves are valid.
        For hard_wall, moves that would exit the lattice are invalid.

        Args:
            coords: Raw (unwrapped) destination coordinates.
            original: Not used for periodic; kept for API consistency.

        Returns:
            Boolean tensor, same shape as coords.
        """
        if self.mode == "periodic":
            return torch.ones_like(coords, dtype=torch.bool)
        else:
            return (coords >= 0) & (coords < self.L)


class InteractionModule:
    """
    Interaction module returning weight tensor M_interaction.

    Base implementation returns ones (no additional interaction).
    Subclass and override `compute` for custom interactions.
    """

    def compute(
        self, O: torch.Tensor, E: torch.Tensor, device: torch.device
    ) -> Dict[int, torch.Tensor]:
        """Compute interaction weight tensors for each direction.

        Args:
            O: Occupancy tensor (B, L, L).
            E: Orientation tensor (B, L, L).
            device: Torch device.

        Returns:
            Dict mapping direction index → weight tensor (B, L, L).
            Default: all ones (no interaction modification).
        """
        B, L, _ = O.shape
        ones = torch.ones(B, L, L, device=device)
        return {d: ones for d in range(4)}


class LatticeABP:
    """
    Lattice Active Brownian Particle simulator.

    Parameters
    ----------
    L : int
        Lattice side length.
    v_plus : float
        Forward hop rate (along orientation).
    v_zero : float
        Lateral hop rate (perpendicular to orientation).
    v_minus : float
        Backward hop rate (against orientation).
    D_rot : float
        Rotational diffusion rate (rate of ±90° turns).
    density : float
        Particle number density ρ = N / L².
    bc_mode : str
        Boundary condition: 'periodic' or 'hard_wall'.
    interaction : InteractionModule or None
        Custom interaction module.
    device : str
        Torch device string.
    seed : int or None
        Random seed.
    """

    def __init__(
        self,
        L: int = 64,
        v_plus: float = 1.0,
        v_zero: float = 0.1,
        v_minus: float = 0.01,
        D_rot: float = 0.1,
        density: float = 0.5,
        bc_mode: str = "periodic",
        interaction: Optional[InteractionModule] = None,
        device: str = "auto",
        seed: Optional[int] = None,
    ):
        self.L = L
        self.v_plus = v_plus
        self.v_zero = v_zero
        self.v_minus = v_minus
        self.D_rot = D_rot
        self.density = density
        self.N = int(density * L * L)  # number of particles

        # Device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Boundary & interaction modules
        self.bc = BoundaryCondition(mode=bc_mode, L=L)
        self.interaction = interaction if interaction is not None else InteractionModule()

        # Direction vectors on device
        self.dir_vecs = DIR_VECTORS.to(self.device)  # (4, 2)

        # Orientation unit vectors as (dy, dx) — same order as dir_vecs
        # e_alpha for orientation α: self.dir_vecs[α]
        # Precompute dot-product lookup: dot(e_alpha, r_{alpha->j})
        # for each orientation α and each movement direction d:
        #   dot = e_alpha · dir_d
        # Shape: (4 orientations, 4 directions)
        self.dot_table = (self.dir_vecs.unsqueeze(0) * self.dir_vecs.unsqueeze(1)).sum(-1)
        # dot_table[α, d] = e_α · d_vec

        # Precompute rate lookup from dot product value
        # dot=+1 → v_plus, dot=0 → v_zero, dot=-1 → v_minus
        self.rate_from_dot = torch.zeros(3, device=self.device)
        self.rate_from_dot[0] = v_minus   # dot = -1 → index 0
        self.rate_from_dot[1] = v_zero    # dot =  0 → index 1
        self.rate_from_dot[2] = v_plus    # dot = +1 → index 2

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        # Péclet number for reference
        self.Pe = (v_plus - v_minus) / (2 * D_rot) if D_rot > 0 else float("inf")

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def init_state(self, B: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize B independent ensembles with random particle placement.

        Args:
            B: Ensemble (batch) size.

        Returns:
            O: Occupancy tensor (B, L, L), dtype long.
            E: Orientation tensor (B, L, L), dtype long.
        """
        L, N = self.L, self.N
        O = torch.zeros(B, L, L, dtype=torch.long, device=self.device)
        E = torch.zeros(B, L, L, dtype=torch.long, device=self.device)

        for b in range(B):
            # Random placement of N particles on L² sites
            perm = torch.randperm(L * L, device=self.device)[:N]
            rows = perm // L
            cols = perm % L
            O[b, rows, cols] = 1
            # Random orientations
            E[b, rows, cols] = torch.randint(0, 4, (N,), device=self.device)

        return O, E

    # ------------------------------------------------------------------
    # Transition rate computation (branch-free, tensor-based)
    # ------------------------------------------------------------------

    def compute_transition_rates(
        self, O: torch.Tensor, E: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[int, Tuple[torch.Tensor, torch.Tensor]]]:
        """Compute transition rates for all particles in all ensembles.

        Uses branch-free tensor masking: no if-else.

        Returns:
            W_all: (B, L, L, 6) — rates for 4 translations + 2 rotations.
            neighbor_info: dict of precomputed neighbor coordinates per direction.
        """
        B, L, _ = O.shape

        # --- Translation rates for 4 directions ---
        W_trans = torch.zeros(B, L, L, 4, device=self.device)
        neighbor_info = {}

        # Create coordinate grids
        rows = torch.arange(L, device=self.device)
        cols = torch.arange(L, device=self.device)
        grid_r, grid_c = torch.meshgrid(rows, cols, indexing="ij")  # (L, L)

        for d in range(4):
            dy, dx = self.dir_vecs[d]
            # Destination coordinates
            dest_r = grid_r + dy  # (L, L)
            dest_c = grid_c + dx

            # Boundary validity mask
            valid_r = self.bc.valid_mask(dest_r, grid_r)  # (L, L)
            valid_c = self.bc.valid_mask(dest_c, grid_c)
            valid = valid_r & valid_c  # (L, L) — broadcast to (B, L, L)

            # Wrap coordinates
            dest_r_w = self.bc.wrap(dest_r)  # (L, L)
            dest_c_w = self.bc.wrap(dest_c)
            neighbor_info[d] = (dest_r_w, dest_c_w)

            # Occupancy at destination: O_j  (B, L, L)
            O_j = O[:, dest_r_w, dest_c_w]  # advanced indexing

            # Dot product of orientation with move direction
            # E has values 0-3; for each orientation value, look up dot with direction d
            # dot_table[E[b,i,j], d] gives the dot product
            dots = self.dot_table[E, d]  # (B, L, L), values in {-1, 0, 1}

            # Map dot products to rates: dot+1 as index into rate_from_dot
            rate_idx = (dots + 1).long()  # {0, 1, 2}
            base_rate = self.rate_from_dot[rate_idx]  # (B, L, L)

            # Excluded volume: multiply by (1 - O_j)
            # Boundary validity: multiply by valid mask
            W_trans[:, :, :, d] = base_rate * (1 - O_j).float() * valid.unsqueeze(0).float()

        # Apply interaction weights
        M_int = self.interaction.compute(O, E, self.device)
        for d in range(4):
            W_trans[:, :, :, d] *= M_int[d]

        # Only occupied sites can have transitions
        occ_mask = O.float()  # (B, L, L)
        W_trans *= occ_mask.unsqueeze(-1)

        # --- Rotation rates: CW (+1) and CCW (-1) ---
        W_rot_cw = self.D_rot * occ_mask   # (B, L, L)
        W_rot_ccw = self.D_rot * occ_mask   # (B, L, L)

        # Combine: (B, L, L, 6) = 4 translations + 2 rotations
        W_all = torch.cat(
            [W_trans, W_rot_cw.unsqueeze(-1), W_rot_ccw.unsqueeze(-1)], dim=-1
        )

        return W_all, neighbor_info

    # ------------------------------------------------------------------
    # Gillespie step (ensemble-parallel)
    # ------------------------------------------------------------------

    def gillespie_step(
        self, O: torch.Tensor, E: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform one Gillespie step for each ensemble member.

        For each ensemble b:
          1. Compute all rates W.
          2. Sample waiting time Δt ~ Exp(R_b).
          3. Select one event proportional to rates.
          4. Execute the event.

        Returns:
            O, E: Updated state tensors.
            dt_vec: (B,) vector of waiting times.
        """
        B, L, _ = O.shape
        W_all, neighbor_info = self.compute_transition_rates(O, E)

        # Flatten spatial + event dims: (B, L*L*6)
        W_flat = W_all.reshape(B, -1)
        R_b = W_flat.sum(dim=1)  # (B,)

        # Waiting time: Δt ~ Exp(R_b)
        u1 = torch.rand(B, device=self.device).clamp(min=1e-30)
        dt_vec = -torch.log(u1) / R_b.clamp(min=1e-30)

        # Select event via cumulative distribution
        probs = W_flat / R_b.unsqueeze(1).clamp(min=1e-30)
        event_idx = torch.multinomial(probs, 1).squeeze(1)  # (B,)

        # Decode event index → (row, col, event_type)
        n_events = 6
        site_idx = event_idx // n_events  # (B,)
        event_type = event_idx % n_events  # (B,)
        row_idx = site_idx // L
        col_idx = site_idx % L

        batch_idx = torch.arange(B, device=self.device)

        # Process translations (event_type 0-3)
        for d in range(4):
            mask = (event_type == d)  # (B,) bool
            if not mask.any():
                continue
            bi = batch_idx[mask]
            ri = row_idx[mask]
            ci = col_idx[mask]

            dest_r_w, dest_c_w = neighbor_info[d]
            dr = dest_r_w[ri, ci]
            dc = dest_c_w[ri, ci]

            # Move particle: clear source, set destination
            O[bi, ri, ci] = 0
            E_val = E[bi, ri, ci].clone()
            E[bi, ri, ci] = 0
            O[bi, dr, dc] = 1
            E[bi, dr, dc] = E_val

        # Process CW rotation (event_type 4)
        mask_cw = (event_type == 4)
        if mask_cw.any():
            bi = batch_idx[mask_cw]
            ri = row_idx[mask_cw]
            ci = col_idx[mask_cw]
            E[bi, ri, ci] = (E[bi, ri, ci] + 1) % 4

        # Process CCW rotation (event_type 5)
        mask_ccw = (event_type == 5)
        if mask_ccw.any():
            bi = batch_idx[mask_ccw]
            ri = row_idx[mask_ccw]
            ci = col_idx[mask_ccw]
            E[bi, ri, ci] = (E[bi, ri, ci] - 1) % 4

        return O, E, dt_vec

    # ------------------------------------------------------------------
    # Tau-leaping step (ensemble-parallel, approximate)
    # ------------------------------------------------------------------

    def tau_leap_step(
        self, O: torch.Tensor, E: torch.Tensor, tau: float = 0.01
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform one tau-leaping step for all ensembles.

        Approximate method: for each site/event, sample number of firings
        from Poisson(W * tau). Conflicts resolved by priority ordering.

        Args:
            O, E: Current state tensors.
            tau: Leap time step.

        Returns:
            O, E: Updated state tensors.
        """
        B, L, _ = O.shape
        W_all, neighbor_info = self.compute_transition_rates(O, E)

        # --- Rotations first (no spatial conflict) ---
        # CW rotations
        W_cw = W_all[:, :, :, 4]  # (B, L, L)
        n_cw = torch.poisson(W_cw * tau)
        rot_mask_cw = (n_cw > 0) & (O == 1)
        # Net rotation: odd number of CW turns
        odd_cw = (n_cw.long() % 2 == 1) & rot_mask_cw
        E = torch.where(odd_cw, (E + 1) % 4, E)

        # CCW rotations
        W_ccw = W_all[:, :, :, 5]
        n_ccw = torch.poisson(W_ccw * tau)
        odd_ccw = (n_ccw.long() % 2 == 1) & (O == 1)
        E = torch.where(odd_ccw, (E - 1) % 4, E)

        # --- Translations: process each direction sequentially ---
        for d in range(4):
            # Recompute rates after each direction to respect exclusion
            W_d = W_all[:, :, :, d]
            n_events = torch.poisson(W_d * tau)
            move_mask = (n_events > 0) & (O == 1)  # (B, L, L)

            if not move_mask.any():
                continue

            dest_r_w, dest_c_w = neighbor_info[d]

            # Check destination is still empty
            O_dest = O[:, dest_r_w, dest_c_w]
            can_move = move_mask & (O_dest == 0)

            if not can_move.any():
                continue

            # Execute moves
            # Gather orientation values
            E_vals = E.clone()

            # For sites that move: clear source
            O = O.clone()
            E = E.clone()

            # Get source coordinates
            src_r = torch.arange(L, device=self.device).unsqueeze(1).expand(L, L)
            src_c = torch.arange(L, device=self.device).unsqueeze(0).expand(L, L)

            for b in range(B):
                mask_b = can_move[b]
                if not mask_b.any():
                    continue
                sr = src_r[mask_b]
                sc = src_c[mask_b]
                dr = dest_r_w[mask_b]
                dc = dest_c_w[mask_b]
                e_vals = E_vals[b, sr, sc]

                O[b, sr, sc] = 0
                E[b, sr, sc] = 0
                O[b, dr, dc] = 1
                E[b, dr, dc] = e_vals

        return O, E

    # ------------------------------------------------------------------
    # Simulation runner
    # ------------------------------------------------------------------

    def simulate(
        self,
        B: int = 1,
        n_steps: int = 10000,
        burn_in: int = 1000,
        method: str = "gillespie",
        tau: float = 0.01,
        save_interval: int = 100,
        show_progress: bool = True,
        initial_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Dict:
        """Run ensemble simulation.

        Args:
            B: Ensemble size.
            n_steps: Number of MC steps after burn-in.
            burn_in: Number of burn-in steps.
            method: 'gillespie' or 'tau_leap'.
            tau: Time step for tau-leaping.
            save_interval: Save state every this many steps.
            show_progress: Show progress bar.
            initial_state: Optional (O, E) initial state.

        Returns:
            Dict with keys:
                'O_traj': (n_saved, B, L, L) occupancy snapshots.
                'E_traj': (n_saved, B, L, L) orientation snapshots.
                'times':  (n_saved, B) simulation times.
        """
        try:
            from tqdm import trange
        except ImportError:
            trange = range

        if initial_state is not None:
            O, E = initial_state
        else:
            O, E = self.init_state(B)

        sim_time = torch.zeros(B, device=self.device)

        # Burn-in
        desc = "Burn-in"
        iterator = trange(burn_in, desc=desc, leave=False) if show_progress else range(burn_in)
        for _ in iterator:
            if method == "gillespie":
                O, E, dt_vec = self.gillespie_step(O, E)
                sim_time += dt_vec
            else:
                O, E = self.tau_leap_step(O, E, tau)
                sim_time += tau

        # Production run
        n_saved = n_steps // save_interval
        O_traj = torch.zeros(n_saved, B, self.L, self.L, dtype=torch.long, device=self.device)
        E_traj = torch.zeros(n_saved, B, self.L, self.L, dtype=torch.long, device=self.device)
        times = torch.zeros(n_saved, B, device=self.device)

        save_counter = 0
        desc = "Simulating"
        iterator = trange(n_steps, desc=desc, leave=False) if show_progress else range(n_steps)
        for step in iterator:
            if method == "gillespie":
                O, E, dt_vec = self.gillespie_step(O, E)
                sim_time += dt_vec
            else:
                O, E = self.tau_leap_step(O, E, tau)
                sim_time += tau

            if (step + 1) % save_interval == 0 and save_counter < n_saved:
                O_traj[save_counter] = O
                E_traj[save_counter] = E
                times[save_counter] = sim_time
                save_counter += 1

        return {
            "O_traj": O_traj.cpu(),
            "E_traj": E_traj.cpu(),
            "times": times.cpu(),
            "O_final": O.cpu(),
            "E_final": E.cpu(),
        }

    # ------------------------------------------------------------------
    # Jammed state detection (for visualization)
    # ------------------------------------------------------------------

    def compute_jammed_mask(self, O: torch.Tensor, E: torch.Tensor) -> torch.Tensor:
        """Compute jammed mask: True where a particle's front neighbor is occupied.

        A particle at site (i,j) with orientation α is jammed if the site
        in direction e_α is occupied.

        Args:
            O: (B, L, L) occupancy.
            E: (B, L, L) orientation.

        Returns:
            jammed: (B, L, L) boolean tensor.
        """
        B, L, _ = O.shape
        jammed = torch.zeros(B, L, L, dtype=torch.bool, device=O.device)

        rows = torch.arange(L, device=O.device)
        cols = torch.arange(L, device=O.device)
        grid_r, grid_c = torch.meshgrid(rows, cols, indexing="ij")

        for d in range(4):
            dy, dx = self.dir_vecs[d].cpu()  # use cpu for indexing
            dest_r = self.bc.wrap((grid_r + dy).to(O.device))
            dest_c = self.bc.wrap((grid_c + dx).to(O.device))

            # Particles with orientation d
            orient_mask = (E == d) & (O == 1)
            # Front neighbor occupied
            front_occ = O[:, dest_r, dest_c]
            jammed |= orient_mask & (front_occ == 1)

        return jammed
