"""
Active Model B — Trajectory Generator (Finite Difference, Midpoint Rule)

Simulates Active Model B (modified Cahn-Hilliard) on a 2D periodic lattice
using finite differences with periodic boundary conditions and the midpoint
(Stratonovich) time-stepping rule.

Dynamics:
    ∂ₜφ = ∇²μ + √(2D) ∇·η

    μ = μ_eq + μ_active
      = (−aφ + bφ³ − κ∇²φ) + λ|∇φ|²

Local entropy production rate density:
    σ(r,t) = μ_active · ∂ₜφ

Features (aligned with generate_trajectories_1d.py):
    - NumPy / PyTorch backend support
    - Batch (parallel) ensemble simulation
    - Memory-efficient on-the-fly mean EPR density computation
    - 3-point EPR density mode (epr_mode="mid")
"""

import numpy as np
from typing import Optional
import argparse

try:
    from tqdm import trange, tqdm
except ImportError:
    trange = range
    def tqdm(iterable, *args, **kwargs):
        return iterable

try:
    import torch
except ImportError:
    torch = None


class ActiveModelB:
    """
    Active Model B on a 2D periodic lattice.

    All spatial operators use finite differences (no FFT).
    EPR density: σ = μ_active · ∂ₜφ.
    """

    def __init__(
        self,
        Lx: int = 64,
        Ly: int = 64,
        dx: float = 1.0,
        a: float = 0.25,
        b: float = 0.25,
        kappa: float = 0.5,
        lam: float = 2.0,
        D: float = 0.001,
        dt: float = 0.01,
        smooth: bool = False,
        backend: str = "numpy",
        use_gpu: bool = False,
        bc: str = "periodic",
        epr_mode: str = "mid",
    ):
        self.Lx = Lx
        self.Ly = Ly
        self.dx = dx
        self.a = a
        self.b = b
        self.kappa = kappa
        self.lam = lam
        self.D = D
        self.dt = dt
        self.smooth = smooth
        self.backend = backend
        self.use_gpu = use_gpu
        self.bc = bc
        self.epr_mode = epr_mode

        # Boundary mode: 'periodic' (default) or 'fixed' (wall)
        self.bc_mode = bc
        self.phi_eq = np.sqrt(self.a / self.b) if self.b > 0 else 0.0

        if backend == 'torch':
            if torch is None:
                raise ImportError("PyTorch is not available.")
            self.device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
        else:
            self.device = None

    # ------------------------------------------------------------------
    # Backend abstraction helpers
    # ------------------------------------------------------------------

    def _zeros(self, shape):
        if self.backend == 'torch':
            return torch.zeros(shape, device=self.device, dtype=torch.float64)
        else:
            return np.zeros(shape, dtype=np.float64)

    def _randn(self, shape):
        if self.backend == 'torch':
            return torch.randn(shape, device=self.device, dtype=torch.float64)
        else:
            return np.random.randn(*shape).astype(np.float64)

    def _roll(self, f, shift, axis):
        if self.backend == 'torch':
            return torch.roll(f, shifts=shift, dims=axis)
        else:
            return np.roll(f, shift, axis=axis)

    def _sum(self, f, axis=None):
        if self.backend == 'torch':
            if axis is None:
                return f.sum()
            return f.sum(dim=axis)
        else:
            return np.sum(f, axis=axis)

    def _mean(self, f, axis=None):
        if self.backend == 'torch':
            if axis is None:
                return f.mean()
            return f.mean(dim=axis)
        else:
            return np.mean(f, axis=axis)

    def _to_numpy(self, f):
        if self.backend == 'torch' and isinstance(f, torch.Tensor):
            return f.detach().cpu().numpy()
        return f

    # ------------------------------------------------------------------
    # Finite-difference spatial operators (periodic BC via roll)
    # Supports shapes (Lx, Ly) or (batch, Lx, Ly)
    # When batched, axis 0 = batch; spatial axes = -2 (x), -1 (y)
    # ------------------------------------------------------------------

    def _laplacian(self, f):
        """5-point discrete Laplacian with periodic BC."""
        return (
            self._roll(f, 2, -2) + self._roll(f, -2, -2)
            + self._roll(f, 2, -1) + self._roll(f, -2, -1)
            - 4.0 * f
        ) / (4 * self.dx ** 2)

    def _biharmonic(self, f):
        dx4 = self.dx ** 4

        term_center = 20.0 * f

        term_near = -8.0 * (
            self._roll(f, 1, -2) + self._roll(f, -1, -2) +
            self._roll(f, 1, -1) + self._roll(f, -1, -1)
        )

        # Diagonal terms: roll in both x and y
        term_diag = 2.0 * (
            self._roll(self._roll(f, 1, -2), 1, -1)
            + self._roll(self._roll(f, 1, -2), -1, -1)
            + self._roll(self._roll(f, -1, -2), 1, -1)
            + self._roll(self._roll(f, -1, -2), -1, -1)
        )

        term_far = 1.0 * (
            self._roll(f, 2, -2) + self._roll(f, -2, -2) +
            self._roll(f, 2, -1) + self._roll(f, -2, -1)
        )

        return (term_center + term_near + term_diag + term_far) / dx4

    def _grad_x(self, f):
        """Central difference ∂ₓf."""
        return (self._roll(f, -1, -2) - self._roll(f, 1, -2)) / (2.0 * self.dx)

    def _grad_y(self, f):
        """Central difference ∂ᵧf."""
        return (self._roll(f, -1, -1) - self._roll(f, 1, -1)) / (2.0 * self.dx)

    def _grad_sq(self, f):
        """|∇f|²."""
        return self._grad_x(f) ** 2 + self._grad_y(f) ** 2

    # ------------------------------------------------------------------
    # Chemical potentials
    # ------------------------------------------------------------------

    def _mu_eq(self, phi):
        """Equilibrium chemical potential: −aφ + bφ³ − κ∇²φ."""
        return -self.a * phi + self.b * phi ** 3 - self.kappa * self._laplacian(phi)

    def mu_active(self, phi):
        """Active chemical potential: λ|∇φ|²."""
        return self.lam * self._grad_sq(phi)

    def _mu_total(self, phi):
        return self._mu_eq(phi) + self.mu_active(phi)

    # ------------------------------------------------------------------
    # Conservative noise
    # ------------------------------------------------------------------

    def _conservative_noise(self, batch_size=None):
        """
        Generate conservative noise increment ∇·ξ for one time step.
        ξ lives on bond midpoints; divergence gives lattice-site noise.
        """
        amp = np.sqrt(2.0 * self.D * self.dt)
        if batch_size is not None:
            shape = (batch_size, self.Lx, self.Ly)
        else:
            shape = (self.Lx, self.Ly)
        xi_x = amp * self._randn(shape)
        xi_y = amp * self._randn(shape)
        # Central divergence
        return (
            (self._roll(xi_x, -1, -2) - self._roll(xi_x, 1, -2)) / (2 * self.dx)
            + (self._roll(xi_y, -1, -1) - self._roll(xi_y, 1, -1)) / (2 * self.dx)
        )

    # ------------------------------------------------------------------
    # Midpoint time stepping (predictor-corrector)
    # ------------------------------------------------------------------

    def _apply_bc(self, phi):
        """Enforce boundary conditions after each step."""
        if self.bc_mode == "fixed":
            if phi.ndim == 3:
                phi[:, 0:4, :] = -self.phi_eq
                phi[:, -4:, :] = +self.phi_eq
            else:
                phi[0:4, :] = -self.phi_eq
                phi[-4:, :] = +self.phi_eq
        return phi

    def step(self, phi):
        batch_size = phi.shape[0] if phi.ndim == 3 else None
        noise = self._conservative_noise(batch_size)
        if self.smooth:
            rhs = (
                -self.a * self._laplacian(phi)
                + self.b * self._laplacian(phi ** 3)
                - self.kappa * self._biharmonic(phi)
                + self.lam * self._laplacian(self._grad_sq(phi))
            )
        else:
            rhs = self._laplacian(self._mu_total(phi))
        phi_new = phi + self.dt * rhs + noise
        phi_new = self._apply_bc(phi_new)
        return phi_new

    # ------------------------------------------------------------------
    # Local entropy production rate density
    # ------------------------------------------------------------------

    def compute_local_epr_density(self, phi_t, phi_tp1):
        """
        Local entropy production rate density (2-point).
        σ(r,t) = - μ_active(mid) · ∂ₜφ / D
        Uses Stratonovich midpoint for μ_active.
        """
        dphi_dt = (phi_tp1 - phi_t) / self.dt
        phi_mid = 0.5 * (phi_t + phi_tp1)
        mu_act = self.mu_active(phi_mid)
        return - mu_act * dphi_dt / self.D

    def compute_local_epr_density_three(self, phi_p, phi_t, phi_n):
        """
        Local entropy production rate density (3-point).
        Uses central difference in time: ∂ₜφ ≈ (φ(t+1) - φ(t-1)) / (2dt)
        and μ_active evaluated at φ(t).
        """
        dphi_dt = (phi_n - phi_p) / (2 * self.dt)
        mu_act = self.mu_active(phi_t)
        return - mu_act * dphi_dt / self.D

    def compute_total_epr(self, phi_t, phi_tp1):
        """Spatially-integrated EPR at one time step."""
        sigma = self.compute_local_epr_density(phi_t, phi_tp1)
        return self._sum(sigma, axis=(-2, -1)) * self.dx ** 2

    def compute_total_epr_three(self, phi_p, phi_t, phi_n):
        """Spatially-integrated EPR (3-point) at one time step."""
        sigma = self.compute_local_epr_density_three(phi_p, phi_t, phi_n)
        return self._sum(sigma, axis=(-2, -1)) * self.dx ** 2

    # ------------------------------------------------------------------
    # Trajectory generation
    # ------------------------------------------------------------------

    def _init_field(self, batch_size=1, mode="circle"):
        """Dispatch to circle or wall initial condition. Returns (batch, Lx, Ly)."""
        if mode == "wall":
            return self._init_field_wall(batch_size)
        else:
            return self._init_field_circle(batch_size)

    def _init_field_circle(self, batch_size=1):
        """
        Circular cluster at center.
        φ_inside  = +√(a/b),  φ_outside = -√(a/b).
        Returns shape (batch_size, Lx, Ly).
        """
        phi_eq = np.sqrt(self.a / self.b) if self.b > 0 else 0.0
        xi = np.sqrt(self.kappa / self.a) if self.a > 0 else 0.0

        cx, cy = self.Lx / 2.0, self.Ly / 2.0
        R = min(self.Lx, self.Ly) / 4.0

        x = np.arange(self.Lx) * self.dx
        y = np.arange(self.Ly) * self.dx
        X, Y = np.meshgrid(x, y, indexing="ij")
        dist = np.sqrt((X - cx * self.dx) ** 2 + (Y - cy * self.dx) ** 2)

        phi_base = -phi_eq * np.tanh((dist - R * self.dx) / (np.sqrt(2.0) * xi * self.dx))
        # Stack into batch
        phi = np.tile(phi_base[None, :, :], (batch_size, 1, 1)).astype(np.float64)

        if self.backend == 'torch':
            phi = torch.tensor(phi, device=self.device, dtype=torch.float64)

        return phi

    def _init_field_wall(self, batch_size=1):
        """
        Planar wall at x = Lx/2.
        Returns shape (batch_size, Lx, Ly).
        """
        phi_eq = np.sqrt(self.a / self.b) if self.b > 0 else 0.0
        xi = np.sqrt(self.kappa / self.a) if self.a > 0 else 0.0

        x = np.arange(self.Lx) * self.dx
        x_center = self.Lx * self.dx / 2.0

        profile = phi_eq * np.tanh((x - x_center) / (np.sqrt(2.0) * xi * self.dx))
        phi_base = np.tile(profile[:, None], (1, self.Ly)).astype(np.float64)
        phi = np.tile(phi_base[None, :, :], (batch_size, 1, 1))

        if self.backend == 'torch':
            phi = torch.tensor(phi, device=self.device, dtype=torch.float64)

        return phi

    def generate_trajectory(
        self,
        n_steps: int,
        initial_phi=None,
        burn_in: int = 1000,
        init_mode: str = "circle",
        show_progress: bool = True,
    ):
        """
        Generate a single trajectory.

        Returns
        -------
        trajectory : np.ndarray of shape (n_steps, Lx, Ly)
        """
        if initial_phi is None:
            phi = self._init_field(1, init_mode)
        else:
            if self.backend == 'torch':
                if not isinstance(initial_phi, torch.Tensor):
                    phi = torch.tensor(initial_phi, device=self.device, dtype=torch.float64)
                else:
                    phi = initial_phi.clone()
                if phi.ndim == 2:
                    phi = phi.unsqueeze(0)
            else:
                phi = initial_phi.copy().astype(np.float64)
                if phi.ndim == 2:
                    phi = phi[None, :, :]

        # Set boundary mode based on init
        if init_mode == "wall":
            self.bc_mode = "fixed"
        else:
            self.bc_mode = "periodic"

        burn_in_iter = trange(burn_in, desc="Burn-in", leave=False) if show_progress else range(burn_in)
        for _ in burn_in_iter:
            phi = self.step(phi)

        trajectory = self._zeros((n_steps, self.Lx, self.Ly))
        step_iter = trange(n_steps, desc="Generating", leave=False) if show_progress else range(n_steps)
        for t in step_iter:
            trajectory[t] = phi[0]
            phi = self.step(phi)

        return self._to_numpy(trajectory)

    def generate_trajectories(
        self,
        n_trajectories: int,
        n_steps: int,
        burn_in: int = 1000,
        show_progress: bool = True,
        init_mode: str = "circle",
    ):
        """
        Generate ensemble of trajectories in parallel (batch mode).

        All n_trajectories are simulated simultaneously using the batch
        dimension, with independent noise for each.

        Returns
        -------
        trajectories : np.ndarray of shape (n_trajectories, n_steps, Lx, Ly)
        """
        phi = self._init_field(n_trajectories, init_mode)

        if init_mode == "wall":
            self.bc_mode = "fixed"
        else:
            self.bc_mode = "periodic"

        burn_in_iter = trange(burn_in, desc="Burn-in Ensemble", leave=False) if show_progress else range(burn_in)
        for _ in burn_in_iter:
            phi = self.step(phi)

        trajectories = self._zeros((n_trajectories, n_steps, self.Lx, self.Ly))

        step_iter = trange(n_steps, desc="Generating Trajectories", leave=False) if show_progress else range(n_steps)
        for t in step_iter:
            trajectories[:, t, :, :] = phi
            phi = self.step(phi)

        return self._to_numpy(trajectories)

    # ------------------------------------------------------------------
    # EPR statistics
    # ------------------------------------------------------------------

    def compute_mean_epr(self, trajectory):
        """Time-averaged total EPR for trajectory(ies).
        
        Handles shapes (T, Lx, Ly) or (M, T, Lx, Ly).
        """
        if trajectory.ndim == 3:
            trajectory = trajectory[None, :, :, :]

        T = trajectory.shape[1]

        if self.backend == 'torch':
            traj_t = torch.tensor(trajectory, device=self.device, dtype=torch.float64)
        else:
            traj_t = trajectory

        total = 0.0
        if self.epr_mode == "mid":
            for t in tqdm(range(T - 2), desc="Computing EPR", leave=False):
                total += self.compute_total_epr_three(traj_t[:, t], traj_t[:, t + 1], traj_t[:, t + 2])
        else:
            for t in tqdm(range(T - 1), desc="Computing EPR", leave=False):
                total += self.compute_total_epr(traj_t[:, t], traj_t[:, t + 1])

        res = total / T
        return self._to_numpy(res)

    def compute_mean_epr_on_the_fly(
        self,
        n_trajectories: int,
        n_steps: int,
        burn_in: int = 1000,
        show_progress: bool = True,
        init_mode: str = "circle",
    ):
        """
        Memory-efficient on-the-fly Mean EPR Density for 2D ensembles.

        Avoids storing full M × T × Lx × Ly trajectory tensors.
        Returns: ensemble_mean_epr_density array of shape (Lx, Ly).
        """
        phi_t = self._init_field(n_trajectories, init_mode)

        if init_mode == "wall":
            self.bc_mode = "fixed"
        else:
            self.bc_mode = "periodic"

        burn_in_iter = trange(burn_in, desc="Burn-in Ensemble", leave=False) if show_progress else range(burn_in)
        for _ in burn_in_iter:
            phi_t = self.step(phi_t)

        ensemble_epr_density = np.zeros((self.Lx, self.Ly), dtype=np.float64)

        step_iter = trange(n_steps, desc="Simulating & Computing On-the-Fly", leave=False) if show_progress else range(n_steps)

        if self.epr_mode == "mid":
            phi_p = phi_t
            phi_t = self.step(phi_t)

            for _ in step_iter:
                phi_n = self.step(phi_t)

                sigma = self.compute_local_epr_density_three(phi_p, phi_t, phi_n)
                sigma = self._to_numpy(sigma)

                # Accumulate spatial mean across M parallel seeds (axis=0)
                ensemble_epr_density += np.mean(sigma, axis=0)

                phi_p = phi_t
                phi_t = phi_n

            ensemble_epr_density /= n_steps
        else:
            for _ in step_iter:
                phi_tp1 = self.step(phi_t)

                sigma = self.compute_local_epr_density(phi_t, phi_tp1)
                sigma = self._to_numpy(sigma)

                ensemble_epr_density += np.mean(sigma, axis=0)
                phi_t = phi_tp1

            ensemble_epr_density /= n_steps

        return ensemble_epr_density

    def compute_epr_density_trajectory(self, trajectory):
        """Local EPR density for every consecutive pair → (T-1, Lx, Ly)."""
        T = trajectory.shape[0]
        epr_maps = np.zeros((T - 1, self.Lx, self.Ly))
        for t in range(T - 1):
            epr_maps[t] = self.compute_local_epr_density(
                trajectory[t], trajectory[t + 1]
            )
        return epr_maps


# ======================================================================
# Save / CLI
# ======================================================================

def save_trajectories(trajectories, output_path, metadata=None):
    save_dict = {"trajectories": trajectories}
    if metadata is not None:
        save_dict["metadata"] = metadata
    np.savez(output_path, **save_dict)
    print(f"Saved trajectories to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Active Model B trajectory generator (FD + midpoint)"
    )

    parser.add_argument("--Lx", type=int, default=64)
    parser.add_argument("--Ly", type=int, default=64)
    parser.add_argument("--dx", type=float, default=1.0)

    parser.add_argument("--a", type=float, default=0.25)
    parser.add_argument("--b", type=float, default=0.25)
    parser.add_argument("--kappa", type=float, default=4.0)
    parser.add_argument("--lam", type=float, default=1.0)
    parser.add_argument("--D", type=float, default=0.01)
    parser.add_argument("--dt", type=float, default=0.01)

    parser.add_argument("--n_trajectories", type=int, default=1)
    parser.add_argument("--n_steps", type=int, default=48000)
    parser.add_argument("--burn_in", type=int, default=24000)
    parser.add_argument("--init_mode", type=str, default="circle",
                        choices=["circle", "wall"],
                        help="Initial condition: 'circle' or 'wall'")
    parser.add_argument("--smooth", action="store_true",
                        help="Use 13-point biharmonic stencil for κ∇⁴φ")

    parser.add_argument("--backend", type=str, default="numpy", choices=["numpy", "torch"])
    parser.add_argument("--use_gpu", action="store_true")
    parser.add_argument("--bc", type=str, default="periodic", choices=["periodic", "fixed"])
    parser.add_argument("--epr_mode", type=str, default="mid", choices=["mid", "standard"])

    parser.add_argument("--output", type=str, default="amb_trajectories.npz")
    parser.add_argument("--seed", type=int, default=None)

    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)
        if torch is not None:
            torch.manual_seed(args.seed)

    print(f"[INFO] Active Model B  ({args.Lx}×{args.Ly})  FD + midpoint")
    print(f"[INFO] Backend: {args.backend}, GPU: {args.use_gpu}")
    print(f"  a={args.a}, b={args.b}, κ={args.kappa}, λ={args.lam}")
    print(f"  D={args.D}, dt={args.dt}")
    print(f"  n_steps={args.n_steps}, burn_in={args.burn_in}")

    model = ActiveModelB(
        Lx=args.Lx, Ly=args.Ly, dx=args.dx,
        a=args.a, b=args.b, kappa=args.kappa,
        lam=args.lam, D=args.D, dt=args.dt,
        smooth=args.smooth,
        backend=args.backend, use_gpu=args.use_gpu,
        bc=args.bc, epr_mode=args.epr_mode,
    )

    trajectories = model.generate_trajectories(
        n_trajectories=args.n_trajectories,
        n_steps=args.n_steps,
        burn_in=args.burn_in,
    )

    sample_epr = model.compute_mean_epr(trajectories)
    if isinstance(sample_epr, np.ndarray) and sample_epr.size > 1:
        mean_val = np.mean(sample_epr)
        print(f"\n[EPR] Mean total EPR over {args.n_trajectories} trajectories: {mean_val:.6f}")
    else:
        print(f"\n[EPR] Mean total EPR: {float(np.mean(sample_epr)):.6f}")

    metadata = vars(args)
    save_trajectories(trajectories, args.output, metadata)
    print("\nDone!")


if __name__ == "__main__":
    main()
