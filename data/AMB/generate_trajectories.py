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

        # Boundary mode: 'periodic' (default) or 'fixed' (wall)
        self.bc_mode = "periodic"
        self.phi_eq = np.sqrt(self.a / self.b) if self.b > 0 else 0.0

    # ------------------------------------------------------------------
    # Finite-difference spatial operators (periodic BC via np.roll)
    # ------------------------------------------------------------------

    def _laplacian(self, f: np.ndarray) -> np.ndarray:
        """5-point discrete Laplacian with periodic BC."""
        return (
            np.roll(f, 2, 0) + np.roll(f, -2, 0)
            + np.roll(f, 2, 1) + np.roll(f, -2, 1)
            - 4.0 * f
        ) / (4 * self.dx ** 2)

    def _biharmonic(self, f: np.ndarray) -> np.ndarray:
        dx4 = self.dx ** 4
    
        term_center = 20.0 * f
    
        term_near = -8.0 * (
            np.roll(f, 1, 0) + np.roll(f, -1, 0) + 
            np.roll(f, 1, 1) + np.roll(f, -1, 1)
        )
    
        term_diag = 2.0 * (
            np.roll(np.roll(f, 1, 0), 1, 1) + np.roll(np.roll(f, 1, 0), -1, 1) +
            np.roll(np.roll(f, -1, 0), 1, 1) + np.roll(np.roll(f, -1, 0), -1, 1)
        )
    
        term_far = 1.0 * (
            np.roll(f, 2, 0) + np.roll(f, -2, 0) + 
            np.roll(f, 2, 1) + np.roll(f, -2, 1)
        )
    
        return (term_center + term_near + term_diag + term_far) / dx4

    def _grad_x(self, f: np.ndarray) -> np.ndarray:
        """Central difference ∂ₓf."""
        return (np.roll(f, -1, 0) - np.roll(f, 1, 0)) / (2.0 * self.dx)

    def _grad_y(self, f: np.ndarray) -> np.ndarray:
        """Central difference ∂ᵧf."""
        return (np.roll(f, -1, 1) - np.roll(f, 1, 1)) / (2.0 * self.dx)

    def _grad_sq(self, f: np.ndarray) -> np.ndarray:
        """|∇f|²."""
        return self._grad_x(f) ** 2 + self._grad_y(f) ** 2

    # ------------------------------------------------------------------
    # Chemical potentials
    # ------------------------------------------------------------------

    def _mu_eq(self, phi: np.ndarray) -> np.ndarray:
        """Equilibrium chemical potential: −aφ + bφ³ − κ∇²φ."""
        return -self.a * phi + self.b * phi ** 3 - self.kappa * self._laplacian(phi)

    def mu_active(self, phi: np.ndarray) -> np.ndarray:
        """Active chemical potential: λ|∇φ|²."""
        return self.lam * self._grad_sq(phi)

    def _mu_total(self, phi: np.ndarray) -> np.ndarray:
        return self._mu_eq(phi) + self.mu_active(phi)

    # ------------------------------------------------------------------
    # Conservative noise
    # ------------------------------------------------------------------

    def _conservative_noise(self) -> np.ndarray:
        """
        Generate conservative noise increment ∇·ξ for one time step.
        ξ lives on bond midpoints; divergence gives lattice-site noise.
        """
        amp = np.sqrt(2.0 * self.D * self.dt)
        xi_x = amp * np.random.randn(self.Lx, self.Ly)
        xi_y = amp * np.random.randn(self.Lx, self.Ly)
        # Forward divergence
        return (
            (np.roll(xi_x, -1, 0) - np.roll(xi_x, 1, 0)) / (2 * self.dx)
            + (np.roll(xi_y, -1, 1) - np.roll(xi_y, 1, 1)) / (2 * self.dx)
        )

    # ------------------------------------------------------------------
    # Midpoint time stepping (predictor-corrector)
    # ------------------------------------------------------------------

    def _apply_bc(self, phi: np.ndarray) -> np.ndarray:
        """Enforce boundary conditions after each step."""
        if self.bc_mode == "fixed":
            # Dirichlet: fix 2 rows on each side (stencil width)
            phi[0:4, :]  = -self.phi_eq
            phi[-4:, :] = +self.phi_eq
        return phi

    def step(self, phi: np.ndarray) -> np.ndarray:
        noise = self._conservative_noise()
        if self.smooth:
            # Use 13-point biharmonic stencil for κ∇⁴φ
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

    def compute_local_epr_density(
        self,
        phi_t: np.ndarray,
        phi_tp1: np.ndarray,
    ) -> np.ndarray:
        """
        Local entropy production rate density.

        σ(r,t) = μ_active(r,t) · ∂ₜφ(r,t)

        Uses Stratonovich midpoint for μ_active.
        """
        dphi_dt = (phi_tp1 - phi_t) / self.dt

        # Ito (total)
        # return - self._mu_total(phi_t) * dphi_dt / self.D

        # Stratonovich (total)
        phi_mid = 0.5 * (phi_t + phi_tp1)
        mu_tot = self._mu_total(phi_mid)
        return - mu_tot * dphi_dt / self.D

        # Stratonovich (act)
        # phi_mid = 0.5 * (phi_t + phi_tp1)
        # mu_act = self.mu_active(phi_mid)
        # return - mu_act * dphi_dt / self.D

    def compute_total_epr(
        self,
        phi_t: np.ndarray,
        phi_tp1: np.ndarray,
    ) -> float:
        """Spatially-integrated EPR at one time step."""
        sigma = self.compute_local_epr_density(phi_t, phi_tp1)
        return np.sum(sigma) * self.dx ** 2

    # ------------------------------------------------------------------
    # Trajectory generation
    # ------------------------------------------------------------------

    def _init_field(self, mode: str = "circle") -> np.ndarray:
        """Dispatch to circle or wall initial condition."""
        if mode == "wall":
            return self._init_field_wall()
        else:
            return self._init_field_circle()

    def _init_field_circle(self) -> np.ndarray:
        """
        Circular cluster at center.
        φ_inside  = +√(a/b),  φ_outside = -√(a/b).
        """
        phi_eq = np.sqrt(self.a / self.b) if self.b > 0 else 0.0
        xi = np.sqrt(self.kappa / self.a) if self.a > 0 else 0.0

        cx, cy = self.Lx / 2.0, self.Ly / 2.0
        R = min(self.Lx, self.Ly) / 4.0

        x = np.arange(self.Lx) * self.dx
        y = np.arange(self.Ly) * self.dx
        X, Y = np.meshgrid(x, y, indexing="ij")
        dist = np.sqrt((X - cx * self.dx) ** 2 + (Y - cy * self.dx) ** 2)

        phi = -phi_eq * np.tanh((dist - R * self.dx) / (np.sqrt(2.0) * xi * self.dx))
        return phi

    def _init_field_wall(self) -> np.ndarray:
        """
        Planar wall at x = Lx/2.
        Left (x < Lx/2) = -√(a/b),  Right (x > Lx/2) = +√(a/b).
        Smooth tanh interface of width ~ √(κ/a).
        """
        phi_eq = np.sqrt(self.a / self.b) if self.b > 0 else 0.0
        xi = np.sqrt(self.kappa / self.a) if self.a > 0 else 0.0

        x = np.arange(self.Lx) * self.dx
        x_center = self.Lx * self.dx / 2.0

        # tanh profile along x, uniform in y
        profile = phi_eq * np.tanh((x - x_center) / (np.sqrt(2.0) * xi * self.dx))
        phi = np.tile(profile[:, None], (1, self.Ly))
        return phi

    def generate_trajectory(
        self,
        n_steps: int,
        initial_phi: Optional[np.ndarray] = None,
        burn_in: int = 1000,
        init_mode: str = "circle",
    ) -> np.ndarray:
        """
        Generate a single trajectory.

        Parameters
        ----------
        init_mode : 'circle' or 'wall'

        Returns
        -------
        trajectory : (n_steps, Lx, Ly)
        """
        phi = self._init_field(init_mode) if initial_phi is None else initial_phi.copy()

        # Set boundary mode based on init
        if init_mode == "wall":
            self.bc_mode = "fixed"
        else:
            self.bc_mode = "periodic"

        burn_in_iter = trange(burn_in, desc="Burn-in", leave=False)
        for _ in burn_in_iter:
            phi = self.step(phi)

        trajectory = np.zeros((n_steps, self.Lx, self.Ly))
        step_iter = trange(n_steps, desc="Generating", leave=False)
        for t in step_iter:
            trajectory[t] = phi
            phi = self.step(phi)

        return trajectory

    def generate_trajectories(
        self,
        n_trajectories: int,
        n_steps: int,
        burn_in: int = 1000,
        show_progress: bool = True,
    ) -> np.ndarray:
        trajectories = np.zeros(
            (n_trajectories, n_steps, self.Lx, self.Ly)
        )
        traj_iter = trange(n_trajectories, desc="Ensemble") if show_progress else range(n_trajectories)
        for i in traj_iter:
            trajectories[i] = self.generate_trajectory(n_steps, burn_in=burn_in)
        return trajectories

    # ------------------------------------------------------------------
    # EPR statistics
    # ------------------------------------------------------------------

    def compute_mean_epr(self, trajectory: np.ndarray) -> float:
        """Time-averaged total EPR for a trajectory of shape (T, Lx, Ly)."""
        T = trajectory.shape[0]
        total = 0.0
        for t in tqdm(range(T - 1), desc="Computing EPR", leave=False):
            total += self.compute_total_epr(trajectory[t], trajectory[t + 1])
        return total / T

    def compute_epr_density_trajectory(
        self, trajectory: np.ndarray
    ) -> np.ndarray:
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

    parser.add_argument("--output", type=str, default="amb_trajectories.npz")
    parser.add_argument("--seed", type=int, default=None)

    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)

    print(f"[INFO] Active Model B  ({args.Lx}×{args.Ly})  FD + midpoint")
    print(f"  a={args.a}, b={args.b}, κ={args.kappa}, λ={args.lam}")
    print(f"  D={args.D}, dt={args.dt}")
    print(f"  n_steps={args.n_steps}, burn_in={args.burn_in}")

    model = ActiveModelB(
        Lx=args.Lx, Ly=args.Ly, dx=args.dx,
        a=args.a, b=args.b, kappa=args.kappa,
        lam=args.lam, D=args.D, dt=args.dt,
        smooth=args.smooth,
    )

    trajectories = model.generate_trajectories(
        n_trajectories=args.n_trajectories,
        n_steps=args.n_steps,
        burn_in=args.burn_in,
    )

    sample_epr = model.compute_mean_epr(trajectories[0])
    print(f"\n[EPR] Mean total EPR (sample traj): {sample_epr:.6f}")

    metadata = vars(args)
    save_trajectories(trajectories, args.output, metadata)
    print("\nDone!")


if __name__ == "__main__":
    main()
