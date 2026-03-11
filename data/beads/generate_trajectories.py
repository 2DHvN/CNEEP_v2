"""
N-Beads Model Trajectory Generator

This module generates trajectories for an N-beads (bead-spring) model in a thermal gradient.
The model consists of N beads connected by harmonic springs, where each bead experiences
a different temperature, creating a nonequilibrium steady state.

The dynamics follow overdamped Langevin equations:
    dx_i/dt = -γ^(-1) * (∂U/∂x_i) + √(2*D_i) * η_i(t)

where:
    - U is the total potential energy (sum of harmonic spring potentials)
    - γ is the friction coefficient
    - D_i = k_B * T_i / γ is the diffusion coefficient for bead i
    - η_i(t) is Gaussian white noise
"""

import numpy as np
import torch
from pathlib import Path
from typing import Optional
import argparse


class NBeadsModel:
    """
    N-Beads (Bead-Spring) Model in a temperature gradient.
    
    Attributes:
        n_beads: Number of beads in the chain.
        k: Spring constant connecting adjacent beads.
        gamma: Friction coefficient.
        temperatures: Temperature at each bead position.
        dt: Integration time step.
    """
    
    def __init__(
        self,
        n_beads: int = 3,
        k: float = 1.0,
        gamma: float = 1.0,
        T_hot: float = 10.0,
        T_cold: float = 1.0,
        dt: float = 0.01,
        boundary: str = "free",
        device: str = "cpu"
    ):
        """
        Initialize the N-Beads model.
        
        Args:
            n_beads: Number of beads in the chain.
            k: Spring constant.
            gamma: Friction coefficient.
            T_hot: Temperature at one end (bead 0).
            T_cold: Temperature at the other end (bead N-1).
            dt: Integration time step.
            boundary: Boundary condition - "free" or "fixed".
            device: Computing device ('cpu' or 'cuda').
        """
        self.n_beads = n_beads
        self.k = k
        self.gamma = gamma
        self.dt = dt
        self.boundary = boundary
        self.device = torch.device(device)
        self.coupling_matrix: torch.Tensor
        self.coupling_matrix_np: np.ndarray
        
        # Linear temperature gradient from T_hot to T_cold
        self.temperatures_np = np.linspace(T_hot, T_cold, n_beads)
        self.temperatures = torch.tensor(self.temperatures_np, dtype=torch.float32, device=self.device)
        
        # Diffusion coefficients D_i = k_B * T_i  (setting k_B = 1)
        self.diffusion_coeffs_np = self.temperatures_np
        self.diffusion_mat_np = np.diag(self.diffusion_coeffs_np)
        
        self.diffusion_coeffs = self.temperatures
        self.diffusion_mat = torch.diag(self.diffusion_coeffs)
        
        # Noise amplitude: sqrt(2 * D_i * dt)
        self.noise_amplitudes = torch.sqrt(2 * self.diffusion_coeffs * dt)
        
        # Build the coupling matrix for spring forces
        self._build_coupling_matrix()

    def _build_cov(self):
        from scipy.linalg import solve_continuous_lyapunov

        # Continuous ODE: dx = (A/gamma)x dt + sqrt(2D) dW
        # So drift M = A/gamma, process noise var Q = 2D
        # Lyapunov eq: M * Cov + Cov * M^T + Q = 0
        drift = self.coupling_matrix_np / self.gamma
        noise_var = 2 * self.diffusion_mat_np

        cov_mat_np = solve_continuous_lyapunov(-drift, noise_var)
        self.cov_mat_np = cov_mat_np
        self.cov_mat = torch.tensor(cov_mat_np, dtype=torch.float32, device=self.device)
        return self.cov_mat_np

    def _build_coupling_matrix(self):
        """Build the coupling matrix for spring forces."""
        n = self.n_beads
        self.coupling_matrix_np = np.zeros((n, n))
        
        for i in range(n):
            self.coupling_matrix_np[i, i] = -2 * self.k

            if i > 0:
                self.coupling_matrix_np[i, i - 1] = self.k
            if i < n - 1:
                self.coupling_matrix_np[i, i + 1] = self.k
                
        self.coupling_matrix = torch.tensor(self.coupling_matrix_np, dtype=torch.float32, device=self.device)

    def compute_forces(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Compute forces on all beads.
        
        Args:
            positions: Current positions of all beads, shape (n_beads,) or (batch, n_beads).
            
        Returns:
            Forces on all beads.
        """
        if positions.ndim == 1:
            return self.coupling_matrix @ positions
        else:
            return positions @ self.coupling_matrix.T
    
    def step(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Perform one Euler-Maruyama integration step.
        
        Args:
            positions: Current positions of all beads.
            
        Returns:
            New positions after one time step.
        """
        forces = self.compute_forces(positions)
        noise = torch.randn_like(positions) * self.noise_amplitudes
        
        # Euler-Maruyama: x_new = x + (F/gamma) * dt + noise
        new_positions = positions + (forces / self.gamma) * self.dt + noise
        
        return new_positions
    
    def generate_trajectory(
        self,
        n_steps: int,
        initial_positions: Optional[np.ndarray] = None,
        burn_in: int = 1000
    ) -> np.ndarray:
        """
        Generate a single trajectory.
        
        Args:
            n_steps: Number of time steps to generate.
            initial_positions: Starting positions. If None, start from zeros.
            burn_in: Number of steps to discard for equilibration.
            
        Returns:
            Trajectory array of shape (n_steps, n_beads).
        """
        if getattr(self, 'cov_mat', None) is None:
            self._build_cov()
            
        if initial_positions is None:
            from torch.distributions.multivariate_normal import MultivariateNormal
            dist = MultivariateNormal(torch.zeros(self.n_beads, device=self.device), self.cov_mat)
            positions = dist.sample()
        else:
            positions = torch.tensor(initial_positions, dtype=torch.float32, device=self.device)
        
        # Burn-in period
        for _ in range(burn_in):
            positions = self.step(positions)
        
        # Generate trajectory
        trajectory = torch.zeros((n_steps, self.n_beads), device=self.device)
        for t in range(n_steps):
            trajectory[t] = positions
            positions = self.step(positions)
        
        return trajectory.cpu().numpy()
    
    def generate_trajectories(
        self,
        n_trajectories: int,
        n_steps: int,
        burn_in: int = 1000,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Generate multiple independent trajectories simultaneously using PyTorch matrix operations.
        
        Args:
            n_trajectories: Number of trajectories to generate.
            n_steps: Number of time steps per trajectory.
            burn_in: Number of burn-in steps.
            show_progress: Whether to show progress.
            
        Returns:
            Array of shape (n_trajectories, n_steps, n_beads) on CPU as NumPy array.
        """
        if getattr(self, 'cov_mat', None) is None:
            self._build_cov()
            
        from torch.distributions.multivariate_normal import MultivariateNormal
        dist = MultivariateNormal(torch.zeros(self.n_beads, device=self.device), self.cov_mat)
        positions = dist.sample((n_trajectories,))
        
        # Burn-in period
        for _ in range(burn_in):
            positions = self.step(positions)
            
        # Generate the trajectories
        trajectories = torch.zeros((n_steps, n_trajectories, self.n_beads), device=self.device)
        for t in range(n_steps):
            trajectories[t] = positions
            positions = self.step(positions)
            
            if show_progress and (t + 1) % max(1, n_steps // 10) == 0:
                print(f"Generating step {t + 1}/{n_steps}", end="\r")
                
        if show_progress:
            print()
            
        return trajectories.transpose(0, 1).cpu().numpy()
    
    def compute_heat_per_bead(self, trajectory: np.ndarray) -> np.ndarray:
        """
        Compute the heat flow into each bead at each time step.
        """
        n_steps = trajectory.shape[0]
        heat_per_bead = np.zeros((n_steps - 1, self.n_beads))
        
        for t in range(n_steps - 1):
            pos = trajectory[t]
            pos_next = trajectory[t + 1]
            
            # Velocity of each bead (using finite difference)
            velocity = (pos_next - pos) / self.dt

            # Stratonovich convention
            mid = (pos + pos_next) / 2
            
            for i in range(self.n_beads):
                # Compute spring force on bead i
                force_i = 0.0

                if i == 0 or i == self.n_beads-1:
                    force_i -= self.k * mid[i]
                
                # Spring from left neighbor (i-1) if exists
                if i > 0:
                    force_i += self.k * (mid[i - 1] - mid[i])
                
                # Spring from right neighbor (i+1) if exists
                if i < self.n_beads - 1:
                    force_i += self.k * (mid[i + 1] - mid[i])
                
                # Heat = Force · velocity · dt (work done on bead i)
                heat_per_bead[t, i] = force_i * velocity[i] * self.dt
        
        return heat_per_bead
    
    def compute_entropy_production_per_bead(self, trajectory: np.ndarray) -> np.ndarray:
        """
        Compute the entropy production decomposed by each bead.
        """
        heat_per_bead = self.compute_heat_per_bead(trajectory)
        
        # σ_i = Q_i / T_i for each bead
        entropy_per_bead = heat_per_bead / self.temperatures_np[np.newaxis, :]
        
        return entropy_per_bead
    
    def compute_entropy_production_rate(self, trajectory: np.ndarray) -> np.ndarray:
        """
        Compute the total instantaneous entropy production rate for a trajectory.
        """
        entropy_per_bead = self.compute_entropy_production_per_bead(trajectory)
        return np.sum(entropy_per_bead, axis=1)
    
    def compute_system_entropy(self, trajectory: np.ndarray) -> np.ndarray:
        """
        Compute the system entropy s(x) = -log p(x) for the trajectory using the steady state distribution.
        """
        if getattr(self, 'cov_mat_np', None) is None:
            self._build_cov()
            
        cov_inv = np.linalg.inv(self.cov_mat_np)
        
        # -log p(x) = 0.5 * x^T Cov^-1 x + 0.5 * n * log(2pi) + 0.5 * log(det(Cov))
        n = self.n_beads
        term1 = 0.5 * np.sum((trajectory @ cov_inv) * trajectory, axis=1)
        term2 = 0.5 * n * np.log(2 * np.pi)
        term3 = 0.5 * np.linalg.slogdet(self.cov_mat_np)[1]
        
        return term1 + term2 + term3
    
    def compute_mean_entropy_production(self, trajectory: np.ndarray) -> float:
        """
        Compute the mean entropy production rate for a trajectory.
        """
        return np.mean(self.compute_entropy_production_rate(trajectory))
    
    def compute_total_entropy_production(self, trajectory: np.ndarray) -> float:
        """
        Compute the total entropy production for a trajectory.
        """
        return np.sum(self.compute_entropy_production_rate(trajectory))


def save_trajectories(
    trajectories: np.ndarray,
    output_path: str,
    metadata: Optional[dict] = None
):
    """
    Save trajectories to file.
    """
    save_dict = {'trajectories': trajectories}
    if metadata is not None:
        save_dict['metadata'] = metadata
    
    np.savez(output_path, **save_dict)
    print(f"Saved trajectories to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Generate N-Beads model trajectories")
    parser.add_argument("--n_beads", type=int, default=2, help="Number of beads")
    parser.add_argument("--n_trajectories", type=int, default=10, help="Number of trajectories")
    parser.add_argument("--n_steps", type=int, default=10000, help="Steps per trajectory")
    parser.add_argument("--burn_in", type=int, default=0, help="Burn-in steps")
    parser.add_argument("--k", type=float, default=1.0, help="Spring constant")
    parser.add_argument("--gamma", type=float, default=1.0, help="Friction coefficient")
    parser.add_argument("--T_hot", type=float, default=10.0, help="Hot reservoir temperature")
    parser.add_argument("--T_cold", type=float, default=1.0, help="Cold reservoir temperature")
    parser.add_argument("--dt", type=float, default=0.01, help="Time step")
    parser.add_argument("--output", type=str, default="trajectories.pt", help="Output file path")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use for simulation")
    
    args = parser.parse_args()
    
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
    
    print(f"Generating trajectories for {args.n_beads}-beads model...")
    print(f"  Temperature gradient: {args.T_hot} -> {args.T_cold}")
    print(f"  Spring constant: {args.k}")
    print(f"  Friction: {args.gamma}")
    print(f"  Time step: {args.dt}")
    print(f"  Trajectories: {args.n_trajectories}")
    print(f"  Steps per trajectory: {args.n_steps}")
    print(f"  Device: {args.device}")
    
    # Create model
    model = NBeadsModel(
        n_beads=args.n_beads,
        k=args.k,
        gamma=args.gamma,
        T_hot=args.T_hot,
        T_cold=args.T_cold,
        dt=args.dt,
        device=args.device
    )
    
    # Generate trajectories
    trajectories = model.generate_trajectories(
        n_trajectories=args.n_trajectories,
        n_steps=args.n_steps,
        burn_in=args.burn_in
    )
    
    # Compute sample entropy production
    sample_ep = model.compute_mean_entropy_production(trajectories[0])
    print(f"\nSample trajectory entropy production: {sample_ep:.4f}")
    
    # Metadata
    metadata = {
        'n_beads': args.n_beads,
        'n_trajectories': args.n_trajectories,
        'n_steps': args.n_steps,
        'k': args.k,
        'gamma': args.gamma,
        'T_hot': args.T_hot,
        'T_cold': args.T_cold,
        'dt': args.dt,
        'device': args.device,
        'temperatures': model.temperatures_np.tolist()
    }
    
    # Save
    output_path = Path(args.output)
    save_trajectories(trajectories, str(output_path.with_suffix('.npz')), metadata)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
