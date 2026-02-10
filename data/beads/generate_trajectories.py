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
        gamma: float = 3.0,
        T_hot: float = 10.0,
        T_cold: float = 1.0,
        dt: float = 0.01,
        boundary: str = "free"
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
        """
        self.n_beads = n_beads
        self.k = k
        self.gamma = gamma
        self.dt = dt
        self.boundary = boundary
        self.coupling_matrix: np.ndarray
        
        # Linear temperature gradient from T_hot to T_cold
        self.temperatures = np.linspace(T_hot, T_cold, n_beads)
        
        # Diffusion coefficients D_i = k_B * T_i / gamma (setting k_B = 1)
        self.diffusion_coeffs = self.temperatures / gamma
        
        # Noise amplitude: sqrt(2 * D_i * dt)
        self.noise_amplitudes = np.sqrt(2 * self.diffusion_coeffs * dt)
        
        # Build the coupling matrix for spring forces
        self._build_coupling_matrix()
    
    def _build_coupling_matrix(self):
        """Build the coupling matrix for spring forces."""
        n = self.n_beads
        self.coupling_matrix = np.zeros((n, n))
        
        for i in range(n):
            if i == 0 or i == n-1:
                self.coupling_matrix[i, i] -= self.k

            if i > 0:
                self.coupling_matrix[i, i - 1] = self.k
                self.coupling_matrix[i, i] -= self.k
            if i < n - 1:
                self.coupling_matrix[i, i + 1] = self.k
                self.coupling_matrix[i, i] -= self.k
    
    def compute_forces(self, positions: np.ndarray) -> np.ndarray:
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
    
    def step(self, positions: np.ndarray) -> np.ndarray:
        """
        Perform one Euler-Maruyama integration step.
        
        Args:
            positions: Current positions of all beads.
            
        Returns:
            New positions after one time step.
        """
        forces = self.compute_forces(positions)
        noise = np.random.randn(*positions.shape) * self.noise_amplitudes
        
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
        if initial_positions is None:
            positions = np.zeros(self.n_beads)
        else:
            positions = initial_positions.copy()
        
        # Burn-in period
        for _ in range(burn_in):
            positions = self.step(positions)
        
        # Generate trajectory
        trajectory = np.zeros((n_steps, self.n_beads))
        for t in range(n_steps):
            trajectory[t] = positions
            positions = self.step(positions)
        
        return trajectory
    
    def generate_trajectories(
        self,
        n_trajectories: int,
        n_steps: int,
        burn_in: int = 1000,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Generate multiple independent trajectories.
        
        Args:
            n_trajectories: Number of trajectories to generate.
            n_steps: Number of time steps per trajectory.
            burn_in: Number of burn-in steps.
            show_progress: Whether to show progress.
            
        Returns:
            Array of shape (n_trajectories, n_steps, n_beads).
        """
        trajectories = np.zeros((n_trajectories, n_steps, self.n_beads))
        
        for i in range(n_trajectories):
            if show_progress and (i + 1) % max(1, n_trajectories // 10) == 0:
                print(f"Generating trajectory {i + 1}/{n_trajectories}")
            trajectories[i] = self.generate_trajectory(n_steps, burn_in=burn_in)
        
        return trajectories
    
    def compute_heat_per_bead(self, trajectory: np.ndarray) -> np.ndarray:
        """
        Compute the heat flow into each bead at each time step.
        
        For bead i, the heat flow Q_i is the work done by spring forces on that bead:
            Q_i = F_i · v_i = [k(x_{i-1} - x_i) + k(x_{i+1} - x_i)] · v_i
        
        For boundary beads:
            - Bead 0: Q_0 = k(x_1 - x_0) · v_0
            - Bead N-1: Q_{N-1} = k(x_{N-2} - x_{N-1}) · v_{N-1}
        
        Args:
            trajectory: Trajectory of shape (n_steps, n_beads).
            
        Returns:
            Heat flow per bead, shape (n_steps - 1, n_beads).
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
        
        Following the formulation from "Attaining entropy production and dissipation 
        maps from Brownian movies via neural networks", the entropy production 
        contribution from bead i is:
            σ_i = Q_i / T_i
        
        where Q_i is the heat flow into bead i and T_i is its temperature.
        
        Args:
            trajectory: Trajectory of shape (n_steps, n_beads).
            
        Returns:
            Entropy production per bead, shape (n_steps - 1, n_beads).
        """
        heat_per_bead = self.compute_heat_per_bead(trajectory)
        
        # σ_i = Q_i / T_i for each bead
        entropy_per_bead = heat_per_bead / self.temperatures[np.newaxis, :]
        
        return entropy_per_bead
    
    def compute_entropy_production_rate(self, trajectory: np.ndarray) -> np.ndarray:
        """
        Compute the total instantaneous entropy production rate for a trajectory.
        
        The total entropy production rate is the sum over all beads:
            σ = Σ_i Q_i / T_i
        
        Args:
            trajectory: Trajectory of shape (n_steps, n_beads).
            
        Returns:
            Total entropy production rate at each time step, shape (n_steps - 1,).
        """
        entropy_per_bead = self.compute_entropy_production_per_bead(trajectory)
        return np.sum(entropy_per_bead, axis=1)
    
    def compute_mean_entropy_production(self, trajectory: np.ndarray) -> float:
        """
        Compute the mean entropy production rate for a trajectory.
        
        Args:
            trajectory: Trajectory of shape (n_steps, n_beads).
            
        Returns:
            Mean entropy production rate.
        """
        return np.mean(self.compute_entropy_production_rate(trajectory))
    
    def compute_total_entropy_production(self, trajectory: np.ndarray) -> float:
        """
        Compute the total entropy production for a trajectory.
        
        Args:
            trajectory: Trajectory of shape (n_steps, n_beads).
            
        Returns:
            Total entropy production.
        """
        return np.sum(self.compute_entropy_production_rate(trajectory))


def save_trajectories(
    trajectories: np.ndarray,
    output_path: str,
    metadata: Optional[dict] = None
):
    """
    Save trajectories to file.
    
    Args:
        trajectories: Array of shape (n_trajectories, n_steps, n_beads).
        output_path: Path to save the file.
        metadata: Optional dictionary of metadata to include.
    """
    save_dict = {'trajectories': trajectories}
    if metadata is not None:
        save_dict['metadata'] = metadata
    
    np.savez(output_path, **save_dict)
    print(f"Saved trajectories to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Generate N-Beads model trajectories")
    parser.add_argument("--n_beads", type=int, default=3, help="Number of beads")
    parser.add_argument("--n_trajectories", type=int, default=100, help="Number of trajectories")
    parser.add_argument("--n_steps", type=int, default=10000, help="Steps per trajectory")
    parser.add_argument("--burn_in", type=int, default=5000, help="Burn-in steps")
    parser.add_argument("--k", type=float, default=1.0, help="Spring constant")
    parser.add_argument("--gamma", type=float, default=1.0, help="Friction coefficient")
    parser.add_argument("--T_hot", type=float, default=2.0, help="Hot reservoir temperature")
    parser.add_argument("--T_cold", type=float, default=1.0, help="Cold reservoir temperature")
    parser.add_argument("--dt", type=float, default=0.01, help="Time step")
    parser.add_argument("--output", type=str, default="trajectories.pt", help="Output file path")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    
    args = parser.parse_args()
    
    if args.seed is not None:
        np.random.seed(args.seed)
    
    print(f"Generating trajectories for {args.n_beads}-beads model...")
    print(f"  Temperature gradient: {args.T_hot} -> {args.T_cold}")
    print(f"  Spring constant: {args.k}")
    print(f"  Friction: {args.gamma}")
    print(f"  Time step: {args.dt}")
    print(f"  Trajectories: {args.n_trajectories}")
    print(f"  Steps per trajectory: {args.n_steps}")
    
    # Create model
    model = NBeadsModel(
        n_beads=args.n_beads,
        k=args.k,
        gamma=args.gamma,
        T_hot=args.T_hot,
        T_cold=args.T_cold,
        dt=args.dt
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
        'temperatures': model.temperatures.tolist()
    }
    
    # Save
    output_path = Path(args.output)
    save_trajectories(trajectories, str(output_path.with_suffix('.npz')), metadata)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
