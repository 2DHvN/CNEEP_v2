"""
Active Model B — 1D Trajectory Generator (Finite Difference, Midpoint Rule)

Simulates Active Model B (modified Cahn-Hilliard) on a 1D periodic lattice
with finite differences and the midpoint (Stratonovich) time-stepping rule.

Dynamics:
    ∂ₜφ = ∇²μ + √(2D) ∇·η
    μ = (−aφ + bφ³ − κ∇²φ) + λ|∇φ|²
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

class ActiveModelB1D:
    def __init__(
        self,
        Lx: int = 500,
        dx: float = 1.0,
        a: float = 0.25,
        b: float = 0.25,
        kappa: float = 4.0,
        lam: float = 1.0,
        D: float = 0.01,
        dt: float = 0.01,
        smooth: bool = False,
        backend: str = "numpy",
        use_gpu: bool = False,
        bc: str = "periodic",
        epr_mode: str = "mid",
    ):
        self.Lx = Lx
        self.dx = dx
        self.L = self.Lx * self.dx
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

        if backend == 'torch':
            if torch is None:
                raise ImportError("PyTorch is not available.")
            self.device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
        else:
            self.device = None

        # Pre-setup matrices eagerly to avoid per-step overhead
        self._setup_matrices()

    def _roll(self, f, shift):
        if self.backend == 'torch':
            return torch.roll(f, shifts=shift, dims=-1)
        else:
            return np.roll(f, shift, axis=-1)

    def _setup_matrices(self):
        N = self.Lx
        dx = self.dx
        
        # 2nd order 1st derivative
        col_1_2 = np.zeros(N)
        col_1_2[1] = 1.0; col_1_2[-1] = -1.0
        col_1_2 /= (2.0 * dx)
        
        # 2nd order 2nd derivative
        col_2_2 = np.zeros(N)
        col_2_2[0] = -2.0; col_2_2[1] = 1.0; col_2_2[-1] = 1.0
        col_2_2 /= (dx**2)
        
        # 8th order 1st derivative
        col_1_8 = np.zeros(N)
        col_1_8[4] = -1.0/280.0; col_1_8[3] = 4.0/105.0; col_1_8[2] = -1.0/5.0; col_1_8[1] = 4.0/5.0
        col_1_8[-4] = 1.0/280.0; col_1_8[-3] = -4.0/105.0; col_1_8[-2] = 1.0/5.0; col_1_8[-1] = -4.0/5.0
        col_1_8 /= dx
        
        if self.backend == 'torch':
            idx_2 = np.concatenate([np.arange(N - 1, N), np.arange(0, 2)])
            idx_8 = np.concatenate([np.arange(N - 4, N), np.arange(0, 5)])
            
            self.kernel_D1_2 = torch.tensor([[[float(x) for x in col_1_2[idx_2]]]], dtype=torch.float64, device=self.device)
            self.kernel_D2_2 = torch.tensor([[[float(x) for x in col_2_2[idx_2]]]], dtype=torch.float64, device=self.device)
            self.kernel_D1_8 = torch.tensor([[[float(x) for x in col_1_8[idx_8]]]], dtype=torch.float64, device=self.device)
        else:
            self.D1_2 = np.zeros((N, N))
            self.D2_2 = np.zeros((N, N))
            self.D1_8 = np.zeros((N, N))
            for i in range(N):
                self.D1_2[i] = np.roll(col_1_2, i)
                self.D2_2[i] = np.roll(col_2_2, i)
                self.D1_8[i] = np.roll(col_1_8, i)
            self.D1_2 = self.D1_2.astype(np.float64)
            self.D2_2 = self.D2_2.astype(np.float64)
            self.D1_8 = self.D1_8.astype(np.float64)

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

    def _pad_mode(self):
        return 'circular' if self.bc == 'periodic' else 'reflect'

    def _apply_bc_numpy(self, f, pad_width):
        if self.bc == 'periodic':
            return np.pad(f, ((0, 0), (pad_width, pad_width)), mode='wrap')
        else:
            return np.pad(f, ((0, 0), (pad_width, pad_width)), mode='edge')

    def _laplacian_2(self, f):
        if not self.smooth:
            return self._grad_x_2(self._grad_x_2(f))

        if self.backend == 'torch':
            f_in = f.unsqueeze(1)
            f_pad = torch.nn.functional.pad(f_in, (1, 1), mode=self._pad_mode())
            res = torch.nn.functional.conv1d(f_pad, self.kernel_D2_2)
            return res.squeeze(1)
        else:
            f_pad = self._apply_bc_numpy(f, 1)
            # Convolution over padded array
            res = np.zeros_like(f)
            col = np.array([1., -2., 1.]) / (self.dx**2)
            for i in range(self.Lx):
                res[:, i] = np.sum(f_pad[:, i:i+3] * col, axis=1)
            return res

    def _grad_x_2(self, f):
        if self.backend == 'torch':
            f_in = f.unsqueeze(1)
            f_pad = torch.nn.functional.pad(f_in, (1, 1), mode=self._pad_mode())
            res = torch.nn.functional.conv1d(f_pad, self.kernel_D1_2)
            return res.squeeze(1)
        else:
            f_pad = self._apply_bc_numpy(f, 1)
            res = np.zeros_like(f)
            col = np.array([-1., 0., 1.]) / (2.0 * self.dx)
            for i in range(self.Lx):
                res[:, i] = np.sum(f_pad[:, i:i+3] * col, axis=1)
            return res

    def _grad_x_8(self, f):
        if not self.smooth:
            return self._grad_x_2(f)
            
        if self.backend == 'torch':
            f_in = f.unsqueeze(1)
            f_pad = torch.nn.functional.pad(f_in, (4, 4), mode='circular')
            res = torch.nn.functional.conv1d(f_pad, self.kernel_D1_8)
            return res.squeeze(1)
        else:
            return np.matmul(f, self.D1_8.T)

    def _mu_eq(self, phi):
        return -self.a * phi + self.b * phi**3 - self.kappa * self._laplacian_2(phi)

    def mu_active(self, phi):
        return self.lam * (self._grad_x_2(phi) ** 2)

    def _mu_total(self, phi):
        return self._mu_eq(phi) + self.mu_active(phi)

    def _conservative_noise(self, batch_size):
        amp = np.sqrt(2.0 * self.D * self.dt / self.dx)
        xi = amp * self._randn((batch_size, self.Lx))
        res = self._grad_x_2(xi)
        if self.backend == 'torch':
            res -= res.mean(dim=-1, keepdim=True)
        else:
            res -= res.mean(axis=-1, keepdims=True)
        return res

    def step(self, phi):
        # 1. Compute components (laplacian inside _mu_eq is 2nd order)
        mu_eq = self._mu_eq(phi)
        mu_act = self.mu_active(phi)
        
        # 2. Compute the current Flux (J)
        # In C repo: Jxeq is -diff_x_8(mueq), Jxact is -diff_x(muact) (2nd order)
        J_eq = -self._grad_x_8(mu_eq)
        J_act = -self._grad_x_8(mu_act)
        
        # amplitude for conservative noise inside J
        amp = np.sqrt(2.0 * self.D / (self.dx * self.dt))
        Lambda = amp * self._randn(phi.shape)

        if self.bc == "Neumann": # conservative and Neumann condition
            J_eq[:, 0] = J_eq[:, -1] = 0
            J_act[:, 0] = J_act[: -1] = 0
            Lambda[:, 0] = Lambda[:, -1] = 0
        
        # 3. Compute divergences (div J)
        # In C repo: divJ = diff_x_8(Jxeq) + diff_x(Jxact) + diff_x_8(Lambdax)
        div_J_eq = self._grad_x_8(J_eq)
        div_J_act = self._grad_x_8(J_act)
        div_Lambda = self._grad_x_8(Lambda)
        
        div_J = div_J_eq + div_J_act + div_Lambda
        
        # 4. Update Equation: dphi/dt = - div J
        return phi - self.dt * div_J

    def compute_local_epr_density(self, phi_t, phi_tp1):
        dphi_dt = (phi_tp1 - phi_t) / self.dt
        phi_mid = 0.5 * (phi_t + phi_tp1)
        mu_act = self.mu_active(phi_mid)
        # The equation for EPR is - mu_act * dphi_dt / D
        return - mu_act * dphi_dt / self.D

    def compute_local_epr_density_three(self, phi_p, phi_t, phi_n):
        dphi_dt = (phi_n - phi_p) / (2 * self.dt)
        mu_act = self.mu_active(phi_t)
        return - mu_act * dphi_dt / self.D

    def compute_total_epr(self, phi_t, phi_tp1):
        sigma = self.compute_local_epr_density(phi_t, phi_tp1)
        if self.backend == 'torch':
            return sigma.sum(dim=-1) * self.dx
        else:
            return sigma.sum(axis=-1) * self.dx

    def compute_total_epr_three(self, phi_p, phi_t, phi_n):
        sigma = self.compute_local_epr_density_three(phi_p, phi_t, phi_n)
        if self.backend == 'torch':
            return sigma.sum(dim=-1) * self.dx
        else:
            return sigma.sum(axis=-1) * self.dx

    def _init_field(self, batch_size):
        phi_eq = np.sqrt(self.a / self.b) if (self.a > 0 and self.b > 0) else 0.0
        xi = np.sqrt(self.kappa / self.a) if self.a > 0 else 0.0
        
        L = self.L
        if self.backend == 'torch':
            x = torch.arange(self.Lx, device=self.device, dtype=torch.float64) * self.dx
            if self.bc == 'periodic':
                phi_base = phi_eq * (torch.tanh((x - L/4) / (np.sqrt(2.0) * xi)) - torch.tanh((x - 3*L/4) / (np.sqrt(2.0) * xi)) - 1.0)
            else:
                # Single domain wall in the center for Neumann
                phi_base = phi_eq * torch.tanh((x - L/2) / (np.sqrt(2.0) * xi))
            
            phi = phi_base.unsqueeze(0).repeat(batch_size, 1)
            
            # For Neumann, we do NOT mean-subtract because the single wall breaks zero mean
            if self.bc == 'periodic':
                phi -= phi.mean(dim=-1, keepdim=True)
        else:
            x = np.arange(self.Lx, dtype=np.float64) * self.dx
            if self.bc == 'periodic':
                phi_base = phi_eq * (np.tanh((x - L/4) / (np.sqrt(2.0) * xi)) - np.tanh((x - 3*L/4) / (np.sqrt(2.0) * xi)) - 1.0)
            else:
                phi_base = phi_eq * np.tanh((x - L/2) / (np.sqrt(2.0) * xi))
                
            phi = np.tile(phi_base, (batch_size, 1))
            if self.bc == 'periodic':
                phi -= phi.mean(axis=-1, keepdims=True)
            
        return phi

    def generate_trajectory(self, n_steps, initial_phi=None, burn_in=1000):
        if initial_phi is None:
            phi = self._init_field(1)
        else:
            if self.backend == 'torch' and not isinstance(initial_phi, torch.Tensor):
                phi = torch.tensor(initial_phi, device=self.device, dtype=torch.float64).unsqueeze(0)
            elif self.backend == 'numpy' and isinstance(initial_phi, torch.Tensor):
                phi = initial_phi.cpu().numpy()[None, :].astype(np.float64)
            else:
                phi = initial_phi.copy()[None, :]
                
        burn_in_iter = trange(burn_in, desc="Burn-in") if show_progress else range(burn_in)
        for _ in burn_in_iter:
            phi = self.step(phi)

        trajectory = self._zeros((n_steps, self.Lx))
        step_iter = trange(n_steps, desc="Generating Trajectory") if show_progress else range(n_steps)
        for t in step_iter:
            trajectory[t] = phi[0]
            phi = self.step(phi)
            
        if self.backend == 'torch':
            return trajectory.cpu().numpy()
        return trajectory

    def generate_trajectories(self, n_trajectories, n_steps, burn_in=0, show_progress=True):
        phi = self._init_field(n_trajectories)

        burn_in_iter = trange(burn_in, desc="Burn-in Ensemble") if show_progress else range(burn_in)
        for _ in burn_in_iter:
            phi = self.step(phi)

        trajectories = self._zeros((n_trajectories, n_steps, self.Lx))

        step_iter = trange(n_steps, desc="Generating Trajectories") if show_progress else range(n_steps)
        for t in step_iter:
            trajectories[:, t, :] = phi
            phi = self.step(phi)

        if self.backend == 'torch':
            return trajectories.cpu().numpy()
        return trajectories

    def compute_mean_epr_on_the_fly(self, n_trajectories, n_steps, burn_in=1000, show_progress=True):
        """
        Extremely memory efficient on-the-fly Mean EPR Density generation for huge M x N ensembles
        Avoids storing full M x T x N trajectory tensors.
        Returns: ensemble_mean_epr_density array of size (Lx,)
        """
        phi_t = self._init_field(n_trajectories)
        
        burn_in_iter = trange(burn_in, desc="Burn-in Ensemble") if show_progress else range(burn_in)
        for _ in burn_in_iter:
            phi_t = self.step(phi_t)

        ensemble_epr_density = np.zeros(self.Lx, dtype=np.float64)

        step_iter = trange(n_steps, desc="Simulating & Computing On-the-Fly", leave=False) if show_progress else range(n_steps)
        if self.epr_mode == "mid":
            phi_p = phi_t
            phi_t = self.step(phi_t)

            for _ in step_iter:
                phi_n = self.step(phi_t)

                sigma = self.compute_local_epr_density_three(phi_p, phi_t, phi_n)
                if hasattr(sigma, 'cpu'):
                    sigma = sigma.cpu().numpy()

                # Accumulate spatial mean across M parallel seeds (axis=0)
                ensemble_epr_density += np.mean(sigma, axis=0)

                phi_p = phi_t
                phi_t = phi_n

            # Average exactly over the number of steps simulated
            ensemble_epr_density /= (n_steps)
        else:
            for _ in step_iter:
                phi_tp1 = self.step(phi_t)

                sigma = self.compute_local_epr_density(phi_t, phi_tp1)
                if hasattr(sigma, 'cpu'):
                    sigma = sigma.cpu().numpy()

                # Accumulate spatial mean across M parallel seeds (axis=0)
                ensemble_epr_density += np.mean(sigma, axis=0)

                phi_t = phi_tp1

            # Average exactly over the number of steps simulated
            ensemble_epr_density /= n_steps
        
        return ensemble_epr_density

    def compute_mean_epr(self, trajectory):
        if trajectory.ndim == 2:
            trajectory = trajectory[None, :, :]
            
        T = trajectory.shape[1]
        
        if self.backend == 'torch':
            traj_t = torch.tensor(trajectory, device=self.device, dtype=torch.float64)
        else:
            traj_t = trajectory
            
        total = 0.0
        # For compute_mean_epr, we don't necessarily need a progress bar, 
        # but if T is very large, it's nice. Let's use tqdm.
        if self.epr_mode == "mid":
            for t in tqdm(range(T - 2), desc="Computing EPR", leave=False):
                total += self.compute_total_epr_three(traj_t[:, t], traj_t[:, t + 1], traj_t[:, t + 2])
        else:
            for t in tqdm(range(T - 1), desc="Computing EPR", leave=False):
                total += self.compute_total_epr(traj_t[:, t], traj_t[:, t + 1])
            
        res = total / T
        if self.backend == 'torch' and isinstance(res, torch.Tensor):
            return res.cpu().numpy()
        return res

def save_trajectories(trajectories, output_path, metadata=None):
    save_dict = {"trajectories": trajectories}
    if metadata is not None:
        save_dict["metadata"] = metadata
    np.savez(output_path, **save_dict)
    print(f"Saved trajectories to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Active Model B 1D trajectory generator")
    parser.add_argument("--L", type=float, default=500.0, help="Total length of the domain")
    parser.add_argument("--Lx", type=int, default=None, help="Number of grid points (overrides L if specified)")
    parser.add_argument("--dx", type=float, default=1.0)
    parser.add_argument("--a", type=float, default=0.25)
    parser.add_argument("--b", type=float, default=0.25)
    parser.add_argument("--kappa", type=float, default=4.0)
    parser.add_argument("--lam", type=float, default=1.0)
    parser.add_argument("--D", type=float, default=0.01)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--bc", type=str, default="periodic", choices=["periodic", "neumann"])

    parser.add_argument("--n_trajectories", type=int, default=1)
    parser.add_argument("--n_steps", type=int, default=48000)
    parser.add_argument("--burn_in", type=int, default=24000)
    parser.add_argument("--smooth", action="store_true")

    parser.add_argument("--backend", type=str, default="numpy", choices=["numpy", "torch"])
    parser.add_argument("--use_gpu", action="store_true")
    
    parser.add_argument("--output", type=str, default="amb_1d_trajectories.npz")
    parser.add_argument("--seed", type=int, default=None)

    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)
        if torch is not None:
            torch.manual_seed(args.seed)

    if args.Lx is None:
        args.Lx = int(args.L / args.dx)

    print(f"[INFO] Active Model B 1D ({args.Lx} points) - Periodic Boundary (Double Wall)")
    print(f"[INFO] Backend: {args.backend}, GPU: {args.use_gpu}")
    model = ActiveModelB1D(
        Lx=args.Lx, dx=args.dx, a=args.a, b=args.b,
        kappa=args.kappa, lam=args.lam, D=args.D, dt=args.dt, smooth=args.smooth,
        backend=args.backend, use_gpu=args.use_gpu, bc=args.bc
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

if __name__ == "__main__":
    main()
