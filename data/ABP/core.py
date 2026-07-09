"""Continuous-space Active Brownian Particles with WCA repulsion.

The simulator follows the notebook prototype in ``data/mips_test.ipynb``:

    dr_i = v0 n_i dt + mobility F_i^WCA dt + sqrt(2 Dt dt) dW_i
    dtheta_i = sqrt(2 Dr dt) dW_i

Particles live in a periodic square box.  The force implementation is fully
tensorized and chunked over source particles, so moderate ensembles run on CUDA
without allocating the full ``B x N x N x 2`` tensor when N is large.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch


def choose_device(device: str | torch.device = "auto") -> torch.device:
    """Resolve ``auto`` to CUDA when available."""
    if isinstance(device, torch.device):
        return device
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def set_seed(seed: Optional[int], device: torch.device) -> None:
    """Set NumPy and torch seeds."""
    if seed is None:
        return
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


@dataclass
class ABPParams:
    """Parameters for overdamped 2D WCA-ABP dynamics."""

    # system
    N: int = 256
    L: float = 32.0
    sigma: float = 1.0

    # WCA interaction
    epsilon: float = 1.0
    mobility: float = 1.0
    force_clip: Optional[float] = 1.0e3
    force_chunk_size: int = 512

    # active Brownian dynamics
    v0: float = 20.0
    Dr: float = 1.0
    Dt: float = 0.0

    # integration
    dt: float = 1.0e-4
    seed: Optional[int] = 0
    device: str = "auto"
    dtype: str = "float32"

    @property
    def phi(self) -> float:
        """Packing fraction N*pi*(sigma/2)^2/L^2."""
        return self.N * math.pi * self.sigma**2 / (4.0 * self.L**2)

    @property
    def rc(self) -> float:
        """WCA cutoff radius."""
        return 2.0 ** (1.0 / 6.0) * self.sigma

    @property
    def Pe(self) -> float:
        """Common ABP Péclet number v0/(Dr*sigma)."""
        if self.Dr <= 0 or self.sigma <= 0:
            return float("inf")
        return self.v0 / (self.Dr * self.sigma)

    @property
    def torch_dtype(self) -> torch.dtype:
        if self.dtype == "float64":
            return torch.float64
        if self.dtype == "float32":
            return torch.float32
        raise ValueError("dtype must be 'float32' or 'float64'.")


class ContinuousABP:
    """Ensemble-parallel continuous ABP simulator."""

    def __init__(self, params: ABPParams | None = None, **kwargs):
        if params is None:
            params = ABPParams(**kwargs)
        elif kwargs:
            merged = params.__dict__.copy()
            merged.update(kwargs)
            params = ABPParams(**merged)

        self.params = params
        self.device = choose_device(params.device)
        self.dtype = params.torch_dtype
        set_seed(params.seed, self.device)

    # ------------------------------------------------------------------
    # Geometry and initialization
    # ------------------------------------------------------------------

    @property
    def L(self) -> float:
        return self.params.L

    @property
    def N(self) -> int:
        return self.params.N

    def minimum_image(self, delta: torch.Tensor) -> torch.Tensor:
        """Apply the square periodic minimum-image convention."""
        return delta - self.L * torch.round(delta / self.L)

    def initialize_lattice(
        self,
        B: int = 1,
        jitter: float = 0.02,
        random_shift: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Place particles on a perturbed square lattice to avoid WCA blow-ups."""
        p = self.params
        n_side = math.ceil(math.sqrt(p.N))
        spacing = p.L / n_side

        xs = torch.arange(n_side, device=self.device, dtype=self.dtype) + 0.5
        ys = torch.arange(n_side, device=self.device, dtype=self.dtype) + 0.5
        xx, yy = torch.meshgrid(xs, ys, indexing="ij")
        base = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)[: p.N] * spacing
        base = base.unsqueeze(0).expand(B, -1, -1).clone()

        if random_shift:
            shift = p.L * torch.rand(B, 1, 2, device=self.device, dtype=self.dtype)
            base = (base + shift) % p.L

        if jitter > 0:
            base = (base + jitter * p.sigma * torch.randn_like(base)) % p.L

        theta = 2.0 * math.pi * torch.rand(B, p.N, device=self.device, dtype=self.dtype)
        return base, theta

    # ------------------------------------------------------------------
    # WCA interaction
    # ------------------------------------------------------------------

    def compute_wca_forces(
        self,
        pos: torch.Tensor,
        *,
        return_potential: bool = False,
        return_min_distance: bool = False,
        chunk_size: Optional[int] = None,
    ):
        """Compute WCA forces with chunked pairwise CUDA tensor operations.

        Parameters
        ----------
        pos:
            Particle positions with shape ``[B, N, 2]``.
        return_potential:
            Also return total WCA potential energy per ensemble.
        return_min_distance:
            Also return the minimum pair distance per ensemble.
        chunk_size:
            Source-particle chunk size.  Defaults to
            ``params.force_chunk_size``.
        """
        if pos.dim() != 3 or pos.shape[-1] != 2:
            raise ValueError("pos must have shape [B, N, 2].")

        p = self.params
        B, N, _ = pos.shape
        chunk = int(chunk_size or p.force_chunk_size or N)
        rc2 = p.rc**2
        sigma2 = p.sigma**2

        forces = torch.zeros_like(pos)
        potential = torch.zeros(B, device=pos.device, dtype=pos.dtype)
        min_r2 = torch.full((B,), float("inf"), device=pos.device, dtype=pos.dtype)

        eps = torch.finfo(pos.dtype).eps
        arange_n = torch.arange(N, device=pos.device)

        for start in range(0, N, chunk):
            end = min(start + chunk, N)
            pos_i = pos[:, start:end, :]
            delta = pos_i[:, :, None, :] - pos[:, None, :, :]
            delta = self.minimum_image(delta)
            r2 = torch.sum(delta * delta, dim=-1)

            src_idx = arange_n[start:end].view(1, -1, 1)
            dst_idx = arange_n.view(1, 1, -1)
            not_self = src_idx != dst_idx
            in_cutoff = (r2 > eps) & (r2 < rc2) & not_self

            r2_safe = torch.where(in_cutoff, r2, torch.ones_like(r2))
            inv_r2 = 1.0 / r2_safe
            sig2_over_r2 = sigma2 * inv_r2
            sig6 = sig2_over_r2**3
            sig12 = sig6**2

            scalar = 24.0 * p.epsilon * (2.0 * sig12 - sig6) * inv_r2
            scalar = torch.where(in_cutoff, scalar, torch.zeros_like(scalar))
            forces[:, start:end, :] = torch.sum(scalar[..., None] * delta, dim=2)

            if return_potential:
                u_pair = 4.0 * p.epsilon * (sig12 - sig6) + p.epsilon
                u_pair = torch.where(in_cutoff, u_pair, torch.zeros_like(u_pair))
                potential = potential + 0.5 * u_pair.sum(dim=(1, 2))

            if return_min_distance:
                pair_r2 = torch.where(not_self & (r2 > eps), r2, torch.full_like(r2, float("inf")))
                min_r2 = torch.minimum(min_r2, pair_r2.amin(dim=(1, 2)))

        if p.force_clip is not None and p.force_clip > 0:
            norm = torch.linalg.norm(forces, dim=-1, keepdim=True).clamp_min(eps)
            scale = torch.clamp(torch.as_tensor(p.force_clip, device=pos.device, dtype=pos.dtype) / norm, max=1.0)
            forces = forces * scale

        outputs = [forces]
        if return_potential:
            outputs.append(potential)
        if return_min_distance:
            outputs.append(torch.sqrt(min_r2))
        if len(outputs) == 1:
            return outputs[0]
        return tuple(outputs)

    def wca_potential_energy(self, pos: torch.Tensor) -> torch.Tensor:
        """Return total WCA potential energy per ensemble."""
        _, potential = self.compute_wca_forces(pos, return_potential=True)
        return potential

    # ------------------------------------------------------------------
    # Dynamics
    # ------------------------------------------------------------------

    def step(
        self,
        pos: torch.Tensor,
        theta: torch.Tensor,
        *,
        return_diagnostics: bool = False,
    ):
        """Advance one Euler-Maruyama step."""
        p = self.params
        forces, potential, min_distance = self.compute_wca_forces(
            pos,
            return_potential=True,
            return_min_distance=True,
        )
        direction = torch.stack([torch.cos(theta), torch.sin(theta)], dim=-1)
        drift = p.v0 * direction + p.mobility * forces

        pos_next = pos + p.dt * drift
        if p.Dt > 0:
            pos_next = pos_next + math.sqrt(2.0 * p.Dt * p.dt) * torch.randn_like(pos)
        pos_next = pos_next % p.L

        theta_next = theta + math.sqrt(2.0 * p.Dr * p.dt) * torch.randn_like(theta)
        theta_next = theta_next % (2.0 * math.pi)

        if return_diagnostics:
            return pos_next, theta_next, {
                "potential": potential,
                "min_distance": min_distance,
                "mean_force_norm": torch.linalg.norm(forces, dim=-1).mean(dim=1),
                "mean_active_speed": torch.full_like(potential, p.v0),
            }
        return pos_next, theta_next

    def simulate(
        self,
        B: int = 1,
        n_steps: int = 1000,
        burn_in: int = 0,
        save_interval: int = 10,
        *,
        initial_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        fieldizer=None,
        show_progress: bool = True,
        save_diagnostics: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Run a simulation and return CPU tensors.

        Saved arrays use time-major layout:

        ``positions``: ``[T, B, N, 2]``
        ``theta``: ``[T, B, N]``
        ``fields``: ``[T, B, C, H, W]`` when a fieldizer is supplied.
        """
        try:
            from tqdm import trange
        except ImportError:  # pragma: no cover
            trange = range

        if save_interval <= 0:
            raise ValueError("save_interval must be positive.")

        if initial_state is None:
            pos, theta = self.initialize_lattice(B)
        else:
            pos, theta = initial_state
            pos = pos.to(device=self.device, dtype=self.dtype)
            theta = theta.to(device=self.device, dtype=self.dtype)

        burn_iter = trange(burn_in, desc="ABP burn-in", leave=False) if show_progress else range(burn_in)
        with torch.no_grad():
            for _ in burn_iter:
                pos, theta = self.step(pos, theta)

        n_saved = n_steps // save_interval + 1
        positions = torch.empty(n_saved, B, self.N, 2, dtype=self.dtype, device=self.device)
        angles = torch.empty(n_saved, B, self.N, dtype=self.dtype, device=self.device)
        times = torch.empty(n_saved, dtype=self.dtype, device=self.device)

        fields = None
        if fieldizer is not None:
            sample = fieldizer.encode(pos, theta)
            fields = torch.empty(
                n_saved,
                *sample.shape,
                dtype=sample.dtype,
                device=sample.device,
            )

        diag: Dict[str, torch.Tensor] = {}
        if save_diagnostics:
            diag = {
                "potential": torch.empty(n_saved, B, dtype=self.dtype, device=self.device),
                "min_distance": torch.empty(n_saved, B, dtype=self.dtype, device=self.device),
                "mean_force_norm": torch.empty(n_saved, B, dtype=self.dtype, device=self.device),
            }

        def save_frame(save_idx: int, step: int) -> None:
            positions[save_idx] = pos
            angles[save_idx] = theta
            times[save_idx] = step * self.params.dt
            if fields is not None:
                fields[save_idx] = fieldizer.encode(pos, theta)
            if save_diagnostics:
                _, potential, min_distance = self.compute_wca_forces(
                    pos,
                    return_potential=True,
                    return_min_distance=True,
                )
                diag["potential"][save_idx] = potential
                diag["min_distance"][save_idx] = min_distance
                diag["mean_force_norm"][save_idx] = torch.linalg.norm(
                    self.compute_wca_forces(pos), dim=-1
                ).mean(dim=1)

        save_idx = 0
        prod_iter = trange(n_steps + 1, desc="ABP simulate", leave=False) if show_progress else range(n_steps + 1)
        with torch.no_grad():
            for step in prod_iter:
                if step % save_interval == 0:
                    save_frame(save_idx, step)
                    save_idx += 1
                if step < n_steps:
                    pos, theta = self.step(pos, theta)

        out = {
            "positions": positions.cpu(),
            "theta": angles.cpu(),
            "times": times.cpu(),
            "params": self.params,
        }
        if fields is not None:
            out["fields"] = fields.cpu()
        for key, value in diag.items():
            out[key] = value.cpu()
        return out

    # ------------------------------------------------------------------
    # Analysis helpers
    # ------------------------------------------------------------------

    def pair_distances(self, pos: torch.Tensor, chunk_size: Optional[int] = None) -> torch.Tensor:
        """Return flattened upper-triangle pair distances for ``B=1`` or batched states."""
        if pos.dim() == 2:
            pos = pos.unsqueeze(0)
        B, N, _ = pos.shape
        chunk = int(chunk_size or self.params.force_chunk_size or N)
        dists = []
        arange_n = torch.arange(N, device=pos.device)
        for start in range(0, N, chunk):
            end = min(start + chunk, N)
            delta = pos[:, start:end, None, :] - pos[:, None, :, :]
            delta = self.minimum_image(delta)
            r = torch.linalg.norm(delta, dim=-1)
            src = arange_n[start:end].view(1, -1, 1)
            dst = arange_n.view(1, 1, -1)
            mask = src < dst
            dists.append(r[mask.expand(B, -1, -1)])
        return torch.cat(dists)
