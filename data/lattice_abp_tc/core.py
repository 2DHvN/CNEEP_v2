"""Thermodynamically consistent lattice Active Brownian particles.

The translational update follows Kim, Kwon, and Baek,
arXiv:2503.16958, Eq. (5), with WCA interaction energy changes and
hop-wise true medium entropy production.
"""

from __future__ import annotations

import math
import warnings
from collections import deque
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch

from ._cuda_backend import (
    CudaBackendUnavailable,
    cuda_backend_buildable,
    cuda_sweep_inplace,
    load_cuda_backend,
    prepare_cuda_sweep_inputs,
)
from ._numba_backend import (
    NUMBA_AVAILABLE,
    active_work_from_theta_vectorized,
    numba_sweep_inplace,
    prepare_sweep_random_inputs,
    require_numba,
)


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
class ThermodynamicLatticeABPParams:
    """Parameters for the thermodynamically consistent lattice ABP method.

    The translational update follows Kim, Kwon, and Baek,
    arXiv:2503.16958, Eq. (5).  The many-body potential is a WCA pair
    potential, evaluated through the exact energy change for each proposed
    single-particle hop.
    """

    # system
    N: int = 256
    L: float = 32.0
    grid_size: int = 128
    sigma: float = 1.0

    # WCA interaction
    epsilon: float = 10.0
    mobility: float = 1.0

    # active Brownian dynamics
    v0: float = 20.0
    Dr: float = 1.0
    Dt: float = 0.1

    # lattice Monte Carlo integration
    dt: float = 1.0e-4
    prefactor: str = "cv"
    strict_probabilities: bool = True
    probability_tolerance: float = 1.0e-6
    shuffle_particles: bool = True

    # initialization and reproducibility
    initial_min_distance: Optional[float] = None
    seed: Optional[int] = 0
    device: str = "auto"
    dtype: str = "float32"
    # Keep the finite-precision Torch reference as the library default. Demos
    # opt into "auto", which selects a device-specific fused exact backend.
    backend: str = "torch"

    @property
    def dl(self) -> float:
        """Lattice spacing."""
        return self.L / float(self.grid_size)

    @property
    def phi(self) -> float:
        """Packing fraction N*pi*(sigma/2)^2/L^2."""
        return self.N * math.pi * self.sigma**2 / (4.0 * self.L**2)

    @property
    def rc(self) -> float:
        """WCA cutoff radius."""
        return 2.0 ** (1.0 / 6.0) * self.sigma

    @property
    def gamma2(self) -> float:
        """Continuum-limit control parameter dl^2/dt."""
        return self.dl * self.dl / self.dt

    @property
    def Pe(self) -> float:
        """ABP Péclet number v0/(Dr*sigma)."""
        if self.Dr <= 0 or self.sigma <= 0:
            return float("inf")
        return self.v0 / (self.Dr * self.sigma)

    @property
    def temperature(self) -> float:
        """Reservoir temperature T = D/mu."""
        return self.Dt / self.mobility

    @property
    def torch_dtype(self) -> torch.dtype:
        if self.dtype == "float64":
            return torch.float64
        if self.dtype == "float32":
            return torch.float32
        raise ValueError("dtype must be 'float32' or 'float64'.")


class ThermodynamicLatticeABP:
    """Discrete-space ABP simulator with thermodynamically consistent hops.

    A single Monte Carlo step is one random sequential sweep through all
    particles followed by an exact angular Brownian update.  Each accepted hop
    contributes the true medium entropy production

    ``Delta S_med = (v dot Delta r - mobility * Delta V) / Dt``,

    where ``Delta V`` is the WCA potential-energy change caused by that hop.
    """

    # Physical coordinates are stored as (x, y).  Sites are integer indices.
    _DIR_SITES_CPU = torch.tensor(
        [[1, 0], [-1, 0], [0, 1], [0, -1]], dtype=torch.long
    )

    def __init__(self, params: ThermodynamicLatticeABPParams | None = None, **kwargs):
        if params is None:
            params = ThermodynamicLatticeABPParams(**kwargs)
        elif kwargs:
            merged = params.__dict__.copy()
            merged.update(kwargs)
            params = ThermodynamicLatticeABPParams(**merged)

        self.params = params
        self._validate_params()
        self.device = choose_device(params.device)
        self.dtype = params.torch_dtype
        self.backend = self._resolve_backend(params.backend)
        set_seed(params.seed, self.device)

        self.dir_sites = self._DIR_SITES_CPU.to(self.device)
        self.dir_vectors = self.dir_sites.to(dtype=self.dtype) * params.dl
        self._kernel_offsets, self._kernel_values = self._build_wca_kernel()
        neighbor_linear = self._build_neighbor_linear_lookup()
        if self.backend == "cuda_fused" and neighbor_linear is not None:
            # The custom kernel accepts compact int32 site indices. The dense
            # long lookup is not retained, so the cache costs only ~2.5 MiB
            # for the default G=96, K=68 geometry.
            self._cuda_neighbor_linear = neighbor_linear.to(torch.int32)
            self._neighbor_linear = None
        else:
            self._cuda_neighbor_linear = None
            self._neighbor_linear = neighbor_linear

    # ------------------------------------------------------------------
    # Parameter and geometry helpers
    # ------------------------------------------------------------------

    @property
    def N(self) -> int:
        return self.params.N

    @property
    def L(self) -> float:
        return self.params.L

    @property
    def grid_size(self) -> int:
        return self.params.grid_size

    @property
    def dl(self) -> float:
        return self.params.dl

    def _validate_params(self) -> None:
        p = self.params
        if p.N <= 0:
            raise ValueError("N must be positive.")
        if p.L <= 0 or p.grid_size <= 0:
            raise ValueError("L and grid_size must be positive.")
        if p.sigma <= 0 or p.epsilon < 0:
            raise ValueError("sigma must be positive and epsilon must be nonnegative.")
        if p.mobility <= 0:
            raise ValueError("mobility must be positive.")
        if p.Dt <= 0:
            raise ValueError("Thermodynamic lattice ABP requires Dt > 0.")
        if p.Dr < 0 or p.dt <= 0:
            raise ValueError("Dr must be nonnegative and dt must be positive.")
        if p.prefactor.lower() not in {"c0", "cv"}:
            raise ValueError("prefactor must be 'c0' or 'cv'.")
        if p.backend.lower() not in {
            "auto",
            "cuda_fused",
            "torch",
            "numba",
        }:
            raise ValueError(
                "backend must be 'auto', 'cuda_fused', 'torch', or 'numba'."
            )
        if p.N > p.grid_size * p.grid_size:
            raise ValueError("N cannot exceed the number of lattice sites.")

    def _resolve_backend(self, backend: str) -> str:
        normalized = backend.lower()
        if normalized == "auto":
            if self.device.type == "cuda":
                if cuda_backend_buildable():
                    try:
                        load_cuda_backend()
                    except CudaBackendUnavailable as exc:
                        warnings.warn(
                            "The fused CUDA backend could not be loaded; "
                            "using the exact fixed-shape Torch CUDA fallback. "
                            f"Reason: {exc}",
                            RuntimeWarning,
                            stacklevel=2,
                        )
                    else:
                        return "cuda_fused"
                else:
                    warnings.warn(
                        "The fused CUDA backend is not buildable (a CUDA "
                        "toolkit/nvcc, compatible host compiler, and Ninja "
                        "are required); using the exact fixed-shape Torch "
                        "CUDA fallback.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                return "torch"
            if self.device.type == "cpu" and NUMBA_AVAILABLE:
                return "numba"
            return "torch"
        if normalized == "cuda_fused":
            if self.device.type != "cuda":
                raise ValueError(
                    "backend='cuda_fused' requires a CUDA device."
                )
            # Explicit selection is fail-fast: never silently replace the
            # requested implementation with a slower backend.
            load_cuda_backend()
            return normalized
        if normalized == "numba":
            if self.device.type != "cpu":
                raise ValueError("backend='numba' requires a CPU device.")
            require_numba()
        return normalized

    def _build_wca_kernel(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Precompute WCA energies for lattice offsets inside the cutoff."""
        p = self.params
        radius = int(math.ceil(p.rc / p.dl))
        offsets = []
        values = []
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                if dx == 0 and dy == 0:
                    continue
                r = p.dl * math.sqrt(dx * dx + dy * dy)
                if r <= 0 or r >= p.rc:
                    continue
                sr6 = (p.sigma / r) ** 6
                values.append(4.0 * p.epsilon * (sr6 * sr6 - sr6) + p.epsilon)
                offsets.append((dx, dy))

        if not offsets:
            return (
                torch.empty(0, 2, dtype=torch.long, device=self.device),
                torch.empty(0, dtype=self.dtype, device=self.device),
            )
        return (
            torch.tensor(offsets, dtype=torch.long, device=self.device),
            torch.tensor(values, dtype=self.dtype, device=self.device),
        )

    def _build_neighbor_linear_lookup(self) -> Optional[torch.Tensor]:
        """Precompute exact periodic WCA-neighbor indices for CUDA gathers.

        Integer geometry does not change during a run.  Caching it removes the
        modulo and coordinate arithmetic from every particle proposal.  The
        lookup is CUDA-only and capped so unusually large grids do not acquire
        an unexpectedly large persistent allocation.
        """
        if (
            self.device.type != "cuda"
            or self.params.epsilon == 0
            or self._kernel_offsets.numel() == 0
        ):
            return None
        entry_count = (
            self.grid_size
            * self.grid_size
            * int(self._kernel_offsets.shape[0])
        )
        max_lookup_bytes = 512 * 1024 * 1024
        index_bytes = torch.empty((), dtype=torch.long).element_size()
        if entry_count * index_bytes > max_lookup_bytes:
            return None

        linear = torch.arange(
            self.grid_size * self.grid_size,
            device=self.device,
            dtype=torch.long,
        )
        x = torch.div(linear, self.grid_size, rounding_mode="floor")
        y = linear.remainder(self.grid_size)
        neighbor_x = (
            x.unsqueeze(1) + self._kernel_offsets[:, 0].unsqueeze(0)
        ).remainder(self.grid_size)
        neighbor_y = (
            y.unsqueeze(1) + self._kernel_offsets[:, 1].unsqueeze(0)
        ).remainder(self.grid_size)
        return (neighbor_x * self.grid_size + neighbor_y).contiguous()

    def minimum_image(self, delta: torch.Tensor) -> torch.Tensor:
        """Apply square periodic minimum-image convention to physical deltas."""
        return delta - self.L * torch.round(delta / self.L)

    def sites_to_positions(self, sites: torch.Tensor) -> torch.Tensor:
        """Convert integer lattice sites to physical cell-center positions."""
        return (sites.to(device=self.device, dtype=self.dtype) + 0.5) * self.dl

    def occupancy_from_sites(self, sites: torch.Tensor) -> torch.Tensor:
        """Build an integer occupancy grid with shape ``[B, G, G]``."""
        if sites.dim() == 2:
            sites = sites.unsqueeze(0)
        if sites.dim() != 3 or sites.shape[-1] != 2:
            raise ValueError("sites must have shape [B, N, 2] or [N, 2].")

        sites = sites.to(device=self.device, dtype=torch.long) % self.grid_size
        B, N, _ = sites.shape
        linear = sites[..., 0] * self.grid_size + sites[..., 1]
        counts = torch.zeros(
            B,
            self.grid_size * self.grid_size,
            dtype=torch.long,
            device=self.device,
        )
        counts.scatter_add_(1, linear, torch.ones(B, N, dtype=torch.long, device=self.device))
        return counts.view(B, self.grid_size, self.grid_size)

    def _occupancy_at(self, occupancy: torch.Tensor, sites: torch.Tensor) -> torch.Tensor:
        """Return occupancy at ``sites`` with output shape ``[B, M]``."""
        if sites.dim() == 2:
            sites = sites.unsqueeze(1)
        B, M, _ = sites.shape
        batch_idx = torch.arange(B, device=self.device).view(B, 1).expand(B, M)
        return occupancy[batch_idx, sites[..., 0] % self.grid_size, sites[..., 1] % self.grid_size]

    def _add_occupancy(self, occupancy: torch.Tensor, sites: torch.Tensor, amount: int) -> None:
        batch_idx = torch.arange(sites.shape[0], device=self.device)
        occupancy[batch_idx, sites[:, 0] % self.grid_size, sites[:, 1] % self.grid_size] += amount

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def init_state(
        self,
        B: int = 1,
        *,
        random_shift: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize particles on a random shifted nonoverlapping sublattice."""
        p = self.params
        min_distance = p.sigma if p.initial_min_distance is None else p.initial_min_distance
        stride = max(1, int(math.ceil(min_distance / p.dl - 1.0e-12)))
        coords_1d = torch.arange(0, p.grid_size, stride, device=self.device)
        xx, yy = torch.meshgrid(coords_1d, coords_1d, indexing="ij")
        candidates = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)

        if candidates.shape[0] < p.N:
            raise ValueError(
                "Not enough nonoverlapping initialization sites. "
                "Decrease N/initial_min_distance or increase L/grid_size."
            )

        sites = torch.empty(B, p.N, 2, dtype=torch.long, device=self.device)
        for b in range(B):
            perm = torch.randperm(candidates.shape[0], device=self.device)[: p.N]
            state = candidates[perm].clone()
            if random_shift:
                shift = torch.randint(0, p.grid_size, (2,), dtype=torch.long, device=self.device)
                state = (state + shift) % p.grid_size
            sites[b] = state

        theta = 2.0 * math.pi * torch.rand(B, p.N, device=self.device, dtype=self.dtype)
        return sites, theta

    # ------------------------------------------------------------------
    # WCA energetics
    # ------------------------------------------------------------------

    def _local_wca_energy(
        self,
        occupancy: torch.Tensor,
        sites: torch.Tensor,
        batch_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return WCA energy of test particles at ``sites`` against occupancy."""
        if sites.dim() == 2:
            sites = sites.unsqueeze(1)
        B, M, _ = sites.shape
        if self._kernel_offsets.numel() == 0 or self.params.epsilon == 0:
            return torch.zeros(B, M, device=self.device, dtype=self.dtype)

        sites = sites.to(device=self.device, dtype=torch.long)
        kernel_size = int(self._kernel_offsets.shape[0])
        if batch_indices is None:
            batch_indices = torch.arange(B, device=self.device)
        batch_idx = batch_indices.view(B, 1, 1).expand(
            B,
            M,
            kernel_size,
        )
        if self._neighbor_linear is not None:
            linear_sites = (
                sites[..., 0].remainder(self.grid_size) * self.grid_size
                + sites[..., 1].remainder(self.grid_size)
            )
            neighbor_linear = self._neighbor_linear.index_select(
                0,
                linear_sites.reshape(-1),
            ).view(B, M, kernel_size)
            occupancy_flat = occupancy.view(B, -1)
            occ = occupancy_flat[
                batch_idx,
                neighbor_linear,
            ].to(dtype=self.dtype)
        else:
            offsets = self._kernel_offsets.view(1, 1, -1, 2)
            query = sites.unsqueeze(2)
            neighbor = (query + offsets) % self.grid_size
            occ = occupancy[
                batch_idx,
                neighbor[..., 0],
                neighbor[..., 1],
            ].to(dtype=self.dtype)
        return torch.sum(occ * self._kernel_values.view(1, 1, -1), dim=-1)

    def potential_energy(self, sites: torch.Tensor, occupancy: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Return total WCA potential energy per ensemble."""
        if sites.dim() == 2:
            sites = sites.unsqueeze(0)
        sites = sites.to(device=self.device, dtype=torch.long) % self.grid_size
        if occupancy is None:
            occupancy = self.occupancy_from_sites(sites)
        else:
            occupancy = occupancy.to(device=self.device)
        return 0.5 * self._local_wca_energy(occupancy, sites).sum(dim=1)

    def pair_distances(self, sites: torch.Tensor, chunk_size: Optional[int] = None) -> torch.Tensor:
        """Return flattened upper-triangle pair distances for lattice states."""
        if sites.dim() == 2:
            sites = sites.unsqueeze(0)
        pos = self.sites_to_positions(sites)
        B, N, _ = pos.shape
        chunk = int(chunk_size or N)
        dists = []
        arange_n = torch.arange(N, device=self.device)
        for start in range(0, N, chunk):
            end = min(start + chunk, N)
            delta = pos[:, start:end, None, :] - pos[:, None, :, :]
            delta = self.minimum_image(delta)
            r = torch.linalg.norm(delta, dim=-1)
            src = arange_n[start:end].view(1, -1, 1)
            dst = arange_n.view(1, 1, -1)
            mask = src < dst
            dists.append(r[mask.expand(B, -1, -1)])
        return torch.cat(dists) if dists else torch.empty(0, device=self.device, dtype=self.dtype)

    # ------------------------------------------------------------------
    # Thermodynamically consistent hop probabilities
    # ------------------------------------------------------------------

    def _cv_factor(self, x: torch.Tensor) -> torch.Tensor:
        """Return ``x * exp(x) / sinh(x)`` with stable limiting forms."""
        finite = torch.isfinite(x)
        safe_x = torch.where(finite, x, torch.zeros_like(x))

        small_values = 1.0 + safe_x + safe_x * safe_x / 3.0
        large_positive_values = 2.0 * safe_x
        large_negative_values = (
            -2.0 * safe_x * torch.exp(2.0 * safe_x)
        )
        mid_values = 2.0 * safe_x / (
            -torch.expm1(-2.0 * safe_x)
        )
        values = torch.where(
            safe_x.abs() < 1.0e-5,
            small_values,
            torch.where(
                safe_x > 50.0,
                large_positive_values,
                torch.where(
                    safe_x < -50.0,
                    large_negative_values,
                    mid_values,
                ),
            ),
        )
        return torch.where(finite, values, torch.zeros_like(values)).clamp_min(0.0)

    def _transition_probabilities_from_delta(
        self,
        active_work: torch.Tensor,
        delta_potential: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Eq. (5) probabilities from active work and WCA ``Delta V``."""
        p = self.params
        affinity = active_work - p.mobility * delta_potential
        x = affinity / (2.0 * p.Dt)
        base = p.dt * p.Dt / (p.dl * p.dl)

        finite = torch.isfinite(x)
        if p.prefactor.lower() == "c0":
            probs = base * torch.exp(torch.where(finite, x, torch.zeros_like(x)))
        else:
            probs = base * self._cv_factor(x)
        probs = torch.where(finite & torch.isfinite(delta_potential), probs, torch.zeros_like(probs))
        return probs.clamp_min(0.0)

    def compute_particle_hop_probabilities(
        self,
        sites: torch.Tensor,
        theta: torch.Tensor,
        particle_idx: int,
        occupancy: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Return hop probabilities and thermodynamic increments for one particle.

        This is mostly a diagnostic helper.  The simulation path computes the
        same quantities while temporarily removing the particle from the
        occupancy grid, so that the proposed new position interacts only with
        the other particles.
        """
        if sites.dim() == 2:
            sites = sites.unsqueeze(0)
        if theta.dim() == 1:
            theta = theta.unsqueeze(0)
        sites = sites.to(device=self.device, dtype=torch.long) % self.grid_size
        theta = theta.to(device=self.device, dtype=self.dtype)
        if occupancy is None:
            occupancy = self.occupancy_from_sites(sites)
        else:
            occupancy = occupancy.clone().to(device=self.device)

        old_sites = sites[:, particle_idx].clone()
        self._add_occupancy(occupancy, old_sites, -1)
        probs, new_sites, delta_v, active_work = self._particle_probabilities_without_self(
            occupancy, old_sites, theta[:, particle_idx]
        )
        self._add_occupancy(occupancy, old_sites, 1)
        return {
            "probabilities": probs,
            "new_sites": new_sites,
            "delta_potential": delta_v,
            "active_work": active_work,
            "medium_ep": (active_work - self.params.mobility * delta_v) / self.params.Dt,
        }

    def _particle_probabilities_without_self(
        self,
        occupancy_without_particle: torch.Tensor,
        old_sites: torch.Tensor,
        theta_i: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B = old_sites.shape[0]
        new_sites = (old_sites.unsqueeze(1) + self.dir_sites.view(1, 4, 2)) % self.grid_size

        old_energy = self._local_wca_energy(
            occupancy_without_particle,
            old_sites.unsqueeze(1),
        )[:, 0]
        new_energy = self._local_wca_energy(occupancy_without_particle, new_sites)
        occupied_destination = self._occupancy_at(occupancy_without_particle, new_sites) > 0
        inf = torch.full_like(new_energy, float("inf"))
        new_energy = torch.where(occupied_destination, inf, new_energy)
        delta_potential = new_energy - old_energy.unsqueeze(1)

        propulsion = self.params.v0 * torch.stack(
            [torch.cos(theta_i), torch.sin(theta_i)], dim=-1
        )
        active_work = torch.sum(
            propulsion.view(B, 1, 2)
            * self.dir_vectors.view(1, 4, 2),
            dim=-1,
        )
        probabilities = self._transition_probabilities_from_delta(active_work, delta_potential)
        return probabilities, new_sites, delta_potential, active_work

    def _particle_probabilities_from_active_work(
        self,
        occupancy_without_particle: torch.Tensor,
        old_sites: torch.Tensor,
        active_work: torch.Tensor,
        batch_indices: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """GPU-friendly proposal calculation with one combined WCA gather."""
        if batch_indices is None:
            batch_indices = torch.arange(
                old_sites.shape[0],
                device=self.device,
            )
        new_sites = (
            old_sites.unsqueeze(1) + self.dir_sites.view(1, 4, 2)
        ) % self.grid_size
        query_sites = torch.cat([old_sites.unsqueeze(1), new_sites], dim=1)
        energies = self._local_wca_energy(
            occupancy_without_particle,
            query_sites,
            batch_indices,
        )
        old_energy = energies[:, 0]
        new_energy = energies[:, 1:]
        occupied_destination = occupancy_without_particle[
            batch_indices.view(-1, 1),
            new_sites[..., 0],
            new_sites[..., 1],
        ] > 0
        new_energy = torch.where(
            occupied_destination,
            torch.full_like(new_energy, float("inf")),
            new_energy,
        )
        delta_potential = new_energy - old_energy.unsqueeze(1)
        probabilities = self._transition_probabilities_from_delta(
            active_work,
            delta_potential,
        )
        return probabilities, new_sites, delta_potential

    def _check_probabilities(self, probabilities: torch.Tensor) -> None:
        p = self.params
        if not torch.isfinite(probabilities).all():
            if p.strict_probabilities:
                raise ValueError(
                    "Invalid lattice-MC probabilities: encountered non-finite "
                    "values. Reduce dt, increase grid spacing, increase Dt, "
                    "or use prefactor='cv'."
                )
            probabilities.nan_to_num_(nan=0.0, posinf=1.0, neginf=0.0)

        prob_sum = probabilities.sum(dim=1)
        max_sum = float(prob_sum.max().detach().cpu())
        if max_sum <= 1.0 + p.probability_tolerance:
            return
        if p.strict_probabilities:
            raise ValueError(
                "Invalid lattice-MC probabilities: total hop probability "
                f"reached {max_sum:.6g} > 1. Reduce dt, increase grid spacing, "
                "increase Dt, or use prefactor='cv'."
            )

        scale = torch.clamp(1.0 / prob_sum.clamp_min(torch.finfo(probabilities.dtype).eps), max=1.0)
        probabilities.mul_(scale.unsqueeze(1))

    # ------------------------------------------------------------------
    # Dynamics
    # ------------------------------------------------------------------

    def _step_inplace_numba(
        self,
        sites: torch.Tensor,
        theta: torch.Tensor,
        occupancy: torch.Tensor,
        *,
        return_ep_map: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Advance one exact sweep with the fused CPU implementation."""
        p = self.params
        B, N, _ = sites.shape
        order, draws = prepare_sweep_random_inputs(
            N,
            B,
            dtype=self.dtype,
            shuffle_particles=p.shuffle_particles,
            legacy_call_order=False,
        )
        active_work = active_work_from_theta_vectorized(
            theta,
            self.dir_vectors,
            p.v0,
        )
        diagnostics = numba_sweep_inplace(
            sites,
            occupancy,
            order,
            draws,
            active_work,
            self._kernel_offsets,
            self._kernel_values,
            dl=p.dl,
            dt=p.dt,
            reservoir_diffusion=p.Dt,
            mobility=p.mobility,
            prefactor=p.prefactor,
            strict_probabilities=p.strict_probabilities,
            probability_tolerance=p.probability_tolerance,
            return_ep_map=return_ep_map,
        )

        # Generate angular noise only after a successful translational sweep,
        # matching the reference RNG order exactly.
        if p.Dr > 0:
            theta_next = (
                theta
                + math.sqrt(2.0 * p.Dr * p.dt) * torch.randn_like(theta)
            ) % (2.0 * math.pi)
        else:
            theta_next = theta
        return sites, theta_next, occupancy, diagnostics

    def _step_inplace_cuda_fused(
        self,
        sites: torch.Tensor,
        theta: torch.Tensor,
        occupancy: torch.Tensor,
        *,
        return_ep_map: bool,
        check_probability_errors: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Advance one random-sequential sweep with the fused CUDA kernel."""
        p = self.params
        order, draws, active_work = prepare_cuda_sweep_inputs(
            theta,
            self.dir_vectors,
            p.v0,
            shuffle_particles=p.shuffle_particles,
        )
        diagnostics = cuda_sweep_inplace(
            sites,
            occupancy,
            order,
            draws,
            active_work,
            self._kernel_offsets,
            self._kernel_values,
            dl=p.dl,
            dt=p.dt,
            reservoir_diffusion=p.Dt,
            mobility=p.mobility,
            prefactor=p.prefactor,
            strict_probabilities=p.strict_probabilities,
            probability_tolerance=p.probability_tolerance,
            return_ep_map=return_ep_map,
            neighbor_linear=self._cuda_neighbor_linear,
            status_check=(
                "sync" if check_probability_errors else "none"
            ),
        )

        # With immediate checking, rotational noise is sampled only after a
        # valid translational sweep. Deferred runs mask the update on failure;
        # that failed interval is discarded at the next save boundary.
        if p.Dr > 0:
            theta_candidate = (
                theta
                + math.sqrt(2.0 * p.Dr * p.dt) * torch.randn_like(theta)
            ) % (2.0 * math.pi)
            if check_probability_errors:
                theta_next = theta_candidate
            else:
                theta_next = torch.where(
                    diagnostics["backend_status"] == 0,
                    theta_candidate,
                    theta,
                )
        else:
            theta_next = theta
        return sites, theta_next, occupancy, diagnostics

    def _step_inplace_torch_dense(
        self,
        sites: torch.Tensor,
        theta: torch.Tensor,
        occupancy: torch.Tensor,
        *,
        return_ep_map: bool = False,
        order: Optional[torch.Tensor] = None,
        draws: Optional[torch.Tensor] = None,
        angular_noise: Optional[torch.Tensor] = None,
        check_probability_errors: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Advance one sweep without CUDA-to-host work inside the particle loop.

        This is the exact random-sequential algorithm used by the reference
        Torch implementation, expressed with fixed-shape gathers, scatters, and
        masks.  In particular, the particle axis is *not* parallelized: each
        accepted hop is committed before the next particle is considered.

        ``order``, ``draws``, and ``angular_noise`` are injectable so optimized
        CUDA implementations can be compared against this path with identical
        stochastic inputs.  Bulk RNG changes only the seeded CUDA stream
        partitioning, not the transition law.
        """
        p = self.params
        B, N, _ = sites.shape
        batch_idx = torch.arange(B, device=self.device)

        if order is None:
            order = (
                torch.randperm(N, device=self.device)
                if p.shuffle_particles
                else torch.arange(N, device=self.device)
            )
        else:
            order = order.to(device=self.device, dtype=torch.long)
            if order.shape != (N,):
                raise ValueError(f"order must have shape ({N},).")

        if draws is None:
            draws = torch.rand(N, B, device=self.device, dtype=self.dtype)
        else:
            draws = draws.to(device=self.device, dtype=self.dtype)
            if draws.shape != (N, B):
                raise ValueError(f"draws must have shape ({N}, {B}).")

        propulsion = p.v0 * torch.stack(
            [torch.cos(theta), torch.sin(theta)],
            dim=-1,
        )
        active_work_all = torch.sum(
            propulsion.unsqueeze(2) * self.dir_vectors.view(1, 1, 4, 2),
            dim=-1,
        )

        # Reorder once, then use a static Python loop index.  This keeps the
        # random sequential dependency while avoiding scalar CUDA indices and
        # the randperm(...).tolist() device synchronization.
        ordered_sites = sites.index_select(1, order).contiguous()
        ordered_active_work = active_work_all.index_select(1, order).contiguous()

        total_ep = torch.zeros(B, device=self.device, dtype=self.dtype)
        active_ep = torch.zeros_like(total_ep)
        wca_ep = torch.zeros_like(total_ep)
        accepted_hops = torch.zeros(B, device=self.device, dtype=torch.long)
        ep_map = (
            torch.zeros(
                B,
                self.grid_size,
                self.grid_size,
                device=self.device,
                dtype=self.dtype,
            )
            if return_ep_map
            else None
        )
        ep_map_flat = ep_map.view(B, -1) if ep_map is not None else None

        # Strict-probability failures are recorded on the device.  Valid
        # trajectories therefore have no host synchronization in this loop.
        # Once a failure is seen, later slots are evaluated but restored
        # without committing a move; one error check is made after the sweep.
        status = torch.zeros((), device=self.device, dtype=torch.int32)
        bad_max_sum = torch.zeros((), device=self.device, dtype=self.dtype)
        tolerance_limit = 1.0 + p.probability_tolerance

        for slot in range(N):
            old_sites = ordered_sites[:, slot].clone()
            occupancy[
                batch_idx,
                old_sites[:, 0],
                old_sites[:, 1],
            ] -= 1
            probabilities, new_sites, delta_v = (
                self._particle_probabilities_from_active_work(
                    occupancy,
                    old_sites,
                    ordered_active_work[:, slot],
                    batch_idx,
                )
            )

            any_nonfinite = ~torch.isfinite(probabilities).all()
            if p.strict_probabilities:
                # Sanitization is used only to keep the deferred-error path
                # numerically well-defined.  A failed slot cannot commit.
                sampling_probabilities = torch.nan_to_num(
                    probabilities,
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )
                probability_sum = sampling_probabilities.sum(dim=1)
                max_sum = probability_sum.max()
                bad_sum = (~any_nonfinite) & (max_sum > tolerance_limit)
                step_status = torch.where(
                    any_nonfinite,
                    torch.ones_like(status),
                    torch.where(
                        bad_sum,
                        torch.full_like(status, 2),
                        torch.zeros_like(status),
                    ),
                )
                was_valid = status == 0
                bad_max_sum = torch.where(
                    was_valid & bad_sum,
                    max_sum,
                    bad_max_sum,
                )
                status = torch.where(was_valid, step_status, status)
                commit_allowed = status == 0
            else:
                sampling_probabilities = torch.nan_to_num(
                    probabilities,
                    nan=0.0,
                    posinf=1.0,
                    neginf=0.0,
                )
                probability_sum = sampling_probabilities.sum(dim=1)
                max_sum = probability_sum.max()
                scale = torch.clamp(
                    1.0
                    / probability_sum.clamp_min(
                        torch.finfo(sampling_probabilities.dtype).eps
                    ),
                    max=1.0,
                )
                sampling_probabilities = torch.where(
                    max_sum > tolerance_limit,
                    sampling_probabilities * scale.unsqueeze(1),
                    sampling_probabilities,
                )
                commit_allowed = torch.ones(
                    (),
                    device=self.device,
                    dtype=torch.bool,
                )

            cumulative = torch.cumsum(sampling_probabilities, dim=1)
            move_idx = (draws[slot].unsqueeze(1) > cumulative).sum(dim=1)
            moved = (move_idx < 4) & commit_allowed
            safe_dir = move_idx.clamp_max(3)

            candidate_sites = new_sites[batch_idx, safe_dir]
            chosen_sites = torch.where(
                moved.unsqueeze(1),
                candidate_sites,
                old_sites,
            )
            chosen_delta_v = delta_v.gather(
                1,
                safe_dir.unsqueeze(1),
            ).squeeze(1)
            chosen_active = ordered_active_work[:, slot].gather(
                1,
                safe_dir.unsqueeze(1),
            ).squeeze(1)

            active_inc = torch.where(
                moved,
                chosen_active / p.Dt,
                torch.zeros_like(chosen_active),
            )
            wca_inc = torch.where(
                moved,
                -(p.mobility * chosen_delta_v) / p.Dt,
                torch.zeros_like(chosen_delta_v),
            )
            total_inc = active_inc + wca_inc

            active_ep += active_inc
            wca_ep += wca_inc
            total_ep += total_inc
            accepted_hops += moved.to(dtype=torch.long)

            if ep_map_flat is not None:
                departing_linear = (
                    old_sites[:, 0] * self.grid_size + old_sites[:, 1]
                )
                ep_map_flat.scatter_add_(
                    1,
                    departing_linear.unsqueeze(1),
                    total_inc.unsqueeze(1),
                )

            ordered_sites[:, slot] = chosen_sites
            occupancy[
                batch_idx,
                chosen_sites[:, 0],
                chosen_sites[:, 1],
            ] += 1

        sites.index_copy_(1, order, ordered_sites)

        if p.strict_probabilities and check_probability_errors:
            status_code = int(status.detach().cpu())
            if status_code == 1:
                raise ValueError(
                    "Invalid lattice-MC probabilities: encountered non-finite "
                    "values. Reduce dt, increase grid spacing, increase Dt, "
                    "or use prefactor='cv'."
                )
            if status_code == 2:
                max_sum_value = float(bad_max_sum.detach().cpu())
                raise ValueError(
                    "Invalid lattice-MC probabilities: total hop probability "
                    f"reached {max_sum_value:.6g} > 1. Reduce dt, increase "
                    "grid spacing, increase Dt, or use prefactor='cv'."
                )

        if p.Dr > 0:
            if angular_noise is None:
                angular_noise = torch.randn_like(theta)
            else:
                angular_noise = angular_noise.to(
                    device=self.device,
                    dtype=self.dtype,
                )
                if angular_noise.shape != theta.shape:
                    raise ValueError(
                        f"angular_noise must have shape {tuple(theta.shape)}."
                    )
            theta_candidate = (
                theta
                + math.sqrt(2.0 * p.Dr * p.dt) * angular_noise
            ) % (2.0 * math.pi)
            if p.strict_probabilities and not check_probability_errors:
                theta_next = torch.where(
                    status == 0,
                    theta_candidate,
                    theta,
                )
            else:
                theta_next = theta_candidate
        else:
            theta_next = theta

        diagnostics: Dict[str, torch.Tensor] = {
            "medium_ep": total_ep,
            "active_medium_ep": active_ep,
            "wca_medium_ep": wca_ep,
            "accepted_hops": accepted_hops,
            "_probability_status": status,
            "_probability_max_sum": bad_max_sum,
        }
        if ep_map is not None:
            diagnostics["medium_ep_map"] = ep_map
        return sites, theta_next, occupancy, diagnostics

    def _step_inplace(
        self,
        sites: torch.Tensor,
        theta: torch.Tensor,
        occupancy: torch.Tensor,
        *,
        return_ep_map: bool = False,
        check_probability_errors: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        if (
            self.backend == "numba"
            and not (torch.is_grad_enabled() and theta.requires_grad)
        ):
            return self._step_inplace_numba(
                sites,
                theta,
                occupancy,
                return_ep_map=return_ep_map,
            )

        if (
            self.backend == "cuda_fused"
            and not (torch.is_grad_enabled() and theta.requires_grad)
        ):
            return self._step_inplace_cuda_fused(
                sites,
                theta,
                occupancy,
                return_ep_map=return_ep_map,
                check_probability_errors=check_probability_errors,
            )

        if (
            self.device.type == "cuda"
            and not (torch.is_grad_enabled() and theta.requires_grad)
        ):
            return self._step_inplace_torch_dense(
                sites,
                theta,
                occupancy,
                return_ep_map=return_ep_map,
                check_probability_errors=check_probability_errors,
            )

        p = self.params
        B, N, _ = sites.shape
        batch_idx = torch.arange(B, device=self.device)

        total_ep = torch.zeros(B, device=self.device, dtype=self.dtype)
        active_ep = torch.zeros_like(total_ep)
        wca_ep = torch.zeros_like(total_ep)
        accepted_hops = torch.zeros(B, device=self.device, dtype=torch.long)
        ep_map = (
            torch.zeros(B, self.grid_size, self.grid_size, device=self.device, dtype=self.dtype)
            if return_ep_map
            else None
        )

        order = (
            torch.randperm(N, device=self.device).tolist()
            if p.shuffle_particles
            else range(N)
        )
        for particle_idx in order:
            old_sites = sites[:, particle_idx].clone()
            theta_i = theta[:, particle_idx]
            self._add_occupancy(occupancy, old_sites, -1)
            probabilities, new_sites, delta_v, active_work = (
                self._particle_probabilities_without_self(
                    occupancy,
                    old_sites,
                    theta_i,
                )
            )
            self._check_probabilities(probabilities)

            cumulative = torch.cumsum(probabilities, dim=1)
            draws = torch.rand(B, device=self.device, dtype=self.dtype)
            move_idx = (draws.unsqueeze(1) > cumulative).sum(dim=1)
            moved = move_idx < 4

            moved_batches = batch_idx[moved]
            moved_dirs = move_idx[moved]
            chosen_sites = old_sites.clone()
            chosen_sites[moved] = new_sites[moved_batches, moved_dirs]

            chosen_delta_v = delta_v[moved_batches, moved_dirs]
            chosen_active = active_work[moved_batches, moved_dirs]
            active_inc = chosen_active / p.Dt
            wca_inc = -(p.mobility * chosen_delta_v) / p.Dt
            total_inc = active_inc + wca_inc

            active_ep[moved] += active_inc
            wca_ep[moved] += wca_inc
            total_ep[moved] += total_inc
            accepted_hops[moved] += 1

            departing_sites = old_sites[moved]
            sites[:, particle_idx] = chosen_sites
            self._add_occupancy(occupancy, chosen_sites, 1)

            if ep_map is not None:
                ep_map[
                    moved_batches,
                    departing_sites[:, 0],
                    departing_sites[:, 1],
                ] += total_inc

        if p.Dr > 0:
            theta_next = theta + math.sqrt(2.0 * p.Dr * p.dt) * torch.randn_like(theta)
            theta_next = theta_next % (2.0 * math.pi)
        else:
            theta_next = theta

        diagnostics: Dict[str, torch.Tensor] = {
            "medium_ep": total_ep,
            "active_medium_ep": active_ep,
            "wca_medium_ep": wca_ep,
            "accepted_hops": accepted_hops,
        }
        if ep_map is not None:
            diagnostics["medium_ep_map"] = ep_map
        return sites, theta_next, occupancy, diagnostics

    def step(
        self,
        sites: torch.Tensor,
        theta: torch.Tensor,
        occupancy: Optional[torch.Tensor] = None,
        *,
        return_diagnostics: bool = False,
        return_ep_map: bool = False,
    ):
        """Advance one lattice-MC step without mutating the input tensors."""
        if sites.dim() == 2:
            sites = sites.unsqueeze(0)
        if theta.dim() == 1:
            theta = theta.unsqueeze(0)
        sites_work = (
            sites.to(device=self.device, dtype=torch.long).clone()
            % self.grid_size
        ).contiguous()
        theta_work = (
            theta.to(device=self.device, dtype=self.dtype)
            .clone()
            .contiguous()
        )
        if occupancy is None:
            occupancy_work = self.occupancy_from_sites(sites_work)
        else:
            occupancy_work = (
                occupancy.to(device=self.device, dtype=torch.long)
                .clone()
                .contiguous()
            )

        sites_next, theta_next, occupancy_next, diagnostics = self._step_inplace(
            sites_work,
            theta_work,
            occupancy_work,
            return_ep_map=return_ep_map,
        )
        if return_diagnostics:
            return sites_next, theta_next, occupancy_next, diagnostics
        return sites_next, theta_next

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
        save_occupancy: bool = True,
        save_exact_medium_ep: bool = True,
        save_ep_maps: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Run a lattice ABP simulation and return CPU tensors.

        Saved arrays are time-major.  ``exact_medium_ep`` has shape
        ``[B, T-1]`` and contains the exact sum of hop-wise Clausius medium EP
        over each saved interval.
        """
        try:
            from tqdm import trange
        except ImportError:  # pragma: no cover
            trange = range

        if save_interval <= 0:
            raise ValueError("save_interval must be positive.")
        if n_steps < 0 or burn_in < 0:
            raise ValueError("n_steps and burn_in must be nonnegative.")

        if initial_state is None:
            sites, theta = self.init_state(B)
        else:
            sites, theta = initial_state
            sites = sites.to(device=self.device, dtype=torch.long) % self.grid_size
            theta = theta.to(device=self.device, dtype=self.dtype)

        occupancy = self.occupancy_from_sites(sites)

        # The CUDA implementations report probability failures through device
        # scalars. During a long simulation, checking those scalars only at a
        # save boundary avoids one forced device synchronization per sweep.
        # On failure the interval's internal state is discarded when the
        # exception is raised; successful dynamics are unchanged.
        defer_cuda_probability_checks = self.device.type == "cuda"
        pending_probability_checks = []

        def record_probability_check(
            step_diagnostics: Dict[str, torch.Tensor],
        ) -> None:
            if not defer_cuda_probability_checks:
                return
            if "backend_status" in step_diagnostics:
                pending_probability_checks.append(
                    (
                        step_diagnostics["backend_status"],
                        step_diagnostics["backend_bad_max_sum"],
                    )
                )
            elif "_probability_status" in step_diagnostics:
                pending_probability_checks.append(
                    (
                        step_diagnostics["_probability_status"],
                        step_diagnostics["_probability_max_sum"],
                    )
                )

        def flush_probability_checks() -> None:
            if not pending_probability_checks:
                return
            status_values = torch.stack(
                [item[0] for item in pending_probability_checks]
            ).detach().cpu()
            failure_indices = torch.nonzero(
                status_values != 0,
                as_tuple=False,
            )
            if failure_indices.numel() == 0:
                pending_probability_checks.clear()
                return

            failure_index = int(failure_indices[0, 0])
            status_code = int(status_values[failure_index])
            bad_sum_tensor = pending_probability_checks[failure_index][1]
            pending_probability_checks.clear()
            if status_code == 1:
                raise ValueError(
                    "Invalid lattice-MC probabilities: encountered non-finite "
                    "values during a CUDA simulation interval. Reduce dt, "
                    "increase grid spacing, increase Dt, or use "
                    "prefactor='cv'."
                )
            if status_code == 2:
                max_sum_value = float(bad_sum_tensor.detach().cpu())
                raise ValueError(
                    "Invalid lattice-MC probabilities: total hop probability "
                    f"reached {max_sum_value:.6g} > 1 during a CUDA "
                    "simulation interval. Reduce dt, increase grid spacing, "
                    "increase Dt, or use prefactor='cv'."
                )
            raise RuntimeError(
                f"Unexpected CUDA probability status {status_code}."
            )

        burn_iter = trange(burn_in, desc="Lattice ABP burn-in", leave=False) if show_progress else range(burn_in)
        with torch.no_grad():
            for burn_idx in burn_iter:
                sites, theta, occupancy, burn_diag = self._step_inplace(
                    sites,
                    theta,
                    occupancy,
                    check_probability_errors=not defer_cuda_probability_checks,
                )
                record_probability_check(burn_diag)
                if (burn_idx + 1) % save_interval == 0:
                    flush_probability_checks()
            flush_probability_checks()

        n_saved = n_steps // save_interval + 1
        sites_traj = torch.empty(n_saved, B, self.N, 2, dtype=torch.long, device=self.device)
        positions = torch.empty(n_saved, B, self.N, 2, dtype=self.dtype, device=self.device)
        angles = torch.empty(n_saved, B, self.N, dtype=self.dtype, device=self.device)
        times = torch.empty(n_saved, dtype=self.dtype, device=self.device)

        occupancy_traj = None
        if save_occupancy:
            occupancy_traj = torch.empty(
                n_saved,
                B,
                self.grid_size,
                self.grid_size,
                dtype=torch.long,
                device=self.device,
            )

        fields = None
        if fieldizer is not None:
            sample = fieldizer.encode(self.sites_to_positions(sites), theta)
            fields = torch.empty(
                n_saved,
                *sample.shape,
                dtype=sample.dtype,
                device=sample.device,
            )

        diag: Dict[str, torch.Tensor] = {}
        if save_diagnostics:
            diag["potential"] = torch.empty(
                n_saved,
                B,
                dtype=self.dtype,
                device=self.device,
            )
        if save_exact_medium_ep:
            diag["accepted_hops"] = torch.empty(
                B,
                n_saved - 1,
                dtype=torch.long,
                device=self.device,
            )

        exact_medium_ep = None
        exact_active_ep = None
        exact_wca_ep = None
        ep_maps = None
        if save_exact_medium_ep:
            exact_medium_ep = torch.empty(B, n_saved - 1, dtype=self.dtype, device=self.device)
            exact_active_ep = torch.empty_like(exact_medium_ep)
            exact_wca_ep = torch.empty_like(exact_medium_ep)
            if save_ep_maps:
                ep_maps = torch.empty(
                    n_saved - 1,
                    B,
                    self.grid_size,
                    self.grid_size,
                    dtype=self.dtype,
                    device=self.device,
                )

        interval_total_ep = torch.zeros(B, device=self.device, dtype=self.dtype)
        interval_active_ep = torch.zeros_like(interval_total_ep)
        interval_wca_ep = torch.zeros_like(interval_total_ep)
        interval_hops = torch.zeros(B, device=self.device, dtype=torch.long)
        interval_ep_map = (
            torch.zeros(B, self.grid_size, self.grid_size, device=self.device, dtype=self.dtype)
            if save_exact_medium_ep and save_ep_maps
            else None
        )

        def save_frame(save_idx: int, step: int) -> None:
            sites_traj[save_idx] = sites
            pos = self.sites_to_positions(sites)
            positions[save_idx] = pos
            angles[save_idx] = theta
            times[save_idx] = step * self.params.dt
            if occupancy_traj is not None:
                occupancy_traj[save_idx] = occupancy
            if fields is not None:
                fields[save_idx] = fieldizer.encode(pos, theta)
            if save_diagnostics:
                diag["potential"][save_idx] = self.potential_energy(sites, occupancy)

        save_idx = 0
        prod_iter = trange(n_steps + 1, desc="Lattice ABP simulate", leave=False) if show_progress else range(n_steps + 1)
        with torch.no_grad():
            for step_idx in prod_iter:
                if step_idx % save_interval == 0:
                    save_frame(save_idx, step_idx)
                    save_idx += 1

                if step_idx < n_steps:
                    sites, theta, occupancy, step_diag = self._step_inplace(
                        sites,
                        theta,
                        occupancy,
                        return_ep_map=save_exact_medium_ep and save_ep_maps,
                        check_probability_errors=(
                            not defer_cuda_probability_checks
                        ),
                    )
                    record_probability_check(step_diag)
                    next_step = step_idx + 1
                    if save_exact_medium_ep:
                        interval_total_ep += step_diag["medium_ep"]
                        interval_active_ep += step_diag["active_medium_ep"]
                        interval_wca_ep += step_diag["wca_medium_ep"]
                        interval_hops += step_diag["accepted_hops"]
                        if interval_ep_map is not None:
                            interval_ep_map += step_diag["medium_ep_map"]

                        if next_step % save_interval == 0:
                            interval_idx = next_step // save_interval - 1
                            if interval_idx < exact_medium_ep.shape[1]:
                                exact_medium_ep[:, interval_idx] = interval_total_ep
                                exact_active_ep[:, interval_idx] = interval_active_ep
                                exact_wca_ep[:, interval_idx] = interval_wca_ep
                                diag["accepted_hops"][:, interval_idx] = interval_hops
                                if ep_maps is not None and interval_ep_map is not None:
                                    ep_maps[interval_idx] = interval_ep_map

                            interval_total_ep.zero_()
                            interval_active_ep.zero_()
                            interval_wca_ep.zero_()
                            interval_hops.zero_()
                            if interval_ep_map is not None:
                                interval_ep_map.zero_()
                    if next_step % save_interval == 0:
                        flush_probability_checks()
            flush_probability_checks()

        out = {
            "sites": sites_traj.cpu(),
            "positions": positions.cpu(),
            "theta": angles.cpu(),
            "times": times.cpu(),
            "params": self.params,
        }
        if occupancy_traj is not None:
            occupancy_cpu = occupancy_traj.cpu()
            out["occupancy"] = occupancy_cpu
            out["O_traj"] = occupancy_cpu
        if fields is not None:
            out["fields"] = fields.cpu()
        for key, value in diag.items():
            out[key] = value.cpu()
        if save_exact_medium_ep:
            out["exact_medium_ep"] = exact_medium_ep.cpu()
            out["exact_active_medium_ep"] = exact_active_ep.cpu()
            out["exact_wca_medium_ep"] = exact_wca_ep.cpu()
            out["exact_medium_ep_rate"] = (exact_medium_ep / (save_interval * self.params.dt)).cpu()
            if ep_maps is not None:
                out["exact_medium_ep_maps"] = ep_maps.cpu()
        return out

    # ------------------------------------------------------------------
    # MIPS helpers
    # ------------------------------------------------------------------

    def coarse_density(self, occupancy: torch.Tensor, box: int = 8) -> torch.Tensor:
        """Return periodic box-averaged occupancy for MIPS diagnostics."""
        if occupancy.dim() == 2:
            occupancy = occupancy.unsqueeze(0)
        occ = occupancy.to(device=self.device, dtype=self.dtype)
        box = max(1, min(int(box), occ.shape[-2], occ.shape[-1]))
        out = torch.zeros_like(occ)
        for dx in range(box):
            rolled_x = torch.roll(occ, shifts=-dx, dims=-2)
            for dy in range(box):
                out += torch.roll(rolled_x, shifts=-dy, dims=-1)
        return out / float(box * box)

    def mips_summary(
        self,
        occupancy: torch.Tensor,
        coarse_box: int = 8,
        *,
        include_coarse: bool = False,
    ) -> Dict[str, float]:
        """Return compact MIPS indicators for the first ensemble in a snapshot."""
        if occupancy.dim() == 3:
            occ = occupancy[0]
        elif occupancy.dim() == 2:
            occ = occupancy
        else:
            raise ValueError("occupancy must have shape [B, G, G] or [G, G].")

        occ_cpu = occ.detach().cpu().numpy().astype(np.float64)
        rho = float(occ_cpu.mean())

        centered = occ_cpu - rho
        spectrum = np.abs(np.fft.fftshift(np.fft.fft2(centered))) ** 2
        k = np.fft.fftshift(np.fft.fftfreq(occ_cpu.shape[0]))
        ky, kx = np.meshgrid(k, k, indexing="ij")
        radius = np.sqrt(kx * kx + ky * ky)
        low = (radius > 0) & (radius <= 0.12)
        mid = (radius > 0.12) & (radius <= 0.35)
        low_mean = float(spectrum[low].mean()) if np.any(low) else 0.0
        mid_mean = float(spectrum[mid].mean()) if np.any(mid) else 0.0

        summary = {
            "site_density": rho,
            "packing_fraction": self.params.phi,
            "largest_site_cluster_fraction": self._largest_cluster_fraction_np(occ_cpu),
            "low_k_ratio": low_mean / (mid_mean + 1.0e-12),
        }
        if not include_coarse:
            return summary

        coarse = self.coarse_density(
            torch.as_tensor(occ_cpu, device=self.device),
            box=coarse_box,
        )
        coarse_np = coarse.detach().cpu().numpy()
        random_std = math.sqrt(
            max(rho * (1.0 - rho), 0.0)
            / float(coarse_box * coarse_box)
        )
        coarse_2d = coarse_np[0] if coarse_np.ndim == 3 else coarse_np
        dense_threshold = rho + random_std
        dense_domains = coarse_2d >= dense_threshold

        summary.update({
            "coarse_std": float(coarse_2d.std()),
            "coarse_std_random": random_std,
            "coarse_std_ratio": float(coarse_2d.std() / (random_std + 1.0e-12)),
            "coarse_q10": float(np.quantile(coarse_2d, 0.10)),
            "coarse_q90": float(np.quantile(coarse_2d, 0.90)),
            "dense_area_fraction": float(dense_domains.mean()),
            "largest_dense_domain_fraction": self._largest_cluster_fraction_np(dense_domains),
        })
        return summary

    def particle_cluster_fraction(
        self,
        sites: torch.Tensor,
        *,
        cluster_distance: Optional[float] = None,
        ensemble_idx: int = 0,
    ) -> float:
        """Return largest particle cluster fraction using physical distances."""
        if sites.dim() == 2:
            sites = sites.unsqueeze(0)
        sites = sites.to(device=self.device, dtype=torch.long) % self.grid_size
        if cluster_distance is None:
            cluster_distance = 1.35 * self.params.sigma

        pos = self.sites_to_positions(sites[ensemble_idx : ensemble_idx + 1])[0]
        delta = pos[:, None, :] - pos[None, :, :]
        delta = self.minimum_image(delta)
        r2 = torch.sum(delta * delta, dim=-1)
        adjacency = (r2 > 0) & (r2 <= cluster_distance * cluster_distance)
        adjacency_np = adjacency.detach().cpu().numpy()

        N = adjacency_np.shape[0]
        visited = np.zeros(N, dtype=bool)
        largest = 0
        for start in range(N):
            if visited[start]:
                continue
            size = 0
            queue = deque([start])
            visited[start] = True
            while queue:
                idx = queue.popleft()
                size += 1
                neighbors = np.flatnonzero(adjacency_np[idx] & ~visited)
                visited[neighbors] = True
                queue.extend(int(n) for n in neighbors)
            largest = max(largest, size)
        return largest / float(max(N, 1))

    def mips_summary_from_sites(
        self,
        sites: torch.Tensor,
        *,
        coarse_box: int = 8,
        include_coarse: bool = False,
        cluster_distance: Optional[float] = None,
        ensemble_idx: int = 0,
    ) -> Dict[str, float]:
        """Return MIPS indicators from particle sites for one ensemble."""
        occupancy = self.occupancy_from_sites(sites)
        summary = self.mips_summary(
            occupancy[ensemble_idx],
            coarse_box=coarse_box,
            include_coarse=include_coarse,
        )
        summary["particle_largest_cluster_fraction"] = self.particle_cluster_fraction(
            sites,
            cluster_distance=cluster_distance,
            ensemble_idx=ensemble_idx,
        )
        return summary

    @staticmethod
    def _largest_cluster_fraction_np(occupancy: np.ndarray) -> float:
        occ = np.asarray(occupancy).astype(bool)
        total = int(occ.sum())
        if total == 0:
            return 0.0

        size_x, size_y = occ.shape
        visited = np.zeros_like(occ, dtype=bool)
        largest = 0
        for x0 in range(size_x):
            for y0 in range(size_y):
                if not occ[x0, y0] or visited[x0, y0]:
                    continue
                cluster_size = 0
                queue = deque([(x0, y0)])
                visited[x0, y0] = True
                while queue:
                    x, y = queue.popleft()
                    cluster_size += 1
                    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nx = (x + dx) % size_x
                        ny = (y + dy) % size_y
                        if occ[nx, ny] and not visited[nx, ny]:
                            visited[nx, ny] = True
                            queue.append((nx, ny))
                largest = max(largest, cluster_size)
        return largest / float(total)

