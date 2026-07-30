"""Optional fused CUDA backend for thermodynamic lattice-ABP sweeps.

This module deliberately has no import-time compilation side effect.  The
extension is built the first time :func:`load_cuda_backend` or
:func:`cuda_sweep_inplace` is called, and the regular PyTorch implementation
can remain a reliable fallback when a CUDA compiler is not installed.

The CUDA kernel fuses one complete random-sequential translational sweep:

* particles are visited in the caller-provided ``order``;
* a particle is removed before its old/new WCA energies are evaluated;
* the four hop probabilities are checked before that ensemble is committed;
* accepted-hop entropy production and the optional departure-site map are
  accumulated in the same order as the reference implementation.

Only the five independent WCA stencil reductions for the current particle are
parallelized.  There is no tau leaping, simultaneous particle update, enlarged
time step, or truncated interaction beyond the caller-provided exact stencil.
Floating-point reductions use a deterministic warp tree, so their last bits
can differ from ATen's or Numba's reduction order even though the simulated
transition rule is unchanged.
"""

from __future__ import annotations

import hashlib
import os
import threading
from pathlib import Path
from types import ModuleType
from typing import Dict, Literal, Optional

import torch

try:
    from torch.utils.cpp_extension import CUDA_HOME, load
except Exception as exc:  # pragma: no cover - depends on the torch install
    CUDA_HOME = None
    load = None
    _CPP_EXTENSION_IMPORT_ERROR: Optional[BaseException] = exc
else:
    _CPP_EXTENSION_IMPORT_ERROR = None


class CudaBackendUnavailable(ImportError):
    """Raised when the optional fused CUDA extension cannot be used."""


_SOURCE_DIR = Path(__file__).resolve().parent / "csrc"
_SOURCES = (
    _SOURCE_DIR / "cuda_sweep.cpp",
    _SOURCE_DIR / "cuda_sweep_kernel.cu",
)
_LOAD_LOCK = threading.Lock()
_EXTENSION: Optional[ModuleType] = None
_LOAD_ERROR: Optional[BaseException] = None


def cuda_backend_buildable() -> bool:
    """Return whether this process appears able to JIT-build CUDA code.

    This is a cheap capability check; it does not compile or load the
    extension.  A ``True`` result therefore does not guarantee that the host
    C++ compiler, Ninja, filesystem permissions, and CUDA architecture are all
    configured correctly.
    """

    return bool(
        torch.cuda.is_available()
        and load is not None
        and CUDA_HOME is not None
        and all(path.is_file() for path in _SOURCES)
        and os.environ.get("LATTICE_ABP_DISABLE_CUDA_EXTENSION", "0") != "1"
    )


def cuda_backend_load_error() -> Optional[BaseException]:
    """Return the cached JIT-load exception, if a load has been attempted."""

    return _LOAD_ERROR


def _extension_name() -> str:
    digest = hashlib.sha256()
    for source in _SOURCES:
        digest.update(source.read_bytes())
    digest.update(torch.__version__.encode("utf-8"))
    digest.update(str(torch.version.cuda).encode("utf-8"))
    digest.update(b"cxx17;-O3;fmad=false;abi=1")
    return f"lattice_abp_cuda_{digest.hexdigest()[:12]}"


def _unavailable_message() -> str:
    if os.environ.get("LATTICE_ABP_DISABLE_CUDA_EXTENSION", "0") == "1":
        return (
            "The fused lattice-ABP CUDA extension is disabled by "
            "LATTICE_ABP_DISABLE_CUDA_EXTENSION=1."
        )
    if not torch.cuda.is_available():
        return "A CUDA-enabled PyTorch runtime and visible CUDA GPU are required."
    if load is None:
        return (
            "torch.utils.cpp_extension is unavailable in this PyTorch install."
        )
    if CUDA_HOME is None:
        return (
            "A CUDA toolkit with nvcc is required to JIT-build the fused "
            "lattice-ABP backend. CUDA_HOME was not detected."
        )
    missing = [str(path) for path in _SOURCES if not path.is_file()]
    if missing:
        return "Missing CUDA backend source file(s): " + ", ".join(missing)
    return "The fused lattice-ABP CUDA extension is unavailable."


def load_cuda_backend(*, verbose: Optional[bool] = None) -> ModuleType:
    """Build (once, in PyTorch's extension cache) and load the CUDA module.

    Fast math is disabled, including fused multiply-add contraction. PyTorch
    chooses the architecture from ``TORCH_CUDA_ARCH_LIST`` when it is set,
    otherwise from the visible GPU. For an NVIDIA L40S the native target is
    compute capability 8.9.
    """

    global _EXTENSION, _LOAD_ERROR
    if _EXTENSION is not None:
        return _EXTENSION
    if _LOAD_ERROR is not None:
        raise CudaBackendUnavailable(
            "The fused lattice-ABP CUDA extension failed to load previously."
        ) from _LOAD_ERROR
    if not cuda_backend_buildable():
        error = CudaBackendUnavailable(_unavailable_message())
        _LOAD_ERROR = error
        raise error

    with _LOAD_LOCK:
        if _EXTENSION is not None:
            return _EXTENSION
        if _LOAD_ERROR is not None:
            raise CudaBackendUnavailable(
                "The fused lattice-ABP CUDA extension failed to load previously."
            ) from _LOAD_ERROR

        if verbose is None:
            verbose = (
                os.environ.get("LATTICE_ABP_CUDA_BUILD_VERBOSE", "0") == "1"
            )
        if os.name == "nt":
            extra_cflags = ["/O2", "/std:c++17"]
        else:
            extra_cflags = ["-O3", "-std=c++17"]

        try:
            assert load is not None  # narrowed by cuda_backend_buildable()
            _EXTENSION = load(
                name=_extension_name(),
                sources=[str(path) for path in _SOURCES],
                extra_cflags=extra_cflags,
                extra_cuda_cflags=[
                    "-O3",
                    "--std=c++17",
                    "--fmad=false",
                ],
                with_cuda=True,
                is_python_module=True,
                verbose=bool(verbose),
            )
        except Exception as exc:  # pragma: no cover - toolchain dependent
            _LOAD_ERROR = exc
            raise CudaBackendUnavailable(
                "Failed to build/load the fused lattice-ABP CUDA extension. "
                "Verify that nvcc, a supported host compiler, Ninja, and the "
                "PyTorch CUDA runtime are compatible."
            ) from exc
        return _EXTENSION


def _require_cuda_contiguous(tensor: torch.Tensor, name: str) -> None:
    if tensor.device.type != "cuda":
        raise ValueError(f"{name} must be a CUDA tensor.")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous.")


def _validate_inputs(
    sites: torch.Tensor,
    occupancy: torch.Tensor,
    order: torch.Tensor,
    draws: torch.Tensor,
    active_work: torch.Tensor,
    kernel_offsets: torch.Tensor,
    kernel_values: torch.Tensor,
    *,
    dl: float,
    dt: float,
    reservoir_diffusion: float,
    mobility: float,
    prefactor: str,
    probability_tolerance: float,
) -> None:
    tensors = {
        "sites": sites,
        "occupancy": occupancy,
        "order": order,
        "draws": draws,
        "active_work": active_work,
        "kernel_offsets": kernel_offsets,
        "kernel_values": kernel_values,
    }
    for name, tensor in tensors.items():
        _require_cuda_contiguous(tensor, name)
        if tensor.device != sites.device:
            raise ValueError("All CUDA backend tensors must be on one device.")

    if sites.dtype != torch.long or sites.ndim != 3 or sites.shape[-1] != 2:
        raise ValueError("sites must be contiguous int64 with shape [B, N, 2].")
    if (
        occupancy.dtype != torch.long
        or occupancy.ndim != 3
        or occupancy.shape[1] != occupancy.shape[2]
    ):
        raise ValueError(
            "occupancy must be contiguous int64 with shape [B, G, G]."
        )
    batch_size, particle_count, _ = sites.shape
    if occupancy.shape[0] != batch_size:
        raise ValueError("sites and occupancy batch dimensions must match.")
    if order.dtype != torch.long or order.shape != (particle_count,):
        raise ValueError("order must be int64 with shape [N].")
    if draws.shape != (particle_count, batch_size):
        raise ValueError("draws must have shape [N, B].")
    if active_work.shape != (batch_size, particle_count, 4):
        raise ValueError("active_work must have shape [B, N, 4].")
    if (
        kernel_offsets.dtype != torch.long
        or kernel_offsets.ndim != 2
        or kernel_offsets.shape[1] != 2
    ):
        raise ValueError("kernel_offsets must be int64 with shape [K, 2].")
    if (
        kernel_values.ndim != 1
        or kernel_values.shape[0] != kernel_offsets.shape[0]
    ):
        raise ValueError("kernel_values must have shape [K].")
    if kernel_values.dtype not in (torch.float32, torch.float64):
        raise ValueError("kernel_values must have dtype float32 or float64.")
    if draws.dtype != kernel_values.dtype:
        raise ValueError("draws and kernel_values must have the same dtype.")
    if active_work.dtype != kernel_values.dtype:
        raise ValueError(
            "active_work and kernel_values must have the same dtype."
        )
    if dl <= 0 or dt <= 0 or reservoir_diffusion <= 0:
        raise ValueError("dl, dt, and reservoir_diffusion must be positive.")
    if mobility <= 0:
        raise ValueError("mobility must be positive.")
    if prefactor.lower() not in {"cv", "c0"}:
        raise ValueError("prefactor must be 'cv' or 'c0'.")
    if probability_tolerance < 0:
        raise ValueError("probability_tolerance must be nonnegative.")


def prepare_cuda_sweep_inputs(
    theta: torch.Tensor,
    dir_vectors: torch.Tensor,
    v0: float,
    *,
    shuffle_particles: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate a sweep's order, uniforms, and active work on the GPU.

    Random values are generated in bulk to avoid ``N`` Python/CUDA dispatches.
    This preserves the exact random-sequential stochastic rule, but a seeded
    trajectory is not promised to be bitwise identical to one produced by
    ``N`` separate ``torch.rand(B)`` calls.
    """

    if theta.device.type != "cuda" or dir_vectors.device != theta.device:
        raise ValueError("theta and dir_vectors must be on the same CUDA device.")
    if not theta.is_contiguous() or not dir_vectors.is_contiguous():
        raise ValueError("theta and dir_vectors must be contiguous.")
    if theta.ndim != 2:
        raise ValueError("theta must have shape [B, N].")
    if dir_vectors.shape != (4, 2):
        raise ValueError("dir_vectors must have shape [4, 2].")
    if theta.dtype not in (torch.float32, torch.float64):
        raise ValueError("theta must have dtype float32 or float64.")
    if dir_vectors.dtype != theta.dtype:
        raise ValueError("theta and dir_vectors must have the same dtype.")

    batch_size, particle_count = theta.shape
    if shuffle_particles:
        order = torch.randperm(particle_count, device=theta.device)
    else:
        order = torch.arange(particle_count, device=theta.device)
    draws = torch.rand(
        particle_count,
        batch_size,
        dtype=theta.dtype,
        device=theta.device,
    )
    propulsion = v0 * torch.stack(
        (torch.cos(theta), torch.sin(theta)),
        dim=-1,
    )
    active_work = torch.sum(
        propulsion.unsqueeze(2) * dir_vectors.view(1, 1, 4, 2),
        dim=-1,
    ).contiguous()
    return order, draws, active_work


def build_neighbor_linear_lookup(
    grid_size: int,
    kernel_offsets: torch.Tensor,
) -> torch.Tensor:
    """Precompute exact periodic WCA neighbor indices for the fused kernel.

    The result has shape ``[G * G, K]`` and dtype ``int32``. Supplying it to
    :func:`cuda_sweep_inplace` removes two integer modulo operations from every
    stencil entry. At ``G=128, K=68`` it occupies about 4.25 MiB, which is
    negligible on a 40 GB L40S and can be reused for the full simulation.
    """

    if grid_size <= 0:
        raise ValueError("grid_size must be positive.")
    _require_cuda_contiguous(kernel_offsets, "kernel_offsets")
    if (
        kernel_offsets.dtype != torch.long
        or kernel_offsets.ndim != 2
        or kernel_offsets.shape[1] != 2
    ):
        raise ValueError("kernel_offsets must be int64 with shape [K, 2].")
    site_count = grid_size * grid_size
    if site_count > torch.iinfo(torch.int32).max:
        raise ValueError("grid_size is too large for an int32 neighbor lookup.")

    linear = torch.arange(
        site_count,
        dtype=torch.long,
        device=kernel_offsets.device,
    )
    center_x = torch.div(linear, grid_size, rounding_mode="floor").unsqueeze(1)
    center_y = torch.remainder(linear, grid_size).unsqueeze(1)
    neighbor_x = torch.remainder(
        center_x + kernel_offsets[:, 0].view(1, -1),
        grid_size,
    )
    neighbor_y = torch.remainder(
        center_y + kernel_offsets[:, 1].view(1, -1),
        grid_size,
    )
    return (neighbor_x * grid_size + neighbor_y).to(torch.int32).contiguous()


def _raise_for_status(
    status: torch.Tensor,
    failed_order_index: torch.Tensor,
    bad_max_sum: torch.Tensor,
) -> None:
    status_code = int(status.item())
    if status_code == 0:
        return
    order_index = int(failed_order_index.item())
    if status_code == 1:
        raise ValueError(
            "Invalid lattice-MC probabilities: encountered non-finite values "
            f"at sweep order index {order_index}. Reduce dt, increase grid "
            "spacing, increase Dt, or use prefactor='cv'."
        )
    if status_code == 2:
        max_sum = float(bad_max_sum.item())
        raise ValueError(
            "Invalid lattice-MC probabilities: total hop probability reached "
            f"{max_sum:.6g} > 1 at sweep order index {order_index}. Reduce dt, "
            "increase grid spacing, increase Dt, or use prefactor='cv'."
        )
    raise RuntimeError(
        f"Unexpected fused CUDA sweep status {status_code} at order index "
        f"{order_index}."
    )


def cuda_sweep_inplace(
    sites: torch.Tensor,
    occupancy: torch.Tensor,
    order: torch.Tensor,
    draws: torch.Tensor,
    active_work: torch.Tensor,
    kernel_offsets: torch.Tensor,
    kernel_values: torch.Tensor,
    *,
    dl: float,
    dt: float,
    reservoir_diffusion: float,
    mobility: float,
    prefactor: str,
    strict_probabilities: bool,
    probability_tolerance: float,
    return_ep_map: bool = False,
    neighbor_linear: Optional[torch.Tensor] = None,
    status_check: Literal["sync", "none"] = "sync",
) -> Dict[str, torch.Tensor]:
    """Run one exact random-sequential translational sweep on CUDA.

    ``sites`` and ``occupancy`` are mutated in place.  Angular Brownian motion
    remains the caller's responsibility and should be sampled only after this
    function succeeds, matching the reference RNG call order.

    This is a low-level trusted-input API. To avoid a device synchronization
    on every sweep, it checks tensor metadata but assumes ``order`` is a
    permutation of ``0..N-1`` and every site coordinate lies in ``0..G-1``.
    :class:`ThermodynamicLatticeABP` maintains those invariants. Direct callers
    must validate them before entering a long CUDA loop.

    ``status_check='sync'`` (the default) performs one device-to-host scalar
    check per sweep and raises the same two classes of probability error as the
    CPU backend.  ``'none'`` removes that synchronization for a prevalidated
    high-throughput run; in that mode callers must inspect
    ``diagnostics['backend_status']`` before relying on the result.  A nonzero
    status means the sweep failed and its mutated tensors must be discarded.
    Strict valid runs use one block per ensemble for L40S throughput. Therefore
    an invalid strict run is not transactional across ensembles: another block
    can have advanced before the shared failure status is observed.
    """

    if status_check not in {"sync", "none"}:
        raise ValueError("status_check must be 'sync' or 'none'.")
    _validate_inputs(
        sites,
        occupancy,
        order,
        draws,
        active_work,
        kernel_offsets,
        kernel_values,
        dl=dl,
        dt=dt,
        reservoir_diffusion=reservoir_diffusion,
        mobility=mobility,
        prefactor=prefactor,
        probability_tolerance=probability_tolerance,
    )
    prefactor_code = 0 if prefactor.lower() == "cv" else 1
    if neighbor_linear is None:
        neighbor_linear = torch.empty(
            0,
            dtype=torch.int32,
            device=sites.device,
        )
    else:
        _require_cuda_contiguous(neighbor_linear, "neighbor_linear")
        if neighbor_linear.device != sites.device:
            raise ValueError(
                "neighbor_linear must be on the same CUDA device as sites."
            )
        expected_shape = (
            occupancy.shape[1] * occupancy.shape[2],
            kernel_offsets.shape[0],
        )
        if (
            neighbor_linear.dtype != torch.int32
            or neighbor_linear.shape != expected_shape
        ):
            raise ValueError(
                "neighbor_linear must be contiguous int32 with shape "
                f"{expected_shape}."
            )
    extension = load_cuda_backend()
    outputs = extension.sweep_cuda(
        sites,
        occupancy,
        order,
        draws,
        active_work,
        kernel_offsets,
        kernel_values,
        neighbor_linear,
        float(dt * reservoir_diffusion / (dl * dl)),
        float(reservoir_diffusion),
        float(mobility),
        prefactor_code,
        bool(strict_probabilities),
        float(probability_tolerance),
        bool(return_ep_map),
    )
    (
        total_ep,
        active_ep,
        wca_ep,
        accepted_hops,
        ep_map,
        status,
        failed_order_index,
        bad_max_sum,
    ) = outputs

    if status_check == "sync":
        _raise_for_status(status, failed_order_index, bad_max_sum)

    diagnostics: Dict[str, torch.Tensor] = {
        "medium_ep": total_ep,
        "active_medium_ep": active_ep,
        "wca_medium_ep": wca_ep,
        "accepted_hops": accepted_hops,
        "backend_status": status,
        "backend_failed_order_index": failed_order_index,
        "backend_bad_max_sum": bad_max_sum,
    }
    if return_ep_map:
        diagnostics["medium_ep_map"] = ep_map
    return diagnostics


__all__ = [
    "build_neighbor_linear_lookup",
    "CudaBackendUnavailable",
    "cuda_backend_buildable",
    "cuda_backend_load_error",
    "cuda_sweep_inplace",
    "load_cuda_backend",
    "prepare_cuda_sweep_inputs",
]
