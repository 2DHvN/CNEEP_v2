"""Optional fused CPU backend for thermodynamic lattice-ABP sweeps.

The regular PyTorch implementation deliberately updates particles in random
sequential order.  On CPU, dispatching the many small tensor operations needed
for every particle dominates the actual WCA calculation.  This module fuses
one translational sweep into a Numba loop while keeping the same update order.

The backend is intentionally low-level and is selected by :mod:`core` for CPU
simulations when ``backend="auto"`` is requested. Callers generate ``order``
and ``draws`` with ``prepare_sweep_random_inputs`` and perform the angular
Brownian update *after* a successful call. This preserves the PyTorch random
sequence used by
``ThermodynamicLatticeABP._step_inplace`` on valid runs:

1. one ``torch.randperm`` (when shuffling is enabled);
2. the same ``N * B`` uniform values as the particlewise reference calls;
3. one ``torch.randn_like(theta)`` call after the translational sweep.

Only CPU, contiguous tensors are accepted.  ``sites`` and ``occupancy`` are
updated in place through zero-copy NumPy views.

The transition rule and WCA stencil are unchanged. As with any switch between
numeric kernels, the compiled scalar reduction and ATen reduction can differ
in their last floating-point bits for specially constructed occupancies; no
interaction, event, or time-step approximation is introduced.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import torch

try:
    from numba import njit
except Exception as exc:  # pragma: no cover - depends on optional dependency
    njit = None
    _NUMBA_IMPORT_ERROR: Optional[BaseException] = exc
else:
    _NUMBA_IMPORT_ERROR = None


NUMBA_AVAILABLE = njit is not None

_DIRECTION_X = np.asarray([1, -1, 0, 0], dtype=np.int64)
_DIRECTION_Y = np.asarray([0, 0, 1, -1], dtype=np.int64)


def _make_sweep_kernel(float_type):
    """Build a dtype-specialized fused sweep kernel."""

    if njit is None:  # pragma: no cover - guarded by module initialization
        return None

    zero = float_type(0.0)
    one = float_type(1.0)
    two = float_type(2.0)
    three = float_type(3.0)
    fifty = float_type(50.0)
    small_cutoff = float_type(1.0e-5)
    positive_inf = float_type(np.inf)
    dtype_eps = float_type(np.finfo(float_type).eps)

    @njit(cache=True, error_model="numpy")
    def sweep_kernel(
        sites,
        occupancy,
        order,
        draws,
        active_work,
        kernel_offsets,
        kernel_values,
        base,
        reservoir_diffusion,
        mobility,
        prefactor_code,
        strict_probabilities,
        probability_tolerance,
        record_ep_map,
        probabilities,
        proposed_sites,
        delta_potential,
        total_ep,
        active_ep,
        wca_ep,
        accepted_hops,
        ep_map,
    ):
        batch_size = sites.shape[0]
        particle_count = sites.shape[1]
        grid_size = occupancy.shape[1]
        kernel_size = kernel_offsets.shape[0]

        # status: 0 = success, 1 = non-finite probability, 2 = sum > 1.
        status = 0
        failed_order_index = -1
        bad_max_sum = zero

        for order_index in range(particle_count):
            particle_index = order[order_index]

            # Match the PyTorch path: remove this particle from every ensemble
            # before calculating any of its old/new local energies.
            for batch_index in range(batch_size):
                old_x = sites[batch_index, particle_index, 0]
                old_y = sites[batch_index, particle_index, 1]
                occupancy[batch_index, old_x, old_y] -= 1

            for batch_index in range(batch_size):
                old_x = sites[batch_index, particle_index, 0]
                old_y = sites[batch_index, particle_index, 1]

                old_energy = zero
                for kernel_index in range(kernel_size):
                    neighbor_x = (
                        old_x + kernel_offsets[kernel_index, 0]
                    ) % grid_size
                    neighbor_y = (
                        old_y + kernel_offsets[kernel_index, 1]
                    ) % grid_size
                    count = float_type(
                        occupancy[batch_index, neighbor_x, neighbor_y]
                    )
                    old_energy = float_type(
                        old_energy + count * kernel_values[kernel_index]
                    )

                for direction in range(4):
                    new_x = (old_x + _DIRECTION_X[direction]) % grid_size
                    new_y = (old_y + _DIRECTION_Y[direction]) % grid_size
                    proposed_sites[batch_index, direction, 0] = new_x
                    proposed_sites[batch_index, direction, 1] = new_y

                    if occupancy[batch_index, new_x, new_y] > 0:
                        # The PyTorch path replaces the destination energy by
                        # +inf after evaluating it.  Skipping that discarded
                        # stencil calculation does not change any output.
                        energy_change = positive_inf
                    else:
                        new_energy = zero
                        for kernel_index in range(kernel_size):
                            neighbor_x = (
                                new_x + kernel_offsets[kernel_index, 0]
                            ) % grid_size
                            neighbor_y = (
                                new_y + kernel_offsets[kernel_index, 1]
                            ) % grid_size
                            count = float_type(
                                occupancy[
                                    batch_index,
                                    neighbor_x,
                                    neighbor_y,
                                ]
                            )
                            new_energy = float_type(
                                new_energy
                                + count * kernel_values[kernel_index]
                            )
                        energy_change = float_type(new_energy - old_energy)

                    delta_potential[batch_index, direction] = energy_change
                    active = active_work[
                        batch_index,
                        particle_index,
                        direction,
                    ]
                    affinity = float_type(
                        active - float_type(mobility * energy_change)
                    )
                    x = float_type(
                        affinity / float_type(two * reservoir_diffusion)
                    )

                    if not np.isfinite(x) or not np.isfinite(energy_change):
                        probability = zero
                    elif prefactor_code == 1:  # c0
                        probability = float_type(base * float_type(np.exp(x)))
                    else:  # cv
                        if abs(x) < small_cutoff:
                            x_squared_over_three = float_type(
                                float_type(x * x) / three
                            )
                            factor = float_type(
                                float_type(one + x) + x_squared_over_three
                            )
                        elif x > fifty:
                            factor = float_type(two * x)
                        elif x < -fifty:
                            factor = float_type(
                                float_type(-two * x)
                                * float_type(np.exp(float_type(two * x)))
                            )
                        else:
                            factor = float_type(
                                float_type(two * x)
                                / float_type(
                                    -np.expm1(float_type(-two * x))
                                )
                            )
                        if factor < zero:
                            factor = zero
                        probability = float_type(base * factor)

                    if probability < zero:
                        probability = zero
                    probabilities[batch_index, direction] = probability

            # Reproduce _check_probabilities before sampling or committing the
            # current particle.  All ensembles are checked together.
            any_nonfinite = False
            for batch_index in range(batch_size):
                for direction in range(4):
                    if not np.isfinite(
                        probabilities[batch_index, direction]
                    ):
                        any_nonfinite = True

            if any_nonfinite:
                if strict_probabilities:
                    status = 1
                    failed_order_index = order_index
                    return (
                        status,
                        failed_order_index,
                        bad_max_sum,
                    )
                for batch_index in range(batch_size):
                    for direction in range(4):
                        value = probabilities[batch_index, direction]
                        if np.isnan(value) or value == -positive_inf:
                            probabilities[batch_index, direction] = zero
                        elif value == positive_inf:
                            probabilities[batch_index, direction] = one

            max_probability_sum = zero
            for batch_index in range(batch_size):
                probability_sum = zero
                for direction in range(4):
                    probability_sum = float_type(
                        probability_sum
                        + probabilities[batch_index, direction]
                    )
                if (
                    batch_index == 0
                    or probability_sum > max_probability_sum
                ):
                    max_probability_sum = probability_sum

            if max_probability_sum > 1.0 + probability_tolerance:
                if strict_probabilities:
                    status = 2
                    failed_order_index = order_index
                    bad_max_sum = max_probability_sum
                    return (
                        status,
                        failed_order_index,
                        bad_max_sum,
                    )

                for batch_index in range(batch_size):
                    probability_sum = zero
                    for direction in range(4):
                        probability_sum = float_type(
                            probability_sum
                            + probabilities[batch_index, direction]
                        )
                    denominator = probability_sum
                    if denominator < dtype_eps:
                        denominator = dtype_eps
                    scale = float_type(one / denominator)
                    if scale > one:
                        scale = one
                    for direction in range(4):
                        probabilities[batch_index, direction] = float_type(
                            probabilities[batch_index, direction] * scale
                        )

            # Sample and commit independently across ensembles only after the
            # joint strict-probability check has passed.
            for batch_index in range(batch_size):
                cumulative = zero
                selected_direction = 4
                draw = draws[order_index, batch_index]
                for direction in range(4):
                    cumulative = float_type(
                        cumulative
                        + probabilities[batch_index, direction]
                    )
                    if (
                        selected_direction == 4
                        and draw <= cumulative
                    ):
                        selected_direction = direction

                old_x = sites[batch_index, particle_index, 0]
                old_y = sites[batch_index, particle_index, 1]
                chosen_x = old_x
                chosen_y = old_y

                if selected_direction < 4:
                    chosen_x = proposed_sites[
                        batch_index,
                        selected_direction,
                        0,
                    ]
                    chosen_y = proposed_sites[
                        batch_index,
                        selected_direction,
                        1,
                    ]
                    chosen_delta = delta_potential[
                        batch_index,
                        selected_direction,
                    ]
                    chosen_active = active_work[
                        batch_index,
                        particle_index,
                        selected_direction,
                    ]
                    active_increment = float_type(
                        chosen_active / reservoir_diffusion
                    )
                    wca_increment = float_type(
                        -float_type(mobility * chosen_delta)
                        / reservoir_diffusion
                    )
                    total_increment = float_type(
                        active_increment + wca_increment
                    )

                    active_ep[batch_index] = float_type(
                        active_ep[batch_index] + active_increment
                    )
                    wca_ep[batch_index] = float_type(
                        wca_ep[batch_index] + wca_increment
                    )
                    total_ep[batch_index] = float_type(
                        total_ep[batch_index] + total_increment
                    )
                    accepted_hops[batch_index] += 1
                    if record_ep_map:
                        ep_map[batch_index, old_x, old_y] = float_type(
                            ep_map[batch_index, old_x, old_y]
                            + total_increment
                        )

                sites[batch_index, particle_index, 0] = chosen_x
                sites[batch_index, particle_index, 1] = chosen_y
                occupancy[batch_index, chosen_x, chosen_y] += 1

        return status, failed_order_index, bad_max_sum

    return sweep_kernel


if NUMBA_AVAILABLE:
    _SWEEP_FLOAT32 = _make_sweep_kernel(np.float32)
    _SWEEP_FLOAT64 = _make_sweep_kernel(np.float64)
else:  # pragma: no cover - depends on optional dependency
    _SWEEP_FLOAT32 = None
    _SWEEP_FLOAT64 = None


def require_numba() -> None:
    """Raise a useful error when the optional Numba backend is unavailable."""

    if NUMBA_AVAILABLE:
        return
    message = (
        "The fused lattice-ABP CPU backend requires numba. "
        "Install numba>=0.60 or use the PyTorch backend."
    )
    raise ImportError(message) from _NUMBA_IMPORT_ERROR


def prepare_sweep_random_inputs(
    particle_count: int,
    batch_size: int,
    *,
    dtype: torch.dtype,
    shuffle_particles: bool,
    legacy_call_order: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate CPU sweep randomness in the legacy PyTorch call order.

    ``draws`` is intentionally populated with ``particle_count`` separate
    ``torch.rand(batch_size)`` calls.  Replacing these calls by one bulk random
    draw can change seeded trajectories on some PyTorch/device combinations.
    """

    if particle_count <= 0 or batch_size <= 0:
        raise ValueError("particle_count and batch_size must be positive.")
    if dtype not in (torch.float32, torch.float64):
        raise ValueError("dtype must be torch.float32 or torch.float64.")

    if shuffle_particles:
        order = torch.randperm(particle_count, device="cpu")
    else:
        order = torch.arange(particle_count, device="cpu")

    if legacy_call_order:
        draws = torch.empty(
            particle_count,
            batch_size,
            dtype=dtype,
            device="cpu",
        )
        for order_index in range(particle_count):
            draws[order_index] = torch.rand(
                batch_size,
                dtype=dtype,
                device="cpu",
            )
    else:
        # This is distributionally identical and substantially faster.  On
        # current PyTorch CPU generators it is also seed-wise identical to the
        # loop above, but legacy_call_order=True remains the conservative
        # choice because RNG implementation details are not a public contract.
        draws = torch.rand(
            particle_count,
            batch_size,
            dtype=dtype,
            device="cpu",
        )
    return order, draws


def active_work_from_theta(
    theta: torch.Tensor,
    dir_vectors: torch.Tensor,
    v0: float,
) -> torch.Tensor:
    """Compute ``v(theta) dot direction`` with legacy particlewise operations.

    The particle loop is deliberate.  It uses the same operation shapes as the
    PyTorch reference path, avoiding device/math-library-dependent differences
    that a single larger trigonometric kernel could introduce.
    """

    if theta.device.type != "cpu" or dir_vectors.device.type != "cpu":
        raise ValueError("The Numba backend only accepts CPU tensors.")
    if theta.dim() != 2:
        raise ValueError("theta must have shape [B, N].")
    if dir_vectors.shape != (4, 2):
        raise ValueError("dir_vectors must have shape [4, 2].")
    if theta.dtype not in (torch.float32, torch.float64):
        raise ValueError("theta must have dtype float32 or float64.")
    if dir_vectors.dtype != theta.dtype:
        raise ValueError("theta and dir_vectors must have the same dtype.")

    batch_size, particle_count = theta.shape
    active_work = torch.empty(
        batch_size,
        particle_count,
        4,
        dtype=theta.dtype,
        device="cpu",
    )
    for particle_index in range(particle_count):
        theta_i = theta[:, particle_index]
        propulsion = v0 * torch.stack(
            [torch.cos(theta_i), torch.sin(theta_i)],
            dim=-1,
        )
        active_work[:, particle_index] = torch.sum(
            propulsion.view(batch_size, 1, 2)
            * dir_vectors.view(1, 4, 2),
            dim=-1,
        )
    return active_work


def active_work_from_theta_vectorized(
    theta: torch.Tensor,
    dir_vectors: torch.Tensor,
    v0: float,
) -> torch.Tensor:
    """Vectorized active-work calculation for the fused CPU backend.

    This has the same mathematical operations as
    :func:`active_work_from_theta`, but lets PyTorch evaluate all particle
    angles in one pair of trigonometric kernels.  It is much faster for B=1.
    The two implementations are bitwise identical for float32/float64 in the
    supported CPU configurations exercised by the backend tests.
    """

    if theta.device.type != "cpu" or dir_vectors.device.type != "cpu":
        raise ValueError("The Numba backend only accepts CPU tensors.")
    if theta.dim() != 2:
        raise ValueError("theta must have shape [B, N].")
    if dir_vectors.shape != (4, 2):
        raise ValueError("dir_vectors must have shape [4, 2].")
    if theta.dtype not in (torch.float32, torch.float64):
        raise ValueError("theta must have dtype float32 or float64.")
    if dir_vectors.dtype != theta.dtype:
        raise ValueError("theta and dir_vectors must have the same dtype.")

    propulsion = v0 * torch.stack(
        [torch.cos(theta), torch.sin(theta)],
        dim=-1,
    )
    return torch.sum(
        propulsion.unsqueeze(2) * dir_vectors.view(1, 1, 4, 2),
        dim=-1,
    )


def _require_cpu_contiguous(
    tensor: torch.Tensor,
    name: str,
) -> None:
    if tensor.device.type != "cpu":
        raise ValueError(f"{name} must be a CPU tensor.")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous.")


def numba_sweep_inplace(
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
) -> Dict[str, torch.Tensor]:
    """Run one exact random-sequential translational sweep on CPU.

    Args:
        sites: Contiguous ``int64`` tensor with shape ``[B, N, 2]``. Mutated.
        occupancy: Contiguous ``int64`` tensor ``[B, G, G]``. Mutated.
        order: The length-``N`` particle permutation for this sweep.
        draws: Categorical uniform draws with shape ``[N, B]``.
        active_work: Precomputed active work with shape ``[B, N, 4]``.
        kernel_offsets: WCA stencil offsets with shape ``[K, 2]``.
        kernel_values: WCA stencil energies with shape ``[K]``.

    Returns:
        A diagnostics dictionary matching ``_step_inplace``.  Angular Brownian
        motion is intentionally left to the caller.

    Notes:
        The compiled arithmetic implements the same transition rule; no
        tau-leaping, larger time step, or interaction approximation is used.
        Reproducible random streams additionally require callers to use
        :func:`prepare_sweep_random_inputs` and to generate the angular
        Gaussian noise only after this function succeeds.
    """

    require_numba()
    for name, tensor in (
        ("sites", sites),
        ("occupancy", occupancy),
        ("order", order),
        ("draws", draws),
        ("active_work", active_work),
        ("kernel_offsets", kernel_offsets),
        ("kernel_values", kernel_values),
    ):
        _require_cpu_contiguous(tensor, name)

    if sites.dtype != torch.long or sites.dim() != 3 or sites.shape[-1] != 2:
        raise ValueError("sites must be contiguous int64 with shape [B, N, 2].")
    if (
        occupancy.dtype != torch.long
        or occupancy.dim() != 3
        or occupancy.shape[-2] != occupancy.shape[-1]
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
        or kernel_offsets.dim() != 2
        or kernel_offsets.shape[1] != 2
    ):
        raise ValueError("kernel_offsets must be int64 with shape [K, 2].")
    if (
        kernel_values.dim() != 1
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

    normalized_prefactor = prefactor.lower()
    if normalized_prefactor not in {"cv", "c0"}:
        raise ValueError("prefactor must be 'cv' or 'c0'.")
    prefactor_code = 0 if normalized_prefactor == "cv" else 1

    dtype = kernel_values.dtype
    numpy_float_type = np.float32 if dtype == torch.float32 else np.float64
    kernel = _SWEEP_FLOAT32 if dtype == torch.float32 else _SWEEP_FLOAT64
    base = numpy_float_type(
        dt * reservoir_diffusion / (dl * dl)
    )
    diffusion_scalar = numpy_float_type(reservoir_diffusion)
    mobility_scalar = numpy_float_type(mobility)

    probabilities = torch.empty(
        batch_size,
        4,
        dtype=dtype,
        device="cpu",
    )
    proposed_sites = torch.empty(
        batch_size,
        4,
        2,
        dtype=torch.long,
        device="cpu",
    )
    delta_potential = torch.empty_like(probabilities)
    total_ep = torch.zeros(batch_size, dtype=dtype, device="cpu")
    active_ep = torch.zeros_like(total_ep)
    wca_ep = torch.zeros_like(total_ep)
    accepted_hops = torch.zeros(
        batch_size,
        dtype=torch.long,
        device="cpu",
    )
    if return_ep_map:
        ep_map = torch.zeros(
            batch_size,
            occupancy.shape[1],
            occupancy.shape[2],
            dtype=dtype,
            device="cpu",
        )
    else:
        ep_map = torch.empty(0, 0, 0, dtype=dtype, device="cpu")

    status, failed_order_index, bad_max_sum = kernel(
        sites.numpy(),
        occupancy.numpy(),
        order.numpy(),
        draws.numpy(),
        active_work.numpy(),
        kernel_offsets.numpy(),
        kernel_values.numpy(),
        base,
        diffusion_scalar,
        mobility_scalar,
        prefactor_code,
        strict_probabilities,
        float(probability_tolerance),
        return_ep_map,
        probabilities.numpy(),
        proposed_sites.numpy(),
        delta_potential.numpy(),
        total_ep.numpy(),
        active_ep.numpy(),
        wca_ep.numpy(),
        accepted_hops.numpy(),
        ep_map.numpy(),
    )

    if status == 1:
        raise ValueError(
            "Invalid lattice-MC probabilities: encountered non-finite "
            "values. Reduce dt, increase grid spacing, increase Dt, "
            "or use prefactor='cv'."
        )
    if status == 2:
        raise ValueError(
            "Invalid lattice-MC probabilities: total hop probability "
            f"reached {float(bad_max_sum):.6g} > 1. Reduce dt, increase "
            "grid spacing, increase Dt, or use prefactor='cv'."
        )
    if status != 0:  # pragma: no cover - defensive against backend corruption
        raise RuntimeError(
            "Unexpected Numba sweep status "
            f"{status} at order index {failed_order_index}."
        )

    diagnostics: Dict[str, torch.Tensor] = {
        "medium_ep": total_ep,
        "active_medium_ep": active_ep,
        "wca_medium_ep": wca_ep,
        "accepted_hops": accepted_hops,
    }
    if return_ep_map:
        diagnostics["medium_ep_map"] = ep_map
    return diagnostics
