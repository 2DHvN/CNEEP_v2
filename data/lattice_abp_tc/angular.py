"""Angular increments for particle-resolved lattice-ABP trajectories."""

import torch


def wrapped_angle_difference(theta1: torch.Tensor, theta0: torch.Tensor) -> torch.Tensor:
    """Return the signed shortest displacement ``theta1 - theta0`` on a circle."""
    if theta1.shape != theta0.shape:
        raise ValueError(
            f"theta shape mismatch: {tuple(theta1.shape)} != {tuple(theta0.shape)}"
        )
    delta = theta1 - theta0
    return torch.atan2(torch.sin(delta), torch.cos(delta))


def particle_angular_increment_map(
    sites0: torch.Tensor,
    sites1: torch.Tensor,
    theta0: torch.Tensor,
    theta1: torch.Tensor,
    grid_size: int,
) -> torch.Tensor:
    """Scatter particle-wise wrapped angular increments onto a lattice.

    Half of each particle's angular increment is assigned to its departure
    site and half to its arrival site.  A particle that does not move therefore
    contributes its full increment at one site.  The symmetric assignment is
    important: swapping the two frames negates the map at every lattice site,
    including for particles that hop.

    Parameters use batched shapes ``sites*: [B,N,2]`` and ``theta*: [B,N]``.
    The returned map has shape ``[B,G,G]`` and is measured in radians.
    """
    grid_size = int(grid_size)
    if grid_size <= 0:
        raise ValueError(f"grid_size must be positive, got {grid_size}")
    if sites0.shape != sites1.shape:
        raise ValueError(
            f"site shape mismatch: {tuple(sites0.shape)} != {tuple(sites1.shape)}"
        )
    if sites0.ndim != 3 or sites0.shape[-1] != 2:
        raise ValueError(f"Expected sites with shape [B,N,2], got {tuple(sites0.shape)}")
    if theta0.shape != theta1.shape or theta0.shape != sites0.shape[:-1]:
        raise ValueError(
            "Expected theta0/theta1 with shape [B,N] matching sites; got "
            f"{tuple(theta0.shape)}, {tuple(theta1.shape)}, and {tuple(sites0.shape)}"
        )
    devices = {sites0.device, sites1.device, theta0.device, theta1.device}
    if len(devices) != 1:
        raise ValueError("sites and theta tensors must be on the same device")

    sites0 = sites0.long()
    sites1 = sites1.long()
    if sites0.numel() and (
        int(torch.minimum(sites0.min(), sites1.min())) < 0
        or int(torch.maximum(sites0.max(), sites1.max())) >= grid_size
    ):
        raise ValueError(f"site index is outside [0, {grid_size})")

    delta_theta = wrapped_angle_difference(theta1, theta0)
    batch_size = theta0.shape[0]
    flat_map = torch.zeros(
        (batch_size, grid_size * grid_size),
        dtype=delta_theta.dtype,
        device=delta_theta.device,
    )
    linear0 = sites0[..., 0] * grid_size + sites0[..., 1]
    linear1 = sites1[..., 0] * grid_size + sites1[..., 1]
    half_delta = 0.5 * delta_theta
    flat_map.scatter_add_(1, linear0, half_delta)
    flat_map.scatter_add_(1, linear1, half_delta)
    return flat_map.view(batch_size, grid_size, grid_size)
