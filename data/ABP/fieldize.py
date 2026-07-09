"""Particle-size-aware field encodings for continuous ABP trajectories."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional

import torch


def recommended_center_grid_size(box_size: float, particle_diameter: float) -> int:
    """Grid size for a hard-core-style center-bin diagnostic.

    If the pixel side length ``dx`` satisfies ``sqrt(2) * dx <= sigma``, two
    particle centers inside the same pixel would be closer than the hard-core
    diameter ``sigma``.  For WCA this is not a strict constraint, but it is a
    useful diagnostic scale.  Count fields can still be used on coarser grids.
    """
    if box_size <= 0 or particle_diameter <= 0:
        raise ValueError("box_size and particle_diameter must be positive.")
    max_dx = particle_diameter / math.sqrt(2.0)
    return int(math.ceil(box_size / max_dx))


@dataclass
class FieldDiagnostics:
    max_center_count: int
    multi_center_pixels: int
    pixel_size: float
    center_exclusive_dx_limit: float
    center_exclusive_rule_ok: bool


class ABPFieldizer:
    """Convert ABP particle centers to Eulerian fields.

    Parameters
    ----------
    box_size:
        Periodic box length.
    grid_size:
        Output field side length.
    particle_diameter:
        ABP/WCA particle diameter ``sigma``.
    mode:
        ``"center"`` bins particle centers into pixel counts. ``"disk"``
        paints the finite particle disk on all pixels whose centers fall inside
        the particle radius. ``"gaussian"`` deposits a count-preserving
        Gaussian cloud around each center.
    include_orientation:
        Add local ``cos(theta)`` and ``sin(theta)`` channels.  For center mode
        these channels are averaged when a coarse grid causes multiple centers
        in a pixel.
    clip_occupancy:
        Clip occupancy/count fields to ``0/1``.
    gaussian_sigma:
        Physical Gaussian width for ``mode="gaussian"``.  If omitted, the
        width defaults to half the particle diameter.
    gaussian_sigma_pixels:
        Gaussian width in pixels.  Overrides ``gaussian_sigma`` when provided.
    gaussian_truncate:
        Gaussian support radius in standard deviations.
    gaussian_normalize:
        If True, normalize each particle's truncated kernel so the first
        channel sums to the number of particles.
    """

    def __init__(
        self,
        box_size: float,
        grid_size: int,
        particle_diameter: float = 1.0,
        *,
        mode: str = "center",
        include_orientation: bool = False,
        clip_occupancy: bool = False,
        gaussian_sigma: Optional[float] = None,
        gaussian_sigma_pixels: Optional[float] = None,
        gaussian_truncate: float = 3.0,
        gaussian_normalize: bool = True,
        dtype: torch.dtype = torch.float32,
    ):
        if mode not in {"center", "disk", "gaussian"}:
            raise ValueError("mode must be 'center', 'disk', or 'gaussian'.")
        if grid_size <= 0:
            raise ValueError("grid_size must be positive.")
        if box_size <= 0 or particle_diameter <= 0:
            raise ValueError("box_size and particle_diameter must be positive.")

        self.box_size = float(box_size)
        self.grid_size = int(grid_size)
        self.particle_diameter = float(particle_diameter)
        self.mode = mode
        self.include_orientation = include_orientation
        self.clip_occupancy = clip_occupancy
        self.gaussian_sigma = gaussian_sigma
        self.gaussian_sigma_pixels = gaussian_sigma_pixels
        self.gaussian_truncate = float(gaussian_truncate)
        self.gaussian_normalize = gaussian_normalize
        self.dtype = dtype

    @property
    def dx(self) -> float:
        return self.box_size / self.grid_size

    @property
    def particle_radius_pixels(self) -> float:
        return 0.5 * self.particle_diameter / self.dx

    @property
    def gaussian_sigma_in_pixels(self) -> float:
        if self.gaussian_sigma_pixels is not None:
            return float(self.gaussian_sigma_pixels)
        sigma = 0.5 * self.particle_diameter if self.gaussian_sigma is None else float(self.gaussian_sigma)
        return sigma / self.dx

    @property
    def n_channels(self) -> int:
        return 3 if self.include_orientation else 1

    def _center_indices(self, pos: torch.Tensor):
        H = self.grid_size
        idx = torch.floor((pos % self.box_size) / self.dx).long() % H
        ix = idx[..., 0]
        iy = idx[..., 1]
        return ix, iy

    def _scatter_center(self, pos: torch.Tensor, theta: Optional[torch.Tensor]):
        B, N, _ = pos.shape
        H = self.grid_size
        ix, iy = self._center_indices(pos)
        linear = ix * H + iy
        ones = torch.ones(B, N, device=pos.device, dtype=self.dtype)

        counts = torch.zeros(B, H * H, device=pos.device, dtype=self.dtype)
        counts.scatter_add_(1, linear, ones)

        occ = counts.clamp(max=1.0) if self.clip_occupancy else counts
        channels = [occ.view(B, H, H)]

        if self.include_orientation:
            if theta is None:
                raise ValueError("theta is required when include_orientation=True.")
            cos_sum = torch.zeros_like(counts)
            sin_sum = torch.zeros_like(counts)
            cos_sum.scatter_add_(1, linear, torch.cos(theta).to(self.dtype))
            sin_sum.scatter_add_(1, linear, torch.sin(theta).to(self.dtype))
            denom = counts.clamp_min(1.0)
            mask = (counts > 0).to(self.dtype)
            channels.append((cos_sum / denom * mask).view(B, H, H))
            channels.append((sin_sum / denom * mask).view(B, H, H))

        return torch.stack(channels, dim=1), counts.view(B, H, H)

    def _disk_offsets(self, device: torch.device) -> torch.Tensor:
        radius = self.particle_radius_pixels
        max_offset = int(math.ceil(radius))
        offsets = []
        for dx in range(-max_offset, max_offset + 1):
            for dy in range(-max_offset, max_offset + 1):
                if math.sqrt(dx * dx + dy * dy) <= radius + 1e-12:
                    offsets.append((dx, dy))
        if not offsets:
            offsets.append((0, 0))
        return torch.tensor(offsets, device=device, dtype=torch.long)

    def _scatter_disk(self, pos: torch.Tensor, theta: Optional[torch.Tensor]):
        B, N, _ = pos.shape
        H = self.grid_size
        ix, iy = self._center_indices(pos)
        offsets = self._disk_offsets(pos.device)
        ox = offsets[:, 0].view(1, 1, -1)
        oy = offsets[:, 1].view(1, 1, -1)

        px = (ix.unsqueeze(-1) + ox) % H
        py = (iy.unsqueeze(-1) + oy) % H
        linear = (px * H + py).reshape(B, -1)

        weights = torch.ones(B, linear.shape[1], device=pos.device, dtype=self.dtype)
        counts = torch.zeros(B, H * H, device=pos.device, dtype=self.dtype)
        counts.scatter_add_(1, linear, weights)

        occ = counts.clamp(max=1.0) if self.clip_occupancy else counts
        channels = [occ.view(B, H, H)]

        if self.include_orientation:
            if theta is None:
                raise ValueError("theta is required when include_orientation=True.")
            K = offsets.shape[0]
            cos_values = torch.cos(theta).to(self.dtype).unsqueeze(-1).expand(B, N, K).reshape(B, -1)
            sin_values = torch.sin(theta).to(self.dtype).unsqueeze(-1).expand(B, N, K).reshape(B, -1)
            cos_sum = torch.zeros_like(counts)
            sin_sum = torch.zeros_like(counts)
            cos_sum.scatter_add_(1, linear, cos_values)
            sin_sum.scatter_add_(1, linear, sin_values)
            denom = counts.clamp_min(1.0)
            mask = (counts > 0).to(self.dtype)
            channels.append((cos_sum / denom * mask).view(B, H, H))
            channels.append((sin_sum / denom * mask).view(B, H, H))

        return torch.stack(channels, dim=1), counts.view(B, H, H)

    def _gaussian_offsets(self, device: torch.device) -> torch.Tensor:
        sigma_px = self.gaussian_sigma_in_pixels
        if sigma_px <= 0:
            raise ValueError("Gaussian sigma must be positive.")
        radius = int(math.ceil(self.gaussian_truncate * sigma_px))
        offsets = [
            (dx, dy)
            for dx in range(-radius, radius + 1)
            for dy in range(-radius, radius + 1)
        ]
        return torch.tensor(offsets, device=device, dtype=torch.long)

    def _scatter_gaussian(self, pos: torch.Tensor, theta: Optional[torch.Tensor]):
        B, N, _ = pos.shape
        H = self.grid_size
        sigma_px = self.gaussian_sigma_in_pixels
        offsets = self._gaussian_offsets(pos.device)
        K = offsets.shape[0]

        coord = (pos % self.box_size) / self.dx
        base = torch.floor(coord).long() % H
        frac = coord - torch.floor(coord)

        ox = offsets[:, 0].view(1, 1, K)
        oy = offsets[:, 1].view(1, 1, K)
        px = (base[..., 0].unsqueeze(-1) + ox) % H
        py = (base[..., 1].unsqueeze(-1) + oy) % H

        # Pixel-center distance from the particle center in pixel units.
        dx_pix = ox.to(pos.dtype) + 0.5 - frac[..., 0].unsqueeze(-1)
        dy_pix = oy.to(pos.dtype) + 0.5 - frac[..., 1].unsqueeze(-1)
        weights = torch.exp(-(dx_pix * dx_pix + dy_pix * dy_pix) / (2.0 * sigma_px * sigma_px))
        if self.gaussian_normalize:
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(weights.dtype).eps)
        weights = weights.to(self.dtype)

        linear = (px * H + py).reshape(B, -1)
        weights_flat = weights.reshape(B, -1)
        counts = torch.zeros(B, H * H, device=pos.device, dtype=self.dtype)
        counts.scatter_add_(1, linear, weights_flat)

        field = counts.clamp(max=1.0) if self.clip_occupancy else counts
        channels = [field.view(B, H, H)]

        if self.include_orientation:
            if theta is None:
                raise ValueError("theta is required when include_orientation=True.")
            cos_values = (
                torch.cos(theta).to(self.dtype).unsqueeze(-1).expand(B, N, K) * weights
            ).reshape(B, -1)
            sin_values = (
                torch.sin(theta).to(self.dtype).unsqueeze(-1).expand(B, N, K) * weights
            ).reshape(B, -1)
            cos_sum = torch.zeros_like(counts)
            sin_sum = torch.zeros_like(counts)
            cos_sum.scatter_add_(1, linear, cos_values)
            sin_sum.scatter_add_(1, linear, sin_values)
            denom = counts.clamp_min(1e-12)
            mask = (counts > 1e-12).to(self.dtype)
            channels.append((cos_sum / denom * mask).view(B, H, H))
            channels.append((sin_sum / denom * mask).view(B, H, H))

        return torch.stack(channels, dim=1), counts.view(B, H, H)

    def encode(
        self,
        pos: torch.Tensor,
        theta: Optional[torch.Tensor] = None,
        *,
        return_counts: bool = False,
    ):
        """Encode positions as ``[B, C, H, W]`` fields."""
        if pos.dim() == 2:
            pos = pos.unsqueeze(0)
            if theta is not None and theta.dim() == 1:
                theta = theta.unsqueeze(0)
        if pos.dim() != 3 or pos.shape[-1] != 2:
            raise ValueError("pos must have shape [B, N, 2] or [N, 2].")
        if theta is not None:
            theta = theta.to(device=pos.device)

        if self.mode == "center":
            fields, counts = self._scatter_center(pos, theta)
        elif self.mode == "disk":
            fields, counts = self._scatter_disk(pos, theta)
        else:
            fields, counts = self._scatter_gaussian(pos, theta)

        if return_counts:
            return fields, counts
        return fields

    def encode_sequence(
        self,
        positions: torch.Tensor,
        theta: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode ``[T, B, N, 2]`` positions as ``[T, B, C, H, W]`` fields."""
        if positions.dim() != 4:
            raise ValueError("positions must have shape [T, B, N, 2].")
        fields = []
        for t in range(positions.shape[0]):
            theta_t = None if theta is None else theta[t]
            fields.append(self.encode(positions[t], theta_t))
        return torch.stack(fields, dim=0)

    def diagnostics(self, pos: torch.Tensor) -> FieldDiagnostics:
        """Return center-bin collision diagnostics for a position batch."""
        if pos.dim() == 2:
            pos = pos.unsqueeze(0)
        if pos.dim() != 3 or pos.shape[-1] != 2:
            raise ValueError("pos must have shape [B, N, 2] or [N, 2].")

        B, N, _ = pos.shape
        H = self.grid_size
        ix, iy = self._center_indices(pos)
        linear = ix * H + iy
        ones = torch.ones(B, N, device=pos.device, dtype=self.dtype)
        counts = torch.zeros(B, H * H, device=pos.device, dtype=self.dtype)
        counts.scatter_add_(1, linear, ones)
        counts = counts.view(B, H, H)

        dx_limit = self.particle_diameter / math.sqrt(2.0)
        return FieldDiagnostics(
            max_center_count=int(counts.max().detach().cpu().item()),
            multi_center_pixels=int((counts > 1).sum().detach().cpu().item()),
            pixel_size=self.dx,
            center_exclusive_dx_limit=dx_limit,
            center_exclusive_rule_ok=self.dx <= dx_limit + 1e-12,
        )

    def diagnostics_dict(self, pos: torch.Tensor) -> Dict[str, float | int | bool]:
        d = self.diagnostics(pos)
        return {
            "max_center_count": d.max_center_count,
            "multi_center_pixels": d.multi_center_pixels,
            "pixel_size": d.pixel_size,
            "center_exclusive_dx_limit": d.center_exclusive_dx_limit,
            "center_exclusive_rule_ok": d.center_exclusive_rule_ok,
        }
