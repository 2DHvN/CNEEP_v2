"""Small diagnostics for lattice-ABP MIPS sanity checks."""

from __future__ import annotations

from collections import deque
from typing import Dict

import numpy as np


def coarse_density_periodic(occupancy: np.ndarray, box: int = 8) -> np.ndarray:
    """Return a periodic box-averaged density field.

    Args:
        occupancy: 2D binary occupancy array.
        box: Side length of the square coarse-graining window.
    """
    occ = np.asarray(occupancy, dtype=np.float64)
    if occ.ndim != 2:
        raise ValueError("occupancy must be a 2D array.")
    if box <= 0:
        raise ValueError("box must be positive.")

    box = min(int(box), occ.shape[0], occ.shape[1])
    out = np.zeros_like(occ, dtype=np.float64)
    for dy in range(box):
        rolled_y = np.roll(occ, -dy, axis=0)
        for dx in range(box):
            out += np.roll(rolled_y, -dx, axis=1)
    return out / float(box * box)


def largest_cluster_fraction(occupancy: np.ndarray, periodic: bool = True) -> float:
    """Return the fraction of occupied sites in the largest connected cluster."""
    occ = np.asarray(occupancy).astype(bool)
    if occ.ndim != 2:
        raise ValueError("occupancy must be a 2D array.")

    total = int(occ.sum())
    if total == 0:
        return 0.0

    Lx, Ly = occ.shape
    visited = np.zeros_like(occ, dtype=bool)
    largest = 0
    neighbors = ((1, 0), (-1, 0), (0, 1), (0, -1))

    for i in range(Lx):
        for j in range(Ly):
            if not occ[i, j] or visited[i, j]:
                continue

            size = 0
            queue = deque([(i, j)])
            visited[i, j] = True

            while queue:
                r, c = queue.popleft()
                size += 1
                for dr, dc in neighbors:
                    nr, nc = r + dr, c + dc
                    if periodic:
                        nr %= Lx
                        nc %= Ly
                    elif nr < 0 or nr >= Lx or nc < 0 or nc >= Ly:
                        continue

                    if occ[nr, nc] and not visited[nr, nc]:
                        visited[nr, nc] = True
                        queue.append((nr, nc))

            largest = max(largest, size)

    return largest / float(total)


def low_k_structure_ratio(occupancy: np.ndarray, low_cut: float = 0.12, high_cut: float = 0.35) -> float:
    """Measure low-wave-number density enhancement in the structure factor."""
    occ = np.asarray(occupancy, dtype=np.float64)
    if occ.ndim != 2:
        raise ValueError("occupancy must be a 2D array.")

    centered = occ - occ.mean()
    spectrum = np.abs(np.fft.fftshift(np.fft.fft2(centered))) ** 2
    Lx, Ly = occ.shape
    ky = np.fft.fftshift(np.fft.fftfreq(Lx))
    kx = np.fft.fftshift(np.fft.fftfreq(Ly))
    grid_y, grid_x = np.meshgrid(ky, kx, indexing="ij")
    radius = np.sqrt(grid_y**2 + grid_x**2)

    low = (radius > 0) & (radius <= low_cut)
    mid = (radius > low_cut) & (radius <= high_cut)
    low_mean = float(spectrum[low].mean()) if np.any(low) else 0.0
    mid_mean = float(spectrum[mid].mean()) if np.any(mid) else 0.0
    return low_mean / (mid_mean + 1e-12)


def summarize_mips_snapshot(
    occupancy: np.ndarray,
    *,
    density: float | None = None,
    coarse_box: int = 8,
    periodic: bool = True,
) -> Dict[str, float]:
    """Compute compact scalar diagnostics for one occupancy snapshot."""
    occ = np.asarray(occupancy, dtype=np.float64)
    rho = float(occ.mean() if density is None else density)
    local_density = coarse_density_periodic(occ, box=coarse_box)
    random_std = np.sqrt(max(rho * (1.0 - rho), 0.0) / float(coarse_box * coarse_box))

    return {
        "density": float(occ.mean()),
        "coarse_std": float(local_density.std()),
        "coarse_std_random": float(random_std),
        "coarse_std_ratio": float(local_density.std() / (random_std + 1e-12)),
        "coarse_q10": float(np.quantile(local_density, 0.10)),
        "coarse_q90": float(np.quantile(local_density, 0.90)),
        "largest_cluster_fraction": largest_cluster_fraction(occ, periodic=periodic),
        "low_k_ratio": low_k_structure_ratio(occ),
    }


def mips_pass(summary_initial: Dict[str, float], summary_final: Dict[str, float]) -> bool:
    """Heuristic pass/fail check for visible MIPS-like aggregation."""
    cluster_growth = summary_final["largest_cluster_fraction"] / (
        summary_initial["largest_cluster_fraction"] + 1e-12
    )
    contrast = summary_final["coarse_q90"] - summary_final["coarse_q10"]
    return bool(
        summary_final["largest_cluster_fraction"] >= 0.35
        and cluster_growth >= 1.25
        and summary_final["coarse_std_ratio"] >= 1.35
        and contrast >= 0.18
    )
