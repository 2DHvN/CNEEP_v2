"""Create ABP sanity-check notebooks."""

import json
import os


def md(cells, src):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": src.splitlines(True)})


def code(cells, src):
    cells.append(
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": src.splitlines(True),
        }
    )


def write_notebook(path, cells):
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.9.0"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"Created: {path}")


def root_cell():
    return """import os
import sys

candidate_roots = [
    os.environ.get("CNEEP_V2_ROOT"),
    os.path.abspath("../.."),
    os.path.abspath(".."),
    os.path.abspath("."),
    "/home/user1/CNEEP_v2",
]

CNEEP_V2_ROOT = None
for candidate in candidate_roots:
    if candidate and os.path.exists(os.path.join(candidate, "data", "ABP", "core.py")):
        CNEEP_V2_ROOT = candidate
        break

if CNEEP_V2_ROOT is None:
    raise RuntimeError("Could not locate CNEEP_v2 root. Set CNEEP_V2_ROOT.")

if CNEEP_V2_ROOT not in sys.path:
    sys.path.append(CNEEP_V2_ROOT)

print("CNEEP_v2 root:", CNEEP_V2_ROOT)"""


def build_sanity():
    cells = []
    md(
        cells,
        "# ABP Fieldization Sanity Check\n\n"
        "This notebook checks a continuous WCA-ABP simulation and the particle-size-aware "
        "mapping from particle centers to Eulerian fields.  The default field is a center "
        "count field, not a clipped 0/1 hard-core occupancy.  A Gaussian cloud option is "
        "also shown for comparison."
    )
    code(cells, root_cell())
    code(
        cells,
        """import math
import numpy as np
import torch
import matplotlib.pyplot as plt

from data.ABP import ABPParams, ContinuousABP, ABPFieldizer, recommended_center_grid_size

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)""",
    )
    md(cells, "## 1. Simulate a small WCA-ABP ensemble")
    code(
        cells,
        """params = ABPParams(
    N=196,
    L=20.0,
    sigma=1.0,
    epsilon=1.0,
    mobility=1.0,
    force_clip=500.0,
    force_chunk_size=256,
    v0=8.0,
    Dr=1.0,
    Dt=0.02,
    dt=2.0e-4,
    seed=7,
    device=device,
)

grid_center = max(32, recommended_center_grid_size(params.L, params.sigma))
fieldizer = ABPFieldizer(
    box_size=params.L,
    grid_size=grid_center,
    particle_diameter=params.sigma,
    mode="center",
    include_orientation=True,
    clip_occupancy=False,
)

print(f"packing fraction phi={params.phi:.3f}, Pe={params.Pe:.2f}")
print(f"grid={grid_center}x{grid_center}, dx={fieldizer.dx:.4f}, sigma/sqrt(2)={params.sigma / math.sqrt(2):.4f}")

sim = ContinuousABP(params)
result = sim.simulate(
    B=2,
    burn_in=200,
    n_steps=600,
    save_interval=30,
    fieldizer=fieldizer,
    show_progress=True,
)

positions = result["positions"]
theta = result["theta"]
fields = result["fields"]
print("positions:", positions.shape)
print("theta:    ", theta.shape)
print("fields:   ", fields.shape)""",
    )
    md(cells, "## 2. Center-count diagnostics")
    code(
        cells,
        """last_pos = positions[-1].to(device)
diag = fieldizer.diagnostics_dict(last_pos)
for k, v in diag.items():
    print(f"{k}: {v}")

print(f"min pair distance / sigma: {(result['min_distance'][-1].min() / params.sigma).item():.3f}")
print("Count fields keep multi-center pixels instead of clipping them to 1.")""",
    )
    md(cells, "## 3. Particle state and center field")
    code(
        cells,
        """frame = -1
ens = 0
pos = positions[frame, ens].numpy()
ang = theta[frame, ens].numpy()
field = fields[frame, ens].numpy()

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
axes[0].scatter(pos[:, 0], pos[:, 1], s=8, alpha=0.8)
stride = max(1, params.N // 60)
axes[0].quiver(
    pos[::stride, 0], pos[::stride, 1],
    np.cos(ang[::stride]), np.sin(ang[::stride]),
    angles="xy", scale_units="xy", scale=4, width=0.004,
)
axes[0].set_xlim(0, params.L)
axes[0].set_ylim(0, params.L)
axes[0].set_aspect("equal")
axes[0].set_title("particles")

im1 = axes[1].imshow(field[0].T, origin="lower", cmap="viridis")
axes[1].set_title("center count")
plt.colorbar(im1, ax=axes[1], fraction=0.046)

orient_mag = np.sqrt(field[1] ** 2 + field[2] ** 2)
im2 = axes[2].imshow(orient_mag.T, origin="lower", cmap="viridis", vmin=0, vmax=1)
axes[2].set_title("orientation-channel magnitude")
plt.colorbar(im2, ax=axes[2], fraction=0.046)

plt.tight_layout()
plt.show()""",
    )
    md(cells, "## 4. Center count vs finite-radius and Gaussian fields")
    code(
        cells,
        """center_fieldizer = ABPFieldizer(
    params.L, grid_center, params.sigma,
    mode="center", include_orientation=False, clip_occupancy=False,
)
disk_fieldizer = ABPFieldizer(
    params.L, grid_center, params.sigma,
    mode="disk", include_orientation=False, clip_occupancy=False,
)
gaussian_fieldizer = ABPFieldizer(
    params.L, grid_center, params.sigma,
    mode="gaussian", include_orientation=False, clip_occupancy=False,
    gaussian_sigma=0.5 * params.sigma,
)

center_field = center_fieldizer.encode(last_pos[:1])[0, 0].cpu().numpy()
disk_field = disk_fieldizer.encode(last_pos[:1])[0, 0].cpu().numpy()
gaussian_field = gaussian_fieldizer.encode(last_pos[:1])[0, 0].cpu().numpy()

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
im0 = axes[0].imshow(center_field.T, origin="lower", cmap="viridis")
axes[0].set_title("center count")
plt.colorbar(im0, ax=axes[0], fraction=0.046)

im1 = axes[1].imshow(disk_field.T, origin="lower", cmap="magma")
axes[1].set_title("disk footprint count")
plt.colorbar(im1, ax=axes[1], fraction=0.046)

im2 = axes[2].imshow(gaussian_field.T, origin="lower", cmap="viridis")
axes[2].set_title("Gaussian cloud")
plt.colorbar(im2, ax=axes[2], fraction=0.046)

plt.tight_layout()
plt.show()

print(f"center count sum:   {center_field.sum():.6f}")
print(f"disk footprint sum: {disk_field.sum():.6f}")
print(f"gaussian sum:       {gaussian_field.sum():.6f}")
print(f"N particles:        {params.N}")""",
    )
    md(cells, "## 5. Basic WCA stability checks")
    code(
        cells,
        """time = result["times"].numpy()
potential = result["potential"].numpy()
min_distance = result["min_distance"].numpy()
mean_force = result["mean_force_norm"].numpy()

fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
axes[0].plot(time, potential)
axes[0].set_ylabel("WCA potential")
axes[0].set_title("Potential should stay finite")

axes[1].plot(time, min_distance / params.sigma)
axes[1].axhline(1.0, color="k", linestyle="--", lw=1, label="sigma")
axes[1].set_ylabel("min distance / sigma")
axes[1].legend()

axes[2].plot(time, mean_force)
axes[2].set_ylabel("mean |F|")
axes[2].set_xlabel("time")

plt.tight_layout()
plt.show()""",
    )
    return cells


def build_steady():
    cells = []
    md(
        cells,
        "# ABP Steady-State Sanity Check\n\n"
        "This notebook runs a longer WCA-ABP trajectory and checks whether simple "
        "observables have reached an approximately stationary regime.  For MIPS-like "
        "settings the density structure may continue to coarsen, so this notebook is "
        "a practical plateau check rather than a proof of stationarity."
    )
    code(cells, root_cell())
    code(
        cells,
        """import math
import numpy as np
import torch
import matplotlib.pyplot as plt

from data.ABP import ABPParams, ContinuousABP, ABPFieldizer, recommended_center_grid_size

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)""",
    )
    md(cells, "## 1. Run a longer trajectory")
    code(
        cells,
        """params = ABPParams(
    N=256,
    L=24.0,
    sigma=1.0,
    epsilon=1.0,
    mobility=1.0,
    force_clip=500.0,
    force_chunk_size=256,
    v0=12.0,
    Dr=1.0,
    Dt=0.01,
    dt=2.0e-4,
    seed=11,
    device=device,
)

grid_size = max(40, recommended_center_grid_size(params.L, params.sigma))
fieldizer = ABPFieldizer(params.L, grid_size, params.sigma, mode="center", include_orientation=False, clip_occupancy=False)
sim = ContinuousABP(params)

result = sim.simulate(
    B=3,
    burn_in=2_000,
    n_steps=6_000,
    save_interval=60,
    fieldizer=fieldizer,
    show_progress=True,
)

print("phi:", params.phi, "Pe:", params.Pe)
print("positions:", result["positions"].shape, "fields:", result["fields"].shape)""",
    )
    md(cells, "## 2. Build observables")
    code(
        cells,
        """time = result["times"].numpy()
potential = result["potential"].numpy().mean(axis=1)
min_distance = result["min_distance"].numpy().mean(axis=1)
mean_force = result["mean_force_norm"].numpy().mean(axis=1)
fields = result["fields"][:, :, 0].numpy()  # [T, B, H, W]

def low_k_power(field_batch, kmax=3):
    vals = []
    for field in field_batch:
        f = field - field.mean()
        spec = np.abs(np.fft.rfft2(f)) ** 2
        low = spec[: kmax + 1, : kmax + 1].sum() - spec[0, 0]
        total = spec.sum() + 1e-12
        vals.append(low / total)
    return float(np.mean(vals))

low_power = np.array([low_k_power(fields[t]) for t in range(fields.shape[0])])

def running_mean(x, window=7):
    window = min(window, len(x))
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode="same")

def plateau_report(name, x):
    q = max(2, len(x) // 4)
    middle = x[-2 * q : -q]
    late = x[-q:]
    scale = np.std(x[-2 * q :]) + 1e-12
    score = abs(late.mean() - middle.mean()) / scale
    print(f"{name:18s} middle={middle.mean():.5e} late={late.mean():.5e} score={score:.3f}")

plateau_report("WCA potential", potential)
plateau_report("min distance", min_distance)
plateau_report("low-k power", low_power)
plateau_report("mean |F|", mean_force)""",
    )
    md(cells, "## 3. Time-series plateau check")
    code(
        cells,
        """fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True)
axes[0].plot(time, potential, alpha=0.45)
axes[0].plot(time, running_mean(potential), lw=2)
axes[0].set_ylabel("WCA U")

axes[1].plot(time, min_distance / params.sigma, alpha=0.45)
axes[1].plot(time, running_mean(min_distance / params.sigma), lw=2)
axes[1].axhline(1.0, color="k", linestyle="--", lw=1)
axes[1].set_ylabel("min r / sigma")

axes[2].plot(time, low_power, alpha=0.45)
axes[2].plot(time, running_mean(low_power), lw=2)
axes[2].set_ylabel("low-k power")

axes[3].plot(time, mean_force, alpha=0.45)
axes[3].plot(time, running_mean(mean_force), lw=2)
axes[3].set_ylabel("mean |F|")
axes[3].set_xlabel("time")

plt.tight_layout()
plt.show()""",
    )
    md(cells, "## 4. Early/middle/late density snapshots")
    code(
        cells,
        """snap_ids = [0, len(time) // 2, len(time) - 1]
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
for ax, t in zip(axes, snap_ids):
    img = fields[t, 0]
    im = ax.imshow(img.T, origin="lower", cmap="viridis")
    ax.set_title(f"t={time[t]:.3f}")
    plt.colorbar(im, ax=ax, fraction=0.046)
plt.tight_layout()
plt.show()""",
    )
    md(cells, "## 5. Center-count audit over saved frames")
    code(
        cells,
        """multi_center_counts = []
max_center_counts = []
for t in range(result["positions"].shape[0]):
    diag = fieldizer.diagnostics_dict(result["positions"][t].to(device))
    multi_center_counts.append(diag["multi_center_pixels"])
    max_center_counts.append(diag["max_center_count"])
multi_center_counts = np.asarray(multi_center_counts)
max_center_counts = np.asarray(max_center_counts)

plt.figure(figsize=(9, 3))
plt.plot(time, multi_center_counts, label="multi-center pixels")
plt.plot(time, max_center_counts, label="max center count")
plt.xlabel("time")
plt.ylabel("count statistic")
plt.title("Center-count audit")
plt.legend()
plt.tight_layout()
plt.show()

print("max multi-center pixels over saved frames:", int(multi_center_counts.max()))
print("max center count over saved frames:", int(max_center_counts.max()))
print("fieldizer dx:", fieldizer.dx)
print("hard-core diagnostic dx limit:", params.sigma / math.sqrt(2.0))""",
    )
    return cells


if __name__ == "__main__":
    here = os.path.dirname(__file__)
    write_notebook(os.path.join(here, "sanity_check_abp.ipynb"), build_sanity())
    write_notebook(os.path.join(here, "steady_state_sanity_check_abp.ipynb"), build_steady())
