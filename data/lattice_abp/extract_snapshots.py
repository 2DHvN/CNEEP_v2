"""Extract key frames from the MIPS video as PNG snapshots."""
import sys, os
sys.path.insert(0, '.')
os.makedirs("output", exist_ok=True)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from core import LatticeABP
import time

L = 64
density = 0.55
v_plus = 10.0
v_zero = 0.5
v_minus = 0.1
D_rot = 0.2
Pe = (v_plus - v_minus) / (2 * D_rot)
N = int(density * L * L)

sim = LatticeABP(L=L, v_plus=v_plus, v_zero=v_zero, v_minus=v_minus,
                 D_rot=D_rot, density=density, seed=123)
O, E = sim.init_state(B=1)

# Snapshot times: initial, mid-burn, post-burn, late
snapshot_steps = [0, 2000, 5000, 8000, 12000, 20000]
snapshot_labels = ["t=0 (Initial)", "t=2k (Nucleation)", "t=5k (Burn-in done)",
                   "t=8k", "t=12k (Coarsening)", "t=20k (Steady MIPS)"]

current_step = 0
snapshots = []

for target in snapshot_steps:
    while current_step < target:
        O, E, _ = sim.gillespie_step(O, E)
        current_step += 1
        if current_step % 2000 == 0:
            print(f"  step {current_step}...")

    jammed = sim.compute_jammed_mask(O, E)
    occ = O[0].cpu().numpy()
    jam = jammed[0].cpu().numpy()

    rgb = np.zeros((L, L, 3))
    rgb[:,:,0] = 0.05; rgb[:,:,1] = 0.07; rgb[:,:,2] = 0.14
    free = (occ == 1) & (~jam)
    rgb[free, 0] = 0.20; rgb[free, 1] = 0.60; rgb[free, 2] = 0.86
    jm = (occ == 1) & jam
    rgb[jm, 0] = 0.91; rgb[jm, 1] = 0.30; rgb[jm, 2] = 0.24

    n_jam = (jam & (occ == 1)).sum()
    snapshots.append((rgb.copy(), n_jam))
    print(f"  Snapshot at step {target}: jammed={n_jam}/{N} ({100*n_jam/N:.1f}%)")

# Plot panel
fig, axes = plt.subplots(2, 3, figsize=(16, 11), facecolor="#0d1117")
fig.suptitle(
    f"MIPS Evolution — L={L}, ρ={density}, Pe={Pe:.1f}",
    color="white", fontsize=18, fontweight="bold", y=0.98,
)

for i, (ax, (rgb, n_jam), label) in enumerate(zip(axes.flat, snapshots, snapshot_labels)):
    ax.imshow(rgb, interpolation="nearest")
    ax.set_title(label, color="white", fontsize=12, fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])
    frac = 100 * n_jam / N
    ax.text(0.02, 0.02, f"Jammed: {frac:.0f}%", transform=ax.transAxes,
            color="white", fontsize=10, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#1a1a2e",
                      edgecolor="#444", alpha=0.85),
            verticalalignment="bottom")

plt.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig("output/mips_snapshots.png", dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
print("\nSaved: output/mips_snapshots.png")
