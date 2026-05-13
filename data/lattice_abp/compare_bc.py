"""Extract final-state comparison snapshots from both BCs."""
import sys, os
sys.path.insert(0, '.')
os.makedirs("output", exist_ok=True)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from core import LatticeABP
import time

L, density = 64, 0.55
v_plus, v_zero, v_minus, D_rot = 10.0, 0.5, 0.1, 0.2
Pe = (v_plus - v_minus) / (2 * D_rot)
N = int(density * L * L)

def build_rgb(occ, jam):
    rgb = np.zeros((L, L, 3))
    rgb[:,:,0]=0.05; rgb[:,:,1]=0.07; rgb[:,:,2]=0.14
    free = (occ == 1) & (~jam)
    rgb[free,0]=0.20; rgb[free,1]=0.60; rgb[free,2]=0.86
    jm = (occ == 1) & jam
    rgb[jm,0]=0.91; rgb[jm,1]=0.30; rgb[jm,2]=0.24
    return rgb

fig, axes = plt.subplots(1, 2, figsize=(16, 8), facecolor="#0d1117")

for i, bc in enumerate(["periodic", "hard_wall"]):
    label = "Periodic BC" if bc == "periodic" else "Hard Wall BC"
    print(f"\nRunning {label}...")
    sim = LatticeABP(L=L, v_plus=v_plus, v_zero=v_zero, v_minus=v_minus,
                     D_rot=D_rot, density=density, bc_mode=bc, seed=123)
    O, E = sim.init_state(B=1)
    for s in range(40000):
        O, E, _ = sim.gillespie_step(O, E)
        if (s+1) % 10000 == 0: print(f"  step {s+1}/40000")

    jammed = sim.compute_jammed_mask(O, E)
    occ = O[0].cpu().numpy()
    jam = jammed[0].cpu().numpy()
    n_jam = (jam & (occ==1)).sum()

    ax = axes[i]
    ax.imshow(build_rgb(occ, jam), interpolation="nearest")
    ax.set_title(f"{label}  (Jammed: {100*n_jam/N:.1f}%)",
                 color="white", fontsize=14, fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])

fig.suptitle(f"MIPS Steady State — L={L}, ρ={density}, Pe={Pe:.1f}",
             color="white", fontsize=18, fontweight="bold", y=0.98)
axes[0].legend(
    handles=[Patch(facecolor="#e74c3c", label="Jammed"),
             Patch(facecolor="#3498db", label="Free")],
    loc="upper left", fontsize=10,
    facecolor="#1a1a2e", edgecolor="#444", labelcolor="white")
plt.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig("output/mips_bc_comparison.png", dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
print("\nSaved: output/mips_bc_comparison.png")
