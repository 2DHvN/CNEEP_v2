"""
Generate MIPS videos to compare exact Gillespie (Tau-leaping OFF) 
and Tau-leaping approximation (Tau-leaping ON).

Starting from t=0 to observe differences in formation and speed.
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.makedirs("output", exist_ok=True)

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Patch
from core import LatticeABP

def build_rgb(occ, jam, L):
    rgb = np.zeros((L, L, 3))
    rgb[:, :, 0] = 0.05; rgb[:, :, 1] = 0.07; rgb[:, :, 2] = 0.14
    free = (occ == 1) & (~jam)
    rgb[free, 0] = 0.20; rgb[free, 1] = 0.60; rgb[free, 2] = 0.86
    jm = (occ == 1) & jam
    rgb[jm, 0] = 0.91; rgb[jm, 1] = 0.30; rgb[jm, 2] = 0.24
    return rgb

def run_method(method, output_path, L=64, density=0.55,
               v_plus=10.0, v_zero=0.5, v_minus=0.1, D_rot=0.2,
               n_frames=600, steps_per_frame=60, tau=0.01,
               seed=123, fps=30):
    
    Pe = (v_plus - v_minus) / (2 * D_rot)
    N = int(density * L * L)
    method_label = "Gillespie (Tau-leaping OFF)" if method == "gillespie" else f"Tau-leap (ON, tau={tau})"
    
    print(f"\n{'='*60}")
    print(f"  [{method_label}] L={L}, ρ={density}, Pe={Pe:.1f}, N={N}")
    print(f"{'='*60}")

    sim = LatticeABP(
        L=L, v_plus=v_plus, v_zero=v_zero, v_minus=v_minus,
        D_rot=D_rot, density=density, bc_mode="periodic",
        device="auto", seed=seed,
    )
    O, E = sim.init_state(B=1)

    print(f"Collecting {n_frames} frames from t=0...")
    frames_occ = []
    frames_jammed = []

    jammed_0 = sim.compute_jammed_mask(O, E)
    frames_occ.append(O[0].cpu().numpy().copy())
    frames_jammed.append(jammed_0[0].cpu().numpy().copy())

    t0 = time.time()
    for f in range(1, n_frames):
        for _ in range(steps_per_frame):
            if method == "gillespie":
                O, E, _ = sim.gillespie_step(O, E)
            else:
                O, E = sim.tau_leap_step(O, E, tau=tau)

        jammed = sim.compute_jammed_mask(O, E)
        frames_occ.append(O[0].cpu().numpy().copy())
        frames_jammed.append(jammed[0].cpu().numpy().copy())

        if f % 100 == 0:
            elapsed = time.time() - t0
            eta = elapsed / f * (n_frames - f)
            n_jam = (jammed[0] & (O[0] == 1)).sum().item()
            print(f"  frame {f}/{n_frames} — jammed: {n_jam}/{N} ({100*n_jam/N:.1f}%) — ETA: {eta:.0f}s")
            
    prod_time = time.time() - t0
    print(f"  Simulation completed in {prod_time:.1f}s")

    print(f"Rendering → {output_path}")
    fig, ax = plt.subplots(figsize=(8, 8), facecolor="#0d1117")
    ax.set_facecolor("#0d1117")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_aspect("equal")

    im = ax.imshow(build_rgb(frames_occ[0], frames_jammed[0], L), interpolation="nearest")

    title = ax.set_title("", color="white", fontsize=13, fontweight="bold", pad=10)
    frac_text = ax.text(
        0.02, 0.02, "", transform=ax.transAxes,
        color="white", fontsize=11, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#1a1a2e", edgecolor="#444", alpha=0.85),
        verticalalignment="bottom",
    )
    ax.legend(
        handles=[Patch(facecolor="#e74c3c", label="Jammed"), Patch(facecolor="#3498db", label="Free")],
        loc="upper right", fontsize=10, facecolor="#1a1a2e", edgecolor="#444", labelcolor="white",
    )

    def update(idx):
        im.set_data(build_rgb(frames_occ[idx], frames_jammed[idx], L))
        n_jam = (frames_jammed[idx] & (frames_occ[idx] == 1)).sum()
        frac_text.set_text(f"Jammed: {100*n_jam/N:.1f}%")
        step = idx * steps_per_frame
        title.set_text(f"{method_label} — step {step:,}")
        return [im, frac_text, title]

    anim = animation.FuncAnimation(fig, update, frames=n_frames, interval=1000//fps, blit=False)
    writer = animation.FFMpegWriter(fps=fps, bitrate=2500)
    anim.save(output_path, writer=writer, dpi=120, savefig_kwargs={"facecolor": fig.get_facecolor()})
    plt.close(fig)

    print(f"  Saved: {output_path}")
    return prod_time

if __name__ == "__main__":
    params = dict(
        L=64, density=0.55, v_plus=10.0, v_zero=0.5, v_minus=0.1, D_rot=0.2,
        n_frames=600, steps_per_frame=60, fps=30
    )

    t1 = run_method("gillespie", "output/mips_gillespie_off.mp4", seed=42, **params)
    t2 = run_method("tau_leap", "output/mips_tau_leap_on.mp4", tau=0.01, seed=42, **params)

    print(f"\n{'='*60}")
    print(f"Comparison Summary:")
    print(f"  Gillespie (Tau-leaping OFF) time: {t1:.1f}s")
    print(f"  Tau-leaping ON (tau=0.01) time:   {t2:.1f}s")
    print(f"  Speedup factor: {t1/t2:.2f}x")
    print(f"{'='*60}")
