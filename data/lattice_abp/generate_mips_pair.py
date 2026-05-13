"""
Generate two MIPS videos: Periodic BC vs Hard Wall BC.

Starting from t=0 (no burn-in) to observe the MIPS formation process.
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


def run_and_save(bc_mode, output_path, L=64, density=0.55,
                 v_plus=10.0, v_zero=0.5, v_minus=0.1, D_rot=0.2,
                 n_frames=800, steps_per_frame=50,
                 seed=123, fps=30):

    Pe = (v_plus - v_minus) / (2 * D_rot)
    N = int(density * L * L)
    bc_label = "Periodic" if bc_mode == "periodic" else "Hard Wall"

    print(f"\n{'='*60}")
    print(f"  [{bc_label} BC]  L={L}, ρ={density}, Pe={Pe:.1f}, N={N}")
    print(f"  Start from t=0, frames={n_frames}, steps/frame={steps_per_frame}")
    print(f"  total steps = {n_frames * steps_per_frame}")
    print(f"{'='*60}")

    sim = LatticeABP(
        L=L, v_plus=v_plus, v_zero=v_zero, v_minus=v_minus,
        D_rot=D_rot, density=density, bc_mode=bc_mode,
        device="auto", seed=seed,
    )
    O, E = sim.init_state(B=1)

    # --- Collect frames ---
    print(f"\nCollecting {n_frames} frames from t=0...")
    frames_occ = []
    frames_jammed = []

    # Save t=0 frame first
    jammed_0 = sim.compute_jammed_mask(O, E)
    frames_occ.append(O[0].cpu().numpy().copy())
    frames_jammed.append(jammed_0[0].cpu().numpy().copy())

    t0 = time.time()
    for f in range(1, n_frames):
        for _ in range(steps_per_frame):
            O, E, _ = sim.gillespie_step(O, E)

        jammed = sim.compute_jammed_mask(O, E)
        frames_occ.append(O[0].cpu().numpy().copy())
        frames_jammed.append(jammed[0].cpu().numpy().copy())

        if f % 100 == 0:
            elapsed = time.time() - t0
            eta = elapsed / f * (n_frames - f)
            n_jam = (jammed[0] & (O[0] == 1)).sum().item()
            print(f"  frame {f}/{n_frames} — jammed: {n_jam}/{N} "
                  f"({100*n_jam/N:.1f}%) — ETA: {eta:.0f}s")
    prod_time = time.time() - t0
    print(f"  Frames collected in {prod_time:.1f}s")

    # --- Render ---
    print(f"\nRendering → {output_path}")
    fig, ax = plt.subplots(figsize=(8, 8), facecolor="#0d1117")
    ax.set_facecolor("#0d1117")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_aspect("equal")

    im = ax.imshow(build_rgb(frames_occ[0], frames_jammed[0], L),
                   interpolation="nearest")

    title = ax.set_title("", color="white", fontsize=13, fontweight="bold", pad=10)
    frac_text = ax.text(
        0.02, 0.02, "", transform=ax.transAxes,
        color="white", fontsize=11, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#1a1a2e",
                  edgecolor="#444", alpha=0.85),
        verticalalignment="bottom",
    )
    ax.legend(
        handles=[Patch(facecolor="#e74c3c", label="Jammed"),
                 Patch(facecolor="#3498db", label="Free")],
        loc="upper right", fontsize=10,
        facecolor="#1a1a2e", edgecolor="#444", labelcolor="white",
    )

    def update(idx):
        im.set_data(build_rgb(frames_occ[idx], frames_jammed[idx], L))
        n_jam = (frames_jammed[idx] & (frames_occ[idx] == 1)).sum()
        frac_text.set_text(f"Jammed: {100*n_jam/N:.1f}%")
        step = idx * steps_per_frame
        title.set_text(f"{bc_label} BC — L={L}, ρ={density}, Pe={Pe:.1f}  |  "
                       f"step {step:,}")
        return [im, frac_text, title]

    anim = animation.FuncAnimation(fig, update, frames=n_frames,
                                   interval=1000 // fps, blit=False)
    writer = animation.FFMpegWriter(fps=fps, bitrate=2500)
    anim.save(output_path, writer=writer, dpi=120,
              savefig_kwargs={"facecolor": fig.get_facecolor()})
    plt.close(fig)

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    duration = n_frames / fps
    print(f"  Saved: {output_path} ({size_mb:.1f} MB, {duration:.1f}s @ {fps}fps)")
    return prod_time


if __name__ == "__main__":
    total_t0 = time.time()

    # Total steps = 100,000, starting from 0.
    params = dict(
        L=64, density=0.70,
        v_plus=10.0, v_zero=0.5, v_minus=0.1, D_rot=0.2,
        n_frames=1000, steps_per_frame=100,
        fps=30,
    )

    # 1) Periodic BC
    t1 = run_and_save(
        bc_mode="periodic",
        output_path="output/mips_periodic_from_zero_high_density.mp4",
        seed=123, **params,
    )

    # 2) Hard Wall BC
    t2 = run_and_save(
        bc_mode="hard_wall",
        output_path="output/mips_hard_wall_from_zero_high_density.mp4",
        seed=123, **params,
    )

    total = time.time() - total_t0
    print(f"\n{'='*60}")
    print(f"  All done! Total time: {total/60:.1f} min")
    print(f"  Periodic:  output/mips_periodic_from_zero.mp4")
    print(f"  Hard Wall: output/mips_hard_wall_from_zero.mp4")
    print(f"{'='*60}")
