"""
Generate MIPS animation video for Lattice ABP.

Uses high-Pe, moderate-density regime to observe
Motility-Induced Phase Separation (MIPS).

Parameters tuned for clear cluster formation:
  - v+ = 10.0, v- = 0.1 → strong persistence
  - D_rot = 0.2 → slow reorientation
  - Pe = (v+ - v-) / (2*D_rot) ≈ 24.75
  - ρ = 0.55 → above critical density for MIPS
  - L = 64 → large enough for macroscopic clusters
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap
from core import LatticeABP


def generate_mips_video(
    L=64,
    density=0.55,
    v_plus=10.0,
    v_zero=0.5,
    v_minus=0.1,
    D_rot=0.2,
    burn_in=5000,
    n_frames=300,
    steps_per_frame=50,
    seed=123,
    output_path="output/lattice_abp_mips.mp4",
    fps=20,
):
    """Run simulation and render MIPS animation."""

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    Pe = (v_plus - v_minus) / (2 * D_rot)
    N = int(density * L * L)

    print("=" * 60)
    print("  MIPS Animation Generator — Lattice ABP")
    print("=" * 60)
    print(f"  L={L}, ρ={density:.2f}, N={N}")
    print(f"  v+={v_plus}, v0={v_zero}, v-={v_minus}")
    print(f"  D_rot={D_rot}, Pe={Pe:.2f}")
    print(f"  Burn-in: {burn_in} steps")
    print(f"  Frames: {n_frames} × {steps_per_frame} steps/frame")
    print(f"  Total production steps: {n_frames * steps_per_frame}")
    print("=" * 60)

    sim = LatticeABP(
        L=L,
        v_plus=v_plus,
        v_zero=v_zero,
        v_minus=v_minus,
        D_rot=D_rot,
        density=density,
        bc_mode="periodic",
        device="auto",
        seed=seed,
    )

    print(f"\nDevice: {sim.device}")

    # Initialize
    O, E = sim.init_state(B=1)

    # Burn-in
    print(f"\nBurn-in ({burn_in} steps)...")
    t0 = time.time()
    for i in range(burn_in):
        O, E, _ = sim.gillespie_step(O, E)
        if (i + 1) % 1000 == 0:
            jammed = sim.compute_jammed_mask(O, E)
            n_jam = (jammed & (O == 1)).sum().item()
            print(f"  step {i+1}/{burn_in} — jammed: {n_jam}/{N} "
                  f"({100*n_jam/N:.1f}%)")
    print(f"  Burn-in done in {time.time()-t0:.1f}s")

    # Collect frames
    print(f"\nCollecting {n_frames} frames...")
    frames_occ = []
    frames_jammed = []

    t0 = time.time()
    for f in range(n_frames):
        for _ in range(steps_per_frame):
            O, E, _ = sim.gillespie_step(O, E)

        jammed = sim.compute_jammed_mask(O, E)
        frames_occ.append(O[0].cpu().numpy().copy())
        frames_jammed.append(jammed[0].cpu().numpy().copy())

        if (f + 1) % 50 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (f + 1) * (n_frames - f - 1)
            n_jam = (jammed[0] & (O[0] == 1)).sum().item()
            print(f"  frame {f+1}/{n_frames} — jammed: {n_jam}/{N} "
                  f"({100*n_jam/N:.1f}%) — ETA: {eta:.0f}s")

    print(f"  Frames collected in {time.time()-t0:.1f}s")

    # --- Render animation ---
    print(f"\nRendering video → {output_path}")

    # Color map: empty=dark, free=blue, jammed=red
    fig, ax = plt.subplots(figsize=(8, 8), facecolor="#0d1117")
    ax.set_facecolor("#0d1117")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")

    # Build RGB image from occupancy + jammed
    def build_frame_rgb(occ, jam):
        """Return (L, L, 3) float RGB array."""
        rgb = np.zeros((L, L, 3))
        # Background: dark navy
        rgb[:, :, 0] = 0.05
        rgb[:, :, 1] = 0.07
        rgb[:, :, 2] = 0.14

        # Free particles: blue (#3498db)
        free = (occ == 1) & (~jam)
        rgb[free, 0] = 0.20
        rgb[free, 1] = 0.60
        rgb[free, 2] = 0.86

        # Jammed particles: red (#e74c3c)
        jammed_mask = (occ == 1) & jam
        rgb[jammed_mask, 0] = 0.91
        rgb[jammed_mask, 1] = 0.30
        rgb[jammed_mask, 2] = 0.24

        return rgb

    img_data = build_frame_rgb(frames_occ[0], frames_jammed[0])
    im = ax.imshow(img_data, interpolation="nearest")

    title = ax.set_title(
        f"Lattice ABP — L={L}, ρ={density}, Pe={Pe:.1f}  |  frame 0",
        color="white", fontsize=13, fontweight="bold", pad=10,
    )

    # Jammed fraction text
    frac_text = ax.text(
        0.02, 0.02, "", transform=ax.transAxes,
        color="white", fontsize=11, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#1a1a2e",
                  edgecolor="#444", alpha=0.85),
        verticalalignment="bottom",
    )

    # Legend patches
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#e74c3c", label="Jammed"),
        Patch(facecolor="#3498db", label="Free"),
    ]
    ax.legend(
        handles=legend_elements, loc="upper right",
        fontsize=10, facecolor="#1a1a2e", edgecolor="#444",
        labelcolor="white",
    )

    def update(frame_idx):
        rgb = build_frame_rgb(frames_occ[frame_idx], frames_jammed[frame_idx])
        im.set_data(rgb)

        n_jam = (frames_jammed[frame_idx] & (frames_occ[frame_idx] == 1)).sum()
        frac = n_jam / N * 100
        frac_text.set_text(f"Jammed: {frac:.1f}%")

        mc_step = burn_in + (frame_idx + 1) * steps_per_frame
        title.set_text(
            f"Lattice ABP — L={L}, ρ={density}, Pe={Pe:.1f}  |  "
            f"step {mc_step:,}"
        )
        return [im, frac_text, title]

    anim = animation.FuncAnimation(
        fig, update, frames=n_frames,
        interval=1000 // fps, blit=False,
    )

    writer = animation.FFMpegWriter(fps=fps, bitrate=2000)
    anim.save(output_path, writer=writer, dpi=120,
              savefig_kwargs={"facecolor": fig.get_facecolor()})
    plt.close(fig)

    file_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\nDone! Video saved: {output_path} ({file_size:.1f} MB)")
    print(f"  {n_frames} frames @ {fps} fps = {n_frames/fps:.1f}s duration")


if __name__ == "__main__":
    generate_mips_video()
