"""
SAOU Field — Magnitude and Direction Animation Generator

Renders trajectory animations for the 2-component (u, v) Ornstein-Uhlenbeck field
by separating it into:
  - Left panel: Field Magnitude heatmap (color-mapped using 'magma')
  - Right panel: Field Direction angle heatmap (color-mapped using 'twilight')
                 with overlaid uniform rotating arrows (quiver plot).

Outputs MP4 videos.

Usage:
    python generate_mag_dir_animation.py
    python generate_mag_dir_animation.py --input saou_trajectories.npz --fps 15
    python generate_mag_dir_animation.py --L 32 --n_steps 20000 --omega0 1.0 --output saou_mag_dir.mp4
"""

import numpy as np
import argparse
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as mcolors

from saou_model import simulate


def save_mag_dir_animation(
    trajectory: np.ndarray,
    fps: int,
    output_path: str,
    arrow_stride: int = 2,
    arrow_scale: float = 25.0,
    max_frames: int = 500,
):
    """
    Save a 2-panel animation:
      - Left: Field Magnitude (heatmap using 'magma')
      - Right: Direction Field (angle heatmap using 'twilight', with rotating arrows overlaid)
    """
    T_total = trajectory.shape[0]
    skip = max(1, T_total // max_frames)
    traj = trajectory[::skip]
    n_frames = traj.shape[0]
    L = traj.shape[1]

    # Compute magnitude and angle for the entire trajectory subset
    u = traj[..., 0]
    v = traj[..., 1]
    mag = np.sqrt(u**2 + v**2)
    angle = np.arctan2(v, u)

    # Set up figure
    # We want a dark theme for a premium look
    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=150)
    fig.patch.set_facecolor('#0f0f12')  # dark slate background
    ax1.set_facecolor('#0f0f12')
    ax2.set_facecolor('#0f0f12')

    # Color limits
    mag_vmin = 0.0
    mag_vmax = max(1.5, mag.max() * 0.8)  # scaling vmax slightly for better contrast

    # Heatmaps
    im_mag = ax1.imshow(
        mag[0],
        vmin=mag_vmin,
        vmax=mag_vmax,
        cmap="magma",
        origin="lower",
        extent=[0, L, 0, L]
    )
    
    # Direction is shown via arrows only, so we set explicitly the limits and aspect
    ax2.set_xlim(0, L)
    ax2.set_ylim(0, L)
    ax2.set_aspect('equal')

    # Titles and labels
    ax1.set_title("Field Magnitude $\\sqrt{u^2 + v^2}$", fontsize=12, pad=10, color='#e0e0e6')
    ax2.set_title("Field Direction $\\theta = \\arctan2(v, u)$", fontsize=12, pad=10, color='#e0e0e6')

    for ax in (ax1, ax2):
        ax.set_xticks([])
        ax.set_yticks([])
        # Set border color to a dark grey
        for spine in ax.spines.values():
            spine.set_color('#2e2e36')

    # Colorbars
    cb1 = fig.colorbar(im_mag, ax=ax1, orientation='vertical', fraction=0.046, pad=0.04)
    cb1.ax.yaxis.set_tick_params(color='#e0e0e6', labelcolor='#e0e0e6')
    cb1.outline.set_edgecolor('#2e2e36')

    # Quiver setup
    # Create coordinate grid for arrows (centering arrows in grid cells)
    y, x = np.meshgrid(np.arange(L) + 0.5, np.arange(L) + 0.5)
    
    # Apply stride to coordinates
    x_sub = x[::arrow_stride, ::arrow_stride]
    y_sub = y[::arrow_stride, ::arrow_stride]

    # Compute normalized arrows for initial frame
    u_init = u[0] / (mag[0] + 1e-8)
    v_init = v[0] / (mag[0] + 1e-8)
    
    u_sub_init = u_init[::arrow_stride, ::arrow_stride]
    v_sub_init = v_init[::arrow_stride, ::arrow_stride]

    # Quiver plot overlaid on the direction map
    # We use semi-transparent white arrows for clean visual design
    q = ax2.quiver(
        x_sub, y_sub,
        u_sub_init, v_sub_init,
        color=(1.0, 1.0, 1.0, 0.8),
        pivot='middle',
        scale=arrow_scale,
        scale_units='width',
        width=0.003
    )

    # Adjust spacing
    fig.tight_layout(pad=1.5)

    # Animation update function
    def update(frame_idx):
        # Update magnitude image
        im_mag.set_data(mag[frame_idx])
        
        # Update quiver arrows
        u_f = u[frame_idx]
        v_f = v[frame_idx]
        mag_f = mag[frame_idx]
        
        u_norm = u_f / (mag_f + 1e-8)
        v_norm = v_f / (mag_f + 1e-8)
        
        u_sub = u_norm[::arrow_stride, ::arrow_stride]
        v_sub = v_norm[::arrow_stride, ::arrow_stride]
        
        q.set_UVC(u_sub, v_sub)
        
        return [im_mag, q]

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=n_frames,
        blit=True
    )

    # Save to file
    writer = animation.FFMpegWriter(fps=fps)
    ani.save(output_path, writer=writer)
    plt.close(fig)
    print(f"[OK] Magnitude/Direction animation saved → {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="SAOU Field — magnitude/direction animation generator"
    )

    # Source: either load from npz or simulate fresh
    parser.add_argument("--input", type=str, default="",
                        help="Path to .npz file from generate_trajectories.py")

    # Simulation params (used if --input is not given)
    parser.add_argument("--L", type=int, default=32)
    parser.add_argument("--radii", type=float, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--amplitudes", type=float, nargs="+", default=[1.0, 0.5, 2.0, 0.0])
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--omega0", type=float, default=0.0)
    parser.add_argument("--T", type=float, default=1.0)
    parser.add_argument("--dt", type=float, default=1e-3)
    parser.add_argument("--n_steps", type=int, default=20000)
    parser.add_argument("--burn_steps", type=int, default=10000)
    parser.add_argument("--sample_every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1)

    # Rendering
    parser.add_argument("--skip_frames", type=int, default=1)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--max_frames", type=int, default=500)
    parser.add_argument("--arrow_stride", type=int, default=1,
                        help="Plot every Nth arrow (default: 1)")
    parser.add_argument("--arrow_scale", type=float, default=25.0,
                        help="Quiver arrow scale (smaller value = larger arrows, default: 25.0)")

    # Outputs
    parser.add_argument("--output", type=str, default="saou_mag_dir_movie.mp4",
                        help="Output movie path (default: saou_mag_dir_movie.mp4)")

    args = parser.parse_args()

    # Load or simulate
    if args.input and os.path.exists(args.input):
        import json as _json
        print(f"[INFO] Loading trajectory from {args.input}")
        data = np.load(args.input, allow_pickle=True)
        trajectory = data["trajectory"]
        params = _json.loads(str(data["params_json"]))
        print(f"[INFO] Trajectory shape: {trajectory.shape}")
    else:
        print("[INFO] No input file; running fresh simulation ...")
        out = simulate(
            L=args.L,
            radii=tuple(args.radii),
            amplitudes=tuple(args.amplitudes),
            gamma=args.gamma,
            omega0=args.omega0,
            T=args.T,
            dt=args.dt,
            n_steps=args.n_steps,
            burn_steps=args.burn_steps,
            sample_every=args.sample_every,
            seed=args.seed,
            record_trajectory=True,
            show_progress=True,
        )
        trajectory = out["trajectory"]
        print(f"[INFO] Trajectory shape: {trajectory.shape}")

    # Generate outputs
    print("[INFO] Rendering magnitude/direction frames ...")
    save_mag_dir_animation(
        trajectory=trajectory,
        fps=args.fps,
        output_path=args.output,
        arrow_stride=args.arrow_stride,
        arrow_scale=args.arrow_scale,
        max_frames=args.max_frames,
    )
    print("\nDone.")


if __name__ == "__main__":
    main()
