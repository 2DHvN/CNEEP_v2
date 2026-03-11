"""
N-Beads Brownian Movie (Pixel-level, RAW-consistent)

- Raw pixel frames generation (data-truth)
- No cv2, no codec-side interpolation
- MP4 generated ONLY via matplotlib viewer
- interpolation = nearest (pixel-faithful)
- Frames + trajectory saved as npz
"""

import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from generate_trajectories import NBeadsModel


# ============================================================
# Raw frame generation
# ============================================================

def generate_brownian_frames(
    trajectory: np.ndarray,
    cell_size: int,
    skip_frames: int=1,
    sigma: float = 2.0
):
    """
    Generate raw pixel frames.
    Returns
    -------
    frames : (T, H, W, 3) uint8
    """
    n_steps, n_beads = trajectory.shape
    A = cell_size

    H = A
    W = n_beads * A

    cell_centers_x = np.array([A // 2 + i * A for i in range(n_beads)])
    cell_center_y = A // 2

    traj_scale = (A / 4) / max(
        abs(trajectory.min()),
        abs(trajectory.max()),
        1.0
    )

    yy, xx = np.meshgrid(
        np.arange(H),
        np.arange(W),
        indexing="ij"
    )

    frames = []

    for positions in trajectory[::skip_frames]:
        frame = np.zeros((H, W, 3), dtype=np.float32)

        for i, x in enumerate(positions):
            cx = cell_centers_x[i] + x * traj_scale
            cy = cell_center_y

            blob = np.exp(
                -((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2)
            )

            frame[..., 1] += blob  # green channel only

        if frame.max() > 0:
            frame /= frame.max()

        frame = (255 * frame).astype(np.uint8)
        frames.append(frame)

    return np.stack(frames)


# ============================================================
# MP4 generation via matplotlib (viewer only)
# ============================================================

def save_frames_as_mp4_mlp(
    frames: np.ndarray,
    fps: int,
    output_path: str
):
    """
    Save frames to mp4 using matplotlib viewer.
    Raw pixel fidelity is preserved.
    """
    T, H, W, _ = frames.shape

    fig, ax = plt.subplots()
    ax.axis("off")

    im = ax.imshow(
        frames[0],
        interpolation="nearest",
        animated=True
    )

    def update(i):
        im.set_array(frames[i])
        return (im,)

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=T,
        interval=1000 / fps,
        blit=True
    )

    writer = animation.FFMpegWriter(
        fps=fps,
    )

    ani.save(output_path, writer=writer)
    plt.close(fig)

    print(f"[OK] MP4 saved via matplotlib → {output_path}")


# ============================================================
# EPR time series & cumulative EP plot
# ============================================================

def save_epr_timeseries(
    trajectory: np.ndarray,
    model: 'NBeadsModel',
    output_path: str,
    skip_frames: int = 1,
):
    """
    Compute total EPR at each time step and save a time-series plot.
    Shows instantaneous EPR and cumulative EP.
    """
    ep_rate = model.compute_entropy_production_rate(trajectory)

    T = len(ep_rate)
    T_sub = T // skip_frames
    
    ep_rate_sub = np.zeros(T_sub)
    cumulative_ep_sub = np.zeros(T_sub)
    
    cumulative_ep = np.cumsum(ep_rate) * model.dt
    
    for i in range(T_sub):
        ep_rate_sub[i] = np.mean(ep_rate[i * skip_frames:(i + 1) * skip_frames])
        cumulative_ep_sub[i] = np.mean(cumulative_ep[i * skip_frames:(i + 1) * skip_frames])

    dt_eff = model.dt * skip_frames
    time_axis = np.arange(T_sub) * dt_eff

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    # Instantaneous EPR
    axes[0].plot(time_axis, ep_rate_sub, lw=0.4, alpha=0.7, color="steelblue")
    axes[0].set_ylabel(r"$\dot{S}_{\mathrm{tot}}(t)$", fontsize=12)
    axes[0].set_title("Total entropy production rate vs time", fontsize=12)
    axes[0].axhline(0, color="k", lw=0.5, ls="--")

    # Cumulative EP
    axes[1].plot(time_axis, cumulative_ep_sub, lw=1.5, color="crimson")
    axes[1].set_ylabel(r"$\sum \Delta S$", fontsize=12)
    axes[1].set_xlabel("Time", fontsize=12)
    axes[1].set_title("Cumulative entropy production", fontsize=12)
    axes[1].axhline(0, color="k", lw=0.5, ls="--")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    print(f"[OK] EPR time series saved → {output_path}")
    print(f"     mean EPR = {ep_rate.mean():.6e}")
    print(f"     final cumulative EP = {cumulative_ep[-1]:.6e}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="N-Beads Brownian Movie (RAW pixel-consistent)"
    )

    parser.add_argument("--n_beads", type=int, default=2)
    parser.add_argument("--n_steps", type=int, default=10000)
    parser.add_argument("--burn_in", type=int, default=10)

    parser.add_argument("--k", type=float, default=1)
    parser.add_argument("--T_hot", type=float, default=10.0)
    parser.add_argument("--T_cold", type=float, default=1.0)
    parser.add_argument("--dt", type=float, default=0.01)

    parser.add_argument("--cell_size", type=int, default=20)
    parser.add_argument("--skip_frames", type=int, default=1)
    parser.add_argument("--sigma", type=float, default=3.0)
    parser.add_argument("--fps", type=int, default=20)

    parser.add_argument("--output", type=str, default="brownian_movie.mp4")
    parser.add_argument("--epr_timeseries", type=str, default="",
                        help="If set, save EPR time series + cumulative EP plot as PNG")
    parser.add_argument("--save_npz", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    np.random.seed(args.seed)

    print(f"[INFO] Generating {args.n_beads}-beads trajectory")

    model = NBeadsModel(
        n_beads=args.n_beads,
        k=args.k,
        T_hot=args.T_hot,
        T_cold=args.T_cold,
        dt=args.dt
    )

    trajectory = model.generate_trajectory(
        n_steps=args.n_steps,
        burn_in=args.burn_in
    )

    mean_epr = model.compute_mean_entropy_production(trajectory)
    print(f"[EPR] Mean total EPR: {mean_epr:.6e}")
    print(f"[INFO] Trajectory shape: {trajectory.shape}")

    frames = generate_brownian_frames(
        trajectory=trajectory,
        cell_size=args.cell_size,
        skip_frames=args.skip_frames,
        sigma=args.sigma
    )

    print(f"[INFO] Frames shape: {frames.shape}")

    save_frames_as_mp4_mlp(
        frames=frames,
        fps=args.fps,
        output_path=args.output
    )

    # --- EPR time series + cumulative EP (optional) ---
    if args.epr_timeseries:
        print("[INFO] Computing EPR time series ...")
        save_epr_timeseries(
            trajectory, model,
            output_path=args.epr_timeseries,
            skip_frames=args.skip_frames,
        )

    if args.save_npz:
        npz_path = os.path.splitext(args.output)[0] + ".npz"
        np.savez(
            npz_path,
            frames=frames,
            trajectory=trajectory,
            cell_size=args.cell_size,
            skip_frames=args.skip_frames,
            sigma=args.sigma
        )
        print(f"[OK] Saved dataset → {npz_path}")

    print("Done.")


if __name__ == "__main__":
    main()

