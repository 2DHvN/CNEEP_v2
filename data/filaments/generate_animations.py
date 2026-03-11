"""
Filament Brownian Movie (Pixel-level, RAW-consistent, FIXED SITES)

- Filament-only visualization (bond density)
- Fixed boundary sites respected
- No cv2, no codec interpolation
- MP4 via matplotlib only
- interpolation = nearest
"""

import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from generate_trajectories import SquareLatticeFilamentModel


# ============================================================
# Filament Gaussian renderer
# ============================================================

def draw_filament_gaussian(p0, p1, sigma, xx, yy):
    x0, y0 = p0
    x1, y1 = p1

    vx = x1 - x0
    vy = y1 - y0
    L2 = vx * vx + vy * vy + 1e-12

    t = ((xx - x0) * vx + (yy - y0) * vy) / L2
    t = np.clip(t, 0.0, 1.0)

    cx = x0 + t * vx
    cy = y0 + t * vy

    dist2 = (xx - cx) ** 2 + (yy - cy) ** 2
    return np.exp(-dist2 / (2 * sigma ** 2))


# ============================================================
# RAW filament frame generation
# ============================================================

def generate_filament_frames(
    trajectory_flat: np.ndarray,
    bonds: np.ndarray,
    lattice_shape: tuple,
    cell_size: int,
    skip_frames: int,
    sigma: float
):
    """
    trajectory_flat : (T, 2N)
    bonds           : (N_bonds, 2)
    lattice_shape   : (Lx, Ly)
    """

    T, D, _ = trajectory_flat.shape
    Lx, Ly = lattice_shape
    N = Lx * Ly

    A = cell_size
    H = (Ly-1) * A
    W = (Lx-1) * A

    yy, xx = np.meshgrid(
        np.arange(H),
        np.arange(W),
        indexing="ij"
    )

    frames = []

    for pos_flat in trajectory_flat[::skip_frames]:
        pos = pos_flat.reshape(N, 2)

        frame = np.zeros((H, W, 3), dtype=np.float32)

        # lattice → pixel coordinates
        pixel_pos = np.zeros((N, 2))
        for i in range(N):
            px = pos[i, 0] * A
            py = pos[i, 1] * A
            pixel_pos[i] = (px, py)

        # render filaments
        for i, j in bonds:
            blob = draw_filament_gaussian(
                pixel_pos[i],
                pixel_pos[j],
                sigma,
                xx,
                yy
            )
            frame[..., 1] = np.maximum(frame[..., 1], blob)

        if frame.max() > 0:
            frame /= frame.max()

        frames.append((255 * frame).astype(np.uint8))

    return np.stack(frames)


# ============================================================
# MP4 generation (viewer only)
# ============================================================

def save_frames_as_mp4_mlp(frames, fps, output_path):
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

    writer = animation.FFMpegWriter(fps=fps)
    ani.save(output_path, writer=writer)
    plt.close(fig)

    print(f"[OK] MP4 saved → {output_path}")


# ============================================================
# EPR time series & cumulative EP plot
# ============================================================

def save_epr_timeseries(
    trajectory: np.ndarray,
    model: 'SquareLatticeFilamentModel',
    output_path: str,
    skip_frames: int = 1,
):
    """
    Compute total EPR at each time step and save a time-series plot.
    Shows instantaneous EPR and cumulative EP (integrated entropy production).
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
# Mean EPR map (per-site, time-averaged)
# ============================================================

def save_mean_epr_map(
    trajectory: np.ndarray,
    model: 'SquareLatticeFilamentModel',
    output_path: str,
    cmap_name: str = "hot",
):
    """
    Compute time-averaged per-site EPR and display as a lattice heatmap.
    """
    ep_per_site = model.compute_entropy_production_per_site(trajectory)

    mean_ep = ep_per_site.mean(axis=0)  # (N,)

    # reshape to lattice for visualization
    ep_grid = mean_ep.reshape(model.Lx, model.Ly)

    fig, ax = plt.subplots(figsize=(6, 5))
    vmax = np.max(np.abs(ep_grid))
    vmax = max(vmax, 1e-12)
    im = ax.imshow(
        ep_grid, origin="lower", cmap=cmap_name,
        vmin=-vmax, vmax=vmax, interpolation="nearest"
    )
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label(r"$\langle \sigma_i \rangle_t$", fontsize=12)
    ax.set_title("Time-averaged per-site EPR", fontsize=12)
    ax.set_xlabel("ix")
    ax.set_ylabel("iy")

    # Mark hot sites
    for i in range(model.N):
        ix = i // model.Ly
        iy = i % model.Ly
        if model.T[i] > model.T.min() + 0.01:
            ax.plot(ix, iy, 'r.', markersize=4)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    print(f"[OK] Mean EPR map saved → {output_path}")
    print(f"     spatial mean ⟨σ⟩ = {mean_ep.mean():.6e}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser("Filament lattice movie (fixed sites)")

    parser.add_argument("--Lx", type=int, default=6)
    parser.add_argument("--Ly", type=int, default=6)
    parser.add_argument("--n_steps", type=int, default=100000)
    parser.add_argument("--burn_in", type=int, default=0)

    parser.add_argument("--k", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=10000.0)
    parser.add_argument("--T_hot", type=float, default=2.0)
    parser.add_argument("--T_cold", type=float, default=1.0)
    parser.add_argument("--dt", type=float, default=0.0001)

    parser.add_argument("--cell_size", type=int, default=10)
    parser.add_argument("--skip_frames", type=int, default=100)
    parser.add_argument("--sigma", type=float, default=1)
    parser.add_argument("--fps", type=int, default=20)

    parser.add_argument("--output", type=str, default="filament_movie.mp4")
    parser.add_argument("--epr_timeseries", type=str, default="",
                        help="If set, save EPR time series + cumulative EP plot as PNG")
    parser.add_argument("--mean_epr_output", type=str, default="",
                        help="If set, save time-averaged per-site EPR map as PNG")
    parser.add_argument("--save_npz", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    np.random.seed(args.seed)

    model = SquareLatticeFilamentModel(
        Lx=args.Lx,
        Ly=args.Ly,
        k=args.k,
        gamma=args.gamma,
        T_hot=args.T_hot,
        T_cold=args.T_cold,
        dt=args.dt
    )

    trajectory = model.generate_trajectory(
        n_steps=args.n_steps,
        burn_in=args.burn_in
    )

    print(f"[INFO] Trajectory shape: {trajectory.shape}")

    # --- Mean EPR ---
    mean_epr = model.compute_mean_entropy_production(trajectory)
    print(f"[EPR] Mean total EPR: {mean_epr:.6e}")

    # --- Filament movie ---
    frames = generate_filament_frames(
        trajectory_flat=trajectory,
        bonds=model.bonds,
        lattice_shape=(args.Lx, args.Ly),
        cell_size=args.cell_size,
        skip_frames=args.skip_frames,
        sigma=args.sigma
    )

    save_frames_as_mp4_mlp(frames, args.fps, args.output)

    # --- EPR time series + cumulative EP (optional) ---
    if args.epr_timeseries:
        print("[INFO] Computing EPR time series ...")
        save_epr_timeseries(
            trajectory, model,
            output_path=args.epr_timeseries,
            skip_frames=args.skip_frames,
        )

    # --- Mean EPR map (optional) ---
    if args.mean_epr_output:
        print("[INFO] Computing mean EPR map ...")
        save_mean_epr_map(
            trajectory, model,
            output_path=args.mean_epr_output,
        )

    # --- Save NPZ ---
    if args.save_npz:
        np.savez(
            os.path.splitext(args.output)[0] + ".npz",
            frames=frames,
            trajectory=trajectory,
            bonds=model.bonds,
            lattice_shape=(args.Lx, args.Ly)
        )

    print("Done.")


if __name__ == "__main__":
    main()

