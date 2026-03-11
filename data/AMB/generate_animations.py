"""
Active Model B — Brownian Movie (density field + EPR map)

- Renders the φ(r,t) density field as a colormap animation
- Optionally renders local entropy production rate density map
- MP4 via matplotlib (same pattern as beads / filaments)
- interpolation = nearest (pixel-faithful)
"""

import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as mcolors

from generate_trajectories import ActiveModelB


# ============================================================
# Frame generation — density field φ
# ============================================================

def generate_density_frames(
    trajectory: np.ndarray,
    skip_frames: int = 1,
    cmap_name: str = "magma",
    symmetric: bool = True,
) -> np.ndarray:
    """
    Render φ(r,t) as RGB frames.

    Parameters
    ----------
    trajectory : (T, Lx, Ly)
    skip_frames : subsample factor
    cmap_name : matplotlib colormap name
    symmetric : if True, clim is symmetric around 0

    Returns
    -------
    frames : (T', Lx, Ly, 3) uint8
    """
    cmap = plt.get_cmap(cmap_name)

    traj_sub = trajectory[::skip_frames]

    if symmetric:
        vmax = np.max(np.abs(traj_sub))
        vmin = -vmax
    else:
        vmin = traj_sub.min()
        vmax = traj_sub.max()

    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    frames = []
    for phi in traj_sub:
        rgba = cmap(norm(phi))  # (Lx, Ly, 4)
        rgb = (rgba[..., :3] * 255).astype(np.uint8)
        frames.append(rgb)

    return np.stack(frames)


# ============================================================
# Frame generation — EPR density map
# ============================================================

def generate_epr_frames(
    trajectory: np.ndarray,
    model: ActiveModelB,
    skip_frames: int = 1,
    cmap_name: str = "gist_heat",
) -> np.ndarray:
    """
    Render local EPR density σ(r,t) as RGB frames.

    Parameters
    ----------
    trajectory : (T, Lx, Ly)
    model : ActiveModelB instance
    skip_frames : subsample factor
    cmap_name : colormap

    Returns
    -------
    frames : (T'-1, Lx, Ly, 3) uint8
    """
    cmap = plt.get_cmap(cmap_name)

    T = trajectory.shape[0]

    epr_maps = []
    for t in range(T - 1):
        sigma = model.compute_local_epr_density(trajectory[t], trajectory[t + 1])
        epr_maps.append(sigma)
    T_sub = (T - 1) // skip_frames
    epr_maps_sub = np.zeros((T_sub, model.Lx, model.Ly))
    for i in range(T_sub):
        epr_maps_sub[i] = np.mean(epr_maps[i * skip_frames:(i + 1) * skip_frames], axis=0)
    epr_maps = epr_maps_sub

    # use symmetric log-scale normalization for EPR
    # (EPR can be positive or negative instantaneously)
    vmax = np.percentile(np.abs(epr_maps), 99)
    vmax = max(vmax, 1e-12)
    norm = mcolors.Normalize(vmin=-vmax, vmax=vmax)

    frames = []
    for sigma in epr_maps:
        rgba = cmap(norm(sigma))
        rgb = (rgba[..., :3] * 255).astype(np.uint8)
        frames.append(rgb)

    return np.stack(frames)


# ============================================================
# Mean EPR density map (time-averaged)
# ============================================================

def save_mean_epr_map(
    trajectory: np.ndarray,
    model: ActiveModelB,
    output_path: str,
    cmap_name: str = "hot",
):
    """
    Compute and save the time-averaged local EPR density map as PNG.

    Parameters
    ----------
    trajectory : (T, Lx, Ly)
    model : ActiveModelB instance
    output_path : path to save PNG
    cmap_name : colormap
    skip_frames : subsample factor for trajectory
    """
    T = trajectory.shape[0]

    mean_epr = np.zeros((model.Lx, model.Ly))
    for t in range(T - 1):
        mean_epr += model.compute_local_epr_density(trajectory[t], trajectory[t + 1])
    mean_epr /= (T - 1)

    fig, ax = plt.subplots(figsize=(6, 5))
    vmax = np.max(mean_epr)
    vmax = max(vmax, 1e-12)
    im = ax.imshow(
        mean_epr,
        origin="lower",
        cmap=cmap_name,
        vmin=0,
        vmax=vmax,
        interpolation="nearest",
    )
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label(r"$\langle \sigma(\mathbf{r}) \rangle_t$", fontsize=12)
    ax.set_title("Time-averaged local EPR density", fontsize=12)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    print(f"[OK] Mean EPR map saved → {output_path}")
    print(f"     max |⟨σ⟩| = {vmax:.6e}")
    print(f"     spatial mean ⟨σ⟩ = {mean_epr.mean():.6e}")


# ============================================================
# EPR time series plot
# ============================================================

def save_epr_timeseries(
    trajectory: np.ndarray,
    model: ActiveModelB,
    output_path: str,
    skip_frames: int = 1,
):
    """
    Compute total EPR at each time step and save a time-series plot.
    Shows instantaneous EPR and cumulative running average.
    """
    T = trajectory.shape[0]

    epr_series = np.zeros(T - 1)
    for t in range(T - 1):
        epr_series[t] = model.compute_total_epr(trajectory[t], trajectory[t + 1])

    T_sub = (T - 1) // skip_frames
    epr_series_sub = np.zeros(T_sub)
    cumulative_epr_sub = np.zeros(T_sub)
    
    cumulative_epr = np.cumsum(epr_series) * model.dt
    
    for i in range(T_sub):
        epr_series_sub[i] = np.mean(epr_series[i * skip_frames:(i + 1) * skip_frames])
        cumulative_epr_sub[i] = np.mean(cumulative_epr[i * skip_frames:(i + 1) * skip_frames])

    dt_eff = model.dt * skip_frames
    time_axis = np.arange(T_sub) * dt_eff

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    # Instantaneous EPR
    axes[0].plot(time_axis, epr_series_sub, lw=0.4, alpha=0.7, color="steelblue")
    axes[0].set_ylabel(r"$\dot{S}_{\mathrm{tot}}(t)$", fontsize=12)
    axes[0].set_title("Total entropy production rate vs time", fontsize=12)
    axes[0].axhline(0, color="k", lw=0.5, ls="--")

    # Cumulative average
    axes[1].plot(time_axis, cumulative_epr_sub, lw=1.5, color="crimson")
    axes[1].set_ylabel(r"$\langle \dot{S} \rangle_{\mathrm{cum}}$", fontsize=12)
    axes[1].set_xlabel("Time", fontsize=12)
    axes[1].set_title("Cumulative EPR", fontsize=12)
    axes[1].axhline(0, color="k", lw=0.5, ls="--")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    print(f"[OK] EPR time series saved → {output_path}")
    print(f"     final cumulative EPR = {cumulative_epr[-1]:.6e}")


# ============================================================
# MP4 generation (matplotlib viewer)
# ============================================================

def save_frames_as_mp4_mlp(
    frames: np.ndarray,
    fps: int,
    output_path: str,
    title: str = "",
):
    """Save frames to mp4 using matplotlib."""
    T, H, W, _ = frames.shape

    fig, ax = plt.subplots()
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=10)

    im = ax.imshow(
        frames[0],
        interpolation="nearest",
        animated=True,
    )

    def update(i):
        im.set_array(frames[i])
        return (im,)

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=T,
        interval=1000 / fps,
        blit=True,
    )

    writer = animation.FFMpegWriter(fps=fps)
    ani.save(output_path, writer=writer)
    plt.close(fig)

    print(f"[OK] MP4 saved → {output_path}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Active Model B Brownian Movie (density + EPR)"
    )

    # model parameters
    parser.add_argument("--Lx", type=int, default=64)
    parser.add_argument("--Ly", type=int, default=64)
    parser.add_argument("--dx", type=float, default=1.0)
    parser.add_argument("--a", type=float, default=0.25)
    parser.add_argument("--b", type=float, default=0.25)
    parser.add_argument("--kappa", type=float, default=4.0)
    parser.add_argument("--lam", type=float, default=1.0)
    parser.add_argument("--D", type=float, default=0.1)
    parser.add_argument("--dt", type=float, default=0.001)

    # trajectory
    parser.add_argument("--n_steps", type=int, default=48000)
    parser.add_argument("--burn_in", type=int, default=50000)
    parser.add_argument("--init_mode", type=str, default="circle",
                        choices=["circle", "wall"],
                        help="Initial condition: 'circle' or 'wall'")
    parser.add_argument("--smooth", action="store_true",
                        help="Use 13-point biharmonic stencil")

    # rendering
    parser.add_argument("--skip_frames", type=int, default=100)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--cmap", type=str, default="magma",
                        help="Colormap for density field")
    parser.add_argument("--epr_cmap", type=str, default="hot",
                        help="Colormap for EPR density")

    # output
    parser.add_argument("--output", type=str, default="amb_movie.mp4")
    parser.add_argument("--epr_output", type=str, default="",
                        help="If set, also generate EPR movie")
    parser.add_argument("--mean_epr_output", type=str, default="",
                        help="If set, save time-averaged EPR map as PNG")
    parser.add_argument("--epr_timeseries", type=str, default="",
                        help="If set, save EPR time series plot as PNG")
    parser.add_argument("--save_npz", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    np.random.seed(args.seed)

    # --- model ---
    model = ActiveModelB(
        Lx=args.Lx, Ly=args.Ly, dx=args.dx,
        a=args.a, b=args.b, kappa=args.kappa,
        lam=args.lam, D=args.D, dt=args.dt,
        smooth=args.smooth,
    )

    print(f"[INFO] Generating AMB trajectory ({args.Lx}×{args.Ly})")
    trajectory = model.generate_trajectory(
        n_steps=args.n_steps,
        burn_in=args.burn_in,
        init_mode=args.init_mode,
    )
    print(f"[INFO] Trajectory shape: {trajectory.shape}")

    mean_epr = model.compute_mean_epr(trajectory)
    print(f"[EPR] Mean total EPR: {mean_epr:.6f}")

    # --- density frames ---
    print("[INFO] Rendering density frames ...")
    density_frames = generate_density_frames(
        trajectory,
        skip_frames=args.skip_frames,
        cmap_name=args.cmap,
    )
    print(f"[INFO] Density frames shape: {density_frames.shape}")

    save_frames_as_mp4_mlp(
        density_frames, args.fps, args.output,
        title="Active Model B  φ(r,t)",
    )

    # --- EPR frames (optional) ---
    if args.epr_output:
        print("[INFO] Rendering EPR density frames ...")
        epr_frames = generate_epr_frames(
            trajectory, model,
            skip_frames=args.skip_frames,
            cmap_name=args.epr_cmap,
        )
        print(f"[INFO] EPR frames shape: {epr_frames.shape}")

        save_frames_as_mp4_mlp(
            epr_frames, args.fps, args.epr_output,
            title="Local EPR density  σ(r,t)",
        )

    # --- mean EPR map (optional) ---
    if args.mean_epr_output:
        print("[INFO] Computing mean EPR density map ...")
        save_mean_epr_map(
            trajectory, model,
            output_path=args.mean_epr_output,
            cmap_name=args.epr_cmap,
        )

    # --- EPR time series (optional) ---
    if args.epr_timeseries:
        print("[INFO] Computing EPR time series ...")
        save_epr_timeseries(
            trajectory, model,
            output_path=args.epr_timeseries,
            skip_frames=args.skip_frames,
        )

    # --- save npz ---
    if args.save_npz:
        npz_path = os.path.splitext(args.output)[0] + ".npz"
        save_dict = dict(
            density_frames=density_frames,
            trajectory=trajectory,
        )
        if args.epr_output:
            save_dict["epr_frames"] = epr_frames
        np.savez(npz_path, **save_dict)
        print(f"[OK] Saved dataset → {npz_path}")

    print("Done.")


if __name__ == "__main__":
    main()
