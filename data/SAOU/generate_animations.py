"""
SAOU Field — Animation Generator

Renders trajectory animations for the 2-component (u, v) Ornstein-Uhlenbeck field:
  - u/v component heatmaps
  - Angular flux and angular velocity maps
  - Dynamic structure factor S(|k|, omega)
  - Chiral dynamic spectrum S_z(|k|, omega)

Outputs MP4 videos and PNG static plots.

Usage:
    python generate_animations.py
    python generate_animations.py --input saou_trajectories.npz --skip_frames 2 --fps 15
    python generate_animations.py --L 32 --n_steps 20000 --omega0 1.0
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


# ============================================================
# Frame generation — u/v components
# ============================================================

def generate_uv_frames(
    trajectory: np.ndarray,
    skip_frames: int = 1,
    cmap_name: str = "RdBu",
    vmin: float = -1.5,
    vmax: float = 1.5,
) -> np.ndarray:
    """
    Render u and v components side-by-side as RGB frames.

    Parameters
    ----------
    trajectory : (T, L, L, 2)
    skip_frames : subsample factor
    cmap_name : matplotlib colormap name
    vmin, vmax : color limits

    Returns
    -------
    frames : (T', L, 2*L, 3) uint8
    """
    cmap = plt.get_cmap(cmap_name)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    traj_sub = trajectory[::skip_frames]
    frames = []
    for x in traj_sub:
        u_rgba = cmap(norm(x[..., 0]))  # (L, L, 4)
        v_rgba = cmap(norm(x[..., 1]))  # (L, L, 4)
        # side-by-side
        combined = np.concatenate([u_rgba[..., :3], v_rgba[..., :3]], axis=1)
        rgb = (combined * 255).astype(np.uint8)
        frames.append(rgb)
    return np.stack(frames)


# ============================================================
# Frame generation — angular quantities
# ============================================================

def generate_angular_frames(
    trajectory: np.ndarray,
    dt_frame: float,
    skip_frames: int = 1,
    cmap_name: str = "RdBu",
) -> np.ndarray:
    """
    Render angular flux and angular velocity side-by-side.

    Parameters
    ----------
    trajectory : (T, L, L, 2)
    dt_frame : time between trajectory frames
    skip_frames : subsample factor

    Returns
    -------
    frames : (T'-1, L, 2*L, 3) uint8
    """
    cmap = plt.get_cmap(cmap_name)
    eps = 1e-8

    traj_sub = trajectory[::skip_frames]
    T = traj_sub.shape[0]

    frames = []
    for i in range(1, T):
        x = traj_sub[i]
        x_prev = traj_sub[i - 1]
        dx = x - x_prev
        x_mid = 0.5 * (x + x_prev)

        u = x_mid[..., 0]
        v = x_mid[..., 1]
        du = dx[..., 0] / (dt_frame * skip_frames)
        dv = dx[..., 1] / (dt_frame * skip_frames)

        angular_flux = u * dv - v * du
        r2 = u**2 + v**2
        angular_velocity = angular_flux / (r2 + eps)

        norm_flux = mcolors.Normalize(vmin=-5.0, vmax=5.0)
        norm_vel = mcolors.Normalize(vmin=-5.0, vmax=5.0)

        flux_rgba = cmap(norm_flux(angular_flux))
        vel_rgba = cmap(norm_vel(angular_velocity))

        combined = np.concatenate([flux_rgba[..., :3], vel_rgba[..., :3]], axis=1)
        rgb = (combined * 255).astype(np.uint8)
        frames.append(rgb)

    return np.stack(frames)


# ============================================================
# MP4 generation
# ============================================================

def save_frames_as_mp4(
    frames: np.ndarray,
    fps: int,
    output_path: str,
    title: str = "",
):
    """Save frames to mp4 using matplotlib."""
    T, H, W, _ = frames.shape

    # Fix: Ensure a minimum figure size so the field isn't dwarfed by text
    fig_width = max(8.0, W / 10.0)
    fig_height = fig_width * (H / W)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=150)
    
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=16, pad=10)

    im = ax.imshow(frames[0], interpolation="nearest", animated=True)
    fig.tight_layout(pad=0.3)

    def update(i):
        im.set_array(frames[i])
        return (im,)

    ani = animation.FuncAnimation(
        fig, update, frames=T, interval=1000 / fps, blit=True,
    )

    writer = animation.FFMpegWriter(fps=fps)
    ani.save(output_path, writer=writer)
    plt.close(fig)

    print(f"[OK] MP4 saved → {output_path}")


# ============================================================
# Matplotlib figure-based animation (richer layout)
# ============================================================

def save_rich_animation(
    trajectory: np.ndarray,
    dt_frame: float,
    fps: int,
    output_path: str,
    max_frames: int = 500,
):
    """
    Save a 3-panel animation: u-component, angular flux, angular velocity.
    """
    T_total = trajectory.shape[0]
    skip = max(1, T_total // max_frames)
    traj = trajectory[::skip]
    n_frames = traj.shape[0]
    dt_eff = dt_frame * skip
    eps = 1e-8

    fig, axs = plt.subplots(1, 3, figsize=(11, 3), dpi=100)

    def animate(i):
        for ax in axs:
            ax.clear()

        x = traj[i]

        if i == 0:
            L = x.shape[0]
            angular_flux = np.zeros((L, L))
            angular_velocity = np.zeros((L, L))
        else:
            x_prev = traj[i - 1]
            dx = x - x_prev
            x_mid = 0.5 * (x + x_prev)

            u = x_mid[..., 0]
            v = x_mid[..., 1]
            du = dx[..., 0] / dt_eff
            dv = dx[..., 1] / dt_eff

            angular_flux = u * dv - v * du
            r2 = u**2 + v**2
            angular_velocity = angular_flux / (r2 + eps)

        axs[0].imshow(x[..., 0], vmin=-1.5, vmax=1.5, cmap="RdBu")
        axs[1].imshow(angular_flux, vmin=-5.0, vmax=5.0, cmap="RdBu")
        axs[2].imshow(angular_velocity, vmin=-5.0, vmax=5.0, cmap="RdBu")

        axs[0].set_title("u component")
        axs[1].set_title(r"$u\dot{v} - v\dot{u}$")
        axs[2].set_title(r"angular velocity $\dot{\theta}$")

        for ax in axs:
            ax.set_xticks([])
            ax.set_yticks([])

    ani = animation.FuncAnimation(fig, animate, frames=n_frames, blit=False)
    writer = animation.FFMpegWriter(fps=fps)
    ani.save(output_path, writer=writer)
    plt.close(fig)
    print(f"[OK] Rich animation saved → {output_path}")


# ============================================================
# Static analysis plots
# ============================================================

def save_angular_timeseries(
    trajectory: np.ndarray,
    dt_frame: float,
    output_path: str,
):
    """
    Plot angular momentum, angular flux per site, and weighted global omega.
    """
    x0 = trajectory[:-1]
    x1 = trajectory[1:]
    dx = x1 - x0
    N = x0.shape[1] * x0.shape[2]

    angular_momentum = np.sum(
        x0[..., 0] * dx[..., 1] - x0[..., 1] * dx[..., 0],
        axis=(1, 2),
    )
    angular_flux_per_site = angular_momentum / (N * dt_frame)

    r2_total = np.sum(x0[..., 0]**2 + x0[..., 1]**2, axis=(1, 2))
    global_weighted_omega = angular_momentum / (dt_frame * (r2_total + 1e-8))

    fig, axes = plt.subplots(1, 3, figsize=(10, 3), dpi=150)

    axes[0].plot(angular_momentum, lw=0.5)
    axes[0].set_title("total angular increment")
    axes[0].axhline(0, color="k", lw=0.8)

    axes[1].plot(angular_flux_per_site, lw=0.5)
    axes[1].set_title("angular flux per site")
    axes[1].axhline(0, color="k", lw=0.8)

    axes[2].plot(global_weighted_omega, lw=0.5)
    axes[2].set_title("weighted global omega")
    axes[2].axhline(0, color="k", lw=0.8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"[OK] Angular time series saved → {output_path}")


def save_dynamic_structure_factor(
    trajectory: np.ndarray,
    dt_frame: float,
    output_path: str,
):
    """
    Compute and plot the dynamic structure factor S(|k|, omega).
    """
    n_t, L, _, d = trajectory.shape

    # Remove time mean
    x = trajectory - trajectory.mean(axis=0, keepdims=True)

    # Windowing
    window_t = np.hanning(n_t)[:, None, None, None]
    xw = x * window_t

    # Fourier transform
    X_kw = np.fft.fftn(xw, axes=(0, 1, 2))
    S = np.sum(np.abs(X_kw)**2, axis=-1)

    freq = np.fft.fftfreq(n_t, d=dt_frame)
    omega = 2 * np.pi * freq
    kx = 2 * np.pi * np.fft.fftfreq(L, d=1.0)
    ky = 2 * np.pi * np.fft.fftfreq(L, d=1.0)
    KY, KX = np.meshgrid(ky, kx, indexing="ij")
    K = np.sqrt(KX**2 + KY**2)

    n_k_bins = L // 2
    k_edges = np.linspace(0, K.max(), n_k_bins + 1)
    k_centers = 0.5 * (k_edges[:-1] + k_edges[1:])

    S_radial = np.zeros((n_t, n_k_bins))
    for b in range(n_k_bins):
        mask = (K >= k_edges[b]) & (K < k_edges[b + 1])
        if np.any(mask):
            S_radial[:, b] = S[:, mask].mean(axis=1)

    S_plot = np.fft.fftshift(S_radial, axes=0)
    omega_plot = np.fft.fftshift(omega)

    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    ax.imshow(
        np.log10(S_plot + 1e-12),
        aspect="auto",
        origin="lower",
        extent=[k_centers[0], k_centers[-1], omega_plot[0], omega_plot[-1]],
        cmap="magma",
    )
    ax.set_xlabel(r"$|k|$")
    ax.set_ylabel(r"$\omega$")
    ax.set_title(r"Dynamic structure factor $S(|k|,\omega)$")
    fig.colorbar(ax.images[0], ax=ax, label=r"$\log_{10} S(|k|,\omega)$")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"[OK] Dynamic structure factor saved → {output_path}")


def save_chiral_spectrum(
    trajectory: np.ndarray,
    dt_frame: float,
    output_path: str,
    shells=None,
):
    """
    Compute and plot the chiral dynamic spectrum S_z(|k|, omega)
    with optional predicted Omega(k) overlay.
    """
    n_t, L, _, d = trajectory.shape

    # Complex chiral field
    z = trajectory[..., 0] + 1j * trajectory[..., 1]
    z = z - z.mean(axis=0, keepdims=True)

    window_t = np.hanning(n_t)[:, None, None]
    zw = z * window_t

    Z_kw = np.fft.fftn(zw, axes=(0, 1, 2))
    S_z = np.abs(Z_kw)**2

    freq = np.fft.fftfreq(n_t, d=dt_frame)
    omega = 2 * np.pi * freq
    kx = 2 * np.pi * np.fft.fftfreq(L, d=1.0)
    ky = 2 * np.pi * np.fft.fftfreq(L, d=1.0)
    KY, KX = np.meshgrid(ky, kx, indexing="ij")
    K = np.sqrt(KX**2 + KY**2)

    n_k_bins = L // 2
    k_edges = np.linspace(0, K.max(), n_k_bins + 1)
    k_centers = 0.5 * (k_edges[:-1] + k_edges[1:])

    S_z_radial = np.zeros((n_t, n_k_bins))
    for b in range(n_k_bins):
        mask = (K >= k_edges[b]) & (K < k_edges[b + 1])
        if np.any(mask):
            S_z_radial[:, b] = S_z[:, mask].mean(axis=1)

    S_z_plot = np.fft.fftshift(S_z_radial, axes=0)
    omega_plot = np.fft.fftshift(omega)

    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    ax.imshow(
        np.log10(S_z_plot + 1e-12),
        aspect="auto",
        origin="lower",
        extent=[k_centers[0], k_centers[-1], omega_plot[0], omega_plot[-1]],
        cmap="magma",
    )

    # Overlay theoretical Omega(k) if shells are provided
    if shells is not None:
        Omega_grid = np.zeros((L, L), dtype=np.float64)
        for sh in shells:
            lam = np.zeros((L, L), dtype=np.complex128)
            for (dy, dx_off), w in zip(sh.offsets, sh.weights):
                lam += w * np.exp(1j * (KX * dx_off + KY * dy))
            c = np.sum(sh.weights)
            Omega_grid += sh.amplitude * (lam.real - c)

        Omega_radial = np.zeros(n_k_bins)
        for b in range(n_k_bins):
            mask = (K >= k_edges[b]) & (K < k_edges[b + 1])
            if np.any(mask):
                Omega_radial[b] = Omega_grid[mask].mean()

        ax.plot(k_centers, Omega_radial, "c-", lw=2, label=r"$\Omega(|k|)$")
        ax.legend()

    ax.set_xlabel(r"$|k|$")
    ax.set_ylabel(r"$\omega$")
    ax.set_title(r"Chiral dynamic spectrum $S_z(|k|,\omega)$")
    fig.colorbar(ax.images[0], ax=ax, label=r"$\log_{10} S_z(|k|,\omega)$")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"[OK] Chiral spectrum saved → {output_path}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="SAOU Field — animation and analysis generator"
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

    # Outputs
    parser.add_argument("--output_uv", type=str, default="saou_uv_movie.mp4",
                        help="u/v component animation (set empty to skip)")
    parser.add_argument("--output_angular", type=str, default="saou_angular_movie.mp4",
                        help="Angular flux/velocity animation (set empty to skip)")
    parser.add_argument("--output_rich", type=str, default="saou_rich_movie.mp4",
                        help="3-panel rich animation (set empty to skip)")
    parser.add_argument("--output_angular_ts", type=str, default="saou_angular_timeseries.png",
                        help="Angular time series plot (set empty to skip)")
    parser.add_argument("--output_dsf", type=str, default="saou_dynamic_sf.png",
                        help="Dynamic structure factor (set empty to skip)")
    parser.add_argument("--output_chiral", type=str, default="saou_chiral_spectrum.png",
                        help="Chiral dynamic spectrum (set empty to skip)")

    args = parser.parse_args()

    # Load or simulate
    if args.input and os.path.exists(args.input):
        import json as _json
        print(f"[INFO] Loading trajectory from {args.input}")
        data = np.load(args.input, allow_pickle=True)
        trajectory = data["trajectory"]
        params = _json.loads(str(data["params_json"]))
        dt_frame = params["dt"] * params["sample_every"]
        shells = None  # cannot reconstruct Shell objects from npz easily
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
        dt_frame = out["params"]["dt"] * out["params"]["sample_every"]
        shells = out["shells"]
        print(f"[INFO] Trajectory shape: {trajectory.shape}")

    # Generate outputs
    if args.output_uv:
        print("[INFO] Rendering u/v component frames ...")
        uv_frames = generate_uv_frames(trajectory, skip_frames=args.skip_frames)
        save_frames_as_mp4(uv_frames, args.fps, args.output_uv, title="SAOU: u | v")

    if args.output_angular:
        print("[INFO] Rendering angular frames ...")
        ang_frames = generate_angular_frames(
            trajectory, dt_frame, skip_frames=args.skip_frames,
        )
        save_frames_as_mp4(
            ang_frames, args.fps, args.output_angular,
            title="SAOU: angular flux | angular velocity",
        )

    if args.output_rich:
        print("[INFO] Rendering rich 3-panel animation ...")
        save_rich_animation(
            trajectory, dt_frame, args.fps, args.output_rich,
            max_frames=args.max_frames,
        )

    if args.output_angular_ts:
        print("[INFO] Computing angular time series ...")
        save_angular_timeseries(trajectory, dt_frame, args.output_angular_ts)

    if args.output_dsf:
        print("[INFO] Computing dynamic structure factor ...")
        save_dynamic_structure_factor(trajectory, dt_frame, args.output_dsf)

    if args.output_chiral:
        print("[INFO] Computing chiral spectrum ...")
        save_chiral_spectrum(trajectory, dt_frame, args.output_chiral, shells=shells)

    print("\nDone.")


if __name__ == "__main__":
    main()
