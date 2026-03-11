"""
Active Model B 1D — Animations and Kymographs

Renders 1D φ(x,t) density and EPR maps.
"""

import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation

try:
    from tqdm import trange, tqdm
except ImportError:
    trange = range
    def tqdm(iterable, *args, **kwargs):
        return iterable

from generate_trajectories_1d import ActiveModelB1D

def save_phi_kymograph(trajectory, dx, dt_eff, output_path):
    T, Lx = trajectory.shape
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(
        trajectory,
        aspect='auto',
        origin='lower',
        extent=[0, Lx*dx, 0, T*dt_eff],
        cmap="magma",
    )
    plt.colorbar(im, ax=ax, label=r"$\phi(x,t)$")
    ax.set_xlabel("x")
    ax.set_ylabel("Time")
    ax.set_title("Density Kymograph")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"[OK] Kymograph saved -> {output_path}")


def save_1d_animation(trajectory, model, fps, skip_frames, output_path):
    traj_sub = trajectory[::skip_frames]
    T, Lx = traj_sub.shape
    x = np.arange(Lx) * model.dx

    fig, ax = plt.subplots(figsize=(8, 4))
    line, = ax.plot(x, traj_sub[0], lw=2, color='navy')
    phi_eq = np.sqrt(model.a / model.b) if (model.a > 0 and model.b > 0) else 0.0
    ax.set_ylim(-1.5 * phi_eq, 1.5 * phi_eq)
    ax.set_xlim(0, Lx * model.dx)
    ax.set_xlabel("x")
    ax.set_ylabel(r"$\phi(x,t)$")
    ax.set_title("Active Model B 1D Density")

    def update(i):
        line.set_ydata(traj_sub[i])
        return (line,)

    ani = animation.FuncAnimation(fig, update, frames=T, interval=1000/fps, blit=True)
    writer = animation.FFMpegWriter(fps=fps)
    ani.save(output_path, writer=writer)
    plt.close(fig)
    print(f"[OK] MP4 saved -> {output_path}")

def save_epr_kymograph(trajectory, model, skip_frames, output_path):
    T = trajectory.shape[0]
    
    epr_maps = []
    for t in tqdm(range(T - 1), desc="Computing EPR Kymograph", leave=False):
        sigma = model.compute_local_epr_density(trajectory[t], trajectory[t + 1])
        if hasattr(sigma, 'cpu'):
            sigma = sigma.cpu().numpy()
        epr_maps.append(sigma)
    epr_maps = np.stack(epr_maps)

    T_sub = (T - 1) // skip_frames
    epr_maps_sub = np.zeros((T_sub, model.Lx))
    for i in range(T_sub):
        epr_maps_sub[i] = np.mean(epr_maps[i * skip_frames:(i + 1) * skip_frames], axis=0)

    fig, ax = plt.subplots(figsize=(6, 5))
    vmax = np.percentile(np.abs(epr_maps_sub), 99)
    vmax = max(vmax, 1e-12)
    im = ax.imshow(
        epr_maps_sub,
        aspect='auto',
        origin='lower',
        extent=[0, model.Lx*model.dx, 0, T_sub*model.dt*skip_frames],
        cmap="hot",
        vmin=-vmax, vmax=vmax,
    )
    plt.colorbar(im, ax=ax, label=r"$\sigma(x,t)$")
    ax.set_xlabel("x")
    ax.set_ylabel("Time")
    ax.set_title("Local EPR Density Kymograph")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"[OK] EPR Kymograph saved -> {output_path}")

def save_mean_phi_map(trajectory, model, output_path):
    mean_phi = np.mean(trajectory, axis=0)
    x = np.arange(model.Lx) * model.dx

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x, mean_phi, lw=2, color='navy')
    phi_eq = np.sqrt(model.a / model.b) if (model.a > 0 and model.b > 0) else 0.0
    ax.axhline(phi_eq, color='k', ls='--', lw=0.8)
    ax.axhline(-phi_eq, color='k', ls='--', lw=0.8)
    ax.set_xlim(0, model.Lx * model.dx)
    ax.set_ylim(-1.5 * phi_eq, 1.5 * phi_eq)
    ax.set_xlabel("x")
    ax.set_ylabel(r"$\langle\phi(x)\rangle_t$")
    ax.set_title("Time-averaged Density Profile")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"[OK] Mean phi map saved -> {output_path}")

def save_mean_epr_map(trajectory, model, output_path):
    T = trajectory.shape[0]
    
    mean_epr = np.zeros(model.Lx)
    for t in tqdm(range(T - 1), desc="Computing Mean EPR Map", leave=False):
        idx_epr = model.compute_local_epr_density(trajectory[t], trajectory[t + 1])
        if hasattr(idx_epr, 'cpu'):
            idx_epr = idx_epr.cpu().numpy()
        mean_epr += idx_epr
    mean_epr /= (T - 1) 

    x = np.arange(model.Lx) * model.dx

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x, mean_epr, lw=2, color='crimson')
    ax.axhline(0, color='k', ls='--', lw=0.8)
    ax.set_xlim(0, model.Lx * model.dx)
    ax.set_xlabel("x")
    ax.set_ylabel(r"$\langle\sigma(x)\rangle_t$")
    ax.set_title("Time-averaged Local EPR Density")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"[OK] Mean EPR map saved -> {output_path}")

def save_epr_timeseries(trajectory, model, skip_frames, output_path):
    T = trajectory.shape[0]

    epr_series = np.zeros(T - 1)
    for t in tqdm(range(T - 1), desc="Computing EPR Timeseries", leave=False):
        idx_total_epr = model.compute_total_epr(trajectory[t], trajectory[t + 1])
        if hasattr(idx_total_epr, 'cpu'):
            idx_total_epr = idx_total_epr.cpu().numpy()
        epr_series[t] = idx_total_epr

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
    axes[0].plot(time_axis, epr_series_sub, lw=1.0, color="steelblue")
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

    print(f"[OK] EPR time series saved -> {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Active Model B 1D Animation generator")
    parser.add_argument("--L", type=float, default=500.0, help="Total length of the domain")
    parser.add_argument("--Lx", type=int, default=None, help="Number of grid points (overrides L if specified)")
    parser.add_argument("--dx", type=float, default=0.1)
    parser.add_argument("--a", type=float, default=0.125)
    parser.add_argument("--b", type=float, default=0.125)
    parser.add_argument("--kappa", type=float, default=8.0)
    parser.add_argument("--lam", type=float, default=2.0)
    parser.add_argument("--D", type=float, default=0.001)
    parser.add_argument("--dt", type=float, default=0.001)

    parser.add_argument("--n_steps", type=int, default=20000)
    parser.add_argument("--burn_in", type=int, default=10000)
    parser.add_argument("--smooth", action="store_true")

    parser.add_argument("--backend", type=str, default="numpy", choices=["numpy", "torch"])
    parser.add_argument("--use_gpu", action="store_true")

    parser.add_argument("--skip_frames", type=int, default=100)
    parser.add_argument("--fps", type=int, default=15)

    parser.add_argument("--output", type=str, default="amb_1d_movie.mp4")
    parser.add_argument("--kymo_output", type=str, default="amb_1d_kymo.png")
    parser.add_argument("--epr_kymo_output", type=str, default="amb_1d_epr_kymo.png")
    parser.add_argument("--mean_phi_output", type=str, default="amb_1d_mean_phi.png")
    parser.add_argument("--mean_epr_output", type=str, default="amb_1d_mean_epr.png")
    parser.add_argument("--epr_timeseries", type=str, default="amb_1d_epr_timeseries.png")
    parser.add_argument("--save_npz", action="store_true")

    args = parser.parse_args()

    if args.Lx is None:
        args.Lx = int(args.L / args.dx)

    model = ActiveModelB1D(
        Lx=args.Lx, dx=args.dx, a=args.a, b=args.b,
        kappa=args.kappa, lam=args.lam, D=args.D, dt=args.dt, smooth=args.smooth,
        backend=args.backend, use_gpu=args.use_gpu
    )

    print(f"[INFO] Generating 1D trajectory ({args.Lx} points) [periodic, double wall]")
    trajectory = model.generate_trajectory(n_steps=args.n_steps, burn_in=args.burn_in)

    print("[INFO] Rendering animations and kymographs")
    if args.kymo_output:
        dt_eff = model.dt * args.skip_frames
        save_phi_kymograph(trajectory[::args.skip_frames], args.dx, dt_eff, args.kymo_output)

    if args.output:
        save_1d_animation(trajectory, model, args.fps, args.skip_frames, args.output)

    if args.epr_kymo_output:
        save_epr_kymograph(trajectory, model, args.skip_frames, args.epr_kymo_output)

    if args.mean_phi_output:
        save_mean_phi_map(trajectory, model, args.mean_phi_output)

    if args.mean_epr_output:
        save_mean_epr_map(trajectory, model, args.mean_epr_output)

    if args.epr_timeseries:
        print("[INFO] Computing EPR time series ...")
        save_epr_timeseries(trajectory, model, args.skip_frames, args.epr_timeseries)

    if args.save_npz:
        npz_path = os.path.splitext(args.output)[0] + ".npz"
        np.savez(npz_path, trajectory=trajectory)
        print(f"[OK] Dataset saved -> {npz_path}")

    print("[INFO] Computing Final Mean EPR ...")
    final_mean_epr = model.compute_mean_epr(trajectory)
    if hasattr(final_mean_epr, 'size') and final_mean_epr.size > 1:
        print(f"\n[EPR] Mean total EPR: {np.mean(final_mean_epr):.6f}\n")
    else:
        print(f"\n[EPR] Mean total EPR: {float(np.mean(final_mean_epr)):.6f}\n")

    print("Done.")

if __name__ == "__main__":
    main()
