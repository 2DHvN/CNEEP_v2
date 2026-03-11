import numpy as np
import argparse
import os
import matplotlib.pyplot as plt

try:
    from tqdm import trange, tqdm
except ImportError:
    trange = range
    def tqdm(iterable, *args, **kwargs):
        return iterable

from generate_trajectories_1d import ActiveModelB1D

def main():
    parser = argparse.ArgumentParser(description="Active Model B 1D Mean EPR Calculator over multiple seeds (Ensemble)")
    parser.add_argument("--L", type=float, default=500.0, help="Total length of the domain")
    parser.add_argument("--Lx", type=int, default=None, help="Number of grid points")
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

    parser.add_argument("--n_seeds", type=int, default=5, help="Number of different seeds/trajectories (Ensemble size)")
    parser.add_argument("--start_seed", type=int, default=0, help="Starting value of seed")
    parser.add_argument("--skip_frames", type=int, default=100, help="Frames to average over for 2D kymographs")

    parser.add_argument("--output", type=str, default="amb_1d_total_epr_results.npy")
    parser.add_argument("--density_output", type=str, default="amb_1d_mean_epr_densities.npy")
    parser.add_argument("--plot_output", type=str, default="amb_1d_mean_epr_density_plot.png")
    parser.add_argument("--epr_kymo_output", type=str, default="amb_1d_ensemble_epr_kymo.png")
    parser.add_argument("--phi_kymo_output", type=str, default="amb_1d_ensemble_phi_kymo.png")

    args = parser.parse_args()

    if args.Lx is None:
        args.Lx = int(args.L / args.dx)

    if args.start_seed != 0:
        np.random.seed(args.start_seed)
        # Assuming torch is available if requested
        try:
            import torch
            torch.manual_seed(args.start_seed)
        except ImportError:
            pass

    model = ActiveModelB1D(
        Lx=args.Lx, dx=args.dx, a=args.a, b=args.b,
        kappa=args.kappa, lam=args.lam, D=args.D, dt=args.dt, smooth=args.smooth,
        backend=args.backend, use_gpu=args.use_gpu
    )

    all_eprs = []
    all_epr_densities = []

    T_sub = (args.n_steps - 1) // args.skip_frames
    
    print(f"[INFO] Active Model B 1D ({args.Lx} points) - [periodic, double wall]")
    print(f"[INFO] Generating ensemble of {args.n_seeds} trajectories simultaneously using {args.backend}...")
    
    # 앙상블 시스템으로 한 번에(Batched) 궤적을 모두 생성
    trajectories = model.generate_trajectories(
        n_trajectories=args.n_seeds,
        n_steps=args.n_steps,
        burn_in=args.burn_in,
        show_progress=True
    )
    
    print("[INFO] Computing mean EPR across the ensemble...")
    # trajectories shape: (n_seeds, n_steps, Lx)
    # compute_mean_epr returns total EPR for each trajectory: shape (n_seeds,)
    all_eprs = model.compute_mean_epr(trajectories)
    
    # Kymograph and internal density calculations
    T = trajectories.shape[1]
    
    # To compute batched local density averages memory efficiently:
    ensemble_epr_kymo = np.zeros((T_sub, args.Lx))
    ensemble_phi_kymo = np.zeros((T_sub, args.Lx))
    all_epr_densities = np.zeros((args.n_seeds, args.Lx))
    
    for t_idx in trange(T_sub, desc="Computing Kymographs"):
        start_t = t_idx * args.skip_frames
        end_t = (t_idx + 1) * args.skip_frames
        
        chunk_epr = np.zeros((args.n_seeds, args.Lx))
        chunk_phi = np.zeros((args.n_seeds, args.Lx))
        
        for t in range(start_t, end_t):
            sigma = model.compute_local_epr_density(trajectories[:, t], trajectories[:, t + 1])
            if hasattr(sigma, 'cpu'):
                sigma = sigma.cpu().numpy()
            
            chunk_epr += sigma
            chunk_phi += trajectories[:, t]
            all_epr_densities += sigma
            
        ensemble_epr_kymo[t_idx] = np.mean(chunk_epr, axis=0) / args.skip_frames
        ensemble_phi_kymo[t_idx] = np.mean(chunk_phi, axis=0) / args.skip_frames

    # Handle remaining steps
    for t in tqdm(range(T_sub * args.skip_frames, T - 1), desc="Computing Remaining Eprs", leave=False):
        sigma = model.compute_local_epr_density(trajectories[:, t], trajectories[:, t + 1])
        if hasattr(sigma, 'cpu'):
            sigma = sigma.cpu().numpy()
        all_epr_densities += sigma
        
    all_epr_densities /= (T - 1)
    
    overall_mean_epr = np.mean(all_eprs)
    overall_std_epr = np.std(all_eprs)

    print("\n--- Final Total EPR Results ---")
    for i in range(args.n_seeds):
        print(f"Trajectory {i+1} Mean Total EPR: {all_eprs[i]:.6f}")
    print(f"\nOverall Mean EPR: {overall_mean_epr:.6e} \u00b1 {overall_std_epr:.6e}")
    print(f"                  {overall_mean_epr:.6f} \u00b1 {overall_std_epr:.6f}")
    
    np.save(args.output, all_eprs)
    np.save(args.density_output, all_epr_densities)
    print(f"[OK] Saved individual total EPRs to {args.output}")

    # 1. Visualization of Time-Avged Local EPR Densities
    print("[INFO] Generating visualization for local EPR densities...")
    x = np.arange(args.Lx) * args.dx
    fig, ax = plt.subplots(figsize=(8, 4))
    
    for i, density in enumerate(all_epr_densities):
        label = "Individual Trajectories" if i == 0 else ""
        ax.plot(x, density, lw=1, alpha=0.3, color='steelblue', label=label)
        
    ensemble_mean_density = np.mean(all_epr_densities, axis=0)
    ax.plot(x, ensemble_mean_density, lw=2, color='crimson', label="Ensemble Mean")
    
    ax.axhline(0, color='k', ls='--', lw=0.8)
    ax.set_xlim(0, args.Lx * args.dx)
    
    # Scale y-axis based on the ensemble mean instead of individual noisy trajectories
    y_min, y_max = np.min(ensemble_mean_density), np.max(ensemble_mean_density)
    y_margin = (y_max - y_min) * 0.15
    if y_margin == 0:
        y_margin = 1e-10
    ax.set_ylim(y_min - y_margin, y_max + y_margin)
    
    ax.set_xlabel("x")
    ax.set_ylabel(r"$\langle\sigma(x)\rangle_t$")
    ax.set_title(f"Time-averaged Local EPR Density ({args.n_seeds} seeds)")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(args.plot_output, dpi=150)
    plt.close(fig)
    print(f"[OK] Saved density plot to {args.plot_output}")

    # Kymograph plot parameters
    dt_eff = args.dt * args.skip_frames
    extent = [0, args.Lx * args.dx, 0, T_sub * dt_eff]

    # 2. Visualization of Ensemble EPR Kymograph
    print("[INFO] Generating visualization for Ensemble EPR kymograph...")
    fig, ax = plt.subplots(figsize=(6, 5))
    vmax = np.percentile(np.abs(ensemble_epr_kymo), 99)
    vmax = max(vmax, 1e-12)
    im = ax.imshow(
        ensemble_epr_kymo,
        aspect='auto',
        origin='lower',
        extent=extent,
        cmap="hot",
        vmin=-vmax, vmax=vmax,
    )
    plt.colorbar(im, ax=ax, label=r"$\langle\sigma(x,t)\rangle_{\mathrm{ens}}$")
    ax.set_xlabel("x")
    ax.set_ylabel("Time")
    ax.set_title(f"Ensemble Mean Local EPR Density Kymograph ({args.n_seeds} seeds)")
    fig.tight_layout()
    fig.savefig(args.epr_kymo_output, dpi=150)
    plt.close(fig)
    print(f"[OK] Saved Ensemble EPR Kymograph to {args.epr_kymo_output}")

    # 3. Visualization of Ensemble Phi Kymograph
    print("[INFO] Generating visualization for Ensemble Phi kymograph...")
    fig, ax = plt.subplots(figsize=(6, 5))
    vmax_phi = np.percentile(np.abs(ensemble_phi_kymo), 99)
    # Ensure some bounds so it doesn't crash on completely uniform arrays
    vmax_phi = max(vmax_phi, 1e-12)
    im2 = ax.imshow(
        ensemble_phi_kymo,
        aspect='auto',
        origin='lower',
        extent=extent,
        cmap="magma",
        vmin=-vmax_phi, vmax=vmax_phi
    )
    plt.colorbar(im2, ax=ax, label=r"$\langle\phi(x,t)\rangle_{\mathrm{ens}}$")
    ax.set_xlabel("x")
    ax.set_ylabel("Time")
    ax.set_title(rf"Ensemble Mean Density Kymograph $\phi(x,t)$ ({args.n_seeds} seeds)")
    fig.tight_layout()
    fig.savefig(args.phi_kymo_output, dpi=150)
    plt.close(fig)
    print(f"[OK] Saved Ensemble Phi Kymograph to {args.phi_kymo_output}")

if __name__ == "__main__":
    main()
