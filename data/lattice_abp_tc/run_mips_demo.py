"""Run a thermodynamically consistent lattice-ABP MIPS demo.

Example:
    python data/lattice_abp_tc/run_mips_demo.py --steps 20000 --save-interval 200
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.lattice_abp_tc import ThermodynamicLatticeABP, ThermodynamicLatticeABPParams


def n_from_packing_fraction(phi: float, box_size: float, sigma: float) -> int:
    return int(round(phi * 4.0 * box_size * box_size / (math.pi * sigma * sigma)))


def draw_occupancy_cells(ax, occupancy: np.ndarray, title: str):
    """Draw binary occupancy as tiles that fill their lattice cells exactly."""
    image = np.asarray(occupancy)
    if image.ndim != 2:
        raise ValueError("occupancy must be a two-dimensional array.")
    nx, ny = image.shape
    mesh = ax.pcolormesh(
        np.arange(nx + 1),
        np.arange(ny + 1),
        (image.T > 0).astype(np.uint8),
        shading="flat",
        antialiased=False,
        cmap="gray_r",
        vmin=0,
        vmax=1,
    )
    ax.set_title(title)
    ax.set_xlim(0, nx)
    ax.set_ylim(0, ny)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    return mesh


def save_snapshot_figure(
    occupancy: torch.Tensor,
    out_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    occ0 = occupancy[0, 0].numpy()
    occ1 = occupancy[-1, 0].numpy()

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)
    for ax, image, title in zip(
        axes,
        (occ0, occ1),
        ("initial occupancy", "final occupancy"),
    ):
        draw_occupancy_cells(ax, image, title)

    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", default="output/lattice_abp_mips")
    parser.add_argument("--L", type=float, default=24.0)
    parser.add_argument("--grid-size", type=int, default=96)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--phi", type=float, default=0.55)
    parser.add_argument("--N", type=int, default=None)
    parser.add_argument("--epsilon", type=float, default=20.0)
    parser.add_argument("--v0", type=float, default=24.0)
    parser.add_argument("--Dr", type=float, default=1.0)
    parser.add_argument("--Dt", type=float, default=0.2)
    parser.add_argument("--dt", type=float, default=1.0e-4)
    parser.add_argument("--prefactor", choices=["cv", "c0"], default="cv")
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--burn-in", type=int, default=0)
    parser.add_argument("--save-interval", type=int, default=100)
    parser.add_argument("--B", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--device",
        default="cuda",
        help="CUDA device to use (default: cuda, intended for the L40S).",
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "cuda_fused", "numba", "torch"],
        default="cuda_fused",
        help=(
            "Default cuda_fused fails fast unless the exact fused L40S path "
            "can be built; auto permits a slower exact Torch fallback."
        ),
    )
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    n_particles = args.N
    if n_particles is None:
        n_particles = n_from_packing_fraction(args.phi, args.L, args.sigma)

    params = ThermodynamicLatticeABPParams(
        N=n_particles,
        L=args.L,
        grid_size=args.grid_size,
        sigma=args.sigma,
        epsilon=args.epsilon,
        v0=args.v0,
        Dr=args.Dr,
        Dt=args.Dt,
        dt=args.dt,
        prefactor=args.prefactor,
        seed=args.seed,
        device=args.device,
        backend=args.backend,
    )
    sim = ThermodynamicLatticeABP(params)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(
        f"device={sim.device}, L={params.L:g}, "
        f"grid={params.grid_size}, dl={params.dl:g}, "
        f"N={params.N}, phi={params.phi:.3f}, Pe={params.Pe:.2f}, "
        f"prefactor={params.prefactor}, backend={sim.backend}"
    )
    result = sim.simulate(
        B=args.B,
        burn_in=args.burn_in,
        n_steps=args.steps,
        save_interval=args.save_interval,
        show_progress=not args.no_progress,
        save_diagnostics=False,
        save_occupancy=True,
        save_exact_medium_ep=True,
    )

    initial_summary = sim.mips_summary_from_sites(
        result["sites"][0],
        include_coarse=False,
    )
    final_summary = sim.mips_summary_from_sites(
        result["sites"][-1],
        include_coarse=False,
    )
    ep_rate = result["exact_medium_ep_rate"].mean(dim=1).numpy()

    np.savez_compressed(
        outdir / "trajectory.npz",
        sites=result["sites"].numpy(),
        occupancy=result["occupancy"].numpy(),
        theta=result["theta"].numpy(),
        times=result["times"].numpy(),
        exact_medium_ep=result["exact_medium_ep"].numpy(),
        exact_medium_ep_rate=result["exact_medium_ep_rate"].numpy(),
    )

    summary = {
        "params": params.__dict__,
        "resolved_backend": sim.backend,
        "initial": initial_summary,
        "final": final_summary,
        "mean_exact_medium_ep_rate_by_ensemble": ep_rate.tolist(),
    }
    with open(outdir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    if not args.no_plot:
        save_snapshot_figure(
            result["occupancy"],
            outdir / "mips_snapshots.png",
        )

    print(json.dumps(summary, indent=2))
    print(f"Saved outputs under {outdir}")


if __name__ == "__main__":
    main()
