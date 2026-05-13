"""
Lattice ABP — Demo & CLI Runner
=================================
Runs a lattice ABP simulation and produces visualizations.

Usage:
    python run_demo.py --L 32 --density 0.6 --v_plus 5.0 --n_steps 5000
"""

import argparse
import time
import sys
import os

import torch
import numpy as np

# Allow running from this directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import LatticeABP, InteractionModule
from visualization import (
    visualize_state,
    visualize_density_evolution,
    visualize_jammed_fraction,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Lattice ABP Simulation Demo")

    # Lattice
    parser.add_argument("--L", type=int, default=32, help="Lattice side length")
    parser.add_argument("--density", type=float, default=0.5, help="Particle density")

    # Rates
    parser.add_argument("--v_plus", type=float, default=5.0, help="Forward hop rate")
    parser.add_argument("--v_zero", type=float, default=1.0, help="Lateral hop rate")
    parser.add_argument("--v_minus", type=float, default=0.2, help="Backward hop rate")
    parser.add_argument("--D_rot", type=float, default=0.5, help="Rotational diffusion rate")

    # Simulation
    parser.add_argument("--B", type=int, default=4, help="Ensemble size")
    parser.add_argument("--n_steps", type=int, default=5000, help="MC steps")
    parser.add_argument("--burn_in", type=int, default=2000, help="Burn-in steps")
    parser.add_argument("--method", type=str, default="gillespie",
                        choices=["gillespie", "tau_leap"])
    parser.add_argument("--tau", type=float, default=0.01, help="Tau-leaping step")
    parser.add_argument("--save_interval", type=int, default=50,
                        help="Save state every N steps")

    # Boundary
    parser.add_argument("--bc", type=str, default="periodic",
                        choices=["periodic", "hard_wall"])

    # Misc
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="./output")

    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    Pe = (args.v_plus - args.v_minus) / (2 * args.D_rot) if args.D_rot > 0 else float("inf")

    print("=" * 60)
    print("  Lattice Active Brownian Particle Simulation")
    print("  Based on: Whitelam, Klymko, Mandal (2018)")
    print("=" * 60)
    print(f"  L={args.L}, ρ={args.density}, N={int(args.density * args.L**2)}")
    print(f"  v+={args.v_plus}, v0={args.v_zero}, v-={args.v_minus}")
    print(f"  D_rot={args.D_rot}, Pe={Pe:.2f}")
    print(f"  BC={args.bc}, Method={args.method}")
    print(f"  Ensemble B={args.B}, Steps={args.n_steps}")
    print(f"  Device={args.device}")
    print("=" * 60)

    # Create simulator
    sim = LatticeABP(
        L=args.L,
        v_plus=args.v_plus,
        v_zero=args.v_zero,
        v_minus=args.v_minus,
        D_rot=args.D_rot,
        density=args.density,
        bc_mode=args.bc,
        device=args.device,
        seed=args.seed,
    )

    # Run simulation
    t0 = time.time()
    results = sim.simulate(
        B=args.B,
        n_steps=args.n_steps,
        burn_in=args.burn_in,
        method=args.method,
        tau=args.tau,
        save_interval=args.save_interval,
        show_progress=True,
    )
    elapsed = time.time() - t0
    print(f"\nSimulation completed in {elapsed:.1f}s")

    O_traj = results["O_traj"]
    E_traj = results["E_traj"]
    n_saved = O_traj.shape[0]
    print(f"Saved {n_saved} snapshots")

    # --- Visualize final state ---
    O_final = results["O_final"]
    E_final = results["E_final"]
    jammed_final = sim.compute_jammed_mask(O_final, E_final)

    n_particles = O_final[0].sum().item()
    n_jammed = (jammed_final[0] & (O_final[0] == 1)).sum().item()
    print(f"\nFinal state (ensemble 0): {n_particles} particles, "
          f"{n_jammed} jammed ({100*n_jammed/max(n_particles,1):.1f}%)")

    show_arrows = args.L <= 48  # hide arrows for large lattices

    fig1, _ = visualize_state(
        O_final, E_final, jammed_final,
        ensemble_idx=0,
        title=f"Lattice ABP — L={args.L}, ρ={args.density}, Pe={Pe:.1f}",
        show_arrows=show_arrows,
        save_path=os.path.join(args.output_dir, "lattice_abp_final.png"),
    )

    # --- Density evolution ---
    fig2 = visualize_density_evolution(
        O_traj,
        ensemble_idx=0,
        n_snapshots=min(6, n_saved),
        save_path=os.path.join(args.output_dir, "lattice_abp_evolution.png"),
    )

    # --- Jammed fraction ---
    fig3 = visualize_jammed_fraction(
        O_traj, E_traj, sim,
        save_path=os.path.join(args.output_dir, "lattice_abp_jammed_frac.png"),
    )

    # --- Save raw data ---
    np.savez_compressed(
        os.path.join(args.output_dir, "lattice_abp_results.npz"),
        O_traj=O_traj.numpy(),
        E_traj=E_traj.numpy(),
        times=results["times"].numpy(),
        params=vars(args),
    )
    print(f"\nResults saved to {args.output_dir}/")
    print("Done!")


if __name__ == "__main__":
    main()
