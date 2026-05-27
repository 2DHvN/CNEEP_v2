"""
SAOU Field — Trajectory Generator (CLI)

Generates trajectories from the Shell-resolved Antisymmetric Ornstein-Uhlenbeck
field model and saves them alongside EPR metrics to an .npz file.

Usage:
    python generate_trajectories.py
    python generate_trajectories.py --L 64 --radii 1 2 4 8 --amplitudes 1.0 0.5 2.0 0.0
    python generate_trajectories.py --omega0 1.0 --n_steps 50000 --output saou_traj.npz
"""

import numpy as np
import argparse
import os
import json

try:
    from tqdm import trange
except ImportError:
    trange = range

from saou_model import simulate


# ======================================================================
# Save utility
# ======================================================================

def save_results(out: dict, output_path: str) -> None:
    """Save simulation results to .npz, excluding non-serialisable objects."""
    save_dict = {}

    # Trajectory
    if out["trajectory"] is not None:
        save_dict["trajectory"] = out["trajectory"]

    # Scalar EPR estimates
    save_dict["epr_rate_est_local"] = np.float64(out["epr_rate_est_local"])
    save_dict["epr_rate_sem_local"] = np.float64(out["epr_rate_sem_local"])
    save_dict["epr_rate_est_by_shell"] = out["epr_rate_est_by_shell"]
    save_dict["epr_rate_sem_by_shell"] = out["epr_rate_sem_by_shell"]
    save_dict["epr_rate_est_total"] = np.float64(out["epr_rate_est_total"])
    save_dict["epr_rate_sem_total"] = np.float64(out["epr_rate_sem_total"])

    # Theory
    save_dict["epr_rate_theory_gram"] = out["epr_rate_theory_gram"]
    save_dict["epr_rate_theory_self_local"] = np.float64(out["epr_rate_theory_self_local"])
    save_dict["epr_rate_theory_self_by_shell"] = out["epr_rate_theory_self_by_shell"]
    save_dict["epr_rate_theory_path_local"] = np.float64(out["epr_rate_theory_path_local"])
    save_dict["epr_rate_theory_path_by_shell"] = out["epr_rate_theory_path_by_shell"]
    save_dict["epr_rate_theory_total"] = np.float64(out["epr_rate_theory_total"])

    # Per-step increments
    save_dict["epr_increment_by_step_local"] = out["epr_increment_by_step_local"]
    save_dict["epr_increment_by_step_shell"] = out["epr_increment_by_step_shell"]
    save_dict["epr_increment_by_step_total"] = out["epr_increment_by_step_total"]

    # Params (as JSON string)
    save_dict["params_json"] = np.array(json.dumps(out["params"]))

    np.savez(output_path, **save_dict)
    print(f"[OK] Results saved → {output_path}")


# ======================================================================
# CLI
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description="SAOU field trajectory generator (Euler-Maruyama)"
    )

    # Lattice
    parser.add_argument("--L", type=int, default=32,
                        help="Linear lattice size (default: 32)")

    # Shell configuration
    parser.add_argument("--radii", type=float, nargs="+", default=[1, 2, 4, 8],
                        help="Upper radii of annular shells (default: 1 2 4 8)")
    parser.add_argument("--amplitudes", type=float, nargs="+", default=[1.0, 0.5, 2.0, 0.0],
                        help="Antisymmetric shell amplitudes a_s (default: 1.0 0.5 2.0 0.0)")

    # Physical parameters
    parser.add_argument("--gamma", type=float, default=1.0,
                        help="Local damping rate (default: 1.0)")
    parser.add_argument("--omega0", type=float, default=0.0,
                        help="Local torque amplitude (default: 0.0)")
    parser.add_argument("--T", type=float, default=1.0,
                        help="Noise temperature (default: 1.0)")

    # Integration
    parser.add_argument("--dt", type=float, default=1e-3,
                        help="Time step (default: 1e-3)")
    parser.add_argument("--n_steps", type=int, default=50000,
                        help="Number of production steps (default: 50000)")
    parser.add_argument("--burn_steps", type=int, default=10000,
                        help="Burn-in steps (default: 10000)")
    parser.add_argument("--sample_every", type=int, default=10,
                        help="Save trajectory every N steps (default: 10)")

    # Misc
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed (default: 0)")
    parser.add_argument("--weight_norm", type=str, default="mean",
                        choices=["mean", "none"],
                        help="Shell weight normalisation (default: mean)")
    parser.add_argument("--no_trajectory", action="store_true",
                        help="Skip recording trajectory (save only EPR stats)")
    parser.add_argument("--output", type=str, default="saou_trajectories.npz",
                        help="Output .npz file path (default: saou_trajectories.npz)")

    args = parser.parse_args()

    # Validate
    if len(args.radii) != len(args.amplitudes):
        parser.error("--radii and --amplitudes must have the same number of elements.")

    print(f"[INFO] SAOU Field  L={args.L}  shells={len(args.radii)}")
    print(f"  radii      = {args.radii}")
    print(f"  amplitudes = {args.amplitudes}")
    print(f"  gamma={args.gamma}, omega0={args.omega0}, T={args.T}")
    print(f"  dt={args.dt}, n_steps={args.n_steps}, burn_steps={args.burn_steps}")
    print(f"  sample_every={args.sample_every}, seed={args.seed}")
    print()

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
        record_trajectory=not args.no_trajectory,
        weight_normalization=args.weight_norm,
        show_progress=True,
    )

    # Print EPR comparison
    print()
    print("=" * 64)
    print("Local + shell EPR rate: estimated vs analytic")
    print("=" * 64)
    print(
        f"{'local torque':14s}  "
        f"est={out['epr_rate_est_local']:10.3f} ± {out['epr_rate_sem_local']:8.3f}   "
        f"theory={out['epr_rate_theory_path_local']:10.3f}"
    )

    for sh, est, sem, th in zip(
        out["shells"],
        out["epr_rate_est_by_shell"],
        out["epr_rate_sem_by_shell"],
        out["epr_rate_theory_path_by_shell"],
    ):
        print(f"{sh.name:14s}  est={est:10.3f} ± {sem:8.3f}   theory={th:10.3f}")

    print(
        f"{'total':14s}  "
        f"est={out['epr_rate_est_total']:10.3f}       "
        f"theory={out['epr_rate_theory_total']:10.3f}"
    )
    print()

    # Save
    save_results(out, args.output)
    print("\nDone!")


if __name__ == "__main__":
    main()
