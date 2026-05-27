"""
SAOU Field — Sanity Check

Runs a short simulation and compares the estimated pathwise EPR rates
against the analytic theory to verify that the model is implemented correctly.

Usage:
    python sanity_check_saou.py
"""

import numpy as np
import sys
import os

from saou_model import simulate


def run_sanity_check():
    """Run a quick sanity check with the default SAOU parameters."""

    print("=" * 70)
    print("SAOU Field — Sanity Check")
    print("=" * 70)
    print()

    # Run with the exact same parameters as the notebook
    out = simulate(
        L=32,
        radii=(1, 2, 4, 8),
        amplitudes=(1.0, 0.5, 2.0, 0.0),
        gamma=1.0,
        omega0=0.0,
        T=1.0,
        dt=1e-3,
        n_steps=20_000,
        burn_steps=10_000,
        sample_every=10,
        seed=1,
        record_trajectory=True,
        show_progress=True,
    )

    # Expected values from the notebook output
    expected_theory = {
        "local torque": 0.000,
        "S0_r(0,1]": 7680.000,
        "S1_r(1,2]": 3648.000,
        "S2_r(2,4]": 14563.556,
        "S3_r(4,8]": 0.000,
        "total": 25891.556,
    }

    print("Local + shell EPR rate: estimated vs analytic")
    print("-" * 70)

    all_ok = True

    # Local torque
    est_local = out["epr_rate_est_local"]
    sem_local = out["epr_rate_sem_local"]
    th_local = out["epr_rate_theory_path_local"]
    print(
        f"{'local torque':14s}  "
        f"est={est_local:10.3f} ± {sem_local:8.3f}   "
        f"theory={th_local:10.3f}"
    )
    if abs(th_local - expected_theory["local torque"]) > 0.01:
        print("  *** THEORY MISMATCH for local torque!")
        all_ok = False

    # Shells
    for sh, est, sem, th in zip(
        out["shells"],
        out["epr_rate_est_by_shell"],
        out["epr_rate_sem_by_shell"],
        out["epr_rate_theory_path_by_shell"],
    ):
        print(f"{sh.name:14s}  est={est:10.3f} ± {sem:8.3f}   theory={th:10.3f}")

        if sh.name in expected_theory:
            if abs(th - expected_theory[sh.name]) > 0.01:
                print(f"  *** THEORY MISMATCH for {sh.name}!")
                all_ok = False

        # Check that estimated is within reasonable range of theory
        # (within 5% or 3 SEM)
        if th != 0 and sem > 0:
            rel_err = abs(est - th) / abs(th)
            z_score = abs(est - th) / sem
            if rel_err > 0.05 and z_score > 3:
                print(f"  *** WARNING: estimate deviates from theory "
                      f"(rel_err={rel_err:.4f}, z={z_score:.1f})")

    # Total
    est_total = out["epr_rate_est_total"]
    th_total = out["epr_rate_theory_total"]
    print(
        f"{'total':14s}  "
        f"est={est_total:10.3f}       "
        f"theory={th_total:10.3f}"
    )
    if abs(th_total - expected_theory["total"]) > 0.1:
        print("  *** THEORY MISMATCH for total!")
        all_ok = False

    print("-" * 70)

    # Verify trajectory shape
    traj = out["trajectory"]
    L = out["params"]["L"]
    expected_n_samples = out["params"]["n_steps"] // out["params"]["sample_every"]
    print(f"\nTrajectory shape: {traj.shape}")
    print(f"  Expected: ({expected_n_samples}, {L}, {L}, 2)")

    if traj.shape != (expected_n_samples, L, L, 2):
        print("  *** TRAJECTORY SHAPE MISMATCH!")
        all_ok = False

    # Verify stationary covariance: <X^2> should be T/gamma = 1.0
    var_u = np.var(traj[..., 0])
    var_v = np.var(traj[..., 1])
    print(f"\nStationary variance check (expected T/gamma = 1.0):")
    print(f"  Var(u) = {var_u:.4f}")
    print(f"  Var(v) = {var_v:.4f}")

    if abs(var_u - 1.0) > 0.1 or abs(var_v - 1.0) > 0.1:
        print("  *** WARNING: variance deviates > 10% from expected")

    # Print Gram matrix
    print(f"\nEPR Gram matrix:")
    gram = out["epr_rate_theory_gram"]
    labels = ["local"] + [sh.name for sh in out["shells"]]
    print(f"{'':14s}  " + "  ".join(f"{l:>14s}" for l in labels))
    for i, l in enumerate(labels):
        row = "  ".join(f"{gram[i, j]:14.3f}" for j in range(gram.shape[1]))
        print(f"{l:14s}  {row}")

    print()
    if all_ok:
        print("✓ All sanity checks PASSED.")
    else:
        print("✗ Some sanity checks FAILED. See details above.")

    return all_ok


if __name__ == "__main__":
    ok = run_sanity_check()
    sys.exit(0 if ok else 1)
