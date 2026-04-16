#!/usr/bin/env python
# coding: utf-8
"""
AMB Sanity Check — PCA-based Degree of Freedom Measurement

Generates AMB 1D trajectories and performs PCA (Principal Component Analysis)
on the ensemble of density field snapshots to measure the effective
"degree of freedom" of the system.

Key Metrics:
  1. Eigenvalue spectrum (sorted in descending order)
  2. Cumulative explained variance ratio
  3. Effective dimension (participation ratio / PR)
     PR = (Σ λ_i)^2 / Σ λ_i^2
     → measures the "effective number of independent components"
  4. Number of components needed to capture 90%, 95%, 99% of variance
"""

import sys
import os
import argparse

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import torch

# ---------- path setup --------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CNEEP_V2_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
if CNEEP_V2_ROOT not in sys.path:
    sys.path.insert(0, CNEEP_V2_ROOT)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from generate_trajectories_1d import ActiveModelB1D


# -----------------------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------------------

def compute_participation_ratio(eigenvalues: np.ndarray) -> float:
    """
    Participation Ratio (PR):
        PR = (Σ λ_i)^2 / Σ λ_i^2

    The PR gives the effective number of significant eigenvalues
    (i.e., the "effective dimension" or "degree of freedom" of the system).
    If all eigenvalues are equal, PR = N (full rank).
    If only one eigenvalue is nonzero, PR = 1.
    """
    total = np.sum(eigenvalues)
    total_sq = np.sum(eigenvalues ** 2)
    if total_sq == 0:
        return 0.0
    return total ** 2 / total_sq


def count_components_for_variance(explained_ratio: np.ndarray,
                                   threshold: float) -> int:
    """Return the smallest k such that cumsum(explained_ratio[:k]) >= threshold."""
    cumsum = np.cumsum(explained_ratio)
    idx = np.searchsorted(cumsum, threshold)
    return int(min(idx + 1, len(explained_ratio)))


# -----------------------------------------------------------------------
# Core analysis
# -----------------------------------------------------------------------

def run_pca_analysis(
    trajectories: np.ndarray,
    n_components: int = None,
    variance_thresholds: list = None,
    normalize: bool = False,
):
    """
    Perform PCA on the AMB snapshot data and return analysis results.

    Parameters
    ----------
    trajectories : np.ndarray
        Shape (M, T, Lx) — M trajectories, T time steps, Lx spatial grid.
        All snapshots are flattened into a (M*T, Lx) matrix before PCA.
    n_components : int, optional
        Number of PCA components to compute.  None → min(n_samples, n_features).
    variance_thresholds : list of float
        E.g. [0.90, 0.95, 0.99].
    normalize : bool, default False
        If True, apply StandardScaler (zero-mean + unit-variance) before PCA.
        This is equivalent to performing PCA on the correlation matrix.
        If False (default), only centering is applied (covariance matrix PCA).

    Returns
    -------
    results : dict
        - eigenvalues           : sorted eigenvalues (descending)
        - explained_variance_ratio : fractional variance per component
        - cumulative_variance   : cumulative variance ratio
        - participation_ratio   : effective DoF (PR)
        - components_for_threshold : {threshold: #components}
        - pca                   : fitted sklearn PCA object
    """
    if variance_thresholds is None:
        variance_thresholds = [0.90, 0.95, 0.99]

    # flatten (M, T, Lx) → (M*T, Lx)
    if trajectories.ndim == 3:
        M, T, Lx = trajectories.shape
        data = trajectories.reshape(M * T, Lx)
    elif trajectories.ndim == 2:
        data = trajectories
    else:
        raise ValueError(f"Unexpected trajectory shape: {trajectories.shape}")

    print(f"[PCA] Data matrix shape: {data.shape}")
    print(f"[PCA] Normalize (StandardScaler): {normalize}")

    if normalize:
        scaler = StandardScaler()
        data = scaler.fit_transform(data)

    if n_components is None:
        n_components = min(data.shape)

    pca = PCA(n_components=n_components)
    pca.fit(data)

    eigenvalues = pca.explained_variance_          # actual eigenvalues
    explained_ratio = pca.explained_variance_ratio_ # fractional
    cumulative = np.cumsum(explained_ratio)

    pr = compute_participation_ratio(eigenvalues)

    comp_for_thr = {}
    for thr in variance_thresholds:
        comp_for_thr[thr] = count_components_for_variance(explained_ratio, thr)

    results = dict(
        eigenvalues=eigenvalues,
        explained_variance_ratio=explained_ratio,
        cumulative_variance=cumulative,
        participation_ratio=pr,
        components_for_threshold=comp_for_thr,
        n_components=n_components,
        pca=pca,
    )
    return results


# -----------------------------------------------------------------------
# Visualization
# -----------------------------------------------------------------------

def plot_pca_results(results: dict, save_path: str = None,
                     show: bool = True):
    """Four-panel PCA summary figure."""
    eigenvalues = results['eigenvalues']
    explained_ratio = results['explained_variance_ratio']
    cumulative = results['cumulative_variance']
    pr = results['participation_ratio']
    comp_thr = results['components_for_threshold']

    n_show = min(len(eigenvalues), 100)  # show at most first 100 components

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"AMB 1D — PCA Degree of Freedom Analysis\n"
        f"Participation Ratio = {pr:.2f}",
        fontsize=14, fontweight='bold'
    )

    # ---- (0,0) Eigenvalue spectrum (linear) ----
    ax = axes[0, 0]
    ax.plot(range(1, n_show + 1), eigenvalues[:n_show],
            'o-', ms=3, lw=1.0, color='steelblue')
    ax.set_xlabel('Principal Component Index')
    ax.set_ylabel('Eigenvalue (Variance)')
    ax.set_title('Eigenvalue Spectrum')
    ax.set_xlim(1, n_show)

    # ---- (0,1) Eigenvalue spectrum (log) ----
    ax = axes[0, 1]
    ax.semilogy(range(1, n_show + 1), eigenvalues[:n_show],
                'o-', ms=3, lw=1.0, color='steelblue')
    ax.set_xlabel('Principal Component Index')
    ax.set_ylabel('Eigenvalue (log scale)')
    ax.set_title('Eigenvalue Spectrum (Log Scale)')
    ax.set_xlim(1, n_show)

    # ---- (1,0) Explained variance ratio ----
    ax = axes[1, 0]
    ax.bar(range(1, n_show + 1), explained_ratio[:n_show],
           color='steelblue', alpha=0.7)
    ax.set_xlabel('Principal Component Index')
    ax.set_ylabel('Explained Variance Ratio')
    ax.set_title('Individual Explained Variance')
    ax.set_xlim(0.5, n_show + 0.5)

    # ---- (1,1) Cumulative variance ----
    ax = axes[1, 1]
    ax.plot(range(1, len(cumulative) + 1), cumulative,
            '-', lw=2, color='crimson')
    for thr, ncomp in comp_thr.items():
        ax.axhline(thr, ls='--', lw=0.8, color='grey')
        ax.axvline(ncomp, ls=':', lw=0.8, color='grey')
        ax.annotate(f'{thr*100:.0f}% → {ncomp} PCs',
                    xy=(ncomp, thr), fontsize=9,
                    xytext=(ncomp + 3, thr - 0.03),
                    arrowprops=dict(arrowstyle='->', color='black'),
                    color='black')
    # Mark PR point
    pr_int = int(round(pr))
    if pr_int <= len(cumulative):
        ax.axvline(pr_int, ls='-', lw=1.5, color='green', alpha=0.7,
                   label=f'PR ≈ {pr:.1f}')
        ax.legend(loc='lower right')
    ax.set_xlabel('Number of Principal Components')
    ax.set_ylabel('Cumulative Explained Variance')
    ax.set_title('Cumulative Explained Variance')
    ax.set_ylim(0, 1.05)

    plt.tight_layout(rect=[0, 0, 1, 0.93])

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[PCA] Figure saved to {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


# -----------------------------------------------------------------------
# Main entry point
# -----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="AMB 1D — PCA Degree of Freedom measurement"
    )

    # AMB model parameters
    parser.add_argument('--Lx', type=int, default=256, help='Grid size')
    parser.add_argument('--dx', type=float, default=1.0)
    parser.add_argument('--a', type=float, default=0.125)
    parser.add_argument('--b', type=float, default=0.125)
    parser.add_argument('--kappa', type=float, default=8.0)
    parser.add_argument('--lam', type=float, default=20.0)
    parser.add_argument('--D', type=float, default=0.1)
    parser.add_argument('--dt', type=float, default=0.01)
    parser.add_argument('--smooth', action='store_true', default=True)
    parser.add_argument('--backend', type=str, default='torch',
                        choices=['numpy', 'torch'])
    parser.add_argument('--use_gpu', action='store_true',
                        default=torch.cuda.is_available())
    parser.add_argument('--bc', type=str, default='periodic')
    parser.add_argument('--epr_mu_active_only', action='store_true',
                        default=False)

    # Trajectory parameters
    parser.add_argument('--n_trajs', type=int, default=500,
                        help='Number of ensemble trajectories')
    parser.add_argument('--n_steps', type=int, default=500,
                        help='Number of time steps per trajectory')
    parser.add_argument('--burn_in', type=int, default=10000,
                        help='Burn-in steps')
    parser.add_argument('--skip', type=int, default=1,
                        help='Sub-sampling interval (take every skip-th frame)')
    parser.add_argument('--seed', type=int, default=42)

    # PCA parameters
    parser.add_argument('--n_components', type=int, default=None,
                        help='Number of PCA components (default: min(samples, features))')

    # Output
    parser.add_argument('--save_path', type=str, default=None,
                        help='Path to save the figure')
    parser.add_argument('--no_show', action='store_true',
                        help='Do not show the plot (useful for batch runs)')

    args = parser.parse_args()

    # ── set seeds ──────────────────────────────────────────────────────
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ── build model ────────────────────────────────────────────────────
    kwargs = dict(
        Lx=args.Lx, dx=args.dx, a=args.a, b=args.b,
        kappa=args.kappa, lam=args.lam, D=args.D, dt=args.dt,
        smooth=args.smooth, backend=args.backend,
        use_gpu=args.use_gpu, bc=args.bc,
        epr_mu_active_only=args.epr_mu_active_only,
    )
    print(f"[INFO] AMB 1D parameters: {kwargs}")
    model = ActiveModelB1D(**kwargs)

    # ── generate trajectories ──────────────────────────────────────────
    print(f"[INFO] Generating {args.n_trajs} trajectories × {args.n_steps} steps "
          f"(burn-in={args.burn_in}) ...")
    trajectories = model.generate_trajectories(
        n_trajectories=args.n_trajs,
        n_steps=args.n_steps,
        burn_in=args.burn_in,
    )
    print(f"[INFO] Trajectory shape: {trajectories.shape}")

    # sub-sample if requested
    if args.skip > 1:
        trajectories = trajectories[:, ::args.skip, :]
        print(f"[INFO] After sub-sampling (skip={args.skip}): {trajectories.shape}")

    # ── PCA analysis ───────────────────────────────────────────────────
    results = run_pca_analysis(
        trajectories,
        n_components=args.n_components,
    )

    # ── print summary ──────────────────────────────────────────────────
    pr = results['participation_ratio']
    print(f"\n{'='*60}")
    print(f"PCA Degree of Freedom Summary")
    print(f"{'='*60}")
    print(f"  Total data points:        {trajectories.shape[0] * trajectories.shape[1]}")
    print(f"  Feature dimension (Lx):   {trajectories.shape[2]}")
    print(f"  Number of PCA components: {results['n_components']}")
    print(f"  Participation Ratio (PR): {pr:.2f}")
    for thr, ncomp in results['components_for_threshold'].items():
        print(f"  Components for {thr*100:.0f}% variance: {ncomp}")
    print(f"{'='*60}\n")

    # ── plot ────────────────────────────────────────────────────────────
    save_path = args.save_path
    if save_path is None:
        save_path = os.path.join(SCRIPT_DIR, 'pca_degree_of_freedom.png')

    plot_pca_results(results, save_path=save_path, show=not args.no_show)


if __name__ == '__main__':
    main()
