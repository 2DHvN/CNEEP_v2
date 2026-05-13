"""
Lattice ABP — Visualization Module
====================================
Renders lattice states with color-coded particles:
  Red:  Jammed (front neighbor occupied)
  Blue: Free (can move forward)

Orientation shown via arrow markers.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import torch
from typing import Optional


# Arrow markers for each orientation: 0=up, 1=right, 2=down, 3=left
ARROW_MARKERS = {0: "^", 1: ">", 2: "v", 3: "<"}
ARROW_DY = {0: -0.15, 1: 0.0, 2: 0.15, 3: 0.0}
ARROW_DX = {0: 0.0, 1: 0.15, 2: 0.0, 3: -0.15}


def visualize_state(
    O: torch.Tensor,
    E: torch.Tensor,
    jammed: torch.Tensor,
    ensemble_idx: int = 0,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
    figsize: tuple = (8, 8),
    show_arrows: bool = True,
    cmap_bg: str = "#1a1a2e",
    color_jammed: str = "#e74c3c",
    color_free: str = "#3498db",
    save_path: Optional[str] = None,
):
    """Visualize a single ensemble's lattice state.

    Args:
        O: (B, L, L) occupancy tensor.
        E: (B, L, L) orientation tensor.
        jammed: (B, L, L) boolean jammed mask.
        ensemble_idx: Which ensemble member to visualize.
        ax: Matplotlib axes (created if None).
        title: Plot title.
        figsize: Figure size.
        show_arrows: Whether to draw orientation arrows.
        cmap_bg: Background color.
        color_jammed: Color for jammed particles.
        color_free: Color for free particles.
        save_path: If provided, save figure to this path.
    """
    # Extract single ensemble
    occ = O[ensemble_idx].cpu().numpy()
    orient = E[ensemble_idx].cpu().numpy()
    jam = jammed[ensemble_idx].cpu().numpy()

    L = occ.shape[0]

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize, facecolor="#0d1117")
    else:
        fig = ax.figure

    ax.set_facecolor(cmap_bg)

    # Draw grid background
    ax.set_xlim(-0.5, L - 0.5)
    ax.set_ylim(L - 0.5, -0.5)  # invert y for matrix convention
    ax.set_aspect("equal")

    # Draw particles
    particle_rows, particle_cols = np.where(occ == 1)

    for r, c in zip(particle_rows, particle_cols):
        is_jammed = jam[r, c]
        color = color_jammed if is_jammed else color_free
        e = orient[r, c]

        # Draw filled circle
        circle = plt.Circle(
            (c, r), 0.4,
            facecolor=color,
            edgecolor="white",
            linewidth=0.5,
            alpha=0.9,
        )
        ax.add_patch(circle)

        # Draw orientation arrow
        if show_arrows:
            marker = ARROW_MARKERS.get(e, "o")
            ax.plot(
                c, r, marker=marker,
                color="white", markersize=6,
                markeredgewidth=0.8,
            )

    # Grid lines
    for i in range(L + 1):
        ax.axhline(i - 0.5, color="#2d2d4e", linewidth=0.3, alpha=0.5)
        ax.axvline(i - 0.5, color="#2d2d4e", linewidth=0.3, alpha=0.5)

    # Legend
    legend_patches = [
        mpatches.Patch(color=color_jammed, label="Jammed (blocked)"),
        mpatches.Patch(color=color_free, label="Free (mobile)"),
    ]
    ax.legend(
        handles=legend_patches,
        loc="upper right",
        fontsize=9,
        facecolor="#1a1a2e",
        edgecolor="#444",
        labelcolor="white",
    )

    # Labels
    if title:
        ax.set_title(title, color="white", fontsize=14, fontweight="bold", pad=10)
    ax.tick_params(colors="white", labelsize=7)
    ax.set_xlabel("Column", color="white", fontsize=10)
    ax.set_ylabel("Row", color="white", fontsize=10)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"Saved: {save_path}")

    return fig, ax


def visualize_density_evolution(
    O_traj: torch.Tensor,
    ensemble_idx: int = 0,
    n_snapshots: int = 6,
    figsize: tuple = (18, 10),
    save_path: Optional[str] = None,
):
    """Plot density field snapshots at several time points.

    Args:
        O_traj: (n_saved, B, L, L) trajectory tensor.
        ensemble_idx: Ensemble member to plot.
        n_snapshots: Number of snapshots to show.
        figsize: Figure size.
        save_path: Path to save figure.
    """
    n_saved = O_traj.shape[0]
    indices = np.linspace(0, n_saved - 1, n_snapshots, dtype=int)

    fig, axes = plt.subplots(
        1, n_snapshots, figsize=figsize,
        facecolor="#0d1117",
    )

    for i, (ax, idx) in enumerate(zip(axes, indices)):
        occ = O_traj[idx, ensemble_idx].cpu().numpy().astype(float)

        cmap = ListedColormap(["#1a1a2e", "#3498db"])
        ax.imshow(occ, cmap=cmap, interpolation="nearest", vmin=0, vmax=1)
        ax.set_title(f"t = {idx}", color="white", fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor("#0d1117")

    fig.suptitle(
        "Density Evolution (Lattice ABP)",
        color="white", fontsize=16, fontweight="bold", y=1.02,
    )
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"Saved: {save_path}")

    return fig


def visualize_jammed_fraction(
    O_traj: torch.Tensor,
    E_traj: torch.Tensor,
    sim,
    figsize: tuple = (10, 5),
    save_path: Optional[str] = None,
):
    """Plot jammed fraction over time, averaged across ensembles.

    Args:
        O_traj: (n_saved, B, L, L)
        E_traj: (n_saved, B, L, L)
        sim: LatticeABP simulator instance.
        figsize: Figure size.
        save_path: Path to save.
    """
    n_saved, B, L, _ = O_traj.shape
    fractions = []

    for t in range(n_saved):
        O_t = O_traj[t].to(sim.device)
        E_t = E_traj[t].to(sim.device)
        jammed = sim.compute_jammed_mask(O_t, E_t)
        n_particles = O_t.sum(dim=(-2, -1)).float()
        n_jammed = (jammed & (O_t == 1)).sum(dim=(-2, -1)).float()
        frac = (n_jammed / n_particles.clamp(min=1)).mean().item()
        fractions.append(frac)

    fig, ax = plt.subplots(figsize=figsize, facecolor="#0d1117")
    ax.set_facecolor("#1a1a2e")
    ax.plot(fractions, color="#e74c3c", linewidth=1.5, alpha=0.8)
    ax.fill_between(range(len(fractions)), fractions, alpha=0.15, color="#e74c3c")
    ax.set_xlabel("Saved Step", color="white", fontsize=12)
    ax.set_ylabel("Jammed Fraction", color="white", fontsize=12)
    ax.set_title("Jammed Fraction over Time", color="white", fontsize=14, fontweight="bold")
    ax.tick_params(colors="white")
    ax.grid(alpha=0.2, color="white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#444")
    ax.spines["bottom"].set_color("#444")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"Saved: {save_path}")

    return fig
