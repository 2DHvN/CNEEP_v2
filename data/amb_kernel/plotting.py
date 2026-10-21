"""Export notebook-ready field movies and signed EP maps without ffmpeg."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import TwoSlopeNorm
import numpy as np
from tqdm.auto import tqdm

from .diagnostics import json_safe


def _symmetric_norm(*arrays):
    limit = max(max(float(np.abs(array).max()) for array in arrays), 1e-12)
    return TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)


def _movie(fig, update, count, path, *, fps, description, show_progress):
    """JSHTML frame export with an explicit upper bound on embedded PNG bytes."""
    seen = set()
    with tqdm(total=count, desc=description, unit="frame", mininterval=.5,
              leave=False, disable=not show_progress) as bar:
        def render(index):
            artists = update(index)
            if index not in seen:
                seen.add(index)
                bar.update(1)
            return artists
        movie = FuncAnimation(fig, render, frames=count, interval=1000 / fps,
                              blit=False, repeat=True, cache_frame_data=False)
        width, height = fig.get_size_inches() * fig.dpi
        # PNG/base64 cannot reasonably exceed twice the raw RGBA canvas size.
        # Set the cap for this finite, deliberately selected video, rather than
        # accepting Matplotlib's default cap that silently drops later frames.
        limit_mb = max(20.0, count * width * height * 8 / 1024 ** 2)
        try:
            with mpl.rc_context({"animation.embed_limit": limit_mb}):
                html = movie.to_jshtml(fps=fps, embed_frames=True, default_mode="loop")
            embedded = html.count("data:image/")
            if len(seen) != count or embedded != count:
                raise RuntimeError(f"Movie export rendered {len(seen)} and embedded {embedded} of {count} frames.")
            path.write_text("<!doctype html><html><head><meta charset='utf-8'></head><body>" +
                            html + "</body></html>\n", encoding="utf-8")
        finally:
            plt.close(fig)


def render_training_visualizations(analysis: dict, output_dir, *, model_id="", fps=10,
                                   show_progress=True) -> dict[str, Path]:
    """Save profile/EP movies, all-pair mean maps, and a signed shell spectrum.

    Maps are local rate contributions whose spatial SUM is extensive EP/time.
    The known map is a finite-lag active-work convention; raw NN maps and shell
    branches need not share that local attribution. No extra N, D, or affine
    normalization factor is applied here: the analyzer has already set units.
    """
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be positive and finite.")
    required = ("video_phi", "video_known_ep_rate", "video_nn_ep_rate",
                "video_time_start", "video_time_mid", "video_time_end",
                "mean_known_ep_rate", "mean_nn_ep_rate")
    arrays = {key: np.asarray(analysis[key]) for key in required}
    phi = arrays["video_phi"]
    if phi.ndim != 3 or len(phi) == 0:
        raise ValueError("video_phi must contain at least one [Ny,Nx] frame.")
    count = len(phi)
    for key in ("video_known_ep_rate", "video_nn_ep_rate"):
        if arrays[key].shape != phi.shape:
            raise ValueError(f"{key} must have the same frame/grid shape as video_phi.")
    for key in ("video_time_start", "video_time_mid", "video_time_end"):
        if arrays[key].shape != (count,):
            raise ValueError(f"{key} must have one time per movie frame.")
    for key in ("mean_known_ep_rate", "mean_nn_ep_rate"):
        if arrays[key].shape != phi.shape[1:]:
            raise ValueError(f"{key} must match the spatial lattice.")
    if not all(np.isfinite(value).all() for value in arrays.values()):
        raise ValueError("Visualizations require finite maps and physical times.")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    paths = {"profile_video_html": destination / "profile_video.html",
             "ep_map_video_html": destination / "ep_map_video.html",
             "time_averaged_ep_map_png": destination / "time_averaged_ep_map.png",
             "ep_spectrum_png": destination / "ep_spectrum.png",
             "visualization_data_npz": destination / "visualization_data.npz",
             "ep_spectrum_csv": destination / "ep_spectrum.csv"}
    prefix = f"{model_id} " if model_id else ""
    fig, axis = plt.subplots(figsize=(5.5, 5), dpi=100)
    image = axis.imshow(phi[0], origin="lower", interpolation="nearest", cmap="RdBu_r",
                        norm=_symmetric_norm(phi))
    axis.set(xlabel="x (lattice site)", ylabel="y (lattice site)")
    fig.colorbar(image, ax=axis, shrink=.82, label=r"physical $\phi$")
    title = axis.set_title("")
    fig.tight_layout()
    def profile_update(index):
        image.set_data(phi[index])
        title.set_text(f"{prefix}field | t={arrays['video_time_start'][index]:.5g}")
        return image, title
    _movie(fig, profile_update, count, paths["profile_video_html"], fps=fps,
           description=f"{prefix}profile video export", show_progress=show_progress)

    known, neural = arrays["video_known_ep_rate"], arrays["video_nn_ep_rate"]
    norm = _symmetric_norm(known, neural)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6), dpi=100, layout="constrained")
    images = [axis.imshow(value[0], origin="lower", interpolation="nearest", cmap="RdBu_r", norm=norm)
              for axis, value in zip(axes, (known, neural))]
    for axis, label in zip(axes, ("Known active work (finite lag)", "Learned NN EP")):
        axis.set(title=label, xlabel="x (lattice site)", ylabel="y (lattice site)")
    fig.colorbar(images[0], ax=axes, shrink=.8, label="local EP rate (per site / time)")
    title = fig.suptitle("")
    def ep_update(index):
        images[0].set_data(known[index]); images[1].set_data(neural[index])
        title.set_text(f"{prefix}EP maps | t ∈ [{arrays['video_time_start'][index]:.5g}, "
                       f"{arrays['video_time_end'][index]:.5g}]")
        return *images, title
    _movie(fig, ep_update, count, paths["ep_map_video_html"], fps=fps,
           description=f"{prefix}EP video export", show_progress=show_progress)

    mean_known, mean_neural = arrays["mean_known_ep_rate"], arrays["mean_nn_ep_rate"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6), dpi=130, layout="constrained")
    norm = _symmetric_norm(mean_known, mean_neural)
    for axis, value, label in zip(axes, (mean_known, mean_neural),
                                 ("Time mean: known active work", "Time mean: learned NN EP")):
        image = axis.imshow(value, origin="lower", interpolation="nearest", cmap="RdBu_r", norm=norm)
        axis.set(title=label, xlabel="x (lattice site)", ylabel="y (lattice site)")
    fig.colorbar(image, ax=axes, shrink=.8, label="mean local EP rate (per site / time)")
    pairs = analysis.get("metadata", {}).get("total_pairs")
    fig.suptitle(f"{prefix}all test pairs" + (f" (n={pairs:,})" if pairs is not None else ""))
    fig.savefig(paths["time_averaged_ep_map_png"], dpi=150)
    plt.close(fig)

    known_labels, nn_labels = list(analysis["known_shell_labels"]), list(analysis["nn_shell_labels"])
    known_rates = np.asarray(analysis["known_shell_rates"], dtype=float)
    nn_rates = np.asarray(analysis["nn_shell_rates"], dtype=float)
    if known_rates.shape != (len(known_labels),) or nn_rates.shape != (len(nn_labels),):
        raise ValueError("Shell labels and rates must have matching lengths.")
    labels = sorted(set(known_labels + nn_labels), key=lambda label: (label == "tail", int(label[1:]) if label.startswith("r") else 0))
    positions = {label: index for index, label in enumerate(labels)}
    fig, axis = plt.subplots(figsize=(8.5, 4.7), dpi=130, layout="constrained")
    axis.bar([positions[label] - .2 for label in known_labels], known_rates, width=.4,
             label="Known active-work attribution (finite lag)", color="darkorange")
    axis.bar([positions[label] + .2 for label in nn_labels], nn_rates, width=.4,
             label="Learned raw NN branch", color="steelblue")
    axis.axhline(0, color="black", linewidth=.8)
    axis.set_xticks(np.arange(len(labels)), labels)
    axis.set(xlabel="Chebyshev shell / physical tail", ylabel="signed extensive EP rate (1 / time)",
             title=f"{prefix}EP shell spectrum")
    axis.legend(fontsize="small")
    fig.savefig(paths["ep_spectrum_png"], dpi=150)
    plt.close(fig)

    with paths["ep_spectrum_csv"].open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("quantity", "shell", "mean_extensive_rate"))
        writer.writeheader()
        for quantity, shell_labels, values in (("known_active_work_finite_lag", known_labels, known_rates),
                                               ("NN_raw_branch", nn_labels, nn_rates)):
            writer.writerows({"quantity": quantity, "shell": label, "mean_extensive_rate": float(value)}
                            for label, value in zip(shell_labels, values))
    data = {key: value for key, value in analysis.items() if isinstance(value, np.ndarray)}
    data.update(known_shell_labels=np.asarray(known_labels), nn_shell_labels=np.asarray(nn_labels),
                known_channel_labels=np.asarray(analysis.get("known_channel_labels", [])),
                metadata_json=np.asarray(json.dumps(json_safe(analysis.get("metadata", {})), ensure_ascii=False)))
    np.savez_compressed(paths["visualization_data_npz"], **data)
    return paths
