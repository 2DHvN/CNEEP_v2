"""Training bridge for the existing scalar ShellForce network.

The physical fields remain scalar. All branches share a single extensive NEEP
objective; projection against known active channels is a separate diagnostic.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import csv
import hashlib
import json
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
from typing import Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from models.NEEP_ShellForce_2D import MultiScaleShellForceCNEEP2D
from .model import laplacian

SPEC_SOURCE_SHA256 = "baea2ddf87ed10ac5c4150617e8a24556f6d6490c54009b671d4254d1a79cf79"
SOURCE_PATH = Path(__file__).resolve().parents[2] / "models" / "NEEP_ShellForce_2D.py"


@dataclass(frozen=True)
class ArchitectureConfig:
    """Keep the established Corr_AMB2D_ShellForce settings across ALL models."""
    n_channel: int = 64
    n_hidden: int = 3
    max_distance: int = 3
    shell_force_activation: str = "elu"

    def options(self):
        if min(self.n_channel, self.n_hidden) < 1 or self.max_distance < 0:
            raise ValueError("Architecture dimensions must be positive (R may be zero).")
        return SimpleNamespace(
            **asdict(self), n_components=1, ep_component_indices=(0,),
            n_external_ep_components=0, include_k0=True, positional=False,
            shell_center_mode="relative_only", shell_relative_mode="learned_absolute",
            shell_force_bias=False, shell_weight_normalization="none", beta=0.0,
        )


@dataclass(frozen=True)
class TrainingConfig:
    iterations: int = 300
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    alpha: float = -0.5
    clip_norm: float = 1.0
    record_every: int = 25
    seed: int = 3
    device: str = "cpu"
    dtype: str = "float64"
    lag_frames: int = 1
    eval_batch_size: int | None = None
    micro_batch_size: int | None = 32

    def __post_init__(self):
        if min(self.iterations, self.batch_size, self.record_every, self.lag_frames) < 1:
            raise ValueError("Iterations, batch, recording interval and lag must be positive.")
        for name in ("eval_batch_size", "micro_batch_size"):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 1):
                raise ValueError(f"{name} must be a positive integer or None.")
        if self.dtype not in {"float32", "float64"}:
            raise ValueError("dtype must be float32 or float64.")
        if not -1 < self.alpha <= 0 or self.learning_rate <= 0 or self.clip_norm <= 0:
            raise ValueError("Require -1 < alpha <= 0, positive learning rate and clip norm.")


def _fields(array, *, check_finite=True):
    x = np.asarray(array)
    if x.ndim != 4 or x.shape[1] != 1 or x.shape[-1] != x.shape[-2]:
        raise ValueError("Expected scalar fields [time, 1, Ny, Nx] on a square lattice.")
    if len(x) < 2 or (check_finite and not np.isfinite(x).all()):
        raise ValueError("Need at least two finite frames.")
    return x


@dataclass
class TrainingSplits:
    train: list[np.ndarray]
    validation: list[np.ndarray]
    test: list[np.ndarray]
    metadata: dict = field(default_factory=dict)


def contiguous_splits(fields, train_fraction=0.6, validation_fraction=0.2):
    """Split frames BEFORE forming pairs; no shared endpoints across splits."""
    x = _fields(fields)
    if not 0 < train_fraction < 1 or not 0 < validation_fraction < 1 - train_fraction:
        raise ValueError("Split fractions must leave a nonempty test split.")
    cut1 = int(len(x) * train_fraction)
    cut2 = int(len(x) * (train_fraction + validation_fraction))
    pieces = [x[:cut1], x[cut1:cut2], x[cut2:]]
    for piece in pieces:
        _fields(piece)
    return TrainingSplits(*[[piece] for piece in pieces], metadata={
        "method": "nonoverlapping_contiguous_frames_before_pairing",
        "frame_ranges_half_open": [[0, cut1], [cut1, cut2], [cut2, len(x)]],
        "caution": "Disjoint endpoints avoid leakage; mixing between blocks still requires stationarity/autocorrelation checks.",
    })


def seed_splits(trajectories: Mapping[int, np.ndarray], train_seeds, validation_seeds, test_seeds):
    """Independent-seed split. Pass trajectories from identical physical settings."""
    groups = [list(train_seeds), list(validation_seeds), list(test_seeds)]
    flat = [seed for group in groups for seed in group]
    if any(not group for group in groups) or len(flat) != len(set(flat)):
        raise ValueError("Train/validation/test seed lists must be nonempty and disjoint.")
    values = [[_fields(trajectories[seed]) for seed in group] for group in groups]
    shapes = {array.shape[1:] for group in values for array in group}
    if len(shapes) != 1:
        raise ValueError("All splits must use the same spatial lattice.")
    return TrainingSplits(*values, metadata={
        "method": "independent_simulation_seeds",
        "train_seeds": groups[0], "validation_seeds": groups[1], "test_seeds": groups[2],
    })


@dataclass(frozen=True)
class AffineNormalizer:
    mean: float
    std: float

    @classmethod
    def fit(cls, training_fields, *, show_progress=False, progress_desc="Train normalization"):
        """Streaming moments: only a small frame chunk is converted to float64."""
        count, mean, m2 = 0, 0.0, 0.0
        with tqdm(training_fields, desc=progress_desc, unit="trajectory",
                  disable=not show_progress, mininterval=.5, leave=False) as bar:
            for array in bar:
                x = _fields(array, check_finite=False)
                for start in range(0, len(x), 64):
                    chunk = np.asarray(x[start:start + 64], dtype=np.float64)
                    if not np.isfinite(chunk).all():
                        raise ValueError("Training fields must be finite.")
                    chunk_count = chunk.size
                    chunk_mean = float(chunk.mean())
                    chunk_m2 = float(np.square(chunk - chunk_mean).sum())
                    delta = chunk_mean - mean
                    total = count + chunk_count
                    m2 += chunk_m2 + delta * delta * count * chunk_count / total
                    mean += delta * chunk_count / total
                    count = total
        if count == 0:
            raise ValueError("Training split must not be empty.")
        std = float(np.sqrt(m2 / count))
        if not np.isfinite(std) or std <= np.finfo(float).eps:
            raise ValueError("Training fields have no finite nonzero variance.")
        return cls(mean, std)

    def transform(self, x):
        return (x - self.mean) / self.std

    def force_to_phi(self, force_z):
        return force_z / self.std


class _Pairs(Dataset):
    """Keep mmap-backed trajectories on disk; copy only a requested pair."""
    def __init__(self, trajectories, lag_frames):
        if not isinstance(lag_frames, int) or lag_frames < 1:
            raise ValueError("lag_frames must be a positive integer.")
        self.trajectories = [_fields(x, check_finite=False) for x in trajectories]
        self.lag = lag_frames
        lengths = [len(x) - lag_frames for x in self.trajectories]
        if not lengths or any(n <= 0 for n in lengths):
            raise ValueError("Each split trajectory must have more frames than lag_frames.")
        self.ends = np.cumsum(lengths)

    def __len__(self):
        return int(self.ends[-1])

    def __getitem__(self, index):
        if index < 0 or index >= len(self):
            raise IndexError(index)
        trajectory = int(np.searchsorted(self.ends, index, side="right"))
        offset = index - (int(self.ends[trajectory - 1]) if trajectory else 0)
        x = self.trajectories[trajectory]
        pair = np.stack((x[offset, 0], x[offset + self.lag, 0])).astype(np.float64, copy=False)
        if not np.isfinite(pair).all():
            raise ValueError("Observation pairs must be finite.")
        return torch.from_numpy(pair)


def source_provenance():
    actual = hashlib.sha256(SOURCE_PATH.read_bytes()).hexdigest()
    return {
        "source": "models/NEEP_ShellForce_2D.py",
        "source_sha256": actual,
        "spec_source_sha256": SPEC_SOURCE_SHA256,
        "matches_spec_source_hash": actual == SPEC_SOURCE_SHA256,
        "class": "MultiScaleShellForceCNEEP2D",
        "input_adapter": "[B,2,1,Ny,Nx] -> [B,2,Ny,Nx] for the existing scalar constructor",
        "architecture_source_modified": False,
    }


def joint_neep_loss(branch_scores, n_sites, alpha=-0.5):
    """Same alpha-NEEP objective as the existing notebook, on EXTENSIVE EP.

    Branch scores are spatial means. N is applied exactly once here. We never
    sum independently optimized per-branch exponential objectives.
    """
    ep = n_sites * branch_scores.sum(dim=1)
    if alpha == 0:
        return (-ep + torch.expm1(-ep)).mean()
    if not -1 < alpha < 0:
        raise ValueError("Use -1 < alpha <= 0.")
    return (-torch.expm1(alpha * ep) / alpha
            + torch.expm1(-(1 + alpha) * ep) / (1 + alpha)).mean()


@dataclass
class FitResult:
    model: torch.nn.Module
    normalizer: AffineNormalizer
    history: list[dict]
    manifest: dict
    checkpoint: Path


def train_shell_force(splits: TrainingSplits, output_dir, architecture=None, config=None,
                      physical_metadata=None, *, show_progress=True, progress_desc=None):
    """Joint training with iteration/validation ETA and bounded activation memory.

    micro_batch_size accumulates one logical batch's mean gradient before a
    single clipping/optimizer step. It does not change the requested batch size.
    """
    started = perf_counter()
    architecture = architecture or ArchitectureConfig()
    config = config or TrainingConfig()
    label = progress_desc or "ShellForce train"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / "best_model_parameter.pth.tar"
    current_path = output_dir / "model_parameter.pth.tar"
    if best_path.exists() or current_path.exists():
        raise FileExistsError("Use a new output directory to preserve existing checkpoints.")
    for group in (splits.train, splits.validation, splits.test):
        if not group:
            raise ValueError("Train, validation and test splits must all be nonempty.")
    normalizer = AffineNormalizer.fit(splits.train, show_progress=show_progress,
                                       progress_desc=f"{label} normalization")
    train_data = _Pairs(splits.train, config.lag_frames)
    val_data = _Pairs(splits.validation, config.lag_frames)
    n = train_data.trajectories[0].shape[-1]
    if n <= 2 * architecture.max_distance:
        raise ValueError("Use N_side > 2*R so the NN shells do not alias periodic offsets.")
    for group in (splits.train, splits.validation, splits.test):
        if any(_fields(x, check_finite=False).shape[-2:] != (n, n) for x in group):
            raise ValueError("All train/validation/test fields must have the same lattice.")
    torch.manual_seed(config.seed)
    rng = torch.Generator().manual_seed(config.seed)
    dtype = getattr(torch, config.dtype)
    options = architecture.options()
    model = MultiScaleShellForceCNEEP2D(options).to(device=config.device, dtype=dtype)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    train_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True, generator=rng)
    eval_batch_size = config.eval_batch_size or config.batch_size
    val_loader = DataLoader(val_data, batch_size=eval_batch_size, shuffle=False)
    micro_batch_size = config.micro_batch_size or config.batch_size
    manifest = {
        "architecture": vars(options), "training": asdict(config),
        "normalization": {**asdict(normalizer), "fit_split": "train_only", "rule": "z=(phi-m)/s; d_phi=d_z/s"},
        "split": splits.metadata, "provenance": source_provenance(),
        "physical_metadata": physical_metadata or {},
        "dataset": {name: {"trajectories": len(arrays),
                            "frames": sum(len(x) for x in arrays),
                            "pairs": sum(len(x) - config.lag_frames for x in arrays)}
                    for name, arrays in (("train", splits.train), ("validation", splits.validation), ("test", splits.test))},
        "n_sites": n * n, "optimizer": "AdamW",
        "loss": "mean[-expm1(alpha*S)/alpha + expm1(-(1+alpha)*S)/(1+alpha)], alpha=0 limit supported",
        "loss_score": "S=N*sum_r spatial_mean(d_z,r*delta_z); no extra scalar or beta factor",
        "gradient_accumulation": "Mean gradient over each logical batch; clip and optimizer step once per batch",
        "input": "phi only; no gradients, phi^2, kernels, spatial coordinates or external increments",
        "status": "training_run_not_physical_validation",
    }
    (output_dir / "learning_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf8")
    history, best_val = [], float("inf")
    batches = iter(train_loader)
    with tqdm(total=config.iterations, desc=label, unit="iteration", mininterval=.5,
              disable=not show_progress) as train_bar:
        for iteration in range(1, config.iterations + 1):
            try:
                pair = next(batches)
            except StopIteration:
                batches = iter(train_loader)
                pair = next(batches)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss_value = 0.0
            for chunk in pair.split(micro_batch_size):
                x = normalizer.transform(chunk).to(device=config.device, dtype=dtype)
                loss = joint_neep_loss(model(x), n * n, config.alpha)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite extensive NEEP loss; inspect learning rate, lag and initialization.")
                weight = len(chunk) / len(pair)
                (loss * weight).backward()
                loss_value += float(loss.detach()) * weight
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_norm, error_if_nonfinite=True)
            optimizer.step()
            if iteration == 1 or iteration % config.record_every == 0 or iteration == config.iterations:
                model.eval()
                val_sum, count = 0.0, 0
                validation_started = perf_counter()
                with torch.no_grad(), tqdm(total=len(val_data), desc=f"{label} validation {iteration}",
                        unit="pair", mininterval=.5, leave=False, disable=not show_progress) as val_bar:
                    for pair in val_loader:
                        for chunk in pair.split(micro_batch_size):
                            x = normalizer.transform(chunk).to(device=config.device, dtype=dtype)
                            val_loss = joint_neep_loss(model(x), n * n, config.alpha)
                            if not torch.isfinite(val_loss):
                                raise FloatingPointError("Nonfinite validation objective.")
                            val_sum += float(val_loss) * len(x)
                            count += len(x)
                            val_bar.update(len(x))
                val_mean = val_sum / count
                history.append({"iteration": iteration, "train_loss": loss_value,
                                "validation_loss": val_mean, "elapsed_seconds": perf_counter() - started,
                                "validation_seconds": perf_counter() - validation_started})
                manifest["elapsed_seconds"] = perf_counter() - started
                state = {
                    "settings": vars(options), "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(), "iteration": iteration,
                    "mean": torch.tensor(normalizer.mean, dtype=torch.float64),
                    "std": torch.tensor(normalizer.std, dtype=torch.float64),
                    "manifest": manifest, "history": history,
                }
                torch.save(state, current_path)
                if val_mean < best_val:
                    best_val = val_mean
                    torch.save(state, best_path)
                train_bar.set_postfix(loss=f"{loss_value:.5g}", val=f"{val_mean:.5g}", refresh=False)
            train_bar.update(1)
    manifest["elapsed_seconds"] = perf_counter() - started
    state = torch.load(best_path, map_location=config.device, weights_only=True)
    model.load_state_dict(state["state_dict"])
    model.eval()
    state["manifest"] = manifest
    state["history"] = history
    torch.save(state, best_path)
    (output_dir / "learning_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf8")
    (output_dir / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf8")
    return FitResult(model, normalizer, history, manifest, best_path)


def load_checkpoint(path, device="cpu"):
    """Load a checkpoint generated by this bridge using tensor-only safe loading."""
    state = torch.load(path, map_location=device, weights_only=True)
    dtype = getattr(torch, state["manifest"]["training"]["dtype"])
    model = MultiScaleShellForceCNEEP2D(SimpleNamespace(**state["settings"])).to(device=device, dtype=dtype)
    model.load_state_dict(state["state_dict"])
    model.eval()
    normalizer = AffineNormalizer(float(state["mean"]), float(state["std"]))
    return FitResult(model, normalizer, state["history"], state["manifest"], Path(path))


def _summary(segments, block_pairs):
    values = np.concatenate(segments)
    blocks = []
    for segment in segments:
        n_blocks = len(segment) // block_pairs
        if n_blocks:
            blocks.extend(segment[:n_blocks * block_pairs].reshape(n_blocks, block_pairs).mean(axis=1))
    se = float(np.std(blocks, ddof=1) / np.sqrt(len(blocks))) if len(blocks) > 1 else None
    return {"mean": float(np.mean(values)), "block_SE": se, "n_pairs": len(values),
            "n_blocks": len(blocks), "block_pairs": block_pairs,
            "uncertainty_status": "block_length_requires_autocorrelation_check"}


def evaluate_shell_force(fit: FitResult, trajectories, physical_model, frame_dt,
                         lag_frames=1, batch_size=32, block_pairs=32, output_dir=None, *,
                         show_progress=True, progress_desc=None, micro_batch_size=32):
    """Report neural EP, endpoint work, and known-channel projection separately.

    The endpoint work uses observation pairs, not integration-step streaming
    truth. Projection uses the left endpoint as an equal-time state and is a
    rate already: it must NOT be divided by lag again. No pair crosses a seed.
    """
    started = perf_counter()
    if micro_batch_size is not None and (not isinstance(micro_batch_size, int) or micro_batch_size < 1):
        raise ValueError("micro_batch_size must be a positive integer or None.")
    if min(lag_frames, batch_size, block_pairs) < 1 or frame_dt <= 0:
        raise ValueError("Lag/batch/block and frame interval must be positive.")
    if isinstance(trajectories, np.ndarray):
        trajectories = [trajectories]
    model, normalizer = fit.model, fit.normalizer
    parameter = next(model.parameters())
    device, dtype = parameter.device, parameter.dtype
    radius = model.max_distance
    trajectories = list(trajectories)
    if not trajectories:
        raise ValueError("Evaluation needs at least one trajectory.")
    total_pairs = 0
    for trajectory in trajectories:
        x = _fields(trajectory, check_finite=False)
        if x.shape[-2:] != physical_model.shape or len(x) <= lag_frames:
            raise ValueError("Physical model shape or observation lag does not match fields.")
        total_pairs += len(range(0, len(x) - lag_frames, lag_frames))
    step_size = min(batch_size, micro_batch_size or batch_size)
    all_values = {}
    with tqdm(total=total_pairs, desc=progress_desc or "ShellForce evaluation", unit="pair",
              mininterval=.5, disable=not show_progress) as evaluation_bar:
        for trajectory in trajectories:
            x = _fields(trajectory)[:, 0]
            if x.shape[-2:] != physical_model.shape or len(x) <= lag_frames:
                raise ValueError("Physical model shape or observation lag does not match fields.")
            local = {}
            # Nonoverlapping observation intervals reduce gratuitous pair overlap.
            starts = np.arange(0, len(x) - lag_frames, lag_frames)
            for begin in range(0, len(starts), step_size):
                ii = starts[begin:begin + step_size]
                pair_np = np.stack((x[ii], x[ii + lag_frames]), axis=1)
                pair = torch.as_tensor(pair_np, dtype=torch.float64)
                with torch.no_grad():
                    z = normalizer.transform(pair).to(device=device, dtype=dtype)
                    raw_rates = model(z).cpu().numpy() * physical_model.N / (frame_dt * lag_frames)
                    stationary_pair = torch.stack((pair[:, 0], pair[:, 0]), dim=1)
                    z_equal = normalizer.transform(stationary_pair).to(device=device, dtype=dtype)
                    forces = model(z_equal, return_forces=True)
                    force_phi = normalizer.force_to_phi(forces.sum(dim=1)[:, 0]).cpu().numpy()
                for r in range(raw_rates.shape[1]):
                    local.setdefault(("NN_raw_branch", "learned", f"r{r}"), []).extend(raw_rates[:, r])
                local.setdefault(("NN_total", "total", "all"), []).extend(raw_rates.sum(axis=1))
                for j, i in enumerate(ii):
                    midpoint = (x[i] + x[i + lag_frames]) / 2
                    delta = x[i + lag_frames] - x[i]
                    mid_channels = physical_model.active_channels(midpoint, max_distance=radius)
                    eq_channels = physical_model.active_channels(x[i], max_distance=radius)
                    lap_force = laplacian(force_phi[j])
                    totals = {"endpoint_midpoint": 0.0, "NN_force_projection": 0.0}
                    for label, nu in mid_channels.items():
                        mechanism, shell = label.split(":", 1)
                        work = -float(np.sum(nu * delta)) / (physical_model.config.D * frame_dt * lag_frames)
                        projection = -float(np.sum(eq_channels[label] * lap_force))
                        local.setdefault(("endpoint_midpoint", mechanism, shell), []).append(work)
                        local.setdefault(("NN_force_projection", mechanism, shell), []).append(projection)
                        totals["endpoint_midpoint"] += work
                        totals["NN_force_projection"] += projection
                    for method, value in totals.items():
                        local.setdefault((method, "total", "all"), []).append(value)
                evaluation_bar.update(len(ii))
            for key, values in local.items():
                all_values.setdefault(key, []).append(np.asarray(values, dtype=float))
    rows = [{"method": key[0], "mechanism": key[1], "shell": key[2],
             **_summary(segments, block_pairs)} for key, segments in all_values.items()]
    result = {"rows": rows, "values_by_trajectory": all_values,
              "observation_interval": frame_dt * lag_frames,
              "elapsed_seconds": perf_counter() - started,
              "note": "Endpoint work is finite-lag; compare separately with streaming integration-step truth. Signed branches are retained."}
    if output_dir is not None:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        with (destination / "nn_evaluation.csv").open("w", newline="", encoding="utf8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        payload = {f"{method}__{mechanism}__{shell}__traj{j}": values
                   for (method, mechanism, shell), segments in all_values.items()
                   for j, values in enumerate(segments)}
        np.savez_compressed(destination / "nn_evaluation_values.npz", **payload)
    return result


def projected_curl_check(fit: FitResult, physical_model, phi, probes=8, seed=73031, *,
                         show_progress=True, progress_desc=None):
    """Random zero-mean bilinear check of J_d-J_d.T + (J_mu-J_mu.T)/D.

    A small sampled residual is a diagnostic, not a proof of global exactness.
    The affine chain rule is applied in both force values and Jacobians.
    """
    if probes < 1:
        raise ValueError("probes must be positive.")
    phi = np.asarray(phi, dtype=np.float64)
    if phi.shape != physical_model.shape:
        raise ValueError("phi must be a single [Ny,Nx] field.")
    parameter = next(fit.model.parameters())
    x = torch.as_tensor(phi, device=parameter.device, dtype=parameter.dtype)
    def force(field):
        z = fit.normalizer.transform(torch.stack((field, field))[None])
        dz = fit.model(z, return_forces=True).sum(dim=1)[0, 0]
        return fit.normalizer.force_to_phi(dz)
    rng = np.random.default_rng(seed)
    residuals, targets, learned = [], [], []
    started = perf_counter()
    with tqdm(range(probes), desc=progress_desc or "Projected curl", unit="probe",
              mininterval=.5, leave=False, disable=not show_progress) as bar:
        for _ in bar:
            u, v = rng.normal(size=phi.shape), rng.normal(size=phi.shape)
            u -= u.mean()
            v -= v.mean()
            u /= np.linalg.norm(u)
            v /= np.linalg.norm(v)
            ut = torch.as_tensor(u, device=x.device, dtype=x.dtype)
            vt = torch.as_tensor(v, device=x.device, dtype=x.dtype)
            _, jdu = torch.autograd.functional.jvp(force, x, ut)
            _, jdv = torch.autograd.functional.jvp(force, x, vt)
            curl_d = float((ut * jdv - vt * jdu).sum().detach())
            curl_mu = float(np.sum(u * physical_model.mu_jvp(phi, v) - v * physical_model.mu_jvp(phi, u)))
            target = -curl_mu / physical_model.config.D
            targets.append(target)
            learned.append(curl_d)
            residuals.append(curl_d - target)
    rms = float(np.sqrt(np.mean(np.square(residuals))))
    target_rms = float(np.sqrt(np.mean(np.square(targets))))
    return {"residual_rms": rms, "target_rms": target_rms,
            "relative_residual": rms / target_rms if target_rms > 1e-14 else None,
            "learned_bilinear": learned, "target_bilinear": targets,
            "n_probes": probes, "seed": seed, "elapsed_seconds": perf_counter() - started,
            "note": "Projected total curl only; raw branch attribution additionally needs branch residual and mean-work tests."}
