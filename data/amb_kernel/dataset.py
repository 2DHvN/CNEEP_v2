"""Independent, disk-backed AMB replica datasets for the existing shell-force trainer.

The production preset follows the LABPM sample counts: 1024 training replicas,
256 validation replicas, and 10 test replicas, each with 1000 observation
intervals (1001 frames). CUDA generates a batch of independent trajectories at
once; CPU retains the reference solver. Saved replicas reopen as read-only
memory maps; no multi-gigabyte ensemble is stacked in RAM.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from tqdm.auto import tqdm

from .diagnostics import json_safe, write_json
from .experiment import RunConfig, run_experiment
from .model import AMBKernelModel, ModelConfig

SPLITS = ("train", "validation", "test")
RUN_ARTIFACTS = ("fields.npy", "times.npy", "kernel.npz", "manifest.json",
                 "diagnostics.json", "ep_blocks.csv", "ep_block_values.npz", "observables.npz")


def resolve_simulation_device(device: str = "auto") -> str:
    """Resolve CPU/CUDA explicitly; a requested unavailable GPU never falls back.

    ``auto`` uses the current CUDA device if available, otherwise the NumPy CPU
    reference. CPU simulation does not require importing PyTorch.
    """
    if not isinstance(device, str):
        raise TypeError("device must be 'auto', 'cpu', 'cuda', or 'cuda:N'.")
    name = device.lower()
    if name == "cpu":
        return "cpu"
    if name != "auto" and name != "cuda" and not (
            name.startswith("cuda:") and name[5:].isdigit()):
        raise ValueError("Simulation device must be 'auto', 'cpu', 'cuda', or 'cuda:N'.")
    import torch
    if not torch.cuda.is_available():
        if name == "auto":
            return "cpu"
        raise RuntimeError("CUDA simulation was requested but this Python environment has no available CUDA device. "
                           "Select a notebook kernel with CUDA-enabled PyTorch and a working NVIDIA GPU, "
                           "or choose device='cpu' explicitly.")
    index = int(name[5:]) if name.startswith("cuda:") else torch.cuda.current_device()
    if index >= torch.cuda.device_count():
        raise ValueError(f"CUDA device index {index} is unavailable (device count: {torch.cuda.device_count()}).")
    return f"cuda:{index}"


@dataclass(frozen=True)
class DatasetConfig:
    """Replica counts and frame contract; duration adequacy still needs validation."""
    train_count: int = 1024
    validation_count: int = 256
    test_count: int = 10
    intervals: int = 1000
    burn_in_steps: int = 10000
    observation_stride: int = 1
    split_seeds: tuple[int, int, int] = (12031, 23041, 34057)

    def __post_init__(self):
        for name in ("train_count", "validation_count", "test_count", "intervals", "burn_in_steps", "observation_stride"):
            value = getattr(self, name)
            minimum = 0 if name == "burn_in_steps" else 1
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}.")
        if len(self.split_seeds) != 3 or len(set(self.split_seeds)) != 3:
            raise ValueError("Provide three distinct split_seeds for train/validation/test.")
        if any(isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0 for seed in self.split_seeds):
            raise ValueError("split_seeds must be nonnegative integers.")

    @classmethod
    def smoke(cls, **overrides):
        """Tiny independent replicas for execution checks, never a production replacement."""
        values = dict(train_count=2, validation_count=2, test_count=2,
                      intervals=32, burn_in_steps=16)
        values.update(overrides)
        return cls(**values)

    @property
    def counts(self) -> dict[str, int]:
        return {"train": self.train_count, "validation": self.validation_count, "test": self.test_count}

    @property
    def frames_per_trajectory(self) -> int:
        return self.intervals + 1

    @property
    def total_replicas(self) -> int:
        return sum(self.counts.values())

    def estimated_storage_bytes(self, n: int, storage_dtype: str = "float64") -> int:
        """Field arrays alone; per-run physical diagnostics require additional disk space."""
        return self.total_replicas*self.frames_per_trajectory*n*n*np.dtype(storage_dtype).itemsize

    def replica_seeds(self) -> dict[str, list[int]]:
        """Reproducible SeedSequence child seeds with an explicit collision check."""
        seeds = {}
        used = set()
        for split_index, (split, root_seed) in enumerate(zip(SPLITS, self.split_seeds)):
            children = np.random.SeedSequence([int(root_seed), split_index]).spawn(self.counts[split])
            values = [int(child.generate_state(1, dtype=np.uint64)[0]) for child in children]
            if len(set(values)) != len(values) or used.intersection(values):
                raise ValueError("Replica seed collision; choose different split_seeds.")
            seeds[split] = values
            used.update(values)
        return seeds


@dataclass
class DatasetResult:
    train: list[np.memmap]
    validation: list[np.memmap]
    test: list[np.memmap]
    manifest: dict
    model: AMBKernelModel
    output_dir: Path

    def training_splits(self):
        """Existing TrainingSplits container without scanning or copying field contents."""
        from .learning import TrainingSplits
        return TrainingSplits(self.train, self.validation, self.test, metadata={
            "method": "independent_simulation_replicas",
            "dataset_manifest": str(self.output_dir / "dataset_manifest.json"),
            "replica_counts": self.manifest["replica_counts"],
            "replica_seeds": self.manifest["replica_seeds"],
            "frames_per_trajectory": self.manifest["frames_per_trajectory"],
            "lag1_pair_counts": self.manifest["lag1_pair_counts"],
            "memory_map_mode": "r",
            "memory_map_paths": {split: [str(self.output_dir / record["path"] / "fields.npy")
                                         for record in self.manifest["replicas"] if record["split"] == split]
                                 for split in SPLITS},
        })

    def close(self):
        """Release mapped files after training/evaluation of one model finishes."""
        for split in SPLITS:
            for array in getattr(self, split):
                if isinstance(array, np.memmap) and array._mmap is not None:
                    array._mmap.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def _kernel_hash(model: AMBKernelModel) -> str:
    return hashlib.sha256(np.ascontiguousarray(model.K).tobytes()).hexdigest()


def _simulator_hashes(*, torch_backend: bool = False) -> dict[str, str]:
    folder = Path(__file__).parent
    names = ("model.py", "experiment.py", "diagnostics.py", "dataset.py")
    if torch_backend:
        names += ("torch_model.py", "torch_experiment.py")
    return {name: hashlib.sha256((folder / name).read_bytes()).hexdigest()
            for name in names}


def _run_template(config: DatasetConfig, template: RunConfig | None) -> RunConfig:
    if template is None:
        template = RunConfig(dt=0.01, observation_lags=tuple(config.observation_stride*k for k in (1, 4, 16)),
                             block_steps=min(1024, config.intervals*config.observation_stride))
    if not isinstance(template, RunConfig):
        raise TypeError("run_config must be a RunConfig or None.")
    # Replica identity owns the seed and frame counts; physical/solver settings
    # such as dt, D (in the model), storage precision and tolerances are retained.
    return replace(template, seed=0, burn_in_steps=config.burn_in_steps,
                   measurement_steps=config.intervals*config.observation_stride,
                   observation_stride=config.observation_stride)


def _identity(model: AMBKernelModel, config: DatasetConfig, template: RunConfig,
              *, torch_backend: bool = False) -> dict:
    return json_safe({"schema_version": 2, "model_parameters": asdict(model.config),
                      "kernel_sha256": _kernel_hash(model), "dataset_config": asdict(config),
                      "run_template": asdict(template),
                      "simulation_backend": "torch" if torch_backend else "numpy",
                      "noise_rng": "numpy_pcg64_per_replica",
                      "simulator_sha256": _simulator_hashes(torch_backend=torch_backend)})


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_replica(path: Path, model: AMBKernelModel, run_config: RunConfig,
                      config: DatasetConfig, *, simulation_backend: str | None = None) -> np.memmap:
    """Validate cached physics, exact frame times and headers without copying fields."""
    missing = [name for name in RUN_ARTIFACTS if not (path / name).is_file()]
    if missing:
        raise ValueError(f"Incomplete cached replica {path}: missing {', '.join(missing)}. Use a new dataset directory.")
    manifest = _read_json(path / "manifest.json")
    if manifest.get("run_status") != "complete":
        raise ValueError(f"Cached replica {path} is incomplete; restart it in a new dataset directory.")
    if simulation_backend is not None and manifest.get("simulation_backend", "numpy") != simulation_backend:
        raise ValueError(f"Cached replica simulation backend differs: {path}")
    if manifest.get("model_parameters") != json_safe(asdict(model.config)):
        raise ValueError(f"Cached replica model parameters differ: {path}")
    if manifest.get("run_config") != json_safe(asdict(run_config)):
        raise ValueError(f"Cached replica run configuration differs: {path}")
    if manifest.get("kernel_sha256") != _kernel_hash(model):
        raise ValueError(f"Cached replica kernel identity differs: {path}")
    with np.load(path / "kernel.npz", allow_pickle=False) as kernel:
        if not np.array_equal(kernel["K"], model.K):
            raise ValueError(f"Cached replica kernel contents differ: {path}")
    diagnostics = _read_json(path / "diagnostics.json")
    if (diagnostics.get("completed_burn_in_steps") != run_config.burn_in_steps or
            diagnostics.get("completed_measurement_steps") != run_config.measurement_steps):
        raise ValueError(f"Cached replica completed step counts differ: {path}")
    expected_times = np.asarray([(config.burn_in_steps+i*config.observation_stride)*run_config.dt
                                 for i in range(config.frames_per_trajectory)])
    times = np.load(path / "times.npy", allow_pickle=False)
    if not np.array_equal(times, expected_times):
        raise ValueError(f"Cached replica observation times differ: {path}")
    fields = np.load(path / "fields.npy", mmap_mode="r", allow_pickle=False)
    expected_shape = (config.frames_per_trajectory, 1, *model.shape)
    if fields.shape != expected_shape or fields.dtype != np.dtype(run_config.storage_dtype):
        fields._mmap.close()
        raise ValueError(f"Cached replica fields must have shape {expected_shape} and dtype {run_config.storage_dtype}: {path}")
    return fields


def generate_dataset(model_or_config, output_dir, dataset_config: DatasetConfig | None = None,
                     run_config: RunConfig | None = None, *, show_progress: bool = True,
                     resume: bool = True, nn_metadata: dict | None = None,
                     device: str = "cpu", simulation_batch_size: int = 32) -> DatasetResult:
    """Generate/reuse independent train/validation/test replicas on CPU or CUDA.

    DatasetConfig controls replica count, burn-in, observation interval count
    and stride. RunConfig supplies dt, solver settings, observation lags and
    storage precision; its seed/count fields are replaced explicitly. Resume
    accepts only complete artifacts with identical physics, kernel, run settings
    and simulator source/backend. Incomplete runs are preserved and rejected.
    ``device='cuda'`` batches up to ``simulation_batch_size`` independent
    trajectories on the GPU using float64 implicit midpoint. ``auto`` selects
    CUDA if available; ``cpu`` uses the original sequential NumPy/SciPy solver.
    Batch size does not change replica seeds and is not a cache identity field.
    """
    if (isinstance(simulation_batch_size, bool) or
            not isinstance(simulation_batch_size, (int, np.integer)) or simulation_batch_size < 1):
        raise ValueError("simulation_batch_size must be a positive integer.")
    selected_device = resolve_simulation_device(device)
    torch_backend = selected_device.startswith("cuda")
    backend = "torch" if torch_backend else "numpy"
    effective_batch_size = int(simulation_batch_size) if torch_backend else 1
    model = model_or_config if isinstance(model_or_config, AMBKernelModel) else AMBKernelModel(model_or_config)
    config = dataset_config or DatasetConfig()
    if not isinstance(config, DatasetConfig):
        raise TypeError("dataset_config must be a DatasetConfig.")
    template = _run_template(config, run_config)
    output = Path(output_dir)
    manifest_path = output / "dataset_manifest.json"
    identity = _identity(model, config, template, torch_backend=torch_backend)
    previous = None
    if manifest_path.exists():
        if not resume:
            raise FileExistsError(f"Dataset already exists: {output}")
        previous = _read_json(manifest_path)
        if previous.get("identity") != identity:
            raise ValueError("Cached dataset identity differs (model/kernel/counts/run settings/backend/source). Use a new output directory.")
    elif output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Nonempty dataset directory has no dataset_manifest.json: {output}")
    seeds = config.replica_seeds()
    plans = []
    for split in SPLITS:
        for index, seed in enumerate(seeds[split]):
            relative = Path(split) / f"replica_{index:05d}"
            plans.append({"split": split, "index": index, "seed": seed, "path": relative.as_posix()})
    cached = set()
    # Validate every cached replica before starting new work. Opening and closing
    # each mmap here checks headers without retaining a complete dataset in RAM.
    for record in plans:
        path = output / record["path"]
        if path.exists():
            if not resume:
                raise FileExistsError(f"Replica already exists: {path}")
            fields = _validate_replica(path, model, replace(template, seed=record["seed"]), config,
                                       simulation_backend=backend)
            fields._mmap.close()
            cached.add(record["path"])
    output.mkdir(parents=True, exist_ok=True)
    if previous is None:
        np.savez_compressed(output / "dataset_kernel.npz", K=model.K)
    else:
        with np.load(output / "dataset_kernel.npz", allow_pickle=False) as kernel:
            if not np.array_equal(kernel["K"], model.K):
                raise ValueError("Cached dataset kernel contents differ.")
    steps_per_replica = template.burn_in_steps+template.measurement_steps
    prior_elapsed = float(previous.get("elapsed_seconds", 0.0)) if previous else 0.0
    start = perf_counter()
    manifest = {"identity": identity, "status": "generating", "model_id": model.config.model_id,
                "model_parameters": asdict(model.config), "kernel_sha256": _kernel_hash(model),
                "dataset_config": asdict(config), "run_template": asdict(template),
                "replica_counts": config.counts, "replica_seeds": seeds,
                "frames_per_trajectory": config.frames_per_trajectory,
                "intervals_per_trajectory": config.intervals,
                "lag1_pair_counts": {split: count*config.intervals for split, count in config.counts.items()},
                "pair_counts_by_lag_steps": {str(lag): {split: count*max(0, config.frames_per_trajectory-lag//config.observation_stride)
                    for split, count in config.counts.items()} for lag in template.observation_lags},
                "estimated_field_storage_bytes": config.estimated_storage_bytes(model.config.n, template.storage_dtype),
                "total_integration_steps": config.total_replicas*steps_per_replica,
                "steps_per_replica": steps_per_replica, "replicas": plans,
                "completed_replicas": len(cached), "cached_replicas_this_invocation": len(cached),
                "generated_replicas_this_invocation": 0, "elapsed_seconds": prior_elapsed,
                "memory_map_mode": "r", "nn_metadata": nn_metadata or {},
                "simulation_backend": backend, "simulation_device": selected_device,
                "simulation_batch_size": effective_batch_size,
                "requested_simulation_batch_size": int(simulation_batch_size),
                "noise_rng": "numpy_pcg64_per_replica",
                "split_method": "independent SeedSequence replica seeds, disjoint across all splits",
                "duration_note": "Burn-in and observation lengths are starting settings, not a stationarity guarantee.",
                "storage_note": "Read-only per-replica memory maps; estimated size excludes diagnostic files."}
    def checkpoint(status, failure=None):
        elapsed = perf_counter()-start
        manifest.update(status=status, elapsed_seconds=prior_elapsed+elapsed,
                        last_invocation_elapsed_seconds=elapsed, failure=failure)
        pending = output / "dataset_manifest.pending.json"
        write_json(pending, manifest)
        pending.replace(manifest_path)
    checkpoint("generating")
    bar = tqdm(total=manifest["total_integration_steps"], initial=len(cached)*steps_per_replica,
               desc=f"{model.config.model_id} dataset", unit="replica-step", disable=not show_progress,
               dynamic_ncols=True, mininterval=.25,
               bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} "
                          "[{elapsed} elapsed, ETA {remaining}, {rate_fmt}{postfix}]")
    try:
        pending = [record for record in plans if record["path"] not in cached]
        for start_index in range(0, len(pending), effective_batch_size):
            batch = pending[start_index:start_index + effective_batch_size]
            if torch_backend:
                from .torch_experiment import run_experiments_batched
                bar.set_description(f"{model.config.model_id} {selected_device} batch={len(batch)}", refresh=False)
            else:
                record = batch[0]
                split, index = record["split"], record["index"]
                bar.set_description(f"{model.config.model_id} {split} {index+1}/{config.counts[split]}", refresh=False)
            previous_step = 0
            def progress(done, total):
                nonlocal previous_step
                bar.update(done-previous_step)
                previous_step = done
                phase = "burn-in" if done <= len(batch)*template.burn_in_steps else "measurement"
                bar.set_postfix_str(f"{phase}, saved={manifest['completed_replicas']}/{config.total_replicas}", refresh=False)
            if torch_backend:
                run_experiments_batched(model, [replace(template, seed=row["seed"]) for row in batch],
                                        [output / row["path"] for row in batch], device=selected_device,
                                        nn_metadata=nn_metadata, progress=progress, show_progress=False)
            else:
                result = run_experiment(model, replace(template, seed=record["seed"]),
                                        output_dir=output / record["path"], nn_metadata=nn_metadata,
                                        progress=progress, show_progress=False)
                del result
            # CUDA batches stream per-replica fields to disk; validation retains
            # only one read-only map at a time before the next batch starts.
            for record in batch:
                fields = _validate_replica(output / record["path"], model, replace(template, seed=record["seed"]),
                                           config, simulation_backend=backend)
                fields._mmap.close()
                manifest["completed_replicas"] += 1
                manifest["generated_replicas_this_invocation"] += 1
                checkpoint("generating")
            bar.set_postfix_str(f"saved={manifest['completed_replicas']}/{config.total_replicas}", refresh=False)
        checkpoint("complete")
    except (Exception, KeyboardInterrupt) as exc:
        checkpoint("incomplete", f"{type(exc).__name__}: {exc}")
        raise
    finally:
        bar.close()
    return load_dataset(output)


def load_dataset(output_dir) -> DatasetResult:
    """Reopen a complete dataset as independent read-only memory-mapped fields."""
    output = Path(output_dir)
    manifest = _read_json(output / "dataset_manifest.json")
    if manifest.get("status") != "complete":
        raise ValueError("Dataset is incomplete; generate_dataset(..., resume=True) may complete missing replicas.")
    config = DatasetConfig(**manifest["dataset_config"])
    with np.load(output / "dataset_kernel.npz", allow_pickle=False) as kernel:
        model = AMBKernelModel(ModelConfig(**manifest["model_parameters"]), kernel=kernel["K"])
    if _kernel_hash(model) != manifest["kernel_sha256"]:
        raise ValueError("Dataset kernel identity differs from its manifest.")
    template = RunConfig(**manifest["run_template"])
    expected_seeds = config.replica_seeds()
    if expected_seeds != manifest["replica_seeds"]:
        raise ValueError("Dataset seed identity differs from its manifest.")
    arrays = {split: [] for split in SPLITS}
    try:
        records = manifest["replicas"]
        expected_records = [(split, i, seed, (Path(split)/f"replica_{i:05d}").as_posix())
                            for split in SPLITS for i, seed in enumerate(expected_seeds[split])]
        actual_records = [(row["split"], row["index"], row["seed"], row["path"]) for row in records]
        if actual_records != expected_records:
            raise ValueError("Dataset replica index/count/path identity differs from its manifest.")
        for record in records:
            fields = _validate_replica(output / record["path"], model,
                                       replace(template, seed=record["seed"]), config,
                                       simulation_backend=manifest.get("simulation_backend"))
            arrays[record["split"]].append(fields)
    except Exception:
        for group in arrays.values():
            for fields in group:
                fields._mmap.close()
        raise
    return DatasetResult(arrays["train"], arrays["validation"], arrays["test"], manifest, model, output)
