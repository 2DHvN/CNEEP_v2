"""Float64, conserved implicit-midpoint experiments and streaming shell EPR.

Only saved observation frames feed the NN. Every integration step contributes
separately to physical active work and the equal-time drift-minus-trace check.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import csv
import hashlib
import json
import platform
from importlib import metadata
from pathlib import Path
from time import perf_counter
from typing import Callable

import numpy as np
from tqdm.auto import tqdm

from .diagnostics import integrated_autocorrelation, json_safe, stationarity_report, summarize_blocks, write_json
from .model import AMBKernelModel, ModelConfig, laplacian


@dataclass(frozen=True)
class RunConfig:
    """Short smoke defaults; choose duration from measured slow times for science.

    observation_lags are expressed in integration steps. All requested lags
    must be multiples of observation_stride, so no unavailable fine lag is
    silently substituted. Stored fields retain the physical scalar coordinate.
    """
    seed: int = 0
    dt: float = 0.001
    burn_in_steps: int = 100
    measurement_steps: int = 400
    observation_stride: int = 1
    observation_lags: tuple[int, ...] = (1, 4, 16)
    block_steps: int = 100
    max_distance: int = 4
    solver_atol: float = 1e-10
    solver_rtol: float = 1e-8
    solver_maxiter: int = 20
    initial_scale: float = 0.05
    storage_dtype: str = "float64"
    initial_mode: str = "random"
    circle_radius_fraction: float = 0.25
    circle_width: float = 1.0
    circle_amplitude: float = 1.0

    def __post_init__(self):
        for name in ("seed", "burn_in_steps", "measurement_steps", "observation_stride",
                     "block_steps", "max_distance", "solver_maxiter"):
            value = getattr(self, name)
            if not isinstance(value, (int, np.integer)) or isinstance(value, bool):
                raise ValueError(f"{name} must be an integer")
            if value < (0 if name in ("seed", "burn_in_steps", "max_distance") else 1):
                raise ValueError(f"Invalid {name}: {value}")
        if not np.isfinite(self.dt) or self.dt <= 0:
            raise ValueError("dt must be positive and finite")
        if self.measurement_steps % self.observation_stride:
            raise ValueError("measurement_steps must be divisible by observation_stride")
        if not self.observation_lags:
            raise ValueError("observation_lags must contain at least one lag")
        for lag in self.observation_lags:
            if not isinstance(lag, (int, np.integer)) or lag <= 0 or lag % self.observation_stride:
                raise ValueError("observation_lags must be positive integer multiples of observation_stride")
        if self.solver_atol <= 0:
            raise ValueError("solver_atol must be strictly positive")
        if self.storage_dtype not in ("float32", "float64"):
            raise ValueError("storage_dtype must be float32 or float64")
        if self.initial_mode not in ("random", "circle"):
            raise ValueError("initial_mode must be 'random' or 'circle'.")
        if not np.isfinite(self.circle_radius_fraction) or not 0 < self.circle_radius_fraction <= 0.5:
            raise ValueError("circle_radius_fraction must be finite and in (0, 0.5].")
        for name in ("circle_width", "circle_amplitude"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive and finite.")
        for name in ("solver_atol", "solver_rtol", "initial_scale"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and nonnegative")


@dataclass
class ExperimentResult:
    fields: np.ndarray
    times: np.ndarray
    ep_blocks: list[dict]
    diagnostics: dict
    manifest: dict
    block_values: dict[str, np.ndarray]
    block_step_counts: np.ndarray
    observables: dict[str, np.ndarray]
    model: AMBKernelModel
    output_dir: Path | None = None

    def save(self, output_dir, *, overwrite: bool = False) -> Path:
        """Save a self-describing run; existing artifacts require explicit overwrite."""
        path = Path(output_dir)
        artifacts = ("fields.npy", "times.npy", "kernel.npz", "manifest.json",
                     "ep_blocks.csv", "diagnostics.json", "ep_block_values.npz", "observables.npz")
        if not overwrite and any((path / name).exists() for name in artifacts):
            raise FileExistsError(f"Run artifacts already exist in {path}")
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / "fields.npy", self.fields, allow_pickle=False)
        np.save(path / "times.npy", self.times, allow_pickle=False)
        np.savez_compressed(path / "kernel.npz", K=self.model.K, Ke=self.model.Ke,
                            Ko=self.model.Ko, Khat=self.model.Khat, Lhat=self.model.Lhat,
                            shell_labels=self.model.shell_labels,
                            physical_parameters=json.dumps(json_safe(asdict(self.model.config))))
        write_json(path / "manifest.json", self.manifest)
        write_json(path / "diagnostics.json", self.diagnostics)
        columns = ("method", "mechanism", "shell", "mean", "SE", "ci95_lower", "ci95_upper",
                   "n_steps", "n_blocks", "n_full_blocks", "duration", "uncertainty")
        with (path / "ep_blocks.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows(json_safe(self.ep_blocks))
        np.savez_compressed(path / "ep_block_values.npz", block_step_counts=self.block_step_counts,
                            **self.block_values)
        np.savez_compressed(path / "observables.npz", **self.observables)
        self.output_dir = path
        return path


class SimulationFailure(RuntimeError):
    """A failed step is not clipped, resampled, or silently restarted."""
    def __init__(self, message: str, result: ExperimentResult):
        super().__init__(message)
        self.result = result


def _aggregates(channels: dict, max_distance: int, zero):
    result = dict(channels)
    mechanisms = sorted({key.split(":")[0] for key in channels})
    total = zero.copy() if isinstance(zero, np.ndarray) else zero
    total_tail = zero.copy() if isinstance(zero, np.ndarray) else zero
    shell_totals = {}
    for mechanism in mechanisms:
        all_value = zero.copy() if isinstance(zero, np.ndarray) else zero
        tail = zero.copy() if isinstance(zero, np.ndarray) else zero
        for key, value in channels.items():
            name, shell = key.split(":")
            if name != mechanism:
                continue
            all_value = all_value + value
            shell_totals[shell] = shell_totals.get(shell, zero) + value
            if shell == "tail" or (shell.startswith("r") and int(shell[1:]) > max_distance):
                tail = tail + value
        result[f"{mechanism}:all"] = all_value
        result[f"{mechanism}:tail"] = tail
        total = total + all_value
        total_tail = total_tail + tail
    result.update({f"total:{shell}": value for shell, value in shell_totals.items()})
    result["total:all"] = total
    result["total:tail"] = total_tail
    return result


def _observable(phi: np.ndarray, model: AMBKernelModel) -> dict:
    centered = phi - phi.mean()
    spectrum = np.abs(np.fft.fft2(centered)) ** 2 / phi.size
    out = {"phi2": float(np.mean(phi ** 2)),
           "gradient_energy_density": float(np.mean(phi * laplacian(phi)) / 2),
           "low_q_structure_factor": float((spectrum[0, 1] + spectrum[1, 0]) / 2),
           "structure_factor_x": float(spectrum[0, 1]),
           "structure_factor_y": float(spectrum[1, 0]),
           "phi_min": float(phi.min()), "phi_max": float(phi.max())}
    if model.config.model_id == "M05":
        u = np.fft.ifft2(model.Khat * np.fft.fft2(phi)).real
        out.update(receiver_u_mean=float(u.mean()), receiver_u2=float(np.mean(u ** 2)),
                   receiver_u4=float(np.mean(u ** 4)))
    return out


def _source_hashes() -> dict:
    folder = Path(__file__).parent
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(folder.glob("*.py"))}


def run_experiment(model_or_config, run_config: RunConfig | None = None,
                   output_dir=None, *, initial_field=None, nn_metadata: dict | None = None,
                   progress: Callable[[int, int], None] | None = None,
                   show_progress: bool = True, progress_desc: str | None = None,
                   progress_details: Callable[[dict], None] | None = None) -> ExperimentResult:
    """Run burn-in and measurement, saving exact lags and per-step active work.

    The equal-time estimator is evaluated at phi_now, independently of the
    midpoint work. Failure artifacts preserve only completed steps, with an
    explicit incomplete status, and SimulationFailure carries the partial run.
    A failed run must be restarted from the beginning with a smaller dt.

    The progress bar includes burn-in and measurement and displays elapsed time
    and ETA. ``progress(done, total)`` retains its original two-argument API.
    ``progress_details(info)`` additionally receives phase, elapsed_seconds and
    eta_seconds after each completed integration step. Set show_progress=False
    when a caller provides an aggregate bar for multiple trajectories.
    """
    started_at = perf_counter()
    model = model_or_config if isinstance(model_or_config, AMBKernelModel) else AMBKernelModel(model_or_config)
    cfg = run_config or RunConfig()
    if not isinstance(cfg, RunConfig):
        raise TypeError("run_config must be a RunConfig")
    if output_dir is not None:
        target = Path(output_dir)
        if any((target / name).exists() for name in ("manifest.json", "fields.npy", "ep_blocks.csv")):
            raise FileExistsError(f"Run already exists in {target}; select a new output directory")
    rng = np.random.default_rng(cfg.seed)
    phi = (model.initial_field(rng, scale=cfg.initial_scale, mode=cfg.initial_mode,
                               circle_radius_fraction=cfg.circle_radius_fraction,
                               circle_width=cfg.circle_width, circle_amplitude=cfg.circle_amplitude)
           if initial_field is None
           else np.array(initial_field, dtype=np.float64, copy=True))
    if phi.shape != model.shape or not np.isfinite(phi).all():
        raise ValueError(f"initial_field must be a finite array of shape {model.shape}")
    if abs(float(phi.mean()) - model.config.phi_bar) > 1e-10:
        raise ValueError("initial_field mean must match model.config.phi_bar; no recentering is performed")
    phi = np.asarray(phi, dtype=np.float64)
    initial_mass = float(phi.sum())
    frames, times = [], []
    observation_rows = []
    block_values, block_counts, block_accumulator = {}, [], {}
    tail_energy = {}
    receiver_covariance = np.zeros((model.config.n // 2 + 1,) * 2)
    receiver_shell_hats = np.fft.fft2(model.K_shells) if model.config.model_id == "M05" else None
    max_mass_error = max_mass_correction = max_residual = 0.0
    max_iterations = 0
    burn_completed = completed = in_block = 0
    passive_start = None
    phi_measure_start = None
    passive_integrals = {}
    gaussian = None
    if model.config.model_id == "M03" and model.config.b == 0:
        gaussian = model.gaussian_reference(max_distance=cfg.max_distance)

    def record_frame(t):
        nonlocal receiver_covariance
        frames.append(phi[None].copy())
        times.append(t)
        observation_rows.append(_observable(phi, model))
        if receiver_shell_hats is not None:
            shells = np.fft.ifft2(receiver_shell_hats * np.fft.fft2(phi)).real.reshape(len(receiver_shell_hats), -1)
            shells = shells - shells.mean(axis=1, keepdims=True)
            receiver_covariance += shells @ shells.T / model.N

    def close_block():
        nonlocal in_block
        if not in_block:
            return
        for key, amount in block_accumulator.items():
            block_values.setdefault(key, []).append(amount / (in_block * cfg.dt))
        block_counts.append(in_block)
        block_accumulator.clear()
        in_block = 0

    def accumulate(key, increment):
        block_accumulator[key] = block_accumulator.get(key, 0.0) + float(increment)

    def finalize(status, failure=None):
        close_block()
        arrays = {k: np.asarray(v, dtype=np.float64) for k, v in block_values.items()}
        counts = np.asarray(block_counts, dtype=np.int64)
        summary = summarize_blocks(arrays, counts, dt=cfg.dt, full_block_steps=cfg.block_steps)
        obs = {key: np.asarray([row[key] for row in observation_rows], dtype=np.float64)
               for key in observation_rows[0]} if observation_rows else {}
        obs["times"] = np.asarray(times, dtype=np.float64)
        stored = np.asarray(frames, dtype=np.float64) if frames else np.empty((0, 1, *model.shape))
        quantization = {"storage_dtype": cfg.storage_dtype, "increment_relative_l2_error": 0.0,
                        "lost_nonzero_increment_fraction": 0.0}
        if cfg.storage_dtype == "float32" and len(stored) > 1:
            exact = np.diff(stored, axis=0)
            rounded = np.diff(stored.astype(np.float32).astype(np.float64), axis=0)
            norm = float(np.linalg.norm(exact))
            quantization["increment_relative_l2_error"] = float(np.linalg.norm(rounded - exact) / norm) if norm else 0.0
            nonzero = exact != 0
            quantization["lost_nonzero_increment_fraction"] = float(np.mean(rounded[nonzero] == 0)) if nonzero.any() else 0.0
        stored = stored.astype(cfg.storage_dtype)
        stationary = stationarity_report(obs, observation_dt=cfg.observation_stride * cfg.dt,
                           burn_in_time=burn_completed * cfg.dt, block_time=cfg.block_steps * cfg.dt,
                           full_blocks=int(np.sum(counts == cfg.block_steps)), model_a=model.config.a)
        ep_rates = arrays.get("midpoint|total|all", np.array([]))[counts == cfg.block_steps]
        ep_stationarity = integrated_autocorrelation(ep_rates, cfg.block_steps * cfg.dt)
        if len(ep_rates) >= 4:
            half = len(ep_rates) // 2
            delta = float(ep_rates[half:].mean() - ep_rates[:half].mean())
            se = float(np.std(ep_rates, ddof=1) * np.sqrt(1 / half + 1 / (len(ep_rates) - half)))
            ep_stationarity.update(half_mean_difference=delta, drift_z=delta / se if se else 0.0)
        exact_zero_epr = bool(ep_rates.size > 0 and np.all(ep_rates == 0))
        ep_stationarity["exact_zero_observed_series"] = exact_zero_epr
        ep_stationarity["autocorrelation_required"] = not exact_zero_epr
        if exact_zero_epr:
            ep_stationarity["reason"] = "Every complete EPR block is exactly zero; autocorrelation is undefined and not required for this null series."
            ep_stationarity["drift_z"] = 0.0
        stationary["epr_block_mean"] = ep_stationarity
        stationary["epr_block_autocorrelation_resolved"] = bool(exact_zero_epr or ep_stationarity["reliable"])
        if (not stationary["epr_block_autocorrelation_resolved"]
                or ep_stationarity.get("drift_z") is None or abs(ep_stationarity["drift_z"]) > 3):
            stationary["sampling_checks_passed"] = False
            stationary["status"] = "not_established"
        tail = {"kernel_tail_L1": float(np.sum(np.abs(model.K)[model.shell_labels > cfg.max_distance]) /
                                             np.sum(np.abs(model.K))),
                "max_distance": cfg.max_distance, "active_tail_ratio": {}, "sigma_tail": [],
                "note": "Signed tail EPR is retained. Energy ratios are not EPR error bounds."}
        for mechanism, energy in tail_energy.items():
            numerator, denominator = energy
            tail["active_tail_ratio"][mechanism] = {"ratio": numerator / denominator if denominator > 0 else None,
                "mean_tail_energy": numerator / completed if completed else None,
                "mean_active_energy": denominator / completed if completed else None}
        tail["sigma_tail"] = [row for row in summary if row["shell"] == "tail"]
        boundary = {}
        if passive_start is not None and phi_measure_start is not None:
            final_channels = model.passive_channels(phi)
            for key, initial_nu in passive_start.items():
                expected = -float(np.sum(phi * final_channels[key] - phi_measure_start * initial_nu)) / (2 * model.config.D)
                measured = passive_integrals.get(key, 0.0)
                boundary[key] = {"integrated_midpoint_work": measured, "minus_delta_F_over_D": expected,
                                 "absolute_chain_rule_error": abs(measured - expected)}
        linear_symbol = model.linear_symbol()
        nonzero_mass = model.Lhat > 1e-14
        real_stiffness = np.real(linear_symbol)[nonzero_mass]
        growth_rate = -model.Lhat[nonzero_mass] * real_stiffness
        linear_stability = {"min_real_stiffness_nonzero": float(real_stiffness.min()),
                            "max_real_growth_rate_nonzero": float(growth_rate.max()),
                            "homogeneous_linearly_stable": bool(real_stiffness.min() > 0),
                            "around_phi_bar": model.config.phi_bar,
                            "note": "Linear stability of the uniform field only; not a nonlinear stability or stationarity test."}
        diagnostics = {"run_status": status, "failure": failure, "completed_burn_in_steps": burn_completed,
                       "completed_measurement_steps": completed, "stationarity": stationary,
                       "linear_stability": linear_stability,
                       "mass": {"initial_total": initial_mass, "final_total": float(phi.sum()),
                                "maximum_absolute_drift": max_mass_error,
                                "maximum_reported_correction": max_mass_correction},
                       "solver": {"maximum_residual_norm": max_residual, "maximum_iterations": max_iterations,
                                  "failed_step_retried": False}, "tail": tail,
                       "passive_quadratic_boundary": boundary, "storage_quantization": quantization,
                       "noise_covariance": "bond-incidence construction; empirical check belongs to algebra tests",
                       "gaussian_reference": gaussian,
                       "receiver_diagnostics": {"attribution_rule": "u_times_ur",
                           "mixed_shell_covariances": receiver_covariance / len(frames) if frames else None,
                           "covariance_definition": "saved-frame mean of spatially centered u_r*u_s",
                           "n_observation_frames": len(frames)} if receiver_shell_hats is not None else None,
                       "epr_interpretation": "Finite-run signed active work. Stationary continuous-time EPR requires independent-seed and dt/lag checks."}
        nn = {"n_components": 1, "ep_component_indices": [0], "n_channel": None,
              "n_hidden": None, "max_distance": cfg.max_distance, "shell_force_activation": None,
              "constructor_configuration": "unconfigured; supply nn_metadata for intended experiment",
              "training_performed": False, "loss": None, "split": None,
              "normalization": "physical phi; no simulator normalization"}
        if nn_metadata:
            nn.update(nn_metadata)
            nn["constructor_configuration"] = "provided_nn_metadata"
        versions = {"python": platform.python_version()}
        for package in ("numpy", "scipy", "torch"):
            try:
                versions[package] = metadata.version(package)
            except metadata.PackageNotFoundError:
                versions[package] = None
        network_source = Path(__file__).resolve().parents[2] / "models" / "NEEP_ShellForce_2D.py"
        nn["source_sha256"] = hashlib.sha256(network_source.read_bytes()).hexdigest() if network_source.exists() else None
        nn["spec_source_sha256"] = "baea2ddf87ed10ac5c4150617e8a24556f6d6490c54009b671d4254d1a79cf79"
        nn["source_note"] = "Repository scalar adapter; source identity/architecture differences are explicit, not assumed identical to the specification file."
        manifest = {"model_id": model.config.model_id, "spec_version": model.spec_version,
                    "seed": cfg.seed, "Nx": model.shape[1], "Ny": model.shape[0], "h": 1.0,
                    "phi_bar": model.config.phi_bar, "D": model.config.D, "dt": cfg.dt,
                    "model_parameters": asdict(model.config), "run_config": asdict(cfg),
                    "observation_lag_steps": list(cfg.observation_lags),
                    "observation_lags": [lag * cfg.dt for lag in cfg.observation_lags],
                    "saved_frame_dt": cfg.observation_stride * cfg.dt,
                    "available_observation_lag_steps": [lag for lag in cfg.observation_lags
                                                          if lag <= max(0, len(frames) - 1) * cfg.observation_stride],
                    "burn_in_rule": "fixed steps; adequacy assessed against measured tau_int, not presumed",
                    "measurement_length": completed * cfg.dt,
                    "requested_measurement_length": cfg.measurement_steps * cfg.dt,
                    "solver": {"method": "implicit midpoint with one fixed conserved-noise draw per step",
                               "atol": cfg.solver_atol, "rtol": cfg.solver_rtol, "maxiter": cfg.solver_maxiter},
                    "epr_sampling": "every integration step; midpoint active work and pre-step drift-minus-trace",
                    "kernel_convention": "FFT origin (0,0); Kf[i]=sum_s K[s] f[i-s]; all Chebyshev shells retained",
                    "normalization": "none; physical scalar phi in one channel",
                    "source_hash": _source_hashes(), "environment_versions": versions, "nn": nn,
                    "kernel_sha256": hashlib.sha256(model.K.tobytes()).hexdigest(),
                    "nn_score_units": "raw branch is a spatial mean; extensive dS=N*sum(branch_score)",
                    "known_limitations": ["finite-size periodic cutoff h=1", "tail not truncated or renormalized",
                         "receiver attribution u*u_r is a convention with cross-shell inputs",
                         "raw NN branch attribution is not assumed equal to physical shell EPR",
                         "short smoke runs do not establish stationary EPR"], "run_status": status}
        result = ExperimentResult(stored, np.asarray(times), summary, json_safe(diagnostics), json_safe(manifest),
                                  arrays, counts, obs, model)
        result.diagnostics["elapsed_seconds"] = result.manifest["elapsed_seconds"] = perf_counter() - started_at
        if output_dir is not None:
            result.save(output_dir)
            # Include artifact writes in the duration retained in both memory
            # and the saved metadata (the final timing refresh is excluded).
            result.diagnostics["elapsed_seconds"] = result.manifest["elapsed_seconds"] = perf_counter() - started_at
            write_json(Path(output_dir) / "diagnostics.json", result.diagnostics)
            write_json(Path(output_dir) / "manifest.json", result.manifest)
        return result

    total_steps = cfg.burn_in_steps + cfg.measurement_steps
    bar = tqdm(total=total_steps, desc=progress_desc or f"{model.config.model_id} seed={cfg.seed}",
               unit="step", mininterval=0.25, dynamic_ncols=True, disable=not show_progress,
               bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} "
                          "[{elapsed} elapsed, ETA {remaining}, {rate_fmt}{postfix}]")
    bar.set_postfix_str("phase=burn-in" if cfg.burn_in_steps else "phase=measurement", refresh=False)
    try:
        for step in range(total_steps):
            if step == cfg.burn_in_steps:
                bar.set_postfix_str("phase=measurement", refresh=False)
                phi_measure_start = phi.copy()
                passive_start = model.passive_channels(phi)
                record_frame(step * cfg.dt)
            old = phi
            next_phi, info = model.step(old, cfg.dt, rng, atol=cfg.solver_atol,
                                       rtol=cfg.solver_rtol, maxiter=cfg.solver_maxiter)
            if not np.isfinite(next_phi).all():
                raise RuntimeError("Integrator returned nonfinite values")
            phi = np.asarray(next_phi, dtype=np.float64)
            max_mass_error = max(max_mass_error, abs(float(phi.sum()) - initial_mass))
            max_mass_correction = max(max_mass_correction, abs(float(info.get("mass_correction", 0.0))))
            max_residual = max(max_residual, float(info.get("residual_norm", 0.0)))
            max_iterations = max(max_iterations, int(info.get("iterations", 0)))
            if step < cfg.burn_in_steps:
                burn_completed += 1
            else:
                midpoint = (old + phi) / 2
                delta = phi - old
                active_mid = _aggregates(model.active_channels(midpoint), cfg.max_distance, np.zeros_like(phi))
                active_old = _aggregates(model.active_channels(old), cfg.max_distance, np.zeros_like(phi))
                trace = _aggregates(model.active_channel_traces(old), cfg.max_distance, 0.0)
                Lmu = laplacian(model.mu(old))
                for key, nu in active_mid.items():
                    mechanism, shell = key.split(":")
                    accumulate(f"midpoint|{mechanism}|{shell}", -np.sum(nu * delta) / model.config.D)
                    equal_time = np.sum(active_old[key] * Lmu) / model.config.D - trace[key]
                    accumulate(f"drift_minus_trace|{mechanism}|{shell}", equal_time * cfg.dt)
                if model.config.model_id == "M02":
                    angular = _aggregates(model.angular_channels(midpoint), cfg.max_distance, np.zeros_like(phi))
                    for key, nu in angular.items():
                        mechanism, shell = key.split(":")
                        if mechanism != "total":
                            accumulate(f"midpoint|{mechanism}|{shell}", -np.sum(nu * delta) / model.config.D)
                accumulate("midpoint|full_mu|all", -np.sum(model.mu(midpoint) * delta) / model.config.D)
                for key, nu in model.passive_channels(midpoint).items():
                    mechanism, shell = key.split(":")
                    work = -float(np.sum(nu * delta)) / model.config.D
                    passive_integrals[key] = passive_integrals.get(key, 0.0) + work
                    accumulate(f"midpoint|{mechanism}|{shell}", work)
                for key, nu in active_old.items():
                    mechanism, shell = key.split(":")
                    if shell == "all":
                        tail_nu = active_old[f"{mechanism}:tail"]
                        pair = tail_energy.setdefault(mechanism, [0.0, 0.0])
                        pair[0] += max(0.0, float(np.sum(tail_nu * laplacian(tail_nu))))
                        pair[1] += max(0.0, float(np.sum(nu * laplacian(nu))))
                completed += 1
                in_block += 1
                if completed % cfg.observation_stride == 0:
                    record_frame((step + 1) * cfg.dt)
                if in_block == cfg.block_steps:
                    close_block()
            bar.update(1)
            if progress is not None:
                progress(step + 1, total_steps)
            if progress_details is not None:
                elapsed = perf_counter() - started_at
                progress_details({"completed_steps": step + 1, "total_steps": total_steps,
                                  "phase": "burn-in" if step < cfg.burn_in_steps else "measurement",
                                  "elapsed_seconds": elapsed,
                                  "eta_seconds": elapsed * (total_steps - step - 1) / (step + 1)})
    except Exception as exc:
        bar.set_postfix_str("phase=failed", refresh=False)
        result = finalize("incomplete", f"{type(exc).__name__}: {exc}")
        raise SimulationFailure(f"Simulation failed after {burn_completed} burn-in and {completed} measurement steps. "
                                "Inspect the preserved partial result and restart with a smaller dt. "
                                f"Cause: {exc}", result) from exc
    finally:
        bar.close()
    return finalize("complete")


def load_experiment(output_dir, *, mmap_mode=None) -> ExperimentResult:
    """Load run artifacts without pickle; reconstruct the exact configured model."""
    path = Path(output_dir)
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    diagnostics = json.loads((path / "diagnostics.json").read_text(encoding="utf-8"))
    with np.load(path / "kernel.npz", allow_pickle=False) as saved_kernel:
        model = AMBKernelModel(ModelConfig(**manifest["model_parameters"]), kernel=saved_kernel["K"])
    with (path / "ep_blocks.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        for key in ("mean", "SE", "ci95_lower", "ci95_upper", "duration"):
            row[key] = float(row[key]) if row[key] else None
        for key in ("n_steps", "n_blocks", "n_full_blocks"):
            row[key] = int(row[key])
    with np.load(path / "ep_block_values.npz", allow_pickle=False) as saved:
        counts = saved["block_step_counts"].copy()
        blocks = {key: saved[key].copy() for key in saved.files if key != "block_step_counts"}
    with np.load(path / "observables.npz", allow_pickle=False) as saved:
        obs = {key: saved[key].copy() for key in saved.files}
    return ExperimentResult(np.load(path / "fields.npy", mmap_mode=mmap_mode, allow_pickle=False),
                            np.load(path / "times.npy", allow_pickle=False), rows, diagnostics,
                            manifest, blocks, counts, obs, model, path)
