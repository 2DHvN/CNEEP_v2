"""Batched float64 midpoint integration with device-side physical diagnostics.

Each replica has its own NumPy PCG64 stream, independent of batch composition.
Only noise is copied to the device each step. Field transfers occur at requested
observation times, and reduced work statistics are transferred once per block.
Observation frames stream to .npy memory maps rather than accumulating in RAM.
"""
from __future__ import annotations

from dataclasses import asdict
import csv
import hashlib
from importlib import metadata
import json
from pathlib import Path
import platform
from time import perf_counter

import numpy as np
import torch
from tqdm.auto import tqdm

from .diagnostics import integrated_autocorrelation, json_safe, stationarity_report, summarize_blocks, write_json
from .experiment import RunConfig, _aggregates, _source_hashes
from .model import AMBKernelModel
from .torch_model import TorchAMBKernelModel


class BatchedSimulationFailure(RuntimeError):
    """No replica retries failed noise; all partial artifacts remain inspectable."""
    def __init__(self, message, output_dirs):
        super().__init__(message)
        self.output_dirs = list(output_dirs)


def _cpu(value):
    return value.detach().cpu().numpy()


def _sum(value):
    return value.sum(dim=(-2, -1))


class _FrameWriter:
    """A private staging map keeps failed runs' final field shape honest."""
    def __init__(self, path, cfg, shape):
        self.path = path
        path.mkdir(parents=True, exist_ok=True)
        self.temporary = path / ".fields.inprogress.npy"
        self.array = np.lib.format.open_memmap(self.temporary, mode="w+", dtype=cfg.storage_dtype,
            shape=(cfg.measurement_steps // cfg.observation_stride + 1, 1, *shape))
        self.count = 0

    def append(self, field):
        self.array[self.count, 0] = field
        self.count += 1

    def close(self):
        if not self.array._mmap.closed:
            try:
                self.array.flush()
            finally:
                self.array._mmap.close()

    def finish(self):
        self.array.flush()
        final = self.path / "fields.npy"
        if self.count == len(self.array):
            self.array._mmap.close()
            self.temporary.replace(final)
        else:
            # A failed run can have fewer frames, including none during burn-in.
            target_shape = (self.count, *self.array.shape[1:])
            if self.count:
                trimmed = np.lib.format.open_memmap(final, mode="w+", dtype=self.array.dtype, shape=target_shape)
                for start in range(0, self.count, 64):
                    trimmed[start:start + 64] = self.array[start:min(start + 64, self.count)]
                trimmed.flush()
                trimmed._mmap.close()
            else:
                np.save(final, np.empty(target_shape, dtype=self.array.dtype), allow_pickle=False)
            self.array._mmap.close()
            self.temporary.unlink()


def _finish_replica(reference, cfg, path, *, index, states, nn_metadata,
                    device, batch_size, status, failure, elapsed_seconds):
    """Write the CPU runner's artifact schema from per-replica scalar reductions."""
    counts = np.asarray(states["block_counts"], dtype=np.int64)
    blocks = {key: np.asarray(values, dtype=np.float64)[:, index]
              for key, values in states["blocks"].items()}
    summary = summarize_blocks(blocks, counts, dt=cfg.dt, full_block_steps=cfg.block_steps)
    times = np.asarray(states["times"], dtype=np.float64)
    obs = {key: np.asarray(values, dtype=np.float64)[:, index]
           for key, values in states["observables"].items()}
    obs["times"] = times
    completed, burn_completed = states["completed"], states["burn_completed"]
    stationarity = stationarity_report(obs, observation_dt=cfg.observation_stride * cfg.dt,
        burn_in_time=burn_completed * cfg.dt, block_time=cfg.block_steps * cfg.dt,
        full_blocks=int(np.sum(counts == cfg.block_steps)), model_a=reference.config.a)
    rates = blocks.get("midpoint|total|all", np.array([]))[counts == cfg.block_steps]
    ep = integrated_autocorrelation(rates, cfg.block_steps * cfg.dt)
    if len(rates) >= 4:
        half = len(rates) // 2
        delta = float(rates[half:].mean() - rates[:half].mean())
        se = float(np.std(rates, ddof=1) * np.sqrt(1 / half + 1 / (len(rates) - half)))
        ep.update(half_mean_difference=delta, drift_z=delta / se if se else 0.0)
    zero_ep = bool(rates.size and np.all(rates == 0))
    ep.update(exact_zero_observed_series=zero_ep, autocorrelation_required=not zero_ep)
    if zero_ep:
        ep.update(drift_z=0.0, reason="Every complete EPR block is exactly zero; autocorrelation is undefined and not required for this null series.")
    stationarity["epr_block_mean"] = ep
    stationarity["epr_block_autocorrelation_resolved"] = bool(zero_ep or ep["reliable"])
    if not stationarity["epr_block_autocorrelation_resolved"] or ep.get("drift_z") is None or abs(ep["drift_z"]) > 3:
        stationarity.update(sampling_checks_passed=False, status="not_established")
    tail = {"kernel_tail_L1": float(np.abs(reference.K)[reference.shell_labels > cfg.max_distance].sum() /
                                       np.abs(reference.K).sum()),
            "max_distance": cfg.max_distance, "active_tail_ratio": {},
            "sigma_tail": [row for row in summary if row["shell"] == "tail"],
            "note": "Signed tail EPR is retained. Energy ratios are not EPR error bounds."}
    for mechanism, energies in states["tail_energy"].items():
        numerator, denominator = energies[:, index]
        tail["active_tail_ratio"][mechanism] = {
            "ratio": numerator / denominator if denominator > 0 else None,
            "mean_tail_energy": numerator / completed if completed else None,
            "mean_active_energy": denominator / completed if completed else None}
    boundary = {}
    for key, expected in states["passive_boundary"].items():
        measured = states["passive_integrals"][key][index]
        boundary[key] = {"integrated_midpoint_work": measured, "minus_delta_F_over_D": expected[index],
                         "absolute_chain_rule_error": abs(measured - expected[index])}
    mask = reference.Lhat > 1e-14
    stiffness = reference.linear_symbol().real[mask]
    quant = states["quantization"][:, index]
    diagnostics = {
        "run_status": status, "failure": failure, "completed_burn_in_steps": burn_completed,
        "completed_measurement_steps": completed, "stationarity": stationarity,
        "linear_stability": {"min_real_stiffness_nonzero": float(stiffness.min()),
            "max_real_growth_rate_nonzero": float((-reference.Lhat[mask] * stiffness).max()),
            "homogeneous_linearly_stable": bool(stiffness.min() > 0), "around_phi_bar": reference.config.phi_bar,
            "note": "Linear stability of the uniform field only; not a nonlinear stability or stationarity test."},
        "mass": {"initial_total": states["initial_mass"][index], "final_total": states["final_mass"][index],
            "maximum_absolute_drift": states["max_mass"][index], "maximum_reported_correction": 0.0},
        "solver": {"maximum_residual_norm": states["max_residual"][index],
            "maximum_iterations": int(states["max_iterations"][index]), "failed_step_retried": False,
            "backend": "torch", "method": states["solver"]},
        "tail": tail, "passive_quadratic_boundary": boundary,
        "storage_quantization": {"storage_dtype": cfg.storage_dtype,
            "increment_relative_l2_error": np.sqrt(quant[1] / quant[0]) if quant[0] else 0.0,
            "lost_nonzero_increment_fraction": quant[3] / quant[2] if quant[2] else 0.0},
        "noise_covariance": "Per-replica bond-incidence noise; empirical covariance tested independently.",
        "gaussian_reference": states["gaussian_reference"],
        "receiver_diagnostics": {"attribution_rule": "u_times_ur",
            "mixed_shell_covariances": states["receiver_covariance"][index] / len(times) if len(times) else None,
            "covariance_definition": "saved-frame mean of spatially centered u_r*u_s",
            "n_observation_frames": len(times)} if reference.config.model_id == "M05" else None,
        "epr_interpretation": "Finite-run signed active work. Stationary continuous-time EPR requires independent-seed and dt/lag checks.",
        "elapsed_seconds": elapsed_seconds, "elapsed_scope": "shared batched integration and device diagnostics"}
    nn = {"n_components": 1, "ep_component_indices": [0], "n_channel": None, "n_hidden": None,
          "max_distance": cfg.max_distance, "shell_force_activation": None,
          "constructor_configuration": "unconfigured; supply nn_metadata for intended experiment",
          "training_performed": False, "loss": None, "split": None,
          "normalization": "physical phi; no simulator normalization"}
    if nn_metadata:
        nn.update(nn_metadata)
        nn["constructor_configuration"] = "provided_nn_metadata"
    network_source = Path(__file__).resolve().parents[2] / "models" / "NEEP_ShellForce_2D.py"
    nn.update(source_sha256=hashlib.sha256(network_source.read_bytes()).hexdigest() if network_source.exists() else None,
              spec_source_sha256="baea2ddf87ed10ac5c4150617e8a24556f6d6490c54009b671d4254d1a79cf79",
              source_note="Repository scalar adapter; source differences from the specification are explicit.")
    versions = {"python": platform.python_version()}
    for package in ("numpy", "scipy", "torch"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    manifest = {"model_id": reference.config.model_id, "spec_version": reference.spec_version,
        "seed": cfg.seed, "Nx": reference.shape[1], "Ny": reference.shape[0], "h": 1.0,
        "phi_bar": reference.config.phi_bar, "D": reference.config.D, "dt": cfg.dt,
        "model_parameters": asdict(reference.config), "run_config": asdict(cfg),
        "observation_lag_steps": list(cfg.observation_lags),
        "observation_lags": [lag * cfg.dt for lag in cfg.observation_lags],
        "saved_frame_dt": cfg.observation_stride * cfg.dt,
        "available_observation_lag_steps": [lag for lag in cfg.observation_lags if lag <= max(0, len(times) - 1) * cfg.observation_stride],
        "burn_in_rule": "fixed steps; adequacy assessed against measured tau_int, not presumed",
        "measurement_length": completed * cfg.dt, "requested_measurement_length": cfg.measurement_steps * cfg.dt,
        "solver": {"method": states["solver"], "atol": cfg.solver_atol, "rtol": cfg.solver_rtol,
                   "maxiter": cfg.solver_maxiter},
        "epr_sampling": "every integration step on device; midpoint active work and pre-step drift-minus-trace",
        "kernel_convention": "FFT origin (0,0); Kf[i]=sum_s K[s] f[i-s]; all Chebyshev shells retained",
        "normalization": "none; physical scalar phi in one channel", "source_hash": _source_hashes(),
        "environment_versions": versions, "nn": nn,
        "kernel_sha256": hashlib.sha256(reference.K.tobytes()).hexdigest(),
        "nn_score_units": "raw branch is a spatial mean; extensive dS=N*sum(branch_score)",
        "known_limitations": ["finite-size periodic cutoff h=1", "tail not truncated or renormalized",
            "receiver attribution u*u_r is a convention with cross-shell inputs",
            "raw NN branch attribution is not assumed equal to physical shell EPR",
            "short smoke runs do not establish stationary EPR", "different numerical backends need not be bitwise identical"],
        "run_status": status, "simulation_backend": "torch", "simulation_device": str(device),
        "simulation_batch_size": batch_size, "noise_rng": "numpy_pcg64_per_replica",
        "elapsed_seconds": elapsed_seconds, "elapsed_scope": diagnostics["elapsed_scope"]}
    np.save(path / "times.npy", times, allow_pickle=False)
    np.savez_compressed(path / "kernel.npz", K=reference.K, Ke=reference.Ke, Ko=reference.Ko,
        Khat=reference.Khat, Lhat=reference.Lhat, shell_labels=reference.shell_labels,
        physical_parameters=json.dumps(json_safe(asdict(reference.config))))
    np.savez_compressed(path / "ep_block_values.npz", block_step_counts=counts, **blocks)
    np.savez_compressed(path / "observables.npz", **obs)
    columns = ("method", "mechanism", "shell", "mean", "SE", "ci95_lower", "ci95_upper",
               "n_steps", "n_blocks", "n_full_blocks", "duration", "uncertainty")
    with (path / "ep_blocks.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(json_safe(summary))
    write_json(path / "diagnostics.json", diagnostics)
    write_json(path / "manifest.json", manifest)


@torch.no_grad()
def run_experiments_batched(model, run_configs, output_dirs, *, device="cuda",
                            nn_metadata=None, progress=None, show_progress=True):
    """Save a synchronous batch of independent runs and return their paths.

    RunConfig values must agree except for seeds. Progress counts combined
    replica steps: total=B*(burn_in_steps+measurement_steps). Each replica's
    PCG64 draws match the NumPy runner regardless of batching. Arithmetic and
    nonlinear solvers can differ numerically between CPU and GPU backends.
    An explicit unavailable CUDA device is an error, never a CPU fallback.
    """
    started = perf_counter()
    configs, paths = list(run_configs), [Path(path) for path in output_dirs]
    if not configs or len(configs) != len(paths):
        raise ValueError("Provide one output directory per nonempty run_config batch.")
    if not all(isinstance(config, RunConfig) for config in configs):
        raise TypeError("Every run_config must be a RunConfig.")
    signatures = [{key: value for key, value in asdict(config).items() if key != "seed"} for config in configs]
    if any(signature != signatures[0] for signature in signatures[1:]):
        raise ValueError("Batched RunConfig values must agree except for seed.")
    if len(set(path.resolve() for path in paths)) != len(paths):
        raise ValueError("Each replica requires a distinct output directory.")
    artifacts = ("fields.npy", "times.npy", "kernel.npz", "manifest.json", "ep_blocks.csv",
                 "diagnostics.json", "ep_block_values.npz", "observables.npz", ".fields.inprogress.npy")
    for path in paths:
        if any((path / name).exists() for name in artifacts):
            raise FileExistsError(f"Run artifacts already exist in {path}; select a new output directory.")
    requested_device = torch.device(device)
    if requested_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA simulation was requested, but torch.cuda.is_available() is False.")
    if requested_device.type == "cuda" and requested_device.index is None:
        requested_device = torch.device("cuda", torch.cuda.current_device())
    if isinstance(model, TorchAMBKernelModel):
        gpu = model
        actual_device = gpu.device
        if actual_device.type == "cuda" and actual_device.index is None:
            actual_device = torch.device("cuda", torch.cuda.current_device())
        if actual_device != requested_device:
            raise ValueError("The supplied TorchAMBKernelModel device differs from device=.")
    else:
        reference = model if isinstance(model, AMBKernelModel) else AMBKernelModel(model)
        gpu = TorchAMBKernelModel(reference, device=requested_device)
    reference, cfg = gpu.reference, configs[0]
    batch = len(configs)
    rngs = [np.random.default_rng(config.seed) for config in configs]
    initial = np.stack([reference.initial_field(rng, scale=config.initial_scale)
                        for rng, config in zip(rngs, configs)])
    phi = torch.as_tensor(initial, dtype=torch.float64, device=gpu.device)
    zero = torch.zeros(batch, dtype=torch.float64, device=gpu.device)
    initial_mass = _sum(phi)
    max_mass, max_residual, max_iterations = zero.clone(), zero.clone(), zero.clone()
    quantization = torch.zeros((4, batch), dtype=torch.float64, device=gpu.device)
    previous_observation = None
    receiver_covariance = torch.zeros((batch, len(reference.radii), len(reference.radii)),
                                      dtype=torch.float64, device=gpu.device)
    khat = torch.as_tensor(reference.Khat, device=gpu.device)
    receiver_hats = torch.as_tensor(np.fft.fft2(reference.K_shells), device=gpu.device)
    block_accumulator, blocks, block_counts = {}, {}, []
    tail_energy, passive_integrals, passive_energy_start = {}, {}, {}
    times, observables, frame_buffer, observable_buffer = [], {}, [], []
    observation_names = ["phi2", "gradient_energy_density", "low_q_structure_factor",
                         "structure_factor_x", "structure_factor_y", "phi_min", "phi_max"]
    if reference.config.model_id == "M05":
        observation_names += ["receiver_u_mean", "receiver_u2", "receiver_u4"]
    completed = burn_completed = in_block = 0
    solver_name = "torch_implicit_midpoint_preconditioned_residual"
    writers = []

    def flush_frames():
        if not frame_buffer:
            return
        fields = _cpu(torch.stack(frame_buffer))
        measures = _cpu(torch.stack(observable_buffer))
        for frame_index in range(len(fields)):
            for replica, writer in enumerate(writers):
                writer.append(fields[frame_index, replica])
            for column, key in enumerate(observation_names):
                observables.setdefault(key, []).append(measures[frame_index, :, column])
        frame_buffer.clear()
        observable_buffer.clear()

    def record_frame(t):
        nonlocal previous_observation, receiver_covariance
        centered = phi - phi.mean(dim=(-2, -1), keepdim=True)
        spectrum = torch.fft.fft2(centered).abs().square() / reference.N
        qx, qy = spectrum[:, 0, 1], spectrum[:, 1, 0]
        values = [phi.square().mean(dim=(-2, -1)), (phi * gpu.laplacian(phi)).mean(dim=(-2, -1)) / 2,
                  (qx + qy) / 2, qx, qy, phi.amin(dim=(-2, -1)), phi.amax(dim=(-2, -1))]
        if reference.config.model_id == "M05":
            transform = torch.fft.fft2(phi)
            u = torch.fft.ifft2(khat * transform).real
            values += [u.mean(dim=(-2, -1)), u.square().mean(dim=(-2, -1)), u.pow(4).mean(dim=(-2, -1))]
            shells = torch.fft.ifft2(receiver_hats[None] * transform[:, None]).real.flatten(2)
            shells = shells - shells.mean(dim=-1, keepdim=True)
            receiver_covariance += torch.bmm(shells, shells.transpose(1, 2)) / reference.N
        if cfg.storage_dtype == "float32" and previous_observation is not None:
            exact = phi - previous_observation
            rounded = phi.float().double() - previous_observation.float().double()
            nonzero = exact != 0
            quantization[0] += _sum(exact.square())
            quantization[1] += _sum((rounded - exact).square())
            quantization[2] += _sum(nonzero)
            quantization[3] += _sum(nonzero & (rounded == 0))
        observation = phi.clone()
        previous_observation = observation
        frame_buffer.append(observation)
        observable_buffer.append(torch.stack(values, dim=1))
        times.append(t)
        # Bounded device staging amortizes transfers even with stride=1.
        if len(frame_buffer) >= 32:
            flush_frames()

    def close_block():
        nonlocal in_block
        if not in_block:
            return
        keys = list(block_accumulator)
        values = _cpu(torch.stack([block_accumulator[key] for key in keys])) / (in_block * cfg.dt)
        for key, value in zip(keys, values):
            blocks.setdefault(key, []).append(value)
        block_counts.append(in_block)
        block_accumulator.clear()
        in_block = 0

    def _finalize(status, failure=None):
        close_block()
        flush_frames()
        passive_boundary = {}
        if passive_energy_start:
            for key, nu in gpu.passive_channels(phi).items():
                passive_boundary[key] = _cpu(-(0.5 * _sum(phi * nu) - passive_energy_start[key]) / reference.config.D)
        if gpu.device.type == "cuda":
            torch.cuda.synchronize(gpu.device)
        states = {"block_counts": block_counts, "blocks": blocks, "times": times, "observables": observables,
            "completed": completed, "burn_completed": burn_completed,
            "initial_mass": _cpu(initial_mass), "final_mass": _cpu(_sum(phi)),
            "max_mass": _cpu(max_mass), "max_residual": _cpu(max_residual), "max_iterations": _cpu(max_iterations),
            "solver": solver_name, "quantization": _cpu(quantization),
            "receiver_covariance": _cpu(receiver_covariance),
            "tail_energy": {key: _cpu(value) for key, value in tail_energy.items()},
            "passive_integrals": {key: _cpu(value) for key, value in passive_integrals.items()},
            "passive_boundary": passive_boundary,
            "gaussian_reference": reference.gaussian_reference(max_distance=cfg.max_distance)
                if reference.config.model_id == "M03" and reference.config.b == 0 else None}
        elapsed = perf_counter() - started
        for index, (writer, config, path) in enumerate(zip(writers, configs, paths)):
            writer.finish()
            _finish_replica(reference, config, path, index=index, states=states, nn_metadata=nn_metadata,
                device=gpu.device, batch_size=batch, status=status, failure=failure, elapsed_seconds=elapsed)

    def finalize(status, failure=None):
        try:
            _finalize(status, failure)
        finally:
            for writer in writers:
                writer.close()

    # Gaussian positivity is checked before creating output files.
    if reference.config.model_id == "M03" and reference.config.b == 0:
        reference.gaussian_reference(max_distance=cfg.max_distance)
    total_steps = cfg.burn_in_steps + cfg.measurement_steps
    bar = tqdm(total=batch * total_steps, desc=f"{reference.config.model_id} {gpu.device} batch={batch}",
        unit="replica-step", mininterval=.25, disable=not show_progress, dynamic_ncols=True,
        bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed} elapsed, ETA {remaining}, {rate_fmt}{postfix}]")
    bar.set_postfix_str("phase=burn-in" if cfg.burn_in_steps else "phase=measurement", refresh=False)
    try:
        writers.extend(_FrameWriter(path, config, reference.shape) for path, config in zip(paths, configs))
        for step in range(total_steps):
            if step == cfg.burn_in_steps:
                bar.set_postfix_str("phase=measurement", refresh=False)
                for key, nu in gpu.passive_channels(phi).items():
                    passive_energy_start[key] = 0.5 * _sum(phi * nu)
                    passive_integrals[key] = zero.clone()
                record_frame(step * cfg.dt)
            noise = torch.as_tensor(np.stack([reference.noise(rng, cfg.dt) for rng in rngs]),
                                    dtype=torch.float64, device=gpu.device)
            next_phi, info = gpu.step_with_noise(phi, cfg.dt, noise, atol=cfg.solver_atol,
                rtol=cfg.solver_rtol, maxiter=cfg.solver_maxiter)
            solver_name = info["solver"]
            if step < cfg.burn_in_steps:
                burn_completed += 1
            else:
                midpoint, delta = (phi + next_phi) / 2, next_phi - phi
                zero_field = torch.zeros_like(phi)
                mid_channels = _aggregates(gpu.active_channels(midpoint), cfg.max_distance, zero_field)
                old_channels = _aggregates(gpu.active_channels(phi), cfg.max_distance, zero_field)
                traces = _aggregates(gpu.active_channel_traces(phi), cfg.max_distance, zero)
                Lmu = gpu.laplacian(gpu.mu(phi))
                increments = {}
                for key, nu in mid_channels.items():
                    mechanism, shell = key.split(":")
                    increments[f"midpoint|{mechanism}|{shell}"] = -_sum(nu * delta) / reference.config.D
                    increments[f"drift_minus_trace|{mechanism}|{shell}"] = (
                        _sum(old_channels[key] * Lmu) / reference.config.D - traces[key]) * cfg.dt
                increments["midpoint|full_mu|all"] = -_sum(gpu.mu(midpoint) * delta) / reference.config.D
                for key, nu in gpu.passive_channels(midpoint).items():
                    mechanism, shell = key.split(":")
                    work = -_sum(nu * delta) / reference.config.D
                    passive_integrals[key] += work
                    increments[f"midpoint|{mechanism}|{shell}"] = work
                if reference.config.model_id == "M02":
                    for key, nu in _aggregates(gpu.angular_channels(midpoint), cfg.max_distance, zero_field).items():
                        mechanism, shell = key.split(":")
                        if mechanism != "total":
                            increments[f"midpoint|{mechanism}|{shell}"] = -_sum(nu * delta) / reference.config.D
                for key, increment in increments.items():
                    if key not in block_accumulator:
                        block_accumulator[key] = increment.clone()
                    else:
                        block_accumulator[key] += increment
                for key, nu in old_channels.items():
                    mechanism, shell = key.split(":")
                    if shell == "all":
                        tail_nu = old_channels[f"{mechanism}:tail"]
                        energy = torch.stack((_sum(tail_nu * gpu.laplacian(tail_nu)).clamp_min(0),
                                              _sum(nu * gpu.laplacian(nu)).clamp_min(0)))
                        if mechanism not in tail_energy:
                            tail_energy[mechanism] = energy
                        else:
                            tail_energy[mechanism] += energy
                completed += 1
                in_block += 1
            phi = next_phi
            max_mass = torch.maximum(max_mass, (_sum(phi) - initial_mass).abs())
            max_residual = torch.maximum(max_residual, info["residual_norm"])
            max_iterations = torch.maximum(max_iterations, info["iterations"].to(dtype=torch.float64))
            if step >= cfg.burn_in_steps:
                if completed % cfg.observation_stride == 0:
                    record_frame((step + 1) * cfg.dt)
                if in_block == cfg.block_steps:
                    close_block()
            bar.update(batch)
            if progress is not None:
                progress(batch * (step + 1), batch * total_steps)
    except (Exception, KeyboardInterrupt) as exc:
        bar.set_postfix_str("phase=incomplete", refresh=False)
        finalize("incomplete", f"{type(exc).__name__}: {exc}")
        if isinstance(exc, KeyboardInterrupt):
            raise
        raise BatchedSimulationFailure(f"Batched simulation failed after {burn_completed} burn-in and "
            f"{completed} measurement steps per replica; partial artifacts were preserved. Cause: {exc}", paths) from exc
    finally:
        bar.close()
    finalize("complete")
    return paths
