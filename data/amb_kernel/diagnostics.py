"""Sampling diagnostics for the fixed-cutoff scalar AMB experiments.

These checks describe a finite run; they do not establish stationarity or a
continuous-time EPR without independent seeds and dt/lag convergence.
"""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def json_safe(value: Any) -> Any:
    """Convert NumPy/dataclass results to strict JSON, using null for missing data."""
    if is_dataclass(value):
        return json_safe(asdict(value))
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, complex):
        return {"real": json_safe(value.real), "imag": json_safe(value.imag)}
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(json_safe(data), ensure_ascii=False, indent=2,
                               allow_nan=False) + "\n", encoding="utf-8")


def integrated_autocorrelation(values, sample_dt: float = 1.0) -> dict:
    """FFT autocorrelation with an initial-positive-pair window.

    tau_int uses the convention 1/2 + sum_{k>0} rho(k). Reliability is only
    a sample-length diagnostic, not proof that unobserved slow modes are absent.
    """
    x = np.asarray(values, dtype=np.float64)
    n = x.size
    result = {"n_samples": int(n), "tau_int": None, "tau_int_samples": None,
              "effective_samples": None, "reliable": False, "window": None}
    if n < 16 or not np.isfinite(x).all():
        result["reason"] = "At least 16 finite observations are required."
        return result
    x = x - x.mean()
    variance = float(np.mean(x * x))
    if variance <= np.finfo(np.float64).tiny:
        result["reason"] = "Observable has zero measured variance."
        return result
    size = 1 << (2 * n - 1).bit_length()
    transform = np.fft.rfft(x, n=size)
    covariance = np.fft.irfft(transform * transform.conj(), n=size)[:n]
    covariance /= np.arange(n, 0, -1)
    rho = covariance / covariance[0]
    tau = 0.5
    window = 0
    found_window = False
    previous_pair = float("inf")
    for k in range(1, n // 2 - 1, 2):
        pair = float(rho[k] + rho[k + 1])
        if pair <= 0:
            found_window = True
            break
        pair = min(pair, previous_pair)
        previous_pair = pair
        tau += pair
        window = k + 1
    tau = max(0.5, tau)
    reliable = bool(n >= 128 and n >= 50 * tau and found_window)
    result.update(tau_int=float(tau * sample_dt), tau_int_samples=float(tau),
                  effective_samples=float(n / (2 * tau)), reliable=reliable,
                  window=int(window), reason=("Length/window heuristic passed."
                      if reliable else "Run is too short or no positive-pair window closed."))
    return result


def stationarity_report(observables: dict, *, observation_dt: float,
                        burn_in_time: float, block_time: float,
                        full_blocks: int, model_a: float) -> dict:
    """Report tau, window drift and sampling requirements without certifying EPR."""
    details = {}
    for key in ("phi2", "gradient_energy_density", "low_q_structure_factor",
                "structure_factor_x", "structure_factor_y"):
        values = np.asarray(observables.get(key, []), dtype=np.float64)
        info = integrated_autocorrelation(values, observation_dt)
        if values.size >= 16 and np.isfinite(values).all():
            half = values.size // 2
            left, right = values[:half], values[half:]
            delta = float(right.mean() - left.mean())
            tau_samples = info["tau_int_samples"]
            if tau_samples is not None:
                variance = float(np.var(values, ddof=1))
                se = np.sqrt(2 * tau_samples * variance * (1 / len(left) + 1 / len(right)))
                z = delta / se if se > 0 else (0.0 if delta == 0 else None)
            else:
                z = None
            info.update(first_half_mean=float(left.mean()), second_half_mean=float(right.mean()),
                        half_mean_difference=delta, drift_z=z)
        details[key] = info
    taus = [v["tau_int"] for v in details.values() if v["tau_int"] is not None]
    slow_tau = max(taus) if taus else None
    reliable = bool(details and all(v["reliable"] for v in details.values()))
    burn_ok = bool(reliable and slow_tau is not None and burn_in_time >= 20 * slow_tau)
    block_ok = bool(reliable and slow_tau is not None and block_time >= 5 * slow_tau)
    drift_ok = bool(reliable and all(v.get("drift_z") is not None and
                                     abs(v["drift_z"]) <= 3 for v in details.values()))
    enough_blocks = full_blocks >= 30
    sampling_passed = burn_ok and block_ok and drift_ok and enough_blocks
    return {"status": "sampling_checks_passed_convergence_not_checked" if sampling_passed
            else "not_established", "stationary_epr_established": False,
            "observables": details, "slowest_measured_tau_int": slow_tau,
            "burn_in_at_least_20_tau": burn_ok, "block_at_least_5_tau": block_ok,
            "at_least_30_full_blocks": enough_blocks, "no_resolved_half_run_drift": drift_ok,
            "full_blocks": int(full_blocks), "sampling_checks_passed": sampling_passed,
            "independent_seed_check": "not_performed_by_single_run",
            "dt_convergence": "not_performed_by_single_run",
            "observation_lag_convergence": "not_performed_by_single_run",
            "coarsening_check": "domain_size_not_measured; inspect morphology" if model_a < 0
                else "not_certified_by_homogeneous_observables",
            "note": "Short runs are smoke tests. Block SE/CI assumes sufficiently long independent blocks; "
                    "sampling checks do not establish dt/lag convergence or exclude unmeasured slow modes."}


def summarize_blocks(block_values: dict, block_step_counts, *, dt: float,
                     full_block_steps: int) -> list[dict]:
    """Mean over all steps; uncertainty uses only complete equal-length blocks."""
    counts = np.asarray(block_step_counts, dtype=np.int64)
    complete = counts == full_block_steps
    n_complete = int(complete.sum())
    n_steps = int(counts.sum())
    rows = []
    for key, values in block_values.items():
        method, mechanism, shell = key.split("|")
        rates = np.asarray(values, dtype=np.float64)
        mean = float(np.dot(rates, counts) / n_steps) if n_steps else None
        se = float(np.std(rates[complete], ddof=1) / np.sqrt(n_complete)) if n_complete >= 2 else None
        rows.append({"method": method, "mechanism": mechanism, "shell": shell,
                     "mean": mean, "SE": se, "ci95_lower": None if se is None else mean - 1.96 * se,
                     "ci95_upper": None if se is None else mean + 1.96 * se,
                     "n_steps": n_steps, "n_blocks": int(counts.size),
                     "n_full_blocks": n_complete, "duration": n_steps * dt,
                     "uncertainty": "provisional_normal_approximation; check block independence"})
    return rows
