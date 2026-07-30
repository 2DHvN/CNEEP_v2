"""Benchmark exact lattice-ABP sweeps on an NVIDIA L40S-class CUDA device.

This measures the in-device simulation loop: initialization, trajectory
storage, plotting, and CPU transfers are outside the timed region.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.lattice_abp_tc import (  # noqa: E402
    ThermodynamicLatticeABP,
    ThermodynamicLatticeABPParams,
)


def n_from_packing_fraction(phi: float, box_size: float, sigma: float) -> int:
    return int(round(phi * 4.0 * box_size * box_size / (math.pi * sigma * sigma)))


def validate_state(
    sim: ThermodynamicLatticeABP,
    sites: torch.Tensor,
    theta: torch.Tensor,
    occupancy: torch.Tensor,
    diagnostics: dict[str, torch.Tensor],
) -> None:
    rebuilt = sim.occupancy_from_sites(sites)
    if not torch.equal(occupancy, rebuilt):
        raise AssertionError("occupancy no longer matches particle sites")
    if not bool(((occupancy >= 0) & (occupancy <= 1)).all().item()):
        raise AssertionError("exclusive lattice occupancy was violated")
    if not bool((occupancy.sum(dim=(-2, -1)) == sim.N).all().item()):
        raise AssertionError("particle number was not conserved")
    if not bool(torch.isfinite(theta).all().item()):
        raise AssertionError("non-finite angle encountered")
    torch.testing.assert_close(
        diagnostics["active_medium_ep"] + diagnostics["wca_medium_ep"],
        diagnostics["medium_ep"],
        rtol=1.0e-5 if sim.dtype == torch.float32 else 1.0e-12,
        atol=1.0e-5 if sim.dtype == torch.float32 else 1.0e-12,
    )


def benchmark_batch(
    args: argparse.Namespace,
    batch_size: int,
) -> dict[str, object]:
    n_particles = args.N
    if n_particles is None:
        n_particles = n_from_packing_fraction(args.phi, args.L, args.sigma)

    params = ThermodynamicLatticeABPParams(
        N=n_particles,
        L=args.L,
        grid_size=args.grid_size,
        sigma=args.sigma,
        epsilon=args.epsilon,
        v0=args.v0,
        Dr=args.Dr,
        Dt=args.Dt,
        dt=args.dt,
        prefactor=args.prefactor,
        strict_probabilities=True,
        seed=args.seed,
        device=args.device,
        dtype=args.dtype,
        backend=args.backend,
    )
    sim = ThermodynamicLatticeABP(params)
    sites, theta = sim.init_state(batch_size)
    occupancy = sim.occupancy_from_sites(sites)

    diagnostics: dict[str, torch.Tensor] | None = None
    deferred_status: list[torch.Tensor] = []
    with torch.no_grad():
        # Warm-up absorbs lazy CUDA initialization and any backend compilation.
        for _ in range(args.warmup):
            sites, theta, occupancy, diagnostics = sim._step_inplace(
                sites,
                theta,
                occupancy,
            )
        torch.cuda.synchronize(sim.device)
        torch.cuda.reset_peak_memory_stats(sim.device)

        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(args.steps):
            sites, theta, occupancy, diagnostics = sim._step_inplace(
                sites,
                theta,
                occupancy,
                check_probability_errors=False,
            )
            if "backend_status" in diagnostics:
                deferred_status.append(diagnostics["backend_status"])
            elif "_probability_status" in diagnostics:
                deferred_status.append(diagnostics["_probability_status"])
        stop.record()
        stop.synchronize()

    assert diagnostics is not None
    if deferred_status:
        status = int(torch.stack(deferred_status).max().cpu())
        if status != 0:
            raise ValueError(
                f"CUDA probability validation failed with status {status}."
            )
    validate_state(sim, sites, theta, occupancy, diagnostics)

    elapsed_seconds = start.elapsed_time(stop) / 1000.0
    ensemble_sweeps = batch_size * args.steps
    particle_updates = ensemble_sweeps * params.N
    return {
        "batch_size": batch_size,
        "particles_per_ensemble": params.N,
        "timed_steps": args.steps,
        "seconds": elapsed_seconds,
        "step_calls_per_second": args.steps / elapsed_seconds,
        "ensemble_sweeps_per_second": ensemble_sweeps / elapsed_seconds,
        "particle_updates_per_second": particle_updates / elapsed_seconds,
        "peak_allocated_gib": torch.cuda.max_memory_allocated(sim.device)
        / (1024.0**3),
        "resolved_backend": sim.backend,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--backend", default="cuda_fused")
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 16, 64])
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--L", type=float, default=24.0)
    parser.add_argument("--grid-size", type=int, default=96)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--phi", type=float, default=0.55)
    parser.add_argument("--N", type=int, default=None)
    parser.add_argument("--epsilon", type=float, default=20.0)
    parser.add_argument("--v0", type=float, default=24.0)
    parser.add_argument("--Dr", type=float, default=1.0)
    parser.add_argument("--Dt", type=float, default=0.2)
    parser.add_argument("--dt", type=float, default=1.0e-4)
    parser.add_argument("--prefactor", choices=["cv", "c0"], default="cv")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    if args.steps <= 0 or args.warmup < 0:
        parser.error("--steps must be positive and --warmup must be nonnegative")
    if any(batch_size <= 0 for batch_size in args.batch_sizes):
        parser.error("all --batch-sizes values must be positive")
    return args


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable; run this benchmark on the L40S host.")

    device = torch.device(args.device)
    properties = torch.cuda.get_device_properties(device)
    header = {
        "device": properties.name,
        "compute_capability": f"{properties.major}.{properties.minor}",
        "visible_memory_gib": properties.total_memory / (1024.0**3),
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "dtype": args.dtype,
        "semantics": (
            "random-sequential single-particle WCA sweeps; no tau leaping, "
            "cutoff reduction, particle subsampling, or changed dt"
        ),
    }
    results = [benchmark_batch(args, batch_size) for batch_size in args.batch_sizes]
    print(json.dumps({"environment": header, "results": results}, indent=2))


if __name__ == "__main__":
    main()
