"""Time exact ensemble simulation after compilation, including output transfer.

From CNEEP_v2: python -m data.labp_m.benchmark --replicas 16 64 --workers 1 4 8
Event pairs: add --sampling event --events 1024 --event-warmup 128
CUDA is included when available. This is a throughput comparison, not a MIPS
or entropy-recovery test. CPU/CUDA use different random-number generators.
"""

from __future__ import annotations

import argparse
import json
from time import perf_counter

import numpy as np

from .core import LABPMConfig, simulate_ensemble, simulation_backend
from .events import simulate_event_ensemble


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replicas", type=int, nargs="+", default=[16, 64])
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 4])
    parser.add_argument("--backends", nargs="+", choices=["cpu", "cuda"], default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lattice-size", type=int, default=24)
    parser.add_argument("--frames", type=int, default=21)
    parser.add_argument("--sampling", choices=["time", "event"], default="time")
    parser.add_argument("--events", type=int, default=1024)
    parser.add_argument("--event-warmup", type=int, default=128)
    parser.add_argument("--sample-dt", type=float, default=0.5)
    parser.add_argument("--burn-time", type=float, default=10.0)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    for values in (args.replicas, args.workers, [args.repeats, args.batch_size, args.frames, args.events, args.event_warmup]):
        if any(value < 1 for value in values):
            parser.error("replicas, workers, repeats, batch-size, frames, events and event-warmup must be positive")
    if args.sampling == "time" and args.frames < 2:
        parser.error("frames must be at least 2 to measure recorded event throughput")

    backends = args.backends
    if backends is None:
        backends = ["cpu"]
        if simulation_backend("auto") == "cuda":
            backends.append("cuda")
    if "cuda" in backends:
        from numba import config as numba_config

        if numba_config.ENABLE_CUDASIM:
            parser.error("CUDA simulator timings do not measure GPU performance")
        simulation_backend("cuda")  # Fail clearly for an explicit unavailable request.

    config = LABPMConfig(lattice_size=args.lattice_size)
    common = dict(burn_time=args.burn_time, seed=19, batch_size=args.batch_size)
    if args.sampling == "event":
        simulator = simulate_event_ensemble
        common.update(n_events=args.events, event_warmup=args.event_warmup)
        tiny = dict(n_events=2, event_warmup=1)
    else:
        simulator = simulate_ensemble
        common.update(n_frames=args.frames, sample_dt=args.sample_dt)
        tiny = dict(n_frames=2, sample_dt=.01)
    print(json.dumps({"config": config.__dict__, "observation": common,
                      "sampling": args.sampling,
                      "note": "timing includes host outputs and transfer; excludes first compilation"}))
    for backend in backends:
        # Compile the same dimensional/dtype signature with a tiny trajectory.
        simulator(config, 1, backend=backend, workers=1,
                  batch_size=args.batch_size, **tiny)
        for count in args.replicas:
            reference = None
            for workers in (args.workers if backend == "cpu" else [None]):
                timings = []
                for _ in range(args.repeats):
                    start = perf_counter()
                    result = simulator(config, count, backend=backend,
                                       workers=workers, **common)
                    timings.append(perf_counter() - start)
                    if reference is None:
                        reference = result
                    # Check reproducibility within the backend. No CPU/CUDA
                    # path equality is expected because their RNGs differ.
                    fields = ("states", "hop_counts", "rotation_counts", "shell_ep")
                    if args.sampling == "event":
                        fields += ("waiting_times", "event_types", "medium_ep", "start_times")
                    for field in fields:
                        np.testing.assert_array_equal(getattr(result, field), getattr(reference, field))
                elapsed = float(np.median(timings))
                recorded_events = int(result.hop_counts.sum() + result.rotation_counts.sum())
                output_bytes = sum(value.nbytes for value in result.__dict__.values())
                print(json.dumps({
                    "backend": backend, "replicas": count, "workers": workers,
                    "batch_size": args.batch_size if backend == "cuda" else None,
                    "seconds": timings, "median_seconds": elapsed,
                    "replicas_per_second": count / elapsed,
                    "recorded_events": recorded_events,
                    "recorded_events_per_second": recorded_events / elapsed,
                    "output_mib": output_bytes / 2**20,
                }), flush=True)
                del result
            del reference


if __name__ == "__main__":
    main()
