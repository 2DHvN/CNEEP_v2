"""Exact transition-indexed ensembles for event-pair NEEP training."""
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

import numpy as np

from .core import LABPMConfig, _integer, simulate_ensemble, simulation_backend
from ._gillespie import _simulate_single


@dataclass(frozen=True)
class LABPMEventResult:
    """States [M,E+1,L,L]; each adjacent pair is exactly one allowed event.

    times [M,E+1] are replica-specific, relative to event collection start;
    waiting_times [M,E] contain actual complete holding times. sources/targets
    [M,E,2] store sparse EP endpoints, and event_types [M,E] use indices 0..5.
    Rotations have equal source/target and zero medium EP. Medium entropy and
    shell entropy are event labels for diagnostics, not supervised targets.
    """
    states: np.ndarray
    times: np.ndarray
    waiting_times: np.ndarray
    escape_rates: np.ndarray
    event_types: np.ndarray
    sources: np.ndarray
    targets: np.ndarray
    medium_ep: np.ndarray
    shell_ep: np.ndarray
    seeds: np.ndarray
    burn_seeds: np.ndarray
    start_times: np.ndarray

    @property
    def hop_counts(self):
        return (self.event_types < 4).astype(np.int8)

    @property
    def rotation_counts(self):
        return (self.event_types >= 4).astype(np.int8)

    @property
    def medium_ep_maps(self):
        """Materialize dense maps on demand (normally only for held-out data)."""
        count, events = self.medium_ep.shape
        size = self.states.shape[-1]
        maps = np.zeros((count, events, size, size), dtype=np.float64)
        replica, event = np.indices((count, events))
        for endpoint in (self.sources, self.targets):
            np.add.at(maps, (replica, event, endpoint[..., 0], endpoint[..., 1]), self.medium_ep * .5)
        return maps


def simulate_event_ensemble(config, n_trajectories, n_events, burn_time=0.0,
                            seed=0, workers=None, initial_states=None, *,
                            backend="cpu", burn_backend=None, batch_size=64,
                            event_warmup=128, progress=False):
    """Collect fixed event counts with variable physical holding times.

    backend selects CPU replica threads or CUDA replica blocks (or auto).
    Physical burn-in uses the same backend unless burn_backend is supplied.
    CUDA retains event loops, waiting times, and sparse entropy on the device;
    batch_size bounds simultaneous replicas and output memory. RNG paths are
    independent of CPU worker count / CUDA batch size within each backend,
    but CPU and CUDA use different RNGs and need not produce matching paths.
    At least one initial event is discarded to start at an event boundary;
    additional event_warmup transitions help relax the embedded jump chain.
    This does not by itself guarantee stationarity of a slowly mixing system.
    Absorbing chains raise instead of generating fabricated no-change pairs.
    Loss samples should be uniform in events. Convert sums to rates using
    total elapsed holding time, never by averaging score/waiting_time.
    """
    if not isinstance(config, LABPMConfig):
        raise TypeError("config must be LABPMConfig")
    count = _integer("n_trajectories", n_trajectories, 1)
    events = _integer("n_events", n_events, 1)
    warmup = _integer("event_warmup", event_warmup, 1)
    seed = _integer("seed", seed, 0)
    workers = min(count, _integer("workers", (os.cpu_count() or 1) if workers is None else workers, 1))
    backend = simulation_backend(backend)
    burn_backend = backend if burn_backend is None else burn_backend
    burned = simulate_ensemble(
        config, count, 1, 1.0, burn_time=burn_time, seed=seed,
        workers=workers, initial_states=initial_states, backend=burn_backend,
        batch_size=batch_size, progress=progress,
    )
    seeds = np.array([child.generate_state(1)[0] for child in
                      np.random.SeedSequence(seed, spawn_key=(1, 0)).spawn(count)], dtype=np.uint32)
    used = set(int(value) for value in burned.seeds)
    for i in range(count):
        value = int(seeds[i])
        while value in used:
            value = (value + 1) % 2**32
        seeds[i] = value
        used.add(value)
    size = config.lattice_size
    physical, betas = config._arrays()
    if backend == "cuda":
        from ._gillespie_cuda import simulate_cuda

        states, trace, shells, starts = simulate_cuda(
            burned.states[:, 0], physical, betas, warmup + events + 1,
            1.0, 0.0, seeds, batch_size=batch_size, progress=progress,
            event_mode=True, event_warmup=warmup,
        )
        starts += float(burn_time)
    else:
        states = np.empty((count, events + 1, size, size), dtype=np.int8)
        trace = np.empty((count, events, 9), dtype=np.float64)
        shells = np.empty((count, events, len(betas) + 2), dtype=np.float64)
        starts = np.empty(count, dtype=np.float64)

        def run(index):
            event_buffer = np.empty((warmup + events, 9), dtype=np.float64)
            output = _simulate_single(burned.states[index, 0], physical, betas,
                                      warmup + events + 1, 1.0, 0.0, int(seeds[index]),
                                      None, event_buffer)
            return output[0][warmup:].copy(), event_buffer, output[2][warmup:].copy()

        def store(index, output):
            snapshot, event_buffer, shell = output
            states[index] = snapshot
            trace[index] = event_buffer[warmup:]
            shells[index] = shell
            starts[index] = float(burn_time) + event_buffer[warmup - 1, 8]

        from tqdm.auto import tqdm
        with tqdm(total=count, desc="LABP-M CPU event replicas", disable=not progress) as bar:
            if workers == 1:
                for i in range(count):
                    store(i, run(i))
                    bar.update(1)
            else:
                # Compile the event signature without serializing a real trajectory.
                _simulate_single(burned.states[0, 0], physical, betas, 1, 1.0, 0.0,
                                 int(seeds[0]), None, np.empty((1, 9)))
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    pending = {pool.submit(run, i): i for i in range(count)}
                    for future in as_completed(pending):
                        store(pending.pop(future), future.result())
                        bar.update(1)
    waits = trace[..., 0].copy()
    times = np.concatenate((np.zeros((count, 1)), np.cumsum(waits, axis=1)), axis=1)
    return LABPMEventResult(
        states=states, times=times, waiting_times=waits,
        escape_rates=trace[..., 1].copy(), event_types=trace[..., 2].astype(np.int8),
        sources=trace[..., 3:5].astype(np.int32), targets=trace[..., 5:7].astype(np.int32),
        medium_ep=trace[..., 7].copy(), shell_ep=shells, seeds=seeds,
        burn_seeds=burned.seeds.copy(), start_times=starts,
    )
