"""Bounded exact-event waiting-time diagnostics from supplied configurations.

This CPU diagnostic reuses the production Gillespie kernel. It does not infer
event times from fixed-time snapshots or change their sampling schedule.
"""

import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .core import LABPMConfig, _integer
from ._gillespie import _simulate_single


def sample_waiting_times(config, initial_states, n_events=20000, seed=47031, workers=None):
    """Return a list of per-replica event traces, continuing supplied states.

    Each trace has waiting_times, escape_rates, event_types and seed fields.
    Event types follow event_rates: forward/back/left/right/turn-left/turn-right.
    These are whole-system inter-event times, including hops and rotations,
    not per-particle residence times. Samples are event-weighted, not uniformly
    sampled physical times. Each replica contributes up to n_events intervals.

    The first event after the supplied snapshot is discarded: its delay is a
    forward recurrence time from an arbitrary observation, not a complete
    interval between two events. Absorbing replicas may return fewer samples
    (including none); infinite, unrealized waits are not plotted as events.
    """
    if not isinstance(config, LABPMConfig):
        raise TypeError("config must be LABPMConfig")
    n_events = _integer("n_events", n_events, 1)
    seed = _integer("seed", seed, 0)
    states = np.asarray(initial_states)
    size = config.lattice_size
    if states.ndim != 3 or states.shape[1:] != (size, size) or len(states) == 0:
        raise ValueError("initial_states must have shape [M,L,L], M >= 1")
    if not np.issubdtype(states.dtype, np.integer) or np.any(states < -1) or np.any(states > 3):
        raise ValueError("initial_states must contain integer values -1..3")
    if np.any(np.sum(states >= 0, axis=(1, 2)) == 0):
        raise ValueError("Each replica must contain at least one particle")
    states = np.array(states, dtype=np.int8, order="C", copy=True)
    workers = min(len(states), _integer("workers", (os.cpu_count() or 1) if workers is None else workers, 1))
    seeds = [int(child.generate_state(1)[0]) for child in np.random.SeedSequence(seed).spawn(len(states))]
    used = set()
    for index, value in enumerate(seeds):
        while value in used:
            value = (value + 1) % 2**32
        seeds[index] = value
        used.add(value)
    physical, betas = config._arrays()

    def run(index):
        trace = np.full((n_events + 1, 3), np.nan, dtype=np.float64)
        # An infinite single-frame target means run until the event buffer is
        # full or the chain absorbs. No large frame/entropy maps are allocated.
        _simulate_single(states[index], physical, betas, 1, 1.0, np.inf, seeds[index], trace)
        complete = trace[1:]
        complete = complete[np.isfinite(complete[:, 0])]
        return {"waiting_times": complete[:, 0].copy(),
                "escape_rates": complete[:, 1].copy(),
                "event_types": complete[:, 2].astype(np.int8),
                "seed": seeds[index]}

    if workers == 1:
        return [run(i) for i in range(len(states))]
    _simulate_single(states[0], physical, betas, 1, 1.0, 0.0, seeds[0], np.empty((1, 3)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(run, range(len(states))))
