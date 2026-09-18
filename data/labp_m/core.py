"""Four-direction modified lattice ABP, simulated by exact Gillespie SSA.

For an empty forward destination, v_plus = forward_rate * exp(sum beta_k n_k).
Here n_k is occupancy k sites along the particle's orientation, k=2,...,R.
All forward sites contribute independently; there is no nearest-neighbor search
or occlusion. Backward/lateral rates are constant, and each 90-degree turn has
rate rotation_rate. Occupied destinations are forbidden. Positions and fields
use (row, column), with orientations 0=up, 1=right, 2=down, 3=left.

The reported medium EP is the exact sum of model log forward/reverse rates,
with orientations even under time reversal. It is not a claim of physical heat
or of equality with irreversibility inferred from temporally coarse frames.
"""

from __future__ import annotations

import math
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from numbers import Integral

import numpy as np

from ._gillespie import DR, DC, RELATIVE_DIRECTIONS, _particle_rates, _simulate_single


def _integer(name, value, minimum):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


def _scalar(name, value, minimum=0.0, strict=False):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite real number")
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite real number") from exc
    if not math.isfinite(value) or value < minimum or (strict and value == minimum):
        comparison = ">" if strict else ">="
        raise ValueError(f"{name} must be finite and {comparison} {minimum}")
    return value


@dataclass(frozen=True)
class LABPMConfig:
    """Rates have units 1/time; shell_weights are dimensionless log-rate boosts.

    shell_weights[j] acts at Chebyshev distance j+2 along the forward ray.
    Increasing weights implement a larger boost from more distant particles.
    An empty tuple gives the ordinary persistent exclusion model. The cutoff
    must be smaller than half the periodic side, preventing wrapped duplicate
    neighbors. rotation_rate is the rate of EACH of the two 90-degree turns;
    the free orientation autocorrelation decays as exp(-2*rotation_rate*t).
    """

    lattice_size: int = 24
    density: float = 0.35
    forward_rate: float = 2.0
    backward_rate: float = 0.2
    lateral_rate: float = 0.2
    rotation_rate: float = 0.1
    shell_weights: tuple[float, ...] = (0.15, 0.30, 0.45)

    def __post_init__(self):
        object.__setattr__(self, "lattice_size", _integer("lattice_size", self.lattice_size, 3))
        for name in ("density", "forward_rate", "backward_rate", "lateral_rate", "rotation_rate"):
            value = _scalar(name, getattr(self, name), strict=name in {
                "density", "forward_rate", "backward_rate"
            })
            object.__setattr__(self, name, value)
        if self.density > 1.0:
            raise ValueError("density must lie in (0, 1]")
        try:
            weights = tuple(_scalar("shell weight", x) for x in self.shell_weights)
        except TypeError as exc:
            raise ValueError("shell_weights must be an iterable of nonnegative numbers") from exc
        object.__setattr__(self, "shell_weights", weights)
        if 2 * self.interaction_radius >= self.lattice_size:
            raise ValueError("interaction radius must be smaller than half lattice_size")
        if self.n_particles < 1:
            raise ValueError("density is too small to place a particle")
        log_max = math.log(self.forward_rate) + sum(weights)
        try:
            maximum = math.exp(log_max) + self.backward_rate + 2 * self.lateral_rate + 2 * self.rotation_rate
        except OverflowError as exc:
            raise ValueError("maximum hopping rate must be finite") from exc
        if not math.isfinite(maximum * self.lattice_size**2):
            raise ValueError("maximum total event rate must be finite")

    @property
    def n_particles(self):
        return int(math.floor(self.density * self.lattice_size**2 + 0.5))

    @property
    def interaction_radius(self):
        return len(self.shell_weights) + 1

    def _arrays(self):
        return (
            np.array([self.forward_rate, self.backward_rate, self.lateral_rate,
                      self.rotation_rate], dtype=np.float64),
            np.asarray(self.shell_weights, dtype=np.float64),
        )


@dataclass(frozen=True)
class LABPMResult:
    """Fixed-time ensemble, all arrays on CPU.

    states: int8 [M,T,L,L], -1 empty, 0..3 orientation. times: [T], relative
    to burn-in. medium_ep: [M,T-1], event-path EP in each saved interval.
    medium_ep_maps: [M,T-1,L,L], half the hop EP on each endpoint.
    shell_ep: [M,T-1,R+1], baseline log(v_plus/v_minus) in slot 0; the
    signed beta_k terms in slots 2..R. This microscopic decomposition is NOT
    uniquely identified with the learned spatial shell spectrum.
    hop_counts, rotation_counts: int64 [M,T-1]. seeds: uint32 [M].
    Frame zero is the state at burn_time; all burn-in EP is discarded.
    """

    states: np.ndarray
    times: np.ndarray
    medium_ep: np.ndarray
    medium_ep_maps: np.ndarray
    shell_ep: np.ndarray
    hop_counts: np.ndarray
    rotation_counts: np.ndarray
    seeds: np.ndarray


def _validated_particles(config, sites, orientations):
    if not isinstance(config, LABPMConfig):
        raise TypeError("config must be LABPMConfig")
    sites = np.asarray(sites)
    orientations = np.asarray(orientations)
    if sites.ndim != 2 or sites.shape[1] != 2 or sites.shape[0] == 0:
        raise ValueError("sites must have shape [N,2], N >= 1")
    if orientations.shape != (sites.shape[0],):
        raise ValueError("orientations must have shape [N]")
    if not np.issubdtype(sites.dtype, np.integer) or not np.issubdtype(orientations.dtype, np.integer):
        raise ValueError("sites and orientations must contain integers")
    size = config.lattice_size
    if np.any(sites < 0) or np.any(sites >= size):
        raise ValueError("sites must lie within the lattice")
    if np.any(orientations < 0) or np.any(orientations > 3):
        raise ValueError("orientations must be in {0,1,2,3}")
    if len(np.unique(sites[:, 0] * size + sites[:, 1])) != len(sites):
        raise ValueError("hard-core sites must be distinct")
    sites = np.array(sites, dtype=np.int64, order="C", copy=True)
    orientations = np.array(orientations, dtype=np.int64, order="C", copy=True)
    occupancy = np.full((size, size), -1, dtype=np.int64)
    occupancy[sites[:, 0], sites[:, 1]] = np.arange(len(sites))
    return sites, orientations, occupancy


def event_rates(config, sites, orientations):
    """Return [N,6] rates: forward, backward, left, right, turn left, turn right."""
    sites, orientations, occupancy = _validated_particles(config, sites, orientations)
    physical, betas = config._arrays()
    return np.stack([
        _particle_rates(sites, orientations, occupancy, i, config.lattice_size, physical, betas)
        for i in range(len(sites))
    ])


def event_medium_ep(config, sites, orientations, particle, event):
    """Return (log rate ratio, shell vector) for one allowed event; no mutation."""
    sites, orientations, occupancy = _validated_particles(config, sites, orientations)
    particle = _integer("particle", particle, 0)
    event = _integer("event", event, 0)
    if particle >= len(sites) or event >= 6:
        raise ValueError("particle or event index is out of range")
    physical, betas = config._arrays()
    before = _particle_rates(sites, orientations, occupancy, particle,
                             config.lattice_size, physical, betas)[event]
    if before == 0:
        raise ValueError("event is forbidden (zero rate)")
    pieces = np.zeros(config.interaction_radius + 1, dtype=np.float64)
    if event >= 4:
        return 0.0, pieces
    row, col = sites[particle]
    orientation = orientations[particle]
    direction = (orientation + RELATIVE_DIRECTIONS[event]) % 4
    target = (sites[particle] + np.array([DR[direction], DC[direction]])) % config.lattice_size
    if event == 0:
        pieces[0] = math.log(config.forward_rate) - math.log(config.backward_rate)
        for index, beta in enumerate(betas):
            k = index + 2
            rr = (row + k * DR[orientation]) % config.lattice_size
            cc = (col + k * DC[orientation]) % config.lattice_size
            pieces[k] = beta * (occupancy[rr, cc] >= 0)
    occupancy[row, col] = -1
    occupancy[tuple(target)] = particle
    sites[particle] = target
    reverse = _particle_rates(sites, orientations, occupancy, particle,
                              config.lattice_size, physical, betas)[(1, 0, 3, 2)[event]]
    if event == 1:
        pieces[0] = math.log(config.backward_rate) - math.log(config.forward_rate)
        for index, beta in enumerate(betas):
            k = index + 2
            rr = (target[0] + k * DR[orientation]) % config.lattice_size
            cc = (target[1] + k * DC[orientation]) % config.lattice_size
            pieces[k] = -beta * (occupancy[rr, cc] >= 0)
    return math.log(before) - math.log(reverse), pieces


def simulation_backend(backend="auto"):
    """Resolve the simulation device independently of the training device.

    ``auto`` chooses CUDA when Numba can access it, otherwise CPU. An explicit
    CUDA request raises if unavailable; execution failures never silently fall
    back to another backend. CPU-only use does not import the CUDA module.
    """
    if backend not in ("cpu", "cuda", "auto"):
        raise ValueError("backend must be 'cpu', 'cuda', or 'auto'")
    if backend == "cpu":
        return "cpu"
    from ._gillespie_cuda import cuda_available

    if cuda_available():
        return "cuda"
    if backend == "cuda":
        raise RuntimeError(
            "CUDA simulation requested but Numba CUDA is unavailable. "
            "Use backend='cpu' or install a compatible CUDA driver/toolkit."
        )
    return "cpu"


def simulate_ensemble(config, n_trajectories, n_frames, sample_dt, burn_time=0.0,
                      seed=0, workers=None, progress=False, initial_states=None,
                      *, backend="cpu", batch_size=64):
    """Simulate independent replicas with exact Gillespie event times.

    workers controls CPU threads across replicas; None uses available CPUs,
    capped by the replica count. Numba releases the GIL. backend is 'cpu',
    'cuda', or 'auto'; batch_size bounds concurrent CUDA replicas/device
    buffers. Each replica retains its own clock and RNG stream. Changing CPU
    workers or CUDA batch_size does not change that backend's seeded paths.
    CPU and CUDA use different RNGs, so paths need not match across backends.
    sample_dt only selects observation times, not an
    integration step. A pending event is retained across frame boundaries.
    Optional initial_states [M,L,L] overrides density-derived initialization;
    each replica must contain at least one particle, with values -1..3.
    """
    if not isinstance(config, LABPMConfig):
        raise TypeError("config must be LABPMConfig")
    count = _integer("n_trajectories", n_trajectories, 1)
    frames = _integer("n_frames", n_frames, 1)
    workers = min(_integer("workers", (os.cpu_count() or 1) if workers is None else workers, 1), count)
    batch_size = _integer("batch_size", batch_size, 1)
    backend = simulation_backend(backend)
    seed = _integer("seed", seed, 0)
    sample_dt = _scalar("sample_dt", sample_dt, strict=True)
    burn_time = _scalar("burn_time", burn_time)
    end = burn_time + (frames - 1) * sample_dt
    if not math.isfinite(end) or (frames > 1 and burn_time + sample_dt == burn_time):
        raise ValueError("observation times must be finite and distinguishable")
    size = config.lattice_size
    if initial_states is not None:
        initial_states = np.asarray(initial_states)
        if initial_states.shape != (count, size, size):
            raise ValueError("initial_states must have shape [M,L,L]")
        if not np.issubdtype(initial_states.dtype, np.integer):
            raise ValueError("initial_states must contain integers")
        if np.any(initial_states < -1) or np.any(initial_states > 3):
            raise ValueError("initial_states must contain only -1..3")
        if np.any(np.sum(initial_states >= 0, axis=(1, 2)) == 0):
            raise ValueError("each initial state must contain at least one particle")
        initial_states = np.array(initial_states, dtype=np.int8, order="C", copy=True)
    physical, betas = config._arrays()
    children = np.random.SeedSequence(seed).spawn(count)
    seeds = np.array([child.generate_state(1)[0] for child in children], dtype=np.uint32)
    # Deterministically avoid even a rare collision in the compiled RNG seeds.
    used = set()
    for i in range(count):
        value = int(seeds[i])
        while value in used:
            value = (value + 1) % (2**32)
        seeds[i] = value
        used.add(value)

    def initial(index):
        if initial_states is not None:
            return initial_states[index]
        rng = np.random.default_rng(children[index])
        chosen = rng.choice(size * size, config.n_particles, replace=False)
        state = np.full(size * size, -1, dtype=np.int8)
        state[chosen] = rng.integers(0, 4, config.n_particles, dtype=np.int8)
        return state.reshape(size, size)

    def run(index):
        return _simulate_single(initial(index), physical, betas, frames,
                                sample_dt, burn_time, int(seeds[index]))

    def pack(states, ep_maps, shell_ep, hops, rotations):
        return LABPMResult(
            states=states, times=np.arange(frames, dtype=np.float64) * sample_dt,
            medium_ep=ep_maps.sum(axis=(-2, -1)), medium_ep_maps=ep_maps,
            shell_ep=shell_ep, hop_counts=hops, rotation_counts=rotations, seeds=seeds,
        )

    if backend == "cuda":
        from ._gillespie_cuda import simulate_cuda

        initial_batch = np.stack([initial(i) for i in range(count)])
        return pack(*simulate_cuda(
            initial_batch, physical, betas, frames, sample_dt, burn_time, seeds,
            batch_size=batch_size, progress=progress,
        ))

    states = np.empty((count, frames, size, size), dtype=np.int8)
    ep_maps = np.empty((count, frames - 1, size, size), dtype=np.float64)
    shell_ep = np.empty((count, frames - 1, config.interaction_radius + 1), dtype=np.float64)
    hops = np.empty((count, frames - 1), dtype=np.int64)
    rotations = np.empty_like(hops)

    def store(index, result):
        states[index], ep_maps[index], shell_ep[index], hops[index], rotations[index] = result

    from tqdm.auto import tqdm
    with tqdm(total=count, desc="LABP-M Gillespie replicas", disable=not progress) as bar:
        if workers == 1:
            for i in range(count):
                store(i, run(i))
                bar.update(1)
        else:
            # Prepare the compiled signature without serializing a full path.
            # Real trajectories (including zero) all start in the worker pool.
            _simulate_single(initial(0), physical, betas, 1, sample_dt, 0.0, int(seeds[0]))
            with ThreadPoolExecutor(max_workers=workers) as pool:
                pending = {pool.submit(run, i): i for i in range(count)}
                for future in as_completed(pending):
                    store(pending.pop(future), future.result())
                    bar.update(1)
    return pack(states, ep_maps, shell_ep, hops, rotations)


def encode_observations(result_or_states):
    """Encode four directional occupancy fields as CPU float32 [M,T,4,L,L]."""
    import torch

    states = result_or_states.states if isinstance(result_or_states, LABPMResult) else np.asarray(result_or_states)
    if states.ndim != 4 or states.shape[-2] != states.shape[-1]:
        raise ValueError("states must have shape [M,T,L,L]")
    if not np.issubdtype(states.dtype, np.integer) or np.any(states < -1) or np.any(states > 3):
        raise ValueError("states must contain integer values -1..3")
    return torch.from_numpy(np.stack([states == q for q in range(4)], axis=2).astype(np.float32))
