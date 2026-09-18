"""Fused exact Gillespie ensemble simulation on a CUDA device.

One 32-thread block owns one replica, including its clock and RNG. The leader
selects and executes an event; the block refreshes the affected particle rates
in parallel. Tree ancestors are then updated by the leader, avoiding concurrent
writes to shared ancestors. Event loops stay on the device, including burn-in
and fixed-time observations. Bounded launches preserve clocks, pending events,
and partial observation intervals between launches. There is no common ensemble
clock, time step, rejection sampling, or approximation of the event process.

CUDA is imported only when requested. Replica chunks bound device allocations;
the returned NumPy arrays still contain the complete ensemble. CUDA uses an
independent xoroshiro128+ stream for each trajectory seed, so results do not
depend on chunk size. Its seeded paths differ from the CPU MT19937 backend.
"""

from functools import lru_cache
import math
from numbers import Integral

import numpy as np


_THREADS = 32
_EVENTS_PER_LAUNCH = 1024


def cuda_available():
    """Whether Numba can use CUDA (or its explicitly enabled CPU simulator)."""
    try:
        from numba import cuda as runtime
        return bool(runtime.is_available())
    except (ImportError, OSError, RuntimeError):
        return False


def _initial_rng_states(seeds):
    """Numba-compatible SplitMix64 initialization of xoroshiro128+ states.

    Python integer arithmetic makes the intended modulo-2**64 operations
    explicit and avoids overflow warnings in the CUDA simulator.
    """
    states = np.empty((len(seeds), 2), dtype=np.uint64)
    mask = (1 << 64) - 1
    for index, seed in enumerate(seeds):
        value = (int(seed) + 0x9E3779B97F4A7C15) & mask
        value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & mask
        value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & mask
        value ^= value >> 31
        states[index, :] = value
    return states


@lru_cache(maxsize=1)
def _get_kernel():
    # The simulator replaces CUDA references in a kernel's globals, so cuda
    # must be a global name rather than a variable captured by these closures.
    global cuda, _nextafter
    from numba import cuda
    from numba import config
    if config.ENABLE_CUDASIM:
        _nextafter = math.nextafter
    else:
        from numba.cuda.libdevice import nextafter
        _nextafter = nextafter

    @cuda.jit(device=True)
    def uniform(rng, replica):
        s0 = rng[replica, 0]
        s1 = rng[replica, 1]
        # Equivalent to (s0+s1) modulo 2**64, without an overflowing NumPy
        # scalar addition when the exact same kernel runs under CUDASIM.
        low_mask = np.uint64(0x7FFFFFFFFFFFFFFF)
        high_mask = np.uint64(0x8000000000000000)
        result = ((s0 & low_mask) + (s1 & low_mask)) ^ ((s0 ^ s1) & high_mask)
        s1 ^= s0
        rng[replica, 0] = ((s0 << np.uint32(55)) | (s0 >> np.uint32(9))) ^ s1 ^ (s1 << np.uint32(14))
        rng[replica, 1] = (s1 << np.uint32(36)) | (s1 >> np.uint32(28))
        return float(result >> np.uint32(11)) * (1.0 / 9007199254740992.0)

    @cuda.jit(device=True)
    def wait_time(total, rng, replica):
        if total <= 0.0:
            return math.inf
        return -math.log1p(-uniform(rng, replica)) / total

    @cuda.jit(device=True)
    def dr(direction):
        if direction == 0:
            return -1
        if direction == 2:
            return 1
        return 0

    @cuda.jit(device=True)
    def dc(direction):
        if direction == 1:
            return 1
        if direction == 3:
            return -1
        return 0

    @cuda.jit(device=True)
    def event_direction(orientation, event):
        offset = 0
        if event == 1:
            offset = 2
        elif event == 2:
            offset = 3
        elif event == 3:
            offset = 1
        return (orientation + offset) % 4

    @cuda.jit(device=True)
    def particle_rate(replica, particle, event, sites, orientations, occupancy,
                      size, physical, betas):
        if event >= 4:
            return physical[3]
        row = sites[replica, particle, 0]
        col = sites[replica, particle, 1]
        orientation = orientations[replica, particle]
        direction = event_direction(orientation, event)
        rr = (row + dr(direction) + size) % size
        cc = (col + dc(direction) + size) % size
        if occupancy[replica, rr, cc] >= 0:
            return 0.0
        if event == 0:
            exponent = 0.0
            for index in range(betas.size):
                distance = index + 2
                rr = (row + distance * dr(orientation) + size) % size
                cc = (col + distance * dc(orientation) + size) % size
                if occupancy[replica, rr, cc] >= 0:
                    exponent += betas[index]
            return math.exp(math.log(physical[0]) + exponent)
        if event == 1:
            return physical[1]
        return physical[2]

    @cuda.jit(device=True)
    def refresh_rates(replica, particle, sites, orientations, occupancy, size,
                      physical, betas, rates, tree, leaf_count):
        total = 0.0
        for event in range(6):
            rate = particle_rate(replica, particle, event, sites, orientations,
                                 occupancy, size, physical, betas)
            rates[replica, particle, event] = rate
            total += rate
        tree[replica, leaf_count + particle] = total

    @cuda.jit
    def kernel(sites, orientations, occupancy, particle_counts, physical, betas,
               rng, rates, tree, affected, marked, states, ep_maps, shell_ep,
               hops, rotations, sample_dt, burn_time, leaf_count, clocks,
               frame_cursors, partial_frames, initialize, event_budget):
        replica = cuda.blockIdx.x
        lane = cuda.threadIdx.x
        if frame_cursors[replica] >= states.shape[1]:
            return
        size = occupancy.shape[1]
        count = particle_counts[replica]
        radius = betas.size + 1
        # control[0]: another event precedes this observation; [1]: affected N.
        control = cuda.shared.array(2, dtype=np.int32)
        clock = cuda.shared.array(1, dtype=np.float64)
        baseline = math.log(physical[0]) - math.log(physical[1])

        if initialize:
            for index in range(lane, 2 * leaf_count, cuda.blockDim.x):
                tree[replica, index] = 0.0
            for particle in range(lane, count, cuda.blockDim.x):
                marked[replica, particle] = 0
            cuda.syncthreads()
            for particle in range(lane, count, cuda.blockDim.x):
                refresh_rates(replica, particle, sites, orientations, occupancy,
                              size, physical, betas, rates, tree, leaf_count)
            cuda.syncthreads()
            level = leaf_count // 2
            while level > 0:
                for index in range(level + lane, 2 * level, cuda.blockDim.x):
                    tree[replica, index] = tree[replica, 2 * index] + tree[replica, 2 * index + 1]
                cuda.syncthreads()
                level //= 2
            if lane == 0:
                clock[0] = wait_time(tree[replica, 1], rng, replica)
        elif lane == 0:
            clock[0] = clocks[replica]
        cuda.syncthreads()

        events_done = 0
        for frame in range(frame_cursors[replica], states.shape[1]):
            target = burn_time + frame * sample_dt
            interval = frame - 1
            if partial_frames[replica] == 0:
                for index in range(lane, size * size, cuda.blockDim.x):
                    row = index // size
                    col = index % size
                    states[replica, frame, row, col] = -1
                    if interval >= 0:
                        ep_maps[replica, interval, row, col] = 0.0
                if interval >= 0:
                    for index in range(lane, radius + 1, cuda.blockDim.x):
                        shell_ep[replica, interval, index] = 0.0
                    if lane == 0:
                        hops[replica, interval] = 0
                        rotations[replica, interval] = 0
            cuda.syncthreads()

            while True:
                if lane == 0:
                    control[0] = 1 if clock[0] <= target else 0
                cuda.syncthreads()
                if control[0] == 0:
                    break
                if lane == 0:
                    mass = uniform(rng, replica) * tree[replica, 1]
                    if mass >= tree[replica, 1]:
                        mass = _nextafter(tree[replica, 1], 0.0)
                    index = 1
                    while index < leaf_count:
                        index *= 2
                        if mass >= tree[replica, index]:
                            mass -= tree[replica, index]
                            index += 1
                    particle = index - leaf_count
                    total = tree[replica, leaf_count + particle]
                    mass = uniform(rng, replica) * total
                    if mass >= total:
                        mass = _nextafter(total, 0.0)
                    cumulative = 0.0
                    event = 0
                    for candidate in range(6):
                        cumulative += rates[replica, particle, candidate]
                        if mass < cumulative:
                            event = candidate
                            break
                    affected[replica, 0] = particle
                    control[1] = 1
                    if event >= 4:
                        turn = -1 if event == 4 else 1
                        orientations[replica, particle] = (orientations[replica, particle] + turn + 4) % 4
                        if interval >= 0:
                            rotations[replica, interval] += 1
                    else:
                        row = sites[replica, particle, 0]
                        col = sites[replica, particle, 1]
                        orientation = orientations[replica, particle]
                        direction = event_direction(orientation, event)
                        new_row = (row + dr(direction) + size) % size
                        new_col = (col + dc(direction) + size) % size
                        forward = rates[replica, particle, event]
                        if interval >= 0 and event == 0:
                            shell_ep[replica, interval, 0] += baseline
                            for beta_index in range(betas.size):
                                distance = beta_index + 2
                                rr = (row + distance * dr(orientation) + size) % size
                                cc = (col + distance * dc(orientation) + size) % size
                                if occupancy[replica, rr, cc] >= 0:
                                    shell_ep[replica, interval, distance] += betas[beta_index]
                        occupancy[replica, row, col] = -1
                        occupancy[replica, new_row, new_col] = particle
                        sites[replica, particle, 0] = new_row
                        sites[replica, particle, 1] = new_col
                        reverse_event = event ^ 1
                        reverse = particle_rate(replica, particle, reverse_event,
                                                sites, orientations, occupancy,
                                                size, physical, betas)
                        if interval >= 0:
                            entropy = math.log(forward) - math.log(reverse)
                            hops[replica, interval] += 1
                            ep_maps[replica, interval, row, col] += 0.5 * entropy
                            ep_maps[replica, interval, new_row, new_col] += 0.5 * entropy
                            if event == 1:
                                shell_ep[replica, interval, 0] -= baseline
                                for beta_index in range(betas.size):
                                    distance = beta_index + 2
                                    rr = (new_row + distance * dr(orientation) + size) % size
                                    cc = (new_col + distance * dc(orientation) + size) % size
                                    if occupancy[replica, rr, cc] >= 0:
                                        shell_ep[replica, interval, distance] -= betas[beta_index]
                        marked[replica, particle] = 1
                        for endpoint in range(2):
                            center_r = row if endpoint == 0 else new_row
                            center_c = col if endpoint == 0 else new_col
                            for axis in range(4):
                                for distance in range(1, radius + 1):
                                    rr = (center_r + distance * dr(axis) + size) % size
                                    cc = (center_c + distance * dc(axis) + size) % size
                                    other = occupancy[replica, rr, cc]
                                    if other >= 0 and marked[replica, other] == 0:
                                        affected[replica, control[1]] = other
                                        control[1] += 1
                                        marked[replica, other] = 1
                cuda.syncthreads()
                for index in range(lane, control[1], cuda.blockDim.x):
                    particle = affected[replica, index]
                    refresh_rates(replica, particle, sites, orientations,
                                  occupancy, size, physical, betas, rates, tree,
                                  leaf_count)
                cuda.syncthreads()
                if lane == 0:
                    for offset in range(control[1]):
                        particle = affected[replica, offset]
                        index = (leaf_count + particle) // 2
                        while index > 0:
                            tree[replica, index] = tree[replica, 2 * index] + tree[replica, 2 * index + 1]
                            index //= 2
                        marked[replica, particle] = 0
                    # Preserve this pending event across observation frames.
                    clock[0] += wait_time(tree[replica, 1], rng, replica)
                cuda.syncthreads()
                events_done += 1
                if events_done >= event_budget:
                    if lane == 0:
                        clocks[replica] = clock[0]
                        frame_cursors[replica] = frame
                        partial_frames[replica] = 1
                    # Every thread has the same event counter and exits here.
                    # Rates, RNG, and the uncompleted output interval survive.
                    return

            for particle in range(lane, count, cuda.blockDim.x):
                row = sites[replica, particle, 0]
                col = sites[replica, particle, 1]
                states[replica, frame, row, col] = orientations[replica, particle]
            cuda.syncthreads()
            if lane == 0:
                frame_cursors[replica] = frame + 1
                partial_frames[replica] = 0
                clocks[replica] = clock[0]
            cuda.syncthreads()

    return kernel


def _device_bytes_per_replica(size, particles, frames, shells, leaf_count):
    """Working arrays plus output arrays, excluding tiny shared inputs."""
    return (
        size * size * 4 + particles * (2 * 4 + 4 + 6 * 8 + 1)
        + 2 * leaf_count * 8 + (1 + 8 * (shells - 1)) * 4 + 16 + 4 + 8 + 8 + 1
        + frames * size * size
        + (frames - 1) * (size * size * 8 + shells * 8 + 2 * 8)
    )


def _simulate_batch(initial, physical, betas, frames, sample_dt, burn_time, seeds):
    from numba import cuda as runtime

    kernel = _get_kernel()
    batch, size, _ = initial.shape
    counts = np.sum(initial >= 0, axis=(1, 2)).astype(np.int32)
    maximum = int(counts.max())
    leaf_count = 1 << (maximum - 1).bit_length()
    sites = np.zeros((batch, maximum, 2), dtype=np.int32)
    orientations = np.zeros((batch, maximum), dtype=np.int32)
    occupancy = np.full((batch, size, size), -1, dtype=np.int32)
    for replica, count in enumerate(counts):
        rows, cols = np.nonzero(initial[replica] >= 0)
        sites[replica, :count, 0] = rows
        sites[replica, :count, 1] = cols
        orientations[replica, :count] = initial[replica, rows, cols]
        occupancy[replica, rows, cols] = np.arange(count, dtype=np.int32)
    outputs = (
        runtime.device_array((batch, frames, size, size), dtype=np.int8),
        runtime.device_array((batch, frames - 1, size, size), dtype=np.float64),
        runtime.device_array((batch, frames - 1, betas.size + 2), dtype=np.float64),
        runtime.device_array((batch, frames - 1), dtype=np.int64),
        runtime.device_array((batch, frames - 1), dtype=np.int64),
    )
    # Explicit transfers make allocations and lifetimes visible; the kernel
    # never falls back to implicit host-array copies at launch or per event.
    device_sites = runtime.to_device(sites)
    device_orientations = runtime.to_device(orientations)
    device_occupancy = runtime.to_device(occupancy)
    device_counts = runtime.to_device(counts)
    device_physical = runtime.to_device(physical)
    device_betas = runtime.to_device(betas)
    device_rng = runtime.to_device(_initial_rng_states(seeds))
    device_rates = runtime.device_array((batch, maximum, 6), dtype=np.float64)
    device_tree = runtime.device_array((batch, 2 * leaf_count), dtype=np.float64)
    device_affected = runtime.device_array((batch, 1 + 8 * (betas.size + 1)), dtype=np.int32)
    device_marked = runtime.device_array((batch, maximum), dtype=np.int8)
    device_clocks = runtime.device_array(batch, dtype=np.float64)
    device_cursors = runtime.to_device(np.zeros(batch, dtype=np.int64))
    device_partial = runtime.to_device(np.zeros(batch, dtype=np.int8))
    initialize = True
    while True:
        kernel[batch, _THREADS](
            device_sites, device_orientations, device_occupancy, device_counts,
            device_physical, device_betas, device_rng, device_rates, device_tree,
            device_affected, device_marked, *outputs, sample_dt, burn_time,
            leaf_count, device_clocks, device_cursors, device_partial, initialize,
            _EVENTS_PER_LAUNCH,
        )
        # One small progress transfer per event batch, never per event. Chunked
        # launches avoid monopolizing a display GPU for the whole simulation.
        if np.all(device_cursors.copy_to_host() >= frames):
            break
        initialize = False
    return tuple(array.copy_to_host() for array in outputs)


def simulate_cuda(initial_states, physical, betas, n_frames, sample_dt, burn_time,
                  seeds, *, batch_size=64, progress=False):
    """Internal CUDA counterpart to repeated ``_simulate_single`` calls.

    Inputs are validated by the public ``simulate_ensemble`` API. ``batch_size``
    is an upper bound on simultaneous replicas, automatically reduced when
    device memory information is available. A replica's RNG state depends only
    on its seed, not its position within a chunk. All replicas retain their own
    Gillespie clocks and use the same fixed physical observation schedule.
    """
    if isinstance(batch_size, (bool, np.bool_)) or not isinstance(batch_size, Integral) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    if not cuda_available():
        raise RuntimeError("CUDA backend requires an available Numba CUDA device; use backend='cpu' on this machine")
    from numba import cuda as runtime
    from numba import config
    from tqdm.auto import tqdm

    count, size, _ = initial_states.shape
    batch_size = min(int(batch_size), count)
    shells = len(betas) + 2
    if not config.ENABLE_CUDASIM:
        maximum = int(np.max(np.sum(initial_states >= 0, axis=(1, 2))))
        leaves = 1 << (maximum - 1).bit_length()
        per_replica = _device_bytes_per_replica(size, maximum, n_frames, shells, leaves)
        free_bytes, _ = runtime.current_context().get_memory_info()
        # Reserve room for compilation/runtime allocations and other consumers.
        budget = int(free_bytes * 0.7)
        if per_replica > budget:
            raise MemoryError(
                "One CUDA replica's working/output arrays exceed 70% of free "
                "device memory; reduce n_frames/lattice_size or use backend='cpu'"
            )
        batch_size = min(batch_size, max(1, budget // per_replica))
    outputs = (
        np.empty((count, n_frames, size, size), dtype=np.int8),
        np.empty((count, n_frames - 1, size, size), dtype=np.float64),
        np.empty((count, n_frames - 1, shells), dtype=np.float64),
        np.empty((count, n_frames - 1), dtype=np.int64),
        np.empty((count, n_frames - 1), dtype=np.int64),
    )
    with tqdm(total=count, desc="LABP-M CUDA Gillespie replicas", disable=not progress) as bar:
        for start in range(0, count, batch_size):
            stop = min(start + batch_size, count)
            result = _simulate_batch(
                initial_states[start:stop], physical, betas, n_frames, sample_dt,
                burn_time, seeds[start:stop],
            )
            for output, chunk in zip(outputs, result):
                output[start:stop] = chunk
            bar.update(stop - start)
    return outputs
