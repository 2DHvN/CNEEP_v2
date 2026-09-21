"""Compiled exact direct SSA with a particle-rate segment tree.

There are no sweeps, fixed integration steps, rejected proposals, or tau leaps.
Only rate dependencies of an executed event are refreshed. Rates, waiting
times, and entropy increments use float64.
"""

import math

import numpy as np
from numba import njit


# Array axes are (row, column); orientations run clockwise from up.
DR = np.array([-1, 0, 1, 0], dtype=np.int64)
DC = np.array([0, 1, 0, -1], dtype=np.int64)
RELATIVE_DIRECTIONS = np.array([0, 2, 3, 1], dtype=np.int64)


@njit(cache=True, nogil=True)
def _particle_rates(sites, orientations, occupancy, particle, size, physical, betas):
    """Events: forward, backward, left, right, turn left, turn right."""
    row, col = sites[particle]
    orientation = orientations[particle]
    rates = np.empty(6, dtype=np.float64)
    for event in range(4):
        direction = (orientation + RELATIVE_DIRECTIONS[event]) % 4
        rr = (row + DR[direction]) % size
        cc = (col + DC[direction]) % size
        if occupancy[rr, cc] >= 0:
            rates[event] = 0.0
        elif event == 0:
            exponent = 0.0
            for index in range(len(betas)):
                distance = index + 2
                rr = (row + distance * DR[orientation]) % size
                cc = (col + distance * DC[orientation]) % size
                if occupancy[rr, cc] >= 0:
                    exponent += betas[index]
            rates[event] = math.exp(math.log(physical[0]) + exponent)
        elif event == 1:
            rates[event] = physical[1]
        else:
            rates[event] = physical[2]
    rates[4] = physical[3]
    rates[5] = physical[3]
    return rates


@njit(cache=True, nogil=True)
def _tree_set(tree, leaf_count, particle, value):
    index = leaf_count + particle
    tree[index] = value
    index //= 2
    while index:
        tree[index] = tree[2 * index] + tree[2 * index + 1]
        index //= 2


@njit(cache=True, nogil=True)
def _tree_sample(tree, leaf_count, mass):
    index = 1
    while index < leaf_count:
        index *= 2
        if mass >= tree[index]:
            mass -= tree[index]
            index += 1
    return index - leaf_count


@njit(cache=True, nogil=True)
def _refresh(particle, sites, orientations, occupancy, size, physical, betas,
             rates, tree, leaf_count):
    rates[particle] = _particle_rates(
        sites, orientations, occupancy, particle, size, physical, betas
    )
    _tree_set(tree, leaf_count, particle, rates[particle].sum())


@njit(cache=True, nogil=True)
def _wait(total):
    if total <= 0.0:
        return np.inf
    return -math.log1p(-np.random.random()) / total


@njit(cache=True, nogil=True)
def _simulate_single(initial, physical, betas, n_frames, sample_dt, burn_time, seed,
                     waiting_trace=None):
    np.random.seed(seed)
    size = initial.shape[0]
    n_particles = int(np.sum(initial >= 0))
    sites = np.empty((n_particles, 2), dtype=np.int64)
    orientations = np.empty(n_particles, dtype=np.int64)
    occupancy = np.full((size, size), -1, dtype=np.int64)
    particle = 0
    for row in range(size):
        for col in range(size):
            if initial[row, col] >= 0:
                sites[particle, 0] = row
                sites[particle, 1] = col
                orientations[particle] = initial[row, col]
                occupancy[row, col] = particle
                particle += 1

    leaf_count = 1
    while leaf_count < n_particles:
        leaf_count *= 2
    tree = np.zeros(2 * leaf_count, dtype=np.float64)
    rates = np.empty((n_particles, 6), dtype=np.float64)
    for particle in range(n_particles):
        _refresh(particle, sites, orientations, occupancy, size, physical,
                 betas, rates, tree, leaf_count)

    radius = len(betas) + 1
    states = np.full((n_frames, size, size), -1, dtype=np.int8)
    ep_maps = np.zeros((n_frames - 1, size, size), dtype=np.float64)
    shell_ep = np.zeros((n_frames - 1, radius + 1), dtype=np.float64)
    hops = np.zeros(n_frames - 1, dtype=np.int64)
    rotations = np.zeros(n_frames - 1, dtype=np.int64)
    affected = np.empty(1 + 8 * radius, dtype=np.int64)
    marked = np.zeros(n_particles, dtype=np.bool_)
    contribution = np.zeros(radius + 1, dtype=np.float64)
    baseline_affinity = math.log(physical[0]) - math.log(physical[1])
    pending_wait = _wait(tree[1])
    next_event_time = pending_wait
    trace_count = 0

    for frame in range(n_frames):
        target = burn_time + frame * sample_dt
        interval = frame - 1
        while next_event_time <= target and math.isfinite(next_event_time):
            event_time = next_event_time
            mass = np.random.random() * tree[1]
            if mass >= tree[1]:
                mass = np.nextafter(tree[1], 0.0)
            particle = _tree_sample(tree, leaf_count, mass)
            particle_total = rates[particle].sum()
            mass = np.random.random() * particle_total
            if mass >= particle_total:
                mass = np.nextafter(particle_total, 0.0)
            cumulative = 0.0
            event = 0
            for candidate in range(6):
                cumulative += rates[particle, candidate]
                if mass < cumulative:
                    event = candidate
                    break

            if waiting_trace is not None and trace_count < len(waiting_trace):
                # Record the rate that generated this waiting time, before
                # changing the state. Diagnostic callers use a single frame
                # and stop after filling this bounded event buffer.
                waiting_trace[trace_count, 0] = pending_wait
                waiting_trace[trace_count, 1] = tree[1]
                waiting_trace[trace_count, 2] = event
                trace_count += 1

            if event >= 4:
                turn = -1 if event == 4 else 1
                orientations[particle] = (orientations[particle] + turn) % 4
                _refresh(particle, sites, orientations, occupancy, size,
                         physical, betas, rates, tree, leaf_count)
                if interval >= 0:
                    rotations[interval] += 1
            else:
                row, col = sites[particle]
                orientation = orientations[particle]
                direction = (orientation + RELATIVE_DIRECTIONS[event]) % 4
                new_row = (row + DR[direction]) % size
                new_col = (col + DC[direction]) % size
                forward_rate = rates[particle, event]
                contribution[:] = 0.0
                if event == 0:
                    contribution[0] = baseline_affinity
                    for index in range(len(betas)):
                        distance = index + 2
                        rr = (row + distance * DR[orientation]) % size
                        cc = (col + distance * DC[orientation]) % size
                        if occupancy[rr, cc] >= 0:
                            contribution[distance] = betas[index]

                occupancy[row, col] = -1
                occupancy[new_row, new_col] = particle
                sites[particle, 0] = new_row
                sites[particle, 1] = new_col
                reverse_event = (1, 0, 3, 2)[event]
                new_rates = _particle_rates(
                    sites, orientations, occupancy, particle, size, physical, betas
                )
                entropy = math.log(forward_rate) - math.log(new_rates[reverse_event])
                if event == 1:
                    contribution[0] = -baseline_affinity
                    for index in range(len(betas)):
                        distance = index + 2
                        rr = (new_row + distance * DR[orientation]) % size
                        cc = (new_col + distance * DC[orientation]) % size
                        if occupancy[rr, cc] >= 0:
                            contribution[distance] = -betas[index]
                if interval >= 0:
                    hops[interval] += 1
                    # Symmetric endpoint gauge negates under hop reversal.
                    ep_maps[interval, row, col] += 0.5 * entropy
                    ep_maps[interval, new_row, new_col] += 0.5 * entropy
                    shell_ep[interval] += contribution

                affected[0] = particle
                marked[particle] = True
                count = 1
                for endpoint in range(2):
                    center_r = row if endpoint == 0 else new_row
                    center_c = col if endpoint == 0 else new_col
                    for axis in range(4):
                        for distance in range(1, radius + 1):
                            rr = (center_r + distance * DR[axis]) % size
                            cc = (center_c + distance * DC[axis]) % size
                            other = occupancy[rr, cc]
                            if other >= 0 and not marked[other]:
                                affected[count] = other
                                count += 1
                                marked[other] = True
                for index in range(count):
                    other = affected[index]
                    _refresh(other, sites, orientations, occupancy, size,
                             physical, betas, rates, tree, leaf_count)
                    marked[other] = False

            # Keep the next event pending across observation boundaries.
            pending_wait = _wait(tree[1])
            next_event_time = event_time + pending_wait
            if waiting_trace is not None and n_frames == 1 and trace_count == len(waiting_trace):
                break

        for particle in range(n_particles):
            row, col = sites[particle]
            states[frame, row, col] = orientations[particle]
    return states, ep_maps, shell_ep, hops, rotations
