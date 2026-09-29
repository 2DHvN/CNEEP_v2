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
def _sensing_contributions(sites, orientations, occupancy, particle, size, physical, betas):
    """Saturated full-shell responses, indexed by k - 2 (never ray sensing)."""
    row, col = sites[particle]
    orientation = orientations[particle]
    angular_sensing = physical[6] != 0.0
    clockwise = (orientation + 1) % 4
    counterclockwise = (orientation - 1) % 4
    plus_weight = (1.0 + physical[5]) * 0.5
    minus_weight = (1.0 - physical[5]) * 0.5
    values = np.zeros(len(betas), dtype=np.float64)
    for index in range(len(betas)):
        if betas[index] == 0.0:
            continue
        distance = index + 2
        count = 0
        count_plus = 0
        count_minus = 0
        # Top/bottom contain the corners; left/right exclude the corners.
        for side in range(2):
            dr = -distance if side == 0 else distance
            for dc in range(-distance, distance + 1):
                other = occupancy[(row + dr) % size, (col + dc) % size]
                if other >= 0:
                    if angular_sensing:
                        count_plus += orientations[other] == clockwise
                        count_minus += orientations[other] == counterclockwise
                    else:
                        count += 1
        for side in range(2):
            dc = -distance if side == 0 else distance
            for dr in range(-distance + 1, distance):
                other = occupancy[(row + dr) % size, (col + dc) % size]
                if other >= 0:
                    if angular_sensing:
                        count_plus += orientations[other] == clockwise
                        count_minus += orientations[other] == counterclockwise
                    else:
                        count += 1
        sensed = plus_weight * count_plus + minus_weight * count_minus if angular_sensing else count
        q = sensed / physical[4]
        values[index] = betas[index] * (-math.expm1(-q))
    return values


@njit(cache=True, nogil=True)
def _boosted_forward_rate(physical, exponent):
    return math.exp(math.log(physical[0]) + exponent)


@njit(cache=True, nogil=True)
def _shell_allocation(sites, orientations, occupancy, particle, size, physical, betas, sign):
    """Exact log-rate allocation: baseline in k=1 and signed G_k in k>=2."""
    values = _sensing_contributions(sites, orientations, occupancy, particle,
                                   size, physical, betas)
    result = np.zeros(len(betas) + 2, dtype=np.float64)
    result[1] = sign * (math.log(physical[0]) - math.log(physical[1]))
    for index in range(len(betas)):
        result[index + 2] = sign * values[index]
    return result


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
            values = _sensing_contributions(sites, orientations, occupancy,
                                           particle, size, physical, betas)
            rates[event] = _boosted_forward_rate(physical, values.sum())
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
                     waiting_trace=None, event_trace=None):
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
    # Event sampling stores sparse endpoints instead of a mostly-zero dense map.
    ep_maps = np.zeros((0 if event_trace is not None else n_frames - 1, size, size), dtype=np.float64)
    shell_ep = np.zeros((n_frames - 1, radius + 1), dtype=np.float64)
    hops = np.zeros(n_frames - 1, dtype=np.int64)
    rotations = np.zeros(n_frames - 1, dtype=np.int64)
    affected = np.empty(n_particles, dtype=np.int64)
    marked = np.zeros(n_particles, dtype=np.bool_)
    contribution = np.zeros(radius + 1, dtype=np.float64)
    pending_wait = _wait(tree[1])
    next_event_time = pending_wait
    trace_count = 0

    for frame in range(n_frames):
        target = burn_time + frame * sample_dt
        interval = frame - 1
        if event_trace is not None and interval >= 0:
            target = np.inf
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

            if event_trace is not None and interval >= 0:
                # tau, pre-jump escape rate, event type, source row/col,
                # target row/col, medium entropy, absolute event time.
                event_trace[interval, 0] = pending_wait
                event_trace[interval, 1] = tree[1]
                event_trace[interval, 2] = event
                event_trace[interval, 3] = sites[particle, 0]
                event_trace[interval, 4] = sites[particle, 1]
                event_trace[interval, 7] = 0.0
                event_trace[interval, 8] = event_time

            if waiting_trace is not None and trace_count < len(waiting_trace):
                # Record the rate that generated this waiting time, before
                # changing the state. Diagnostic callers use a single frame
                # and stop after filling this bounded event buffer.
                waiting_trace[trace_count, 0] = pending_wait
                waiting_trace[trace_count, 1] = tree[1]
                waiting_trace[trace_count, 2] = event
                trace_count += 1

            row, col = sites[particle]
            new_row, new_col = row, col
            if event >= 4:
                turn = -1 if event == 4 else 1
                orientations[particle] = (orientations[particle] + turn) % 4
                if interval >= 0:
                    rotations[interval] += 1
            else:
                orientation = orientations[particle]
                direction = (orientation + RELATIVE_DIRECTIONS[event]) % 4
                new_row = (row + DR[direction]) % size
                new_col = (col + DC[direction]) % size
                forward_rate = rates[particle, event]
                contribution[:] = 0.0
                if event == 0 and interval >= 0:
                    contribution = _shell_allocation(sites, orientations, occupancy,
                                                     particle, size, physical, betas, 1.0)

                occupancy[row, col] = -1
                occupancy[new_row, new_col] = particle
                sites[particle, 0] = new_row
                sites[particle, 1] = new_col
                if interval >= 0:
                    reverse_event = (1, 0, 3, 2)[event]
                    new_rates = _particle_rates(
                        sites, orientations, occupancy, particle, size, physical, betas
                    )
                    entropy = math.log(forward_rate) - math.log(new_rates[reverse_event])
                    if event == 1:
                        contribution = _shell_allocation(sites, orientations, occupancy,
                                                         particle, size, physical, betas, -1.0)
                    hops[interval] += 1
                    # Symmetric endpoint gauge negates under hop reversal.
                    if event_trace is None:
                        ep_maps[interval, row, col] += 0.5 * entropy
                        ep_maps[interval, new_row, new_col] += 0.5 * entropy
                    else:
                        event_trace[interval, 7] = entropy
                    shell_ep[interval] += contribution

            # A turn changes the sensed heading for every nearby particle;
            # a hop changes sensing and exclusions around both endpoints.
            # Refresh complete Chebyshev neighborhoods, including the mover.
            affected[0] = particle
            marked[particle] = True
            count = 1
            for endpoint in range(1 if event >= 4 else 2):
                center_r = row if endpoint == 0 else new_row
                center_c = col if endpoint == 0 else new_col
                for dr in range(-radius, radius + 1):
                    for dc in range(-radius, radius + 1):
                        other = occupancy[(center_r + dr) % size, (center_c + dc) % size]
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
            if event_trace is not None and interval >= 0:
                event_trace[interval, 5] = sites[particle, 0]
                event_trace[interval, 6] = sites[particle, 1]
                break  # Exactly one transition between each saved state pair.
            if waiting_trace is not None and n_frames == 1 and trace_count == len(waiting_trace):
                break

        if event_trace is not None and interval >= 0 and hops[interval] + rotations[interval] == 0:
            raise ValueError("The chain absorbed before the requested event count; event pairs cannot be padded.")

        for particle in range(n_particles):
            row, col = sites[particle]
            states[frame, row, col] = orientations[particle]
    return states, ep_maps, shell_ep, hops, rotations
