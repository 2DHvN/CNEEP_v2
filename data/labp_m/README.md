# LABP-M: modified lattice active particles

Four discrete orientations, hard-core exclusion, periodic boundaries, and an
exact Gillespie direct method. Independent replicas run in parallel CPU threads
using a Numba kernel; each replica executes one event at a time. Training can
run on CUDA or CPU. No GPU simulation, tau-leaping, sweeps, or fixed integration
time step is used.

## Interaction and events

Orientations are `0=up, 1=right, 2=down, 3=left`; arrays use `(row, column)`.
For an empty forward destination,

```text
v_plus(i) = forward_rate * exp(sum_{k=2..R} beta_k * n(r_i + k e_i))
```

`shell_weights[j]` is `beta_(j+2)`. Every occupied site on the forward ray
contributes, including those behind another particle. There is no nearest
particle selection or occlusion. Increasing coefficients give more distant
particles a larger boost. These axial offsets have Chebyshev distance `k`.
There is no normalization by the number of neighbors. No sensed particles
means `v_plus = forward_rate`.

The backward rate is `backward_rate`, each lateral rate is `lateral_rate`,
and each clockwise/counterclockwise 90-degree turn has `rotation_rate`.
All rates have units inverse time; the free orientation autocorrelation is
`exp(-2 * rotation_rate * t)`. An occupied destination always blocks a hop.
Forward and backward rates must be positive so every allowed forward hop has
a reverse. Lateral and rotation rates may be zero. The cutoff must obey
`2*R < lattice_size`, preventing periodic self/duplicate sensing.

## Ensemble simulation

Run from `CNEEP_v2`, or put that directory on `sys.path`:

```python
from data.labp_m import LABPMConfig, simulate_ensemble, encode_observations

config = LABPMConfig(
    lattice_size=24, density=0.35,
    forward_rate=2.0, backward_rate=0.2,
    lateral_rate=0.2, rotation_rate=0.1,
    shell_weights=(0.15, 0.30, 0.45),  # distances 2, 3, 4
)
result = simulate_ensemble(
    config, n_trajectories=16, n_frames=1001,
    sample_dt=0.01, burn_time=100.0,
    seed=17, workers=4, progress=True,
)
video = encode_observations(result)  # CPU torch [16,1001,4,24,24]
```

`sample_dt` is an observation interval, not a numerical integration step.
States are sampled at `burn_time + j*sample_dt` without executing a future
event early. The pending next event survives frame boundaries. Returned times
are relative to burn-in. An absorbing state produces repeated frames with
zero subsequent events. The first call includes Numba compilation time.

Initial positions/orientations and event RNG streams are independent across
replicas. `seed` controls all replicas and results do not depend on `workers`.
Optional `initial_states` is integer `[M,L,L]`, with `-1` empty and `0..3`
occupied orientations; it overrides random initialization and the requested
density. Inputs are not mutated.

A segment tree selects particles proportional to their six-event total rate
in `O(log N)` time. A hop only invalidates rates of the moved particle and
particles along the affected row/column rays. Rotation only changes that
particle's rates. This avoids a full lattice-wide rate calculation per event.

## Reference irreversibility and outputs

For each actual hop, the simulator evaluates

```text
ds = log(w(X -> X')) - log(w(X' -> X))
```

The reverse rate uses the **post-hop** state and unchanged orientation.
Symmetric turns/lateral hops contribute zero. This is the model medium path
entropy under even-orientation reversal, not a physical heat claim. The
stationary system-entropy boundary has zero ensemble mean but not generally
zero value on an individual interval.

With `M` replicas and `T` frames:

| Field | Shape | Meaning |
|---|---|---|
| `states` | `[M,T,L,L]` | int8, empty `-1`, occupied direction `0..3` |
| `times` | `[T]` | observation times relative to burn-in |
| `medium_ep` | `[M,T-1]` | sum of actual hop log-rate ratios |
| `medium_ep_maps` | `[M,T-1,L,L]` | half each hop contribution at each endpoint |
| `shell_ep` | `[M,T-1,R+1]` | baseline in slot 0; interaction terms in 2..R |
| `hop_counts`, `rotation_counts` | `[M,T-1]` | actual event counts |
| `seeds` | `[M]` | compiled trajectory RNG seeds |

Maps sum to `medium_ep`; shell terms sum to it up to floating-point rounding.
Slot 1 is zero: immediate exclusion is a rate constraint, not a separate log
rate term. The microscopic baseline slot 0 is not identified with the learned
KNEEP local branch. Learned shell allocation depends on the observation and
architecture, and is not guaranteed to equal this microscopic decomposition.

`event_rates` and `event_medium_ep` expose small-state reference calculations
without changing the supplied particles. Event indices are forward, backward,
left, right, turn left, turn right.

## Training notebook

Open [Corr_labpm.ipynb](../../notebooks/Corr_labpm.ipynb). It defaults to **16
training replicas**, 4 validation replicas, and 4 test replicas with distinct
seeds; it rejects a single training replica. Pairs never cross replica
boundaries. Four directional occupancy channels preserve the discrete state.
The model uses exclusive Chebyshev shell branches.

Set `SAMPLE_DT` in the notebook to control the fixed physical lag of every
training pair. Sampling frequency is `1/SAMPLE_DT`. `OBSERVATION_TIME` controls
the post-burn duration, and `N_FRAMES = OBSERVATION_TIME/SAMPLE_DT + 1` is derived
automatically (the duration must be an integer multiple of the interval).
For example, duration 5 and `SAMPLE_DT=0.005` give 1001 frames at frequency 200;
using `SAMPLE_DT=0.01` gives 501 frames over the same duration. All splits use
the same lag, checked against the actual saved times, and cache keys include
the sampling settings. Gillespie waiting times remain variable internally;
they are never passed to training as variable lags. A frame with no intervening
events repeats the state; no state interpolation is performed.

The notebook compares held-out observed irreversibility with event-path EP.
Fixed-time frames hide intermediate events, so their inferred irreversibility
need not recover all the path EP. Local maps and individual learned shell
contributions are descriptive and can be signed. The network returns spatial
means; the notebook multiplies branch scores by `L*L` before the NEEP objective
and sums raw local maps for full-system entropy. Its final force heads start
at zero, giving the zero-score baseline without lattice-size-dependent forces.
`LABPM_SMOKE=1` selects a short end-to-end
simulation/training/plotting run for validation, not scientific conclusions.

## MIPS sanity notebook

Open [sanity_check_labpm.ipynb](sanity_check_labpm.ipynb) in this data directory.
It compares forward sensing with the same model at zero sensing strength,
using multiple replicas and matched initial configurations. It includes:

- Coarse density snapshots, every replica's final occupancy, and an animation.
- Periodic largest-cluster fraction, density contrast, and low-q power over time.
- Late-time density histograms and radially averaged density structure factors.
- Random configurations with identical particle count as a reference, and
  standard errors computed across independent replicas.

`SAMPLE_DT` and `OBSERVATION_TIME` control its fixed-time observation grid too.
Defaults use L=40, 4 replicas/model, duration 500, and sampling interval 1;
`BURN_TIME=0` makes relaxation from a random configuration visible. These are
screening settings, not a claimed MIPS coexistence point. Inspect late-time
drift, coarse-graining scale, and system-size dependence before interpreting
clustering as phase separation. No automatic pass/fail threshold is imposed.

`LABPM_SMOKE=1` runs a small check with animation disabled. Outputs go to
`results/labpm_mips/` (override with `LABPM_OUTPUT_DIR`); optional
`SAVE_TRAJECTORIES=True` also saves the full fixed-time ensemble.

Run simulator tests from the project directory:

```text
python -m unittest discover -s tests -p test_labp_m.py
```
