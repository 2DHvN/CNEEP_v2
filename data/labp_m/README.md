# LABP-M: modified lattice active particles

Four discrete orientations, hard-core exclusion, periodic boundaries, and an
exact Gillespie direct method. Independent replicas run in parallel CPU threads
or in a fused CUDA kernel. Each replica keeps its own event clock and executes
one event at a time. Training and simulation devices are selected independently.
There is no tau-leaping, sweep, or fixed numerical integration time step.

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
from data.labp_m import LABPMConfig, simulate_ensemble, encode_observations, simulation_backend

config = LABPMConfig(
    lattice_size=24, density=0.35,
    forward_rate=2.0, backward_rate=0.2,
    lateral_rate=0.2, rotation_rate=0.1,
    shell_weights=(0.15, 0.30, 0.45),  # distances 2, 3, 4
)
result = simulate_ensemble(
    config, n_trajectories=16, n_frames=1001,
    sample_dt=0.01, burn_time=100.0,
    seed=17, backend="auto", workers=4, batch_size=64, progress=True,
)
video = encode_observations(result)  # CPU torch [16,1001,4,24,24]
```

`sample_dt` is an observation interval, not a numerical integration step.
States are sampled at `burn_time + j*sample_dt` without executing a future
event early. The pending next event survives frame boundaries. Returned times
are relative to burn-in. An absorbing state produces repeated frames with
zero subsequent events. The first call includes Numba compilation time.

Initial positions/orientations and event RNG streams are independent across
replicas. `seed` controls all replicas. Within a backend, results do not depend
on CPU `workers` or CUDA `batch_size`. CPU and CUDA use different event RNGs,
so they sample the same CTMC distribution but do not produce identical seeded
paths. The random initial configurations are the same across backends.
Optional `initial_states` is integer `[M,L,L]`, with `-1` empty and `0..3`
occupied orientations; it overrides random initialization and the requested
density. Inputs are not mutated.

A segment tree selects particles proportional to their six-event total rate
in `O(log N)` time. A hop only invalidates rates of the moved particle and
particles along the affected row/column rays. Rotation only changes that
particle's rates. This avoids a full lattice-wide rate calculation per event.

### Selecting parallel execution

- `backend="cpu"` (API default): compiled Numba simulations in a thread pool.
  `workers=None` uses available logical CPUs, capped by the replica count;
  `workers=1` runs serially. A short initialization compiles the kernel before
  all replicas, including replica zero, are submitted to the pool.
- `backend="cuda"`: one CUDA block per replica. A leader selects and executes
  the next event; block threads refresh the affected particle rates. Independent
  replica clocks need not agree, while the observation grid is shared. The
  event loop stays on the device, without a Python call for every jump.
- `backend="auto"`: uses CUDA if Numba reports it available, otherwise CPU.
  This selects an available device, not the empirically fastest backend.
  `simulation_backend("auto")` exposes that choice. An explicit unavailable
  CUDA request raises an error. Runtime/compilation failures are surfaced,
  not silently replaced by a CPU run.

`batch_size` controls how many replicas are sent to CUDA together; it is not a
physical timestep or an event approximation. CPU ignores this setting. Every
replica retains its own RNG state, so splitting the same ensemble into smaller
CUDA batches preserves its paths. Returned arrays are still NumPy arrays on
the host; large ensembles require memory for the complete requested output,
including float64 spatial entropy maps.

The CUDA launcher automatically reduces the replica batch if its estimated
working/output buffers exceed 70% of currently free device memory. Kernel
launches process at most 1024 events per replica, then resume from the saved
clock, RNG, rates, and partially accumulated observation interval. Only a
small progress array returns to the host between these launches. This keeps a
long burn-in or observation interval from becoming one simulation-long kernel
launch, without changing the CTMC or dropping pending events.

CUDA simulation requires a working **Numba CUDA** installation, including a
compatible NVIDIA driver and CUDA compilation libraries. CUDA-enabled PyTorch
alone is not sufficient. Rates, clocks, and entropy use float64 on both
backends. GPU throughput depends on the device, number of replicas, lattice
size, and output volume; a small ensemble can be faster on CPU.

Both notebooks expose `SIMULATION_BACKEND` (environment override
`LABPM_BACKEND`, default `auto`), `CUDA_BATCH_SIZE`, and `WORKERS` (default
`None`, all available CPUs; environment override `LABPM_WORKERS`). The training
notebook records the resolved simulation backend and simulator source hashes
in its cache metadata, independently of its PyTorch training device.

To compare throughput on the target machine after compilation:

```text
python -m data.labp_m.benchmark --replicas 16 64 --workers 1 4 8
python -m data.labp_m.benchmark --backends cuda --replicas 64 128 --batch-size 64
```

The first command includes CUDA only when available. Timing includes allocation
and copying results to the host, and the script checks seeded reproducibility
within each backend. Reported event counts exclude discarded burn-in events;
elapsed time includes burn-in. Compare replica throughput for equal settings.

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
python -m unittest discover -s tests -p test_labp_m_cuda.py
```

CUDA-specific tests skip when no device is available. For a small correctness
check without GPU hardware, start a **fresh** process with
`NUMBA_ENABLE_CUDASIM=1` and run the second command. Numba's CUDA simulator
checks the algorithm and thread coordination; it does not validate device
compilation or measure real GPU performance. The benchmark rejects simulated
CUDA timings.
