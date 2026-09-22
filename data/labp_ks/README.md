# LABP-KS: direction-selective shell sensing

`labp_ks` is a separate lattice active-particle model with exact continuous-time
Gillespie simulation. The existing `labp_m` model is unchanged. It uses the same
four-channel observations, periodic exclusion, CPU replica parallelism, CUDA
replica parallelism, and transition-pair training workflow.

## Dynamics

Headings run clockwise: `0=up, 1=right, 2=down, 3=left`. For particle `i` with
heading `c`, let `n_plus[k]` and `n_minus[k]` count neighbors with headings
`(c+1)%4` and `(c-1)%4` on the **entire exclusive Chebyshev shell** at distance
`k`. Shell corners are counted once; there is no ray, nearest-neighbor selection,
occlusion, force toward neighbors, or alignment torque.

```text
q[k] = ((1 + angular_bias)/2 * n_plus[k]
      + (1 - angular_bias)/2 * n_minus[k]) / sensing_scale
G[k] = shell_weights[k-2] * (1 - exp(-q[k]))
v0   = forward_rate - backward_rate
w_forward = backward_rate + v0 * exp(sum(G))
w_backward = backward_rate
w_left = w_right = lateral_rate
w_turn_left = w_turn_right = rotation_rate
```

Occupied hop destinations have rate zero. Rotation never changes a particle's
position. `forward_rate` is the **total forward rate without sensing**, not the
propulsion-only rate. With `backward_rate = lateral_rate = D`, the diffusive
baseline `D` remains fixed; only propulsion changes. Discrete propulsion jumps
still contribute shot noise, so this does not fix total displacement variance.

`shell_weights[j]` controls `k=j+2`. The default `(0, 1, 0)` activates only `k=3`.
`sensing_scale` is a neighbor-count scale, not a perimeter normalization. All
weights are nonnegative. The exponential boost is bounded by `exp(sum(weights))`.
The sensing radius must be smaller than half the periodic lattice side.

Default experiment: `L=30`, density `0.15`, forward/backward/lateral rates
`8/0.1/0.1`, each rotation rate `0.01`, `sensing_scale=1`, `angular_bias=1`.
Sparse density and bounded sensing are starting parameters for inference;
neither phase separation nor a nonlocal learned spectrum is assumed.

## Exact sampling and entropy

Every replica has an independent clock and RNG stream. CPU uses Numba plus
replica threads; CUDA uses one block per replica and keeps event loops on the
device. Both implement direct SSA with rate trees and float64 rates/clocks.
There is no tau-leaping or simultaneous particle update.

An event refreshes every potentially affected particle's rate: both spatial
endpoint neighborhoods for a hop, and the entire sensing neighborhood for a
rotation. The latter is essential because another particle's heading is part of
the sensed field. Refreshing only the rotating particle would produce an
incorrect CTMC even though its own rotation rate is constant.

`simulate_ensemble` records states at fixed physical intervals, retaining pending
events across observation boundaries. `simulate_event_ensemble` records exactly
one actual transition per pair. It first relaxes in physical time, discards a
specified number of events, and records complete event-to-event holding times.
The event count is uniform across replicas; physical durations differ. Neither a
chosen burn time nor a short event warm-up proves stationarity.

The medium-entropy convention treats orientations as even under reversal:

```text
medium increment = log(w(X -> Y)) - log(w(Y -> X))
```

The reverse rate is evaluated in the **post-event state**, including the changed
sensing environment. Rotations have zero medium increment because their two
rates are equal, although rotations affect later hopping rates. This is exact
for the implemented model; it is not a claim of thermodynamic heat.

`shell_ep` is an explicit diagnostic allocation. Slot zero contains the signed
bare affinity `log(forward_rate/backward_rate)`. A forward hop allocates
`log(w_forward/forward_rate)` to shell `k` in proportion to `G[k]/sum(G)`; a
backward hop uses the negative allocation in the post-event state. When the
sum is zero the allocation is zero. The slots sum to the exact medium increment,
but are **not uniquely defined shell entropies or supervised spectrum labels**.
The fixed diffusion floor makes the total log-rate boost nonlinear in `sum(G)`.

For event training, sample pairs uniformly in events. Convert an accumulated
score into a physical-time rate using `sum(score)/sum(waiting_times)`, never the
average of `score/waiting_time`. Event-boundary states follow the jump-chain
measure rather than the physical-time stationary measure. NEEP learns an event
forward/reverse ratio including a stationary boundary term; its per-event value
need not equal the medium label.

## Angular asymmetry and the learned spectrum

`angular_bias = chi` changes the direction-response matrix:

```text
J[c,c+1] = (1+chi)/2
J[c,c-1] = (1-chi)/2
```

At `chi=0` the two perpendicular directions contribute equally and `J=J.T`.
At `chi=1` only the next clockwise heading is sensed; `chi=-1` reverses that
cycle. The total angular weight stays one. Neither value introduces a torque or
changes the rotation noise. Nonzero `chi` breaks reflection symmetry while
preserving quarter-turn symmetry of the square lattice.

There is a useful representability argument for the existing ShellForce model.
For a linear shell force `F=A*x_mid`, its score is

```text
s = <A*(x0+x1)/2, x1-x0>.
```

If `A` is symmetric, this is exactly the boundary difference
`(<x1,A*x1> - <x0,A*x0>)/2`. It telescopes on a closed trajectory and has zero
stationary mean. For an even spatial shell kernel, a nonsymmetric angular
matrix makes a nonsymmetric `A` available using the existing four occupancy
channels. Such a force can produce a nonzero score on closed cycles. This is
why angular asymmetry may make useful nonlocal features accessible without
changing the NN architecture.

This argument is limited: the actual NN is nonlinear and can learn other
operators, and its optimum must fit the actual event distribution. Symmetric
sensing is not an equilibrium control; active hopping remains irreversible.
Propulsion-only dynamics can still contain affinities outside the model's
representational capacity. Increasing `abs(chi)` therefore need not monotonically
increase either inference accuracy or any one learned branch. Signed branches
can compensate one another, and their sum is what the training objective sees.

For comparison, use no sensing, `chi=0`, `0.5`, `1`, and optionally `-1`, with
fixed noise rates, shell weights, density, and training budget. Use independent
training seeds and replica-level uncertainty. At fixed angular weight sum,
changing `chi` also changes count fluctuations, saturation, and stationary
structure: it is not a perfectly matched motility distribution. Record escape
rates, entropy rates, and structural diagnostics alongside learned spectra.

A spatial reflection maps `chi` to `-chi`. Thus scalar stationary entropy rates
should agree between these mirror models, subject to sampling/mixing. A
Chebyshev-shell radius is unchanged by reflection; no sign reversal of a scalar
shell spectrum is required. Independently learned decompositions may differ
because they are not uniquely identified.

The Corr notebook also evaluates a **frozen-model** ablation that removes the
scores of `k>=2` branches. Worsening held-out objective is evidence that the fitted
model uses these branches. It is not proof that a separately retrained local
model cannot do equally well, nor proof of a unique physical range assignment.

## Experiments

- [sanity_check_labpks.ipynb](sanity_check_labpks.ipynb): fixed-time ensembles for
  no sensing, symmetric sensing, and cyclic sensing; snapshots, periodic cluster
  statistics, local-density histograms, structure factors, relaxation checks,
  and complete waiting-time diagnostics (`escape_rate * wait ~ Exp(1)`). MIPS
  diagnostics indicate morphology; a single finite system does not establish a
  phase transition.
- [Corr_labpks.ipynb](../../notebooks/Corr_labpks.ipynb): independent train,
  validation, and test ensembles; event-pair ShellForce training and signed
  Chebyshev spectrum. Defaults are 100 training replicas by 1000 events,
  20 validation and 10 test replicas, and physical burn time 5000. Source/config
  hashes separate incompatible caches.

Both notebooks support `LABPKS_SMOKE=1` for a short execution check. The default
simulation backend is `auto`; an explicit `cuda` request fails clearly if CUDA
is unavailable. A CUDA runtime failure is not silently replaced by a CPU run.
CUDA batches bound concurrent device buffers, not the total returned dataset.
Within each backend, changing CPU workers or CUDA batching preserves seeded
paths; CPU and CUDA use different RNG algorithms.

```python
from data.labp_ks import LABPKSConfig, simulate_event_ensemble

config = LABPKSConfig(angular_bias=1.0, shell_weights=(0.0, 1.0, 0.0))
data = simulate_event_ensemble(
    config, n_trajectories=100, n_events=1000, burn_time=5000,
    backend="auto", batch_size=64, seed=12031, event_warmup=128,
)
```

Run the independent rate and SSA checks from `CNEEP_v2`:

```text
python -m unittest discover -s tests -p test_labp_ks.py -v
```

With `NUMBA_ENABLE_CUDASIM=1` set before starting Python, the same suite checks
the CUDA algorithm without a GPU. CUDASIM checks indexing, event logic, and
synchronization paths; it does not certify native GPU compilation or performance.
