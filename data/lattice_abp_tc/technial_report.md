# Five-colour WCA lattice-ABP update

## Scope

This implementation uses the five-colour sublattice schedule of the authors'
reference CUDA implementation instead of a random-sequential particle sweep.
One simulation step consists of five translational substeps followed by one
angular Brownian update.  The site colour is

\[
c(x,y)=(x+3y)\bmod 5.
\]

`grid_size % 5 != 1` is required so that periodic cardinal neighbours have
different colours.  The production grid `G=32` satisfies this condition.

## One translational substep

For colour `c`, all particles whose *departure* site has colour `c` are
selected from the same pre-substep state.

1. Keep the occupancy field fixed and evaluate every selected particle's four
   WCA energy changes in one batched gather.  The particle's old site is
   masked in that gather, exactly matching its removal from occupancy.
2. Construct Eq. (5) probabilities and draw one categorical outcome
   (`+x`, `-x`, `+y`, `-y`, or stay) for every selected particle.
3. Commit all selected outcomes together: remove every selected departure
   site and add every final site.
4. Accumulate active, WCA, and total medium EP at the departure site.

The next colour starts from the just-committed occupancy field.  Therefore the
WCA field is recalculated five times per simulation step, once per colour; it
is never reused across a commit.

## Hard-core safety

An occupied destination has `Delta V=+inf` and zero hop probability.  Since
cardinal neighbours have different colours, selected sources cannot be
nearest-neighbour sites and two selected cardinal hops cannot claim the same
destination.  Synchronous commit therefore preserves one particle per site.

## WCA qualification

The original public reference code uses hard exclusion.  This project adds a
finite-range WCA energy.  Rates within a colour are evaluated from a frozen
occupancy snapshot, exactly as required by a synchronous coloured update.
Consequently, if two same-colour hops jointly create or remove a WCA pair,
their independent single-particle `Delta V` values do not include that joint
change.  This is a finite-time-step difference from a serial single-particle
kernel and must be treated as a discretisation error.

For the current production values `L=16`, `G=32`, and `sigma=0.5`,
`dl=0.5` and `r_c=0.561...`; WCA contains only the four cardinal nearest
neighbours.  The model is therefore a hard-core, nearest-neighbour repulsive
lattice gas rather than a spatially resolved continuum WCA potential.

To make the coloured WCA update closer to a serial WCA process, reduce `dl`
and use a colour schedule whose same-colour sites remain farther apart than
two possible hop lengths plus `r_c`, or implement a joint multi-particle
transition rule.  Do not describe the present five-colour WCA path as an
exact finite-`dt` WCA integrator.

## Backend policy

The pre-existing Numba and fused CUDA backends implement a random-sequential
particle sweep.  They are bypassed by `update_scheme="five_color"`; the
Torch path runs on CPU tensors or CUDA tensors and keeps the five commits
explicit.  This avoids silently applying a different update law merely to
obtain a fused kernel.  The Torch path uses a `[B, N, 4, K]` neighbour gather,
where `K` is the runtime WCA stencil length, so enlarging the cutoff does not
restore a particle loop.

## Required checks

- occupancy is binary and sums to `N` after every substep;
- `active_EP + wca_EP == medium_EP` for every saved interval;
- the five-colour result approaches a random-sequential reference as `dt` is
  decreased, for fixed physical parameters;
- repeat this comparison at the target density and WCA strength, not only in
  the dilute limit.
