# R12 Fork-Core Theory Audit

**Status:** rejected as a new primitive; retained as a mathematical control and
as a source of falsifiable merge-certification bounds.

**Implementation authority:** none. This document authorizes no data build,
model change, fit, score, or GPU job.

## 1. Decision

The Fork-Core Quotient (FCQ) does not define a new state ontology. Its exact
form is a residual-state transducer. Its approximate form is an approximate
information state or predictive-state representation specialized to a chosen
late-query protocol class.

The useful residue is geometric:

1. pairwise-compatible compressed histories need not admit one shared state;
2. in finite-dimensional convex signature spaces, global compatibility has an
   exact finite witness size;
3. pairwise tests incur a sharp worst-case radius inflation;
4. the required bit budget is controlled by predictive dimension, update
   expansion, horizon, and target error.

These results improve the R12 falsifier, but they are not evidence that Shohin
reasons and they do not justify naming a new mechanism.

## 2. Restricted predictive object

Let `A` be a finite continuation-action alphabet and `Y` a finite answer
alphabet. An adaptive protocol of horizon at most `H` chooses its next action
from prior answers:

```
pi_t : Y^(t-1) -> A union {stop}.
```

After history `h`, protocol `pi` induces a joint answer-transcript law
`P_h^pi`. Fix a finite-dimensional vector space `Q` of transcript functions
that contains the allowed cylinder indicators and constants and is closed
under left residuals. This explicitly excludes arbitrary late INDEX queries
unless their indicators are in `Q`.

The restricted predictive signature is

```
s(h) = (P_h^pi)_(pi in Pi[Q,H])
```

with metric

```
||s - s'||_Pi = sup_pi TV(P^pi, P'^pi).
```

Let `C` be the closed convex set of coherent signatures. Convexity corresponds
to mixing causal kernels with a hidden initial seed. Since signatures are
normalized linear functionals on `Q`, their affine dimension `d` is at most
`dim(Q) - 1`.

## 3. Joint fork operator

For admissible event generators `E = {e_1, ..., e_r}`, define

```
J(h) = (s(h), s(h e_1), ..., s(h e_r)).
```

Let `K` be the convex set of admissible center tuples. Without an imposed
dynamics graph, `K` is a subset of `C^(r+1)` and can have affine dimension up
to `(r+1)d`. If a shared affine update family is imposed,

```
K = {(q, T_e1 q, ..., T_er q) : q in C},
```

then its affine dimension is at most `d`.

For positive answer and update tolerances `alpha` and `beta`, use the normalized
product norm

```
||(v_0, ..., v_r)||_(alpha,beta)
  = max(||v_0||_Pi / alpha, max_i ||v_i||_Pi / beta).
```

For a finite proposed merge fiber `F`, define its Fork-Core radius

```
rho(F) = inf_(c in K) max_(h in F) ||J(h) - c||_(alpha,beta).
```

The fiber has one valid shared current-and-successor center exactly when
`rho(F) <= 1`.

## 4. Finite witness theorem

Let `D = affdim(K)` and assume the metric balls induced inside `K` are convex.
Then

```
rho(F) = max_{S subset F, |S| <= D+1} rho(S).
```

**Proof.** For a proposed radius `t`, each history defines the convex set

```
K intersect closed_ball(J(h), t).
```

The full fiber has radius at most `t` exactly when all these sets intersect.
Helly's theorem in the `D`-dimensional affine hull says that intersection is
equivalent to intersection of every subfamily of at most `D+1` sets. Taking the
smallest feasible `t` gives the identity.

This theorem does not make FCQ a new primitive. It converts a global merge
claim into a bounded-arity falsifier when the relevant predictive dimension is
known.

## 5. Pairwise tests are quantitatively insufficient

Let `d_F = affdim(conv(J(F))) >= 1`, and suppose every pair in `F` has normalized
radius at most one. Then

```
rho(F) <= 2 d_F / (d_F + 1).
```

The constant is sharp. Pairwise validity bounds every pairwise distance by two.
The barycenter of any `k <= d_F + 1` points lies within
`2(k-1)/k` of each point. Applying the finite witness theorem gives the bound.
This recovers the classical finite-dimensional Jung/Bohnenblust radius
constant; it is not a novel geometric inequality.

The smallest obstruction has three histories and three answer atoms. With

```
s(h_i) = delta_i in Delta_3,
```

every pair has total-variation radius `1/2`, while one center for all three
requires radius `2/3`. Pairwise contrastive training can therefore certify
every edge and still create an invalid merged state.

More generally, `D+1` simplex vertices have pair radius `1/2` and global radius
`D/(D+1)`, giving the sharp inflation ratio `2D/(D+1)` and showing that witness
arity `D+1` is necessary.

## 6. Horizon and bit law

Suppose the reachable signatures lie in a `d`-dimensional norm ball of radius
`R`, every event residual is `L`-Lipschitz, and every update is requantized with
error at most `delta`. After `t` updates,

```
error_t <= delta * sum_(j=0)^t L^j.
```

A `delta`-net has at most `(1 + 2R/delta)^d` elements. Therefore error at most
`epsilon` through horizon `H` is achievable with the covering upper bound

```
b <= ceil(d log2(1 + 2 R S_H / epsilon)),
S_H = sum_(j=0)^H L^j.
```

The qualitative regimes are decisive:

- `L < 1`: horizon-independent bit growth is possible;
- `L = 1`: required bits grow like `d log H`;
- `L > 1`: required bits grow linearly in `H` at rate `d log L`.

Worst-case token conditioning is not generally contractive. For

```
P = (p, 0, 1-p),  Q = (p-delta, delta, 1-p),
```

conditioning on the first two atoms expands TV distance from `delta` to
`delta/p`. Any contraction claim must therefore restrict rare continuations,
use a probability-weighted metric, or be explicitly average-case.

## 7. Collapse and prior-art audit

The representation itself collapses completely:

- finite exact signatures plus event updates are a residual machine and
  transition monoid;
- future-test signatures are predictive-state representations;
- zero-radius equivalence is restricted probabilistic bisimulation;
- lossy signature coding is causal/predictive rate-distortion;
- a learned signature metric is metric representation learning.

The common-center requirement is already implicit in the single shared reward
and update kernels of [Approximate Information State for Approximate Planning
and Reinforcement Learning in Partially Observed
Systems](https://www.jmlr.org/papers/volume23/20-1165/20-1165.pdf). Composable
future tests and recursive predictive-state updates are explicit in [Compressed
Predictive States](https://www.jmlr.org/papers/volume15/hamilton14a/hamilton14a.pdf).
Lossy compression of causal states is covered by [Causal Rate-Distortion for
Infinite-Order Markov Processes](https://arxiv.org/abs/1412.2859).

The June 2026 paper [History, Hypergraphs, and Memory: The Exact Complexity of
Deviation-Rational Control](https://openreview.net/forum?id=oNLGDwZo5d) already
proves that pairwise compatibility can hide higher-order memory gaps and gives
a Helly certificate for one-state memory in a convex controller simplex. The
radius factor above is an application of classical Jung/Bohnenblust geometry.
No novelty claim is allowed for FCQ, the Helly certificate, or the sharp radius
constant without a substantially stronger delta and a complete literature
review.

## 8. Falsifiable consequences

1. If a measured joint fork cloud has verified effective affine dimension two,
   every global incompatibility must have a triple witness up to the declared
   approximation residual. A genuine irreducible four-history violation refutes
   the dimension estimate or convexity assumptions.
2. At a fixed continuation class, required state bits should scale with
   `d_eff log2(1/epsilon)`. Horizon scaling should plateau for contractive
   modes, grow logarithmically near `L=1`, and become linear when `L>1`.
3. Adding `n` independently addressable late INDEX bits requires at least `n`
   state bits for uniform error below one half. Apparent sublinear storage must
   be using query restriction, external access, or nonuniform error.

## 9. Next mathematical problem

Do not implement FCQ. The unresolved object is **coherent action extension**:
whether a family of event maps can be extended from exact causal states to a
lower-complexity ambiguity space while preserving all event-monoid relations,
not merely extending each generator independently. Injective or hyperconvex
hulls can extend individual nonexpansive maps, but independent extensions need
not compose coherently off the original state space.

R12 advances only if that simultaneous extension problem yields either a new
resource theorem, a smallest obstruction that changes the training target, or
a uniform learned realization with a measured advantage over AIS/PSR controls.
