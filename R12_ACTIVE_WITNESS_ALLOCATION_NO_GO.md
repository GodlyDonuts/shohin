# R12 Active Witness Allocation No-Go

**Status:** exact partial separation and exact collapse. Adaptive target queries
can beat passive/random allocation, but residual-witness supervision has no
oracle-complexity advantage over a fair active answer-only learner when every
derived label is computed from the same counted transcript.

## 1. Active versus passive theorem

Let a target threshold be `theta in {1,...,N}` and let an ordinary answer query
at `x in {1,...,N-1}` return

```
O_theta(x) = 1[x >= theta].
```

Adaptive binary search identifies `theta` with `ceil(log2 N)` one-bit answers,
and this is optimal because a depth-`m` binary decision tree has at most `2^m`
leaves.

Any nonadaptive schedule of `m` locations partitions the `N` possible
thresholds into at most `m+1` answer transcripts. Under the uniform target
prior, even the optimal decoder therefore has

```
P(theta_hat = theta) <= (m + 1) / N.
```

Success at least `1-delta` requires `m >= (1-delta)N - 1`, and worst-case exact
identification requires `N-1` calls. This is a real `Theta(log N)` versus
`Theta(N)` active/passive separation.

## 2. Active answer-only simulation theorem

Suppose a WGRQ policy chooses query `x_t` from the public transcript

```
T_(t-1) = (x_1,y_1,...,x_(t-1),y_(t-1))
```

and receives the ordinary answer `y_t=O_theta(x_t)`. If every merge,
separation, collision, or witness label is computed from those public queries
and counted answers, an active answer-only learner can:

1. run the identical query-selection policy;
2. submit the identical ordinary answer query;
3. receive the identical answer;
4. compute the identical derived labels;
5. perform the identical model update.

Induction on `t` gives identical transcripts, parameters, and outputs for every
target and random seed. Thus WGRQ has no strict oracle or sample advantage over
the fair active answer-only class.

If WGRQ instead receives exact residual-equivalence labels, target-selected
counterexamples, hidden state IDs, or simulator-produced witness identities,
it has a stronger oracle. Equal call counts do not restore fairness. The ledger
must count oracle semantics, returned information bits, query-description bits,
target-dependent witness-search work, and adaptive rounds.

## 3. Smallest exhaustive audit

`N=4` is minimal. Adaptive binary search and active answer-only both identify
all four targets in two calls. Every nonadaptive two-call schedule induces at
most three transcripts, so uniform-prior exact success is at most `3/4`.
Enumerating all depth-two adaptive trees and all nonadaptive schedules can only
verify this identity; it cannot rescue a WGRQ oracle advantage.

## 4. Decision

Reject WGRQ as an oracle-complexity or finite-sample invention relative to
active answer-only supervision. Preserve adaptive allocation as a known data
acquisition control. A remaining CPU board may test only a narrower neural
optimization claim under frozen oracle transcripts, favorable active controls,
and a complete information ledger.
