# R12 Canonical Residual Naming Control

**Status:** exact positive reconstruction control; rejected as an R12
invention. It identifies the minimal observable quotient, not hidden causal
coordinates, and therefore defines a symbolic ceiling for WGRQ rather than a
new reasoning mechanism.

## 1. Setup

Let a deterministic Moore system have at most `n` reachable states, a known
reset state `s0`, finite event alphabet `Sigma`, and exact observable outputs.
For a history `u` and continuation `v`, write

```
H(u, v) = output(delta(s0, uv)).
```

Two histories are residual-equivalent when their rows agree on every suffix:

```
u == v  iff  H(u, w) = H(v, w) for every w in Sigma*.
```

This is the observable causal quotient. It may have fewer states than the
physical or originally named system.

## 2. Finite determining-suffix theorem

If the minimal observable quotient has `r <= n` states, any two distinct
residual states have a distinguishing suffix of length at most `r-2`.

Proof sketch: start with the partition induced by immediate outputs and refine
it by one event predecessor step at a time. Every strict refinement increases
the number of blocks. Starting from at least two blocks and ending with `r`
blocks requires at most `r-2` strict refinements. If two states remain equal on
all suffixes of that length, no later refinement can separate them.

Consequently, endpoint observations for prefixes and suffixes whose combined
length is at most `2n-2` suffice to distinguish every reachable residual state:

- every state has a shortest access word of length at most `n-1`;
- suffixes of length at most `n-2` determine its residual row;
- appending one event to an access word and comparing the resulting row
  reconstructs every quotient transition.

The exact bound is deliberately conservative. Smaller systems may stabilize
earlier, but no score may assume that without measuring the refinement depth.

## 3. Canonical reconstruction

Enumerate words in shortlex order. For each observed residual row, choose its
shortlex-minimal access word as the canonical state name. Then:

1. merge histories with identical determining-suffix rows;
2. assign each class its shortlex-minimal access word;
3. for every class and event, append the event and look up the resulting row;
4. attach the immediate Moore output to the class.

This reconstructs a minimal deterministic observable machine uniquely up to
isomorphism. The chosen access-word names make the serialized reconstruction
canonical relative to the declared alphabet order and output encoding.

## 4. What cannot be identified

The procedure cannot recover arbitrary original hidden labels, axes, or state
coordinates. A global relabeling of hidden states preserves every observation.
If two physical states have identical future observable behavior, no endpoint
experiment can separate them at all; the correct reconstruction merges them.

Therefore a training target that asks a neural model to reproduce privileged
hidden state IDs supplies extra supervision. It is not evidence that the model
discovered a task-native causal representation.

## 5. WGRQ consequence

The canonical residual machine is the symbolic ceiling and custody oracle for
the WGRQ board:

- witness generation must use only observable residual differences;
- merge labels must be invariant to randomized hidden-state and alphabet
  relabeling;
- the score must compare behavior and canonical residual partitions, never
  privileged simulator coordinates;
- target-oracle calls used to obtain distinguishing suffixes are counted;
- a neural arm receives no endpoint observation unavailable to its matched
  controls.

If the board can be solved by directly tabulating these rows within the
declared retained-state or source budget, the partition-refinement control has
won. WGRQ may still claim an optimization or oracle-allocation advantage over
matched neural controls, but not a new state object.

## 6. Decision

Retain canonical residual naming as an exact reconstruction control and as the
source of score labels for finite boards. Reject it as an R12 primitive: it is
classical minimal-machine reconstruction, cannot identify hidden coordinates,
and does not establish neural extrapolation beyond the observed determining
set.
