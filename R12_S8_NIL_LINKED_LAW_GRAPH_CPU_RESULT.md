# R12 S8 Nil-Linked Law Graph: CPU Result

**Date:** 2026-07-19
**Source/preregistration commit:** `81fb6b0`
**Post-commit seed:** `4822478724546321200`
**Decision:** `admit_s8_nil_linked_law_graph_preregistration`

## Coverage

The frozen falsifier evaluates 3,520 depth-three-through-eight programs over
440 hidden coordinate systems:

- all 120 permutations at modulus 5;
- 128 deterministic permutations at modulus 7;
- 128 deterministic permutations at modulus 11; and
- 64 deterministic permutations at modulus 13.

Every program has at least two contextual laws. Event records are stored under
a random node permutation independent of their executable order.

## Results

| Arm | Exact state | Answer |
|---|---:|---:|
| **Nil-linked treatment** | **3,520/3,520 = 100.000%** | **100.000%** |
| Storage-order shortcut | 357/3,520 = 10.142% | 37.614% |
| Reversed event links | 174/3,520 = 4.943% | 32.500% |
| Deranged operation cards | 37/3,520 = 1.051% | 23.580% |
| One-witness unit completion | 129/3,520 = 3.665% | 28.693% |
| State reset per event | 72/3,520 = 2.045% | 28.835% |
| Early nil after one event | 70/3,520 = 1.989% | 29.432% |

Changing every node's storage ID while preserving predicted links leaves all
3,520 treatment states and answers unchanged. All 3,520 executable paths differ
from raw storage order. Treatment remains exact separately at every modulus;
every causal/shortcut arm remains below its frozen 40% state ceiling. All ten
preregistered CPU gates pass.

## Interpretation

The interface is complete: initial state, law cards, entity/card bindings,
entry pointer, next-event links, nil termination, and query are sufficient for
the confirmed S7 dynamics to reproduce an independent reference executor. The
large causal collapses establish that links are not decorative metadata and
that schedule, two-witness law evidence, persistent state, and terminal depth
all affect the result.

This is a mechanics result, not a neural reasoning result. The graph fields in
this falsifier are gold. A neural experiment is authorized only after the
whole-source board builder, sub-16M graph compiler, training exclusions,
favorable ordinary parser, causal controls, and frozen assessor are committed.
No S8 development or confirmation board exists yet.

The architectural boundary remains explicit: categorical argmax/equality,
nil-linked traversal with a node-count safety bound, the confirmed S7 cyclic
compiler, and categorical pop-insert mutation are hard runtime operations.

## Artifact

CPU report SHA-256:
`c98bd96ef66289fe580523a20116c62c96bef77ef69b7c55eebd2c94630b3aeb`
