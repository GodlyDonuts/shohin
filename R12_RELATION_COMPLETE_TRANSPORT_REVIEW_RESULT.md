# R12 Relation-Complete Transport Review Result

**Decision:** finite `S_3` identification mechanics `GO`; uniform neural
reasoning mechanism, resource advantage, preregistration, fitting, and H100
allocation `NO-GO`.

## Reviewed claim

The candidate proposed globally enforced Coxeter relations as a way to recover
missing transitions with fewer labeled endpoints than an unconstrained atlas.
The finite `S_3` falsifier correctly derives:

- 76 involutions on six labels;
- 120 globally relation-valid transitive actions;
- equality of those actions with labeled regular-action relabelings;
- unique completion of one erased canonical edge;
- a target-specific four-edge identifying set for the canonical table.

Those are valid finite statements. They do not establish a uniform neural
sample-efficiency or reasoning advantage.

## Uniform theorem

Let `N = m!` and let the adjacent transpositions of `S_m` act transitively on
an `N`-state carrier.

1. Orbit-stabilizer gives a trivial stabilizer, so every such action is
   regular.
2. Up to conjugacy, the regular action is unique. If semantic carrier labels do
   not matter, zero transition anchors are required to identify the action.
3. On a fixed labeled carrier there are `(N - 1)!` distinct regular action
   tables, because the centralizer of the regular action has size `N`.
4. Exact semantic labeling therefore remains the unresolved resource. Direct
   state labels require `N - 1` labels; transition anchors have a target-
   specific lower bound `ceil((N - 1) / 2)` and a spanning-tree upper bound
   `N - 2`.
5. A uniform learner that identifies every labeled action requires at least
   `ceil(log_(N-1)((N-1)!)) = N - Theta(N / log N)` transition queries in the
   worst case.

The semantic identification cost is therefore `Theta(m!)`. The exact
coefficient is not needed to decide the neural lane.

## Scaling ledger

| `m` | States `N` | Untied edges | Uniform anchor bounds | Global relation applications |
|---:|---:|---:|---:|---:|
| 3 | 6 | 12 | exact target-specific minimum 4 | 60 |
| 4 | 24 | 72 | 17 to 22 | 528 |
| 5 | 120 | 480 | 95 to 118 | 4,560 |
| 6 | 720 | 3,600 | 611 to 718 | 41,760 |

There are `m(m-1)/2` Coxeter relation schemas. Exhaustively enforcing them
from every state costs `m! * (2m(m-1) - 2)` transition applications.

## Matched-control collapse

The apparent target-bit reduction survives only against an untied atlas.

- An untied atlas stores `m!(m-1)` successors and pays factorial state
  alignment.
- A relation-aware tied recurrence on an atomic carrier has the same
  `(N - 1)!` gauge ambiguity.
- A recurrence with permutation coordinates needs only `O(m log m)` state bits
  and one shared adjacent-swap rule.
- A hard-coded coordinate update swaps positions `i` and `i+1` and requires no
  learned transition atlas or relation oracle.

The favorable recurrence and hard-coded controls remove the claimed advantage.
Relation consistency may still be a useful regularizer, but it is not a new
reasoning primitive.

## Omitted resources in the candidate ledger

The current 36-target-bit versus 12-target-bit comparison does not charge:

- the selected anchor indices;
- the semantic carrier-to-permutation decoder;
- supplied carrier size and transitivity;
- generator-token and presentation semantics;
- factorial relation-oracle applications;
- query decoding from arbitrary state labels;
- the group operation used to generate supervision.

If relation consistency is architectural, the favorable tied recurrence must
receive it. If it is supervised, relation-oracle generation and optimization
must be counted.

## Gate table

| Gate | Decision |
|---|---|
| Finite `S_3` enumeration and erased-edge completion | `GO` |
| Target-specific four-edge `S_3` identification | `GO` |
| Uniform `S_m` reasoning primitive | `NO-GO` |
| Resource advantage over favorable recurrence | `NO-GO` |
| Neural preregistration | `NO-GO` |
| Neural source/data/fitting/H100 | `NO-GO` |
| Autonomous Shohin reasoning or novelty claim | `NO-GO` |

## Preservation boundary

Preserve the finite `S_3` artifact as an exact identifiability certificate and
possible relation-consistency regularizer. An optional CPU closure may solve
the exact `S_4` anchor coefficient, but it cannot overturn the factorial
scaling result and has no capability priority.

The highest-leverage Shohin frontier remains natural-language compilation,
common-mode operation-selection errors, internal state actuation, recurrent
consumption, and termination.

