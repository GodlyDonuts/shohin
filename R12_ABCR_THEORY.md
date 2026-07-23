# Apical-Basal Critical Resonance

**Status:** architectural hypothesis; no capability claim

## Problem

Shohin's current reasoning experiments expose three separable bottlenecks:

1. learned local operations can be strong while autonomous composition fails;
2. monotone recurrent fields cannot retract a defeated hypothesis; and
3. rewrite search without a query enumerates possibilities but does not decide
   which computation answers the problem.

Apical-Basal Critical Resonance (ABCR) is a query-directed, reversible neural
proof field. It combines bottom-up support, top-down demand, transient
episode-local bindings, conflict backflow, and exchangeable hypothesis lanes.
The proposal is inspired by two-compartment cortical models, concurrent belief
propagation, competitive constraint networks, and focused proof search. Those
ingredients are not individually novel. The hypothesis under test is their
specific source-deleted, equivariant, model-owned combination.

ABCR must not use a host matcher, symbolic scheduler, branch enumerator,
executor, semantic verifier, target state, or convergence oracle.

## Operational Claim

Given:

- an anonymous typed rule hypergraph;
- anonymous evidence records;
- one anonymous query record; and
- fixed recurrent compute,

ABCR should:

1. propagate evidence forward as basal support;
2. propagate the query backward as apical demand;
3. form episode-local variable/object bonds;
4. activate only rules whose support and demand agree through the same bond;
5. maintain competing hypotheses in exchangeable lanes;
6. inhibit and revise bonds responsible for contradictions;
7. commit only stable support-demand closures; and
8. halt or abstain when its own obligations are resolved or irreducibly
   conflicted.

A clean cross-family pass would establish a bounded general reasoning
mechanism. It would not establish unrestricted intelligence or natural-language
reasoning.

## State

For batch `b`, lane `l`, record `i`, rule `r`, and variable `v`, recurrent state
contains:

```text
S[b,l,i,h]     basal support
D[b,l,i,h]     apical demand
T[b,l,i,h]     tentative state
C[b,l,i,h]     conflict/inhibition
B[b,l,r,v,i]   transient variable-to-record bond
J[b,l,r,h]     rule resonance
Q[b,l,h]       lane query state
H[b,l]         lane halt potential
```

Rule and graph geometry are source-deleted tensors. Every symbol, rule,
variable, record, and lane identity is freshly reindexed per episode.

`S`, `D`, `T`, and `C` have no privileged slot order. `B` is an episode-local
soft partial matching with explicit unbound capacity. Lanes share all weights.

## Recurrent Dynamics

Let `premise(r)` and `conclusion(r)` be typed anonymous incidence tensors, not
host-executed semantics.

### Basal support

Evidence and tentative conclusions send forward proposals:

```text
support_match[r] =
    MatchPremises(S, B, premise(r))
```

`MatchPremises` is a learned equivariant contraction. It receives incidence,
types, equality structure, support state, and bonds. It receives no legal mask
or host binding.

### Apical demand

The query initializes `D`. Active conclusions send obligations backward:

```text
demand_match[r] =
    MatchConclusion(D, B, conclusion(r))
```

Demand is not an answer hint. It specifies which consequences are relevant.
Query-shuffle twins must redirect the activated proof field.

### Multiplicative resonance

A rule becomes active only when support and demand agree through the same
binding:

```text
J[r] = sigmoid(
    Wj(
        support_match[r]
        * demand_match[r]
        * binding_consistency[r]
        - conflict_pressure[r]
    )
)
```

The elementwise product is causal, not decorative. An additive dual-stream
control receives the same tensors, parameter count, and compute but replaces
the product with a learned sum.

### Transient bonds

Bindings update through evidence, demand, and resonance:

```text
B_next = PartialSinkhorn(
    leak_b * B
    + proposal_b(S, D, J, rule_graph)
    - conflict_to_bond(C)
)
```

The partial normalization enforces only competition and an explicit unbound
state. It does not match a rule. Equality and repeated-variable constraints are
learned from anonymous incidence twins.

### Tentative and committed state

Tentative conclusions are reversible:

```text
T_next =
    leak_t * T
    + forward_proposal(J, B, rule_graph)
    - retract(C)
```

Committed support uses a differentiable hard event after stability:

```text
stable = agreement(S, D, T, B) * low_conflict(C)
write = straight_through(stable > threshold)
S_next = max(S_evidence, S, write * T)
```

Only evidence and stable closures are monotone. Tentative hypotheses and bonds
remain retractable.

### Critical conflict

Incompatible overlapping proposals induce inhibition:

```text
C_next =
    leak_c * C
    + incompatible(T, B, rule_graph)
    + duplicate_lane_pressure(T)
```

Conflict flows to the bindings and rules that caused it. A no-conflict-backflow
control keeps the same conflict computation but prevents it from changing
`B`, `J`, or `T`.

### Lane exchange and coalescence

Lanes are exchangeable phase states, not host-created branches. A learned
competition mechanism amplifies distinct low-conflict hypotheses and
coalesces equivalent lanes:

```text
lane_affinity = EquivariantStateSimilarity(S, D, T, B)
lane_gate = compete_and_coalesce(lane_affinity, C, Q)
```

No canonical graph hash or host equivalence test is available in the model
process.

### Halt and abstention

Halt is a learned function of:

- unresolved demand;
- tentative activity;
- conflict energy;
- bond motion;
- rule resonance;
- lane diversity; and
- state velocity.

```text
energy_t =
    unresolved(D)
    + activity(T)
    + conflict(C)
    + velocity(S,D,T,B)

halt = hard_event(H > threshold)
```

Terminal/nonterminal delay twins, cyclic twins, and underdetermined twins are
required. A fixed recurrence cap is only a fail-closed safety bound.

## Energy Interpretation

ABCR is not required to minimize one scalar energy, but a useful diagnostic is:

```text
E =
    E_unmet_demand
    + E_unsupported_claims
    + E_binding_inconsistency
    + E_conflict
    + E_duplicate_lanes
    + E_state_velocity
```

Successful reasoning should reduce all terms except during deliberate
hypothesis splitting. Energy is never used as a host convergence test. It is
logged for causal diagnosis and may supervise the learned halt head.

## Parameter Budget

The protected Shohin trunk has 125,081,664 parameters. The initial ABCR budget
is 32,000,000 added parameters:

| Component | Ceiling |
|---|---:|
| record/rule encoders | 6M |
| support and demand dynamics | 8M |
| bond dynamics | 6M |
| conflict and lane dynamics | 6M |
| state reader and halt | 2M |
| Shohin interface adapters | 4M |
| complete system | 157,081,664 |

The complete system remains below 200M. Larger size is not evidence; every
promoted treatment needs a parameter- and compute-matched generic recurrence.

## Training

Training is staged without exposing a privileged full proof trace to the
autonomous score path:

1. **Bond mechanics:** anonymous repeated-variable, type, and occurrence twins.
2. **One-rule resonance:** set-valued valid activations under query changes.
3. **Reversible composition:** two-to-six-rule episodes with hard recurrent
   state and decaying teacher forcing.
4. **Critical competition:** forks, delete effects, contradictions, cycles,
   and underdetermined queries.
5. **Autonomous closure:** final-answer, abstention, and halt supervision only.
6. **Cross-family confirmation:** five fresh seeds with two entire held-out
   families.

Training may use independent-oracle labels in an offline process. Evaluation
may not contain the oracle, its source, its traces, or any derived legal-action
mask.

## Families

The first rotation uses:

- Horn closure;
- forward and backward dataflow;
- typed occurrence rewriting;
- delete-effect planning; and
- algebraic normalization.

For each confirmation, two families are absent from all optimization. Every
local operator must appear in training, while the held-out family changes its
composition, state topology, and query semantics.

## Controls

1. forward-only AHRF-style support;
2. additive support/demand without multiplicative resonance;
3. no conflict backflow;
4. no persistent transient bonds;
5. one hypothesis lane;
6. shuffled query;
7. shuffled rule cards preserving statistics;
8. parameter/FLOP-matched generic recurrent slots;
9. reset recurrent state each step;
10. fixed-deadline readout; and
11. Shohin trunk-zero intervention.

## Gates

All five seeds must achieve:

- at least 95% exact in every held-out family;
- at least 90% exact at doubled depth and increased graph width;
- at least 95% halt or abstention on cyclic and underdetermined cases;
- 100% storage, symbol, rule, lane, and branch-order reindex invariance;
- at least 99% correct query, donor-state, binding, and RHS intervention
  responses;
- treatment at least 20 percentage points above every causal control;
- paired 95% lower confidence bound above a 10-point advantage;
- no host repair, family-specific inference head, evaluation fine-tuning,
  source access, or custody failure; and
- complete parameter count below 200M.

Reject ABCR if:

- a generic recurrent control matches it;
- it requires gold schedules or host matching;
- conflict backflow is causally inert;
- query interventions do not redirect computation;
- tentative states never retract on counterfactual twins;
- family-specific adapters are required; or
- language integration succeeds only by bypassing the frozen runtime.

## Evidence Ladder

1. deterministic mechanics and tensor-custody tests;
2. isolated one-rule causal interventions;
3. autonomous within-family composition;
4. held-out renderer and depth transfer;
5. held-out task-family transfer;
6. frozen Shohin language compilation and reading;
7. natural-task confirmation with manual transcript review.

No rung may be described using the claim of a later rung.

