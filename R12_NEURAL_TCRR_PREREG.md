# Neural Typed Critical-Pair Rewrite Reactor

**Status:** preregistration; CPU mechanics admitted, neural claim untested

## Claim Boundary

N-TCRR-1 asks whether a neural system can receive only episode-local,
source-deleted typed rewrite declarations and an initial term graph, then
autonomously enumerate the exact reachable normal-form set and cycle witnesses.
The system must emit occurrence-specific graph transactions, manage branches,
and select its own halt.

A pass establishes bounded architecture-native nonmonotone rewrite reasoning.
It does not establish language understanding or genuine general reasoning.
Those claims require a separately frozen language compiler and transfer to
unseen natural task families without changing the reactor.

The committed CPU mechanics and independent audit are only an executable
semantic specification:

- `pipeline/typed_critical_pair_rewrite_board.py`
- `pipeline/audit_typed_critical_pair_rewrite_board.py`
- `artifacts/r12/tcrr_mechanics_521058d.json`

They may generate training labels and assess sealed evaluation transcripts, but
they may not be present in the neural evaluation process.

## Why A New State Machine Is Required

AHRF is intentionally monotone: facts are written once and retained. Typed
rewriting requires deletion, replacement, capacity reclamation, alternative
successors, and mixed cyclic and terminating paths. Extending the AHRF latch
with exceptions would obscure these causal requirements. N-TCRR therefore uses
an explicit transaction state whose mutation semantics can be independently
audited.

## Frozen Geometry

| Quantity | Value |
|---|---:|
| Graph slots per branch | 16 |
| Branch lanes | 8 |
| Rules per episode | at most 8 |
| Nodes per rule side | at most 12 |
| Constructor arity | at most 3 |
| Occurrence path depth | at most 8 |
| Recurrent safety bound | 64 |
| Hidden width | 256 |
| Added parameter ceiling | 16,000,000 |
| Complete Shohin ceiling | 200,000,000 |

The first implementation uses shared slot-, rule-, branch-, constructor-, and
type-equivariant weights. It contains:

1. six graph/rule encoding rounds;
2. four transaction-decoder rounds;
3. an agenda and branch controller;
4. a visited-state comparator;
5. a terminal normal-form bank;
6. a cycle-witness bank; and
7. a learned halt head.

The protected Shohin trunk may provide frozen renderer-record embeddings.
Reasoning-state mutation remains entirely inside N-TCRR. A trunk-zero
intervention measures whether those embeddings contribute causally.

## Source-Deleted Tensor Contract

```text
graph_active       [B,K,N]
graph_root         [B,K,N+1]
node_kind          [B,K,N,3]
node_constructor   [B,K,N,C]
node_type          [B,K,N,Y]
node_children      [B,K,N,A,N+1]
branch_active      [B,K]

lhs_kind           [B,R,P,3]
lhs_constructor    [B,R,P,C]
lhs_type           [B,R,P,Y]
lhs_children       [B,R,P,A,P+1]
lhs_variable_eq    [B,R,P,V]

rhs_kind           [B,R,P,3]
rhs_constructor    [B,R,P,C]
rhs_type           [B,R,P,Y]
rhs_children       [B,R,P,A,P+1]
rhs_bound_variable [B,R,P,V]
rhs_delete         [B,R]
```

Constructor, type, rule, variable, slot, and branch identities are freshly
permuted per episode. No global semantic ID, family label, source text,
episode class, oracle state, expected count, schedule, or legal-action mask is
available to the evaluated model.

At each tick, the model emits:

```text
agenda branch
mode = STEP | FORK | ACCEPT_NORMAL | ACCEPT_CYCLE | HALT
rule pointer
root-relative occurrence path
next occupancy
next constructor and type references
next child pointers
next root
optional second successor for FORK
```

Occurrence paths are semantic. A shared DAG node reached through two paths has
two rewrite occurrences; changing one path must not silently mutate the other.

## Rule-Blind Committer

A fixed non-neural committer installs a predicted transaction. It may enforce
only:

- tensor shape and pointer range;
- declared type compatibility;
- reachability and acyclicity of each graph value;
- branch and slot capacity;
- conservation of live graph records; and
- exact installation of the packet the model emitted.

The committer may not inspect rule cards, pattern-match, bind variables, choose
a redex, construct an RHS, rank branches, repair a packet, test semantic
equivalence, detect a normal form, detect a cycle, or decide halt. Invalid
transactions remain incorrect observations.

## Custody

Evaluation runs in an allowlisted directory containing only:

- the frozen neural checkpoint;
- the neural runtime and rule-blind committer;
- source-deleted packet files; and
- exact source and checkpoint receipts.

The production and independent CPU oracles, board generators, training data,
targets, schedules, and expected outputs must not exist in that process or
filesystem.

The model's raw transactions are sealed before a one-access assessor loads the
independent oracle. Assessment never returns information to the model.

## Board

The current 14 audited CPU episodes remain untouched mechanics tests. A new
procedural board is required for neural work:

| Partition | Episodes | Purpose |
|---|---:|---|
| local-transition train | 48,000 | one-step match, bind, delete, and graph deltas |
| autonomous train | 24,000 | two-to-six-step hard rollouts |
| composition development | 4,000 | unseen rule co-occurrences and depths 7-10 |
| renderer development | 4,000 | unseen slot and rule layouts |
| family confirmation | 8,000 | typed-stack and dataflow rewriting |

Training families are algebraic normalization, Boolean simplification, and
list/tree rewriting. Typed-stack reduction and dataflow rewriting are withheld
in full. Every local primitive appears in training, while confirmation combines
them in unseen motifs such as capacity release followed by nested redex
creation, critical forks, and mixed cyclic/terminating paths.

No exact graph, graph-isomorphism class, normalized rule window, or rule-pair
composition may cross partitions.

Mandatory causal twins include:

- RHS-pointer twins with identical marginal statistics;
- two root-to-shared-node occurrence twins;
- capacity 16 versus capacity 15 twins;
- branch-order twins;
- constructor/type/rule/storage reindex twins; and
- cyclic-plus-terminating twins.

## Optimization

The frozen objective is:

```text
L = 1.00 L_legal_set
  + 2.00 L_successor_graph
  + 0.50 L_variable_binding
  + 0.50 L_occurrence_path
  + 1.00 L_terminal_set
  + 0.50 L_branch_coverage
  + 0.50 L_cycle_witness
  + 0.25 L_halt
  + 0.10 L_equivariance
  + 10.0 L_invalid_soft
```

`L_legal_set` is set-valued: negative log probability mass over all legal
actions, not one oracle-chosen schedule. Successor-graph loss minimizes over
storage-equivalent layouts. Terminal-set loss uses bipartite matching between
predicted and target normal forms.

Training phases:

1. fit one-step rule, occurrence, binding, and delta prediction;
2. roll out argmax transactions and decay teacher forcing to zero;
3. freeze the local motor and fit agenda, branch coverage, cycle witnesses,
   terminal collection, and halt;
4. jointly polish at low learning rate using only hard recurrent state; and
5. run five independent confirmation seeds without changing thresholds or
   board generation.

No primitive name, intermediate host state, single privileged trajectory, or
fixed answer schedule may supervise the autonomous score path.

## Matched Controls

Every promoted treatment requires:

1. **generic recurrence:** same state, outputs, parameters, and compute, but
   rule cards are reduced to object-marginal summaries;
2. **physical-slot reactor:** selects storage slots rather than root-relative
   occurrences;
3. **no writeback:** predicts every tick from the initial graph;
4. **greedy reactor:** retains at most one successor while preserving branch
   compute;
5. **shuffled RHS:** evaluation intervention preserving arity, type, rule
   count, and graph statistics;
6. **fixed deadline:** disables learned halt and reads at tick 64; and
7. **trunk zero:** zeros Shohin-provided record embeddings.

## Gates

All five seeds must pass:

- at least 99.5% exact unseen one-step successors;
- at least 95% exact complete outcome sets on canonical development;
- at least 90% exact in every unseen-composition, renderer, and held-out-family
  cell;
- at least 99% learned halt with at most 1% safety exhaustion;
- 100% capacity conservation, typing, reachability, and acyclicity;
- 100% slot, rule, constructor, type, and branch reindex invariance;
- at least 99% correct RHS-twin, occurrence-twin, and capacity-twin responses;
- treatment at least 20 percentage points above every matched learned control;
  and
- paired 95% lower confidence bound above a 10-point treatment advantage.

Hard rejection occurs if:

- canonical development is below 80%;
- either held-out family is below 60%;
- any conservation or custody violation occurs;
- rule-card or writeback interventions have weak causal effects; or
- treatment-control separation is below 10 percentage points.

Passing these gates authorizes a separately frozen language-interface transfer
experiment. It does not by itself authorize a claim of genuine general
reasoning.

