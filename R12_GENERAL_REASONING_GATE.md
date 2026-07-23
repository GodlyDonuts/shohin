# R12 General Reasoning Gate

**Status:** UROM-3 rejected pre-neural; categorical-phase register machine in
train-only optimization development; no reasoning claim
**Active architecture:** Query-blind Equivariant Relation-Algebra Register
Machine (`QERARM`)
**Retired negative control:** Uniform Relational Object Machine (`UROM-3`)
**Protected base:** Shohin raw pretrain step 300,000
**Base SHA-256:** `211d6b2cddf0c2cf8b12cb0b2d73f9c4440d85f6f531018080c8afd35b2f66a6`
**Strict system limit:** fewer than 200,000,000 unique parameters
**Last updated:** 2026-07-23

## 1. Objective

The objective is not to make Shohin imitate visible chain-of-thought text. It is
to give Shohin a model-owned mechanism that can:

1. infer episode-local rules and bindings from a source it has not memorized;
2. commit those rules to a private object file;
3. lose access to source tokens, residuals, and KV state;
4. update private state over multiple dependent steps;
5. answer a query disclosed only after execution;
6. transfer the same mechanism across unseen names, renderers, rules,
   cardinalities, lengths, program topologies, and task families; and
7. respond causally to rule, state, order, and query interventions.

A finite synthetic board cannot prove unrestricted intelligence. The promotion
claim is therefore narrower and falsifiable: **one resource-bounded,
source-deleted mechanism exhibits systematic transfer beyond every finite
label, renderer, rule, length, topology, and task-family table available in
training.**

## 2. Why Existing Results Are Insufficient

The project has established useful components, but no previous result meets the
objective:

- S7 composed unseen laws perfectly inside a fixed cyclic topology.
- S9.1 compiled bounded occurrence graphs well but missed invariance gates.
- SD-CST v1.3 achieved fresh-renderer source-deleted execution inside a fixed
  three-object ontology.
- ER-CST achieved high witness-equality accuracy but enumerated finite `S3`
  cards.
- ER-TT removed the card enumeration and made execution exact, but neural
  packet compilation failed.
- S4-TPT supplies noncommutative dynamic binding mechanics but consumes
  host-materialized semantic tensors.

The unresolved problem is the seam between language and private execution:
compile a new object system faithfully, delete the source, then operate on the
compiled objects without a host parser, answer packet, or external scheduler.

## 3. Retired UROM-3 Architecture

`train/general_relational_object_machine.py` implements the first unified
mechanics slice. Independent hostile review rejects it as a reasoning
candidate: its executor is a fixed bounded Boolean-relation VM and its
apparently different task families do not require different algorithms.
It remains only an audited compiler/executor negative control.

```text
program source
  -> frozen Shohin residuals
  -> occurrence decoder
  -> source-value identity carriers
  -> DeletedRelationalProgram
  -> source/residual/KV deletion boundary
  -> shared relational recurrence
  -> terminal relational state

late query source (disclosed after terminal commitment)
  -> frozen Shohin residuals
  -> DeletedRelationalQuery
  -> model-owned relational reader
  -> answer distribution
```

### 3.1 Private Object File

The executor accepts only:

- episode cardinality;
- initial object-state relation;
- episode-local rule cards;
- rule-active bits;
- event-to-rule bindings; and
- event kinds (`APPLY`, `STOP`, `NOOP`).

It does not accept source IDs, source masks, source memory, pointer logits,
identity carriers, parser spans, family IDs, targets, final states, answers,
verifier output, or retry feedback.

The hard score-bearing object file is 649 categorical bytes per row at the
current maximum geometry.

### 3.2 Dual Occurrence/Identity Compiler

The compiler uses decoder slots to locate occurrences, but episode identity is
carried only by a weighted read from source values. Slot embeddings therefore
cannot directly encode an opaque entity or rule name.

Relations are constructed by comparing source-derived carriers:

- initial occurrences against declaration occurrences;
- up to 24 source/destination edge occurrences per rule against declarations;
  an edge-active head forms their differentiable union; and
- each event opcode against episode rule opcodes.

This avoids the ordinal branch's exact raw-byte equality oracle.

### 3.3 Shared Executor

A relation card is a binary many-to-many map over episode-local objects. One
recurrent operation, Boolean relation composition, supports all proposed
families:

```text
next_state[i,k] = OR_j(selected_relation[i,j] AND current_state[j,k])
```

`STOP` transfers live state into a persistent halted state. Later events cannot
modify halted mass. The late reader selects a terminal position only after the
terminal state has committed.

There is no Python branch controlled by a semantic rule value, no
generated-token feedback, and no retry/repair loop. The recurrent update is
nevertheless a fixed host-authored PyTorch relation-composition algorithm. It
is not a learned model-owned state-update law, so it cannot establish the
target capability.

### 3.4 Exact Production Parameter Ledger

The ledger was instantiated against the real immutable 300k checkpoint:

| Component | Unique parameters |
|---|---:|
| Frozen Shohin trunk | 125,081,664 |
| UROM compiler and object heads | 13,323,046 |
| **Complete system** | **138,404,710** |
| **Headroom below 200M** | **61,595,290** |

The compiler is the only trainable component in this first slice. The
relational executor and reader are tensor architecture, not host-side semantic
code and not separately learned answer tables.

## 4. UROM-3 Board Rejection

The labels `episodic_transport`, `graph_agenda`, and `constraint_dataflow`
change relation distributions and surface renderers, but each target is the
same matrix product. They therefore do not constitute task-family transfer.
The initial split also coupled family and cardinality, and the late reader
could not identify an opaque query name from the query alone because no
declaration dictionary crossed the deletion boundary.

The board generator now factorizes family/cardinality cells and the hard
runtime rejects out-of-cardinality state and queries. Those repairs preserve a
valid negative control but do not reopen UROM-3 for neural training.

## 5. Split Contract

Every semantic world receives independent random relations and opaque names.
Canonical world hashes and graph-isomorphism hashes must be disjoint. Renderer,
length, and answer distributions must be balanced within semantic orbits.

| Split | Cardinality | Rule depth | Program length | Topology |
|---|---:|---:|---:|---|
| Train | 4-6 | 1-3 | 2-8 | chains and shallow forks |
| Development | 7 | 4-5 | 9-16 | diamonds, nested joins, simple cycles |
| Confirmation | 8 | 6-8 | 17-32 | strongly connected, repeated, nested, hybrid |

The current architecture has a strict maximum cardinality of eight. Promotion
beyond G2 requires a successor with shape-polymorphic cardinality, not merely a
larger fixed maximum.

Each development and confirmation split must contain:

- a fresh-rule-only stratum;
- a fresh-renderer-only stratum;
- a fresh-length/topology-only stratum;
- a fresh-family-composition stratum; and
- an all-axes-at-once stratum.

## 6. Matched Arms

1. **Structured treatment:** Shohin compiler plus relation-tied object machine.
2. **Favorable dense control:** identical compiler, source, labels, updates,
   state width, recurrence slots, and at least as many trainable parameters;
   relation composition is replaced by unconstrained learned transitions.
3. **Family-specialized control:** separate executors with the same aggregate
   parameter budget.
4. **Finite motor control:** packet/prefix lookup under the same parameter and
   object-file bit budget.
5. **Oracle ceilings:** gold object plus shared executor, and predicted object
   plus gold executor. These localize failure and never count as reasoning.

A structured-versus-dense comparison is invalid unless the dense arm reaches
at least 99% training and 95% in-distribution joint accuracy.

## 7. Causal Tests

Required interventions:

- entity, relation-storage, register, and event-node reindexing;
- complete alpha-renaming and renderer paraphrase;
- wrong-law substitution with a separately calculated counterfactual;
- relation-card, intermediate-state, and terminal-state transplantation;
- state reset, relation deletion, binding deletion, and event-order reversal;
- equivalent commuting programs versus noncommuting order twins;
- source, residual, and KV poisoning after commitment;
- late-query rotation with terminal-state invariance;
- post-`STOP` suffix mutation, forced-alive, and early-stop tests; and
- type-compatible relation-card transplantation across task families.

## 8. Promotion Gates

### G0: Architectural Custody

- complete system below 200M by unique-parameter identity;
- executor interface contains no source or pointer evidence;
- source-deleted hard rollout is bit-invariant to post-seal source mutation;
- exact relation composition, halt, late-query, gradients, and interventions;
- independent CPU implementation agrees on every exhaustive small case.

### G1: Single-Family Systematic Transfer

- gold-object executor at least 99.5%;
- predicted object, trajectory, halt, terminal state, and answer scored
  separately;
- at least 90% joint accuracy on unseen rules, renderers, and lengths for each
  family in isolation;
- every deletion and post-`STOP` gate passes at 100%.

### G2: Cross-Scale And Cross-Topology Transfer

- at least 85% joint accuracy on all-axis held-out cases;
- at least 95% noncommuting order-twin separation;
- at least 99.5% equivalent-program invariance;
- treatment exceeds a qualified dense control by at least ten points.

### G3: Shared Multi-Family Execution

- at least 85% joint accuracy per family;
- at least 90% macro average;
- at least 75% on unseen hybrid programs;
- one executor and object schema, with no renderer- or family-specific head;
- five independent seeds pass individually.

### G4: Natural-Language Transfer

- post-training examples teach interface use without teaching confirmation
  answers or rules;
- interactive transcripts show internally consistent multi-step state use;
- public reasoning benchmarks improve over the frozen 300k base and over a
  parameter-matched post-training control;
- causal internal-state interventions predictably alter natural-language
  answers.

Only passing G0-G4 supports a claim of genuine general reasoning for Shohin.
Passing G0-G3 supports a narrower claim of systematic relational reasoning.

## 9. Current Evidence

As of 2026-07-23:

- the real 300k frozen checkpoint loads with the required immutable hash;
- the UROM compiler attaches without modifying the trunk;
- the complete system is 138,404,710 parameters;
- the combined UROM/QERARM mechanics suite passes 33 focused tests;
- hard two-rule composition is exact;
- arbitrary many-to-many Boolean relation composition is exact;
- post-`STOP` suffixes are inert;
- changing only the late query changes the answer but not terminal state;
- state transplantation has the predicted causal effect;
- gradients reach every soft object-file field and the late query; and
- sealed execution is invariant to mutation of the original soft compiler
  outputs.

UROM-3 is **rejected before H100 use**. Its mechanics are retained because a
clean negative control is useful, not because more compiler optimization is
expected to turn fixed relation composition into general reasoning.

## 10. Active Successor: QERARM

`train/equivariant_relation_register_machine.py` implements the current
falsifier. A source-deleted packet contains only cardinality and six relation
registers: raw relations `A`, `B`, identity, and three empty writable
registers. The late query is absent during execution. A learned,
object-permutation-invariant controller selects operation, operands,
destination, phase transition, and `HALT`.

Every candidate operation is evaluated tensorially:

- composition, union, intersection, difference, converse, copy, clear,
  identity, and fixed-point expansion;
- only registers 3-5 are writable;
- categorical phase is model state, not a host program counter;
- missing halt is invalid and remains in the denominator; and
- packet state outside the declared object square is rejected.

The current separator is `TC(A) \ TC(B)`. Difference is antitone in `B`, so a
monotone union/closure machine cannot solve it. The development board changes
both graph size and required fixed-point depth: training uses cardinalities
3-5 and depths 2-4; development uses 6-7 and depths 5-6; confirmation is
reserved at cardinality 8 and depth 7. No operation schedule, closure,
trajectory, halt time, or answer-equivalent field enters the machine.

The active controller receives both normalized action-change mass and a
scale-free maximum-change signal, preventing fixed-point decisions from
depending on the `1/n^2` magnitude of one new edge. The default 512-wide,
three-layer categorical-phase controller adds exactly 2,829,341 parameters.
With the protected trunk, the complete system is 127,911,005 parameters and
leaves 72,088,995 parameters below 200M.

Four score-free optimizer probes are negative and retained. A fifth
development-only probe is the first successful learned-executor signal:

| Probe | Train joint | Development joint | Diagnostic |
|---|---:|---:|---|
| naive hard | 0% | 0% | immediate halt collapse |
| soft/hard GRU | 0% | 0% | answer shortcut without work state |
| teacher GRU | 38.2813% | 0% | training sequence memorization |
| Markov affordance | 0% | 0% | no phase separation |
| categorical phase, mean-change only | 100% | 94.2708% | residual cardinality-7 convergence errors |

Their report SHA-256 values are, respectively,
`4499c3621422e3b51e72a0cb91d544faba0c53da23d9f3d93c713f87a5068b0d`,
`c64823623f6b56c77ecdd7375eb73be2f2716f1f99beff385aaf8003b430c300`,
`e1908349ce9f22d980650ab4acc3ef881651f48eb4bae26ab41c286987776824`,
and
`48dfeab5ce54265ed4ae9a7bbdad8146fca5da0386d99bb2e85b959e994c63cb`.
The fifth checkpoint/report SHA-256 values are
`39781187bcf0f7a6baeda01fc27890180fdacd6e38aa34c8f918de79c88dcd90`
and
`eff8a0fdd36e2c5f81cf0d3027e9db40c618c8331064c18d1954b54baa0d909e`.
Its 214,589-parameter controller reaches 768/768 train joint and 181/192
development joint. All 11 failures are cardinality-seven rows, principally
misclassified fixed-point transitions. This is a bounded development result,
not a promotion or confirmation result.

## 11. Immediate Work

1. run the scale-free stability revision against the same development-only
   contract;
2. require hard exact work-register, answer, and halt extrapolation rather
   than a low continuous loss;
3. localize every residual action/phase/halt error before opening any new
   split;
4. if it passes development, add query-blind state/action transplants,
   operation ablations, and a parameter/FLOP-matched generic recurrent
   control;
5. only then connect the learned executor to a source-deleted Shohin compiler;
6. keep confirmation sealed until source, board, thresholds, matched controls,
   and independent assessment are frozen.
