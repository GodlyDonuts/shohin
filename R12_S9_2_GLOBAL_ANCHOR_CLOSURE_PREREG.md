# R12 S9.2 Global Anchor Closure Preregistration

**Status:** CPU mechanics and fresh board admitted; no neural score access

**Parent:** S9.1 is closed at 2,025/2,048 = 98.877% exact graph, state,
and answer. All 2,025 emitted graphs are exact. Its 23 failed rows retain no
partial-span diagnostics, so root-anchor failure is a hypothesis rather than an
established diagnosis. Operation recoding also invalidates one graph and changes
two canonical graphs. Confirmation remains sealed.

## Narrow claim under test

S9.2 asks whether finite global assignment over model role logits can replace
S9.1's independent positive-argmax root selection, and whether alpha consistency
over hard negative competitors can remove operation-recode instability. This is
a bounded language-to-graph compiler test. It does not add arithmetic,
recurrence, a search-and-repair loop, or a general reasoning claim.

## Frozen ownership boundary

The model owns every candidate role score and therefore the selected roster
size, card count, event count, root spans, children, entry, successor/nil, and
query. The host exposes only the already declared finite surface grammar:

- roster size is one of 5, 7, or 11;
- card count is one of 2, 3, or 4;
- event count is one through 8;
- root role blocks occur in the declared source order;
- selected spans cannot overlap; and
- one entry and one query are required.

The optimizer must not read candidate targets, row modulus/depth/cards/nodes,
gold spans, exact-byte classes, graph validity, executor output, state, answer,
or retry feedback. Candidate targets are stripped from its inference view.
`compile_quotient` is called at most once after the root and child assignments
are irrevocable. A failed compile is final; no lower-scoring assignment may be
tried.

The fresh board builder resolves `source_commit` to the current clean tracked
Git `HEAD`. Training and evaluation then require that every frozen scientific
runtime path is byte-identical to that ancestor commit. Evaluation separately
hash-binds the base and tokenizer to both checkpoint and board. Before reading
`development.jsonl`, it exclusively creates a deterministic board-hash access
ledger under `artifacts/r12/access_ledgers`; an existing ledger is a permanent
failure. A different output directory or copied board path cannot create a
second development read in the same scientific repository.

## Global assignment theorem

For every grammar hypothesis `(m, c, d)`, form the ordered root template

```text
entity.roster^m, position.roster^m, state.entity^m,
card.operation^c, entry.tag, event.tag^d, query.position.
```

For candidate interval `i` and role `r`, the only neural score is

```text
delta(i, r) = role_logit(i, r) - role_logit(i, none).
```

A deterministic interval Viterbi algorithm finds the maximum summed delta
among ordered, non-overlapping assignments for all admitted templates. Ties are
resolved by stable candidate order and then ascending `(m, c, d)`. The system
abstains unless the winning total is strictly positive. Existing S9.1 local
top-eight child assignment then runs once around the selected card/event roots.

If the gold root assignment is feasible and uniquely outscores every legal
alternative, each gold child tuple uniquely wins its local assignment, and the
winning total is positive, then the emitted graph is exact by the already
admitted S9 quotient theorem. If an alpha recode induces a candidate bijection
and preserves every compared score difference, deterministic argmax commutes
with the recode and the canonical graph is identical. This is a finite MAP
closure theorem, not a theorem of semantic or general reasoning.

## Alpha objective

All arms retain S9.1's aligned positive log-softmax orbit loss. The treatment
adds a coordinate-free competitor term: for every non-`none` role and each
paired original/recoded example, sort the top eight sampled negative
`sigmoid(role-none)` margins and apply mean-squared error between the two sorted
vectors. The full training loss is

```text
weighted role CE + 0.25 * (positive orbit MSE + hard-negative orbit MSE).
```

Sorting makes the negative comparison independent of token-coordinate changes
under retokenization. No candidate identity or gold negative alignment is used.

## Frozen neural architecture and budget

The protected 300k trunk, closed S8.1 initializer, five-layer width-384 S9
encoder, exact-byte occurrence-class mean, relation head, width-four proposals,
S7 generator, and S8 graph/runtime are unchanged. S9.2 adds zero trainable
parameters; the complete system remains exactly 134,580,264 parameters.

Every trained arm receives exactly 24,000 unique sources, original plus recoded
views, 48,000 charged views, batch 64, 750 updates, and 128 sampled negatives
per view. The arms are:

1. treatment: class messages plus positive and hard-negative orbit loss;
2. positive-orbit-only: the exact S9.1 objective under the new board;
3. no-class: equal architecture/budget without class messages;
4. shuffled: one paired-consistent role permutation per source; and
5. layout-only: lexical span tokens masked, class messages disabled, with the
   same architecture, budget, and labels.

Layer 19, width 384, eight attention heads, five encoder layers, FF width
1,408, learning rate `1e-3`, 50 warmup updates, clip 1.0, top-eight hard
negatives, and orbit weight 0.25 are exact constants. The assessor validates
these values plus every arm's unique-source/view/update/batch/negative budget,
class-message mode, orbit mode, and masking mode; an equal parameter count
alone cannot conceal a changed architecture or optimizer.

At evaluation, treatment is decoded by both global S9.2 and frozen local-root
S9.1 decoders using identical logits. Unconstrained S9 decoding is also
reported.

## Required pre-board CPU falsifiers

Before drawing source, board, or training seeds:

1. interval Viterbi equals exhaustive enumeration on at least 10,000 reduced
   synthetic cases;
2. oracle and operation-recoded oracle logits reconstruct all 2,048 closed S9
   mechanics rows exactly;
3. lowering one required gold root below `none` breaks S9.1 root selection but
   is recovered globally on all rows;
4. adding one high-positive spurious root breaks greedy cardinality but not the
   global assignment;
5. uniform logits abstain on all rows;
6. correct counts with flat-positive role logits remain below 10% exact;
7. shuffled roles with correct score distributions remain below 10% exact;
8. every row has at least two distinct complete syntax-valid assignments;
9. a wrong high-margin legal root remains wrong or rejected, never repaired;
10. a wrong high-margin count is followed or rejected, never repaired;
11. poisoning row metadata and candidate targets leaves the selected assignment
    byte-identical;
12. optimizer instrumentation records zero compile/executor calls and the full
    decoder calls compile at most once;
13. hard-negative orbit loss is zero under identical score multisets and
    positive with finite gradients after a single competitor changes; and
14. all deterministic, unit, bytecode, style, and finite-gradient tests pass.

Closed rows are mechanics-only and may not be neurally rescored.

The full falsifier ran once with seed `7509220561492772015`; all 17 gates pass.
Oracle and recoded oracle are 2,048/2,048 exact, flat-positive and shuffled
controls are 0/2,048, every wrong high-margin root/count intervention is
followed without repair, and 10,000/10,000 reduced exhaustive cases match the
Viterbi result. Report SHA-256 is
`91d653c7e2a131ad7e21319dd72a52dee95c00520fbd59771fe5f7a08fe52e24`.

Scientific source commit
`38c934cf9f360e1fd13258c23be310e948cafba1` precedes fresh board seed
`3823077847356570601` and independent training seed `1277007704479652588`.
The admitted board has 48,000/2,048/2,048 train/development/sealed-confirmation
rows, 52,096 executor agreements, zero exact/13-gram/name overlap, and access
`0/0`. Report SHA-256 is
`f22401e82690f8240abe89d6083e3a387619243bc68bb9d9e380540d90b1899e`.
The development and confirmation files remain unopened after generation.

## Sole fresh-development gates

The board is drawn only after source, tests, falsifier, assessor, and launcher
are committed. One development read must satisfy every gate below before any
sealed confirmation access:

- at least 2,031/2,048 exact graph, state, and answer, which is at least a
  0.25-point improvement over closed S9.1;
- every valid emitted graph is exact;
- global decoding strictly beats the same-logit S9.1 local-root ablation;
- root spans and `(m, c, d)` counts are at least 99% exact;
- positive-orbit-only is reported, and the treatment must have zero operation-
  recode graph failures even if its ordinary exactness ties that arm;
- every originally valid row has a valid recoded counterpart with bit-identical
  canonical graph, recurrent state, and answer;
- layout-only, shuffled, uniform, and source-free exact graph are each below
  10%;
- treatment beats no-class exact graph by at least five percentage points;
- inherited storage/class reindex, causal intervention, depth, budget,
  parameter, hash, access, and single-read gates all pass; and
- development/confirmation access is exactly `1/0` after assessment.

The one-read gate additionally requires the immutable access-ledger hash in
the evaluation artifact and cannot be satisfied by changing the result output
directory.

Failure closes S9.2 without rescoring or opening confirmation. Passing permits
one separately frozen confirmation read with unchanged bytes and gates.

## Honest frontier boundary

Even confirmation would establish only robust bounded compilation into the
existing S7/S8 machine. It would not establish arbitrary language, new program
topologies, alias resolution, negation/quotation, self-generated decomposition,
or general reasoning. The next independent stage is a causal grammar firewall:
reordered clauses, same-layout counterfactual bindings, quoted/negated decoys,
argument-order changes, and ontology-word removal with a trained layout-only
control. S9.2 must not be presented as a substitute for that stage.
