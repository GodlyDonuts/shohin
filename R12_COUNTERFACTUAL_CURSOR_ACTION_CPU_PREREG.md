# R12 Counterfactual Cursor-Action CPU Preregistration

**Status:** persistent mechanics board admitted and independently audited. The
optional final-block/head-zero Q-intervention path and separately serializable
192-parameter sidecar pass focused CPU and existing inference regressions. No
score-bearing model fit or GPU job has been run.

**Depends on:** `R12_COUNTERFACTUAL_CURSOR_ACTION_THEORY.md` and
`R12_OPERATION_SELECTION_LIKELIHOOD_RESULT.md`.

## 1. Purpose and allowed claim

This CPU package may establish only that a finite operation-order board is
balanced, causally identifiable against named shortcuts, and capable of
rejecting a broken controller before neural training. It is not evidence that
Shohin reasons or that the proposed finite-state cursor is novel.

The only later neural hypothesis allowed from this package is:

> On untouched operation-order renderers and operands, orbit-interchange loss
> improves exact cursor-conditioned action selection over an
> information-identical ordinary-loss controller with matched state,
> parameters, updates, and compute.

## 2. Frozen symbolic board geometry

The board contains:

```
24 operation-order permutations
x 5 renderer/prefix families
x 5 cursor states (four operations plus DONE)
= 600 cells.
```

Every source contains exactly one clause for each of `add`, `subtract`,
`multiply`, and `remainder`. Inside one renderer, every permutation reuses the
same four clause strings and operands; only clause order changes. Cursor is a
separate field and is never serialized into source text by the CPU board.

Across renderers, start value and the `(operation, operand)` sequence are also
identical. Only syntax and clause wording differ, so renderer-invariance pairs
are content-matched rather than value-matched by assumption.

The five renderers must differ in syntax while retaining an auditable
one-to-one clause map. Renderer IDs, operand tuples, source strings, clause
spans, permutation IDs, cursor values, target actions, and pair memberships are
all serialized. No model output or score may influence generation.

The model-exposure allowlist is exact. Selector training/evaluation may expose
only row field `source` plus the separate internal `cursor` tensor. One-call
evaluation may expose only `source` and initializes side state to
`(cursor=0, phase=SELECT)`. IDs, renderer/permutation metadata, start value,
operation order, clause spans, targets, and target indices are gold-only and
must cause the loader to fail if requested as model inputs.

## 3. Mandatory exact audits

The independent auditor reconstructs every target from source clause spans and
the permutation, without trusting target fields. It must prove:

- exactly 600 unique cells, 120 unique sources, and five cells per source;
- all 24 permutations in every renderer;
- one occurrence of every operation clause per source;
- no source-text difference across the five cursor interventions;
- 120 occurrences of every global target including DONE;
- six occurrences of every operation at each nonterminal cursor within each
  renderer;
- DONE in every and only every `c=4` cell;
- complete five-way cursor interchange groups;
- complete adjacent-transposition and cross-renderer pair maps;
- no duplicate source/cursor key, malformed row, or unregistered field;
- deterministic canonical row ordering and stable SHA-256 hashes.

One mismatch rejects this version. The auditor must be a separate source file
and reconstruct the board rather than accepting self-attested booleans.

## 4. Exact symbolic controllers

Before any learner exists, the auditor evaluates deterministic symbolic arms:

| Arm | Frozen expected score |
|---|---:|
| Oracle source + cursor | `600/600` |
| Global constant | `120/600` |
| Best source-only | `120/600` |
| Best renderer-only | `120/600` |
| Best cursor-only | `240/600` |
| Best renderer + cursor | `240/600` |
| Exact controller with cursor clamped to zero | `120/600` |
| Exact controller with fixed five-cycle derangement | `0/600` |

The source-only ceiling is computed per source, not approximated from global
counts. The cursor-only and renderer-plus-cursor ceilings are solved by exact
enumeration. Unique top-1 is required; ties are failures, not fractional credit.

The collapse test exhaustively checks all 12 `(cursor, phase)`/HALT states
against all eight token-event classes through one-hot transition matrices. It
then constructs the five-state selector cursor and proves exact agreement
between:

1. explicit cursor lookup;
2. a tied finite-state recurrence;
3. a fixed hard pointer into a cursor table; and
4. a clamped positional table when every operation has one fixed-duration
   controller step.

This successful reduction rejects a primitive-novelty claim. It does not reject
the later training-protocol hypothesis.

The frozen implementation surface is:

```
pipeline/generate_counterfactual_cursor_action_board.py
pipeline/audit_counterfactual_cursor_action_board.py
pipeline/test_counterfactual_cursor_action_board.py
pipeline/counterfactual_cursor_action_contract_v1.json
```

The generator and auditor bind the SHA-256 of a separate declarative semantic
contract. The auditor does not import the generator. It reconstructs source order,
targets, spans, pair maps, shortcut ceilings, FSM transitions, and the folded
query projection independently. It binds both canonical and physical-file
board hashes in its report.

## 5. Split contract for a later neural canary

The 600-cell board is a mechanics board and may not become confirmation data.
A later data generator must freeze disjoint development and confirmation
domains before model initialization:

- disjoint renderer templates and lexical paraphrases;
- disjoint operand tuples and starting values;
- all 24 operation orders in every split;
- shared-prefix variable-length schedules of two, three, and four operations
  for the separate DONE/EOS gate;
- exact token audits for operation labels and the future COMMIT marker;
- zero overlap with public eval prompts and existing Shohin training rows under
  normalized exact and preregistered n-gram checks.

The frozen first location is head 0 of the final block, Q-only, with a centered
three-bit code and 192-scalar bias-free sidecar. It is not selected by a score
search. Any later location change is a new version with a new development and
confirmation contract. No layer, head, seed, renderer, threshold, or checkpoint
shopping is allowed.

## 6. Matched neural arms required before H100 authorization

Every learned arm receives byte-identical sources, cursors, labels, batching,
optimizer, number of updates, and initialization seed:

1. **Orbit-interchange treatment:** action CE plus cursor-interchange,
   adjacent-order equivariance, and renderer-invariance losses.
2. **Ordinary-loss control:** identical cursor mechanism and trainable
   parameters, with action CE only.
3. **Relation-sham control:** treatment tensors and coefficients with frozen
   wrong relation pairings.
4. **Source-only control:** equal trainable parameters and compute, with the
   same 192-scalar projection evaluated under one fixed centered cursor code
   for every row; its weights remain trainable and receive gradients.
5. **Favorable cursor-table control:** an unconstrained eight-entry by 64-wide
   explicit cursor table with 512 parameters and the same labels.
6. **Ordinary text-cursor LoRA control:** the same source and semantic cursor,
   with cursor rendered by a frozen textual suffix and a favorable rank-one
   LoRA on the final-head Q slice (`576 + 64 = 640` trainable scalars), trained
   by ordinary action CE. This tests whether conventional adaptation with more
   parameters solves the literal ordinal-copy task without the event sidecar.

All arms must log trainable scalars, retained bits, dtype, source/cache bytes,
examples, oracle calls, training FLOPs or a fixed proxy, inference FLOPs,
sequential token depth, external memory, and external execution. Missing or
unequal resources reject the information-matched comparisons. Treatment,
ordinary-loss, relation-sham, and source-only training FLOP proxies must match
within 1%; the larger cursor-table and text-LoRA arms are favorable ceilings
and report their excess explicitly.

## 7. Frozen selector decision rule

The later confirmation must report both cell accuracy and exact five-action
groups. A treatment GO requires all of:

- at least 95% unique-top-1 cell accuracy on each untouched renderer;
- at least 90% exact five-action source groups, including DONE;
- at least 95% of all 2,400 directed cursor-interchange pairs switch to the
  donor cursor's target;
- at least 95% adjacent-order equivariance separately on all 360 affected and
  540 unaffected cell pairs;
- at least 99% renderer invariance over all 1,200 unordered content-matched
  renderer/cursor pairs;
- at least +10 percentage points over the ordinary-loss control and relation
  sham on exact source groups. The comparison uses 20,000 deterministic paired
  bootstrap replicates with seed `2026071504`, resampling source groups within
  renderer; the simultaneous one-sided 95% lower bound for both differences
  must be strictly above zero;
- constant and deranged cursor ablations within two points of their symbolic
  20% and 0% predictions after conditioning on treatment-correct groups;
- at least `520/704` on the immutable raw atomic executor gate, and no family
  may regress by more than five percentage points from its raw baseline.

A near miss is a NO-GO. It cannot trigger a threshold, seed, renderer, loss
weight, or adapter-location change under this version.

## 8. Separate one-call and halt gate

Even a selector GO does not authorize a reasoning claim. One additional frozen
experiment must start from one source prompt and use one uninterrupted model
call. The model must emit the four correct operation labels in order, produce a
COMMIT event after each completed step, emit DONE at the source-dependent end,
then emit tokenizer EOS. No host component may choose operations, parse prose
to advance the cursor, supply state, repair output, force DONE, or force EOS.

The primary gate is exact operation sequence plus immediate DONE/EOS on every
case. Arithmetic results and carried state are reported separately. If the
selector passes but this gate fails, the result is a learned action policy, not
autonomous reasoning.

## 9. Score-blind custody

The generator, independent auditor, tests, contract, frozen hashes, seeds, and
thresholds must be committed before any score-bearing run. Confirmation results
are written exclusively, fsynced, hashed, and made read-only. A score-free
receipt containing job identity, implementation/data/checkpoint hashes, row and
forward counts, and output SHA-256 must be mirrored and committed before the
score-bearing artifact is opened.

The protected flagship output path and checkpoints are read-only inputs. No
canary may share its output directory or modify the live data stream.

Persistent board generation additionally refuses a dirty implementation
surface. The board records the pre-board Git commit and SHA-256 of the theory,
preregistration, declarative contract, generator, auditor, and tests. The
auditor verifies every hash against both the live file and `git show` at that
commit. A board generated before that clean commit is inadmissible.

## 10. Mechanics and implementation result

The immutable board contains 600 cells, 120 sources, 180 adjacent-order pairs,
and 24 renderer groups. The independent audit reproduced the frozen symbolic
scores exactly:

| Arm | Exact result |
|---|---:|
| Oracle source + cursor | `600/600` |
| Cursor-only | `240/600` |
| Renderer + cursor | `240/600` |
| Source/global/renderer/clamped | `120/600` |
| Deranged cursor | `0/600` |

It also passed all 96 finite-state transition assertions and all 320 exact
query-folding assertions. Board SHA-256 is
`02a202070efa45f14c4e53b7d7f532d98791c7eef9daf438b02d31cc0ec6ab95`;
audit SHA-256 is
`c64951a1369b3dd29ca7e651840e5644e8445c7236cf994c3e54f05ca4a844b2`.

`train/model.py` now accepts an optional Q delta at one named layer/head. With
the argument omitted, old checkpoints still load strictly and the original
path is unchanged. `train/counterfactual_cursor_action.py` supplies the frozen
event FSM and a zero-initialized centered-three-bit projection. The production
projection has exactly `3 * 64 = 192` trainable scalars; all base parameters are
frozen. Six focused tests cover strict-load/zero-delta parity, code geometry,
event transitions, prompt-boundary alignment, gradient isolation, and cached
decode/full-replay equivalence. Existing causal-KV, batched-generation,
recurrent-inference, and masked-loss regressions also pass.

This is a mechanics result only. It does not show that Shohin uses the cursor,
selects the correct action, executes arithmetic, carries state, or halts. A
neural run remains forbidden until the matched-arm data generator, immutable
split hashes, loader allowlist, and score-blind result receipt are frozen.
