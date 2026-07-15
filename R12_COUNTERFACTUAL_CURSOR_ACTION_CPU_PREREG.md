# R12 Counterfactual Cursor-Action CPU Preregistration

**Status:** implementation candidate complete for the final clean-commit freeze.
The symbolic mechanics, disjoint neural generator/auditor, typed exposure
loader, six-arm trainer, score-blind evaluator, independent scorer, and focused
mutation tests pass locally. No persistent neural canary, score-bearing model
result, fit, or GPU job has been run under this version.

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

The neural-canary geometry is exact:

| Split | Renderers | Operand packs | Sources | Cells |
|---|---:|---:|---:|---:|
| train | 6 | 8 | 1,152 | 5,760 |
| development | 2 | 4 | 192 | 960 |
| confirmation | 5 | 8 | 960 | 4,800 |

Every renderer/pack combination contains all 24 operation permutations and
every source contains all five cursor interventions. Confirmation contains
192 five-renderer content groups and 1,440 canonical adjacent-transposition
pairs. Development is an integrity diagnostic only; it may not select a seed,
loss weight, threshold, adapter location, epoch count, or checkpoint.

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
   explicit cursor table with 512 parameters and the same labels. Five entries
   (320 scalars) are active in the selector canary and three entries (192
   scalars) are inactive future-state capacity; both counts are reported.
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

The first fit is fixed to raw `best_step260000.pt`, step `260000`, SHA-256
`91d5288f184fc5230516add9851ac1a8815d3369ffd816cd7d0c03d8bafc741d`.
The seed is `2026071506`. Each arm receives four epochs over the same 288
canonical relation units: 1,152 optimizer updates, 60 rows per update, and
69,120 repeated row presentations total. The unit graph exposes every one of
the 5,760 unique train cells exactly three times per epoch.

The optimizer is AdamW with learning rate `0.01`, 50-update linear warmup,
cosine decay to `0.1` of peak, betas `(0.9, 0.95)`, epsilon `1e-8`, zero weight
decay, and gradient clipping at 1.0. Base weights are frozen. The frozen prefix
is cached only through the block before the final block; the trainable path
then executes the exact final block and tied full-vocabulary output projection.
Action CE is full-vocabulary CE. Relation terms use the five preregistered
action-token logits only after subtracting each row's mean, so they constrain
relative action evidence without changing under an arbitrary common logit
offset. Cursor interchange uses a unit donor-target margin; adjacent and
renderer terms use centered-logit mean squared error. The relation-sham arm
uses the same graph and coefficient with every cursor relation rotated locally
by exactly `+1 mod 5`. The ordinary-loss and source-only arms still execute the
same relation graph with coefficient zero. Thus treatment, ordinary-loss,
relation-sham, and source-only have identical fixed compute proxies and shared
192-scalar initialization; the manifest must prove both facts before any arm is
evaluable.

## 7. Frozen selector decision rule

The later confirmation must report both cell accuracy and exact five-action
groups. A treatment GO requires all of:

- at least 95% unique-top-1 cell accuracy on each untouched renderer;
- at least 90% exact five-action source groups, including DONE;
- at least 95% of all 19,200 directed cursor-interchange pairs switch to the
  donor cursor's target;
- at least 95% adjacent-order equivariance separately on all 2,880 affected and
  4,320 unaffected cell pairs;
- at least 99% renderer invariance over all 9,600 unordered content-matched
  renderer/cursor pairs;
- at least +10 percentage points over the ordinary-loss control and relation
  sham on exact source groups. The comparison uses 20,000 deterministic paired
  cluster-bootstrap replicates with seed `2026071504`, resampling the 192
  content-matched pack/permutation groups and carrying all five renderers in a
  sampled cluster together; the simultaneous one-sided 95% lower bound for
  both differences must be strictly above zero;
- constant and deranged cursor ablations within two points of their symbolic
  20% and 0% predictions after conditioning on treatment-correct groups;
- at least `520/704` on the immutable raw atomic executor gate, and no family
  may regress by more than five percentage points from its raw baseline.

A near miss is a NO-GO. It cannot trigger a threshold, seed, renderer, loss
weight, or adapter-location change under this version.

Full-vocabulary unique-top-1 is the primary selector decision. Restricted
five-action unique-top-1 is reported as a diagnostic and may not rescue a
full-vocabulary failure. Relation gates require both the declared relation and
correct endpoint predictions; common wrong answers do not count as
equivariance or invariance.

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

The six arms train serially into one exclusive read-only output tree. A
read-only training manifest binds the base, canary, audit, tokenizer,
implementation commit, every adapter artifact hash, every initial/final state
hash, update count, parameter count, relation coefficient, and fixed compute
proxy. Evaluation refuses an adapter unless all six artifacts exist, re-hash
to the manifest, and the four information-matched arms have one initialization
and identical compute ledgers.

The evaluator receives confirmation prompts and cursor interventions but no
gold action. It emits only full-vocabulary top-1 metadata and the five frozen
action logits under canonical, clamped-zero, and deranged-cycle conditions.
Each arm writes a separate exclusive read-only raw artifact and score-free
receipt. All six receipts and their exact SHA-256 values must be mirrored and
committed before the independent scorer is invoked. The scorer imports no
evaluator code, re-hashes the base, manifest, adapters, canary, audit, live
inference code, raw artifacts, and receipts, then applies the frozen full-vocab
decision and 20,000 paired bootstrap replicates. A selector pass still records
the atomic executor and one-call DONE/EOS gates as pending and never authorizes
a reasoning claim.

Persistent canary generation additionally refuses a dirty implementation
surface. The canary records the pre-generation Git commit and SHA-256 of the
theory, preregistration, declarative contract, generator, auditor, loader,
objective, adapter factory, trainer, evaluator, scorer, jobs, and focused
tests. The auditor verifies every hash against both the live file and `git
show` at that commit. A canary generated before that clean commit is
inadmissible.
