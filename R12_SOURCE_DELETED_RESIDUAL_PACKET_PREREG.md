# RSP-C1: Source-Deleted Residual Packet Control

**Status:** frozen conditional contract on 2026-07-15 before the 256-case
source-scheduled confirmation result was available and before any RSP board,
training data, fit, or score existed.

RSP-C1 is authorized to generate a board, acquire data, fit, or evaluate only
if `R12_SOURCE_SCHEDULED_REASONING_CONFIRMATION.md` reports
`advance_to_internalization=true` and an independent recomputation agrees with
every locked score and integrity gate. A near miss closes RSP-C1 without a fit.

This is a bounded arithmetic controller experiment. It is not an R12 novelty
claim, a general reasoning claim, a latent-reasoning claim, or evidence of a
new computational primitive.

## 1. Locked inputs and motivation

The prerequisite confirmation is bound to:

- board SHA-256
  `19a84165f15b19911fc8ef229022e47753833d703d77d1e8cc25db9dfc993474`;
- canonical cases SHA-256
  `4afc6c4b0c271ea2f723078ab183e8d1ac1851fd1728898384ef52275887b0e4`;
- raw-260k checkpoint SHA-256
  `91d5288f184fc5230516add9851ac1a8815d3369ffd816cd7d0c03d8bafc741d`;
- tokenizer SHA-256
  `87532df5c121753de3b29194e1f9e3de47986d3f5359548fdf93606773a233d4`.

Development evidence found a renderer-indexed arithmetic executor under
`Problem/Work`: 44/55 independent atomic transitions, 10/20 externally
scheduled model-carried chains, and six of six crossed-state interventions
following the displayed state. A separate exploratory raw-260k interaction on
2026-07-15 correctly wrote equation traces for two fresh three-step programs,
but ignored a requested compact packet and repeated the packet prompt instead
of updating it. The observed split is therefore:

1. a partially learned native arithmetic executor;
2. no reliable source compiler, residual-state interface, or recurrent packet
   updater.

RSP-C1 tests that split directly. It does not teach arithmetic to the
controller.

## 2. Mathematical object and prediction

Let `q` render an initial integer `x` and a finite program
`w = a_1 ... a_L`, where each instruction is `add n`, `multiply n`, or
`subtract n`. Let the canonical residual packet be

```text
State: x
Plan: a_i; ...; a_L
```

The learned compiler `C` maps `q` to the initial packet. The learned updater
`U` receives a packet and an observed executor result `y`, drops exactly the
first instruction, and emits either the next packet or `Answer: y`.

The source-blind runtime reads the first model-authored instruction, renders a
single native `Problem/Work` call to the immutable raw-260k executor `E`, and
transports `E`'s parsed integer to `U`. It performs no arithmetic, planning,
repair, search, ranking, or verification.

For a transition-closed task set, exact execution through every reachable
length follows by induction if and only if:

1. `C` emits the exact initial state and residual program;
2. packets separate behaviorally different residual configurations;
3. the updater commutes with one executor step;
4. the empty residual plan halts and copies the final observed state exactly.

A collision between behaviorally different packets or a reachable failure of
the one-step commutation law creates a finite counterexample. This is the
experiment's falsifiable factorization, not an assertion of extra model
capacity.

If compiler, updater, executor, and halt accuracies are `c_L`, `u`, `e`, and
`h`, the stationary-error prediction is:

```text
P(match external trajectory at length L) = c_L * h * u^L
P(match gold trajectory at length L)     = c_L * h * (u * e)^L
```

The observed length curve must be reported against this prediction. Any fixed
local error eventually destroys long-chain accuracy. RSP-C1 therefore tests a
bounded controller and history deletion, not constant-size universal memory.

## 3. Exact packet and prompt grammar

Allowed operations are the three ASCII forms:

```text
add N
multiply N
subtract N
```

The only valid packet is:

```text
State: N
Plan: OP; OP; ...; OP
```

The compiler prompt is:

```text
Problem: SOURCE
Compile only the execution packet.
Packet:
```

The update prompt is:

```text
Packet:
State: N
Plan: OP; OP; ...; OP
Observed result: Y
Next packet:
```

The updater must emit the packet with `State: Y` and the first operation
removed. After the last operation it must emit exactly:

```text
Answer: Y
```

Leading zeros, signs on positive integers, comments, alternative operation
spellings, extra fields, repeated instructions, or additional non-whitespace
text make the call invalid. No forgiving canonicalization is allowed after a
model call.

## 4. Frozen board generation

Board seed is `2026071503`. Generate exactly 256 unique cases in fixed stratum
order, 64 per stratum. Every intermediate mathematical state must be positive.
Evaluation questions, semantic programs, complete mathematical trajectories,
and final answers are unique.

Training-domain values are:

- initial state: 10 through 99;
- add operand: 2 through 25;
- multiply operand: 2 through 7;
- subtract operand: 2 through 25.

Training lengths are 2, 3, and 4. The operation bigrams `multiply -> add` and
`subtract -> multiply` are absent from training. All other bigrams are eligible.

The four evaluation strata are:

1. `renderer_ood`: length 3, training-domain values, only seen bigrams, and a
   reserved source template absent from training;
2. `value_ood`: length 3, a training source template, only seen bigrams,
   initial state 100 through 299, add/subtract operands 26 through 75, and
   multiply operands 8 through 12;
3. `order_ood`: length 3 or 4, training-domain values and templates, with
   exactly one held-out bigram; 32 cases use `multiply -> add` and 32 use
   `subtract -> multiply`;
4. `length_ood`: length 5, training-domain values and templates, and only
   seen bigrams.

The reserved renderer wording is:

```text
Initialize the value to N. Apply these instructions in order: CLAUSES.
```

Training templates may use only the separately enumerated training renderer
set. The board is generated exactly once, written read-only, and its file and
canonical-row SHA-256 values are hardcoded into every downstream consumer
before training-data generation begins.

## 5. Training corpus and absence of arithmetic supervision

Training seed is `2026071504`; synthetic updater-observation seed is
`2026071505`. Generate exactly 4,096 semantic programs:

- 1,024 length-2 programs;
- 2,048 length-3 programs;
- 1,024 length-4 programs.

Every training semantic program, source, packet, complete mathematical
trajectory, and final answer must be disjoint from the frozen board. Abort
rather than lower a count or weaken a disjointness check.

Each program contributes one compiler row. It also contributes one updater
row per instruction. Updater `State` and `Observed result` integers are sampled
independently of the program arithmetic. Every updater row must satisfy:

```text
Observed result != mathematically applying the first operation to State
```

and neither integer may equal any frozen evaluation answer. Thus the updater
teaches only exact state transport, residual-plan deletion, and halting. No
training completion contains a correct arithmetic transition or the
mathematical final answer of its source program.

The independent admission audit must recompute and require:

- exact row counts and length balance;
- zero malformed compiler or updater rows;
- zero correct arithmetic updater transitions;
- zero duplicate normalized training prompts;
- zero normalized semantic-program overlap with evaluation;
- zero exact source-prompt or packet overlap with evaluation;
- zero shared complete mathematical trajectories;
- zero evaluation-answer occurrences in consumed response fields;
- zero reserved-renderer occurrences in training;
- zero normalized 13-token source n-gram overlap;
- exact completion-prompt token-boundary agreement with `sft.py`;
- exact treatment/sham prompt, response-token-count, supervised-token-count,
  packed-sequence-count, and forward-token-count equality.

## 6. Causally matched compiler arms

Sham-permutation seed is `2026071506`. There are two inferential arms:

1. `treatment`: every source maps to its exact canonical initial packet;
2. `sham`: every source maps to another program's canonical packet under a
   deterministic derangement.

Updater rows are byte-identical in the two arms. Compiler prompts are
byte-identical and only their completions differ.

The generator must create sham strata with no singleton and match all of:

- program length;
- exact operation-type sequence;
- source-template identifier;
- digit-width vector for initial state and every operand;
- tokenized packet length;
- mathematical final-answer digit width.

Every sham mapping must have no fixed point, a different semantic program, a
different complete trajectory, and a different final answer. The independent
audit reconstructs the permutation and rejects any mismatch. It may not trust
a generator-supplied `sham_valid` field.

This sham preserves packet vocabulary, response length, operation locations,
and optimizer exposure while removing correct source-to-number binding. An
ordinary CoT arm is allowed later only as a non-causal capacity reference.

## 7. Frozen fits

Use paired seeds `2026071511` and `2026071512`. Each treatment/sham pair starts
from the exact raw-260k checkpoint and uses the same seed and record order.
The trainer must expose the seed through a command-line argument and persist it
in metadata; hardcoded seed 1337 is not sufficient.

Fit contract:

- full-model completion-masked SFT;
- exact `completion_prompt` used at inference;
- pack length 128;
- batch size 64;
- 10 complete epochs;
- Muon LR `8e-4`;
- Adam LR `2e-4`;
- warmup 50 updates;
- gradient clip 1.0;
- no early stopping, evaluation, or score-dependent selection;
- isolated output directories and no flagship paths;
- treatment and sham must have exactly equal updates, packed forward-token
  positions, and supervised target tokens.

The final epoch checkpoint is claim-bearing. Earlier epoch files are training
telemetry and may not be selected by evaluation.

## 8. Source-deleted runtime

Every call is greedy and starts with a fresh KV cache.

1. A controller call receives the natural-language source and emits one
   packet.
2. The compiler call terminates. The source string, prompt tokens, and KV
   cache are destroyed.
3. A source-blind interpreter receives only the model packet.
4. It parses the first model-authored instruction and renders one atomic
   `Problem/Work` prompt to an immutable raw-260k executor.
5. It parses only the last integer on the first nonempty executor line.
6. A fresh controller call receives only the packet and observed executor
   integer, then emits the next packet or final answer.
7. The loop has at most five executor transitions. Any parse failure,
   noncanonical packet, skipped/repeated operation, extra field, or premature
   halt fails immediately. There is no retry.

The interpreter may perform regex parsing, exact string transport, list-head
selection, and fixed prompt rendering. It may not add, subtract, multiply,
divide, compare candidate answers, retain the source, inspect gold data,
repair output, search, rank, or provide verifier feedback.

The evaluator must also run:

- immutable raw external source scheduling on the same board;
- an oracle-packet controller loop that supplies only the exact initial packet;
- a teacher-forced updater board using source-free packets and synthetic
  observations;
- packet-swap interventions in which the post-compiler runtime follows a
  packet from a different source while the original source remains deleted.

## 9. Raw transcripts and independent scoring

Generation artifacts contain raw prompts, raw responses, token counts, stop
reasons, model/checkpoint/tokenizer hashes, and call order. They contain no
trusted correctness booleans or aggregate success metrics.

Two independently implemented scorers must hardcode and verify every board,
checkpoint, tokenizer, data, audit, and transcript digest. Each scorer reparses
all packets and executor lines, replays the mathematical programs, recomputes
all trajectories and exact two-sided McNemar tests, and agrees on every integer
count and probability to `1e-12`. A self-rehashed substitute artifact is not
admissible.

The append-only resource ledger reports separately by model and arm:

- model calls;
- prompt tokens;
- sampled tokens including sampled EOS;
- decoded tokens;
- supervised completion tokens;
- packed forward-token positions;
- calls not issued after parse failure;
- retries, repairs, searches, and verifier-feedback calls, all fixed at zero.

## 10. Locked metrics and gates

Primary metrics are:

- `compile_exact`: exact initial packet;
- `update_exact`: exact state copy and residual-plan deletion;
- `oracle_packet_loop`: exact loop from a supplied gold initial packet;
- `strict_closed_loop`: exact compile, every update, halt, and final copy;
- `external_trajectory_match`: complete emitted state sequence equals external
  scheduling with the same raw executor;
- `gold_answer`: final state equals the mathematical answer.

RSP-C1 advances only if the immutable prerequisite confirmation passes and
both treatment seeds independently satisfy all of:

```text
raw external scheduler gold answers       >= 128 / 256
oracle-packet exact closed loops           >= 230 / 256
initial compilation                        >= 224 / 256
conditional packet-update accuracy         >= 95%
strict source-deleted closed loops          >= 192 / 256
per-stratum compilation                     >= 52 / 64
per-stratum strict closed loop              >= 40 / 64
final external-trajectory mismatches        <= 8 / 256
treatment - sham compilation                >= 30 percentage points
treatment - sham strict closed loop         >= 25 percentage points
```

Treatment must beat sham in every stratum. In both paired seeds, exact
two-sided McNemar `p < 0.01` is required separately for compilation and strict
closed-loop success. The two independent scorers must agree.

The packet-swap diagnostic requires at least 60/64 complete trajectories to
follow the swapped packet rather than any original-source trajectory.

The measured complete-trajectory length curve must be reported beside
`c_L * h * u^L`. It is diagnostic rather than a tunable gate.

## 11. Interpretation and next boundary

- Failed oracle-packet loops make the experiment uninterpretable: the updater
  did not learn exact recurrence.
- Exact compilers with failed updates identify recurrence as the bottleneck.
- Packet trajectories matching external scheduling but wrong gold answers
  identify the immutable raw executor as the bottleneck.
- CoT success with packet failure would reject the separable-controller
  hypothesis under this scale and budget.
- Treatment and sham both succeeding must be treated as leakage or scorer
  failure until independently disproven.
- Passing RSP-C1 establishes only learned source compilation and compact
  source-deleted state control through a counted external recurrence loop.

One-call internalization is a separate stronger experiment. It cannot inherit
an RSP-C1 claim and must freeze its own board, matched sham, token/FLOP budget,
and exact plan/equation/answer gates before training.
