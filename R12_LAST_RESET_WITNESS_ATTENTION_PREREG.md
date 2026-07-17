# R12 Last-Reset Witness Attention Calibrated Exploratory Rejection

**Protocol:** `R12-LRWA-CPU-v2-calibrated`

**Status:** **CALIBRATED_EXPLORATORY_REJECTION.** This is not an outcome-naive
preregistration. Implementation and outcome calibration occurred before this
contract was frozen. The mechanics and negative learning outcome are retained
only as an exploratory audit. They must never be represented as preregistered,
canonical, prospectively frozen, promotion-grade, or independent evidence.

The legacy filename ends in `_PREREG.md`; that filename is not evidence of
preregistration and confers no timestamp, immutability, or prospective status.
This calibrated contract authorizes only deterministic local CPU replay and
hostile validation. It authorizes no H100 job, Shohin checkpoint mutation,
architecture integration, capability claim, or future promotion decision.

**Decision boundary:** Last-Reset Witness Attention is known decimal
carry-lookahead, last-write retrieval, and reset-monoid attention. It is not a new computational primitive.
A passing finite mechanics board shows only an alternative bounded-depth
implementation of the same endpoint map as serial carry recurrence. The scaled
learned board is calibrated non-promotion evidence. Its negative result rejects
this exploratory factorization on this board. A counterfactual positive value
of the nomination expression would still be a descriptive signal only and
could never authorize a GPU launch, replication claim, or architecture
promotion.

## 1. Calibrated exploratory question

Shohin's locked evidence shows a late digit-bearing residual but unreliable
serialization, carry consumption, and integrated multi-call execution. The
narrow question here is whether decimal carry/borrow, whose local transition
belongs to a three-element reset monoid, admits a better credit-assignment
topology than serial recurrence when:

- models receive raw operation and local decimal-digit inputs;
- no host-computed `K/P/G` status is a learned-model input;
- no generated token or generated-token KV entry is consumed as state;
- no result tape, intermediate answer digit, host ALU result, or external
  schedule is provided;
- treatment, serial recurrence, and generic dense attention have exactly the
  same trainable parameter count; and
- useful FLOPs are approximately matched and reported rather than inferred
  from wall time.

The finite oracle mechanics board may name `K/P/G` statuses because it is a
separately labeled algebra audit, not a learned-model interface. The scaled
learning board receives no status labels or status inputs; it is trained only
from endpoint classes generated offline from raw digits.

## 2. Exact reset monoid

For incoming carry or borrow bit `c`, define:

```text
K(c) = 0
P(c) = c
G(c) = 1
```

For addition, raw digits `(a,b)` map to:

```text
K when a+b <= 8
P when a+b == 9
G when a+b >= 10
```

For subtraction, where the state bit is borrow, they map to:

```text
K when a-b >= 1
P when a-b == 0
G when a-b <= -1
```

These statuses are oracle labels on the finite mechanics board only. A learned
compiler must infer any useful partition from raw operation/digit features.

Given a status word `s_0...s_(q-1)` and initial bit `c_0`, serial recurrence is:

```text
c_(i+1) = s_i(c_i)
```

The witness endpoint is:

```text
j* = max {j < q : s_j is K or G}
c_q = 0 if s_j* is K
c_q = 1 if s_j* is G
c_q = c_0 if no such j exists
```

The last reset overwrites every earlier state and all later `P` events preserve
it. Therefore witness and serial endpoints must be bit-identical on every
`K/P/G` word. A mismatch is an implementation error, not evidence of greater
expressivity.

## 3. Exact deployment parameter budget

The deployment-side candidate and both controls use the same compiler and
motor tensor shapes, including biases.

### Compiler `C`: `1153 -> 4096 -> 4096 -> 4`

```text
1153*4096 + 4096 =  4,726,784
4096*4096 + 4096 = 16,781,312
4096*4    + 4    =     16,388
compiler total         21,524,484
```

### Motor `M`: `1154 -> 1024 -> 12`

```text
1154*1024 + 1024 = 1,182,720
1024*12   + 12   =    12,300
motor total           1,195,020
```

### Strict cap

```text
frozen Shohin base       125,081,664
compiler                  21,524,484
motor                      1,195,020
added                     22,719,504
system total             147,801,168
strict cap               150,000,000
remaining                  2,198,832
```

The total is strictly below 150M. Tied embeddings in the base are counted once.
The candidate, ordinary recurrent control, and generic dense-attention control
each receive exactly `22,719,504` added trainable parameters. No arm receives a
filler tensor, dead parameter, private output head, extra positional table, or
unmatched optimizer state.

### Deployment arm semantics

- **Witness treatment:** compiler outputs are interpreted as `K/P/G/PAD`
  logits; a hard in-model last-reset retrieval selects the carry/borrow for a
  genuine late source query; the motor consumes current raw-source residuals
  and that one bit.
- **Serial recurrent control:** the identical tensors are deployed through the
  serial update `c=(1-g)c+gv`. It receives the same source residuals, examples,
  labels, precision, optimizer schedule, and motor.
- **Dense-attention control:** the identical four compiler outputs are used as
  generic dense routing keys/queries/values; the same motor predicts the same
  12 endpoint classes. There are no extra attention parameters.

Candidate-trained compiler and motor weights, when deployed through the exact
serial endpoint evaluator with oracle-hard statuses, must be bit-identical on
the finite board. This is an equivalence gate, not a performance claim.

## 4. Fixed finite mechanics boards

All raw case records are retained in the structured JSON report. Every summary
must be independently reconstructed from those records, including recomputing
the oracle transition from raw inputs. Editing a summary and its content hash
without editing all consistent raw evidence must fail validation.

### 4.1 Exhaustive reset words

Exhaust:

```text
word alphabet: K, P, G
word lengths: 1 through 10
initial carry/borrow: 0 and 1
query positions: 0 through word length, inclusive
word cases: 177,144
query observations: 1,860,042
```

Witness and serial traces must agree on all `1,860,042` observations.

Fixed newline-delimited raw-case serialization commitment:

```text
sha256 c5ca6f3ddb6a5192c37527240424563891c0b092ec96593353de9805bda983e7
```

### 4.2 Exhaustive raw local cells

Exhaust:

```text
2 operations * 10 a digits * 10 b digits * 2 incoming bits = 400 cells
```

For each case, independently recompute the finite-board status, output digit,
and outgoing carry/borrow. Status is retained only in this finite mechanics
section.

```text
sha256 d4fd99e9deae46c6d32ac993a0236cebf7d896d8316bf018d55b9f5a0baf076b
```

### 4.3 Toggle-event negative control

Add a fourth transition only to the negative-control algebra:

```text
T(c) = 1-c
```

The candidate has only `K/P/G` transition classes. The complete two-input
truth table evaluates every fixed alias of `T` to `K`, `P`, or `G`:

```text
K alias accuracy = 50%
P alias accuracy =  0%
G alias accuracy = 50%
best reset-only candidate <= 75%
four-function recurrent control = 100%
```

This is a functional-class negative control. No training result can reinterpret
a fixed reset action as a toggle on both inputs.

```text
sha256 620abd2eec5eff480d42be47ebd04e563a34e28f15d693f7d7c9de826b328461
```

### 4.4 Fixed interventions

The mechanics package retains raw evidence for:

- deterministic position reversal on every `K/P/G` word of lengths 2 through
  8 and both initial bits;
- deterministic gate rotation and reset-value rotation/flip on the same board;
- 400 raw-digit `K` versus `G` donor swaps across both operations and every
  suffix cell;
- 1,200 same-status raw donor shams;
- 56 earlier-reset changes shadowed by a later reset; and
- 50 generated-prefix corruption shams, with generated text absent from the
  witness function signature.

Required gates:

- position, gate, and value perturbations are each non-vacuous;
- all 400 K/G donor cases recompute and selectively change the suffix endpoint;
- all 1,200 same-status shams are invariant;
- all 56 shadowed-witness shams are invariant; and
- all 50 generated-prefix corruptions are exactly invariant.

```text
sha256 6801acdd5e4319e439f8b776d5d6e431e6dc6e5b1c00d858b28b4a2177326463
```

## 5. Calibrated bounded CPU learning board

This board is intentionally small and is not Shohin evidence. Its design and
outcome were calibrated before this contract was frozen. It is retained to
make the negative result reproducible and mutation-resistant, not to establish
prospective evidence. It cannot authorize an H100 job, a fresh replication, or
architecture promotion even if every counterfactual score gate passes.

### 5.1 Raw inputs and endpoint labels

Each source row contains only:

```text
operation one-hot (2)
a digit one-hot (10), or all-zero at terminal
b digit one-hot (10), or all-zero at terminal
terminal flag (1)
constant one (1)
```

The model also receives the initial carry/borrow bit and the genuine query
position mask. It does not receive `K/P/G`, intermediate carry, answer digits,
generated text, generated-token KV, a host-computed schedule, or a result tape.

Offline dataset construction uses decimal arithmetic to produce one of 12
endpoint classes:

- classes `0..9`: queried source-column output digit;
- class `10`: terminal carry/borrow zero;
- class `11`: terminal carry/borrow one.

Host arithmetic is allowed only in offline label generation and finite
mechanics verification. The learned forward pass has zero host ALU calls.

### 5.2 Scaled matched models

All three arms use the same tensors:

```text
raw encoder: 24 -> 32 -> 32
router:      32 -> 4
motor:       33 -> 12
trainable parameters per arm: 2,396
```

All tensors participate in each arm's forward graph. Same-seed arms begin from
byte-identical state dictionaries. No unused parameter padding is permitted.
The static multiply-add/routing operation-count heuristic at width 32 has
maximum to minimum ratio at most `1.01`; the current formula gives
approximately `1.001`. This is not an executed-graph measurement, profiler
result, hardware FLOP measurement, or wall-clock measurement. It may be used
only in the calibrated exploratory decision expression.

### 5.3 Data, optimization, and commitments

```text
data seed:                 20,260,717
train examples:                 1,536
train widths:                     4, 8
evaluation examples/width:         256
evaluation widths:           4, 8, 16, 32
model seeds:             1701, 1702, 1703
updates/arm/seed:                   96
batch size:                        128
AdamW learning rate:             0.003
AdamW weight decay:             0.0001
precision/device:          float32 / CPU
```

Split commitments:

```text
train  f1f2f4a1e4d15425a5b6d323813c99f4af5de4f17ca15dbe1f716433a3c355bd
w4     986bb8c02cf9ca4e3b16bb59b7e125d4f6d55af634b506891108d27f25dc1b79
w8     8154eb246435c8b746224a37906b39f802397f21f0f11c34636e50b9f80ef425
w16    f74e70d420e42d8be4ba9091b133233ce8410f0eec08a694417896c97849d69f
w32    2b2d8f1fbc207f39d04e5b8eccaf2c7a6de0b541adc1c3c5b5a5a6f883bd573a
```

Any mismatch aborts before training or report publication. All 9,216 raw
evaluation rows `(arm,seed,width,example_id,target,prediction)`, all loss
traces, initial/final model hashes, and per-width outcomes are retained. Scores
are independently recomputed from raw rows; duplicate or missing identities
fail closed.

### 5.4 Calibrated exploratory decision expression

Primary width is 32. The counterfactual exploratory nomination expression is
true only if all of:

1. witness median endpoint exactness at least 95%;
2. witness-minus-serial at least 15 percentage points on at least two of three
   fixed seeds;
3. dense attention not within two percentage points of witness;
4. serial recurrence does not win or tie witness; and
5. exact parameter matching and static heuristic operation-count ratio at most
   1.01.

If dense attention is within 2pp, reject the mechanism-specific factorization
claim. If serial wins or ties, reject the optimization claim. If all arms are
weak, retain only the finite mechanics identity and close the learned board.
The exact expression must include both exact parameter matching and the
heuristic ratio `<=1.01`; neither may be inferred from a reported decision
string. On the calibrated data the expression is false and the decision is
`CALIBRATED_EXPLORATORY_REJECTION`. Even if it were true, its only allowed
label would be `CALIBRATED_EXPLORATORY_NOMINATION_SIGNAL_ONLY`; it would not
authorize replication, H100 use, Shohin integration, or any prospective claim.

## 6. Hostile equivalence audit

The report and any interpretation must retain all of these statements:

1. Last-reset witness retrieval is known carry-lookahead and reset-monoid
   factorization, not a novel reasoning primitive.
2. With correct hard statuses, its endpoint is exactly serial recurrence.
3. A generic dense attention or parallel prefix has the same bounded-depth
   resource advantage. If dense closes within 2pp, the specific factorization
   claim dies.
4. A false late reset can hijack the endpoint; local status errors still
   compound with length.
5. The observed DRS `+31` residual effect at one late position does not prove
   every source column exposes a learnable status feature.
6. Adding `T(c)=1-c` destroys the last-reset theorem. A generic transition
   product or recurrence is then required.
7. Any host operand parser, host-computed status, host schedule, prompt carry,
   generated-token state, or answer-bearing tape is automatic rejection.
8. A score increase without raw causal donor selectivity and sham invariance is
   not evidence for the proposed mechanism.

## 7. Structured calibrated report and fail-closed publication

The versioned calibrated report:

- uses sorted, ASCII, no-NaN compact JSON with a terminal newline;
- contains complete raw mechanics and learning-evaluation evidence;
- independently regenerates mechanics, splits, row identities and targets,
  medians, per-seed deltas, decisions, losses, predictions, and model states;
- binds raw boards, splits, model states, loss traces, report content, and the
  current source/prereg/test bytes with SHA-256;
- rejects duplicate, missing, reordered, malformed, or inconsistent evidence;
- rejects any scientific, authorization, resource, budget, protocol, status,
  claim-boundary, hyperparameter, or gate mutation after self-hash refresh;
- is written with exclusive create, fsync, exact byte reopen, and local mode
  `0444`;
- refuses overwrite; and
- records `gpu_launch_authorized=false` and
  `architecture_promotion_authorized=false` unconditionally.

The current three-file SHA-256 bindings establish only which local bytes built
and validated the report. They do not establish a trusted timestamp, source
immutability, external attestation, or historical ordering. Mode `0444` is a
local publication property, not an immutable-artifact claim.

The finite mechanics report may use host arithmetic because it is an auditor.
The resource claim is specifically zero host arithmetic calls during learned
model forward/inference.

## 8. Locked outcome language

Allowed after a valid finite pass:

> Last-reset witness retrieval is mechanically equivalent to serial carry on
> the fixed reset-monoid board and has the expected toggle-event limitation.

Allowed description of the calibrated negative outcome:

> On the calibrated bounded CPU endpoint board, witness accuracy was weak and
> generic dense attention closed within two percentage points. The exploratory
> factorization is rejected on this board.

Forbidden:

- "new reasoning primitive";
- "Shohin reasons";
- "autonomous arithmetic";
- "architecture proven";
- "SoTA"; or
- "preregistered result";
- "canonical evidence";
- "outcome-naive test"; or
- any GPU, parameter-scale, natural-language, or general-reasoning claim.

No H100 launch is part of this protocol.
