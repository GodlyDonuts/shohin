# R12 Vocabulary-Aligned Microcode Transducer Theory

**Protocol schema:** `R12-VAMT-THEORY-v2`

**Status:** THEORY AND RESOURCE HYPOTHESIS ONLY. This document is not an
executable preregistration. It freezes no implementation bytes, data generation,
split, seed, optimizer, threshold, checkpoint, or score. It authorizes no model
fit, H100 allocation, checkpoint mutation, or capability claim.

Schema v1, SHA-256
`55206f603101e982cb91b81a675ef11143dc0b9fc82af0129cc45e298c802ef9`,
was independently rejected. Schema v2 narrows subtraction to nonnegative
results, defines operand spans and fixed in-graph mechanics, completes the
serializer state, charges structured supervision, and repairs the comparator
and resource boundaries. No authorization transfers from v1.

## 1. Decision in one sentence

Shohin should not be asked to emit arithmetic traces as ordinary text and hope
that composition emerges. The next bounded hypothesis is to compile source
language into pointers and categorical instructions, execute those instructions
with one position-tied model-owned digit/carry transition, and serialize the
result through a tied vocabulary-aligned motor. The system is an ordinary
recurrent transducer, not a new computational primitive; the potentially useful
claim is a width-independent identification and interface-learnability advantage
over named, resource-matched controls.

## 2. Frozen empirical boundary

The following observations are constraints on this theory.

| Observation | Frozen result | Consequence |
|---|---:|---|
| Raw flagship | 300,000 steps, 125,081,664 parameters | Preserve as immutable base. |
| Raw public board | GSM8K maj@4 4%, GSM8K pass@1 2%, MATH-500 2%, HumanEval 3.66%, MBPP 0% | More raw pretraining did not unlock broad reasoning. |
| Fresh language compiler probe | 0/6 exact under zero-shot, two-shot, and constrained two-shot prompting | Ordinary generation is not a usable compiler. |
| Oracle-compiled frozen DRS | 28/34 exact transitions; 5/6 chains closed; 2/6 exact terminal states | Local execution exists, but transport is not reliable. |
| Fresh serializer probe | 2/6 native, 0/6 rule, 0/6 one-shot | Terminal readout is independently missing. |
| Referential pointer compiler | 43.5% fresh answer and 38.4% fresh program exact in r5 | Text-only dynamic binding is real but below an autonomous gate. |
| NL SCEB | 15.7% closed loop with host arithmetic | Operation signal exists; internal execution does not. |
| Result-digit motor | 57/250 base to 59/250 treatment; width-eight 0/50 | A wide output motor alone does not close the system. |
| Tokenizer | Every decimal digit is one token; `-` is also one token | Numeric literals can be selected by learned pointers without host integer parsing. |

The last tokenizer observation was checked directly against
`artifacts/shohin-tok-32k.json`. It is a representation fact, not permission for
the runtime to parse or calculate numbers.

## 3. Capability object

The first confirmation family is deliberately bounded:

```text
T_max = 256 source tokens
L_max = 8 instructions
W_max = 16 accumulator digits plus one terminal-carry slot
opcodes = {LOAD, ADD, SUB, HALT}
```

Every admitted gold `SUB` has a nonnegative intermediate and final result. A
terminal borrow under an admitted gold program is therefore an invalid state
and a scored failure. Signed arithmetic, comparison, branching,
multiplication, and division are outside schema v2.

For a bounded question `x`, define a latent program

```text
P(x) = (instruction_0, ..., instruction_(L-1)).
```

An instruction contains only categorical fields:

```text
(opcode, span_start, span_end).
```

`span_start` and `span_end` are inclusive token addresses satisfying
`0 <= span_start <= span_end < T`. Every token in an admitted span is one of the
ten tokenizer digit tokens and is stored most-significant first in the source.
The compiler predicts both endpoints. The host does not find, validate, repair,
or convert the span at inference. An invalid predicted span is a model failure.

`LOAD` copies the selected digit categories to the accumulator. `ADD` and `SUB`
read the accumulator and selected source span from least-significant to
most-significant digit, zero-padding only after the model-produced source cursor
passes `span_start`. `HALT` ends instruction execution. There is one accumulator
and no branch or destination field in schema v2.

The complete mutable machine state at microstep `t` is

```text
q_t = (pc_t, phase_t, source_cursor_t, carry_t,
       accumulator_t, invalid_t, halt_t).
```

The accumulator is a categorical digit tape with one explicit terminal-carry
slot. The transition is

```text
q_(t+1) = U_theta(q_t, P(x), source_tokens).
```

The learned add/sub table in `U_theta` is reused at every digit position and
every arithmetic instruction. `LOAD` is a fixed categorical copy. The
serializer is another tied transition with complete state

```text
r_t = (read_cursor_t, seen_nonzero_t, halt_t):
```

```text
(emit_t, output_symbol_t, r_(t+1)) =
    S_psi(accumulator_T[read_cursor_t], at_last_t, r_t).
```

One registered model forward contains the entire recurrent graph. Zero-parameter
tensor operations inside that graph may gather through a hard pointer, compare
one-hot pointer identity, apply a fixed one-position shift permutation, select a
categorical write with a hard mask, advance a one-hot program counter, and stop
on model-produced `HALT`. These operations and their FLOPs/state bytes must be
reported as part of the architecture. They may not inspect digit values or
choose semantic actions. There is no branch in schema v2.

Outside the registered forward, the host may allocate tensors, invoke the model
once, and decode returned token IDs. It may not advance a program counter,
interpret an instruction, gather an operand, move a cursor, write a destination,
compute an opcode, parse a literal, calculate a result digit or carry, repair a
state, retry, or consult a verifier.

## 4. Three separable interfaces

### 4.1 Pointer compiler

The compiler reads transformer residuals and produces all `L_max` categorical
instruction slots in parallel. Each slot predicts an opcode and inclusive start
and end pointers over all source positions. Numeric operands are represented by
those source spans, not by a 1,000-way value classifier and not by generated
decimal text. No formatting-derived event span or gold digit mask may enter the
positive inference path.

The compiler must be equivariant to:

- entity renaming;
- reordering of independent introductory clauses;
- replacement of a numeric literal by a new literal with the same syntactic
  role;
- insertion of semantically neutral text;
- width growth of a pointed numeric span.

No structural span, operation, pointer, or destination supplied by the offline
generator may enter inference except as a training target or held-out score.

### 4.2 Position-tied executor

The first executable family is decimal addition and nonnegative-subtraction
microcode. Its learned atomic
transition domain is

```text
operation in {ADD, SUB}
left_digit in {0, ..., 9}
right_digit in {0, ..., 9}
incoming_carry_or_borrow in {0, 1}.
```

There are exactly `2 * 10 * 10 * 2 = 400` local contexts and 20 joint
`(result_digit, next_carry)` outcomes. The same learned transition is reused at
all positions. `LOAD`, pointer shift, program-counter shift, categorical gather,
and masked write are fixed in-graph mechanics and are separately counted. For an
admitted `SUB`, terminal borrow must be zero. Multiplication, division,
comparison, branching, signed subtraction, and general word-problem programs
are outside the first confirmation generation. They may enter only through a
later theory and gate, not by silently extending this one.

### 4.3 Tied serializer

The serializer reads only the final model-owned accumulator, its complete
`(read_cursor, seen_nonzero, halt)` state, and the fixed `at_last` pointer
relation. It scans most-significant to least-significant position. Its 40 atomic
contexts are `(seen_nonzero, digit, at_last)` and its outputs are
`(emit_or_skip, next_seen_nonzero, halt_or_continue, digit_symbol)`. It writes a
vocabulary-aligned residual and uses Shohin's tied output head to emit digits and
stop. Leading zeros are skipped; the final zero is emitted for the number zero.
An addition terminal carry occupies a real accumulator slot. There is no sign
output in schema v2.

The serializer is forbidden from reading the original question, gold answer,
offline trace, host-parsed integer, or verifier result.

## 5. Capability and resource theorems

### 5.1 Width-independent transition identification

**Theorem 1.** Let `T` be a deterministic local digit transition over finite
context set `C`. Suppose operand endpoints, digit direction, zero padding,
cursor initialization, cursor shift, and terminal handling follow the fixed
schema-v2 mechanics. If one tied executor applies the same exact `T` at every
position, and training identifies `T(c)` for every `c in C`, then addition and
admitted nonnegative subtraction are exact at every finite width for which the
source and accumulator tapes are well-formed.

**Proof.** At position zero, the executor receives a context in `C` and emits
the exact result digit and next carry. Assume positions `0..p-1` are exact. The
carry entering position `p` is therefore exact, and the fixed pointer relation
selects the correct source digit or post-span zero, so the context at `p` is in
`C` with its correct fields. Exactness of `T` gives the correct digit and
successor carry. Induction reaches the terminal position. The addition terminal
carry is already a component of the final state and therefore needs no host
reconstruction. An admitted subtraction has zero terminal borrow by definition;
a nonzero terminal borrow is an invalid-state failure, not a negative result.

For decimal add/subtract, complete local identification uses 400 contexts,
independent of width.

**Corollary 1.** In the favorable position-untied comparator family
`T_0, ..., T_(w-1)`, any position/context pair absent from training can be
changed without affecting training loss. Exact identification over width `w`
therefore requires coverage proportional to `400w`, unless the comparator adds
a tying or equivariance assumption. This is an identification result relative
to the named untied family, not a separation from transformers, Neural GPUs, or
other tied recurrent models.

### 5.2 Dynamic pointer output capacity

**Proposition 2.** If source-position representations are distinguishable and a
query state identifies the desired source relation, a shared pointer scorer can
represent a distribution over any of `T` source positions through a dynamic
`T`-way softmax. A single fixed `K`-class operand-identity head has at most `K`
distinct outputs and therefore aliases more than `K` operand identities.

This is the standard Pointer Network advantage, not a new result. It explains
why SCEB's frozen 1,000-class operand head is the wrong positive control for
unseen values. It says nothing about whether the compiler can learn the correct
pointer, and it does not apply to a compositional digit decoder or
autoregressive copier. Both are mandatory favorable controls.

### 5.3 Width-independent serialization

**Theorem 3.** Suppose the serializer starts at the most-significant accumulator
slot with state `(seen_nonzero=0, halt=0)`, the fixed cursor relation supplies
the correct digit and `at_last` bit, and one tied serializer is exact on all 40
contexts `(seen_nonzero, digit, at_last)`. Then it emits the canonical unsigned
decimal serialization of every finite well-formed accumulator, including zero
and a nonzero terminal-carry slot.

**Proof.** Before the first nonzero digit, every nonterminal zero maps to
`skip, seen_nonzero=0`. The first nonzero maps to `emit` and permanently sets
`seen_nonzero=1`; every later digit is emitted. If all digits are zero, the
`at_last` context emits one zero. The `at_last` transition halts after that
decision. Induction over cursor positions yields exactly the canonical unsigned
digit string. This theorem assumes the complete state and fixed cursor
mechanics; it does not reduce to symbol/terminal classification alone.

### 5.4 Conditional vocabulary-alignment advantage

Let `B` be a symbol subspace and `J_k` the linearized downstream map at consumer
`k`. For a desired local output displacement `y` in the range of `J_k B`, the
minimum-norm exact preimage is `(J_k B)^+ y`. The weaker norm bound

```text
||delta|| >= ||y|| / sigma_max(J_k B)
```

is necessary but not sufficient to characterize attainable target directions.
For a fixed one-dimensional target direction with measured gain `g`, the
required perturbation norm is `margin / g`. Therefore a vocabulary-aligned basis
has a local actuation-energy advantage only if its measured target-direction
gain, reachable rank, conditioning, and label preservation are better.

The diagnostic rotation is not an in-subspace basis change, which would preserve
singular values. Let `B_align` contain normalized token-aligned digit/opcode
directions. Let `Q` be a frozen ambient orthogonal map chosen before measurement
so `B_rot = Q B_align` has the same Gram matrix but large principal angles from
`span(B_align)`. The frozen-trunk diagnostic learns no inverse. In any later
trained control, both arms receive identical trainable bridges and adapters, so
the rotated arm is allowed to learn an inverse at the charged optimization cost.
Both serializers use an equal learned projection into the same frozen tied
output head. Across multiple consumers, no multiplicative claim is allowed
unless every intermediate Jacobian restriction and nonlinear operating point is
measured.

This is a conditional linear-algebra statement, not evidence that Shohin has
the needed workspace. The 2026 global-workspace study reports preferential
broadcast of vocabulary-aligned directions in larger LMs and uses random
orthogonal rotations as controls. Shohin must reproduce the relevant gain,
label-preservation, and causal-swap result locally before vocabulary alignment
may be treated as more than a representational choice.

## 6. Axiomatic primitive and exact collapse

The state and operators above define an ordinary finite Mealy transducer with a
source-addressing function. Its declared machine state may refine the minimal
Nerode quotient; no minimality or state-equivalence claim is made. For bounded
tape width, program length, precision, and recurrent steps, the complete
mechanism can be unrolled into a finite acyclic circuit.

Consequences:

1. VAMT is not a new state ontology or computational class.
2. The pointer compiler reduces to a Pointer Network or equivalent attention
   copier.
3. The executor reduces to a tied finite-state digit transducer and is closely
   related to Neural GPU and neural program-interpreter constructions.
4. The serializer reduces to a tied copy transducer.
5. Vocabulary alignment is a coordinate and learnability hypothesis, not a new
   primitive.

The only admissible positive is therefore package-level and resource-relative:

> At equal total trainable parameters, retained state, source access, training
> examples, training FLOPs, inference FLOPs, sequential depth, and external
> execution, the pointer/tied/aligned package improves exact held-out program
> execution over the preregistered favorable controls.

If the advantage disappears against a tied recurrent pointer control, the
specific VAMT hypothesis is rejected even if both systems solve the board.

## 7. Prior-art boundary

Known work already covers every broad component:

- [Pointer Networks](https://arxiv.org/abs/1506.03134) provide dynamic
  input-position outputs and length extrapolation.
- [Neural GPUs Learn Algorithms](https://arxiv.org/abs/1511.08228) use tied
  recurrent computation for algorithm learning and long arithmetic.
- [Neural Programmer-Interpreters](https://arxiv.org/abs/1511.06279) use a
  recurrent core, program memory, execution traces, and compositional programs.
- [Neural Arithmetic Logic Units](https://arxiv.org/abs/1808.00508) impose
  arithmetic inductive bias for numerical extrapolation.
- [Verbalizable Representations Form a Global Workspace in Language Models](https://transformer-circuits.pub/2026/workspace/index.html)
  reports vocabulary-aligned, broadcast representations and random-rotation
  controls in larger language models.

VAMT must not be called a novel pointer network, neural computer, arithmetic
unit, program interpreter, workspace, or recurrent primitive. The narrow open
delta is whether a small pretrained language model can reuse its existing
token-aligned residual geometry as the interface between a text compiler and a
position-tied learned transducer more efficiently than matched arbitrary-basis
or monolithic alternatives.

## 8. Exact maximum parameter ledger

The frozen base has 125,081,664 unique parameters. The strict project ceiling is
`<150,000,000`, so at most 24,918,335 additional parameters are admissible.

### 8.1 Primary minimal realization

The primary falsifiable realization is deliberately small. It reuses the
300,493-parameter R4-style referential compiler allocation, including the 8,000
local transition logits, but it may not reuse R4's host-supplied structural
spans. Global start/end pointer heads replace that inference shortcut.

```text
R4-style compiler and 400-context table                 300,493
boundary head 256->3                                        771
digit key 256->128                                       32,768
two slot start/end queries, 128->128 each                32,768
event start/end queries, two 256->128 maps               65,536
tied 13->64->13 serializer                                1,741
                                                        -------
additional parameters                                   434,077
base plus VAMT-min                                  125,515,741
strict headroom                                      24,484,258
```

The 13 serializer symbols are digits `0..9`, `BLANK`, `EOS`, and `INVALID`.
The serializer count is

```text
Linear(13,64,bias)   13*64 + 64      896
Linear(64,13,bias)   64*13 + 13      845
                                      -----
                                      1,741
```

An executable implementation must instantiate and recount this graph. The
existing R4 source is evidence and a favorable control, not drop-in positive
code, because its formatting-derived spans are forbidden here.

### 8.2 Optional maximum realization

The following is a capacity ceiling, not the primary experiment and not
permission to instantiate it. It may be considered only if VAMT-min establishes
the mechanism while the independently scored compiler remains capacity-limited:

| Component | Formula | Parameters |
|---|---:|---:|
| Late-block LoRA, blocks 18-29, rank 64 over q/k/v/o/gate/up/down | `12 * 64 * 10,176` | 7,815,168 |
| Compiler norm/trunk/role heads/pointer keys | exact listed dimensions below | 1,928,344 |
| Compiler GRU, initialization, opcode/role/stop/pointer-query heads | exact listed dimensions below | 3,031,578 |
| Tied executor symbol interfaces and `512->2048->2048->32` cell | exact listed dimensions below | 5,394,720 |
| Tied serializer symbol embedding, `256->512` GRU, residual/stop/init heads | exact listed dimensions below | 2,007,362 |
| **Additional total** |  | **20,177,172** |
| **Base plus treatment** |  | **145,258,836** |
| **Headroom below strict ceiling** | `149,999,999 - 145,258,836` | **4,741,163** |

The `10,176` LoRA coefficient per rank and block is

```text
q     576 + 576       1,152
k     576 + 192         768
v     576 + 192         768
o     576 + 576       1,152
gate  576 + 1536      2,112
up    576 + 1536      2,112
down  1536 + 576      2,112
                         -----
                        10,176
```

The compiler ledger is

```text
LayerNorm(576)                                      1,152
Linear(576,1024,bias)                             590,848
Linear(1024,1024,bias)                          1,049,600
role heads 1024->16 and 1024->8                    24,600
pointer key 1024->256                             262,144
GRUCell(input=1024, hidden=512)                 2,362,368
initial state 1024->512                           524,800
opcode 512->16                                      8,208
role 512->8                                         4,104
stop 512->2                                         1,026
pointer query 512->256                            131,072
                                                    ---------
                                                   4,959,922
```

The executor ledger is

```text
token projection 576->128                          73,728
opcode embedding 16*128                              2,048
carry embedding 2*128                                  256
phase embedding 8*128                                1,024
register-symbol embedding 32*128                     4,096
LayerNorm(512)                                       1,024
Linear(512,2048,bias)                            1,050,624
Linear(2048,2048,bias)                           4,196,352
Linear(2048,32,bias)                                65,568
                                                    ---------
                                                   5,394,720
```

The maximum serializer ledger is

```text
symbol embedding 13*256                               3,328
GRUCell(input=256,hidden=512)                     1,182,720
residual head 512->576,bias                          295,488
stop head 512->2,bias                                  1,026
initial state 1024->512,bias                         524,800
                                                    ---------
                                                   2,007,362
```

A later executable preregistration must derive one canonical module graph and
recount this ledger from instantiated tensors. Padding a control with unused
parameters is insufficient; every treatment parameter must have a declared
counterpart or favorable control allocation.

### 8.3 State, source, target, and compute ledger

For `T_max=256`, `L_max=8`, and `W_max=16`, the minimal hard inference state is
stored canonically as:

```text
program opcodes:              8 * uint8                 8 bytes
program span starts:          8 * uint16               16 bytes
program span ends:            8 * uint16               16 bytes
pc, phase:                    2 * uint8                 2 bytes
source cursor:                1 * uint16                2 bytes
carry, invalid, halt:         3 * uint8                 3 bytes
accumulator including carry: 17 * uint8                17 bytes
serializer cursor/seen/halt:  3 * uint8                 3 bytes
                                                       --------
retained program and private state                     67 bytes
```

The immutable source is at most `256 * uint16 = 512` bytes. The returned output
buffer is at most `17 * uint16 = 34` bytes plus a 17-byte emit mask. Temporary
logits, one-hot straight-through tensors, autograd storage, base KV/residual
state, and allocator overhead are not retained-state bits but must be reported
as peak bytes separately by every executable arm.

At maximum bounds, the fixed executor performs exactly `L_max * W_max = 128`
instruction-position cycles, even after an early `HALT`; inactive cycles are
masked but still charged. VAMT-min performs one 400-context table gather per
active ADD/SUB cycle. The minimal serializer performs at most 17 calls to a
`13->64->13` MLP, or `17 * (13*64 + 64*13) = 28,288` matrix MACs, plus fixed
mask/shift operations. The optional maximum executor performs
`512*2048 + 2048*2048 + 2048*32 = 5,308,416` matrix MACs per cycle, or
679,477,248 at 128 cycles. Compiler/base FLOPs depend on admitted source length
and must be reported per actual token with identical padding and cycle charging
in matched arms.

Structured supervision is a resource. Every report must count target bits,
oracle-generated fields, and loss terms in addition to examples and FLOPs. No
comparison may attribute a gain to architecture when only VAMT received program,
transition, pointer, or serializer labels.

## 9. Mandatory controls

The first executable protocol must include all of the following.

1. **Primary pointer recurrent control:** a conventional pointer compiler plus
   tied recurrent executor with identical structured targets, target bits,
   oracle calls, state, cycles, parameter allocation, and equal or favorable
   compute. This is the strongest known-component control.
2. **Primary rotated-bus control:** identical module graph, targets, and
   initialization spectrum. A frozen ambient orthogonal map sends the aligned
   basis to a large-principal-angle subspace while preserving its Gram matrix.
   No fixed inverse is supplied. Any learned inverse uses the same charged
   bridges/adapters available to treatment.
3. **Secondary monolithic LM control:** same base, examples, parameter ceiling,
   optimizer, updates, and training FLOPs. It receives the same structured
   program, transition, pointer, and serializer targets through matched
   auxiliary heads in addition to answer/text loss. If exact target-bit and
   oracle matching is impossible, its comparison is descriptive only and
   cannot support architectural attribution.
4. **Untied-position control:** independent position transitions with treatment
   parameters reallocated favorably, used to test width identification.
5. **Shuffled compiler binding:** source spans and op labels are permuted within
   matched strata while true answer scoring remains unchanged.
6. **Shuffled transition table:** preserves label frequencies but destroys the
   arithmetic transition law.
7. **Shuffled serializer map:** preserves output frequencies but breaks the
   register-to-token relation.
8. **Gate-off identity:** disabling every new interface must reproduce the
   frozen base logits byte-for-byte at the declared precision.
9. **Oracle ceilings:** gold compiler only, gold executor only, and gold
   serializer only. These are diagnostics and cannot be reported as autonomous
   capability.

All arms must report the full resource vector:

```text
(parameters, retained bits, precision, source bytes, training examples,
 oracle calls, training FLOPs, inference FLOPs, sequential depth,
 external memory, external execution).
```

Every primary arm receives the same structured target tensors, reductions, and
loss weights. Target bits and offline oracle fields are charged explicitly.
Program labels are never visible as model inputs. If a target cannot be exposed
to one arm without changing that arm's semantics, the affected comparison is
descriptive rather than causal.

### 9.1 Existing development supervision and fresh-confirmation requirement

The compiler may derive training-only labels from
`artifacts/sft/role_equivariant_microcode_v3.jsonl` (288,000 rows / 48,000
programs) and the source-scheduled development board. Rows must be grouped by
`equivalence_id`; all semantic views and register permutations stay in one
split. The model input is only `question`. Structured operations, values,
pointers, query, and halt are loss-only labels.

The executor may derive component-pure labels from
`artifacts/sft/digitwise_factor_v1_train.jsonl` by projecting only
`(opcode,left_digit,right_digit,carry_in)` to
`(result_digit,carry_out)`. Width, absolute position, result prefix, gold state,
and expected register are forbidden executor inputs. The serializer may use
only final model-owned register projections; the existing final prompt is
forbidden because it exposes the original operand tape.

All existing microcode, cursor, source-scheduled, digitwise-factor, and "fresh"
boards are development-only because their scores or contents have already been
read. After theory, implementation, thresholds, and split policy freeze, a new
sealed confirmation generation is mandatory. Compiler splits group by
`equivalence_id`; executor/serializer splits group by `episode_id` and operand
tape. Confirmation must cross width, value, paraphrase, entity permutation,
opcode composition, and carry-run length without changing carry prevalence
between arms.

## 10. Finite collapse tests before neural code

Before any neural implementation, one CPU-only falsifier must:

1. enumerate all 400 add/sub transition contexts;
2. prove the tied truth table executes exact widths 1 through 64 by induction
   and exhaustive bounded replay;
3. construct an untied comparator that agrees on every observed position and
   fails arbitrarily on the first unseen position;
4. verify pointer equivariance under entity renaming, literal replacement, and
   neutral-token insertion, including inclusive start/end relocation, unequal
   span widths, least-significant-first reads, and post-span zero padding;
5. verify serializer exactness for all 40 complete-state contexts and exhaustive
   unsigned tapes through a bounded width;
6. count every persistent bit and every fixed runtime operation;
7. reject any runtime path that parses an integer or calls arithmetic,
   correction, search, retry, or verification code;
8. demonstrate exact reduction to an ordinary Mealy transducer and finite
   unrolling, preserving the claim boundary.

The CPU falsifier may use an explicit categorical truth-table lookup as the
exact realization being analyzed, but it must count every such lookup as
external symbolic execution. It may show that no arithmetic occurs inside a
replay after the table is frozen; it may not report zero external execution or
model-owned reasoning. A CPU pass establishes mechanics only. It does not
authorize a Shohin fit.

## 11. Score-blind staged gates

An executable preregistration may advance only in this order.

### Stage A: compiler

- exact digit-span pointer at least 99% on held lexical templates;
- exact opcode/role/pointer program at least 90% in-distribution and 75% on the
  frozen joint value/width/paraphrase split;
- at least 10 percentage points over shuffled binding and at least 5 points over
  the favorable fixed-value head;
- all entity-renaming and literal-replacement causal swaps move the intended
  pointer and no unrelated pointer.

### Stage B: executor

- 400/400 local contexts exact;
- zero transition errors over frozen widths 1 through 64 and held value strata;
- every admitted nonnegative subtraction ends with terminal borrow zero, and
  every nonadmitted negative subtraction is rejected rather than serialized;
- causal carry swaps change exactly the successor contexts predicted by the
  transition law;
- no position-specific parameter or source leak.

### Stage C: serializer

- every symbol/terminal context exact;
- zero errors on frozen random registers through width 64, including leading
  carry and zero;
- register-symbol swaps redirect exactly the corresponding output token.

### Stage D: integration

- all three model-owned interfaces active, with no oracle component;
- at least 60% exact on the frozen composed board and at least 10 points over
  every primary matched control;
- no regression larger than 3 points on the frozen direct-language preservation
  board;
- treatment advantage survives width, value, paraphrase, entity-permutation,
  and longer-program strata separately;
- direct transcripts show correct intermediate register state and correct final
  serialization, not only answer extraction.

The numerical thresholds above are theory defaults only. They acquire standing
only if frozen before candidate training and confirmation generation.

## 12. Stop conditions

Reject VAMT without rescue-by-renaming if any of the following occurs:

- the pointer/tied recurrent favorable control matches treatment;
- vocabulary alignment has no preregistered gain or causal-swap advantage over
  the rotated basis;
- compiler errors dominate even with oracle executor and serializer;
- tied executor fails one complete local context after convergence;
- terminal carry is reconstructed outside the model-owned register;
- host code computes or repairs any semantic field;
- width performance depends on training every tested position;
- full exactness rises only through parser, extraction, retry, verifier, or
  answer-format leniency;
- parameter, state, source, or compute matching cannot be made exact or
  favorable to the controls.

## 13. Current authorization

Schema v1's independent reviewer returned theory `NO-GO`, CPU falsifier
`RESTRICTED GO` for counterexample discovery only, and neural implementation
`NO-GO`. The first v1 symbolic all-pass artifact is consequently invalid as a
positive gate. Schema v2 is a repair candidate, not an inherited approval.

This theory attempts a complete capability object, named comparator family,
resource claim, exact-collapse boundary, and prior-art boundary. It does not yet
complete the R12 gates. The next permitted work is:

1. independent adversarial review of this theory;
2. one symbolic CPU falsifier for Sections 5, 6, and 10;
3. a canonical instantiated parameter ledger without fitting any weights;
4. only after those survive, an executable score-blind preregistration.

No Shohin checkpoint, data mix, remote scheduler state, or accelerator resource
is authorized by this document.
