# R12 Shared In-Model Transition Circuit Theory

**Protocol schema:** `R12-STC-THEORY-SCHEMA-v3`

Any externally reported hash for schema v2 identifies superseded bytes. This
schema has no embedded self-hash. An externally computed SHA-256 may identify
the exact v3 theory artifact, but it is not an implementation, generator,
secret, instrumentation, or executable-protocol commitment under Section 12.

**Status:** THEORY AND PROTOCOL SCHEMA ONLY; **NOT AN EXECUTABLE
PREREGISTRATION**. This file does not freeze implementation source, a data
generator, support/exclusion rules, secret commitment, artifact hashes, exact
training denominators, instrumentation bytes, or a report validator. It
therefore authorizes no checkpoint change, CPU result, neural implementation,
data generation, fit, autonomous score, accelerator job, H100 launch,
capability claim, or novelty claim. Sections 12-15 specify what a later
executable preregistration would have to bind before any run.

**Decision in one sentence:** the `578->4096->4096->12` MLP is only a local
whole-pair predictor unless a fixed runtime initializes and advances an opaque
state, proves that the committed carry is the carry consumed by the successor,
and excludes generated output from that path; the resulting complete system is
an ordinary finite-state/Mealy recurrent machine with a learned local table,
not learned end-to-end reasoning.

## 1. Frozen empirical boundary

This dossier takes the following facts as constraints, not targets to reinterpret
or improve post hoc.

| Fact | Frozen value | Consequence |
|---|---:|---|
| Base Shohin unique parameters | `125,081,664` | Tied embeddings are counted once. |
| Total parameter ceiling | strictly `<150,000,000` | Equality is failure. |
| Maximum additional parameters | `24,918,335` | `150,000,000 - 1 - 125,081,664`. |
| Post-DRS late residual | strong causal digit signal | A digit actuator is plausible. |
| Post-DRS carry path | materially weaker | Carry writing and later consumption remain open. |
| Wide digit motor | `576->4096->4096->10`, `19,185,674` parameters | Teacher-forced fit is perfect; autonomous value is still pending. |
| Carry motor | separately review-gated | It supplies no result or reusable artifact to this theory lane. |
| Host arithmetic | forbidden | No `apply_op`, parsed integer update, carry computation, or result reconstruction may enter inference. |
| Generated result tape | forbidden as a causal solver | Emitted symbols may be scored after the run but may not update, repair, schedule, or feed the next transition. |

The canonical r3 diagnosis in `R12_DRS_CAUSAL_CYCLE_RESULT.md` is the local
starting point: digit response was stronger than carry response, only `9/50`
integrated two-call cycles were exact, and generated-token KV was explicitly
excluded from an admissible successor. The packet-on-lattice result is also
binding: moving one carry bit with cursor support is exactly an ordinary finite
recurrent transducer, not a new computational primitive.

The exploratory digit motor's perfect teacher-forced fit is an actuator ceiling.
It is not evidence that its output is consumed, that errors do not compound, or
that the model has acquired an autonomous transition law.

## 2. Exact question and claim boundary

The bounded question is:

> Can one shared learned circuit convert the frozen post-DRS local residual into
> an atomic digit/carry transition, commit the carry to a minimal private state,
> and support width/value-OOD closed-loop execution without any generated token,
> parsed tape, or host arithmetic entering the successor computation?

This schema does not yet define matched training. The strongest admissible
future positive claim is therefore about the **entire PEDC interface and
training package**, not atomic commit in isolation:

> Under one later executable protocol that freezes the complete training and
> runtime resource vectors, the PEDC package gives a reproducible causal and
> behavioral advantage over its primary ordinary-transducer control on the
> canonical digitwise transition family.

A PEDC-component attribution is allowed only if that later protocol makes
PEDC and R-SEQ identical in ordered data, semantic targets, loss terms,
coefficients and reductions, argmax gradient estimator, warmup, optimizer,
initialization and update schedule, model calls, sequential depth, precision,
FLOPs, and allocated state/source/trace/output/KV bytes, leaving
agreement-or-fault commit versus R-SEQ's fixed aggregate commit as the only
treatment variable. The resulting learned weight trajectories need not be
equal. If exact protocol matching is impossible, any positive is package-level
and cannot be attributed specifically to atomic commit, dual-view agreement,
or site conditioning.

Even a complete pass would not establish:

- a new computational class;
- a new state ontology;
- learned operation selection or planning;
- learned cursor scheduling or general halting;
- arbitrary-width execution beyond the tested source representation;
- a natural-language source compiler;
- a conventional decimal answer formatter;
- learned end-to-end execution, because fixed runtime logic supplies
  initialization, addressing, recurrence, cursor movement, END handling,
  emission order, and halt;
- broad reasoning, SoTA, or a world-first primitive.

## 3. The bare shared trunk

Let `h in R^576` be the frozen block-29 residual at one transition aperture. Let
the two exact site codes be

```text
s_D = (1,0)    digit view
s_C = (0,1)    carry view.
```

The candidate trunk is

```text
f_theta(h,s) = W3 SiLU(W2 SiLU(W1 [h;s] + b1) + b2) + b3,

[h;s] in R^578,
f_theta(h,s) in R^12.
```

Each 12-logit output is split as

```text
f_theta(h,s)[0:10]   digit logits for {0,...,9}
f_theta(h,s)[10:12]  next-carry/borrow logits for {0,1}.
```

Both site views predict the complete pair. The treatment does not use a digit
view that predicts only a digit and a carry view that predicts only carry. That
weaker construction permits two unrelated classifiers while making their
shared parameter tensor look mechanistically meaningful.

### 3.1 Exact parameter ledger

```text
Layer 1 weights       578 * 4096             2,367,488
Layer 1 bias                    4096              4,096
Layer 2 weights      4096 * 4096            16,777,216
Layer 2 bias                    4096              4,096
Layer 3 weights        4096 * 12                49,152
Layer 3 bias                      12                 12
                                                  -----------
Shared trunk trainable parameters              19,202,060
Frozen Shohin unique parameters               125,081,664
                                                  -----------
Total unique parameters                       144,283,724
Distance to 150,000,000                         5,716,276
Maximum further spend under strict `<150M`      5,716,275
```

The shared trunk is exactly `16,386` parameters larger than the current digit
motor:

```text
two added input features:  2 * 4096       = 8,192
two added output classes:  2 * 4096 + 2   = 8,194
                                            ------
                                            16,386
```

The hard latch, categorical register, cursor shift, END test, attention
firewall, and fixed scatter into the existing digit/carry vocabulary rows have
zero trainable parameters. They are runtime state and computation and must be
reported separately. A learned register embedding, learned address head,
learned halt head, extra layer norm, calibration scalar, or trainable router is
not part of this count. Adding any one of them changes the protocol and is a
NO-GO until separately counted and preregistered.

The remaining `5,716,275` spendable parameters are deliberately unused. Padding
does not make a control stronger, and no second causal hypothesis currently
justifies another learned module.

### 3.2 What the 12 outputs can and cannot represent

The two hard argmaxes can represent all `10 * 2 = 20` deterministic
`(digit,next_carry)` pairs. A 20-way joint head is therefore unnecessary for a
deterministic local transition.

The factorized head cannot represent an arbitrary calibrated joint probability
over those 20 pairs. Under residual ambiguity it supplies marginals, not
correlation. Separate cross-entropy can therefore assign mass to an incoherent
pair even when each marginal looks good. The autonomous claim is consequently
based only on hard pair exactness and full trajectories, never marginal NLL.

The trunk also cannot create information absent from `h`. Width does not repair
a representation collision, and site conditioning does not supply operation,
operands, incoming carry, or cursor unless the frozen in-model aperture already
delivers them.

## 4. Proposed mechanism: pre-emission dual-view commit

The treatment is named **pre-emission dual-view commit** (`PEDC`) only to make
its intervention surface unambiguous. The name is not a primitive-novelty
claim.

### 4.1 Read-only source and private state

The source is a canonical, immutable, column-addressable input:

```text
S = (operation, a[0:w], b[0:w], END).
```

Digits are least-significant first for this bounded board. Converting arbitrary
natural-language numerals into `S` is outside this protocol. The source grammar
router may identify delimiters and source positions, but it may not calculate a
digit, carry, cursor value, width-dependent answer, or result.

The complete cross-column mutable state is

```text
q_t = (p_t, c_t),

p_t  one hard cursor support over source columns plus END,
c_t  one hard carry/borrow bit.
```

Initialization is exact and supplied by the fixed runtime:

```text
p_0 = e_0       one-hot support on the least-significant source column
c_0 = 0         no incoming carry or borrow
q_0 = (e_0, 0).
```

`q_0` is created inside `forward_episode` after the immutable source layout has
been validated and before the first local residual is computed. The host does
not pass `p_0` or `c_0`, and neither value is learned or inferred from emitted
text. A source with no column zero is invalid before execution.

Three different quantities must not be conflated at width `w`:

```text
semantic state set                  Q_w = {0,...,w} x {0,1}
number of valid semantic states     |Q_w| = 2(w + 1)
exact information capacity          log2(2(w + 1)) bits
minimum fixed binary storage        ceil(log2(2(w + 1))) bits
one-hot physical allocation         (w + 1) cursor cells + 1 carry cell
```

For widths `4`, `6`, `8`, and `10`, exact information capacities are
`log2(10)`, `log2(14)`, `log2(18)`, and `log2(22)`, approximately `3.3219`,
`3.8074`, `4.1699`, and `4.4594` bits. Minimum fixed binary storage is
`4`, `4`, `5`, and `5` bits; a literal one-hot-plus-carry implementation
allocates `6`, `8`, `10`, and `12` binary cells before byte/dtype packing.
The valid-state information capacity, minimum coding length, allocated tensor
cells, and actual bytes must all be reported separately. None is constant in
width, and allocated bytes are not evidence that every raw tensor pattern is a
valid semantic state.

No result digit, result prefix, continuous residual, logit vector, source copy,
token id, generated KV entry, retry count, verifier value, or hidden history is
allowed in `q_t`.

### 4.2 Frozen transition aperture

One private transition query is formed inside the model from immutable `S` and
hard `q_t`. The cursor selects the current source column; the carry bit is
injected as a hard two-valued private lane. The query may expose exactly

```text
(operation, a[p_t], b[p_t], c_t)
```

to the local residual. Width, terminality, prior result digits, emitted text,
future columns, and absolute cursor value are masked from the local trunk. The
fixed controller may inspect whether the cursor's next support is `END`, but
that bit may not enter `f_theta`.

The frozen Shohin backbone then produces one block-29 residual:

```text
h_t = H_frozen(S, q_t) in R^576.
```

The private query may reuse frozen token embeddings and frozen transformer
weights. It may not introduce trainable state embeddings. If the frozen
backbone cannot expose the local tuple under this aperture, that is an empirical
NO-GO, not permission to leak source or cursor fields around the bottleneck.

### 4.3 Dual-view prediction and atomic agreement

Both views are evaluated from the identical `h_t`, before any token is emitted:

```text
z_D = f_theta(h_t, s_D)
z_C = f_theta(h_t, s_C)

d_D = argmax z_D[0:10]       c_D = argmax z_D[10:12]
d_C = argmax z_C[0:10]       c_C = argmax z_C[10:12].
```

Commit is legal only if

```text
(d_D,c_D) = (d_C,c_C).
```

Disagreement enters an internal fault terminal and scores the episode wrong.
There is no retry, vote, oracle, fallback, or host choice. On agreement, the
pair is hard-latched:

```text
L_t = (d_t, c_(t+1)).
```

The two-site agreement is an error-detection constraint, not a proof of
correctness. Shared weights can make the same wrong prediction twice. Its value
is that it creates an explicit protocol-schema invariant and prevents two
disagreeing hard pairs from silently committing. A correct site-ignoring
whole-pair predictor can satisfy the invariant legitimately; agreement does not
prove that either site code is used.

### 4.4 Update before emission

The successor register is committed before any observable symbol:

```text
p_(t+1) = SHIFT(p_t)
q_(t+1).carry = COPY_CARRY(L_t.carry)
q_(t+1) = (p_(t+1), q_(t+1).carry).
```

`COPY_CARRY` is the identity on one hard bit. It has exactly one authorized
input, `L_t.carry`, and no default, stale-register, source-derived, emitted-token,
or host fallback. The next aperture is defined as

```text
h_(t+1) = H_frozen(S, p_(t+1), q_(t+1).carry).
```

The provenance graph therefore proves a structural path

```text
L_t.carry -> q_(t+1).carry -> h_(t+1) -> f_theta(h_(t+1),s),
```

with no other authorized carry input. This proves that the committed bit is the
bit presented to the successor. It does **not** prove that the learned successor
is behaviorally sensitive to that bit. Functional consumption requires the
carry-latch and stale/precomputed-carry interventions in Sections 8, 13, and 14.

**Proposition: latch provenance and successor consumption.** For every
nonterminal step, `COPY_CARRY` is the identity, so direct substitution gives

```text
q_(t+1).carry = L_t.carry
h_(t+1) = H_frozen(S, SHIFT(p_t), L_t.carry).
```

Because the successor aperture has no other carry argument, the carry value it
consumes is exactly `L_t.carry`; a stale register, source-precomputed carry, or
emitted carry cannot satisfy these equations. This is a proof of structural
consumption for the specified machine. Behavioral use by the learned local map
is a separate empirical premise: on the carry-sensitive witnesses, the next
hard pair must change under a latch flip, while the stale and precomputed
implementations must produce the exact failures specified in Section 8.4.

`SHIFT` is a fixed on-device one-hot permutation over source-column support. The
next-support `END` test controls terminal emission and halt. It is hard-coded
scheduling, not learned planning.

**Base case.** For every valid source of width at least one, fixed initialization
gives `q_0=(e_0,0)`. The first aperture therefore receives exactly
`(operation,a[0],b[0],0)`. If the corresponding one of the 200 zero-carry local
cells is exact, the latch is `L_0=(d_0,c_1)`. `SHIFT(e_0)=e_1` (or `END` at
width one), and `COPY_CARRY` writes exactly `c_1`; hence
`q_1=(e_1,c_1)` or `(END,c_1)`. This establishes the induction base used by
Theorem 5. A CPU base-case gate must enumerate all `200/200` zero-carry cells;
an autonomous report must retain the actual `q_0`, `L_0`, and `q_1` bytes for
every episode.

The model runtime must expose a single opaque call boundary:

```text
forward_episode(source_tokens) -> write_only_emissions
```

The host may launch that call and receive the final emissions. It may not loop
over columns, inspect or mutate `q_t`, parse a prediction, increment a cursor,
compute a carry, select a retry, or decide when the arithmetic episode halts.
All recurrence, `SHIFT`, commit, and END handling execute inside the model
runtime on device.

This fixed runtime performs essential execution. It validates the source
layout, creates `q_0`, selects the current column, applies `SHIFT`, copies the
carry latch, detects END, orders emissions, enforces fault halt, and blocks
output feedback. The learned trunk supplies only a local pair prediction over a
400-cell domain. The complete mechanism is therefore a hand-scaffolded FST/
Mealy/RNN with a learned local table, not an end-to-end learned algorithm.

### 4.5 Write-only emission and the causal firewall

After `q_(t+1)` is committed, the digit and carry views may scatter the hard
latch into the existing output vocabulary. These symbols are observational:

```text
for t in source-column order:
    emit typed digit d_t
    emit typed carry c_(t+1)
after the last pair:
    emit typed END and halt
```

The loop above is model-internal fixed emission control, not a host loop. The
typed carry in the last pair is the terminal carry/borrow; it is not emitted a
second time.

- they may be logged and compared with a target after the episode;
- they may not be appended to source KV visible to a later transition;
- they may not be parsed into a new prompt or state;
- they may not be reversed, accumulated, or converted to an integer by host
  code for a claim-bearing score;
- changing, deleting, or replacing every emitted symbol must leave all later
  private registers and transition predictions unchanged.

The primary endpoint is the exact least-significant-first transition stream and
the internal halt boundary. A conventional most-significant-first answer would
require a separately learned formatter with its own retained-state ledger and
preregistration. This dossier does not authorize one.

Transcript intervention must occur online **inside** the opaque call. Before
launch, the evaluator selects one of three frozen modes:

```text
PASS       expose the emitted symbol only to the output sink
DROP       suppress the emitted symbol
REPLACE    substitute a precommitted equal-length symbol stream
```

The internal interceptor sits after latch-to-symbol scatter and before any
token/KV/store that a later transition could read. `REPLACE` bytes are fixed
before the call from a future protocol-bound seed and position only; they may
not depend on generated symbols, logits, latch values, or failures. The host
cannot change mode or replacement bytes during the call. Audit copies of
`q_t`, `L_t`, and emissions may leave the call only through a write-only trace
buffer that is not addressable by the transition graph. A future executable
preregistration must hash the interceptor source, replacement generator, trace
schema, and provenance validator.

### 4.6 Why this is stronger than a merged motor

A merged motor changes two token distributions. PEDC additionally imposes four
causal facts:

1. digit and carry are functions of one pre-emission residual;
2. both site views must predict the whole transition pair;
3. successor carry is committed before serialization;
4. future computation is structurally independent of serialized output.

An output-dependent serializer will fail the firewall, but an output-independent
Mealy/RNN can satisfy all four facts. Firewall success therefore excludes an
output feedback path; it does not distinguish PEDC from ordinary recurrence.
R-SEQ is the primary control for that distinction. None of these facts makes the
learned local law correct; finite and autonomous gates test that separately.

## 5. Theorem and no-go dossier

### Theorem 1: minimal state under current-column-only access

Assume the local successor may read only the operation, the **current** operand
digits selected by the cursor, and recurrent state; it may not read lower-order
operand columns, a source prefix, prior outputs, or a precomputed prefix
summary. Under this premise, one incoming carry/borrow bit is sufficient to
determine `(digit,next_carry)`. It is necessary because addition with
`a+b=9` maps incoming carry zero and one to different digits and different
outgoing carries; subtraction with `a=b` gives the analogous borrow witness.

The necessity statement is false without the current-column-only premise. A
dense predictor with access to all lower-order source columns can recompute the
incoming carry from immutable operands and need not transport a recurrent carry
bit. D-ALL is the favorable control for exactly that alternative.

If cursor location is not otherwise supplied by source position or an external
schedule, the state must distinguish `w+1` cursor values. Combined with carry,
the valid state set has `2(w+1)` members, exact information capacity
`log2(2(w+1))`, and minimum fixed binary storage
`ceil(log2(2(w+1)))`. Prior result digits are unnecessary for future local
arithmetic conditional on `(S,p,c)`.

**Consequence:** a result tape is unnecessary state. A treatment that improves
only when prior emitted digits are visible has learned serialization or
recomputation, not the declared minimal transition.

### Lemma 2: shared weights do not imply a shared transition

On the two-point site domain `{s_D,s_C}`, a sufficiently wide nonlinear MLP can
use the site bits as a gate and approximate two unrelated functions of `h` in
disjoint hidden subspaces. The `4096`-wide trunk has more than enough capacity
to do so on the finite DWS support.

**Consequence:** parameter sharing, joint training, or perfect fit at both sites
is not evidence of a common causal computation. Whole-pair cross-site agreement,
same-residual evaluation, and atomic commit are mandatory.

### Lemma 3: residual collision no-go

If two reachable local states `x` and `x'` require different hard transition
pairs but produce the same admissible residual and site code,

```text
(h(x),s) = (h(x'),s),
```

then every deterministic trunk predicts the same output for both. At least one
must be wrong.

**Consequence:** increasing motor width cannot repair a missing carry, cursor,
or operand distinction in the frozen aperture. A collision witness is an
architecture NO-GO.

### Theorem 4: atomic-firewall serialization exclusion

Assume `q_(t+1)` is a deterministic function only of `(S,q_t,L_t)`, is committed
before emission, and the future transition graph has no path from emitted
symbols or generated-token KV to `q_(t+1)` or later residuals. Then replacing
the complete emitted transcript after every step cannot change any future
private state or prediction.

**Proof:** the first successor is equal by the update dependency restriction.
Inductively, equal source and equal private state produce equal next residual,
latch, and successor. Emission never enters the induction hypothesis. QED.

**Consequence:** transcript mutation is an exact structural test. Passing it
proves only that text is not the executor, not that the private transition is
arithmetically correct.

### Theorem 5: initialized local closure gives bounded exact iteration

Assume:

1. `q_0=(e_0,0)` exactly;
2. source addressing exposes the current column only;
3. all 400 local `(operation,a,b,incoming_carry)` cells are exact;
4. `q_(t+1).carry=COPY_CARRY(L_t.carry)` with no alternate carry path;
5. the successor aperture receives `q_(t+1).carry` as its only carry input; and
6. `SHIFT` visits each source column in order and then END exactly once.

The base case is established in Section 4.4. For the induction step, suppose
`q_t=(e_t,c_t)` contains the exact incoming carry. Premises 2 and 3 produce the
exact `L_t=(d_t,c_(t+1))`. Premises 4 and 5 make that committed carry, rather
than a stale, source-precomputed, or emitted carry, the only carry presented to
the next local map. Premise 6 advances to `e_(t+1)` or END. Thus `q_(t+1)` is
exact, and induction yields the exact transition stream for every finite source
in the declared width bound.

The proof is conditional on functional dependence in premise 3. Wiring alone
does not show consumption. The mandatory latch-flip board holds the next source
tuple carry-sensitive and requires the next hard pair to follow an intervention
on `L_t.carry`; stale and source-precomputed controls must fail that board.

**Consequence:** any neural failure after those premises appear to pass reveals
an interface, addressing, numerical, or hidden-channel error. It may not be
explained away as an arithmetic exception.

### No-go 6: finite success is compatible with a motor table

There are only 400 local decimal cells. A 19.2M-parameter network can memorize
them, and any fixed maximum width can be unrolled into a finite dense circuit.
No finite board can prove that the learned representation is uniquely a state
or exclude an unrestricted finite lookup table.

**Consequence:** the only reopenable claim is a resource-bounded learnability or
generalization advantage over named controls. The 400-entry table is a required
upper bound, not an opponent that a correct treatment is expected to beat.

## 6. Training-package schema, not a frozen objective

No training is authorized by this theory file. The prior version's statement
that arms matched data, calls, updates, and objectives was unsupported and is
retracted. This schema supplies candidate objective components but does not
freeze their weights, denominators, schedule, or gradient estimator.

The package-level hypothesis is:

> The already strong DRS digit residual plus the complete PEDC runtime,
> supervision, hard-state, and output-erasure package may improve causal carry
> consumption and closed-loop transfer relative to a separately specified
> ordinary-transducer package.

Candidate objective components are:

1. digit CE and carry CE from `z_D`;
2. digit CE and carry CE from `z_C`;
3. symmetric distribution agreement between corresponding digit and carry
   partitions of `z_D` and `z_C`;
4. exact hard-pair agreement penalty;
5. autonomous two-step loss after a fixed teacher-forced warmup;
6. carry-swap donor-following loss;
7. cursor-swap address-following loss;
8. transcript deletion/replacement invariance;
9. same-local-tuple invariance across width, terminality, and result history.

Before an executable comparison exists, a later preregistration must freeze all
of the following for PEDC and R-SEQ:

```text
ordered training-row identities and canonical bytes
train/development/confirmation denominators
batch composition and exact batch order for every seed
optimizer, hyperparameters, initialization, dtype, and update count
every loss term, reduction, coefficient, and denominator
teacher-forced warmup length and the exact transition to on-policy rollout
hard-argmax forward policy
hard-argmax backward policy: stop-gradient or a named exact STE
gradient clipping, accumulation, scaling, and overflow policy
checkpoint selection rule fixed before development scores
base calls, trunk calls, and recurrent microsteps per column
sequential depth and synchronization points
analytical and measured train/inference FLOPs
source, state, trace, emitted-output, and generated-KV bytes
online transcript-interceptor mode and replacement bytes
```

Equivalent-objective matching means each arm receives the same semantic target
set and the same weighted objective under the same denominator. If an objective
cannot be defined for both arms without changing its semantics, it is a package
difference and must be named as such; it cannot be hidden under "matched
training." In particular, an agreement penalty available only to PEDC makes the
comparison package-level unless R-SEQ receives a semantically equivalent
constraint.

Forbidden objectives or inputs are:

- a host-computed local result, carry, next cursor, terminal answer, or repaired
  state at inference;
- target carry or digit injected into the recurrent path;
- generated-token KV or parsed output used as state;
- a verifier, rejection sampler, retry loop, beam selection, or post-hoc best
  trace;
- a hidden result prefix or continuous 12-logit vector retained across steps;
- undisclosed differences in labels, examples, update counts, objectives,
  warmup, gradient policy, selection, or confirmation access.

Teacher forcing may be a warmup and diagnostic only after its exact duration is
frozen. Sequence generation is known
to have a teacher-forcing/inference mismatch; scheduled-sampling work addresses
that mismatch but does not prove state closure ([Mihaylova and Martins,
2019](https://arxiv.org/abs/1906.07651)). Every future claim-bearing endpoint
must be hard, closed-loop, and unpatched.

## 7. Control hierarchy and future matching contract

The only currently exact cross-arm match is the proposed learned parameter
count: the frozen base plus a newly initialized
`578->4096->4096->12` trunk with `19,202,060` trainable parameters. No call,
data, loss, update, depth, FLOP, or byte match exists until a later executable
protocol freezes and validates it.

### R-SEQ: primary ordinary-transducer control

R-SEQ is the primary causal control. It is an ordinary output-independent
Mealy/RNN with exact `q_0`, a hard `(cursor,carry)` register, fixed source
addressing, fixed `SHIFT`/END runtime, and the same output firewall. It uses the
same `578->4096->4096->12` trunk, hence exactly `19,202,060` learned parameters,
and computes `h_t`, `z_D`, and `z_C` through the same per-column tensor
definitions and call graph as PEDC. Numerical values may differ after training.
Its fixed commit rule is

```text
d_R = argmax (z_D[0:10]  + z_C[0:10])
c_R = argmax (z_D[10:12] + z_C[10:12])
L_t^R = (d_R,c_R).
```

It writes `c_R` through the same `COPY_CARRY` path before emission. Its emitted
symbols do not feed its state, so it can legitimately pass every
transcript-firewall test. Logit summation is fixed, adds no parameters or
retained cross-step state, and makes R-SEQ favorable on view disagreements:
R-SEQ may continue with a correct aggregate where PEDC must fault.

For an exact comparison, one audited per-column code path computes both
per-view argmaxes, the agreement bit, both logit sums, and both aggregate
argmaxes in **both** arms. A frozen selector chooses agreement-or-fault for PEDC
and `L_t^R` for R-SEQ; the unselected candidate cannot enter `q_(t+1)`. Both
arms use one base residual, two ordered trunk invocations, the same scratch and
trace tensors, and the same private-state allocation. This construction makes
exact call, sequential-depth, analytical-FLOP, and byte matching possible; it
does not freeze those quantities in this theory file. Both site outputs must
retain identical whole-pair targets, loss terms, coefficients, and reductions;
R-SEQ may not gain a lighter objective by dropping the cross-site outputs.

A future implementation must realize PEDC and R-SEQ through one audited code
path and freeze the equality vector in Section 6. The executable protocol must
make base calls, trunk calls, sequential depth, objective semantics, analytical
FLOPs, and allocated state/source/trace/output bytes exactly equal, not merely
within an observed tolerance. Measured wall time and hardware counters are
reported but are not substitutes for that analytical equality.

R-SEQ is the decisive control. PEDC-specific advantage is admissible only if
R-SEQ is validly matched and PEDC beats it under the later byte-bound
comparative gates.
If R-SEQ ties or wins, retain the simpler ordinary transducer and reject the
PEDC-specific claim.

### G-SER: favorable unmatched generic output control

G-SER fires the same-size trunk at actual digit and carry grammar positions and
may see ordinary generated prefix/KV. It can use different base calls,
sequential depth, retained KV, and training objectives. It is therefore
parameter-matched but resource- and interface-unmatched unless a future
protocol proves otherwise.

An output-dependent G-SER instance should fail online transcript interception.
A G-SER instance that ignores generated output can pass; behaviorally it has
become an output-independent transducer rather than evidence that the firewall
distinguishes architecture names. G-SER is contextual evidence, not the primary
PEDC attribution control.

### D-ALL: favorable unmatched dense recomputation control

D-ALL receives the cursor and dense access to immutable operation and all
operand columns on every step. It can recompute carry from the complete
lower-order source prefix rather than transport it. Its source access and state
resource differ essentially from PEDC, so it is parameter-matched but
resource-unmatched.

D-ALL tests the premise of Theorem 1. If it ties or wins, recurrent carry is not
needed on this board. That is a useful systems result but not a PEDC-specific
causal comparison.

### Required nonmatched incumbents and ceilings

These additional arms are reported honestly and never called equivalent:

- frozen base;
- current `19,185,674`-parameter digit motor, after its autonomous report is
  sealed;
- separately reviewed carry motor, only after its own custody gate clears;
- admissible digit-plus-carry motor bundle at its actual parameter count;
- learned 400-cell table with the same source address and emission wrapper;
- hard decimal oracle used only as an external upper-bound auditor, charged as
  external execution and never called a treatment.

The carry lane's data, reports, or unpublished artifacts may not be imported to
train or select this lane.

### 7.1 Resource receipt

Before any fit, each arm must bind numeric values for:

```text
base_unique_parameters
added_trainable_parameters
total_unique_parameters
runtime_state_semantic_variables_and_cardinality_by_width
runtime_state_exact_information_capacity_by_width
runtime_state_minimum_binary_storage_by_width
runtime_state_allocated_cells_by_width
runtime_state_allocated_bytes_by_width_and_dtype
source_cache_bytes
trace_buffer_bytes
emitted_output_bytes
generated_token_kv_bytes_visible_to_transition
retained_result_tape_bits
training_examples
supervised_digit_labels
supervised_carry_labels
counterfactual_pairs
optimizer_updates
loss_terms_weights_reductions_hash
warmup_and_on_policy_schedule_hash
hard_argmax_forward_backward_policy
training_flops
base_calls_per_column
trunk_calls_per_column
inference_flops_per_column
sequential_depth
oracle_calls_at_inference
external_execution_calls
host_arithmetic_calls
parser_repair_calls
retry_calls
```

For a PEDC-specific claim, PEDC and R-SEQ must match every applicable field
exactly under independently recomputed canonical receipts. If one field cannot
be equalized without changing the control's semantics, the comparison is
package-level and M-gates for PEDC-specific advantage are ineligible. G-SER and
D-ALL are favorable unmatched controls and publish their actual resource
vectors without padding. Equal parameter count alone is never described as a
matched experiment.

## 8. Causal interventions and negative controls

The following tests are mandatory in any future executable protocol. Aggregate
accuracy cannot substitute for them.

### 8.1 Carry interchange

Pair recipient and donor cases with identical operation, current operand digits,
cursor support, and END status but opposite incoming carry. Swap only the hard
carry bit. The complete local pair must follow the donor carry. No donor source,
residual, token, or result history may transfer.

The CPU board contains exactly 400 different-carry donor swaps and 400
same-carry donor shams: `2` operations times `2` recipient carry classes times
`100` current operand pairs. A future learned-board generator must preserve
these four strata in each confirmation regime.

### 8.2 Cursor interchange

Hold source and carry fixed and move only hard cursor support. The selected
operand column and subsequent support must follow the donor cursor. The carry
must remain the recipient carry.

### 8.3 Digit-latch sham

After atomic commit, replace only the write-only digit latch with a different
digit before emission. The visible digit must change, while `q_(t+1)` and every
future private transition remain bit-identical. This distinguishes ephemeral
output from recurrent state.

### 8.4 Carry-latch intervention

Replace only committed `c_(t+1)` before register write. The current visible
digit remains unchanged; the next transition must follow the intervention when
the next local tuple distinguishes carry. Double intervention restores the
baseline.

The CPU successor-consumption board is exact:

```text
q_0 carry                                      0
first-column pairs per operation             100
carry-sensitive second pairs per operation    10
operations                                     2
total two-column latch-flip cases           2,000
```

The second pair is carry-sensitive by construction: addition uses every
`a_1+b_1=9` pair and subtraction uses every `a_1=b_1` pair. For each case:

1. run the first transition from exact `q_0` and retain audit copy `L_0`;
2. run the baseline successor;
3. flip only `L_0.carry` before `COPY_CARRY` and run the successor;
4. verify `q_1.carry` equals the intervened bit byte-for-byte;
5. require the second hard pair to equal the oracle under the intervened carry
   and to differ from baseline; and
6. flip the latch twice and require exact baseline restoration.

All six checks must pass `2,000/2,000`. This is the concrete proof obligation
that the successor functionally consumes `L_0.carry`, not merely that a bit is
stored.

Two planted negatives are mandatory on the same board:

- **stale carry:** write `q_1.carry=q_0.carry=0`; it must follow the flipped
  latch in exactly `900/2,000` cases and fail in `1,100/2,000`;
- **source-precomputed carry:** recompute the unintervened first-column carry
  from `(operation,a_0,b_0,0)` and ignore `L_0`; it must follow the flipped
  latch in exactly `0/2,000` cases.

The stale denominator is algebraic, not empirical. Under incoming carry zero,
exactly 45 of the 100 addition pairs produce carry one and exactly 45 of the
100 subtraction pairs produce borrow one. Crossing each first pair with ten
carry-sensitive second pairs gives `2 * 45 * 10 = 900` baseline-one cases and
`1,100` baseline-zero cases. After the latch flip, stale zero therefore matches
exactly the former 900 cases. The unintervened precomputed bit is the complement
of the flipped latch in all 2,000 cases, giving exact zero following.

The source-precomputed oracle exists only inside the CPU negative control and
auditor and is charged as external execution. It is forbidden in every learned
treatment. A future neural protocol must generate a separate, balanced,
precommitted latch-flip board and run both stale and precomputed/baseline-clamp
controls; Section 14 gives the required candidate denominator schema.

### 8.5 Transcript deletion and adversarial replacement

Run each episode through the internal `PASS`, `DROP`, and `REPLACE` modes from
Section 4.5. All private register bytes, later transition pairs, and halt
location must be identical. Only the write-only output sink may differ. Running
three separate host-side prompts after parsing an earlier response is not this
test and is invalid.

### 8.6 Site-code controls

- swap `s_D` and `s_C` after training;
- replace both with one code;
- shuffle site codes within identical local tuples;
- zero one view after the other has committed;
- train without whole-pair cross-view supervision.

A site-ignoring whole-pair predictor is legitimate: it can set the two
site-input columns effectively to zero and return the same correct pair from
both views. Therefore invariance under swapping codes, replacing both with one
code, or zeroing site features does **not** falsify local transition, carry
consumption, or the PEDC package. It shows only that site conditioning was
unused and forbids a site-conditioning contribution claim.

Likewise, cross-view agreement can be vacuous when both views are identical.
If removing whole-pair agreement loses no advantage, withhold the
agreement-specific claim; do not convert that silence into an autonomous
NO-GO. PEDC-specific advantage is decided against R-SEQ, not by demanding that
site codes matter.

### 8.7 Structural cheating controls

The CPU validator must reject implementations that are behaviorally correct but
retain any of:

- prior result digits;
- a continuous residual or logits across columns;
- source identity beyond immutable read-only source;
- generated token ids or KV;
- a hidden step counter separate from cursor support;
- width or terminality in the local trunk input;
- a host-updated cursor/carry object;
- a parser callback, decimal helper, verifier, retry flag, or answer accumulator.

### 8.8 Representation and shortcut negatives

- same local tuple, different width/terminality/result prefix: transition pair
  invariant;
- different local target, identical or collision-forced residual: at least one
  failure, as required by Lemma 3;
- shuffled carry labels within nuisance strata: autonomous carry following
  collapses;
- shuffled digit labels: digit exactness collapses;
- carry-zero policy: fails every frozen carry-one witness;
- cursor-zero policy: fails source-column interchange;
- planted output-dependent serializer: must fail `DROP`/`REPLACE` when its
  generated history is its causal state;
- output-independent Mealy/RNN: may pass the firewall and must be admitted as
  an ordinary-transducer positive control rather than mislabeled a serializer
  failure;
- stale carry and source-precomputed carry: must produce the exact latch-flip
  failures in Section 8.4.

## 9. Distinguishing local transition from serialization

The dossier uses the following operational definitions.

**Serialization:** a module changes the probability of a target symbol at a
grammar location. Its prediction may depend on teacher tokens, generated
history, or a state prepared elsewhere. It need not update anything that a
future computation consumes.

**Local transition:** a module maps the admissible current local state to a
hard output and a hard successor state; the successor is later consumed, and
the complete cycle remains correct after all emitted text is deleted.

The shared trunk counts as a local transition only if all are true:

1. local tuple interventions change the hard pair selectively;
2. the PEDC treatment's two whole-pair views satisfy its declared agreement
   rule, whether or not site codes are used;
3. the committed carry, not a stale/source-precomputed carry or emitted carry
   token, controls the next step on the exact latch-flip board;
4. cursor swaps control source selection without a host update;
5. transcript deletion leaves the private trajectory unchanged;
6. autonomous two-step and full-trace gates pass;
7. the full package is interpreted against R-SEQ as the primary ordinary
   transducer, with G-SER and D-ALL reported as favorable unmatched controls.

Perfect teacher-forced fit satisfies none of conditions 2 through 7 by itself.

## 10. Expressive and systems limits

1. **Finite local table.** Decimal add/sub has 400 local cells. The trunk is
   massively overparameterized relative to that table.
2. **Frozen representation ceiling.** The trunk cannot separate residual
   collisions.
3. **No learned address claim.** The source grammar and cursor shift are fixed
   architecture supplied equally to controls.
4. **No learned halt claim.** END handling is fixed. Fault halt only detects
   internal disagreement.
5. **No probabilistic joint model.** The 10-way and 2-way heads are factorized.
6. **Bounded source format.** Natural-language parsing and decimal reformatting
   are outside the experiment.
7. **No answer tape.** Write-only least-significant-first emissions are scored
   directly. A host-produced conventional integer is not a model endpoint.
8. **Fixed precision.** Hard argmax and register bytes must be identical in the
   declared inference precision. Soft state during claim-bearing evaluation is
   forbidden.
9. **No primitive separation.** RNNs, finite transducers, and fixed-depth dense
   unrolling can realize the same bounded function.
10. **No inference from spare budget.** Unused parameter headroom is not latent
    evidence for a larger architecture.
11. **Essential fixed executor.** The runtime, not the learned trunk, performs
    initialization, source addressing, cursor movement, carry copying, END
    handling, emission scheduling, fault halt, and output isolation. Calling
    the package "in-model" locates those operations inside one opaque model
    runtime; it does not make them learned.
12. **Learned local table.** On this board the trainable trunk's semantic job is
    extensionally a 400-entry local transition table represented by a large
    MLP. Closed-loop success would validate the interface and fitted table, not
    end-to-end discovery of arithmetic execution.

## 11. Honest novelty and prior-art boundary

The architecture's known components include tied MLPs, finite-state recurrence,
hard categorical state, redundant consistency checks, fixed cursor movement,
masked attention, and write-only observation. Recurrent neural execution and
algorithm learning are established, including [Neural
GPUs](https://arxiv.org/abs/1511.08228), [Neural
Programmer-Interpreters](https://arxiv.org/abs/1511.06279), and [Universal
Transformers](https://arxiv.org/abs/1807.03819). Persistent transformer state is
also established through architectures such as [Recurrent Memory
Transformer](https://arxiv.org/abs/2207.06881) and [Block-Recurrent
Transformers](https://arxiv.org/abs/2203.07852). External differentiable memory
predates all of these in [Neural Turing
Machines](https://arxiv.org/abs/1410.5401).

Therefore PEDC is not claimed as a new computational primitive, new recurrence
class, or new memory ontology. A bounded prior-art search does not justify such
a claim.

The untested Shohin-facing package hypothesis is the exact conjunction below:

```text
one post-DRS pre-emission residual
  -> two site-conditioned whole-pair views
  -> hard agreement or fail-stop
  -> atomic carry write before output
  -> minimal private cursor/carry recurrence
  -> complete output-token causal erasure.
```

This conjunction predicts exact behavior under latch swaps, output deletion,
and state interchange that the current motors do not enforce. Site-code
ablation is diagnostic only because a site-ignoring whole-pair predictor is
valid. The possible contribution is a bounded package-level optimization and
interface result. A PEDC-specific attribution requires an exactly matched
R-SEQ comparison; if R-SEQ ties or matching fails, that narrower claim is
closed.

## 12. Requirements for a future executable preregistration

No board, denominator, seed, generator, secret, source hash, or instrumentation
is frozen by this theory schema. Consequently Sections 13 and 14 are recommended
gate schemas, not executable decisions.

Before any CPU or learned run, a separate immutable executable preregistration
must bind at least:

1. exact implementation paths and SHA-256 hashes for treatment, every control,
   generator, interceptor, provenance checker, report writer, and validator;
2. canonical source serialization, tokenizer identity, source-position map,
   local-cell ordering, and byte normalization;
3. exact train/development/confirmation support intervals, exclusion rules,
   decontamination algorithm, retry policy, and denominators;
4. one secret commitment domain and formula, one entropy source, one reveal,
   and a rule that any second secret, reseed, or regenerated board invalidates
   the experiment;
5. exact row-generation algorithm, row ordering, stratum counts, board hashes,
   and independently replayed generator receipts;
6. exact training package from Section 6 and exact PEDC/R-SEQ equality receipts
   from Section 7;
7. instrumentation that records audit copies of `q_0`, every `L_t`, every
   `q_(t+1)`, source-address support, interceptor mode/bytes, hard predictions,
   logits, fault state, and halt without exposing that trace to execution;
8. exact denominator and expected result for every CPU negative, including
   stale and source-precomputed carry;
9. a finite preservation-prompt board with exact bytes and hash, with the
   preservation claim limited to those bytes; and
10. canonical report serialization, content hash, mode/custody rules, and an
    independent validator that recomputes every aggregate from row evidence.

One acceptable **candidate** confirmation shape, which is not frozen here, is:

```text
fit_w4                 400 episodes
fit_w6                 400 episodes
value_ood_w4           400 episodes
value_ood_w6           400 episodes
width_ood_w8           400 episodes
width_ood_w10          400 episodes
                       ------------
total                2,400 episodes
```

Under that candidate, each regime would contain 200 additions and 200
nonnegative subtractions. Additions would split `100/100` by terminal carry
zero/one. Subtractions would split `100/100` by absence/presence of at least one
intermediate borrow. Fit/value-OOD scalar supports, complete operand pairs, and
episode traces would be disjoint, and widths 8 and 10 would be absent from all
training rows. None of those facts is binding until generator bytes and board
hashes are frozen.

The candidate seed set is:

```text
1337
7331
20260717
```

An executable protocol should apply every threshold separately to every seed
unless it explicitly says `mean`, prohibit best-seed selection, and reveal its
confirmation board once only after every arm checkpoint and resource receipt is
immutable. These become enforceable only when bound to exact bytes.

Reports must retain row-level inputs, hard predictions, both site-view logits,
agreement state, register states, cursor supports, intervention identities,
emitted symbols, and exact failure location. Aggregates without independently
recomputable rows are invalid. This is a report-schema requirement, not evidence
that the missing instrumentation currently exists.

## 13. Recommended executable CPU gate schema

The proposed CPU falsifier is an integration and collapse audit with a planted
oracle local transition. It does not fit Shohin and cannot establish
learnability. It may not run under this theory file: a separate executable
preregistration must first bind every item in Section 12.

Under that future binding, **CPU GO requires every condition C0-C17:**

| Gate | Exact requirement |
|---|---|
| C0 parameter ledger | Recompute `19,202,060` added and `144,283,724` total; reject `>=150,000,000`. |
| C1 executable binding | All Section 12 source, generator, denominator, instrumentation, report, and validator hashes are present and independently revalidated before execution. |
| C2 callable surface | Local trunk provenance is exactly `(h,site)`; local aperture provenance is exactly `(op,a_p,b_p,c)`; no width, terminality, result, history, or host object. |
| C3 initialization/base case | Every canonical episode starts with byte-exact `q_0=(e_0,0)`; all `200/200` zero-carry local cells produce exact `L_0` and exact copied `q_1`. |
| C4 local oracle | `400/400` hard `(digit,next_carry)` cells exact in both site views with `400/400` agreement. |
| C5 context invariance | `6,400/6,400` observations exact over 400 cells crossed with the 16 contexts below. |
| C6 canonical two-column closure | `20,000/20,000` trajectories exact: `2` operations times `10^4` two-column operand assignments, all from exact zero-carry `q_0`. |
| C7 commit order | Digit-first, carry-first, and no-emission schedules produce byte-identical successor registers in all `20,000` canonical trajectories. |
| C8 online transcript firewall | Internal `PASS`, `DROP`, and precommitted `REPLACE` modes produce byte-identical private trajectories and halt in all `20,000` canonical cases. |
| C9 carry interchange | `400/400` different-carry donor swaps follow donor carry and `400/400` same-carry shams are invariant. |
| C10 latch-to-successor consumption | On the exact 2,000-case board in Section 8.4, `q_1.carry` follows the intervened `L_0.carry`, the second hard pair follows it, and double flip restores baseline in `2,000/2,000`. |
| C11 stale/precomputed negatives | Stale carry follows the flipped latch in exactly `900/2,000`; source-precomputed carry follows it in exactly `0/2,000`; any other count is implementation or board drift. |
| C12 cursor interventions | All `80,000` width-two selected-column observations follow swapped cursor support while preserving carry. |
| C13 digit-latch sham | Every changed digit latch changes only current emission; `20,000/20,000` successor trajectories are unchanged. |
| C14 structural state | Reflection finds exactly cursor support plus one carry bit, zero retained result bits, zero continuous state, and zero generated KV bytes visible to transition. |
| C15 host boundary | For the candidate and admissible positive controls, host arithmetic, parser repair, host state update, verifier, retry, and external executor calls are each exactly zero during model-side inference. Fixed in-runtime `q_0`, `SHIFT`, `COPY_CARRY`, END, emission, and firewall operations are separately counted as essential execution. Planted violating negatives disclose their forbidden calls or sidecar bytes and cannot qualify as positives. |
| C16 negative controls | Result-history, hidden-step, stale-source, generated-KV, carry-zero, cursor-zero, planted output-dependent serializer, stale-carry, and source-precomputed-carry cheats are each rejected by their designated gate; the output-independent RNN positive is admitted. |
| C17 reproducibility | Two fresh processes produce byte-identical canonical reports and all resource receipts recompute independently. |

The 16 C5 contexts are the exact Cartesian product

```text
width             in {4,6,8,10}
terminality       in {nonterminal,terminal}
prior-result view in {all-zero,adversarial-nonzero}
```

The adversarial prefix is generated deterministically from the local-cell index
and is guaranteed to differ from the all-zero prefix. Neither prefix may enter
the local trunk provenance.

Any single failure is CPU **NO-GO**. A complete executable binding plus CPU pass
would authorize only independent review of a possible neural preregistration.
It would not authorize training or an accelerator.

## 14. Recommended autonomous gate schema

No learned experiment is executable under this file. A later experiment is
admissible only after an executable protocol binds Section 12, the CPU gates
pass, an independent review clears the bytes, data/resource receipts are
immutable, and the digit-motor autonomous result is sealed. The carry motor may
enter only after its separate reviewer authorizes it; this lane may not bypass
that gate.

If the candidate 2,400-episode board in Section 12 is adopted, the future
generator must also create a dedicated `1,200`-case successor-consumption board:

```text
6 regimes * 200 cases
per regime:
  100 additions with second-column a_1+b_1=9
  100 subtractions with second-column a_1=b_1
  within each operation, baseline L_0.carry balanced 50/50
```

Every case begins from exact `q_0`, flips only `L_0.carry`, and runs through the
online opaque interface. The board, balancing algorithm, and denominator are
only candidate requirements until bound to generator and board hashes.

The future generator must also seal one negative-control-only sidecar bit per
case: the unintervened baseline `L_0.carry`. That sidecar is absent from
`source_tokens`, treatment, PEDC, and R-SEQ. The planted stale control writes
constant zero; the planted precomputed control writes the sealed sidecar and
ignores the runtime latch. The latter is intentionally forbidden external
state, must be charged as one sidecar bit per case plus its generator/auditor
execution, and exists only to prove that the gate rejects precomputed carry.
Because baseline carry is balanced `50/50` within each operation and the latch
is flipped, stale zero follows in exactly `600/1,200` register writes and the
sealed baseline bit follows in exactly `0/1,200`.

### 14.1 Absolute learned-package gates

Under the candidate denominators, every seed must satisfy all of A0-A16:

| Gate | Exact requirement |
|---|---|
| A0 executable binding | Every Section 12 generator, secret, support, denominator, source, instrumentation, report, and validator field is bound to immutable bytes before fitting. |
| A1 parameter and source identity | Added parameters exactly `19,202,060`; total exactly `144,283,724`; frozen base and all scientific source hashes match the executable protocol. |
| A2 initialization/base case | All `2,400/2,400` episodes begin with byte-exact `q_0=(e_0,0)`; every first aperture reads column zero/carry zero; every report contains `q_0`, `L_0`, and copied `q_1`. |
| A3 site agreement | PEDC has `100%` hard whole-pair agreement on every scored transition and intervention. Site-code use is not required and cannot be inferred from agreement. |
| A4 complete local table | `400/400` local cells exact in both whole-pair views after training. |
| A5 context invariance | At least `99.9%` same-local-tuple invariance separately by width, terminality, and result-history stratum defined by the hashed generator. |
| A6 one-step pair | At least `99.5%` aggregate and `99.0%` in every confirmation regime. |
| A7 two-step pair | At least `95%` exact in every regime and at least `95%` separately on carry/borrow-boundary windows. |
| A8 full trace | At least `90%` exact source-to-END transition streams separately in all six regimes. |
| A9 carry interchange | At least `99%` different-carry donor following and `99.5%` same-carry sham invariance in every regime. |
| A10 latch-to-successor consumption | On all `1,200` dedicated cases, `q_1.carry` equals intervened `L_0.carry` byte-for-byte; the second hard pair follows the intervention in at least `99%` per regime; double flip restores baseline in at least `99.5%` per regime. |
| A11 stale/precomputed controls | On the balanced board, stale `q_1.carry=0` follows the flipped latch in exactly `600/1,200` register writes and the sealed-sidecar precomputed carry in exactly `0/1,200`; their behavioral second-pair following is at most `60%` and `5%`, respectively. |
| A12 cursor causality | At least `99%` selected-column and successor-support following in every regime. |
| A13 online output erasure | Internal `PASS`, `DROP`, and precommitted `REPLACE` modes give `100%` identical private register bytes, hard future pairs, and halt locations. Separate host-side reruns do not count. |
| A14 no-result/external path | For PEDC and R-SEQ, retained result-tape bits, generated-token KV bytes visible to transition, host arithmetic calls, external execution calls, parser repairs, verifier calls, and retries are all exactly zero; fixed runtime execution operations are nonzero and separately counted. Planted violating negatives disclose their forbidden resources and are not eligible positives. |
| A15 finite preservation board | On the exact hashed preservation board, router fires are `0/N` and treatment logits are bit-identical to the frozen base in `N/N`; `N` and all prompt bytes must be frozen before fitting, and the preservation claim is limited to those `N` cases. |
| A16 custody | All row evidence, seeds, checkpoints, traces, reports, and resource receipts are complete, immutable, and independently recomputable. |

The `90%` width-10 trace gate cannot be replaced by local accuracy. If
`r_1,...,r_10` are the step-conditioned success probabilities along successful
prefixes, chain factorization gives trace success `prod_i r_i`; `90%` trace
success therefore requires their geometric mean to be at least
`0.90^(1/10) = 98.9519%`. It does **not** imply that every individual step has
at least `98.9519%` accuracy, and it does not assume independent errors.

### 14.2 Comparative mechanism gates

Passing A0-A16 is an **engineering GO** for the entire bounded PEDC package, not
a PEDC-component result or learned-reasoning result.

R-SEQ validity must not import PEDC's absolute treatment floors. In particular,
R-SEQ is **not** required to pass A3-A12, A7's `95%` two-step floor, or A8's
`90%` trace floor. A future executable protocol must instead freeze a
development-only R-SEQ validity board, row-disjoint from both training and
confirmation, and bind all of the following criteria before fitting:

1. strict checkpoint load, the scheduled update count, finite losses/logits,
   a post-fit parameter hash different from initialization, deterministic
   replay, and every M1 equality receipt must pass;
2. byte-exact `q_0`, `SHIFT`, `COPY_CARRY`, END, online interceptor, and
   no-host/no-result-path audits must pass independently of arithmetic score;
3. a frozen rational-logit selector board must exercise agreement and
   disagreement cases, reproduce the exact R-SEQ sum-logit rule, and trace its
   selected carry byte through `COPY_CARRY` into `q_(t+1)`; the stale and
   sealed-sidecar implementations must produce their preregistered failures;
4. on a uniformly weighted 400-cell development local board, R-SEQ must predict
   all ten digits and both carry values and exceed, by at least `5.0` percentage
   points on every seed, each of its same-seed untrained initialization, the
   best fixed-pair predictor, and the planted carry-zero policy in one-step hard
   pair exactness; and
5. on the 20 carry-sensitive event pairs (`10` addition pairs with `a+b=9` and
   `10` subtraction pairs with `a=b`), changing only incoming carry must change
   R-SEQ's hard pair in at least `15/20`, while deterministic duplicate queries
   must be byte-identical in `40/40` cells.

These criteria are independent of PEDC scores and confirmation outcomes. They
reject an unloaded, untrained, nonfinite, constant, carry-blind, dead-selector,
or forbidden-path control without imposing any confirmation two-step or trace
accuracy floor. They therefore place no algebraic lower bound on the R-SEQ
scores used by M3-M4. If R-SEQ fails one criterion, the outcome is NO-GO for
PEDC-specific attribution; the failed arm may not be replaced or retrained
after confirmation is visible.

A PEDC-specific interface claim additionally requires all of M1-M4. The
separate shared-package efficiency claim additionally requires M5:

| Gate | Exact requirement |
|---|---|
| M1 control completion | R-SEQ completes with exact equality receipts for every Section 6/7 match field. G-SER and D-ALL complete with valid actual receipts and are labeled favorable unmatched controls. |
| M2 primary-control validity | R-SEQ satisfies all five prebound development-only validity criteria above on every seed. No A3-A12 treatment floor is imported, and all confirmation scores are reported unconditionally. |
| M3 two-step advantage | PEDC exceeds exactly matched R-SEQ by at least `10.0` percentage points in mean confirmation two-step exactness, with a positive difference on every seed. |
| M4 OOD trace advantage | PEDC exceeds exactly matched R-SEQ by at least `10.0` points in the mean of the four value/width-OOD full-trace regimes, with a positive difference on every seed. |
| M5 incumbent efficiency | PEDC exceeds the best admissible unpadded digit/carry motor bundle by at least `5.0` points on width-8/10 mean full-trace exactness, with a positive difference on every seed. |

There is no post-fit selector-swap promotion gate. Under A3, each PEDC view has
the same hard winner on every scored transition. With the same frozen tie rule,
that winner also wins the componentwise logit sum, so replacing the selector on
the fitted PEDC checkpoint is behaviorally identical. It is a diagnostic only.
The R-SEQ scored in M3-M4 is the single separately trained, fully bound primary
arm from M1-M2; no additional retrained selector arm is introduced.

If PEDC passes A0-A16 but R-SEQ ties, retain the simpler ordinary transducer and
record **NO-GO for PEDC-specific advantage**. If exact PEDC/R-SEQ matching fails,
only a full-package comparison is admissible. If D-ALL ties, dense
recomputation remains sufficient. If G-SER ties, inspect its firewall path: an
output-dependent instance is serialization; an output-independent instance is
another recurrent/transducer realization. If only teacher-forced or one-step
gates pass, no closed-loop claim is allowed. If M1-M4 pass but M5 fails, the
PEDC-specific interface result may stand, but the shared-package efficiency
claim is withheld.

## 15. Final recommendation

```text
CPU NOW:
  NO-GO to execute. GO only to author and independently review a separate
  executable preregistration that binds Section 12 and C0-C17 to exact bytes.

NEURAL OR AUTONOMOUS RUN NOW:
  NO-GO. This theory schema freezes neither an experiment nor matched training;
  the CPU protocol is absent, the digit-motor autonomous result is pending under
  the stated premise, and the carry lane remains separately review-gated.

H100 NOW:
  NO-GO. This file authorizes no accelerator launch now or by implication.

EVENTUAL ENGINEERING GO:
  only under a later executable protocol if every A0-A16 gate passes on every
  frozen seed. The claim is a bounded FST/RNN package with a learned local table,
  not learned end-to-end reasoning.

EVENTUAL PEDC-SPECIFIC INTERFACE GO:
  only if every A0-A16 and M1-M4 gate passes, PEDC/R-SEQ matching is exact, and
  no hidden result tape, generated-token recurrence, host arithmetic, repair,
  retry, or missing primary control exists.

EVENTUAL SHARED-PACKAGE EFFICIENCY GO:
  only if the PEDC-specific interface gate passes and M5 also passes.

ANY OTHER OUTCOME:
  NO-GO for PEDC-specific attribution; preserve the result as package-level
  localization or retain the simplest ordinary control that actually passes.
```
