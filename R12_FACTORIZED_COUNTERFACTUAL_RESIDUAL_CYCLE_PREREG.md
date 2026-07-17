# R12 Factorized Counterfactual Residual Cycle Preregistration

**Status:** SECOND REPAIRED DRAFT AFTER INDEPENDENT NO-GO. The v2 CPU
structural falsifier passes locally, but a fresh independent review is still
required.
The larger packet remains dormant unless the strictly smaller carry-only
writer/reader control fails. No neural implementation, training run,
accelerator job, architecture promotion, capability result, reasoning result,
SoTA claim, or novelty claim is authorized until this document and a numeric
resource board are frozen in a clean source snapshot.

**Protocol:** `R12-FCRC-CPU-v2`

## 1. Evidence boundary from the canonical r3 diagnosis

Canonical DRS causal-cycle job `691847` is mechanically valid. Its immutable
report is `artifacts/evals/drs_causal_cycle_post_drs_r3.json`, SHA-256
`0b927fee009de5e5cf87971ecaf390c716d6d9acb5644cabe3c176f6da9d4e7a`.
The locked facts relevant to this proposal are:

| r3 endpoint | Result | Consequence here |
|---|---:|---|
| direct two-token ceiling | `50/50` | the existing decoder can express each target state |
| irrelevant-transplant invariance | `49/50` | the tested late site is mostly context-specific |
| counterfactual residual first state | `14/50` | residual-to-token writing/serialization fails |
| integrated two-call cycle | `9/50` | the residual is not an autonomous state cycle |
| teacher-forced base carry | `30/50` | carry consumption is materially weak |
| teacher-forced base digit | `45/50` | digit response is stronger than carry response |
| same-target rescue | `5/12`, overall `-14pp` | no native residual rescue signal |

The admissible diagnosis is narrow: a causally active late digit-bearing
residual and a sufficient token motor exist, but reliable state writing,
carry update, and unpatched multi-step consumption do not. R3 did not show that
a compact state is learned, portable, sufficient, or better than ordinary
recurrence or SFT.

### Post-hoc carry-path localization

The following re-slicing was specified after the primary r3 result and is
therefore secondary, post-hoc localization. It does not change any frozen r3
decision or serve as a preregistered primary endpoint. The v2 falsifier derives
it directly from the immutable r3 bytes and rejects any artifact whose SHA-256
differs from the canonical value above.

- In the counterfactual both-site arm, the carry hook fired in `50/50` cases,
  but the later digit hook was reached in only `16/50`; the generated carry
  token usually failed to switch, so generation diverged before the digit site.
- Conditional on reaching that digit site, the full counterfactual state was
  exact in `14/16 = 87.5%` cases.
- Direct target-token forcing reached both sites in `50/50` and was exact in
  `50/50`.
- Width-8 transcripts often contained the correct next digit while carry
  remained stuck at one.

Conditioning on site reach is selection-biased and cannot prove that carry is
the only defect. It does make a smaller carry-path repair a mandatory control:
the packet architecture is not justified if a dedicated learned carry
writer/reader or feature-amplification adapter closes the same endpoints.

## 2. Terminal-carry identifiability no-go theorem

Let one decimal transition input be

```text
x = (op, a, b, c, p, w),    tau(x) = 1[p = w - 1].
```

Let the intended transition be

```text
T(x) = (d(x), c'(x)).
```

Define a terminal-zero alternative

```text
T0(x) = (d(x), c'(x) * (1 - tau(x))).
```

**Theorem.** If every terminal addition in a training set `D` has
`c'(x)=0`, then `T(x)=T0(x)` for every `x` in `D`. On every terminal-overflow
addition with `c'(x)=1`, `T(x) != T0(x)`. Therefore an observational objective
whose labels are only `D` gives the two laws identical empirical risk and
cannot identify the intended terminal-carry law without an additional
inductive restriction or added support.

**Proof.** On nonterminal examples, `tau=0`, so the definitions are equal. On
terminal training additions, the premise gives `c'=0`, so multiplying by
`1-tau=0` does not change the already-zero bit. On terminal-overflow additions,
`tau=1` and `c'=1`, so `T0` changes the bit from one to zero. This constructs
two hypotheses with identical restrictions to `D` and different restrictions
to the omitted support. QED.

The canonical r3 report does not establish an exact training-support count, so
this preregistration makes no such empirical claim. The independent factorial
data audit must measure and bind the support used by each learned arm. The CPU
falsifier supplies only the finite local witness: the intended and terminal-zero
laws agree on all `100/100`
addition cells with next carry zero and disagree on all `100/100` omitted
addition cells with next carry one.

This theorem is about identifiability, not optimization. More epochs on the
same support cannot select `T` over `T0`. Adding terminal-carry examples tests
the data remedy. Removing terminality from the local operator's causal inputs
tests an architectural remedy. Neither remedy is guaranteed to learn.

## 3. Bounded candidate architecture

FCRC is a tied recurrent transducer over read-only source memory and one hard
packet. It emits a least-significant-first trace, not a conventional decimal
answer. It is not the smallest admissible repair: with an independently supplied
cursor, decimal carry requires and admits one arithmetic bit. Phase is also
derivable from cursor plus `END`. The carry-only rank-8 writer/reader control is
therefore run first; FCRC cannot advance merely because its own endpoints pass.

### 3.1 Read-only source

The source contains only:

```text
S = (op, a[0:w], b[0:w], END).
```

The source encoder or immutable source KV may be computed once. Generated
symbols are never appended to that KV. The source is counted as external
read-only input memory and is not called a compact packet.

### 3.2 Hard packet

The complete cross-cycle mutable state is

```text
q_t = (p_t, c_t, phase_t)
phase in {RUN, FINAL, HALT}.
```

The schema has exactly three scalar fields. It is fixed in field count, not in
information capacity. For width `w`, its allocated logical capacity is

```text
ceil(log2(w + 1)) + 1 + ceil(log2(3)) bits.
```

This is `5, 6, 6, 7` bits at widths `2, 4, 6, 8`. Any claim of constant memory
that omits the logarithmic cursor is false. A list, tensor with width-dependent
rank, result prefix, operand copy, per-position slot, or dynamically added
field is packet growth and an automatic rejection.

Every categorical value must be a canonical built-in scalar. Python integer,
string, tuple, tensor, or packet subclasses are forbidden because an apparently
one-bit value can otherwise carry an unbounded hidden payload. Runtime packet,
address, local-result, emission, and step records reject noncanonical scalar or
container types before use.

### 3.3 Learned hard address interface

The address module is

```text
A_theta(H_source, p_t) -> hard_one_hot(op_t, a_t, b_t) or END.
```

Only the hard categorical output crosses into the local operator. No soft
address logits, source residual, width embedding, terminal bit, absolute
position embedding, result prefix, or generated-token residual may bypass the
hard interface. Straight-through gradients are allowed during training, but
the forward causal path must be the exact categorical value used at inference.
Address accuracy is scored separately so local arithmetic cannot hide address
errors.

### 3.4 Position-blind local operator

The only local map is

```text
F_theta(op_t, a_t, b_t, c_t) -> hard_one_hot(d_t, c_(t+1)).
```

Its callable and tensor dependency surface contains exactly those four inputs.
It cannot receive `p`, `w`, `END`, terminality, source identity, source hidden
state, result prefix, decode position, phase, or emitted-token history. Thus
the terminal-zero alternative in Section 2 is outside this local hypothesis
class. The controller must copy `c_(t+1)` into the packet identically at
terminal and nonterminal positions.

There are only `2 * 10 * 10 * 2 = 400` local input cells. The operator is
extensionally equivalent to a 400-entry lookup table. Learning this table is
not by itself reasoning and cannot support a novelty claim.

### 3.5 Fixed packet update and late residual actuator

For `phase=RUN`, the architecture performs:

```text
(op_t, a_t, b_t) = A_theta(H_source, p_t)
(d_t, c_next)    = F_theta(op_t, a_t, b_t, c_t)
emit d_t through M_theta at a frozen late-layer site
p_next            = p_t + 1
phase_next        = FINAL if p_next reaches END else RUN
q_(t+1)           = (p_next, c_next, phase_next)
```

For `phase=FINAL`, the same late actuator emits a typed terminal-carry symbol
from `c_t` and changes phase to `HALT`. The deterministic cursor increment and
phase schedule are supplied architectural control logic. They are not learned
planning and must be reported as such.

`M_theta` receives only the hard local result or the typed final carry. It may
write a late residual and token logits, but its emitted token is write-only.
The next cycle receives the immutable source and `q_(t+1)`, never a generated
token, generated-token KV entry, result tape, parsed text, verifier result, or
host-repaired state.

The primary output is exactly `w` least-significant-first digit symbols plus
one typed terminal carry/borrow symbol. Reversing that trace into a conventional
integer with host code is external execution and cannot count as autonomous
answer generation. A later learned formatter requires its own state and gates.

## 4. Training hypothesis and counterfactual constraints

The bounded hypothesis is:

> Given terminal-carry-complete support, hard source addressing, a
> position-blind local operator, and an explicitly trained late actuator, FCRC
> may learn a more reliable two-step carry cycle and width/value transfer than
> a text-mediated DRS policy under a fully reported resource vector.

The experiment must train all learned modules. A hard-coded address, decimal
operator, next-state table, carry update, or token choice is an oracle and
disqualifies the neural result.

Permitted objectives are:

1. address categorical loss on `(op,a[p],b[p],END)`;
2. local digit and next-carry categorical loss;
3. late actuator token loss;
4. same-local-tuple consistency across position, width, and terminality;
5. carry-swap counterfactual loss, with all other local fields fixed;
6. same-target actuator swap and different-target actuator swap losses;
7. autonomous unpatched two-step loss after a teacher-forcing warmup.

Every control receives the same labeled examples, counterfactual pairs, and
supervised targets unless the difference is the explicitly named treatment.
Oracle packets or target residuals are forbidden at claim-bearing evaluation.

## 5. Required controls and equivalences

### 5.1 Computational equivalences

- At any fixed maximum width and precision, FCRC is a deterministic finite-state
  transducer with read-only input. It defines no new computational class.
- An ordinary RNN, GRU, tied recurrent transformer, Universal Transformer, or
  sufficiently large lookup table can simulate it.
- A transformer with explicit state tokens can simulate the packet while
  charging those tokens and their KV memory.
- Fixed-depth unrolling simulates the complete board at a fixed maximum width.
- The 400-cell local operator is exactly table-equivalent.
- A hard packet is a coordinate choice for recurrent state, not proof of an
  ontologically distinct workspace.
- Counterfactual swaps are causal supervision, not a new reasoning primitive.

No primitive novelty, world-first, general reasoning, or SoTA claim is allowed
without separate prior-art evidence and a demonstrated resource advantage.

### 5.2 Matched controls

The following arms are mandatory:

1. **FCRC treatment.** Exact architecture in Section 3.
2. **Token-SFT control.** Same base checkpoint, corpus rows, supervised tokens,
   counterfactual examples, optimizer updates, optimizer, and seeds. It uses
   ordinary visible state tokens and reports generated KV bytes.
3. **Generic recurrent control.** Same immutable source encoder, address
   outputs, packet cardinality, recurrent steps, late actuator, data, seeds,
   optimizer updates, and parameter/FLOP allocation. Its update is a generic
   jointly learned recurrent map rather than the factorized local map.
4. **Carry-only rank-8 writer/reader control.** Keep the existing text-mediated
   DRS cycle and add no hard packet, address module, local arithmetic module, or
   digit adapter. At layer 29, a rank-8 additive adapter may act only at fixed
   protocol carry-write sites and at the corresponding next-call carry-read
   sites. Site masks come only from fixed protocol delimiters. It receives no
   target carry, oracle direction, repair signal, or verifier at inference.
   All parameters and FLOPs are counted, and its ordinary generated-token KV
   remains visible in the resource vector. This is the mandatory small control
   motivated by the post-hoc `14/16` localization.
5. **400-entry learned table control.** Same address and actuator; a learned
   categorical table replaces `F_theta`. This is an explicit collapse control.
6. **Hard oracle upper bound.** Hard address and decimal transition. It is
   charged as external execution and is never a treatment.

If the carry-only adapter or generic recurrence closes the result, the larger
packet architecture has no demonstrated advantage even if the resulting
engineering artifact is useful. If matched token SFT closes the result, no
architecture efficiency claim is allowed. If the learned table closes the
local-operator result, the expected finite-table collapse is confirmed and no
special claim about `F_theta` is allowed; that outcome alone does not test the
packet against the non-packet controls.

## 6. Resource-vector accounting

Before any neural launch, each arm must publish a numeric immutable receipt:

```text
R = (
  base_parameters,
  added_trainable_parameters,
  train_examples,
  input_tokens,
  supervised_tokens,
  counterfactual_pairs,
  optimizer_updates,
  training_flops,
  inference_flops_per_cycle,
  recurrent_cycles_per_example,
  sequential_depth,
  packet_fields,
  packet_bits_by_width,
  immutable_source_cache_bytes,
  generated_token_kv_bytes,
  result_tape_bits,
  emitted_symbols,
  oracle_calls,
  external_executor_calls,
  wall_time,
  accelerator_type
).
```

The FCRC and generic recurrent arms must match examples, tokens, pairs, updates,
seeds, packet cardinality, cycles, and sequential depth exactly; added
parameters and measured train/inference FLOPs must be within `1%`. Padding must
be reported separately and never described as useful computation. The token
SFT arm must match data and updates exactly; its different state and KV costs
remain visible rather than being scalarized away.

The carry-only rank-8 control must use the same checkpoint, data rows,
supervised targets, counterfactual pairs, updates, optimizer, and seeds. It must
not be padded to the larger FCRC parameter or FLOP budget: its smaller resource
vector is part of the control's advantage and must remain visible.

Use at least three frozen seeds: `1337`, `7331`, and `20260717`. No arm may be
selected by best seed. Report every seed and the mean.

The CPU positive has the explicit vector boundary:

- zero trainable parameters;
- three packet fields and `5/6/6/7` logical bits at widths `2/4/6/8`;
- zero result-tape bits and zero generated-token KV causal bits;
- `w` hard-coded address calls, `w` decimal calls, `w+1` actuator calls,
  and `w+1` fixed controller transitions;
- `3w+1` hard-coded substitutes for modules that must be learned in a neural
  arm, and `4w+2` total external execution/control calls;
- `w+1` sequential steps;
- `2w` read-only operand digit symbols plus operation and endpoint controls;
- `w+1` emitted symbols.

It is therefore an oracle mechanics witness, not learned reasoning.

## 7. Structural collapse tests and shams

Every implementation must pass all conditions before training:

1. packet reflection returns exactly `(cursor, carry, phase)` with fixed scalar
   fields and no dynamic payload;
2. the address source surface is exactly `(source,cursor)` and the local
   operator surface is exactly `(op,a,b,c)`;
3. the digit and terminal-carry actuator surfaces receive only their frozen
   local-result or packet input;
4. all 400 local cells are invariant under nonterminal cursor changes,
   terminal/nonterminal, widths 4/6/8, result-prefix, and generated-history
   changes;
5. the terminal-zero negative is detected on every affected carry-one cell;
6. dedicated cursor, width-6, width-8, result-prefix, and generated-history
   leakers are detected;
7. terminal and nonterminal controller paths copy the same `c_next` on all 400
   local cells;
8. observer returns and fake KV payloads cannot affect any later packet or
   emission;
9. adding a result-tape field is rejected;
10. carry, cursor, and phase each have a collision witness showing why removing
   the field changes behavior;
11. table equivalence and ordinary-RNN equivalence are explicitly admitted.

Neural shams are frozen as:

- same local tuple, different width and terminality;
- same local tuple, randomized already-emitted prefix;
- same packet and source, generated history dropped versus randomized;
- same-target late residual transplant;
- different-target digit transplant;
- different-target carry transplant;
- packet interchange between matched-address cases, where continuation must
  follow the donor packet's carry and cursor without importing donor source;
- zeroed packet adapter and parameter-count-matched dead adapter.

Any soft side channel around a hard category, including address logits or
continuous source residuals passed to `F_theta`, is a structural failure even
if the shams happen to pass empirically.

The CPU source audit is deliberately narrow: it binds the context operator,
local operator, address source, digit actuator, and terminal actuator to their
original in-module callable identities. Each must have no closure cells,
defaults, function attributes, nonlocals, or mutable globals and must have an
exact allowlist of referenced globals and attribute names. This is source-bound
evidence under a frozen file hash, not a theorem about arbitrary Python purity
and not a substitute for a tensor-provenance audit in the neural
implementation. Alternate same-signature callables automatically fail even if
a finite behavior sample happens to be invariant.

## 8. Frozen evaluation boards and neural gates

The existing 1,500-row DRS heldout set has already been inspected and is a
development board only. It cannot be the sole promotion board. Use a sealed
commit-reveal protocol for a fresh confirmation board with `300` cases in each
regime:

```text
fit_w4, fit_w6, value_ood_w4, value_ood_w6, width_ood_w8
```

Both operands in each regime must belong to that regime's declared scalar
support (inclusive interval unions):

```text
fit_w4:       [1,000, 3,999] U [6,000, 8,999]
value_ood_w4: [4,000, 5,999] U [9,000, 9,999]
fit_w6:       [100,000, 399,999] U [600,000, 899,999]
value_ood_w6: [400,000, 599,999] U [900,000, 999,999]
width_ood_w8: [40,000,000, 59,999,999] U [90,000,000, 99,999,999]
```

The union of every declared fit scalar is disjoint from the union of every
declared value-OOD scalar, including cross-width comparisons. Pair-level
decontamination remains necessary but is not sufficient for this gate.

Each regime must be operation-balanced. Addition must be balanced between
terminal carry zero and one. Valid nonnegative subtraction must retain terminal
borrow zero while balancing examples with and without intermediate borrows. No
training example may share a complete operand pair with confirmation. The
generator, source hashes, normalization contract, and numeric resource board
must be immutable before fitting. Then a separate CPU custodian job draws a
256-bit secret from kernel entropy, publishes only
`SHA256(b"FCRC-confirm-v2\n" + secret)` by exclusive-create mode `0444`, and
stores the unrevealed secret outside every training snapshot. The exact seed
job ID, script hash, commitment, and one-output/no-retry rule are recorded before
any fit starts. Training sees the commitment but not the secret or board.

Only after every candidate checkpoint is immutable may the custodian reveal the
single secret, verify the commitment, and generate the board once. The board
seed is `SHA256(b"FCRC-board-v2\n" + secret + J)`, where `J` is canonical JSON
of the frozen source and resource hashes. The reveal, generator, board, and
normalization receipt are immutable and independently replayed. A second seed,
pre-fit reveal, manual filtering, reseeding, or failed attempt followed by a new
secret invalidates the experiment. This is an honest-process custody boundary,
not remote attestation against a malicious same-UID actor.

### 8.1 Mechanical preflight GO

`train/fcrc_falsifier.py` must report every gate true. The repaired v2
implementation currently reports `28/28` gates true locally:

- context invariance `400/400` local classes;
- terminal-zero negative detected on `100/100` affected cells;
- cursor, width-6, width-8, result-prefix, and history negatives each detected
  on `400/400` cells;
- terminal and nonterminal carry updates `400/400` exact;
- autonomous two-step oracle mechanics `20,000/20,000`, including `9,000`
  carry/borrow boundary cases;
- full mechanics rollouts `500/500`, `100/100` in every named regime; each
  regime has exactly 25 addition/carry-zero, 25 addition/carry-one, 25 valid
  subtraction/no-intermediate-borrow, and 25 valid
  subtraction/with-intermediate-borrow cases;
- all declared fit/value-OOD scalar-support intersections are empty, all
  `1,000` generated operands belong to their declared support, and the observed
  fit versus value-OOD scalar intersection is empty;
- exact non-subclass scalar/container validation plus `15` address/actuator
  negatives covering mutable globals, closures, nonlocals, defaults, and
  function attributes; the three mutable-global variants demonstrably change
  traces but still fail static admission;
- immutable-r3 post-hoc derivation: carry site reached `50/50`, digit site
  reached `16/50`, and reached-plus-full-exact `14/16`;
- mechanics board SHA-256
  `c8eb388c21414f36b6aae099a3ccd39e1119f7e7171fcf9a9043890fb689d949`;
- table-collapse SHA-256
  `553a6015bdce2c455acb42f4b07e689d5cada87dd9b1dc2e5b2e07fe0f8499e4`.

This is only a local CPU mechanics pass. It does not authorize an isolated
learned pilot until the fresh independent review, clean source freeze, resource
freeze, confirmation custody, and rank-8 negative required by Section 9 all
exist. It is not a neural GO.

### 8.2 Learned pilot GO

All thresholds apply separately to every seed unless explicitly stated:

1. structural gates remain exact after integration;
2. hard address accuracy is at least `99%` in every confirmation regime;
3. one-step local `(digit,carry)` exactness is at least `98%` over the complete
   400-cell board;
4. autonomous, unpatched two-step exactness is at least `90%` balanced
   aggregate and `80%` in every regime;
5. autonomous two-step exactness on carry/borrow-boundary cases is at least
   `80%` in every regime;
6. full source-to-HALT trace exactness is at least `60%` balanced aggregate and
   `50%` separately in `value_ood_w4`, `value_ood_w6`, and `width_ood_w8`;
7. no evaluation call uses oracle packets, teacher tokens, target residuals,
   parsing repair, verifier selection, retry, generated-token KV, or host state
   updates;
8. width, terminality, result-prefix, and generated-history shams are at least
   `99%` invariant in every regime;
9. FCRC exceeds matched token SFT, generic recurrence, and the carry-only
   rank-8 writer/reader control by at least
   `10` percentage points on mean autonomous two-step exactness and mean full
   OOD trace exactness, with a positive difference on every seed;
10. no packet/resource receipt changes after launch.

Passing endpoints 1-8 but failing endpoint 9 is an **engineering GO** for the
best-performing arm and a **NO-GO for an FCRC-specific advantage claim**.
Failing any of endpoints 1-8 or 10 is a neural **NO-GO**. Thresholds may not be
relaxed after observing results.

## 9. Exact GO / NO-GO boundary

**GO to prepare an isolated neural pilot** only if the CPU report is fully
true, a fresh independent review clears the repaired v2 source, all three source
files are frozen in a clean snapshot, the commit-reveal confirmation custody is
allocated, every resource-vector field is numeric for all arms, **and the
smaller carry-only rank-8 writer/reader has already failed its own frozen causal
gates**. The confirmation board itself remains unrevealed until all candidate
checkpoints are immutable.

**GO for FCRC as a mechanism** only if every learned gate in Section 8.2
passes, including the matched-control deltas on every seed.

**NO-GO** immediately if any of the following occurs:

- packet fields or capacity grow beyond the declared formula;
- a result tape, generated-token KV, parsed output, retry loop, verifier, host
  repair, hidden source residual, soft address side channel, or target injection
  affects a successor;
- terminality, width, absolute position, or result prefix reaches the local
  operator;
- the address or arithmetic transition is hard-coded in a neural treatment;
- the fresh board or resource receipt is missing or changes after fitting;
- autonomous two-step or OOD gates miss their frozen threshold;
- the carry-only adapter, generic recurrence, or token SFT closes the
  preregistered advantage;
- only a conventional answer produced by an external trace reverser is scored;
- only best-seed, aggregate-only, or post-hoc-selected results are reported.

The strongest possible claim from a clean pass is a resource-bounded empirical
advantage for this factorization on the frozen decimal transduction board. It
would not establish general reasoning, a new computational primitive, or
state-of-the-art intelligence per parameter.

## 10. Current authority boundary

Authorized by this draft:

1. edit and test only `train/fcrc_falsifier.py` and
   `train/test_fcrc_falsifier.py` against this mechanics contract;
2. run CPU unit tests, the deterministic falsifier, Ruff, `py_compile`, and
   diff checks;
3. review and tighten the draft before a clean freeze.

Not authorized: data generation, checkpoint loading, neural code, SFT,
accelerator submission, result promotion, runbook edits, or any capability or
novelty claim.
