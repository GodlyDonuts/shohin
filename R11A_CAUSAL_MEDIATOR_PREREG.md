# R11a v3 Preregistration: Minimal Internal Causal-Mediator Canary

> **Status:** Versions 1 and 2 failed independent review and are superseded.
> Version 3 remains preregistration-only. No v3 implementation, generated
> board, model, job, score, checkpoint, or capability result exists under this
> document.
>
> **v2 review record:** The frozen v2 artifact has SHA-256
> `11ca769036b2bc85eebd47a950e51b2bc87158668e47ba3cc5e38e4fdae68408`.
> Its independent review disposition was **NO-GO**. No v2 result may be carried
> forward; v3 is a new preregistration contract.
>
> **Scope:** One isolated wrapper around the immutable raw-200k Shohin GPT.
> R11a tests only whether a source-written latent state supports source-free,
> crossed, late-bound affine reads and whole-state causal interventions.
>
> **Authority:** This text revision does not authorize implementation, code
> changes, data generation, CPU or GPU allocation, or access to or modification
> of any live job, existing checkpoint, score, data stream, or other file. The
> protected training writer and all of its artifacts remain out of scope.
>
> **Supersession:** For R11a, the architecture, counts, data contract, controls,
> gates, and claim boundary below supersede broader or conflicting provisions
> in `R11_INTERNAL_WORKSPACE_PREREG.md`.

## 1. Frozen decision and claim boundary

R11a is the smallest internal causal-mediator canary admitted after review. A
source-only pass writes six private 96-wide slots. One tied writer cell updates
them for four rounds. A structurally separate source-free query pass reads the
same state after decoder blocks 12, 18, and 24. Queries are disclosed only
after the state value has been produced. The primary intervention crosses two
different affine queries and two different worlds, so neither a shared query
template nor a receiver-answer default can satisfy the test.

R11a does not test reportability, natural-language state updates, transfer
families, held-out consumers, broadcast, a global workspace, general
reasoning, consciousness, context compression, or public-benchmark transfer.
Those terms are prohibited in an R11a result.

The only positive claim available after every gate passes is the exact wording
in Section 18. Until then, R11a is an unvalidated design.

## 2. Immutable base and isolated wrapper

All layer numbers are one-based.

| Frozen property | Value |
|---|---:|
| Base checkpoint | Immutable raw-200k Shohin GPT |
| Decoder blocks | 30 |
| Residual width | 576 |
| Query / KV heads | 9 / 3 |
| Native head width | 64 |
| SwiGLU width | 1,536 |
| Vocabulary | 32,768 |
| Native context | 2,048 |
| Tied base parameters, including native QK norms | 125,081,664 |
| R11a adapter parameters | 1,607,334 |
| Total resident parameters | 126,688,998 |

The base checkpoint, embeddings, native QK norms, block norms, final norm, and
tied language-model head are frozen. R11a is a separate wrapper and checkpoint
namespace. It may store only adapter tensors plus immutable manifests.

With `r11a_mode=False`, all LoRA deltas, writer code, and reader code are
bypassed. On identical device, dtype, token IDs, masks, and cache state, logits
and KV tensors must be bitwise equal to the unwrapped base. An R11a checkpoint
must never overwrite, augment in place, or be loaded into a flagship
checkpoint.

## 3. Structurally separate differentiable API

The treatment forward surface is closed to exactly these four calls; there is
no generic `read`, hidden overload, or ambient model context:

```text
W = write(source_ids, source_mask, source_lengths)

logits = teacher_read(
    W,
    query_ids, query_mask, query_lengths,
    answer_prefix_ids, answer_prefix_mask, answer_prefix_lengths,
    reader_site_mask
)

next_logits, decode_session = begin_read(
    W, query_ids, query_mask, query_lengths, reader_site_mask
)

next_logits, decode_session = decode_step(
    decode_session, generated_token_ids
)
```

All IDs are rank-two integer tensors. Each mask is a same-shape boolean tensor;
each length is a rank-one integer tensor equal to that row's mask sum. True mask
bits must be one contiguous left prefix of exactly the declared length. Empty
sources and queries, left padding, holes, and length inference from a pad token
ID are forbidden. `reader_site_mask` is one batch-global frozen condition over
sites 12, 18, and 24; it cannot vary by example or carry metadata.

`write` has no query, answer, target, donor, coefficient, or metadata argument.
It returns only the numeric `[B,6,96]` tensor `W`, never source KV or a source
handle. `teacher_read` and `begin_read` have no source IDs, source mask, source
lengths, source cache, source position, or source-side handle. In the treatment
arm, the forward value of `W` is complete before any query tensor is made
available to either call.

`teacher_read` allocates and asserts a fresh empty decoder KV cache internally,
forms exactly `query || answer_prefix`, applies readers only to true query
positions, returns logits, and returns no cache or session. `begin_read` also
allocates and asserts a fresh empty cache, prefills only the true query tokens,
applies readers only during that prefill, and returns the logits predicting the
first answer token plus a `DecodeSession`.

Any temporary W/reader context inside `teacher_read` or `begin_read` is stack-
local to that call and is cleared before return. Neither call may write W,
source, reader state, or a callable into a module attribute, global,
thread-local, callback registry, cache manager, or other ambient location.
Autograd-saved tensors reachable only backward through returned training logits
are governed by the backward-only exception below.

At return from `begin_read`, `DecodeSession` is a query-only decode cache whose
complete field whitelist is ordinary decoder KV, its key-validity mask, cache
lengths, and next positions. Those values are produced solely by the query
prefill. The session contains no `W` tensor or alias, reader callable or reader
state, source tensor or alias, source/query IDs, closure, mutable model buffer,
metadata, RNG state, or audit registry handle. `decode_step` accepts only that
session and the just-generated token IDs. It has no `W`, source, mask, length,
reader-site, or metadata argument; all readers are hard-disabled, and it appends
only ordinary generated-token KV and position state to the session. EOS and the
eight-token cap are enforced by the caller solely from generated tokens.

No KV entry, residual, activation, mask, position counter, cache object, RNG
state, closure, or mutable buffer from `write` may be reused by `teacher_read`,
`begin_read`, or `decode_step`. The exact `W` tensor is the only per-example
object allowed to cross from `write` into `teacher_read` or `begin_read`. Model
parameters and frozen architecture constants are shared globally and are not
per-example transport. Free generation may reuse only the decode session created
by `begin_read`.

During training, autograd-saved source tensors may remain reachable only through
`W.grad_fn` for backward. They are not read-forward inputs, cannot be looked up
by `teacher_read` or `begin_read`, and cannot affect forward logits except
through the numeric value of W. This retained gradient history is part of the W
object, not a second source transport channel.

"Complete" or "frozen before query disclosure" refers to the forward value,
not to autograd. `W`, source taps, and the complete writer graph must remain
differentiable until answer CE backpropagation finishes. Calls to `detach`,
`item`, NumPy conversion, serialization, host execution, or a stop-gradient on
any source tap or writer state are forbidden.

The production treatment path computes and caches one `W` per source rendering.
The exact cached tensor is reused across queries. Audit hooks may capture or
replace the whole tensor, but model inference may not inspect its coordinates or
use them to choose a query, operation, or answer. There is no source-visible
read mode, source-plus-W mode, source-cache fallback, alternate API, or implicit
session field anywhere in v3.

## 4. Frozen adapter architecture

### 4.1 LoRA

Rank-8 LoRA is active only in R11a mode on the bias-free `q`, `k`, `v`, `o`,
`gate`, `up`, and `down` projections in blocks 7 through 18. Source and query
passes share the same LoRA tensors.

- Rank `r = 8`.
- Alpha `alpha = 8`.
- Scale `alpha / r = 1` exactly.
- LoRA dropout is zero.
- No LoRA bias is added.
- For a base matrix with shape `[out, in]`, `A` has shape `[8, in]` and `B`
  has shape `[out, 8]`.

The per-block count is 81,408 and the 12-block count is 976,896.

### 4.2 Source encoding and right-padding mask

The source transport is right padded. For each row, `source_lengths[b]` is
explicit, `source_mask[b,t]` is true exactly when
`0 <= t < source_lengths[b]`, and every later position is masked regardless of
its token ID. Left padding is forbidden.

The source runs through blocks 1 through 18. Residuals after blocks 6, 12, and
18 are captured without detachment. One shared learned 576-wide RMSNorm with
`eps = 1e-6` is applied at all three taps. Shared bias-free projections produce
96-wide source keys and values. A learned 96-wide tap ID for 6, 12, or 18 is
added to both projected streams before concatenation:

```text
K_src = concat(N_src(H_6) K_w + e_6,
               N_src(H_12) K_w + e_12,
               N_src(H_18) K_w + e_18)

V_src = concat(N_src(H_6) V_w + e_6,
               N_src(H_12) V_w + e_12,
               N_src(H_18) V_w + e_18)
```

In treatment, sham, reset, and query-only source writes, only true source
positions may be writer keys or values. Pad positions, dummy suffix positions,
query positions, and answer-poison positions receive negative infinity before
writer softmax. Each row must contain at least one true source token. Changing
right-pad width or pad token IDs must not change `W`. Private-query is the sole
explicit diagnostic exception: Section 7 reclassifies its concatenated row
query as true writer input and therefore cannot support the treatment claim.

### 4.3 Four tied writer rounds

The state contains six slots of width 96:

```text
W^0 in R[B, 6, 96]
```

`W^0` is one learned six-slot seed. There are exactly four rounds with one
shared parameter set, no round embeddings, no adaptive halting, and no
round-specific parameters:

```text
E^r = MHA(Q = N_cross(W^(r-1)) Q_w,
          K = K_src,
          V = V_src,
          source_mask)

U^r = W^(r-1) + tanh(g_cross) * (E^r O_w)
V^r = U^r + tanh(g_self) * MHA_self(N_self(U^r))
W^r = V^r + tanh(g_mlp) * SwiGLU(N_mlp(V^r))
```

Writer cross-attention and self-attention each use three heads of width 32.
Every workspace attention uses a fixed `1 / sqrt(32)` logit scale. Attention
logits and softmax are evaluated in fp32 and outputs are cast back to the model
dtype. The writer MLP width is 256. There is no workspace QK norm, no RoPE, no
relative-position bias, and no attention or MLP dropout. The three 96-wide
writer pre-norms use `eps = 1e-6`.

The treatment state is `W = W^4`.

### 4.4 Query-prefill-only readers

The source-free decoder input is `query || answer[:-1]` during teacher forcing.
The same `W` is available at reader sites after blocks 12, 18, and 24, but a
reader may modify only query-token residuals. Reader output is identically zero
on every teacher-forced answer-prefix position.

During free generation, readers operate only inside the one `begin_read` query
prefill. They are structurally unavailable to `decode_step` and therefore
disabled for every generated position. Workspace-mediated query residuals may
influence later tokens only through the ordinary decoder layers and the
query/generated decoder cache created from that prefill.

At site `l`:

```text
R_l = MHA(Q = N_query_l(X_l[query]) Q_l,
          K = N_state_l(W) K_l,
          V = N_state_l(W) V_l) O_l

X_l[query] = X_l[query] + tanh(g_l) * R_l
```

Each reader has three heads of width 32 and fixed `1 / sqrt(32)` scaling. Its
attention softmax is fp32. There is no reader QK norm beyond the two explicit
pre-norms, no RoPE, no positional bias, and no dropout. Answer-prefix and
generated-position reader call counts must both be zero.

### 4.5 Frozen initialization

Initialization uses CPU generator seed `20260715` and a fixed, lexicographic
parameter traversal recorded in the manifest.

| Tensor class | Initialization |
|---|---|
| LoRA `A` | Independent normal, mean 0, std 0.02 |
| LoRA `B` | All zeros |
| Writer and reader projection matrices | Independent normal, mean 0, std 0.02 |
| Six slot seeds | Independent normal, mean 0, std 0.02 |
| Three source-tap IDs | Independent normal, mean 0, std 0.02 |
| Every added RMSNorm weight | All ones |
| Writer gates | `atanh(0.1)` |
| Reader gates | Zero |

All added projections are bias-free. Initialization is performed in fp32
before casting. All matched arms must begin from the same byte-identical
adapter state and initialization hash.

### 4.6 Exact parameter count

| Trainable component | Count |
|---|---:|
| Rank-8 LoRA in blocks 7-18 | 976,896 |
| Writer cross-attention | 129,024 |
| Writer self-attention | 36,864 |
| Writer SwiGLU | 73,728 |
| Shared source RMSNorm | 576 |
| Three writer 96-wide pre-norms | 288 |
| Six slot seeds | 576 |
| Three source-tap IDs | 288 |
| Three writer scalar gates | 3 |
| Three readers, 129,697 each | 389,091 |
| **Total adapter** | **1,607,334** |
| **Tied base plus adapter** | **126,688,998** |

One reader contains 129,024 projection weights, a 576-wide query pre-norm, a
96-wide state pre-norm, and one scalar gate. Native base QK norms remain frozen;
no QK norms are added to workspace attentions.

The cached state is exactly 576 scalars per example.

## 5. Frozen deterministic data and target contract

Only the two-register domain specified here and in Appendix A is admitted. No
parent-document data rule, implicit template, library RNG default, or
implementation-dependent iteration order may fill a gap in this section.

### 5.1 Board sizes and exact OOD factors

| Board | Matched pairs | Semantic worlds | Composition |
|---|---:|---:|---|
| Train | 2,048 | 4,096 | Fit factors only |
| Calibration | 512 | 1,024 | 128 each wording, length, query, and full OOD |
| Post-calibration confirmation | 1,024 | 2,048 | 256 each wording, length, query, and full OOD |

An A/B pair is the indivisible generation, split, training, scoring, and
bootstrap unit. Train and calibration source worlds must all be generated,
source-admitted, and hash-frozen before any query template, affine coefficient,
rendered query, or serialized target for either board is sampled or computed.
Confirmation cardinalities describe a board generated only after a passing
calibration decision hash is frozen and its deterministic seed is derived.

| Stratum | Source/query wording | Event depth | Affine coefficients |
|---|---|---|---|
| Train/fit | `F0`, `F1`, `QF0`, `QF1` | fit | `C_fit` |
| Wording OOD | `H0`, `H1`, `QH0`, `QH1` | fit | `C_fit` |
| Length OOD | `F0`, `F1`, `QF0`, `QF1` | long | `C_fit` |
| Query OOD | `F0`, `F1`, `QF0`, `QF1` | fit | `C_ood` |
| Full OOD | `H0`, `H1`, `QH0`, `QH1` | long | `C_ood` |

For first or final edits, fit depth is sampled uniformly from `{2,3,4}`. For
interior edits it is sampled uniformly from `{3,4}`. Long depth is sampled
uniformly from `{6,8,12}` for every edit position. These are the only OOD axes;
single-axis strata change exactly the named factor. Canonical pair
serializations must be unique across all boards.

### 5.2 SHA-256 counter PRF and seed process

The public 32-byte seeds, interpreted from lowercase hexadecimal, are:

```text
train       a1b2cc46d0384b6f1c0402e10c7603fbafab0565e7aeeb6839866e9a019dd204
calibration 720dbcd7becb4f6c4ca2f85063e1df2817f116a969be1a52388c2e68417dc254
```

For a 32-byte seed `S`, an ASCII domain `D` containing no NUL, and counter `i`,
the only random-byte primitive for board generation, epoch shuffling, and
bootstrapping is:

```text
CTR(S,D,i) = SHA256(
    ASCII("R11A-V3-CTR") || 0x00 || S ||
    U16BE(len(ASCII(D))) || ASCII(D) || U64BE(i)
)
```

Lengths are byte lengths. `U16BE` and `U64BE` are unsigned big-endian. A
uniform draw from `[0,n)` takes `u = U64BE(CTR(...)[0:8])`, sets
`L = floor(2^64/n)*n`, rejects `u >= L`, increments the counter, and otherwise
returns `u mod n`. Fisher-Yates uses these unbiased draws from last index to
first. Every ordered bank below is sampled by index. Board generation, epoch
shuffling, and bootstrapping use no `random`, NumPy, PyTorch, hash-table order,
locale, clock, process ID, operating-system entropy, or hardware RNG. The
separate adapter initialization RNG is exactly Section 4.5 and is not used for
any board, shuffle, or bootstrap draw.

Mandatory golden vector: with the train seed, domain
`board=train/stratum=fit/pair=00000000/field=depth` (49 bytes), and counter
zero, `CTR` is exactly
`f239199fff4e4c8de03d37fd593daf2b6789cece0406fb249c1a95495f2c339a`.

Pair domains and stratum-schedule domains have exactly these forms:

```text
board=<board>/stratum=<stratum>/pair=<eight-digit-zero-padded-index>/field=<field>
board=<board>/stratum=<stratum>/scope=stratum/field=<field>
```

`board` is exactly `train`, `calibration`, or `confirmation`; `stratum` is
exactly `fit`, `wording`, `length`, `query`, or `full`. Each complete domain has
its own counter beginning at zero. The only stratum fields are
`cell_extra_order`, `slot_order`, `register_bits/<class>`, and
`direction_bits/<class>`. The source-stage fixed pair fields are `depth` and
`edit_index`. The source-attempt fields, each ending `/attempt=<a>`, are
`initial_x`, `initial_y`, `event_kind/<tt>`, `event_operand/<tt>`,
`edit_addsub_kind`, `edit_abs`, `edit_sign`, `edit_mag_anchor`,
`edit_mag_other`, `edit_transfer`, and `edit_signed_operand`, where `<tt>` is a
two-digit zero-based event index. The query-stage fixed pair fields are
`template_q0`, `template_qA`, and `template_qB`; the only redraw fields are
`coeff_A/redraw=<j>` and `coeff_B/redraw=<j>`.

`<j>` and `<a>` are unsigned decimal with no leading zeros except literal `0`;
pair index is always eight digits. A field unused by an edit/event is not drawn.
Distinct fields use distinct domains, so rejection in one stream cannot shift
another stream. Before a board's `SOURCE_FREEZE` ledger record, only its
stratum, source-fixed, and source-attempt domains may be consumed. After that
record, no source domain for that board may ever be consumed again; only its
query-template and coefficient-redraw domains may be consumed. For train and
calibration, both `SOURCE_FREEZE` records must exist before either board consumes
a query-stage domain.

The only non-board CTR domains are
`fit/epoch=<0..7>/shuffle`, using the train seed, and
`bootstrap/board=<board>/stratum=<stratum>/replicate=<0..9999>/draw`, using the
Section 12 assessor seed. In the bootstrap form, `board` is exactly
`calibration` or `confirmation` and `stratum` is exactly `wording`, `length`,
`query`, or `full`. Counters begin at zero independently for every complete
domain. Any unlisted CTR domain, field, draw, seed substitution, or cross-domain
counter reuse is forbidden.

Version 3 specifies no secret-custody process. The confirmation seed is
deterministic, and the frozen public 32-byte salt is:

```text
SALT_confirm = 0143ce90e4c3f9efad7ec8060948035b88690c997fae5723d10a08eef3380fa4
```

After, and only after, Section 14 freezes a passing calibration decision record
and its 32-byte hash `H_decision`, derive:

```text
S_confirm = SHA256(
    ASCII("R11A-V3-POST-CALIBRATION-CONFIRM") || 0x00 ||
    SALT_confirm || H_decision
)
```

`SALT_confirm` and `H_decision` above are raw 32-byte values decoded from their
lowercase hexadecimal records. The generating process records exactly one
derivation event containing the domain bytes, salt, `H_decision`, and resulting
`S_confirm`; auditor recomputation must match byte-for-byte and is not a new
candidate seed. There is no nonce, timestamp, entropy input, secret, alternate
domain, alternate salt, seed search, or reseed. This construction is auditable
ordering only: it makes no claim of secrecy, unpredictability, blinding, or
untouched randomness. Section 14 defines and limits the resulting confirmation
gate.

### 5.3 Exact initial, event, and operand banks

Initial `x` and `y` are independent uniform draws from the ordered integer bank
`I = [-32,-31,...,31,32]`. The ordered signed operand bank is
`K = [-9,-8,...,-1,1,2,...,9]`; the ordered transfer bank is
`M = [1,2,...,9]`. The ordered event-kind bank is:

```text
ADD_X, ADD_Y, SUB_X, SUB_Y, MOVE_X_Y, MOVE_Y_X,
MERGE_X_Y, MERGE_Y_X, SWAP_X_Y
```

Their atomic integer semantics are exactly:

```text
ADD(r,k):       r <- r + k                     where k in K
SUB(r,k):       r <- r - k                     where k in K
MOVE(src,dst,m): src <- src - m; dst <- dst + m  where m in M
MERGE(src,dst): dst <- dst + src; src <- 0
SWAP:           (x,y) <- (y,x)
```

The two assignments in `MOVE` and `MERGE` use the pre-event values atomically.
Every intermediate value after the complete atomic event and every final value
must be in `[-2048,2048]`. Python-style unbounded integer arithmetic is used for
the check; overflow, floating point, wraparound, clipping, and saturation are
forbidden.

### 5.4 Exact edit schedule and collision query

The ordered edit classes are `[sign,magnitude,role,operation_kind]`; positions
are `[first,interior,final]`. For a stratum of `N` pairs, repeat each of the 12
ordered class/position cells `floor(N/12)` times. Give one extra occurrence to
the first `N mod 12` cells in a PRF-shuffled permutation of the 12 cells, then
PRF-shuffle the resulting `N`-slot schedule. Thus cell counts differ by at most
one by construction, before any world or answer exists.

The first shuffle uses `cell_extra_order`; the second uses `slot_order` and
starts from the repeated cells in class-major, position-minor order. Within
each stratum and edit class containing `n` slots, create the exact binary list
of `ceil(n/2)` zeros followed by `floor(n/2)` ones and independently shuffle it
under `register_bits/<class>` and `direction_bits/<class>`. Consume each list
in ascending final pair-slot order for that class. Register bit zero means `x`
and one means `y`. Direction bit zero means the A-to-B direction listed below;
one exchanges A and B. These schedules, not post-hoc filtering, set sign,
magnitude, role, and operation directions.

Depth is one unbiased draw from the applicable Section 5.1 bank. `first` fixes
edit index zero; `final` fixes index `depth-1`; `interior` draws `edit_index`
uniformly from `[1,depth-2]`. The edit event is then:

- `sign`: draw ADD/SUB uniformly with `edit_addsub_kind` and magnitude
  uniformly from `[1,...,9]` with `edit_abs`; direction zero is `+m` in A and
  `-m` in B;
- `magnitude`: the same `ADD` or `SUB` and sign, with distinct magnitudes. For
  direction zero, draw the A magnitude uniformly from `[1,...,8]` with
  `edit_mag_anchor` and the B magnitude uniformly from `[A+1,...,9]` with
  `edit_mag_other`; direction one exchanges A/B. Draw ADD/SUB with
  `edit_addsub_kind` and sign from `[-1,+1]` in that order with `edit_sign`;
- `role`: draw `m` uniformly from `M` with `edit_transfer`; direction zero is
  `MOVE(r,s,m)` in A versus `MOVE(s,r,m)` in B;
- `operation_kind`: draw `k` uniformly from `K` with `edit_signed_operand`;
  direction zero is `ADD(r,k)` in A versus `SUB(r,k)` in B.

At every non-edit site, event kind is an independent uniform draw from the
nine-kind bank under `event_kind/<tt>` and its operand is an independent
uniform draw from `K` or `M` under `event_operand/<tt>` as applicable. At the
edit site, only the edit-specific fields above are consumed. The scheduled
direction determines A/B orientation; no answer-dependent orientation or
balancing is allowed.

`q0` is fixed from event syntax, never selected by examining numeric answers.
For a role edit, `q0(x,y)=x+y`. For every other edit, start a symbolic
different-coordinate marker `d` at edited register `r` and process only events
after the edit: ADD, SUB, and MOVE leave `d` unchanged; SWAP exchanges it;
MERGE(src,dst) sets `d=dst`. Then `q0` asks for the other final register. This
construction makes `q0(A)=q0(B)` whenever the pair remains distinct. The same
q0 token sequence is used for the two worlds.

The two q0 rows already present in the eight-row matrix are the paired
collision negative-control. They are not clean-plus-two-extra reads and are
never included in donor-follow, receiver-follow, or swap specificity.

### 5.5 Source freeze precedes every query and coefficient

No affine coefficient, query template ID, rendered query, affine target,
serialized answer, or query/answer histogram may exist while source candidates
are being generated or selected. The complete source phase in Section 5.6 is
board-wide, not pair-local: every source pair in train and calibration is
accepted and both source manifests are hash-frozen before the first query-stage
domain is invoked for either board. A source hash is never replaced after that
boundary. The syntax-derived q0 equality relation in Phase S is a source
invariant; no q0 query or answer bytes exist there.

Only after those source freezes, every pair receives two affine queries:

```text
qA(x,y) = cA1*x + cA2*y + cA0
qB(x,y) = cB1*x + cB2*y + cB0
```

The exact ordered coefficient banks are Cartesian products in the displayed
factor order:

```text
C_fit = [-3,-2,-1,1,2,3]
        x [-3,-2,-1,1,2,3]
        x [-9,-6,-3,0,3,6,9]

C_ood = [-7,-5,5,7]
        x [-7,-5,5,7]
        x [-20,-10,0,10,20]
```

Tuples are ordered lexicographically by the displayed list positions. After the
source freeze, q0, qA, and qB template IDs are sampled once from their separate
fixed domains and become immutable. At coefficient redraw index `j`, qA and qB
are unbiased uniform draws from independent `field=coeff_A/redraw=j` and
`field=coeff_B/redraw=j` streams. The query-stage checks in Section 5.6 may
reject that coefficient pair and increment only `j`; both tuples are redrawn
together. They may never redraw a template, source attempt, initial value,
event, operand, schedule bit, depth, rendering, or pair slot.

The first coefficient pair surviving every query/answer constraint is frozen.
Exhausting redraw indices `0` through `2^32-1` hard-fails the board. Reseeding,
source regeneration, source replacement, source reordering, selecting among
multiple source boards, or retaining a source because its later answers look
favorable is forbidden.

For notation only, `Eval(q,W)` means `begin_read(W,q,...)` followed by greedy
`decode_step` calls; it is not another API. The crossed evaluation matrix is:

```text
clean: Eval(qA,W_A) -> qA(A)
clean: Eval(qB,W_B) -> qB(B)
cross: Eval(qA,W_B) -> qA(B)
cross: Eval(qB,W_A) -> qB(A)
```

There is no shared q1. Metric name `q1` means the qA/qB affine family.

### 5.6 Two-phase generation, admission, and immutable ledgers

Generation has exactly two phases. Pair-slot order is preserved in every
artifact, and no later sort, replacement, score-based filtering, or balancing
is allowed.

**Phase S: source generation and source admission.** Construct all train source
pairs and then all calibration source pairs. For each board and stratum, freeze
the edit-cell, slot, register, and direction schedules. For pair slot `p`, draw
depth and edit index once. For source attempt `a`, draw only initial values,
event kinds, and operands from domains ending `/attempt=a`. Apply these checks
in order; the first failure increments only `a`:

1. Replay both worlds and reject any intermediate or final value outside
   `[-2048,2048]`.
2. Require identical initials, equal depth, exactly one differing canonical
   event at the scheduled site, legal edit semantics, and unequal final
   two-register states.
3. Derive the syntax-fixed q0 selector and require its selected final value to
   agree across A and B. This is a source-state invariant only; do not render a
   q0 query or serialize its answer.
4. Render both Appendix A source renderings exactly. Require nonempty source
   tokenizations, source-only native-context compliance, and solver-equivalent
   semantics between `S^0` and `S^1`.
5. Require a new Appendix A source-pair serialization and new normalized source
   bytes against every earlier pair in the current board and every board whose
   source manifest is already frozen: train, then calibration, then confirmation.
6. Accept the first surviving source attempt. Exhausting attempts `0` through
   `2^32-1` hard-fails the board; no query-stage fact may be consulted.

After all source slots are accepted, an independent source-admission pass
replays those six checks from the exact source bytes. It emits one strict-ASCII
JSONL source manifest per board with lexicographically sorted keys, no optional
whitespace, lowercase hexadecimal byte fields, one record per line in pair-slot
order, and one final LF. The manifest must contain:

- the preregistration, generator, solver, tokenizer, and Appendix A hashes;
- the board seed, stratum schedules, complete ordered source-domain/counter
  ledger, every attempted `a`, and each ordered rejection reason;
- for each accepted pair, its slot, accepted `a`, edit metadata, exact source-
  pair serialization, both world serializations, both source renderings, source
  IDs/masks/lengths, final states, q0 selector, and hashes of every byte/tensor
  representation; and
- a terminal record with
  `SOURCE_FREEZE = SHA256(ASCII("R11A-V3-SOURCE-FREEZE") || 0x00 ||
  all preceding manifest bytes)`.

The terminal record itself is excluded from `all preceding manifest bytes`.
The entire source bundle is then immutable. Train and calibration each require
an independently reproduced `SOURCE_FREEZE`, and both terminal records must be
frozen before Phase Q begins for either board.

**Phase Q: query, coefficient, and answer admission.** Rehash the frozen source
bundle first. In pair-slot order, sample each q0/qA/qB template ID exactly once.
Render and serialize q0 only now. Any coefficient-independent failure, including
an empty q0 tokenization, q0 answer length over eight tokens including EOS,
full-prompt context failure, prompt duplication, or any q0 frequency cap,
hard-fails the board without changing a source or template.

For each pair, start `j=0`, draw both coefficient tuples, and apply these checks
in order. The first failure records its reason, verifies the source-pair hash is
unchanged, increments only `j`, and redraws both coefficient tuples:

1. Require qA and qB tuples to differ.
2. Evaluate them on the already frozen final states and require all four values
   `qA(A)`, `qA(B)`, `qB(A)`, and `qB(B)` to be pairwise distinct.
3. Serialize all four targets by Section 5.7. Require one common token length
   including EOS and at most eight tokens for each target.
4. Render qA and qB exactly. Require nonempty, distinct qA/qB token sequences,
   full source/query/answer native-context compliance, and solver agreement with
   the frozen final states.
5. Require a new final canonical pair serialization and no exact full
   source/query prompt duplicate against any accepted pair in any board whose
   query phase already exists.
6. Apply every coefficient-target uniqueness and prospective affine frequency
   cap in Section 5.9 against earlier accepted pair slots.
7. Accept and freeze the first surviving coefficient pair.

The query manifest contains the frozen `SOURCE_FREEZE`, the hash of every source
pair before and after every template draw and coefficient redraw, the complete
ordered query-domain/counter ledger, each `j` and rejection reason, all final
queries/targets/tokenizations/histograms, and the final board hash. It must show
zero source-domain calls after `SOURCE_FREEZE` and byte-identical source hashes
at every redraw. Independent admission replays both manifests and fails closed
if any query/answer constraint selected, regenerated, reordered, or changed a
source. Confirmation uses the same two phases after its seed derivation.

Canonical source and final semantic serializations are the ASCII grammar in
Appendix A.

### 5.7 Rendering, normalization, and target serialization

Appendix A is the complete frozen renderer/query appendix. `S^0` always uses
`F0` or `H0`; `S^1` always uses the corresponding disjoint `F1` or `H1` family.
Renderer choice is therefore fixed by stratum and rendering index, independent
of coefficients and answers. Event order and semantics are unchanged.

All generated source, query, canonical, and target text is strict 7-bit ASCII
with LF line endings and no trailing spaces or trailing LF. Integer text is
exactly `0` for zero, otherwise optional ASCII `-` followed by base-10 digits,
with no `+`, commas, decimal point, leading zero, surrounding whitespace, or
locale formatting.

For integer `z`, the answer token sequence is exactly:

```text
tokenizer.encode(ascii_integer(z), add_special_tokens=False) || [eos_token_id]
```

Source and query IDs are separately
`tokenizer.encode(exact_ascii_string, add_special_tokens=False)`. No BOS, EOS,
or other special token is inserted in a source or query; EOS is appended only
to the answer as shown. Production right padding uses token ID zero with a false
source-mask bit. The tokenizer file, token-ID mapping, `eos_token_id`, and
tokenizer hash are frozen in the manifest. Exact-match generation includes EOS.

Normalization `N` lowercases ASCII, replaces every maximal run of bytes in
`[0x09,0x0a,0x0b,0x0c,0x0d,0x20]` with one ASCII space, then strips leading and
trailing spaces. Skeleton normalization additionally replaces every maximal
match of ASCII regex `(?<![A-Za-z0-9_])-?[0-9]+(?![A-Za-z0-9_])` with literal
`<int>`, from left to right without overlap. The admission code must use these
byte operations, not Unicode or locale APIs. Fit wording is intentionally reused
by length/query OOD; only wording/full OOD require disjoint source and query
template skeleton IDs. Exact full prompts and canonical semantic pairs remain
disjoint across all boards.

### 5.8 Exactly eight answer-only sequences

The treatment training objective and the common primary evaluation matrix
contain exactly these eight unique rows per pair:

| Row | State supplied to reader | Query | Target | Role |
|---:|---|---|---|---|
| 1 | `W_A^0` | q0 | q0(A) | Paired collision control A |
| 2 | `W_B^0` | q0 | q0(B) | Paired collision control B |
| 3 | `W_A^0` | qA | qA(A) | Clean derived |
| 4 | `W_B^0` | qB | qB(B) | Clean derived |
| 5 | `W_B^0` | qA | qA(B) | Cross-world transplant |
| 6 | `W_A^0` | qB | qB(A) | Cross-world transplant |
| 7 | `W_A^1` | qB | qB(A) | Alternate-rendering read |
| 8 | `W_B^1` | qA | qA(B) | Alternate-rendering read |

Rows 7-8 deliberately use the opposite affine query from rows 3-4. This makes
the sham training control in Section 7 unique without changing the six affine
query and answer lengths. There are no report, update, transfer, rationale,
latent-coordinate, attention, state-reconstruction, KL, probe, or
automatic-language losses.

### 5.9 Query-only and answer-frequency controls

For each board stratum, Phase Q maintains separate target histograms for q0
rows, diagonal affine rows, crossed affine rows, alternate-rendering affine
rows, and all six affine rows. For a family with final denominator `D`, no exact
ASCII target may exceed `ceil(0.05*D)` rows. Intended repeats of the same
world/query target across canonical and alternate renderings count as separate
rows.

The complete q0 histogram is computed only after the source manifest is frozen.
Because q0 is coefficient-independent, a q0 cap failure hard-fails the board;
it may not redraw a source, source attempt, schedule, q0 selector, or query
template. Affine caps are tested prospectively in pair-slot order. An affine cap
failure rejects only the current coefficient pair and advances `j` under
Section 5.6, with the source and all three query template IDs unchanged.

Across distinct semantic worlds in one stratum, the same `(coefficient tuple,
exact target)` pair may not recur in two different A/B pairs, regardless of
qA/qB label or wording template. For every coefficient tuple with at least ten
distinct semantic-world targets, its modal target share must be at most 10%.
The query manifest publishes these histograms by q0/affine family, qA/qB side,
query template, coefficient tuple, answer sign, and answer-token length, along
with the unchanged source hash for every rejected coefficient pair.

For metric `M` in stratum `s`, define the answer-frequency baseline
`B_freq(M,s)` as the accuracy of the single most frequent exact target among
the exact read units scored by `M`, with ties resolved by ASCII byte order.
Admission requires `B_freq <= 5%` for retrieval, derived, donor-follow,
paraphrase, and flexible-q1 families. The fully trained query-only arm and the
treatment readerless lesion are separate empirical controls; their bound-based
limits are in Gate 4. Passing histogram caps alone cannot substitute for those
controls.

### 5.10 Exact admission audit

Before any fit, an independent implementation and reviewer must fail closed
unless they verify:

1. Public seed bytes, counter-PRF golden vectors, unbiased bank sampling,
   every listed RNG domain, source/query phase separation, schedule
   construction, and pair order.
2. Exact train/calibration cardinalities, eight rows per pair, two worlds per
   pair, two renderings per world, OOD factors, and balance schedules.
3. Both source manifests were independently reproduced and hash-frozen before
   any train/calibration query-template or coefficient domain was invoked; the
   ordered ledgers contain no source-domain call after either `SOURCE_FREEZE`.
4. Exact source-only solver replay, atomic event semantics, exactly one legal
   edit, syntax-fixed q0 collision, unequal states, source rendering equivalence,
   source uniqueness, and unchanged source bytes across every query redraw.
5. Query templates were drawn once only after both source freezes; every
   query/answer rejection either hard-failed without mutation or redrew both
   coefficient tuples only; no source was regenerated, selected, replaced, or
   reordered. Verify four-answer distinctness and equal affine answer lengths.
6. Exact Appendix A bytes, ASCII integer/EOS targets, normalization, source and
   final semantic disjointness, prompt disjointness, context limits, and
   right-padding masks.
7. Every frequency cap, conditional query histogram, and published
   answer-frequency baseline, with no model-informed admission.
8. Query, target, donor, coefficient, and metadata fields are absent from each
   treatment writer input object; metadata deletion and poison leave writer
   inputs unchanged.
9. SHA-256 hashes for the preregistration, base, tokenizer, generator, solver,
   independent admission implementation,
   Appendix A banks, coefficient banks, train/calibration source manifests,
   train/calibration query manifests and boards, admission output, wrapper,
   target builder, initialization, and bootstrap assessor. No confirmation
   seed, source manifest, query manifest, or board exists before Section 14.

This audit is a prerequisite if implementation is separately authorized. This
document itself authorizes neither code work nor an H100 or other accelerator
allocation.

## 6. Manual target and generation contract

The wrapper must call the base for logits without passing `targets`; this avoids
the base model's built-in `zloss`. R11a computes CE manually.

Let nonempty query tokens be `Q = [q_0, ..., q_(m-1)]` and the canonical answer,
including EOS, be `A = [a_0, ..., a_(n-1)]`. The decoder input is:

```text
Q || A[:-1]
```

Only the following next-token positions receive targets:

```text
logit at Q[m-1]       predicts A[0]
logit at A[k-1]       predicts A[k], for k = 1..n-1
all other positions   use ignore_index
```

Loss is fp32 mean token cross-entropy on these next-answer positions only,
with no label smoothing and no z-loss. Each sequence CE is first averaged over
its own answer tokens. The pair loss is:

```text
L_pair = 0.25 * mean(CE rows 1-2)
       + 0.25 * mean(CE rows 3-4)
       + 0.25 * mean(CE rows 5-6)
       + 0.25 * mean(CE rows 7-8)
```

Thus every one of the eight sequences has weight `1/8`.

Primary accuracy is free-running exact generation, not teacher-forced token
accuracy. Decoding is greedy, temperature zero, with no sampling, beam,
verifier, repair, parser, or constrained decoder. Generation calls
`begin_read(W,Q,...)` with a fresh empty KV cache and greedily chooses the first
answer token from its returned `next_logits`. Every later token is produced only
by `decode_step(decode_session, generated_token_ids)`, with no W or source
argument and all readers disabled. Generation stops at EOS or the frozen eight-
token cap and must exactly match the target token sequence including EOS.
Missing EOS, extra text, a rationale, or alternate numeric formatting is wrong.

Mandatory target tests are:

- perturb one current target while holding decoder inputs fixed; logits remain
  bitwise identical and only that CE term changes;
- perturb any future answer token; logits and CE terms up through the earlier
  prediction remain bitwise identical under causal masking;
- append gold-answer tokens to a masked writer poison suffix; `W` remains
  bitwise identical;
- compare teacher-forced argmax and stepwise cached logits at every answer
  position under identical prefixes;
- inspect every returned `DecodeSession` against the Section 3 field whitelist,
  delete or poison all auditor-owned W/source references after `begin_read`, and
  require unchanged finite generated logits and tokens;
- run exact free generation and verify that reader hooks never fire in
  `decode_step` or on a teacher-forced answer-prefix position.

## 7. Frozen training arms and common primary evaluation

Five arms are fit once. They share the immutable base, 1,607,334 adapter
parameters, initialization bytes, pair order, decoder query/target token and
padding shapes, optimizer, update count, precision, and eight sequence losses.
Every difference defined in this section is a training-arm intervention only;
it does not define an arm-specific primary evaluation row, target, query/world
pairing, or substitute matrix. Every frozen trained arm's primary evaluation
uses all eight rows of the common crossed treatment matrix in Section 5.8,
including the donor-world rows 5-6. No sham training substitution is retained as
a primary evaluation substitution.

| Arm | Training-only intervention |
|---|---|
| Treatment | Source-only `W`; exact rows 1-8 in Section 5.8 |
| Sham-interchange | Training rows 5-6 are unique same-world alternate-rendering reads: `Eval(qA,W_A^1)->qA(A)` and `Eval(qB,W_B^1)->qB(B)`; all other training rows match treatment |
| Private-query | Query tokens are unmasked writer K/V and a fresh query-conditioned state is written for each row |
| Reset | For each source, all four tied writer cells start from the same `W^0`; their four outputs are averaged, so no round consumes a previous round |
| Query-only | All writes execute and are discarded; every example reads the same learned six-slot `W^0`, with no source-dependent state |

The sham's rows 5-6 are not duplicates: its `W_A^1,qA` and `W_B^1,qB` reads
differ from rows 7-8, which use `W_A^1,qB` and `W_B^1,qA`. Treatment and sham
have identical source-call token/pad lengths, decoder query multisets, sequence
counts, query lengths, and target-token lengths. Their semantic target support
and writer-to-loss reuse still differ, so the sham is a diagnostic comparator
only.

At common primary evaluation, each arm retains its frozen trained state-
construction mechanism but receives the same row's source rendering, query, and
target. Treatment and sham construct the named source-only recurrent W; reset
constructs its named source-only averaged reset W; query-only executes and
discards the named write and supplies its one global learned `W^0`. Thus rows
5-6 are true donor-world crossed rows for all four arms.

Private-query is defined separately and exhaustively. For each common primary
row, take the source rendering named by the state column of Section 5.8, append
that row's query token sequence with no separator or special token, and call the
same three-argument `write` API with the concatenated IDs, its right-prefix mask,
and summed length. All true source and query positions are unmasked writer K/V.
The resulting fresh query-conditioned W is supplied to `begin_read` with that
same row query. Therefore its eight primary writes are, in row order:

```text
S_A^0||q0, S_B^0||q0, S_A^0||qA, S_B^0||qB,
S_B^0||qA, S_A^0||qB, S_A^1||qB, S_B^1||qA
```

The common targets remain exactly q0(A), q0(B), qA(A), qB(B), qA(B), qB(A),
qB(A), and qA(B). Private-query is deliberately non-reusable and does not meet
the treatment source-only boundary. In its strict-reuse diagnostic, the primary
row-1 q0-conditioned state is reused for qA and the primary row-2 q0-conditioned
state is reused for qB; neither state may be rebuilt. The query-only arm is a
trained leakage/frequency control, not a parameter-matched claim that source
computation is unnecessary.

No control can support causal-training attribution. Forward call counts and
shapes are matched where stated, but backward graphs and objective support
cannot be perfectly matched. Treatment-control deltas are reported with whole-
pair intervals as diagnostics only; they are not pass gates and cannot justify
"causal training," "caused by interchange training," or an equivalent claim.

A one-round diagnostic evaluates the trained treatment with captured `W^1`.
It is not a separately trained arm and cannot replace the reset diagnostic.

## 8. Exact per-pair call and objective ledger

Logical calls may be batched, but none may be removed, fused across semantic
rows, or conditionally skipped. A "reader-site application" is one configured
site executing on one read sequence, regardless of query token count. Reader
applications on answer-prefix/generated positions are separately fixed at zero.

### 8.1 Training ledger

Let `n0` be the q0 target length including EOS and `n1` the common length of
all four affine targets including EOS for that pair. For every pair in every
optimizer exposure:

| Operation/objective | Treatment | Sham | Private-query | Reset | Query-only |
|---|---:|---:|---:|---:|---:|
| Source block-1:18 calls | 8 | 8 | 8 | 8 | 8 |
| Result-bearing writes | 4 | 4 | 8 | 4 | 0 |
| Discard-only writes | 4 | 4 | 0 | 4 | 8 |
| Writer-round cells executed | 32 | 32 | 32 | 32 | 32 |
| Backward-connected writer-round cells | 16 | 16 | 32 | 16 | 0 |
| Fresh-empty-cache `teacher_read` calls | 8 | 8 | 8 | 8 | 8 |
| Query-prefill reader-site applications | 24 | 24 | 24 | 24 | 24 |
| Answer-prefix reader applications | 0 | 0 | 0 | 0 | 0 |
| Scalar sequence CE means | 8 | 8 | 8 | 8 | 8 |
| Answer-token CE terms | `2*n0+6*n1` | `2*n0+6*n1` | `2*n0+6*n1` | `2*n0+6*n1` | `2*n0+6*n1` |
| CE sequences connected to source encoding or writer-round output | 8 | 8 | 8 | 8 | 0 |
| CE sequences connected to six-slot seed parameter | 8 | 8 | 8 | 8 | 8 |
| CE attachments per result-bearing write | `[3,3,1,1]` | `[2,2,2,2]` | `[1,1,1,1,1,1,1,1]` | `[3,3,1,1]` | `[]` |
| CE sequences connected to query-side path | 8 | 8 | 8 | 8 | 8 |
| Pair-loss scalars | 1 | 1 | 1 | 1 | 1 |

Treatment, sham, and reset result-bearing writes are exactly `S_A^0`, `S_B^0`,
`S_A^1`, and `S_B^1`; four additional source-only writes are discarded.
Private-query performs one result-bearing write for each row. Query-only
performs eight source/write calls but discards every result and uses the one
shared learned `W^0` parameter value for all pairs and rows. Its eight CEs may
train that global seed and the query/reader path, but have zero path to a source
encoding or writer-round output. All treatment result-bearing `W` values are
complete before any query tokens are disclosed to `teacher_read`. Dummy outputs
cannot replace or alter them.

The exact call order for treatment, sham, and reset is the four result-bearing
sources in the order above, followed by one source-only dummy rerun of each in
the same order. Query-only executes that same eight-source order with every
output discarded. All eight calls in these four arms finish before any query
tokens are supplied to `teacher_read`. Private-query instead executes rows 1
through 8 in Section 5.8 order, combining that row's named source rendering with
that row's query in its writer; this timing and writer-token-length difference
is part of the training diagnostic and is not compute matched.

The attachment vectors are in that same four-source order, or row order for
private-query. They count scalar sequence CE means, not answer-token terms.

Every arm uses the Section 6 loss, so CE is the only objective and every
sequence mean has weight `1/8`. Matching the 32 forward round-cell executions
does not establish equal FLOPs, memory, kernels, gradient communication, or
optimization pressure. In particular, the numbers of backward-connected
writer graphs are 4, 4, 8, 4, and 0. Any timing or memory comparison is
descriptive and must disclose these graph differences.

### 8.2 Primary evaluation ledger by trained arm

Each arm executes this complete ledger per pair. There is no backward objective
during evaluation. For all five arms, the eight primary read rows, their order,
source rendering, query, target, and clean/cross role are exactly Section 5.8.
In particular, rows 5-6 are `qA` on donor world B and `qB` on donor world A for
every arm; the sham's same-world training substitutions are forbidden here.
Every primary sequence starts with `begin_read` and continues only through
`decode_step`.

| Operation | Treatment | Sham | Private-query | Reset | Query-only |
|---|---:|---:|---:|---:|---:|
| Source block-1:18 calls | 8 | 8 | 8 | 8 | 8 |
| Result-bearing writes | 4 | 4 | 8 | 4 | 0 |
| Discard-only writes | 4 | 4 | 0 | 4 | 8 |
| Writer-round cells | 32 | 32 | 32 | 32 | 32 |
| Primary all-site `begin_read` calls | 8 | 8 | 8 | 8 | 8 |
| Primary reader-site applications | 24 | 24 | 24 | 24 | 24 |
| q0 reads included in the eight | 2 | 2 | 2 | 2 | 2 |
| Extra q0 reads | 0 | 0 | 0 | 0 | 0 |

Treatment, sham, reset, and query-only construct their common-matrix states as
specified in Section 7. Private-query makes the exact eight row-conditioned
writes listed there: for a crossed row, the donor source and receiver-named
query are concatenated in its writer, and the same query is then prefilled by
`begin_read`. It may not use the diagonal source, omit the writer-side query, or
reuse a state from another primary row.

If row `r` emits `g_r` tokens before EOS or the eight-token cap, where
`1 <= g_r <= 8`, it executes exactly `g_r-1` `decode_step` calls. Thus each
arm's primary pair executes exactly `sum_r(g_r-1)` decode steps over its eight
rows, and every one has zero reader-site applications. No unused step is run
after EOS or after token eight.

The treatment primary all-site rows are counted exactly once. Its four
canonical writes cache `W^1` and `W^4`; every lesion below reuses those exact
objects and adds no write:

| Treatment condition | Additional writes | Additional rounds | Reads | Reader-site applications |
|---|---:|---:|---:|---:|
| Primary sites 12+18+24 | included above | included above | 8 | 24 |
| Block-24 lesioned, sites 12+18 | 0 | 0 | 8 | 16 |
| Block-24 only | 0 | 0 | 8 | 8 |
| Readerless, all sites disabled | 0 | 0 | 8 | 0 |
| One-round `W^1`, sites 12+18+24 | 0 | 0 | 8 | 24 |

Thus the four treatment lesions add exactly 32 reads after the eight-read
primary; there is no second all-sites pass hidden in the lesion ledger. The
block-24 lesion is mandatory. There is no source-visible evaluation and no
source-plus-W read.

The private-query strict-reuse diagnostic reuses the q0-conditioned `W_A^0`
and `W_B^0` already produced for primary rows 1-2 and adds exactly two
`begin_read` sequences, zero writes, zero rounds, and six reader-site
applications: qA on the former and qB on the latter. Treatment strict-two-query
reuse is computed from primary rows 1-4 and adds no calls. No other trained-arm
diagnostic adds a read.

### 8.3 Boundary-invariance audit ledger

In deterministic fp32 audit mode, source-cache clear, swap, and poison each
rerun the same eight primary treatment reads using the same four cached `W^4`
objects: zero additional writes, zero rounds, eight reads, and 24 reader-site
applications per condition. q0 remains rows 1-2, not extra reads.

The interleaving test uses a fixed two-pair fixture. Both the baseline schedule
and interleaved schedule execute exactly eight result-bearing writes, 32 writer
rounds, 16 generation sequences, and 48 reader-site applications. Baseline is
`write(P),Eval(P),write(R),Eval(R)`; interleaved is
`write(P),write(R),Eval(P),Eval(R)`. Restored outputs must be bitwise equal.

## 9. Frozen optimizer and fit

- Optimizer: AdamW only.
- Betas: `(0.9, 0.95)`.
- Epsilon: `1e-8`.
- Weight decay: `0.01` on LoRA and projection matrices; zero on norm weights,
  slot seeds, tap IDs, and scalar gates.
- Peak learning rate: `3e-4`.
- Updates 1-128: linear warmup from zero to `3e-4`.
- Updates 129-1638: constant `3e-4`.
- Updates 1639-2048: linear decay to `3e-5`.
- Effective batch: eight complete A/B pairs, or 64 answer sequences.
- Fit length: 2,048 optimizer updates, exactly eight deterministic epochs over
  the 2,048 training pairs.
- Epoch shuffle: Section 5.2 Fisher-Yates over pair indices with the train seed
  and exact domains `fit/epoch=0/shuffle` through `fit/epoch=7/shuffle`; each
  counter begins at zero independently.
- Precision: bf16 forward/backward, fp32 attention softmax, loss, gradient norm,
  AdamW moments, and master adapter weights.
- Global adapter gradient clip: 1.0.
- Base parameters: `requires_grad=False` and absent from the optimizer.

Gradient accumulation may reduce microbatch size only before any capability
score and must preserve effective batch, pair integrity, pair order, calls,
updates, and schedule. No second fit, early stopping, optimizer substitution,
learning-rate sweep, seed sweep, loss reweighting, or checkpoint selection is
allowed.

## 10. No-oracle inference contract

Allowed model inputs are:

- source token IDs, right-padding mask, and explicit source lengths in `write`;
- `W`, query IDs/mask/lengths, answer-prefix IDs/mask/lengths, and one batch-
  global reader-site mask in `teacher_read`;
- `W`, query IDs/mask/lengths, and one batch-global reader-site mask in
  `begin_read`;
- only the whitelisted `DecodeSession` and generated token IDs in
  reader-disabled `decode_step`;
- the model-written `W`, as the only per-example object crossing from `write`
  to a fresh-empty-cache `teacher_read` or `begin_read`;
- fixed architecture constants and learned parameters.

Forbidden model inputs or inference operations are:

- semantic states, x/y values, parsed numbers, coefficients outside their raw
  query tokens, operation labels, edit labels, register-role labels, renderer
  IDs, pair IDs, donor flags, or answer metadata;
- solver outputs, affine calculations, equations, ASTs, programs, candidate
  sets, confidence certificates, or retrieval decisions;
- a host parser, host executor, query router, answer repair, or state-coordinate
  inspection;
- source KV, source residuals, source positions, source masks, source IDs,
  source lengths, a source-side closure, or any source cache in `teacher_read`,
  `begin_read`, `DecodeSession`, or `decode_step`;
- W, a W alias, any reader state, or a reader-site mask in `DecodeSession` or
  `decode_step`;
- teacher rationale, chain of thought, pause token, latent output token, report
  text, or gold intermediate state.

Structured records may be used offline only to generate, admit, stratify, and
score examples. The treatment writer collator must construct a fresh object
containing only source IDs, source mask, and lengths. Metadata deletion and
metadata poison must leave W and logits bitwise unchanged. Private-query and
query-only are explicit diagnostic departures from the treatment input contract
and cannot support the primary claim.

## 11. Mechanics and compute canaries

### 11.1 CPU and tiny-model contracts

All tests below must pass before a GPU canary:

1. Exact parameter counts and initialization hash.
2. Workspace-disabled base logits and cache identity.
3. Source-only mask, right-padding, and all-masked-suffix tests.
4. Target-shift, current/future perturbation, and closed-API free-generation
   tests, including `DecodeSession` field inspection and W/source deletion after
   `begin_read`.
5. API reflection and ambient-state audits showing only the four Section 3
   signatures, no hidden overload, and no W/source/reader context retained after
   `teacher_read` or `begin_read`; query-prefill-only reader hook counts and zero
   `decode_step` reader calls.
6. Exact per-pair call-ledger counters for all arms.
7. Counter-PRF, source-before-query domain ledger, source-freeze rehash,
   coefficient-only redraw, pair construction, solver replay, rejection-order,
   rendering, normalization, target, admission, metric, and whole-pair
   bootstrap golden tests.
8. Whole-state swap, paired q0 collision, alternate-rendering swap, block-24
   lesion, readerless, and one-round intervention tests on a deterministic tiny
   model.
9. Fresh-cache entry assertions and source-cache clear, swap, poison, and
   interleaving invariance tests using the exact Section 8.3 ledger.

The treatment must also pass these exact query-isolation tests in deterministic
fp32 audit mode. After restoring row order, `torch.equal` must hold for W under:

- q0, qA, qB, and an empty query;
- three fixed manifest-listed query-token fixtures and gold-answer poison
  suffixes behind the writer mask;
- every admitted right-pad width and at least three alternate pad token IDs;
- single-row versus mixed-length batching;
- arbitrary batch permutation and inverse permutation.

Every `teacher_read` and `begin_read` entry hook must observe
`past_key_values is None` or an exactly empty cache with zero sequence length
and position index zero. Every `decode_step` entry hook must observe only a
Section 3 whitelisted session and must assert that readers are disabled.
Source-side caches are not returned by the production `write` API. Audit
instrumentation may retain source scratch only outside the model API for these
tests:

- **clear:** delete all source IDs, masks, residuals, and KV scratch after W is
  produced;
- **swap:** exchange the auditor-owned source scratch registries for the two
  fixture pairs without exchanging W;
- **poison:** overwrite retained floating source scratch with quiet NaNs and
  integer scratch with the repeated byte pattern `0xA5`;
- **interleaving:** execute the two schedules in Section 8.3.

For clear, swap, and poison, all eight generated token sequences and all
teacher-forced logits must be bitwise equal to baseline; no poisoned value may
be read and all outputs must remain finite. Interleaving outputs must be
bitwise equal after restoring pair order. Any source-scratch lookup during
`teacher_read`, `begin_read`, or `decode_step` is an immediate failure even if
outputs happen to match.

`autograd.grad(W, query_input_embeddings, allow_unused=True)` must be absent or
exactly zero. A sentinel `requires_grad` audit at H6, H12, and H18 must show a
valid writer path, and nonzero finite CE gradients must reach the undetached H12
and H18 taps after reader gates open. `W.grad_fn` must be present in training.
Base parameter gradients must remain absent.

### 11.2 Sixteen-update H100 canary

If separately authorized outside this document, only after CPU/tiny success may
one isolated H100 run a 16-update bf16 treatment canary. This section is a
requirements contract, not allocation or execution authority. The canary is a
mechanics test, not a capability score. It must show:

- real CUDA and bf16 allocation on the intended device;
- 16 finite losses, logits, workspaces, and gradient norms;
- no OOM, allocator retry, CPU offload, or activation offload;
- only the 1,607,334 adapter parameters change;
- unchanged base hash and correct adapter save/reload hash;
- exact reader-position and call-ledger counters;
- nonzero writer-path gradients after reader gates open.

Timing uses one device, frozen canary shapes, 20 warmups, 100 measured
iterations, CUDA events, and synchronization. Peak memory is measured after a
reset on the same shapes. The canary fails if any cap is exceeded:

1. Clean workspace prefill median wall time or peak allocated VRAM is greater
   than `1.15x` the direct base on the frozen source/query length mix.
2. Cached decode time per generated token is greater than `1.10x` base.
3. A step with one clean and one crossed-transplant read is greater than
   `2.25x` the clean-workspace step.
4. The full eight-row pair ledger does not fit without changing the frozen
   effective batch or using offload.

These are treatment feasibility caps, not evidence of compute equivalence
between trained arms. Control-arm backward graphs differ as disclosed in
Section 8.1 and need not have equal time or memory. No cap may be relaxed in
response to a promising loss.

## 12. Metrics and whole-pair bootstrap

All primary metrics use free-running exact generation over every admitted pair.
There is no clean-correct filtering. qA and qB directions are both retained.

- `retrieval`: exact q0 row accuracy over rows 1-2.
- `derived`: exact diagonal accuracy for `Eval(qA,W_A)` and `Eval(qB,W_B)`.
- `donor_follow`: exact crossed accuracy for `Eval(qA,W_B)` and
  `Eval(qB,W_A)` against donor-state targets.
- `receiver_follow`: crossed outputs equal the corresponding diagonal receiver
  targets `qA(A)` or `qB(B)`.
- `swap_specificity`: `donor_follow - receiver_follow`, in percentage points,
  using qA/qB only.
- `q0_collision`: the pair-level indicator that both existing q0 rows 1-2 are
  exact for their shared target. It introduces no additional read.
- `paraphrase`: exact row accuracy for alternate-rendering rows 7-8,
  `Eval(qB,W_A^1)` and `Eval(qA,W_B^1)`.
- `strict_two_query_reuse`: for each world, both q0 and its own affine query are
  correct using the exact same cached W object without recomputation. The
  metric averages the two world-level indicators per pair.
- `q1_side_donor`: donor-follow computed separately for the qA and qB crossed
  directions, pooled only after each side is reported.
- `flexible_q1`: exact accuracy over the two diagonal and two crossed affine
  reads.
- `readerless_M`: treatment accuracy for target-matching metric `M` with every
  reader disabled.
- `query_only_M`: query-only-arm accuracy for the same target-matching metric.
- `B_freq(M,s)`: the frozen answer-frequency baseline from Section 5.9.

The whole A/B pair is the only bootstrap resampling unit. Both worlds, both
renderings, q0, qA, qB, both cross directions, and every lesion stay together.
For each board stratum, use exactly 10,000 percentile-bootstrap replicates,
sampling its pairs with replacement. The assessor uses the Section 5.2 counter
PRF with seed:

```text
de5e446280f10d6d69cf6a2034c95a9c75da50401eaaa5941179be33a214a54d
```

and domains
`bootstrap/board=<calibration|confirmation>/stratum=<wording|length|query|full>/replicate=<0..9999>/draw`.
For pooled metrics, resample independently within each stratum and preserve
stratum sizes. All paired conditions on the same board/stratum use the identical
resampled pair indices. Row-, token-, world-, and direction-level bootstraps are
forbidden.

Sort the 10,000 replicate values ascending. With one-indexed nearest-rank
`Q(p)=value[ceil(p*10000)]`, define `LB95=Q(0.025)` and `UB95=Q(0.975)`; point
estimates use the full admitted denominator. Every positive accuracy or
retention threshold in this document is applied to `LB95`. Every negative-
pathway ceiling for receiver-follow, readerless, or query-only accuracy is
applied to `UB95`. The q0 collision negative-control has a preregistered
high-accuracy expectation, so its positive minimum uses `LB95`. Difference and
ratio thresholds use the corresponding paired-bootstrap `LB95`. Both bounds
and point estimates are always reported. No alternate interval, filtering rule,
or denominator may replace a failed bound.

These intervals quantify pair sampling for this one fixed checkpoint and its
fixed evaluation boards only. With one initialization and one fit per arm they
do not estimate training-seed variance, retraining success probability, or the
reliability of the training procedure. Inference and all allowed wording are
limited to this exact frozen trained artifact.

## 13. Calibration gates

All gates are conjunctive. "Single OOD" means each of wording, length, and
query OOD independently; none may subsidize another. "Full" means full OOD.
All table entries labeled `LB95` or `UB95` use Section 12 whole-pair bounds, not
point estimates.

### Gate 0: admission

Every item in Section 5.10 passes, every pre-fit hash is recorded, and both
train/calibration source manifests demonstrably precede both query manifests.
The public Section 5.2 confirmation domain and salt and the Section 14 decision-
record schema are frozen, but no confirmation seed, manifest, board, or score
exists. Failure blocks training.

The preceding no-confirmation-exists clause applies to the pre-fit Gate 0. On
the Section 14 confirmation rerun, it is replaced by exact verification of the
single seed derivation, source-before-query manifests, immutable-source redraw
ledger, and final confirmation hashes.

### Gate 1: mechanics, isolation, and compute

Every CPU/tiny and 16-update canary item in Section 11 passes. In addition:

- active adapter count is exactly 1,607,334 and resident count is 126,688,998;
- all reader hooks are confined to query prefill;
- every `teacher_read` and `begin_read` begins with an empty cache, only W
  crosses from `write`, and every `decode_step` receives neither W nor source;
- every returned `DecodeSession` matches its whitelist and every reader is
  disabled during `decode_step`;
- no hidden read overload or ambient W/source/reader context exists after a
  read call returns;
- all W, source-cache clear/swap/poison/interleaving, padding, metadata, and
  batch-order invariance tests are bitwise exact;
- query-input gradient to W is zero and source writer taps are not detached;
- the manual CE target contract passes and built-in zloss is never invoked;
- all values and gradients are finite.

Failure blocks the full fit or invalidates it if discovered later.

### Gate 2: clean source-free capability

| Metric | Every single OOD `LB95` | Full OOD `LB95` |
|---|---:|---:|
| Retrieval q0 | >=85% | >=80% |
| Clean derived qA/qB | >=75% | >=65% |
| Strict two-query reuse | >=65% | >=55% |

### Gate 3: crossed whole-W mediation and collision controls

| Metric | Every single OOD bound | Full OOD bound |
|---|---:|---:|
| Donor-follow qA/qB `LB95` | >=80% | >=70% |
| Receiver-follow qA/qB `UB95` | <=15% | <=15% |
| Swap specificity `LB95` | >=55 points | >=55 points |

Additionally, on every OOD stratum:

- q0 paired-collision `LB95` is at least 90%;
- same-world alternate-rendering `LB95` is at least 90%;
- q0 is excluded from specificity and donor metrics;
- qA-side donor-follow `LB95` and qB-side donor-follow `LB95` are each at least
  75%, with both sides reported separately.

### Gate 4: late-reader exclusion and W necessity

On full OOD, sites 12+18 with block 24 lesioned must:

1. have paired-bootstrap `LB95 >= 0.80` for
   `donor_12_18 / donor_all`, defining a replicate's ratio as zero if its
   all-site denominator is zero;
2. independently pass every full-OOD threshold in Gates 2 and 3.

Block-24-only results are diagnostic and can never satisfy or substitute for
this gate. This is a late-reader exclusion only. Readers at blocks 12 and 18
may compute or encode query-conditioned answer state after attending to W; v3
does not localize the affine computation inside W and does not claim to exclude
an early query-conditioned answer representation. The allowed conclusion is
only that W mediates flexible late-bound reads without requiring the block-24
reader.

For each OOD stratum `s` and each target-matching metric `M` in `{retrieval,
derived, donor_follow, paraphrase, flexible_q1}`, both controls must satisfy:

```text
UB95(readerless_M) <= min(10%, B_freq(M,s) + 2 percentage points)
UB95(query_only_M) <= min(10%, B_freq(M,s) + 2 percentage points)
```

A mere drop from treatment is insufficient. Failure of either absolute ceiling
fails Gate 4.

### Diagnostic comparators, not gates

Treatment-sham, treatment-private-query, treatment-reset, and treatment-one-
round differences are reported with paired whole-pair point estimates and 95%
bounds on every OOD stratum, using the common Section 5.8 primary evaluation
matrix for every trained arm. No minimum delta is required and none supports
causal-training attribution, recurrence necessity, compute equivalence, or a
claim about which objective caused the treatment result. Four rounds remain a
fixed architecture setting.

## 14. Deterministic post-calibration confirmation gate

This gate is an auditable deterministic extension of a frozen calibration
decision. It is not described as secret, random, blinded, independently
custodied, unpredictable, or untouched. The only allowed sequence is:

1. Finish all five one-time fits and all calibration scoring, including the
   common crossed primary matrix and every preregistered diagnostic.
2. Emit the complete calibration report as strict-ASCII JSONL with every object
   key sorted lexicographically, no optional whitespace, fixed metric/stratum/arm
   array order, lowercase 64-hex hashes, and one final LF. Every numeric point,
   bound, difference, ratio, count, and timing is serialized exactly as a
   reduced rational object `{"den":<positive_integer>,"num":<integer>}`;
   zero is `{"den":1,"num":0}`. Floating, decimal, exponent, NaN, infinity,
   and negative-zero spellings are forbidden. Timestamps, nonces, hostnames,
   paths, comments, free text, and unused fields are forbidden. Freeze the
   report SHA-256 and every input artifact hash.
3. Emit exactly this ordered strict-ASCII decision record, substituting
   lowercase 64-hex values and literal `PASS` or `FAIL`, with one LF after every
   line including the last:

```text
R11A-V3-CALIBRATION-DECISION
prereg_sha256=<64hex>
base_sha256=<64hex>
tokenizer_sha256=<64hex>
generator_sha256=<64hex>
solver_sha256=<64hex>
admission_sha256=<64hex>
wrapper_sha256=<64hex>
target_builder_sha256=<64hex>
assessor_sha256=<64hex>
train_source_manifest_sha256=<64hex>
train_query_manifest_sha256=<64hex>
train_board_sha256=<64hex>
calibration_source_manifest_sha256=<64hex>
calibration_query_manifest_sha256=<64hex>
calibration_board_sha256=<64hex>
initialization_sha256=<64hex>
treatment_checkpoint_sha256=<64hex>
sham_checkpoint_sha256=<64hex>
private_query_checkpoint_sha256=<64hex>
reset_checkpoint_sha256=<64hex>
query_only_checkpoint_sha256=<64hex>
calibration_report_sha256=<64hex>
gate0=<PASS|FAIL>
gate1=<PASS|FAIL>
gate2=<PASS|FAIL>
gate3=<PASS|FAIL>
gate4=<PASS|FAIL>
decision=<PASS|FAIL>
```

   `decision` must be `PASS` if and only if all five gate fields are `PASS`.
   Define `H_decision = SHA256(exact decision-record bytes)` and freeze both the
   record and hash. If the decision is `FAIL`, stop; no confirmation seed or
   board is created.
4. For a passing decision only, derive `S_confirm` exactly once by Section 5.2
   and freeze the derivation record before any confirmation domain is invoked.
   Independent recomputation must agree. From this derivation onward, no refit,
   checkpoint replacement, checkpoint selection, code change, report change,
   manifest change, threshold change, decision-record change, domain/salt
   change, or new calibration score is permitted. Any required change kills v3;
   it cannot produce a revised `H_decision` or second seed within v3.
5. In one deterministic generation invocation, run confirmation Phase S for all
   four strata, independently source-admit it, and freeze its `SOURCE_FREEZE`.
   Only then run Phase Q, where query/answer failures may redraw both coefficient
   tuples only and every source hash remains fixed. Independently admit and
   freeze the query manifest, answers, and final board. A failure kills v3;
   reseeding, source regeneration, board replacement, repair, or a second
   generation invocation is forbidden.
6. Score the already frozen treatment artifact and controls exactly once using
   the common Section 5.8 matrix. Seed and board bytes may be audited and are not
   claimed to be hidden; the no-change rule in step 4, not secrecy, is the only
   post-calibration protection. Partial or exploratory scoring is forbidden.

Confirmation must pass Gates 0-4 with identical thresholds, bounds,
definitions, decoding, lesions, common trained-arm evaluation matrix, and
frozen checkpoints. Gate 0 is rerun against the derivation and both confirmation
manifests. No threshold, renderer, bank, target rule, reader site, state, round,
loss, optimizer, decoding rule, or intervention may change. A confirmation
failure closes v3; a second seed, board, fit, or score is forbidden.

Passing this gate supports evidence only on this one deterministic
post-calibration board. It does not establish secrecy, blinded holdout validity,
training-seed robustness, retraining reliability, or confirmation over a random
board distribution.

## 15. Kill gates and prohibited rescues

R11a is killed as formulated by any of the following:

1. A query, answer, coefficient field, donor field, or semantic metadata can
   affect treatment W.
2. `teacher_read` or `begin_read` begins with source KV/state, any per-example
   object other than W crosses from `write`, `DecodeSession` contains a forbidden
   field, a hidden overload or ambient W/source/reader context exists,
   `decode_step` accepts W/source/reader state, or any source-cache clear/swap/
   poison/interleaving test changes an output.
3. A source tap or W is detached, or query embeddings have a gradient path to W.
4. A reader acts on any teacher-forced answer-prefix position or during any
   `decode_step`.
5. Block 24 is required to salvage a failed 12+18 full-OOD gate.
6. Built-in zloss, any non-CE objective, target leakage, or incorrect causal
   shifting is used.
7. Parameter counts, initialization, LoRA scaling, attention scaling, call
   ledger, optimizer, or compute caps differ from this document.
8. Clean accuracy passes while crossed donor, receiver, specificity, q0
   collision, alternate-rendering, reuse, readerless, or query-only bounds fail.
9. A used-board miss motivates changing data, pair selection, queries, layers,
   slots, width, rounds, losses, schedule, masks, decoding, or thresholds.
10. Any query template, coefficient, rendered query, or serialized target is
    sampled or computed before the required source freezes; a query/answer
    constraint regenerates, selects, replaces, reorders, or changes a source or
    template; a source domain is called after `SOURCE_FREEZE`; or a manifest
    fails to prove byte-identical sources through every coefficient redraw.
11. The confirmation seed is derived before a passing `H_decision`, from any
    alternate record/domain/salt, or more than once; any fit, artifact, code,
    report, manifest, gate, or decision input changes after derivation; or the
    confirmation board is generated or scored more than once.
12. Any R11b report, update, transfer, broadcast, or public-benchmark work is
    started before a complete R11a pass.

The following are not rescues: clean-correct filtering, q0 in specificity,
shared-query swaps, teacher-forced accuracy, probe decodability, cosine
similarity, nonzero gradients, a block-24-only effect, lower thresholds, more
data, more rounds, more slots, base unfreezing, parser use, an alternate decision
hash, a different salt, or a second seed.

A mechanics bug found before any capability score may be corrected only under
a new version identifier, with regenerated boards, repeated independent
admission, and all hashes frozen again. After any capability score, such a
change is a new experiment and cannot inherit R11a results.

## 16. Execution order

This document authorizes no implementation, file change, data generation, job,
CPU run, or GPU allocation. If those actions receive separate authority later,
the implementation must execute in this order:

1. Implement only isolated wrapper, generator, assessor, and unit-test paths;
   freeze all required generator, solver, admission, wrapper, target-builder,
   assessor, preregistration, base, tokenizer, and Appendix hashes before
   generation.
2. Pass CPU and tiny-model contracts, including target, query isolation, closed
   API, fresh cache, decode-session whitelist, source-cache invariance, late
   reader, intervention, RNG-domain ledger, bootstrap, and exact call-ledger
   tests.
3. Run Phase S only for every train source pair and every calibration source
   pair. Independently source-admit both boards and freeze both source manifests
   and `SOURCE_FREEZE` hashes. Do not invoke any query-template or coefficient
   domain and do not compute any rendered query or serialized target.
4. Only after both source freezes, run Phase Q for train and calibration. Freeze
   template IDs once, permit only coefficient-pair redraws, independently admit
   the query manifests, and freeze final board/answer hashes.
5. Recheck base bypass identity, parameter counts, initialization, right-padding
   masks and lengths, no-oracle collators, source immutability ledgers, and all
   pre-fit hashes.
6. If separately authorized, run the isolated 16-update H100 mechanics and
   compute/VRAM canary. Stop on any failed prerequisite or canary conjunct.
7. Train treatment, sham-interchange, private-query, reset, and query-only
   exactly once in separate output directories.
8. Freeze checkpoint hashes, then score calibration and all preregistered
   diagnostics with one assessor. Every trained arm uses the common crossed
   Section 5.8 primary evaluation matrix.
9. Emit and freeze the canonical calibration report, decision record, and
   `H_decision`. Stop if any gate fails.
10. On a pass only, derive and record `S_confirm` once. From this point, refit or
    any change to an artifact, code hash, report, manifest, gate input, or
    decision record is forbidden.
11. Generate confirmation Phase S once, source-admit and hash-freeze it, then
    run Phase Q with coefficient-only redraws. Independently admit and freeze
    the final board, score it once, and stop if any gate fails.
12. Only a full R11a calibration and deterministic post-calibration confirmation
    pass may permit a separately preregistered R11b.

The protected pretraining job, optimizer, data stream, output directories,
checkpoint names, and job scripts must not be inspected, canceled, modified,
resumed, or reused by this execution chain.

## 17. R11b firewall

R11b report, update, transfer, held-out-consumer, multi-family, broadcast, and
global-workspace experiments are forbidden unless R11a passes every calibration
and deterministic post-calibration confirmation gate. Passing a mechanics
canary, clean gate, donor gate, or calibration alone is insufficient.

Even after an R11a pass, R11b requires a new preregistration, new data admission,
new hashes, and its own confirmation design with explicitly limited claims. No
R11a metric may be silently relabelled as reportability, updateability, transfer,
or broadcast.

## 18. Strongest allowed wording

If and only if every calibration and deterministic post-calibration confirmation
gate passes, the strongest allowed statement is:

> For this single frozen trained artifact on preregistered two-register
> synthetic worlds, an isolated 1,607,334-parameter adapter around the immutable
> 125,081,664-parameter raw-200k Shohin GPT wrote a 576-scalar source-only W that
> was the sole per-example object entering fresh-cache reads; W mediated
> crossed, flexible late-bound affine answers and whole-W donor interventions
> under the frozen tests, including exclusion of the block-24 reader.

This wording does not place the affine calculation inside W. Blocks 12 and 18
may encode query-conditioned answer state after reading W. It also makes no
claim about causal effects of the training objective, recurrence necessity,
compute equivalence, retraining reliability, or any artifact other than the
fixed fitted checkpoint tested here.

The confirmation seed was deterministically derived from the frozen calibration
decision hash and public salt. Therefore this wording makes no claim of a
secret, random, blinded, independently custodied, unpredictable, or untouched
confirmation set.

No stronger workspace, broadcast, report, update, transfer, general-reasoning,
or public-benchmark claim is authorized.

Until every gate passes, the only allowed statement is:

> R11a is a preregistered minimal causal-mediator canary. It has not established
> a capability result.

## Appendix A. Frozen ASCII grammar and templates

This appendix is normative. Quoted strings below denote their exact ASCII
contents after substituting Section 5.7 integers. `\n` denotes one LF byte.
Source assembly is `header + "\n" + event_1 + ... + "\n" + event_depth`, with
one event per line and no final LF.

### A.1 Canonical semantic serialization

Canonical event strings are:

```text
ADD_X(<k>)       ADD_Y(<k>)       SUB_X(<k>)       SUB_Y(<k>)
MOVE_X_Y(<m>)    MOVE_Y_X(<m>)    MERGE_X_Y        MERGE_Y_X
SWAP_X_Y
```

Angle brackets above indicate substitution and are not emitted. A world is:

```text
x0=<int>;y0=<int>;events=[<event_0>,<event_1>,...,<event_d-1>]
```

The source pair is one line with no spaces or final LF and contains no query
template or affine coefficient:

```text
A{<world_A>}|B{<world_B>}|edit=<class>@<zero-based-index>|q0=<X|Y|SUM>
```

Only in Phase Q, the final pair serialization appends:

```text
<source_pair>|qA=(<c1>,<c2>,<c0>)|qB=(<c1>,<c2>,<c0>)
```

`class` is exactly `sign`, `magnitude`, `role`, or `operation_kind`. Literal
brackets, braces, parentheses, commas, semicolons, at signs, equals signs, and
vertical bars are emitted exactly as shown.

### A.2 Fit source renderers

`F0`:

```text
header = "Initial values: x = {x0}; y = {y0}."
ADD = "Add {k} to {r}."
SUB = "Subtract {k} from {r}."
MOVE = "Move {m} from {src} to {dst}."
MERGE = "Move all of {src} into {dst}; {src} becomes 0."
SWAP = "Swap x and y."
```

`F1`:

```text
header = "At the start, register x holds {x0}, while register y holds {y0}."
ADD = "{r} increases by {k}."
SUB = "{r} decreases by {k}."
MOVE = "Transfer {m} units out of {src} and into {dst}."
MERGE = "Combine all units in {src} with {dst}, then clear {src}."
SWAP = "Exchange the contents of x and y."
```

### A.3 Held-out source renderers

`H0`:

```text
header = "Set the opening ledger to x:{x0} and y:{y0}."
ADD = "Raise {r} by {k}."
SUB = "Lower {r} by {k}."
MOVE = "Take {m} from {src}; place it in {dst}."
MERGE = "Pour {src}'s entire value into {dst} and reset {src} to zero."
SWAP = "Let x take y's value and y take x's value."
```

`H1`:

```text
header = "The two counters begin as follows: x has {x0}; y has {y0}."
ADD = "Increase counter {r} using {k}."
SUB = "Reduce counter {r} using {k}."
MOVE = "Shift {m} units from counter {src} over to counter {dst}."
MERGE = "Add counter {src} completely into counter {dst}, leaving counter {src} at 0."
SWAP = "Trade the values of counters x and y."
```

### A.4 Fit query templates

`QF0`:

```text
q0_x = "After all events, what is x?\nAnswer:"
q0_y = "After all events, what is y?\nAnswer:"
q0_sum = "After all events, what is x + y?\nAnswer:"
affine = "After all events, compute ({c1} * x) + ({c2} * y) + ({c0}).\nAnswer:"
```

`QF1`:

```text
q0_x = "Using the final register values, give x.\nAnswer:"
q0_y = "Using the final register values, give y.\nAnswer:"
q0_sum = "Using the final register values, give x plus y.\nAnswer:"
affine = "Using the final register values, evaluate {c1}*x + {c2}*y + {c0}.\nAnswer:"
```

### A.5 Held-out query templates

`QH0`:

```text
q0_x = "When the sequence is complete, report x.\nResult:"
q0_y = "When the sequence is complete, report y.\nResult:"
q0_sum = "When the sequence is complete, report x plus y.\nResult:"
affine = "When the sequence is complete, calculate ({c1} times x) plus ({c2} times y) plus ({c0}).\nResult:"
```

`QH1`:

```text
q0_x = "Use the ending counters to state x.\nValue:"
q0_y = "Use the ending counters to state y.\nValue:"
q0_sum = "Use the ending counters to state x + y.\nValue:"
affine = "Use the ending counters to evaluate {c1}*x + {c2}*y + {c0}.\nValue:"
```

For each pair, only after the board's required `SOURCE_FREEZE`, q0, qA, and qB
independently draw one of the two available query template IDs for their stratum
exactly once. The q0 semantic variant (`x`, `y`, or `sum`) was already fixed only
by the source-syntax rule in Section 5.4; it is not redrawn. No unlisted prefix,
suffix, separator, instruction, event numbering, whitespace, or punctuation is
permitted.
