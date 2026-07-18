# R12 Orthogonal Carry Serializer Curriculum Preregistration

**Protocol:** `R12-OCSC-v8-external-bootstrap-permanent-evidence-candidate`

**Status:** qualification-source repair candidate only. Fresh independent
exact-byte review may authorize local source inspection and, separately, local
or two-host Linux/Lustre qualification of these mechanics. Author tests do not
authorize qualification. Bundle publication, consumer integration, fitting,
evaluation, H100 or other GPU use, promotion, and scientific claims remain
`NO-GO`. They require separate real, reviewed implementations for a consumer,
trainer, evaluator, report validator, parameter ledger, and train/eval exclusion
ledger, plus a separate execution preregistration. This repair authorizes no
cluster access, remote action, corpus publication, model run, or job submission.

**Owned implementation surface:**

```
R12_ORTHOGONAL_CARRY_SERIALIZER_CURRICULUM_PREREG.md
pipeline/generate_orthogonal_carry_serializer_curriculum.py
pipeline/test_generate_orthogonal_carry_serializer_curriculum.py
pipeline/run_orthogonal_carry_serializer_curriculum.py
```

The generator may read `train/digitwise_protocol.py` only. It may not modify
that file or any other existing file.

The reviewed arithmetic oracle is frozen to exactly:

```
path    train/digitwise_protocol.py
bytes   8690
SHA-256 37cd76751eb4146f85268d6c0e44d946eb353ee03605ceb25f4bda97e4c00813
```

The runner file is an ISO C11/Python polyglot. The only authoritative entry is
an independently compiled external C executable made from the exact reviewed
runner bytes. Before starting Python, that executable opens its own actual
executable, the Python interpreter, a reviewed runtime manifest, the pinned
checkout root, all five consumed sources, and every held runtime file. Absolute
paths and checkout-relative source paths are traversed component by component
with descriptor-relative `O_NOFOLLOW`; no `resolve()` or `realpath()` result is
used as pre-open authority. Pre-existing symlink aliases, `..` escapes,
nonregular files, hard links, world-writable files, hash drift, and component
identity changes are rejected.

The external executable hashes its own actual bytes, not a caller-supplied
launcher string or an imported Python constant. It verifies the exact Python
bytes and `shohin-ocsc-external-runtime-closure-v1` manifest before any project
Python import. Runtime records are sorted exact `role<TAB>sha256<TAB>absolute-
path` lines with roles `bootstrap`, `python`, `runtime-held`, or
`runtime-inventory`. Held code and native-image descriptors survive into Python.
The C executable passes a hash-chained attestation through an anonymous retained
descriptor, starts the pinned interpreter with `-I -S -B -c`, and executes the
reviewed runner payload from its retained descriptor. The minimal Python launch
code is part of the compiled executable and is therefore covered by the actual
executable hash; an internal module constant or caller-constructed context has
no standing.

Only after that external attestation does the runner import Python helpers,
validate every retained descriptor and live component identity, configure the
authenticated distribution paths, compile the retained generator bytes, and
inject the source/runtime descriptors. Before any oracle byte is compiled, the
generator verifies the retained oracle descriptor, live pathname identity,
frozen byte length, and frozen SHA-256, then compiles only that snapshot. Source
and runtime descriptors are re-read after the action. Direct generator or runner
execution is rejected and has no authority. A second decimal
state parser, validator, microstep executor, state replay, and terminal-answer
reader is implemented independently inside the reviewed generator and replays
all 48,000 emitted corpus rows. It does not call the reviewed oracle. Every
state, response, local target, carry/borrow, and answer must agree exactly
between the imported oracle and independent replay. A coherently wrong mutable
oracle therefore cannot certify itself.

## 1. Fixed evidence and claim boundary

The design is motivated by these observations, not by OCSC results:

1. Original DRS can write and consume local carry or borrow in its grammar.
2. The original 439,865 rows include 39,985 final-answer rows; all 39,985
   terminal states have `c=0`.
3. Terminal-only data restores carry sensitivity but overpredicts `c=1`.
4. Broad width data harms serializer exactness.
5. In a development-only balanced 16-case field intervention under one-cache,
   EOS-suppressed continuation, nominal second-state exactness was `5/16`.
   Carry-flip full-target exactness was `7/16`, but output changed only `3/16`.
   Six of eight true-carry-one nominal cases were wrong while coincidentally
   matching the carry-flipped target. Written-result flips changed output
   `7/16`, while full-target exactness was `1/16`.
6. On eight carry interventions using frozen weights, full-history continuation
   with stale `S0` retained and generated `S1` produced only `1/8` paired causal
   carry responses. A fresh prompt containing the same `S1` but no `S0`
   produced `7/8`.
7. Independently replayed four-arm evidence is frozen at
   `artifacts/eval_history/digitwise_factorial_v4_four_arm_replay_v5_de45ace_20260718.json`,
   SHA-256
   `d08b17a4fdaf031205ca445bb01f72a2983010e5eb929e6f13ab46409fa5c42f`.
   In the WIDTH to TERM+WIDTH state-exact flow, 179 of 249 gains are frozen
   terminal-carry class `10` (active carry/borrow before the final digit and
   cleared after it), while 200 of 211 losses are class `00`. Subtraction class
   `00` alone accounts for 158 of 211 losses. The largest signed cells are add
   width four class `10` at `+65/-4`, subtraction width four class `10` at
   `+94/-7`, and subtraction width four class `00` at `+36/-107`.
   Symmetric source-arm field reconstruction places `c` at the first mismatch
   in 247 of 249 gains and 207 of 211 losses. Among class-10 gains, `c` is
   involved in 177 of 179. Addition width-four class `10` has 65 gains, all
   `c`-only; subtraction width-four class `10` splits its 94 gains into 90
   `c`-only, two `c+r`, and two `r`-only. Among the losses, `c` is involved in
   196 of 200 class-00 regressions and 155 of 158 subtraction class-00
   regressions. Subtraction width-four class `00` splits into 99 `c`-only,
   seven `c+r`, and one `r`-only first mismatches; addition width-four class
   `00` splits into 23 `c`-only and one `r`-only. Many failures begin before the
   final digit and therefore are not merely final emission errors.
   Directional reconstruction shows that 239 of the 249 WIDTH source errors
   repaired by TERM+WIDTH are expected-carry-one predicted as zero, eight are
   expected-zero predicted as one, and two preserve the carry prediction while
   changing another field. In the opposite direction, 196 of 211 TERM+WIDTH
   regressions are expected-zero predicted as one, 11 are expected-one
   predicted as zero, and four preserve the carry prediction. TERM+WIDTH thus
   shifts a carry threshold through the optimum rather than uniformly improving
   state execution.
8. The superseding direct frozen-parent diagnostic is frozen at
   `artifacts/eval_history/drs_terminal_width_sweep_v2_w2_w10_20260718_mps.json`,
   SHA-256
   `db6056e66310ed7d56509403d40f7549d016294a014c0c4527173b4005210520`,
   against the same DRS parent
   `d79e9df26caecb9801118d1bf68bd7b85381a06b256f23478acffe40a2108459`
   under MPS greedy decoding and frozen-protocol oracles. Widths two through ten
   use matched positive-final-carry and negative/no-final-carry arms with
   identical lower history within each width; only the final operand digits
   differ. The positive arm is `0/9` strict-exact transitions and `0/9` exact
   serializers, but tolerant raw-field reconstruction finds `c` exact at widths
   two and three (`2/9`) and `r` exact in five of nine. Its failure sets are
   `z`-only at widths two/three, `c`-only at widths four/five/seven, `c+r` at
   widths six/eight/nine, and `p+c+r+z` at width ten. The negative arm has raw
   `c` exact `9/9`, raw `r` exact `5/9`, strict transition exact `5/9`, and
   serializer exact `5/9`; its serializer is exact at widths two through six
   and then `0/4` at widths seven through ten. Strict-parser nulls therefore
   cannot be interpreted as every field being wrong. This matched deterministic
   interaction sweep is a diagnosis, not a benchmark, training result, or broad
   reasoning claim.
9. The teacher-forced terminal-carry residual-swap diagnostic is frozen at
   `artifacts/eval_history/drs_terminal_carry_residual_swap_w2_w10_20260718_mps.json`,
   SHA-256
   `4183b8c381e559b23c41b88c8c8cc3b3d0e0b41c03b3dea4786df98a7676590f`.
   At layer 29, positive-versus-negative carry separation is positive in eight
   of nine widths, with an inversion at width six. The absolute `c=1` logit
   exceeds `c=0` only at widths two and three. This shows a broadly present late
   carry direction with a miscalibrated absolute decision boundary; it is not
   autonomous reasoning, a benchmark, or a trained intervention.

Observation 5 demonstrates default-carry-zero confounding. Observation 6 shows
a large compound history-versus-fresh-prompt sensitivity despite available
arithmetic, but it does not identify source presence, cache reset, token
position, or framing separately. Consequently,
nominal target accuracy, counterfactual target accuracy, raw output change,
and paired target-switch are distinct readouts. Raw counterfactual accuracy
has no causal authority and cannot compensate for a failed target-switch gate.
Local carry supervision alone does not establish robustness to that compound
canonicalization package. Source-specific attribution requires SCERT.

Observation 7 is an almost pure carry-threshold tradeoff, not an aggregate-score
target.
Any later GPU preregistration must report gains and losses separately by frozen
terminal carry class, operation, and width. It must retain genuine class `10`
carry/borrow resolution while reducing first-mismatch `c`-field regressions on
matched class `00`; a class-10 improvement cannot compensate for a class-00
regression, and an aggregate exactness increase cannot satisfy this gate. The
carry-policy gate and serializer-exactness gate are separate and
noncompensatory in both directions. OCSC must calibrate matched feasible
carry-positive and carry-negative contexts. Additional positive-carry
supervision without its matched negative-carry control is contraindicated and
cannot be introduced by a later execution preregistration.

Observation 8 separates universal terminal-carry inclusion from
length-generalized reverse readout. A candidate must preserve the frozen
parent's negative-arm serializer competence at widths two through six while
separately repairing positive-arm carry/write behavior and serializer readout
at every width. Aggregate width balancing, carry-only improvement, or
TERM+WIDTH-only gains cannot compensate for a polarity/width failure. Tolerant
raw field-set strata and strict state exactness are both mandatory readouts;
neither may be inferred from the other.

Observation 9 localizes a likely calibration problem but does not authorize a
residual-edit claim. A future intervention must still pass autonomous strict
state and serializer gates; teacher-forced swap direction alone has no
promotion authority.

The narrow hypothesis is that a complete feasible local basis, matched
prefix interventions, and a shortcut-resistant serializer relation can improve
exact DWS execution. Natural-language parsing, ordinary-language arithmetic,
composition, autonomous control, and broad transfer remain fully deferred.

## 2. Exact 24,000-row OCSC corpus

```
22,500 transition rows
+1,500 serializer rows
=24,000 rows
```

Widths are exactly `3,4,5,6,7`. Each width has 900 transition cells and five
contexts per cell, hence 4,500 transition rows.

| Role | Cells per width | Enumeration |
|---|---:|---|
| initial | 200 | `op x a[0] x b[0]`, incoming `c=0` |
| interior | 400 | `op x incoming c x a[p] x b[p]` |
| terminal add | 200 | incoming `c x a[p] x b[p]` |
| terminal sub | 100 | `c=0,a>=b` gives 55; `c=1,a>b` gives 45 |

Initial incoming `c=1` and terminal-subtraction outgoing borrow `1` are
impossible and are always `N/A`. Interior cells use enumeration-order
round-robin over positions `1..w-2`; per-position counts differ by at most one.

### 2.1 Matched context construction

Every noninitial cell has three reachable anchors at contexts `0,1,2`.
Contexts `3,4` are clones of contexts `0,1`, respectively. A clone changes
exactly one already-written `r[q]`, where `q<p`. Input states and one-step
targets differ only at that same `r[q]`; operation, width, position, carry,
operand tapes, terminal flag, active result digit, and outgoing carry are
identical. These 7,000 pairs are `local_prefix_intervention`.

Initial states have no writable prefix. Their `0<->3` and `1<->4` pairs remain
solver-reachable and differ at exactly one unprocessed suffix operand digit.
They are 2,000 `initial_suffix_context_invariance` pairs and are never labeled
prefix interventions.

All target states are reconstructed with `apply_microstep`. Reachable rows
must equal solver replay; interventional rows may differ from replay only in
the written result prefix.

## 3. Serializer corpus and anti-shortcut contract

Each width has five deterministic translation orbits, ten translations per
orbit, and two orientations:

```
50 reversal pairs x 2 orientations x 3 valid slices = 300 rows/width
300 x 5 widths = 1,500 rows
```

Valid slices are `add_c0`, `add_c1`, and `sub_c0`. Paired tapes share identical
operands, and neither tape is the natural arithmetic result. The generator
rejects palindromes, reversal-equivalent orbits, constant-except-one patterns,
fewer than three distinct digits when width permits, fewer than two adjacent
difference values, affine modular-difference patterns, and insufficient
pairwise Hamming separation against translations and reversals. Minimum
orbit separation is `1,2,2,3,3` for widths `3,4,5,6,7`.

Every width has 100 unique tapes, exact ten-per-digit marginals at every LSF
position, ten tapes with an LSF zero, ten with a most-significant zero, and
100 rows in each valid slice. `add_c1` answers have length `w+1`.

The relation graph contains 750 positive reversal pairs and 750 deterministic
cross-pair mismatches. Every serializer batch has two positive pairs and two
crossed mismatches in the same main forward. Reversal alignment maps tape
position `p` to `w-1-p`; positions omitted by ordinary-integer leading-zero
rendering are explicitly absent. Counterfactual edges retain only aligned
positions with different target digits and use a JS margin of `1/10`, so a
constant or uniform candidate distribution cannot minimize the package.

## 4. Slot-matched IID control

The IID control also has exactly 24,000 rows: 22,500 newly drawn reachable
transitions and byte-identical references to the shared 1,500 serializer rows.
Each IID transition is matched one-to-one to an OCSC slot on width, role,
position, operation, context index, completion-mask indices, nonpadding-token
count, and supervised-position count. IID operands are deterministic draws;
they do not enumerate the complete local basis.

`B-A` is therefore only the slot-matched curriculum contrast. Serializer
content, timing, bytes, and dose are identical.

## 5. Independent `[8,256]` main batches

There are exactly 4,875 shared skeleton slots: 4,500 transition slots and 375
serializer slots. The skeleton is deterministic round-robin stratified by
kind, width, role, position, operation, and incoming carry where applicable.

- Transition batch: five real row lanes and three dummy lanes.
- Serializer batch: four real row lanes and four dummy lanes.
- Every batch shape is exactly `[8,256]`, or 2,048 main positions.
- Every lane has its own causal mask and position reset `0..token_count-1`.
- Attention and KV state are block diagonal across lanes. Cross-row attention
  is forbidden.
- Pair endpoints share this one main forward. Extra endpoint forwards are
  forbidden.

Each slot receipt losslessly emits 256 `u32` token IDs, 256 `u8` attention
values, 256 `u8` completion values, 256 `u8` field IDs, 256 `u8` raw units,
and 256 little-endian `u16` position IDs. The 2,560 raw bytes are compressed
with zlib level 9, base64 encoded, and bound by compressed and uncompressed
SHA-256 hashes. Dummy lanes contain the pad token and zero masks. Pack records
bind all eight slot payloads and emit exact pair-to-lane activation maps.

Production tokenization remains bound to:

```
SHA-256 87532df5c121753de3b29194e1f9e3de47986d3f5359548fdf93606773a233d4
bytes   2309567
```

Every decimal digit must encode as one distinct token, and every active target
digit must align to exactly one causal target token. Any lossy frame,
boundary-crossing token, overlength lane, or ambiguous digit alignment is a
hard rejection.

## 6. Paired 5,120-update schedules

Within a seed, every run cell uses the same 4,875-slot stratified skeleton
before any repeat. Across seeds, the deterministic within-stratum order and
balanced repeat selection differ. Each seed has its own committed cycle hash,
update-RNG-stream hash, and order-invariant SHA-256 of its sorted 245-member
repeat-set identity; all three values must be distinct across all three seeds.
Different ordering of the same 245 members is a hard rejection. Exactly 245
prebound skeletons repeat per seed, so 4,630 packs occur once and 245 occur
twice. Every repeat set contains 19 serializer, 50 initial, four inactive
noninitial, 86 active carry-zero, and 86 active carry-one transition slots.
Thus active noninitial local-pair dose remains balanced.

Per run and seed, every schedule row emits the exact skeleton, pack, occurrence,
row presentations, nonpadding tokens, supervised positions, raw units, main
positions, resident and active relation IDs, replay ID, and update RNG seed.
A and B have equal resource values at every update within a seed. Serializer
pack IDs and batch bytes are identical for every cell at identical positions
within that seed. Seed-level batch order is intentionally different and is the
replication variation; rows and updates are never treated as replications.

Unique active noninitial local pairs are exactly `3,450` at incoming carry 0
and `3,450` at incoming carry 1. Scheduled presentations are exactly `3,622`
for each carry value. Initial invariance pairs are outside this balance because
initial carry 1 is impossible.

## 7. Six cells and three paired seeds

The old combined `D-C` label is removed because simultaneous local and
serializer activation does not identify either relation effect.

| Cell | Corpus | Weights | Local relation | Serializer relation |
|---|---|---|---|---|
| A | IID control | uniform | off | off |
| B | OCSC | uniform | off | off |
| M00 | OCSC | field | off | off |
| M10 | OCSC | field | on | off |
| M01 | OCSC | field | off | on |
| M11 | OCSC | field | on | on |

Paired seeds are exactly `2026071801`, `2026071802`, and `2026071803`. Within a
seed, `M00/M10/M01/M11` use identical parent bytes, data, packs, replay,
update RNG, dropout state, and main forwards. They differ only in relation
activation. Across seeds, deterministic stratified batch permutations, repeat
sets, and update RNG streams differ. Each seed is analyzed first as one paired
replication. Pooling rows or updates as independent replication is forbidden.
Within-seed factorial bookkeeping is:

```
local      = (M10-M00) + (M11-M01)
serializer = (M01-M00) + (M11-M10)
interaction = M11-M10-M01+M00
```

These are not pooled across seeds and are secondary to the paired carry-switch
gate for the local package. No cell has promotion authority in this CPU
preregistration.

## 8. Exact scheduled weights

Field raw units are `default=4`, `op=2`, `w=2`, `p=5`, `c=8`, `a=2`, `b=2`,
`r=6`, `z=5`, and serializer answer `=8`. Uniform runs use the emitted
completion mask as unit-one raw weights. For every seed and cell, the generator
recomputes from scheduled emitted vectors:

```
N = scheduled supervised-position count
R = scheduled raw-unit sum
scale = reduced rational N/R
weight_i = raw_i * N/R
```

The emitted proof requires `sum(weight_i)=N` and exact mean `1/1`. Tests decode
the slot payloads and independently recompute `N`, `R`, the reduced fraction,
and the equality.

## 9. Frozen future execution contract, still unauthorized

The required parent checkpoint SHA-256 is:

```
d79e9df26caecb9801118d1bf68bd7b85381a06b256f23478acffe40a2108459
```

Each run must independently load exact parent model/config bytes with fresh
optimizer state. All parent parameters are trainable. One future H100, BF16
parameters/activations, FP32 gradients/loss, batch `[8,256]`, no accumulation,
dropout zero, TF32 off, deterministic algorithms on, cuDNN benchmark off, and
`torch.compile` off are frozen.

Muon receives every trainable 2-D parameter whose name contains neither
`tok` nor `head`: LR `1/50`, momentum `19/20`, Nesterov on, five
Newton-Schulz steps, coefficients `3.4445,-4.7750,2.0315`, no decay. AdamW
receives all other trainable parameters: LR `3/1000`, betas `9/10,19/20`,
epsilon `1e-8`, no decay. Group overlap or omission is fatal. The LR schedule
uses 200 linear-warmup updates, is flat through update 4095, then linearly
decays for 1,024 updates to multiplier `1/10`. Global gradient clipping is 1.
Any skipped or nonfinite update invalidates the run.

## 10. Frozen losses

Main loss is weighted completion-only cross entropy divided by the sum of
normalized weights.

Replay uses the frozen parent in eval mode and exact full-vocabulary
`KL(parent||run_cell)`. The mask contains positions `t` whose replay attention
at `t` and `t+1` is valid. KL is accumulated in FP32 over vocabulary and valid
positions, then divided by valid-position count. Its coefficient is `1/10`.

Local relation loss uses final-normalized hidden states before the LM head and
the emitted causal prediction indices. Candidate logits are restricted to
carry `{0,1}` or digit `{0..9}` and normalized within that set. A local pair is
half the sum of carry JS and active-result-digit JS; eligible active pairs are
averaged. Eligibility is restricted to factorial-active noninitial
`local_prefix_intervention` pairs. Initial suffix-invariance pairs remain
resident for audit but their activation is always zero, and they contribute
zero to every carry numerator.

Serializer positive loss is mean JS over emitted visible reversal-aligned digit
positions. Counterfactual loss is mean `max(0,1/10-JS)` over mismatched aligned
positions. Serializer loss is positive mean plus counterfactual mean.

```
L_total = L_main
        + (1/10) L_KL
        + (1/4) (I_local L_local + I_serializer L_serializer)
```

There is no coefficient-sum renormalization. An inactive component is exactly
zero. Auxiliary relations consume only the main `[8,256]` forward.

## 11. Replay and reserved evaluation

The input registry has exactly 4,096 records:

| Use | DRS | non-DWS | Total |
|---|---:|---:|---:|
| replay | 640 | 640 | 1,280 |
| development | 704 | 704 | 1,408 |
| hidden-confirmation registry reserve | 704 | 704 | 1,408 |

Replay occurs every fourth update, alternates families, and consumes each
prebound replay prompt once per run. All cells and seeds use the same replay ID
at the same update. Normalized-prompt and semantic-signature identities must be
globally unique across replay, development, and `secret_confirmation` registry
records. Training, replay, development, secret-confirmation commitments, and
the later hidden opening must be pairwise disjoint by both identities.

### 11.1 Carry-switch canonicalization-package board

Exactly 170 active noninitial base sites use 680 of the 704 development DRS
prompt commitments. Every width has 17 true-carry-zero and 17 true-carry-one
sites; within each width/carry slice, roles are ten interior, five terminal
add, and two terminal sub. Each base site defines a `2x2` carry/package board:

1. nominal incoming carry with old `S0` present before current `S1`;
2. flipped incoming carry with old `S0` present before current `S1`;
3. nominal incoming carry in a fresh prompt containing only the same `S1`;
4. flipped incoming carry in a fresh prompt containing only the same `S1`.

The history-retained package uses one-cache EOS-suppressed continuation. The
fresh-current-state package starts a fresh cache; its `S1` bytes are identical
and no `S0` bytes are present. This compound contrast simultaneously changes
old-source presence, cache reset, absolute token positions, and framing. It is
not a source-only effect. Source-specific attribution is deferred to a separate
SCERT source-by-cache-by-position-by-framing factorial.

Each carry endpoint reconstructs its own target. An authenticated opening must
store the exact site ID, endpoint and package condition, prompt bytes, endpoint
token bytes and frozen-tokenizer IDs, target-token and causal-prediction
positions, ordered `(active_digit,outgoing_carry)` target, and per-row parse and
exact-score contract. Secret rows are not exposed by this CPU artifact. The
following integer counts are reported separately by package condition:

1. nominal full-target exact;
2. counterfactual full-target exact, descriptive only;
3. raw output bytes changed, descriptive only;
4. paired local target-switch exact: both outputs parse and each emitted
   `(active_digit,outgoing_carry)` equals its own ordered target tuple;
5. canonicalization-package joint exact: all four outputs parse and match the
   local target encoded by current `S1` for their carry endpoint.

For every M11 seed, history-retained-package and fresh-current-state-package
target-switch must each be at least `166/170`; package-joint exactness must be
at least `163/170`; and every width/true-carry slice must be at least `16/17`.
These gates are noncompensatory. Counterfactual full-target accuracy, including
high accuracy under the fresh package, cannot satisfy a failed history or joint
gate. These results cannot authorize a source-specific claim.

## 12. Hidden Merkle confirmation

The hidden direct board has exactly 3,600 rows: 2,100 transitions and 1,500
serializers. Per width, transition geometry is:

| Site type | Sites | Rows/site | Rows |
|---|---:|---:|---:|
| initial suffix invariance | 50 | 2 | 100 |
| interior prefix x carry | 40 | 4 | 160 |
| terminal add prefix x carry | 25 | 4 | 100 |
| terminal sub prefix x carry | 15 | 4 | 60 |

Every noninitial site contains the exact Cartesian product of prefix
`anchor/intervention` and carry endpoint `c0/c1`. At each carry endpoint, the
prefix intervention changes exactly one earlier `r[q]`, `q<p`; every other
source field and byte is fixed, and the target differs only at the same
already-written byte. At each prefix variant, the carry pair changes only `c`;
both one-step targets are reconstructed and their active target digits must
switch. Initial pairs instead change exactly one unprocessed suffix operand
digit, remain reachable at `c0`, and are never carry interventions.

For every noninitial site, the verifier independently replays the operands from
position zero to `p` and requires the one anchor carrying the solver-natural
carry bit to equal that replay byte-for-byte before any prefix intervention is
considered. It is not sufficient for the row merely to carry the correct
reachability label. Corrupting any earlier prefix byte in both arms is fatal.

Full-state uniqueness is also insufficient. For every width, hidden transition
sites mechanically enforce all of the following:

- initial and interior sites contain both addition and subtraction with at
  least `16/50` and `13/40` sites per operation, respectively;
- terminal-add is all addition and terminal-sub is all subtraction;
- interior sites cover every legal position `1..w-2`, with position counts
  differing by at most one;
- active left and right digits cover `0..9` at least twice for initial/interior
  and at least once for terminal-add;
- feasible terminal-sub domains cover active left digits `1..9` and active
  right digits `0..8` at least once;
- each natural site tuple `(op,p,c,a[p],b[p])` is unique; and
- every anchor endpoint tuple
  `(op,p,c,a[p],b[p],target_digit,outgoing_carry)` is unique.

These requirements are frozen inside the hidden geometry commitment. Varying
irrelevant tape context while repeating one local arithmetic transition cannot
satisfy the opening verifier.

Every width/slice serializer cell has 100 rows. Each width has 50 six-row sites:
one operand pair, two non-palindromic reversal tapes, and all three valid
slices. The opening verifier independently enforces identical pair operands,
operation, width and site, natural-result exclusion, unique semantic
signatures, exactly five ten-translation orbits, 100 unique tapes, exact digit
marginals, zero marginals, non-affine adjacent differences, anti-shortcut
diversity, and Hamming separation. Duplicate signatures are fatal.

Hidden rows are not emitted by the generator. The commitment contains only
board ID, exact geometry, leaf count, Merkle algorithm/root, a domain-separated
custodian-opening commitment, and no secret rows. Every opened row includes
exact site/pair IDs, endpoint bytes and token IDs, prediction and target-token
positions, targets, and its tokenizer-bound scoring contract.

Canonical opening rows use ASCII sorted-key compact JSON without a newline in
the leaf payload. For ordinal `i` and canonical row bytes `R`:

```
leaf_i = SHA256("R12-OCSC-HIDDEN-LEAF-v1\0" || u64be(i)
                || u64be(len(R)) || R)
node_l = SHA256("R12-OCSC-HIDDEN-NODE-v1\0" || u64be(level)
                || left || right)
root   = SHA256("R12-OCSC-HIDDEN-ROOT-v1\0" || u64be(leaf_count)
                || tree_top)
```

Odd levels duplicate the last node. The verifier rejects arbitrary roots,
wrong order/ordinals, duplicate IDs, malformed schemas, noncanonical JSON,
noncanonical DWS state strings (including whitespace and leading-zero scalar
encodings accepted by a merely syntactic parser), tampered leaves,
two-position prefix mutations, wrong geometry, invalid DWS witnesses,
collapsed or palindromic serializer tapes, duplicate signatures, and differing
pair operands. On opening, hidden normalized prompts and semantic
signatures are checked against final train, replay, development, and
`secret_confirmation` sets.

Post-publication manifest copying is forbidden as custody evidence. Before the
bundle directory exists, an external custodian must issue one canonical
Ed25519-signed prepublication commitment over the exact output path and parent
directory device/inode/mode/owner-UID identity, mode, artifact inventory, pad token,
tokenizer/registry/hidden-commitment paths, sizes, hashes, file device/inodes,
custody-root paths and device/inodes, and the complete source manifest. The
production trusted public key is frozen as:

```
4d67229fe6b9c62f95ae9208284735fcb4c410e2efb2e4f8a6d935b762887e08
```

Its private key is not present in this repository. Test mode uses a distinct
frozen test-only key with no production authority. This custodian signature is
necessary but no longer sufficient for publication.

A second, distinct external Ed25519 independent-review receipt is mandatory.
Its signed review request binds the exact bytes and SHA-256 of each of the four
owned sources separately; the frozen oracle path, bytes, and SHA-256; the full
tokenizer, prompt-registry, and hidden-commitment custody contracts; the exact
prepublication request object and hash; the signed physical output parent and
canonical pathname; and the byte length and SHA-256 of all ten expected output
files. Its decision is limited to `approve-cpu-publication-contract-only` and
cannot authorize GPU execution, promotion, or a claim. The review receipt is a
sole `0444`, one-link file under its own `0555` custody root and remains outside
the closed ten-file bundle. Generation, ordinary verification, and hidden-
opening verification all require it.

The test review key is distinct from both prepublication keys. It can authorize
test mode only. No production independent-review public key, private key, or
receipt is present or invented in these bytes: the production review trust root
is deliberately unset, so any production review receipt, including one signed
by the test key, is rejected. Freezing a real external production review public
key and supplying a matching exact-byte receipt require a later reviewed source
change.

The generator verifies both signatures. The resulting manifest binds
the immutable receipt path, bytes, SHA-256, request SHA-256, custodian, sequence,
and signer key. Later bundle and hidden-opening verification require that same
prepublication receipt, reconstruct its signed request from live source/input
bytes, and separately reconstruct and verify the independent review request
from the ten bundle files.
Generation captures the tokenizer, registry, hidden commitment, reviewed source
manifest, prepublication receipt, and independent-review receipt exactly once
before publication authentication. Those same immutable byte-and-identity
snapshots are threaded through
`build_artifacts`, its final authenticated-request check, staging, and the last
pre-rename check; no phase may authenticate one pathname opening and consume a
later opening. Independent verification likewise captures one input set and
uses those exact objects for signed-request authentication and deterministic
reconstruction. A caller-supplied receipt dictionary is not trusted.

The tokenizer, registry, hidden opening, hidden commitment, custodian opening,
prepublication commitment, and independent-review receipt each reside as the
only `0444`, one-link file under its own `0555` directory. The bundle directory
is `0555` and its artifacts are `0444`. A late receipt, signer substitution,
test-to-production signer reuse, changed source closure,
substitute registry, moved or self-rehashed bundle, unauthenticated opening,
extra root file, symlink, or hard link is fatal.
CPU request construction, generation, deterministic bundle verification, and
hidden-opening verification all enforce this custody-root contract. Each
custody read pins the root directory, inventories it through that descriptor,
opens the sole child with `openat`/`O_NOFOLLOW`, reads one inode, verifies
`fstat` before and after, verifies the directory entry still names that inode,
then verifies the original and resolved root paths still name the pinned root.
Loaded bytes, hashes, and sizes therefore come from one mechanically stable
snapshot rather than a validate-then-reopen pathname sequence.

Bundle verification similarly opens the requested `0555` bundle directory
once, inventories it through that pinned descriptor, opens all ten `0444`,
one-link artifacts relative to it, and retains the directory and artifact
descriptors through manifest validation, artifact hashing, deterministic byte
reconstruction, and the final directory-path/inode/inventory check. Hidden-
opening verification reuses that same pinned bundle object, including the
already authenticated manifest, train corpora, and replay bytes, through the
contamination comparison and its final check. It may not verify one bundle and
then reopen a replacement bundle by pathname for hidden-set comparison. It also
captures the tokenizer, registry, hidden commitment, prepublication receipt,
and independent-review receipt once before bundle authentication, passes those
exact snapshots into bundle verification, and uses the same tokenizer/registry/
commitment bytes for hidden replay and contamination checks. An A-to-B-to-A
same-path swap between those
phases is therefore either irrelevant to consumed bytes or rejected by the
manifest identity contract and final custody check.

## 13. Integer gates, not execution authority

The three immutable diagnostic evidence identities are part of the
machine-readable `evaluation_gate_contract()` and may not be replaced by a
same-schema or self-rehashed file:

```text
artifacts/eval_history/digitwise_factorial_v4_four_arm_replay_v5_de45ace_20260718.json
d08b17a4fdaf031205ca445bb01f72a2983010e5eb929e6f13ab46409fa5c42f

artifacts/eval_history/drs_terminal_width_sweep_v2_w2_w10_20260718_mps.json
db6056e66310ed7d56509403d40f7549d016294a014c0c4527173b4005210520

artifacts/eval_history/drs_terminal_carry_residual_swap_w2_w10_20260718_mps.json
4183b8c381e559b23c41b88c8c8cc3b3d0e0b41c03b3dea4786df98a7676590f
```

Every seed separately requires `B-A >=60` direct correct, `M00-B >=-6` direct
correct, `M00-B >=10` serializer correct, nonnegative direct brackets
`M10-M00` and `M11-M01`, and serializer brackets `M01-M00 >=10` and
`M11-M10 >=10`. There is no pooled row-level numerator.

The primary local effect is hidden noninitial paired carry target-switch exact
at M11. Every seed requires `M11 >=392/400`, `M11-M01 >=8`, `M11-M00 >=8`,
and `M10-M00 >=0`. All three seed gates must pass. Initial suffix-invariance
rows contribute exactly zero. The M11 hidden target-switch gate is also
`79/80` per width, `198/200` interior, `124/125` terminal add, and `75/75`
terminal sub; these are noncompensatory.

Replay top-1 parent-match gates, per seed and cell, are `1268/1280` overall and
`634/640` in each family.

The immutable replay-v5 dissociation is also a required noncompensatory readout.
Relative to the frozen parent, every candidate must report exact-state gains and
losses for terminal classes `00`, `01`, `10`, and `11`, split by operation and
width. Class `10` must have strictly more gains than losses overall. Class `00`
must have gains at least equal to losses both overall and within subtraction.
Directional carry decisions are scored on matched parent/candidate rows. For
expected-carry-one rows, repairs of parent `1->0` errors must strictly outnumber
new candidate `1->0` errors. Independently, for expected-carry-zero rows,
repairs of parent `0->1` errors must strictly outnumber new candidate `0->1`
errors. Same-carry other-field gains and losses are reported separately and
cannot compensate for either directional carry gate. Effective main and
auxiliary supervision counts and normalized weights must be reported for
matched feasible carry-positive and carry-negative contexts; a positive-only
carry supplement is forbidden.
Among the matched class-00 regressions, the count whose first mismatch involves
the `c` field must be strictly below the frozen v5 references of `196/200`
overall and `155/158` within subtraction. The report must split `c`-only,
`c+r`, and `r`-only first mismatches by operation and width and state whether
the first mismatch occurs before the terminal digit. These `c`-field reduction
gates cannot be offset by any class-10 gain or aggregate exactness increase. A
future source-bound GPU preregistration may tighten these floors before
execution but may not pool or remove these slices after seeing a candidate
result.

The frozen width-two-through-ten sweep is a separate development nonregression
board. Before strict parsing, a tolerant frozen extractor must report the exact
raw mismatch field set over `p,c,r,z` separately for each polarity and width;
parse failure may not null or synthesize those field counters. Strict terminal
state exactness is then scored independently: all nine positive-final-carry
rows and all nine negative/no-final-carry rows must be exact, including final
`r`, `c`, `p`, and `z`. Positive-arm serializers must include the terminal
carry and be exact at every width (`9/9`). Negative-arm serializers at widths
two through six must preserve the parent floor (`5/5`), while widths seven
through ten form a separate reverse-readout generalization gate (`4/4`). No
width, polarity, or raw field may borrow credit from another slice, and no raw
field, transition, residual-swap, or serializer gate may compensate for another.
This board does not replace the larger hidden confirmation board and has no
independent promotion authority.

The layer-29 residual-swap board is a third independent calibration gate. At
every width two through ten, signed positive-minus-negative separation must be
strictly positive and positive-arm absolute `c1-c0` logit separation must be
strictly positive; the frozen width-six inversion must be repaired. Those
teacher-forced directions have no independent authority and cannot compensate
for either the autonomous strict-state gate or autonomous serializer gate.
Width pooling is forbidden.

Hidden gates are `3564/3600` overall, `99/100` in every serializer width/slice,
and `ceil(0.99*n)/n` in every nonempty transition
`width/role/carry/prefix-variant` slice. Initial suffix-invariance is `248/250`.
Impossible slices remain `N/A`.
## 14. Bundle, source, and permanent-evidence policy

The closed test bundle has exactly ten files: both corpora, relations, replay,
tokenization receipts, packs, schedule, commitments, audit, and manifest. The
manifest lists exactly the nine non-manifest artifacts. Duplicate JSON keys,
nonfinite values, unknown keys, unsafe paths, symlinks, nonregular files, hard
links, unsafe modes, and uncommitted bytes are rejected. Tokenizer, registry,
hidden commitment, prepublication receipt, and independent-review receipt are
bound by exact bytes, SHA-256, descriptor identity, and custody-root identity.

The source manifest is `shohin-ocsc-source-manifest-v4`. It binds the four owned
files and `train/digitwise_protocol.py`, the
`shohin-ocsc-bootstrap-source-identity-v2` record, and
`shohin-ocsc-runtime-closure-v3`. The bootstrap identity contains the actual
external executable and runtime-manifest descriptors and hashes, the C
attestation hash, checkout-root identity, source descriptors, pinned Python
identity, and external runtime inventory. A caller-built Python dictionary,
internal launcher constant, direct import, or direct runner/generator invocation
cannot produce an authoritative source-bound CLI result.

The runtime closure binds the pinned interpreter; Python implementation,
version, ABI, and isolation flags; loaded stdlib and distribution modules; full
installed-file inventories for `cryptography` and `tokenizers`; imported
native images; and the external pre-attestation inventory. On Linux, executable
native mappings and `/proc/self/exe` must match the already pinned device/inode
identities. Every consumed loaded-module path must have appeared in the external
manifest before the generator compiled. Shadow modules, lazy imports outside the
inventory, path replacement, and in-place byte drift fail closed.

All path opens in the external bootstrap, runner, and generator use lexical
absolute paths only to choose descriptor-relative components. Each component is
opened with `O_NOFOLLOW` and checked against its directory entry before the next
component is consumed. Checkout sources are opened relative to the pinned root
descriptor. No pre-open `resolve()` or `realpath()` is authority. Existing
symlink aliases, source symlinks, root aliases, `..` escapes, and post-pin
component or final-file replacement are rejected.

`publish_bundle()` remains the one publication mechanics path. Production mode
unconditionally rejects until the separately reviewed downstream implementations
listed in section 15 exist. Test-mode qualification mechanics still require the
signed output-parent identity, deterministic stage/journal/lease names, a kernel
`flock` lease, descriptor-relative `O_EXCL|O_NOFOLLOW` writes, file fsync
after chmod, stage fsync, atomic no-replace rename, parent fsync, strict
descriptor/inode readback, deterministic reconstruction, and two complete
verification passes. The no-replace primitive is
`renameatx_np(RENAME_EXCL)` on macOS and
`renameat2(RENAME_NOREPLACE)` on Linux; unsupported filesystems fail closed.

Evidence retention is permanent. Failed or partial stages, journals, leases,
broker requests, copied broker events, broker receipts, derived reports,
markers, and final receipts are never removed or rewritten by an ordinary
publication, restart, error, or qualification path. A successful canonical
test tree also retains its journal and lease. A partial staged restart verifies
the journal/tree binding, stops, and requires a new externally committed output
identity. A canonical restart may verify the completed tree idempotently but
retains its journal and lease. Torn or absent journals, orphan metadata,
ambiguous stage-plus-canonical state, foreign children, coherent same-UID
forgeries, replacement inodes, collisions, and injected I/O failures remain in
place and fail closed.

`capture_retained_publication_evidence()` performs descriptor-only inventory and
readback without mutation. The five exact crash points are:

```text
stage-created-before-journal
journal-durable-before-first-artifact
partial-artifact-write
stage-fsync-before-rename
canonical-before-parent-fsync
```

The first point records an absent journal; all later points require the journal.
Every point requires the retained lease. The final point requires the retained
canonical tree; the other four require the retained stage. Tree, child, journal,
lease, and parent identities are rechecked after full reads.

The current test inventory is exactly 56 collected tests. It includes hostile
regressions for actual bootstrap-byte drift, runner/source drift, direct
execution, pre-attestation environment shadowing, caller-built contexts, root
and source symlink aliases, final-file and path-component replacement, runtime
shadowing, Linux executable/native-image substitution, no-replace collisions,
both-host lease contention, externally launched broker/report source actions,
all five exact hard-crash points, partial writes, post-rename
fsync failure, foreign and coherent-forgery children, broker omission,
publication-path omission, re-signed malformed crash evidence, boolean and count
forgery, report/marker forgery, and permanent evidence retention. Author tests
remain local evidence only.

## 15. Honest downstream boundary

No real bundle consumer, trainer, evaluator, report validator, parameter ledger,
or train/eval exclusion implementation exists in this repair. Their reviewed
source paths and SHA-256 values are null in
`consumer_interface_contract()`. `nonexecuting_consumer_contract()` can only
validate and canonically copy a reference request. It cannot open a bundle,
construct tensors, fit weights, evaluate a checkpoint, validate a scientific
report, allocate parameters to optimizer groups, prove train/eval exclusion, or
authorize publication.

Bundle publication, consumer compatibility, fitting, evaluation, checkpoint or
metric production, H100 or other GPU use, promotion, and scientific claims stay
`NO-GO` until all six implementations are real, source-bound, independently
reviewed, hash-frozen, and tested against a real reviewed bundle. A separate
execution preregistration and parameter ledger are also mandatory. A local or
Linux/Lustre filesystem qualification receipt cannot satisfy any of these
missing gates.

## 16. Unexecuted two-host Linux/Lustre qualification source

No remote or two-host qualification has been run for these bytes. Fresh
independent exact-byte review is mandatory before even a local qualification
attempt. That review may authorize only source inspection and local or
Linux/Lustre filesystem qualification. It may not authorize scientific bundle
publication, model consumption, fitting, evaluation, GPU work, or claims.

`qualification_contract()` is
`shohin-ocsc-linux-lustre-two-host-qualification-source-v2` with status
`unexecuted-source-contract-only`. Its authority fields for qualification,
bundle publication, consumer integration, fit/evaluation, GPU use, and
scientific claims are all false. The executable qualification-only actions are:

```text
--print-source-manifest
--qualification-output-dir
--qualification-broker-transfer-event
--qualification-write-evidence-package
```

`--qualification-output-dir` additionally accepts exactly one reviewed
`--qualification-crash-point` value from the five-point inventory below. It is
rejected for every other action.

### 16.1 External bootstrap prerequisite

An independent reviewer must compile the exact reviewed polyglot runner as C to
an external absolute path and freeze the resulting executable bytes:

```bash
cc -std=c11 -O2 -Wall -Wextra -Werror -pedantic -x c \
  /absolute/reviewed/shohin/pipeline/run_orthogonal_carry_serializer_curriculum.py \
  -o /absolute/reviewer-controlled/ocsc-external-bootstrap
```

The reviewer must separately create and approve one canonical ASCII runtime
manifest. Its first line and sorted record format are:

```text
shohin-ocsc-external-runtime-closure-v1
bootstrap<TAB><sha256><TAB><absolute bootstrap path>
python<TAB><sha256><TAB><absolute Python path>
runtime-held<TAB><sha256><TAB><absolute imported code/native path>
runtime-inventory<TAB><sha256><TAB><absolute non-code inventory path>
```

The manifest must include the complete transitive runtime and native-image
closure needed by the runner and generator. It is reviewed input, not generated
inside the project Python process. The bootstrap executable, Python, manifest,
and every record must be one non-world-writable regular file with one hard link.
The checkout root and every component must be a real directory, not a symlink.

All qualification-source commands have this external prefix:

```bash
readonly OCSC_BOOTSTRAP=/absolute/reviewer-controlled/ocsc-external-bootstrap
readonly OCSC_RUNTIME=/absolute/reviewer-controlled/runtime.manifest
readonly OCSC_PYTHON=/absolute/reviewer-controlled/python
readonly OCSC_ROOT=/absolute/reviewed/shohin
readonly OCSC_RUNNER="$OCSC_ROOT/pipeline/run_orthogonal_carry_serializer_curriculum.py"

/usr/bin/env -i LC_ALL=C PYTHONDONTWRITEBYTECODE=1 \
  "$OCSC_BOOTSTRAP" \
  --bootstrap-sha256 '<approved actual bootstrap SHA-256>' \
  --runtime-manifest "$OCSC_RUNTIME" \
  --runtime-manifest-sha256 '<approved runtime-manifest SHA-256>' \
  --python "$OCSC_PYTHON" --python-sha256 '<approved Python SHA-256>' \
  --runner "$OCSC_RUNNER" --checkout-root "$OCSC_ROOT" \
  --runner-sha256 '<approved runner SHA-256>' \
  --prereg-sha256 '<approved preregistration SHA-256>' \
  --generator-sha256 '<approved generator SHA-256>' \
  --tests-sha256 '<approved tests SHA-256>' \
  --oracle-sha256 37cd76751eb4146f85268d6c0e44d946eb353ee03605ceb25f4bda97e4c00813 \
  --profile qualification -- <one reviewed qualification-only action>
```

Replacing the final action with `--print-qualification-contract` before the
`--` prints the runner contract after C attestation and accepts no generator
arguments. `-- --print-source-manifest` forces the complete generator runtime
and source closure, but both inspection results have zero authority.

### 16.2 Required execution paths

After a separate review authorizes qualification, node A and distinct node B
must run the exact prefix against the same Lustre mount source and output-parent
device/inode. Node A uses `--qualification-output-dir` and the real frozen
tokenizer, registry, commitment, external test-mode prepublication commitment,
and independent-review receipt. Node B must contend while node A holds the real
kernel lease and must record live exclusion. Both nodes must use distinct host,
FQDN, kernel, publisher, sequence, and nonce identities. A same-host process or
fork is local mechanics evidence only.

Deliberate process deaths must cover all five crash points. Each attempt uses a
new externally committed output identity. No failed attempt is reused or
altered. Restart observation uses the same `publish_bundle()` and
`capture_retained_publication_evidence()` paths. Linux must exercise the real
`renameat2(RENAME_NOREPLACE)` branch on Lustre, file fsync after chmod, stage
and parent fsync, descriptor/inode readback, collision rejection, live flock
exclusion, and canonical restart verification.

The crash selector is inside `publish_bundle()`, not a parallel simulator. It
sends `SIGKILL` after the stage is opened but before its journal, after the
journal and parent are durable, after a durable strict prefix of the first
artifact, after the complete stage is chmoded and fsynced, or after the
no-replace rename but before parent fsync. Canonical restart fsyncs the parent
before strict descriptor readback. The other four restart states fail closed
and retain the exact stage, journal when present, lease, and children.

The production broker path is
`execute_qualification_broker_transfer()`, exposed by
`--qualification-broker-transfer-event`. It validates one signed raw event,
writes an immutable signed broker request with no replacement, copies the exact
event with descriptor-relative no-replace I/O, fsyncs the file after chmod,
fsyncs the parent, performs descriptor/inode and exact-byte readback, then
writes an immutable signed broker receipt. Request, copied event, and receipt
remain at their original paths and inodes.

`--qualification-write-evidence-package` accepts only
`shohin-ocsc-qualification-raw-evidence-request-v1`: reviewer identity,
sequence, nonce, command, raw signed events, signed broker requests, signed
broker receipts, and its payload hash. It has no fields for checks, booleans,
summaries, numerators, or denominators. The source derives and no-replace writes
the report, marker, and final receipt. The signing key is a separately pinned
`0400` custody file supplied with `--qualification-signing-key`. No production
qualification key is configured in these bytes, so production mode remains
fail-closed.

The derived report, marker, and final receipt are each written no-replace into
a distinct deterministic one-file custody root. Each file is descriptor-read
back against its directory entry, each root is fsynced and frozen at mode
`0555`, and the package parent is fsynced. The externally attested action then
revalidates the final signature, all events, both broker records, report, and
marker before returning its zero-authority verification record. Any partial
custody root remains permanently retained.

### 16.3 Event-derived machine contract

Each raw event is
`shohin-ocsc-qualification-event-v1`, signed in domain
`R12-OCSC-QUALIFICATION-EVENT-v1\0`. It binds qualification, host, FQDN,
kernel, per-host sequence and previous-event hash, event type, nonce, source
manifest, Lustre mount source/path, output-parent device/inode, and structured
details. Booleans are forbidden. `evidence_sha256` is recomputed from the other
detail fields. Per-host sequences start at one with no gaps and hash-chain every
signed event.

Exactly one event is required for each check below:

```text
external_bootstrap_source_bound
all_consumed_source_bytes_pinned
runtime_closure_complete_before_action
real_tokenizer_registry_consumed
cross_node_distinct_hosts
same_lustre_mount_and_output_inode
production_broker_transfer_complete
publication_path_complete
renameat2_noreplace_real
descriptor_relative_io
file_fsync_after_chmod
stage_and_parent_fsync
kernel_flock_live_exclusion
stale_lease_observed_without_mutation
all_crash_evidence_permanently_retained
canonical_death_recovery
collision_rejected
path_substitution_rejected
coherent_forgery_foreign_child_preserved
partial_child_preserved
foreign_replacement_preserved
runtime_shadow_import_rejected
injected_io_fail_closed
strict_full_readback
report_marker_receipt_event_derived
permanent_evidence_inventory_recorded
```

The broker-transfer check additionally requires exactly one
`shohin-ocsc-qualification-broker-request-v1` and one
`shohin-ocsc-qualification-broker-receipt-v1`, both signed and hash-bound to
that event and to the same source manifest. The publication-path event must
contain this exact ordered path:

```text
production-broker-transfer
publish_bundle-stage-no-replace
file-fsync-after-chmod
stage-fsync
rename-noreplace
parent-fsync
descriptor-inode-readback
```

`derive_qualification_report()` produces
`shohin-ocsc-qualification-derived-report-v1`. Its `check_evidence` values are
event/request/receipt ID lists, never booleans. Its summary is mechanically fixed
to 26 raw events, one broker request, one broker receipt, 26 derived checks, five
retained crash records, and two distinct hosts. `qualification_marker()`
revalidates these counts and IDs. Neither accepts a caller denominator or success
map.

The final canonical receipt is
`shohin-ocsc-linux-lustre-qualification-receipt-v3` with exactly:

```text
schema
qualification_id, reviewer_id, sequence, nonce_hex
command, command_sha256
source_manifest, source_manifest_sha256
raw_events
broker_requests
broker_receipts
derived_report
marker
claim_boundary
signature_algorithm, signer_public_key_hex, signature_hex
```

Its signature domain is
`R12-OCSC-LINUX-LUSTRE-QUALIFICATION-v3\0`. Verification checks the signature,
revalidates every raw event and broker chain, re-derives the report and marker,
and requires exact recursive equality. A re-signed forged count, check,
publication path, crash inventory, event, report, or marker is rejected.

### 16.4 Decision boundary

Author tests establish source behavior only. They do not establish a Lustre
mount, distinct physical hosts, external reviewer identity, or qualification
receipt. The honest result after author verification can be at most `GO` for
fresh independent exact-byte source rereview and qualification-source review.
Actual local or Linux/Lustre qualification remains `NO-GO` until that review
explicitly authorizes it. Remote qualification is never authorized by author
tests.

Even a valid future qualification receipt has
`qualification_authority=false` in this repair and cannot authorize production
bundle publication or any downstream scientific action. Consumer integration,
fit/eval, H100, model reports, promotion, and claims remain `NO-GO` until the
separate implementations and reviews in section 15 exist.
