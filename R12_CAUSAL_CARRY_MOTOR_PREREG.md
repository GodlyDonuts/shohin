# R12 causal carry motor preregistration

**Status:** amended frozen design; no canonical fit is authorized until the
eight-shard extractor, merger, tests, and batch job pass independent review.
The first monolithic extraction allocation demonstrated that exact batch-one
features cannot finish inside one bounded H100 window and produced no artifact.
The amendment changes only execution custody, not rows, features, fit budget,
arms, or decision gates. This experiment is intentionally smaller than FCRC and
must run first.

## 1. Question

The post-DRS causal cycle isolated a narrow failure.  On the frozen 50-case
boundary board, the native model produced the exact first state in 38/50 cases.
Every native failure first diverged at the serialized `c=` value.  At a
teacher-forced prefix, the newly written result digit was correct in 50/50
cases.  Supplying only the target carry and result digit tokens made the whole
state exact in 50/50 cases.  An oracle residual transplant did not reliably
move the carry token, while the next call changed its active result digit in
the counterfactual direction in 40/50 paired cases.

This experiment asks one question:

> Does the frozen late residual contain enough information for a tiny learned
> output motor, activated only at the grammar-defined carry site, to serialize
> the correct carry and thereby improve autonomous multi-step execution?

The experiment does **not** test a general workspace, broad language reasoning,
or a new computational class.

## 2. Frozen inputs

| Input | SHA-256 |
|---|---|
| `train/sft_digitwise_recurrent_v2_200k_r3/sft_ep1.pt` | `d79e9df26caecb9801118d1bf68bd7b85381a06b256f23478acffe40a2108459` |
| `artifacts/evals/digitwise_recurrent_v2_heldout.jsonl` | `89ce11b36ff2f56e83cda72a1f07b1a90f4a3dc3803c69db2779a27219712646` |
| `artifacts/shohin-tok-32k.json` | `87532df5c121753de3b29194e1f9e3de47986d3f5359548fdf93606773a233d4` |
| `artifacts/evals/drs_causal_cycle_post_drs_r3.json` | `0b927fee009de5e5cf87971ecaf390c716d6d9acb5644cabe3c176f6da9d4e7a` |

The checkpoint is a 30-layer, 576-wide, 125.1M-parameter DRS model. Its
canonical checkpoint-step identity is the exact JSON string `"sft_ep1"`; an
integer epoch/update surrogate or any other string is invalid. Every base
parameter, tied embedding/output weight, normalization weight, and cache rule is
frozen.

## 3. Minimal state theorem and equivalence boundary

For a fixed-width decimal add/sub transition with the immutable operand tapes,
result prefix, operation, and externally serialized cursor available, the only
cross-position arithmetic state is one carry/borrow bit.  This follows directly
from the ripple transition

`(a[p], b[p], c[p]) -> (r[p], c[p+1])`.

Previously written result digits are not needed to compute the next local
transition.  If the cursor were not externally available, additional control
state would be required; this experiment does not remove that cursor.

The proposed motor is equivalent to a small grammar-gated mixture-of-output-head
adapter.  It is not a new computational primitive.  Its useful causal claim,
if it passes, is narrower: a frozen general language backbone can hold the
local arithmetic variable while a tiny specialized motor repairs the
representation-to-action bottleneck without broad weight updates.

## 4. Architecture

Let `h` be the frozen residual after block 29 at the final prefix token and let
`l` be the frozen vocabulary logits.  The treatment motor is

`m(h) = W_up * SiLU(W_down * h + b_down) + b_up`,

where `W_down` is `8 x 576` and `W_up` is `2 x 8`.  With both bias vectors, the
motor has exactly 4,634
trainable parameters.  It adds its two outputs only to the tokenizer's single
character `0` and `1` logits.  All other logits remain bit-identical.

The motor is active only when all of these conditions hold:

1. generation is inside a DWS microstep response, not the prompt;
2. the already generated canonical response prefix ends exactly after `;c=`;
3. the prefix before `;c=` has the canonical `dws:op=...;w=...;p=...` shape;
4. the site has not already fired for the response.

The router reads grammar and position only.  It may not inspect operands,
desired carry, a solver result, a residual donor, verifier output, or future
tokens.  Router use is reported as one external binary site-classification call
per generated response.  The learned motor, not the router, chooses the carry.

## 5. Fit data and budget

The fit generator is solver-backed but the solver is available only while
constructing labels.  It samples complete add/sub episodes, derives each
reachable interior state by replaying the canonical decimal transition, and
creates the exact teacher-forced prefix ending after `;c=`.  Fit widths are 4
and 6.  Core and held-out prompt phrasings are balanced.  Exact prompt hashes in
the frozen 1,500-episode development board are rejected from fit.

The admitted fit board must satisfy:

- equal counts of next carry 0 and 1 within every admitted
  `(operation, width, position, prompt_style, current_carry)` stratum;
- position zero admits only its reachable current carry zero; later positions
  admit both current-carry values;
- terminal subtraction is excluded from fit because a valid nonnegative
  subtraction can only emit terminal borrow zero there; it remains an explicit
  evaluation stratum rather than becoming a one-class position shortcut;
- no duplicate prefix token sequence;
- single-token, prefix-stable targets for `0` and `1`;
- no development-board prompt hash;
- a complete manifest binding generator, tokenizer, checkpoint, row order,
  stratum counts, and token-length histogram.

Frozen residual features are extracted once with the base in evaluation mode,
no gradient, and the exact prompt-prefill plus incremental KV-cache path used
during generation.  Both learned arms receive the same features, initialization,
optimizer, number of updates, batch order, and full-vocabulary cross-entropy.
The frozen exact budget is 2,000 AdamW updates, batch 512, learning rate
`3e-3`, weight decay `1e-4`, seed `20260717`.  The implementation must fail
closed if it cannot supply exactly that many full batches; no early stopping or
hyperparameter choice may use confirmation data.

Canonical residual extraction is frozen at batch size one.  This is slower than
batched extraction but is required because bf16 GEMM shape changes produced
small, reproducible residual differences between batch-one generation and
batched extraction.  Every fit and evaluation feature therefore follows the
same batch-one prompt-prefill plus incremental-cache path as inference, and the
batch size is part of the immutable canonical budget.

Canonical fit extraction is partitioned into exactly eight deterministic
shards by sorted global row index modulo eight. Each shard therefore contains
exactly 8,192 of the frozen 65,536 rows. It regenerates and verifies the entire
board, records the global indices and row-identity hash, and publishes one
exclusive-create immutable feature artifact. Failed or missing shards may be
rerun independently; completed shards are never appended to or overwritten.

Every shard additionally re-extracts the same four sentinels, one from each
`(prompt_token_length, prefix_token_length)` shape. The merger requires exact
tensor equality for all sentinel features across all eight H100 processes,
exact source/input contracts, exact artifact receipts, a unique complete shard
index set, zero primary-row overlap, zero coverage gaps, expected labels,
identical vocabulary/token identities and dtypes, and a deterministic merged
feature hash. The motor fit starts only after those checks. Sharding changes no
forward call: each primary and sentinel row is still evaluated alone through
the canonical cached path.

Before any plan publication, extraction, or fit, a separate Stokes custodian
publishes the immutable confirmation commitment specified below. Only after
that commitment is sealed does a CPU-only Stokes job publish one immutable
canonical plan at the sole allowed absolute path
`/lustre/fs1/home/sa305415/shohin/artifacts/carry_motor/canonical_${SOURCE_COMMIT}/plan.json`.
Python and both wrappers derive this path from the reviewed 40-hex commit and
reject every caller-selected alternate root. The plan
binds the four absolute frozen-input paths and hashes, the exact sealed
confirmation-commitment path, bytes, receipt, and parsed document, exact
reviewed source manifest, ordered confirmation-exclusion identities and digest,
the exact checkpoint-step string `"sft_ep1"`, generated board and row order, every
modulo-shard membership and fixed artifact path, sentinel identities, tokenizer/model
dimensions, exact PyTorch/CUDA/H100 runtime, fit schedule, shuffled-label
assignment, the seed-derived initial motor state hash, and the complete frozen
development-selection contract in Section 7. It also freezes the exact teacher
scoring contract `h100_bfloat16_batch1_apply_motor_logits_v1`. Before publication
or validation comparison, Python normalizes the complete expected plan through
strict finite JSON serialization and duplicate-key-rejecting parsing. Thus JSON
object keys are strings and serializable scalar subclasses become primitive JSON
values in both memory and the sealed bytes. Validation compares those normalized
strict JSON payloads rather than Python object equality, so JSON booleans,
integers, and floats are never interchangeable. The plan root is an exact
non-symlink mode-`0555` directory; `plan.json` is a regular non-symlink
mode-`0444` file with `st_nlink == 1`; its eight shard, fit, development
evaluation, and confirmation evaluation directories are empty mode `0700`
before their sole writer runs. The root has no children other than `plan.json`
and those eleven planned directories.

Canonical objects use the distinct closed-world audit tags
`causal_carry_motor_plan_v6`,
`causal_carry_motor_feature_shard_v6_canonical`, and
`causal_carry_motor_fit_v8_canonical_sharded`. Development and confirmation
reports use `causal_carry_motor_development_eval_v7` and
`causal_carry_motor_confirmation_eval_v2`. The legacy monolithic command is
development-only and cannot emit a canonical tag. Each canonical stage accepts
only the fixed planned path, exact
lowercase plan SHA-256, exact tensor keys/shapes/dtypes/finiteness/vocabulary
identity, and `NVIDIA H100 PCIe` runtime. Extract, fit, and evaluation each
enforce the exact one-visible-H100 runtime inside Python; planned evaluation has
no CPU/MPS/noncanonical bypass. Each writer stages bytes outside the planned
output directory, hard-links the final mode-`0444` artifact by exclusive create,
then seals the exact one-file directory. A crash after publication but before
directory sealing is recoverable only when the mode-`0700` directory contains
exactly the complete mode-`0444` artifact and that artifact passes its full
canonical validator. Siblings, staging remnants, writable artifacts, and
partial schemas fail closed. No sidecar may substitute for the one-file tensor
receipt. The fit bundle additionally binds
all eight sorted shard receipts, merged tensor hash, exact treatment/control
schedule, control-label hash, initial state, arm state schemas, and arm state
hashes. Its linear diagnostic, every feature-metric arm, all counts,
accuracies, finite values, schedule receipt, and claim boundary have closed-world
validators. Every fit-time feature-metric arm retains complete per-row tensor
evidence derived from the merged shards; no fit aggregate is admitted from a
summary alone. That evidence retains the exact merged float32 hidden tensor and
bf16 base carry logits. Every canonical fit-bundle validation, including
recovering an already published fit and validating a fit before either downstream
evaluator, first performs the complete eight-path shard preflight, then binds and
loads all eight exact sealed snapshots, validates and merges their tensors, and
recomputes both merged-payload hashes. The validator requires the saved merge
object to equal that replay, derives hidden, base carry logits, targets, and
non-carry maxima directly from the replayed bytes, and requires every retained
tensor to equal those values. It then loads the exact treatment and shuffled
state dictionaries on the canonical H100 and recomputes each learned delta one
row at a time with batch size one through the same `apply_motor_logits`
arithmetic used by autonomous decoding. Each float32 delta is cast to bf16 and
added to the retained bf16 base row before exact tensor equality and any row
prediction or summary are admitted. The retained source-feature hash includes
hidden, base logits, labels, non-carry maxima, token identities, and deployment
dtype and must equal the hash independently produced by the replayed sealed
shards. A self-consistent retained-feature rewrite cannot be accepted by changing
bundle-local hashes while shard receipts remain unchanged. Shard arrival order
must not change the merged feature or receipt hash.

Before constructing `BoundInput` for a canonical plan, Python validates the
exact lexical commit-bound plan path, plan/root file types, modes and link
counts, the closed-world child set, and each directory's empty, recoverable, or
sealed lifecycle state. A recoverable state is only a mode-`0700` one-file
directory containing its complete regular non-symlink mode-`0444`, one-link
artifact. Fit is forbidden unless every descriptor is its exact planned shard
path and is already a regular non-symlink mode-`0444`, one-link file inside its
mode-`0555` one-file shard directory. Python first validates all eight paths
completely, without constructing any shard `BoundInput` and without calling
`torch.load`. Only after that pass succeeds does a second ordered pass
immediately recheck a shard, construct its `BoundInput`, and load its private
snapshot. Thus an invalid shard 1 prevents even shard 0 from being bound or
loaded. A nonempty fit requires all shards sealed; a nonempty development
evaluation requires the fit sealed; and a nonempty confirmation evaluation
requires the development evaluation sealed. These rules preserve crash recovery
without admitting linked or writable inputs.

The H100 wrapper repeats the fit-input check before Python: for each of the
eight exact planned shard paths it requires a regular non-symlink mode-`0444`
file with `st_nlink == 1`, the sole child of its regular non-symlink
mode-`0555` shard directory. Python remains authoritative and performs the
complete pass plus the immediate second-pass checks described above.

Both Slurm wrappers require a clean exact-commit checkout, compare their private
spooled script bytes with the reviewed wrapper, and derive the scientific source
manifest from committed bytes before invoking Python. Before `nvidia-smi`, CUDA
availability checks, or a CUDA tensor allocation, the H100 wrapper runs the
`validate-confirmation` Python subcommand with `CUDA_VISIBLE_DEVICES` empty. That
CPU-only path binds the frozen inputs and sources, parses the commitment with
duplicate-key and non-finite rejection, derives the complete exclusion contract,
and requires exact semantic equality. This remains process and
filesystem custody within the stated same-UID trust boundary; it is not remote
attestation against a malicious scheduler or account owner.

## 6. Arms

1. **Base:** no motor.
2. **Treatment:** correct carry labels.
3. **Shuffled-label control:** labels permuted deterministically within each
   nuisance stratum while preserving exact class counts.
4. **Dead motor:** the treatment architecture with all deltas forced to zero.
5. **Linear diagnostic:** a separately reported two-class linear probe on frozen
   features.  It is diagnostic only and is never inserted into generation.

The treatment and shuffled checkpoints must bind the same initial-parameter
hash before their first update.  Gate-off treatment logits must be exactly equal
to base logits, not merely numerically close.

## 7. Boards and custody

### Development

The existing 1,500-episode DRS held-out file is development data because it has
already been inspected repeatedly.  It contains fit-width, value-OOD, and
width-8 regimes. Before any canonical plan, fit, or result, the only admitted
development selection is frozen to these 300 records in source-file order:

- `fit_width`: first 50 `fit_w4` plus first 50 `fit_w6`;
- `value_ood`: first 50 `value_ood_w4` plus first 50 `value_ood_w6`;
- `width_8`: first 100 `width_ood_w8`.

The ordered identity list records selection index, original zero-based source
index, episode ID, source split, and named regime. Its immutable SHA-256 is
`0a68fe542306ae954696c3346cb9c6dcfff14e638e9dd26f0996d46c27e5e80b`.
The exact 300 identities, digest, three named counts of 100, and five source
counts of 50/50/50/50/100 are repeated verbatim in `plan.json` and the
development report. The validator independently reconstructs this contract
from the frozen episode bytes and rejects any identity, order, count, digest,
or plan/report disagreement. Canonical evaluation has no caller-controlled
selection or `--per-regime` option. Development results can reject a candidate
but cannot alone establish the claim.

### Confirmation

Before the canonical plan exists, and therefore before extraction or fit, the
code and confirmation-board generator hashes are frozen. A Stokes custodian
creates exactly 32 secret bytes, stores them outside every trainer, plan,
extractor, and fit input path, and publishes only `SHA256(secret)` in
`/lustre/fs1/home/sa305415/shohin/artifacts/carry_motor/confirmation_commitments/commitment_${SOURCE_COMMIT}/commitment.json`.
The containing directory is regular non-symlink mode `0555` with exactly that
one child; the JSON file is regular non-symlink mode `0444` with
`st_nlink == 1`. The lowercase SHA-256 of the commitment bytes is separately
supplied as its receipt. Trainers and fit jobs never read the secret.

The commitment is one exact JSON object with no additional keys:

```json
{
  "audit": "causal_carry_motor_confirmation_commitment_v4",
  "canonical": true,
  "source_contract": {
    "git_commit": "<reviewed lowercase 40-hex commit>",
    "manifest_sha256": "<reviewed scientific-source manifest SHA-256>"
  },
  "generator_source_contract": {
    "schema": "causal_carry_motor_confirmation_generator_v4",
    "entrypoint": "train/causal_carry_motor.py:generate_confirmation_board",
    "sources": {
      "train/causal_carry_motor.py": "<SHA-256 from the reviewed source snapshot>",
      "train/digitwise_protocol.py": "<SHA-256 from the reviewed source snapshot>"
    },
    "manifest_sha256": "<stable JSON SHA-256 of the exact sources mapping>"
  },
  "exclusion_contract": {
    "audit": "causal_carry_motor_confirmation_exclusions_v1",
    "episodes_sha256": "89ce11b36ff2f56e83cda72a1f07b1a90f4a3dc3803c69db2779a27219712646",
    "cycle_sha256": "0b927fee009de5e5cf87971ecaf390c716d6d9acb5644cabe3c176f6da9d4e7a",
    "prompt_count": 33700,
    "operand_count": 3000,
    "identity_count": 36700,
    "identities": ["<the exact ordered identity objects specified below>"],
    "identity_sha256": "df2d7fc97f22b9bd8987141095f95ec2cf0240f4c4bf463f53996f82ef6c1f00"
  },
  "secret_sha256": "<lowercase SHA256 of the unrevealed 32-byte secret>",
  "timing": "published_before_canonical_plan_extraction_and_fit",
  "claim_boundary": "This pre-fit commitment freezes generator identity, the exact development/cycle exclusion identities, and SHA256(secret). It contains no secret, confirmation board, score, or capability result."
}
```

The exclusion list is a unique sorted set derived internally from the exact
frozen inputs. It first contains every unique prompt identity as exactly
`{"kind":"prompt","sha256":"<lowercase digest>"}` sorted by digest. Prompt
identities cover both canonical prompt styles for every factual and
counterfactual development transition plus both calls of every frozen cycle
case. It then contains every unique operand identity sorted by its digest as
exactly
`{"kind":"operand","operation":"add|sub","width":<integer>,"left":<integer>,"right":<integer>,"sha256":"<stable JSON digest of operation/width/left/right>"}`.
Counts, input hashes, the complete ordered list, and its digest are mandatory;
no extra key is admitted.

Python rejects duplicate JSON keys, non-finite values, any secret-bearing or
extra field, source/entrypoint/hash disagreement, malformed digest, alternate
path, receipt mismatch, writable or linked bytes, or non-closed-world custody.
Plan, extract, fit, and development evaluation all bind the same bytes before
CUDA setup or planned-artifact consumption. `plan.json`, every shard, the fit
bundle, and the development report carry the commitment receipt; after plan
publication, a rewritten commitment cannot be admitted by recomputing caller
flags because it no longer equals the immutable plan. The plan repeats the exact
exclusion contract outside the embedded commitment as a redundant equality
check.

After the treatment and shuffled checkpoints and development report are
immutable and hash-recorded, the secret is revealed once as exactly 32 raw bytes
in one absolute regular non-symlink mode-`0400`, one-link file. The secret path
is forbidden in every other H100 mode and the secret bytes never enter a plan or
result. The frozen public entrypoint is exactly
`generate_confirmation_board(secret_bytes, bound_inputs, frozen_sha256,
commitment_document, plan_path, plan_sha256, plan_document)`. It has no defaults
and does not accept caller-provided episode text, cycle text, exclusions, rows,
or seed.

Before deriving a row, the entrypoint verifies all four bound input paths and
hashes against the immutable plan and the exact preregistered hashes; independently
hashes the episode and cycle bytes held by their `BoundInput` snapshots; derives
the complete exclusion contract internally; requires its exact equality in the
commitment and plan; requires `SHA256(secret)` to equal the committed digest;
and binds the exact commitment path, receipt, and document. Independently of the
caller's already parsed plan object, it derives the sole plan path from the
committed source revision, validates the full closed-world plan lifecycle,
constructs a new `BoundInput` for that exact regular non-symlink mode-`0444`
one-link `plan.json`, checks the supplied receipt, parses only those bound bytes,
and requires the supplied `plan_document` to equal that parsed object exactly.
It rechecks the bound plan path and bytes before returning. An alternate path or
receipt, including `/hostile/plan.json` with an `eeee...` digest, cannot identify
a board. Its seed binds `SHA256(secret)`, both frozen-input hashes, and the
exclusion-identity hash. Thus the committed secret and frozen bytes admit
exactly one board. An alternate 32-byte secret or any changed source record,
including episode index 1499, fails before canonical evaluation. No retry or
alternate secret is permitted. This is an honest same-UID custody boundary, not
cryptographic isolation from the account owner.

The confirmation board contains fresh values at widths 4 and 6 plus unseen
widths 8 and 10, balanced add/sub operations, carry outcomes, positions, and
prompt styles. It rejects every internally derived development/cycle prompt and
operand identity.

The returned board is one exact object containing only generator audit,
committed secret digest, all four planned frozen-input path/hash identities, the
exact commitment binding, complete exclusion contract, plan path/receipt, all
256 ordered rows, and the stable JSON row digest. Every row retains its index,
ID, width regime, operation, width, style, target carry, position, operand and
prompt digests, prompt, exact expected state, selected transition, and complete
solver-derived episode. The episode retains its ID, split, operation, width,
operands, style, initial state, every expected state, and final answer. Canonical
validation independently regenerates this full object and requires exact
equality; accepting only a caller-supplied row digest is forbidden.

`confirmation-eval` is the sole post-reveal command. It requires the sealed fit
and development report, regenerates the board before CUDA model setup, and
writes only
`canonical_${SOURCE_COMMIT}/confirmation_eval/evaluation.json`. The result
repeats the secret digest, all frozen-input identities, complete exclusion
contract, exact commitment, plan identity, full board, and row digest. It also
binds the fully validated development report path and SHA-256, so its frozen
cycle, direct-call, and exact token/logit preservation gates remain part of the
decision. For all 256 rows and all four arms it retains complete teacher-forced
evidence, one-step raw calls, and autonomous per-episode raw calls. Validators
replay every token-boundary ledger, derive every one-step and autonomous total,
and reject any row or binding rewrite. The confirmation directory may become
recoverable or sealed only after the development directory is sealed.

## 8. Evaluations

All arms are evaluated with identical deterministic greedy decoding.

The frozen H100 path emits bfloat16 base logits.  Motor deltas are computed in
float32, cast to the base-logit dtype, and added in that dtype in fitting,
teacher-forced scoring, and autonomous decoding. Global top-one uses the
decoder's deterministic lowest-token-ID tie break. No global-rank metric is
preregistered or reported.

Every treatment and shuffled teacher-forced score used by fit, development, or
confirmation is executed on the canonical single visible `NVIDIA H100 PCIe` one
row at a time with an exact `(1, 576)` hidden input. There is no all-row CPU or
batched motor-head decision path. Each row calls the same `apply_motor_logits`
function as autonomous generation, so clone, float32 motor forward, per-column
bf16 cast, and bf16 addition order are identical. Every teacher report records
the immutable `h100_bfloat16_batch1_apply_motor_logits_v1` contract, deployment
dtype, and complete per-row top-one evidence; validators reject a missing or
altered execution contract.

1. Teacher-forced next-carry accuracy and exact global-vocabulary top-one
   accuracy. Every development row retains its complete frozen identity and
   target, both adjusted carry-token logits, the exact maximum non-carry token
   ID and logit, redundant carry/global predictions and correctness bits, and
   the exact token-boundary router site/fire decision. The validator reconstructs
   targets from the frozen selected episodes and tokenizer, derives each
   lowest-token-ID tie break, prediction, correctness bit, site/fire count, and
   every aggregate. Sparse competitor lists and global rank are forbidden.
   Canonical fit retains one ordered identity object per planned row, true and
   shuffled-control target tensors, exact non-carry maximum ID/logit tensors,
   and for every arm the complete adjusted two-carry-logit tensor plus redundant
   target-token, prediction, correctness, site, and fire tensors. Their source
   feature payload hash binds the merged shards. The retained exact hidden and
   base tensors are first compared with the independently replayed shard bytes,
   then replayed row-by-row on H100 through the exact treatment or shuffled state
   with float32 motor computation, bf16 cast, then bf16 addition. The validator
   requires every adjusted tensor to equal this recomputation before deriving
   each row and summary; a trusted adjusted tensor or summary-only fit report is
   invalid.
2. Exact one-step canonical state. Confirmation retains and reparses the raw
   generated call for every one of its 256 rows and every arm.
3. Autonomous full episode: every state exact, terminal answer exact, and first
   failure position. For every one of the 300 selected development episodes and
   every one of the 256 secret-derived confirmation episodes in every arm, the
   applicable report retains every ordered model call, raw prompt, raw response, and
   per-call router-site and motor-fire count, including the final-answer call
   after a closed state loop. The validator replays every transition against
   the frozen episode, parses every raw state and final answer, and derives each
   compact episode record and every named-regime aggregate. Compact accounting
   and totals are redundant claims only; they are never evidence for one
   another. The 15-item transcript view must equal the first 15 full evidence
   records and is only a redundant prefix sample. Each raw call additionally retains the exact
   generated token IDs, every decoded prefix at which the decoder actually made
   a next-token decision, that boundary's router-site and motor-fire booleans,
   and the exact EOS, sequence-cap, complete-answer, or max-token stop reason.
   The validator re-decodes those token boundaries with the frozen tokenizer
   and never invents character boundaries; a token decoding to `==` cannot
   create a skipped intermediate `=` site.
4. The frozen 50-case boundary causal cycle, with no oracle token or residual.
   All 50 cases are retained, not sampled. Each case retains every raw call and
   its per-call site/fire counts plus redundant case aggregates. The validator
   parses both responses when the first call is exact, derives first-, second-,
   and integrated correctness, sums per-call router accounting into each case,
   and derives all global cycle totals. A two-call case reporting only one
   aggregate router opportunity is invalid even when mutable global totals are
   rewritten to agree.
5. Width/value breakdown, especially width 8 and secret width 10.
6. Router fire count, false-fire count, and responses that never reach the site.
7. Twelve frozen researcher-written direct interactions, preserving every raw
   generated call with the same token-boundary evidence and deriving all
   identities, responses, targets, predictions, success, and site/fire totals:
   four complete carry/borrow chains, two terminal transitions,
   two source-deleted continuations beginning from an interior state, two exact
   state-reuse replays, and two explicit review prompts that include the prior
   proposed state before continuing to a final answer.  Review prompts are not
   canonical router sites and therefore test review behavior without granting
   an extra motor intervention.
8. A non-DWS preservation set proving zero router fires and exact gate-off
   identity. Every base and treatment call retains its exact generated token-ID
   sequence and, at every generated boundary, the full-logit tensor dtype, shape,
   byte count, byte SHA-256, and identity SHA-256. The validator requires token
   IDs and all boundary logit identities to be exactly equal; decoded-text
   equality is only redundant. Distinct sequences such as `[818, 0]` and
   `[41, 41, 0]` fail even if both decode to `==`.

## 9. Preregistered decision

Treatment receives a **mechanism GO** only if the one-shot confirmation result
passes its fresh-row gates and its exact bound development report passes the
frozen cycle, direct, and preservation gates:

- next-carry accuracy is at least 95%, at least 15 percentage points above base,
  and at least 15 points above shuffled control;
- exact one-step state improves by at least 15 points over base and introduces
  no new pre-carry divergence class;
- autonomous full-episode exactness improves by at least 20 points overall and
  by at least 15 points at unseen widths, with no fit-width regression larger
  than 2 points;
- the frozen boundary cycle improves from 9/50 to at least 25/50 without oracle
  intervention;
- shuffled control does not meet the treatment thresholds;
- non-DWS false fires are zero and gate-off identity is exact;
- at least 8/12 fresh direct episodes have exact complete state traces and final
  answers.

If carry accuracy passes but full episodes do not, the motor is recorded as a
successful **writer/actuator repair only**.  The next experiment may then test a
separate rank-8 carry consumer at the active result-digit site, using matched
writer-only, reader-only, joint, and sham arms.  Larger FCRC remains dormant
unless the carry-only motor is negative or a writer success leaves a measured
consumer bottleneck.

## 10. Exact collapse and finite falsifiers

Before H100 use, CPU tests must prove:

- gate false and dead motor produce exact base logits;
- only the two carry-token logits can change at a true site;
- a canonical prefix fires once while malformed, prompt-side, and non-DWS
  suffixes never fire;
- same-shape shuffled labels preserve stratum counts;
- a synthetic hidden board with carry encoded in a known nonlinear rank-8 basis
  is learned by treatment but not shuffled control;
- removing carry information from that board collapses both learned arms to
  chance;
- frozen input hash, output no-replace, receipt, and router-accounting failures
  stop execution;
- plan, extract, and fit stop before CUDA or shard loading when the immutable
  pre-fit confirmation commitment is absent, malformed, source-mismatched, or
  rewritten after binding;
- the same secret and frozen inputs admit one board, caller-supplied exclusion
   arguments are impossible, and any deleted/reordered/recomputed exclusion
   identity fails semantic commitment validation before CUDA;
- an alternate 32-byte secret, a rewrite of frozen episode index 1499, or any
  confirmation row/digest rewrite fails against the commitment and plan; the
  public generator also rejects an alternate plan path/receipt or any supplied
  plan document that differs from its independently bound bytes;
- a skipped character inside a multi-character decoded token never becomes a
  router boundary, while any generated-token, decoded-prefix, decision-count,
  or stop-reason mutation fails;
- teacher-forced target/prediction/raw-maximum deletions and direct-call/site
  rewrites fail even when their mutable aggregate totals are also recomputed;
- fit-time teacher evidence rejects a deleted row field or tensor and recomputes
  treatment and shuffled logits from exact shard-replayed hidden/base tensors
  and fitted state; a hidden rewrite with all bundle-local hashes, logits, and
  summaries recomputed still fails while the sealed shard receipts are unchanged;
  replacing a learned row with `[120,-120]` and recomputing every prediction and
  summary still fails;
- a 576-dimensional batch-sensitive motor fixture makes a batched head choose
  token 0 and singleton calls choose token 1, and proves every canonical teacher
  decision uses only `(1, 576)` motor forwards;
- preservation rejects decoded-text aliases with unequal token IDs and any
  unequal per-boundary full-logit identity;
- mutating a selected development identity fails even if its mutable identity
  digest is recomputed;
- plan, shard, fit, development, and confirmation schemas preserve the exact
  checkpoint-step JSON string `"sft_ep1"`; an integer or alternate-string rewrite
  fails publication validation even if the report and caller expectation are
  changed together;
- rewriting an unsampled treatment episode's compact accounting and all regime
  totals fails when the retained raw calls do not support the rewrite;
- every cycle total is recomputed from all 50 per-call ledgers, including a
  falsifier for a two-call/one-site case claim;
- writable, symlinked, or multiply linked plans and symlinked or multiply
  linked shards fail before canonical binding or fit; an invalid shard 1 fails
  during the complete first pass before shard 0 can be bound or loaded; fit
  recovery and both downstream evaluators cannot validate a bundle without the
  same complete eight-shard replay.

Canonical launch requires an explicit reviewed Git commit.  Before importing
the experiment, the Slurm wrapper compares every scientific source, this
preregistration, the tests, and the wrapper itself byte-for-byte with that
commit and derives a stable source-manifest SHA-256.  The Python process then
snapshots those same files plus checkpoint, tokenizer, boards, cycle evidence,
and the sealed confirmation commitment into private immutable bytes before
consumption, verifies the wrapper manifest, and records commit plus manifest in
the motor bundle. Evaluation requires exact source-contract equality, bundle SHA-256, and treatment/shuffled
tensor hashes.  Outputs use a new one-purpose directory sealed mode `0555`
after read-only artifacts are fsynced.  This is process-local custody, not
remote attestation against a malicious same-UID actor that can alter process
memory or permissions.

No benchmark, reasoning, novelty, or workspace claim is authorized by a fit
loss or development score.
