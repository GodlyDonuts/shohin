# R12 causal carry motor preregistration

**Status:** frozen design draft; no H100 fit is authorized until the code, tests,
and batch job pass independent review.  This experiment is intentionally smaller
than FCRC and must run first.

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

The checkpoint is a 30-layer, 576-wide, 125.1M-parameter DRS model.  Every base
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
width-8 regimes.  Development results can reject a candidate but cannot alone
establish the claim.

### Confirmation

Before fitting, the code and confirmation-board generator hashes are frozen.
A Stokes custodian creates a 256-bit secret, stores it outside all trainer input
paths, and publishes only `SHA256(secret)`.  Trainers and fit jobs do not read
the secret.  After the treatment and shuffled checkpoints are immutable and
hash-recorded, the secret is revealed once and deterministically creates the
confirmation board.  No retry or alternate secret is permitted.  This is an
honest same-UID custody boundary, not cryptographic isolation from the account
owner.

The confirmation board contains fresh values at widths 4 and 6 plus unseen
widths 8 and 10, balanced add/sub operations, carry outcomes, positions, and
prompt styles.  It rejects exact fit/development prompt and operand-pair hashes.

## 8. Evaluations

All arms are evaluated with identical deterministic greedy decoding.

1. Teacher-forced next-carry accuracy, exact global-vocabulary top-one accuracy,
   and exact global rank on evaluation features.  Fit-time reports may omit rank
   to avoid retaining the full vocabulary matrix, but may not report a capped
   rank as global.
2. Exact one-step canonical state.
3. Autonomous full episode: every state exact, terminal answer exact, and first
   failure position.
4. The frozen 50-case boundary causal cycle, with no oracle token or residual.
5. Width/value breakdown, especially width 8 and secret width 10.
6. Router fire count, false-fire count, and responses that never reach the site.
7. Twelve frozen researcher-written direct interactions, preserving raw
   transcripts: four complete carry/borrow chains, two terminal transitions,
   two source-deleted continuations beginning from an interior state, two exact
   state-reuse replays, and two explicit review prompts that include the prior
   proposed state before continuing to a final answer.  Review prompts are not
   canonical router sites and therefore test review behavior without granting
   an extra motor intervention.
8. A non-DWS preservation set proving zero router fires and exact base token/logit
   identity when the motor is inactive.

## 9. Preregistered decision

Treatment receives a **mechanism GO** only if all are true on the one-shot
confirmation board:

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
  stop execution.

Canonical launch requires an explicit reviewed Git commit.  Before importing
the experiment, the Slurm wrapper compares every scientific source, this
preregistration, the tests, and the wrapper itself byte-for-byte with that
commit and derives a stable source-manifest SHA-256.  The Python process then
snapshots those same files plus checkpoint, tokenizer, boards, and cycle
evidence into private immutable bytes before consumption, verifies the wrapper
manifest, and records commit plus manifest in the motor bundle.  Evaluation
requires exact source-contract equality, bundle SHA-256, and treatment/shuffled
tensor hashes.  Outputs use a new one-purpose directory sealed mode `0555`
after read-only artifacts are fsynced.  This is process-local custody, not
remote attestation against a malicious same-UID actor that can alter process
memory or permissions.

No benchmark, reasoning, novelty, or workspace claim is authorized by a fit
loss or development score.
