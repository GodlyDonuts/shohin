# R12 Raw-260k Operation-Selection Likelihood Preregistration

**Status:** `FROZEN / UNRUN / ADAPTIVE EXPLORATORY DECOMPOSITION`

**Freeze date:** 2026-07-15 America/New_York

**Execution boundary at freeze:** the immutable source board and tokenizer were
read on CPU to freeze source geometry, prompts, token boundaries, hashes, and
resource counts. The Shohin checkpoint was not loaded into a model, no Shohin
forward pass was made, no existing GPU result was opened, and no job was
submitted.

## 1. Question and claim boundary

The already-frozen operation-cursor diagnostic asks the model to freely decode
strict JSON containing an operation and operand, and in one arm a next state.
That interface mixes operation selection with JSON compliance, operand copying,
multi-token decoding, EOS behavior, and decode-stop behavior.

This diagnostic asks only:

> At each frozen operation cursor, which of four exact one-token operation
> candidates has the largest next-token logit under each of three exposures?

The three exposures are:

1. the full natural-language source and an explicit zero-based cursor;
2. the oracle residual suffix whose head is being selected; and
3. that same residual suffix plus the oracle current numeric state.

This is a restricted next-operation preference diagnostic. It does not test
operand selection, arithmetic execution, state update, residual deletion,
free decoding, termination, autonomous recurrence, or general reasoning. A
correct unique top-1 result means only that the gold operation has the largest
logit among the four fixed candidates in that prompt. It does not authorize
training, promotion, a production submission, or a bottleneck claim.

This is not held-out confirmation. Its interface was designed after observing
the earlier free-decoding cursor failure on the same raw-260k model and source
board. The positional row subset remains score-blind, but the diagnostic design
is adaptive. Only `full_source_cursor` can measure source-conditioned operation
preference. Both residual-suffix arms expose the gold operation literally as the
first JSON label and are label-copy controls only.

## 2. Immutable bindings

| Object | Frozen identity |
|---|---|
| Source board | `artifacts/evals/source_scheduled_reasoning_confirmation_v1.json` |
| Source artifact SHA-256 | `19a84165f15b19911fc8ef229022e47753833d703d77d1e8cc25db9dfc993474` |
| Source rows SHA-256 | `4afc6c4b0c271ea2f723078ab183e8d1ac1851fd1728898384ef52275887b0e4` |
| Score-blind subset rows SHA-256 | `c48ad18103b7971e7cd3c29be172ed40baccaa10d5d255011a22d3c023dc17e6` |
| Raw-260k checkpoint SHA-256 | `91d5288f184fc5230516add9851ac1a8815d3369ffd816cd7d0c03d8bafc741d` |
| Required checkpoint step | `260000` |
| Required context length | `2048` |
| Required architecture fields | `n_layer=30`, `d_model=576`, `n_loop=1` |
| Tokenizer | `artifacts/shohin-tok-32k.json` |
| Tokenizer SHA-256 | `87532df5c121753de3b29194e1f9e3de47986d3f5359548fdf93606773a233d4` |
| Candidate manifest SHA-256 | `1e25e46807928f7ca9af1a3ea601181513e9cfe677e21f08632b295a62a75b89` |
| Expanded prompt manifest SHA-256 | `0a85a49f6ba818d51f6b74a48129f48e9e9d6bcd02dde152c31d626c68a472d6` |
| Tokenized prompt manifest SHA-256 | `2e910eb13ad2200cee82d713174f43c95d5c8a8544b9089b5b6de2fff976aaa6` |

The code freeze commit is supplied as `FROZEN_COMMIT`. A later evidence commit
may add only the canonical pre-score receipt; its full SHA-1 and the receipt
SHA-256 are late-bound execution inputs because a commit cannot contain its own
identity. Both must exist in the read-only bundle before scoring, and both are
embedded in the result. They cannot change prompts, candidates, code, or scores.

The source reader uses `O_NOFOLLOW` where available, requires a regular file
with no write bits, rejects duplicate JSON keys and non-finite constants, and
requires the complete artifact hash before any model load. It independently
parses every natural-language question, checks the source schedule, replays all
704 operations, and requires the frozen answer and stratum metadata.

The checkpoint and tokenizer may live at different absolute paths, but their
bytes must match the hashes above. The checkpoint metadata must also match the
step, context, and architecture bindings. There is no CPU or non-H100 fallback
for a canonical run.

## 3. Frozen source geometry and subset

Subset selection is inherited unchanged from
`R12_OPERATION_CURSOR_DIAGNOSTIC.md`: within each immutable 64-row family
block, take positional rows `000` through `015`, preserving family and row
order. No model score, response, or output influences selection.

| Family | Cases | Transitions |
|---|---:|---:|
| `multiply_subtract` | 16 | 32 |
| `base_conversion` | 16 | 64 |
| `sequential_state` | 16 | 48 |
| `modular_update` | 16 | 32 |
| **Total** | **64** | **176** |

The frozen denominator by zero-based schedule index is:

| Index | Transitions |
|---:|---:|
| `0` | 64 |
| `1` | 64 |
| `2` | 32 |
| `3` | 16 |

The frozen denominator by gold operation is:

| Operation | Transitions |
|---|---:|
| `add` | 64 |
| `subtract` | 32 |
| `multiply` | 64 |
| `remainder` | 16 |

## 4. One-token candidate contract

Each prompt ends immediately after the ASCII word `is`. Candidate text includes
exactly one leading space and no trailing byte. Under the bound tokenizer each
candidate is exactly one token, both alone and at every one of the 528 prompt
boundaries.

| Candidate order | Operation | Exact candidate text | Token ID |
|---:|---|---|---:|
| 1 | `add` | ` add` | `820` |
| 2 | `subtract` | ` subtract` | `5498` |
| 3 | `multiply` | ` multiply` | `4307` |
| 4 | `remainder` | ` remainder` | `7486` |

All four operation names appear once in the common candidate line of every
prompt. Candidate tokens are validation targets only: none is appended to the
model input, and all four logits come from the same final prompt position in a
single forward.

## 5. Frozen prompts

All text is ASCII. Newlines shown below are one LF byte. Prompts have no trailing
newline or trailing space after the final `is`.

### A. `full_source_cursor`

```text
Task: Select the operation at the supplied cursor.
Source: {complete source question}
Step index (zero-based): {index}
Candidate operations: add, subtract, multiply, remainder.
The operation at that cursor is
```

Exposed: the complete natural-language question and explicit zero-based cursor.
Not exposed: source schedule fields, current intermediate state, residual
suffix, final answer, model output, score, or verifier feedback.

### B. `residual_suffix_head`

```text
Task: Select the first operation in the supplied residual suffix.
Residual suffix (read-only JSON): {compact oracle suffix}
Candidate operations: add, subtract, multiply, remainder.
The residual head operation is
```

The suffix uses compact JSON such as
`[["multiply",5],["subtract",7]]`. Exposed: the suffix beginning at the
tested transition. Not exposed: the full question, cursor, current state,
final answer, model output, score, or verifier feedback.

### C. `residual_suffix_oracle_state`

```text
Task: Select the first operation in the supplied residual suffix.
Current state (oracle-supplied for this arm): {integer state}
Residual suffix (read-only JSON): {compact oracle suffix}
Candidate operations: add, subtract, multiply, remainder.
The residual head operation is
```

Exposed: the same oracle suffix and the independently replayed oracle current
state. This arm still scores only the operation; it does not ask for or score
an operand or next state.

Calls are stateless. No response or score from one prompt enters any later
prompt. The expanded prompt manifest hashes all 176 transitions, gold labels,
states, residual suffixes, and 528 exact prompt strings. The tokenized manifest
additionally hashes every prompt token ID and every verified one-token candidate
suffix at every boundary.

## 6. Teacher-forced scoring and accuracy

For prompt tokens `P`, candidate operation `o`, fixed candidate token ID `t_o`,
and final-position vocabulary logits `z(P)`, one inference-only forward computes:

```text
candidate_logit(o) = z(P)[t_o]
restricted_log_probability(o) =
    candidate_logit(o) - logsumexp(candidate_logit(c) for c in four candidates)
```

Only the four gathered candidate logits and their four-way normalization are
preserved. Full-vocabulary rank and top token are not reported. The prediction
is the operation with the unique largest candidate logit. An exact tie has no
prediction and is incorrect. There is no threshold, calibrated cutoff,
candidate reranking, retry, or score-conditioned branch.

Every arm record stores the exact prompt and prompt token IDs, all four
candidate texts/IDs/logits/restricted log probabilities/probabilities, the gold
margin to the best incorrect candidate, the top set, prediction, tie status,
and correctness.

Accuracy is emitted only as integer numerator/denominator objects. Exact
accuracy and tie counts are recomputed:

- globally;
- for each source family;
- for each zero-based step index; and
- for each gold operation.

The exclusive read-only result contains no floating summary percentage, p-value,
confidence interval, pass threshold, inferred bottleneck, or promotion gate.

## 7. Frozen resource ledger

| Resource | Exact value |
|---|---:|
| Unique full source rows / transitions audited per reconstruction | 256 / 704 |
| Selected cases / transitions | 64 / 176 |
| Arms per transition | 3 |
| Prompts scored / model forwards | 528 / 528 |
| Candidate classes per prompt | 4 |
| Candidate logit values / teacher-forced alternatives scored | 2,112 / 2,112 |
| `full_source_cursor` prompt tokens | 10,868 |
| `residual_suffix_head` prompt tokens | 9,731 |
| `residual_suffix_oracle_state` prompt tokens | 12,561 |
| Total model-input token positions | 33,160 |
| Maximum prompt length | 79 |
| Candidate tokens appended to model input | 0 |
| Checkpoint / tokenizer / source hash passes | 4 / 4 / 4 |
| Implementation hash passes | 5 |
| Tokenizer loads / model loads | 2 / 1 |
| H100 preflight allocations | 1 |
| Generated / sampled / training tokens | 0 / 0 / 0 |
| Retries / repairs / searches / threshold searches | 0 / 0 / 0 / 0 |
| Verifier feedback / external generation calls | 0 / 0 |
| Quarantine result files created | 1 |
| Preserved result copies / read-only receipts | 2 / 1 |
| Mutable sidecars | 0 |
| Authenticated pre-score remote verifications | 1 |
| Read-only Git bundles | 1 |
| Temporary bare Git repositories | 1 |
| Temporary pre-score receipt files | 1 |
| Mutable scheduler log files | 1 |

There is no batching, padding, KV reuse, candidate-specific forward, candidate
token input, generation API, sampling API, optimizer, backward pass, training
data write, retry loop, or downstream submission.

## 8. Pre-model freeze and post-model custody

The canonical wrapper must first receive a full `FROZEN_COMMIT` that was pushed
before execution. Because the GitHub repository is private and no credential may
be copied to a compute node, the pre-score controller first verifies the commit
with authenticated `gh api` and an exact `origin/main` match, writes a
score-unobserved receipt, commits and pushes that receipt, then creates a
read-only Git bundle containing both commits. The wrapper verifies the bundle's
object graph, requires bundle `main` to equal the supplied evidence commit,
requires `FROZEN_COMMIT` to be its ancestor, reconstructs and validates the
pre-score receipt, and requires every deployed implementation file's bytes to
equal the corresponding frozen Git object. This does not require credentials or
modify Git metadata in the deployed Newton tree. A runtime hash without this
commit/receipt/bundle chain is inadmissible. The result binds `FROZEN_COMMIT`,
the evidence commit, and the pre-score receipt SHA-256.

The executable must then complete this order before `torch.load`:

1. reject an existing, symlinked, or wrongly named output;
2. load and fully audit the read-only source artifact;
3. reconstruct the positional subset and all 176 transition states/suffixes;
4. verify candidate and expanded prompt manifest hashes;
5. read the checkpoint and tokenizer once through no-follow regular-file
   descriptors, hash those byte snapshots, and later load only those same
   in-memory bytes;
6. snapshot and hash the preregistration, evaluator, tests, wrapper, model
   loader, inherited cursor contract, and inherited cursor geometry
   implementation; compile and execute `model.py` only from that in-memory byte
   snapshot, never through import lookup or reusable bytecode;
7. load only the bound tokenizer, verify all candidate IDs and all 2,112
   one-token prompt boundaries, then verify the tokenized manifest and counts;
8. require exactly one visible CUDA device, allocate one preflight tensor, and
   require that its device name contains `H100`; and
9. load the hash-bound checkpoint once, verify metadata, disable parameter
   gradients, and enter evaluation mode.

After all 528 forwards, the evaluator hashes the checkpoint, tokenizer, source,
and every implementation file again. Any path-level change aborts before
output, while the already-loaded checkpoint and tokenizer remain the exact
pre-model byte snapshots. Candidate logits are gathered as float32 without
autocast.

The evaluator first creates one private quarantine
`raw260k_operation_selection_likelihood_*.json` under a mode-`0700` temporary
directory. Creation uses `O_EXCL` and `O_NOFOLLOW` where available, refuses
overwrite, serializes ASCII JSON with non-finite values forbidden, flushes and
`fsync`s, changes mode to `0444`, verifies no write bits remain, and attempts a
parent-directory `fsync`. The evaluator prints no score or summary. The wrapper
then runs the no-model full preserved-result reconstruction against the
quarantined artifact. Only after that audit passes does it atomically hard-link
the artifact into the final result path, remove the quarantine name, create an
exclusive hash-verified copy outside the result directory, and create an
exclusive read-only receipt binding the result, mirror, frozen commit, and
Slurm job. The wrapper still prints no score. Mode `0444` is write protection,
not absolute immutability. Canonical interpretation additionally requires the
receipt to be mirrored locally and pushed before any score is read. The mutable
Slurm log is counted separately and has no evidentiary standing by itself.

## 9. Isolated H100 wrapper

`train/jobs/probe_operation_selection_likelihood.sbatch` requests exactly one
node, one task, one `nvidia_h100_pcie` GPU, four CPUs, 64 GiB RAM, and two hours
on `normal`. It excludes `evc34`. It requires an explicitly supplied fresh
output path directly under `artifacts/eval_history`, checks all three immutable
input hashes, verifies the read-only pre-score Git bundle and receipt, verifies
every implementation byte against `FROZEN_COMMIT`, invokes only this evaluator into a private quarantine,
invokes the evaluator's no-model full preserved-result audit mode, and only then
publishes the result, mirror, and receipt. It contains no `sbatch`, training,
retry, score-conditioned branch, or downstream command and is not submitted by
this change.

## 10. CPU verification boundary

From `train/`, the focused test command is:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest -v test_probe_operation_selection_likelihood.py
```

The tests do not load the checkpoint or instantiate Shohin. They validate the
real hash-bound source and tokenizer, all geometry and manifest hashes, every
prompt/candidate boundary, exact resource accounting, one-forward/four-logit
scoring with unique/wrong/tied synthetic logits, grouped accuracy denominators,
result mutation rejection, checkpoint metadata rejection, exclusive read-only
output, exact-byte source compilation without import caches, and the
non-training/quarantine/mirror/receipt H100 wrapper contract.
