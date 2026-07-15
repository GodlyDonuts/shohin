# R12 Raw-260k Updater Candidate-Likelihood Preregistration

**Status:** implementation frozen; diagnostic not run. This document does not
authorize a model launch, checkpoint modification, generation pass, training
job, or commit.

## 1. Question and claim boundary

The existing raw-260k updater transcript fails all six joint state-and-tail
updates under free decoding. This diagnostic asks one narrower question:

> When exact updater alternatives are teacher-forced, is the correct residual
> state/packet the model's preferred content, with failure attributable to
> sequence decoding or immediate termination, or is the correct update itself
> not preferred?

The word `latent` in the machine-readable diagnoses means only **unique top-1
normalized likelihood among the five fixed candidate strings**. It does not
mean a decoded hidden state, a causal neural mechanism, autonomous recurrence,
or free-running ability. The candidates are evaluator-authored. This is a
six-prompt development diagnostic, not a benchmark or training gate.

## 2. Immutable bindings

| Object | Required identity |
|---|---|
| Prompt source | `artifacts/eval_history/raw260k_updater_subskill_probe_20260715_mps.json` |
| Prompt source SHA-256 | `4505602994a0e337b99359e580a6f2f04fad4d365b2dac59f4c339fac13a7593` |
| Prompt source schema | `raw260k_updater_subskill_probe_v1` |
| Prompt rows / source calls | `12 / 12` |
| Checkpoint step | `260000` |
| Checkpoint SHA-256 | `91d5288f184fc5230516add9851ac1a8815d3369ffd816cd7d0c03d8bafc741d` |
| Tokenizer SHA-256 | `87532df5c121753de3b29194e1f9e3de47986d3f5359548fdf93606773a233d4` |
| EOS token / ID | `<|endoftext|>` / `0` |
| Candidate manifest SHA-256 | `0e01fc54abfe63dcfd063fa6d5a1e4ed46b57aef617580d2cc286db839b3ba98` |
| Tokenized manifest SHA-256 | `13528bacb21d8bca006b434283830d9a0790225dd9a7ddcdcbc8f02e7bac8a99` |

The prompt source must be a regular, non-symlink file with no write bits. The
reader uses `O_NOFOLLOW` where available, verifies the complete file SHA-256,
rejects duplicate JSON keys and non-finite JSON, then checks every binding,
row ID, row kind, prompt byte string, original prompt-token count, and the
absence of prompt truncation. A copied or self-rehashed replacement is not
accepted.

The checkpoint and tokenizer paths may differ across machines, but their bytes
must match the hashes above. The loaded checkpoint must still report step
`260000` and context length `2048`.

## 3. Frozen source prompts

All twelve source prompts are validated in this order. Newline escapes below
denote one LF byte; there is no trailing newline after `Work:` or
`Next packet:`.

```text
copy_25  [copy_state, 18]  "Problem: The observed result is 25. Write the new state.\nWork:"
copy_75  [copy_state, 18]  "Problem: The observed result is 75. Write the new state.\nWork:"
copy_64  [copy_state, 18]  "Problem: The observed result is 64. Write the new state.\nWork:"
delete_a [delete_head, 33] "Problem: The plan is add 6; multiply 3; subtract 11. The first operation is complete. Write only the remaining operations.\nWork:"
delete_b [delete_head, 33] "Problem: The plan is multiply 4; subtract 13; add 8. The first operation is complete. Write only the remaining operations.\nWork:"
delete_c [delete_head, 32] "Problem: The plan is subtract 5; add 9; multiply 2. The first operation is complete. Write only the remaining operations.\nWork:"
joint_a  [joint_natural, 37] "Problem: The plan was add 6; multiply 3; subtract 11. The first operation produced 25. Write the new state and remaining plan.\nWork:"
joint_b  [joint_natural, 38] "Problem: The plan was multiply 4; subtract 13; add 8. The first operation produced 108. Write the new state and remaining plan.\nWork:"
joint_c  [joint_natural, 36] "Problem: The plan was subtract 5; add 9; multiply 2. The first operation produced 37. Write the new state and remaining plan.\nWork:"
packet_a [joint_packet, 35] "Packet:\nState: 19\nPlan: add 6; multiply 3; subtract 11\nObserved result: 25\nNext packet:"
packet_b [joint_packet, 36] "Packet:\nState: 27\nPlan: multiply 4; subtract 13; add 8\nObserved result: 108\nNext packet:"
packet_c [joint_packet, 34] "Packet:\nState: 42\nPlan: subtract 5; add 9; multiply 2\nObserved result: 37\nNext packet:"
```

Only `joint_a`, `joint_b`, `joint_c`, `packet_a`, `packet_b`, and `packet_c`
are scored. The copy/delete rows remain mandatory custody evidence: deleting,
reordering, or changing any of the twelve rows invalidates the source before a
model can load.

## 4. Exact candidate bytes

Every scored prompt has exactly these classes, in this order:

1. `correct_residual_packet_state_tail`
2. `unchanged_source_packet`
3. `consumed_head_replay`
4. `arithmetic_execution_continuation`
5. `reordered_tail`

`correct_residual_packet_state_tail` copies the observation into `State` and
deletes exactly the consumed plan head. `unchanged_source_packet` preserves the
old state and full source plan. `consumed_head_replay` repeats only the already
completed arithmetic transition. `arithmetic_execution_continuation` executes
the residual tail instead of serializing it. `reordered_tail` has the right
observed state and tail elements in the wrong order.

The same five exact strings are used for each natural/packet pair. Every string
starts with exactly one LF and has no trailing whitespace or EOS byte.

```text
joint_a and packet_a
correct_residual_packet_state_tail = "\nState: 25\nPlan: multiply 3; subtract 11"
unchanged_source_packet             = "\nState: 19\nPlan: add 6; multiply 3; subtract 11"
consumed_head_replay                = "\n19 + 6 = 25"
arithmetic_execution_continuation   = "\n25 * 3 = 75\n75 - 11 = 64"
reordered_tail                      = "\nState: 25\nPlan: subtract 11; multiply 3"

joint_b and packet_b
correct_residual_packet_state_tail = "\nState: 108\nPlan: subtract 13; add 8"
unchanged_source_packet             = "\nState: 27\nPlan: multiply 4; subtract 13; add 8"
consumed_head_replay                = "\n27 * 4 = 108"
arithmetic_execution_continuation   = "\n108 - 13 = 95\n95 + 8 = 103"
reordered_tail                      = "\nState: 108\nPlan: add 8; subtract 13"

joint_c and packet_c
correct_residual_packet_state_tail = "\nState: 37\nPlan: add 9; multiply 2"
unchanged_source_packet             = "\nState: 42\nPlan: subtract 5; add 9; multiply 2"
consumed_head_replay                = "\n42 - 5 = 37"
arithmetic_execution_continuation   = "\n37 + 9 = 46\n46 * 2 = 92"
reordered_tail                      = "\nState: 37\nPlan: multiply 2; add 9"
```

The implementation stores these as literal tuples. It does not call an
arithmetic solver, packet renderer, parser, model, or source response to create
or alter a candidate.

## 5. Pre-model freeze order

The executable must complete this sequence before `torch.load`:

1. Refuse an existing or symlink output path.
2. Read and verify the immutable twelve-row source artifact.
3. Canonicalize the literal candidate manifest and require SHA-256
   `0e01fc54abfe63dcfd063fa6d5a1e4ed46b57aef617580d2cc286db839b3ba98`.
4. Hash the complete checkpoint and tokenizer files against Section 2.
5. Load only the bound tokenizer.
6. Tokenize each exact `prompt + candidate` concatenation. The independently
   tokenized prompt must be an exact token prefix; boundary retokenization is a
   hard failure. No candidate may be empty or exceed context length.
7. Require EOS ID `0` and tokenized-manifest SHA-256
   `13528bacb21d8bca006b434283830d9a0790225dd9a7ddcdcbc8f02e7bac8a99`.
8. Preflight the requested accelerator, then and only then load the model.

The tokenized manifest fixes 518 candidate tokens. The six prompt lengths are
`37, 38, 36, 35, 36, 34`; replayed across five candidates they contribute
1,080 prompt positions.

## 6. Teacher-forced scores

For prompt tokens `P = (p_1, ..., p_m)`, candidate tokens
`C = (c_1, ..., c_n)`, and EOS token `e`, one forward receives `P || C`.
No BOS, generated token, separator, or automatic space is inserted. Candidate
strings already contain their leading LF.

The gathered candidate log likelihoods are:

```text
l_i = log p(c_i | P, c_1, ..., c_(i-1))
L_total = sum_i l_i
L_normalized = L_total / n
L_eos = log p(e | P, C)
L_complete = L_total + L_eos
```

The result stores candidate token IDs, every `l_i`, `L_total`,
`L_normalized`, `L_eos`, `L_complete`, and the complete-sequence normalized
score. At the EOS position it additionally stores EOS vocabulary rank, whether
EOS is the unique top-1 token, the top token ID/log likelihood, and the EOS
margin to top-1. Ties are not unique wins.

All scores use inference-mode float32 logits on either MPS or CUDA. There is no
CPU fallback and no mixed-precision autocast. `--device auto` chooses CUDA
first, then MPS; an unavailable or failed accelerator allocation aborts rather
than silently changing device. Device identity is recorded because small
backend numerical differences are possible.

## 7. Locked decision rule

`L_normalized` is the primary fixed-content preference metric. `L_total`
separately exposes sequence-length/decoding pressure. EOS is a termination
test, and `L_complete` is the exact candidate-plus-EOS sequence score.

For each prompt, apply this hierarchy without thresholds or post-hoc changes:

1. If correct is not the unique `L_normalized` winner, report
   `correct_update_not_preferred`.
2. Otherwise, if correct is not the unique `L_total` winner, report
   `correct_update_latent_but_loses_sequence_decoding`.
3. Otherwise, if EOS is not unique top-1 immediately after the correct
   candidate, report `correct_update_latent_but_loses_termination`.
4. Otherwise, if correct is not the unique `L_complete` winner, report
   `correct_update_latent_but_loses_joint_eos_likelihood`.
5. Otherwise report `correct_update_and_termination_preferred`.

If any of six prompts is `correct_update_not_preferred`, the aggregate is
`correct_update_not_consistently_preferred`. If all six pass every condition,
it is `correct_update_and_termination_preferred`. Every other all-content-win
outcome is
`correct_update_likelihood_preferred_but_decoding_or_termination_loses`.

This hierarchy prevents a favorable EOS score from rescuing absent content
preference and prevents normalized likelihood alone from being described as a
free-decoding success.

## 8. Exact resource ledger

| Resource | Frozen value |
|---|---:|
| Source prompt rows read | 12 |
| Scored prompt rows | 6 |
| Candidate classes per prompt | 5 |
| Candidate sequence evaluations | 30 |
| Model forward calls | 30 |
| Prompt tokens replayed | 1,080 |
| Supervised candidate tokens | 518 |
| Supervised EOS tokens | 30 |
| Teacher-forced target tokens | 548 |
| Forward token positions | 1,598 |
| Checkpoint hash passes / model loads | 1 / 1 |
| Tokenizer hash passes / tokenizer loads | 1 / 1 |
| Accelerator preflight allocations | 1 |
| Generated / sampled / training tokens | 0 / 0 / 0 |
| Retries / repairs | 0 / 0 |
| Candidate searches / threshold searches | 0 / 0 |
| Verifier feedback / external generation calls | 0 / 0 |

Each candidate is one independent forward. There is no KV state carried
between candidates or prompts, batching with shared padding, generation,
beam search, reranking, adaptive candidate edit, retry, parser feedback, or
score-conditioned exclusion.

## 9. Output custody

The only output schema is `raw260k_updater_candidate_likelihood_v1`. It binds
the source, candidate manifest, tokenized manifest, checkpoint, tokenizer,
step, EOS, and device; records all per-token scores and diagnoses; and includes
the complete resource ledger.

Output creation uses `O_EXCL` and `O_NOFOLLOW` where available, refuses
overwrite, writes ASCII JSON with non-finite values forbidden, flushes and
`fsync`s the descriptor, changes mode to `0444`, verifies no write bits remain,
and attempts a parent-directory `fsync`. A partial or pre-existing path is not
accepted as a result.

## 10. Tests and execution boundary

`train/test_probe_updater_candidate_likelihood.py` contains no checkpoint load.
Its toy scorer forces and verifies the distinct not-preferred,
sequence-decoding, termination, joint-EOS, and full-preference branches. It
also verifies the 30-call zero-generation ledger, literal candidate contract,
manifest hash, real source binding, byte-tamper rejection, metadata-binding
rejection, file-hash rejection, and exclusive read-only JSON creation.

The implementation supports a future explicitly authorized invocation through
`train/probe_updater_candidate_likelihood.py` with exact `--ckpt`,
`--tokenizer`, `--out`, and optional `--device auto|mps|cuda` arguments. This
preregistration freezes that invocation surface but records no run or score.
