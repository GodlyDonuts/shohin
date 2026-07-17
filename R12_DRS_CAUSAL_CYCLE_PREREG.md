# R12 DRS Causal Cycle Preregistration

**Status:** frozen read-only diagnostic; no architecture, SFT, or reasoning
claim is authorized by this document.

## Question and exact boundary

Post-DRS residual replacement moves held-out digit logits by about `+31` at
layers 17--29 and carry logits by about `+2.96` at layer 29. That proves a
local value-bearing residual exists. It does not identify which part of a
multi-step cycle fails: native production, state-line serialization, text
transport, or next-step response.

For the current inference API, each DRS call invokes `GPT.forward` from token
ids with no hidden-state argument from the preceding call. If `H_t` denotes all
hidden tensors in call `t`, `Y_t` its emitted text, and `P` the fixed canonical
state parser/renderer, then

```text
H_t -> Y_t -> P(Y_t) -> H_(t+1)
```

is the only cross-call causal path. Conditional on the forwarded token ids,
`H_t` has zero direct influence on `H_(t+1)`. A successful residual-authored
first state followed by a successful unpatched second call is therefore
positive end-to-end evidence. A failed second call is not, by itself, localized
consumer evidence because it also includes new residual production and state
serialization. The probe consequently scores teacher-forced second-call local
digit/carry predictions separately from full replay.

This is a decomposition theorem, not a new reasoning primitive.

## Frozen inputs and board

| Input | Frozen SHA-256 |
|---|---|
| `train/sft_digitwise_recurrent_v2_200k_r3/sft_ep1.pt` | `d79e9df26caecb9801118d1bf68bd7b85381a06b256f23478acffe40a2108459` |
| `artifacts/evals/digitwise_recurrent_v2_heldout.jsonl` | `89ce11b36ff2f56e83cda72a1f07b1a90f4a3dc3803c69db2779a27219712646` |
| `artifacts/shohin-tok-32k.json` | `87532df5c121753de3b29194e1f9e3de47986d3f5359548fdf93606773a233d4` |

The claim-bearing scientific source closure is also frozen:

| Source | Frozen SHA-256 |
|---|---|
| `train/probe_drs_causal_cycle.py` | `af1551723b9b056fbdb66f7fe1dece5819c7d6dcf887608299373ceff1a772e5` |
| `train/digitwise_protocol.py` | `708489d61c212c402e1533a1483e77bf3fd2d1a057ce924321bb19e4888461f6` |
| `train/eval_suite.py` | `d6f70b8828c967d7f59fae842f3320c6378ae42d5d8fa7b16e0e82ff5620e5e6` |
| `train/model.py` | `45fc0dc46ceb0f91d08e3f671cbe9ef202ea212e72d5bba8b77356c3fb0983d4` |
| `train/probe_digitwise_workspace.py` | `fb545450a93bbc04aac1549efd0a70b863f50e458fd95993cf9935bbe4a53ace` |
| `train/jobs/probe_drs_causal_cycle.sbatch` | `7fc09fa925fb725838ddbb3efac6f61d401cf7d02e07f1a64f2f0ce216d4122a` |

The batch job copies exactly those five sources and all three frozen inputs to
a fresh job-private snapshot, makes the snapshot read-only, verifies every
hash there, and executes only the copied probe with the copied inputs. The
canonical report records the copied scientific-source hashes, Slurm job ID,
and snapshot identity. The snapshot is deleted after the process exits. This
closes hash-then-reopen drift inside the ordinary trusted Slurm/filesystem
boundary; it is not remote attestation against a malicious same-UID process.

Use transition index 2 only. For each of the five frozen regimes, select the
first ten episode IDs satisfying

```text
apply_microstep(s).carry != apply_microstep(flip_carry(s)).carry.
```

The resulting 50 records are true decimal-boundary cases: the first altered
state changes the carry consumed by the following local computation. Each
record also receives an independent same-regime donor matched on operation,
width, position, input carry, next carry, and next digit.

The claim-bearing run is fixed at layer 29, greedy BF16-autocast cached decoding,
96 new tokens, and fresh output
`artifacts/evals/drs_causal_cycle_post_drs_r2.json`. Input hashes are checked
before model loading. The job accepts no scientific override. Output uses
exclusive same-directory publication and refuses existing files.
Canonical mode additionally requires CUDA, the exact output path, a Slurm job
ID, and the verified private snapshot. Every noncanonical invocation must use
the explicit development-only flag; it reports all scientific decisions as
`null`, even if its raw mechanics resemble a pass.

## Frozen interventions

All generation uses the same KV-cached stopping contract as the public greedy
evaluator. A residual hook applies only to the final current token at an exact
token-prefix match. At layer 29 it can affect the next emitted token but cannot
enter a later transformer block or survive as hidden state.

For each record run:

1. **baseline:** no intervention;
2. **identity:** base residuals replace themselves at carry and digit sites;
3. **same-target rescue:** independent same-label donor residuals at both sites;
4. **carry only:** counterfactual carry residual, scored against a carry-only
   hybrid state;
5. **digit only:** counterfactual digit residual, scored against a digit-only
   hybrid state;
6. **both:** counterfactual carry and digit residuals, scored against the full
   counterfactual next state;
7. **token ceiling:** directly force only those two target tokens, then let the
   model serialize everything else;
8. **irrelevant sham:** transplant carry/digit residuals from a state differing
   only in an already-written result digit, while the active transition target
   is unchanged.

Every arm records requested-site reach separately by arm and field. Low reach
is a behavioral pre-field serialization outcome, not mechanical invalidity.
The identity control additionally compares full generated token IDs and exact
teacher-forced logits.
The irrelevant sham is scored causally in the base prefix: patched digit/carry
argmax must remain invariant, and the sham-generated token sequence is compared
directly with baseline. Unpatched donor-context invariance remains diagnostic
only and cannot satisfy the sham gate.

## Independent second-call tests

Regardless of first-arm success, run unpatched calls from both the reachable
base next state and the canonical carry-counterfactual next state. Score:

- teacher-forced next digit and carry argmax;
- paired active-digit switching under the changed carried bit;
- whole successor-state greedy exactness;
- the integrated two-call cycle, requiring the residual-authored first state
  and the unpatched counterfactual successor both to be exact.

For every parsed first state, canonicalize and hash the actual next-prompt token
ids and the intended next-prompt token ids separately. Exact equality is the
transport endpoint; no hidden state is forwarded.

## Mechanical validity

The report is invalid unless all conditions hold:

- exactly 50 records, ten from each named regime;
- every selected record satisfies the carry-boundary predicate;
- checkpoint, heldout, and tokenizer hashes equal the frozen values;
- every teacher-forced identity patch has zero max absolute logit delta;
- every identity generated token sequence equals baseline exactly;
- the output path was exclusively published.
- execution is canonical CUDA BF16 from the verified private snapshot and the
  exact frozen output/configuration;
- all five scientific source hashes equal the table above.

Site reach, parse rate, and all capability scores are outcomes rather than
validity gates.

## Frozen decision thresholds

| Endpoint | Threshold / diagnosis |
|---|---|
| counterfactual both-site first state | `>=50%` balanced aggregate and `>=30%` in every regime is write/serialization pass |
| direct two-token ceiling first state | `>=80%` aggregate and `>=70%` in every regime is non-field serialization pass |
| paired unpatched second-call active-digit switch | `>=70%` aggregate and `>=50%` in every regime, with all base/counterfactual carry/digit fields `>=70%` aggregate and `>=50%` per regime, is local next-step response pass |
| irrelevant residual sham | transplanted carry+digit argmax invariance and sham-token equality are each `>=90%` aggregate and `>=80%` in every regime |
| same-target rescue among baseline failures | `>=50%` and `>=+20pp` overall signals weak native residual production |

Every primary endpoint is emitted both on the balanced 50-case aggregate and
separately for each ten-case regime. No aggregate pass can hide a zeroed regime.
Same-target rescue remains aggregate-only because a regime may have zero
baseline failures; its per-regime numerator and nullable denominator are still
reported.

Interpretation is conditional. A high token ceiling with low residual-authored
exactness identifies the residual-to-token interface. High first-state and
paired second-call scores with weak baseline rollout identify native state
production/control as the remaining DRS bottleneck. A low teacher-forced
second-call score means next-step response is already broken before whole-state
serialization; a high teacher-forced score but low full successor exactness
localizes that branch to serialization.

## Equivalence and claim dossier

The probe supplies oracle residuals or oracle target tokens and is therefore an
external causal intervention, not autonomous reasoning. It adds no parameters,
persistent hidden bits, training examples, learned updater, or executor. It
cannot establish an advantage over SFT, recurrence, retrieval, fast weights,
hard registers, external execution, or finite unrolling. Its only admissible
claim is localization within the existing DRS text-mediated cycle.

Any architecture experiment selected by this result requires its own
resource-vector comparison, exact collapse test, finite falsifier, and matched
controls.
