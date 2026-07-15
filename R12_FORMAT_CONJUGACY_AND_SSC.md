# R12 Format-Conjugacy No-Go and Source-Scheduled Continuation Diagnostic

**Status:** unlabeled format-only training rejected as a reasoning mechanism.
Source-Scheduled Continuation (SSC) is admitted only as a counted diagnostic of
selector/halting failure, not as a new primitive or promotion candidate.

## 1. Evidence boundary

Raw 260k continuation confirmation finds 8/20 strict first-segment answers
under two solved worked examples, 4/20 under direct QA, and 1/20 under bare
expressions. The four discordant worked wins and zero direct wins give two-sided
exact McNemar `p=0.125`. More importantly, worked prompts contain target-
relevant solved states and answers, so they are not an unlabeled change of
format.

Sequential add/multiply/subtract is the only robust family: 5/5 worked and 4/5
direct. Correct intermediates are present in 12/20 direct and 12/20 worked
responses even though finals are less reliable. Base conversion is 0/5 in
every mode. This supports a narrow selector/halting diagnosis, not a general
reasoning claim.

## 2. Unlabeled format-orbit no-go

Consider any learner that receives only fixed base weights, unlabeled problems,
known format maps, model output distributions, and target-independent
randomness. It receives no answer, verifier result, target-dependent reward, or
privileged state.

Two semantic target worlds can share the identical unlabeled problems and
format orbits while assigning complementary binary answers. The learner sees
the same transcript and therefore emits the same model in both worlds; its
errors across the two worlds sum to one. Thus format orbits alone cannot
guarantee correctness or strict improvement. They can only reorganize
information already present in the weights.

Agreement losses reduce to prompt consistency regularization. Transported
same-model targets are self-distillation. Model-authored chains are
self-training/CoT SFT. Voting is prompt ensembling/self-consistency. Hard format
maps are canonicalization/equivariant sharing. None is a new source of target
information.

## 3. SSC diagnostic

Factor a generated procedural case into a public operation schedule and hidden
numeric states:

```
selector: choose the next requested operation or STOP
executor: predict U_operation(current_state).
```

SSC deterministically copies only the operation schedule from the structured
source. It asks the frozen model for one next state per operation, parses one
integer, and carries that model-produced integer forward. It supplies no
intermediate state value, final answer, verifier feedback, repair, search, or
retry.

If forced transition `t` has error at most `epsilon_t`, the ordinary union
bound gives

```
P(entire chain correct) >= 1 - sum_t epsilon_t.
```

SSC can therefore reveal an executor hidden by operation selection or stopping
failures. It is constrained decoding with an external schedule and parser. Its
controller code, state, calls, generated tokens, context, and sequential depth
must be counted.

## 4. Interpretation

- If SSC fails individual transitions, Shohin lacks the executor for that
  family; answer-boundary tuning cannot fix it.
- If SSC succeeds while ordinary decoding fails, the missing component is
  selector/halting behavior under that structured contract.
- If only solved demonstrations succeed, classify the effect as in-context
  trace imitation.
- Any attempt to amortize SSC into weights must compete with a deterministic
  compiler, true-trace SFT, prompt consistency, and self-distillation at matched
  resources.

The pilot may reuse the immutable 20-case confirmation manifest and is
post-hoc diagnostic evidence only. A claim-bearing follow-up requires a sealed
1,024-case, four-family board with unseen renderers, corrupted demonstrations,
operation swaps, exact ordered intermediates, stopping checks, and a frozen
resource ledger.

## 5. Raw-260k SSC result: 2026-07-15

The frozen diagnostic is negative. Across the same 20 cases and 55 scheduled
one-operation calls, raw 260k obtained:

```
first transition correct:  0 / 20
all transitions correct:   0 / 20
final chains correct:      0 / 20
```

Every family is 0/5 on the first scheduled transition. However, this is a
renderer failure rather than clean executor evidence: **16/20 first outputs
and 43/55 outputs overall are exactly `input_state + 1`**. The prompt says both
"Return only the next integer" and "Next state", so the frozen model usually
selects a literal successor-integer continuation rather than applying the
named operation. The immutable artifact is
`artifacts/eval_history/raw260k_ssc_diagnostic_20260715_mps.json`, SHA-256
`a152e85294d02173a697e29d8537bf4b53428d747d16c7e3baf692095d9b6a2f`.

This rejects the narrow claim that Shohin exposes a source-free atomic executor
under the preregistered `Current state` / `Requested operation` contract. It
does **not** distinguish missing arithmetic from a renderer that overwhelmingly
selects the wrong lexical transition. The 20-case confirmation already shows
strong renderer dependence, so a fixed no-demonstration format matrix is
allowed as a post-hoc access diagnostic; it must score every arm, cannot select
a winning prompt after seeing answers, and cannot establish a reasoning
mechanism by itself.

## 6. Frozen-format access matrix

The allowed matrix evaluated all three formats on every one of the 55 atomic
transitions and all 20 model-carried chains, with no demonstrations, retries,
repair, search, or verifier feedback:

```
renderer          atomic transitions   full model-carried chains
Question/Answer       40 / 55                    7 / 20
bare equation          8 / 55                    1 / 20
Problem/Work          44 / 55                   10 / 20
```

`Problem/Work` is strongest in every aggregate. Its family results are base
conversion 16/20 atomic and 1/5 chains, modular update 7/10 and 2/5,
multiply-subtract 6/10 and 2/5, and sequential state 15/15 and 5/5. By
operation it reaches add 19/20, multiply 15/20, subtract 8/10, and remainder
2/5. The immutable artifact is
`artifacts/eval_history/raw260k_atomic_operation_formats_20260715_mps.json`,
SHA-256
`b33c26b3963296c0d97b2a6d3332c0be18af40f460137c25652b881824a1ca4b`.

This is the first source-free multi-call capability foothold in R12: a fixed
deterministic schedule plus model-produced visible states improves strict final
chains from 4/20 direct to 10/20. The controller still imports the operation
schedule, parser, repeated model calls, and sequential state carry, so this is
a capability-system result rather than autonomous Shohin reasoning.

## 7. Causal renderer interchange

A crossed-prefix diagnostic held the requested operation fixed while making
the visible state disagree with the state implied by the source context. In
all six add/multiply/subtract cells, the displayed-state candidate received
higher summed log probability than the source-implied candidate; the minimum
absolute margin was 0.79386. There were 18 candidate sequence evaluations and
no generated or training tokens. The artifact is
`artifacts/eval_history/raw260k_renderer_interchange_20260715_mps.json`,
SHA-256
`963177139b6abb333710f0db19a521c341a039fce3f65743ebdd698be6f12170`.

This establishes a narrow causal fact: under a familiar equation-like chart,
the model reads and transforms the displayed state rather than merely replaying
the source-implied answer. It does not localize a parser or prove a reusable
latent program. Prompt canonicalization, visible scratchpad recurrence, and
ordinary process supervision remain resource-preserving explanations.

## 8. Fresh confirmation

`R12_SOURCE_SCHEDULED_REASONING_CONFIRMATION.md` freezes a new 256-case board
before model evaluation: 64 cases each for multiply-subtract, base conversion,
sequential state, and modular update. Board SHA-256 is
`19a84165f15b19911fc8ef229022e47753833d703d77d1e8cc25db9dfc993474`;
its canonical 256-row payload hash is
`4afc6c4b0c271ea2f723078ab183e8d1ac1851fd1728898384ef52275887b0e4`.
Newton job `689535` evaluates direct, whole-work, oracle-state atomic, and
model-carried scheduled arms against immutable raw 260k. Passing can authorize
only an internalization experiment; it cannot itself establish standalone or
latent reasoning.
