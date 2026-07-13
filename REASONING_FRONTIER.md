# Reasoning Frontier

## Current Diagnosis

The raw 200k checkpoint is not displaying an unmeasured reasoning capability.
Direct, fresh interaction is `1/7` initially, `0/7` after self-review, `1/7`
with a supplied correct intermediate fact, and `0/7` after compact-state reuse.
It produces incorrect arithmetic, non-executing code scaffolds, and repeated
Markdown. Forced-choice likelihood also ranks the correct answer first in only
`1/7` cases. This is a failure to select and execute a reliable action, not
just a benchmark extraction or visible-chain-of-thought problem.

The live model is a 30-layer, 576-wide, 125-135M parameter transformer. Its
pretraining path is healthy but historically dominated by math/code sources.
That is useful for symbols but weak for answer-mode control and natural-language
parsing. The future language-balanced corpus is a required data transition at
a natural handoff, not a speculative fix for the active run.

## Hypothesis: Proof-Carrying Deliberation

The next distinctive mechanism is **proof-carrying deliberation (PCD)**. A
tiny model should not be expected to invent and maintain a long free-form chain
of thought. It may be able to build a small executable thought one local action
at a time if it learns both sides of the action:

1. **Propose:** emit a typed, compact next state.
2. **Verify:** inspect a grammar-valid candidate state and identify whether it
   is the single legal successor, including the first violated local field.
3. **Deliberate:** generate several candidates itself, ask itself to verify
   each, and select only according to its own verdict. The controller only
   carries exact model text and enforces syntax; it never computes, repairs, or
   inserts a correct candidate.
4. **Compact:** periodically replace raw deltas with a model-authored proof
   block whose fields are independently locally checkable on the next turn.

This differs from ordinary chain-of-thought distillation. The model is trained
on counterfactual *near misses* that are the same length and grammar as correct
states, so it cannot win from style, answer position, or malformed text. A
verifier that cannot distinguish these cases is not useful, even if it can
recite a state template.

## First Falsification Gate

`train/probe_transition_verifier.py` is the raw feasibility probe. It presents
balanced valid and grammar-valid invalid DRS transitions. Invalid candidates
change only a local digit, carry/borrow, or immutable operand tape. It reports
accuracy by wording, label, and near-miss type from verbatim completions.

This probe does **not** train the model and does not yet establish PCD. It
answers a narrower question: is local verification materially easier for the
raw model than free-form state generation? The answer determines whether a
counterfactual verifier curriculum is worth an isolated H100 ablation.

### Gate 0 Result: Raw Verification Is Also Absent

The raw 200k checkpoint fails the first probe. On 48 balanced, grammar-valid
DRS transitions it emits no usable verdicts (`0/48`). Its verbatim responses
are bare digit strings or repeated document fragments. That alone could be an
answer-mode failure, so the exact two completions were also scored by mean
token likelihood. The model prefers `verdict=valid` on every case, yielding
exactly `24/48 = 50%` on the balanced labels. It has no raw local-verification
signal under this contract.

Artifact: `artifacts/eval_history/transition_verifier_likelihood_raw200k_20260713_mps.json`,
MD5 `fb7bbdbb1fa16104117f09c6c3faa07c`.

The consequence is not to declare PCD successful by construction. It becomes
a conditional supervised experiment: only consider it after the current DRS
SFT proves that the model can learn a core local transition. If DRS cannot do
that, there is no basis to expect a jointly trained generator/critic loop to
bootstrap itself.

## Required Causal Evidence Before Any Claim

A future PCD ablation must keep these gates:

- Solver-generated train and held-out operand tapes, widths, vocabulary, and
  controller wording are disjoint; every controller prompt is overlap-audited.
- Negative states are grammar-valid and balanced by error type. Label order,
  wording, and candidate position must be randomized.
- The evaluator uses candidates sampled by the model itself. It may not give
  the model a solver-supplied correct option at inference.
- Report greedy generation, sampled generation, and model-verifier reranking
  on the identical candidate pool. The external solver scores afterward only.
- Require improvement on held-out state transitions, complete closed loops,
  paired counterfactuals, and natural wording. A template-only score cannot
  advance the mechanism.
- Run a label-shuffled verifier control with the same data, steps, and compute.
  If it performs equally well, the verifier learned a surface prior rather
  than an executable invariant.

Even a passing PCD result would establish only narrow, model-authored
algorithmic deliberation. Broad natural-language reasoning remains a separate
claim and needs direct-interaction and public held-out evidence.
