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

## New Hypothesis: Counterfactual Bisimulation Compiler

The repeated negative results identify a sharper problem than "the model needs
more chain of thought." The model has not learned a representation that is
*causally sufficient* for future work. A state string can be reproduced as a
template, and an answer can be imitated from a familiar prompt, without the
state actually carrying the facts needed to update, query, or explain a new
situation. V7, VRWM, the semantic-capsule raw controls, and the continuous
packet controls each exposed a different version of this loophole.

The next mechanism to investigate, conditional on the current DRS learnability
gate, is a **Counterfactual Bisimulation Compiler (CBC)**. It uses the model's
own token output as a compact recurrent register, but makes that register
answer to four linked obligations rather than one formatting target:

1. **Compile:** map a natural-language history to a canonical typed state.
2. **Advance:** map that state plus one new event to the next typed state after
   the history has been removed.
3. **Read out:** answer several previously unseen questions using only the
   typed state, never the original history.
4. **Explain the delta:** given two adjacent states, name the single event or
   field change that connects them.

For every episode, a paired paraphrase describes the same world with unrelated
surface wording and an independently generated counterfactual changes one
causal fact. The canonical state must be identical for paraphrases and differ
only in the affected fields for the counterfactual. This is the operational
meaning of *bisimulation* here: equivalent descriptions must induce the same
future behavior under every held-out event/query; a changed fact must change
only the future behavior that depends on it. The external generator and
verifier create labels during training and score results afterward, but the
runtime controller only transports exact model text, drops the source, and
enforces grammar. It does not answer a query, repair a state, rank candidates,
or inject a correct field.

### The Crucial New Constraint: State Interchange

The easy way to fake a state curriculum is to answer from the current prompt
and treat the rendered state as decoration. CBC therefore adds an *interchange*
operation that is not present in ordinary chain-of-thought SFT. For one latent
world, generate two independently worded histories, compile each to a state,
then give the first state's output to a query drawn from the second history.
Because the worlds are semantically identical, the answer must remain correct.
For a counterfactual world, perform the same exchange after changing exactly
one causal fact; now exactly the dependent answers must change. The model is
not shown the original history for either readout.

This turns "does the model emit a plausible state?" into a causal intervention:

1. Same-world interchange must preserve answers across wording.
2. Cross-world interchange must fail in the exact directions predicted by the
   changed fact, rather than merely changing output style.
3. Zeroed, shuffled, and counterfactually mismatched states must lose the
   corresponding advantage on the *same* queries.
4. An inverse-delta prompt must recover the changed field from adjacent states,
   so a lossy answer-only summary cannot pass by accident.

The resulting metric is a **state-necessity margin**: normal model-authored
state accuracy minus matched zeroed/shuffled/mismatched-state accuracy, with
the paraphrase and counterfactual rows reported separately. A high ordinary
answer score with no margin rejects the mechanism. This is the central
distinction from all earlier state experiments in this project.

### Why This Is Different From Earlier State Work

- **Not V7:** V7 rewards a rendered state/answer contract. CBC requires the
  state to survive source deletion and support multiple forward, inverse, and
  query tasks that were not present in the compiler prompt.
- **Not the rejected capsule control:** that was a raw-capability test. CBC is
  a supervised curriculum and treats compilation, transition, and decoding as
  mutually constraining tasks rather than assuming a raw model already knows
  the protocol.
- **Not continuous memory:** the retained object is readable model-authored
  text. Its content, causal effect, and failure modes can be independently
  audited. A shuffled or zeroed state control can therefore falsify the claim.
- **Not ordinary CoT:** a long rationale is not sufficient. The compact state
  must make new predictions after the rationale and source are gone.

### Curriculum, Not a One-Shot SFT

CBC should be staged only after DRS tells us whether the current model can
learn a local symbolic transition at all:

1. **Primitive executor:** DRS establishes exact local transition learning on
   a fixed canonical syntax. This is the active gate, not an assumed ability.
2. **Semantic compiler:** two to four natural-language facts compile into a
   compact state; paraphrase pairs, distractors, and randomized field names
   block lexical copying.
3. **Recurrent world model:** only the prior state plus a new event is
   available for each update. Training alternates forward update, inverse
   delta, and query readout examples so no single answer template dominates.
4. **Compaction under pressure:** after several updates, the model emits a
   shorter canonical state. The dropped trace is never reintroduced. Held-out
   questions include facts that are not asked during compaction, making an
   answer-only summary insufficient.
5. **Interchange before self-check:** generated states must solve queries from
   a separate paraphrase of the same world and fail predictably when swapped
   with a counterfactual world. This is the first point at which a state can be
   called causally useful rather than merely well formatted.
6. **Self-check only after competence:** a verifier/repair role is trained on
   balanced grammar-valid near misses and then asked to judge *model-sampled*
   states. This is where PCD can become a component, not a premise.

The phase-3/4 data must progressively randomize entities, field order,
paraphrases, distractors, operation order, and query wording. It should also
include reversible pairs: state-to-language descriptions and language-to-state
compilation must agree on the same held-out world. This is deliberately a
harder requirement than exact state formatting, because the target property is
semantic invariance rather than a learned serialization.

### Preregistered Advancement Gates

CBC is not authorized for a flagship change on a positive training loss or a
default-template score. Before a follow-on stage, report all of the following
on disjoint worlds, vocabularies, field names, and controller prompts:

- DRS core and held-out results, including first transition, full loop, final
  answer, and paired intervention. A weak primitive means we first compare the
  lower-copy ADL curriculum rather than build a semantic stack on sand.
- Compilation equality for paraphrase pairs and minimal, causal state change
  for counterfactual pairs.
- Source-free forward updates, inverse-delta accuracy, and multiple unseen
  query readouts from the same generated state.
- Same-world state interchange plus cross-world counterfactual interchange.
  Normal versus zeroed, shuffled, and mismatched-state margins must be
  measured on the identical generated states. No margin means the state is
  decorative, irrespective of answer accuracy.
- A label-shuffled verifier control with matched compute. Equal performance
  rejects the claimed self-checker.
- Fresh direct interaction with ordinary arithmetic, code, logic, and state
  questions. Transfer is a requirement, not an aspirational extrapolation.

This is a new project hypothesis, not a claim of field-wide novelty or a claim
that the model already possesses the mechanism. Its value is that it gives a
small model a concrete route from language to a compact, revisable, causally
testable token state. If the gates fail, the failure will identify whether the
barrier is primitive execution, semantic compilation, recurrence, compression,
or self-verification instead of producing another ambiguous SFT score.

## Conditional Hypothesis: Dual-Code Reversible Deliberation

The missing ingredient may be neither a longer trace nor a larger hidden
packet. A small model can make a locally plausible but globally wrong state
transition, then repeat that error with high confidence. An ordinary
self-critique is weak evidence because the same decoder can reproduce its own
mistake. The proposed remedy is **Dual-Code Reversible Deliberation (DCRD)**:
the model carries one compact state in two deliberately incompatible token
codes and must close a reversible loop across them before accepting a step.

For each episode, the generator creates two canonical encodings of the same
machine state. They use independent field order, delimiters, role names, and
digit symbols. A short static codebook may be retained at every turn, but the
problem history may not. The codebook is randomized per episode and held-out
codebooks contain symbols, orders, and bindings never used during training.
That prevents a second rendering from becoming an identity-copy shortcut.

The single model learns four tagged operations:

1. `FWD-A`: advance an A-code state by one local transition.
2. `A->B`: transcode the resulting state into the unrelated B-code.
3. `REV-B`: recover the preceding B-code state from the B-code successor.
4. `B->A`: transcode the recovered predecessor back into A-code.

At inference, the controller forwards exact model text between these calls,
checks only that the last A-code string is byte-identical to the A-code input,
and either accepts the proposed successor or abstains. It never computes a
transition, chooses a digit, repairs a state, ranks candidates, or supplies a
correct alternative. A later extension may use bounded sampling to obtain
another proposed successor, but it may never use a solver at runtime.

This is not a proof of correctness. A wrong transition can still be internally
reversible. Its purpose is narrower and testable: force the model to represent
state semantics through two non-copying channels, then turn agreement into a
reliability signal rather than an ungrounded natural-language self-critique.

### Why It Is Not Another Formatting Control

- **Different syntax is functional, not cosmetic.** The A/B codebooks and
  field orders vary per episode, and a held-out codebook is required. A model
  that memorizes a fixed serialization cannot transcode or close the loop.
- **The round trip has an asymmetric failure surface.** `FWD-A` and `REV-B`
  operate in opposite temporal directions; the model must preserve enough
  information for an inverse operation after source deletion.
- **Acceptance is evaluated as a decision, not as a trace score.** Report
  unconditional accuracy, accepted accuracy, coverage, and the improvement in
  accepted accuracy over an equal-call baseline. A mechanism that merely
  abstains is rejected.
- **It has adversarial interventions.** Replace one B-code field, swap a B
  state from a matched counterfactual episode, or permute the B codebook.
  Correct acceptance must fall and dependent answers must change in exactly the
  solver-predicted direction. Invariance to these changes means the second
  code is decorative.

### Required Controls And Advancement Gate

DCRD is conditional on a positive DRS core execution result. It must remain an
isolated experiment with a fresh output directory and no flagship path. Before
it can motivate a semantic CBC compiler, all of the following must hold on
unseen operands, widths, controller wording, and per-episode codebooks:

- The basic DRS model must beat raw zero on first transitions, closed loops,
  final answers, and paired counterfactual interventions. Otherwise DCRD only
  adds an elaborate checksum to a nonexistent executor.
- DCRD's accepted states must have materially higher exact solver accuracy than
  its unfiltered proposals *and* an equal-call control that simply repeats the
  A-code lane. Coverage must be reported with confidence intervals.
- `A->B->A` alone, a shuffled B-codebook, and an identity-format B lane are
  matched controls. Equal performance rejects the claimed semantic second
  channel.
- The model must fail closed under a one-field B-code corruption and reject
  counterfactual B-code interchange whenever the query depends on the changed
  field. Passing ordinary answers without this sensitivity is a failure.
- A forward/reverse round trip may never be reported as a reasoning score by
  itself. Only correct accepted answers on held-out tasks and fresh interactive
  probes count as capability evidence.

If this fails, we learn whether the bottleneck is primitive transition
learning, codebook binding, inverse dynamics, or correlated self-error. If it
passes, it supplies a compact, model-authored state and an internal
error-detection signal that can be carried into CBC's language compiler. This
is a project hypothesis, not a novelty claim and not authorization to modify
the live pretraining run.

### Implementation Status

The CPU-only protocol substrate is implemented in
`train/dual_code_reversible_protocol.py`. It provides deterministic per-episode
A/B codebooks, channel-specific serialization grammars, strict code-specific
parsers, source-free prompt builders, and a solver-only inverse transition for
data construction and scoring. Train and
held-out codebooks use disjoint alias vocabularies and structurally distinct
instruction interfaces. The generator and independent auditor bind prompt
style to codebook vocabulary for the training corpus, while the protocol still
permits crossed style/codebook combinations for future attribution controls.
This makes literal train/held-out n-gram overlap an auditable data failure
rather than a hidden template confound.

`pipeline/generate_dual_code_reversible_v1.py` and its independent companion
`pipeline/audit_dual_code_reversible_v1.py` now construct and semantically
recompute every forward, transcode, inverse, and readout target. A local
1,000-episode preflight generated 21,000 training rows plus 200 held-out paired
counterfactual episodes: the auditor found 0 invalid rows or episodes, 0
normalized duplicates, 0 exact held-out prompt hits, and 0 literal 13-gram
hits. The smaller end-to-end contract has the same result. These are only
generator/auditor checks: no durable DCRD corpus has been admitted and no
controller rollout, SFT, or GPU job has been submitted. The DRS causal chain
still decides whether this branch is worth launching.

`train/test_dual_code_reversible_protocol.py` exercises codebook separation,
encode/decode round trips, canonical-state leakage rejection, prompt-style
binding, and 120 randomized inverse-transition cases. The implementation is a
precondition for a later causal experiment, not evidence that the model can use
the protocol.
