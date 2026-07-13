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

## Workspace Hypothesis: A Small Reportable Register, Not More Narration

The [global-workspace study](https://transformer-circuits.pub/2026/workspace/index.html)
is relevant to Shohin, but it is not evidence that a 125M model already has a
usable workspace. Its useful operational claim is narrower: deliberate
reasoning is associated with a **small, reportable, selectively used
representation** that can be written once and consumed by multiple downstream
computations. The study also distinguishes this from ordinary automatic
processing and tests it with causal interventions, rather than treating an
eloquent explanation as evidence of thought.

That maps directly onto the current DRS result. DRS v2 gets the first local
state right on 497/500 core episodes yet ends only 275/500 closed loops; it can
compute a local action but does not reliably transport the evolving state. The
original DRS carrier also needlessly makes the model rewrite immutable operand
tapes on every turn. The static-tape recurrent-register (STRR) control removes
that copy burden: the controller re-sends immutable evidence verbatim, while
the model emits only `p,c,r,z`, the compact mutable register. The controller
does not calculate, repair, rank, or choose that register.

STRR is therefore the first workspace-style experiment, not a latent-reasoning
claim. It advances only if all of these are true on held-out tapes, wording,
and paired counterfactuals:

1. **Write:** the model emits the exact next compact register from the fixed
   tape and preceding register.
2. **Maintain:** model-emitted registers, not solver states, survive the full
   closed loop.
3. **Broadcast:** the same terminal register supports distinct readouts
   (final result and indexed-digit queries), rather than merely the response
   template that produced it.
4. **Intervene:** swapping one operand in a paired counterfactual changes the
   resulting model-authored state and final answer in the predicted direction;
   malformed, zeroed, shuffled, or mismatched registers fail on the same
   readouts.
5. **Generalize:** results hold under disjoint values, widths, and natural
   wording. A default-syntax score is not a workspace result.

Only a positive STRR result justifies the next step: a semantic compiler that
maps natural-language facts into this compact register and then tests
state interchange across paraphrases. A negative STRR result would instead
localize the bottleneck below semantic reasoning, in primitive recurrent state
transport itself. We will not imitate the paper's Jacobian lens prematurely;
after a positive behavioral gate, a lightweight late-layer logit-lens trace
can test whether a stable, reportable register has emerged inside the tiny
model. The behavioral causal tests remain decisive.

### Raw Workspace-Patching Baseline: No Simple Broadcast Register

`train/probe_digitwise_workspace.py` is the first diagnostic built from this
hypothesis. It does not train, generate a solver state, or claim to implement
the paper's Jacobian lens. On a teacher-forced DRS transition, it captures the
last-position residual after a selected block, replaces it with the residual
from a matched held-out transition whose correct next carry or digit differs,
and measures whether the target log-odds move toward that source state's
answer. A genuine result must be directional under symmetric A-to-B and B-to-A
swaps; a generic perturbation cannot satisfy that condition consistently.

The raw-200k local-MPS baseline is negative. Its frozen artifact
`artifacts/evals/digitwise_workspace_raw200k_mps_p4_layers.json` has SHA-256
`78b5efa4f3f7fe3ef10104de8d02fdee67f253c805c58f214a4cd1985c495875`.
It evaluates five held-out regimes, four matched pairs per regime, both carry
and digit fields, and symmetric directions: 40 directions per field/layer.
Carry swap deltas are only `+0.001` to `+0.028` log-odds with 18-22/40
directions positive; digit deltas are `-0.042` to `+0.0002` with 14-20/40
positive. A 10-direction smoke had seemed positive, but the expanded matched
sample removed it. Therefore the raw model does **not** expose a stable,
last-position, broadcastable local-state direction under this probe.

This does not say the model has no internal arithmetic features: the state can
be distributed across positions or represented nonlinearly. It does provide a
specific, preregistered contrast for the DRS/STRR interventions. Post-DRS
probe `687578` is queued after the existing wording, direct-interaction, and
NLL evidence chain with the identical 80-direction configuration. STRR may
advance only if its behavioral closed-loop gates and this matched diagnostic
are interpreted together; neither alone is a reasoning claim.

### Restricted Jacobian Digit Lens: A More Specific Causal Diagnostic

Whole-residual swaps are intentionally blunt: a negative result can mean that
the relevant state is distributed, that the swapped residual carries too many
unrelated features, or that there is no reusable state direction at all. The
paper's J-lens suggests a more selective test, but reproducing its full
cross-position, cross-corpus Jacobian construction would be unjustified for a
125M model before we establish a behavioral primitive.

`train/probe_restricted_jacobian_digit_lens.py` therefore implements a bounded
middle ground. On a frozen held-out DRS split, it averages the gradient from a
selected block's last prompt-position activation to each one-token *next-state
digit* logit. The discovery episode IDs are hash-separated from the evaluation
episode IDs. It then measures two disjoint evaluation conditions:

1. **Readout:** can the ten averaged directions rank the correct next digit
   above chance on new episodes?
2. **Causal swap:** on matched pairs with the same local operation, width,
   position, and carry but different correct output digits, does swapping only
   the two corresponding gradient-direction coordinates shift the target's
   next-token log odds toward the source digit more than a fixed shuffled-label
   control?

The job wrapper is tested but unsubmitted. This is not a full J-lens, not a
semantic workspace probe, and not evidence of general reasoning. Its purpose
is purely diagnostic: distinguish a reusable token-specific direction from a
template-bound or distributed state representation after the active DRS direct
interaction, NLL, and whole-residual patch chain has finished. A positive
restricted result still cannot authorize CWI or a capability claim without the
already-preregistered behavioral, counterfactual, and multi-readout gates.

## Conditional Technique: Counterfactual Workspace Induction

The paper's counterfactual-reflection result motivates a distinct follow-on
experiment, **Counterfactual Workspace Induction (CWI)**. It is deliberately
not another request for the model to print a chain of thought. Starting from a
checkpoint that can already execute a local register transition, CWI would
append a *training-only* reflection turn after a fixed-tape local-state context:
"Which one invariant distinguishes the legal next register from this
grammar-valid foil?" The supervised continuation names the concrete local
operator, input digits, carry/borrow, result digit, and immutable fields that
must be preserved. Loss is computed only on that appended reflection; at
evaluation, the reflection question is absent and the model must perform the
ordinary direct state update.

The critical foil is not malformed text. It is a state-shaped candidate that
changes exactly one semantic field: an incorrect carry, a wrong `r[p]`, an
unjustified program-counter change, or a rewritten immutable tape. Thus the
reflection cannot be solved from style or grammar. If it transfers to the
unreflected task, it would be evidence that training a reportable disposition
changed the intermediate computation used for action, the limited phenomenon
the paper tests at scale.

CWI is **conditional** on a positive STRR primitive gate. It must be compared
from the same STRR checkpoint and token budget against: (1) a syntax-only
reflection with no local arithmetic content, (2) a reflection-label permutation
control, and (3) an equal-compute direct-transition continuation. Advancement
requires an improvement on held-out unreflected state loops and paired
counterfactuals, no loss of distinct register readouts, and a matched positive
change in the residual-patching diagnostic. A reflection that only improves its
own prompted explanation is rejected. This would make CWI a test of
workspace-shaped computation, not a new narration style.

## Conditional Representation Control: Token-Native Delta Ledger

The static-tape register removes immutable input copying, but its next state
is still a **21-token** BPE continuation (`dwr:p=...;c=...;r=...;z=...`). The
existing textual append-ledger delta is shorter but still costs **14 tokens**.
At the observed state-error rates, those serial output decisions are a
plausible exposure-error bottleneck independent of arithmetic. The proposed
**Token-Native Delta Ledger (TNDL)** therefore encodes one model-authored
transition as exactly three *existing* atomic special tokens, in fixed field
order: next position, carry/borrow, and result digit. It does not add a
tokenizer entry, alter the model architecture, or let a controller calculate
anything. The controller only validates that exactly three code tokens were
emitted, retains their exact sequence, and supplies the last emitted triple on
the next update; the model must still derive the carry and digit from the
unchanged operand tape. The full emitted ledger is supplied only for final
readout. Because this carrier has intentionally tiny finite entropy, final
prompts repeat an opaque hash of the immutable tape between triples. This
prevents a train and held-out prompt from sharing a long carrier substring;
the controller never decodes, predicts, or computes with the hash, and the
same rendering is required in all matched controls.

This deliberately differs from DCRD. DCRD asks the small model to bind random
natural-language codebooks and perform reversible translations, which tests a
much harder semantic-binding hypothesis. TNDL instead holds the mapping fixed
and isolates whether the previous negative results are dominated by state
serialization length. It is also a more atomic version of ADL: no textual
field labels, result-tape rewrites, or controller-generated arithmetic are
introduced. Its initial scope is width 4/6 train and width-8 held-out; any
claim about longer context must wait for a successful first-level ledger.

TNDL is useful only if it is compared against a text ADL and static-tape
register from the same raw checkpoint, update budget, operands, held-out
counterfactuals, and controller wording. A positive result requires materially
higher complete closed-loop and paired-intervention accuracy, not merely more
well-formed three-token responses. A second fixed permutation of the ten code
tokens is required before claiming that the result is not an accidental
association with the tokenizer's pre-existing special-token semantics. A
passing first-level carrier would be a primitive transport result, not proof
of language reasoning or context scaling; only then can it be combined with
the CWI and semantic compiler gates.

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

### CBC Protocol/Audit Preflight: 2026-07-13 15:24 EDT

The CBC substrate now exists as an isolated CPU-only protocol,
`train/bisimulation_compiler_protocol.py`, plus a generator and independent
auditor.  The protocol uses a canonical `cbc:key=value;key=value` carrier and
a distinct `cbc-delta:` grammar.  It has two source-description compilation
interfaces, source-free update and inverse-delta prompts, and a final query
that receives only the carrier.  Every held-out episode carries a paired
counterfactual whose initial first field changes by one while its operation
sequence is identical and its final answer must differ.

The medium local preflight generated **1,000** train episodes, **16,000**
train rows, and **120** held-out paired-counterfactual episodes.  The auditor
independently reconstructed every compilation target, state update, delta,
readout, shared operation sequence, and counterfactual relation.  It found
**0** invalid train rows, **0** invalid held-out episodes, **0** normalized
duplicate prompts, and **0** exact or literal 13-gram train/held-out prompt
hits.  Corruption tests prove that it rejects both changed train targets and a
semantically valid-looking counterfactual with a mismatched operation sequence.
No
durable corpus, SFT checkpoint, controller rollout, or GPU job has been
created.  CBC remains conditional on a positive DRS causal result.

The matching transport-only controller is also preflighted.  It accepts only
a parsed model-emitted state, renders the next source-free prompt around that
text, and halts on an incorrect or malformed emission.  Its test covers
primary rollout, inverse-delta checks, same-world compiler interchange, and a
cross-world counterfactual mismatch on the identical query.  A bad first
state terminates the run; it is not canonicalized into a solver answer or
repaired.  This makes CBC's later state-necessity measurement executable
rather than an informal data claim.

### DRS v2 Coverage Diagnosis: 2026-07-13 15:38 EDT

The new read-only position-coverage audit resolves a material ambiguity in the
ongoing DRS core evaluation.  The immutable v2 train corpus has **zero**
transition inputs containing digits **3–9** at the most-significant position
of either operand tape for width 4 or width 6; those positions were limited
to values below `3000` and `300000`.  In contrast, each `value_ood` regime
uses `7000–9999` or `700000–999999`.  All **600** paired value-OOD local
transition contexts per width therefore have an unseen exact local arithmetic
context, and the audit records **1,200** unseen digit-position events per
value-OOD regime.  Width-8 is a true compositional extrapolation with **4,800**
unseen local contexts.

This does not excuse a poor result; it prevents a false conclusion.  DRS v2
can only establish in-distribution fixed-register execution.  Any next DRS
curriculum must stratify digit support by width, position, operand tape,
operation, and carry/borrow context before its value-OOD result can be used as
evidence about algorithmic generalization.  No revised corpus or GPU job is
created before the current serialized DRS chain finishes.

### DRS v3 Minimal Transition Basis: 2026-07-13 15:45 EDT

The corrective candidate is deliberately not simply “more random arithmetic.”
`generate_digitwise_basis_v3.py` constructs complete arithmetic episodes whose
designated transition enumerates every **reachable** local tuple of `(width,
operation, position, carry/borrow, left digit, right digit)` for width 4 and
width 6. It keeps full operand and result tapes, so the model still has to
preserve a recurrent state rather than answer a disconnected lookup question.
The paired held-out sets use unseen full tapes at the same local-support basis
(`recombine_w4`, `recombine_w6`) and an unseen width (`width_ood_w8`).

Its independent admission audit rechecks every arithmetic row and held-out
counterfactual, then independently requires all **3,400** reachable contexts.
The medium local preflight with two tape variants produced **6,800** complete
episodes and **77,946** rows: 0 malformed rows/episodes, 0 normalized duplicate
prompts after deterministic deduplication, 0 exact or literal 13-gram split
hits, and all 3,400 contexts present. A corruption test deleting every
instance of one otherwise valid context is rejected. This is a staged
learnability control, not a model result, and it has no durable corpus, SFT,
or GPU allocation before the current DRS core/held-out/direct evidence chain
has finished.

The future one-epoch launch path is now static-tested but deliberately
unsubmitted. `sft_digitwise_basis_v3.sbatch` independently binds the candidate
data and held-out SHA-256 values to its admission audit, requires all 3,400
contexts and the three prescribed held-out regimes, rejects any contamination
or structural counter, and proves the exact inference/SFT prompt boundary
before using CUDA. This makes a later causal test reproducible; it does not
promote the hypothesis, create a durable corpus, or reserve a GPU.

### Static-Tape Recurrent Register (STRR): 2026-07-13 16:04 EDT

DRS exposes a second representation confound besides its missing local
contexts. The previous `dws:` state makes every self-authored transition copy
the immutable `a` and `b` operand tapes, even though only the control register
and result tape change. That can reward long-string reproduction more than
local execution. **STRR** factorizes these roles: the original problem is a
fixed `dwt:` tape containing opcode, width, and the two operand tapes; the
model emits only the evolving `dwr:` register with `p`, `c`, `r`, and `z`.

The transport-only controller is deliberately constrained. It re-sends the
unchanged tape from the episode and forwards only a parsed model-emitted
register. It never applies the arithmetic transition, replaces a malformed
register, or chooses among outputs. Thus the test asks whether preserving
immutable evidence in context lets a small model carry a compact dynamic state
more reliably, rather than delegating arithmetic to the controller.

Its independently checked medium preflight uses the same **6,800** complete
episodes / **77,946** rows / **3,400** reachable local contexts and **120**
paired held-out counterfactual episodes as the matched basis smoke. The
generator and auditor report 0 malformed rows or episodes, duplicate prompts,
counterfactual mismatches, exact split hits, or literal 13-gram hits; deleting
all examples of one valid local context makes admission fail. The matched
closed-loop evaluator is static-tested and can retain capped success/failure
transcripts separately for each regime. Its matching staged SFT wrapper binds
the immutable data and held-out hashes to that audit, verifies exact inference
and SFT prompt-token boundaries, refuses existing outputs, and requires a real
CUDA allocation. STRR is still an unsubmitted candidate: it has no durable
corpus, SFT checkpoint, or GPU allocation. It becomes admissible only after the
running v2 core, held-out wording, and transcript chain distinguish coverage
failure from a deeper execution failure.

### Complete DRS v3 Basis Artifact: 2026-07-13 16:20 EDT

The full eight-variant coverage control is now immutable and mirrored locally
and on Newton, but has not been submitted for SFT. It has **27,200** complete
episodes / **311,127** deduplicated rows, covers **3,400 / 3,400** reachable
local contexts, and keeps **900** paired held-out episodes (300 each of
`recombine_w4`, `recombine_w6`, and `width_ood_w8`). Its independent audit
reports zero invalid rows or episodes, normalized duplicate train prompts,
missing contexts, exact train/held-out prompt collisions, or held-out
13-gram collisions. The train and held-out SHA-256 values are respectively
`b785866bf24813272d346e4a3bb717d4156b01a59a4dd8ccaf450733267368f6` and
`f2fcfcae41b55aa82dd360036bd8c9c00ed6e4ca442debec1c85ed282e50dfe1`.
This artifact tests the coverage confound in the observed v2 value OOD gap; it
does not establish algorithmic generalization by itself and remains gated on
the active transcript evidence chain.

### DRS v2 Core Result: 2026-07-13 16:25 EDT

The canonical held-out core evaluation is a positive narrow mechanism result
and a negative generalization result. The isolated checkpoint gets **275/500**
final answers: 100/100 on fit width 4, 98/100 on fit width 6, 34/100 and
43/100 on the two unseen-value regimes, and 0/100 on unseen width 8. However,
the first model-authored state is correct on **497/500** episodes, including
98/100 width-8 episodes. The model begins from valid local arithmetic but
accumulates errors over later turns. This shifts the immediate causal priority:
the complete v3 basis remains needed to isolate unseen interior contexts, but
the first corrective SFT should be **STRR**, which removes immutable tape
rewriting and thereby directly tests multi-step state transport. This remains a
conditional decision until held-out wording and transcript probes complete.

### Complete STRR Artifact: 2026-07-13 16:23 EDT

The full static-tape/recurrent-register corpus is immutable and mirrored
locally/Newton: **27,200** episodes / **311,127** deduplicated rows, **3,400 /
3,400** required local contexts, and **900** paired held-out episodes split
evenly across recombine widths 4/6 and width-8 OOD. Its independent admission
audit reports zero invalid rows or episodes, normalized duplicate prompts,
counterfactual mismatches, missing contexts, exact hits, or 13-gram hits.
Train SHA-256 is `82245615f0849c3270f99f2db85c604ff46cb2c3dfb14f0ab3660dff3eb0d3ec`;
held-out SHA-256 is `a699ac58ad8184f4dc23dcfa317cd6e7b8f7d4ef453dcbf1ae21201901e0948a`.
It is not trained or allocated and should only be submitted as an isolated
transport control after the current transcript gate.

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
