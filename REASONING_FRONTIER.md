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

## Semantic Bootstrap: Learn One Natural-Language State Primitive Before a Broad Mix

The raw 200k operator transcript and the completed V9 decision eliminate an
important ambiguity: the model does not currently turn a two-field natural
language record into a reusable state, and a large broad mix did not repair
that. It is therefore premature to ask it to reason over long contexts or to
judge an elaborate proof. **V10A** was the next independent test: bridge-only
SFT from the immutable raw 200k checkpoint on the admitted semantic-bridge
corpus. It covers only five solver-verified families: product adjustment,
state chains, base conversion, continuation from a verified fact, and repair
of a wrong computation.

V10A is deliberately a *learnability* ablation. A good in-distribution loss or
visible `<think>` block is irrelevant. It earns a second stage only if it
improves both (a) the value/template-disjoint five-family held-out bridge
evaluator and (b) a fresh direct source-drop/reuse interaction that is not
part of the bridge corpus. This separates “the model can imitate a concise
calculation trace” from “the model can form a small semantic object and use it
after the original story is absent.” Only the latter makes semantic capsules,
context compaction, or CWI scientifically defensible.

The bridge evaluator is still only within-family generalization, so V10A also
faces a separately generated, **evaluation-only semantic-composition suite**:
product-to-chain, base-then-adjust, verified-fact-to-chain, repair-to-chain,
and source-dropped named-state updates. Its values, terms, question forms, and
operation compositions are outside bridge training; an independent audit
rejects exact or word-13-gram overlap with the full bridge corpus. Passing the
bridge evaluator while failing this suite is a narrowly formatted curriculum
result, not semantic state competence.

### V10A Outcome: Reject the Family-Trace Hypothesis

The isolated one-epoch V10A checkpoint fit its 200,000 bridge rows (loss
`1.0546 -> 0.0126`) but did not acquire a semantic state primitive. Its
checkpoint-bound 500-case bridge score was **123/500** answers and **121/500**
solver-equation trace contracts: base conversion 35/100, fact continuation
62/100, product adjustment 3/100, state chain 15/100, and trace repair 8/100.
The separate source-dropped cross-family suite was **4/500**, all four in
repair-to-chain; the other four families, including named-state source drop,
were 0/100. The direct seven-case interview improved only to 3/7 initial,
1/7 review, 3/7 supplied-fact use, and 1/7 state reuse. These are learned
family responses, not a reportable state that survives source removal.

V10A therefore blocks the semantic capsule, CWI, KV-anchor, and ISL branches.
No formatting score, low training loss, or apparently explanatory `<think>`
text can reopen those branches without a source-deleted, multi-consumer pass.

### Next Basis: Two-Value Semantic Transport

The right next experiment is smaller than ISL. V10A confounds language-to-state
transport with multiplication, base conversion, and long family traces. The
new **semantic-basis transport** candidate asks only for a natural-language
record to compile to `ledger:P=<integer>;Q=<integer>`, after which the source is
removed. The same model-emitted ledger must support an add-to-P transition and
two independent consumers (`P-Q` and `P+Q`). Train and held-out splits differ
in values, language, field labels, and domains.

This is not an ISL claim and not a context-scaling claim. It must first pass a
closed-loop evaluator that forwards only the model's exact emitted ledger, then
pass paired state swaps, zeroed/mismatched-ledger controls, and held-out source
language. The Stokes CPU builder is only an audited data-admission step; no SFT
may start until this basis evaluator and its counterfactual controls are bound
to the generated artifacts.

### Exact-Carrier Correction: Semantic-Basis V2

The first admitted semantic-basis corpus remains preserved as a data-quality
artifact, but it cannot be used for the causal experiment. Its compile,
reflection, and update completions contained reasoning prose followed by a
ledger line. Any controller that extracts, parses, or reprints that substring
would become an unmeasured semantic component, so a good score would not prove
that the model's own emission is portable.

V2 is a distinct, immutable candidate with only five targets per episode:
`compile -> ledger:P=<int>;Q=<int>`, `reflect ->` the identical exact ledger,
`update ->` the next exact ledger, and two `answer=<int>` consumers. The
consumer prompts receive the updated raw model emission by one literal-string
replacement and have no access to the source description. The 150,000-row
train / 5,000-row held-out build has distinct train/held-out values, labels,
domains, and every phase's wording. It also enforces uniqueness of both source
and post-update ledgers, so no downstream prompt is duplicated accidentally.

The controller rejects any non-full carrier or answer and never calculates,
normalizes, or repairs a model output. The held-out evaluator requires:

1. Correct exact compile and reflection from two source descriptions.
2. A raw compile emission to drive an update, and that raw update emission to
   drive both arithmetic consumers.
3. Two normal episodes to pass before cross-episode interchange is tested;
   the donor's literal model-produced update string is then placed in the
   receiver's source-deleted consumer prompts.
4. A zero-carrier non-recreation control and an evaluator-created P/Q mismatch
   that must produce the counterfactual answers rather than the original
   answers. These controls are explicitly never called model-authored state.

This is the minimum behavioral analogue of a workspace-style claim: reportable
content, multiple downstream readers, and causal swaps. A pass would still be
only a narrow synthetic transport result. It would justify an isolated learning
ablation, not a broad-reasoning or context-scaling conclusion.

The immutable raw-200k MPS smoke is the pre-learning anchor: both direct and
inference-aligned `Question:/Answer:` four-pair probes are **0/8** correct
compile emissions, **0/8** correct reflection emissions, and therefore 0/8
exact reportability or downstream transport. Its raw continuations are generic
pretraining-style prose (for example, "The first step is ..."), not a ledger.
This is a useful negative: the base model does not already implement the
requested output interface, so any later success must be judged against this
fixed checkpoint and must still survive the controls rather than being called
recovered latent reasoning. The queued full H100 baseline uses that same
standard prompt surface, eliminating SFT/evaluation boundary ambiguity.

The full H100 baseline now corroborates that anchor rather than merely being
consistent with it. Read-only run `687792` evaluated 100 deterministic held-out
pairs from `best_step200000.pt` on the standard Q/A surface: **0/200** correct
compile emissions, **0/200** correct reflection emissions, **0/200** exact
reportability matches, **0/200** correct updates, **0/200** normal strict
transports, and **0/100** model-authored interchange, mismatch, and strict
causal pairs. All 100 per-pair raw transcripts are retained in the
hash-verified artifact SHA-256
`e4a96192abc528bad1a8c7ed4e5f275dc5bdb1080a2ac36a4e65a19026a8067e`; they
show source paraphrase, generic explanation, or prompt continuation, not a
single full ledger. This makes the next SFT a clean **learnability** ablation,
not an attempt to recover an unmeasured raw latent skill.

The isolated learnability result closes the broad V2 claim. One epoch from the
same immutable 200k checkpoint reaches **198/200** exact compile emissions,
**200/200** exact reflection emissions, and **198/200** identical carriers on
the full held-out evaluator. Yet it completes only **23/200** state updates and
**6/200** complete update-plus-two-reader episodes; no pair contains two normal
strict episodes, so all **100** model-authored swap, zero, mismatch, and strict
causal outcomes are zero. The full transcript artifact is SHA-256
`b643241ea154b49482627e9c6c2e73d20ad17b64422cf5341c06702c7327505e`.

This is a useful negative rather than a confusing mixed result: the model can
report and reproduce an exact carrier under two independently worded source
prompts, but cannot reliably operate on that carrier once source information
is removed. It is not flexible multi-reader state, and it is not evidence for
a workspace. The completed train-only diagnostic confirms that this is not an
evaluator boundary failure: it reaches **200/200** compile/reflection/equality,
**194/200** updates, **160/200** normal strict transports, **65/100** raw
model-authored swaps, and **48/100** full causal passes, with artifact SHA-256
`b13050b50345834cf0ce861f23facb0c43a3e9753649ee3d0a410001355171ce`.

The present held-out split changes source wording, labels/domains, value range,
and delta range together. The completed factorial *evaluation* matrix therefore
held three factors fixed while changing one: language, P/Q magnitude, or update
delta. It finds **1/100** strict causal passes for language-only, **3/100** for
values-only, and **2/100** for delta-only, versus **48/100** on train-only
episodes. Each condition still compiles and reflects almost perfectly, so this
is not a format or controller failure: it is a three-axis failure of semantic,
numeric, and operator invariance. The first follow-up may target wording
invariance, but no reflection data or context mechanism is justified yet.

### External Workspace Paper: What It Changes

The 2026 global-workspace paper is a useful experimental standard, not a
turnkey recipe for a 125M model. Its central criteria are stronger than
verbalization: a candidate representation must be reportable, deliberately
modulable, used in intermediate computation, flexibly reused by distinct
downstream readers, and selectively necessary for the resulting behavior. V2
was designed as a small behavioral proxy for the reportability, multi-reader,
and intervention portions of that standard. Its failed held-out operation gate
means Shohin does not yet warrant a workspace claim.

The paper's Jacobian lens is a corpus-averaged, per-layer causal readout, not a
logit-lens screenshot. A faithful implementation would need a reproducible
prompt corpus, averaged Jacobian maps, layer/position selection, and a
pre-registered activation intervention. The prior restricted four-layer digit
lens was negative. A full lens build is therefore deferred until a behavioral
primitive passes a source-deleted causal transport gate; otherwise it risks
finding correlations in a model that cannot use the proposed content.

Its counterfactual-reflection result is directly relevant only as a later
ablation: supervise an interrupted reflection continuation, score the original
uninterrupted context with no reflection request, and compare against a
token-budget-matched neutral auxiliary control. It is not evidence that
visible `think` tokens create reasoning, nor evidence that the technique will
transfer from the paper's large model to this one.

### Conditional Candidate: Paraphrase-Equivariant State Alignment

If the factor matrix confirms that wording transfer, rather than only numeric
extrapolation, is the bottleneck, test a representation-level objective rather
than another larger response-format corpus. Each train episode already provides
two independently worded source prompts (`compile` and `reflect`) with the
same exact latent P/Q state. At the final prompt token, capture a designated
mid-layer residual for each prompt and add a normalized alignment loss between
the two states while retaining the ordinary completion-only next-token loss.

The aim is deliberately narrow: force distinct descriptions of the same facts
to write a common *internal* state before the ledger is emitted. It is neither
a latent-token rollout nor a claim that the aligned vector is a workspace.
The isolated ablation must use the immutable raw-200k checkpoint and compare:

1. CE-only on the identical paired corpus and update budget.
2. CE plus same-state residual alignment.
3. CE plus a length- and batch-matched **different-state** pairing control.

All three retain the same ledger targets. A benefit is credible only if the
same-state arm improves the source-language-only causal gate over both
controls, then retains a measured fraction of that gain under the values-only
and delta-only gates. Diagnostics must report feature cosine similarity for
same-state versus different-state prompts, feature norms/variance to rule out
collapse, ordinary token loss, and every causal control. No reflection or
context-compaction claim can be made from representation alignment alone.

The paired SFT objective is not, by itself, a mechanistic result. The matching
read-only `eval_paraphrase_state_causality.py` therefore uses a full replay at
every decode token and replaces only the answer-boundary residual after the
selected block. This avoids reusing a pre-patch KV cache whose keys would make
the intervention ambiguous. It evaluates identity replacement, same-ledger
compile/reflect exchange, and different-ledger exchange. A viable state result
requires all of the following: identity replacement is neutral; same-ledger
exchange preserves the target ledger; and different-ledger exchange increases
the donor-ledger likelihood or exact report. Even that result establishes only
causal influence on ledger *report*, not flexible downstream use.

The local raw-200k baseline is cleanly negative over four bidirectional
language-only pairs: **0/8** baseline or same-state exact reports, **0/8**
mismatch donor reports, and zero positive donor-vs-target mismatch margins.
Equivalent and distinct prompt-boundary states are almost indistinguishable
at this layer (mean cosine **0.9747** versus **0.9734**); the mean
donor-minus-target mismatch log probability is **-13.50**. Artifact SHA-256:
`ac6f42ffa36e089afa2ab2da1a9b9b0087287393ab54e9cb9728b15d5852af60`.
The earlier one-pair smoke remains preserved as a path check. This small
baseline is still not a high-power estimate; the aligned, CE-only, and
wrong-state models must each receive the same 50-pair audit after their normal
behavioral transfer gate.

### Conditional Follow-Up: Contrastive State Geometry

The raw baseline also exposes why positive-pair alignment may be too weak:
same and different states have nearly identical cosine geometry. The trainer
therefore supports an optional symmetric InfoNCE term over a distinct-ledger
batch: compile must identify its own reflect state among the batch, and vice
versa. Unlike positive-only alignment, this simultaneously attracts equivalent
descriptions and repels other ledger states. It logs positive and hardest
negative cosine separately, and it refuses duplicate ledger states in a batch.

This is a conditional successor, not a replacement for the attraction-only
canary. First compare CE-only, same-state attraction, and wrong-state
attraction with identical data and updates. If same-state attraction fails its
language causal gate while the raw geometry remains collapsed, the next
isolated candidate is same-state attraction plus contrastive geometry. It must
beat all three prior arms on language transfer and activation-exchange
criteria, then be evaluated on values and delta. A successful contrastive loss
or lower same-state cosine is not a capability result.

### New Hypothesis: Native Residual Relay

The earlier continuous-memory and CPR branches are closed: they added learned
slots or packet machinery, then failed shuffled-source causal controls. The
next experiment must not add a second model around Shohin and call that
reasoning. **Native Residual Relay (NRR)** instead uses one residual the
existing transformer already computes. There are no relay parameters, slots,
state parser, external readout, or source K/V cache in the downstream pass.

For a source description `S`, encode `S` only through a selected layer `L` and
take that layer's final residual `h(S)`. The remaining blocks then process a
fresh sequence `[h(S), event, query, answer]`; source tokens never enter that
suffix computation. This creates a physical information cut: gradients can
teach `h(S)` to be a useful compact state, but the suffix cannot retrieve a
forgotten lexical source through attention. At inference the identical native
two-pass operation is used. The ordinary `GPT.forward` and flagship remain
unchanged.

NRR is deliberately stronger than response-level state SFT. Each synthetic
world supplies independently worded equivalent sources, a counterfactual
source with one changed fact, source-free forward events, inverse-delta
questions, and two distinct readouts. A model passes only when all of these
are measured on held-out language/value/delta regimes:

1. a relay from either equivalent source gives the same correct downstream
   answers;
2. a counterfactual relay changes the answers in the predicted direction;
3. a zero relay and a shuffled-world relay fail materially; and
4. the source text is absent from the suffix by construction, verified by an
   execution-level no-KV/no-source unit test.

The current implementation is limited to the no-parameter relay primitive and
its hard-cut test. It has **no checkpoint, data corpus, H100 run, or result**
yet. The PSA comparison remains useful as an inexpensive test of whether a
standard representation objective already solves the problem; NRR is the
separate, more ambitious causal-bottleneck route if it does not.

### Conditional Next Hypothesis: Counterfactual Reflection Route

An exact external carrier, even if it passes V2, would still be an explicit
tool-use skill. The next question is whether a small model can be trained to
*prepare a useful state without being rewarded for printing a reasoning trace
on the ordinary answer path*. This is the project-specific adaptation of the
paper's counterfactual-reflection idea.

For each solver-verified source record, construct two continuations that share
the entire source prefix:

1. **Direct route:** request the final source-visible answer and supervise only
   `answer=<integer>`; no ledger and no `<think>` token is allowed in the
   target.
2. **Interrupted reflection route:** ask what exact portable state the model
   would report if stopped before answering, and supervise only the strict
   `ledger:P=<integer>;Q=<integer>` carrier.

The practical augmentation arm receives both routes. Its control receives only
direct routes, resampled to match supervised answer-token count, updates,
learning-rate schedule, source records, and prompt lengths. That tests whether
reflection improves a direct-answer foundation rather than merely adding more
supervision. A stricter paper-faithful arm starts from that same direct-answer
foundation and then receives **only** reflection-turn loss; a length- and
token-matched neutral auxiliary continuation is its control. Neither arm is
ever asked for a reflection at ordinary evaluation time.

Held-out scoring first asks the direct route only. A reflection benefit is
credible only when it improves unseen-label direct answers **without** emitting
a ledger, beats its token-matched control, and also succeeds when explicitly
interrupted and routed through the exact-carrier transport gate. This creates a
falsifiable route to an internal preparatory representation rather than
equating visible text with thought.

If the reflection model merely improves the interrupted route but not the
ordinary direct route, it is an output-format skill and is rejected. If both
models improve equally, the auxiliary reflection branch has no demonstrated
value. Any later residual/cache intervention must be evaluated against these
behavioral controls; neither a probe nor a logit lens is accepted as a shortcut
to a thinking claim.

The CPU-only substrate in `train/counterfactual_reflection_protocol.py` now
makes this contrast mechanically testable without creating a corpus or
allocating a GPU. It defines a source-visible direct answer, a counterfactual
interrupted reflection whose response is only an exact post-change state, and a
fixed-shape neutral auxiliary continuation that contains no source-specific
numeric task state. A future data builder must prove tokenizer-level target
budget matching before the control is eligible. Its future source-dropped consumers may forward one full model-authored
state by literal replacement but cannot parse, calculate, repair, or choose it.
`train/test_counterfactual_reflection_protocol.py` covers those boundaries.
This is protocol groundwork only; data generation, SFT, and evaluation remain
blocked on a positive exact-carrier causal result. V2's held-out failure leaves
that gate closed.

### Conditional Context Mechanism: Reversible Semantic Checkpoints

Only after exact transport and the reflection control have a positive causal
result should the project test bounded-context scaling. The candidate is a
**reversible semantic checkpoint**: after a fixed number of events, the model
authors one bounded ledger plus two independently checkable readouts. The old
history is discarded, the ledger is used as the sole prefix for the next
window, and a later query must recover the same state across two consumers.

The checkpoint is not trusted because it is short. At every reset the evaluator
must test a model-authored checkpoint swap between two histories, a zero
checkpoint, a P/Q mismatch, and a replay from the original history. It must
also account for total source tokens, checkpoint tokens, mutable tokens,
prefill work, retained KV bytes, and task accuracy as context length increases.
The mechanism earns a context-scaling claim only if it maintains causal state
utility after resets at a fixed prompt budget; it is otherwise just lossy
summarization.

### Conditional Next Primitive: Interchangeable Semantic Ledger

If V10A learns its five families but fails the cross-family composition suite,
the diagnosis is not simply "needs more examples." Its current traces are
family-specific prose: a multiplication trace, a place-value trace, and a
repair trace have no enforced shared object that a later operation must consume.
That allows separate local programs without an interchangeable semantic state.

The proposed **Interchangeable Semantic Ledger (ISL)** is a deliberately small,
token-native state interface that a model must author and then use. A single
ordinary-language record is compiled to a canonical ledger containing named
values, operation-ready values, and immutable identity fields. The source
record is then absent. A second prompt can ask a distinct consumer to update
the ledger, answer a different query about it, or evaluate a counterfactual.
The controller only forwards exact model-emitted text; it never normalizes a
value, selects a field, or performs an operation.

The requirement is *interchangeability*, not ledger formatting:

1. Multiple unrelated source descriptions must compile to the same typed
   ledger when they denote the same state.
2. The same model-authored ledger must support at least two disjoint consumers
   such as an arithmetic update and an indexed/value query.
3. A paired counterfactual ledger swap must change each consumer's output in
   the solver-predicted direction; zeroed, mismatched, syntax-only, and
   label-permuted controls must fail on the same consumers.
4. Evaluation independently holds out values, source language, field names,
   downstream operation combinations, and multi-step lengths. No exact ledger
   syntax is sufficient without the causal swap result.

V10A failed its own bridge holdout and failed composition, so ISL is explicitly
held. A richer ledger would only add a template before the model has shown that
it can transport even two simple semantic values. ISL can become a controlled
ablation only after the two-value basis passes closed-loop source deletion,
multi-consumer use, and paired counterfactual controls.

If ISL later passes source-deleted, multi-consumer, and counterfactual gates,
the ledger becomes the only admissible input to a context-scaling experiment.
Its exact tokens can be held as a KV anchor, and periodic re-anchoring must be
model-authored and pass the same swap/zero controls. That would measure a real
bounded state-compression mechanism without claiming an extended context window
or allowing an external summarizer to do the reasoning.

### Post-Bridge Semantic Capsule Gate

The existing semantic-capsule corpus is not another raw capability test. Raw
and broad V9 both score zero because neither can initially form a valid capsule;
those controls reject a claim that ordinary pretraining or generic reasoning
formatting already supplies context compression. The corpus remains valuable as
the next **serial** mechanism test after a semantic primitive has been taught.

`sft_semantic_capsule_v11a.sbatch` therefore refuses to start from raw weights.
It requires a V10A checkpoint together with its full 500-case bridge result and
full 500-case cross-family composition result, each bound to that exact
checkpoint and its immutable evaluation data. Admission requires at least
250/500 bridge answers, 200 solver-derived intermediate-equation contracts, at
least 25 such contracts in every bridge family, at least 40 bridge answers in
every family, at least 50/500 cross-family answers, and at least five
composition answers in every family. These are deliberately stronger than
the rejected V9 signal and stop the capsule corpus from laundering a narrow
template result into a context-scaling claim.

Only then does one isolated capsule epoch teach source-deleted write, update,
repair, and readout actions. The SFT is completion-bound to the exact
controller prompt carried in every row; it does not add a second generic
`Question:/Answer:` wrapper. Its held-out 4/8/12-step controller evaluation
uses model-generated capsules only. CBC follows only if this result is nonzero:
the capsule protocol measures persistence across resets, while CBC's paired
counterfactual compiler measures whether a resulting state is causally
interchangeable across worlds. Neither result alone establishes broad reasoning.

### Conditional Engineering Substrate: Causal KV Anchors

The normal cache is a useful engineering mechanism but **not** a context
compression result. Once a model has authored a discrete semantic anchor, its
exact tokens can be prefetched once and their KV cache retained while later
events and model-authored state updates are appended. This eliminates repeated
prompt transmission and repeated prefix projection, but it still has a linear
attention-cache footprint and does not by itself extend the model's context
window or create reasoning ability.

`train/causal_kv_anchor.py` is intentionally a small, no-training substrate
for this future experiment. It transports only exact tokens; no controller
parses, summarizes, selects, computes, or repairs their meaning. Because this
model's cached-attention fast path is causally exact only for a single new token
at a time, all updates are serially appended and mechanically compared against
full replays of the same token history. The original root cache remains
immutable so a matched anchor swap or zero-cache control cannot be hidden by
in-place state mutation.

This may be tested only after V10A and the semantic-capsule route establish
nonzero, source-deleted semantic transport. A valid experiment must compare:

1. **Exactness:** cached serial decoding and full replay have matching logits
   for every appended token; otherwise it is an invalid inference path.
2. **Semantics:** a model-authored anchor improves held-out source-deleted
   readout over a no-anchor control, while replacing it with a matched
   counterfactual anchor changes the answer in the solver-predicted direction.
3. **No controller shortcut:** token histories are forwarded verbatim; the
   controller is prohibited from converting facts to states or selecting among
   anchor candidates.
4. **Net resource accounting:** report original prompt tokens, model-authored
   anchor tokens, mutable tokens, cache bytes, prefill work, and full-replay
   work. `resource_accounting` reports exact token-position and per-layer
   causal-attention-pair counts for cached serial append versus full replay;
   GPU wall time remains a separate measured quantity. KV reuse is useful only
   if it saves end-to-end session work without concealing a longer context or
   another model call.
5. **Periodic re-anchoring:** any attempt to exceed a fixed context budget must
   ask the model to author a new compact anchor and rerun the same swap/zero
   controls. Copying an external summary into a fresh cache is disallowed.

This separates a real potential systems gain (persistent exact attention to a
model-authored state) from the rejected continuous-packet branches and from a
false claim of unlimited context. It becomes a context-scaling mechanism only
if model-authored re-anchoring preserves counterfactually useful state across
resets.

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

The raw-200k four-layer baseline is negative. From 80 hash-separated discovery
gradients (eight per digit), all four layers have exactly 20/200 top-1 readout
on separate contexts, the ten-way chance count. Their 40-direction matched
causal effects above a shuffled-label control are +0.108, +0.217, +0.240, and
+0.286 log-odds at layers 13/17/21/25, respectively, but their descriptive
SEMs are +0.252, +0.339, +0.326, and +0.330 with only 20-21/40 directions
favoring the signal. The immutable artifact is
`artifacts/eval_history/restricted_jacobian_digit_lens_raw200k_mps_l13_17_21_25_d8_r20_p4.json`,
md5 `d5c61ead369acc0e1fbf0daf6006cb53`. Thus the raw model has no detectable
reusable, verbalizable next-digit direction under this restricted method. That
is a constraint on the project hypothesis, not a full J-lens result or proof
that all internal state is absent; a distributed or nonlinear code can evade
the test.

The job wrapper remains an isolated diagnostic, not a semantic workspace probe
or evidence of general reasoning. It should next run only on a checkpoint that
first passes V10A's behavioral semantic-primitive gates. A positive restricted
result still cannot authorize CWI or a capability claim without the
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

`train/counterfactual_workspace_protocol.py` now makes the reflection premise
mechanically precise. Starting from a fixed tape and current register, it
derives the legal successor and then constructs a candidate that is
grammar-valid but differs in exactly one of four semantic ways: carry, active
result digit, program counter, or immutable tape. The supervision target
reports the verdict and the expected/observed value at the active position.
Thus the reflection cannot succeed by detecting malformed text or predicting a
constant `illegal` label.

The CPU-only builder/auditor pair now has a full local dry-run against the
admitted factor corpus: 682,957 train reflection rows and 52,200 held-out
reflection rows, with every row semantically rederived by an auditor that does
not import the builder. It found 0 malformed rows, duplicate identities,
normalized duplicate prompts, exact prompt hits, or 13-gram train/held-out
hits; it covered all 3,400 legal local contexts and preserved all 26,100
base/counterfactual held-out foil pairs. The durable Stokes wrapper refuses to
write over artifacts and requests CPU only. This is still corpus admission,
not an SFT, checkpoint, H100 allocation, workspace result, or relaxation of
the STRR/CWI gates above.

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
No model checkpoint, controller rollout, or GPU job has been created. CBC
remains conditional on a positive causal-result gate rather than a format
score.

The matching transport-only controller is also preflighted.  It accepts only
a parsed model-emitted state, renders the next source-free prompt around that
text, and halts on an incorrect or malformed emission.  Its test covers
primary rollout, inverse-delta checks, same-world compiler interchange, and a
real cross-world counterfactual carrier swap on the identical source-free
query.  The swap takes the *model-emitted counterfactual terminal state* and
requires the query to produce the counterfactual answer rather than the normal
answer.  Re-reading the normal state would only restate ordinary rollout
accuracy and is explicitly not counted as a causal result.  A bad first state
terminates the run; it is not canonicalized into a solver answer or repaired.
This makes CBC's later state-necessity measurement executable rather than an
informal data claim.

### CBC Build and Evaluation Readiness

The CPU-only Stokes build wrapper creates a fresh candidate only after an
independent audit passes: it requires zero malformed rows or held-out episodes,
zero exact and 13-gram train/held-out prompt hits, all five training roles
(two compilers, update, inverse delta, readout), and all 4/8/12-step held-out
regimes.  The initial build uses 4,000 episodes per train domain and 200 per
held-out domain, which exceeds 100,000 training rows without consuming a GPU.
Stokes job `738468` completed this build without consuming a GPU. The immutable
candidate contains **16,000** train episodes / **256,000** training rows and
**600** held-out paired-counterfactual episodes: 198 length-4, 201 length-8,
and 201 length-12. The independent audit found zero invalid rows or held-out
episodes, duplicates, exact split prompts, or literal 13-gram split hits.
The train / held-out SHA-256 values are
`6013f5118b00c3b88afbe2af892b7e25867a4a5e5a2d1c5882ee635564326c02` /
`163e60398f239ab4058129ef135350d3d5509ea1dc309417d9f85dabbdf59256`.
The full data and the two small admission records are mirrored locally and on
Newton. This is data admission only, not an SFT or capability result.

`train/eval_counterfactual_bisimulation.py` now measures the controller with a
checkpoint on a deterministic balanced slice.  It records compilation A/B,
source-deleted closed loops, inverse-delta checks, same-world interchange,
counterfactual interchange, and the true cross-world carrier intervention.
No SFT should be proposed from CBC unless those held-out metrics show a
nonzero causal state-necessity margin over malformed, swapped, and
counterfactual-mismatched carriers.

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
