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

### Conditional Direct Counterfactual Reflection: Operator Semantics, Not a Hidden Carrier

The direct-only operator-anchor experiment is the first clean test of whether
the previous COTA failure was caused by paired-answer grammar rather than by
the entire direct trace curriculum. Only if its pre-registered gate preserves
ordinary direct decoding and produces a bounded operator signal may the next
experiment run.

That follow-up is intentionally different from the closed source-dropped
ledger/workspace branches. It keeps the complete natural-language problem in
context and never transports a hidden state. During training only, an external
interruption reverses one named operation and asks for the operation labels,
the state immediately before it, and the exact counterfactual next state. The
original task's answer is *not* a target on that interruption. Normal
evaluation asks only the original direct question, with no reflection request.

The experiment must have two otherwise identical arms:

1. **Numeric reflection:** the interruption target contains the task-derived
   counterfactual state.
2. **Neutral structural control:** it retains the identical operation-label
   and fixed-width reflection surface, but both state fields are zeros and so
   contain no task-derived numeric information.

The comparison therefore asks a falsifiable question: does supervising a
counterfactual numeric consequence improve unreflected direct operation
selection beyond reflection grammar and operation-name exposure alone? A
credible result requires the numeric arm to beat the neutral arm on the frozen
wording/value/full factor suite and direct transcripts, with no response-mode
leakage, arithmetic/base collapse, or RG regression. It would still establish
only a bounded operator-semantics improvement, not general intelligence or a
workspace.

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

That matched matrix is now complete and rejects PSA. CE-only, same-state, and
wrong-state score respectively **1/100**, **2/100**, and **1/100** strict
language-only causal passes. Their full-replay 50-pair activation audits are
all zero on baseline/identity/same exact target reports, mismatch exact donor
reports, and positive mismatch donor margins. Same-state attraction reduces
its own objective but does not beat the deliberately wrong-state control; both
train to cosine about 0.9998. The 2/100 is well within this small test's noise
and has no causal support. Values/delta scoring and contrastive PSA are
therefore not justified. This closes PSA as an ordinary representation-loss
route and leaves NRR as the next causal-bottleneck test.

### Conditional Follow-Up: Contrastive State Geometry

The raw baseline also exposes why positive-pair alignment may be too weak:
same and different states have nearly identical cosine geometry. The trainer
therefore supports an optional symmetric InfoNCE term over a distinct-ledger
batch: compile must identify its own reflect state among the batch, and vice
versa. Unlike positive-only alignment, this simultaneously attracts equivalent
descriptions and repels other ledger states. It logs positive and hardest
negative cosine separately, and it refuses duplicate ledger states in a batch.

This hypothesis is now closed without a contrastive run. The matched controls
showed that an attraction objective can make even deliberately wrong states
nearly identical while leaving causal behavior null. A stronger geometric loss
would only optimize the same unvalidated surrogate. NRR changes the actual
information path instead of adding another similarity term.

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

The no-parameter relay primitive and its hard-cut test passed only as
infrastructure. Its CPU-only v1 corpus is admitted on shared Stokes/Newton
storage: 30,000 train rows (SHA-256
`bac1e8d041abbfefa892056302a8d78c14abd0d31dd1694e9bc92aefac2fe03c`) and
2,000 held-out rows (`1d8b633713fff41b331e7c2728e9c0aa3ae307a7b99622c526d99d6dc84120f2`),
with zero duplicate prompts and zero exact or word-13-gram cross-split hits.
The 12-update H100 launch canary exercised gradients and serialization from
immutable raw-200k weights; it was not a capability result.

**Closed result, 2026-07-14: NRR v1 is rejected.** The two full isolated
one-epoch arms, `L=13` (`688533`, 7,465 updates, checkpoint md5
`f721645e5b5c38622cf2bc55563957b9`) and `L=19` (`688534`, 7,465 updates,
checkpoint md5 `1616cf2eb21f656e4e781093fb524dfe`), both score **0/500** on
the frozen combined held-out causal evaluation. That is 0 normal, paraphrase,
counterfactual, direct-bypass, and strict-causal answers for both arms.
The relay is not inert: for L19, replacing it with zero or a shuffled relay
changes the emitted answer on 499/500 and 500/500 cases respectively, and a
counterfactual source changes the prediction on 355/500 cases. But it has no
semantic success: it neither preserves the answer across paraphrase nor
updates it correctly under a counterfactual. A L19 in-distribution diagnostic
is also inadequate: only 21/200 normal, 23/200 paraphrase, 13/200
counterfactual, and 2/200 strict-causal cases. Low training loss therefore
represented local token formatting and near-number imitation, not a usable
latent state. The separately admitted language/value/delta factor suite is
retained for methodology but is not worth H100 time after this primary gate.
No continuation or recurrence may be built from this mechanism.

### Conditional Extension: Native Relay Recurrence (Blocked)

A one-step relay is only a compression test. The actual context-scaling
hypothesis is to reuse the transformer tail as a recurrent state transition
without adding an RNN, memory slots, a controller, or a serialization channel:

`h_0 = Encode_L(source)`

`h_(t+1) = Tail_(L+1..N)([h_t, event_t])[-1]`

`answer = Tail_(L+1..N)([h_T, query])`

The recurrent state would be native and every transition would use the same
frozen architecture and source-free hard cut. That hypothesis is now
**blocked, not pending**: NRR v1 scored 0/500 held-out strict causal and 2/200
on the in-distribution diagnostic, far below the advancement gate below.
Implementing recurrence would merely compound an unlearned state channel.
Retain these specifications as a falsification record, but spend no further
training time on native-relay recurrence unless a materially different
one-step state mechanism independently clears the same gate.

For the current one-step depth sweep, "substantial" is pre-registered as at
least **300/500 strict causal** cases on the frozen combined held-out set,
with each of normal, paraphrase, and counterfactual correctness at least
350/500 and each zero/shuffled relay recreating the normal answer at most
25/500. A candidate meeting that bar must still clear the newly separated
language, values, delta, and combined factor sets before recurrence is
implemented. The full-source bypass remains diagnostic only and cannot satisfy
any of these thresholds.

### New Hypothesis: Counterfactual Residual Algebra

NRR showed that a single source residual can influence a suffix without
becoming a semantic state. The next hypothesis therefore does **not** ask for
another answer conditioned on another hidden vector. It asks the model to make
an *intervention* in residual space work across unrelated worlds.

**Counterfactual Residual Algebra (CRA)** exports a short tape of the last
native residuals from a fixed, ordinary source anchor, rather than adding slots
or parameters. Let `Z(x, y)` be that tape for a world with two latent facts.
For three independently rendered sources,

`A = (p, q_a)`, `A' = (p + d, q_a)`, and `B = (r, q_b)`,

the source-free suffix receives only

`Z(B) + Z(A') - Z(A)`

and a question about the unshown target world `(r + d, q_b)`. It must answer
several readouts (field, sum, difference, and later affine readouts). Thus a
successful model cannot merely encode an answer-like number in one residual:
the residual *difference* for a fact change must transfer over a different
background, and the suffix must decode the composed result with all source text
and all source K/V states absent.

This is intentionally an end-to-end causal objective, not a cosine,
clustering, attraction, probe, or text-state loss. The only supervised target
is the answer produced after the residual intervention. It is also distinct
from the failed DRS/CPR/PSA/NRR branches: no string is emitted or parsed as
state; no learned slot, controller, or packet is added; and a wrong residual
algebra operation has a solver-verifiable wrong answer.

The first candidate begins with a small no-carry arithmetic curriculum so the
test isolates semantic composition rather than the raw model's known multi-
digit arithmetic deficit. It must then clear all of the following before any
larger-value, event-transition, or recurrent version exists:

1. at least **300/500** strict compositional-causal cases on a frozen combined
   held-out set;
2. at least **350/500** correct each for normal source renderings,
   independent paraphrases, and a counterfactual `d` substitution;
3. no more than **25/500** answers recreated by a zero or shuffled residual
   tape; and
4. separate language, value, delta, query-family, and two-edit commutativity
   factor evaluations, all constructed before training.

This is a project-specific falsification attempt, not a claim of a new
general technique. If the residual algebra does not pass the first primitive,
it closes with NRR rather than acquiring a recurrence, a public benchmark, or
a post-hoc story.

### Conditional Fallback: Paired Counterfactual Discrimination

Raw-model geometry gives this first CRA arm a specific, falsifiable failure
mode: the residual differences for `+d` and `-d` are almost collinear even
though their answers must diverge. Ordinary one-target CE could therefore
lower loss by making the source-free suffix sensitive to a broad
"there was an edit" template without making the *direction* of that edit
functional.

The conditional **paired CRA** fallback keeps the exact native tape, hard cut,
source corpus, and no-extra-parameter rule. For each episode it decodes both
`Z(B) + Z(A') - Z(A)` and
`Z(B) + Z(A'_{cf}) - Z(A)`. Besides CE for both solver answers, it applies a
per-example margin only at the output distribution:

`NLL(correct | tape) + m < NLL(opposite-counterfactual-answer | tape)`.

This is not an activation-attraction loss and does not assert that residual
vectors should have a particular cosine. It only rejects a model that assigns
the same completion preference to both causal interventions. Each example's
margin is computed independently, so errors cannot cancel across a minibatch.
The fallback can run only after a fully evaluated ordinary CRA rejection. It
must then clear the same behavioral combined and factor gates; improved
training loss or teacher-forced margin alone cannot advance a context or
reasoning claim.

### Conditional Next Mechanism: Counterfactual Chart Closure (C3)

If paired CRA learns the sign of an edit in-distribution but fails the language
or combined factors, the likely defect is deeper than a missing contrastive
answer: a residual difference is still tied to the particular wording that
produced it. The next candidate is therefore **Counterfactual Chart Closure
(C3)**. A *chart* is simply one natural-language rendering of the same small
two-field world; it is not a learned module, an external state carrier, or a
new model parameter.

For a source state `A`, an edit `d`, a donor `B`, and two independently
rendered charts `alpha` and `beta`, C3 trains and evaluates only functional
output constraints such as:

`Z(B^gamma) + [Z((A+d)^alpha) - Z(A^alpha)]`

and

`Z(B^gamma) + [Z((A+d)^beta) - Z(A^beta)]`.

Both source-free tails must answer the same target world `B+d`. Crucially, a
closed cross-chart path must recover the donor answer:

`Z(B^gamma) + [Z((A+d)^alpha) - Z(A^alpha)] + [Z(A^beta) - Z((A+d)^beta)]`.

The model is never rewarded for a residual cosine, a vector norm, a parser
output, or an explanatory string. It is rewarded only when independently
compiled paths cause the correct tail answer, and it is penalized when a
same-shaped but semantically wrong path reaches that answer. This turns
surface-language invariance from a post-hoc probe into a *path-independence*
requirement on the causal operation itself.

C3 is deliberately conditional on a diagnostic paired-CRA outcome, not on
low loss. A C3 corpus may be admitted only if it has disjoint value ranges,
chart vocabularies, question forms, and exact source bundles across splits.
Its minimum behavioral gate is pre-registered before any training: on a
frozen 500-case jointly held-out suite, at least 300 strict cases must get
both independently compiled edit paths and the cross-chart closed path right;
each direct edit path must reach 350/500; and zero, shuffled, chart-mismatched,
or wrong-inverse paths may recreate the correct answer on at most 25/500.
Separate language, values, edit-magnitude, donor-chart, and two-edit
commutation factors remain mandatory. A pass would establish only a
transportable source-free intervention primitive, not general reasoning. A
failure would close residual-algebra work instead of inviting another format
or geometry loss.

### Next Admitted Research Question: Finite-Query Residual Basis (FQRB)

The completed CRA factor matrix changes the question. Its support-matched
value control keeps every answer string in the training vocabulary but remains
at zero strict causal cases, while the delta factor reaches 208/500 strict.
The primary defect is therefore not merely an unseen answer token and not
primarily the sign of an edit: the model does not yet carry a reusable numeric
source state through the residual composition.

**Finite-Query Residual Basis (FQRB)** tests that prerequisite without asking a
small model to emit an unbounded integer. Each source still contains a base
world, an edited base world, and a donor world, and the decoder still receives
only the source-free composition

`Z(donor) + [Z(edited) - Z(base)]`.

Instead of a single direct numeral, independently sampled suffix consumers ask
for one bounded, solver-derived property of the composed target: signed tens
digit, ones digit, sign, parity, or the relation between the two target
fields. The answer alphabet is fixed and fully present in training. For every
episode the normal and counterfactual edits are selected to change that
consumer's answer, so an edit-insensitive tape cannot pass by returning a
constant class.

This is not a parser, an external calculator, a vector-alignment objective, a
new parameter, or a visible trace. It is functional tomography: the same
source-free native state must support several incompatible finite readouts.
The multi-consumer condition matters. A tape that answers `parity` but cannot
also answer `ones` and `relation` has not established a reusable number state;
it has learned a query-specific classifier.

The admitted train split uses signed two-digit source fields and a fixed finite
answer alphabet. Its first held-out split uses unseen full source bundles and
unseen source wording but no unseen answer classes. Exact three-source bundles
and held-out prompt n-grams must remain absent. A later dedicated magnitude
factor, rather than the first combined score, will move source fields to
three-digit values while retaining the same finite output alphabet. Evaluation
will re-use each encoded source triple for all five suffix consumers, then test
normal, paraphrase, counterfactual, zero, whole-group shuffled, and wrong-query
controls.

Before a full arm is submitted, CPU generation and a separate audit must prove
the coverage and split claims. A future one-epoch isolated arm can advance only
if a frozen 500-episode combined evaluation has at least 300 strict episodes,
each consumer is at least 350/500 on its applicable direct path, all five
consumer answers are jointly correct on at least 300 episodes, and zero,
shuffled, or wrong-query tapes recreate a correct answer on at most 25/500.
The same thresholds apply to the answer-support-matched and language factors;
the three-digit magnitude factor is reported separately as the first true
numeric-length generalization test. A pass would show only a bounded causal
numeric basis. It would be a necessary but still insufficient precursor to a
general reasoning claim.

### Conditional FQRB Ablation: Phase-Aligned Anchor Tapes (PAAT)

FQRB source records are ordinary token sequences, so their terminal anchors
can land at different RoPE positions when a signed or multi-digit value changes
tokenization. Residual arithmetic across `base`, `edited`, and `donor` then
adds states from different positional frames, while the tail may decode that
same tape from positions zero onward. That is a concrete mechanism failure
hypothesis, not an explanation after the fact.

**Phase-Aligned Anchor Tapes (PAAT)** is a zero-parameter ablation. It
right-aligns each source into the same fixed zero-embedded positional window,
so all ordinary source tokens and the terminal anchor use a common endpoint.
The source-free suffix then continues from the anchor's true RoPE positions.
The inserted prefix has no token ids, learned vector, semantic content,
attention mask exception, controller, or external computation. It merely makes
the residual coordinates being added commensurate.

PAAT is eligible only if the current FQRB arm fails its combined or
source-tuple gate. Its experiment must preserve the frozen corpus, layer,
tape length, model initialization, batch/update count, optimizer schedule,
and evaluator; `source_window` is the only changed variable and is bound into
the checkpoint metadata. The same combined, core, magnitude, zero, shuffle,
wrong-query, and transcript gates apply. Equal or worse performance rejects
positional misalignment as the limiting explanation. A positive result would
still establish only a bounded phase-consistent latent basis, not a reasoning
system.

### Conditional Next Hypothesis: Ephemeral-Codebook Latent Interrogation (ECLI)

FQRB's five readers are stronger than one fixed numeral head, but they remain
fixed readers. A model could still learn five template-specific classifiers
whose outputs happen to depend on a source tape. **ECLI** adds a late-binding
test before any semantic or reflection claim: for every source world, the
source-free suffix supplies a fresh arbitrary binding table from each of the
thirteen FQRB semantic classes to an opaque code word. The model must return
the code word, never the semantic class directly.

The source triple and native composition remain unchanged:

`Z(donor) + [Z(edited) - Z(base)]`.

All five consumers of a world share one codebook, while every world receives a
different permutation drawn from sixteen ordinary code words. The source is
absent from the binding-table suffix. Thus a successful output has two
separable requirements: recover the correct semantic property from the tape,
then use the *current* query-local table to bind that property to a code word.
The codebook is not a parser or tool: it is literal prompt text, and all
targets are ordinary next-token targets.

The held-out split must make source bundles, wording, and complete codebook
permutations disjoint from training while retaining the same code-word
vocabulary. Every row also has a codebook-swap control: with the identical
source-free tape and question, two semantic entries of the table are swapped,
and the required output must change to the newly bound code. Normal,
paraphrase, counterfactual, zero, whole-group shuffle, wrong-query, and
codebook-swap controls all count in a group-strict score.

ECLI is eligible **only** after FQRB passes its combined held-out and unseen
source-tuple gates. Its frozen 500-world admission threshold is at least 350
correct cases on every consumer for normal, paraphrase, counterfactual, and
codebook-swap paths; at least 300 worlds jointly strict across all five
consumers; and at most 25 zero, shuffled, wrong-query, or codebook-swap
normal-answer recreations. A pass would establish only a bounded,
late-bound latent interrogation primitive: it would be evidence that a query
can modulate how one source-free state is read. It would not establish
open-ended reasoning, language understanding, or a general workspace.

### Conditional Research Direction: Latent Interrogation Cascade (LIC)

The missing ingredient after a late-bound readout is **intermediate use**. A
model can answer arbitrary probes about a source-free tape and still fail to
prepare that state before an ordinary direct answer. **Latent Interrogation
Cascade (LIC)** is a project-specific attempt to bridge that gap without
teaching visible chain-of-thought or giving the evaluator a parser.

LIC has two strictly separated routes over the same solver-generated world:

1. **Interrogation route:** encode the world once, remove it, and answer a
   randomized sequence of late-bound finite probes from the native tape. Probe
   order, binding table, and selected intermediate property vary per world.
2. **Silent-action route:** encode the ordinary source-visible question, append
   a fixed small number of differentiable native latent-rollout states, then
   supervise only the ordinary final answer. No probe text, binding table,
   ledger, or `<think>` target appears on this route.

The proposed training objective couples the routes only through the model's
shared weights. It does not copy a probe answer into the direct prompt, add a
controller, or decode a model-produced state externally. The key comparison is
a compute- and token-matched **neutral-latent control**: it receives the same
number of latent rollouts and direct-answer updates, but its auxiliary suffixes
are source-independent neutral continuations rather than counterfactual
interrogations. If both improve equally, LIC has no evidence of a reasoning
benefit.

LIC is not eligible until ECLI has passed its multi-reader, codebook-swap, and
source-control gate. A future pass requires direct-answer improvement on
unseen source language and unseen query compositions *without* a codebook or
probe prompt at evaluation, exceeding the neutral-latent control, and
remaining sensitive to a pre-registered source-state intervention. That would
be evidence for a small, silent preparatory computation. It would still fall
well short of a claim of open-ended reasoning, and a failure would reject the
interrogation-to-action bridge rather than invite a larger trace corpus.

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

### Conditional Context Primitive: Causal Residual Count-Sketch (CRCS)

The rejected packet-memory and semantic-ledger routes share a structural
weakness: they ask the small model to serialize a complete state before it has
shown a transferable internal state. **Causal Residual Count-Sketch (CRCS)**
starts from the opposite end. It retains no model-authored text and introduces
no learned memory slot. Instead, each fixed-format event is encoded once to a
native anchor tape, then placed into a fixed number of deterministic signed
residual lanes:

`S[b] = sum_i sign(i,b) * Z(event_i)` for `b = 1..B`.

The event ordinal determines its public, nonsemantic hash lane and sign; the
controller never chooses a fact by meaning, computes an answer, or rewrites a
state. A later source-free query receives the same fixed-width lane tape plus
its ordinary query tokens. Multiple independent lanes make interference a
measurable capacity property rather than an opaque learned-memory claim. The
model must learn to use the query and native lanes to recover one event or
combine two events. The representation, not an external lookup, carries the
content.

CRCS is deliberately more demanding than ordinary retrieval. Its first
solver-derived curriculum would ask late-bound finite questions about one
event, then about a two-event relation, with all original events removed. The
held-out suite must grow from four training events to eight and sixteen events
at the *same* lane budget, change event wording and assignments, and use
disjoint codebook permutations. It must compare the signed multi-lane sketch
to a token/compute-matched flat residual sum; otherwise an apparent gain could
be ordinary extra capacity rather than structured compaction.

The causal controls are required at each length: zero every lane, shuffle
event-to-ordinal assignments, invert a lane's sign pattern, replace a query
with a mismatched event query, and swap exactly one event between paired
histories. A positive result needs a source-free margin over all controls,
per-event and two-event readout, and a non-collapsing length curve at fixed
lane width. It must also report retained residual bytes and prefill/decoding
work. Passing would establish only a bounded, fixed-width native context
sketch, not unbounded memory or general reasoning.

CRCS is not currently admissible. It requires ECLI to establish that a
source-free latent state can be interrogated through a current query and
binding table. That prerequisite prevents a count-sketch failure from being
misread as a hashing problem when the model cannot yet read one latent state.

The CPU-only `pipeline/generate_causal_residual_count_sketch_v1.py` is staged
for that gate, but no CRCS training data or GPU job is authorized yet. It
refuses any parent assessment other than
`bounded_ecli_late_binding_candidate`, then constructs 12,000 four-event
training histories and 500 held-out histories of eight or sixteen events.
Each history has five consumer questions, a fresh opaque codebook, a
counterfactual event-edit answer, and a same-history codebook-swap answer.
The builder rejects non-changing interventions and records zero exact-history,
codebook, and semantic 13-gram train/held-out overlap before it writes data.
`train/test_generate_causal_residual_count_sketch_v1.py` fixes those
admission and split-audit conditions. This is reproducible curriculum
groundwork, not evidence for CRCS or a claim that the model can reason.
`pipeline/watch_ecli_crcs_admission.sh` is the corresponding one-shot
CPU-only continuation: it may build that audited corpus only after the exact
ECLI assessment is present and positive. It never submits a CRCS training job;
any learned context claim remains separately gated on a later model result.

### Conditional Direct-Transfer Test: Counterfactual Workspace Reflection (CWR)

The raw 200k transcript audit shows that Shohin has neither a dependable
visible scratchpad nor a useful reportable intermediate state. The FQRB/ECLI
branch tests whether a narrow native state can exist without those behaviors.
Only if both stages pass, the next question is whether that state can change
ordinary, source-visible reasoning rather than merely answer an artificial
suffix. **Counterfactual Workspace Reflection (CWR)** is a narrow test of
that transfer.

For a frozen FQRB source triple and ordinary direct question, CWR appends a
counterfactual interruption, such as asking what five semantic facts should
be held in mind before answering. Training computes loss only on the
interruption's reflection, which must name the complete donor-after-edit
state. It never computes loss on the direct answer. At evaluation, the
interruption is absent: the model receives the ordinary source-visible
question and must give the answer directly. A result can therefore not be
explained by having trained that answer completion in the target context.

The arm must use held-out source bundles, wording, query templates, and
counterfactual source edits. It requires a direct answer change under the
edited source, failure under whole-source shuffle or source zeroing, and a
matched placebo-reflection arm whose reflection describes a different world.
It additionally reports a reflection-probed held-out score, but that score is
diagnostic only: the primary endpoint is a source-visible direct answer with
no reflection instruction. The same checkpoint must improve the existing
seven-task transcript audit without a reflection prompt before it can be
called a general capability gain.

CWR is intentionally not a generic chain-of-thought or answer-distillation
recipe. The prediction is mechanistic: if a reportable latent basis exists,
supervising its *future counterfactual report* should make those concepts
available while the preceding direct answer is formed. If FQRB or ECLI fails,
there is no evidence that the model owns such a carrier and CWR remains
blocked rather than becoming another ungrounded SFT run.

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

## Active Architecture Hypothesis: Verbalizable Recurrent Workspace

The next isolated experiment is a **Verbalizable Recurrent Workspace (VRW)**.
It targets a failure shared by the closed soft-token, source-dropped packet,
typed-state, and operator-trace branches: adding a carrier or teaching a trace
does not prove that the frozen decoder can read a causally necessary internal
state. VRW changes that interface while leaving the 125M base immutable.

The design is informed by the sparse, reportable workspace observations in
<https://transformer-circuits.pub/2026/workspace/index.html>, but it does not
claim to implement that paper's J-lens. Its top-k normalized unembedding basis
is an explicit late-layer verbalizability approximation and is evaluated as an
engineering hypothesis.

### Mechanism

- The pretrained base is frozen. The ordinary source prompt remains visible to
  all upper transformer blocks, so the experiment does not manufacture a hard
  source-removal bottleneck that the base was never trained to cross.
- At block boundary 19, four 96-wide slots cross-attend only to prompt
  residuals. One shared GRU cell updates those slots four times. Recurrent
  depth therefore adds compute without adding step-specific parameters.
- Answer tokens never enter scratch construction. The state is read only at
  answer-predicting positions through query-conditioned slot attention.
- The readout is projected onto the top eight frozen normalized unembedding
  directions with nonnegative mixture weights, rescaled to local residual RMS,
  and applied through one signed scalar gate. The gate starts at exactly zero,
  making the initialized wrapper exactly equal to the frozen base.
- Only adapter parameters are optimized. Checkpoints store the small adapter,
  immutable base/data hashes, architecture values, and an exact initial-adapter
  hash rather than copying or drifting the base.

### Causal Controls

The recurrent candidate and reset control use the same checkpoint, data,
seed, shape-bucketed batches, optimizer, parameter count, initialization, and
four cell executions. In the reset arm every execution starts from the learned
initial slots; their identical outputs are averaged so every call participates
in backward while no information can accumulate across steps.

Held-out evaluation keeps the source visible and measures teacher-forced NLL,
token accuracy, and exact answer sequences under:

1. adapter disabled;
2. one recurrent step;
3. the adapter's trained depth and trained recurrence mode;
4. recurrent candidate reset at inference;
5. zero scratch state; and
6. scratch states shuffled between matched-shape examples.

The locked comparator requires all of the following before autoregressive
evaluation: at least 0.05 fit-IID NLL advantage over the reset fit; at least
0.03 depth-OOD NLL advantage; at least 0.05 state-necessity margin over the
strongest zero/shuffle control; at least 0.03 within-model depth-OOD advantage
over one-step/reset inference; and exact-sequence wins in at least two of four
held-out regimes. Passing these gates only admits state-swap generation and
manual interaction. It is not itself a reasoning result. Failure closes VRW
without scaling it or modifying the protected pretraining writer.

### Implementation

The isolated implementation is in `train/causal_recurrent_scratch.py`, with
paired trainer, NLL evaluator, locked comparator, Slurm wrappers, and CPU unit
contracts. The first canary is capped at 8,192 admitted answer-only operator
examples per arm (1,024 updates at batch size eight) from immutable 200k. No
VRW checkpoint or capability result exists until both arms complete and the
hash-bound comparator runs.

### VRW Result: 2026-07-14 13:08 EDT

The bounded experiment is complete and **VRW is rejected**. The recurrent and
reset arms used the same immutable 200k base, admitted answer-only corpus,
8,192 examples, 1,024 updates, seed, 297,217 adapter parameters, and exact
initial-adapter SHA-256. The reset arm was trained and evaluated in reset mode;
the evaluator did not accidentally enable recurrence.

On 224 held-out cases, reset beats recurrence by 0.26736 fit-IID NLL and
0.26769 depth-OOD NLL. Within the recurrent adapter, four steps improve
depth-OOD NLL over one step/reset by only 0.01446, below the locked 0.03 gate.
Shuffling state between matched-shape examples changes all-case NLL by only
0.00377, below the 0.05 state-necessity gate. Neither model gets one exact
answer sequence in any regime. Every advancement gate is false, so no
autoregressive state-swap evaluation is permitted.

This rejects the specific hypothesis that a sparse token-aligned readout plus
final-answer supervision is enough to identify a useful recurrent workspace.
The reset adapter's substantially better NLL shows that the readout can learn a
one-step prompt-conditioned correction. Repeated GRU updates instead erase or
homogenize useful information, and the near-invariance to shuffled state shows
that the learned recurrent state is not prompt-specific enough to mediate an
answer. Future work must supervise or structurally identify local state
transitions; adding more answer-only recurrent depth is blocked.

## Active Architecture Hypothesis: Causal Microcode Bottleneck

The strongest positive mechanism evidence is not a public benchmark score. In
the Digitwise Recurrent Scratchpad, the first local state was correct on
497/500 episodes, including 98/100 width-eight cases, while repeated textual
rollout ended with only 275/500 correct answers. The model can select a local
transition, but serializing and rereading an exact state accumulates errors.
VRW then showed that final-answer loss does not identify a prompt-specific
recurrent latent state: reset beat recurrence and shuffled state was nearly
invariant. Causal Microcode Bottleneck (CMB) tests a different decomposition.

### Mechanism And Claim Boundary

- The 125M base is frozen. Its layer-19 hidden state at each event-line end and
  the final query-line end feeds shared operation/query classifiers. The
  compiler predicts one of nine register-relative operations per event and one
  of five readout operations per query.
- A deterministic lexical frontend extracts only the two standalone initial
  integers, at most one standalone integer per event, and line boundaries. It
  does not classify operations, bind registers, execute arithmetic, or select
  the answer. This supplied structure is reported rather than hidden.
- Execution uses two eight-digit categorical registers. Add and subtract are
  performed by one learned table over operation, carry/borrow, left digit, and
  right digit. Its complete basis has 2 x 2 x 10 x 10 = 400 contexts and 20
  digit/carry outputs. Move, merge, swap, and queries compose those local
  transitions without generating intermediate text.
- This is a narrow neuro-symbolic executor experiment. A pass would show that
  a tiny frozen LM can compile paraphrased language into an exact reusable
  internal program when lexical number extraction and an execution substrate
  are supplied. It would not show broad language reasoning or autonomous
  algorithm discovery.

### Locked Admission And Causal Gates

The immutable training source is the 96,000-row latent-operator answer-only
corpus at depths one through four. The evaluation board contains 896 disjoint
cases: fit-IID, depth-only OOD at depths 5/6/8, language-only OOD, and full OOD
with new language, labels, numeric range, and depth. Before H100 use, a CPU
admission independently replays every structured program, checks lexical and
structured values agree, rejects every negative or eight-digit-overflowing
intermediate register, proves the oracle answer, and binds train, evaluation,
and tokenizer hashes.

Advancement requires every condition below:

1. The learned local transition table is exactly 400/400.
2. Answer accuracy is at least 70% fit-IID, 60% depth OOD, 50% language OOD,
   and 40% full OOD.
3. At least 50% of complete operation-plus-query programs are exactly right.
4. Answer accuracy exceeds a depth-matched shuffled-program intervention by at
   least 20 percentage points.
5. Oracle programs execute to the gold answer on 100% of the board.

Passing these gates permits only an output decoder bridge and fresh manual
interaction. Failure is decomposed into semantic compilation, query binding,
local arithmetic, depth composition, or causal-program sensitivity. It does
not authorize scaling an answer-only recurrent adapter or changing the
protected flagship.

### R1 Result: 2026-07-14 13:22 EDT

The compiler, categorical executor, trainer, evaluator, admission audit, and
Slurm wrappers passed their mechanical gates. CPU admission `738772` checked
96,000 train and 896 evaluation rows, all 400 local arithmetic contexts, every
intermediate register, and every oracle answer. Its report SHA-256 is
`893386103c2484308769e24ab94001d5b6026a38095d038bf8e42ccc6a841fa2`.

H100 `688994` trained a 225,742-parameter compiler/table from immutable 200k on
32,768 examples and 2,048 updates; no base parameter was trainable. Read-only
evaluation `688995` provides a split result:

- fit-IID: 251/256 answers and 250/256 exact programs;
- depth-5/6/8 OOD under training language: 155/192 answers and 150/192 exact
  programs;
- unseen language at training depths: 19/256 answers and 8/256 exact programs;
- full unseen language/range/depth: 1/192 answers and 0 exact programs;
- overall: 426/896 answers versus 45/896 after depth-matched program shuffling.

The result is real but narrow. Categorical execution avoids the recurrent
text-state decay and composes programs past training depth. The 42.5-point
answer margin over shuffled programs shows that the compiled program is
causal. Yet the operation/query compiler is surface-bound: held-out synonyms
map systematically to trained but wrong opcodes, and unseen query wording
often defaults to register 1. Four locked gates fail, including language and
full OOD, so no decoder bridge is authorized.

The next admissible test is **Paired Semantic-Equivalence Compilation**. It
must render the same structured program through two training-only paraphrase
and entity-label views, train on both, and explicitly align corresponding
operation/query distributions. Held-out words, templates, and domains remain
untouched. This isolates whether the frozen base contains enough semantic
geometry for a small head to learn an equivalence quotient. Merely adding the
held-out templates would be leakage and is forbidden. A matched classification
arm without the equivalence loss is required before attributing any gain to
the new constraint.

### R2 Locked Protocol

R2 begins from 48,000 immutable structured programs and renders two views of
each. The views use different operation and query paraphrase families and
different entity labels, while retaining identical register-relative opcodes,
numeric operands, initial register values, query opcode, and answer. Sixteen
training-only domain vocabularies rotate through both views, preventing view
identity from being inferred from labels. The exact held-out event/query
phrases and all held-out domain labels remain excluded.

The two H100 arms must match base, data, admission report, seed, layer, hidden
width, pair order, batch size, pair count, schedule, and initial-adapter hash:

- diverse-pair control: classification plus the same 400-cell ALU basis loss,
  equivalence weight 0;
- semantic-equivalence candidate: identical objectives plus symmetric KL
  between corresponding event and query distributions, weight 0.2.

Both arms retain the r1 absolute gates. Attribution to semantic equivalence
additionally requires at least five percentage points over the control on
combined language/full answer accuracy and all-case exact-program accuracy,
while fit-IID and depth-OOD answer accuracy may regress by at most three
points. If only the control passes, the result supports language-diverse
compiler supervision but rejects the equivalence-loss claim. If neither
passes, CMB remains a same-language symbolic executor and no decoder bridge is
allowed.

Before either r2 arm trains, eight hand-authored interaction prompts are
frozen in `artifacts/evals/categorical_microcode_manual_v1.jsonl`. They cover
all nine opcodes, depths four through seven, and domains and wording absent
from both r1 and the deterministic r2 render banks. Exact-table replay is 8/8.
The same transcript inspector runs r1, the diverse-pair control, and the
equivalence candidate, reporting the full predicted program rather than only
an aggregate score. These interactions are diagnostic and cannot override the
896-case locked board.

### R2 Result: Output Equivalence Is Not Semantic Identification

R2 completed with immutable 96,000-row paired data and matched control versus
KL=0.2 arms. The control/candidate scores are respectively fit 44/256 versus
46/256, depth OOD 4/192 versus 1/192, language OOD 44/256 versus 44/256, full
OOD 9/192 versus 12/192, and exact programs 30/896 versus 30/896. Combined
language plus full answer accuracy changes only 11.83% to 12.50%. Both solve
1/8 frozen hand-authored cases, the same newsroom program. The locked
comparator rejects both absolute capability and equivalence attribution.

The negative result is mechanistically useful. Assigning two paraphrases the
same opcode labels already pushes both output distributions toward the same
one-hot target, so symmetric output KL contributes little independent
information. It does not force the trunk to separate operation kind from who
acts on whom. Replacing rather than replaying the r1 language also caused
catastrophic anchor forgetting. Yet operation-kind recognition on held-out
language increased from r1's 255/640 to about 374/640, while register and query
roles remained poor. The next intervention targets that remaining factor
rather than scaling the redundant objective.

### R3 Locked Protocol: Counterfactual Role-Equivariant Compilation

R3 is an anchor-preserving, representation-level causal test:

- Every structured program has six views: the exact r1 anchor, two disjoint
  training-only paraphrases, and an exact register-permuted version of each.
  The permutation swaps initial register values and every event/query role but
  keeps key order fixed. It is an automorphism of the two-register executor,
  so the scalar answer is preserved while every non-symmetric role label
  flips.
- The compiler predicts operation kind (add/sub/move/merge/swap) separately
  from destination register, and query kind (read/sum/difference) separately
  from selected register. Deterministic composition maps those factors back to
  the original nine opcodes and five query codes.
- Kind and role use separate learned feature projections. Each role head emits
  one signed scalar `s` and constructs logits `[s, -s]`, so negating the role
  feature implements the exact two-register `Z2` action. The candidate aligns
  kind features under register exchange and aligns role features to the
  negative of their counterfactual. This is representation-level structure,
  not the output-KL redundancy that failed in r2.
- The matched control sees all six views and all factor labels. The candidate
  sees byte-identical data/order and additionally aligns normalized event/query
  features across semantic views, preserves kind distributions under register
  permutation, and requires role distributions to swap.
- Independent CPU admission must bind source/data/evaluation/tokenizer hashes,
  prove every six-view group and permutation signature, execute every oracle
  program, reject negative/overflowing states, and find zero exact or 13-gram
  held-out-language overlap before any H100 allocation.
- Both arms retain all CMB absolute gates. Attribution requires at least five
  points over the factorized control on combined language/full answers and
  all-program exactness, with no more than three points fit/depth regression.
  The frozen eight-case manual board then remains a diagnostic gate before any
  decoder bridge.

This is not generic symbolic leakage: numbers and line boundaries remain the
same disclosed deterministic frontend as r1, but operation kind and argument
binding remain neural. A pass would establish narrow language-to-program
equivariance, not broad autonomous reasoning.

The treatment strength is frozen before the full fits: semantic feature
alignment weight **0.5** and permutation-equivariance weight **1.0** for the
candidate, versus **0/0** for the matched control. Both use 48,000 complete
six-view groups, batch four groups, exactly 12,000 updates, seed 20260714, and
the same initial adapter hash. The 64-group mechanics canary established finite
losses and gradients for the superseded output-logit implementation only; it
did not tune these weights against capability. The signed-feature revision
must pass a new isolated mechanics canary before either full arm starts.

The first full construction is preserved as a rejected admission result.
Although its automorphisms, executor replay, width, and held-out-language tests
were clean, it contained 988 duplicate normalized questions and three exact
fit-IID prompts. The corrected construction never edits a rendered row. It
scans the larger immutable source in order and admits a whole six-view group
only when every rendered question is unique within the group, disjoint from
previously selected groups, and not an exact held-out prompt. Its build report
binds the selected source-index sequence and skip counts. Training remains
blocked until the rebuilt 288,000-row artifact passes every independent gate.

The corrected build selected 48,000 groups from 48,372 source rows and passed
the structural, automorphism, oracle, uniqueness, exact-prompt, held-out
language, and public-evaluation gates. A regime-aware scanner then observed
95,972 allowed anchor-boilerplate overlaps and zero forbidden rows, but its
first version incorrectly required all 96,000 anchors to overlap. The 28 clean
anchors were ordinary orchard examples with no shared 13-gram. The repaired
logic still requires exactly two anchor rows per program, permits overlap only
from anchors against fit/depth diagnostics, and rejects every exact prompt or
language/full/manual overlap; it no longer treats absence of overlap as an
error. The immutable data SHA-256 is
`9f97e9339f665de27d99195d5b4f61c8c09681ea268cd4459a5e212b8875267f`.

The corrected full-text and response-contract pass is complete. A fresh
signed-feature mechanics canary then trained 64 groups for 16 updates with a
frozen base and finite gradients. Its initial semantic/permutation losses were
0.0999/1.8672, confirming that the representation-level counterfactual term is
not the near-zero redundant logit term it replaced. The matched full control
and candidate are jobs `689070` and `689071`.

### R3 Result: Local Equivariance Is Not Referential Binding

Both arms completed their locked 12,000-update schedules and all read-only
evaluations. Control/candidate results are:

| Slice | Control answers | Candidate answers | Control exact programs | Candidate exact programs |
|---|---:|---:|---:|---:|
| fit IID | 256/256 | 251/256 | 255/256 | 250/256 |
| depth OOD | 166/192 | 167/192 | 160/192 | 160/192 |
| language OOD | 52/256 | 60/256 | 17/256 | 32/256 |
| full OOD | 20/192 | 19/192 | 3/192 | 3/192 |
| all | 494/896 | 497/896 | 435/896 | 445/896 |

The candidate therefore gains only **1.56 percentage points** on the locked
language+full answer aggregate and **1.12 points** on all-program exactness.
Both are below the preregistered five-point attribution gates, both absolute
language/full gates fail, and both frozen eight-case direct interactions are
0/8 answers and 0/8 exact programs. Comparator `738798` correctly records
`reject_role_equivariant_compiler_r3`; no decoder bridge is authorized.

The component errors are more informative than the aggregate. On language
OOD, signed equivariance improves operation-kind accuracy from 69.22% to
79.69%, query-kind accuracy from 45.31% to 66.41%, and joint non-sum query
role accuracy from 2.91% to 25.58%. It does **not** improve operation role
given the right kind: 55.70% control versus 55.61% candidate. Merge remains
nearly unreadable at 7/119 correct kinds, while the move kind rises from
43/122 to 116/122. Full-OOD local factors improve but exact programs do not;
both arms are 0/64 exact at depth 8. The candidate repairs some semantic
categories and query geometry, but each unresolved role or kind error poisons
the deterministic execution chain.

This rejects the hypothesis that stronger role-equivariance pressure alone is
the missing mechanism. The role bit is currently an absolute class predicted
from a line-ending hidden state. Unseen entity names require a *relational
identity match* between the introduction, each event, and the final query;
class-level antisymmetry cannot manufacture that match. The next mechanism
must bind dynamic entities before it classifies operations, and it must expose
enough redundant evidence that one local mistake does not destroy an entire
program.

### R4 Hypothesis: Binding-First Referential Slot Compilation

The next bounded architecture candidate is a two-slot referential compiler,
not a larger loss coefficient or another output agreement term:

1. A small token-level tagger, supervised only during training, identifies the
   two entity mentions in the introductory clause and their later mentions.
   At evaluation it receives question text only; structured keys may score the
   tagger but may never be supplied as input features.
2. Two dynamic slot vectors are pooled from the predicted introductory
   mentions. Event/query role logits are pointer similarities to those slots,
   not fixed `role_0`/`role_1` classifiers. Swapping the two slots therefore
   swaps roles by construction.
3. Operation/query kind is read from token-span attention after projecting out
   slot identity. This directly targets the observed move/merge confusion and
   avoids relying on one punctuation-position hidden state.
4. Training uses complete entity-renaming orbits, including nonce labels, so
   lexical familiarity cannot identify a register. A matched control gets the
   same token encoder, parameters, examples, and update budget but replaces
   pointer binding with an equally sized absolute-role head.
5. Held-out scoring keeps the existing 896-case and eight-case boards and adds
   tagger, pointer, and depth-conditioned exactness diagnostics. Decoder-bridge
   authorization remains the original absolute gates plus a matched gain; no
   component score alone can promote it.

This is a new causal claim: dynamic slot identity, rather than stronger
surface invariance, should make role transport survive unseen nouns and
templates. It should first be implemented and falsified on a CPU mechanics
smoke and tiny isolated H100 canary before any full matched fit.

The text-only mechanics are now implemented. Intro slots and per-line target
mentions are soft attention distributions over token spans. Mention labels
supervise those distributions during training, but `classify_text` has no key
or target-label input. Role-pointer logits compare a projected raw-token
identity pooled from the predicted target to identities pooled from the two
predicted intro slots. The matched absolute-role control instantiates the same
modules and parameter count and receives the same mention supervision, but its
role logits come from the selected contextual mention rather than a slot
identity match. This separates relational reference from extra span capacity.

Stokes job `738806` admitted the complete existing frozen substrate: 288,000
training rows, 896 held-out rows, and eight manual rows, with zero mention
alignment or structural failures. The admission binds train/eval/manual and
tokenizer hashes and records 581,562 role-bearing training operations plus
192,774 role-bearing training queries. Its SHA-256 is
`c9758c191d6dc0754547c14ad554a7479cc5375f25e6259e9eefa409f30847fc`.
The full comparison is preregistered before any H100 result: absolute versus
pointer, mention-loss weight 1.0, same seed/init/data/order/schedule and equal
parameter count; original absolute CMB gates; at least five points on both
language+full answers and all-program exactness; at least ten points on
language+full operation role conditioned on a correct kind; and at most three
points fit/depth regression. The 64-group pointer canary has now passed.
Isolated job `689101` completed 16 finite updates with the base frozen and
wrote a locally hash-verified, CPU-loadable adapter (SHA-256
`1867b5b7f3c25c93785e4e924d43ce68a7e6c40d544e1a3348d3d5e85ac029d2`).
Matched full jobs `689104` (absolute control) and `689105` (pointer candidate)
use 300,493 parameters each and share the same initial adapter SHA-256
`fd1d2b04607b1d0c81c12551ea9d7667b91b9260453e862370e540344619fabb`.
Both have cleared finite startup telemetry. Held-out and manual evaluations are
dependency-gated; no R4 capability or decoder-bridge conclusion exists until
the locked comparator reports.

R4 is now complete. The locked comparator formally rejects promotion because
the pointer candidate reaches only 51/192 = 26.56% full OOD, below the 40%
absolute floor. That rejection must stand. The matched intervention is still
the strongest causal component result in this program: language OOD rises
29/256 -> 139/256, full OOD 2/192 -> 51/192, all exact programs 469/896 ->
624/896, and language+full operation-role accuracy conditioned on a correct
kind rises 57.92% -> 100%. The candidate retains perfect fit/depth behavior.
Comparator SHA-256 is
`890a19c1d9eaad04b5d09b5216f2622a01036ba140c11455bc6837bc23a79d54`.

The factor record changes the mechanism diagnosis. Dynamic reference binding
is no longer hypothetical; it works. Remaining OOD error is dominated by
argument structure: 116/123 language and 206/243 full subtraction events using
the unseen construction `take ... away from` are classified as move. Both
phrases describe loss from an entity, but subtract has one entity argument and
move has a source and destination. R4's kind head pools a line independently
of that incidence graph. It therefore resolves the noun correctly and still
chooses the wrong transformation.

### Parallel Diagnostic: Exact Future-Jacobian Workspace

The 2026 Jacobian-lens result changes one measurement assumption without
changing any R4 gate. Our earlier immediate logit-lens and residual-patching
nulls do not test the paper's object: the average causal map from a source
residual to *all current and future* final-block residuals. Shohin may lack such
a map, but that must be measured rather than inferred from immediate
unembedding.

`train/jacobian_workspace.py` implements the exact row-batched estimator for
Shohin's custom transformer. At every valid target position it injects one
output-coordinate cotangent, backpropagates to selected source-layer
residuals, averages source positions, and freezes every model parameter. The
first canary is raw `best_step200000.pt`, one deterministic 48-token prompt,
source layers 5/9/13/17/21/25/28, final block target, and no weight or data
write. Its unit contract proves that one-row and four-row batching agree on a
tiny transformer, matrices are finite, transport shapes are correct, and no
model parameter receives a gradient.

Even a clean matrix is **not** a workspace result. Advancement requires: (1)
stable directions across disjoint prompt samples; (2) a mid-layer band where
future-Jacobian readouts recover unspoken intermediate concepts better than an
immediate-logit control; (3) coordinate swaps that redirect a downstream
conclusion in both directions; (4) zero/shuffled/non-Jacobian controls; and (5)
evidence that the same sparse directions support more than one operation.

The readout selection rule is frozen before its H100 run. Two disjoint
eight-document lens fits must have >=0.90 whole-matrix cosine at every fitted
layer. On the existing 896-case operator board, operation/query kind is scored
at the same line-ending residual under future-Jacobian and immediate-logit
readout. Among layers 13/17/21/25, select the layer with the largest combined
language/full MRR gain; it advances only if future MRR is >=1.25x immediate
MRR and future top-10 accuracy improves by >=10 points. The separate eight-case
manual board remains untouched for the bidirectional causal-swap test. This is
a diagnostic selection rule, not a capability metric.

If those gates and R4 binding both pass, the next architecture candidate is a
**Sparse Jacobian Recurrent Workspace**: bind text to dynamic entity slots,
write only a top-k future-verbalizable state into a recurrent workspace,
broadcast it through a shared block, and train counterfactual interruption
probes to report the hidden state without requiring visible chain-of-thought
in ordinary inference. Context scaling would retain the sparse workspace plus
source provenance across chunks while dropping raw source tokens. Normal,
zero, shuffled, concept-swap, source-dropped length, and direct transcript
controls are mandatory. This is a conditional mechanism proposal, not an
authorized fit or a reasoning claim.

#### Frozen Readout Result: Stable Map, No Semantic Workspace

The exact readout gate completed in job `689118` and **failed**. Independent
future-Jacobian fits remained highly reproducible across disjoint prompt
samples, but reproducibility did not imply semantic usefulness. The frozen
selection rule chose layer 13: on 2,304 language/full operation and query
concept targets, future-Jacobian MRR was 0.0002588 versus 0.0001535 for the
immediate-logit control (1.69x), while both had 0% top-10 and 0% top-100
accuracy. The required +10 percentage-point top-10 gain was therefore absent.
The hash-bound report is
`artifacts/diagnostics/jacobian_readout_raw200k_p16_v1.json`, SHA-256
`dd173d677748d4b08113c02c4664c4fcca533f1ab5028c77a25062b28362533e`.

This separates two claims that must not be conflated. Raw Shohin has a stable
average future-causal transport map, but the map does not expose the unspoken
operation/query concepts needed by the referential compiler. A coordinate
swap would consequently manipulate an unreadable rank-tail direction and
would not test a meaningful workspace. The preregistration therefore blocks
the swap and blocks direct use of this map in a recurrent bridge. Any next
workspace experiment must explicitly install and causally validate semantic
state; it cannot assume that raw pretraining already produced one.

### R5 Hypothesis: Future-Effect Argument Algebra

A literature check rules out a weak novelty claim. [Adaptive recurrence,
algorithmic supervision, discrete latent anchors, and explicit error
correction](https://openreview.net/forum?id=8bFgEyRLrO), [dynamic entity
memory](https://arxiv.org/abs/1708.00781), and [compositional latent
programs](https://openreview.net/forum?id=N99odDSTM7) all have direct prior
art. R5 does not claim novelty for those ingredients. Its narrower hypothesis
is that a tiny reasoner should carry a **function over future consequences**,
not an unconstrained vector or a generated rationale.

For the admitted two-register domain, every event is exactly a 3x3 homogeneous
affine operator over `[entity_0, entity_1, 1]`. Chronological matrix products
compose arbitrary event chunks, while query row operators read the final
answer. `train/future_effect_algebra.py` now proves this contract over all 896
held-out programs and proves that separately compiled chunks yield the same
operator after source text is discarded. This is exact mathematics, not a
trained-model result.

The proposed learned object combines two causal structures:

1. R4-style dynamic slots identify entity identity without absolute names.
2. A text-derived argument-incidence graph identifies which slots participate
   in each event and in what relation.
3. The event encoder emits an operator whose identity is supervised by its
   effects over multiple initial-state and future-query probes, rather than by
   a single arbitrary opcode label.
4. Operators compose associatively into a fixed-size source-droppable state.
5. Counterfactual one-argument/two-argument edits must change exactly the
   corresponding future effects; zero and shuffled operators must fail.

The existing R4 board is now development data. A post-hoc label-assisted arity
rule raises the frozen pointer result from 139 to 213/256 language and roughly
52 to 120/192 full while changing no fit/depth case. That result motivates R5
but cannot score it. Before any H100 fit, freeze a fresh lexical split and an
equal-parameter control that sees the same tokens, slots, examples, and update
budget but predicts unconstrained labels or vector state. Advancement requires
all original absolute compiler gates, a matched gain on the fresh split,
counterfactual operator necessity, associative chunk invariance, and a fresh
manual board. Passing those would establish a compact causal program state,
not broad language reasoning.

The first text-only argument-graph implementation is now frozen. For each event
line, it compares every projected token identity with the two dynamically
predicted introductory slots. A threshold of **0.80**, fixed before fresh
scoring, infers one versus two participating entities and masks operation kinds
with incompatible arity. Structured keys remain supervision/audit labels only;
inference receives token states and formatting-derived line spans. On the old
development board this changes the unchanged R4 pointer adapter to 252/256 fit,
173/192 depth, 226/256 language, 146/192 full, and 773/896 exact programs. That
is a strong engineering signal but not confirmatory evidence because R4's error
analysis selected the intervention.

The confirmatory contract was therefore committed before reading a fresh score.
The new board retains the 448 fit/depth preservation controls and replaces all
448 language/full cases with new greenhouse, depot, laboratory, and library
domains, three unseen surface forms per operation, new introductions, and new
queries. It must have zero exact or 13-gram overlap with both R4 training and
development. The same pointer adapter is evaluated twice, once raw and once
with the frozen argument graph. Advancement requires all original absolute
gates, >=70% fresh language answers, >=55% fresh long-composition answers,
>=15 percentage points fresh answer gain, >=10 points fresh exact-program gain,
>=95% fresh arity accuracy, complete five-operation/three-query coverage, and
no more than 10 points fit/depth regression. These are intentionally difficult:
a failure closes this parser intervention instead of inviting threshold tuning.

Even a pass authorizes only the next matched mechanism test. That test will
replace arbitrary event-label prediction with **future-effect identification**:
two text encoders must be compared under equal parameter/update/data budgets,
one emitting ordinary class/vector state and one emitting an operator selected
by its effects across counterfactual initial states and future queries. The
operator arm must additionally pass zero/shuffle necessity, argument-edit
specificity, associative chunk composition, and source-dropping transport.

The exact algebra now also has an **error-correcting future-effect code**
reference. Eight fixed counterfactual states crossed with eight future-query
probes produce 64 scalar effects for a nine-coordinate affine operator. The
valid signatures therefore occupy a strict linear subspace: projection yields
the nearest valid operator and the orthogonal residual is an explicit error
syndrome. The CPU contract recovers every clean operator, exactly corrects one
arbitrarily corrupted scalar by leave-one-measurement decoding, and preserves
chronological composition after both chunks are independently decoded. This is
not claimed as a world-first coding-theory primitive. Its research claim is
narrower and testable: a tiny language compiler may generalize better when its
hidden computation is trained as redundant observable future behavior rather
than as an arbitrary opcode or unconstrained vector.

If R5 admission passes, the matched treatment must control for redundancy as
well as parameter count. Both arms will emit the same-width code and use fixed
decoders; the treatment code consists of structured state/query effects, while
the control uses a frozen random full-rank encoding of the same operator. A
gain can then be attributed to future-effect geometry rather than extra output
channels. Numeric values must eventually be inferred from text as well; until
that separate value-binding gate passes, the experiment remains an operation-
semantics/compiler result rather than autonomous text-only execution.

#### Frozen R5 Result: Arity Transfers, Capability Does Not

The confirmatory result rejects R5. The board and both admissions passed, and
the raw/argument evaluations were concurrent on separate H100s with identical
base, pointer adapter, data, and hash-bound admissions. Raw versus argument-
constrained fresh answers are 196/448 versus 195/448; exact programs are 174/448
versus 172/448. Language changes 146/256 -> 142/256 and full changes 50/192 ->
53/192. None of the frozen gain or absolute gates passes.

This is not a failure to detect the graph. Fresh arity accuracy is 96.61%.
Instead, the graph is too coarse: only 115/408 raw operation-kind errors cross
the unary/binary partition, while 293 remain within it. The intervention changes
130 kinds, correcting 21, harming 23, and replacing one wrong kind with another
in 86 cases. At the answer level it fixes seven cases and breaks eight. Fresh
add is only 140/353 before intervention, with 99 add->subtract and 114 add->merge
errors; move has 67 move->merge and 27 move->swap errors, while swap has 99
swap->merge errors. These are different future transformations with equal or
overlapping argument incidence. Threshold tuning cannot provide the missing
semantic relation, so R5 is closed rather than rescued.

### R6 Hypothesis: Counterfactual Effect-Coded Operators

R6 is a new experiment, not a continuation that bypasses R5's failed gate. Its
premise is that a tiny compiler should identify an event by the entire function
it induces over possible states and future queries, including the event's
numeric value, instead of choosing an operation noun and receiving the value
from structured data. The current Hadamard-derived 64-effect code is exactly
conditioned: its nine operator coordinates are orthogonal, clean codes project
with zero syndrome, and one corrupted scalar is exactly recoverable in the CPU
reference. Every existing 896 program round-trips through the code and split
chunks compose after independent decode.

The confirmatory comparison must separate future-effect semantics from extra
width and coding redundancy. Treatment and control receive the same dynamic
slots, token states, parameters, output width, examples, optimizer schedule,
and update budget. The treatment uses the fixed state/query effect code; the
control uses a frozen random orthogonal 64x9 code with equal condition number
and the same decoder capacity. The R6 board must be generated only after those
arms and hashes are frozen, with unseen language, unseen numeric values,
held-out probe combinations, longer compositions, counterfactual one-event
edits, source dropping, zero/shuffled code controls, and a fresh manual
interaction. Structured operation values are forbidden at inference. A pass
would establish a narrow learned effect algebra; broad model reasoning would
still require transport into ordinary decoding and unrelated domains.

#### Pre-Fit Correction: Basis Coding Is Not the Mechanism

The random-code comparison above is superseded before any H100 fit. An exact
CPU contract now proves why: for any full-rank 64x9 treatment and control
codebooks, decoding to a common 3x3 operator and re-encoding gives a fixed
linear transport between codes. Chronological composition commutes with that
transport exactly. Equal singular values make the map an isometry on the valid
nine-dimensional subspace. A structured effect code versus random orthogonal
code is therefore a coordinate change, not a distinct reasoning mechanism.
No result from such a comparison could justify the intended causal claim.

R6 is refined to an **Active Counterfactual Distinction Loop**. A shared neural
head receives one event's text representation plus one counterfactual state and
future-query probe and predicts the resulting scalar effect. The cell maintains
a distribution over lawful event operators, selects the unobserved probe that
maximally partitions the currently plausible hypotheses, obtains one more
text-conditioned effect prediction, and repeats only while ambiguity remains.
The committed operator is then composed into a fixed 3x3 carried state and the
event text can be dropped. The internal reasoning trace is thus an auditable
sequence of questions of the form "what would this event do under this state
and future readout?", not generated prose or an arbitrary latent vector.

The exact oracle mechanics cover 597 distinct hypotheses: six numeric operator
families over values 1--99 plus three structural operators. The active policy
identifies every operator in at most three probes, averaging 1.838. A
deterministic random-probe policy also eventually identifies every operator but
averages 2.822 and requires as many as 13 probes. This is only a decision-tree
upper bound; it does not show that Shohin can answer a selected counterfactual
from text.

The neural causal test will therefore train one probe-conditioned effect head,
not two code-basis heads. Read-only inference compares active and random probe
schedulers on the byte-identical model with the same maximum number of probe
calls. A one-pass pointer compiler, zeroed effects, shuffled predicted effects,
and oracle effects are required controls. The board remains frozen only after
the head, scheduler, numeric range, tolerance, maximum latent steps, hashes,
and value/language/composition splits are fixed. Advancement requires unseen
values and language, counterfactual edit specificity, longer source-dropped
composition, a material active-over-random gain, and direct transcript evidence.
This makes adaptive future distinction operational while avoiding a basis-
renaming result.

The first isolated CUDA mechanics canary is clean. Job `689183` initialized
from the immutable raw-200k checkpoint and the frozen R4 pointer adapter,
compiled the complete admitted 288,000-row substrate, then ran 16 updates over
64 groups. It reports 466,894 trainable adapter parameters, zero trainable base
parameters, finite effect loss 0.0526 at step zero, inherited operation kind and
role accuracy 1.0 on the canary sample, and a finite pre-clip gradient norm
9.881 under the locked 1.0 clip. Training took 13 seconds after preprocessing
and the saved adapter is CPU-loadable, finite, contains no base-model tensors,
and hash-matches Newton/local at SHA-256
`b27805f489cd39069c5d3b919d113d38d2441b27f63ac70ba4d4c0187724a929`.
This certifies mechanics only; a longer development fit must show that effect
loss and gradient norms settle before any fresh-board generation.

The old-board development decision is executable and frozen before its score.
`train/evaluate_future_effect_r6.py` separates the 448 fresh language/full cases
from 448 pinned fit/depth controls. Authorization to generate one untouched R6
board requires, simultaneously: active >=55% fresh answers, >=50% fresh exact
programs, and >=65% operation recovery; >=60% language and >=40% full answers;
>=80% fit and >=60% depth preservation; at least +10 points over the unchanged
R5 raw fresh answer and exact-program counts; at least +5 points answers/exact
and operations over random with equal calls; at least +10 points
answers/exact over the better zero/shuffled control; >=80% oracle answers and
exact programs; >=95% query accuracy; and finite held-out-probe MSE no greater
than max(2x train MSE, 1.0). Any failed conjunct closes this R6 head before a
fresh board exists.

An exact pre-score board audit also bounds the runtime scheduler itself. Using
true event effects but the evaluator's actual three-step top-64 approximation,
fresh language/full reaches 382/448 answers (85.27%) and 365/448 exact programs
(81.47%). The frozen >=80% oracle floors are therefore attainable but close to
the mechanism's current ceiling. This prevents a learned head from being judged
against 100% while also preventing a weak compiler ceiling from excusing it.

A separate **R6b posterior scheduler** is frozen before any R6a learned output
is read. It does not change or rescue the R6a decision. Instead of discarding
all but a hard top-64 list after each noisy effect, it retains a Gaussian score
posterior over all 597 operators and selects the next probe by maximum weighted
partition entropy. Its assumed effect-noise scale is fixed at 1.0 and its effect
bin width at 2.0. Under deterministic equal-noise CPU mechanics, three posterior
probes recover 100% of operators with exact effects and 92.46% at noise 0.5,
versus 88.27% for R6a's hard top-64 scheduler. The implementation is
`train/future_posterior_distinction.py`.

R6a must be scored and recorded first under its existing gate. R6b may then be
run read-only on the byte-identical adapter and old development board only; it
receives the same three scalar-effect calls and must face the same random,
zero, shuffled, absolute-capability, and held-out-fidelity gates. It cannot
advance a fresh board unless its policy and comparator are frozen independently
before that evaluation. This separation prevents a stronger inference rule
from being used as a post-hoc relabeling of R6a.

#### Conditional Context Extension: Distinction-Certified Context Folding

A bounded literature search finds important adjacent work. [UNComp](https://openreview.net/forum?id=28oMPC5bcE)
uses uncertainty to vary hidden-state and KV-cache compression; [Compile to
Compress](https://openreview.net/forum?id=NjbMkeaOKD) uses compiler failure modes
to compress theorem-proving search history; [Proof-Carrying
Numbers](https://openreview.net/forum?id=455AaEhQbu) fail-closes numeric display
through external verification; and [Selection-Inference](https://arxiv.org/abs/2205.09712)
alternates model-generated natural-language selection and inference. These block
novelty claims for uncertainty-aware compression, verifier-backed outputs, or
alternating selection alone. The bounded search did not find their exact
combination with a learned scalar counterfactual-effect interface, active
posterior operator distinction, independent unused-probe certification, and
associative source-dropped operator folding. That absence is not proof of
world-first novelty; the causal mechanism and empirical result remain the claim.

If and only if the learned active policy clears its frozen active-over-random
and causal-control gates, the same mechanism has a non-token context extension.
An event may leave context only when its selected counterfactual observations
reduce the lawful hypothesis set to one operator and a separately predicted,
previously unused probe agrees with that operator. The accepted event operator
then composes into a fixed 3x3 chronological state. Ambiguous or independently
inconsistent events retain their source instead of being silently compressed.

The exact CPU reference in `train/counterfactual_context_folding.py` admits all
597 oracle event certificates using at most three selected probes plus one
independent validation probe. It folds 4,096 chronological events, discards all
event sources, and retains the same nine-scalar operator and answers as direct
execution. Independently folded chunks merge associatively. Empty evidence is
rejected as ambiguous and a corrupted validation effect is rejected before
folding.

This is a proof-carrying algebra contract, not a learned context result. It is
intentionally stronger than retaining an opaque latent or model-authored text
ledger: source deletion is conditional on a falsifiable future-effect witness.
The neural mechanism may advance only after the R6 head demonstrates calibrated
held-out-probe certificates. A context-scaling claim would additionally require
source-dropped length transfer beyond the native window, injected-certificate
corruption that causes retention or reopening, equal-model raw-context controls,
and measured retained-state, prefill, and accuracy curves. Until then the
nine-scalar result is an oracle upper bound.

#### R6 Outcome: Causal Use Without Semantic Transport

R6a is rejected before any fresh confirmatory board. Its isolated 466,894-
parameter effect adapter completed 12,000 updates and the frozen evaluator
scored 896 old-board cases under five policies with exactly three effect calls
per event. On the 448 development-fresh language/full cases, active probing
recovers 610/1,856 operations, compared with 407 for equal-call random, 78 for
shuffled effects, and 24 for zero effects. The +10.94 percentage-point operation
gain over random is real causal evidence that the selected interventions are
used. It is not reasoning evidence: active reaches only 36/448 answers and
19/448 exact programs, while learned effect MSE rises from 0.33 on fit and 0.84
on depth to about 130.74 on unseen language/full.

The failure is representational, not merely schedular. Of 1,856 active fresh
operations, 884 have the wrong opcode and another 362 have the right opcode but
the wrong value; subtraction is essentially absent. Query binding is only
342/448. R6b's stronger posterior scheduler therefore remains an unrun CPU
mechanics result: choosing a better question cannot repair an answer interface
whose unseen semantic response is off-scale. No R6 fresh board, source-drop,
context-folding, or broad capability claim is authorized.

### R7 Hypothesis: Interventional Semantic Quotients

R7 changes the observable instead of tuning R6. The central hypothesis is that
a natural-language operation can be identified by its **causal response field**
inside a frozen language model. For an unknown event, the evaluator makes
matched finite interventions to visible initial values, event values, and
entity roles, then measures how the final future-token hidden state changes at
frozen layers 5, 11, 17, 23, and 29. It constructs the same nonlinear finite-
difference signature for lawful canonical operator hypotheses. The predicted
operator is the canonical hypothesis whose intervention signature best matches
the unknown event.

This is not a learned operation classifier, a chain-of-thought prompt, a logit
lens, or R6's supervised scalar-effect head. It uses no gradients and trains no
weights. It asks whether two descriptions implement the same transformation by
comparing what the frozen network itself does under matched interventions.
The active policy spends a frozen budget of two intervention channels chosen to
maximize pairwise candidate separation. Equal-budget random-channel,
unintervened direct-hidden-similarity, and shuffled-signature policies prevent a
generic representation or extra-compute gain from being called causal
semantics.

The adjacent literature prevents an unqualified novelty claim. [CausaLM](https://arxiv.org/abs/2005.13407)
trains counterfactual representations for causal explanation; [Passive Learning
of Active Causal Strategies](https://arxiv.org/abs/2305.16183) studies learned
intervention policies; [Model-based Interactive Semantic
Parsing](https://arxiv.org/abs/1910.05389) uses a world model and clarification;
and [Selection-Inference](https://arxiv.org/abs/2205.09712) alternates language
selection and inference. A bounded search did not find the exact combination of
canonical operator hypotheses, nonlinear future-hidden finite-difference
fields, actively selected matched textual interventions, and an independent
unused-intervention certificate. That is an informed project-novelty hypothesis,
not proof that the method is world-first.

The first canary is deliberately small and read-only: exactly 108 development
events, 12 for each of nine opcodes, restricted to the already-used language and
full regimes. It uses visible identifier strings and numeric literals supplied
by the board to construct lexical candidates; therefore a pass is only a
semantic-identification canary, not a complete text-only reasoner. Before any
score is read, advancement is frozen as all of: active opcode accuracy at least
45%; active at least five points above random and direct hidden similarity;
active at least 15 points above shuffled signatures; and at least seven of nine
opcodes reaching 4/12. A pass authorizes only a full evaluation on the already-
used R5 board. It does not authorize training, fresh data, source deletion,
reasoning, or context-scaling claims. A failure closes this R7 observable rather
than triggering threshold tuning.
