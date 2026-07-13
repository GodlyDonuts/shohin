# Latent Reasoning and Memory Research

## Status

This document records a research program, not a capability claim. Shohin does
not yet demonstrate reliable broad reasoning, visible working, or semantic
context retention. Every proposed mechanism remains isolated from the flagship
pretraining writer until it beats a matched non-latent control on held-out
transfer and survives direct interaction.

## Literature Boundary

The first continuous-latent pilot is not claimed as a new architecture:

- [Coconut](https://arxiv.org/abs/2412.06769) feeds a final hidden state back
  as a continuous thought input. Shohin's `latent_rollout.py` is a constrained,
  answer-only control in that family.
- [Recurrent Memory Transformer](https://arxiv.org/abs/2207.06881) passes
  dedicated memory tokens between segments.
- [Compressive Transformer](https://arxiv.org/abs/1911.05507) compresses old
  activations into a secondary memory.
- [Associative Recurrent Memory Transformer](https://arxiv.org/abs/2407.04841)
  studies recurrent transformer memory for very long contexts.

The research opportunity is therefore not to rename a latent token mechanism.
It is to establish a small-model, verifier-backed protocol showing whether a
fixed continuous packet can retain and update useful information after the
source text is unavailable, while preserving semantic transfer under strict
resource constraints.

## Current Evidence

The first matched 180k pilots use the same raw checkpoint, data hash,
`24,000` selected examples, `1,500` updates, seed, optimizer, batch size, and
answer-only target. The only intended difference is the progressive number of
continuous feedback tokens: `L=0` control versus `L<=4` pilot.

The original full-OOD test simultaneously changed wording, names, value range,
and depth. It is deliberately retained as a hard end gate, but cannot diagnose
which transfer failed. The factorized v2 read-only evaluation has `896` rows:

| Regime | What changes from training | Rows |
| --- | --- | ---: |
| `fit_iid` | nothing except examples | 256 |
| `depth_ood` | composition depth 5/6/8 | 192 |
| `language_ood` | domain, labels, and event wording | 256 |
| `full_ood` | language, value range, and depth | 192 |

The matched `L=0` control completed the v2 diagnostic with these exact results:

| Regime | Exact accuracy |
| --- | ---: |
| `fit_iid` | 129/256 = 50.39% |
| `depth_ood` | 27/192 = 14.06% |
| `language_ood` | 34/256 = 13.28% |
| `full_ood` | 0/192 = 0.00% |

This proves that the low answer-only loss was not a parser failure: the model
learned part of the in-template program. It also proves that this narrow
curriculum is not sufficient for semantic or compositional reasoning.

The paired continuous-feedback checkpoint was evaluated with the same v2
suite at `L=0`, `L=2`, and `L=4`:

| Model / decode steps | Fit IID | Depth OOD | Language OOD | Full OOD |
| --- | ---: | ---: | ---: | ---: |
| matched control, `L=0` | 50.39% | 14.06% | 13.28% | 0.00% |
| feedback pilot, `L=0` | 41.41% | 6.77% | 10.94% | 0.00% |
| feedback pilot, `L=2` | 44.92% | 9.38% | 11.72% | 0.00% |
| feedback pilot, `L=4` | 46.48% | 12.50% | 11.72% | 0.00% |

The feedback pilot loses to the matched control in every regime. Its original
full-OOD report is also 0/600 at every evaluated decode depth `L=0/1/2/4/8`.
The simple final-hidden-state feedback route is therefore rejected for
reasoning and context-scaling promotion. It remains a documented negative
control, not a future flagship component.

## Next Mechanism: Source-Dropping Memory Packet

The next architecture must be materially different from the current feedback
loop. The current loop retains the full source prompt at every latent step, so
it offers extra computation but **not context compression**.

The proposed isolated experiment is a **source-dropping memory packet**:

1. Split source evidence into bounded token chunks.
2. Prepend a fixed bank of `M` continuous memory slots and append `M` learned
   write slots to each source chunk.
3. Read the write-slot hidden states as the next memory packet and recursively
   carry only that packet to the next chunk.
4. After the final chunk, delete every source token. Decode the answer from
   the `M` continuous slots plus a fresh query only.
5. Train end-to-end on the answer, with no serialized `think`, `state`,
   capsule, external execution, answer injection, or selection oracle.

This is a fixed-capacity read/write recurrent memory experiment. It is not
claimed to be novel in isolation. Its project-specific contribution would be a
small-model, source-removal protocol with explicit semantic, compression, and
behavioral gates.

## Required Falsification Gates

No source-dropping packet may be called useful unless all of these are met:

1. **Fit gate:** a fresh exact-prompt-disjoint in-distribution split proves the
   model learned the task rather than an answer prior.
2. **Source-removal gate:** decoder input contains only memory slots and query;
   an assertion records that no source token IDs or KV cache are present.
3. **Memory ablation:** `M=0`, detached-state, and shuffled-state controls
   under the same update/data budget establish that the packet carries causal
   information.
4. **Length gate:** held-out chunk counts exceed training counts while source
   tokens stay unavailable after their chunk is written.
5. **Semantic gate:** labels, domains, event language, and values change
   independently. One aggregate full-OOD number is never enough.
6. **Behavior gate:** direct transcripts must show correct intermediate
   information use and final answer; a format token or final answer alone is
   insufficient.
7. **Generalization boundary:** passing a synthetic memory task is still not a
   broad reasoning promotion. Public and human-style held-out evaluations
   remain separate requirements.

## Immediate Decision Rule

- The completed feedback pilot did not materially exceed the matched control
  on any regime, so it is rejected as a useful latent-computation path.
- Only the source-dropping packet can test the constrained-context objective;
  its source-removal and memory-ablation gates are mandatory before training.

## Source-Packet Pilot Result

The answer-only source-packet screen is rejected. M0 (zero slots) and M1
(eight slots) each used `24,000` selected examples, `6,000` updates, the same
raw-180k checkpoint, seed, optimizer schedule, and exact answer target. M1's
held-out normal/zero/shuffled scores were `6/384`, `6/384`, and `9/384`.
Normal lost to shuffled on IID (`4/96` versus `5/96`) and had no positive
margin on length plus language transfer (`2/192` versus `3/192` controls),
with zero positive chunk or query-kind margins.

The predeclared advancement rule was:

1. It beats **each** of M0, zeroed packet, and shuffled packet by at least 15
   percentage points on the fit-IID regime.
2. It beats those same controls by at least 5 percentage points on the combined
   length-OOD and language-OOD regimes; a fit-only result is not context scale.
3. It has a positive normal-packet margin on at least three chunk counts and
   two distinct query kinds; one favorable slice is not context scaling.
4. Read-only transcript inspection shows the source-free decoder uses the
   actual retained value rather than a fixed answer prior.

The screen did reject the answer-only packet. Low loss and formatted answers
are insufficient evidence of retained information.

## If Answer-Only Memory Fails: Certified Latent Ledger

The next bounded mechanism is a **Certified Latent Ledger (CLL)**, not a hidden
external program. The writer still receives source chunks only once and the
decoder still receives only continuous slots plus a fresh natural-language
query. The change is the training signal:

1. After each written chunk, sample several solver-verified readback queries
   about the current record (both individual values and a relation such as a
   sum or difference). The model must answer from the packet alone.
2. Reuse the *same* final packet for multiple fresh queries. This makes a
   packet useful only if it retains information rather than associating one
   query template with one answer distribution.
3. Train counterfactual source pairs that share a prefix and query but differ
   by one final event. A correct packet must separate the two consequences;
   the evaluator checks this exact pairwise distinction.
4. Keep all supervision token-level and verifier-backed. There is no controller
   that executes, repairs, selects, injects, or serializes the ledger at
   inference time.
5. Evaluate normal, zero, shuffled, slot-drop, longer-length, paraphrased, and
   counterfactual conditions separately. CLL is rejected if normal packets do
   not cause the measured advantage.

This is a stronger route to constrained-context scaling because it trains the
continuous state as an information-bearing interface throughout a sequence,
while preserving the hard source-removal boundary. It still would establish
narrow retained-information reasoning first, not broad intelligence.

### CLL implementation boundary

`pipeline/generate_certified_latent_ledger_v1.py` creates solver-recomputed
readback probes at every source prefix plus final-event counterfactual pairs.
`pipeline/audit_certified_latent_ledger_v1.py` independently replays every
operation and query, requires two valid variants for every counterfactual pair,
and rejects any exact or 13-gram train/held-out **input-prompt** collision. Each
discarded source chunk has an inert record tag solely to make the input split
auditable; it is never present in the query or target answer and has no answer
relation. CPU-only build `687132` completed in 153 seconds and independently
admitted `223,996` train rows, `7,936` held-out rows, `16,000` train pairs,
and `384` held-out pairs with zero invalid, duplicate, exact-overlap,
13-gram-overlap, or malformed-pair findings. The matched raw-190k CLL M0/M1
screen is `687134`/`687135`; neither can touch the flagship.

`train/eval_source_dropping_memory.py --all-heldout` retains every held-out
ledger row instead of independently sampling rows that could split a pair. For
a bounded but pair-safe screen, `--counterfactual-pairs-per-regime N` adds `N`
complete deterministic pairs from each regime to the ordinary balanced
readback selection; it never evaluates one counterfactual variant alone. The
report records this selection, accuracy by ledger stage and probe kind, and the
rate at which both members of each counterfactual pair are correct and receive
different predicted values. When pairs exist,
`compare_source_dropping_memory.py` adds a required 10-point normal-packet
advantage over M0, zeroed, and shuffled controls. This is still a narrow
retained-information gate, not a general-reasoning result.

### CLL v1 efficiency correction

The initial v1 corpus is a valid correctness artifact, but it is not an efficient tiny-model
experiment. Read-only tokenization measured mean train chunk length **213.36** tokens (20k-row
sample; p50 208, p95 241) and mean held-out chunk length **194.93** tokens (all rows; p50 192,
p95 227), largely because each discarded boundary carried sixteen opaque seal words. The pending
M0/M1/evaluator/comparator chain `687134`-`687138` was canceled before it allocated a GPU.

`compact_v2` preserves the solver-recomputed readbacks, source-removal boundary, counterfactual
pairs, and exact/13-gram split audit, but represents each inert tag as `Reference` plus twelve
single-character words. With `shohin-tok-32k.json` this is exactly 14 tokens including punctuation
and supplies a record-unique first 13-token ngram. The v1 artifacts are retained unchanged. V2
must independently pass its audit and a full token-cost report before matched H100 training is
resubmitted; a correct but unnecessarily long synthetic protocol is not a useful reasoning result.

CPU build `687141` completed the fresh v2 paths in 85 seconds with the same 223,996/7,936
train/held-out rows and 16,000/384 pairs. Its independent audit has zero invalid, duplicate,
exact-overlap, 13-gram-overlap, or malformed-pair findings; train/eval SHA-256 are
`8760df867b4da98dcc84b356eea8e0d70922e3280e868193216173603814c08c` and
`dfcf852c0ca57b6bb58f4f7c1a775221e33e97aa261d79478cc3bf36e82e5fc7`. The local/Newton
hash-matched token report (md5 `89ac960e6eb12b0f2dbe25f42a9ee3d1`) confirms whole-corpus
train chunk/source means **228.60/511.63 -> 41.44/92.74** tokens and held-out means
**194.93/660.24 -> 37.43/126.79**. This earns exactly one matched raw-190k M0/M1 screen:
`687144` (no slots) versus `687145` (eight slots), each 24,000 examples / 6,000 updates;
pair-safe evaluators `687146/687147` and locked comparator `687148` followed only on success.
The screen is now complete and **rejected**. M0 is a valid 11/631 (1.743%) no-memory floor in
every mode. M1 normal/zero/shuffled is 16/631 (2.536%) / 4/631 (0.634%) / 12/631 (1.902%),
but the locked comparator finds only a 0.79-point fit margin, a 0.63-point combined
length/language margin, two positive chunk counts, and 0/128 correct-and-different intervention
pairs. It fails every required gate except the weak query-kind count. The result establishes that
the source-free writer can influence a few formatted fit answers, not that it retains a causally
useful, compositional state.

## If CLL Fails: Latent State Algebra

CLL may still fail because answer losses reward the right token without making
the memory geometry represent the same state consistently. The next unsubmitted
idea is **Latent State Algebra (LSA)**: keep the same source-free decoder, but
use verified equivalent and counterfactual records to constrain the memory
packet itself during training.

1. Two distinct verified update sequences that finish in the same ledger state
   should yield equivalent normalized packets; a lightweight training-only
   projection is aligned across those equivalent records.
2. Counterfactual records sharing a prefix but differing in one final update
   should have packet deltas that decode the signed verified state change.
3. The ordinary source-free readback loss remains primary. The alignment and
   delta objectives are removed at inference; there is no symbolic executor,
   textual scratchpad, or external controller.
4. It must be matched against CLL without those auxiliaries, with the same
   source-removal, zero, shuffled, length, language, and pairwise tests.

This is intentionally a falsifiable architectural bet, not a claim that a
continuous packet is already semantic. CLL now rejects answer-only packets
while showing a small normal-versus-zero fit gap. That is weak enough to reject
the route, but sufficient to justify exactly one denser, matched geometry
screen before abandoning source-free packet memory.

### LSA preregistration

LSA is not more slots or a hidden symbolic controller. It tests whether a
fixed continuous packet can acquire the algebra of a verified state while the
only inference interface remains `packet + fresh query -> answer`.

1. Generate matched episode quartets. Equivalent pairs use different natural-
   language update sequences that end in exactly the same two-variable state
   through operation commutation and delta decomposition. Intervention pairs
   share a prefix and differ in a verified final delta. Anti-equivalent pairs
   retain similar surface form but finish in different states.
2. Keep ordinary source-free readback cross-entropy primary. Add training-only
   packet objectives: equivalent packets align after a learned projection;
   intervention packet differences predict the signed normalized state delta;
   and a contrastive term separates packets whose verified states differ. A
   small probe may read numeric state only for these losses and is discarded
   with the projection before inference.
3. Prevent a cosmetic solution. The paired batch must include random
   mismatches, permuted state-code controls, and a zero-auxiliary CLL control.
   Equivalent alignment without state separation is collapse; accurate probe
   output without source-free decoder transfer is an auxiliary-only shortcut.
4. The LSA candidate must beat a matched answer-only control, zeroed packet,
   shuffled source, shuffled pair assignments, and permuted state-code controls
   on the same held-out length, language, equivalence, and intervention regimes.
   A narrow pass remains constrained-context evidence, not autonomous general
   reasoning.

The prediction is diagnostic. No CLL margin means the writer/decoder has not
learned a usable channel and LSA should not be funded yet. CLL readback with
weak equivalence or counterfactual transfer identifies exactly the state-
geometry defect that LSA targets. A CLL pair pass followed by weak direct
reasoning means retained context is necessary but not sufficient and moves the
research program toward learned latent deliberation rather than more memory.

`train/latent_state_algebra.py` now contains the focused training-only loss
primitive; `train/test_latent_state_algebra.py` verifies alignment, state,
delta, separation, and shape invariants. Equivalent rows alone receive
alignment/contrastive pressure; verified interventions retain state/delta
supervision and now also have a packet-space separation margin, preventing a
state probe from hiding distinct values in a collapsed packet direction.

The resulting, still-unsubmitted LSA v1 screen is fully wired before any GPU
allocation: `pipeline/generate_latent_state_algebra_v1.py` produces paired
commutation/intervention records, `pipeline/audit_latent_state_algebra_v1.py`
recomputes every state and checks pair structure plus exact/13-gram splits,
and `train/latent_state_algebra_train.py` exposes verified, shuffled-pair,
permuted-state-code, and zero-auxiliary modes. `train/compare_latent_state_algebra.py`
locks a 10-point fit/equivalence/intervention and 5-point length/language gate
against the matched answer-only model. The generator/auditor/trainer/comparator
have only passed CPU smoke contracts so far; no LSA data, checkpoint, or GPU
result exists yet.

## Original Directions After LSA

If LSA proves a stable semantic packet, the next architectural experiment is
a Causal Latent Workspace: a small fixed recurrent workspace that repeatedly
updates only the packet from the question and current packet before decoding.
It would be trained on verified state transitions with halting fixed during the
first screen, then compared with equal-compute untied token decoding. No text
chain of thought, controller, retrieval system, or external executor would be
available at inference. This remains a design hypothesis, not an implemented
or claimed capability.

## Causal Prefix Readback Result and Direction Change

CPR was the decisive decoder-interface test after CLL and LSA. It kept the
source-removal boundary but asked the language decoder to answer a fresh,
solver-verified question from every intermediate continuous packet. Three
matched arms used the immutable 190k checkpoint and the same 32,000-pair /
7,163-update surface: verified readbacks, shuffled complete readback labels,
and an equal-work replicated-final control.

The locked held-out comparison rejects the mechanism. The verified normal
packet score is **161/10,752 = 1.497%**, exactly equal to the shuffled-source
control; the equal-work replay scores **193/10,752 = 1.795%**. The verified
model loses to the strongest control by **1.273 points** on fit IID, **0.493
points** on length OOD, **0.347 points** on language OOD, and **0.439 points**
on full OOD. It has zero positive preregistered gates. Verified and
shuffled-label training losses were also nearly identical. This means the
model did not learn a causal, decoder-readable continuous packet channel even
on the matched task. CPR is rejected; stage 2 is prohibited.

This changes the research direction. The negative results consistently show
that a 125M decoder is not reliably learning to write and read a semantic
continuous state, while broad SFT traces can teach formatting without correct
intermediate computation. The next experiment must reduce the primitive
transition burden rather than add another continuous interface.

## Next Hypothesis: Digitwise Recurrent Scratchpad

The next candidate is a **Digitwise Recurrent Scratchpad**, a discrete,
model-authored microstate protocol. It is a design hypothesis only; no model
has been trained on it.

1. The state is a fixed-width digit tape, least-significant digit first, plus
   a program counter, carry/borrow bit, and immutable operation digits.
2. Each generated turn performs exactly one local rewrite: read one digit and
   carry, emit the next digit and carry, advance the program counter, and copy
   the remaining tape. The model emits every next state; a controller may only
   transport that emitted text to the next fixed turn.
3. The first screen uses canonical operations only. It must prove unseen-value
   and longer-width arithmetic before a separate language-to-program bridge is
   attempted. Natural-language parsing is not allowed to hide a failed
   executor.
4. Evaluation must include train-fit generation, teacher-forced transitions,
   self-authored closed loops, random-tape and constant-output controls,
   transition interventions, unseen width, and later unseen lexical wrappers.

This differs from VRWM: VRWM asked the model to perform whole-register updates
in one generation. The tape, counter, and carry reduce each generation to a
finite local transition, preserve a strict bounded context, and make every
state transition verifier-checkable. A passing result would still be narrow
algorithmic-execution evidence, not general reasoning. Only after the discrete
skeleton passes should learned compression or a language bridge be considered.

### DRS v1 split decision and v2 rebuild

The first full CPU-only candidate was generated on Stokes before any model
allocation. It contains 439,865 train rows and 1,500 held-out paired
counterfactual episodes over fit width 4/6, value OOD width 4/6, and width OOD
8. The independent audit recomputed every transition and counterfactual. It
found zero invalid rows, duplicate normalized train prompts, or exact prompt
hits, but it found **27 held-out 13-gram hits**, so the candidate is rejected.

The overlap was real rather than fixed instructional wording: an `add` training
episode and a `sub` held-out episode could carry the same width and operand
tapes. The operation field can occur outside an arbitrary 13-token state
window, so an exact initial-state signature was insufficient. The rejected v1
JSONL files and audit remain immutable. The correction reserves the
`(width,left,right)` operand-tape signature across operations and across both
members of each counterfactual pair. The audit now emits bounded overlap
examples for future diagnosis. Unit tests and a 1,000-episode smoke build pass
with zero exact and 13-gram overlap.

Stokes `738122` built a separately named v2 candidate and `738123` passed its
independent audit: 439,865 train rows, 1,500 held-out paired episodes, 19,800
controller prompts, and zero invalid/duplicate/exact/13-gram findings. The
train/eval SHA-256 are
`381b8bbf3a4eddb7b08b0f9d4b08ea3ce65e1f0ec48de930632d54417c2f7f35` and
`89ce11b36ff2f56e83cda72a1f07b1a90f4a3dc3803c69db2779a27219712646`.

The resulting GPU test is deliberately causal and narrow. `687348` measures
raw-200k self-authored closed loops under held-out wording, and `687362` then
measures the same episodes under the core wording. Only after both succeeds
does `687363` perform one isolated DRS SFT epoch from the same
`best_step200000.pt`; `687364` and `687365` evaluate the SFT result under
core and held-out wording respectively. The initial `687350`/`687351`
children were canceled before start because they lacked the raw-core control.
This is a data-admission correction and an experiment launch, not a model
result. Any positive result must beat raw by regime, preserve state through
paired counterfactuals, and show response diversity rather than a fixed
template.

### Append-only Delta Ledger candidate

The direct 200k transcript shows that full-state DRS asks too much of the
untrained decoder in one generation: copy a long record, select the local
operands, perform arithmetic, write a digit, update carry, and update a
program counter. The **Append-only Delta Ledger (ADL)** is a separate candidate
that decomposes this burden without letting the controller solve anything.

For a fixed immutable operand tape, each model turn emits only
`adl:step=<n>;d=<digit>;c=<carry>`. Every four exact model-authored delta lines
are then presented back to the model for one compaction turn,
`adl:block=<n>;digits=<lsf digits>;c=<carry>`. The controller only schedules the
fixed program, transports exact parsed lines, and drops raw deltas after a
model-authored block exists; it never calculates, rewrites, repairs, ranks, or
chooses a digit/carry/block. The next prompt sees immutable operands, retained
blocks, and any live deltas. This gives a first-level 4x reduction in retained
execution trace and makes the later recursive block-of-block extension a
measurable, rather than aspirational, context-scaling step.

`train/append_ledger_protocol.py`, controller, generator, and independent
audit are implemented. The initial 40-episode local smoke produced 640 train
rows and 20 paired held-out episodes across fit, value-OOD, and width-OOD
regimes; the audit recomputed every row/episode and passed with zero malformed
rows, duplicate normalized prompts, exact prompt overlap, or 13-gram overlap.
An earlier smoke initially found 147 generic delta-record n-gram collisions;
the protocol now binds each retained record on both sides to an immutable
base-derived identifier, without adding computed controller content. That
correction reduced the overlap gate to zero. This is not yet a model result or
a broad-reasoning claim. Full CPU data generation is allowed, but GPU SFT must
wait for the DRS core-vs-heldout result to establish whether ADL addresses a
real interface failure.

The complete candidate entered CPU-only admission on Stokes as build `738186`
with independent dependent audit `738187`, after host-local imports and the
same generator/controller tests passed. It remains deliberately outside the
GPU path: full artifact hashes, transition recomputation, all held-out prompt
overlap checks, and the DRS diagnosis must exist before any isolated ADL SFT
can be considered.

The full build passed admission. `738186` produced 384,000 train rows and
1,000 paired held-out episodes. Independent audit `738187` recomputed all
rows/episodes and found zero invalid rows, duplicate normalized prompts, exact
held-out prompt hits, or 13-gram hits over 42,000 held-out controller prompts.
The train, held-out, and audit SHA-256s are respectively
`ef317dd5aed85fa83add40a637c52232f4b4daf626e609f88926cb358113cbec`,
`3117ec5072134a9bade424499be9ee3a3e504e4f26deec445c3b5b1baeccaca0`, and
`5d0e2acd2cfc042de7c76266d048987c347d1e5d22b05e79232fce8ea5c9258f`.
The data is ready on Newton for an isolated future experiment, but remains
unsubmitted while DRS measures whether the bottleneck is local execution or
the lexical interface.

The evaluation and SFT surfaces are prepared but unsubmitted:
`eval_append_ledger.py` separately scores first delta, all delta and block
transitions, exact chains, final answers, paired counterfactuals, response
diversity, and core versus held-out wording. The corresponding SFT wrapper
hash-binds the audit and uses the data's exact `completion_prompt` rather than
wrapping it in another `Question/Answer` frame. This readiness work changes no
GPU queue and lets ADL become a clean next comparison if DRS local execution
does not transfer.

Raw ADL does not already know this interface. On sixteen fixed first-step
probes (eight tapes under core and held-out wording), it ranks the correct one
of twenty possible digit/carry records first **0** times, with mean correct
rank **10.688**; its top candidate is always the generic `d=0;c=0`. The
diagnostic is intentionally not an external decoder: it only measures model
likelihood of fixed grammar-valid completions. This confirms that ADL should be
judged as a supervised learnability experiment, not claimed as a hidden
pretrained reasoning path.

### Decoder Recognition Check

Before interpreting a compact-state curriculum as merely an output adapter, the
raw 200k model was also asked to *rank* fixed candidate answers without any
free generation. Correct candidates compete against matched wrong candidates
under the same plain question prefix. It selects the correct candidate first
on just 1/7 fresh cases (mean rank 2.571), including preferring `yes` to the
correct `no` on a fresh syllogism. The artifact is
`forced_choice_raw200k_20260713_mps.json` (md5
`7b6bcdd58f6420703fcb0b6bbbfa3afd`). This does not say the model cannot know
anything under another prompt contract; it does say a broadly reliable answer
representation is not waiting behind a simple emission or `<think>` adapter.
That strengthens the local-execution curriculum hypothesis: first teach a
small set of verifier-backed transitions, then test retention and compaction
before asking for language-level extrapolation.
