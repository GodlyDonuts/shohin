# Frontier Agent Plans: Shohin Evidence Review

**Status:** external-plan analysis complete; no neural run authorized by this
review alone.

**Reviewed source:** `FRONTIER_AGENT_PLANS.md`

**Original download SHA-256:**
`3d82ece081a899750c64a5a63df50e7b0405b2621f7cfd98b3e84c8ab4d942fc`

**Workspace copy SHA-256:**
`2d18b364cbad8999eaac8c30b373d4aebd4d858b8608fb33e6868c433f7124d3`

The workspace copy is content-identical to the download with one normalized
terminal newline added. The download contains 343 lines, 7,070 words, and
55,141 bytes. It contains four labeled submissions but only three distinct
plans: the Meta AI and ChatGPT Deep Research sections are identical except
that the ChatGPT copy omits Meta's final two sentences.

## 1. Executive Verdict

The plans are valuable as a design-space survey, but none is ready to launch
as written.

All three distinct plans correctly identify Shohin's central failure:

> The base has useful local execution and a real late residual signal, but it
> does not reliably compile language into operations, update and consume state,
> or halt autonomously.

Their strongest shared recommendation is also sound: stop forcing compilation,
execution, state transport, correction, and halting to share one undifferentiated
language-model loss and residual stream. Separate those interfaces physically
and educationally, then test their integration.

The plans become weaker when they move from diagnosis to mechanism. Several
components are analogies rather than update laws; several reintroduce a host
executor or fixed scheduler; and most bundles change too many variables at
once. Predictive coding, Hopfield attraction, vector quantization, an SSM,
LoRA, a reservation station, RL, and a checksum do not become a reasoner merely
by being assembled together.

The best actionable synthesis is therefore not any complete submitted stack.
It is a staged **compiler-to-state-machine post-training program**:

1. Train and independently gate a pointer-grounded natural-language compiler.
2. Train a source-deleted recurrent executor with an explicit state interface.
3. Train state consumption and halting as separate objectives and parameter
   islands.
4. Integrate only components that pass their own fresh-transfer and causal
   gates.
5. Compare every architectural addition with a favorable equal-resource
   recurrent control.

This direction is compatible with the immutable 300k base and the less-than
150M total parameter ceiling. It is a post-training protocol first, not an
immediate claim of a new computational primitive.

## 2. What The Plans Got Right

### 2.1 They used the correct empirical diagnosis

The plans are grounded in the most important Shohin results:

- DRS first-state accuracy is 497/500 while complete final accuracy is only
  275/500.
- Source-scheduled execution reaches 115/256 while autonomous whole generation
  reaches 9/256.
- Oracle-compiled frozen DRS transitions are 28/34, while fresh natural-language
  compilation is 0/6.
- A late digit residual is causally strong, but the wide digit motor changes
  autonomous accuracy only from 61/250 to 63/250.
- Typed controller v1 learns DONE but not reliable arithmetic; mixing formats in
  v2 destroys DONE.
- R9c shows that 88.83% of wrong operation decisions are agreed-wrong
  common-mode errors.

That evidence does support decomposition into compiler, executor, state,
consumer, and halt interfaces.

### 2.2 They recognized that final pretraining is a substrate, not a reasoner

None of the plans argues that another undifferentiated raw-token extension is
the main answer. That agrees with the 120k, 168.75k, and 300k plateau. The
important next educational phase is controlled post-training on the immutable
300k checkpoint.

### 2.3 They focused on organization rather than parameter count alone

Shohin has 24,918,335 parameters of nominal room below the strict 150M ceiling.
The proposals use that room for interfaces, state, and control rather than a
generic width increase. That is directionally appropriate, provided the added
mechanism earns its cost against a parameter-matched dense or recurrent arm.

## 3. Source-By-Source Review

### 3.1 Gemini Deep Research

**Proposed stack:** a Global Workspace router, 16 memory tokens, a Modern
Hopfield layer, adaptive recurrence, partial attention-to-SSM conversion, a
programmatic finite-state controller, and denoising recursion training.

**Useful elements:**

- Deep-start halting initialization is a concrete, testable ACT control.
- Memory tokens plus tied recurrence define a clear state carrier.
- Denoising self-generated state is a legitimate robustness objective.
- The proposal explicitly budgets components under the parameter ceiling.

**Scientific problems:**

1. **It is an architecture bundle, not one hypothesis.** A positive result
   could not be attributed to workspace routing, recurrence, denoising,
   memory, the SSM conversion, or deterministic gating.
2. **The FSA violates the native boundary as written.** It intercepts invalid
   tokens, prunes them, and programmatically forces transitions and completion.
   That is an external scheduler/repair controller, so it can be a ceiling or
   safety wrapper but not evidence that Shohin owns control and halt.
3. **Hopfield attraction does not define mathematical validity.** The proposed
   update is attention-like retrieval. Without a learned and independently
   verified codebook, the nearest attractor may simply be the nearest common
   mistake. Shohin's dominant errors are semantic and common-mode, not merely
   small metric perturbations.
4. **The SSM conversion addresses efficiency more directly than reasoning.**
   Replacing ten pretrained attention layers is a high-risk representation
   surgery, and no distinguishing prediction connects constant-size fading
   memory to correct operation selection.
5. **The parameter ledger is internally inconsistent.** `256 * 576 * 32` is
   4,718,592, not 4,608,000. `5 * 576 * 576` is 1,658,880, not 1,536,000. Even
   using the proposal's own row values gives a modified footprint of
   120,749,376, not 120,669,376, before accounting for the replacement SSM
   parameters. The budget must be rebuilt from actual modules.
6. **Several numerical claims are not supported inside the supplied file.**
   The stated shallow-halt frequency, 34% compute saving, 2.3x decode speedup,
   and sub-0.5% priming cost are not tied to reproducible Shohin evidence here.

**Verdict:** `NO-GO_AS_BUNDLED`. Retain deep-start ACT and denoising as ordinary
matched controls. Do not replace attention layers or install the host FSA as a
reasoning mechanism.

### 3.2 Grok

**Proposed stack:** a predictive-coding sidecar, a learned dynamical controller,
modular primitive adapters, bistable or cellular state, and a compact microcode
interface.

**Useful elements:**

- It correctly treats the post-DRS residual as a possible actuator aperture,
  not proof of a complete workspace.
- It emphasizes source-deleted rollout and counterfactual state swaps.
- It proposes physically separated modules and losses, directly addressing
  the typed-v1/v2 gradient-conflict result.
- It keeps the host boundary explicit in its proposed tests.

**Scientific problems:**

1. **Prediction error is not automatically a controller.** A sidecar trained
   to predict the next state can collapse to an auxiliary recurrent adapter.
   It must show that its state is both causally necessary and more useful than
   an equal-resource next-state model without the predictive-coding framing.
2. **The compiler remains underspecified.** A setpoint or latent goal does not
   explain how unseen paraphrases become exact ordered operations and operand
   bindings.
3. **The modular proposal risks importing the operation ontology.** If the
   router is handed named add/multiply modules or a gold schedule, the hard
   part has moved into supervision rather than been solved.
4. **Bistability protects a selected state but cannot correct a wrong semantic
   selection.** This is the same distinction exposed by the common-mode error
   result and the noise-stable-action no-go.
5. **PCMW still combines several hypotheses.** Predictive coding, modular
   routing, bistability, recurrence, and learned halt need individual ablations
   before a combined run has scientific meaning.

**Verdict:** `PROMISING_DECOMPOSITION`, but not a frozen experiment. The most
valuable extract is phase-separated parameter islands plus a source-deleted
state interface. Predictive error should enter as one ablation, not as the
unquestioned foundation.

### 3.3 Meta AI and ChatGPT Deep Research

These are one distinct submission, not two independent votes.

**Proposed stack:** temporal compiler/executor LoRAs, a three-field pointer
codon, a quantized abacus, read/write buses, a reservation station, an
RL-trained halt controller, a trajectory checksum, dual compiler views, and a
negative-program discriminator.

#### Pointer codon and temporal differentiation

This is the strongest component. An explicit
`[operation, source-A pointer, source-B pointer]` interface directly targets
the observed binding and compilation failures. Separating compiler and
executor parameters and stopping gradients at the interface could prevent the
response-mode interference seen in typed v2.

The biological error-correction argument is not valid as stated. The three
fields are different semantic fields, not three redundant transmissions of the
same symbol, and mapping `add`, `plus`, and `sum` to one embedding is canonical
classification rather than an error-correcting code. The mechanism can still
be useful without that analogy.

The proposed LoRA count is not auditable until the exact target matrices,
ranks, biases, and shared parameters are listed. The phase notation also must
state whether compiler and executor are separate forward calls, depth phases
inside one call, or training phases.

#### Quantized abacus and reservation station

Explicit categorical state is testable, but categorical representation alone
has already proved insufficient in packet/tape controls. The required new
evidence is an autonomous source-deleted transition law that consumes its own
previous state.

The sentence "carry becomes trivial" hides the central problem. If overflow
and carry propagation are fixed rules, the architecture contains an arithmetic
executor. If they are learned, then carry is not trivial and must pass the same
width/value/order transfer gates as any other updater. Likewise, a reservation
station that decides readiness and execution order is a scheduler whose state,
logic, and supervision must be counted.

Read/write parameter separation is a plausible treatment for destructive
interference. It should be compared against an equally large ordinary recurrent
adapter with the same state bits and sequential depth.

#### RL halt and checksum

Training a separate halt policy with a different objective is worth testing.
The typed-controller evidence makes loss separation more plausible than merely
adding another DONE token.

The checksum does not detect the proposed common-mode semantic error. A rolling
hash of the chosen operations and intermediate digits can be perfectly
consistent with a completely wrong operation sequence. It only detects
corruption relative to that wrong trajectory unless there is an independent,
source-derived expected invariant. If an external process supplies that
expected invariant, its information and computation must be counted.

"Distance to closed form" is also undefined. If computing it requires knowing
the correct terminal state, it is an answer oracle. A valid halt energy must be
computed only from model-owned state and must be tested against premature and
late halt twins.

#### Dual views and negative selection

Two tokenizations or reading directions are not guaranteed to be independently
grounded; they may reproduce the same lexical error. This proposal is closely
related to the existing pre-emission dual-view commit lane and therefore cannot
be treated as unexplored merely because it uses immunology or market language.

A wrong-program discriminator is potentially useful for compiler post-training,
but it must be trained on source-preserving hard minimal pairs. Randomly wrong
programs are usually easy negatives and may teach formatting artifacts rather
than semantic rejection. The discriminator may reject or abstain; it may not
search for and install a corrected program without that repair work being
counted.

#### Proposed phase gates

The numerical gates are directionally useful, but Phase 0 is ill-defined: an
untrained pointer-codon/VQ architecture cannot be expected to improve a 0/6
compiler "with zero GPU training." CPU training would still be training and
must be specified. Phase 1 changes abacus state, bus separation, scheduling,
and training together. Phase 2 changes optimizer, controller, halt, checksum,
and reward together. Both need factorial ablations or a narrower first run.

**Verdict:** `EXTRACT_AND_REWRITE`. Preserve the pointer-grounded compiler,
parameter/loss separation, and explicit halt policy. Reject the biological
coding claim, fixed carry shortcut, unchecked reservation station, and checksum
claim as written.

## 4. Consensus Does Not Equal Evidence

The submissions converge on four themes:

| Theme | Evidence status in Shohin |
|---|---|
| Separate compiler from executor | Strongly motivated by compiler 0/6 and typed-v2 interference |
| Use explicit mutable state | Necessary but already known insufficient by itself |
| Add recurrent or adaptive compute | Ordinary strong control; prior recurrence alone did not win |
| Train halt separately | Strongly motivated by halt-first and typed-controller results |

This consensus is useful for prioritization. It is not a mechanism proof. The
plans were given the same ledger and therefore inherit the same diagnosis. The
Meta and ChatGPT sections are duplicates, further reducing the apparent number
of independent endorsements.

## 5. Highest-Value Components

### Tier A: worth formal preregistration

1. **Pointer-grounded compiler.** Train exact operation and source-span
   binding on frozen natural-language families. Score exact AST/program
   equality on unseen paraphrases, role names, values, operation orders, and
   distractors. Compare with a parameter-matched ordinary sequence classifier.
2. **Phase-separated compiler/executor parameters.** Use separate adapters or
   modules and an explicit stop-gradient interface. Compare with one jointly
   trained adapter using the same total parameters, examples, updates, and
   inference depth.
3. **Source-deleted recurrent state executor.** After compilation, remove the
   natural-language source and require the model to update, retain, and consume
   its own state. Compare categorical, continuous, and favorable recurrent
   controls under equal state bits.
4. **Independent halt policy.** Train halt/continue on prefix twins where the
   same local answer token appears in nonterminal and terminal contexts. Test
   premature halt, late halt, output recoding, and state swaps.

### Tier B: useful ablations, not foundations

- predictive-error auxiliary loss;
- denoising of self-generated state;
- VQ state versus continuous state;
- deep-start ACT bias;
- read/write parameter separation;
- hard-negative program discrimination.

### Tier C: reject or retain only as ceilings

- host FSA token pruning or forced completion;
- fixed carry propagation or host readiness scheduling;
- an unchecked rolling checksum as semantic correction;
- wholesale attention-to-SSM replacement as a reasoning intervention;
- a full multi-component architecture launch without component gates;
- claims that Hopfield attraction, quantization, or bistability alone creates
  valid mathematical state.

## 6. Recommended First Program

The first experiment should test education and interface separation before
installing a large cognitive architecture.

### Stage A: 300k compiler post-training

Freeze the 300k checkpoint and train only a small compiler island. Its output
is a typed operation with source-span pointers and an explicit terminal marker.
No arithmetic target or answer token is included in this stage.

Primary gate:

- exact full program on a frozen, family-balanced board;
- per-field operation and pointer accuracy;
- order-twin and renamed-role accuracy;
- zero source/target leakage;
- no score based only on parseability.

Controls:

- equal-parameter plain classifier;
- text-token program SFT without pointers;
- shuffled-pointer control;
- source-only lexical-family control.

### Stage B: isolated state transition post-training

Train an executor island from exact compiled programs and explicit previous
state. Remove the natural-language source after state creation. The model must
produce the next internal state without host arithmetic.

Primary gate:

- width/value/order transfer;
- complete multi-step trajectory exactness, not only first state;
- donor-state following;
- zero/shuffled/complement state degradation;
- improvement over a favorable equal-resource tied recurrent control.

### Stage C: state consumption and halt

Freeze the compiler and executor before generating this board. Train a separate
consumer/halt island on terminal/nonterminal twins and state reuse. The same
answer-like local configuration must sometimes continue and sometimes halt.

Primary gate:

- exact halt location;
- no cap-hit or post-answer replay;
- source-deleted reuse;
- output-recoding stability;
- causal dependence on the committed state.

### Stage D: autonomous integration

Only after A-C pass individually, compose the islands and allow limited joint
calibration. Evaluate full natural-language-to-answer rollout with no oracle
program, schedule, arithmetic, repair, or result selection. Report both the
product of component accuracies and observed end-to-end exactness; a large gap
is itself an integration failure.

## 7. Decision

The external plans materially improve our hypothesis inventory, especially by
reinforcing staged compiler/executor/halt separation. They do not overturn the
existing Shohin diagnosis and do not justify an immediate H100 architecture
run.

The research decision is:

> **ADMIT the pointer-grounded, phase-separated post-training program for
> formal preregistration. Keep predictive coding, VQ state, denoising, ACT, and
> read/write buses as individually matched ablations. Reject the submitted
> integrated stacks as scientifically underidentified.**

No claim in this review establishes native reasoning, novelty, or expected
benchmark gain. It identifies the smallest experiments that can convert the
plans from analogies into causal evidence.
