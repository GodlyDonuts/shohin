# R11 Preregistration: Interchange-Trained Internal Broadcast Workspace

> **Status:** broad design rejected as the first canary by adversarial review.
> No R11 model has been built, trained, scored, or shown to work. This document
> is retained as possible R11b background only and does not authorize code,
> H100 use, touching the protected writer, or any capability claim. The narrower
> crossed-query R11a preregistration must be frozen and pass CPU mechanics first.
>
> **Scope:** one new, isolated architecture experiment for the immutable Shohin
> `best_step200000.pt` base. R11 is not an R10 continuation, an external
> executor, a context-compression system, or a chain-of-thought recipe.
>
> **As read:** 2026-07-14. `AGENT_RUNBOOK.md` and
> `REASONING_FRONTIER.md` are the operational and evidence sources of truth.

## 1. Decision in one paragraph

This broad draft proposed an **Interchange-Trained Recurrent Broadcast Workspace
(IT-RBW)** inside the exact 30-layer, 576-wide Shohin decoder. A source-only
pass writes six private 96-wide latent slots. One weight-tied neural cell
refines those slots four times. A source-free query pass reads the same slots
through cross-attention adapters at three separated decoder depths and emits an
ordinary answer with Shohin's existing tied output head. The state is never
serialized as text, never decoded into opcodes, and never executed by host
code. Solver metadata may create training targets and counterfactual pairs, but
the model forward at inference receives only token IDs, lengths, masks, and its
own latent state. The central falsifiable claim is not that an extra adapter
improves accuracy. It is that one bounded, model-written state is necessary,
causally interchangeable between examples, updateable without text rollout,
and reusable by readers that were not jointly trained with its writer.

## 2. What the existing evidence forbids

R11 is constrained by the complete negative chain, not motivated by ignoring
it.

| Evidence | Result that must be preserved | R11 consequence |
|---|---|---|
| Earlier VRW internal-state test | Four recurrent slots trained from final-answer loss were nearly invariant to matched state shuffling (`0.00377` all-case NLL change), and reset beat recurrence. | Final-answer loss alone is not allowed to identify the workspace. R11 trains and scores counterfactual state transplants directly. |
| R6 active distinction | Active probes beat equal-call random by `+10.94` operation points, yet produced only `36/448` answers and `19/448` exact programs; unseen-language/full effect MSE was about `130.74`. | Causal use of an intervention is not semantic transport. R11 must pass source-deleted, language/value/depth OOD state transfer, not just a policy margin. |
| R7 first-order response fields | Active was `32/108`, below direct hidden similarity at `46/108`; numeric events were especially weak. | Do not assume a frozen derivative exposes the missing state. R11 installs a trainable state interface. |
| R8 mixed curvature | Curvature was nonzero but scored `26/108`, below random pairs (`28/108`) and direct (`46/108`). | Larger hidden responses are not evidence of aligned semantics. No layer, pair, or normalization search is part of R11. |
| R9a static recurrence | Static orbit replay had an exact one-shot form and transformed-label cross-entropy equivalence. | Repeated calls count only if state-dependent attention changes later evidence and R=4 beats equal-call reset and one-round controls. Weight tying alone is not a mechanism result. |
| R9c dynamic syndrome | Dynamic paths were real, but treatment operation/answer accuracy (`78.29% / 47.77%`) lost to static, no-syndrome, and shuffled-goal controls. `88.83%` of wrong operations were common-mode. Its expected-operator syndrome was also non-injective for executed categorical decisions. | Do not use agreement, confidence, or two views of one local classifier as a certificate. R11 uses an independently scored donor-state intervention whose target cannot be reconstructed from the receiver text. |
| R10 ACAW | As of this document, R10 is pre-score. It is explicitly an external neuro-symbolic test: calibrated candidate sets, exact affine composition, parsed numeric slots, a product tree, and host-side certificates. | R10 is not a failed score, but it is disqualified as evidence of internal thinking. R11 may use none of its version-space, affine, parser, or certificate machinery in the model forward. |

The positive R4 fact is also retained: dynamic entity pointer binding improved
language/full transfer substantially. R11 may learn relational binding through
the model, but it may not receive entity IDs, mention tags, operation labels,
query labels, or numeric slots at inference.

## 3. How the 2026 workspace paper is used

The July 2026 study
[Verbalizable Representations Form a Global Workspace in Language Models](https://transformer-circuits.pub/2026/workspace/index.html)
is an experimental standard, not an architecture recipe. It treats a
workspace-like representation as reportable, deliberately modulable,
causally used in intermediate computation, flexibly reusable by different
downstream functions, and selective rather than required for routine language
processing.

R11 maps those criteria to preregistered tests:

| Paper criterion | R11 operational test |
|---|---|
| Reportability | A natural-language report query reads the same latent state used by non-report consumers; a donor-state transplant must change the report to the donor state. |
| Directed modulation | A natural-language update segment changes the private state, after which two independent consumers must read the updated result. |
| Intermediate causal use | Replacing the receiver workspace with a donor workspace must redirect the receiver's answer to the solver-defined donor counterfactual while visible query tokens stay fixed. |
| Flexible reuse | One source-written state is reused without recomputation by multiple consumers, including a consumer trained on explicit states but never jointly trained with latent workspace states. |
| Selectivity | Workspace ablation must strongly hurt flexible composition while leaving matched copy, local lookup, syntax, and ordinary language NLL nearly unchanged. |

The paper primarily studies much larger production models and explicitly leaves
model-size scaling unresolved. Shohin's exact future-Jacobian map was stable but
had `0%` top-10 and `0%` top-100 semantic recovery on the registered concept
targets. R11 therefore does not use a J-lens basis, token-named coordinates, or
post-hoc workspace-layer discovery. The layer sites below are architecture
choices frozen before fitting. The paper's counterfactual-reflection training is
also not the R11 treatment: no interrupted reflection turn or verbal reflection
target appears on ordinary R11 examples.

## 4. Hypothesis and claim boundary

### 4.1 Hypothesis

A 125.1M decoder can learn a small common representational interface if all of
the following are true during training:

1. Source information must cross a private fixed-capacity bottleneck before an
   answer can be produced.
2. The identical state must serve multiple query-conditioned consumers.
3. Whole-state counterfactual transplants are trained and evaluated, making
   state identity causally consequential rather than merely decodable.
4. The state can be updated by a new text command and reread without producing
   or consuming an intermediate rationale.
5. Mid-layer LoRA updates are allowed to install the write/read interface; the
   experiment does not assume raw pretraining already contains it.

### 4.2 What a pass would mean

A full pass would establish only this narrow result:

> On frozen, source-deleted synthetic language domains, a parameter-efficient
> Shohin variant learned one bounded internal latent state that was written from
> text, causally transplanted, updated, and reused by multiple neural readers.

It would not establish general reasoning, hidden human-like thought, a global
workspace in the philosophical sense, broad benchmark gains, safe context
compression, irreversible source deletion, or a publication-level novelty
claim.

## 5. Established ingredients versus project-novel composition

### 5.1 Established ingredients

The individual components have substantial prior art:

- Iterative cross-attention into a compact latent bottleneck is established by
  [Perceiver](https://arxiv.org/abs/2103.03206) and query-based decoding by
  [Perceiver IO](https://arxiv.org/abs/2107.14795).
- Iteratively updated latent slots are established by
  [Slot Attention](https://arxiv.org/abs/2006.15055).
- Weight-tied recurrent transformer computation is established by the
  [Universal Transformer](https://arxiv.org/abs/1807.03819), and persistent
  memory tokens by the
  [Recurrent Memory Transformer](https://arxiv.org/abs/2207.06881).
- Low-rank adaptation of frozen transformer weights is established by
  [LoRA](https://arxiv.org/abs/2106.09685).
- Counterfactual activation replacement as a training objective is established
  by
  [Interchange Intervention Training](https://arxiv.org/abs/2112.00826), and
  activation patching is an established causal diagnostic.
- Continuous latent reasoning and extra hidden computation already have direct
  prior art in
  [Coconut](https://arxiv.org/abs/2412.06769) and
  [pause-token training](https://arxiv.org/abs/2310.02226).
- The reportability, modulation, causal-use, broadcast, and selectivity tests
  are adapted from the 2026 global-workspace study cited above.

### 5.2 Project-novel composition under test

The bounded project hypothesis is their composition for this exact model:

1. A source-only 18-block pass over the existing Shohin decoder writes exactly
   six 96-wide slots from three preselected depth taps.
2. One state-conditioned, weight-tied neural cell refines those slots four
   times without visible tokens or step-specific parameters.
3. A source-free 30-block query pass reads that state after blocks 12, 18, and
   24 through three independent depth interfaces.
4. Whole-state interchange training, multi-consumer reuse, held-out consumer
   recombination, and a text-command state update jointly prevent final-answer
   correction from standing in for a workspace.
5. The evidence package requires both source-free necessity and source-visible
   selectivity, plus matched no-interchange, reset, and private-query controls.

This is a project-novel experimental composition, not a world-first claim. A
broader literature review would be required before stronger novelty language.

## 6. Exact integration with Shohin

All layer numbers in this document are **one-based**.

| Frozen base property | Value |
|---|---:|
| Decoder blocks | 30 |
| Residual width | 576 |
| Query / KV heads | 9 / 3 |
| Head width | 64 |
| SwiGLU width | 1,536 |
| Vocabulary | 32,768 |
| Native context | 2,048 |
| Tied base parameters | 125,081,664 |

The base checkpoint is immutable. Workspace mode is a separate forward path.
When `workspace_mode=False`, every LoRA delta and workspace reader is bypassed,
and logits must be bitwise equal to the current base path under the same dtype
and device. R11 checkpoints contain adapters and a complete base/config/data
manifest, not a modified flagship checkpoint.

Rank-8 LoRA is active only in workspace mode on every bias-free attention and
MLP projection in blocks 7 through 18. The same LoRA tensors are used in source
and query passes. All original base tensors, embeddings, norms, and output head
remain frozen.

LoRA `A` matrices use the fixed seed's normal initialization with standard
deviation `0.02`; LoRA `B` matrices start at zero. Reader scalar gates start at
zero. The three writer residual gates start at `atanh(0.1)` so source-dependent
state updates are nondegenerate as soon as the reader gates open. All matched
arms must share the resulting exact initial-adapter hash.

## 7. Concrete forward pass

### 7.1 Runtime inputs

The primary API is one model call:

```text
logits = model.forward_workspace(source_ids, query_ids, answer_prefix_ids)
```

`source_ids` and `query_ids` are raw tokenizer outputs. The model call owns the
latent tensor; production inference does not serialize or expose it. Audit mode
may capture or replace the tensor at named intervention hooks.

For the synthetic canary, source and query are already distinct natural-language
fields. For an ordinary single-question prompt, the complete user question is
the source and the fixed text `Answer the problem.` is the query. A literal
field boundary is an encoder/decoder protocol, not semantic metadata. No host
code extracts entities, values, operations, equations, or answers.

### 7.2 Source encoding

The treatment's workspace is source-only. For exact matched-arm training, the
18-block source-side call processes `[source || query]`, but the treatment masks
every query position out of the workspace keys and values. Because the query is
a causal suffix, it cannot change any source-position residual. The
private-query control differs only by unmasking those already-computed query
positions. Production treatment inference may omit that redundant suffix and
cache one state per source.

Residuals after Shohin blocks 6, 12, and 18 are captured. Shared source key and
value projections are applied to source positions only, then one learned
96-wide layer ID is added to both projected streams:

```text
K_src = concat(
    RMSNorm(H_6[source])  K_w + e_6,
    RMSNorm(H_12[source]) K_w + e_12,
    RMSNorm(H_18[source]) K_w + e_18
)

V_src = concat(
    RMSNorm(H_6[source])  V_w + e_6,
    RMSNorm(H_12[source]) V_w + e_12,
    RMSNorm(H_18[source]) V_w + e_18
)

K_src, V_src in R[B, 3*T_source, 96]
```

Padding and query positions are masked in treatment. No answer or answer-prefix
token enters this call.

### 7.3 Recurrent write cell

The private workspace has six slots of width 96:

```text
W^0 in R[B, 6, 96]
```

`W^0` is a learned six-slot seed for a fresh source. For an update segment,
`W^0` is the previous model-written workspace. There are exactly four rounds,
with all parameters tied and no round embeddings.

For `r = 1..4`:

```text
E^r = Attention(
    Q = RMSNorm(W^(r-1)) Q_w,
    K = K_src,
    V = V_src
)

U^r = W^(r-1) + tanh(g_write) * (E^r O_w)
V^r = U^r + tanh(g_self) * SelfAttention(RMSNorm(U^r))
W^r = V^r + tanh(g_mlp) * SwiGLU(RMSNorm(V^r))
```

The workspace attention has three heads of width 32. Its MLP width is 256.
Source keys and values are projected once and reused across rounds. Each round's
attention query depends on the state written by the prior round, so later
evidence selection is state-dependent. The exact need for recurrence remains an
empirical claim and is killed if the one-round or equal-call reset controls
match it.

The final state is `W = W^4`. It is not normalized into a vocabulary basis,
decoded to text, assigned opcode coordinates, or passed to external code.

### 7.4 Natural-language state update

An update episode calls the same internal source encoder and the same four-round
write cell on a raw text command, initializing the slots with the previous
state:

```text
W_after = Write(command_ids, W_before)
```

The command may say, for example, that one named quantity receives three more
units or that two named containers exchange contents. The host does not parse
or execute it. The update is accepted as evidence only if downstream readers
and state transplants follow the model-written post-update state on unseen
language, values, and compositions.

### 7.5 Query and answer pass

The source is absent from the primary reader pass. Query tokens plus the
teacher-forced answer prefix run through all 30 Shohin blocks. After blocks 12,
18, and 24, an independent three-head cross-attention reader injects the same
workspace:

```text
R_l = Attention(
    Q = RMSNorm(X_l) Q_l,
    K = RMSNorm(W) K_l,
    V = RMSNorm(W) V_l
) O_l

X_l <- X_l + tanh(g_l) * R_l
```

The reader gate at each site starts at zero. The standard remaining blocks,
final RMSNorm, and tied language-model head produce answer logits. There is no
workspace-specific classifier or symbolic output head.

During teacher forcing, `W` is computed from source tokens only and is fixed for
the answer sequence, preventing completion leakage. During autoregressive
inference, it is cached beside the ordinary KV cache. A second workspace write
occurs only at an explicit natural-language update segment, never after seeing a
gold answer.

### 7.6 Source-visible audit mode

A separately named audit mode gives the reader the original source and query in
the normal decoder context. It is not the primary capability score. It exists to
measure causal mediation and selectivity under a setting where a bypass is
possible:

- clamp the workspace while editing the source;
- ablate readers on flexible versus automatic tasks;
- verify that the workspace is not merely required because the architecture
  hid all source tokens.

Primary source-free scores and source-visible audit scores are never pooled.

### 7.7 Why this is not ordinary CoT or R10

- No rationale, scratch text, pause token, soft token, or continuous output
  token is inserted into the autoregressive sequence.
- No intermediate target is a chain of reasoning. The only verbal state target
  appears when the query explicitly asks for a report.
- The private state is updated by neural cross-attention, self-attention, and an
  MLP, not by matrix algebra, a parser, a search procedure, a candidate set, or
  an external executor.
- The host transports an opaque tensor only inside one model forward. It never
  reads its values to decide an operation.

## 8. Frozen parameter budget

All projections are bias-free. LoRA rank is eight. The table is exact for the
specified design.

| Trainable component | Count |
|---|---:|
| Rank-8 LoRA on `q/k/v/o/gate/up/down` in 12 blocks (`81,408` per block) | 976,896 |
| Workspace cross-attention | 129,024 |
| Workspace self-attention | 36,864 |
| Workspace SwiGLU (`96 -> 256 -> 96`) | 73,728 |
| Workspace norms, six seeds, three layer IDs, and three scalar gates | 1,731 |
| Three depth readers, including norms and scalar gates (`129,697` each) | 389,091 |
| **Total trainable R11 parameters** | **1,607,334** |
| **Base plus R11 resident parameters** | **126,688,998** |

R11 adds `1.285028%` relative to the 125,081,664-parameter base and remains well
under the project sub-200M limit. Increasing slot count, slot width, LoRA rank,
reader sites, or recurrent rounds is forbidden after a used-board score. Any
such change is a new experiment.

The live workspace itself is exactly `6 * 96 = 576` scalars: 1,152 bytes in
bf16 or 2,304 bytes in fp32 per example, excluding temporary attention
activations. This is a capacity statement, not a context-compression claim.

## 9. Frozen compute budget

Let `L_s` be source length and `L_q` query plus answer-prefix length. The
matched experimental source-side call includes a query suffix so the
private-query control changes a mask rather than token count. Ignoring the
small adapters, worst-case base-block token work is:

```text
R11:    18 * (L_s + L_q) + 30 * L_q
direct: 30 * (L_s + L_q)
```

The deployable treatment can cache `W` from a source-only call and therefore
uses `18 * L_s + 30 * L_q`; the larger expression above is the fairness budget
for the preregistered neural comparison.

Workspace projection and attention add `O(4 * 6 * 3L_s * 96)` attention work,
and each query token attends to six slots at three layers. There is no quadratic
attention over added latent tokens.

Hard measured budgets for the first production canary are:

1. Clean workspace prefill wall time and peak allocated CUDA memory must each be
   no more than `1.15x` the direct base on the frozen source/query length mix.
2. Cached decode time per generated token must be no more than `1.10x` the base.
3. A treatment training step containing one clean and one transplant reader
   pass must be no more than `2.25x` the clean-workspace step.
4. Recurrent rounds are fixed at four. There is no learned halting threshold,
   adaptive replay, beam search, sampling, or verifier call.
5. Every matched neural arm executes the same source blocks, four workspace
   calls, three reader sites, reader-token count, and optimizer-update count.

Measured violations kill the design before a capability score. Budgets may not
be relaxed because the model appears promising.

## 10. Inference contract: no oracle metadata

### 10.1 Inputs allowed to reach the model

- source token IDs and padding mask;
- query/answer-prefix token IDs and causal mask;
- a previous model-written workspace only for an explicit update segment;
- fixed architecture constants and learned parameters.

### 10.2 Inputs forbidden at inference

- operation, query, role, entity, mention, line, value, or state labels;
- structured programs, ASTs, equations, parsed integers, candidate sets, or
  affine transforms;
- solver outputs, gold intermediate states, confidence certificates, or
  retrieval decisions;
- donor identity or counterfactual answer metadata;
- visible or hidden chain-of-thought text supplied by a teacher.

Structured records may exist in the offline generator solely to produce target
answers, pair examples, stratify boards, and audit correctness. The collator
must build a fresh minimal object containing only token IDs, masks, and targets.
A metadata-poison test must randomize every unused structured field and show
bitwise-identical logits and workspaces. A second test must delete all such
fields and still run the complete forward.

The fixed source/query boundary is disclosed input protocol, analogous to an
encoder-decoder boundary. It contains no semantic interpretation. For ordinary
prompts the entire prompt is the source, eliminating any need for a parser.

## 11. Training tasks

### 11.1 Core source-state domain

The first admitted domain is a text-only two-register world, selected because
R4-R10 provide strong comparators and exact offline verification. Each source
contains:

- two introduced entities with integer quantities;
- two to four training events expressed in natural language;
- the existing exact event semantics: add/subtract change one target by a
  stated value; move subtracts a stated value from one source and adds it to one
  target; merge adds the complete source value into the target while preserving
  the source; swap exchanges the two values;
- entity labels, templates, domains, values, and event order rendered into raw
  text.

The writer sees only this text. It must leave one workspace usable by these
query consumers:

1. value of the first referenced entity;
2. value of the second referenced entity;
3. signed difference;
4. greater/equal comparison;
5. exact two-value report, only when explicitly requested;
6. sum, used as the held-out workspace/consumer recombination described below.

The state is also passed through one natural-language update and then queried by
two consumers. All targets are answer-only. Signed states are allowed. The
generator admits only episodes whose every intermediate and post-update value
lies in `[-2048, 2048]`; that bound is identical across partitions and checked
before rendering. Silent clamping, wraparound, or saturation is forbidden.

### 11.2 Held-out consumer recombination

The sum reader is grounded during training on explicit natural-language state
descriptions placed directly in a source-visible decoder context with workspace
readers disabled, so the decoder learns the function without receiving `W`.
It is never trained on a workspace-written state. Conversely, the workspace
writer is trained with the other consumers but never with sum as a latent-state
consumer. The used and confirmation boards combine them for the first time:

```text
source text -> private W -> unseen W-plus-sum pairing -> answer
```

This is stronger than withholding query wording. It tests whether the state is
in a common reader-usable format rather than a bank of answers for jointly
trained heads.

### 11.3 Transfer families

R11 cannot advance on arithmetic alone. The same architecture and objective
must also train and score without family-specific modules on:

- a referential inventory domain with three named containers, moves, removals,
  swaps, membership, count, and equality readers;
- a tiny straight-line code domain with named variables, assignment, add,
  subtract, swap, and variable/output readers.

At least one complete transfer family must pass the full OOD and causal gates
independently. No family-specific parser, head, slot, or loss is allowed.

### 11.4 Frozen split factors

Training groups use event/program depths 2 through 4 and values 0 through 39.
Evaluation partitions are disjoint:

| Partition | Changed factors |
|---|---|
| Fit | New programs and values, training render banks, depths 2-4 |
| Language OOD | Unseen event/query wording, domains, and entity labels |
| Value OOD | Values 40-99 and unseen update magnitudes, training language/depth |
| Depth OOD | Depths 6, 8, and 12, training language/value range |
| Full OOD | New language, values, labels, domains, depths, and update/consumer pairings |

The held-out sum pairing appears only on evaluation boards. Update operations
are individually seen in training, but at least half of update/consumer
combinations on full OOD are composition-held out.

### 11.5 Dataset and board sizes

- Training: exactly 32,768 semantic source groups: 16,384 register, 8,192
  inventory, and 8,192 straight-line-code groups. Each group supplies four
  clean latent consumers, one report request in only 25% of groups, one update,
  two post-update consumers, and one donor pairing.
- Explicit-reader grounding: 16,384 disjoint explicit-state groups, including
  sum, with no latent workspace state.
- Used board: 3,840 source groups, exactly 256 per
  `(partition, family)` cell across five partitions and three families.
- Confirmation board: 3,072 source groups, exactly 256 per
  `(partition, family)` cell across language, value, depth, and full OOD. It has
  no fit partition.
- Automatic/selectivity board: 1,024 answer-free or locally answerable prompts,
  balanced across copy, syntax, next-token continuation, and one-step lookup.

Before a fit, independent admission must prove exact solver replay, partition
factor contracts, donor/receiver answer differences, unique normalized
sources/queries, zero exact prompt overlap, zero forbidden 13-gram overlap, and
zero public-evaluation overlap. Train, used, confirmation, selectivity,
tokenizer, base, code, and admission hashes must be written into this document
or an immutable manifest referenced here before any H100 job is authorized.
This design document alone does not supply that authorization.

## 12. Objective and training schedule

### 12.1 Losses

Every loss is mean token cross-entropy over answer tokens only unless stated
otherwise:

```text
L = 1.00 * L_clean
  + 1.00 * L_transplant
  + 0.50 * L_update
  + 0.50 * L_reader_ground
  + 0.25 * L_report
  + 0.10 * L_automatic_KL
```

- `L_clean`: ordinary source-to-workspace-to-query answers for the seen
  consumers.
- `L_transplant`: compute `W_A` and `W_B`, hold receiver query tokens fixed,
  replace `W_A` by `W_B`, and train against the offline solver answer for
  applying the receiver query to donor state B.
- `L_update`: apply a text command to `W`, then answer two post-update queries.
- `L_reader_ground`: answer from explicit visible state text with all workspace
  readers disabled. This grounds held-out consumer functions without pairing
  them with a latent state.
- `L_report`: answer an explicit report request through the same decoder and
  workspace readers. It is not active on normal answer prompts.
- `L_automatic_KL`: with source-visible audit mode, keep automatic-board logits
  close to the immutable base. The base-disabled workspace path remains exactly
  bypassable regardless of this term.

There is no loss on slot coordinates, attention maps, operation labels,
intermediate numeric states, latent rounds, or rationales. The solver specifies
only expected output behavior under clean and intervened worlds.

### 12.2 Fixed first fit

- Base: immutable `best_step200000.pt` with all 125,081,664 parameters frozen.
- Trainable tensors: exactly the 1,607,334 parameters in Section 8.
- Optimizer: AdamW, betas `(0.9, 0.95)`, weight decay `0.01` on matrix tensors
  and zero on norms, seeds, and scalar gates.
- Peak learning rate: `3e-4`; 128-update linear warmup; stable through 80% of
  updates; linear decay to `3e-5` over the final 20%.
- Batch: 16 semantic groups plus eight explicit-reader-grounding groups per
  optimizer update, shape-bucketed; gradient accumulation may change microbatch
  size but not either semantic batch size.
- Updates: exactly 2,048, one pass over 32,768 source groups.
- Precision: bf16 forward/backward, fp32 loss and gradient norm, clip at 1.0.
- Seed: `20260715` for data order and adapter initialization.
- Hardware: one isolated H100 per arm after a real CUDA/bf16 allocation smoke.

No second epoch, learning-rate sweep, slot-width sweep, layer sweep, or round
sweep is authorized by a near miss.

## 13. Matched neural arms

All arms share base, tokenizer, data, semantic-group order, active parameter
count, initialization hash, optimizer, updates, source/query token counts, four
workspace calls, three reader calls, and maximum compute.

| Arm | Difference from treatment | Question answered |
|---|---|---|
| Treatment | Full objective and one reusable source-only workspace | Does the proposed mechanism work? |
| No-interchange | `L_transplant` is replaced by a second clean reader pass with equal target tokens and compute | Are clean multi-task targets sufficient without causal state training? |
| Reset recurrence | Every round starts from the same `W^0`; four outputs are averaged before reading | Does information accumulation matter, or is this an extra prompt-conditioned adapter? |
| Private-query memory | The already-computed query suffix is unmasked in workspace cross-attention and a fresh state is built per consumer; treatment processes the same suffix but masks it | Does a reusable broadcast state beat equally capable query-local computation? |

Two diagnostic baselines are reported but cannot substitute for the matched-arm
attribution:

- conditional LoRA-only direct decoding with source and query visible;
- a compute-matched latent/pause-token baseline with no workspace transplant
  objective.

If a control cannot be made active-parameter and compute matched, the mismatch
must be reported and that control cannot support a causal attribution.

## 14. Causal intervention protocol

### 14.1 Pair construction

Donor and receiver are matched on token-length bucket, domain, depth, query
consumer, and answer-token length. Their solver states and correct answers must
differ. Pair eligibility is fixed by the generator before any model runs. The
primary metric uses every generator-eligible pair, not only pairs the clean
model answers correctly. A clean-correct subset is secondary and explicitly
named.

For receiver `A`, donor `B`, and receiver query `q_A`:

```text
clean:       y_hat_A  = Read(q_A, W_A)
transplant:  y_hat_AB = Read(q_A, W_B)
target:      y_AB     = SolverAnswer(q_A, state_B)
```

Primary state-swap metrics are:

- `donor_follow`: `y_hat_AB == y_AB`;
- `receiver_follow`: `y_hat_AB == y_A`;
- `swap_specificity`: donor-follow rate minus receiver-follow rate;
- donor logit-margin change relative to clean;
- exact all-consumer donor episode, requiring every query to follow the same
  donor state.

### 14.2 Write evidence

The workspace counts as written only if all of these hold:

1. Whole-state donor replacement redirects answers to the donor counterfactual.
2. A one-fact source edit changes `W` enough that patching edited `W` into the
   original reader produces the edited answer.
3. Clamping clean `W` while applying the source edit removes most of the answer
   effect in source-visible audit mode.
4. Two independently rendered descriptions of the same world produce states
   interchangeable for all consumers, while different worlds do not.

Cosine similarity, linear-probe accuracy, and nonzero gradients are diagnostics
only. They cannot satisfy the write gate.

### 14.3 Read evidence

The workspace counts as read only if:

1. Zero, batch-shuffled, and matched-norm Gaussian states each reduce flexible
   accuracy substantially without a donor-specific direction.
2. The donor state produces the donor-specific answer, not merely any changed
   answer.
3. Disabling all three reader gates destroys most flexible performance.
4. At least two separated reader sites have a nontrivial causal contribution;
   a block-24-only motor correction is insufficient.
5. A held-out consumer and a post-update consumer both respond to the same
   transplanted state.

### 14.4 Frozen intervention conditions

Every board is run under:

1. clean `W^4`;
2. `W^1` only;
3. four-round reset state;
4. all-zero state;
5. within-bucket shuffled state;
6. matched-norm Gaussian state;
7. whole-state donor transplant;
8. donor transplant at only reader 12, only reader 18, only reader 24, and all
   readers;
9. reader-12 lesion, reader-18 lesion, reader-24 lesion, and all-reader lesion;
10. clean-state clamp under a one-fact source edit in source-visible audit mode.

No slot subset, arbitrary rotation, layer, or intervention strength is selected
after scores are visible.

## 15. Preregistered advancement gates

All gates are conjunctive. Percentages are computed separately by partition and
family; no partition may subsidize another.

### Gate 0: mechanics and isolation

1. Workspace-disabled logits are bitwise equal to the immutable base.
2. Changing or appending answer tokens cannot change source-written `W` beyond
   `1e-6` max absolute fp32 audit error.
3. Metadata deletion and poison tests leave `W` and logits bitwise unchanged.
4. Base gradients are absent; trainable parameter count is exactly `1,607,334`.
5. All states, losses, logits, and gradients are finite; local/Newton adapter
   hashes match.
6. The measured compute and memory caps in Section 9 pass.

Any failure blocks all capability fitting or scoring.

### Gate 1: clean capability

The table applies independently to the core register family and to the
all-family macro average. A transfer family is not silently averaged into the
core result; its separate minimum is stated below. Treatment answer accuracy
must reach:

| Partition | Minimum answer accuracy |
|---|---:|
| Fit | 85% |
| Language OOD | 65% |
| Value OOD | 60% |
| Depth OOD | 60% |
| Full OOD | 45% |

In addition:

- strict all-four-seen-consumer accuracy is at least 40% on full OOD;
- at least one non-arithmetic transfer family reaches 40% full-OOD answers and
  30% strict all-consumer episodes;
- treatment exceeds the best matched arm by at least 5 percentage points on
  full-OOD answers and strict episodes, with a paired 95% bootstrap interval
  excluding zero;
- fit and depth accuracy are no more than 3 points below the strongest matched
  arm.

### Gate 2: causal state write/read

On generator-eligible pairs, independently for the register family and for each
transfer family used to support a claim:

- observed donor-follow is at least 70% on language, value, and depth OOD and
  at least 55% on full OOD;
- the corresponding Wilson 95% lower bounds are at least 65% and 50%;
- swap specificity is at least 40 points in every OOD partition;
- donor-follow exceeds the strongest zero, shuffle, Gaussian, reset, and
  no-interchange condition by at least 25 points;
- all-reader ablation reduces flexible full-OOD accuracy by at least 25 points;
- reader-12 and reader-18 lesions each reduce either full-OOD answer accuracy
  or donor-follow by at least 3 points, preventing an exclusively final-site
  motor explanation;
- clean-state clamping mediates at least 70% of the one-fact source-edit effect
  in source-visible audit mode.

A clean accuracy gain with failed donor following is a rejection, not partial
workspace evidence.

### Gate 3: recurrence

- Four rounds exceed one round and equal-call reset by at least 5 points on
  depth/full strict accuracy and by at least 5 points on donor-follow.
- Prompt-pair workspace distances do not collapse: median between-world distance
  must exceed median same-world/paraphrase distance by at least `2x` on every
  OOD partition.
- Round-4 attention is not identical to round-1 attention within `1e-4` mean
  absolute difference on at least 90% of sources.

Failure rejects recurrent refinement. A successful one-pass bottleneck would
require a new preregistration and cannot inherit the R11 claim.

### Gate 4: broadcast, report, and update

- Every seen register consumer reaches at least 50% full-OOD accuracy. Every
  consumer in a claimed transfer family must independently meet the same floor.
- The held-out sum/workspace pairing reaches at least 45% full-OOD accuracy and
  at least 50% donor-follow, each at least 10 points above no-interchange.
- The identical cached workspace hash is used for all consumer queries in an
  episode; recomputing a query-specific state is forbidden.
- Explicit report accuracy is at least 60% full OOD, and transplanted reports
  follow the donor at least 60% of the time.
- Post-update two-reader strict accuracy is at least 45% full OOD; transplanting
  `W_after` must follow the donor post-update answers at least 55%, while
  transplanting `W_before` must not.

### Gate 5: selectivity and ordinary behavior

In source-visible audit mode:

- all-reader ablation reduces flexible multi-step accuracy by at least 20
  points;
- the same ablation changes automatic-board accuracy by no more than 3 points;
- automatic-board mean NLL rises by no more than `0.05`;
- adapter-enabled fluent generations may not introduce workspace report syntax,
  paired-answer grammar, or state labels on ordinary prompts.

Workspace-disabled mode remains bitwise base-equivalent regardless of this
gate.

### Gate 6: untouched confirmation

The confirmation board is scored once after all used-board decisions are final.
It must independently satisfy Gates 1 through 5 with the same thresholds,
except that no fit slice exists. At least 300 generator-eligible full-OOD donor
transplants must be present. No threshold, layer, round, loss weight, slot count,
or decoding rule may change between used and confirmation boards.

## 16. Kill gates and prohibited rescues

R11 is closed as formulated if any of the following occurs:

1. Any inference path requires structured state, parsed numbers, operation
   labels, solver calls, R10 candidate sets, or host-side execution.
2. Any rationale, pause token, latent output token, or teacher CoT is needed for
   the primary score.
3. The state is decodable but donor transplants do not produce donor-specific
   behavior.
4. Clean gains are matched by no-interchange, reset, or private-query controls.
5. Recurrence fails its gate, even if four rounds have a lower training loss.
6. Only the last reader site matters, indicating a motor correction rather than
   broadcast use across decoder depth.
7. Held-out consumer recombination or the post-update gate fails.
8. Full-OOD language, value, depth, or transfer-family floors fail.
9. Automatic behavior or ordinary output mode degrades past Gate 5.
10. The exact parameter or compute budget is exceeded.
11. A used-board near miss motivates changing layers, slot width/count, rounds,
    loss weights, pair selection, or intervention strength.

The following are explicitly not rescues:

- adding more slots or rounds;
- unfreezing the base;
- searching J-lens layers or latent coordinates;
- using R10 exact algebra to interpret or execute `W`;
- lowering OOD or donor-follow thresholds;
- reporting train/IID accuracy, probe decodability, or nonzero gradients as the
  mechanism result;
- moving to a public benchmark before the internal gates pass.

Mechanics bugs discovered before any score may be corrected only with a version
bump, new hashes, repeated admissions, and regenerated score-blind boards.

## 17. Threats that remain even after a pass

1. **Synthetic-domain limitation.** The state could implement a narrow learned
   simulator for the admitted families. Public math, code, and logic transfer
   would remain unproved.
2. **Protocol dependence.** Source/query separation is an architectural
   contract. Success would not show that arbitrary chat text self-segments into
   workspace episodes.
3. **Opaque encoding.** Whole-state interchange establishes causal semantics at
   the state level, not a human-interpretable meaning for individual slots or
   coordinates.
4. **Finite capacity.** Six 96-wide slots may work only within the tested state
   complexity. No length-unbounded or context-compression result follows.
5. **Training-created interface.** R11 deliberately installs a workspace-like
   interface. It would not show that raw Shohin had one before training.
6. **No consciousness claim.** Functional workspace criteria do not imply
   phenomenal experience or any philosophical conclusion.

## 18. Execution order and live-training firewall

This document authorizes no current job. A future implementation must proceed
in this order:

1. Build only isolated R11 modules and CPU/tiny-model unit contracts.
2. Generate train, used, confirmation, transplant, and selectivity artifacts;
   run independent admissions; freeze all hashes before a model score.
3. Prove base bypass identity, causal masking, metadata deletion, exact active
   parameter matching, and measured compute budgets.
4. Run a 16-update CUDA/bf16 mechanics canary from a read-only local copy of
   `best_step200000.pt`.
5. Run the four matched 2,048-update arms in separate output directories.
6. Run dependency-held read-only used-board evaluations and one CPU assessor.
7. Stop on any failed conjunct. Only a full used-board pass permits the one
   untouched confirmation run.

The protected pretraining job, its optimizer, output path, numbered
checkpoints, data stream, and job scripts are out of scope. R11 must never be
loaded into, resumed from, or written beside the live writer. No existing
training job is to be canceled, requeued, modified, or inspected as part of
this design task.

## 19. Authorized wording if every gate passes

The strongest allowed statement is:

> In preregistered synthetic language domains, an isolated 1.61M-parameter
> adapter for the 125.08M-parameter Shohin base learned a 576-scalar private
> state that was source-written, donor-interchangeable, updateable, selectively
> necessary, and reusable by multiple neural consumers without visible
> chain-of-thought or inference-time oracle metadata.

Until those gates pass, the only accurate statement is:

> R11 is a falsifiable design for an internal trainable workspace. It has not
> been implemented or validated, and no capability claim exists.
