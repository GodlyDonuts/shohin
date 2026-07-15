# R12 Preregistration: Operation Workspace Jacobian Diagnostic

**Status:** `FROZEN / UNRUN`

**Freeze date:** 2026-07-15 America/New_York

**Execution status at freeze:** checkpoint bytes were hashed, and CPU-only
tokenizer, prompt-construction, arithmetic, intervention-math, and tiny random
model mechanics were checked. The raw-260k checkpoint was not loaded into a
model and no Shohin forward pass, backward pass, generation, score read, GPU
job, or submission occurred.

This is a new, isolated mechanistic diagnostic for the immutable raw-260k
checkpoint. It addresses the narrow consume-and-transport failure documented
in `R12_RAW260K_INTERACTION_CAPABILITY_MAP.md`: the model can often execute a
supplied local operation under `Problem/Work`, but does not reliably consume a
visible state, advance a plan, and select the next operation itself.

## 1. Claim boundary

The diagnostic asks only whether the correct label among `add`, `multiply`,
`subtract`, and `remainder` is:

1. detectable at a source residual in a frozen token-directed averaged
   future-logit Jacobian readout;
2. selected by the model's later restricted candidate logits; and
3. causally redirectable by one bounded, norm-matched activation-coordinate
   swap.

It does **not** test or establish reasoning, a general semantic operation
representation, a global workspace, autonomous recurrence, broad transport,
consciousness, or benchmark capability. An outcome called "not detected" is
operationally absent from this preregistered readout; it is not proof that no
other representation exists.

The methodological inspiration is Anthropic's 2026
[Verbalizable Representations Form a Global Workspace in Language Models](https://transformer-circuits.pub/2026/workspace/index.html).
That paper defines a layer map by averaging current-and-future residual
Jacobians across contexts and reads token directions through the unembedding.
It also swaps two J-lens coordinates with a pseudoinverse. This diagnostic uses
a cheaper token-directed derivative of future candidate-logit contrasts. It
does not compute the paper's full `d_model x d_model` `J_l`, sparse J-space, or
global-workspace battery.

## 2. Frozen custody

All values below must match before a canonical run begins.

| Object | Frozen identity |
|---|---|
| Raw checkpoint | `train/flagship_out/ckpt_0260000.pt` |
| Checkpoint step | `260000` |
| Checkpoint SHA-256 | `91d5288f184fc5230516add9851ac1a8815d3369ffd816cd7d0c03d8bafc741d` |
| Tokenizer | `artifacts/shohin-tok-32k.json` |
| Tokenizer SHA-256 | `87532df5c121753de3b29194e1f9e3de47986d3f5359548fdf93606773a233d4` |
| Expected architecture | `n_layer=30`, `d_model=576`, `n_loop=1` |
| Frozen spec SHA-256 | `5c807955be3d94cf40f04179bdf31fbefb000765885d409d05448f22d2c9b398` |
| Probe canonical freeze SHA-256 | `00b50f82a5aa15d6734c5edeee0fa9630b7bb1c8f74efa84b03324bcd557b4a0` |
| Probe file SHA-256 | `c7a8b1626604c8b7c2eeb4fc065fcc7076c84c0e98cb83064f39bbecffd5c95c` |
| Direction prompt manifest SHA-256 | `45c335658ceb5d95dd7b054f53dd2d48e5f8396bb02116a2fa7a0df7ed534f2c` |
| Evaluation prompt manifest SHA-256 | `da931fc25f3d6b9c804ab3af77de08141620e477caf5b27c9b565134d7ff437a` |
| Case payload SHA-256 | `a3628c3495d23f68e7d8c6ddea6ee96c46405776210e17462100c3a3f114e2f3` |
| Prior raw-260k capability map SHA-256 | `c8dd535cacd0fcf87c88a0d1b88b5b872dc7c22212fc047891019aace9bc8cad` |

The canonical probe source replaces only its embedded canonical-hash value
before hashing itself, then fails closed on any other source edit. The isolated
job wrapper additionally checks the ordinary byte SHA-256 of this preregistration
and the probe before Python starts. The result JSON records ordinary hashes and
byte counts for the preregistration, probe, test, wrapper, checkpoint, and
tokenizer.

No result file may be overwritten. The probe has no CLI controls for layers,
candidate tokens, prompts, cases, seed, swap strength, controls, or thresholds.

## 3. Frozen candidate tokens

The output query ends in `The next operation is`. Each candidate is exactly one
token under the frozen tokenizer, including its leading space.

| Operation | Exact token text | Token ID | Donor | Same-norm sham pair | Permuted-label readout control |
|---|---|---:|---|---|---|
| add | ` add` | `820` | multiply | subtract, remainder | subtract |
| multiply | ` multiply` | `4307` | subtract | remainder, add | remainder |
| subtract | ` subtract` | `5498` | remainder | add, multiply | add |
| remainder | ` remainder` | `7486` | add | multiply, subtract | multiply |

Every evaluation source contains every candidate token exactly once. Correct
labels therefore cannot win merely because one operation word is absent from
the source.

## 4. Frozen prompts

### 4.1 Direction contexts

For each candidate, substitute the bare operation word into all six source
templates. The source anchor is the final token of the source. The fixed future
suffix follows it.

```text
A controller records the operation {operation}.
The record has been read and retained.
```

```text
The named state update is {operation}.
The update has been placed in memory.
```

```text
For this process, the chosen operation is {operation}.
The choice is now fixed.
```

```text
A worksheet labels the pending action {operation}.
The label has been registered.
```

```text
The operator card says {operation}.
The card has been inspected.
```

```text
The transition rule selected for later use is {operation}.
The rule has been stored.
```

Fixed future suffix:

```text
When asked to report it, the operation is
```

There are `4 * 6 = 24` direction contexts. They contain no evaluation source,
state, operand sequence, case ID, score, or generated response.

### 4.2 Evaluation source and query

Each held-out source has the exact form:

```text
Problem: Start with {start}. Follow this plan in order: {four rendered operations}.
Current state: {state_after_a_nonempty_prefix}.
The current state was produced after completing a prefix of that plan.
```

The source anchor is the final period. The exact later suffix is:

```text
Which operation comes next?
The next operation is
```

The prompt does not reveal the prefix length. The model must align the visible
state with a unique state in the source plan, then transport the next operation
through the later query. Every trajectory has five unique integer states. Each
plan uses each operation exactly once.

## 5. Frozen held-out boards

`completed` is the number of operations already applied. `state` is derived by
the exact integer solver. The two boards are disjoint, have 12 cases each, and
contain exactly three correct instances of each candidate.

| ID | Board | Start | Ordered plan | Completed | State | Correct | Donor |
|---|---|---:|---|---:|---:|---|---|
| primary-add-01 | primary | 7 | multiply 3; add 8; subtract 5; remainder 11 | 1 | 21 | add | multiply |
| primary-add-02 | primary | 31 | subtract 9; multiply 2; add 7; remainder 13 | 2 | 44 | add | multiply |
| primary-add-03 | primary | 25 | remainder 7; multiply 5; subtract 3; add 12 | 3 | 17 | add | multiply |
| primary-multiply-01 | primary | 18 | add 7; multiply 4; subtract 9; remainder 13 | 1 | 25 | multiply | subtract |
| primary-multiply-02 | primary | 40 | subtract 6; add 5; multiply 3; remainder 17 | 2 | 39 | multiply | subtract |
| primary-multiply-03 | primary | 29 | remainder 8; subtract 2; add 9; multiply 6 | 3 | 12 | multiply | subtract |
| primary-subtract-01 | primary | 11 | multiply 5; subtract 14; add 6; remainder 13 | 1 | 55 | subtract | remainder |
| primary-subtract-02 | primary | 27 | add 8; multiply 3; subtract 10; remainder 19 | 2 | 105 | subtract | remainder |
| primary-subtract-03 | primary | 50 | remainder 17; add 9; multiply 4; subtract 7 | 3 | 100 | subtract | remainder |
| primary-remainder-01 | primary | 22 | add 11; remainder 8; multiply 5; subtract 3 | 1 | 33 | remainder | add |
| primary-remainder-02 | primary | 35 | multiply 2; subtract 9; remainder 11; add 6 | 2 | 61 | remainder | add |
| primary-remainder-03 | primary | 17 | add 10; multiply 4; subtract 5; remainder 9 | 3 | 103 | remainder | add |
| replication-add-01 | replication | 9 | multiply 4; add 13; remainder 11; subtract 2 | 1 | 36 | add | multiply |
| replication-add-02 | replication | 44 | remainder 13; multiply 6; add 5; subtract 8 | 2 | 30 | add | multiply |
| replication-add-03 | replication | 63 | subtract 12; remainder 10; multiply 7; add 4 | 3 | 7 | add | multiply |
| replication-multiply-01 | replication | 16 | subtract 7; multiply 5; add 8; remainder 14 | 1 | 9 | multiply | subtract |
| replication-multiply-02 | replication | 28 | remainder 9; add 6; multiply 5; subtract 3 | 2 | 7 | multiply | subtract |
| replication-multiply-03 | replication | 52 | add 9; subtract 4; remainder 12; multiply 8 | 3 | 9 | multiply | subtract |
| replication-subtract-01 | replication | 14 | add 12; subtract 7; multiply 3; remainder 10 | 1 | 26 | subtract | remainder |
| replication-subtract-02 | replication | 33 | multiply 3; remainder 14; subtract 5; add 11 | 2 | 1 | subtract | remainder |
| replication-subtract-03 | replication | 48 | remainder 13; multiply 5; add 4; subtract 9 | 3 | 49 | subtract | remainder |
| replication-remainder-01 | replication | 19 | multiply 3; remainder 10; subtract 2; add 7 | 1 | 57 | remainder | add |
| replication-remainder-02 | replication | 37 | subtract 8; add 6; remainder 9; multiply 4 | 2 | 35 | remainder | add |
| replication-remainder-03 | replication | 21 | add 5; multiply 3; subtract 4; remainder 8 | 3 | 74 | remainder | add |

Neither board is development data. Both are independently scored confirmation
boards. No model result may alter either board or the gates below.

## 6. Token-directed future-logit directions

Frozen source layers are:

```text
5, 9, 13, 17, 21, 25, 28
```

For candidate `c`, direction context `k`, source anchor `a_k`, future report
logit position `q_k > a_k`, and layer `l`, define the contrastive gradient:

```text
g[l,c,k] = grad_h(l,a_k) (
    z(q_k,c) - mean_{j != c} z(q_k,j)
)
```

Normalize each context gradient before averaging, then normalize the average:

```text
v[l,c] = normalize(mean_k(normalize(g[l,c,k])))
```

This estimates only four rows of an average future-logit Jacobian. It is
token-directed, context-averaged, contrastive, and causal to first order. It is
not the identity approximation used by a naive logit lens, and it is not the
paper's full future-residual Jacobian.

Every model parameter has `requires_grad=False`. Autograd starts by detaching
the layer-5 block output and making that residual require gradients. Gradients
are retained only for the seven frozen source residuals at one source anchor.

## 7. Frozen readout

The preregistered readout band is layers `13, 17, 21`. Layer selection after a
score read is forbidden.

At each layer, compute the four inner products `<h[l,a], v[l,c]>`, subtract
their candidate mean, and divide by their candidate RMS. Average those four
normalized score vectors across the three readout layers. The readout prediction
is the largest resulting candidate score, with candidate-order tie breaking.

The fixed permuted-label control shifts each target by two places in candidate
order. It uses the same residual and directions, so it cannot gain from a
different forward pass.

## 8. Frozen causal swap

Only layer `17` and only the source anchor are causally patched. For correct
direction `v_c`, donor direction `v_d`, and `V=[v_c v_d]`, compute:

```text
coords = pinv(V) h
delta_signal = V (swap(coords) - coords)
```

The sham performs the identical operation with the two candidate directions
unused by the correct/donor pair. Signal and sham deltas are independently
scaled to the smaller of their two raw L2 norms and `0.05 * ||h||`. They have
exactly matched requested L2 norm and never exceed 5% of the clean source
residual norm. The probe measures the realized post-cast deltas inside both
hooks: each must remain between 0.01% and 5% of its clean residual norm, and
their realized L2 norms must match within 2%. A degenerate, rounded-away, or
linearly dependent pair invalidates the run.

The causal endpoint is the change in the later candidate-logit contrast:

```text
(donor logit - correct logit)_patched
    - (donor logit - correct logit)_clean
```

Signal is compared against the same endpoint under the norm-matched sham. The
intervention never patches the query position, output position, token IDs, KV
cache, weights, or any other residual position.

## 9. Frozen gates

All gates apply independently to each 12-case board. `Binomial p` values are
exact one-sided tails. Ties fail the causal sign count.

| Gate | Per-board requirement |
|---|---|
| Jacobian-aligned readout | at least `8/12` top-1; at least `+4` over permuted-label top-1; mean correct-vs-best-other normalized margin at least `0.10`; chance `Binomial(12, 0.25)` tail at most `0.05` |
| Operational absence | at most `5/12` readout top-1; at most `+1` over permuted-label top-1; mean readout margin at most `0.0` |
| Output selected | at least `8/12` clean restricted-logit top-1 |
| Output not selected | at most `5/12` clean restricted-logit top-1 |
| Causal swap | mean signal effect at least `0.20` logit; mean signal-minus-sham at least `0.10` logit; positive signal-minus-sham on at least `10/12`; one-sided `Binomial(12, 0.5)` sign tail at most `0.05`; every patch at most 5% relative L2 |

The final outcome is ordered to give causal evidence precedence:

1. **C: bounded swap causally redirects operation.** Both boards pass the
   causal gate.
2. **B: Jacobian-aligned operation present but not selected.** C fails, both
   boards pass readout, and both boards pass output-not-selected.
3. **A: operation not detected in the frozen Jacobian-aligned readout.** C
   fails and both boards pass the strict operational-absence gate.
4. **D: mixed or inconclusive.** Every other combination, including one-board
   replication failures, readout-present-and-output-selected without causal
   evidence, near-threshold effects, or conflicting controls.

These categories do not convert a null into proof of neural nonexistence.

## 10. Frozen controls

1. **Equal lexical exposure:** all four candidate token IDs occur exactly once
   in every evaluation source.
2. **No stage label:** prompts provide a current state, not a completed-step
   count. Every trajectory is unique.
3. **Independent direction contexts:** no direction source hash overlaps an
   evaluation source hash.
4. **Independent boards:** primary and replication source hashes are disjoint,
   balanced, and separately gated.
5. **Permuted-label readout:** fixed two-place derangement, evaluated from the
   same scores.
6. **Norm-matched causal sham:** swaps the two unused operation directions at
   the same layer, position, and L2 norm.
7. **Clean baseline:** every patched endpoint is differenced from its own clean
   prompt logits.
8. **No post-hoc layer selection:** only layers 13/17/21 define readout and only
   layer 17 defines the causal claim.
9. **No generation or parsing:** all endpoints are teacher-forced next-token
   logits restricted to the four frozen one-token candidates.
10. **No score-bearing noncanonical run:** CPU or MPS execution may test
    mechanics but is labeled noncanonical. Only the frozen CUDA path may emit
    the preregistered outcome as canonical evidence.

## 11. Resource ledger

The expected canonical run has exactly:

| Resource | Frozen count or bound |
|---|---:|
| Direction contexts | 24 |
| Held-out cases | 24 |
| Model forwards | `24 + 24 clean + 24 signal + 24 sham = 96` |
| Model backwards | 24 |
| Candidate logit reads | 384 |
| Generated tokens | 0 |
| Optimizer steps | 0 |
| Trained parameters | 0 |
| Patched layers per case | 1 |
| Patched token positions per case | 1 |
| Realized patch L2 | at least 0.01% and at most 5% of clean anchor residual L2; signal/sham ratio within 2% of 1.0 |
| GPU request in isolated wrapper | one H100, maximum 2 hours, no submission command |

The result must additionally record wall time, model parameter count,
checkpoint/tokenizer/source bytes and SHA-256 values, actual forward/backward
counts, actual prompt token counts, device, peak CUDA allocation where
available, and process maximum RSS. Full model logits are computed by the model,
but only the four preregistered token logits are read as endpoints.

## 12. Run validity and stopping rules

A run is invalid and produces no admissible verdict if any of the following
occurs:

- any checkpoint, tokenizer, source, canonical-source, spec, prompt, or case
  hash differs;
- checkpoint step or architecture differs;
- any candidate is not exactly one token in both isolation and query context;
- a prompt crosses its source/suffix token boundary;
- lexical candidate counts, trajectory uniqueness, split balance, or split
  disjointness fail;
- `n_loop != 1`, a requested layer is unavailable, a gradient is zero/nonfinite,
  swap directions are dependent, or signal/sham cannot be norm matched;
- any realized patch falls below 0.01%, exceeds 5%, or misses the 2% norm-match tolerance;
- deterministic CUDA prerequisites are absent;
- any output path already exists;
- any prompt, token, layer, control, bound, threshold, or outcome rule changes
  after a Shohin model score has been observed.

OOM, timeout, deterministic-kernel failure, or infrastructure failure is
`INVALID/NO RESULT`, never outcome A. No retry may change the scientific
contract; an exact rerun may change only machine, scheduler metadata, or fresh
output filename.

## 13. Forbidden interpretations and follow-ons

No result from this probe authorizes training, checkpoint promotion, production
invocation, a new GPU submission, a global-workspace claim, or a reasoning
claim. It does not adjudicate whether operation words are compositional beyond
these prompts. A future experiment may be proposed only after this frozen
result is read, and must use new files and a new preregistration rather than
editing this contract.

The wrapper in `train/jobs/probe_operation_workspace_jacobian.sbatch` is an
isolated reproducibility artifact. Creating it is not authorization to submit
it. No job was submitted as part of this freeze.
