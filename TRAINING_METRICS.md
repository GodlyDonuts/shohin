# Shohin Training Metrics Ledger

This is the auditable metrics companion to [AGENT_RUNBOOK.md](AGENT_RUNBOOK.md).
It records confirmed measurements, their source artifacts, and the distinction between
training progress, corpus capacity, and capability. It is not a substitute for the
runbook's operational instructions.

**Last refreshed:** 2026-07-15 02:55 EDT
**Flagship source of truth:** Newton Slurm job `686732`,
`/lustre/fs1/home/sa305415/shohin/logs/flagship2_686732.out`
**Checkpoint source of truth:** capture the numbered checkpoint at its milestone, promote
`best_step<step>.pt`, and verify the local full checkpoint. The trainer may subsequently reap
the numbered file under its retention policy; the ledger records which copies remain durable.

## Definitions

| Metric | Meaning | Do not interpret it as |
|---|---|---|
| **Step** | One optimizer update. | One input document or one unique data pass. |
| **Nominal update tokens** | `step * global_tokens_per_update`; global update size is fixed at 524,288 tokens. | Unique source tokens learned without repetition. Restarts before the forward-stream fix could revisit earlier shard prefixes. |
| **Active corpus capacity** | Token count in the manifests mounted by the current flagship's `SHARDS` list. | The number of tokens already presented to the model, nor a claim that every row is equally sampled. |
| **Checkpoint milestone** | A numbered full checkpoint at an exact step. | A benchmark or capability improvement. |
| **Admitted data** | A source with a reviewed manifest and all required quality/decontamination gates. | A downloaded, probed, or partially generated source. |
| **Capability evidence** | A frozen benchmark or held-out, independently audited transfer result. | Training loss, an in-distribution generator score, or best-of-N samples. |

## Flagship Pretraining

| Field | Confirmed value |
|---|---|
| Model | 125.1M trained parameters; frozen 32k tokenizer; 2,048-token sequence length |
| Active job / node | `686732` on `evc34`, two H100s, 4 CPU cores; one protected writer with no capability-experiment output sharing. |
| Start / scheduled end | 2026-07-14 03:52:10 / 2026-07-17 03:52:10 EDT (Slurm allocation; not a completion guarantee) |
| Microbatch / accumulation | `world=2`, `BS=32`, `ACC=4` |
| Global tokens per update | `2 * 32 * 4 * 2,048 = 524,288` |
| Absolute training target | 300,000 steps |
| Resume point | `ckpt_0217250.pt` to step 217,251 with fresh optimizer rewarmup and stream generation 1 |
| Latest checkpoint milestone | **260,000** steps = **136,314,880,000 nominal update tokens**. Newton promoted `best_step260000.pt` and local `train/flagship_out/ckpt_0260000.pt` match at MD5 `301082250e15c26820790ec7ff7730a0`; the local file is complete at 1,076,597,546 bytes. The rolling numbered Newton file aged out after promotion. Before resuming from the promoted artifact, the complete 1,035,468,800-byte local prefix was independently verified against the remote prefix. |
| Last observed live step | At least 262,060 = 137,394,913,280 nominal update tokens |
| Last observed throughput | 283,710 tokens/s, approximately 24.512B nominal tokens/day at that sustained rate |
| Latest loss / gradient norm | step 262,060: loss 1.8301; gnorm 0.10; LR 0.0033 |
| Direct H100 telemetry | No intrusive telemetry task was added at this milestone. The established two-H100 configuration remains `BS32/ACC4`; current sustained throughput is about 1.85x the prior one-H100 154.3k tok/s band. |
| Post-handoff health | Startup guard events at 217,569--217,573 and 217,643 recovered into the normal band; the later isolated 234,419 event recovered at 234,420. Isolated gnorm skips at 258,239 and 261,479 recovered on the immediately following updates. Logged updates through 262,060 are finite with normal gradient norms and no persistent skip, loader, CUDA, NCCL, or DDP error. |
| Two-H100 handoff validation | `686734` first established world-2 transport; live `686732` then resumed the exact writer at step 217,251 and has sustained roughly 285--287k tok/s after rewarmup. This is now production throughput, not a canary extrapolation. |

The current live flagship's data stream is frozen for the life of `686732`. Do not add
new shards, alter weights, or apply an experimental runtime optimization. The job has
already logged `world=2`, passed real CUDA/NCCL execution, and preserves the exact
524,288-token global update with 250-step checkpoints.

### Raw-Checkpoint Direct Interaction: 200k vs 252.5k vs 260k

The same seven fresh transcript-first prompts were run greedily on the verified local
`ckpt_0200000.pt`, `ckpt_0252500.pt`, and `ckpt_0260000.pt` checkpoints. This is a
qualitative capability audit, not a public benchmark.

| Checkpoint | Initial | Independent review | Supplied verified fact | Valid compact-state reuse |
|---|---:|---:|---:|---:|
| Raw 200k | 1/7 | 0/7 | 1/7 | 0/7 |
| Raw 252.5k | 1/7 | 0/7 | 1/7 | 0/7 after transcript audit |
| Raw 260k | 1/7 | 0/7 | 1/7 | 0/7 |

The automated artifact originally marked 252.5k reuse as 1/7 because the old yes/no
scorer found a repeated `No mar...` premise inside malformed state text. It did not emit
or reuse a valid compact state, so the human-forensic result is 0/7. Arithmetic, base
conversion, sequential state, sorting, string insertion, and Python all remain wrong.
The 252.5k model more often emits locally relevant fragments, such as the correct
`29 * 16 = 464`, but does not perform the required next operation. This is weak evidence
of better local completion, not multi-step reasoning. Canonical local artifact:
`artifacts/eval_history/manual_raw_200k_vs_252500_local_mps.json`, SHA-256
`169564bde33eeb21a0f224147d38b7cc972cb8215e598627774e43aea111eed3`.

The 260k probe still reports `29*16=496`, treats base-6 `425` as decimal
`425`, omits the multiplication in the sequential update, copies the unsorted
input, fails insertion, and emits verbose code/template continuations. Only the
simple syllogism passes. This is no capability movement from 200k or 252.5k.
Artifact: `artifacts/eval_history/manual_capability_raw260k_20260715_mps.json`,
SHA-256 `42590202834294cea182821f09613503c5ca91f6a1676d020d9f2cc2100c0aac`.

## Current Reasoning-Mechanism Frontier

These are experiment-state measurements, not capability scores.

| Mechanism | Verified state | Capability status |
|---|---|---|
| R10 ACAW/VSPT | Exact noncommutative composition, rank-six exact-rational ambiguity, monotone replay, fail-closed overflow, fixed-size commitments, and canonical serialized-store accounting pass local mechanics. The replacement finite-board contract uses 800 calibration rows and 1,840 factorial confirmation rows, exact operation/query/depth cells, at least 10 accepted cases per cell, at least 400 per confirmation partition, and zero false certificates. Local checks passed, but a second custody audit found seven claim-blocking failures: self-attesting manifests, job-only clean-code enforcement, selectable seeds/R5 input, rehashable score substitution, hash/read TOCTOU, incomplete batch/device/determinism identity, and source changes between admission identities. | **Dormant control; no score read; second audit NO-GO.** Preserve the mechanics and hardening as comparator evidence. Do not resume its score chain unless an R12 contract explicitly requires it. |
| R11a causal mediator | V3 closes the v2 source/query sampling, cached-generation, common-evaluation, and confirmation-derivation contract defects on paper. Its six source-derived slots, tied recurrent writer, and query readers remain established recurrence/memory machinery rather than a new primitive. | **Dormant favorable control; no implementation, board, fit, score, or GPU job.** |
| R12 mathematical invention frontier | Exact extensional states are conjugate to the residual transducer. Fork-core compression, coherent profile extension, arbitrary late-query compression, secret sharing, finite relation loss, unrestricted MDL, hidden-coordinate locality, passive general-matroid learning, uncoded noise stability, commutator/holonomy state recovery, private-bank causal-address revelation, query-kernel factorization, active verifier queries, and dynamic-frontier compression each have an explicit theorem/collapse record. The latest CAR audit is decisive: centralized preprocessing can store a composite table or segment tree, so its private-bank communication gap is not a neural reasoning separation. | **Theory-only; current candidates rejected or retained only as controls.** No code, data, fit, score, CPU board, or GPU job is authorized. The next candidate must prove a uniform advantage over the strongest structure-aware centralized comparator, not raw history or an artificially one-round baseline. |

The research decision rule is stricter: infrastructure, training loss, local mechanics, and a
decodable hidden state are not reasoning. R10 and R11 are dormant controls. R12 requires a uniform
resource-scaled capability, future-equivalent state invariance, future-distinguishable state
separation, causal necessity, and a comparator-relative theorem before implementation.

## Overnight Comparison Snapshot: 2026-07-13 11:36 EDT

This is the explicit before-sleep reference point for the next custody check.

| Surface | Verified state |
|---|---|
| Flagship | `685084` is `RUNNING` on `evc22`, 2d 07h elapsed, one H100, `BS=32 ACC=8`, 4 CPUs. |
| Training progress | Step **200,560**; **105,151,201,280** nominal update tokens; **154,293 tok/s**. Recent loss/gnorm remains finite and in band. |
| Stability | The step-200,387 gnorm outlier recovered at the next logged step. No divergence, data-loader, CUDA, or checkpoint error observed. |
| Durable recovery | Hash-matched 200k full checkpoint on Newton/local: md5 `510d57df578447986b40e20029511b9d`. Next mandatory promotion is 210k. |
| Frontier gate | LSA and CPR are rejected. CPR's verified normal packet accuracy is identical to shuffled-source at **161/10,752 = 1.497%**; all five preregistered comparator gates are false. No source-free continuous-packet claim survives. |
| Corpus expansion | DCLM `686529` completed 25,000,001,792 tokens / 250 shards but is unadmitted pending a fresh scan. OpenMath PT completed 5,000,000,144 tokens / 50 shards and is likewise future-handoff-only. Stokes FineWeb r2 `738030` is live through shard 53, roughly 5.4B tokens. None is in the active stream. |
| Next protected transition | `686732` is dependency-held after the flagship: two H100s, `BS=32 ACC=4`, same 524,288-token update. It must not affect the live writer. |

### Post-Snapshot Research Update: 2026-07-13 06:32 EDT

The locked LSA comparator `687172` **rejected** the verified-geometry candidate. The candidate's
fit-IID margin over the strongest control was **+1.04pp** against a required +10pp; combined
length/language OOD was **+0.09pp** against +5pp; equivalent-pair margin was **+0.52pp** against
+10pp; intervention pairs were **0/576** for every arm. It won only two chunk counts, not the
required three. This is not retained-state evidence and does not justify LSA stage 2.

The source-free causal-prefix-readback replacement is now the active isolated experiment. The first
submission (`687216`-`687218`) correctly refused the CPR-specific audit before model loading because
the trainer requires the hash-bound generic LSA admission audit; its never-satisfiable evaluators were
canceled. The corrected arms start from the immutable 190k raw checkpoint and use that generic audit
plus the separately preserved CPR protocol audit: `687223` verified readbacks, `687224` shuffled
complete readback labels, and `687225` equal-work replicated-final readbacks. All three reached finite
step-80 losses after warmup on the identical 32,000-pair / 7,163-update surface. Read-only held-out
successors `687226`-`687228` are `afterok`-held and use separate outputs. None shares the flagship
checkpoint writer, its data stream, or its output tree.

### CPR Training Milestone: 2026-07-13 09:34 EDT

All matched CPR arms completed cleanly from immutable `best_step190000.pt`: verified `687223` in
2h50m, shuffled-label `687224` in 2h52m, and equal-work final replay `687225` in 2h38m. Their
checkpoints are separate and hash-recorded: verified `7d84282c3daaa4a238db821ff8c69ed3`, shuffled
`162282f3db4d7f6ce064b252be5bc35d`, replay `6c4f11caeae9701d7fd09fef957833a3`. Full held-out
source-free readback evaluations `687226`-`687228` are running; no outcome is inferred from their
partial normal-mode counts.

### CPR Decision: 2026-07-13 11:36 EDT

All three held-out evaluator reports and the locked four-control comparator are complete. CPR is
**rejected**. The verified packet model's normal source-free readback is **161/10,752 = 1.497%**,
exactly equal to its shuffled-source control; the equal-work replay is higher at **193/10,752 =
1.795%**. The fit-IID margin is **-1.273pp**, length OOD **-0.493pp**, language OOD **-0.347pp**,
and full OOD **-0.439pp** against the strongest control. Every preregistered gate is false. Training
losses were likewise nearly indistinguishable across verified and shuffled-label arms, so this is a
failure to establish a decoder-readable, label-dependent packet channel, not merely an OOD miss.
Reports are preserved locally under `artifacts/evals/causal_prefix_readback_*_190k.json`; no CPR stage
2 or flagship integration is authorized.

### DRS and Direct-Interaction Update: 2026-07-13 12:04 EDT

The fixed seven-case transcript-first probe was run directly against the verified local full
`ckpt_0200000.pt`, with greedy 32-token decoding. It is **1/7 initial, 0/7 self-review,
1/7 verified intermediate fact, and 0/7 compact-state reuse**, the same directional result as the
190k probe. The only correct initial/fact case is the simple syllogism. This is interaction-level
evidence that ordinary reasoning has not visibly improved over the last 10k steps; it is not a
public benchmark result. Artifact md5: `eb3e06aa2039ceb77adf17dbc3301fd3`.

The first Digitwise Recurrent Scratchpad CPU build `738117` wrote **439,865** immutable train rows
and **1,500** held-out paired counterfactual episodes. Its independent read-only audit `738120`
recomputed every row/episode and found zero invalid rows, duplicate normalized prompts, or exact
prompt hits, but **27 13-gram hits**. Inspection showed a genuine split leak: train and held-out
episodes could reuse the same `(width,left,right)` operand tape under different operations, leaving
the operation token outside a 13-gram window. The candidate is rejected before GPU use. Its data
SHA-256 is `de6e4f798357484fb8496c396ecd930effb9b969e7dd9a16861ac5ced121102a` and held-out SHA-256
is `b831e43d87a7594464d3721212cbcb049bdfe7e5b7657680e0147b1020c3a72f`; those rejected artifacts
remain immutable. The corrected split reserves operand tapes across operations and counterfactuals,
and local 1,000-episode smoke audit is 0 invalid, 0 duplicate, 0 exact, and 0 13-gram overlap.
Stokes `738122` constructs a separately named v2 candidate and `738123` independently audits it.

### DRS v2 Admission and Matched Causal Chain: 2026-07-13 12:16 EDT

The fresh v2 candidate passed the required independent Stokes audit before any GPU job was submitted:
**439,865** train rows, **1,500** held-out paired counterfactual episodes, five 300-episode regimes,
and **19,800** held-out controller prompts. The audit found **0** invalid rows, duplicate normalized
prompts, exact held-out prompt hits, or 13-gram held-out overlaps. V2 train/eval SHA-256 are
`381b8bbf3a4eddb7b08b0f9d4b08ea3ce65e1f0ec48de930632d54417c2f7f35` and
`89ce11b36ff2f56e83cda72a1f07b1a90f4a3dc3803c69db2779a27219712646`.

The authorized DRS GPU evidence path now separates execution from wording transfer:
`687348 -> 687362 -> 687363 -> {687364, 687365}`. It runs raw `best_step200000.pt` with held-out
wording, raw `best_step200000.pt` with core wording on the identical episodes, one DRS-only SFT epoch
from exactly that checkpoint, then matched post-SFT core and held-out evaluations. The former children
`687350`/`687351` were canceled before allocation or output because they lacked the raw-core control.
The jobs use a separate output tree, exclude evc22, and cannot modify the live corpus or flagship
writer. We will report first-transition, closed-loop state, final-answer, paired-counterfactual, and
response-diversity results by regime; broad reasoning is not inferred from any DRS score.

### Append-only Delta Ledger Pre-Admission Smoke: 2026-07-13

ADL is a separate, untrained candidate for reducing recurrent output burden:
the model emits a short digit/carry delta per local step, then compacts exactly
four model-authored deltas into a retained block. The transport-only controller
never computes, repairs, or chooses state content. Its 40-episode smoke wrote
**640** train rows and **20** paired held-out episodes across five regimes. The
independent audit passed with **0** malformed rows, duplicate normalized
prompts, exact prompt hits, or 13-gram overlap. An initial 147-hit n-gram audit
failure was corrected before admission by binding retained prompt records to a
base-derived immutable identifier; the output grammar remains short and no
model-produced arithmetic is added by the controller. This is data/protocol
evidence only. No ADL GPU job is authorized before DRS identifies whether
whole-state copying is the actual failure locus.

### ADL CPU Admission Launch: 2026-07-13 12:49 EDT

The full, separately named ADL corpus is now being generated on Stokes CPU job
**`738186`**, after `py_compile`, controller tests, and generator/audit smoke all
passed on that host. Its dependent independent audit is **`738187`**. No result
from these jobs is training data until the audit records full train/held-out
counts, artifact SHA-256s, recomputed transition validity, duplicate prompts,
exact overlaps, and 13-gram overlap. The CPU jobs neither read nor write the
flagship output and do not authorize a GPU SFT.

### ADL Full Admission: 2026-07-13 13:04 EDT

Stokes `738186` completed **384,000** immutable train rows and **1,000**
paired held-out episodes, evenly distributed across five 200-episode regimes.
Data/held-out SHA-256 are
`ef317dd5aed85fa83add40a637c52232f4b4daf626e609f88926cb358113cbec` and
`3117ec5072134a9bade424499be9ee3a3e504e4f26deec445c3b5b1baeccaca0`.
Independent audit `738187` passed: 0 invalid rows/episodes, duplicate prompts,
exact prompt hits, or 13-gram overlaps across **42,000** held-out controller
prompts. Its report SHA-256 is
`5d0e2acd2cfc042de7c76266d048987c347d1e5d22b05e79232fce8ea5c9258f`.
This admits the data/protocol only; GPU training remains intentionally gated on
the active DRS core-versus-heldout diagnosis.

### DRS Exact-Prompt Repair: 2026-07-13 13:04 EDT

Pending DRS SFT `687363` and held evaluations `687364/687365` were canceled
before allocation or artifacts because their Slurm-snapshotted script would
have added a second `Question/Answer` wrapper around every already-complete
protocol prompt. Replacement `687375 -> {687376,687377}` uses the same raw
200k checkpoint, data, and dependencies, but the SFT job now validates the
stored prompt boundary and uses `--prompt-override-field completion_prompt`.
This prevents an otherwise confounded execution experiment; it is not a model
result.

### Raw DRS and ADL Primitive Diagnostics: 2026-07-13 13:05 EDT

Raw 200k DRS held-out `687348` completed all 500 episodes with **0** first
transitions, exact state loops, final answers, paired counterfactuals, and
paired interventions in every regime. It emitted 434 unique first responses
with a mode count of 67, mostly malformed Markdown or copied prompt fragments;
this is a true untrained baseline rather than a constant-answer artifact.

To test whether ADL merely exposes an already-known local primitive, a separate
non-generative likelihood probe ranked all 20 grammar-valid first digit/carry
records for eight fixed tapes under both core and held-out wording. The correct
record is top-1 on **0/16**, with mean rank **10.688/20**; `d=0;c=0` wins every
prompt. Artifact md5: `9ae4c88aca13079fe69036a47b88e597`. The result rejects
pre-existing raw microstep competence, while keeping ADL viable as an isolated
supervised learnability and compaction test.

### Direct Candidate-Likelihood Diagnosis: 2026-07-13 13:03 EDT

The non-benchmark forced-choice probe scores fixed candidate completions after
the exact same plain `Question/Answer` prompt, separating answer recognition
from free decoding. Raw `ckpt_0200000.pt` ranks the correct candidate first on
only **1/7** fresh cases, with a mean correct rank of **2.571**: arithmetic
3/4, base conversion 4/4, state update 3/4, linear equation 2/4, sort/dedup
1/4, string insertion 3/4, logic 2/2. This rules out the specific hypothesis
that the weak greedy transcript is only an emission-format failure. It is not
a claim about every possible prompt contract or general reasoning. Artifact:
`artifacts/eval_history/forced_choice_raw200k_20260713_mps.json`, md5
`7b6bcdd58f6420703fcb0b6bbbfa3afd`.

### DRS Core-Wording Completion: 2026-07-13 13:42 EDT

The raw canonical-interface control `687362` completed cleanly on an isolated
H100 (42m12s, exit 0) against the same 500 paired DRS episodes as raw
held-out-wrapped `687348`. It records **0/500** first transitions, exact
closed-loop states, final answers, counterfactual finals, and paired
interventions in every 100-episode `fit_w4`, `fit_w6`, `value_ood_w4`,
`value_ood_w6`, and `width_ood_w8` regime. It attempted exactly one transition
per normal branch before failing, so there is no unreported partial-loop gain.
The result has 34 unique first responses with a mode count of 265, not a
constant-answer collapse. The artifact is
`artifacts/evals/digitwise_recurrent_v2_raw200k_core_p100.json`, MD5
`20a5d4cc4a776ee3ffb9220f288f4f6a`.

Together with `687348`'s zero held-out-wrapped score, this rejects the
hypothesis that the raw model contains an executable DRS primitive behind a
lexical interface. It does **not** reject supervised learnability. The
isolated, exact-prompt-bound one-epoch SFT `687375` started only after this
clean control, from `best_step200000.pt`, and its core/held-out children are
still dependency-held. The result cannot alter the flagship pretrain.

### DRS CUDA Recovery and Causal-Workspace Refinement: 2026-07-13 14:18 EDT

The first hash-bound, exact-prompt DRS SFT allocation `687419` on evc44 passed
both data and completion-boundary preflights but then failed before model load
at `torch.empty(..., device='cuda')` with CUDA-capable devices busy or
unavailable. It produced no checkpoint, batches, loss, or capability result.
This is an infrastructure non-result, not a negative DRS result. The isolated
DRS SFT/evaluation exclusions now include evc44; they are not a flagship-wide
node policy.

Before reusing H100 time, `687428` requested a real H100 allocation on idle
evc49 and passed a CUDA tensor plus bfloat16 matmul in 28 seconds. Fresh,
non-overlapping paths are now chained as `687430 -> {687431,687432}` on that
verified node: one exact-boundary SFT epoch from `best_step200000.pt`, then
parallel matched core and held-out DRS evaluations. Only those child results can
answer whether supervised local execution transfers across wording.

The follow-on Counterfactual Bisimulation Compiler hypothesis now has a stricter
falsification condition: model-authored states must be interchangeable between
unseen paraphrases of the same world, must change downstream answers in the
predicted direction when swapped with a one-fact counterfactual world, and must
beat zeroed/shuffled/mismatched-state controls by a recorded state-necessity
margin. This remains a staged research specification, not a model result or an
authorized flagship modification.

### DRS v2 Isolated SFT Completion: 2026-07-13 14:46 EDT

The uncompiled replacement SFT **`687459`** completed cleanly on isolated H100
`evc49`, from immutable `best_step200000.pt`, with no access to the flagship
output tree. It consumed the hash-bound DRS v2 training data (SHA-256
`381b8bbf3a4eddb7b08b0f9d4b08ea3ce65e1f0ec48de930632d54417c2f7f35`):
**439,865** rows, **51,131,402** packed tokens, and **10,623,342** masked
answer tokens (21% of the packed surface) in **24,966** 2,048-token sequences.
One epoch was exactly **1,561** optimizer updates in **1,115 seconds**. Loss
fell from `0.6846` at step zero to a near-final logged `0.0115`; this is
training-fit evidence only. The isolated artifact is
`train/sft_digitwise_recurrent_v2_200k_r3/sft_ep1.pt`, MD5
`6f30db16208d274229950b17662dda01`.

The causal decision chain is serialized, not inferred from this loss:
**`687460`** runs the 500-episode source-free *core* evaluation,
**`687461`** runs the same counterfactual episodes under held-out wording only
after a clean core result, **`687462`** runs fresh direct raw-versus-SFT
interaction, and **`687463`** records the independent raw NLL monitor. The
SFT cannot alter active pretraining or its data writer. No DRS capability or
reasoning claim is authorized until the evaluator outputs and direct transcript
are inspected.

### DCRD Generator/Auditor Preflight: 2026-07-13 15:02 EDT

This is dataset infrastructure, not a training result. The conditional
Dual-Code Reversible Deliberation branch now has a separate deterministic
generator and independent semantic auditor. Its 1,000-episode local preflight
produced **21,000** train rows and **200** held-out paired counterfactual
episodes. The auditor recomputed every transition/readout and found **0**
invalid train rows, **0** invalid held-out episodes, **0** normalized duplicate
prompts, **0** exact held-out prompt hits, and **0** literal 13-gram hits.
Train and held-out use both disjoint codebook aliases and incompatible prompt
interfaces; A/B also use distinct serialization grammars. This result was
achieved by removing the shared template rather than waiving the overlap gate.
No durable corpus, SFT checkpoint, controller,
or GPU job exists for DCRD; submission remains conditional on the full DRS
causal decision chain.

### CBC Generator/Auditor Preflight: 2026-07-13 15:24 EDT

CBC is a prepared, source-free context-compiler experiment, not a training
result.  Its medium local preflight generated **1,000** train episodes,
**16,000** rows, and **120** held-out paired-counterfactual episodes.  The
independent audit recomputed each compiler target, update, inverse delta,
readout, shared normal/counterfactual operation sequence, and one-fact
counterfactual relation.  It reported **0** invalid train rows, **0** invalid
held-out episodes, **0** normalized duplicate prompts, **0** exact prompt
overlaps, and **0** literal 13-gram overlaps.  The corpus has not been
materialized as a durable artifact and no CBC SFT/GPU job is authorized until
the DRS core, held-out wording, and direct-interaction gates complete.
The companion transport-only controller test passes source-free rollout,
inverse-delta, same-world interchange, and counterfactual mismatch checks;
an incorrect first model state halts rather than being repaired.

### DRS v2 Position Coverage Audit: 2026-07-13 15:38 EDT

`pipeline/audit_digitwise_position_coverage.py` ran read-only against the
immutable v2 corpus.  It found four missing train marginal cells: digits
**3–9** never occur in the most-significant `a` or `b` position at width 4 or
width 6.  Consequently each value-OOD regime contains **1,200** unseen
digit-position events and **600** unseen exact local transition contexts
across its 300 paired held-out episodes; width-8 contains **9,600** and
**4,800**, respectively.  The fit regimes have zero unseen digit-position and
zero unseen local-context events.  These counts define the defect a later
position-balanced DRS curriculum must repair; they are not a model score.

### DRS v3 Minimal Transition-Basis Preflight: 2026-07-13 15:45 EDT

The staged v3 candidate is a full-episode local-context basis, not a magnitude
band. Its independent medium preflight uses **6,800** complete episodes and
**77,946** rows with two tape variants. It covers all **3,400** independently
enumerated reachable width-4/6 local decimal contexts and reports 0 malformed
rows/episodes, normalized duplicate prompts after deduplication, exact split
hits, or 13-gram split hits. Its held-out set has 40 episodes each for
`recombine_w4`, `recombine_w6`, and `width_ood_w8`. Removing all training
instances of one still-semantic local context makes the admission audit fail.
No durable v3 artifact or training job has been created. The isolated launch
contract is static-tested: it will hash-bind the corpus and held-out set to the
v3 audit, require all 3,400 contexts and all three held-out regimes, reject
any structural or contamination counter, and prove the exact inference/SFT
prompt boundary before CUDA. This is reproducibility infrastructure, not a
training or capability result.

### STRR Factorized-Register Preflight: 2026-07-13 16:04 EDT

The Static-Tape Recurrent Register control holds immutable operand evidence in
a fixed `dwt:` prompt field and asks the model to emit only the evolving
`dwr:` register. Its medium CPU preflight has **6,800** complete episodes,
**77,946** deduplicated rows, **3,400 / 3,400** independently required and
covered local contexts, and **120** paired held-out counterfactual episodes
(40 each of `recombine_w4`, `recombine_w6`, and `width_ood_w8`). The
independent admission audit reports 0 invalid rows/episodes, normalized
duplicates, counterfactual mismatches, missing contexts, exact split hits, and
13-gram split hits. It is not a model score and has no durable data, SFT, or
GPU job. The static-tested evaluator forwards model-emitted registers only;
its staged SFT wrapper is audit-hash-bound and has not been submitted. Future
factor evaluations retain capped successful and failed transcripts per regime
so aggregate accuracy cannot hide a parse, transition, or transport failure.

### DRS v3 Complete-Transition-Basis Artifact: 2026-07-13 16:20 EDT

The durable CPU-only v3 corpus has **27,200** solver-derived episodes and
**311,127** deduplicated SFT rows (eight tape variants for each of the 3,400
reachable local decimal contexts). It reserves **900** paired held-out episodes:
300 each in `recombine_w4`, `recombine_w6`, and `width_ood_w8`. The independent
admission audit is clean: 0 invalid train rows, invalid held-out episodes,
normalized duplicate prompts, missing contexts, exact split hits, or 13-gram
split hits. Data SHA-256 is
`b785866bf24813272d346e4a3bb717d4156b01a59a4dd8ccaf450733267368f6`; held-out
SHA-256 is `f2fcfcae41b55aa82dd360036bd8c9c00ed6e4ca442debec1c85ed282e50dfe1`.
It has no SFT checkpoint, score, or GPU submission. Its purpose is a causal
coverage control for DRS v2, not a claim of reasoning.

### DRS v2 Core Closed-Loop Result: 2026-07-13 16:25 EDT

From the isolated DRS SFT checkpoint, canonical core wording yields **275 / 500
(55.0%)** final answers. By regime: fit width 4 **100 / 100**, fit width 6
**98 / 100**, unseen-value width 4 **34 / 100**, unseen-value width 6
**43 / 100**, and unseen width 8 **0 / 100**. The paired counterfactual
correct-and-different totals are 100, 97, 32, 40, and 0 respectively.

This is explicitly not a binary learned/unlearned outcome. First emitted
microstates are correct on **497 / 500** episodes (100, 100, 100, 99, 98 by
regime), while later state transport fails. For example width 8 preserves 353
correct transition responses across 453 attempted before failure but never
reaches a correct final. This is the evidence for prioritizing the static-tape
register transport control before interpreting a full-basis v3 result.

### STRR Complete Artifact: 2026-07-13 16:23 EDT

The full factorized corpus mirrors the v3 basis scale: **27,200** episodes,
**311,127** rows, **3,400 / 3,400** required/covered contexts, and **900**
paired held-out episodes. Its admission audit has zero invalid rows/episodes,
normalized duplicates, counterfactual mismatches, missing contexts, exact
train/held-out hits, or 13-gram overlap. Train SHA-256:
`82245615f0849c3270f99f2db85c604ff46cb2c3dfb14f0ab3660dff3eb0d3ec`;
held-out SHA-256:
`a699ac58ad8184f4dc23dcfa317cd6e7b8f7d4ef453dcbf1ae21201901e0948a`.
This is data admission, not a score or SFT result.

## Checkpoint and Disaster-Recovery Inventory

| Milestone | Numbered checkpoint at milestone | Newton durable copy | Local full checkpoint | MD5 | State |
|---|---|---|---|---|---|
| 170k | `ckpt_0170000.pt` | `best_step170000.pt` | `train/flagship_out/ckpt_0170000.pt` | `7ad139b6b9b537a5a3e65978f8296419` | Verified Newton + local |
| 180k | Observed and hashed, then reaped by trainer retention | `best_step180000.pt` | `train/flagship_out/ckpt_0180000.pt` | `a592a8bd46163eb1427fe64460be0c6a` | Two durable verified copies |
| 190k | `ckpt_0190000.pt` | `best_step190000.pt` | `train/flagship_out/ckpt_0190000.pt` | `3e195aaf44a14259797c49d7f80d9c7f` | Verified Newton + local |
| 200k | `ckpt_0200000.pt` | `best_step200000.pt` | `train/flagship_out/ckpt_0200000.pt` | `510d57df578447986b40e20029511b9d` | Verified Newton + local |
| 252.5k | `ckpt_0252500.pt` | `best_step252500.pt` | `train/flagship_out/ckpt_0252500.pt` | `1769bb0a8a06d4565df001f0521db99e` | Post-250k recovery point; verified Newton + local |
| 260k | Numbered file subsequently reaped | `best_step260000.pt` | `train/flagship_out/ckpt_0260000.pt` | `301082250e15c26820790ec7ff7730a0` | Verified promoted Newton + local full checkpoint |

All rows above are full optimizer checkpoints, not model-only exports. The next local DR
target is 270k, or the newest clean checkpoint before any natural handoff.

## Current Active Pretraining Corpus

These are the exact current `SHARDS` inputs for `686732`, taken from their manifests.
They are decontaminated against the project evaluation n-gram set at shard construction.

| Source | Tokens | Shards | Documents seen | Documents kept | Eval-contamination drops |
|---|---:|---:|---:|---:|---:|
| FineMath-4+ | 2,000,001,108 | 10 | 1,265,604 | 1,258,975 | 2,445 |
| OpenWebMath | 14,063,689,153 | 71 | 6,315,233 | 6,224,492 | 5,080 |
| CodeParrot-Clean Python | 16,762,327,600 | 84 | 5,361,373 | 5,358,977 | 2,396 |
| FineMath-3+ | 25,000,004,410 | 125 | 13,478,404 | 13,407,172 | 8,575 |
| **Active total** | **57,826,022,271** | **290** | **26,420,614** | **26,249,616** | **18,496** |

`openmath_pt` is intentionally **not** in the live job: it is a future-handoff-only
manifest with 5,000,000,144 tokens in 50 shards (12,662,236 kept of 12,828,009 seen;
165,773 evaluation-contaminated rows dropped). Its existence is not permission to change
the running `SHARDS` list.

### Equal-Domain Exposure Accounting: 2026-07-13 14:18 EDT

At live step **203,240**, the fixed 524,288-token update implies
**106,556,293,120 nominal update tokens**, or **851.77 nominal tokens per
125.1M trainable parameters**. This is not a unique-token claim: it counts
replay and does not reconstruct the historical loader cursor. It is nevertheless
useful because `ShardLoader` is confirmed to round-robin equally over the four
mounted directories when no explicit weights are passed.

Under that equal-domain policy, each directory receives about
**26,639,073,280 nominal tokens** by this point. Relative to its manifest, the
corresponding expected capacity-equivalents are FineMath-4 **13.32x**,
OpenWebMath **1.894x**, CodeParrot Python **1.589x**, and FineMath-3 **1.066x**.
Across all sources this is **1.843x** total mounted-corpus capacity. The figures
are an exposure-risk diagnostic, not proof that any source is memorized or that
the model is overtrained. They do make two gates non-optional before a long
continuation of the same mix: compare fixed held-out English/code NLL at 200k
against the 170k baseline, and finish/admit the planned language sources before
the next natural data-mix handoff. The healthy writer remains untouched.

## Reasoning and Code Data Gates

| Asset / job | Latest measured state | Admission status |
|---|---|---|
| Frozen V8 SFT candidate | 699,928 valid rows: math 292,944, procedural 374,659, code 7,250, teacher 25,075. SHA-256 `da94f9f6aae1d69a12633241b3971f6cfc68f7a7edbc788b956063ec5a70fc72`. | Isolated SFT experiment only, never flagship data. |
| V8 full-text decontamination | All `question`, `response`, and `completion_prompt` text audited: 0 malformed rows, 0 exact-eval rows, 0 13-gram-eval rows. | Passes lexical gate, not a capability claim. |
| V8 2,048-token packing | 73,273 packed sequences: math 43,767, procedural 24,847, code 2,660, teacher 1,999. Maximum replay factor 2.755x (code). | Meets preflight data gate; held for the isolated raw-to-V8 transfer chain. |
| TACO shuffled all-test audit `686584` | Last durable log: 400/3,000 selected candidates passed all supplied bounded stdin/stdout tests; 1,605 source rows scanned. The active pre-fix partial file is not treated as durable. | In progress. Success path is `686585 -> 686586`; non-success retry is `686659 -> 686660 -> 686661` with immutable input and all tests retained. |
| Verifier rollout `686536` | 78,654 emitted rollout rows at ledger refresh; generator log had reached 5,100/10,000 prompts and 81,600 sampled candidates. | In progress. It is not training data until the tail, global dedup, exact packing, and >=3,000 packed-512-sequence gate succeed. |
| OpenMathReasoning COT selector `686672` | Under full problem+trace decontamination, final-answer verification, individual limits, and an exact combined 2,048-token SFT limit: 326/10,000 rows retained. Rejections: 9,398 long traces, 17 long combined examples, 198 answer mismatches, 1 exact-problem hit, 45 13-gram hits, 8 duplicate problems. | Inspection-only. No bulk candidate is authorized until yield, data balance, and source-specific quality review are recorded. |
| 25B DCLM / FineWeb replacements | FineWeb job `686530` completed only 4,599,748,648 tokens because it used `sample-10BT`; it is explicitly rejected as a 25B replacement. Corrected Stokes CPU job `738030` uses `sample-100BT`, writes only `fineweb_edu_25b_r2.partial`, enforces a >=24.5B manifest-token floor, and last verified 1 shard / 100,001,543 tokens. Newton DCLM `686529` remains live at 188 partial 100M-token shards (about 18.8B tokens), with transient Hugging Face 503 retries. | Not admitted; no partial or pilot output may enter a future relaunch. |
| VRWM r3 transition SFT | 497,274 unique solver-checked rows, 0 malformed rows, duplicate prompts, or full-text evaluation overlaps; 18,013 packed 2,048-token sequences. SHA-256 `b2a688e1f7aa6c79dd65ed1944fa5dc00cd022acfc793896ecf4696c94d4089f`. One epoch `686742` wrote `sft_ep1.pt` (MD5 `90607e7307187c2ad4839d48dfa3a0c6`). Full default p80 closed-loop result: 43/400. | Rejected as template-bound: held-out paraphrase p10 is 0/50. |
| VRWM r4 controlled ablation | Both state-only and deterministic-scratch branches: 513,902 audited rows, 0 malformed/duplicate/public-overlap rows; state SHA-256 `cfab3c0c06cd5eba419d42cd52937ab7159e8f30acc2bc1202375ea38c162e58`, scratch SHA-256 `0df3d86471ccc675ad2dea07bb19cd7ffd97adde5c78b3e92b7fb1581c7d7b10`. | State: 32/400 default, 2/400 semantic. Scratch: 120/400 default, 21/400 semantic. Narrow executable-state evidence only; not general reasoning or promotion. |
| VRWM r5 repair curriculum | 1,409,072 audited rows / 68,347 packed sequences, 139,976,150 total SFT tokens and 38,629,088 answer tokens. SHA-256 `011282f032963a40b8b39ab9572808de1d3473ef2b57ef727526fb9d00985c76`; zero malformed, duplicate, exact-eval, or 13-gram-eval rows. SFT `686820` completed on evc37 in 1,303s; its locally and remotely preserved checkpoint md5 is `ef99f8c2ab5835c8229bcd4f36fb8789`. | Rejected for broad promotion. Semantic p80 first-pass is 17/400, below r4 scratch's 21/400; remaining default/self-repair jobs are diagnostic-only. |

## Capability and Monitoring Baselines

These numbers are deliberately retained as baselines, not marketing claims. The current
model has **not** met the project reasoning target.

| Checkpoint / model | GSM8K | MATH-500 | HumanEval | MBPP | Interpretation |
|---|---:|---:|---:|---:|---|
| Raw `best_step168750.pt` | maj@4 5/100; pass@1 2/100 | 2/100 | 7/164 | 0/100 | Current broad public baseline; weak general reasoning and code. |
| V4 r3 isolated SFT | maj@4 5/100; pass@1 14/100 | 1/100 | 2/164 | 0/100 | Rejected: narrow procedural improvement did not transfer to broad math/code. |
| V5 primitive isolated SFT | maj@4 10/100; greedy 9/100 | 3/100 | 2/164 | 0/100 | Rejected: arithmetic-format gain with code regression versus raw. |

Additional independent evidence:

- Raw 120k balanced held-out Reasoning-Gym baseline: **29/800 = 3.625%**.
- V4 r3 matched held-out procedural score: **209/800 = 26.125%**. This is diagnostic
  transfer, not a clean data-only attribution and not broad-reasoning evidence.
- Fresh manual raw-180k interaction probe, 7 hand-authored cases with greedy 32-token
  completions: **1/7 initial, 0/7 review, 1/7 supplied-fact, 0/7 state reuse**. The sole
  correct answer was the simple syllogism. It is a transcript-level directional check rather
  than a formal comparison to the prior 128-token probe, but shows no visible reasoning jump.
  Artifact: `artifacts/eval_history/manual_capability_raw180k_20260712_mps32.json`, MD5
  `cc6332a5c99d6cbf6ba2f8987ae58cc0`.
- Raw-260k continuation-mode confirmation uses 20 fresh, fixed-seed cases and
  immutable transcripts. Strict first-segment final accuracy is **4/20 direct
  QA**, **1/20 bare expression**, and **8/20 two-example worked continuation**.
  The only robust family is sequential add/multiply/subtract: **4/5 direct**
  and **5/5 worked**, with all required intermediates present. Multiply-
  subtract is 1/5 worked, modular update 2/5 worked, and base conversion 0/5
  in every mode. Transcript SHA-256 is
  `f333c8f54383c411813551bc2001077b88e49514923b76c3cfe0331e9fd6bb47`;
  hash-bound corrected assessment SHA-256 is
  `058aa9dafdc741efc181e6377db5d46b233875504b4b4b6d92837a0db71ea62b`.
  This is narrow procedural competence plus response-mode brittleness, not a
  broad reasoning score.
- Raw-200k counterfactual verifier feasibility probe: over **48** balanced, grammar-valid local state
  transitions, free verdict generation is **0/48** and fixed-completion likelihood chooses `valid` for
  every case, therefore **24/48 = 50.0%**. This is negative evidence against a hidden self-checking
  ability; it must not be used as a verifier without supervised training and a label-shuffled control.
  Artifact: `artifacts/eval_history/transition_verifier_likelihood_raw200k_20260713_mps.json`, MD5
  `fb7bbdbb1fa16104117f09c6c3faa07c`.
- VRWM raw H100 p80 control is **0/400** exact first transitions and **0/400** closed-loop programs.
  r4 scratch increases the isolated protocol to **120/400** default-prompt closed-loop programs, but only
  **21/400** under the reserved semantic prompt form and only **3/80** at default length 32. This is a
  bounded, generated-state transition policy, not evidence that the base model now thinks through ordinary
  questions. The r5 self-repair comparison must improve the semantic and long-horizon rows without a
  controller-side correction before it can advance beyond research.
- Matched direct operator interview (eight fresh non-VRWM questions): raw 180k is **1/8** initial and
  **1/8** when supplied a correct intermediate fact; r5 is **0/8** initial, review, supplied-fact,
  valid-state, and reuse. Its verbatim outputs are synthetic `check:` / `wm:` transitions even for logic
  and Python requests. This is response-mode collapse, so r5 must not be compared on public boards or
  considered a broad-reasoning checkpoint. Artifacts: raw
  `generalization_interview_raw180k_mps_20260712_r3.json` MD5 `c4cef6117b53965776eae259868bedbb`; r5
  `generalization_interview_vrwm_r5_180k_mps_20260712.json` MD5 `e67c6b589e6fb5d9171472129a3873c5`.
- Fixed raw-170k monitor results: WikiText-103 test NLL **3.9648849**, PPL **52.7142** over
  301,056 targets; CodeContests test NLL **1.3537146**, PPL **3.8718** over 145,408 targets.
  They are trend monitors only. The code monitor is not source-disjointness proof, so
  HumanEval/MBPP and execution-based held-out tests remain decisive.

The VRWM r5 paired first-pass/self-repair gate is the immediate context-scaling measurement. The separate
raw-180k -> V8 SFT -> board/interview chain remains the broad-capability measurement. Neither branch can
be promoted on loss, formatting, generator holdouts, or a single benchmark movement alone.

## 260k Reasoning Sprint Ledger: 2026-07-15

### Protected pretraining denominator

- Model: 125,081,664 trained parameters.
- Exact step 260,000 nominal update tokens: **136,314,880,000**
  (`260000 * 524288`). This counts replay and is not a unique-token claim.
- Mounted decoded-token manifest capacity: **57,826,022,271** across 290
  shards: FineMath-4+ 2,000,001,108; OpenWebMath 14,063,689,153;
  CodeParrot-Clean Python 16,762,327,600; FineMath-3+ 25,000,004,410.
- Nominal update-token / mounted-capacity ratio at 260k: **2.3572x**. Because
  the loader round-robins directories rather than weighting by manifest size,
  this aggregate ratio is not a per-source exposure estimate.
- Durable raw checkpoint: Newton `best_step260000.pt` and local
  `train/flagship_out/ckpt_0260000.pt`, 1,076,597,546 bytes, MD5
  `301082250e15c26820790ec7ff7730a0`.
- Live continuation `686732`: two H100s, `BS=32`, `ACC=4`, exact same 524,288
  tokens/update. At 2026-07-15 04:26 EDT it was healthy through step 264,890 at
  about 283.38k tok/s. Four gnorm skips at 264,031--264,034 recovered without
  intervention by 264,040.

### Raw-260k capability accounting

| Evidence | Calls / cases | Strict result | Artifact SHA-256 |
|---|---:|---:|---|
| Frozen continuation-mode confirmation | 60 generations / 20 cases | direct 4/20; bare 1/20; worked 8/20 | transcript `f333c8f54383c411813551bc2001077b88e49514923b76c3cfe0331e9fd6bb47`; assessor `058aa9dafdc741efc181e6377db5d46b233875504b4b4b6d92837a0db71ea62b` |
| Failed `Next state` SSC renderer | 55 calls / 20 chains | 0/20 chains; 43/55 outputs equal input+1 | `a152e85294d02173a697e29d8537bf4b53428d747d16c7e3baf692095d9b6a2f` |
| Three-renderer source-free matrix | 330 calls / 20 cases | `Problem/Work` 44/55 atomic, 10/20 chains | `b33c26b3963296c0d97b2a6d3332c0be18af40f460137c25652b881824a1ca4b` |
| Causal renderer interchange | 18 candidate-sequence scores / 6 cells | displayed state favored 6/6, min margin 0.79386 | `963177139b6abb333710f0db19a521c341a039fce3f65743ebdd698be6f12170` |
| Fresh source-scheduled confirmation | 256 cases / 704 transitions / at most 1,920 model calls | running as Newton `689535`; locked gates in `R12_SOURCE_SCHEDULED_REASONING_CONFIRMATION.md` | board `19a84165f15b19911fc8ef229022e47753833d703d77d1e8cc25db9dfc993474` |

The 10/20 result is not autonomous model reasoning. The controller imports the
public operation schedule, parses integers, carries model-produced state, and
makes one model call per operation. Those resources remain part of every claim.

### Matched SFT control accounting

- DRS complete-basis SFT `689524`: 311,127 examples, 36,516,108 total tokens,
  7,650,920 answer tokens, 17,830 packed 2,048-token sequences, one epoch and
  1,115 updates. Training completed in 846 seconds after model start. Locked
  900-case evaluation is `689525`.
- STRR factorized static-tape SFT `689526`: 311,127 examples, 36,532,447 total
  tokens, 4,123,456 answer tokens, 17,838 packed sequences, one epoch and 1,115
  updates. Training completed in 824 seconds. Locked 900-case evaluation is
  `689527`.
- Both start from immutable raw `best_step200000.pt`; neither writes the
  flagship output. The first attempts `689496/689498` timed out before their
  first update because a pure prompt-boundary check imported the full PyTorch
  module over Lustre. The lightweight `train/sft_encoding.py` correction was
  smoke-tested at 1.78 seconds before clean resubmission.

### WGRQ Stage-A CPU accounting

Stokes `739105 -> 739106` generated and independently replay-audited exactly
18,432 committed episodes, four histories per episode, 32 ordinary one-bit
answers per episode, and **589,824 answer calls**. The immutable files total
302,572,503 bytes and are mode 0444 on Stokes. Their transcript, call-ledger,
generation-report, and audit-report SHA-256 values are respectively:

```
ae2849db5d57fda36e2e2fd634ce6e1d0f11eaed7fefe8d9ce722f016f28295a
251d85432d845c31ce64da1adae132fa8df8f6a63b5db744654b519f2413c9e8
12c1e54f23b27f3a97a86857b723fec3573f5d558b7528e1615c55746899befb
8f5fac80e0c50bdc807287599f8468194431f3612d6d79a1331f51a073fa2dd4
```

No fit was launched. An adversarial implementation audit found a broken
relation-sham stratum contract, an independent-audit bypass, and a scorer that
could accept arbitrary checkpoint bytes plus hand-authored success rows. The
locked preregistration therefore closes v1 before its planned 60 fits.

## Update Protocol

At each 10k checkpoint milestone:

1. Confirm the exact numbered checkpoint exists at the milestone, copy it to the corresponding
   `best_step<step>.pt`, and record the remote MD5. Expect the trainer to reclaim old numbered
   files; `best_step` and the verified local copy are the required durable artifacts.
2. Transfer to `train/flagship_out/ckpt_<step>.pt.part` (or a resumable equivalent). Verify
   the local MD5 against Newton before atomically renaming it without `.part`.
   `scripts/preserve_flagship_checkpoint.sh <step>` performs remote `best_step` promotion,
   resumable `sftp reget`, matching-MD5 verification, and atomic local rename in one command.
3. Update the pretraining table with the exact step, nominal update-token count, latest
   throughput/loss/gnorm, and local DR status. Do not infer unique-data exposure from step count.
4. Update each data row only from a saved manifest, hash-bound report, or completed job log.
   Label running work as in progress and never count an unflushed partial as admitted data.
5. Add a terse append-only milestone to `AGENT_RUNBOOK.md`, sync both documents to Newton,
   and commit/push docs and safe code only. Never commit checkpoints, `.env`, or live writer output.

## Primary Evidence Paths

- Local runbook: `AGENT_RUNBOOK.md`
- Local checkpoints: `train/flagship_out/`
- Newton checkpoints/log: `/lustre/fs1/home/sa305415/shohin/train/flagship_out/` and
  `/lustre/fs1/home/sa305415/shohin/logs/flagship_685084.out`
- Active corpus manifests: `/lustre/fs1/home/sa305415/shohin/artifacts/shards/*/manifest.json`
- Frozen V8 reports: `/lustre/fs1/home/sa305415/shohin/artifacts/sft/sft_mix_reasoning_v8_candidate_r2.*.r3.json`
- External-source selection reports: `/lustre/fs1/home/sa305415/shohin/artifacts/source_probes/`
