# Shohin Native Reasoning Master Ledger

**Purpose:** This is the durable, theory-facing record of what Shohin is, what
we mean by native reasoning, what has been tried, what happened, and what a new
theory must explain. It is written so that a researcher can propose a new
mechanism without first reconstructing several days of experiments from the
runbook and result files.

**Status:** Living document. The protected raw-pretraining anchor is complete
at 300,000 steps. Native multi-step reasoning is not yet established.

**Last updated:** 2026-07-18 19:25 EDT.

**Operational source of truth:** [AGENT_RUNBOOK.md](AGENT_RUNBOOK.md)

**Training metrics source of truth:** [TRAINING_METRICS.md](TRAINING_METRICS.md)

**Research charter:** [R12_REASONING_INVENTION_CHARTER.md](R12_REASONING_INVENTION_CHARTER.md)

**Coverage rule:** This ledger includes every substantive data, training,
architecture, controller, state-transport, probe, motor, curriculum, or theory
lane that reached a scientific decision or materially changed the diagnosis.
Pure infrastructure retries and jobs canceled before model execution are
included only when the failure exposed a scientific or custody confound; the
full job-by-job chronology remains in `AGENT_RUNBOOK.md`.

---

## 1. What We Are Trying To Do

The goal is to give **Shohin native reasoning** while keeping the complete
model under 150 million parameters.

The intended system should accept a natural-language problem, determine what
operations are required, maintain and update its own task state, use that state
across multiple dependent steps, and emit a correct terminal answer. The model
must do this itself. A host program may tokenize input and decode output, but it
may not solve the task on the model's behalf.

The long-term product target is a small, verifiable reasoning specialist for
math, code, and logic. The immediate scientific target is narrower and more
fundamental:

> Demonstrate a model-owned, causally necessary computation cycle that
> generalizes beyond its training templates and beats favorable matched
> controls.

### 1.1 The native reasoning contract

A result counts as native reasoning only if Shohin owns all five interfaces:

1. **Compilation:** Convert the natural-language problem into the required
   operation sequence or executable internal program.
2. **State creation:** Construct the state needed to solve the problem without
   receiving a gold intermediate state.
3. **State transition:** Apply the correct operation to the current state and
   replace it with the correct next state.
4. **State reuse and control:** Consume the updated state over later steps,
   select the next operation, detect completion, and avoid replay loops.
5. **Serialization:** Return the requested answer exactly and halt.

The following are useful diagnostics or ceilings, but **do not** count as
native reasoning:

- a host supplies the operation schedule, cursor, state, carry, or answer;
- a host executes arithmetic or code between model calls;
- an external verifier searches, repairs, retries, or selects the correct
  candidate at inference;
- the model receives gold intermediate states at inference;
- a decoder extracts a fact that the model cannot autonomously use;
- a model succeeds only on a memorized wording, fixed width, or fixed value
  range;
- a trace looks thoughtful but its intermediate values are wrong;
- a hidden probe is accurate without a causal intervention showing that the
  represented state is required by held-out consumers.

### 1.2 Required evidence for a positive claim

A promoted mechanism must satisfy all of the following:

- **Fresh transfer:** New paraphrases, values, widths, lengths, and operation
  order twins are frozen before evaluation.
- **Source deletion:** After the model forms its state, the original source is
  removed where the hypothesis says the state should be sufficient.
- **Causal necessity:** Correct state swaps help; wrong, shuffled, zeroed, and
  norm-matched swaps hurt in the predicted direction.
- **Autonomous rollout:** No oracle schedule, state, host arithmetic, host
  executor, or verifier repair is present.
- **Matched controls:** Compare against static SFT, an ordinary recurrent
  model, retrieval, a favorable tied recurrence, and host execution whenever
  those are equivalent or stronger baselines.
- **Resource accounting:** Count total parameters, retained state bits, source
  bytes, target bits, examples, optimizer updates, training and inference
  compute, sequential depth, and external work.
- **Direct interaction:** Inspect complete transcripts in addition to aggregate
  scores. A benchmark score alone can hide loops, leakage, invalid formatting,
  or a host-owned solution.
- **Preservation:** Broad language/code capability and previously established
  skills may not silently collapse in exchange for one template-local gain.

### 1.3 Why this is difficult at Shohin's scale

Shohin is small enough that it often learns local token associations and
individual arithmetic transitions without learning a robust controller. The
evidence repeatedly shows a gap between **local competence** and
**composition**:

- correct first local DRS state on 497/500 core episodes, but only 275/500
  complete final answers;
- 44.92% with an externally supplied schedule, versus 3.52% for whole-problem
  autonomous generation;
- a strong post-DRS residual digit signal, but almost no autonomous gain from a
  digit motor;
- operation and query factors can improve while exact full programs remain
  wrong;
- raw pretraining loss continues to look healthy while public reasoning scores
  remain in the low single digits.

The project is therefore not trying to make the model narrate longer. It is
trying to install or discover a compact control-and-state mechanism that the
model can compile, update, consume, and terminate by itself.

---

## 2. Shohin Technical Profile

### 2.1 Base architecture

The immutable 300k checkpoint instantiates the following decoder-only GPT:

| Field | Value |
|---|---:|
| Unique trained parameters | **125,081,664** |
| Allowed total for experimental architecture | **less than 150,000,000** |
| Remaining nominal parameter headroom | **24,918,335** |
| Vocabulary | 32,768 |
| Layers | 30 |
| Model width | 576 |
| Feed-forward width | 1,536 |
| Feed-forward activation | SwiGLU |
| Query heads | 9 |
| Key/value heads | 3 |
| Head dimension | 64 |
| Context length | 2,048 tokens |
| Positional encoding | RoPE, theta 50,000 |
| Attention details | Grouped-query attention and QK normalization |
| Embeddings | Input/output embeddings tied |
| Auxiliary loss | z-loss 0.0001 |
| Base recurrence | One transformer pass (`n_loop=1`) |

The training stack uses bf16, `torch.compile`, Muon for matrix parameters,
AdamW for the remaining parameters, guarded gradient-norm skips, and WSD-style
learning-rate scheduling. The final two-H100 continuation used data parallelism
only; it did not change the model architecture.

### 2.2 Final pretraining run

| Field | Verified value |
|---|---:|
| Final step | **300,000 / 300,000** |
| Global tokens per optimizer update | **524,288** |
| Nominal token exposures | **157,286,400,000** |
| Mounted manifest capacity | **57,826,022,271 decoded tokens** |
| Aggregate exposure/capacity ratio | **2.7200x** |
| Final logged loss | **1.6554** |
| Final logged gradient norm | **0.11** |
| Final logged learning rate | **0.0005** |
| Final sustained throughput | **281,959 tokens/s** |
| Final job | Newton `686732`, two H100s, completed cleanly |

The 157.286B figure is nominal update-token exposure, not unique data. The
loader interleaves sources and the run includes corpus replay. It is incorrect
to call this 157B unique training tokens.

### 2.3 Mounted pretraining corpus

| Source | Decoded tokens | Shards | Role |
|---|---:|---:|---|
| FineMath4+ | 2,000,001,108 | 10 | Curated mathematical text |
| OpenWebMath | 14,063,689,153 | 71 | Web mathematical text |
| CodeParrot Clean Python | 16,762,327,600 | 84 | Python code |
| FineMath3+ | 25,000,004,410 | 125 | Expanded mathematical text |
| **Total** | **57,826,022,271** | **290** | Active 60k-to-300k stream |

Shard manifests include evaluation n-gram filtering for the math, web-math,
code, and FineMath3 sources. This reduces direct contamination risk but does
not prove source-level disjointness for every public benchmark.

### 2.4 Checkpoint custody

The immutable raw 300k checkpoint is:

| Property | Value |
|---|---|
| Local path | `train/flagship_out/ckpt_0300000.pt` |
| Newton names | `ckpt_0300000.pt`, `best_step300000.model.pt` |
| Size | 500,448,522 bytes |
| MD5 | `60de77c31b449060ff0417d8db16d3b0` |
| SHA-256 | `211d6b2cddf0c2cf8b12cb0b2d73f9c4440d85f6f531018080c8afd35b2f66a6` |
| Terminal artifact | Model-only, no optimizer state |
| Resume requirement | Fresh optimizer rewarmup |

No flagship writer is active. All reasoning experiments must use isolated
output paths and must not mutate this checkpoint.

### 2.5 Final raw capability baseline

The final standardized public board used 100 math examples, GSM8K majority@4,
and the complete 164-problem HumanEval set:

| Checkpoint | GSM8K maj@4 | GSM8K pass@1 | MATH-500 | HumanEval | MBPP |
|---|---:|---:|---:|---:|---:|
| Raw 120k | 2/100 | 1/100 | 3/100 | 7/164 | 0/100 |
| Raw 168.75k | 5/100 | 2/100 | 2/100 | 7/164 | 0/100 |
| **Raw 300k** | **4/100** | **2/100** | **2/100** | **6/164** | **0/100** |

The differences are one or two items in either direction. The honest result is
a low-single-digit plateau, not broad improvement from continued raw
pretraining.

### 2.6 Direct interaction with raw 300k

On the fixed seven-case, five-turn interaction protocol:

| Checkpoint | Initial | Review | Supplied fact | Compact-state reuse |
|---|---:|---:|---:|---:|
| Raw 200k | 1/7 | 0/7 | 1/7 | 0/7 |
| Raw 260k | 1/7 | 0/7 | 1/7 | 0/7 |
| **Raw 300k** | **1/7** | **0/7** | **1/7** | **0/7** |

A separate researcher-selected raw-300k interview scored 1/6 semantically and
0/6 under the requested output contracts. The model can sometimes emit a
correct visible arithmetic chain, but it does not reliably bind operations to
the current state, replace the state, reuse a source-deleted state, obey output
contracts, review itself, or halt.

Primary evidence:

- [RAW300K_INTERACTION_RESULT.md](RAW300K_INTERACTION_RESULT.md)
- [RAW300K_FREEFORM_INTERACTION_RESULT.md](RAW300K_FREEFORM_INTERACTION_RESULT.md)
- [R12_RESEARCHER_INTERVIEW_RESULT.md](R12_RESEARCHER_INTERVIEW_RESULT.md)
- [R12_RESEARCHER_ADAPTIVE_INTERACTION_RESULT.md](R12_RESEARCHER_ADAPTIVE_INTERACTION_RESULT.md)

---

## 3. Current Best Scoreboard, With Claim Boundaries

The highest numbers are not necessarily the most native systems. This table
orders the important systems by result while exposing who owns the reasoning
loop.

| System | Exact score | What owns the computation | Scientific class |
|---|---:|---|---|
| Source-scheduled continuation (SSC) | **115/256 = 44.92%** | Host supplies source and operation cursor; model executes local steps | External-control ceiling |
| SCEB typed closed loop | **65/256 = 25.39%** | Small controller heads plus host arithmetic/register bus | Architectural control, not native |
| Halt-first decoding | **61/256 = 23.83%** | Model weights unchanged; decoder stops at first valid answer | Decode-policy control |
| Typed controller v1 | **42/256 = 16.41%** | Model emits typed operations and DONE | Weak autonomous joint emission |
| NL SCEB | **8/51 = 15.69%** | Model predicts operations from natural language; host executes them | Partial compiler signal, external executor |
| Whole Problem/Work raw 260k | **9/256 = 3.52%** | Model owns full generation | Native baseline, weak |
| Direct raw 260k | **16/256 = 6.25%** | Model owns full generation | Native direct baseline, weak |
| Raw 300k fixed interaction | **1/7 initial** | Model owns full generation | Qualitative native baseline |

The best complete model-owned result so far is still weak. The 44.92% SSC
score is valuable because it proves that Shohin has substantial local execution
ability when the missing controller is supplied. It is not an autonomous
reasoning score.

---

## 4. Working Decomposition Of The Failure

The present evidence decomposes the missing capability as follows:

| Component | Best direct evidence | Current conclusion |
|---|---|---|
| Natural-language compiler | Fresh component probe: **0/6** parseable compiled programs | Missing |
| Local arithmetic executor | Oracle-compiled frozen DRS: **28/34** transitions; SSC local add/multiply often above 79% | Real but narrow |
| Persistent state representation | DRS post-training digit swaps: **10/10** positive at layers 17, 21, 25, 29 with about **+31 delta log-odds** | Real late-layer signal |
| Carry representation | Layer 29: **10/10**, mean **+2.96 delta log-odds** | Present but weaker |
| State actuator/serializer | Result motor adds only **0.8 points** to full loop; terminal component probe **2/6** | Missing/reliably weak |
| Cursor/operation selection | Full source+cursor likelihood **80/176**, controls **64/176**, but cursor changes only 1/112 relevant choices and predicts no multiply/remainder | Lexical cue, not scheduler |
| Recurrent consumption | DRS first state 497/500 but final 275/500; complete basis 63/900 finals | Compounding transport failure |
| Halt/DONE | Raw runs hit caps; typed v1 learns DONE but not arithmetic; typed v2 destroys DONE | Missing and entangled |
| Self-review | Fixed raw probes remain 0/7 | Missing |
| Broad transfer | Raw/SFT/code boards remain low; width-8 DRS is zero | Missing |

This diagnosis rules out the simplest stories:

- The problem is not only insufficient arithmetic knowledge.
- The problem is not only output formatting.
- The problem is not only the absence of a hidden workspace.
- The problem is not solved by exposing longer chain-of-thought text.
- The problem is not solved by more raw tokens under the current architecture
  and data mixture.

The strongest surviving interpretation is:

> Shohin has fragments of a local executor and a trainable late-layer state
> signal, but it lacks a robust language-to-program compiler and a closed-loop
> controller that updates, consumes, and terminates that state.

---

## 5. Complete Experiment Ledger

The entries below are grouped by the scientific question they tested. `GO`
means only the explicitly named gate passed. It never implies broad reasoning.

### 5.1 Raw pretraining, optimization, and scaling

| Attempt | Result | Decision / lesson | Evidence |
|---|---|---|---|
| 60k divergence repair and domain-interleaved loader | Stable 60k completion | Training infrastructure fixed; not a reasoning result | `AGENT_RUNBOOK.md` |
| 60k to 300k continuation | 300k complete, 157.286B nominal exposures | Durable raw anchor | [RAW300K_INTERACTION_RESULT.md](RAW300K_INTERACTION_RESULT.md) |
| One-H100 BS16/ACC16 | About 148k tok/s, 99-100% utilization | Stable baseline | `AGENT_RUNBOOK.md` |
| One-H100 BS32/ACC8 | About 154.5k tok/s, about 64 GB | Adopted at natural handoff; modest gain | `AGENT_RUNBOOK.md` |
| One-H100 BS64/ACC4 | OOM at 78.93 GiB | Rejected | `AGENT_RUNBOOK.md` |
| Two-H100 BS16/ACC8 | About 260-275k tok/s | Validated DDP path | `AGENT_RUNBOOK.md` |
| Two-H100 BS32/ACC4 | About 282k tok/s final | Production continuation, same global update | `AGENT_RUNBOOK.md` |
| Whole-update CUDA graphs | Clean canary gained only about 1.8% and removed guard/observability | Rejected for integration | `AGENT_RUNBOOK.md` |
| Two-pass recurrence ablation (`n_loop=2`) | Loss 2.4899 versus 2.4890 control; 286k versus 472.7k tok/s | Stable but much slower, with no measured capability gain | `AGENT_RUNBOOK.md` |
| Raw 80k/120k/168.75k/300k boards | Low-single-digit, non-monotonic changes | Scaling did not yield broad reasoning | `AGENT_RUNBOOK.md`, [TRAINING_METRICS.md](TRAINING_METRICS.md) |
| Raw 200k/252.5k/260k/300k direct interviews | Fixed strict scores essentially flat | More raw tokens did not create reliable state reuse or self-correction | [RAW300K_INTERACTION_RESULT.md](RAW300K_INTERACTION_RESULT.md) |
| Raw 300k future-Jacobian repeat | Reproducible geometry, but **0% top-10 and 0% top-100** on 2,304 decisive targets | No usable vocabulary-aligned semantic workspace emerged | [R12_JACOBIAN_WORKSPACE_LONGITUDINAL_RESULT.md](R12_JACOBIAN_WORKSPACE_LONGITUDINAL_RESULT.md) |

### 5.2 General SFT and verified data curricula

| Attempt | Result | Decision / lesson |
|---|---|---|
| Original curated SFT baseline | GSM8K 6/100, MATH 0/100, HumanEval 4/164, MBPP 0/100 | Weak recipe; do not repeat blindly |
| Frozen core SFT mix | 97,439 rows; 0 malformed, 0 duplicate questions, 0 exact eval-prompt hits; dropped 206 exact and 741 eval 13-gram overlaps | Established minimum data-quality bar |
| Reasoning v2 mix | 349,449 rows, including 83,611 execution-verified RG traces | Clean frozen training source |
| Reasoning v2 one-epoch pilot from 120k | GSM pass@1 improved, but MATH/code did not improve consistently; later board 14 GSM pass@1, 6 MATH, 6 HE, 0 MBPP | Narrow arithmetic-format learning, not broad promotion |
| V4 invalid staging attempts | One used stale mixture weights; one exposed 461/3,542 legacy BPE prompt-prefix mismatches; both stopped before a valid artifact | Custody and prompt-boundary failures caught before promotion |
| V4 source-balanced SFT | GSM maj/pass 5/14, MATH 1, HE 2, MBPP 0 | Rejected broad recipe |
| V5 primitive SFT | Primitive 272/700; public GSM 10/9, MATH 3, HE 2, MBPP 0 | Teaches selected primitives but regresses code and does not create general composition |
| V6 response-contract SFT | Contract holdout 20/245 to 142/245; deep interaction 4/8 initial and 0/8 reuse | Learned review/scaffold contracts but produced invalid compact calculations |
| V7 typed-state SFT | 307/420 answers and 169/280 exact states versus raw 21/420 and 0/280 | Typed repair/reuse format is learnable; independent interaction was only 1/8 and 0 reuse |
| V8 large audited mix | 699,928 rows prepared; direct 0/8 and trace 0/12 | Clean scale did not create visible native reasoning |
| V9 broad plus semantic memory | GSM maj 18/200, pass 21/200, MATH 5/200, HE 5/164, MBPP 1/200; direct 1/8; trace tags 9/12 but 0 correct | Format and narrow benchmark movement; rejected semantic-memory claim |
| V10A family-trace bridge | 123/500 answers, 121/500 format-local state; source-dropped cross-family 4/500 | Learned family syntax, not portable state |
| Behavior-preserving retention SFT | RG 31/800 to 143/800; visible trace 0/12 to 3/12; state OOD 0/72 to 6/72, but arithmetic 7/100 and base conversion 2/100 | Retained as narrow baseline, rejected as broad operator reasoning |
| Operator-trace v2 COTA | Template-local primitive 97/100, but arithmetic/base 0/100, RG 58/800, trace 0/12 | Rejected; paired-answer grammar and catastrophic specialization |
| Direct-only broad anchor | RG 173/800, but trace 0/12, state OOD 4/72, arithmetic 8/100, base 3/100 | Rejected; removing response-mode leakage did not create transport |
| Matched reflection vs neutral data | 57,015 rows/arm, audited and token matched | CPU-admitted but dormant because prerequisite failed |
| Frozen broad/primitive/RG sources | Broad-v4 643,595 rows; primitives 210,000; RG-v4 374,659; OpenMath PT 5B tokenized tokens | Durable data assets; admission and scale are not capability evidence |
| Verified teacher distillation | HY3 generated about 25.2k rows before harness/provider failure; Nemotron about 1.78k; GLM provider unavailable | Preserve clean snapshots; no live-writer training; provider lanes paused |

The repeated SFT pattern is important: supervised formatting can improve a
specific family or response mode while destroying DONE, code, arithmetic, or
transfer. Lower loss and a longer rationale are not sufficient.

### 5.3 Source-scheduled execution and decoder controls

| Attempt | Result | Decision / lesson | Evidence |
|---|---|---|---|
| Raw 260k direct | 16/256 = 6.25% | Weak autonomous baseline | [R12_SOURCE_SCHEDULED_REASONING_CONFIRMATION_RESULT.md](R12_SOURCE_SCHEDULED_REASONING_CONFIRMATION_RESULT.md) |
| Raw 260k whole Problem/Work | 9/256 = 3.52% | Full composition collapses | Same |
| Source-scheduled continuation | 115/256 = 44.92%; oracle-state atomic 534/704 = 75.85% | Strong local executor ceiling; host owns schedule | Same |
| Source-scheduled failure taxonomy | 96 wrong first operation, 37 first-arithmetic errors, 71 loop/replay, 36 reached then lost answer, 214/256 loop signatures | Compiler, updater, and halt are separate failures | [R12_SOURCE_SCHEDULED_FAILURE_TAXONOMY.md](R12_SOURCE_SCHEDULED_FAILURE_TAXONOMY.md) |
| First-integer offline rescore | Corrected truncation/scoring issue | Evaluation repair only | [R12_SSC_FIRST_INTEGER_OFFLINE_RESULT.md](R12_SSC_FIRST_INTEGER_OFFLINE_RESULT.md) |
| Halt-first live decoding | 61/256 = 23.83% | Termination policy matters; no new reasoning weights | [R12_SSC_HALT_FIRST_LIVE_RESULT.md](R12_SSC_HALT_FIRST_LIVE_RESULT.md) |
| Updater-candidate likelihood | 0/6 wins | No hidden updater recovered by simple ranking | [R12_UPDATER_CANDIDATE_LIKELIHOOD_RESULT.md](R12_UPDATER_CANDIDATE_LIKELIHOOD_RESULT.md) |

### 5.4 Typed controllers and controller/executor separation

| Attempt | Result | Decision / lesson | Evidence |
|---|---|---|---|
| Typed controller v1 | 42/256 = 16.4%; DONE 86.3%; atomic step 27.3% | Format and DONE learned; arithmetic/controller composition remained weak | [R12_TYPED_CONTROLLER_V1_RESULT.md](R12_TYPED_CONTROLLER_V1_RESULT.md) |
| Typed controller v2 | 0.8%; DONE 0%; atomic step 26.6% | Mixing native Compute SFT destroyed the typed mode | [R12_TYPED_CONTROLLER_V2_RESULT.md](R12_TYPED_CONTROLLER_V2_RESULT.md) |
| Host execution of LM-emitted steps | About 1.2% | LM step emission ignored cursor; host arithmetic cannot rescue wrong control | `REASONING_ATTACK_PLAN.md` |
| SCEB typed closed loop | 65/256 = 25.4% | Discrete heads can control a host register bus; still external math | [R12_SCEB_RESULTS.md](R12_SCEB_RESULTS.md) |
| NL SCEB | 8/51 = 15.7%; op+DONE step about 62% | Some natural-language op signal, but no autonomous executor | Same |
| RegisterAugmentedGPT / discrete controller heads | Implemented and used as controls | Useful architecture substrate, not a reasoning success by itself | `train/register_augmented_gpt.py`, `train/discrete_controller_heads.py` |

### 5.5 Digitwise Recurrent Scratchpad (DRS) and matched recurrence

#### DRS v2 core

- 439,865 train rows.
- 51,131,402 packed tokens.
- 10,623,342 masked answer tokens.
- 24,966 packed sequences and 1,561 updates.
- Training loss moved from 0.6846 to 0.0115.
- Final exact answers: **275/500 = 55%**.
- First model-authored state: **497/500**.
- Width-4 fit: 100/100; width-6 fit: 98/100.
- Value OOD width 4: 34/100; width 6: 43/100; width 8: 0/100.

This is the clearest evidence that local execution can be learned while serial
state transport collapses.

#### Complete-basis DRS v3 versus static-tape STRR

| Arm | First transition | Correct attempted transitions | Exact final | Closed-loop state | Paired intervention |
|---|---:|---:|---:|---:|---:|
| DRS complete basis | 533/900 | 1,259/2,088 | 63/900 | 71/900 | 47/900 |
| STRR static tape | 365/900 | 653/1,537 | 15/900 | 16/900 | 7/900 |

Both arms score 0/300 exact finals on unseen width. Recurrence helps local
transition learning but does not solve length generalization or full-chain
transport.

Evidence:

- [R12_RECURRENT_CONTROLS_RESULT.md](R12_RECURRENT_CONTROLS_RESULT.md)
- `REASONING_FRONTIER.md`, DRS sections

#### Related DRS data/protocol lanes

| Attempt | Result | Decision |
|---|---|---|
| DRS v1 data | 27 held-out 13-gram hits caused by operand-tape reuse | Rejected before GPU |
| DRS v2 split repair | 0 malformed, duplicate, exact, or 13-gram overlap | Admitted |
| DRS held-out wording transfer | Core 275/500 versus held-out wording 125/500 | Some executor transfer, but a 30-point wording drop and width-8 zero |
| Append-only Delta Ledger (ADL) | 384,000 train rows, 1,000 held-out episodes, clean audit | Data admitted; GPU stayed gated because raw primitive top-1 was 0/16 |
| Static Tape Recurrent Register (STRR) | Exact data/control artifact and neural result above | Favorable recurrence control, not broad success |
| Dual-Code Reversible Deliberation (DCRD) | Clean protocol/preflight | No durable positive fit; remains unpromoted |
| Counterfactual Bisimulation Compiler (CBC) | Clean CPU build/evaluation readiness | No autonomous positive result |

### 5.6 Residual workspace and causal motor experiments

| Attempt | Result | Decision / lesson | Evidence |
|---|---|---|---|
| Raw-200k residual digit/carry swaps | Near-zero causal action | Raw model had no simple tested digit workspace | [R12_DRS_WORKSPACE_PROBE_POST_RESULT.md](R12_DRS_WORKSPACE_PROBE_POST_RESULT.md) |
| Post-DRS digit residual | Layers 17/21/25/29 all 10/10 positive, mean about +31 delta log-odds | DRS installed a real late-layer digit broadcast | Same |
| Post-DRS carry residual | Layer 29 10/10 positive, +2.96 mean | Carry signal exists, weaker than digit | Same |
| DRS causal cycle | Baseline first state 38/50; counterfactual residual 14/50; same-target 31/50; direct two-token ceiling 50/50 | Residual write/serialization and carry consumption fail | [R12_DRS_CAUSAL_CYCLE_RESULT.md](R12_DRS_CAUSAL_CYCLE_RESULT.md) |
| Wide result-digit motor, teacher-forced fit | Fit board reached 100%; shuffled true-label control 61.7% in exploratory fit | Residual is readable; fit is not autonomous reasoning |
| Wide result-digit motor, frozen held-out | Digit top-1 91.36% vs base 90.86%; first transition identical 203/250; full loop 63/250 vs base 61/250 | **Rejected** as autonomous actuator | [R12_CAUSAL_RESULT_DIGIT_MOTOR_RESULT.md](R12_CAUSAL_RESULT_DIGIT_MOTOR_RESULT.md) |
| Carry-motor recovery | Source/custody and confirmation-contract defects | `NO-GO`; no valid neural result | [R12_CAUSAL_CARRY_MOTOR_RECOVERY_PREREG.md](R12_CAUSAL_CARRY_MOTOR_RECOVERY_PREREG.md) |

The surviving result is subtle but important: the workspace is not wholly
absent after DRS. The model can carry a digit direction in late residual space.
The failure is converting that signal into a reliable multi-step actuator,
carry update, next-step consumer, and halt decision.

### 5.7 Operation selection, cursor, and compiler probes

| Attempt | Result | Decision / lesson | Evidence |
|---|---|---|---|
| Four-candidate direct likelihood | Correct top-1 1/7, mean rank 2.571 | Correct answer often is not merely hidden behind free decoding | `TRAINING_METRICS.md` |
| Operation-selection likelihood | Full source+cursor 80/176 vs 64/176 controls; add 145, subtract 31, multiply/remainder 0 | Lexical family signal, not operation order | [R12_OPERATION_SELECTION_LIKELIHOOD_RESULT.md](R12_OPERATION_SELECTION_LIKELIHOOD_RESULT.md) |
| Strict operation cursor | 0/528 parseable correct, all hit generation cap | No raw textual cursor |
| Cursor-action neural six-arm study | Restricted action about 20%, full vocabulary 0%, exact groups 0 | `NO-GO` | [R12_COUNTERFACTUAL_CURSOR_ACTION_NEURAL_RESULT.md](R12_COUNTERFACTUAL_CURSOR_ACTION_NEURAL_RESULT.md) |
| Final-token linear readout | Development 43.13%, non-DONE 28.91%, exact groups 0; cursor-only 40% | Weak factor readout, no full state | [R12_CURSOR_READOUT_ACTUATION_RESULT.md](R12_CURSOR_READOUT_ACTUATION_RESULT.md) |
| Token-tape diagnostic | Best cursor-specific 57.81%, non-DONE 47.27%, maximum 11/192 exact groups | Rejects tested single-query linear tape only | [R12_CURSOR_TOKEN_TAPE_RESULT.md](R12_CURSOR_TOKEN_TAPE_RESULT.md) |
| Fresh compiler/executor/serializer component probe | Compilation 0/6; oracle DRS execution 28/34; terminal serialization 2/6 | Compiler and serializer are independently missing | `AGENT_RUNBOOK.md` |
| Operation workspace Jacobian | Norm-matching/validity gate failed closed | No valid causal result | [R12_OPERATION_WORKSPACE_JACOBIAN_RESULT.md](R12_OPERATION_WORKSPACE_JACOBIAN_RESULT.md) |
| Raw-300k longitudinal Jacobian | Reproducible map, decisive semantic top-10/top-100 both zero | No vocabulary-aligned workspace promotion | [R12_JACOBIAN_WORKSPACE_LONGITUDINAL_RESULT.md](R12_JACOBIAN_WORKSPACE_LONGITUDINAL_RESULT.md) |

### 5.8 Microcode, binding, and operator-program experiments

| Attempt | Result | Decision / lesson |
|---|---|---|
| Causal Microcode Bottleneck R1 | Fit 251/256 answers and 250/256 programs; depth OOD 155/192 and 150/192; language OOD 19/256 and 8/256; full OOD 1/192 and 0 programs | Learns structured local compiler but not language transfer |
| CMB R2 output-equivalence control | Candidate/control exact programs both 30/896; combined answer 12.50% vs 11.83% | Output equivalence is not semantic identification |
| CMB R3 role equivariance | Better local operation/query factors, but exact programs remain weak and depth-8 0/64 | Local equivariance is not referential binding |
| R4 binding-first slot compiler | Language 29/256 to 139/256; full 2/192 to 51/192; exact programs 469/896 to 624/896 | Major binding gain, but still not autonomous reasoning |
| R5 future-effect argument algebra | Fresh answers 196/448 vs 195/448; exact programs 174/448 vs 172/448 despite 96.61% arity | Arity transfers; capability does not |
| R6 effect-coded operators | Structured error-correcting code mechanics | No broad positive capability result |
| R7/R8 curvature and geometric operator studies | Counterfactual curvature 26/108 versus random 28/108; numeric chance | Hidden curvature was not operator-aligned |
| R9a static orbit recurrence | Exact collapse to ordinary objective/control | Not a new mechanism |
| R9b bidirectional operator trees | Exact CPU mechanics and noncommutative paths | Mechanics only |
| R9c dynamic directional syndrome | Treatment operation/answer 78.29%/47.77%, below static/no-syndrome controls; full OOD 30.21%; 88.83% common-mode wrong operations | Formally rejected |
| R10 ACAW/VSPT | Strong exact mechanics, but custody could be forged and score chain was never read | Dormant control, second audit `NO-GO` |
| R11 internal workspace | Six slots, width 96, 1,607,334 trainable parameters feasible; v1/v2 contract defects; v3 paper-only and equivalent to known recurrence | Dormant favorable control |

Detailed chronological evidence for this family is in
[REASONING_FRONTIER.md](REASONING_FRONTIER.md).

### 5.9 Prefix packets, latent states, and source-deleted transport

| Attempt | Result | Decision / lesson |
|---|---|---|
| Continuous latent rollout | Matched `L=0` 190/896 versus `L=4` 173/896 | Extra latent iterations changed outputs but lost to the answer-only control |
| Source-dropping memory / CLL | Packet M1 6/384 normal, 6 zero, 9 shuffled; CLL 16/631 normal, 4 zero, 12 shuffled, interventions 0/128 | Executable memory path, no causal retained-information advantage |
| Latent State Algebra (LSA) | Fit margin +1.04 points, OOD +0.09, equivalent-pair +0.52, interventions 0/576 | Rejected |
| Semantic-basis transport v2 | Train diagnostic 160/200 strict and 48/100 causal; held-out 6/200 strict and 0/100 causal | In-distribution carrier, multi-axis OOD collapse |
| Causal Prefix Readback (CPR) | Verified packet 161/10,752 = 1.497%, exactly equal shuffled; replay 193/10,752 | Rejected; packet not causally read |
| Native Residual Relay (NRR) | 0/500 held-out strict causal, 2/200 related diagnostic | Rejected/blocked |
| Finite-Query Residual Basis (FQRB) | Combined held-out only 7/4/16 correct of 2,500; zero strict groups; zero/shuffle often recreate answers | Closed negative |
| Source-deleted residual packet C1/C2 | Custody/reproducibility closure, no valid positive score | Closed |
| Post-commit packet transport v2 | `NO-GO` | Rejected |
| Post-commit packet transport v3 | Exact symbolic transport mechanics pass | Protocol result only; no learned native reasoning |
| Post-commit interface falsifier v1 | Static state 83,521/83,521 versus motor 4,913/83,521 | Exact evaluator can separate reusable state from a finite motor; no learned packet |
| Causal KV anchors | Exact token/KV transport substrate | Engineering control only; no semantic state claim |
| Verbalizable Recurrent Workspace (VRW/VRWM) | R3 43/400 default and 0/50 paraphrase; r4 state 32/400 default and 2/400 semantic, scratch 120/400 and 21/400; r5 semantic 17/400 | Narrow syntax/state gains, poor semantic transport; rejected |

### 5.10 Conflict localization and curriculum allocation

| Attempt | Result | Decision / lesson | Evidence |
|---|---|---|---|
| Conflict-Driven Residual Localization (CDRL) CPU cores | Correctly strips residual-neutral distractors and preserves free-word negative controls | Valid mechanics; not a reasoning primitive | [R12_CONFLICT_DRIVEN_RESIDUAL_LOCALIZATION.md](R12_CONFLICT_DRIVEN_RESIDUAL_LOCALIZATION.md) |
| CDRL neural curriculum | Core minus full **-77.59 points**; core minus random -2.05; core minus hard -77.83 | Closed negative; full/hard curricula dominate | [R12_CDRL_NEURAL_OPTIMIZATION_RESULT.md](R12_CDRL_NEURAL_OPTIMIZATION_RESULT.md) |

### 5.11 VAMT bounded-machine line

The Vocabulary-Aligned Microcode Transducer (VAMT) tried to turn the
compiler/executor/serializer decomposition into a bounded discrete machine.

| Version | Result | Decision |
|---|---|---|
| VAMT v1 | Independent theory/source review found contract and equivalence defects | Rejected |
| VAMT v2 | 15 tests pass, but they do not execute the declared full program machine | Theory `NO-GO`; CPU restricted to counterexample work; no neural |
| VAMT v3 | 152 programs, 20,672 executor cycles, 2,584 serializer cycles, all 400 executor and 40 serializer contexts; 187,332 added params; total 125,268,996; 246 state bits; 20,402,304 nominal MACs | Bounded mechanics `GO`; theory, novelty, neural fit, H100, and reasoning `NO-GO` |

VAMT v3 is a correct finite external-symbolic machine, but its programs are
host-constructed, its resource vector omits important costs, and its fixed
digit permutation is equivalent to a known pointer/Mealy controller. It is a
useful executable specification, not the missing reasoning primitive.

Evidence:

- [R12_VAMT_V2_REVIEW_RESULT.md](R12_VAMT_V2_REVIEW_RESULT.md)
- [R12_VAMT_V3_BOUNDED_MACHINE_THEORY.md](R12_VAMT_V3_BOUNDED_MACHINE_THEORY.md)
- [R12_VAMT_V3_REVIEW_RESULT.md](R12_VAMT_V3_REVIEW_RESULT.md)

### 5.12 Cross-domain fault correction and relation-complete transport

Three analogies were tested:

1. **Triadic replication:** Recovers 12/12 independent single-lane bit flips,
   but fails 4/4 common-mode semantic errors. This is a repetition code after
   semantic selection; it cannot repair three copies of the same wrong plan.
2. **Pure reversible transport:** 6,000 comparisons over `F_5` show zero
   contraction. A bijection preserves a state error unless a decoder,
   invariant, or extra provenance is added.
3. **Global relation syndromes:** The original no-go was wrong because it
   checked relations only at the identity. Global `S_3` relations uniquely
   repair the missing edge.

The repaired finite theorem is real:

- 76 involutions on six labels;
- 120 globally relation-valid transitive `S_3` actions;
- one erased edge has a unique globally valid completion;
- four target-specific transition anchors identify the canonical labeled
  `S_3` action.

However, the uniform `S_m` analysis closes the neural lane:

- up to conjugacy, a transitive action on `m!` states is already the regular
  action, so zero transition anchors are needed if semantic labels do not
  matter;
- on a fixed labeled carrier there are `(m! - 1)!` gauge-equivalent regular
  tables;
- identifying the semantic labels requires `Theta(m!)` anchors/queries;
- a favorable permutation-coordinate recurrence uses only `O(m log m)` state
  bits and a shared swap rule;
- the apparent gain survives only against a deliberately weak untied atlas;
- the current ledger omits semantic label alignment, selected anchor indices,
  relation-oracle generation, presentation semantics, and factorial relation
  enforcement.

**Decision:** Preserve the `S_3` artifact as a finite identifiability
certificate or possible regularizer. Do not allocate a neural fit or H100 run.

Evidence:

- [R12_CROSS_DOMAIN_FAULT_CHANNEL_REVIEW_RESULT.md](R12_CROSS_DOMAIN_FAULT_CHANNEL_REVIEW_RESULT.md)
- [R12_RELATION_COMPLETE_TRANSPORT_HYPOTHESIS.md](R12_RELATION_COMPLETE_TRANSPORT_HYPOTHESIS.md)
- [R12_RELATION_COMPLETE_TRANSPORT_REVIEW_RESULT.md](R12_RELATION_COMPLETE_TRANSPORT_REVIEW_RESULT.md)
- `pipeline/relation_complete_transport_falsifier.py`

### 5.13 Other prepared but unpromoted mechanism lanes

| Lane | State | Reason it did not advance |
|---|---|---|
| Orthogonal Carry Serializer Curriculum (OCSC) | Source mechanics were reviewed | Qualification `NO-GO`; no neural fit |
| EOS-suppressed / DWS single completion | Local tests pass | Source contract relied on self-attested fields and false reopen conditions |
| Carry motor recovery | Repaired source attempted | Confirmation/custody boundary still invalid |
| SCERT | Preregistered structured residual execution | No admitted positive fit |
| WGRQ | 18,432 episodes and 589,824 answers acquired for the CPU object | Gate-vacuity/equivalence audit closed the lane before 60 planned neural fits |
| PCRT/PCFT | CPU/theory falsifiers | Exact collapse/equivalence or adversarial audit blocked neural advancement |
| ACW/packet-memory Track S | Discrete workspace mechanics explored | No autonomous result that beats matched recurrence/host controls |
| Addressed Categorical Workspace (ACW) fit | 4,096 histories, 57,344 labels, 3,400 updates; loss 2.806958 to 2.828255 | Execution/custody pipeline ran, but no scored development matrix or capability-ranked checkpoint |

---

## 6. Mathematical And Theoretical Work Completed

R12 changed the project from architecture-first experimentation to
theory-first falsification. Many appealing mechanisms were rejected before GPU
use because they reduce to an existing transducer or hide the answer in an
unaccounted resource.

### 6.1 Identifiability and information no-gos

| Result family | What it closes |
|---|---|
| Closed late query | A late query cannot recover source information that no accessible state retained |
| Closed deliberation | Target-independent extra recurrence cannot create missing source mutual information |
| Active verifier query | Verification helps only when the query returns new information; a verifier cannot certify an unknown state for free |
| Active witness allocation | Adaptive supervision does not remove the information needed to distinguish futures |
| Hidden-coordinate identifiability | Ordinary examples cannot identify an arbitrary hidden coordinate system without interventions/anchors |
| MDL identifiability | The shortest consistent program need not be the intended extrapolation law |
| Self-authenticating state | A state cannot prove its own semantic correctness without an external binding |
| Secret-shared bootstrap | Splitting a hidden state across shares does not create information or semantic identity |
| Finite-state versus motor | A finite answer-specific motor bundle can fit consumers without becoming reusable reasoning state |

Primary documents include:

- `R12_CLOSED_LATE_QUERY_NO_GO.md`
- `R12_CLOSED_DELIBERATION_NO_GO.md`
- `R12_ACTIVE_VERIFIER_QUERY_NO_GO.md`
- `R12_ACTIVE_WITNESS_ALLOCATION_NO_GO.md`
- `R12_HIDDEN_COORDINATE_IDENTIFIABILITY_NO_GO.md`
- `R12_MDL_IDENTIFIABILITY_NO_GO.md`
- `R12_SELF_AUTHENTICATING_STATE_NO_GO.md`
- `R12_SECRET_SHARED_CAUSAL_BOOTSTRAP_NO_GO.md`
- `R12_FINITE_STATE_VS_MOTOR_NO_GO.md`

### 6.2 Equivalence and resource no-gos

| Result family | What it closes |
|---|---|
| Compiler prior | Recurrence is not a fair invention if compilation is externally supplied |
| Dynamic frontier | Exact memory scales with the number of future-distinguishable active states |
| Query kernel / query distribution | Average-case compression still pays for the mass of queries that must be answered |
| Structured residual resource law | A compact latent description does not eliminate certification, compiler, or decoder cost |
| Coherent action | Exact globally consistent actions are ordinary equivariant transition systems in another representation |
| Holonomy state | Closed-loop curvature does not by itself identify a reusable causal state |
| Axiomatic presentation | Relations reduce labels only when semantic alignment and relation enforcement are counted |
| Commutator factorization | Pairwise commutation does not determine arbitrary operator semantics |
| Polynomial-coded action | Error-correcting interpolation works as an external algorithm but does not supply native semantics |
| Noise-stable action | Robust encoding plus noiseless repair is equivalent to an ordinary logical action with extra resources |
| Fork core / forked transport | Exact future-equivalent states collapse to the task's causal quotient/transducer |
| Minimax causal broadcast | Fitting a finite consumer bundle identifies only the bundle's observable quotient, not a universal workspace |

Primary documents include:

- `R12_COMPILER_PRIOR_NO_GO.md`
- `R12_DYNAMIC_FRONTIER_NO_GO.md`
- `R12_QUERY_KERNEL_FACTORIZATION_NO_GO.md`
- `R12_QUERY_DISTRIBUTIONAL_CONTEXT_NO_GO.md`
- `R12_STRUCTURED_RESIDUAL_RESOURCE_LAW.md`
- `R12_COHERENT_ACTION_THEORY.md`
- `R12_HOLONOMY_STATE_NO_GO.md`
- `R12_AXIOMATIC_PRESENTATION_NO_GO.md`
- `R12_COMMUTATOR_FACTORIZATION_NO_GO.md`
- `R12_POLYNOMIAL_CODED_ACTION_NO_GO.md`
- `R12_NOISE_STABLE_ACTION_NO_GO.md`
- `R12_FORK_CORE_THEORY.md`
- `R12_FORKED_STATE_TRANSPORT_PREREG.md`
- `R12_MINIMAX_CAUSAL_BROADCAST_SUBSPACE_NO_GO.md`

### 6.3 Controls and preregistration machinery

The project also built a large control surface so future positive results are
harder to fake accidentally:

- WGRQ and gate-vacuity audits;
- PCFT/PCRT exact-collapse and adversarial audits;
- SCERT execution contract;
- task-quotient and mixed-difference preregistrations;
- separating-query-basis and shared-transition-circuit theories;
- self-canonicalizing epoch retirement;
- canonical residual naming control;
- local reversible rule control;
- post-commit interface and packet falsifiers;
- explicit source, data, checkpoint, seed, and score custody checks.

These are research infrastructure, not capability results.

The full preregistration inventory includes the following ideas. Their presence
means a hypothesis was specified or audited, not that it worked:

| Family | Artifacts / current boundary |
|---|---|
| Query and quotient mechanisms | `R12_WGRQ_CPU_PREREG.md`, `R12_TASK_QUOTIENT_LIFTING_PREREG.md`, `R12_SEPARATING_QUERY_BASIS_THEORY.md`; CPU/theory controls only |
| Residual transition mechanisms | `R12_MIXED_DIFFERENCE_RESIDUAL_TRANSDUCER_PREREG.md`, `R12_SHARED_TRANSITION_CIRCUIT_THEORY.md`, `R12_PRESENTATION_CLOSED_RESIDUAL_TRANSPORT_PREREG.md`; no accepted neural result |
| Counterfactual state mechanisms | `R12_COUNTERFACTUAL_CONJUGATE_COMMIT_HYPOTHESIS.md`, `R12_FACTORIZED_COUNTERFACTUAL_RESIDUAL_CYCLE_PREREG.md`, `R12_COUNTERFACTUAL_CURSOR_ACTION_CPU_PREREG.md`; tested cursor arm was negative, broader claims remain unestablished |
| Packet and lattice mechanisms | `R12_PACKET_ON_LATTICE_CARRY_CELL_PREREG.md`, `R12_CONTRACTIVE_PACKET_RECURRENCE_PREREG.md`, post-commit packet v2/v3; exact mechanics exist, no learned autonomous packet |
| Witness and retirement mechanisms | `R12_LAST_RESET_WITNESS_ATTENTION_PREREG.md`, `R12_SELF_CANONICALIZING_EPOCH_RETIREMENT_THEORY.md`; theory/preregistration only |
| Categorical workspace mechanisms | `R12_ADDRESSED_CATEGORICAL_WORKSPACE_PREREG.md`, `R12_OPERATOR_BALANCED_COMMIT_BISIMULATION_PREREG.md`; no accepted broad capability result |
| Carry and serializer mechanisms | `R12_CAUSAL_CARRY_MOTOR_PREREG.md`, carry recovery, OCSC, DWS/EOS; source/qualification or autonomous gates failed |
| Proof/format mechanisms | `R12_FORMAT_CONJUGACY_AND_SSC.md`, typed-controller internalization, SCERT; useful decomposition and controls, no accepted native reasoner |

---

## 7. Discoveries That Survive All Current Evidence

These are the durable findings a new theory should preserve and explain.

### 7.1 Local execution is substantially stronger than autonomous composition

SSC's 44.92% versus whole-problem 3.52%, DRS's 497/500 first states versus
275/500 finals, and oracle-compiled 28/34 transitions establish this beyond a
single benchmark.

### 7.2 A real late-layer digit workspace can be trained

Post-DRS residual swaps produce about +31 delta log-odds toward the source
digit at four late layers. This is causal, not merely linearly decodable.

### 7.3 Readable state is not sufficient

The wide digit motor reaches near-perfect fit-board serialization but changes
autonomous full-loop accuracy by only 0.8 points. A state needs an actuator,
carry logic, recurrent consumer, compiler binding, and halt policy.

### 7.4 Natural-language compilation is the largest measured bottleneck

Fresh compilation is 0/6; cursor probes are near control and omit multiply and
remainder; binding-first architectures can improve factor scores but do not
produce exact full programs at hard transfer.

### 7.5 Termination is a real independent bottleneck

Many raw and source-scheduled generations hit the cap or continue after a
correct intermediate answer. Halt-first decoding raises exactness substantially
without changing weights, while typed SFT can learn DONE and then lose it.

### 7.6 Most current errors are common-mode semantic errors

Replication cannot fix three learners choosing the same wrong operation. R9c
found 88.83% of wrong operation decisions were agreed-wrong common-mode errors.
Any error-correction theory must act before or during semantic selection, not
only replicate a selected action.

### 7.7 Raw scaling under the current recipe plateaued

The 120k, 168.75k, and 300k boards and fixed direct interaction are essentially
flat. Continuing the same data/architecture is not the leading reasoning
hypothesis.

### 7.8 Template-local SFT gains are cheap and misleading

Several curricula improve one narrow surface while degrading code, arithmetic,
DONE, source-deleted transfer, or direct interaction. A new method must report
full preservation and transfer, not only its trained grammar.

---

## 8. Hypotheses That Are Closed Or Strongly Disfavored

Do not reopen these without a materially different causal prediction:

1. **More raw pretraining alone will make the current model reason.** The 300k
   plateau rejects this as the leading plan.
2. **Longer visible chain of thought is sufficient.** The model often imitates
   trace shape while computing incorrectly.
3. **A hidden digit coordinate is sufficient.** DRS has one; autonomous cycles
   still fail.
4. **A linear probe demonstrates a workspace.** Readout without causal use is
   insufficient.
5. **External scheduling demonstrates native reasoning.** SSC is a ceiling,
   not a model-owned controller.
6. **Host arithmetic plus learned operation heads is native reasoning.** SCEB
   is a control because the host register bus executes the math.
7. **Replication fixes reasoning errors.** It cannot fix common-mode wrong
   semantic selection.
8. **Pure reversibility repairs wrong state.** A bijection preserves the error
   without a decoder or extra information.
9. **Relation consistency removes semantic alignment cost.** Uniform `S_m`
   analysis retains factorial labeled-state identification and loses to a
   favorable coordinate recurrence.
10. **A bounded symbolic machine is itself a new neural reasoning primitive.**
    VAMT v3 is correct mechanics but equivalent to known external control.
11. **Conflict-core-only training is more efficient.** CDRL lost by about 78
    points to full/hard curricula.
12. **A finite consumer motor proves universal state.** It can fit an
    answer-specific observable quotient without reusable computation.

---

## 9. Current Frontier For New Theories

A useful new theory should target at least one of these open interfaces while
remaining honest about the others.

### 9.1 Language-to-program binding

Find a mechanism that converts paraphrased natural language into an exact,
compositional operation program. It must distinguish order twins, survive
renamed roles, and operate on unseen values and lengths. Supplying the program
from the host does not test this.

### 9.2 Self-updating state with an internal actuator

Use the demonstrated DRS late-layer signal, but require the model to transform
it into the next state without a host ALU. A successful intervention must
improve held-out autonomous cycles and lose that gain under shuffled or
complement ablations.

### 9.3 Joint controller-executor learning without gradient conflict

Typed v1 learned DONE but not arithmetic; typed v2 preserved atomic accuracy
while destroying DONE. A new mechanism may need physically separated losses,
timescales, parameter subspaces, or update phases, followed by a causally
tested integration interface.

### 9.4 Common-mode semantic error correction

Error correction must detect a wrong operation choice before all lanes agree
on it. Candidate inspirations include disagreement over independently grounded
views, invariant violations whose checks do not require the answer, or a
learned uncertainty object that is causally tied to future-distinguishable
states. The verifier cannot simply solve or repair the task externally.

### 9.5 Termination as a learned control primitive

The model needs a completion condition tied to its internal state, not just a
text token imitated from SFT. A useful theory should predict when the halt
state becomes causally available and how it remains stable under output
recoding and paraphrase.

### 9.6 Architecture changes under the 150M cap

Architecture changes are explicitly allowed. Up to about 24.9M parameters may
be added if the total remains below 150M. Parameter count alone is not the
constraint. A proposal must state why its mechanism should outperform a
matched recurrent or static control and which current failure it changes.

---

## 10. Template For A New Theory

Use this checklist when bringing a new idea. A theory that cannot fill these
fields is not ready for a neural experiment.

```text
Theory name:

1. Target failure
   Which measured Shohin failure does this address?

2. Capability object
   What exact behavior should the model gain?

3. State and update law
   What is retained, how is it updated, and who performs the update?

4. Distinguishing causal prediction
   What intervention succeeds only if this mechanism is real?

5. Information source
   Where does every answer-relevant bit enter the system?

6. Native boundary
   What does the host do, and what is it forbidden to do?

7. Equivalence dossier
   Does this collapse to SFT, recurrence, retrieval, a finite atlas, a
   hard-coded algorithm, a verifier, or host execution?

8. Favorable matched controls
   What strongest ordinary method receives the same parameters, data, compute,
   state bits, and inference depth?

9. Resource vector
   Parameters, state bits, source bytes, target bits, examples, oracle work,
   training FLOPs, inference FLOPs, sequential depth, external work.

10. Finite CPU falsifier
    What exact small board can disprove the mechanism before GPU use?

11. Held-out generalization
    Which paraphrases, values, widths, lengths, recodings, and order twins are
    hidden before training?

12. Advancement gate
    Exact thresholds and failure conditions frozen before scores are read.

13. Direct interaction plan
    Which full transcripts will reveal loops, copying, invalid state, or host
    dependence that an aggregate score could hide?

14. Preservation gate
    Which broad language, math, and code capabilities may not regress?
```

### 10.1 Fast rejection questions

Before investing in implementation, ask:

- Does the host already know the operation, state, carry, or answer?
- Is the proposed state just a re-encoding of a finite-state transducer?
- Would a favorable ordinary recurrence receive the same structural prior?
- Are semantic labels or coordinate alignment supplied for free?
- Does the method improve only teacher-forced fit or also autonomous rollout?
- Can zero/shuffled state reproduce the same answer?
- Is the result invariant to output recoding, unseen language, and unseen
  length?
- Does the verifier add information or merely check what the model already
  knows?
- Are all training/oracle/resource costs counted?
- What concrete outcome would make us abandon the idea?

---

## 11. Artifact Map

### Base model and capability

- [AGENT_RUNBOOK.md](AGENT_RUNBOOK.md)
- [TRAINING_METRICS.md](TRAINING_METRICS.md)
- [RAW300K_INTERACTION_RESULT.md](RAW300K_INTERACTION_RESULT.md)
- [RAW300K_FREEFORM_INTERACTION_RESULT.md](RAW300K_FREEFORM_INTERACTION_RESULT.md)
- [REASONING_FRONTIER.md](REASONING_FRONTIER.md)
- [REASONING_ATTACK_PLAN.md](REASONING_ATTACK_PLAN.md)

### Source scheduling and controllers

- [R12_SOURCE_SCHEDULED_REASONING_CONFIRMATION_RESULT.md](R12_SOURCE_SCHEDULED_REASONING_CONFIRMATION_RESULT.md)
- [R12_SOURCE_SCHEDULED_FAILURE_TAXONOMY.md](R12_SOURCE_SCHEDULED_FAILURE_TAXONOMY.md)
- [R12_SSC_HALT_FIRST_LIVE_RESULT.md](R12_SSC_HALT_FIRST_LIVE_RESULT.md)
- [R12_TYPED_CONTROLLER_V1_RESULT.md](R12_TYPED_CONTROLLER_V1_RESULT.md)
- [R12_TYPED_CONTROLLER_V2_RESULT.md](R12_TYPED_CONTROLLER_V2_RESULT.md)
- [R12_SCEB_RESULTS.md](R12_SCEB_RESULTS.md)

### DRS, recurrence, workspace, and motors

- [R12_RECURRENT_CONTROLS_RESULT.md](R12_RECURRENT_CONTROLS_RESULT.md)
- [R12_DRS_WORKSPACE_PROBE_POST_RESULT.md](R12_DRS_WORKSPACE_PROBE_POST_RESULT.md)
- [R12_DRS_CAUSAL_CYCLE_RESULT.md](R12_DRS_CAUSAL_CYCLE_RESULT.md)
- [R12_CAUSAL_RESULT_DIGIT_MOTOR_RESULT.md](R12_CAUSAL_RESULT_DIGIT_MOTOR_RESULT.md)
- [R12_CAUSAL_CARRY_MOTOR_RECOVERY_PREREG.md](R12_CAUSAL_CARRY_MOTOR_RECOVERY_PREREG.md)

### Compiler, cursor, and Jacobian diagnostics

- [R12_OPERATION_SELECTION_LIKELIHOOD_RESULT.md](R12_OPERATION_SELECTION_LIKELIHOOD_RESULT.md)
- [R12_OPERATION_CURSOR_RESULT.md](R12_OPERATION_CURSOR_RESULT.md)
- [R12_COUNTERFACTUAL_CURSOR_ACTION_NEURAL_RESULT.md](R12_COUNTERFACTUAL_CURSOR_ACTION_NEURAL_RESULT.md)
- [R12_CURSOR_READOUT_ACTUATION_RESULT.md](R12_CURSOR_READOUT_ACTUATION_RESULT.md)
- [R12_CURSOR_TOKEN_TAPE_RESULT.md](R12_CURSOR_TOKEN_TAPE_RESULT.md)
- [R12_JACOBIAN_WORKSPACE_LONGITUDINAL_RESULT.md](R12_JACOBIAN_WORKSPACE_LONGITUDINAL_RESULT.md)

### Packets and state transport

- [R12_SOURCE_DELETED_RESIDUAL_PACKET_PREREG.md](R12_SOURCE_DELETED_RESIDUAL_PACKET_PREREG.md)
- [R12_SOURCE_DELETED_RESIDUAL_PACKET_C1_CLOSURE.md](R12_SOURCE_DELETED_RESIDUAL_PACKET_C1_CLOSURE.md)
- [R12_RESIDUAL_PACKET_C2_REPRO_AUDIT_RESULT.md](R12_RESIDUAL_PACKET_C2_REPRO_AUDIT_RESULT.md)
- [R12_POST_COMMIT_PACKET_TRANSPORT_V2_RESULT.md](R12_POST_COMMIT_PACKET_TRANSPORT_V2_RESULT.md)
- [R12_POST_COMMIT_PACKET_TRANSPORT_V3_RESULT.md](R12_POST_COMMIT_PACKET_TRANSPORT_V3_RESULT.md)

### Theory and current frontier

- [R12_REASONING_INVENTION_CHARTER.md](R12_REASONING_INVENTION_CHARTER.md)
- [R12_VAMT_V3_REVIEW_RESULT.md](R12_VAMT_V3_REVIEW_RESULT.md)
- [R12_CROSS_DOMAIN_FAULT_CHANNEL_REVIEW_RESULT.md](R12_CROSS_DOMAIN_FAULT_CHANNEL_REVIEW_RESULT.md)
- [R12_RELATION_COMPLETE_TRANSPORT_HYPOTHESIS.md](R12_RELATION_COMPLETE_TRANSPORT_HYPOTHESIS.md)
- [R12_RELATION_COMPLETE_TRANSPORT_REVIEW_RESULT.md](R12_RELATION_COMPLETE_TRANSPORT_REVIEW_RESULT.md)
- [R12_CDRL_NEURAL_OPTIMIZATION_RESULT.md](R12_CDRL_NEURAL_OPTIMIZATION_RESULT.md)
- [R12_MINIMAX_CAUSAL_BROADCAST_SUBSPACE_NO_GO.md](R12_MINIMAX_CAUSAL_BROADCAST_SUBSPACE_NO_GO.md)

---

## 12. Mandatory Maintenance Protocol

This file is part of experiment completion, not optional documentation.

Every future experiment must update this master ledger before it is considered
closed. The responsible agent must:

1. Update the `Last updated` timestamp.
2. Add the experiment to the appropriate ledger table.
3. Record the exact checkpoint, data, code, seed, job, and result-artifact
   identity in the dedicated result file or runbook.
4. State the score and denominator, not only a percentage.
5. Label the result `GO`, `NO-GO`, `REJECTED`, `CONTROL`, `DIAGNOSTIC`, or
   `UNRESOLVED` at the exact claim boundary.
6. State what the host did at inference.
7. Add any surviving discovery to Section 7.
8. Add any falsified hypothesis to Section 8.
9. Update the current frontier in Section 9 when the result changes the next
   highest-leverage question.
10. Add or update the artifact link in Section 11.
11. Append a one-line change-log entry below.
12. Run `git diff --check`, commit the master and result documentation, and
    push safe code/docs without secrets.

Agents taking custody must read this file after `AGENT_RUNBOOK.md` and before
proposing or launching a reasoning experiment.

### Change log

| Date | Change |
|---|---|
| 2026-07-18 | Created the master native-reasoning ledger from the immutable 300k flagship, public boards, direct interactions, SFT history, DRS/controller/workspace experiments, R9-R12 mechanism studies, and the latest VAMT/relation reviews. |
