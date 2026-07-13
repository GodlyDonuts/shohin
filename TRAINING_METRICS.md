# Shohin Training Metrics Ledger

This is the auditable metrics companion to [AGENT_RUNBOOK.md](AGENT_RUNBOOK.md).
It records confirmed measurements, their source artifacts, and the distinction between
training progress, corpus capacity, and capability. It is not a substitute for the
runbook's operational instructions.

**Last refreshed:** 2026-07-13 11:27 EDT
**Flagship source of truth:** Newton Slurm job `685084`,
`/lustre/fs1/home/sa305415/shohin/train/flagship_out/log_r0.jsonl`  
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
| Active job / node | `685084` on `evc22`, one H100, 4 CPU cores. Dependency-held natural successor `686732` requests two H100s/four CPUs only after `685084` exits; it does not share the live writer. |
| Start / scheduled end | 2026-07-11 03:51:47 / 2026-07-14 03:51:47 EDT (Slurm allocation; not a completion guarantee) |
| Microbatch / accumulation | `BS=32`, `ACC=8` |
| Global tokens per update | `32 * 8 * 2,048 = 524,288` |
| Absolute training target | 300,000 steps |
| Resume point | `ckpt_0141500.pt` to step 141,501 |
| Latest checkpoint milestone | **200,000** steps = **104,857,600,000 nominal update tokens** |
| Last observed live step | 200,360 = 105,046,343,680 nominal update tokens |
| Last observed throughput | 154,294 tokens/s, approximately 13.33B nominal tokens/day at that sustained rate |
| Latest loss / gradient norm | step 200,360: loss 1.6012; gnorm 0.09; LR 0.0050 |
| Direct H100 telemetry | 05:07 EDT: 100% compute utilization, 59% memory-controller utilization, 63,767 / 81,559 MiB VRAM, 280.12 / 310 W, 63 C. Low unused VRAM is deliberate: BS64 was OOM; BS32 is the largest verified single-GPU microbatch. |
| Post-190k health | One isolated guard skip at step 193,437 (gnorm 2.21 versus 0.13 EMA) recovered at the next logged step. Surrounding recent steps remain in the normal range; no divergence or persistent skip exists. |
| Two-H100 handoff revalidation | `686734` on evc37, 320 bounded updates from `best_step180000.pt`, `world=2`, `BS=32`, `ACC=4`, fresh optimizer, stream generation 1. Exit 0 with no CUDA/NCCL/DDP error; compile-free late windows 291.7-293.9k tok/s (about 1.90x the live one-H100 rate). One terminal gnorm guard skip at step 180,319 was not followed by an in-canary recovery step, so this is throughput/transport evidence only. |

The current live flagship's data stream is frozen for the life of this job. Do not add
new shards, alter weights, or apply an experimental runtime optimization to `685084`.

At the natural handoff only, `686732` will use `NG=2, BS=32, ACC=4`: the same
`2 * 32 * 4 * 2,048 = 524,288` global tokens/update, fresh optimizer rewarmup,
250-step checkpoints, and the audited distinct data-stream generation. It excludes
the CUDA-preflight failures `evc26,31,36,43,50`; it must log `world=2` and pass a
real CUDA/NCCL health check before its throughput is counted.

## Overnight Comparison Snapshot: 2026-07-13 11:27 EDT

This is the explicit before-sleep reference point for the next custody check.

| Surface | Verified state |
|---|---|
| Flagship | `685084` is `RUNNING` on `evc22`, 2d 07h elapsed, one H100, `BS=32 ACC=8`, 4 CPUs. |
| Training progress | Step **200,360**; **105,046,343,680** nominal update tokens; **154,294 tok/s**. Recent loss/gnorm remains finite and in band. |
| Stability | No new guard skip after the recovered step-193,437 outlier. No divergence, data-loader, CUDA, or checkpoint error observed. |
| Durable recovery | Hash-matched 200k full checkpoint on Newton/local: md5 `510d57df578447986b40e20029511b9d`. Next mandatory promotion is 210k. |
| Frontier gate | LSA is rejected. CPR has three matched training arms complete; verified and shuffled-label evaluator reports are complete, while replay `687228` remains in held-out shuffled-source evaluation. Comparator is not submitted until all three reports exist. |
| Corpus expansion | DCLM `686529` completed 25,000,001,792 tokens / 250 shards but is unadmitted pending a fresh scan. OpenMath PT completed 5,000,000,144 tokens / 50 shards and is likewise future-handoff-only. Stokes FineWeb r2 `738030` is live at 52 partial 100M-token shards, about 5.2B. None is in the active stream. |
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

## Checkpoint and Disaster-Recovery Inventory

| Milestone | Numbered checkpoint at milestone | Newton durable copy | Local full checkpoint | MD5 | State |
|---|---|---|---|---|---|
| 170k | `ckpt_0170000.pt` | `best_step170000.pt` | `train/flagship_out/ckpt_0170000.pt` | `7ad139b6b9b537a5a3e65978f8296419` | Verified Newton + local |
| 180k | Observed and hashed, then reaped by trainer retention | `best_step180000.pt` | `train/flagship_out/ckpt_0180000.pt` | `a592a8bd46163eb1427fe64460be0c6a` | Two durable verified copies |
| 190k | `ckpt_0190000.pt` | `best_step190000.pt` | `train/flagship_out/ckpt_0190000.pt` | `3e195aaf44a14259797c49d7f80d9c7f` | Verified Newton + local |
| 200k | `ckpt_0200000.pt` | `best_step200000.pt` | `train/flagship_out/ckpt_0200000.pt` | `510d57df578447986b40e20029511b9d` | Verified Newton + local |

All rows above are full optimizer checkpoints, not model-only exports. The next local DR
target is 210k, or the newest clean checkpoint before any natural handoff.

## Current Active Pretraining Corpus

These are the exact current `SHARDS` inputs for `685084`, taken from their manifests.
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
