# Shohin Training Metrics Ledger

This is the auditable metrics companion to [AGENT_RUNBOOK.md](AGENT_RUNBOOK.md).
It records confirmed measurements, their source artifacts, and the distinction between
training progress, corpus capacity, and capability. It is not a substitute for the
runbook's operational instructions.

**Last refreshed:** 2026-07-12 16:17 EDT  
**Flagship source of truth:** Newton Slurm job `685084`,
`/lustre/fs1/home/sa305415/shohin/train/flagship_out/log_r0.jsonl`  
**Checkpoint source of truth:** numbered checkpoint plus `best_step<step>.pt`, each
MD5-verified on Newton and locally before a local row is marked verified.

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
| Active job / node | `685084` on `evc22`, one H100, 4 CPU cores |
| Start / scheduled end | 2026-07-11 03:51:47 / 2026-07-14 03:51:47 EDT (Slurm allocation; not a completion guarantee) |
| Microbatch / accumulation | `BS=32`, `ACC=8` |
| Global tokens per update | `32 * 8 * 2,048 = 524,288` |
| Absolute training target | 300,000 steps |
| Resume point | `ckpt_0141500.pt` to step 141,501 |
| Latest checkpoint milestone | **180,000** steps = **94,371,840,000 nominal update tokens** |
| Last observed live step | 180,090 = 94,419,025,920 nominal update tokens |
| Last observed throughput | 154,302 tokens/s, approximately 13.33B nominal tokens/day at that sustained rate |
| 180k loss / gradient norm | loss 1.6526; gnorm 0.103; LR 0.0050 |
| Post-180k health | One gnorm guard skip at step 180,030 (1.04 versus EMA 0.12), followed immediately by normal 0.09-0.13 gnorm steps. This is a recovered guard event, not a divergence. |

The current live flagship's data stream is frozen for the life of this job. Do not add
new shards, alter weights, or apply an experimental runtime optimization to `685084`.

## Checkpoint and Disaster-Recovery Inventory

| Milestone | Newton numbered checkpoint | Newton promoted copy | Local full checkpoint | MD5 | State |
|---|---|---|---|---|---|
| 170k | `ckpt_0170000.pt` | `best_step170000.pt` | `train/flagship_out/ckpt_0170000.pt` | `7ad139b6b9b537a5a3e65978f8296419` | Verified Newton + local |
| 180k | `ckpt_0180000.pt` | `best_step180000.pt` | `train/flagship_out/ckpt_0180000.pt` | `a592a8bd46163eb1427fe64460be0c6a` | Verified Newton + local |

All rows above are full optimizer checkpoints, not model-only exports. The next local DR
target is 190k, or the newest clean checkpoint before any natural handoff.

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
| 25B DCLM / FineWeb replacements | No final manifests or scan approvals at this refresh. | Not admitted; do not use partial output in a future relaunch. |

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
- Fixed raw-170k monitor results: WikiText-103 test NLL **3.9648849**, PPL **52.7142** over
  301,056 targets; CodeContests test NLL **1.3537146**, PPL **3.8718** over 145,408 targets.
  They are trend monitors only. The code monitor is not source-disjointness proof, so
  HumanEval/MBPP and execution-based held-out tests remain decisive.

The serialized raw-180k -> V8 SFT -> board/interview decision chain is the next capability
measurement. V8 cannot be promoted on loss, formatting, generator holdouts, or a single
benchmark movement alone.

## Update Protocol

At each 10k checkpoint milestone:

1. Confirm the exact numbered checkpoint exists on Newton, copy it to the corresponding
   `best_step<step>.pt`, and record the remote MD5.
2. Transfer to `train/flagship_out/ckpt_<step>.pt.part` (or a resumable equivalent). Verify
   the local MD5 against Newton before atomically renaming it without `.part`.
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
