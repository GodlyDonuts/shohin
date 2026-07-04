# Compute — budget, options, and the Newton reality

## FLOP budget

Training compute ≈ **6 · N · D**. For N = 1.3e8 params, D = 1–2e11 tokens → **≈ 1e20 FLOPs**, inside a
2–3e20 budget. On **8× H100 (bf16)** the ~100–200B-token pretrain is roughly **one overnight run**. Full
campaign ≈ **3–6 × 24h sessions** on that node:

| phase | share of budget |
|---|---:|
| **teacher trace generation + rejection sampling** (run the 1–2B teacher over problem banks, keep correct + short) | ~30% |
| main pretrain + reasoning mid-train (the MobileLLM-R1-beating run) | ~28% |
| data-mix / short-vs-long-CoT ablations (the actual science risk) | ~24% |
| SFT / RL (optional) / eval | ~18% |

The single overnight main run is the *smallest* cost; budget discipline is about **limiting ablation
sprawl** and trace-generation over-runs, not the flagship train.

## The hardware reality (as of planning)

We do **not** currently have an 8× H100 node available. On **UCF Newton**, our account (`skattel` /
`arcc_pi_skattel`) is associated only with `normal`, `ood`, `preemptable`:

- `highgpu` (8× H100 80GB nodes evc101-104) — **exists and is up, but our account can't submit there.**
- `short` / `ucfit` (4× H200 NVL) — also not associated with our account.
- What we *can* use: `normal` / `preemptable` → **H100 PCIe, 2 per node** (+ V100s). Flaky-H100 gotcha on
  some nodes; `preemptable` jobs can be killed.

### Three paths

| path | what it means | cost/effort |
|---|---|---|
| **Ask the PI for `highgpu` access** | proper 8× H100 node — the clean answer | free; needs an email to ARCC / the PI; may take a day |
| **Use `normal` (2× H100 PCIe)** | ~¼ throughput → a 24h-on-8×H100 run ≈ ~4 days on 2× H100 (a 130M model is small, still feasible) | free, slower, preemptable risk |
| **Cloud rent 8× H100** (Lambda / RunPod / etc.) | exactly the 24h scenario | ~$500–700 per full run |

**Recommendation:** email the PI for `highgpu` first (free, clean) — worth doing early since it may take a
day. Meanwhile, **milestone 1 (eval harness + no-KD baseline) needs only modest compute** and can start on
`normal` (2× H100) or a small cloud spot. Cloud 8× H100 is the fast fallback for the full KD run.

## Notes

- FSDP2 + flash-attention + bf16 on 8× H100 is standard, well-trodden — throughput is *not* a research risk.
- **Teacher trace generation + rejection sampling** (running Qwen3-1.7B / DeepSeek-R1-Distill-1.5B over
  math/code/logic problem banks, keeping only correct + short traces) is the meaningful one-time cost — plan
  for teacher inference throughput and trace storage, *not* logit shards. RL rollouts (if we run the optional
  A/B) add generation cost on top.

---

## Stokes (available now — the data-build machine)

The 8×H100 run waits on a node (above), but **data can be built now** on UCF ARCC's **Stokes** cluster
(`sa305415@stokes.ist.ucf.edu`, PI group `arcc_pi_skattel`). Confirmed environment (2026-07):

| resource | reality |
|---|---|
| access | SSH; **key auth disabled server-side** → connection multiplexing (`ssh stokes`: authenticate once/session, nothing stored on disk) |
| internet egress | ✅ direct to Hugging Face (`HTTP/2 200`) — datasets pull without a proxy |
| scheduler | SLURM; large CPU partitions (`normal`, `preemptable`) for parallel tokenization |
| **storage** | `/lustre/fs1` Lustre; **personal quota = 1 TB** (hard 1010 GB). PI group dir `/lustre/fs1/groups/skattel` exists but is shared/near-full. **No quota increase available right now.** |
| Python | system 3.6.8 (too old) → miniforge env at `/lustre/fs1/home/sa305415/shohin/miniforge3` (env `data`, py3.11) |
| GPUs | none on login; GPU partition not yet located — irrelevant until the flagship run |

**Consequence for the plan:** 580B uint16 shards ≈ 1.16 TB **exceeds the 1 TB quota**, so we do **not**
materialize the web bulk on Lustre. Build now (fits easily): the tokenizer, the Reasoning-Gym procedural
corpus, SFT curation, decontamination, and held-out evals. Stream the commodity web pretrain later, straight
to the GPU node's local NVMe (the ≥4 TB scratch §9 calls for). See [pipeline/](pipeline/).
