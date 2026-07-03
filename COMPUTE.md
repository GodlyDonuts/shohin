# Compute — budget, options, and the Newton reality

## FLOP budget

Training compute ≈ **6 · N · D**. For N = 1.3e8 params, D = 2e11 tokens → **≈ 1.6e20 FLOPs**, inside a
2–3e20 budget. On **8× H100 (bf16)** the ~200B-token pretrain is roughly **one overnight run**. Full
campaign ≈ **3–6 × 24h sessions** on that node:

| phase | share of budget |
|---|---:|
| teacher-logit precompute (one teacher pass + top-k store) | ~30% |
| main pretrain + decay (the SmolLM2-beating run) | ~28% |
| data-mix ablations (the actual science risk) | ~24% |
| SFT / DPO / eval | ~18% |

The single overnight main run is the *smallest* cost; budget discipline is about **limiting ablation
sprawl**, not the flagship train.

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
- The teacher-logit precompute (running Qwen3-1.7B / Llama-3.2-1B once over the corpus, storing top-k
  logits) is a meaningful one-time cost — plan storage for the top-k logit shards.
