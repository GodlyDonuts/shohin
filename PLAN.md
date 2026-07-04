# The Shohin build plan (stage by stage)

> ⤴ **Superseded by [MASTER_PLAN.md](MASTER_PLAN.md)** (the plan of record). Kept as background; where numbers
> differ (token budget ~100–200B here vs ~580B there, targets, optimizer), **the master plan wins.**

Recipe for a ~130M dense **reasoning specialist** that beats MobileLLM-R1-140M on verifiable reasoning
(math/code/logic). Confidence is HIGH for short-CoT-distillation + rejection-sampling; LOW / against-trend for
RL-at-130M and looped-reasoning-at-130M (flagged inline).

---

## 0. Framework — PyTorch (unchanged)

The 2026 edge is data + distillation infra — short-CoT trace generation, rejection sampling, bf16 tensor-core
throughput, 8-GPU data parallelism — none deliverable quickly on the custom stack. → **nanoGPT / TorchTitan-class
trainer + FSDP2.** Keep the Psi C++ stack as the parallel craft track; ternary-QAT the winner there afterward.

## 1. Architecture — ~130M dense, reasoning-tuned

| component | choice | why |
|---|---|---|
| shape | **~30 layers × ~576 dim** (deep-and-thin) | depth helps reasoning; MobileLLM's deep-thin edge |
| attention | **GQA** (~9 heads / 3 KV groups) | cheap KV, near-MHA quality |
| block | **SwiGLU** · **RMSNorm** · **RoPE** | standard modern block |
| **tokenizer** | **compact ~32k, English+code+math, single-digit numbers** | frees ~60M params from the embedding table into reasoning depth (see [DATA.md](DATA.md)) |
| embeddings | **tied** | with a 32k vocab, tying is cheap and clean |
| context | 2k pretrain → **extend to 8–16k for CoT** | reasoning traces need room; MobileLLM-R1 runs 32k |

Param check: 30×576 with SwiGLU/GQA ≈ **~107M** transformer + **~18M** (32k tied embedding) ≈ **~125M**. A 151k
vocab would instead be ~87M of embeddings, forcing ~12 layers — gutting the very depth reasoning needs.

**Novelty bet (optional, high-risk/high-reward): looped / latent reasoning.** Ouro / LoopLM (2502.17416,
2510.25741) buy reasoning *depth by recurrently reusing a small parameter set* instead of adding params —
explicitly pitched for param-constrained models. Unproven at ≤200M (no verified GSM8K numbers there), so it's a
**separate prototype track, not plan-of-record.** If it fires, it's the difference between "MobileLLM-R1 done
better" and a genuinely novel result — the "groundbreaking" upside.

## 2. Pretrain — reasoning-tilted base (~100–200B tok, WSD)

Web language floor + **heavy math/code from the start** (~45–55% web / 25–30% code / 20–25% math). Goal: not a
generalist, a **reasoning substrate.** WSD (stable → decay). See [DATA.md](DATA.md).
→ *expected: the foundation the reasoning phases build on.*

## 3. Reasoning mid-training — the decisive phase

Short-CoT, rejection-sampled, **correct-only** traces from the ~1–2B teacher; curriculum easy→hard with
gradually growing trace length; short:long ≈ 4:1. This is where the win is made.
→ *expected: the bulk of the lift over MobileLLM-R1.*

## 4. SFT — instruction + verifiable reasoning

Instruction-tune on short-CoT reasoning + verifiable tasks; filter out long/agentic traces (capacity).
Establish the **logic/deduction axis MobileLLM-R1 never reported** — our cleanest uncontested SoTA lane.

## 5. RL (RLVR / GRPO) — optional A/B only

On verifiable rewards, on top of the SFT'd base. Honest A/B; **expect ≈0 gain at 130M**; drop if flat. Not the
headline. *(Reverses our initial RL-first instinct — see [STRATEGY.md](STRATEGY.md).)*

## 6. Compute / time

6·N·D with N ≈ 1.3e8, D ≈ 1–2e11 ≈ **~1e20 FLOPs** — well inside budget. The new dominant cost is **teacher
trace generation + rejection sampling** (running Qwen3-1.7B / R1-Distill-1.5B over problem banks, keeping only
correct + short traces) — this *replaces* the old logit-precompute line. Overnight-ish pretrain on 8×H100; the
campaign cost is trace generation + ablations. See [COMPUTE.md](COMPUTE.md).

## 7. Milestones (de-risk before the full spend)

1. **Pin the reasoning eval harness**; re-run MobileLLM-R1-140M ourselves ([TARGETS.md](TARGETS.md)).
2. **Reasoning-tilted base** (~130M, compact vocab, ~100–150B tok). Prove the substrate + infra.
3. **The one A/B that matters — short-CoT vs long-CoT distillation** on identical bases. Answers "does short-CoT
   clear MobileLLM-R1 at 140M?" cheaply. *(The de-risk question is short-vs-long CoT — **not** does-RL-help; RL
   is demoted.)*
4. *(Stretch, parallel)* looped-reasoning prototype.

## 8. Risks

| severity | risk | mitigation |
|---|---|---|
| **High** | **capacity wall** — GSM8K may cap in the 20s no matter the recipe | pick low-base/high-headroom axes (MATH-500, unreported logic); short-CoT + compact vocab for max reasoning params |
| High→Med | **RL doesn't help at 130M** (likely) | don't stake the headline on it; distillation is the engine |
| Medium | **long-CoT traces don't fit 130M** | compress + rejection-sample; measure trace lengths locally |
| Medium | **looped-reasoning unproven ≤200M** | keep as optional prototype, not plan-of-record |
| Medium | **contamination inflates a "win"** | decontaminate vs test sets; report the check |
| Low | throughput / infra on PyTorch | FSDP2 + flash-attn + bf16 is standard, not a research risk |

---

**Primary sources:** MobileLLM-R1 (2509.24945) · small-model learnability gap (2502.12143) · Qwen3 (2505.09388)
· DeepSeek-R1 (2501.12948) · Phi-4-mini-reasoning (2504.21233) · OpenR1 / OpenThoughts (2506.04178) · looped
reasoning (2502.17416, 2510.25741) · knowledge capacity (2404.05405) · GRPO-at-0.5B (simpleRL-reason, HKUST).
