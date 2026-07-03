# The 100M build plan (stage by stage)

Concrete recipe for a ~130M dense model that beats SmolLM2-135M on the winnable axes. Sources are listed at
the bottom; confidence is HIGH for the data + logit-KD-at-fixed-tokens claims, LOW / against-trend for
KD-at-130M and reasoning-at-100M (flagged inline).

---

## 0. Framework decision — PyTorch, not the custom stack

The entire 2026 edge comes from three things the Psi C++ stack cannot deliver quickly: **pretraining-time
logit distillation** (no KD infra), **bf16 tensor-core throughput** (custom stack is fp32 → ~5–15× slower),
and **8-GPU data parallelism** (custom stack is single-GPU). Rebuilding that infra (~8–14 person-weeks)
forfeits the very levers that produce the win. → **Use a TorchTitan / nanoGPT-class trainer + FSDP2**
(bf16, flash-attention, mature multi-GPU DP). Iterate on **data + distillation**, where the real science
risk is. Keep the custom stack as a parallel craft track; ternary-QAT the winner there afterward.

## 1. Architecture — ~130M dense

| component | choice | why |
|---|---|---|
| shape | **~30 layers × 576 dim** (deep-and-thin) | MobileLLM: deep-thin adds +2.7% commonsense at 125M |
| attention | **GQA**, ~9 heads / 3 KV groups | cheap KV, near-MHA quality |
| block | **SwiGLU** MLP · **RMSNorm** · **RoPE** | Qwen3 / LFM2-class standard block |
| embeddings | **tied** input/output | at 130M, tying saves ~12% of params for the layers |
| vocab / tokenizer | **~49k, reuse the teacher's BPE** (SmolLM2 / Llama-3) | teacher-compatible logits → full soft-label KD, no vocab-projection hack |
| context | 2k pretrain → extend late | long context is not where the win is |

**Pick the teacher's tokenizer FIRST, then build the student around it** — a shared vocab is what lets you
do full soft-label KD without a lossy projection.

## 2. Training pipeline — ~200B tokens, WSD (stable → decay)

**Stage A — pretrain (stable), ~160B tok (80%).** Anchor on **Nemotron-CC-HQ** (empirically the #1 dataset
at 130M scale, +5.6 MMLU vs DCLM) blended with **DCLM-baseline** (commonsense) + **FineWeb-Edu** (knowledge).
Mix ≈ **78% web / 15% code (Stack-Edu) / 7% math (FineMath4+, MegaMath)**. **Run logit-KD throughout.**
→ *expected: the foundation + most of the KD lift.*

**Stage B — mid-train (decay), ~40B tok (20%).** Linear LR decay to ~0. Shift the mix toward quality:
≈ **60% web / 22% code / 15% math (+OpenMathReasoning) / 3% Cosmopedia-v2 synthetic**, plus a slice of
instruction data. This is the SmolLM3 decay-phase move that reliably lifts benchmarks.
→ *expected: +2–4 pts across the commonsense suite.*

**Stage C — SFT (short).** Instruction-tune on **SmolTalk-style** data with complex/long-CoT tasks filtered
out (capacity). Cap reasoning data at **25–50%** of the mix.
→ *expected: IFEval into the 30s–40s (SmolLM2-Instruct ≈ 29.9).*

**Stage D — DPO (optional, short).** Light preference optimization on distilled preference pairs.
→ *expected: small IFEval / helpfulness bump.*

**Reasoning phase — SKIP (or minimal).** Do **not** build a long-CoT thinking-mode. If used at all, a tiny
(~5–15B tok) short-rationale phase as a *data ingredient*, never as the model's identity.
→ *expected: neutral-to-negative on the graded (multiple-choice) axes.*

## 3. Distillation — the decisive lever

- **Method:** **offline soft-label (logit) KD.** Precompute a converged teacher's **top-k truncated logits**
  (top-0.95, k≈50) over the corpus; train the student with **KL loss at α≈0.9 KD weight** under the WSD
  schedule. *(MSE loss hurts −7.6%; online KD is worse than offline converged-teacher logits.)*
- **Teacher:** a strong but **CLOSE** model — **Qwen3-1.7B or Llama-3.2-1B**, not a 30B frontier model. A
  130M student is ~7–13% of a 1–2B teacher — right at the empirical effectiveness floor (student ≥ ~10% of
  teacher). A larger teacher widens the capacity gap and helps less.
- **Honest asterisk:** KD is unproven below 330M and *larger students benefit more*; at 130M the marginal
  gain of logit-KD over simply training on the teacher's generated text may be small. So **DATA quality is
  a co-equal lever, not an afterthought.**

## 4. Compute / time

Core math: 6·N·D with N=1.3e8, D=2e11 ≈ **1.6e20 FLOPs** — inside a 2–3e20 budget. On 8×H100 in bf16 the
~200B-token pretrain is roughly **one overnight run**. The real cost sinks are one-time **teacher-logit
precompute** (~30%) and **data-mix ablations** (~24%), not the flagship train. Full campaign (precompute →
baseline → 2–3 ablations → final run → post-train) ≈ **3–6 × 24h sessions on 8×H100.** See
[COMPUTE.md](COMPUTE.md) for the hardware options (we don't have 8×H100 on Newton yet).

## 5. First milestone (de-risk before the full spend)

1. **Pin the eval harness** — fixed commonsense suite (HellaSwag, ARC-e/c, PIQA, WinoGrande, OBQA, CSQA);
   re-run SmolLM2-135M on it ourselves. Never compare to quoted aggregates (different task sets).
2. **Baseline pretrain — no KD** — ~130M arch, ~100–150B tokens of the Nemotron-CC-HQ / DCLM / FineWeb-Edu
   mix, WSD. Goal: land within a couple points of SmolLM2-135M. Proves data + arch + infra.
3. **Add KD on top — attribute the lift** — identical config + offline logit-KD from the ~1–2B teacher. If
   the commonsense mean moves meaningfully → distillation thesis validated at our scale. If not → we spent
   little and learned the real answer.

This ordering isolates the one unproven variable (KD below 330M) as a cheap A/B, and never conflates a
data-mix win with a distillation win. **If the baseline can't reach SmolLM2, no KD will** — better to learn
that on a half-budget run.

## 6. Risks

| severity | risk | mitigation |
|---|---|---|
| **High** | **Chasing the wrong axes** (MMLU/GSM8K/long-CoT) — the most expensive failure | keep them off the scoreboard; report only for no-regression |
| High | **Does logit-KD help at 130M?** (the biggest uncertainty — tested only ≥330M) | milestone-3 A/B answers it cheaply; fallback = data + post-train win |
| Medium | **Data-quality edge shrinks at 130M** — our 200B-vs-2T token deficit may be hard to erase with quality alone | raises the burden on KD; lean on Nemotron-CC-HQ + decay-phase quality |
| Medium | **Teacher/student vocab mismatch** breaks full soft-label KD | lock the student to the teacher's BPE early |
| Low | **Throughput / infra on PyTorch** | FSDP2 + flash-attn + bf16 on 8×H100 is standard, not a research risk |

---

**Primary sources:** SmolLM2 (arXiv 2502.02737) · MobileLLM (2402.14905) / MobileLLM-R1 (2509.24945) ·
Nemotron-CC (2412.02595) · open-sci-ref-0.01 (2509.09009) · Pretraining Distillation design space
(2410.16215) · Scale or Reason? (2509.22193) · Small Model Learnability Gap (2502.12143) · Physics of LMs
3.3 / knowledge capacity (2404.05405) · LFM2 (2511.23404) · Qwen3 (2505.09388) · Gemma 3 (2503.19786) ·
SmolLM3 blog.
