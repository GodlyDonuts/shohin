# Targets & eval — what "beat SmolLM2" means, in numbers

**The bar:** SmolLM2-135M base (arXiv 2502.02737), trained on 2T tokens — the strongest fully-open ~135M
base model. Grade on a **fixed commonsense suite re-run on our own harness** against SmolLM2-135M. (Quoted
aggregate means from different papers use different task sets and are **not comparable** — always re-run.)

## Pre-registered targets

| benchmark | SmolLM2-135M | our target | Δ | verdict |
|---|---:|---:|---:|---|
| **HellaSwag** | 42.1 | **48–52** | +6–10 | **win (headline)** |
| ARC (avg e/c) | 43.9 | ≥50 | +6 | win |
| PIQA | 68.4 | 71–73 | +3–5 | win |
| WinoGrande | 51.3 | ≥55 | +4 | win |
| OpenBookQA | 34.6 | ≥38 | +3 | win |
| CommonsenseQA | 33.9 | ≥37 | +3 | win |
| commonsense-suite mean | ~49 | **≥53–55** | +4–6 | win |
| — | | | | |
| MMLU (cloze) | 31.5 | ~30–35 | no regress | **capacity-capped** |
| TriviaQA | 4.1 | ~4 | no regress | **capacity-capped** |
| GSM8K (5-shot) | 1.4 | low single digits | no regress | **capacity-capped*** |
| IFEval (after SFT) | ~29.9 (Instruct) | 30s–40s | — | instruction-following |

## Winnable vs capacity-capped

**Winnable (better data + distillation move these):** HellaSwag, PIQA, ARC-e/c, WinoGrande, OpenBookQA,
CommonsenseQA, instruction-following (IFEval), per-token data efficiency.

**Capacity-capped (report, don't chase):** MMLU (factual recall), TriviaQA (closed-book knowledge),
GSM8K/MATH (multi-step math), long-chain reasoning.

**Why:** LMs store ~2 bits/param (Allen-Zhu, arXiv 2404.05405) → ~130M ≈ 30 MB of facts. Even Qwen3-0.6B
(5× the params, 36T tokens) only reaches MMLU 52.8 — so knowledge is unreachable at 130M by any recipe.

## The honest asterisks

- **\* Math is technically movable — but we decline the trade.** Meta's MobileLLM-R1-140M (arXiv 2509.24945)
  hits GSM8K 16.3 at 140M via narrow math/code mid-train + long-CoT SFT — but its commonsense average (44.3)
  *trails* SmolLM2 (50.7 on that suite). Buying math costs the commonsense axes we want. So GSM8K stays a
  no-regression report, not a target.
- **The headline win is unproven at 135M.** No public sub-200M base model has cleanly beaten SmolLM2-135M on
  commonsense *purely by method* — the demonstrated 2025–26 wins (Qwen3-0.6B, LFM2.5-350M, Gemma-3-270M) are
  2.5–4.5× larger. Treat the +6–10 HellaSwag target as **ambitious-but-unproven at 135M**, not a solved
  recipe.

## Eval harness (milestone 0)

Stand up a fixed harness (lm-evaluation-harness or equivalent) with: HellaSwag, ARC-easy, ARC-challenge,
PIQA, WinoGrande, OpenBookQA, CommonsenseQA — plus MMLU/TriviaQA/GSM8K/IFEval for no-regression tracking.
**Re-run SmolLM2-135M on it ourselves** and record the numbers; that is the scoreboard for every subsequent
run. Same few-shot settings, same scoring, every time.
