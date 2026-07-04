# Gemini's plan, adapted to Shohin's goals

Gemini's proposal ([GEMINI_PLAN.md](GEMINI_PLAN.md)) has genuinely sophisticated architecture instincts wrapped
around **one disqualifying decision.** This doc keeps the good ideas, fixes the flaw, and layers everything onto
Shohin's proven, data-first base ([MASTER_PLAN.md](MASTER_PLAN.md)).

## The fatal flaw — and the fix

Gemini's model **cannot read or write natural language.** Three choices enforce this: a **3,000-token
logic-only vocabulary** (no English), the **"Compressed Symbolic Thought" pipeline** that strips every English
word from traces, and **reward functions that penalize natural language**. But **our entire target suite is
English**: GSM8K and MATH are word problems; knights-and-knaves, zebra puzzles, and riddles are natural-language
logic. A model that can't read the question can't solve it — it's a calculator that can't read.

**The fix:** keep our **32k English+code+math tokenizer** and reason in **natural-language short-CoT**, with
optional **program-of-thought (`<code>`)** for steps that benefit from execution. Natural-language reasoning —
including riddles and NL logic — is a **first-class target**, our cleanest uncontested-SoTA lane, not "filler."

Gemini took a *good* instinct — concise, symbol-dense reasoning — to a fatal extreme. **We already sit in the
correct middle ground:** our Reasoning-Gym execution traces are *semi-symbolic but still readable*
(`8453 - 3863 = 4590 ; 4590 - 7439 = -2849`) — exactly Gemini's density **without** losing the ability to read
a prompt.

## Benefits we keep (mapped to Shohin)

| Gemini idea | why it's good | how we adopt it | status |
|---|---|---|---|
| **Linear-attention + local-attention hybrid** (CARVE/KDA/Gated-DeltaNet + MLA) | a *real* 2025–26 SoTA direction — **Qwen3.5 itself is a Gated-DeltaNet hybrid**; cheap long context | ablation-gated option; if adopted prefer **[Gated DeltaNet-2](https://arxiv.org/abs/2605.22791)** (official NVlabs code, validated 1.3B) over the newer single-author **CARVE** ([2606.27229](https://arxiv.org/pdf/2606.27229)). Default stays our proven GQA transformer. We don't need 1M context (tasks are short) → efficiency/depth, not a requirement | **bet** |
| **Weight-shared depth** (MASA / layer folding) | "depth without params" — *exactly* our looped-reasoning thesis | **elevate** — adaptive atom-sharing is a stronger version of our depth-via-recurrence bet | **bet (upgraded)** |
| **Verifier-guided test-time search** (verifier + best-of-N; optional MCTS/SMC) | the single biggest small-model multiplier (TinyGSM: 125M + verifier → 63% GSM8K) | **keep — already our verifier@16 plan.** Default = best-of-N + same-size verifier; MCTS/SMC = advanced option for verifiable tasks | **core** |
| **Critic-free GRPO + AVSPO** (advantage-collapse fix) | clean VRAM-light RL; AVSPO is a genuinely useful stabilizer | keep as our **(demoted) RL polish** stage; fold in AVSPO/ISPO *if* we run GRPO. Evidence still says RL is polish, not engine | **core (as polish)** |
| **Concise / reasoning-dense reward + dual code/NL format** | matches our short-CoT thesis | keep the conciseness reward + `<think>` NL / `<code>` PoT dual format; **remove the anti-NL penalty** | **core (de-clawed)** |
| **Curriculum: syntax → single-step self-correct → multi-step + backtracking** | self-correction / backtracking data is valuable | keep the curriculum shape; **add bug-injection + self-correction traces** to our data. Anchor on verified RG + short-CoT, not symbol-only | **keep** |
| **Symbolic compression of pure-arithmetic traces** | density where language adds nothing | keep as an *additional* format for pure-arithmetic/algorithmic steps (our RG tracers already do this) — **never exclusive** | **keep (bounded)** |
| **Test-Time Training (TTT) layers** | adaptive inference | park as a high-risk research option, unproven at 135M | **optional** |

## What we reject (and why)

- **3k logic-only vocabulary** → kills NL. Use our 32k.
- **CST stripping all NL** → kills question-reading. Keep NL short-CoT + optional PoT.
- **Rewards penalizing natural language** → would literally train language out of the model.
- **Novel-everything infra as the default** (CARVE + MASA + TTT + MCTS + SMC all at once) → too much unproven
  risk simultaneously; conflicts with our strategy. Adopt novelties **one at a time via the 30M "Mame"
  ablation ladder**, each earning its place by a clear proxy win.
- **1M+ context as a core requirement** → solving a problem we don't have (our tasks are hundreds–low-thousands
  of tokens).

## The real disagreement (philosophy)

Gemini spends its **entire risk budget on unproven architecture.** Shohin deliberately uses a **proven recipe**
(Muon / modded-nanogpt speedrun) so the risk budget goes to **data + distillation**, where the science actually
is. Synthesis: **proven base by default; Gemini's best ideas as ablation-gated bets.** Two of them —
**verifier + best-of-N** and **verifiable-reward RL** — are mature enough to keep as *core*.

## How it maps onto what we've already built

- **Vocab flaw → already fixed:** our 32k tokenizer exists and balances NL + code/math + single-digit numbers.
- **Semi-symbolic-in-NL traces → already built:** our RG execution traces are the correct density.
- **Verifier + best-of-N, GRPO/RLVR, short→long curriculum →** already in the master plan.
- **New from Gemini worth adding:** MASA-style weight-shared depth (upgrades the looped-reasoning bet), AVSPO
  (RL stabilizer), self-correction / bug-injection data (curriculum stage 2), and — as advanced bets — the
  linear-attention hybrid and MCTS/SMC steering.

## Concrete changes to the master plan

1. **Architecture:** add "weight-shared depth (MASA, [2508.04581](https://arxiv.org/abs/2508.04581))" and
   "linear-attention hybrid ([Gated DeltaNet-2](https://arxiv.org/abs/2605.22791))" as named ablation-gated
   bets. *The primitives are verified real — but all are 2025–26, validated at ≥0.5–1.3B, none at 135M; gate
   them at the 30M "Mame" proxy, and prefer GDN-2 (official code) over the newer CARVE.*
2. **Post-training:** fold **AVSPO** ([2605.21125](https://arxiv.org/abs/2605.21125), ICML 2026,
   advantage-collapse mitigation, tested 0.5–14B) into the optional GRPO stage.
3. **Data:** add a **self-correction / bug-injection** trace slice (Gemini's curriculum stage 2), generated +
   verified.
4. **Non-negotiable:** keep the 32k tokenizer, NL short-CoT, and **NL logic / riddles as first-class** —
   explicitly reject the symbol-only path.
5. **Test-time:** verifier + best-of-N stays the headline multiplier; MCTS/SMC noted as an advanced option.

## Bottom line

Gemini's architecture is clever and several pieces are worth stealing — **weight-shared depth, verifier-guided
search, critic-free RL with collapse mitigation, concise-dense rewards.** But its central choice, a
**language-free symbolic engine**, is disqualifying for a model whose whole target suite is written in English.
**We keep the machinery, restore the language, and add the good ideas to our proven, data-first base as
ablation-gated bets.**
