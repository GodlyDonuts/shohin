# SHOHIN-135M — Master Plan

### A 72-hour, 8×H100 run to take the open ≤150M reasoning crown

**Version 1.0 — July 3, 2026**

> **Canon & status.** This is the **plan of record**. It subsumes the thematic docs
> [PLAN.md](PLAN.md), [DATA.md](DATA.md), and [TARGETS.md](TARGETS.md) (kept as background; where numbers
> differ, *this file wins*). Two things it does **not** repeat, and that remain required reading:
> [STRATEGY.md](STRATEGY.md) — the two-track Psi mission + why Shohin pivoted to a reasoning specialist; and
> [COMPUTE.md](COMPUTE.md) — the hardware reality (**we do not yet have 8×H100 access on Newton — see the
> hard prerequisite in §1**).
>
> **Reconciliation with our grounding pass (mid-2026).** Our independent research landed the same core calls
> (compact 32k vocab, single-digit numbers, depth>width, RL-as-polish, looped-block as a risky extra) and set
> a *grounded* GSM8K floor of ~22–30 for a **general** short-CoT approach. The higher targets below (T2 40% /
> T3 60%) are **not a contradiction** — they are what you get by adding the two multipliers our pass named as
> decisive: **(1) heavy synthetic GSM-style data** (TinyGSM / OpenMathInstruct-2) and **(2) a same-size
> verifier + best-of-N**. The engine is data; the multiplier is the verifier; RL is polish.

---

## 0. The claim, stated precisely

**Target claim:** *Shohin-135M is the strongest open reasoning model at ≤150M parameters*, measured on a
public, decontaminated suite: GSM8K, GSM8K-Platinum, a Reasoning-Gym battery (held-out seeds and
difficulties), knights-and-knaves, Countdown, cryptarithms, MATH-500, and MBPP — with general-ability
benchmarks (ARC, HellaSwag, PIQA, WinoGrande, LAMBADA) reported alongside to show the specialization didn't
lobotomize the base.

**What we do NOT claim:** general SOTA at 135M across all tasks (that fight is with SmolLM2's 2T tokens and
isn't the thesis), anything about models >150M, and no number that can't be reproduced from the released
harness.

### Win-condition tiers (decide these now, before training)

| Tier | Condition | Why it matters |
|---|---|---|
| **T1 — Must hit** | GSM8K pass@1 ≥ 15% (≈10× SmolLM2-135M-Instruct); beat L20-Edu-135M on every reported benchmark; publish the first ≤150M Reasoning-Gym battery scores | Floor for a credible release |
| **T2 — Target** | GSM8K pass@1 ≥ 40% after SFT; ≥ 60% with verifier best-of-16; win ≥70% of RG task families vs. any open ≤150M model | Headline territory |
| **T3 — Stretch** | Beat Llama-3.2-1B-Instruct on GSM8K (7.4× param handicap); match SmolLM2-135M's six-task general mean (0.4917) | The screenshot that goes viral |

Reference points that make T2/T3 believable rather than fantasy: **TinyGSM** showed a 125M model reaching ~63%
GSM8K with a same-size verifier (specialist, synthetic data); SmolLM2-135M-Instruct sits around 1–2% GSM8K;
the June 2026 **L20-Edu-135M** single-GPU study (13B tokens) posted a six-task mean of 0.4150 vs.
SmolLM2-135M's 0.4917 — and we have ~45× its token budget.

> **Note (grounding).** T1 is our *grounded floor* (consistent with the ~22–30 range from the prior research
> pass — comfortably clears it). T2/T3 are the **ambitious bet** and they live or die on the synthetic-data +
> verifier levers, not on RL. Treat T2/T3 as earned, not assumed — the verifier@16 number is a separate,
> clearly-labeled line from pass@1, never blended.

---

## 1. Budget math

**Hardware:** 8×H100 SXM ≈ 7.91 PFLOPS peak BF16 (dense).

> **HARD PREREQUISITE (blocker).** This entire plan assumes an **8×H100 node for ~72h**. Per
> [COMPUTE.md](COMPUTE.md), our Newton account (`skattel` / `arcc_pi_skattel`) currently **cannot submit to
> `highgpu`**. No node → no run. Resolve *before* the pre-window program (§9) via one of: PI email for
> `highgpu` access (free, ~1 day), or a cloud 8×H100 rental (~$500–700/run, exactly the 72h scenario). This
> is Open Question #1 and it gates everything downstream.

**Model cost per token** (Shohin-135M, ~132M params incl. tied embeddings):
- Weight FLOPs: 6N ≈ 7.9e8
- Attention (fwd+bwd), seq 2048, sliding-window 1024 on 24/32 layers, full attention every 4th layer: ≈ 2.8e8
- **Total ≈ 1.07e9 FLOPs/token** at 2k context (≈1.4e9 at the 8k anneal context)

**Throughput planning table (pretraining, 2k context):**

| MFU | Tokens/hour (cluster) | 47h Stage-1 yield |
|---|---|---|
| 33% | 8.7B | 410B |
| 40% (planning number) | 10.6B | 500B |
| 45% | 11.9B | 560B |

**Central estimate for the full run: ~580B tokens** (≈500B Stage-1 pretrain + ≈80B Stage-2 anneal at 8k
context, which runs slower per token). Envelope: 480–700B. That is ~25–35% of SmolLM2-135M's 2T tokens —
closed by a Muon-era recipe, 2025–26 data, and specialization. **MFU ≥ 35% is a hard launch gate** (§9).

---

## 2. The 72-hour schedule

Design principle: **the window is an execution of decisions already made, not a lab.** Every hyperparameter
below has a default; ablations (§9) may override them before launch, never during.

| Hours | Phase | Detail |
|---|---|---|
| H0–H0.5 | Launch gate | Throughput check vs. rehearsal number (±10%), checkpoint-write + resume smoke test, W&B live |
| H0.5–H47 | **Stage 1 pretrain** | ~500B tokens, seq 2048, WSD stable phase, batch ramp 0.5M→2M tokens over first 10B |
| H47–H57 | **Stage 2 reasoning anneal** | WSD decay phase; mix shift (§6.2); context 2048→8192 (RoPE θ 50k→500k); ~80B tokens |
| H57–H60 | **SFT** | Two variants in parallel (4 GPUs each): NL-CoT-heavy vs. PoT-heavy mix; pick winner on dev set |
| H59–H61 | **Verifier** | While SFT-v2 finishes: sample K=16 solutions/problem from SFT-v1 on 2 GPUs, auto-label, train Shohin-Verify-135M |
| H61–H67 | **RLVR (GRPO)** | Reasoning-Gym curriculum + GSM8K-train + own generators; verifiable rewards only |
| H67–H69 | **Full evals** | Entire suite ×{base, SFT, RL} checkpoints; best-of-N sweeps N∈{1,4,16} |
| H69–H72 | **Reserve** | Re-run weakest stage, extend RLVR, or extra anneal from a WSD checkpoint — decided by the eval table |

**If the 72h is splittable**, the clean cut is after H57 (end of anneal): base model is complete and safe;
post-training resumes any time. If a node dies, WSD means *every* checkpoint during the stable phase is a
valid model — resume or decay-early are both fine.

Checkpoint every 20 minutes to local NVMe, async rsync of every 3rd checkpoint off-node. Loss-spike policy: if
loss > rolling μ+4σ, auto skip-batch; two spikes in 500 steps → drop Muon LR 30% and continue.

---

## 3. Architecture — Shohin-135M

Deep-thin, because depth carries reasoning at small scale (MobileLLM finding; SmolLM2 and L20-Edu both use
30×576). The tokenizer dividend (§4) funds two extra layers under the same param budget.

| Component | Spec | Rationale |
|---|---|---|
| Layers × width | **32 × 576** | Two layers deeper than SmolLM2-135M at fewer total params |
| Attention | 9 Q heads (d=64), 3 KV (GQA) | SmolLM2/L20-proven shape |
| Attention pattern | Sliding window 1024 on 24 layers; **full attention every 4th layer** | ~2.5× cheaper attention; 8 global layers preserve long-range hops |
| Doc masking | Intra-document attention only (FlexAttention block mask) | Speedrun-standard; free quality |
| FFN | Squared-ReLU, d_ff 2304 | Speedrun-proven; SwiGLU-1536 is the ablation alternative |
| Norms | RMSNorm pre-norm + **QK-norm** | Stability insurance for Muon at high LR |
| Position | RoPE, θ=50k (Stage 1) → 500k (anneal, 8k ctx) | Standard extension path |
| Logits | z-loss 1e-4 (or softcap 30 — ablate) | Spike insurance |
| Embeddings | Tied, vocab 32,768 | §4 |
| Params | **≈132M** (18.9M embed + 113.3M blocks) | Headline: "fewer params than SmolLM2-135M" |

**Ablation-gated extras** (default OFF; each earns entry only by winning at 30M proxy scale by a clear margin,
§9): value embeddings (speedrun trick), multi-token-prediction aux head, plus two verified from the Gemini-plan
review ([GEMINI_PLAN_ADAPTED.md](GEMINI_PLAN_ADAPTED.md)):

- **Weight-shared / recurrent depth — *the* reasoning bet (flagship).** Reasoning is depth-bound, so recurring
  a shared block buys effective depth (~30 logical from ~10 physical) without spending params. Two flavors:
  - *Fixed depth:* Ouro-style looped block / **MASA** dictionary-atom sharing ([2508.04581](https://arxiv.org/abs/2508.04581)).
  - *Adaptive "extended thinking":* **latent-space recurrent depth** — recur a *variable* number of iterations
    per token (harder → more), the Huginn/recurrent-depth ([2502.05171](https://arxiv.org/abs/2502.05171)) /
    Coconut ([2412.06769](https://arxiv.org/abs/2412.06769)) / Ouro lineage. **This is a test-time-compute knob
    that gives extended thinking WITHOUT emitting long token chains — sidestepping the small-model long-CoT
    learnability wall (our biggest reasoning constraint).** A second multiplier, orthogonal to verifier+best-of-N.
  Keep it **additive** to the visible token-CoT — latent recursion adds *internal* compute, it does not replace
  the `<think>` trace, so we keep auditable / verifiable / decontaminatable reasoning + the SFT data. Highest
  reasoning upside of anything here; **unproven at 135M** and recurrent-depth training is unstable
  (truncated-BPTT / curriculum) → gated, prove at 30M "Mame" proxy first.
- **Linear-attention hybrid** — Gated DeltaNet-2 ([2605.22791](https://arxiv.org/abs/2605.22791), official
  code) ± CARVE, with sparse full-attention layers. Buys **context length we don't need** (our tasks are short)
  and its fixed-state compression **risks precise-math fidelity** → kept only as a throughput/efficiency
  ablation, **not** a reasoning play. Low priority for a short-context reasoner.

**Explicitly rejected:** MoE (routing overhead + MFU loss at 100M-total scale, and per-active-param framing
invites goalpost accusations vs. dense SmolLM2); from-scratch framework (fork modded-nanogpt, port GQA/RoPE/SWA
onto its Muon+FlexAttention+compile core); full-run logit distillation (teacher forwards cost 2×+ the whole
window even with a 0.6B teacher — sequence-level distillation via data instead).

---

## 4. Tokenizer

Train a **32,768-token BPE** (byte fallback) on a 20GB sample of the actual mix (≈60% English edu / 25% math /
15% code), with:

- **Single-digit number splitting** — measurably improves small-model arithmetic; non-negotiable for this thesis
- Code whitespace tokens (indent runs), LaTeX-aware pre-tokenization kept simple
- Reserved tokens: `<think>`, `</think>`, `<code>`, `</code>`, chat template, verifier labels

Why 32k instead of SmolLM2's 49k: at d=576, the 49k tied embedding costs 28.3M params (21% of the model). 32k
costs 18.9M — the ~9.4M saved buys the two extra transformer layers. English+code+math only needs far less
vocab than multilingual coverage; slightly worse compression on general web text is an acceptable trade for a
reasoning specialist.

**Fallback (pre-decided):** if the 30M-scale A/B shows the custom tokenizer losing on bits-per-byte *and*
GSM-proxy accuracy, ship SmolLM2's tokenizer and drop to 30 layers. No mid-window tokenizer decisions.

---

## 5. Optimizer and training configuration

| Item | Default | Notes |
|---|---|---|
| Hidden 2D matrices | **Muon**: lr 0.02, momentum 0.95, Newton-Schulz 5, wd 0 | lr swept at 30M scale {0.01, 0.02, 0.04}; Moonlight-style wd 0.1 is the ablation |
| Embeddings / head / norms / scalars | AdamW: lr 3e-3, β(0.9, 0.95), wd 0 on embed/norms | |
| Schedule | **WSD**: 2B warmup → constant → linear decay to 0.1× across the H47–H57 anneal | Any stable-phase checkpoint is usable; enables anneal forking |
| Batch | Global 2M tokens (ramp from 0.5M over first 10B) | 128 seq/GPU at 2048 |
| Precision | bf16 params + fp32 master/accum, grad clip 1.0 | |
| Parallelism | Plain DDP + torch.compile + FlexAttention | No FSDP needed at 132M; keep it boring |
| Dataloader | Pre-tokenized uint16 memmap shards on local NVMe, shuffle-buffer, prefetch, per-source sampling weights | Starvation test in rehearsal |
| SFT | lr 1e-3, small batch (~128k tokens), 2–3 epochs, cosine | SmolTulu found high LR/batch ratios favor reasoning tasks (ARC, GSM8K) at exactly 135M scale |
| GRPO | group 16, lr 2e-6, no KL, clip-higher (DAPO-style), temp 1.0, max gen 768 | Reward = strict verifier match + small capped format bonus |

---

## 6. Data plan (this is where the model is actually built)

Governing insight from the OpenMathInstruct-2 ablations, which transfer directly to a 135M student: **concise
solution formats beat verbose ones** (their concise CoT beat Llama's format by 3.9% while 40% shorter),
strong-teacher data beats on-policy weak-student data, SFT tolerates ~20% noise (answer-match filtering
suffices), and **question diversity drives scaling gains**. Every corpus decision below follows from these.

### 6.1 Stage 1 — ~500B tokens (broad substrate, seq 2048)

| Source | Share | ~Tokens | Notes |
|---|---|---|---|
| FineWeb-Edu (classifier ≥3, deduped) | 55% | 275B | The English backbone — puzzles arrive in natural language; this is non-negotiable |
| DCLM-baseline, FW-Edu-classifier-filtered | 8% | 40B | Diversity supplement (SmolLM2's own trick) |
| **Math 20%** — MegaMath-Web-Pro + MegaMath-Web (≈75B), FineMath-4+ / InfiWebMath-3+ (≈20B), OpenWebMath (≈5B) | 20% | 100B | MegaMath: 371B-token open math corpus (web + Stack-V2 code + 80B synthetic) — the reservoir this plan drinks from |
| **Code 14%** — Stack-Edu (Python-heavy: Py/C/C++/JS) + MegaMath-Code | 14% | 70B | Code-in-pretraining → reasoning is one of the better-replicated findings |
| Cosmopedia-v2 textbook-style + FLAN-style | 3% | 15B | Formatting diversity |

### 6.2 Stage 2 — reasoning anneal, ~80B tokens (seq 8192, WSD decay)

| Source | Share | Notes |
|---|---|---|
| Highest-grade FineWeb-Edu | 30% | Keeps the base from drifting |
| Math, hard-weighted | 30% | FineMath-4+ (2–3 epochs is fine), MegaMath-QA + synthetic, OpenMathInstruct-2 reformatted as documents (14M solutions ≈ 4–5B tokens), AugGSM8K-style augmentation |
| Code, math-interleaved | 20% | Stack-Edu Python + MegaMath text-code blocks |
| **Procedural reasoning corpus (ours)** | 15% | §6.3 — the differentiator |
| Instruct/chat (SmolTalk-family subset) | 5% | Template familiarity before SFT |

### 6.3 The procedural reasoning corpus — our edge

Reasoning-Gym provides **100+ generator+verifier pairs** across algebra, arithmetic, computation, cognition,
geometry, graph theory, logic, and games, with parametric difficulty and infinite instances — NeurIPS 2025
Spotlight, already used by NVIDIA's ProRL and Nemotron work. We use it three ways: anneal documents, SFT
traces, and RLVR environments.

Pre-window (CPU-only, cheap), generate **5–10B tokens of solution-trace documents**: for each task family,
emit the generator's instance + a templated step-by-step derivation from the *solver's* internal path (BFS
traces for mazes, truth-table elimination for knights-and-knaves, search traces for Countdown, constraint
propagation for the cryptarithm solver you already built with z3 for the Nemotron challenge — it drops in as a
first-class generator+verifier). This is execution-trace pretraining: infinite, verifiable, **decontaminated by
construction**, and costs zero teacher FLOPs.

**Held-out discipline:** freeze a set of RG task configs + seeds now as eval-only. They never appear in any
training stage. This is what makes the generalization claim clean.

### 6.4 SFT corpus (~1.5–3B unique tokens)

- **OpenMathInstruct-2** subset: concise solutions (<400 tokens), difficulty-stratified, ~2–4M pairs (14M
  pairs / 600K unique questions available, CC-BY-4.0, Llama-405B-generated)
- **TinyGSM** Python-solution subset → the PoT format
- RG solution traces (training configs only), knights-and-knaves, Countdown, cryptarithms
- Fresh teacher top-up (optional, ~$100–300 pre-window): Qwen3-8B generating *short* traces on RG + augmented
  GSM-style problems, rejection-sampled by verifiers. **Trace-length ceiling ~512 tokens** — a 135M student
  chokes on R1-length reasoning; match trace complexity to student capacity
- Small SmolTalk2 slice for instruction-following retention
- Dual format with tags: `<think>short NL chain</think>answer` and `<code>python</code>` — eval picks per
  task, verifier reranks

### 6.5 Decontamination (the claim's armor)

13-gram Bloom-filter pass of **every** training token (all stages, SFT, RL prompts) against every eval set;
publish hit counts and the script. Report GSM8K-Platinum alongside GSM8K. Procedural evals use fresh seeds —
contamination-proof by construction. An undergrad SOTA claim gets audited like a resume bullet; this section
is why it survives.

---

## 7. Post-training

**SFT (H57–H60).** Two mixes in parallel on 4 GPUs each — (a) NL-CoT-heavy, (b) PoT-heavy — 2–3 epochs, pick
on a 1k-problem dev slice. High LR/small batch per the SmolTulu 135M finding.

**Verifier (H59–H61).** Shohin-Verify-135M: init from the anneal checkpoint, train as a binary judge (problem
+ candidate → correct-token probability) on K=16 self-samples per training problem, auto-labeled by answer
checking. This is the TinyGSM mechanism — **the single biggest test-time multiplier available to a small
model.** Report pass@1 and verifier@16 as separate, clearly labeled numbers.

**RLVR / GRPO (H61–H67).** verl (or TRL+vLLM) with rewards from RG verifiers + exact-match GSM checking + our
own verifiers. Curriculum: start at easy configs (arithmetic, propositional logic), ramp difficulty on a fixed
schedule. Strict correctness reward; format bonus capped low to prevent reward hacking. **AVSPO**
([2605.21125](https://arxiv.org/abs/2605.21125), ICML 2026) is folded in to fix GRPO *advantage collapse* —
all-correct or all-wrong groups give zero gradient under binary verifiable rewards; it injects virtual reward
samples to restore the signal (−58–63% collapse, +4–6pp across 0.5–14B). Precedent at exactly
this scale: L20-Edu-135M ran RLVR on GSM8K at 134.5M params. **Expectation discipline: RLVR is polish
(single-digit to low-teens relative gains + robustness), not the engine — the engine is §6.** RG verifiers are
what make RLVR well-posed here (clean rewards + curriculum), which is why it earns a scheduled slot despite
the general "RL barely helps at 130M" finding.

---

## 8. Evaluation protocol and the baseline table

All baselines **re-run by us** under one lm-eval-harness fork with pinned prompts/configs — never compare our
numbers to someone else's harness. Published anchor (L20-Edu-135M paper, June 2026):

| Model | ARC-C | ARC-E | HellaSwag | LAMBADA | PIQA | WinoGrande | Mean₆ |
|---|---|---|---|---|---|---|---|
| SmolLM2-135M | .297 | .585 | .430 | .429 | .684 | .525 | **.492** |
| SmolLM-135M | .288 | .561 | .427 | .376 | .682 | .527 | .477 |
| L20-Edu-135M | .287 | .496 | .324 | .260 | .615 | .508 | .415 |
| GPT-2 / OPT-125M / Pythia-160M | — | — | — | — | — | — | .35–.41 |

> **Verify-before-trust.** The L20-Edu-135M anchor (arXiv 2606.22189, June 2026) is past our research
> verification window — treat its numbers as *claimed* until we re-run the model ourselves under our harness
> (which the protocol above already requires). If it doesn't reproduce, T1's "beat L20-Edu" clause re-baselines
> against SmolLM2/SmolLM/Pythia instead.

Reasoning suite (the actual fight): GSM8K (0-shot CoT + 5-shot), GSM8K-Platinum, RG battery (held-out seeds
*and* held-out difficulty levels, per-family reporting), knights-and-knaves, Countdown, cryptarithms (fresh),
MATH-500 (expect low — report anyway), MBPP + HumanEval. Comparison set: SmolLM2-135M(-Instruct), SmolLM-135M,
MobileLLM-125M, Pythia-160M, GPT-2, L20-Edu-135M, and for the param-handicap rows: Gemma-3-270M, Qwen3-0.6B,
Llama-3.2-1B-Instruct.

Report per-family RG results, not just an average — the generalization story ("trained on families A–F configs
1–3, evaluated on configs 4–5 and families G–H") is the scientific contribution beyond the GSM8K headline.

---

## 9. Pre-window program (T-minus 4 weeks → T-0)

The run is won here. Budget: ~$150–400 in rentals; everything else is CPU/free-tier.

**T-4w → T-3w — Data acquisition + tokenizer**
- Download: FineWeb-Edu (350BT sample + top-up), DCLM subset, MegaMath (371B), Stack-Edu, FineMath/InfiWebMath,
  OpenWebMath, Cosmopedia-v2, OpenMathInstruct-2, TinyGSM, SmolTalk2. Multi-TB; start day one; mirror to a
  Modal volume as backup
- Train tokenizer; freeze after the A/B (below)
- Tokenize everything to uint16 shards (~1.3TB for 650B tokens) — parallel CPU on Modal or a big local box
- **Hard dependency to confirm now: ≥4TB local NVMe on the cluster nodes, and ability to pre-stage data before
  the window opens**

**T-3w → T-1w — Ablation ladder** (rented single 4090/A6000, or Modal; 30M-param "Mame" proxies, ~10–14 runs ×
2–4h)
1. Muon LR sweep {0.01/0.02/0.04} + Adam LR pairing
2. Tokenizer A/B (custom-32k vs SmolLM2-49k) on bits-per-byte + GSM-proxy
3. Squared-ReLU vs SwiGLU; z-loss vs softcap
4. Depth check: 32×576 vs 28×640 at matched params
5. Two anneal-fork mix ratios (math/procedural share)
6. Extras gate: value embeddings, MTP, **weight-shared depth (looped / MASA)**, and — efficiency only —
   **GDN-2 linear-attn hybrid** — in only on a clear proxy win; **prioritize depth-sharing** (the reasoning lever)
7. SFT format pilot: NL-CoT vs PoT vs dual on a 30M student — sets the H57 priors

**T-1w — Systems rehearsal**
- Generate the procedural corpus (CPU); generate teacher top-up traces if budget allows
- Full-stack dry run, ideally 1h on the real cluster: measure MFU (**gate: ≥35%, i.e. ≥9B tok/h — if below
  after tuning, pre-decided fallbacks: shrink global-attention layer count, then 30 layers**), kill-and-resume
  drill, dataloader saturation test, W&B dashboards, eval harness smoke test on SmolLM2-135M (validates the
  baseline table)
- Freeze configs. Write the launch runbook: exact commands, decision gates, abort criteria

**Decision hygiene:** after T-0, no untested ideas. Anything clever that shows up mid-window goes in a NOTES.md
for run 2.

---

## 10. Risk register

| Risk | Likelihood | Mitigation |
|---|---|---|
| **8×H100 access not secured** | **Open (§1)** | **Blocker — resolve at T-4w: PI email for `highgpu` or cloud rental. Everything else is moot without it.** |
| MFU shortfall | Medium | Rehearsal gate + pre-decided arch fallbacks; SWA already cuts attention cost |
| Dataloader starvation | Medium | Pre-tokenized memmap + prefetch; saturation test in rehearsal |
| Loss spikes (Muon, high LR) | Medium | QK-norm, z-loss, clip 1.0, auto skip-batch + LR-dip policy |
| Node failure | Low–Med | 20-min checkpoints, async off-node copies, WSD = every checkpoint valid |
| Non-contiguous window | Unknown | Clean split point at H57; confirm with cluster admin now |
| SFT overfits templates | Medium | Dual formats, held-out generator configs, per-family eval |
| RL reward hacking | Medium | Strict verifiers, capped format bonus, manual transcript audits at H63 |
| Tokenizer regression | Low | A/B gate + pre-decided fallback to SmolLM2 tokenizer |
| Contamination accusation | Certain (someone will check) | §6.5: published decontam script + counts, Platinum variant, fresh-seed procedural evals |
| GSM8K gain doesn't generalize | Medium | The RG battery with held-out families *is* the test; report honestly per-family |
| L20-Edu anchor doesn't reproduce | Low–Med | Re-run ourselves (§8); re-baseline T1 against SmolLM2/Pythia if needed |

---

## 11. Release and writeup (the actual ROI)

Ship within a week of the run, while it's fresh:

1. **HF org:** Shohin-135M-Base, -Instruct, -Verify, tokenizer, full model cards with the mix table
2. **Tech report** (arXiv-ready): budget math, mix, ablation table, decontam receipts, per-family results,
   negative results included — the "auditable" framing is the credibility play
3. **Public W&B** run + the eval-harness fork with pinned configs
4. **Blog + thread:** the one-chart version — GSM8K vs params, Shohin's dot far above the ≤150M cluster
5. **Outreach hooks:** Modal (data prep / trace generation on Modal), NVIDIA (built on OpenMathInstruct-2 + the
   Nemotron-adjacent RLVR stack), Liquid (efficiency-per-param is their entire product thesis), HF (SmolLM2
   comparison done respectfully — they're the audience most likely to signal-boost)

Naming system in the release: **Mame** for the 30M ablation proxies, **Shohin** for the 135M flagship — the
bonsai size-class scheme, explained in one README line. *(The sibling custom C++/ternary track keeps the name
**Psi** — see [STRATEGY.md](STRATEGY.md).)*

---

## 12. Open questions (answer before T-4w)

1. **8×H100 access — secured how (Newton `highgpu` grant vs. cloud rental)?** ← the blocker (§1)
2. Is the 72h contiguous, or splittable (ideal split: 57h + 15h)?
3. Node-local NVMe capacity, and can data be pre-staged before the window?
4. Can we get ~1h on the actual cluster for the rehearsal gate?
5. Pre-window cash budget ceiling ($150–400 covers ablations + teacher traces)?

---

## References

- SmolLM2 (COLM 2025): https://arxiv.org/abs/2502.02737 — 135M trained on 2T tokens; mixture design
- L20-Edu-135M (June 2026): https://arxiv.org/abs/2606.22189 — auditable 135M baseline, RLVR at this scale *(post-cutoff; verify)*
- MegaMath (COLM 2025): https://arxiv.org/abs/2504.02807 — 371B-token open math corpus
- OpenMathInstruct-2 (NVIDIA): https://arxiv.org/abs/2410.01560 — 14M pairs; format/teacher/noise ablations
- Reasoning-Gym (NeurIPS 2025 Spotlight): https://arxiv.org/abs/2505.24760 — 100+ generators/verifiers
- TinyGSM: https://arxiv.org/abs/2312.09241 — 125M + verifier on GSM8K
- SmolTulu: https://arxiv.org/abs/2412.08347 — LR/batch ratio vs reasoning at 135M
- modded-nanogpt: https://github.com/KellerJordan/modded-nanogpt — Muon + speedrun stack
- MobileLLM: https://arxiv.org/abs/2402.14905 — depth > width at sub-billion scale
