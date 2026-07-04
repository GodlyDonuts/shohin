# Gemini's proposal (verbatim, cleaned) — "The 150M Parameter Reasoning Engine"

> **Source:** a Gemini-authored plan the user shared (formatting/LaTeX normalized for readability; substance
> unchanged). This is preserved for reference. Our adapted, goals-aligned version is in
> [GEMINI_PLAN_ADAPTED.md](GEMINI_PLAN_ADAPTED.md).
>
> **Sources verified (2026-07)** — I initially flagged these as possibly fabricated; that was **wrong**,
> corrected after web-checking. They are real, recent papers: **Gated DeltaNet-2**
> ([arXiv 2605.22791](https://arxiv.org/abs/2605.22791), Hatamizadeh/Choi/Kautz, NVIDIA, May 2026,
> [official code](https://github.com/NVlabs/GatedDeltaNet-2)) · **CARVE**
> ([arXiv 2606.27229](https://arxiv.org/pdf/2606.27229), S. Dutta, Jun 2026) · **MASA** = "Share Your Attention"
> ([arXiv 2508.04581](https://arxiv.org/abs/2508.04581), AAAI) · **AVSPO**
> ([arXiv 2605.21125](https://arxiv.org/abs/2605.21125), ICML 2026) · **ISPO** (Jun 2026, unchecked). Only
> **Compressed Symbolic Thought** is a bespoke framework, not a validated method — and it is the source of the
> fatal flaw. **Caveat that still holds: all are 2025–26 and validated at 0.5–1.3B+, none at 135M — real ≠
> proven at our scale.**

*A Production-Ready Engineering Playbook for Sub-Quadratic Infinite-Horizon Algorithmic Execution.*

## 1. Compressed Linear Hybrid Architecture
Softmax attention is O(T²) and the KV cache grows linearly, which is unsustainable at a 1M+ token target and
leaves no parameter budget for deep logic at 150M. So replace the transformer backbone with a **sub-quadratic
hybrid**: hardware-optimized **linear-recurrence layers** + sparse **Multi-Head Latent Attention (MLA)**, with
weight sharing to maximize depth.

**Linear state-tracking primitives compared:**

| Primitive | Complexity | Decay | Erase axis | Write axis | Parallel-solve |
|---|---|---|---|---|---|
| Mamba-2 | O(T) | scalar/channel | none | scalar/head | High (parallel scan) |
| Gated DeltaNet | O(T) | scalar | scalar/head | scalar/head | Moderate (sequential/WY) |
| Kimi Delta Attention (KDA) | O(T) | channel-wise | scalar/head | scalar/head | Moderate |
| Gated DeltaNet-2 | O(T) | channel-wise | channel-wise (key) | channel-wise (value) | Low (serial solver) |
| **CARVE (proposed)** | O(T) | channel-wise | channel-wise (key) | scalar/head | High (chunkwise WY-GEMM) |

**CARVE** ("Content-Aware Recurrent with Value Efficiency") restricts active erase strictly to the *key* axis,
keeping the intra-chunk coupling matrix uniform across value channels → parallel chunkwise GEMM. State update:
`S_t = (I − k_t (b_{c,t} ⊙ k_t)ᵀ) D_t S_{t−1} + k_t (w_t ⊙ v_t)ᵀ`, with channel-wise decay `D_t`, key-selective
content-aware erase gate `b_{c,t}`, scalar write gate `w_t`. The erase gate is a low-rank projection of a
chunk-level memory trace `m_c` (the averaged sequence-mixing output, read back from HBM to avoid reloading the
state matrix): `b_{c,t} = σ(b_x + U_b(m_c))`, `U_b` init to zero.

**MLA** blocks interleave in a **3:1 ratio** (3 CARVE : 1 MLA) for exact retrieval. Hidden state is projected to
a compressed latent `c_t^{KV}` (`d_c ≪ d_model`), cached, and keys/values reconstructed on the fly. **Decoupled
RoPE**: non-positional keys stay in latent space; position is applied to a dedicated low-dim channel; final
`k_t = [k_t^C ; k_t^R]`, `q_t = [q_t^C ; q_t^R]`.

**MASA (Matrix Atom Sharing in Attention):** deep cross-layer weight sharing — each layer's matrix is a linear
combination of `S` shared "dictionary atoms" (`W_l = Σ_s c_{l,s} D_s`) rather than identical tying. Reduces
sequence-mixer params ~66.7% while keeping representational diversity.

**Parameter budget (150,421,504 params, effective 30 layers):**

| Component | Spec | Params | Effective depth |
|---|---|---|---|
| Tied embedding | 3000 × 1024 | 3,072,000 | shared w/ output |
| Physical CARVE | 8× (d_model 1024, d_ff 2730) | 125,968,384 | 24 logical (3-folded) |
| Physical MLA | 2× (r_kv 192, r_q 192) | 21,381,120 | 6 logical |
| RMSNorm | d_model 1024 | 1,072,000 | per layer |

## 2. Logic-Dense Vocabulary + Symbolic Thought
**Hyper-minimalist 3,000-token, logic-exclusive vocabulary** — only code syntax, math operators, structural
delimiters, and abstract variable primitives (`var_0`, `const_0`, `def`, `return`, `assert`, `+`, `==`, …). A
128k vocab would cost `128000×1024 = 131M` params (87% of budget); 3k costs `3000×1024 = 3.07M`, reclaiming
**128M params** reallocated to `d_model` and `d_ff`.

**Compressed Symbolic Thought (CST):** raw verbose reasoning traces are compiled into symbol-only traces via a
regex/AST pipeline that (1) maps phrases to operators, (2) **strips conversational filler** ("let's see",
"wait", "ah"), (3) binds entity names to abstract variable indices, (4) verifies via AST. E.g. *"Now we add
three to the variable, which gives eight"* → `var_0 = var_0 + 3 => 8`.

## 3. Inference-Time Compute
- **Test-Time Training (TTT) layers:** treat the sequence-mixing state as fast weights `W_t` updated by a
  self-supervised gradient step during the forward pass; dynamic learnable inner LR; dual-form mini-batch
  (b=16) for >40% FLOPs utilization.
- **Value-Guided Tree Search (MCTS)** with PUCT selection (`c_puct → 0` at inference) + **Sequential Monte
  Carlo (SMC) steering**: track reasoning paths as weighted particles, resample via a Feynman-Kac potential
  `G_t = exp(λ · R_verify(s_t))` from an **external verifier** — backtrack dynamically without growing params.

## 4. Alignment — critic-free GRPO
**GRPO** (G=8 completions/prompt, rule-based reward, group-standardized advantage `Â_i = (r_i − μ)/(σ+ε)`), no
critic → half the VRAM. **Advantage-collapse mitigation:** when all rewards are equal (σ=0), **AVSPO** injects a
virtual sample (virtual success into an all-fail group, virtual failure into an all-pass group) to restore
gradient; **ISPO** adds sequence/token-level intrinsic rewards (Conditional IFD).

**Reward metrics:** sandboxed Docker execution (+1.0), AST parse stability (+0.4), **concise formatting (−0.5,
penalizes NL filler / "let's", "firstly", "the answer is")**, reasoning density = math-op ratio (+0.8).

**Curriculum:** 0–50B tok AST grammar/syntax anchoring; 50–80B single-step debugging + assertion (inject bugs,
reward correction); 80–100B multi-step reasoning, scale G, step-level value feedback (backtracking).

## 5. Architecture code (as provided)
A `UnifiedLogicHybridLayer` (PyTorch) fusing CARVE recurrence + MLA + a folded SwiGLU FFN, routed by
`layer_idx % 4` (3 CARVE : 1 MLA). *(Full code block reproduced from the proposal; omitted here for brevity —
see the share link. Key shapes: d_model=1024, d_ff=2730, h=16, d_h=64, r_kv=r_q=192.)*

## Unified specs
| Metric | Value |
|---|---|
| Active params | 150,421,504 |
| Effective depth | 30 layers (3-fold MASA) |
| Vocabulary | **3,000 logic-only tokens** |
| Max context | 1,000,000+ tokens |
| FLOPs efficiency | >40% (dual-form mini-batch TTT) |
| GRPO rollouts | G=8, critic-free |
