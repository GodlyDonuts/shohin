# R12 CDRL Neural Optimization Preregistration

**Status:** AUTHORIZED NEURAL OPTIMIZATION BOARD ONLY. This is not a reasoning,
workspace, Shohin-adapter, language-bridge, or primitive-novelty claim. It tests
Conjecture C from `R12_CONFLICT_DRIVEN_RESIDUAL_LOCALIZATION.md` on a fixed tiny
updater class with matched resources.

**Protocol:** `R12-CDRL-NEURAL-v1`

**Cluster:** Newton H100 (`normal`, one GPU). Isolated output under
`artifacts/r12/cdrl_neural_v1/`. Must not write flagship, ACW, or shared SFT
paths.

## 1. Frozen claim

> At equal parameters, labels, and optimizer updates, training a GRU residual
> predictor on Nerode cores (`core`) exceeds each of `full`, `rand`, and
> `hard` on depth-OOD exact state accuracy by at least **+5.0 percentage
> points** in the median over three locked seeds.

Failing any control by that margin is a reject. A tie or win only over `full`
is a reject.

## 2. Family and data

Heisenberg mod `M=5` with identity padding `P` (same algebra as the CPU
mechanics suite). Train histories: length `8..16` inclusive, formed by placing
`1..4` essential events from `{A,B,C}` and filling the rest with `P`. Held-out
depth-OOD: length `20..28` with the same essential-count band. Fit-IID eval:
held-out length `8..16` with disjoint RNG stream.

Seeds: `2026071601`, `2026071602`, `2026071603`.

Labels per arm: **12,288**. Updates: **2,400**. Batch size: **256**.

## 3. Arms (matched)

| Arm | Training histories |
|---|---|
| `full` | raw padded histories |
| `core` | lex-min residual-preserving cores |
| `rand` | random subsequence of length `\|core\|` (same indices RNG per example id) |
| `hard` | raw histories; after update 800, replace the next 1,600 updates' sampling
distribution by the top-quartile CE examples from a frozen probe pass |

All arms share identical GRU width (`h=64`), embedding (`d=32`), parameter
count, optimizer (AdamW lr `1e-3`, wd `0.01`), and loss (sum of three
coordinate CE heads).

## 4. Metrics and gates

Primary: depth-OOD exact state accuracy (`x,y,z` all correct).
Secondary (reported, not gated): fit-IID exact accuracy; mean coordinate accuracy.

Promotion (`advance=true`) only if for the median seed:

```text
acc_core - acc_full  >= 0.05
acc_core - acc_rand  >= 0.05
acc_core - acc_hard  >= 0.05
```

Otherwise `advance=false`. No threshold shopping after scores are read.

## 5. Explicit non-claims

- Not Shohin reasoning, SFT, or base-model improvement
- Not ACW Track S/C
- Not a new state ontology
- Not authorization to scale to language

## 6. Artifact layout

```text
artifacts/r12/cdrl_neural_v1/
  prereg_sha256.txt
  seed_{SEED}/
    arm_{ARM}/
      model.pt
      train_receipt.json
    eval.json
  decision.json
```

One attempt per seed/arm output path; refuse overwrite.
