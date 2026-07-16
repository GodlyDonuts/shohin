# R12 Raw-300k Future-Jacobian Workspace Longitudinal Preregistration

**Status:** **COMPLETED, SEMANTIC GATE FAILED.** Jobs `690028`, `690030`, and
`690031` executed the frozen repeat. The selected raw-300k future-Jacobian
readout has zero top-10 and zero top-100 accuracy on all 2,304 language/full
targets, so no coordinate swap or raw-workspace mechanism is authorized. See
`R12_JACOBIAN_WORKSPACE_LONGITUDINAL_RESULT.md`.

## 1. Question

The raw-200k exact future-Jacobian map was reproducible but semantically
unusable on the frozen referential board: future-Jacobian MRR exceeded the
immediate-logit MRR in the rank tail, while both methods recovered zero of
2,304 language/full concept targets in the top 100. No coordinate swap was
authorized.

The July 2026 global-workspace study explicitly leaves model-size and
pretraining-emergence scaling unresolved. Raw 300k therefore receives one
strict longitudinal repeat:

> Under identical code, prompts, seeds, layers, target board, and decision
> gates, did another 100k pretraining updates create a semantically readable
> future-causal workspace in Shohin?

This is a measurement of an existing method, not an R12 invention.

## 2. Immutable model and inputs

| Item | Frozen value |
|---|---|
| raw-300k checkpoint | `train/flagship_out/ckpt_0300000.pt` |
| checkpoint SHA-256 | `211d6b2cddf0c2cf8b12cb0b2d73f9c4440d85f6f531018080c8afd35b2f66a6` |
| tokenizer SHA-256 | `87532df5c121753de3b29194e1f9e3de47986d3f5359548fdf93606773a233d4` |
| lens prompts SHA-256 | `fbe8687d618550d2251b397d436abb30a990d5f0b4e7c25cfe85c7265aa251d8` |
| evaluation board SHA-256 | `b9106b3233c62592dba5f244d5fdec5474d56c813b6d39dc0fe8ee77a441039c` |

The checkpoint is read-only and model-only. No optimizer or training state is
needed because this diagnostic freezes every parameter.

## 3. Exact repeat contract

Two independent averaged future-residual Jacobians use the existing committed
implementation `train/jacobian_workspace.py` with:

```text
source layers = 5,9,13,17,21,25,28
target layer = 29
prompt count = 8 per fit
prompt seeds = 20260714 and 20260715
maximum sequence length = 128
skip first = 16
dimension batch = 8
```

The same prompt seeds intentionally reproduce the raw-200k prompt samples so
the checkpoint comparison is paired. The two samples remain disjoint from each
other. The existing committed `train/eval_jacobian_workspace.py` scores the
same 896 examples and selects only among layers 13, 17, 21, and 25 by the
frozen language/full MRR-gain rule.

Fresh outputs are exactly:

```text
artifacts/diagnostics/jacobian_workspace_raw300k_p8_v1.pt
artifacts/diagnostics/jacobian_workspace_raw300k_p8_v2.pt
artifacts/diagnostics/jacobian_readout_raw300k_p16_v1.json
```

No raw-200k development row, 300k transcript, or output token may alter the
implementation, layers, seeds, labels, selected-layer rule, or gates.

## 4. Frozen gates

The original gates are retained without relaxation:

1. the two raw-300k lens matrices have whole-matrix cosine at least 0.90 at
   every fitted layer;
2. at the predeclared selected layer, future-Jacobian language/full MRR is at
   least 1.25 times immediate-logit MRR;
3. future-Jacobian language/full top-10 accuracy exceeds immediate-logit
   top-10 accuracy by at least 10 percentage points.

Only all three permit a separately preregistered causal swap. Top-100 or MRR
improvement without the top-10 gate is a semantic failure, exactly as at 200k.

The longitudinal report also records, without changing the decision:

- raw-200k versus raw-300k MRR, top-10, top-100, and median rank;
- cross-checkpoint matrix cosine and right-singular-subspace overlap by layer;
- exact artifact hashes and Slurm logs.

## 5. Interpretation boundary

- **Fail:** raw next-token pretraining did not make this exact J-lens semantic
  workspace readable by 300k. Close raw-Jacobian swaps at this checkpoint.
- **Pass:** only authorizes a fresh, bidirectional, donor-swap causal
  preregistration with zero, shuffled, norm-matched, and non-Jacobian controls.
- **Either result:** does not establish hidden thinking, autonomous reasoning,
  context compaction, or a new mechanism.

The paper reports the workspace primarily in much larger production models and
states that smaller models may have a smaller, less reliable, or absent one.
Shohin must earn the causal premise rather than inherit it from model scale.
