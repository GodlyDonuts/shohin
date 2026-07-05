# Shohin pretrain divergence — diagnosis & fix

## Symptom
During the 135M flagship pretrain, training loss held at ~1.0–1.2 for hundreds of steps,
then **cliffed from ~1.1 to ~3.0 within a single logging interval** and pinned at a finite
~3.0 (perplexity ~20) forever after. Not NaN, not a slow drift — a step function. Recurred
across at least five runs.

## What it was NOT (ruled out by controlled experiments)
| Hypothesis | Test | Result |
|---|---|---|
| Muon learning rate | swept 0.02 → 0.01 → 0.005 → 0.003 | diverged every time |
| AdamW learning rate | 3e-3 vs 1e-3 | diverged every time |
| Loaded optimizer momentum | `--fresh-opt` (reset Muon/AdamW state + rewarmup), job 678316 | diverged at the **same** step |
| Bad / corrupt data | full shard scan (`pipeline/scan_shards.py`, job 678310) | corpus clean & homogeneous; the exact shard at the divergence point decodes to normal English (GMAT/finemath) |

Key observation: divergence **timing tracks the data seed** (seed 1337 → ~step 7500; seed
777 → ~step 6400). The seed only controls shard order, so the trigger is tied to *which batch*
is seen *when* — but the batch content itself is clean.

## Root cause (confirmed by per-step grad-norm instrumentation, job 678334)
Logging gradient norm every step exposed the mechanism. Healthy baseline grad-norm ≈ 0.06–0.12.
At the cliff (seed 777):

```
step 6379  loss 0.96  gnorm 0.06     healthy
step 6380  loss 1.14  gnorm 1.01  ←  GRADIENT SPIKE ~15x baseline, but loss still NORMAL
step 6381  loss 1.15  gnorm 0.06
step 6382  loss 2.59  gnorm 0.30  ←  cliff: the step-6380 update has landed
step 6383+ loss ~3.1                 model wrecked
```

A specific (clean) batch produces a gradient whose **norm** spikes to ~15× baseline while its
**loss** stays normal. The trainer's guard only watched *loss*, so it applied this update — and
that single anomalous-direction update knocked the model into a degenerate basin.

Why the model *usually* survived: earlier grad-norm spikes at steps 6296 (gnorm 1.77) and 6349
(gnorm 1.51) coincided with *high loss*, so the loss-guard skipped them and the model recovered.
The fatal one (6380) was the spike whose loss happened to look fine. This also explains every
"failed fix": LR, momentum, and data were never the variable — the guard's blind spot was.

Muon amplifies the damage: its Newton–Schulz orthogonalization makes the update magnitude
roughly constant regardless of gradient size, so an anomalous-direction gradient is applied at
full strength.

## The fix
**Grad-norm pre-update guard** (`train/train.py`, `--gnorm-mult`, default 8.0): skip the
optimizer step when the current gradient norm exceeds 8× its running EMA — evaluated *before*
the update is applied, so a spike batch is dropped at the right moment (the loss-spike guard
only fires one step late, after the damage). With baseline ≈ 0.08 the threshold sits near 0.64,
far above normal variation (occasional 0.28) and well below the fatal spikes (1.0–1.77), so it
drops only the ~2–3% of steps that are genuine outliers and leaves normal training untouched.

Supporting hardening already in place:
- Loss-spike guard refuses any update with loss > 2× EMA (no capitulation cap — the old
  `skips < 5` cap was itself applying bad updates every 6th step and finishing off the model).
- Non-finite (NaN/Inf) updates always skipped.
- 300-consecutive-skip circuit breaker ends a run cleanly (best ckpt preserved, no diverged
  checkpoint saved) so a true divergence surfaces instead of burning GPU.

## Status
Fix deployed as job **678361** (resume from `ckpt_6000`, seed 777, the order that reliably
diverged at step 6400). Verification criterion: a `[skip:gnorm]` at ~step 6380 with loss holding
~1.0 through 6400+, then continued stable training. Best checkpoints (steps 4000/5000/6000,
loss ~1.0) preserved on-cluster and downloaded locally throughout.

## Follow-ups worth considering
- If rare residual spikes still slip through: add a Muon trust-ratio clamp (bound each update so
  ||update|| ≤ c·||weight||) as defense-in-depth. `--no-muon` bisection is wired and ready.
- Investigate *what* about the spike batches produces a 15× gradient (long low-EOS documents?
  dense LaTeX/number sequences?) — not required for stability, but could inform data packing.
