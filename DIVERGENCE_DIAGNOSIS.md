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

## UPDATE — the grad-norm spike was a correlate, NOT the cause
Deploying the grad-norm pre-update guard (job 678361) DID skip the spike — the log shows
`[skip:gnorm] step 6380 gnorm 0.97` — **and the model diverged anyway at the same step 6382**,
with near-identical loss (2.59) to the unguarded run. The guard fired correctly on every spike
(6296/6338/6349/6357/6380) yet the cliff persisted. Therefore the ~15× gradient spike is a
*symptom that co-occurs near the cliff*, not the trigger. The actual damaging update is step
**6381**, which has a completely normal gradient norm — so the destabilizing step is invisible to
any grad-norm threshold. A normal-magnitude gradient producing a catastrophic update points at
**Muon's orthogonalization** (Newton–Schulz turns any gradient into a full-magnitude update, so a
benign gradient in a bad/ill-conditioned direction lands at full force) — or, less likely, a
forward/backward numerical issue on that specific batch.

## Muon bisection — ALSO ruled out Muon
Job 678367 (`--no-muon`, pure AdamW) cliffed at the **same** step 6382 with the same pattern. So the
divergence is fully **optimizer-independent** (Muon AND AdamW). The trainer's loss-guard then froze
the model, which did not recover — and the frozen, essentially-healthy model scored loss ~3.0 on
every subsequent batch. fp32 loss was already in place (`model.py`: `logits.float()` before
cross-entropy), so numerics were not it either.

## ROOT CAUSE (confirmed) — 200M-token monodomain blocks + domain-shift shock
Decoding the exact divergence batch (`pipeline/peek_batch.py`, token 199.8M in the seed-777 order)
put it at the **end of `code_python/shard_00063`** — and shard 0 in that order is a **200M-token shard
of pure Python code**. The loader shuffled *shard order* but read each shard's 200M tokens
**contiguously**, so from the resume point the model trained on **200M consecutive tokens of pure
code** (steps 6001–6381), over-specialized to code, then hit the finemath boundary at step 6382 and
took a loss shock (2.67) it couldn't absorb. This explains everything:
- optimizer/LR/momentum-independent → it's the data schedule, not the optimizer;
- the frozen model scores ~3.0 on *math* → it's code-specialized, not damaged;
- timing tracks the seed → the seed places the domain boundary at different steps (1337→~7500,
  777→~6400);
- decoded data is "clean" → because it *is* clean; the problem is the ordering, not the content.

## FIX — domain-interleaved dataloader
`train/data.py` rewritten to keep one continuous read stream **per domain** and assemble every batch
round-robin across domains, so each batch is a code+math+web blend and the model never sees a
200M-token monodomain block. This removes the domain-shift cliff and is standard good practice
(mixed-domain batches train better — also a quality win for the final model). Verified locally: every
batch blends all domains, correct shapes and x/y shift. Deployed as job **678379** (resume ckpt_6000,
normal Muon config). Verification: stable loss with NO cliff well past the old danger zone.

## Hardening retained (good regardless)
- Grad-norm pre-update guard (`--gnorm-mult`, default 8×EMA) — skips genuine gradient outliers.
- Loss-spike guard with no capitulation cap + non-finite skip + 300-skip circuit breaker.
- Per-step grad-norm logging.
- `--fresh-opt`, `--no-muon` diagnostic switches.

## Hardening retained regardless (not the fix, but keep)
**Grad-norm pre-update guard** (`train/train.py`, `--gnorm-mult`, default 8.0): skips a step whose
gradient norm exceeds 8× its running EMA, *before* applying it. Baseline ≈ 0.08 → threshold ≈ 0.64,
so it drops only the ~2–3% genuine outliers and leaves normal training untouched. Good robustness
to keep, even though it does not by itself resolve this divergence.

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
