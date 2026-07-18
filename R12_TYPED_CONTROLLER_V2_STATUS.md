# R12 Typed Controller v2 — in flight

**Job:** `691792` on Newton (evc TBD). Protocol `R12-TYPED-CONTROLLER-v2`.

## Setup

- Init: `train/sft_typed_controller_v1_200k_r1/sft_ep1.pt` (immutable flagship untouched)
- Data: seed `2026071701`, **8k train / 256 heldout** (heldout-first; rollout-prompt leak ban only)
- Mix weights: `atomic=0.20 rollout=0.25 resume=0.10 native_atomic=0.45`
- SFT: 2 epochs, no `torch.compile`
- Eval: `eval_typed_controller_v1.py` (fixed early-stop)

## Gates (locked)

| Gate | Threshold |
|---|---|
| Typed rollout | ≥35% |
| Typed − direct | ≥10pp |
| Atomic step | ≥50% (relaxed climb vs v1 70%) |
| Done rate | ≥80% |
| Beats v1 rollout | ≥21.4% (+5pp over 16.4%) |

## Prior failed submits

| Job | Issue |
|---|---|
| `691785` | train∩heldout prompt overlap |
| `691786` | generator dropped rollout/resume via prompt dedupe; cancelled |
| `691787` | synced wrong path; cancelled |
| `691790` | heldout fill failed (native/resume leak ban exhausted base_conversion) |

Remote smoke after fix: 60k train rows with `{atomic, native_atomic, resume, rollout}` all present.

## Artifacts (expected)

- `artifacts/r12/typed_controller_v2/{train,heldout,audit,eval_sft,decision}.json*`
- `train/sft_typed_controller_v2_from_v1_r1/sft_ep2.pt`
