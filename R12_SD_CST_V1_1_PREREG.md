# R12 SD-CST v1.1 Optimization-Correction Preregistration

**Status:** pre-board atomic optimization correction qualified on H100; no v1
development or confirmation bytes were opened; v1.1 source freeze pending

## Frozen diagnosis

SD-CST v1 job `693954` passed source, board, base, tokenizer, H100, and bf16
preflight on `evc36`, then stopped before writing a checkpoint because the motor
finished its fixed 2,000-update fit below 78/78 certificate exactness. It did not
open development or confirmation.

The failure is an optimization defect in the fully supervised finite atomic
component, not evidence about language compilation or recurrent reasoning. The
old implementation also set the recorded motor and reader seeds after their
modules had already been initialized. A CPU replication over independent
explicit initializations found:

| Component schedule | Exact final seeds | Worst final cells |
|---|---:|---:|
| motor AdamW, lr 0.025, 2,000 updates | 30/32 | 76/78 |
| motor AdamW, lr 0.003, 1,000 updates | 32/32 | 78/78 |
| reader AdamW, lr 0.04, 1,200 updates | 63/64 | 12/18 |
| reader AdamW, lr 0.005, 500 updates | 64/64 | 18/18 |

The high-rate arms frequently reached exact fit early and then left it after the
loss was already near floating-point zero. The correction therefore reduces
both learning rates and budgets rather than checkpoint-selecting an early
iterate.

## Sole authorized changes

1. Reset each parameterized motor/reader child from its recorded component-local
   seed immediately before constructing its optimizer.
2. Fit the motor for exactly 1,000 full-table AdamW updates at learning rate
   `0.003`, betas `(0.9, 0.95)`, and zero weight decay.
3. Fit the reader for exactly 500 full-table AdamW updates at learning rate
   `0.005`, betas `(0.9, 0.95)`, and zero weight decay.
4. Require a pre-board H100 canary over at least 64 fresh component seeds to end
   at 78/78 motor and 18/18 reader exactness for every seed.
5. Version checkpoint, board, evaluation, assessment, protocol, and access-ledger
   schemas as v1.1 and draw a new board seed and training seed only after source
   commit.

No architecture, parameter count, compiler data, compiler optimizer, board task,
surface family, split size, evaluator, threshold, causal intervention, control,
or claim boundary changes. The complete system remains 134,306,714 parameters.
The complete v1 preregistration remains incorporated by reference except for the
three replaced optimization settings above.

Failure of the H100 multi-seed canary rejects these settings before board
generation. Passing it admits one new clean-HEAD board. A v1.1 neural failure
after development access closes that board without rescore, exactly as in v1.

## Pre-board H100 result

Job `693956` completed on NVIDIA H100 PCIe `evc22` in 1m44s using fresh seeds
`2073833426` through `2073833489`. All 64/64 seeds ended at exactly 78/78 motor
cells and 18/18 reader cells. The worst final motor loss was
`2.5690799247968243e-6`; the worst final reader loss was
`1.3245475827261544e-7`. The report SHA-256 is
`472ff05ba4ef4dc4cb3956d8d69574f4b2ada8663224fa94c336a3c9de156433`.
The canary had no code path or argument for a base checkpoint, tokenizer,
source board, development split, or confirmation split. These settings are
therefore admitted for one post-source-commit fresh board.
