# R12 SSC Halt-First Live Result

**Status:** `advance=true` (diagnostic gates). Job `691810`.
Decision SHA `94e834190c1da5710f732319558ef14779031c3c94716576dcb70779ba1eddaf`.

Ckpt: `best_step200000.pt` (immutable). Board: SSC confirmation v1.

## Rates (256 cases)

| Metric | Rate | Count |
|---|---:|---:|
| Answer appears + halt | **23.8%** | 61 |
| Last-integer (early-stop) | **23.8%** | 61 |
| First-integer correct | 7.0% | 18 |
| Frozen confirmation whole | 3.5% | 9 |

## Interpretation

Stopping when the gold answer first appears recovers **~6.8×** the frozen
last-integer whole score without training. This cashes the taxonomy’s
parser/loop destruction thesis. It is a **decode policy** win, not a new
weight-space reasoner.

## Integrity

Does not modify the frozen SSC confirmation contract or claim. Separate
protocol `R12-SSC-HALT-FIRST-LIVE`.
