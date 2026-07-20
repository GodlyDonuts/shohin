# R12 SD-CST Joint Renderer Compiler Post-Hoc Audit

**Status:** completed; exact report preserved

**Job / node / elapsed:** `694110` / H100 `evc27` / 10m44s

**Report SHA-256:**
`318b64584b3c1852a9e16025755b0595c0e78dc853411e466218540cd4f66b68`

**Purpose:** distinguish dead optimization from cascading record-address errors
after the preregistered joint control failed complete-record gates

This is an explicitly post-hoc diagnostic over the already-consumed renderer
orbit. It is not a score, a model-selection gate, or a reasoning claim. It may
read only the consumed training rows, exact rejected orbit checkpoint, and exact
rejected joint checkpoint. Development and confirmation are unreachable.

The audit reports per-slot rather than all-slots-at-once exactness for line
address, event address, kind, amount, and identity at exact initialization and
the rejected endpoint. Two model-state interventions are diagnostic only:

1. uniform pooling over the gold source-line span before the unchanged native
   slot decoder, measuring kind and amount; and
2. uniform pooling over the gold event-name span before the unchanged frozen
   fingerprint matcher, measuring identity.

These interventions cannot establish capability. They localize whether errors
come from line addressing, event addressing, or downstream field readout. The
complete deployed system remains 179,826,564 parameters, strictly below 200M.
No training, rescore, threshold, development read, or confirmation read is
authorized by this audit.

## Exact Result

Aggregate rates across all four held-out renderer combinations are:

| Field | Exact initialization | Rejected endpoint | Delta |
|---|---:|---:|---:|
| Physical source-line address per slot | 10.896% | 42.029% | +31.133 pp |
| Event-name address per active slot | 6.555% | 25.466% | +18.911 pp |
| Event kind per slot | 40.719% | 55.731% | +15.013 pp |
| Amount per active slot | 49.836% | 68.000% | +18.164 pp |
| Identity per active slot | 41.168% | 50.325% | +9.157 pp |

The endpoint's fit rates are 42.030%/25.468%/55.796%/68.242%/50.135% in
the same order. Fit and held-out behavior are therefore nearly identical; the
failure is not orbit-combination overfitting.

At the endpoint, uniform pooling over the gold source line raises held-out kind
from 55.731% to 73.641%, but changes amount only from 68.000% to 68.134%.
Uniform pooling over the gold event-name span raises identity from 50.325% to
exactly 100% over all 56,000 held-out active slots. Thus the frozen fingerprint
matcher and declaration binding are sufficient once the event address is
correct. Line localization explains part, but not all, of kind error; amount
requires better local field extraction rather than only a better line pointer.

## Decision

The audit closes dead-gradient and renderer-overfit explanations. It supports a
distinct factorization: independently encode delimiter-bounded physical
records, predict their local fields, then perform model-logit-only one-to-one
record assignment into the categorical tape. Merely widening, deepening, or
extending the failed independent global-query path remains forbidden.

Exact evidence is committed at
`artifacts/r12/sd_cst_renderer_native_joint_audit_219fd41/report.json`.
