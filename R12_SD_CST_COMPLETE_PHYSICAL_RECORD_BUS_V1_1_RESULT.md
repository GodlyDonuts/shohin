# R12 SD-CST Complete Physical-Record Front-End v1.1 Result

**Decision:** `reject_declaration_key_repair`

**Claim boundary:** consumed-training declaration-address mechanics only. No
development or confirmation bytes were addressed, and this is not a native
reasoning result.

## 1. Immutable contract

- scientific source commit:
  `b93b17b3ee5c096509cd1ab0d903ef7a9287d3a3`;
- execution/seed receipt commit:
  `8b1e05d52d94d347350e91780643c501a8ec492e`;
- raw beacon: `14330060956843215829`;
- signed-safe seed: `5106688919988440021`;
- consumed train SHA-256:
  `b7756dbf8d4401dbc5fb897dee53f68758e27200b1ce0d2387631f2f0205ec25`;
- complete-local v1 parent SHA-256:
  `30b75305031b1e2f67a24f98b4907d2d65bc847310ea406d25f34c7b9611e1b4`;
- sole job: `694203`, H100 `evc22`, completed cleanly in 8m09s;
- scored access: development `0`, confirmation `0`.

The clean Newton capsule was at the exact execution receipt. All three compiler
parents and the consumed train file hash-matched before Slurm admission. Runtime
preflight verified bf16 H100 allocation and the exact parameter certificate.

## 2. Exact system

| Quantity | Count |
|---|---:|
| immutable Shohin trunk | 125,081,664 |
| complete compiler | 66,573,580 |
| trainable declaration repair | 297,216 |
| categorical motor | 19,206 |
| categorical reader | 835 |
| **complete deployed system** | **191,675,285** |
| **strict-200M headroom** | **8,324,715** |

Only the six-query declaration table, declaration-query projection, and new
declaration-key projection trained. The exact eight-tensor v1 query endpoint,
88-tensor physical record bus, joint parent, matcher, tape, executor, motor,
reader, and Shohin trunk remained frozen under a byte-identical digest.

## 3. Result

| Minimum held-out renderer | V1 | V1.1 |
|---|---:|---:|
| binding pointer | 5.15% | **61.00%** |
| initial-occurrence pointer | 0% | **0%** |
| initial state | 15.00% | **48.00%** |
| identity | 82.20% | **98.90%** |
| complete packet | 13.45% | **47.45%** |
| query / query pointer | 100% / 100% | **100% / 100%** |
| event pointer / kind / amount | 100% | **100%** |

The dedicated declaration key is causally useful as a repair package: it adds
55.85 points to minimum binding-pointer exactness and 34.0 points to complete
packets while every successful frozen path remains exact. Fit and heldout are
again close. Minimum fit packet is 47.975%, and initial-pointer exactness is
0% on two fit renderers and below 0.41% on the others. This is not a parity
generalization failure.

The remaining error is sharply asymmetric. Binding-address loss falls to
4.043, identity is nearly exact, and initial state improves. Initial-occurrence
address loss remains 4.418 and no held-out renderer exceeds 0.55% all-three
pointer exactness. A shared declaration key can expose declaration names but
does not provide reliable occurrence/order selection for the repeated initial
list.

## 4. Gate accounting

Six of twelve gates pass: query, query pointer, event pointer, excluded-state
preservation, strict-200M size, and zero scored access. Fit packet, heldout
packet, initial state, binding pointer, initial-occurrence pointer, and the
combined kind/identity/amount gate fail; identity misses its 99% minimum by
0.1 point. The exact decision is `reject_declaration_key_repair`.

Do not add epochs or widen v1.1. The next decision must follow a read-only
per-slot confusion audit over the same consumed heldout rows. In particular,
it must distinguish whether initial queries select declaration occurrences,
the wrong repeated initial occurrence, or non-entity bytes before proposing
separate key banks, role-conditioned nonlinear keys, or a local occurrence
parser.

## 5. Preserved artifacts

- local checkpoint:
  `train/sd_cst_complete_physical_record_bus_v1_1_pilot_5106688919988440021/compiler.pt`;
- local report:
  `train/sd_cst_complete_physical_record_bus_v1_1_pilot_5106688919988440021/report.json`;
- committed report copy:
  `artifacts/r12/sd_cst_complete_physical_record_bus_v1_1_pilot_5106688919988440021.report.json`;
- Newton output:
  `/lustre/fs1/home/sa305415/shohin_sd_cst_complete_physical_record_bus_v1_1_pilot_5106688919988440021/`;
- checkpoint SHA-256:
  `46697b3942fdfd2edfec06cea6cb119ad507adcdee4a99330fd06bc79e5b3e88`;
- report SHA-256:
  `73d470bb1a6b9331b3d46fcd56d3e6acb500c768717cc829ff0f567092babcaf`.

Local and Newton hashes match exactly.
