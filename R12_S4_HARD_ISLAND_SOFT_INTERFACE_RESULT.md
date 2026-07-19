# R12 S4 Hard-Island / Soft-Interface Result

## Decision

**Confirmed. Promote S4 v5 as the best bounded native S4 reasoning baseline.**

All preregistered development and confirmation gates pass. The hash-bound promotion manifest is
`artifacts/r12/s4_hard_island_soft_interface_v5.promoted.json`.

## Custody

- Source/preregistration freeze: commit `e9a962b`.
- Board freeze: commit `2101832`.
- Production development seed: `14465012970954709091`.
- Board: 2,048 rows / 512 matched groups, depths 3--8, maximum 340 tokens.
- Data SHA-256: `4d43b050892fc26a712e2c97414e84da3721c7949b98d8862b239e6b2f051c7a`.
- Board report SHA-256: `8b9461bc86771f50b4f72b58bda7e09286ab0ac3830175cc0f28540279db434f`.
- Safe archive SHA-256: `eec8ec5e277dd39791ce1816d175d979f465363718dcf9a1d429b242a151efe1`.
- Every exact/13-gram/name/factor/roster-token-multiset gate passes against source data and closed
  v2--v4 boards.
- Development access: one. Confirmation access: zero.
- Serial baseline/treatment/control/assessor job: `693177`, completed on `evc24` in 83 seconds.

## Qualified development result

| Arm | Count | Roster | Query | Exact program | Exact state | Correct answer |
|---|---:|---:|---:|---:|---:|---:|
| Frozen S4 v1 | **100%** | 93.70% strict | 93.70% strict | 1919/2048 = 93.70% | 93.70% | 93.70% |
| **S4 v5 hybrid** | **100%** | **2022/2048 = 98.73%** | **100%** | **1985/2048 = 96.92%** | **1996/2048 = 97.46%** | **2009/2048 = 98.10%** |
| V5 + roster rotation | 100% | unchanged | 100% | **0/2048** | 0.68% | 5.42% |
| V5 + event-region rotation | 100% | unchanged | 100% | **0/2048** | 13.67% | 26.51% |

V5 improves exact programs by **3.22 percentage points absolute** over its identical-board v1
baseline. Exact program accuracy by depth 3--8 is 97.38%, 97.38%, 98.24%, 97.06%, 96.76%, and
94.71%; every preregistered depth gate passes. Surface-family accuracy is 96.68--97.46%.

The two zero-program interventions establish that neither hard islands nor set carriers are merely
diagnostics. The successful decomposition is: model-owned kind clock -> monotone local region ->
complete hard entity/literal islands -> vocabulary-aligned soft roster identity -> soft query ->
locked exact S3 execution. No v5 parameter was trained.

Baseline, treatment, assessment, and log SHA-256 values are respectively
`103cc7e07d6be5bb355e8944ffc565c9cf7ba06941413044aac9546f47d87986`,
`ca3cb11f0eb6871e1ce2b64efb94a380deffdbbbb72ef0064a164b5d95216ece`,
`41a2dd2eab37c1976803d49e36f1a4ae35b62e8568ebefc3159c118383ab2eb5`, and
`c4e874e16c469387adeb5c66702d3174b32ac3fccea560624dcb042a8f8a7ac7`. The assessor decision is
`qualify_s4_v5_for_fresh_confirmation`.

## Confirmation

The sole disjoint confirmation board uses seed `14809014609581254328`, data SHA
`b16534b3c41d21737370f0eb852cb6c53d75e81d661d6d9592927709551a08cf`, and report SHA
`ce0e2671c8dcd07b5b789798da0325b94066087ce4222082267919f57afdb261`. Job `693178` completes the
unchanged serial protocol on `evc24`.

| Confirmation arm | Exact program | Exact state | Correct answer |
|---|---:|---:|---:|
| Frozen S4 v1 | 1913/2048 = 93.41% | 93.41% | 93.41% |
| **S4 v5 hybrid** | **2003/2048 = 97.80%** | **2015/2048 = 98.39%** | **2020/2048 = 98.63%** |
| V5 + roster rotation | **0%** | 0.68% | 3.76% |
| V5 + event-region rotation | **0%** | 12.50% | 26.12% |

Confirmation program accuracy by depth 3--8 is 97.09%, 98.55%, 98.82%, 96.18%, 98.82%, and
97.35%. Surface accuracy is 97.66--98.05%. The assessor records
`confirm_s4_v5_hard_island_soft_interface` at SHA-256
`41fb41a7aa70139c5a05c321b949bb5ad245e26d1506445910cdbb542e34b407`.

Confirmation baseline, treatment, and log SHA-256 values are
`0293aeeca82660a827c6197fd30b84512d66084925b364cf64152bc39442aaa6`,
`d4d1b2b928aec723f7421147d4d14b386ebcbd014db4d2d2db83636b989c44f0`, and
`cb2d179110c4ef4c31f01d7f1a99748d11ad0235e76d1c521c37f9583a6341b2`.

## Claim boundary

This is a confirmed known-atom parser/executor result and promoted bounded baseline. It is not
unseen operation semantics, open-ended planning, learned halt, free-form reasoning, public benchmark
improvement, or a novelty claim.
