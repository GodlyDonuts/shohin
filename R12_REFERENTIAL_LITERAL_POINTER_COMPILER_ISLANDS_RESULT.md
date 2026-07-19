# R12 Complete-Compiler Parameter-Islands Result

**Protocol:** `r12_referential_literal_pointer_compiler_v1_3_islands_development`
**Decision:** **PARTIAL MECHANISM WIN; FAIL FROZEN PROMOTION GATE**

## Run and custody

Job `692992` completed on a verified H100 PCIe on `evc28` with exit code zero in 10m10s. It used
the frozen one-epoch / 1,514-update v1.1 training schedule, seed `2026071805`, 8,658,701 trainable
compiler parameters, and 133,740,365 total parameters. Shohin remained frozen.

| Item | SHA-256 |
|---|---|
| initial adapter state | `1e9301b55eecf1b1be4598a2e0e3a25bb09d989c2a8d7f37fc09dc310d599abc` |
| final adapter state | `93d3aef45e818c61f9bb7857df45cb3086ba1b7e2b4d6478049913f3dc0516b3` |
| adapter file | `63f735d7fa275b8347e25d3645914fc24d08e24b0c412f27c292a561ebb65af2` |
| development result | `873254e5dde203ea89f2b55b68948ec64f13a3e0536e9d38281f88ad0539205d` |
| Slurm log | `d017fbd983100d37303f5dfb5d05d9d84693501644379b114af43f4cb8564d99` |

The adapter, result, and log are hash-matched between Newton and the Mac. Confirmation remained
absent from Newton and access is zero.

## Frozen gates

| Metric | Gate | Result | Pass |
|---|---:|---:|---:|
| operation-kind accuracy | >=90% | **61.328%** | no |
| initial-state joint exact | >=45% | **45.996%** | yes |
| answer accuracy | >=45% | **43.311%** | no |
| semantic-program exact | >=30% | **23.438%** | no |
| full pointer exact | >=10% | **10.645%** | yes |
| paraphrase answer | >=45% | **59.766%** | yes |
| canonical answer | >=35% | **32.422%** | no |
| canonical + paraphrase both pointer-exact | >=1/512 | **0/512** | no |

Three of eight gates pass. The arm is not promoted and v1.1 confirmation remains sealed.

## Surface result

| Surface | Kind | Initial joint | Program exact | Answer |
|---|---:|---:|---:|---:|
| canonical | 46.289% | 45.508% | 15.820% | 32.422% |
| order twin | 53.711% | 44.336% | 19.336% | 41.016% |
| binding twin | 46.289% | 44.922% | 15.820% | 40.039% |
| paraphrase | **99.023%** | **49.219%** | **42.773%** | **59.766%** |

The separate semantic island fixes v1.2's optimization collapse: training kind loss reaches zero,
and exposed-development answer accuracy rises from 18.994% to 43.311%. It also exceeds the v1.1
answer result by 13.916 percentage points and lifts initial-state joint exact by 27.148 points.

The failure is now lexical and surface-specific. The island transfers almost perfectly to the
unseen `front/rear` paraphrase renderer but not to the unseen `earlier/later` canonical renderer.
Two fixed training renderer families do not identify renderer-invariant lexical semantics, even
with 96,000 examples.

## Scientific consequence

Parameter islands are supported as an optimization mechanism: structural and semantic paths can
learn concurrently without the v1.2 chance-classifier collapse. They are not yet a complete
compiler result. Repeating seeds or tuning against this exposed development split is unauthorized.

The next experiment must change the educational support, not the H100 budget: train on many
factorized combinations of intro frames, argument orders, direction lexemes, operation ordinals,
distractor placements, and query frames; evaluate unseen combinations separately from truly unseen
lexemes; and compare v1.3 against favorable ordinary parser and oracle controls.
