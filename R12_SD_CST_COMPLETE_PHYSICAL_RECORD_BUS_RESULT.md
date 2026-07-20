# R12 SD-CST Complete Physical-Record Front-End Result

**Decision:** `reject_or_revise_complete_local_front_end`

**Claim boundary:** consumed-training interface mechanics only. Development and
confirmation were not addressed. This is not a native-reasoning or fresh-language
result.

## 1. Immutable contract

- scientific source commit:
  `6294ea90f8b9e308edde9cad4d4b276c729961ae`;
- execution/seed receipt commit:
  `205f6b732d9d6dfa2518830d5ea2f470e3089faa`;
- sole seed: `4564290739472553435`;
- consumed train SHA-256:
  `b7756dbf8d4401dbc5fb897dee53f68758e27200b1ce0d2387631f2f0205ec25`;
- joint-parent checkpoint SHA-256:
  `4b842e4c2d0d608c32f0fd113b404866be7269676084cdac9b1a00d43cdd298d`;
- physical-record parent SHA-256:
  `89ab7d7417918e72da60028e6d5936908a3ee29c0981f5fdac9dc385c3099419`;
- sole Slurm job: `694199`, H100 `evc37`, completed cleanly in 6m19s;
- scored accesses: development `0`, confirmation `0`.

Runtime preflight verified the clean exact source, all input hashes, bf16 H100
allocation, and the parameter certificate before optimization.

## 2. Exact system

| Quantity | Count |
|---|---:|
| immutable Shohin trunk | 125,081,664 |
| complete compiler | 66,426,124 |
| new trainable completion parameters | 594,435 |
| all local-front-end parameters, frozen plus new | 11,701,265 |
| categorical motor | 19,206 |
| categorical reader | 835 |
| **complete deployed system** | **191,527,829** |
| **strict-200M headroom** | **8,472,171** |

Only ten `local_*` tensors trained. All 88 retained `record_*` tensors and every
joint-parent/executor tensor remained frozen under a byte-identical excluded-state
digest. The complete forward succeeds when both inherited global encoders are
replaced by raising sentinels.

## 3. Result

The sole run completed the fixed two epochs / 3,000 updates. The local late-query
path converged to 100% query and query-pointer accuracy on every fit and held-out
renderer. The frozen physical record bus also retained 100% line, event-pointer,
kind, and amount accuracy. The declaration-local path did not converge:

| Minimum across held-out renderers | Initial | Endpoint |
|---|---:|---:|
| query | 34.4% | **100%** |
| query pointer | 0% | **100%** |
| event pointer | 100% | **100%** |
| binding pointer | 0% | **5.15%** |
| initial-occurrence pointer | 0% | **0%** |
| initial state | 15.6% | **15.0%** |
| identity | 0.1% | **82.2%** |
| complete packet | 0% | **13.45%** |

Fit is nearly identical: minimum fit binding pointer is 5.24%, initial pointer
is 0%, and packet is 13.81%. This is not renderer holdout overfitting. Endpoint
binding-address and initial-entity-address losses remain 4.545 and 4.600, close
to a uniform distribution over the declaration record. More epochs are not an
admissible interpretation or repair.

## 4. Mechanistic audit

All declaration tensors changed materially: the declaration-query table moved
2.573 in L2 norm and its query projection moved 4.686. On a held-out family at
the endpoint, their gradients remain large (3.189 and 5.152), while the query
path gradients are approximately zero after reaching exactness. Optimization is
therefore active rather than disconnected.

The declaration queries are forced to address six declaration occurrences
through `record_entity_key`, a frozen key projection learned only for event-line
entity extraction. That projection is sufficient for the retained event bus but
does not expose a usable basis for declaration/initial occurrence addressing.
The result localizes the remaining failure to declaration address geometry. It
does not justify retraining the successful record encoder, increasing depth, or
opening a scored split.

## 5. Gate accounting

Six gates pass: held-out query, query pointer, event pointer, frozen-state
preservation, strict-200M size, and zero scored access. The fit/held-out packet,
initial-state, declaration-pointer, initial-pointer, and combined
kind/identity/amount gates fail. The preregistered decision is therefore
`reject_or_revise_complete_local_front_end`.

The smallest admissible successor is a separate training-only contract that:

1. loads and freezes the v1 endpoint's exact local query path;
2. retains and freezes the perfect physical event bus;
3. resets the two failed declaration-query tensors;
4. adds one declaration-local key projection; and
5. trains only those three declaration tensors under the same consumed-row
   partitions, schedule, preservation checks, and absolute gates.

That successor is an optimization/representation repair, not a fresh-board or
reasoning claim.

## 6. Preserved artifacts

- local checkpoint:
  `train/sd_cst_complete_physical_record_bus_pilot_4564290739472553435/compiler.pt`;
- local report:
  `train/sd_cst_complete_physical_record_bus_pilot_4564290739472553435/report.json`;
- committed report copy:
  `artifacts/r12/sd_cst_complete_physical_record_bus_pilot_4564290739472553435.report.json`;
- Newton output:
  `/lustre/fs1/home/sa305415/shohin_sd_cst_complete_physical_record_bus_pilot_4564290739472553435/`;
- checkpoint SHA-256:
  `30b75305031b1e2f67a24f98b4907d2d65bc847310ea406d25f34c7b9611e1b4`;
- report SHA-256:
  `c06348d53d5c9fd3b4fa79e6c9eb9e3720b106834d0326c1d1d979db67c8fff2`.

Local and Newton hashes match exactly.
