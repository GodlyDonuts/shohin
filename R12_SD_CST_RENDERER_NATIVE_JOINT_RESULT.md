# R12 SD-CST Joint Renderer-Memory Program Decoder Result

**Decision:** `reject_or_revise_renderer_native_joint_control`

**Claim boundary:** consumed training rows only; favorable conventional compiler
control, not a fresh generalization, novelty, or reasoning result

## Frozen Run

| Item | Value |
|---|---|
| Source commit | `102ab3f5172e9a6c86d1045d61c0e1ce66f159e2` |
| Seed | `6795424534800881443` |
| Newton job / node | `694099` / `evc36` H100 |
| Updates / elapsed | 3,000 / 358.651 seconds |
| Complete / trainable parameters | 179,826,564 / 32,782,853 |
| Strict-200M headroom | 20,173,436 |
| Development / confirmation accesses | `0 / 0` |

Exact input SHA-256 values:

- consumed training rows: `b7756dbf8d4401dbc5fb897dee53f68758e27200b1ce0d2387631f2f0205ec25`;
- parent Renderer-Orbit checkpoint: `2e019b81406bb90e539665271c9893a0e568e0177396243ac427f17d8ca51eca`.

The run trained exactly the 35 native decoder tensors plus byte/position
embeddings, all eight orbit encoder layers, and orbit normalization. Every
query motor, binding module, packet head, executor, motor, reader, and Shohin
trunk tensor remained frozen. The excluded-state digest remained identical.

## Outcome

Minimum exact rates across the four held-out renderer combinations are:

| Field | Exact rate | Frozen gate |
|---|---:|---:|
| Initial state | 100.00% | 95% |
| Declaration pointer | 100.00% | 99% |
| Initial-occurrence pointer | 100.00% | 99% |
| Query | 100.00% | 99% |
| Query pointer | 100.00% | 99% |
| Kind | 0.30% | 95% |
| Active identity | 0.15% | 90% |
| Amount | 1.45% | 95% |
| Source-line pointer | 0.00% | 95% |
| Event-occurrence pointer | 0.00% | 90% |
| Complete packet | 0.00% | 80% |

This is not merely renderer holdout failure. All four fit renderers also have
0% complete-record source-line pointers, event pointers, tapes, and packets.
Their complete-record kind, identity, and amount exact rates remain very low;
these aggregate rates do not by themselves imply per-slot chance. Event-support rate rises from
5.963% after epoch one to 20.819% after epoch two, while line loss remains
5.304 and kind loss 0.946. The final total loss is 14.634, slightly worse than
14.480 after epoch one. Seven of fifteen frozen gates fail.

## Interpretation

The experiment rejects the specific hypothesis that ordinary joint
co-adaptation of a 32.8M-parameter byte transformer and structured program
heads is sufficient to recover even the finite fit renderer orbit under this
loss/interface contract. More layers, epochs, or a relaxed threshold are not
authorized as a continuation of this run.

The preserved 100% query, declaration binding, initial binding, and initial
state paths remain useful evidence: dedicated content-addressed interfaces can
survive renderer changes. The failure is concentrated in program segmentation,
slot typing, and event binding. It does not implicate the exact categorical
executor and does not establish an impossibility below 200M.

Before proposing a new reasoning mechanism, the next step is an implementation
audit that distinguishes a clean optimization/credit-assignment failure from a
hidden gradient, batching, or objective defect. Any successor must be a new
preregistered contract, not an extension or rescore of this pilot.

That post-hoc audit is now complete. All trainable groups changed and every
relevant loss has a finite gradient path. On all held-out renderer rows, final
per-slot line/event-address/kind/amount/identity are 42.029%/25.466%/55.731%/
68.000%/50.325%, nearly identical to fit. Gold line pooling raises kind to
73.641% but does not improve amount; gold event-span pooling makes identity
100%. See `R12_SD_CST_RENDERER_NATIVE_JOINT_AUDIT.md`. The successor must change
record/address factorization, not add capacity or epochs to this contract.

## Preserved Evidence

- checkpoint SHA-256:
  `4b842e4c2d0d608c32f0fd113b404866be7269676084cdac9b1a00d43cdd298d`;
- report SHA-256:
  `cefb33e81d42b69b8e088e0ea79926c4557ecca79db3731efa83b44241d6f7ff`;
- local checkpoint/report:
  `train/sd_cst_renderer_native_joint_pilot_6795424534800881443/`;
- committed exact report:
  `artifacts/r12/sd_cst_renderer_native_joint_pilot_6795424534800881443/report.json`;
- Newton checkpoint/report:
  `/lustre/fs1/home/sa305415/shohin_sd_cst_renderer_native_joint_pilot_6795424534800881443/`.
