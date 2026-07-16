# R12 Cursor Readout / Actuation Result

**Decision:** cursor-indexed linear readout **NO-GO**. This result does not
reject distributed token-level representations and does not authorize a
reasoning, internal-cursor, compositionality, novelty, or actuation claim.

## Custody

- frozen implementation commit: `e2e4bb703304ebe2ce11554c8b7c97ef0d3aa928`;
- raw 260k base SHA-256:
  `91d5288f184fc5230516add9851ac1a8815d3369ffd816cd7d0c03d8bafc741d`;
- confirmation-free development view SHA-256:
  `24abd93737be57c6792a1d44c8f2e3a28d7c5fbc1666b083383350f410ce6ec9`;
- independent view audit SHA-256:
  `33fb4792ed0a8027d49de157c295cb9ba651cdd9c59ab5cfa04a71e99af8ea25`;
- runtime SHA-256:
  `af7da54fd23ac1f7a64766438ba72d14591ae96d495da2306cc535da875d7f7c`;
- Newton job: `689952`, isolated one-H100 run on `evc30`;
- immutable result SHA-256:
  `fda4fc47f63ffae0f1b085e527230b3242d8899649dbefb8f8b50d72cbaae433`.

The model process received only 5,760 train cells and 960 development cells.
The source canary, source audit, tokenizer paths, and confirmation rows were not
provided to it. Every input was copied to a private node-local directory and
re-hashed before load. The result preserves readout/calibrator tensors,
standardization vectors, per-example development predictions, margins, delta
norms, runtime versions, and code/input bindings.

## Scores

| Arm | Train restricted | Development restricted | Exact five-step groups | Calibrated full vocab | Alpha | Beta | Median delta L-inf |
|---|---:|---:|---:|---:|---:|---:|---:|
| pre-final joint | 98.75% | 43.13% | 0/192 | 43.13% | 2.016 | 21.393 | 88.764 |
| post-final joint | 97.43% | 41.67% | 0/192 | 41.88% | 1.193 | 21.374 | 67.954 |
| pre-final source-only | 20.00% | 20.00% | 0/192 | 19.79% | ~0 | 22.127 | 22.127 |
| post-final source-only | 20.00% | 20.00% | 0/192 | 19.79% | ~0 | 22.093 | 22.093 |
| cursor-only | 40.00% | 40.00% | 0/192 | 40.00% | 0.745 | 21.348 | 21.958 |

Both development renderers agree: the pre-final joint readout scores 42.50%
and 43.75%; the post-final readout scores 42.08% and 41.25%. The apparent
overall lift includes the deterministic DONE cell. On the four non-DONE cells,
pre-final joint accuracy is 222/768 = 28.91%, only 3.91 percentage points above
the 25% operation-choice shortcut. No source has all five actions correct.

The base target action is below the best non-action vocabulary token by median
5.379 logits on development. The scalar calibrators force an action token to
win all 960 joint rows, but only by applying very large direct-logit changes;
they cannot repair incorrect restricted class selection. Their full-vocabulary
accuracy therefore remains essentially the restricted accuracy.

## Interpretation

The final selector position does not expose a renderer-invariant, linearly
readable operation-order code, even when an oracle cursor selects an independent
classifier. The train/development gap shows template/operand overfit rather
than a stable joint representation. This falsifies the next simplest theory
after the Q-only sidecar failure: the correct action is not merely present as a
small linear code at the final token waiting for a stronger vocabulary write.

It does **not** show that source order is absent from the network. The prompt
tokens necessarily carry the operations, and the relevant state may remain
distributed across token positions instead of being consolidated at the final
selector position. The next bounded diagnostic should therefore compare a
cursor-conditioned token-tape retrieval readout against matched source-only and
cursor-only controls. A pass would justify testing a direct residual/write
adapter; a failure would close this external-cursor branch.

V1 development contains all 24 operation permutations, so no result in this
chain demonstrates unseen-permutation extrapolation. Any v2 score-bearing
experiment must withhold permutations and generate fresh operands/renderers.

## Operational Note

Python completed and atomically froze the result before Slurm marked the batch
failed. The non-scientific failure was the cleanup trap attempting to delete an
intentionally read-only exported source tree. The launcher is repaired to make
the private staging directory owner-writable during cleanup and to export each
committed file directly. No rerun is needed because the immutable result was
complete, hash-verified, and mirrored before that post-result cleanup error.
