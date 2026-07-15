# R12 Counterfactual Cursor-Action Mechanics Result

**Status:** persistent mechanics board admitted and independently audited. The
optional final-block/head-zero Q-intervention path and separately serializable
192-parameter sidecar pass focused CPU and existing inference regressions. No
score-bearing model fit or GPU job has been run.

The immutable board contains 600 cells, 120 sources, 180 adjacent-order pairs,
and 24 renderer groups. The independent audit reproduced the frozen symbolic
scores exactly:

| Arm | Exact result |
|---|---:|
| Oracle source + cursor | `600/600` |
| Cursor-only | `240/600` |
| Renderer + cursor | `240/600` |
| Source/global/renderer/clamped | `120/600` |
| Deranged cursor | `0/600` |

It also passed all 96 finite-state transition assertions and all 320 exact
query-folding assertions. Board SHA-256 is
`02a202070efa45f14c4e53b7d7f532d98791c7eef9daf438b02d31cc0ec6ab95`;
audit SHA-256 is
`c64951a1369b3dd29ca7e651840e5644e8445c7236cf994c3e54f05ca4a844b2`.

`train/model.py` accepts an optional Q delta at one named layer/head. With the
argument omitted, old checkpoints still load strictly and the original path is
unchanged. `train/counterfactual_cursor_action.py` supplies the frozen event FSM
and a zero-initialized centered-three-bit projection. The production projection
has exactly `3 * 64 = 192` trainable scalars; all base parameters are frozen.
Focused tests cover strict-load/zero-delta parity, code geometry, event
transitions, prompt-boundary alignment, gradient isolation, and cached
decode/full-replay equivalence. Existing causal-KV, batched-generation,
recurrent-inference, and masked-loss regressions also pass.

## Neural-canary preflight

The disjoint generator and independent auditor now reconstruct these exact
split geometries without loading a model:

| Split | Renderers | Packs | Sources | Cells | Training units |
|---|---:|---:|---:|---:|---:|
| Train | 6 | 8 | 1,152 | 5,760 | 288 |
| Development | 2 | 4 | 192 | 960 | 144 |
| Confirmation | 5 | 8 | 960 | 4,800 | 288 |

Every four-pack block is a Latin rotation: each operand value appears exactly
once under each operation, so operand magnitude has zero deterministic
operation identity inside a split. Train, development, and confirmation use
disjoint numbers and disjoint renderer IDs. The exposure contract separates
prompt-row tokens from the cursor side-state and from gold-only labels. Seven
generator/auditor tests mutate targets, tokenization, ordering, integer types,
pair maps, exposure fields, hashes, evalgrams, and Latin balance; all pass.

The favorable controls now have executable mechanics: an eight-entry cursor
table has 512 total and 320 reachable scalars, while the rank-one final-head
text-cursor LoRA has 640 scalars. The 192-scalar source-only arm has only 64
gradient-active dimensions when its cursor is clamped, so total, reachable,
and active capacity must be reported separately rather than called exactly
capacity matched.

This data package remains a draft rather than frozen evidence. The exact typed
loader, six-arm relation equations and matched forward counts, base-checkpoint
and tokenizer binding, full-vocabulary plus restricted-label evaluator, and
score-blind publication receipt are not implemented yet.

This is a mechanics result only. It does not show that Shohin uses the cursor,
selects the correct action, executes arithmetic, carries state, or halts. A
neural run remains forbidden until the matched-arm data generator, immutable
split hashes, typed loader allowlist, six-arm trainer, independent evaluator,
and score-blind result receipt are frozen in one committed implementation.
