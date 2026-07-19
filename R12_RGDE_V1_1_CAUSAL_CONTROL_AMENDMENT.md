# R12 RGDE v1.1 Causal-Control Mechanics Amendment

**Status:** frozen after positive-arm evaluation and before any replacement
control score. No model is refit by this amendment.

## Defect

Jobs `693118--693121` completed cleanly, but inspection of the preregistered
row-rotation intervention exposed a scorer defect. The factorized development
file is organized in four-surface semantic quartets and `make_batches` also
forms small length buckets. Rotating one row inside a batch therefore often
selected another surface with the same operation program or query position.
The old `intervention_rows` counter measured a changed row index, not a changed
semantic field.

The original `development_predicted_shuffled.json` and
`development_query_shuffled.json` are retained as mechanical diagnostics but
are inadmissible for gates 7--8. The treatment, gold, untied, and composed-arm
scores are unaffected. No weights, packet, targets, positive evaluator, data,
seed, or advancement threshold changes.

## Frozen repair

The evaluator constructs one deterministic global permutation before batch
inference. It groups rows by the declared intervention key, rotates the sorted
groups by the largest group size, and asserts that every destination key
differs from its source key. For operation intervention the key is the complete
two-operation structured program. For query intervention it is the requested
position. It then compiles destination and source rows independently and
replaces only the bounded operation tuple or query vector.

On all 2,048 development rows, both global permutations are bijections and all
2,048 semantic keys differ. Structured keys select a causal control source
only; no gold state, transition, or answer enters the executor.

The replacement control job uses the already frozen tied executor state
`d31fd3e6150dd352cd0eea5063f960393e9017ac3ab729b5c536fd1f1c432184`.
It writes new filenames and refuses overwrite. It does not train or modify any
parameter.

Frozen source SHA-256 values are:

| Object | SHA-256 |
|---|---|
| packet/derangement helper | `5ec8666047666a7e5c4124f31a390c9e05160381e80b1e60911b40d35fd908c3` |
| evaluator | `e54721ff84b3d8ccecb4f8bd963ce19fdaf7e85de635d75599907d2667ac8c65` |
| tests | `3677e23a9b3f1e9bbaf145336f24dacfb88fd59b3e18fbf040b6eaed38dc0bb3` |
| Slurm control job | `3b1cde695ca538ac8a9482b6b3b4281d675563620ef3042433ca48f21d725cd5` |

Fifteen focused CPU tests, a full 2,048-row derangement audit for each key,
Ruff, `py_compile`, shell syntax, and `git diff --check` pass.
