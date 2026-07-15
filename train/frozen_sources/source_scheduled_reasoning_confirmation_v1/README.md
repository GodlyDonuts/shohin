# Frozen Source-Scheduled Confirmation Runtime

These files are byte-for-byte copies of the implementation admitted by raw
260k confirmation job `689542`, which started on Newton at
`2026-07-15T04:47:26-04:00`.

The live repository changed `train/model.py` after that job started to add
masked-loss and resume-contract guards. The independent assessor therefore
must not rehash the moving live path. It rehashes this directory instead.

| File | SHA-256 |
|---|---|
| `contract.md` | `cdca05d9a99a0e341661534433eae5f1351049b4b97f7bf235dcc2cc29edb39d` |
| `generator.py` | `4817472dc3ba0e31b3aed6f91ff01c0fe06cba827828cf5b46cd2b0ed79ccf29` |
| `evaluator.py` | `acd0b895bf73ceeaab37e70330b5a80097027dc3bb6365b848e63567f93247ca` |
| `job.sbatch` | `a34b06ec4da24c5267856f5347a2893fdb64d210a3b56b81df3cb23492122c15` |
| `model_loader.py` | `3f0d092fed269a2ca7556a878fbcf12ebbc0a901911c79fdbc37b6b08dab9284` |

The four non-loader files match the admitted local and Newton sources. The
loader was copied directly from Newton while job `689542` was still running
and its hash matched the value recorded by the evaluator. This archive does
not replace or execute as the current training implementation.
