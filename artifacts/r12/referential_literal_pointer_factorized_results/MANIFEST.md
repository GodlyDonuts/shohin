# Factorized Complete-Compiler Development Evidence

The authoritative assessment is `assessment.json`, SHA-256
`ca8cab2ef9dbaa9d894857438e72193476259fd659e8423b85af47e13e37fc0d`.
It binds the shared base/data/report/tokenizer identity, all five arm metadata
records, all compositional and lexical-OOD scores, all oracle ceilings, every
raw result SHA-256, and the frozen decision. Its decision is
`retain_as_conventional_compiler_baseline_confirmation_sealed`.

The compressed JSON and Slurm files in this directory are deterministic
`gzip -n` backups of the exact raw outputs mirrored from Newton. Their
uncompressed result SHA-256 values are recorded in `assessment.json`. The
uncompressed Slurm SHA-256 values are:

| Job | SHA-256 |
|---:|---|
| `693048` | `5a2ef4e0cae833b63c780d07b990807753533046e5e52bbeba4da94288496638` |
| `693049` | `188f54f44ac34ebf72107420e78f47721e18c692cc5b8b5af6720e72dcacdb7f` |
| `693098` | `e6359d08ddae1fe7a316a241e0b30b1c35fffc12f0a421bf1e75322c58f02cc8` |
| `693101` | `b638bd3223dbe710e39a79da74b794931e30f35a195588157a796e2dd152ef02` |
| `693102` | `848aa90fd8f424bf10a30b6b9c345b7bad3ac86814fd5cae01fc933a28f1fd73` |
| `693099` oracle | `1b00a93203cf840bdaf0092c9517dc4120e3cb0278e1a33359184f754d5124e6` |
| `693100` oracle | `414f09bb5d593977717161aa34e4eca2ce92e220a06e3e05279e8cc962c70c59` |

The adapters are intentionally not stored in Git. They remain hash-matched on
Mac and Newton:

| Arm | Adapter SHA-256 |
|---|---|
| free slots | `54147ccd7d8a25abe4fbf846d8fc899f0eef8555ba72fa2562ad5214989cc248` |
| structured | `b5756095630830f1c258c1a66976d77919493eeb843d9f864d6321c34e9cbbef` |
| parameter islands | `e1d0dbcc385996f7576cb51fdb2204601dbb198028583dbf67c90772375e7acf` |
| ordinary tagger | `747a559b827c6d114943c091b9dea5b4b90cef7af13aa5003b8435c092d24991` |
| shuffled islands | `9d772b5de25daa1f85e0cc7041e4c6fc5d196c9255d5b84a62f08af58c11146b` |

Confirmation bytes were never copied to Newton, evaluated, assessed, or
backed up here.
