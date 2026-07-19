# R12 Referential Literal-Pointer Corpus Result

**Decision:** **FROZEN DATA PASS; NEURAL COMPILER FIT MAY PROCEED.**

**Claim boundary:** this is a synthetic data-custody and leakage-audit result.
It is not a model score, compiler result, native-reasoning result, executor
result, halt result, or novelty claim.

## 1. Build chronology

The corpus generator was committed before the confirmation seed was opened:

| Event | Result |
|---|---|
| generator freeze `aad5ecf` | source/tests committed and pushed; no confirmation seed or row existed |
| confirmation seed opened | `3072310916827575206` |
| first invocation | aborted before creating a row or directory because 4,200 equal-width nonce names were requested but only 4,135 exist |
| capacity repair `ae01b54` | pools reduced to 3,100 train + 400 development + 600 confirmation names; semantics unchanged |
| second invocation | all in-memory semantic/audit gates passed, but a literal `{}.jsonl` path caused sequential split writes to retain only the final confirmation file |
| output-path repair `b064ce1` | distinct-path test added; source committed and pushed before retry |
| final invocation | all three splits written under the same fixed seeds; all gates passed |

The malformed-run final `{}.jsonl` is byte-identical to the repaired
`confirmation.jsonl`. Therefore the output-path repair changed only artifact
placement, not confirmation content. No model score, fit, checkpoint, or row-
level confirmation inspection occurred before either repair.

## 2. Frozen corpus

| Split | Groups | Rows | Renderer families | Uncompressed SHA-256 |
|---|---:|---:|---|---|
| train | 24,000 | 96,000 | `forge`, `route` | `f47c6d6ce316be6765641f61a294481605fa53c7b12388741fb753c238b2f36e` |
| development | 512 | 2,048 | `archive`, `tableau` | `20611bf4ddbdb42d7e2f9dd76759b86f3f4dd16d5942f207bf7b325984da5ad6` |
| confirmation | 1,024 | 4,096 | `docket`, `procession` | `84005921b5fca93f9c2567655c4345bced78fc74ed7f49c8f72189b9f87fbf03` |

Every group contains canonical, independent paraphrase, token-bag-matched order
twin, and token-bag-matched binding twin surfaces. The pre-fit v1.1 amendment
requires ten exact source targets per row: three initial-order entity pointers,
two operation kinds, two operation entity pointers, two literal pointers, and
one query pointer.

Aggregate acquisition ledger:

- 102,144 rows;
- 8,538,572 source tokens;
- 30,870,736 UTF-8 source bytes;
- 1,021,440 target pointer labels;
- zero teacher calls;
- zero checkpoint reads;
- zero production-evaluation answer reads.

## 3. Leakage and structural gates

All gates pass:

- no duplicate source in any split;
- every group passes paraphrase, order, binding, answer-separation, and exact
  token-bag checks;
- every expected pointer span exists and has at least one Shohin token;
- train/development/confirmation have zero exact-prompt overlap;
- every split pair has zero normalized word 13-gram overlap;
- every split pair has zero entity-name overlap;
- every split pair has zero renderer overlap;
- named shortcut Bayes ceilings are at or below `1/3` in every split.

Confirmation matched-surface ceilings are token bag `1024/3072 = 33.33%`,
absolute pointer positions `120/3072 = 3.91%`, source token length `19/3072 =
0.62%`, and renderer identity `3/3072 = 0.10%`.

The corpus is still synthetic and supplies a typed ontology. Passing these
gates does not certify natural-language semantics outside this bounded machine.

## 4. Evidence identity and backup

| Artifact | SHA-256 |
|---|---|
| v1.1 amendment | `f7d8f6f23ceb2f91d33c8a46340e10e298a2b1aa39ca6b1b5e6264d80bbcd72a` |
| final generator | `fd211d6c31be6440a1ef3451632f17c5a3af89781af088fb0caffc8b26d6561f` |
| tokenizer | `87532df5c121753de3b29194e1f9e3de47986d3f5359548fdf93606773a233d4` |
| v1.1 report | `176435d8c544948468f81cb23dc65ff51bf8010af212fb737984bbed1d1265cc` |
| train gzip (`gzip -9 -n`) | `4ef7a4b2b73d07c99bd19effb832fa4897bf60c90b5e044691aadaff6b2d0fd9` |
| development gzip (`gzip -9 -n`) | `abc991626545aa0fab6d3419e0689d5475a5052fa05a9a664cf1cbff7a8cda30` |
| confirmation gzip (`gzip -9 -n`) | `70db90f5ac3b6a8d5ebdc77ade90301d9c64043122734e9ab4c2861f1c80cc18` |

The compressed archives and v1.1 report are the GitHub backup. The uncompressed
files remain local working copies. The v1.1 JSONL files are byte-identical to
v1; only the audit and model obligations changed. Decompression must reproduce
the exact uncompressed hashes above before any fit or score.

## 5. Authorization

The isolated compiler pilot may now train on `train` and select only on
`development`. Confirmation stays sealed until all arm weights, hyperparameters,
selection rules, and deployment identities are frozen. The first confirmation
opening must score every preregistered arm once. No executor or HALT integration
is authorized by this data result.
