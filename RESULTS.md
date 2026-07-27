# Shohin 300K Results

> This ledger reports the **raw pretrained checkpoint only**, before
> instruction or reasoning post-training. The separately controlled bounded
> research extension is summarized in
> [`RESEARCH_SUMMARY.md`](RESEARCH_SUMMARY.md); its scores are not substituted
> for raw benchmark pass@1.

## Completed run

The raw flagship completed exactly 300,000 optimizer updates.

| Measurement | Final value |
|---|---:|
| Unique trained parameters | 125,081,664 |
| Tokens per update | 524,288 |
| Nominal token exposures | 157,286,400,000 |
| Mounted corpus capacity | 57,826,022,271 tokens |
| Final logged loss | 1.6554 |
| Final logged gradient norm | 0.11 |
| Final learning rate | 0.0005 |
| Final throughput | 281,959 tokens/s |
| Continuation wall time | 153,869 seconds |

The final two-H100 throughput was approximately 1.85 times the established
one-H100 band of 154.3K tokens/s while preserving the same global token batch.

Nominal token exposure counts optimizer-update tokens, including replay. It
must not be represented as 157.3B unique training tokens.

## Final benchmark protocol

The final evaluation used the protected step-300,000 model, CUDA, seed
`20260712`, 100 examples for GSM8K/MATH-500/MBPP, all 164 HumanEval problems,
up to 256 generated tokens, and four samples for GSM8K majority voting.

| Benchmark | Metric | Score |
|---|---|---:|
| GSM8K | majority@4 | 4/100 (4.0%) |
| GSM8K | pass@1 | 2/100 (2.0%) |
| MATH-500 | pass@1 | 2/100 (2.0%) |
| HumanEval | pass@1 | 6/164 (3.66%) |
| MBPP | pass@1 | 0/100 (0.0%) |

The complete evaluation log is committed at
`artifacts/eval_history/pretrain_300000_final_692787.log`, SHA-256
`cb3e10be87ac3ef086fcb90bfd39fa1d505352ca3f8a5a6de35a1cec70e146a5`.

## Checkpoint identity

The 300K checkpoint is not stored in Git because it is approximately 500 MB.
The preserved model-only artifact has:

| Property | Value |
|---|---|
| Bytes | 500,448,522 |
| MD5 | `60de77c31b449060ff0417d8db16d3b0` |
| SHA-256 | `211d6b2cddf0c2cf8b12cb0b2d73f9c4440d85f6f531018080c8afd35b2f66a6` |

## Interpretation

The final benchmark movements relative to earlier checkpoints were within one
or two examples. Raw pretraining alone did not create reliable instruction
following or broad mathematical/code reasoning at this scale.

That negative result is part of the record. Shohin's demonstrated result is a
reproducible from-scratch small-model program: custom tokenizer and corpus,
stable mixed-domain training through 300K updates, fixed-global-batch
multi-H100 scaling, checkpoint custody, and executable benchmark evaluation.
