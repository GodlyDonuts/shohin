# Shohin

### A 125M-parameter language model built and trained from scratch

![Parameters](https://img.shields.io/badge/parameters-125.1M-4c6ef5)
![Training](https://img.shields.io/badge/training-300K_steps-7950f2)
![Corpus](https://img.shields.io/badge/corpus-57.8B_tokens-0ca678)
![Throughput](https://img.shields.io/badge/throughput-282K_tok%2Fs-f59f00)
![Hardware](https://img.shields.io/badge/hardware-2%C3%97_H100-e8590c)

Shohin is a dense decoder-only transformer implemented and pretrained from
scratch in PyTorch. The project covers the complete small-model stack:
tokenizer training, corpus preparation, model architecture, distributed
pretraining, checkpoint custody, contamination auditing, and executable
evaluation.

This public repository contains the completed 300K-step pretraining baseline.
Later research is summarized only at the result level while its implementation
remains private during active development.

## At a glance

| | |
|---|---:|
| **Trained parameters** | 125,081,664 |
| **Optimizer steps** | 300,000 |
| **Decontaminated corpus capacity** | 57,826,022,271 tokens |
| **Nominal training exposure** | 157,286,400,000 tokens |
| **Tokenizer** | custom 32,768-token BPE |
| **Context length** | 2,048 tokens |
| **Global batch** | 524,288 tokens/update |
| **Production throughput** | 281,959 tokens/s on 2× H100 |
| **Scaling result** | 1.85× the established single-H100 throughput |

Nominal exposure includes corpus replay and is not a unique-token claim.

## Engineering highlights

### Built from scratch

- Implemented a 30-layer, deep-and-thin GQA transformer with RMSNorm, SwiGLU,
  RoPE, QK normalization, tied embeddings, and KV-cached generation.
- Trained a compact 32K English, mathematics, and Python tokenizer rather than
  inheriting a tokenizer or pretrained embedding table.
- Combined Muon for hidden projection matrices with AdamW for embeddings,
  output, and normalization parameters.

### Stabilized long-running pretraining

Recurring loss cliffs were traced to abrupt transitions between roughly
200M-token monodomain blocks. The data path was rebuilt around independent
domain streams and mixed-domain batches, removing the boundary shocks and
remaining stable through 300K optimizer steps.

The asynchronous loader overlaps Zstandard decompression with GPU execution,
uses deterministic but generation-specific stream seeds across restarts, and
guarantees representation from every active domain in each batch.

### Scaled without changing the experiment

The trainer uses bf16, PyTorch DDP, gradient accumulation, `torch.compile`,
guarded optimizer steps, resumable checkpoints, and a
warmup-stable-decay schedule. Scaling from one to two H100s preserved the exact
524,288-token global update while increasing sustained throughput from the
154.3K-token/s band to **281,959 tokens/s**.

### Made the evidence reproducible

- Exact and n-gram benchmark-overlap filtering, duplicate checks, and
  source-level admission gates before training.
- Hash-verified checkpoints, frozen tokenizer identity, exact configuration
  validation on resume, and immutable final evaluation logs.
- Seeded GSM8K and MATH-500 evaluation plus sandboxed, timeout-bound execution
  for HumanEval and MBPP.
- Reproducible Slurm launch and recovery paths for single- and multi-H100 runs.

## Selected research outcomes

A separately controlled extension studies whether a small language model can
compile bounded language into an executable representation and reuse a learned
state across multiple operations. The implementation, task generator, and
detailed mechanism are not included in this public baseline.

| Frozen held-out result | Score |
|---|---:|
| Exact executable programs | **2,003 / 2,048 (97.80%)** |
| Exact recurrent state | **2,015 / 2,048 (98.39%)** |
| Correct final answers | **2,020 / 2,048 (98.63%)** |
| Exact programs under either causal rotation control | **0 / 2,048** |

A second experiment replaced a fixed transition table with a
**4,934-parameter learned module** trained on only six primitive transition
examples. It composed **36/36 never-trained two-step transitions** and reached
**97.61% exact state** and **98.10% answer accuracy** end to end, matching the
exact-executor ceiling on the same frozen protocol.

These are bounded synthetic-system results, not claims of unrestricted
reasoning or public-benchmark state of the art. The public, non-proprietary
evidence boundary is documented in
[`RESEARCH_SUMMARY.md`](RESEARCH_SUMMARY.md).

## Baseline architecture

```text
32K BPE tokens
      │
      ▼
30 × [RMSNorm → GQA + RoPE → residual
      RMSNorm → SwiGLU MLP → residual]
      │
      ▼
tied language-model head
```

| Component | Configuration |
|---|---|
| Hidden / feed-forward width | 576 / 1,536 |
| Query / key-value heads | 9 / 3 |
| Head dimension | 64 |
| Position encoding | RoPE, theta 50,000 |
| Objective | next-token cross-entropy + z-loss |
| Optimizers | Muon + AdamW |

See [`ARCHITECTURE.md`](ARCHITECTURE.md) for the baseline implementation map.

## Repository map

| Path | Purpose |
|---|---|
| [`train/model.py`](train/model.py) | Transformer architecture and KV-cached forward path |
| [`train/train.py`](train/train.py) | bf16, DDP, compilation, optimization, and checkpointing |
| [`train/data.py`](train/data.py) | Asynchronous domain-interleaved shard loader |
| [`train/muon.py`](train/muon.py) | Muon optimizer implementation |
| [`train/eval_suite.py`](train/eval_suite.py) | GSM8K and MATH-500 evaluation |
| [`train/eval_code.py`](train/eval_code.py) | Sandboxed HumanEval and MBPP execution |
| [`pipeline/`](pipeline) | Tokenizer, shard, and benchmark preparation |
| [`RESULTS.md`](RESULTS.md) | Audited raw-pretraining measurements and checkpoint hashes |

## Reproduce the public baseline

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run a small local smoke model:

```bash
cd train
python train.py \
  --size tiny \
  --shard-dirs /path/to/math /path/to/web /path/to/code \
  --steps 50 \
  --batch-size 2 \
  --out smoke_out
```

Launch the Shohin configuration on two GPUs:

```bash
cd train
torchrun --standalone --nproc_per_node=2 train.py \
  --size shohin \
  --shard-dirs /path/to/math /path/to/web /path/to/code \
  --steps 300000 \
  --batch-size 32 \
  --grad-accum 4 \
  --compile \
  --out flagship_out
```

Run the baseline contracts:

```bash
PYTHONPATH=train python train/test_batched_generation.py
PYTHONPATH=train python train/test_data_stream_resume.py
```

## Raw-checkpoint boundary

The preserved 300K artifact is the raw pretrained base, before instruction or
reasoning post-training. Its final benchmark board is reported in
[`RESULTS.md`](RESULTS.md) without converting best-of-N, qualitative samples,
or later research components into raw pass@1 claims.
