# Shohin: a from-scratch 125M language model

Shohin is a dense decoder-only transformer implemented and pretrained from
scratch in PyTorch. This repository preserves the architecture, distributed
training stack, tokenizer, evaluation harness, and final evidence for the
completed **300,000-step raw-pretraining run**.

## Final training snapshot

| Property | Value |
|---|---:|
| Trained parameters | 125,081,664 |
| Transformer layers | 30 |
| Model / feed-forward width | 576 / 1,536 |
| Query / key-value heads | 9 / 3 |
| Context length | 2,048 tokens |
| Tokenizer | custom 32,768-token BPE |
| Optimizer | Muon for hidden matrices; AdamW for embeddings, head, and norms |
| Global batch | 524,288 tokens per update |
| Completed updates | 300,000 |
| Nominal token exposures | 157,286,400,000 |
| Mounted corpus capacity | 57,826,022,271 decoded tokens |
| Final two-H100 throughput | 281,959 tokens/s |

“Nominal token exposures” includes data replay and is not a unique-token
claim. Aggregate exposure was 2.72 times the mounted corpus capacity.

## What was built

- A deep-and-thin GQA transformer with RMSNorm, SwiGLU, RoPE, QK
  normalization, and tied input/output embeddings.
- A custom English, mathematics, and code BPE tokenizer with 32K tokens.
- Mixed-domain streaming from compressed `uint16` shards. Every batch contains
  multiple domains, preventing the recurring loss cliffs observed when
  200M-token single-domain shards were consumed sequentially.
- Single-node distributed training with PyTorch DDP, bf16, gradient
  accumulation, compilation, guarded optimizer steps, deterministic stream
  generations, and resumable checkpoints.
- A Muon/AdamW optimizer split and a warmup-stable-decay learning-rate
  schedule.
- Reproducible GSM8K, MATH-500, HumanEval, and MBPP evaluation with seeded
  generation and sandboxed execution for code.
- Hash-bound final checkpoint and benchmark records.

The implementation is intentionally compact. Start with
[`train/model.py`](train/model.py), [`train/train.py`](train/train.py), and
[`train/data.py`](train/data.py). A detailed component map is in
[`ARCHITECTURE.md`](ARCHITECTURE.md), and the final measured results are in
[`RESULTS.md`](RESULTS.md).

## Quick start

Install the runtime:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Train a small smoke model from tokenized `.u16.zst` shards:

```bash
cd train
python train.py \
  --size tiny \
  --shard-dirs /path/to/domain-a /path/to/domain-b \
  --steps 50 \
  --batch-size 2 \
  --out smoke_out
```

Launch the Shohin configuration with two GPUs:

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

The published evaluation data lives under `artifacts/evals`. For example:

```bash
cd train
python eval_suite.py \
  --ckpt /path/to/checkpoint.pt \
  --tokenizer ../artifacts/shohin-tok-32k.json \
  --task gsm8k \
  --data ../artifacts/evals/gsm8k.jsonl
```

Run the baseline tests:

```bash
PYTHONPATH=train python train/test_batched_generation.py
PYTHONPATH=train python train/test_data_stream_resume.py
```

## Final raw benchmark board

| Benchmark | Result |
|---|---:|
| GSM8K majority@4 | 4/100 |
| GSM8K pass@1 | 2/100 |
| MATH-500 pass@1 | 2/100 |
| HumanEval pass@1 | 6/164 |
| MBPP pass@1 | 0/100 |

These results are reported without inflating the conclusion: the 300K raw
checkpoint is a durable pretraining base, not a competitive reasoning model.
The run demonstrates the complete model-building, data, scaling, stability,
custody, and evaluation system; post-training was not applied to this
checkpoint.
