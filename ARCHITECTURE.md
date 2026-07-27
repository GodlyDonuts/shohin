# Shohin 300K Architecture

## Model

Shohin is a pre-norm, decoder-only causal transformer. Its trained
configuration is:

| Component | Configuration |
|---|---|
| Vocabulary | 32,768 |
| Sequence length | 2,048 |
| Layers | 30 |
| Hidden width | 576 |
| Feed-forward width | 1,536 |
| Attention | 9 query heads, 3 key/value heads |
| Head dimension | 64 |
| Position encoding | RoPE, theta 50,000 |
| Normalization | RMSNorm; QK normalization |
| Activation | SwiGLU |
| Embeddings | input and output weights tied |
| Biases | none in attention or MLP projections |
| Training loss | next-token cross-entropy plus `1e-4` z-loss |
| Unique parameters | 125,081,664 |

Grouped-query attention reduces the key/value projection and cache cost while
preserving nine query heads. PyTorch scaled dot-product attention selects the
available optimized attention kernel. The inference path supports a KV cache.

The block is:

```text
x = x + GQA(RMSNorm(x))
x = x + Down(SiLU(Gate(x)) * Up(x))
```

## Tokenizer

`artifacts/shohin-tok-32k.json` is the frozen custom 32K tokenizer used by the
run. It was trained for the model's English, mathematics, and Python mixture,
keeping the embedding table compact relative to the model.

Tokenizer SHA-256:
`87532df5c121753de3b29194e1f9e3de47986d3f5359548fdf93606773a233d4`.

## Data path

Training data is stored as Zstandard-compressed arrays of `uint16` token IDs.
The asynchronous loader maintains an independent shuffled stream for each
domain and constructs mixed-domain batches while decompression overlaps GPU
work.

This loader was a consequential stability change. Whole-shard consumption had
created abrupt transitions between roughly 200M-token monodomain blocks,
repeatedly producing loss shocks. Domain-interleaved batches removed that
failure mode and remained stable through 300K steps.

On restart, a generation-derived deterministic seed selects a fresh stream
ordering. This avoids silently replaying the beginning of the same shuffled
stream when a Slurm job resumes from a checkpoint.

## Optimization and distribution

Two optimizers divide parameters by role:

- Muon updates two-dimensional hidden projection matrices using momentum and
  five Newton–Schulz orthogonalization iterations.
- AdamW updates token embeddings, the tied language-model head, normalization
  parameters, and other non-Muon parameters.

Training uses bf16 autocast, gradient accumulation, a
warmup-stable-decay schedule, global gradient-norm guards, and optional
`torch.compile`. DDP keeps the global token batch fixed as the run scales from
one to two H100 GPUs.

The production two-GPU configuration used 32 sequences per rank, four
accumulation microsteps, and 2,048 tokens per sequence:

```text
2 ranks * 32 sequences * 4 microsteps * 2,048 tokens
= 524,288 tokens per optimizer update
```

## Checkpoint behavior

Full checkpoints contain model state, optimizer state when applicable, exact
model configuration, step, and data-stream generation. Resume validation
fails closed if the checkpoint configuration differs from the requested
architecture.

The terminal 300K artifact is model-only and read-only. Any continuation must
start with a fresh optimizer rewarmup.

## Evaluation

The evaluation layer separates:

- mathematical answer extraction and seeded self-consistency for GSM8K and
  MATH-500; and
- sandboxed code execution with timeouts for HumanEval and MBPP.

The final board's exact invocation context and output are preserved in
`artifacts/eval_history/pretrain_300000_final_692787.log`.
