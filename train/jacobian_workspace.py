#!/usr/bin/env python3
"""Fit an exact future-residual Jacobian lens for the Shohin transformer.

The estimator follows the reduction used by the Jacobian Lens reference
implementation: every valid target position receives the same one-hot output
cotangent, then source-position gradients are averaged.  A row therefore
measures how one source-layer residual coordinate affects that coordinate at
all current-and-future target positions.

This module is diagnostic only.  It freezes every model parameter and never
writes a model checkpoint or training example.

Estimator definition adapted from Anthropic's Apache-2.0 ``jlens`` reference:
https://github.com/anthropics/jacobian-lens
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

import torch
from tokenizers import Tokenizer

from categorical_microcode import sha256_file
from model import GPT, GPTConfig


def valid_position_mask(seq_len: int, skip_first: int) -> torch.Tensor:
    if skip_first < 0:
        raise ValueError("skip_first must be nonnegative")
    mask = torch.zeros(seq_len, dtype=torch.bool)
    mask[skip_first:seq_len - 1] = True
    if not mask.any():
        raise ValueError("sequence is too short for the requested position mask")
    return mask


def _resolve_layers(model: GPT, source_layers, target_layer):
    count = len(model.blocks)
    target = target_layer + count if target_layer < 0 else target_layer
    sources = sorted({layer + count if layer < 0 else layer for layer in source_layers})
    if not 0 <= target < count:
        raise ValueError("target layer is out of range")
    if not sources or sources[0] < 0 or sources[-1] >= target:
        raise ValueError("source layers must be nonempty and precede the target layer")
    return sources, target


def _residual_graph(model: GPT, ids: torch.Tensor, source_layers, target_layer):
    """Return selected block outputs while starting autograd at the first source."""
    start = min(source_layers)
    x = model.tok(ids)
    cos = model.cos[: ids.shape[1]].to(x.device)
    sin = model.sin[: ids.shape[1]].to(x.device)
    selected = {}
    for index, block in enumerate(model.blocks):
        if index == start:
            x = x.detach().requires_grad_(True)
        x, _ = block(x, cos, sin)
        if index in source_layers or index == target_layer:
            selected[index] = x
        if index == target_layer:
            break
    return selected


def jacobian_for_ids(
    model: GPT,
    input_ids: torch.Tensor,
    source_layers,
    *,
    target_layer: int = -1,
    dim_batch: int = 8,
    skip_first: int = 16,
):
    """Compute exact per-layer average future Jacobians for one token sequence."""
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("input_ids must have shape [1, sequence]")
    if dim_batch < 1:
        raise ValueError("dim_batch must be positive")
    sources, target = _resolve_layers(model, source_layers, target_layer)
    d_model = model.cfg.d_model
    mask = valid_position_mask(input_ids.shape[1], skip_first)
    valid = mask.nonzero(as_tuple=True)[0].to(input_ids.device)
    repeated = input_ids.expand(dim_batch, -1)
    residuals = _residual_graph(model, repeated, sources, target)
    target_activation = residuals[target]
    source_activations = [residuals[layer] for layer in sources]
    cotangent = torch.zeros_like(target_activation)
    batch_indices = torch.arange(dim_batch, device=input_ids.device)
    jacobians = {
        layer: torch.zeros(d_model, d_model, dtype=torch.float32) for layer in sources
    }
    passes = math.ceil(d_model / dim_batch)
    for pass_index, start in enumerate(range(0, d_model, dim_batch)):
        width = min(dim_batch, d_model - start)
        cotangent.zero_()
        cotangent[
            batch_indices[:width, None], valid[None, :], start + batch_indices[:width, None]
        ] = 1.0
        gradients = torch.autograd.grad(
            target_activation,
            source_activations,
            grad_outputs=cotangent,
            retain_graph=pass_index < passes - 1,
        )
        for layer, gradient in zip(sources, gradients):
            jacobians[layer][start:start + width] = (
                gradient[:width, valid, :].float().mean(dim=1).cpu()
            )
    return jacobians, int(mask.sum())


def transport(jacobian: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
    """Transport source residuals into the final-block residual basis."""
    return residual.float() @ jacobian.float().T


def spectrum_metrics(jacobian: torch.Tensor):
    singular = torch.linalg.svdvals(jacobian.float())
    mass = singular / singular.sum().clamp_min(1e-12)
    cumulative = mass.cumsum(0)
    rank_90 = int((cumulative < 0.90).sum().item()) + 1
    rank_99 = int((cumulative < 0.99).sum().item()) + 1
    entropy_rank = float(torch.exp(-(mass * mass.clamp_min(1e-30).log()).sum()).item())
    return {
        "frobenius_norm": float(jacobian.float().norm().item()),
        "spectral_norm": float(singular[0].item()),
        "rank_90_nuclear_mass": rank_90,
        "rank_99_nuclear_mass": rank_99,
        "entropy_effective_rank": entropy_rank,
    }


def load_prompts(path, count, seed):
    rows = []
    with open(path) as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            text = row.get("text")
            if not isinstance(text, str) or not text.strip():
                raise ValueError("invalid text at line {}".format(line_number))
            rows.append((line_number, text))
    if count > len(rows):
        raise ValueError("requested more prompts than the input contains")
    return random.Random(seed).sample(rows, count)


def atomic_save(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.{}".format(os.getpid()))
    torch.save(payload, temporary)
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--source-layers", default="5,9,13,17,21,25,28")
    parser.add_argument("--target-layer", type=int, default=-1)
    parser.add_argument("--prompt-count", type=int, default=8)
    parser.add_argument("--max-seq-len", type=int, default=128)
    parser.add_argument("--skip-first", type=int, default=16)
    parser.add_argument("--dim-batch", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260714)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("Jacobian workspace fitting requires CUDA")
    if Path(args.out).exists():
        raise SystemExit("refusing existing output")

    checkpoint = torch.load(args.base, map_location="cpu", weights_only=False)
    model = GPT(GPTConfig(**checkpoint["cfg"])).to("cuda").eval()
    model.load_state_dict(checkpoint["model"])
    model.requires_grad_(False)
    tokenizer = Tokenizer.from_file(args.tokenizer)
    layers = [int(value) for value in args.source_layers.split(",") if value.strip()]
    layers, target = _resolve_layers(model, layers, args.target_layer)
    selected = load_prompts(args.prompts, args.prompt_count, args.seed)
    sums = {
        layer: torch.zeros(model.cfg.d_model, model.cfg.d_model, dtype=torch.float32)
        for layer in layers
    }
    records = []
    for ordinal, (line_number, prompt) in enumerate(selected, 1):
        encoding = tokenizer.encode(prompt)
        token_ids = encoding.ids[: args.max_seq_len]
        ids = torch.tensor([token_ids], dtype=torch.long, device="cuda")
        start = time.perf_counter()
        with torch.enable_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            per_prompt, valid_positions = jacobian_for_ids(
                model,
                ids,
                layers,
                target_layer=target,
                dim_batch=args.dim_batch,
                skip_first=args.skip_first,
            )
        for layer in layers:
            sums[layer].add_(per_prompt[layer])
        record = {
            "ordinal": ordinal,
            "line_number": line_number,
            "text_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "tokens": len(token_ids),
            "valid_positions": valid_positions,
            "seconds": time.perf_counter() - start,
            "max_normalized_frobenius": max(
                matrix.norm().item() / math.sqrt(model.cfg.d_model)
                for matrix in per_prompt.values()
            ),
        }
        records.append(record)
        print("[jacobian-workspace] " + json.dumps(record, sort_keys=True), flush=True)

    jacobians = {layer: matrix / len(records) for layer, matrix in sums.items()}
    metadata = {
        "audit": "shohin_future_jacobian_workspace_v1",
        "claim_boundary": (
            "Read-only average future-residual Jacobians. This does not establish a workspace, "
            "reasoning, or a trainable context-scaling mechanism without separate causal tests."
        ),
        "base": os.path.realpath(args.base),
        "base_sha256": sha256_file(args.base),
        "base_step": checkpoint.get("step"),
        "tokenizer": os.path.realpath(args.tokenizer),
        "tokenizer_sha256": sha256_file(args.tokenizer),
        "prompts": os.path.realpath(args.prompts),
        "prompts_sha256": sha256_file(args.prompts),
        "prompt_count": len(records),
        "seed": args.seed,
        "source_layers": layers,
        "target_layer": target,
        "d_model": model.cfg.d_model,
        "max_seq_len": args.max_seq_len,
        "skip_first": args.skip_first,
        "dim_batch": args.dim_batch,
        "records": records,
        "spectrum": {str(layer): spectrum_metrics(matrix) for layer, matrix in jacobians.items()},
    }
    atomic_save({"metadata": metadata, "jacobians": jacobians}, args.out)
    print("[jacobian-workspace] saved {}".format(args.out), flush=True)


if __name__ == "__main__":
    main()
