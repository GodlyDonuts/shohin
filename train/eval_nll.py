#!/usr/bin/env python3
"""Measure token-weighted NLL on fixed, named text monitors.

This is a pretraining diagnostic, not a capability benchmark and never a
training-data writer.  Each input is evaluated independently so a curriculum
handoff can distinguish broad-language regression from math/code changes.  The
input JSONL must already be frozen and held outside the training shard paths.
"""

import argparse
import contextlib
import hashlib
import json
import math
import re
from pathlib import Path

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from model import GPT, GPTConfig


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_input_spec(spec):
    """Parse ``label=path`` without allowing ambiguous monitor labels."""
    if "=" not in spec:
        raise ValueError(f"input must be label=path: {spec}")
    label, raw_path = spec.split("=", 1)
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", label):
        raise ValueError(f"invalid monitor label: {label}")
    path = Path(raw_path)
    if not path.is_file():
        raise FileNotFoundError(f"monitor input is missing: {path}")
    return label, path


def text_rows(path, field):
    """Yield nonempty text rows in deterministic file order."""
    with path.open(errors="replace") as source:
        for number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{number}") from exc
            text = row.get(field)
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"missing nonempty {field!r} at {path}:{number}")
            yield text


def token_blocks(token_sequences, seq_len, max_sequences):
    """Pack documents with EOS already appended into fixed next-token blocks."""
    if seq_len < 2:
        raise ValueError("seq_len must be at least two")
    buffer, emitted = [], 0
    for sequence in token_sequences:
        buffer.extend(sequence)
        while len(buffer) >= seq_len + 1:
            yield buffer[:seq_len + 1]
            del buffer[:seq_len]
            emitted += 1
            if max_sequences and emitted >= max_sequences:
                return


def load_model(path, device):
    checkpoint = torch.load(path, map_location="cpu")
    model = GPT(GPTConfig(**checkpoint["cfg"])).to(device).eval()
    model.load_state_dict(checkpoint["model"])
    return checkpoint, model


def amp_context(device):
    return torch.autocast("cuda", dtype=torch.bfloat16) if device == "cuda" else contextlib.nullcontext()


@torch.inference_mode()
def batch_nll(model, inputs, targets, device):
    """Return pure cross entropy, excluding any training-only auxiliary loss."""
    with amp_context(device):
        logits, _ = model(inputs)
    return F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), targets.reshape(-1))


@torch.inference_mode()
def evaluate(model, tokenizer, path, field, eos_id, device, batch_size, max_sequences):
    token_sequences = (
        tokenizer.encode(text).ids + [eos_id]
        for text in text_rows(path, field)
    )
    blocks = token_blocks(token_sequences, model.cfg.seq_len, max_sequences)
    batch, total_loss, total_tokens, sequences = [], 0.0, 0, 0
    for block in blocks:
        batch.append(block)
        if len(batch) < batch_size:
            continue
        inputs = torch.tensor([row[:-1] for row in batch], device=device)
        targets = torch.tensor([row[1:] for row in batch], device=device)
        loss = batch_nll(model, inputs, targets, device)
        count = targets.numel()
        total_loss += float(loss) * count
        total_tokens += count
        sequences += len(batch)
        batch = []
    if batch:
        inputs = torch.tensor([row[:-1] for row in batch], device=device)
        targets = torch.tensor([row[1:] for row in batch], device=device)
        loss = batch_nll(model, inputs, targets, device)
        count = targets.numel()
        total_loss += float(loss) * count
        total_tokens += count
        sequences += len(batch)
    if not total_tokens:
        raise ValueError(f"monitor has no full {model.cfg.seq_len}-token blocks: {path}")
    nll = total_loss / total_tokens
    return {
        "input": str(path),
        "sequences": sequences,
        "tokens": total_tokens,
        "nll": nll,
        "perplexity": math.exp(min(nll, 30.0)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--input", action="append", required=True,
                        help="fixed monitor input in label=path form; repeat per domain")
    parser.add_argument("--out", required=True)
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-sequences-per-input", type=int, default=256)
    args = parser.parse_args()
    if args.batch_size < 1 or args.max_sequences_per_input < 1:
        raise SystemExit("batch size and max sequences must be positive")

    specs = [parse_input_spec(spec) for spec in args.input]
    labels = [label for label, _ in specs]
    if len(set(labels)) != len(labels):
        raise SystemExit("monitor labels must be unique")
    output = Path(args.out)
    if output.exists():
        raise SystemExit(f"refusing to overwrite monitor result: {output}")

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    tokenizer = Tokenizer.from_file(args.tokenizer)
    eos_id = tokenizer.token_to_id("<|endoftext|>")
    if eos_id is None:
        raise SystemExit("tokenizer has no <|endoftext|> token")
    checkpoint, model = load_model(args.ckpt, device)
    domains = {}
    for label, path in specs:
        result = evaluate(
            model, tokenizer, path, args.text_field, eos_id, device,
            args.batch_size, args.max_sequences_per_input,
        )
        result["input_sha256"] = sha256(path)
        domains[label] = result
    total_tokens = sum(row["tokens"] for row in domains.values())
    weighted_nll = sum(row["nll"] * row["tokens"] for row in domains.values()) / total_tokens
    result = {
        "audit": "pretrain_nll_v1",
        "checkpoint": args.ckpt,
        "step": checkpoint.get("step"),
        "device": device,
        "text_field": args.text_field,
        "max_sequences_per_input": args.max_sequences_per_input,
        "domains": domains,
        "weighted_nll": weighted_nll,
        "weighted_perplexity": math.exp(min(weighted_nll, 30.0)),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".partial")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(output)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
