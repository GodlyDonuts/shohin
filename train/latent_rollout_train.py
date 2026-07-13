#!/usr/bin/env python3
"""Train a continuous latent-rollout model on answer-only operator examples.

This trainer is intentionally separate from ``sft.py``.  It has no packed text
CoT, no external state, and no changes to the flagship writer.  Each batch
contains examples with identical prompt/answer token lengths, avoiding padding
and making the paired latent-versus-non-latent control mechanically matched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from collections import defaultdict

import torch
from tokenizers import Tokenizer

from latent_rollout import supervised_latent_loss
from model import GPT, GPTConfig
from muon import Muon, split_params


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def progressive_latent_steps(step: int, total_steps: int, maximum: int) -> int:
    """Use progressively deeper rollouts after the base answer path is stable."""
    if maximum < 0:
        raise ValueError("maximum must be non-negative")
    if maximum == 0:
        return 0
    fraction = step / max(1, total_steps)
    if fraction < 0.15:
        return 0
    if fraction < 0.35:
        return 1
    if fraction < 0.60:
        return min(2, maximum)
    return maximum


def load_examples(path: str, tokenizer, eos_id: int, max_latent_steps: int, max_examples: int = 0):
    """Tokenize exact question/answer continuations and reject overlength rows."""
    examples = []
    skipped = defaultdict(int)
    with open(path) as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            question, response = row.get("question"), row.get("response")
            if not isinstance(question, str) or not isinstance(response, str):
                skipped["missing_fields"] += 1
                continue
            prompt_ids = tokenizer.encode(question).ids
            answer_ids = tokenizer.encode(" " + response.strip()).ids
            if not prompt_ids or not answer_ids:
                skipped["empty_tokens"] += 1
                continue
            if len(prompt_ids) + max_latent_steps + len(answer_ids) > 2048:
                skipped["overlength"] += 1
                continue
            examples.append({
                "prompt": prompt_ids,
                "answer": answer_ids,
                "depth": int(row.get("depth", 0)),
                "family": str(row.get("family", "")),
                "line": line_number,
            })
            if max_examples and len(examples) >= max_examples:
                break
    if not examples:
        raise ValueError("no fitting latent rollout examples in {}".format(path))
    if eos_id is None or eos_id < 0:
        raise ValueError("tokenizer eos token is missing")
    return examples, dict(sorted(skipped.items()))


def bucketed_batches(examples, batch_size: int, seed: int):
    """Return shuffled full batches with exactly matching token shapes.

    No padding is inserted.  The same source/seed/batch size therefore gives a
    paired control and latent condition the same examples in the same batches.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    buckets = defaultdict(list)
    for index, example in enumerate(examples):
        buckets[(len(example["prompt"]), len(example["answer"]))].append(index)
    rng = random.Random(seed)
    batches = []
    dropped = 0
    for key in sorted(buckets):
        indices = list(buckets[key])
        rng.shuffle(indices)
        usable = (len(indices) // batch_size) * batch_size
        dropped += len(indices) - usable
        batches.extend(indices[offset:offset + batch_size] for offset in range(0, usable, batch_size))
    rng.shuffle(batches)
    if not batches:
        raise ValueError("no full batches; reduce --batch-size")
    return batches, {
        "buckets": len(buckets),
        "full_batches": len(batches),
        "dropped_examples": dropped,
    }


def lr_scale(step: int, total_steps: int, warmup: int) -> float:
    if step < warmup:
        return step / max(1, warmup)
    progress = (step - warmup) / max(1, total_steps - warmup)
    return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))


def make_batch(examples, indices, device: str):
    prompts = torch.tensor([examples[index]["prompt"] for index in indices], dtype=torch.long, device=device)
    answers = torch.tensor([examples[index]["answer"] for index in indices], dtype=torch.long, device=device)
    return prompts, answers


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init", required=True, help="pretrained model checkpoint")
    parser.add_argument("--data", required=True, help="frozen latent-operator training JSONL")
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-latent-steps", type=int, default=4,
                        help="0 is the matched non-latent control")
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--lr-muon", type=float, default=2e-3)
    parser.add_argument("--lr-adam", type=float, default=5e-4)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--eos", default="<|endoftext|>")
    parser.add_argument("--log-every", type=int, default=20)
    args = parser.parse_args()
    if args.epochs <= 0 or args.max_latent_steps < 0 or args.max_examples < 0:
        raise SystemExit("epochs and latent/example limits must be non-negative, with epochs positive")
    if not torch.cuda.is_available():
        raise SystemExit("latent rollout training requires a CUDA allocation")
    if os.path.exists(args.out) and os.listdir(args.out):
        raise SystemExit("refusing non-empty output directory: {}".format(args.out))
    os.makedirs(args.out, exist_ok=True)

    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    device = "cuda"
    tokenizer = Tokenizer.from_file(args.tokenizer)
    eos_id = tokenizer.token_to_id(args.eos)
    examples, skipped = load_examples(args.data, tokenizer, eos_id, args.max_latent_steps, args.max_examples)
    batches, batch_report = bucketed_batches(examples, args.batch_size, args.seed)
    total_steps = args.epochs * len(batches)
    print(json.dumps({
        "latent_rollout": "continuous_v1",
        "examples": len(examples),
        "skipped": skipped,
        "batch_report": batch_report,
        "total_steps": total_steps,
        "max_latent_steps": args.max_latent_steps,
        "data_sha256": sha256_file(args.data),
    }, sort_keys=True), flush=True)

    checkpoint = torch.load(args.init, map_location="cpu")
    cfg = GPTConfig(**checkpoint["cfg"])
    model = GPT(cfg).to(device)
    model.load_state_dict(checkpoint["model"])
    muon_parameters, adam_parameters = split_params(model)
    opt_muon = Muon(muon_parameters, lr=args.lr_muon)
    opt_adam = torch.optim.AdamW(adam_parameters, lr=args.lr_adam, betas=(0.9, 0.95), weight_decay=0.0)
    print("[latent] init={} step={} params={:.1f}M".format(
        args.init, checkpoint.get("step"), model.num_params() / 1e6), flush=True)

    started, step = time.time(), 0
    for epoch in range(args.epochs):
        epoch_batches, _ = bucketed_batches(examples, args.batch_size, args.seed + epoch)
        for indices in epoch_batches:
            prompt_ids, answer_ids = make_batch(examples, indices, device)
            latent_steps = progressive_latent_steps(step, total_steps, args.max_latent_steps)
            scale = lr_scale(step, total_steps, args.warmup)
            for group in opt_muon.param_groups:
                group["lr"] = args.lr_muon * scale
            for group in opt_adam.param_groups:
                group["lr"] = args.lr_adam * scale
            opt_muon.zero_grad(set_to_none=True)
            opt_adam.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, loss, _ = supervised_latent_loss(model, prompt_ids, answer_ids, latent_steps, eos_id)
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite latent rollout loss at step {}".format(step))
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            if not torch.isfinite(grad_norm):
                raise RuntimeError("non-finite latent rollout gradient at step {}".format(step))
            opt_muon.step()
            opt_adam.step()
            if step % args.log_every == 0:
                examples_per_second = args.batch_size * (step + 1) / max(1e-6, time.time() - started)
                print(
                    "[latent] epoch={} step={}/{} L={} loss={:.4f} gnorm={:.3f} lr={:.6f} ex/s={:.1f}".format(
                        epoch, step, total_steps, latent_steps, loss.item(), float(grad_norm),
                        args.lr_muon * scale, examples_per_second,
                    ),
                    flush=True,
                )
            step += 1
        output = os.path.join(args.out, "latent_ep{}.pt".format(epoch + 1))
        if os.path.exists(output):
            raise RuntimeError("refusing to overwrite {}".format(output))
        torch.save({
            "model": model.state_dict(),
            "cfg": cfg.__dict__,
            "step": "latent_ep{}".format(epoch + 1),
            "latent_rollout": {
                "protocol": "continuous_hidden_soft_token_v1",
                "init": args.init,
                "data": args.data,
                "data_sha256": sha256_file(args.data),
                "max_latent_steps": args.max_latent_steps,
                "progressive_schedule": "0-15%=0,15-35%=1,35-60%=min(2,max),60-100%=max",
                "seed": args.seed,
                "epoch": epoch + 1,
                "updates": step,
                "batch_size": args.batch_size,
                "claim_boundary": "Answer-only continuous-state operator training; not a broad reasoning claim.",
            },
        }, output)
        print("[latent] saved {}".format(output), flush=True)
    print("[latent] done {} updates in {:.0f}s".format(step, time.time() - started), flush=True)


if __name__ == "__main__":
    main()
