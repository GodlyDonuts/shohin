#!/usr/bin/env python3
"""Train the no-parameter Native Residual Relay (NRR) in an isolated output.

Each example is split into source text and a later event/query suffix. The
model encodes source only through one layer, forwards the native final residual
to a fresh suffix pass, and receives answer loss there. No source ids, source
embeddings, or source KV cache enter the suffix computation.
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

from model import GPT, GPTConfig
from muon import Muon, split_params
from native_residual_relay import supervised_relay_loss


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_examples(path, tokenizer, seq_len, include_paraphrases=True):
    examples, skipped = [], defaultdict(int)
    for line_number, line in enumerate(open(path), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("schema") != "native_residual_relay_v1" or row.get("split") != "train":
            skipped["schema"] += 1
            continue
        sources = [row.get("source")]
        if include_paraphrases:
            sources.append(row.get("paraphrase_source"))
        suffix, response = row.get("suffix_prompt"), row.get("response")
        if not isinstance(suffix, str) or not isinstance(response, str) or not all(isinstance(item, str) for item in sources):
            skipped["fields"] += 1
            continue
        suffix_ids = tokenizer.encode(suffix).ids
        answer_ids = tokenizer.encode(" " + response.strip()).ids
        for source_kind, source in enumerate(sources):
            source_ids = tokenizer.encode(source).ids
            if not source_ids or not suffix_ids or not answer_ids:
                skipped["empty"] += 1
                continue
            if 1 + len(suffix_ids) + len(answer_ids) > seq_len:
                skipped["overlong_suffix"] += 1
                continue
            examples.append({
                "source": source_ids, "suffix": suffix_ids, "answer": answer_ids,
                "shape": (len(source_ids), len(suffix_ids), len(answer_ids)),
                "episode_id": row["episode_id"], "source_kind": source_kind,
            })
    if not examples:
        raise ValueError("no fitting NRR examples")
    return examples, dict(sorted(skipped.items()))


def bucketed_batches(examples, batch_size, seed):
    if batch_size <= 0:
        raise ValueError("batch size must be positive")
    buckets = defaultdict(list)
    for index, example in enumerate(examples):
        buckets[example["shape"]].append(index)
    rng, batches, dropped = random.Random(seed), [], 0
    for shape in sorted(buckets):
        indices = list(buckets[shape])
        rng.shuffle(indices)
        usable = len(indices) // batch_size * batch_size
        dropped += len(indices) - usable
        batches.extend(indices[offset:offset + batch_size] for offset in range(0, usable, batch_size))
    rng.shuffle(batches)
    if not batches:
        raise ValueError("no complete NRR batches; reduce --batch-size")
    return batches, {"buckets": len(buckets), "full_batches": len(batches), "dropped_examples": dropped}


def make_batch(examples, indices, device):
    def tensor(field):
        return torch.tensor([examples[index][field] for index in indices], dtype=torch.long, device=device)
    return tensor("source"), tensor("suffix"), tensor("answer")


def lr_scale(step, total_steps, warmup):
    if step < warmup:
        return step / max(1, warmup)
    progress = (step - warmup) / max(1, total_steps - warmup)
    return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--audit", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--layer", type=int, default=19)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr-muon", type=float, default=2e-3)
    parser.add_argument("--lr-adam", type=float, default=5e-4)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--max-batches", type=int, default=0,
                        help="cap complete exact-shape batches per epoch for a hardware canary")
    parser.add_argument("--eos", default="<|endoftext|>")
    parser.add_argument("--log-every", type=int, default=20)
    args = parser.parse_args()
    if args.epochs <= 0 or args.batch_size <= 0 or args.layer < 0 or args.max_batches < 0:
        raise SystemExit("epochs, batch size, layer, and max batches must be non-negative")
    if not torch.cuda.is_available():
        raise SystemExit("NRR training requires CUDA")
    if os.path.exists(args.out):
        raise SystemExit("refusing existing output: {}".format(args.out))
    audit = json.load(open(args.audit))
    data_sha = sha256_file(args.data)
    required = ("duplicate_train_prompts", "duplicate_heldout_prompts", "train_heldout_exact_prompt_hits", "train_heldout_13gram_hits")
    if audit.get("audit") != "native_residual_relay_v1" or audit.get("train_sha256") != data_sha or any(audit.get(key) for key in required):
        raise SystemExit("NRR audit does not admit requested data")

    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    tokenizer = Tokenizer.from_file(args.tokenizer)
    eos_id = tokenizer.token_to_id(args.eos)
    if eos_id is None:
        raise SystemExit("tokenizer EOS token missing")
    checkpoint = torch.load(args.init, map_location="cpu")
    cfg = GPTConfig(**checkpoint["cfg"])
    if not args.layer < cfg.n_layer - 1 or cfg.n_loop != 1:
        raise SystemExit("invalid NRR layer or unsupported recurrent model")
    examples, skipped = load_examples(args.data, tokenizer, cfg.seq_len)
    if args.max_examples:
        examples = examples[:args.max_examples]
    batches, batch_report = bucketed_batches(examples, args.batch_size, args.seed)
    if args.max_batches:
        batches = batches[:args.max_batches]
    total_steps = args.epochs * len(batches)
    print(json.dumps({"nrr": "native_residual_relay_v1", "examples": len(examples), "skipped": skipped,
                      "batch_report": batch_report, "steps": total_steps, "layer": args.layer,
                      "max_batches": args.max_batches,
                      "data_sha256": data_sha}, sort_keys=True), flush=True)

    model = GPT(cfg).to("cuda")
    model.load_state_dict(checkpoint["model"])
    muon_params, adam_params = split_params(model)
    opt_muon = Muon(muon_params, lr=args.lr_muon)
    opt_adam = torch.optim.AdamW(adam_params, lr=args.lr_adam, betas=(0.9, 0.95), weight_decay=0.0)
    started, step = time.time(), 0
    for epoch in range(args.epochs):
        epoch_batches, _ = bucketed_batches(examples, args.batch_size, args.seed + epoch)
        if args.max_batches:
            epoch_batches = epoch_batches[:args.max_batches]
        for indices in epoch_batches:
            source, suffix, answer = make_batch(examples, indices, "cuda")
            scale = lr_scale(step, total_steps, args.warmup)
            for group in opt_muon.param_groups:
                group["lr"] = args.lr_muon * scale
            for group in opt_adam.param_groups:
                group["lr"] = args.lr_adam * scale
            opt_muon.zero_grad(set_to_none=True)
            opt_adam.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, loss, relay, _ = supervised_relay_loss(model, source, suffix, answer, args.layer, eos_id)
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite NRR loss at {}".format(step))
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            if not torch.isfinite(grad_norm):
                raise RuntimeError("non-finite NRR gradient at {}".format(step))
            opt_muon.step()
            opt_adam.step()
            if step % args.log_every == 0:
                print("[nrr] epoch={} step={}/{} loss={:.4f} gnorm={:.3f} relay_norm={:.3f} lr={:.6f} {}s".format(
                    epoch, step, total_steps, loss.item(), float(grad_norm), relay.float().norm(dim=-1).mean().item(),
                    args.lr_muon * scale, int(time.time() - started)), flush=True)
            step += 1
    os.makedirs(args.out)
    output = os.path.join(args.out, "nrr_ep1.pt")
    torch.save({
        "model": model.state_dict(), "cfg": cfg.__dict__, "step": "nrr_ep1",
        "native_residual_relay": {
            "layer": args.layer,
            "data_sha256": data_sha,
            "source_present_at_suffix": False,
            "extra_trainable_parameters": 0,
            "paraphrase_sources": True,
            "train_examples": len(examples),
            "max_batches_per_epoch": args.max_batches,
            "init_sha256": sha256_file(args.init),
        },
    }, output)
    print("[nrr] wrote {} after {} steps in {}s".format(output, step, int(time.time() - started)), flush=True)


if __name__ == "__main__":
    main()
