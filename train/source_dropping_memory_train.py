#!/usr/bin/env python3
"""Train an isolated fixed-slot, source-dropping continuous memory packet.

Input rows must contain ``chunks`` (a non-empty list of source strings), a
fresh ``query`` string, and an answer-only ``response``. Every selected batch
has exactly matching chunk/query/answer token shapes; no padding token becomes
an accidental memory channel. The final answer loss is computed after source
removal by ``SourceDroppingMemory``.
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
from source_dropping_memory import SourceDroppingMemory


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_examples(path, tokenizer, slots, max_chunks, seq_len):
    examples = []
    skipped = defaultdict(int)
    with open(path) as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            chunks, query, response = row.get("chunks"), row.get("query"), row.get("response")
            if not isinstance(chunks, list) or not chunks or not all(isinstance(chunk, str) for chunk in chunks):
                skipped["invalid_chunks"] += 1
                continue
            if not isinstance(query, str) or not isinstance(response, str):
                skipped["missing_query_or_response"] += 1
                continue
            if len(chunks) > max_chunks:
                skipped["too_many_chunks"] += 1
                continue
            chunk_ids = [tokenizer.encode(chunk).ids for chunk in chunks]
            query_ids = tokenizer.encode(query).ids
            answer_ids = tokenizer.encode(" " + response.strip()).ids
            if not query_ids or not answer_ids or any(not chunk for chunk in chunk_ids):
                skipped["empty_tokens"] += 1
                continue
            if any(len(chunk) + 2 * slots > seq_len for chunk in chunk_ids):
                skipped["overlong_chunk"] += 1
                continue
            if slots + len(query_ids) + len(answer_ids) > seq_len:
                skipped["overlong_answer"] += 1
                continue
            shape = (len(chunk_ids), tuple(len(chunk) for chunk in chunk_ids), len(query_ids), len(answer_ids))
            examples.append({
                "chunks": chunk_ids,
                "query": query_ids,
                "answer": answer_ids,
                "shape": shape,
                "line": line_number,
            })
    if not examples:
        raise ValueError("no fitting source-dropping memory examples in {}".format(path))
    return examples, dict(sorted(skipped.items()))


def bucketed_batches(examples, batch_size, seed):
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    buckets = defaultdict(list)
    for index, example in enumerate(examples):
        buckets[example["shape"]].append(index)
    rng = random.Random(seed)
    batches, dropped = [], 0
    for shape in sorted(buckets):
        indices = list(buckets[shape])
        rng.shuffle(indices)
        usable = (len(indices) // batch_size) * batch_size
        dropped += len(indices) - usable
        batches.extend(indices[offset:offset + batch_size] for offset in range(0, usable, batch_size))
    rng.shuffle(batches)
    if not batches:
        raise ValueError("no full batches; reduce --batch-size or improve shape coverage")
    return batches, {"buckets": len(buckets), "full_batches": len(batches), "dropped_examples": dropped}


def make_batch(examples, indices, device):
    chunks = torch.tensor([examples[index]["chunks"] for index in indices], dtype=torch.long, device=device)
    query = torch.tensor([examples[index]["query"] for index in indices], dtype=torch.long, device=device)
    answer = torch.tensor([examples[index]["answer"] for index in indices], dtype=torch.long, device=device)
    return chunks, query, answer


def lr_scale(step, total_steps, warmup):
    if step < warmup:
        return step / max(1, warmup)
    progress = (step - warmup) / max(1, total_steps - warmup)
    return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))


def non_model_state(memory):
    return {
        name: value.detach().cpu()
        for name, value in memory.state_dict().items()
        if not name.startswith("model.")
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--slots", type=int, default=8)
    parser.add_argument("--max-chunks", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr-muon", type=float, default=2e-3)
    parser.add_argument("--lr-adam", type=float, default=5e-4)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--eos", default="<|endoftext|>")
    parser.add_argument("--log-every", type=int, default=20)
    args = parser.parse_args()
    if args.slots < 0 or args.max_chunks <= 0 or args.epochs <= 0:
        raise SystemExit("slots must be non-negative; max-chunks and epochs must be positive")
    if not torch.cuda.is_available():
        raise SystemExit("source-dropping memory training requires a CUDA allocation")
    if os.path.exists(args.out) and os.listdir(args.out):
        raise SystemExit("refusing non-empty output directory: {}".format(args.out))
    os.makedirs(args.out, exist_ok=True)

    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    tokenizer = Tokenizer.from_file(args.tokenizer)
    eos_id = tokenizer.token_to_id(args.eos)
    if eos_id is None:
        raise SystemExit("tokenizer EOS token is missing")
    init = torch.load(args.init, map_location="cpu")
    cfg = GPTConfig(**init["cfg"])
    examples, skipped = load_examples(args.data, tokenizer, args.slots, args.max_chunks, cfg.seq_len)
    batches, batch_report = bucketed_batches(examples, args.batch_size, args.seed)
    total_steps = args.epochs * len(batches)
    print(json.dumps({
        "source_dropping_memory": "fixed_slots_v1",
        "examples": len(examples),
        "skipped": skipped,
        "batch_report": batch_report,
        "total_steps": total_steps,
        "slots": args.slots,
        "max_chunks": args.max_chunks,
        "data_sha256": sha256_file(args.data),
    }, sort_keys=True), flush=True)

    model = GPT(cfg).to("cuda")
    model.load_state_dict(init["model"])
    memory = SourceDroppingMemory(model, args.slots, args.max_chunks).to("cuda")
    muon_parameters, adam_parameters = split_params(memory)
    opt_muon = Muon(muon_parameters, lr=args.lr_muon)
    opt_adam = torch.optim.AdamW(adam_parameters, lr=args.lr_adam, betas=(0.9, 0.95), weight_decay=0.0)
    started, step = time.time(), 0
    for epoch in range(args.epochs):
        epoch_batches, _ = bucketed_batches(examples, args.batch_size, args.seed + epoch)
        for indices in epoch_batches:
            chunks, query, answer = make_batch(examples, indices, "cuda")
            scale = lr_scale(step, total_steps, args.warmup)
            for group in opt_muon.param_groups:
                group["lr"] = args.lr_muon * scale
            for group in opt_adam.param_groups:
                group["lr"] = args.lr_adam * scale
            opt_muon.zero_grad(set_to_none=True)
            opt_adam.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, loss, _, _ = memory.supervised_loss(chunks, query, answer, eos_id)
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite source-dropping memory loss at step {}".format(step))
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(memory.parameters(), args.clip)
            if not torch.isfinite(grad_norm):
                raise RuntimeError("non-finite source-dropping memory gradient at step {}".format(step))
            opt_muon.step()
            opt_adam.step()
            if step % args.log_every == 0:
                examples_per_second = args.batch_size * (step + 1) / max(1e-6, time.time() - started)
                print(
                    "[source-memory] epoch={} step={}/{} loss={:.4f} gnorm={:.3f} lr={:.6f} ex/s={:.1f}".format(
                        epoch, step, total_steps, loss.item(), float(grad_norm), args.lr_muon * scale,
                        examples_per_second,
                    ),
                    flush=True,
                )
            step += 1
    output = os.path.join(args.out, "source_memory_ep1.pt")
    torch.save({
        "model": memory.model.state_dict(),
        "cfg": cfg.__dict__,
        "step": "source_memory_ep1",
        "source_dropping_memory": {
            "protocol": "fixed_slots_source_removed_v1",
            "init": args.init,
            "data": args.data,
            "data_sha256": sha256_file(args.data),
            "slots": args.slots,
            "max_chunks": args.max_chunks,
            "seed": args.seed,
            "updates": step,
            "batch_size": args.batch_size,
            "source_present_at_decode": False,
            "claim_boundary": "Isolated source-removal memory experiment; not a broad reasoning claim.",
        },
        "memory_state": non_model_state(memory),
    }, output)
    print("[source-memory] saved {}".format(output), flush=True)
    print("[source-memory] done {} updates in {:.0f}s".format(step, time.time() - started), flush=True)


if __name__ == "__main__":
    main()
