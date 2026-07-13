#!/usr/bin/env python3
"""Train source-free packet memory with verified latent-state geometry.

This is an isolated research trainer.  Answer decoding receives only the
learned packet and a fresh query.  Numeric state targets and pair relations are
training-only signals; they are never serialized into decoder prompts or used
by inference.  Zero-auxiliary, shuffled-pair, and permuted-state-code modes
are explicit causal controls for the same data and answer loss.
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

from latent_state_algebra import LatentStateAlgebra
from model import GPT, GPTConfig
from muon import Muon, split_params
from source_dropping_memory import SourceDroppingMemory


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_admits_training(audit, data_sha256: str) -> bool:
    failures = (
        "invalid_train_rows", "invalid_eval_rows", "duplicate_train_prompts", "duplicate_eval_prompts",
        "duplicate_train_references", "duplicate_eval_references",
        "train_eval_exact_prompt_hits", "train_eval_13gram_hits", "invalid_train_pairs", "invalid_eval_pairs",
    )
    return (
        audit.get("audit") == "latent_state_algebra_v1"
        and audit.get("train_sha256") == data_sha256
        and not any(audit.get(key) for key in failures)
        and set(audit.get("train_pair_kinds", {})) == {"equivalent", "intervention"}
        and set(audit.get("eval_pair_kinds", {})) == {"equivalent", "intervention"}
    )


def tokenize_row(row, tokenizer, slots: int, max_chunks: int, seq_len: int):
    chunks, query, response = row.get("chunks"), row.get("query"), row.get("response")
    if not isinstance(chunks, list) or not chunks or not all(isinstance(chunk, str) for chunk in chunks):
        raise ValueError("invalid source chunks")
    if not isinstance(query, str) or not isinstance(response, str):
        raise ValueError("invalid query or response")
    if len(chunks) > max_chunks:
        raise ValueError("too many chunks")
    chunk_ids = [tokenizer.encode(chunk).ids for chunk in chunks]
    query_ids = tokenizer.encode(query).ids
    answer_ids = tokenizer.encode(" " + response.strip()).ids
    if not query_ids or not answer_ids or any(not chunk for chunk in chunk_ids):
        raise ValueError("empty tokenization")
    if any(len(chunk) + 2 * slots > seq_len for chunk in chunk_ids):
        raise ValueError("source chunk plus slots exceeds sequence length")
    if slots + len(query_ids) + len(answer_ids) > seq_len:
        raise ValueError("answer context exceeds sequence length")
    state = row.get("state")
    if not isinstance(state, list) or len(state) != 2 or int(row.get("state_scale", 0)) <= 0:
        raise ValueError("invalid training-only state target")
    return {
        "chunks": chunk_ids,
        "query": query_ids,
        "answer": answer_ids,
        "state": [float(value) / float(row["state_scale"]) for value in state],
        "shape": (len(chunk_ids), tuple(len(chunk) for chunk in chunk_ids), len(query_ids), len(answer_ids)),
    }


def load_pairs(path, tokenizer, slots: int, max_chunks: int, seq_len: int):
    grouped = defaultdict(list)
    skipped = defaultdict(int)
    with open(path) as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            try:
                pair_id = row["pair_id"]
                member = row["pair_member"]
                pair_kind = row["pair_kind"]
                if member not in {"a", "b"} or pair_kind not in {"equivalent", "intervention"}:
                    raise ValueError("invalid pair metadata")
                item = tokenize_row(row, tokenizer, slots, max_chunks, seq_len)
                item["member"] = member
                item["pair_kind"] = pair_kind
                item["line"] = line_number
                grouped[pair_id].append(item)
            except (KeyError, TypeError, ValueError) as exc:
                skipped[str(exc)] += 1
    pairs = []
    for pair_id, rows in sorted(grouped.items()):
        if len(rows) != 2 or {row["member"] for row in rows} != {"a", "b"}:
            skipped["incomplete_pair"] += len(rows)
            continue
        first, second = sorted(rows, key=lambda row: row["member"])
        if first["pair_kind"] != second["pair_kind"]:
            skipped["mixed_pair_kind"] += 2
            continue
        pairs.append({
            "a": first,
            "b": second,
            "equivalent": first["pair_kind"] == "equivalent",
            "shape": (first["shape"], second["shape"]),
        })
    if not pairs:
        raise ValueError("no fitting complete latent-state-algebra pairs")
    return pairs, dict(sorted(skipped.items()))


def bucketed_batches(pairs, batch_size: int, seed: int):
    if batch_size <= 1:
        raise ValueError("batch-size must be at least two for pair contrastive supervision")
    buckets = defaultdict(list)
    for index, pair in enumerate(pairs):
        buckets[pair["shape"]].append(index)
    rng = random.Random(seed)
    batches, dropped = [], 0
    for shape in sorted(buckets, key=str):
        indices = list(buckets[shape])
        rng.shuffle(indices)
        usable = (len(indices) // batch_size) * batch_size
        dropped += len(indices) - usable
        batches.extend(indices[offset:offset + batch_size] for offset in range(0, usable, batch_size))
    rng.shuffle(batches)
    if not batches:
        raise ValueError("no full batches; reduce --batch-size or improve pair shape coverage")
    return batches, {"buckets": len(buckets), "full_batches": len(batches), "dropped_pairs": dropped}


def limit_batches(batches, max_pairs: int, batch_size: int):
    if max_pairs <= 0:
        return batches
    count = max_pairs // batch_size
    if count <= 0:
        raise ValueError("max-pairs must cover at least one complete batch")
    return batches[:count]


def make_side_batch(pairs, indices, side: str, device: str):
    examples = [pairs[index][side] for index in indices]
    chunk_count = len(examples[0]["chunks"])
    chunks = tuple(
        torch.tensor([example["chunks"][chunk] for example in examples], dtype=torch.long, device=device)
        for chunk in range(chunk_count)
    )
    query = torch.tensor([example["query"] for example in examples], dtype=torch.long, device=device)
    answer = torch.tensor([example["answer"] for example in examples], dtype=torch.long, device=device)
    state = torch.tensor([example["state"] for example in examples], dtype=torch.float32, device=device)
    return chunks, query, answer, state


def make_pair_batch(pairs, indices, device: str):
    chunks_a, query_a, answer_a, state_a = make_side_batch(pairs, indices, "a", device)
    chunks_b, query_b, answer_b, state_b = make_side_batch(pairs, indices, "b", device)
    equivalent = torch.tensor([pairs[index]["equivalent"] for index in indices], dtype=torch.bool, device=device)
    return chunks_a, query_a, answer_a, state_a, chunks_b, query_b, answer_b, state_b, equivalent


def nonidentity_permutation(size: int, device: str):
    """Return a batch permutation that cannot leave every verified pair intact."""
    order = torch.randperm(size, device=device)
    identity = torch.arange(size, device=device)
    if torch.equal(order, identity):
        order = torch.roll(order, shifts=1)
    return order


def lr_scale(step: int, total_steps: int, warmup: int) -> float:
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


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--audit", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--slots", type=int, default=8)
    parser.add_argument("--max-chunks", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-pairs", type=int, default=0)
    parser.add_argument("--lr-muon", type=float, default=2e-3)
    parser.add_argument("--lr-adam", type=float, default=5e-4)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--pair-mode", choices=("verified", "shuffled"), default="verified")
    parser.add_argument("--state-mode", choices=("verified", "permuted"), default="verified")
    parser.add_argument("--weight-alignment", type=float, default=0.1)
    parser.add_argument("--weight-contrastive", type=float, default=0.05)
    parser.add_argument("--weight-separation", type=float, default=0.1)
    parser.add_argument("--weight-state", type=float, default=1.0)
    parser.add_argument("--weight-delta", type=float, default=0.5)
    parser.add_argument("--zero-auxiliary", action="store_true")
    parser.add_argument("--eos", default="<|endoftext|>")
    parser.add_argument("--log-every", type=int, default=20)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.slots <= 0 or args.max_chunks <= 0 or args.epochs <= 0:
        raise SystemExit("slots, max-chunks, and epochs must be positive")
    if not torch.cuda.is_available():
        raise SystemExit("latent-state-algebra training requires a CUDA allocation")
    if os.path.exists(args.out) and os.listdir(args.out):
        raise SystemExit("refusing non-empty output directory: {}".format(args.out))
    weights = {
        "alignment": args.weight_alignment,
        "contrastive": args.weight_contrastive,
        "separation": args.weight_separation,
        "state": args.weight_state,
        "delta": args.weight_delta,
    }
    if any(value < 0 for value in weights.values()):
        raise SystemExit("auxiliary weights must be non-negative")
    if args.zero_auxiliary:
        weights = {name: 0.0 for name in weights}

    audit = json.load(open(args.audit))
    data_sha256 = sha256_file(args.data)
    if not audit_admits_training(audit, data_sha256):
        raise SystemExit("latent-state-algebra audit does not admit requested data")
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    tokenizer = Tokenizer.from_file(args.tokenizer)
    eos_id = tokenizer.token_to_id(args.eos)
    if eos_id is None:
        raise SystemExit("tokenizer EOS token is missing")
    init = torch.load(args.init, map_location="cpu")
    cfg = GPTConfig(**init["cfg"])
    pairs, skipped = load_pairs(args.data, tokenizer, args.slots, args.max_chunks, cfg.seq_len)
    batches, batch_report = bucketed_batches(pairs, args.batch_size, args.seed)
    batches = limit_batches(batches, args.max_pairs, args.batch_size)
    batch_report["selected_batches"] = len(batches)
    batch_report["selected_pairs"] = len(batches) * args.batch_size
    total_steps = args.epochs * len(batches)
    print(json.dumps({
        "latent_state_algebra": "source_removed_v1",
        "pairs": len(pairs),
        "skipped": skipped,
        "batch_report": batch_report,
        "total_steps": total_steps,
        "slots": args.slots,
        "max_chunks": args.max_chunks,
        "data_sha256": data_sha256,
        "pair_mode": args.pair_mode,
        "state_mode": args.state_mode,
        "weights": weights,
    }, sort_keys=True), flush=True)

    model = GPT(cfg).to("cuda")
    model.load_state_dict(init["model"])
    memory = SourceDroppingMemory(model, args.slots, args.max_chunks).to("cuda")
    auxiliary = LatentStateAlgebra(cfg.d_model, state_dim=2).to("cuda")
    muon_memory, adam_memory = split_params(memory)
    muon_auxiliary, adam_auxiliary = split_params(auxiliary)
    opt_muon = Muon(muon_memory + muon_auxiliary, lr=args.lr_muon)
    opt_adam = torch.optim.AdamW(
        adam_memory + adam_auxiliary, lr=args.lr_adam, betas=(0.9, 0.95), weight_decay=0.0,
    )
    started, step = time.time(), 0
    for epoch in range(args.epochs):
        epoch_batches, _ = bucketed_batches(pairs, args.batch_size, args.seed + epoch)
        epoch_batches = limit_batches(epoch_batches, args.max_pairs, args.batch_size)
        for indices in epoch_batches:
            (
                chunks_a, query_a, answer_a, state_a,
                chunks_b, query_b, answer_b, state_b, equivalent,
            ) = make_pair_batch(pairs, indices, "cuda")
            scale = lr_scale(step, total_steps, args.warmup)
            for group in opt_muon.param_groups:
                group["lr"] = args.lr_muon * scale
            for group in opt_adam.param_groups:
                group["lr"] = args.lr_adam * scale
            opt_muon.zero_grad(set_to_none=True)
            opt_adam.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, answer_loss_a, packet_a, _ = memory.supervised_loss(chunks_a, query_a, answer_a, eos_id)
                _, answer_loss_b, packet_b, _ = memory.supervised_loss(chunks_b, query_b, answer_b, eos_id)
                answer_loss = 0.5 * (answer_loss_a + answer_loss_b)
                aux_packet_b, aux_state_b = packet_b, state_b
                if args.pair_mode == "shuffled":
                    order = nonidentity_permutation(packet_b.shape[0], packet_b.device)
                    aux_packet_b, aux_state_b = packet_b[order], state_b[order]
                aux_state_a = state_a
                if args.state_mode == "permuted":
                    order = nonidentity_permutation(state_a.shape[0], state_a.device)
                    aux_state_a, aux_state_b = state_a[order], aux_state_b[order]
                losses = auxiliary.losses(packet_a, aux_packet_b, aux_state_a, aux_state_b, equivalent)
                auxiliary_loss = LatentStateAlgebra.total(losses, weights)
                loss = answer_loss + auxiliary_loss
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite latent-state-algebra loss at step {}".format(step))
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(list(memory.parameters()) + list(auxiliary.parameters()), args.clip)
            if not torch.isfinite(grad_norm):
                raise RuntimeError("non-finite latent-state-algebra gradient at step {}".format(step))
            opt_muon.step()
            opt_adam.step()
            if step % args.log_every == 0:
                pairs_per_second = args.batch_size * (step + 1) / max(1e-6, time.time() - started)
                print(
                    "[lsa] epoch={} step={}/{} loss={:.4f} answer={:.4f} aux={:.4f} "
                    "state={:.4f} delta={:.4f} align={:.4f} sep={:.4f} gnorm={:.3f} lr={:.6f} pair/s={:.1f}".format(
                        epoch, step, total_steps, loss.item(), answer_loss.item(), auxiliary_loss.item(),
                        losses["state"].item(), losses["delta"].item(), losses["alignment"].item(),
                        losses["separation"].item(), float(grad_norm), args.lr_muon * scale, pairs_per_second,
                    ),
                    flush=True,
                )
            step += 1
    output = os.path.join(args.out, "latent_state_algebra_ep1.pt")
    torch.save({
        "model": memory.model.state_dict(),
        "cfg": cfg.__dict__,
        "step": "latent_state_algebra_ep1",
        "source_dropping_memory": {
            "protocol": "fixed_slots_source_removed_v1",
            "init": args.init,
            "data": args.data,
            "data_sha256": data_sha256,
            "slots": args.slots,
            "max_chunks": args.max_chunks,
            "seed": args.seed,
            "updates": step,
            "batch_size": args.batch_size,
            "source_present_at_decode": False,
            "claim_boundary": "Isolated source-free latent-state-algebra experiment; not broad reasoning.",
        },
        "latent_state_algebra": {
            "state_dim": 2,
            "weights": weights,
            "pair_mode": args.pair_mode,
            "state_mode": args.state_mode,
            "state_scale": 256,
            "training_only": True,
        },
        "memory_state": non_model_state(memory),
        "latent_state_algebra_state": auxiliary.state_dict(),
    }, output)
    print("[lsa] saved {}".format(output), flush=True)
    print("[lsa] done {} updates in {:.0f}s".format(step, time.time() - started), flush=True)


if __name__ == "__main__":
    main()
