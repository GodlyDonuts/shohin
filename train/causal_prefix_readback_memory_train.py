#!/usr/bin/env python3
"""Train source-free packets to support semantic readback at every prefix.

This is a bounded follow-up for the specific failure mode where an auxiliary
state probe fits a packet but the language decoder cannot use that packet until
the final source boundary.  Each source write is therefore followed by an
answer loss from the *same* language decoder on a fresh, solver-derived query.
No decoder pass receives source tokens.

The required controls keep the same data, initialization, optimizer, batch
order, number of decoder losses, and decode-token distribution:
``verified`` uses each prefix packet with its true readback; ``shuffled``
attaches a different example's readback to each prefix packet; and
``replicated-final`` repeats the ordinary final query from the final packet.
The last condition isolates temporal readback feedback from merely doing more
decoder work.  This is isolated research, never the flagship trainer.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections import defaultdict

import torch
from tokenizers import Tokenizer

from causal_prefix_readback import prefix_readback_targets, validate_readback_targets
from latent_state_algebra_train import (
    audit_admits_training,
    bucketed_batches,
    limit_batches,
    lr_scale,
    non_model_state,
    nonidentity_permutation,
    sha256_file,
    tokenize_row,
)
from model import GPT, GPTConfig
from muon import Muon, split_params
from source_dropping_memory import SourceDroppingMemory


READBACK_MODES = ("verified", "shuffled", "replicated-final")


def tokenize_readbacks(row, tokenizer, slots, seq_len):
    targets = prefix_readback_targets(row["initial"], row["operations"], row["keys"], row["state"])
    validate_readback_targets(targets, len(row["chunks"]))
    readbacks = []
    for target in targets:
        query = tokenizer.encode(str(target["query"])).ids
        answer = tokenizer.encode(" " + str(target["answer"])).ids
        if not query or not answer:
            raise ValueError("empty tokenization in prefix readback")
        if slots + len(query) + len(answer) > seq_len:
            raise ValueError("prefix readback exceeds decoder context")
        readbacks.append({"query": query, "answer": answer, "target": target})
    return readbacks


def load_readback_pairs(path, tokenizer, slots, max_chunks, seq_len):
    """Load complete audited pairs and bind deterministic prefix readbacks."""
    grouped, skipped = defaultdict(list), defaultdict(int)
    with open(path) as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            try:
                pair_id, member, pair_kind = row["pair_id"], row["pair_member"], row["pair_kind"]
                if member not in {"a", "b"} or pair_kind not in {"equivalent", "intervention"}:
                    raise ValueError("invalid pair metadata")
                item = tokenize_row(row, tokenizer, slots, max_chunks, seq_len)
                readbacks = tokenize_readbacks(row, tokenizer, slots, seq_len)
                if len(readbacks) != len(item["chunks"]):
                    raise ValueError("readback count does not match tokenized chunks")
                item.update({
                    "member": member,
                    "pair_kind": pair_kind,
                    "readbacks": readbacks,
                    "line": line_number,
                })
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
        pairs.append({"a": first, "b": second, "shape": (first["shape"], second["shape"])})
    if not pairs:
        raise ValueError("no fitting complete causal-prefix-readback pairs")
    return pairs, dict(sorted(skipped.items()))


def make_side_batch(pairs, indices, side, device):
    examples = [pairs[index][side] for index in indices]
    chunk_count = len(examples[0]["chunks"])
    chunks = tuple(
        torch.tensor([example["chunks"][chunk] for example in examples], dtype=torch.long, device=device)
        for chunk in range(chunk_count)
    )
    final_query = torch.tensor([example["query"] for example in examples], dtype=torch.long, device=device)
    final_answer = torch.tensor([example["answer"] for example in examples], dtype=torch.long, device=device)
    readback_targets = tuple(
        tuple(example["readbacks"][prefix] for example in examples)
        for prefix in range(chunk_count)
    )
    return chunks, final_query, final_answer, readback_targets


def make_pair_batch(pairs, indices, device):
    return (*make_side_batch(pairs, indices, "a", device), *make_side_batch(pairs, indices, "b", device))


def readback_label_assignment(readbacks, mode):
    """Preserve or shuffle complete query/answer labels across receiver packets."""
    if mode not in READBACK_MODES:
        raise ValueError("unknown readback mode")
    if not readbacks:
        raise ValueError("readback labels must be non-empty")
    if mode == "verified":
        return tuple(readbacks)
    if mode != "shuffled":
        raise ValueError("label assignment is only defined for verified or shuffled modes")
    order = nonidentity_permutation(len(readbacks), "cpu").tolist()
    return tuple(readbacks[index] for index in order)


def grouped_readback_loss(memory, packet, readbacks, eos_id, mode):
    """Decode variable-length semantic readbacks without padding source packets.

    The packet writer is still batched by the original source/final-answer
    shape.  Only source-free decoder readbacks are split into their compatible
    query/answer shapes, preventing extra length buckets from discarding
    otherwise valid source trajectories.
    """
    assigned = readback_label_assignment(readbacks, mode)
    groups = defaultdict(list)
    for receiver, target in enumerate(assigned):
        groups[(len(target["query"]), len(target["answer"]))].append((receiver, target))
    losses = []
    for _, items in sorted(groups.items()):
        receivers = [receiver for receiver, _ in items]
        queries = torch.tensor([target["query"] for _, target in items], dtype=torch.long, device=packet.device)
        answers = torch.tensor([target["answer"] for _, target in items], dtype=torch.long, device=packet.device)
        _, loss, _ = memory.supervised_loss_from_packet(packet[receivers], queries, answers, eos_id)
        losses.append(loss)
    return torch.stack(losses).mean()


def mean_readback_loss(memory, trace, final_packet, final_query, final_answer, readback_targets, eos_id, mode):
    """Score all prefix readbacks without exposing any source token to decoding."""
    losses = []
    if not trace or len(trace) != len(readback_targets):
        raise ValueError("readback trace and labels must have the same nonzero length")
    for packet, targets in zip(trace, readback_targets):
        if mode == "replicated-final":
            _, loss, _ = memory.supervised_loss_from_packet(final_packet, final_query, final_answer, eos_id)
        else:
            loss = grouped_readback_loss(memory, packet, targets, eos_id, mode)
        losses.append(loss)
    return torch.stack(losses).mean()


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
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--readback-mode", choices=READBACK_MODES, default="verified")
    parser.add_argument("--readback-weight", type=float, default=1.0)
    parser.add_argument("--eos", default="<|endoftext|>")
    parser.add_argument("--log-every", type=int, default=20)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.slots <= 0 or args.max_chunks <= 0 or args.epochs <= 0 or args.batch_size <= 1:
        raise SystemExit("slots, max-chunks, epochs, and batch-size must be valid")
    if args.readback_weight < 0:
        raise SystemExit("readback weight must be non-negative")
    if not torch.cuda.is_available():
        raise SystemExit("causal-prefix-readback training requires a CUDA allocation")
    if os.path.exists(args.out) and os.listdir(args.out):
        raise SystemExit("refusing non-empty output directory: {}".format(args.out))
    data_sha256 = sha256_file(args.data)
    audit = json.load(open(args.audit))
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
    pairs, skipped = load_readback_pairs(args.data, tokenizer, args.slots, args.max_chunks, cfg.seq_len)
    batches, batch_report = bucketed_batches(pairs, args.batch_size, args.seed)
    batches = limit_batches(batches, args.max_pairs, args.batch_size)
    total_steps = args.epochs * len(batches)
    print(json.dumps({
        "causal_prefix_readback": "source_removed_v1",
        "pairs": len(pairs), "skipped": skipped, "batch_report": batch_report,
        "total_steps": total_steps, "data_sha256": data_sha256,
        "readback_mode": args.readback_mode, "readback_weight": args.readback_weight,
    }, sort_keys=True), flush=True)

    model = GPT(cfg).to("cuda")
    model.load_state_dict(init["model"])
    memory = SourceDroppingMemory(model, args.slots, args.max_chunks).to("cuda")
    muon_memory, adam_memory = split_params(memory)
    opt_muon = Muon(muon_memory, lr=args.lr_muon)
    opt_adam = torch.optim.AdamW(adam_memory, lr=args.lr_adam, betas=(0.9, 0.95), weight_decay=0.0)
    started, step = time.time(), 0
    for epoch in range(args.epochs):
        epoch_batches, _ = bucketed_batches(pairs, args.batch_size, args.seed + epoch)
        epoch_batches = limit_batches(epoch_batches, args.max_pairs, args.batch_size)
        for indices in epoch_batches:
            (
                chunks_a, final_query_a, final_answer_a, readback_targets_a,
                chunks_b, final_query_b, final_answer_b, readback_targets_b,
            ) = make_pair_batch(pairs, indices, "cuda")
            scale = lr_scale(step, total_steps, args.warmup)
            for group in opt_muon.param_groups:
                group["lr"] = args.lr_muon * scale
            for group in opt_adam.param_groups:
                group["lr"] = args.lr_adam * scale
            opt_muon.zero_grad(set_to_none=True)
            opt_adam.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                packet_a, trace_a = memory.encode(chunks_a, return_trace=True)
                packet_b, trace_b = memory.encode(chunks_b, return_trace=True)
                _, final_loss_a, _ = memory.supervised_loss_from_packet(packet_a, final_query_a, final_answer_a, eos_id)
                _, final_loss_b, _ = memory.supervised_loss_from_packet(packet_b, final_query_b, final_answer_b, eos_id)
                final_loss = 0.5 * (final_loss_a + final_loss_b)
                readback_loss_a = mean_readback_loss(
                    memory, trace_a, packet_a, final_query_a, final_answer_a,
                    readback_targets_a, eos_id, args.readback_mode,
                )
                readback_loss_b = mean_readback_loss(
                    memory, trace_b, packet_b, final_query_b, final_answer_b,
                    readback_targets_b, eos_id, args.readback_mode,
                )
                readback_loss = 0.5 * (readback_loss_a + readback_loss_b)
                loss = final_loss + args.readback_weight * readback_loss
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite causal-prefix-readback loss at step {}".format(step))
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(memory.parameters(), args.clip)
            if not torch.isfinite(grad_norm):
                raise RuntimeError("non-finite causal-prefix-readback gradient at step {}".format(step))
            opt_muon.step()
            opt_adam.step()
            if step % args.log_every == 0:
                pair_rate = args.batch_size * (step + 1) / max(1e-6, time.time() - started)
                print(
                    "[causal-prefix] epoch={} step={}/{} loss={:.4f} final={:.4f} readback={:.4f} "
                    "gnorm={:.3f} lr={:.6f} pair/s={:.1f}".format(
                        epoch, step, total_steps, loss.item(), final_loss.item(), readback_loss.item(),
                        float(grad_norm), args.lr_muon * scale, pair_rate,
                    ), flush=True,
                )
            step += 1
    output = os.path.join(args.out, "causal_prefix_readback_ep1.pt")
    torch.save({
        "model": memory.model.state_dict(), "cfg": cfg.__dict__, "step": "causal_prefix_readback_ep1",
        "source_dropping_memory": {
            "protocol": "fixed_slots_source_removed_causal_prefix_readback_v1",
            "init": args.init, "data": args.data, "data_sha256": data_sha256,
            "slots": args.slots, "max_chunks": args.max_chunks, "seed": args.seed,
            "updates": step, "batch_size": args.batch_size, "source_present_at_decode": False,
            "claim_boundary": "Isolated source-free prefix-readback memory experiment; not broad reasoning.",
        },
        "causal_prefix_readback": {
            "target_protocol": "solver_recomputed_source_free_prefix_readback_v1",
            "readback_mode": args.readback_mode, "readback_weight": args.readback_weight,
            "decoder_readback_at_every_prefix": True,
            "equal_decoder_work_control": args.readback_mode == "replicated-final",
            "training_only": True,
        },
        "memory_state": non_model_state(memory),
    }, output)
    print("[causal-prefix] saved {}".format(output), flush=True)
    print("[causal-prefix] done {} updates in {:.0f}s".format(step, time.time() - started), flush=True)


if __name__ == "__main__":
    main()
