#!/usr/bin/env python3
"""Train source-free memory with solver-verified state targets at every write.

This is an isolated follow-up to final-state-only latent algebra.  The writer
receives a loss after every source chunk, reducing delayed credit assignment;
the answer decoder still sees only the final continuous packet and a fresh
query.  It is not a flagship trainer and must be evaluated against matched
final-only, shuffled-prefix-label, and no-memory controls.
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

from latent_state_algebra import LatentStateAlgebra
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
from prefix_state_supervision import prefix_state_targets, prefix_trajectory_losses
from source_dropping_memory import SourceDroppingMemory


def prefix_targets_for_row(row):
    targets = prefix_state_targets(
        row["initial"], row["operations"], row["keys"], int(row["state_scale"]),
    )
    final = [float(value) / float(row["state_scale"]) for value in row["state"]]
    if len(targets[-1]) != len(final) or any(abs(a - b) > 1e-8 for a, b in zip(targets[-1], final)):
        raise ValueError("final prefix state does not match audited row state")
    return targets


def load_prefix_pairs(path, tokenizer, slots, max_chunks, seq_len):
    """Load the admitted LSA pair split with exact state targets per prefix."""
    grouped, skipped = defaultdict(list), defaultdict(int)
    with open(path) as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            try:
                pair_id = row["pair_id"]
                member, pair_kind = row["pair_member"], row["pair_kind"]
                if member not in {"a", "b"} or pair_kind not in {"equivalent", "intervention"}:
                    raise ValueError("invalid pair metadata")
                item = tokenize_row(row, tokenizer, slots, max_chunks, seq_len)
                targets = prefix_targets_for_row(row)
                if len(targets) != len(item["chunks"]):
                    raise ValueError("prefix target count does not match chunks")
                item.update({
                    "member": member,
                    "pair_kind": pair_kind,
                    "prefix_targets": targets,
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
        pairs.append({
            "a": first,
            "b": second,
            "shape": (first["shape"], second["shape"]),
        })
    if not pairs:
        raise ValueError("no fitting complete prefix-state pairs")
    return pairs, dict(sorted(skipped.items()))


def make_side_batch(pairs, indices, side, device):
    examples = [pairs[index][side] for index in indices]
    chunk_count = len(examples[0]["chunks"])
    chunks = tuple(
        torch.tensor([example["chunks"][chunk] for example in examples], dtype=torch.long, device=device)
        for chunk in range(chunk_count)
    )
    query = torch.tensor([example["query"] for example in examples], dtype=torch.long, device=device)
    answer = torch.tensor([example["answer"] for example in examples], dtype=torch.long, device=device)
    prefix = torch.tensor([example["prefix_targets"] for example in examples], dtype=torch.float32, device=device)
    return chunks, query, answer, prefix


def make_pair_batch(pairs, indices, device):
    return (*make_side_batch(pairs, indices, "a", device), *make_side_batch(pairs, indices, "b", device))


def shuffled_targets(targets, mode):
    if mode == "verified":
        return targets
    if mode == "shuffled":
        return targets[nonidentity_permutation(targets.shape[0], targets.device)]
    raise ValueError("unknown prefix target mode")


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
    parser.add_argument("--prefix-mode", choices=("verified", "shuffled"), default="verified")
    parser.add_argument("--weight-state", type=float, default=1.0)
    parser.add_argument("--weight-delta", type=float, default=0.5)
    parser.add_argument("--zero-auxiliary", action="store_true")
    parser.add_argument("--eos", default="<|endoftext|>")
    parser.add_argument("--log-every", type=int, default=20)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.slots <= 0 or args.max_chunks <= 0 or args.epochs <= 0 or args.batch_size <= 1:
        raise SystemExit("slots, max-chunks, epochs, and batch-size must be valid")
    if args.weight_state < 0 or args.weight_delta < 0:
        raise SystemExit("prefix auxiliary weights must be non-negative")
    if not torch.cuda.is_available():
        raise SystemExit("prefix-state memory training requires a CUDA allocation")
    if os.path.exists(args.out) and os.listdir(args.out):
        raise SystemExit("refusing non-empty output directory: {}".format(args.out))
    data_sha256 = sha256_file(args.data)
    audit = json.load(open(args.audit))
    if not audit_admits_training(audit, data_sha256):
        raise SystemExit("latent-state-algebra audit does not admit requested data")
    weights = {"state": args.weight_state, "delta": args.weight_delta}
    if args.zero_auxiliary:
        weights = {name: 0.0 for name in weights}
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    tokenizer = Tokenizer.from_file(args.tokenizer)
    eos_id = tokenizer.token_to_id(args.eos)
    if eos_id is None:
        raise SystemExit("tokenizer EOS token is missing")
    init = torch.load(args.init, map_location="cpu")
    cfg = GPTConfig(**init["cfg"])
    pairs, skipped = load_prefix_pairs(args.data, tokenizer, args.slots, args.max_chunks, cfg.seq_len)
    batches, batch_report = bucketed_batches(pairs, args.batch_size, args.seed)
    batches = limit_batches(batches, args.max_pairs, args.batch_size)
    total_steps = args.epochs * len(batches)
    print(json.dumps({
        "prefix_state_memory": "source_removed_v1",
        "pairs": len(pairs), "skipped": skipped, "batch_report": batch_report,
        "total_steps": total_steps, "data_sha256": data_sha256, "prefix_mode": args.prefix_mode,
        "weights": weights,
    }, sort_keys=True), flush=True)

    model = GPT(cfg).to("cuda")
    model.load_state_dict(init["model"])
    memory = SourceDroppingMemory(model, args.slots, args.max_chunks).to("cuda")
    state_probe = LatentStateAlgebra(cfg.d_model, state_dim=2).to("cuda")
    muon_memory, adam_memory = split_params(memory)
    muon_probe, adam_probe = split_params(state_probe)
    opt_muon = Muon(muon_memory + muon_probe, lr=args.lr_muon)
    opt_adam = torch.optim.AdamW(adam_memory + adam_probe, lr=args.lr_adam, betas=(0.9, 0.95), weight_decay=0.0)
    started, step = time.time(), 0
    for epoch in range(args.epochs):
        epoch_batches, _ = bucketed_batches(pairs, args.batch_size, args.seed + epoch)
        epoch_batches = limit_batches(epoch_batches, args.max_pairs, args.batch_size)
        for indices in epoch_batches:
            chunks_a, query_a, answer_a, targets_a, chunks_b, query_b, answer_b, targets_b = make_pair_batch(pairs, indices, "cuda")
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
                _, answer_loss_a, _ = memory.supervised_loss_from_packet(packet_a, query_a, answer_a, eos_id)
                _, answer_loss_b, _ = memory.supervised_loss_from_packet(packet_b, query_b, answer_b, eos_id)
                answer_loss = 0.5 * (answer_loss_a + answer_loss_b)
                losses_a = prefix_trajectory_losses(trace_a, shuffled_targets(targets_a, args.prefix_mode), state_probe.predict_state)
                losses_b = prefix_trajectory_losses(trace_b, shuffled_targets(targets_b, args.prefix_mode), state_probe.predict_state)
                state_loss = 0.5 * (losses_a["state"] + losses_b["state"])
                delta_loss = 0.5 * (losses_a["delta"] + losses_b["delta"])
                auxiliary_loss = weights["state"] * state_loss + weights["delta"] * delta_loss
                loss = answer_loss + auxiliary_loss
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite prefix-state loss at step {}".format(step))
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(list(memory.parameters()) + list(state_probe.parameters()), args.clip)
            if not torch.isfinite(grad_norm):
                raise RuntimeError("non-finite prefix-state gradient at step {}".format(step))
            opt_muon.step()
            opt_adam.step()
            if step % args.log_every == 0:
                pair_rate = args.batch_size * (step + 1) / max(1e-6, time.time() - started)
                print(
                    "[prefix-state] epoch={} step={}/{} loss={:.4f} answer={:.4f} aux={:.4f} "
                    "state={:.4f} delta={:.4f} gnorm={:.3f} lr={:.6f} pair/s={:.1f}".format(
                        epoch, step, total_steps, loss.item(), answer_loss.item(), auxiliary_loss.item(),
                        state_loss.item(), delta_loss.item(), float(grad_norm), args.lr_muon * scale, pair_rate,
                    ),
                    flush=True,
                )
            step += 1
    output = os.path.join(args.out, "prefix_state_memory_ep1.pt")
    torch.save({
        "model": memory.model.state_dict(), "cfg": cfg.__dict__, "step": "prefix_state_memory_ep1",
        "source_dropping_memory": {
            "protocol": "fixed_slots_source_removed_prefix_state_v1", "init": args.init, "data": args.data,
            "data_sha256": data_sha256, "slots": args.slots, "max_chunks": args.max_chunks,
            "seed": args.seed, "updates": step, "batch_size": args.batch_size,
            "source_present_at_decode": False,
            "claim_boundary": "Isolated prefix-supervised source-free memory experiment; not broad reasoning.",
        },
        "prefix_state_supervision": {
            "state_dim": 2, "state_scale": 256, "prefix_mode": args.prefix_mode,
            "weights": weights, "training_only": True,
        },
        "memory_state": non_model_state(memory), "prefix_state_probe": state_probe.state_dict(),
    }, output)
    print("[prefix-state] saved {}".format(output), flush=True)
    print("[prefix-state] done {} updates in {:.0f}s".format(step, time.time() - started), flush=True)


if __name__ == "__main__":
    main()
