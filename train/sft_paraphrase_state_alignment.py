#!/usr/bin/env python3
"""Isolated SFT with paraphrase-equivariant prompt-state alignment.

This trainer is for the semantic-basis experiment only.  Completion loss stays
the primary objective.  The optional alignment term makes the residual at one
answer-boundary layer agree for compile/reflect prompt pairs that encode the
same ledger.  ``--pair-mode mismatch`` is a deliberately non-semantic control.
It never changes the flagship trainer or a pretraining output.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from model import GPT, GPTConfig
from muon import Muon, split_params
from sft import build_packed


def expand_paths(items):
    paths = []
    for item in items:
        paths.extend(sorted(glob.glob(item)) if any(char in item for char in "*?[") else [item])
    return paths


def collect_state_pairs(paths):
    """Return independently worded compile/reflect prompts for the same ledger."""
    episodes = defaultdict(dict)
    for path in paths:
        with open(path) as source:
            for line_number, line in enumerate(source, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("schema") != "semantic_basis_transport_v2":
                    continue
                phase = row.get("phase")
                if phase in {"compile", "reflect"}:
                    episode_id = row.get("episode_id")
                    if not episode_id or phase in episodes[episode_id]:
                        raise ValueError("invalid paired row at {}:{}".format(path, line_number))
                    episodes[episode_id][phase] = row
    pairs = []
    for episode_id, rows in sorted(episodes.items()):
        if set(rows) != {"compile", "reflect"}:
            continue
        left, right = rows["compile"], rows["reflect"]
        if left.get("response") != right.get("response"):
            raise ValueError("pair {} has different state targets".format(episode_id))
        pairs.append({
            "episode_id": episode_id,
            "state": str(left["response"]),
            "left_prompt": "Question: {}\nAnswer:".format(left["question"]),
            "right_prompt": "Question: {}\nAnswer:".format(right["question"]),
        })
    if not pairs:
        raise ValueError("no compile/reflect state pairs found")
    return pairs


def tokenize_pairs(pairs, tokenizer, seq_len):
    encoded = []
    for pair in pairs:
        left = tokenizer.encode(pair["left_prompt"]).ids
        right = tokenizer.encode(pair["right_prompt"]).ids
        if not left or not right or max(len(left), len(right)) > seq_len:
            continue
        encoded.append({**pair, "left_ids": left, "right_ids": right})
    if not encoded:
        raise ValueError("no state pairs fit the model sequence length")
    return encoded


def pad_prompts(batch, field, pad_id, device):
    lengths = torch.tensor([len(item[field]) for item in batch], device=device, dtype=torch.long)
    ids = torch.full((len(batch), int(lengths.max())), pad_id, device=device, dtype=torch.long)
    for index, item in enumerate(batch):
        ids[index, :len(item[field])] = torch.tensor(item[field], device=device, dtype=torch.long)
    return ids, lengths


def mismatch_order(batch):
    """Pair every item with a different ledger for a matched non-semantic control."""
    if len(batch) < 2:
        raise ValueError("mismatch control requires at least two pairs")
    order = list(range(1, len(batch))) + [0]
    if any(batch[index]["state"] == batch[order[index]]["state"] for index in range(len(batch))):
        raise ValueError("mismatch control encountered duplicate state")
    return order


def select_distinct_pair_indices(pairs, order, cursor, count):
    """Choose a deterministic batch with no duplicate ledger targets."""
    if count <= 0:
        raise ValueError("pair batch size must be positive")
    if len({item["state"] for item in pairs}) < count:
        raise ValueError("not enough distinct ledger states for requested pair batch")
    selected, states, scanned = [], set(), 0
    while len(selected) < count and scanned < len(order):
        index = int(order[(cursor + scanned) % len(order)])
        scanned += 1
        if pairs[index]["state"] in states:
            continue
        selected.append(index)
        states.add(pairs[index]["state"])
    if len(selected) != count:
        raise ValueError("could not form a distinct-state pair batch")
    return selected, (cursor + scanned) % len(order)


class BoundaryCapture:
    """Capture one residual stream without changing the GPT forward contract."""
    def __init__(self, model, layer):
        if not 0 <= layer < len(model.blocks):
            raise ValueError("capture layer out of range")
        if model.cfg.n_loop != 1:
            raise ValueError("state-alignment experiment requires n_loop=1")
        self.hidden = None
        self.handle = model.blocks[layer].register_forward_hook(self._hook)

    def _hook(self, _module, _inputs, output):
        self.hidden = output[0]

    def boundary(self, model, ids, lengths):
        self.hidden = None
        model(ids)
        if self.hidden is None:
            raise RuntimeError("capture hook did not run")
        return self.hidden[torch.arange(ids.shape[0], device=ids.device), lengths - 1]

    def close(self):
        self.handle.remove()


def alignment_statistics(left, right):
    left32, right32 = left.float(), right.float()
    cosine = F.cosine_similarity(left32, right32, dim=-1)
    loss = (1.0 - cosine).mean()
    joined = torch.cat((left32, right32), dim=0)
    return loss, {
        "cosine": cosine.mean().detach(),
        "norm": joined.norm(dim=-1).mean().detach(),
        "variance": joined.var(dim=0, unbiased=False).mean().detach(),
    }


def contrastive_statistics(left, right, temperature):
    """Pairwise same-state identification with all other ledgers as negatives."""
    if left.shape[0] < 2:
        raise ValueError("contrastive alignment requires at least two distinct state pairs")
    if temperature <= 0:
        raise ValueError("contrastive temperature must be positive")
    left_norm = F.normalize(left.float(), dim=-1)
    right_norm = F.normalize(right.float(), dim=-1)
    similarity = left_norm @ right_norm.T
    labels = torch.arange(left.shape[0], device=left.device)
    loss = 0.5 * (
        F.cross_entropy(similarity / temperature, labels)
        + F.cross_entropy(similarity.T / temperature, labels)
    )
    negative = similarity.masked_fill(torch.eye(len(labels), device=left.device, dtype=torch.bool), -1.0)
    return loss, {
        "positive_cosine": similarity.diag().mean().detach(),
        "negative_cosine": negative.max(dim=-1).values.mean().detach(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init", required=True)
    parser.add_argument("--data", nargs="+", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--pair-batch-size", type=int, default=8)
    parser.add_argument("--capture-layer", type=int, default=19)
    parser.add_argument("--align-weight", type=float, default=0.05)
    parser.add_argument("--contrastive-weight", type=float, default=0.0)
    parser.add_argument("--contrastive-temperature", type=float, default=0.1)
    parser.add_argument("--pair-mode", choices=("same", "mismatch", "none"), default="same")
    parser.add_argument("--lr-muon", type=float, default=2e-3)
    parser.add_argument("--lr-adam", type=float, default=5e-4)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--eos", default="<|endoftext|>")
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--log-every", type=int, default=20)
    args = parser.parse_args()
    if args.epochs <= 0 or args.batch_size <= 0 or args.pair_batch_size <= 0:
        raise ValueError("epochs and batch sizes must be positive")
    if args.align_weight < 0 or args.contrastive_weight < 0:
        raise ValueError("alignment weights must be non-negative")
    if args.pair_mode == "none" and (args.align_weight or args.contrastive_weight):
        raise ValueError("CE-only control requires zero alignment weights")
    if args.pair_mode != "same" and args.contrastive_weight:
        raise ValueError("contrastive alignment is defined only for same-state pairs")
    if args.contrastive_weight and args.pair_batch_size < 2:
        raise ValueError("contrastive alignment requires pair batch size at least two")
    if os.path.exists(args.out):
        raise SystemExit("refusing existing output: {}".format(args.out))

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    elif args.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("--device cuda requested but CUDA is unavailable")
    elif args.device == "mps" and not torch.backends.mps.is_available():
        raise ValueError("--device mps requested but MPS is unavailable")
    else:
        device = args.device
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    paths = expand_paths(args.data)
    tokenizer = Tokenizer.from_file(args.tokenizer)
    eos_id = tokenizer.token_to_id(args.eos)
    checkpoint = torch.load(args.init, map_location="cpu")
    cfg = GPTConfig(**checkpoint["cfg"])
    model = GPT(cfg).to(device)
    model.load_state_dict(checkpoint["model"])
    print("[psa] init={} step={} params={:.1f}M device={} mode={} layer={} align={} contrastive={}".format(
        args.init, checkpoint.get("step"), model.num_params() / 1e6, device, args.pair_mode,
        args.capture_layer, args.align_weight, args.contrastive_weight), flush=True)

    X, Y, _ = build_packed(paths, tokenizer, cfg.seq_len, ["question"], ["response"], eos_id, args.max_examples)
    if not len(X):
        raise ValueError("no packed completion sequences")
    pairs = tokenize_pairs(collect_state_pairs(paths), tokenizer, cfg.seq_len)
    if args.max_examples:
        pairs = pairs[:args.max_examples]
    if args.pair_mode == "mismatch" and len(pairs) < 2:
        raise ValueError("mismatch control requires at least two pairs")
    print("[psa] packed_sequences={} paired_prompt_states={}".format(len(X), len(pairs)), flush=True)

    muon_params, adam_params = split_params(model)
    opt_muon = Muon(muon_params, lr=args.lr_muon)
    opt_adam = torch.optim.AdamW(adam_params, lr=args.lr_adam, betas=(0.9, 0.95), weight_decay=0.0)
    total_steps = args.epochs * (len(X) // args.batch_size)
    rng = np.random.default_rng(args.seed)
    capture = BoundaryCapture(model, args.capture_layer)
    step, started = 0, time.time()

    def lr_scale(index):
        if index < args.warmup:
            return index / max(1, args.warmup)
        progress = (index - args.warmup) / max(1, total_steps - args.warmup)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))

    try:
        for epoch in range(args.epochs):
            order = rng.permutation(len(X))
            pair_order = rng.permutation(len(pairs))
            pair_cursor = 0
            for begin in range(0, len(order) - args.batch_size + 1, args.batch_size):
                selected = order[begin:begin + args.batch_size]
                x = torch.from_numpy(X[selected]).to(device)
                y = torch.from_numpy(Y[selected]).to(device)
                scale = lr_scale(step)
                for group in opt_muon.param_groups:
                    group["lr"] = args.lr_muon * scale
                for group in opt_adam.param_groups:
                    group["lr"] = args.lr_adam * scale
                opt_muon.zero_grad(set_to_none=True)
                opt_adam.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
                    _, ce_loss = model(x, y)
                    align_loss = ce_loss.new_zeros(())
                    contrastive_loss = ce_loss.new_zeros(())
                    stats = {
                        "cosine": ce_loss.detach() * 0, "norm": ce_loss.detach() * 0,
                        "variance": ce_loss.detach() * 0, "positive_cosine": ce_loss.detach() * 0,
                        "negative_cosine": ce_loss.detach() * 0,
                    }
                    if args.pair_mode != "none":
                        indices, pair_cursor = select_distinct_pair_indices(
                            pairs, pair_order, pair_cursor, args.pair_batch_size,
                        )
                        batch = [pairs[index] for index in indices]
                        left_ids, left_lengths = pad_prompts(batch, "left_ids", eos_id, device)
                        right_batch = batch if args.pair_mode == "same" else [batch[index] for index in mismatch_order(batch)]
                        right_ids, right_lengths = pad_prompts(right_batch, "right_ids", eos_id, device)
                        left_hidden = capture.boundary(model, left_ids, left_lengths)
                        right_hidden = capture.boundary(model, right_ids, right_lengths)
                        align_loss, align_stats = alignment_statistics(left_hidden, right_hidden)
                        stats.update(align_stats)
                        if args.contrastive_weight:
                            contrastive_loss, contrastive_stats = contrastive_statistics(
                                left_hidden, right_hidden, args.contrastive_temperature,
                            )
                            stats.update(contrastive_stats)
                    total_loss = ce_loss + args.align_weight * align_loss + args.contrastive_weight * contrastive_loss
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                opt_muon.step()
                opt_adam.step()
                if step % args.log_every == 0:
                    print("[psa] epoch={} step={}/{} ce={:.4f} align={:.4f} contrast={:.4f} total={:.4f} cos={:.4f} pos={:.4f} neg={:.4f} norm={:.3f} var={:.5f} lr={:.5f} {}s".format(
                        epoch, step, total_steps, ce_loss.item(), align_loss.item(), contrastive_loss.item(), total_loss.item(),
                        stats["cosine"].item(), stats["positive_cosine"].item(), stats["negative_cosine"].item(),
                        stats["norm"].item(), stats["variance"].item(),
                        args.lr_muon * scale, int(time.time() - started)), flush=True)
                step += 1
            os.makedirs(args.out, exist_ok=True)
            torch.save({
                "model": model.state_dict(), "cfg": cfg.__dict__, "step": "psa_ep{}".format(epoch + 1),
                "state_alignment": {"mode": args.pair_mode, "capture_layer": args.capture_layer,
                                    "align_weight": args.align_weight, "contrastive_weight": args.contrastive_weight,
                                    "contrastive_temperature": args.contrastive_temperature, "pairs": len(pairs)},
            }, os.path.join(args.out, "psa_ep{}.pt".format(epoch + 1)))
            print("[psa] saved epoch {}".format(epoch + 1), flush=True)
    finally:
        capture.close()
    print("[psa] done {} steps in {}s".format(step, int(time.time() - started)), flush=True)


if __name__ == "__main__":
    main()
