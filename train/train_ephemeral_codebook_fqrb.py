#!/usr/bin/env python3
"""Train isolated late-bound FQRB interrogation from an immutable checkpoint.

The only writer is a fresh ``train/ephemeral_codebook_fqrb_*`` directory.  A
suffix can see the current arbitrary codebook and the query, but never a
source world.  Source worlds influence it only through the native
``donor + edited - base`` residual tape.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time

import torch
from tokenizers import Tokenizer

from counterfactual_residual_algebra import supervised_algebra_loss
from model import GPT, GPTConfig
from muon import Muon, split_params
from train_counterfactual_residual_algebra import bucketed_batches, load_examples, make_batch


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def lr_scale(step: int, total_steps: int, warmup: int) -> float:
    if step < warmup:
        return step / max(1, warmup)
    progress = (step - warmup) / max(1, total_steps - warmup)
    return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--audit", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--layer", type=int, default=19)
    parser.add_argument("--tape-len", type=int, default=0)
    parser.add_argument("--source-window", type=int, default=-1,
                        help="must match the FQRB parent's positional window; -1 inherits it")
    parser.add_argument("--source-anchor", default="\nEnd state record:")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr-muon", type=float, default=2e-3)
    parser.add_argument("--lr-adam", type=float, default=5e-4)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=2026071418)
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--eos", default="<|endoftext|>")
    parser.add_argument("--log-every", type=int, default=20)
    args = parser.parse_args()
    if args.epochs <= 0 or args.batch_size <= 0 or args.layer < 0 or args.max_batches < 0:
        raise SystemExit("epochs, batch size, layer, and max batches must be valid")
    if not torch.cuda.is_available() or os.path.exists(args.out):
        raise SystemExit("CUDA required and output must be fresh")
    audit = json.load(open(args.audit))
    data_sha = sha256_file(args.data)
    required_zero = (
        "duplicate_train_prompts", "duplicate_heldout_prompts", "train_heldout_exact_prompt_hits",
        "train_heldout_exact_source_bundle_hits", "train_heldout_codebook_hits",
        "train_heldout_semantic_13gram_hits", "bad_train_group_cardinality", "bad_heldout_group_cardinality",
    )
    if (
        audit.get("audit") != "ephemeral_codebook_fqrb_v1"
        or audit.get("mechanism") != "ephemeral_codebook_fqrb_v1"
        or audit.get("fqrb_parent_decision") != "bounded_fqrb_basis_candidate_magnitude_and_interaction_still_required"
        or audit.get("train_sha256") != data_sha
        or any(audit.get(key) for key in required_zero)
    ):
        raise SystemExit("ephemeral-codebook audit does not admit requested data")
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    tokenizer = Tokenizer.from_file(args.tokenizer)
    eos_id = tokenizer.token_to_id(args.eos)
    anchor_ids = tokenizer.encode(args.source_anchor).ids
    tape_len = args.tape_len or len(anchor_ids)
    if eos_id is None or not anchor_ids or tape_len != len(anchor_ids):
        raise SystemExit("EOS missing or tape length does not exactly match source anchor")
    checkpoint = torch.load(args.init, map_location="cpu")
    cfg = GPTConfig(**checkpoint["cfg"])
    if not args.layer < cfg.n_layer - 1 or cfg.n_loop != 1:
        raise SystemExit("invalid layer or unsupported recurrent model")
    parent_metadata = checkpoint.get("counterfactual_residual_algebra", {})
    parent_window = int(parent_metadata.get("source_window", 0))
    source_window = parent_window if args.source_window < 0 else args.source_window
    if source_window != parent_window:
        raise SystemExit("ECLI source window must match its FQRB parent")
    examples, skipped = load_examples(args.data, tokenizer, cfg.seq_len, anchor_ids)
    source_lengths = [max(len(example[field]) for field in ("base", "edited", "counter_edited", "donor")) for example in examples]
    if source_window and (source_window < max(source_lengths) or source_window < tape_len):
        raise SystemExit("source window must fit every source and the native anchor tape")
    code_lengths = {
        len(tokenizer.encode(" " + json.loads(line)["response"]).ids)
        for line in open(args.data) if line.strip()
    }
    if len(code_lengths) != 1:
        raise SystemExit("ephemeral code words must have a uniform tokenizer length")
    if args.max_examples:
        examples = examples[:args.max_examples]
    batches, batch_report = bucketed_batches(examples, args.batch_size, args.seed)
    if args.max_batches:
        batches = batches[:args.max_batches]
    total_steps = args.epochs * len(batches)
    print(json.dumps({"mechanism": "ephemeral_codebook_fqrb_v1", "examples": len(examples), "skipped": skipped,
                      "batch_report": batch_report, "steps": total_steps, "layer": args.layer, "tape_len": tape_len,
                      "source_window": source_window, "max_source_tokens": max(source_lengths),
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
            base, edited, donor, suffix, answer = make_batch(examples, indices, "cuda")
            scale = lr_scale(step, total_steps, args.warmup)
            for group in opt_muon.param_groups:
                group["lr"] = args.lr_muon * scale
            for group in opt_adam.param_groups:
                group["lr"] = args.lr_adam * scale
            opt_muon.zero_grad(set_to_none=True)
            opt_adam.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, loss, tape, _ = supervised_algebra_loss(
                    model, base, edited, donor, suffix, answer, args.layer, tape_len, eos_id, source_window,
                )
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite loss at {}".format(step))
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            if not torch.isfinite(gnorm):
                raise RuntimeError("non-finite gradient at {}".format(step))
            opt_muon.step()
            opt_adam.step()
            if step % args.log_every == 0:
                print("[ecfqrb] epoch={} step={}/{} loss={:.4f} gnorm={:.3f} tape_norm={:.3f} lr={:.6f} {}s".format(
                    epoch, step, total_steps, loss.item(), float(gnorm), tape.float().norm(dim=-1).mean().item(),
                    args.lr_muon * scale, int(time.time() - started)), flush=True)
            step += 1
    os.makedirs(args.out)
    output = os.path.join(args.out, "ecfqrb_ep1.pt")
    metadata = {
        "layer": args.layer, "tape_len": tape_len, "source_anchor": args.source_anchor, "source_window": source_window,
        "data_sha256": data_sha, "source_present_at_suffix": False, "extra_trainable_parameters": 0,
        "composition": "donor + edited - base", "paraphrase_bundles": True,
        "train_examples": len(examples), "max_batches_per_epoch": args.max_batches,
        "init_sha256": sha256_file(args.init),
    }
    torch.save({
        "model": model.state_dict(), "cfg": cfg.__dict__, "step": "ephemeral_codebook_fqrb_ep1",
        "counterfactual_residual_algebra": metadata,
        "ephemeral_codebook_fqrb": {**metadata, "mechanism": "ephemeral_codebook_fqrb_v1"},
    }, output)
    print("[ecfqrb] wrote {} after {} steps in {}s".format(output, step, int(time.time() - started)), flush=True)


if __name__ == "__main__":
    main()
