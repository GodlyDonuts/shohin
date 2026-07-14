#!/usr/bin/env python3
"""Read-only teacher-forced diagnostic for Counterfactual Residual Algebra.

This never counts as a behavioral or reasoning score.  It measures whether a
composed native tape assigns lower answer NLL to the solver-correct normal or
counterfactual completion than to its paired foil before greedy decoding can
hide a weak latent preference.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from counterfactual_residual_algebra import (
    compose_counterfactual_tape,
    encode_residual_tape,
    algebra_suffix_logits,
)
from latent_rollout import build_answer_targets
from model import GPT, GPTConfig


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def answer_nll(model, tape, suffix_ids, answer_ids, tape_len, layer, eos_id):
    """Return mean answer-only NLL after a fixed continuous tape."""
    suffix = torch.cat((model.tok(suffix_ids), model.tok(answer_ids)), dim=1)
    logits = algebra_suffix_logits(model, tape, suffix, layer, tape_len)
    targets = build_answer_targets(answer_ids, tape_len + suffix_ids.shape[1], 0, eos_id)
    return float(F.cross_entropy(
        logits.float().reshape(-1, logits.shape[-1]), targets.reshape(-1), ignore_index=-1,
    ).item())


def encode(model, tokenizer, source, layer, tape_len):
    ids = torch.tensor([tokenizer.encode(source).ids], dtype=torch.long, device="cuda")
    return encode_residual_tape(model, ids, layer, tape_len)


def resolve_metadata(checkpoint, tokenizer, args):
    metadata = checkpoint.get("counterfactual_residual_algebra")
    if isinstance(metadata, dict):
        if metadata.get("source_present_at_suffix") is not False:
            raise SystemExit("checkpoint does not certify source-free residual algebra")
        if metadata.get("extra_trainable_parameters") != 0 or metadata.get("composition") != "donor + edited - base":
            raise SystemExit("checkpoint does not certify the native CRA mechanism")
        return metadata, int(metadata["layer"]), int(metadata["tape_len"])
    if not args.allow_raw:
        raise SystemExit("checkpoint does not certify source-free residual algebra; raw requires --allow-raw")
    anchor_ids = tokenizer.encode(args.source_anchor).ids
    tape_len = args.tape_len or len(anchor_ids)
    if not anchor_ids or tape_len != len(anchor_ids):
        raise SystemExit("raw CRA baseline requires tape length equal to exact source anchor")
    layer = int(args.layer)
    return {
        "raw_baseline": True,
        "layer": layer,
        "tape_len": tape_len,
        "source_anchor": args.source_anchor,
        "source_present_at_suffix": False,
        "extra_trainable_parameters": 0,
        "composition": "donor + edited - base",
    }, layer, tape_len


def mean(rows, field):
    return sum(row[field] for row in rows) / len(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--split", default="heldout")
    parser.add_argument("--max-examples", type=int, default=500)
    parser.add_argument("--allow-raw", action="store_true")
    parser.add_argument("--layer", type=int, default=19)
    parser.add_argument("--tape-len", type=int, default=0)
    parser.add_argument("--source-anchor", default="\nEnd state record:")
    args = parser.parse_args()
    if not torch.cuda.is_available() or Path(args.out).exists():
        raise SystemExit("CUDA required and output path must be fresh")
    checkpoint = torch.load(args.ckpt, map_location="cpu")
    model = GPT(GPTConfig(**checkpoint["cfg"])).to("cuda").eval()
    model.load_state_dict(checkpoint["model"])
    tokenizer = Tokenizer.from_file(args.tokenizer)
    eos_id = tokenizer.token_to_id("<|endoftext|>")
    if eos_id is None:
        raise SystemExit("tokenizer EOS missing")
    metadata, layer, tape_len = resolve_metadata(checkpoint, tokenizer, args)
    rows = [json.loads(line) for line in open(args.data) if line.strip()]
    rows = [row for row in rows if row.get("schema") == "counterfactual_residual_algebra_v1" and row.get("split") == args.split]
    rows = rows[:args.max_examples]
    if not rows:
        raise SystemExit("no CRA rows for requested split")

    results, tapes = [], []
    with torch.no_grad():
        for index, row in enumerate(rows):
            base = encode(model, tokenizer, row["base_source"], layer, tape_len)
            edited = encode(model, tokenizer, row["edited_source"], layer, tape_len)
            counter_edited = encode(model, tokenizer, row["counterfactual_edited_source"], layer, tape_len)
            donor = encode(model, tokenizer, row["donor_source"], layer, tape_len)
            normal_tape = compose_counterfactual_tape(base, edited, donor)
            counter_tape = compose_counterfactual_tape(base, counter_edited, donor)
            pbase = encode(model, tokenizer, row["paraphrase_base_source"], layer, tape_len)
            pedited = encode(model, tokenizer, row["paraphrase_edited_source"], layer, tape_len)
            pdonor = encode(model, tokenizer, row["paraphrase_donor_source"], layer, tape_len)
            paraphrase_tape = compose_counterfactual_tape(pbase, pedited, pdonor)
            suffix = torch.tensor([tokenizer.encode(row["suffix_prompt"]).ids], dtype=torch.long, device="cuda")
            normal_answer = torch.tensor([tokenizer.encode(" " + row["response"]).ids], dtype=torch.long, device="cuda")
            counter_answer = torch.tensor([tokenizer.encode(" " + row["counterfactual_response"]).ids], dtype=torch.long, device="cuda")
            result = {
                "episode_id": row["episode_id"],
                "normal_target_nll": answer_nll(model, normal_tape, suffix, normal_answer, tape_len, layer, eos_id),
                "normal_counterfactual_nll": answer_nll(model, normal_tape, suffix, counter_answer, tape_len, layer, eos_id),
                "counterfactual_normal_nll": answer_nll(model, counter_tape, suffix, normal_answer, tape_len, layer, eos_id),
                "counterfactual_target_nll": answer_nll(model, counter_tape, suffix, counter_answer, tape_len, layer, eos_id),
                "paraphrase_target_nll": answer_nll(model, paraphrase_tape, suffix, normal_answer, tape_len, layer, eos_id),
            }
            result["normal_margin"] = result["normal_counterfactual_nll"] - result["normal_target_nll"]
            result["counterfactual_margin"] = result["counterfactual_normal_nll"] - result["counterfactual_target_nll"]
            result["paraphrase_margin"] = result["normal_counterfactual_nll"] - result["paraphrase_target_nll"]
            result["paired_directional"] = bool(
                result["normal_margin"] > 0 and result["counterfactual_margin"] > 0 and result["paraphrase_margin"] > 0
            )
            tapes.append(normal_tape)
            results.append(result)
            print("[cra-nll] {}/{}".format(index + 1, len(rows)), flush=True)
        for index, result in enumerate(results):
            row = rows[index]
            suffix = torch.tensor([tokenizer.encode(row["suffix_prompt"]).ids], dtype=torch.long, device="cuda")
            normal_answer = torch.tensor([tokenizer.encode(" " + row["response"]).ids], dtype=torch.long, device="cuda")
            result["zero_target_nll"] = answer_nll(model, torch.zeros_like(tapes[index]), suffix, normal_answer, tape_len, layer, eos_id)
            result["shuffled_target_nll"] = answer_nll(
                model, tapes[(index + 1) % len(tapes)], suffix, normal_answer, tape_len, layer, eos_id,
            )
            result["zero_margin"] = result["zero_target_nll"] - result["normal_target_nll"]
            result["shuffle_margin"] = result["shuffled_target_nll"] - result["normal_target_nll"]
            result["strict_directional"] = bool(result["paired_directional"] and result["zero_margin"] > 0 and result["shuffle_margin"] > 0)

    summary_fields = (
        "normal_target_nll", "normal_counterfactual_nll", "counterfactual_normal_nll", "counterfactual_target_nll",
        "paraphrase_target_nll", "normal_margin", "counterfactual_margin", "paraphrase_margin", "zero_margin", "shuffle_margin",
    )
    report = {
        "audit": "counterfactual_residual_algebra_teacher_forced_v1",
        "claim_boundary": "Teacher-forced likelihood is diagnostic only and cannot establish behavioral reasoning.",
        "checkpoint": args.ckpt,
        "checkpoint_metadata": metadata,
        "data": args.data,
        "data_sha256": sha256_file(args.data),
        "split": args.split,
        "rows": len(results),
        "summary": dict((field, mean(results, field)) for field in summary_fields),
        "paired_directional": sum(bool(row["paired_directional"]) for row in results),
        "strict_directional": sum(bool(row["strict_directional"]) for row in results),
        "results": results,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print("[cra-nll] summary=" + json.dumps(report["summary"], sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
