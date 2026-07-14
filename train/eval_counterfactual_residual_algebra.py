#!/usr/bin/env python3
"""Read-only causal evaluation for Counterfactual Residual Algebra checkpoints."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from tokenizers import Tokenizer

from counterfactual_residual_algebra import (
    compose_counterfactual_tape,
    compose_two_edit_counterfactual_tape,
    encode_residual_tape,
    generate_from_algebra_tape,
)
from model import GPT, GPTConfig


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def encode_tape(model, tokenizer, source, layer, tape_len):
    ids = torch.tensor([tokenizer.encode(source).ids], dtype=torch.long, device="cuda")
    return encode_residual_tape(model, ids, layer, tape_len)


def compose_decode(model, tokenizer, base_source, edited_source, donor_source, suffix, layer, tape_len, eos_id, max_new):
    base = encode_tape(model, tokenizer, base_source, layer, tape_len)
    edited = encode_tape(model, tokenizer, edited_source, layer, tape_len)
    donor = encode_tape(model, tokenizer, donor_source, layer, tape_len)
    tape = compose_counterfactual_tape(base, edited, donor)
    suffix_ids = torch.tensor([tokenizer.encode(suffix).ids], dtype=torch.long, device="cuda")
    tokens = generate_from_algebra_tape(model, tape, suffix_ids, layer, tape_len, eos_id, max_new)
    return tokenizer.decode(tokens.tolist(), skip_special_tokens=False).strip(), tape


def compose_two_edit_decode(model, tokenizer, base_source, primary_edited_source, secondary_edited_source, donor_source, suffix, layer, tape_len, eos_id, max_new):
    base = encode_tape(model, tokenizer, base_source, layer, tape_len)
    primary = encode_tape(model, tokenizer, primary_edited_source, layer, tape_len)
    secondary = encode_tape(model, tokenizer, secondary_edited_source, layer, tape_len)
    donor = encode_tape(model, tokenizer, donor_source, layer, tape_len)
    tape = compose_two_edit_counterfactual_tape(base, primary, secondary, donor)
    suffix_ids = torch.tensor([tokenizer.encode(suffix).ids], dtype=torch.long, device="cuda")
    tokens = generate_from_algebra_tape(model, tape, suffix_ids, layer, tape_len, eos_id, max_new)
    return tokenizer.decode(tokens.tolist(), skip_special_tokens=False).strip(), tape


def decode_tape(model, tokenizer, tape, suffix, layer, tape_len, eos_id, max_new):
    suffix_ids = torch.tensor([tokenizer.encode(suffix).ids], dtype=torch.long, device="cuda")
    tokens = generate_from_algebra_tape(model, tape, suffix_ids, layer, tape_len, eos_id, max_new)
    return tokenizer.decode(tokens.tolist(), skip_special_tokens=False).strip()


def score_result(result):
    if result["expected"] == result["counterfactual_expected"]:
        raise ValueError("counterfactual target must differ from normal target")
    result["normal_correct"] = result["normal"] == result["expected"]
    result["paraphrase_correct"] = result["paraphrase"] == result["expected"]
    result["counterfactual_correct"] = result["counterfactual"] == result["counterfactual_expected"]
    result["zero_recreates_normal"] = result["zero"] == result["expected"]
    result["shuffle_recreates_normal"] = result["shuffled"] == result["expected"]
    result["strict_causal"] = bool(
        result["normal_correct"] and result["paraphrase_correct"] and result["counterfactual_correct"]
        and not result["zero_recreates_normal"] and not result["shuffle_recreates_normal"]
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--split", default="heldout")
    parser.add_argument("--max-examples", type=int, default=500)
    parser.add_argument("--max-new", type=int, default=12)
    parser.add_argument("--allow-raw", action="store_true", help="evaluate immutable raw baseline with explicit tape settings")
    parser.add_argument("--layer", type=int, default=19, help="raw-baseline export layer")
    parser.add_argument("--tape-len", type=int, default=0, help="raw-baseline tape length; zero uses source anchor")
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
    metadata = checkpoint.get("counterfactual_residual_algebra")
    if isinstance(metadata, dict):
        if metadata.get("source_present_at_suffix") is not False:
            raise SystemExit("checkpoint does not certify source-free residual algebra")
        if metadata.get("extra_trainable_parameters") != 0 or metadata.get("composition") != "donor + edited - base":
            raise SystemExit("checkpoint does not certify the native CRA mechanism")
        layer, tape_len = int(metadata["layer"]), int(metadata["tape_len"])
    elif args.allow_raw:
        anchor_ids = tokenizer.encode(args.source_anchor).ids
        tape_len = args.tape_len or len(anchor_ids)
        if not anchor_ids or tape_len != len(anchor_ids):
            raise SystemExit("raw CRA baseline requires tape length equal to exact source anchor")
        layer = int(args.layer)
        metadata = {
            "raw_baseline": True,
            "layer": layer,
            "tape_len": tape_len,
            "source_anchor": args.source_anchor,
            "source_present_at_suffix": False,
            "extra_trainable_parameters": 0,
            "composition": "donor + edited - base",
        }
    else:
        raise SystemExit("checkpoint does not certify source-free residual algebra; pass --allow-raw only for raw baseline")
    rows = [json.loads(line) for line in open(args.data) if line.strip()]
    rows = [row for row in rows if row.get("schema") == "counterfactual_residual_algebra_v1" and row.get("split") == args.split][:args.max_examples]
    if not rows:
        raise SystemExit("no CRA rows for requested split")
    results, tapes = [], []
    with torch.no_grad():
        for index, row in enumerate(rows):
            if row.get("mode") == "two_edit":
                normal, tape = compose_two_edit_decode(
                    model, tokenizer, row["base_source"], row["primary_edited_source"], row["secondary_edited_source"], row["donor_source"],
                    row["suffix_prompt"], layer, tape_len, eos_id, args.max_new,
                )
                paraphrase, paraphrase_tape = compose_two_edit_decode(
                    model, tokenizer, row["paraphrase_base_source"], row["paraphrase_primary_edited_source"], row["paraphrase_secondary_edited_source"], row["paraphrase_donor_source"],
                    row["suffix_prompt"], layer, tape_len, eos_id, args.max_new,
                )
                counterfactual, _ = compose_two_edit_decode(
                    model, tokenizer, row["base_source"], row["counterfactual_primary_edited_source"], row["secondary_edited_source"], row["donor_source"],
                    row["suffix_prompt"], layer, tape_len, eos_id, args.max_new,
                )
            else:
                normal, tape = compose_decode(
                    model, tokenizer, row["base_source"], row["edited_source"], row["donor_source"],
                    row["suffix_prompt"], layer, tape_len, eos_id, args.max_new,
                )
                paraphrase, paraphrase_tape = compose_decode(
                    model, tokenizer, row["paraphrase_base_source"], row["paraphrase_edited_source"], row["paraphrase_donor_source"],
                    row["suffix_prompt"], layer, tape_len, eos_id, args.max_new,
                )
                counterfactual, _ = compose_decode(
                    model, tokenizer, row["base_source"], row["counterfactual_edited_source"], row["donor_source"],
                    row["suffix_prompt"], layer, tape_len, eos_id, args.max_new,
                )
            zero = decode_tape(model, tokenizer, torch.zeros_like(tape), row["suffix_prompt"], layer, tape_len, eos_id, args.max_new)
            tapes.append(tape)
            results.append({
                "episode_id": row["episode_id"], "normal": normal, "paraphrase": paraphrase,
                "counterfactual": counterfactual, "zero": zero, "expected": row["response"],
                "counterfactual_expected": row["counterfactual_response"],
                "same_tape_cosine": float(torch.nn.functional.cosine_similarity(
                    tape.float().reshape(1, -1), paraphrase_tape.float().reshape(1, -1), dim=-1,
                ).item()),
            })
            print("[cra-eval] {}/{}".format(index + 1, len(rows)), flush=True)
        for index, result in enumerate(results):
            result["shuffled"] = decode_tape(
                model, tokenizer, tapes[(index + 1) % len(tapes)], rows[index]["suffix_prompt"], layer, tape_len, eos_id, args.max_new,
            )
    for result in results:
        score_result(result)
    fields = ("normal_correct", "paraphrase_correct", "counterfactual_correct", "zero_recreates_normal", "shuffle_recreates_normal", "strict_causal")
    summary = {field: sum(bool(row[field]) for row in results) for field in fields}
    report = {
        "audit": "counterfactual_residual_algebra_v1", "checkpoint": args.ckpt, "step": checkpoint.get("step"),
        "checkpoint_metadata": metadata, "data": args.data, "data_sha256": sha256_file(args.data), "split": args.split,
        "rows": len(results), "summary": summary,
        "mean_same_tape_cosine": sum(row["same_tape_cosine"] for row in results) / len(results),
        "results": results,
        "claim_boundary": "A strict pass establishes only source-free residual algebra on this solver-verified suite.",
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print("[cra-eval] summary=" + json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
