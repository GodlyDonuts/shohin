#!/usr/bin/env python3
"""Read-only R7 operation canary using nonlinear future-state interventions."""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path

import torch
from tokenizers import Tokenizer

from categorical_microcode import opcode_for, sha256_file
from counterfactual_influence_quotient import (
    informative_channel_order,
    normalized_signature,
    operation_intervention_bundle,
    random_channel_order,
    rank_direct_candidates,
    rank_signature_candidates,
)
from model import GPT, GPTConfig


POLICIES = ("active", "random", "direct", "shuffled")


def encode_future_states(model, tokenizer, texts, layers):
    encodings = [tokenizer.encode(text) for text in texts]
    lengths = torch.tensor([len(encoding.ids) for encoding in encodings], device="cuda")
    if int(lengths.min()) <= 0 or int(lengths.max()) > model.cfg.seq_len:
        raise ValueError("intervention text exceeds model sequence contract")
    width = int(lengths.max().item())
    ids = torch.zeros((len(encodings), width), dtype=torch.long, device="cuda")
    for row, encoding in enumerate(encodings):
        ids[row, :len(encoding.ids)] = torch.tensor(encoding.ids, dtype=torch.long, device="cuda")
    selected = []
    x = model.tok(ids)
    cos = model.cos[:width].to(x.device)
    sin = model.sin[:width].to(x.device)
    batch = torch.arange(ids.shape[0], device=ids.device)
    positions = lengths - 1
    for index, block in enumerate(model.blocks):
        x, _ = block(x, cos, sin)
        if index in layers:
            selected.append(x[batch, positions].float())
    if len(selected) != len(layers):
        raise ValueError("not all registered layers were observed")
    return torch.stack(selected, dim=1), tuple(len(encoding.ids) for encoding in encodings)


def compatible_channels(tokenizer, bundles):
    common = set(bundles[0]["interventions"])
    for bundle in bundles[1:]:
        common.intersection_update(bundle["interventions"])
    compatible = []
    for channel in sorted(common):
        valid = True
        for bundle in bundles:
            baseline = len(tokenizer.encode(bundle["baseline"]).ids)
            intervention = len(tokenizer.encode(bundle["interventions"][channel]).ids)
            if baseline != intervention:
                valid = False
                break
        if valid:
            compatible.append(channel)
    return tuple(compatible)


def select_items(rows, limit_per_opcode, regimes):
    counts = Counter()
    selected = []
    for row_index, row in sorted(enumerate(rows), key=lambda item: str(item[1].get("reference", ""))):
        if row.get("eval_regime") not in regimes:
            continue
        for operation_index, operation in enumerate(row["operations"]):
            target = opcode_for(operation, row["keys"])
            if counts[target] >= limit_per_opcode:
                continue
            selected.append((row_index, operation_index, target))
            counts[target] += 1
    if set(counts.values()) != {int(limit_per_opcode)} or len(counts) != 9:
        raise ValueError("canary does not contain the exact balanced nine-opcode contract")
    return selected, dict(sorted(counts.items()))


def summarize(records):
    output = {}
    for policy in POLICIES:
        by_opcode = Counter()
        for record in records:
            if record["predictions"][policy] == record["target_opcode"]:
                by_opcode[record["target_opcode"]] += 1
        output[policy] = {
            "correct": sum(by_opcode.values()),
            "total": len(records),
            "by_opcode_correct": dict(sorted(by_opcode.items())),
        }
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--admission", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--layers", default="5,11,17,23,29")
    parser.add_argument("--channels", type=int, default=2)
    parser.add_argument("--limit-per-opcode", type=int, default=12)
    parser.add_argument("--regimes", default="language_ood,full_ood")
    parser.add_argument("--seed", type=int, default=20260714)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("R7 influence evaluation requires CUDA")
    if Path(args.out).exists():
        raise SystemExit("refusing existing output")
    layers = tuple(map(int, args.layers.split(",")))
    if layers != (5, 11, 17, 23, 29):
        raise SystemExit("R7 canary layers changed")
    if args.channels != 2 or args.limit_per_opcode != 12:
        raise SystemExit("R7 canary budget changed")
    regimes = tuple(args.regimes.split(","))
    if regimes != ("language_ood", "full_ood"):
        raise SystemExit("R7 canary regimes changed")

    data_sha256 = sha256_file(args.data)
    admission = json.load(open(args.admission))
    if not admission.get("all_checks_pass") or admission.get("eval_sha256") != data_sha256:
        raise SystemExit("R7 admission does not bind evaluation data")
    rows = [json.loads(line) for line in open(args.data) if line.strip()]
    items, balance = select_items(rows, args.limit_per_opcode, set(regimes))
    tokenizer = Tokenizer.from_file(args.tokenizer)

    checkpoint = torch.load(args.base, map_location="cpu")
    model = GPT(GPTConfig(**checkpoint["cfg"])).to("cuda").eval()
    model.load_state_dict(checkpoint["model"])
    if max(layers) >= len(model.blocks):
        raise SystemExit("R7 registered layer is absent")

    records = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for item_number, (row_index, operation_index, target) in enumerate(items, 1):
            row = rows[row_index]
            original = operation_intervention_bundle(row, operation_index)
            candidate_names = tuple(original["candidate_lines"])
            candidates = [
                operation_intervention_bundle(row, operation_index, name)
                for name in candidate_names
            ]
            bundles = [original] + candidates
            channel_names = compatible_channels(tokenizer, bundles)
            if len(channel_names) < args.channels:
                raise SystemExit("fewer than two token-length-matched interventions")
            texts = []
            for bundle in bundles:
                texts.append(bundle["baseline"])
                texts.extend(bundle["interventions"][name] for name in channel_names)
            states, token_lengths = encode_future_states(model, tokenizer, texts, layers)
            stride = 1 + len(channel_names)
            states = states.reshape(len(bundles), stride, len(layers), -1)
            signatures = torch.stack([
                normalized_signature(bundle_states[0], bundle_states[1:])
                for bundle_states in states
            ])
            original_signature = signatures[0]
            candidate_signatures = signatures[1:]
            active_order = informative_channel_order(candidate_signatures)
            random_order = random_channel_order(len(channel_names), args.seed + row_index * 97 + operation_index)
            active_rank, active_score = rank_signature_candidates(
                original_signature, candidate_signatures, active_order[:args.channels],
            )
            random_rank, random_score = rank_signature_candidates(
                original_signature, candidate_signatures, random_order[:args.channels],
            )
            direct_rank, direct_score = rank_direct_candidates(states[0, 0], states[1:, 0])
            shuffled_order = tuple(reversed(active_order))
            shuffled_original = original_signature.index_select(
                0, torch.tensor(shuffled_order, device=original_signature.device),
            )
            shuffled_rank, shuffled_score = rank_signature_candidates(
                shuffled_original, candidate_signatures, active_order[:args.channels],
            )
            predictions = {
                "active": candidate_names[int(active_rank[0])],
                "random": candidate_names[int(random_rank[0])],
                "direct": candidate_names[int(direct_rank[0])],
                "shuffled": candidate_names[int(shuffled_rank[0])],
            }
            records.append({
                "row_index": row_index,
                "operation_index": operation_index,
                "reference": row.get("reference", ""),
                "regime": row.get("eval_regime"),
                "target_opcode": target,
                "candidate_opcodes": candidate_names,
                "value": original["value"],
                "channel_names": channel_names,
                "active_channel_order": tuple(channel_names[index] for index in active_order),
                "random_channel_order": tuple(channel_names[index] for index in random_order),
                "predictions": predictions,
                "scores": {
                    "active": [float(value) for value in active_score.tolist()],
                    "random": [float(value) for value in random_score.tolist()],
                    "direct": [float(value) for value in direct_score.tolist()],
                    "shuffled": [float(value) for value in shuffled_score.tolist()],
                },
                "token_lengths": token_lengths,
            })
            if item_number % 12 == 0:
                print("[isq-r7] items={}/{}".format(item_number, len(items)), flush=True)

    summary = summarize(records)
    if any(not math.isfinite(score) for record in records for scores in record["scores"].values() for score in scores):
        raise SystemExit("R7 produced a non-finite score")
    report = {
        "protocol": "interventional_semantic_quotient_canary_r7",
        "base": str(Path(args.base).resolve()),
        "base_sha256": sha256_file(args.base),
        "data": str(Path(args.data).resolve()),
        "data_sha256": data_sha256,
        "admission_sha256": sha256_file(args.admission),
        "layers": layers,
        "intervention_budget": args.channels,
        "limit_per_opcode": args.limit_per_opcode,
        "regimes": regimes,
        "selection_balance": balance,
        "summary": summary,
        "records": records,
        "inference_contract": (
            "Predictions consume problem text, the two visible identifier strings, and visible numeric "
            "literals. Structured operation/query labels are used only after prediction for scoring."
        ),
        "claim_boundary": (
            "This used-board canary tests whether nonlinear future-state intervention signatures contain "
            "operator semantics. It cannot establish reasoning, transfer, or context compression."
        ),
    }
    output = Path(args.out)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output)
    print(json.dumps(summary, sort_keys=True), flush=True)
    print("[isq-r7] wrote {}".format(output), flush=True)


if __name__ == "__main__":
    main()
