#!/usr/bin/env python3
"""Read-only R8 canary for mixed counterfactual curvature binding."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import random
from collections import Counter
from pathlib import Path

import torch
from tokenizers import Tokenizer

from categorical_microcode import sha256_file
from counterfactual_influence_quotient import (
    event_candidates,
    normalized_curvature,
    operation_intervention_text,
    rank_direct_candidates,
    rank_signature_candidates,
)
from eval_counterfactual_influence_quotient import encode_future_states, select_items
from model import GPT, GPTConfig


POLICIES = ("curvature", "random_pairs", "direct", "shuffled_curvature")
R7_SHA256 = "2531c6f5b0166feab75a02ac4061fb96e1f773e072c4a690a72436d8a106cfbd"


def common_channels(tokenizer, row, operation_index, candidate_names):
    channels = ("initial_0", "initial_1", "event_value", "event_roles", "query_roles")
    compatible = []
    for channel in channels:
        lengths = []
        valid = True
        for candidate in (None,) + tuple(candidate_names):
            try:
                baseline = operation_intervention_text(row, operation_index, candidate)
                variant = operation_intervention_text(row, operation_index, candidate, (channel,))
            except ValueError:
                valid = False
                break
            lengths.append((len(tokenizer.encode(baseline).ids), len(tokenizer.encode(variant).ids)))
        if valid and all(left == right for left, right in lengths):
            compatible.append(channel)
    return tuple(compatible)


def summarize(records):
    output = {}
    for policy in POLICIES:
        by_opcode = Counter()
        by_group = Counter()
        for record in records:
            if record["predictions"][policy] == record["target_opcode"]:
                by_opcode[record["target_opcode"]] += 1
                by_group[record["operator_group"]] += 1
        output[policy] = {
            "correct": sum(by_opcode.values()),
            "total": len(records),
            "by_opcode_correct": dict(sorted(by_opcode.items())),
            "by_group_correct": dict(sorted(by_group.items())),
        }
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--admission", required=True)
    parser.add_argument("--r7-report", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--layers", default="5,11,17,23,29")
    parser.add_argument("--pairs", type=int, default=2)
    parser.add_argument("--limit-per-opcode", type=int, default=12)
    parser.add_argument("--regimes", default="language_ood,full_ood")
    parser.add_argument("--seed", type=int, default=20260714)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("R8 curvature evaluation requires CUDA")
    if Path(args.out).exists():
        raise SystemExit("refusing existing output")
    layers = tuple(map(int, args.layers.split(",")))
    if layers != (5, 11, 17, 23, 29) or args.pairs != 2 or args.limit_per_opcode != 12:
        raise SystemExit("R8 registered mechanics changed")
    regimes = tuple(args.regimes.split(","))
    if regimes != ("language_ood", "full_ood"):
        raise SystemExit("R8 canary regimes changed")
    if sha256_file(args.r7_report) != R7_SHA256:
        raise SystemExit("R8 is not bound to the frozen R7 diagnosis")
    r7 = json.load(open(args.r7_report))
    if r7.get("protocol") != "interventional_semantic_quotient_canary_r7":
        raise SystemExit("invalid R7 reference")

    data_sha256 = sha256_file(args.data)
    admission = json.load(open(args.admission))
    if not admission.get("all_checks_pass") or admission.get("eval_sha256") != data_sha256:
        raise SystemExit("R8 admission does not bind evaluation data")
    rows = [json.loads(line) for line in open(args.data) if line.strip()]
    items, balance = select_items(rows, args.limit_per_opcode, set(regimes))
    tokenizer = Tokenizer.from_file(args.tokenizer)
    checkpoint = torch.load(args.base, map_location="cpu")
    model = GPT(GPTConfig(**checkpoint["cfg"])).to("cuda").eval()
    model.load_state_dict(checkpoint["model"])
    if max(layers) >= len(model.blocks):
        raise SystemExit("R8 registered layer is absent")

    records = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for item_number, (row_index, operation_index, target) in enumerate(items, 1):
            row = rows[row_index]
            event_line = [
                line for line in row["question"].splitlines()
                if line.lstrip().startswith(("Step ", "Event ")) and ":" in line
            ][operation_index]
            value, candidates = event_candidates(event_line, row["keys"])
            candidate_names = tuple(candidates)
            channels = common_channels(tokenizer, row, operation_index, candidate_names)
            semantic_channel = "event_value" if value else "event_roles"
            binding_pairs = ((semantic_channel, "initial_0"), (semantic_channel, "initial_1"))
            if not all(left in channels and right in channels for left, right in binding_pairs):
                raise SystemExit("R8 binding pair is not token compatible")
            all_pairs = tuple(itertools.combinations(channels, 2))
            binding_indices = tuple(all_pairs.index(tuple(sorted(pair, key=channels.index))) for pair in binding_pairs)
            random_indices = list(range(len(all_pairs)))
            random.Random(args.seed + row_index * 97 + operation_index).shuffle(random_indices)
            random_indices = tuple(random_indices[:args.pairs])

            hypotheses = (None,) + candidate_names
            texts = []
            for candidate in hypotheses:
                texts.append(operation_intervention_text(row, operation_index, candidate))
                texts.extend(
                    operation_intervention_text(row, operation_index, candidate, (channel,))
                    for channel in channels
                )
                texts.extend(
                    operation_intervention_text(row, operation_index, candidate, pair)
                    for pair in all_pairs
                )
            states, token_lengths = encode_future_states(model, tokenizer, texts, layers)
            stride = 1 + len(channels) + len(all_pairs)
            states = states.reshape(len(hypotheses), stride, len(layers), -1)
            curvatures = []
            curvature_norms = []
            for hypothesis_states in states:
                first = torch.stack([
                    hypothesis_states[1 + channels.index(left)] for left, _ in all_pairs
                ])
                second = torch.stack([
                    hypothesis_states[1 + channels.index(right)] for _, right in all_pairs
                ])
                joint = hypothesis_states[1 + len(channels):]
                curvature, norms = normalized_curvature(hypothesis_states[0], first, second, joint)
                curvatures.append(curvature)
                curvature_norms.append(norms)
            curvatures = torch.stack(curvatures)
            curvature_norms = torch.stack(curvature_norms)
            original_curvature, candidate_curvatures = curvatures[0], curvatures[1:]
            curvature_rank, curvature_score = rank_signature_candidates(
                original_curvature, candidate_curvatures, binding_indices,
            )
            random_rank, random_score = rank_signature_candidates(
                original_curvature, candidate_curvatures, random_indices,
            )
            direct_rank, direct_score = rank_direct_candidates(states[0, 0], states[1:, 0])
            shuffled_original = original_curvature.clone()
            shuffled_original[list(binding_indices)] = original_curvature[list(reversed(binding_indices))]
            shuffled_rank, shuffled_score = rank_signature_candidates(
                shuffled_original, candidate_curvatures, binding_indices,
            )
            predictions = {
                "curvature": candidate_names[int(curvature_rank[0])],
                "random_pairs": candidate_names[int(random_rank[0])],
                "direct": candidate_names[int(direct_rank[0])],
                "shuffled_curvature": candidate_names[int(shuffled_rank[0])],
            }
            records.append({
                "row_index": row_index,
                "operation_index": operation_index,
                "reference": row.get("reference", ""),
                "regime": row.get("eval_regime"),
                "target_opcode": target,
                "operator_group": "numeric" if value else "structural",
                "candidate_opcodes": candidate_names,
                "channels": channels,
                "binding_pairs": binding_pairs,
                "random_pairs": tuple(all_pairs[index] for index in random_indices),
                "predictions": predictions,
                "scores": {
                    "curvature": [float(number) for number in curvature_score.tolist()],
                    "random_pairs": [float(number) for number in random_score.tolist()],
                    "direct": [float(number) for number in direct_score.tolist()],
                    "shuffled_curvature": [float(number) for number in shuffled_score.tolist()],
                },
                "curvature_norm_mean": float(curvature_norms[0].mean().item()),
                "token_lengths": token_lengths,
            })
            if item_number % 12 == 0:
                print("[ccb-r8] items={}/{}".format(item_number, len(items)), flush=True)

    summary = summarize(records)
    if any(not math.isfinite(score) for record in records for scores in record["scores"].values() for score in scores):
        raise SystemExit("R8 produced a non-finite score")
    report = {
        "protocol": "counterfactual_curvature_binding_canary_r8",
        "base": str(Path(args.base).resolve()),
        "base_sha256": sha256_file(args.base),
        "data": str(Path(args.data).resolve()),
        "data_sha256": data_sha256,
        "admission_sha256": sha256_file(args.admission),
        "r7_report_sha256": R7_SHA256,
        "r7_reference": {policy: r7["summary"][policy]["correct"] for policy in ("active", "direct")},
        "layers": layers,
        "pair_budget": args.pairs,
        "limit_per_opcode": args.limit_per_opcode,
        "regimes": regimes,
        "selection_balance": balance,
        "summary": summary,
        "records": records,
        "claim_boundary": (
            "This used-board canary tests a mixed second-order operator-semantic observable. "
            "It cannot establish execution, reasoning, source deletion, or context scaling."
        ),
    }
    output = Path(args.out)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output)
    print(json.dumps(summary, sort_keys=True), flush=True)
    print("[ccb-r8] wrote {}".format(output), flush=True)


if __name__ == "__main__":
    main()
