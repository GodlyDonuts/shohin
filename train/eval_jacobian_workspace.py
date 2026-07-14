#!/usr/bin/env python3
"""Compare Shohin future-Jacobian and immediate-logit semantic readouts."""

from __future__ import annotations

import argparse
import collections
import json
import os
from pathlib import Path

import torch
from tokenizers import Tokenizer

from categorical_microcode import sha256_file
from model import GPT, GPTConfig
from referential_slot_microcode import compile_referential_example
from role_equivariant_microcode import OPERATION_KINDS, QUERY_KINDS, factor_operation, factor_query


OPERATION_WORDS = {
    "add": " add",
    "sub": " subtract",
    "move": " move",
    "merge": " merge",
    "swap": " swap",
}
QUERY_WORDS = {name: " " + name for name in QUERY_KINDS}


def load_lens(path):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    metadata = payload.get("metadata", {})
    if metadata.get("audit") != "shohin_future_jacobian_workspace_v1":
        raise ValueError("invalid Jacobian workspace artifact")
    return payload


def stability(left, right):
    output = {}
    for layer in left["metadata"]["source_layers"]:
        a = left["jacobians"][layer].float()
        b = right["jacobians"][layer].float()
        _, _, va = torch.linalg.svd(a, full_matrices=False)
        _, _, vb = torch.linalg.svd(b, full_matrices=False)
        output[str(layer)] = {
            "frobenius_cosine": float(torch.nn.functional.cosine_similarity(
                a.flatten(), b.flatten(), dim=0,
            ).item()),
            "relative_delta": float(((a - b).norm() / ((a.norm() + b.norm()) / 2)).item()),
            "right_subspace_overlap_k16": float((va[:16] @ vb[:16].T).pow(2).sum().div(16).item()),
            "right_subspace_overlap_k32": float((va[:32] @ vb[:32].T).pow(2).sum().div(32).item()),
            "right_subspace_overlap_k64": float((va[:64] @ vb[:64].T).pow(2).sum().div(64).item()),
        }
    return output


def rank_summary(records, method, layer, regimes=None):
    selected = [record for record in records if record["layer"] == layer]
    if regimes is not None:
        selected = [record for record in selected if record["regime"] in regimes]
    ranks = [record[method + "_rank"] for record in selected]
    if not ranks:
        raise ValueError("empty rank summary")
    ordered = sorted(ranks)
    return {
        "cases": len(ranks),
        "mrr": sum(1.0 / rank for rank in ranks) / len(ranks),
        "top1": sum(rank == 1 for rank in ranks) / len(ranks),
        "top10": sum(rank <= 10 for rank in ranks) / len(ranks),
        "top100": sum(rank <= 100 for rank in ranks) / len(ranks),
        "median_rank": ordered[len(ordered) // 2],
    }


def token_id(tokenizer, text):
    encoding = tokenizer.encode(text)
    if len(encoding.ids) != 1:
        raise ValueError("workspace label is not one token: {!r}".format(text))
    return encoding.ids[0]


def all_batches(examples, batch_size):
    buckets = collections.defaultdict(list)
    for index, example in enumerate(examples):
        buckets[len(example.compiled.ids)].append(index)
    return [
        indices[offset:offset + batch_size]
        for _, indices in sorted(buckets.items())
        for offset in range(0, len(indices), batch_size)
    ]


def residual_ranks(model, residual, target_ids, jacobian):
    immediate_logits = model.head(model.norm(residual)).float()
    transported = residual.float() @ jacobian.to(residual.device).float().T
    future_logits = model.head(model.norm(transported)).float()
    targets = torch.tensor(target_ids, dtype=torch.long, device=residual.device).unsqueeze(1)
    immediate_target = immediate_logits.gather(1, targets)
    future_target = future_logits.gather(1, targets)
    return (
        (immediate_logits > immediate_target).sum(dim=1).add(1).cpu().tolist(),
        (future_logits > future_target).sum(dim=1).add(1).cpu().tolist(),
        immediate_logits.argmax(dim=1).cpu().tolist(),
        future_logits.argmax(dim=1).cpu().tolist(),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--lens-a", required=True)
    parser.add_argument("--lens-b", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("Jacobian workspace evaluation requires CUDA")
    if Path(args.out).exists():
        raise SystemExit("refusing existing output")

    left, right = load_lens(args.lens_a), load_lens(args.lens_b)
    matched = (
        left["metadata"]["base_sha256"] == right["metadata"]["base_sha256"] == sha256_file(args.base)
        and left["metadata"]["tokenizer_sha256"] == right["metadata"]["tokenizer_sha256"]
        == sha256_file(args.tokenizer)
        and left["metadata"]["source_layers"] == right["metadata"]["source_layers"]
        and left["metadata"]["target_layer"] == right["metadata"]["target_layer"]
    )
    if not matched:
        raise SystemExit("lens artifacts do not bind the same base/tokenizer/layers")
    left_lines = {record["line_number"] for record in left["metadata"]["records"]}
    right_lines = {record["line_number"] for record in right["metadata"]["records"]}
    if left_lines & right_lines:
        raise SystemExit("lens prompt samples are not disjoint")
    layers = left["metadata"]["source_layers"]
    jacobians = {
        layer: (left["jacobians"][layer].float() + right["jacobians"][layer].float()) / 2
        for layer in layers
    }
    stability_report = stability(left, right)

    tokenizer = Tokenizer.from_file(args.tokenizer)
    rows = [json.loads(line) for line in Path(args.data).read_text().splitlines() if line.strip()]
    examples = [compile_referential_example(row, tokenizer) for row in rows]
    operation_tokens = {
        index: token_id(tokenizer, OPERATION_WORDS[name]) for index, name in enumerate(OPERATION_KINDS)
    }
    query_tokens = {
        index: token_id(tokenizer, QUERY_WORDS[name]) for index, name in enumerate(QUERY_KINDS)
    }
    checkpoint = torch.load(args.base, map_location="cpu", weights_only=False)
    model = GPT(GPTConfig(**checkpoint["cfg"])).to("cuda").eval()
    model.load_state_dict(checkpoint["model"])
    model.requires_grad_(False)
    batches = all_batches(examples, args.batch_size)
    records = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for batch_number, indices in enumerate(batches, 1):
            ids = torch.tensor(
                [examples[index].compiled.ids for index in indices], dtype=torch.long, device="cuda",
            )
            x = model.tok(ids)
            cos = model.cos[: ids.shape[1]].to(x.device)
            sin = model.sin[: ids.shape[1]].to(x.device)
            for layer, block in enumerate(model.blocks):
                x, _ = block(x, cos, sin)
                if layer not in jacobians:
                    continue
                selected_residuals = []
                target_ids = []
                descriptors = []
                for local, index in enumerate(indices):
                    wrapped = examples[index]
                    compiled = wrapped.compiled
                    for event, position, opcode in zip(
                        range(len(compiled.operation_targets)),
                        compiled.operation_positions,
                        compiled.operation_targets,
                    ):
                        kind, _ = factor_operation(opcode)
                        selected_residuals.append(x[local, position])
                        target_ids.append(operation_tokens[kind])
                        descriptors.append((index, "operation", event, OPERATION_KINDS[kind]))
                    query_kind, _ = factor_query(compiled.query_target)
                    selected_residuals.append(x[local, compiled.query_position])
                    target_ids.append(query_tokens[query_kind])
                    descriptors.append((index, "query", 0, QUERY_KINDS[query_kind]))
                residual = torch.stack(selected_residuals)
                immediate_rank, future_rank, immediate_top, future_top = residual_ranks(
                    model, residual, target_ids, jacobians[layer],
                )
                for descriptor, target, irank, frank, itop, ftop in zip(
                    descriptors, target_ids, immediate_rank, future_rank, immediate_top, future_top,
                ):
                    index, item_type, item_index, label = descriptor
                    records.append({
                        "example_index": index,
                        "reference": examples[index].compiled.reference,
                        "regime": examples[index].compiled.regime,
                        "item_type": item_type,
                        "item_index": item_index,
                        "label": label,
                        "target_token_id": target,
                        "layer": layer,
                        "immediate_rank": irank,
                        "future_rank": frank,
                        "immediate_top_token_id": itop,
                        "future_top_token_id": ftop,
                    })
            if batch_number % 20 == 0 or batch_number == len(batches):
                print("[jacobian-readout] {}/{} batches".format(batch_number, len(batches)), flush=True)

    summary = {}
    language_full = {"language_ood", "full_ood"}
    for layer in layers:
        summary[str(layer)] = {
            method: {
                "all": rank_summary(records, method, layer),
                "language_full": rank_summary(records, method, layer, language_full),
            }
            for method in ("immediate", "future")
        }
    eligible = [layer for layer in layers if 13 <= layer <= 25]
    selected_layer = max(
        eligible,
        key=lambda layer: (
            summary[str(layer)]["future"]["language_full"]["mrr"]
            - summary[str(layer)]["immediate"]["language_full"]["mrr"]
        ),
    )
    selected = summary[str(selected_layer)]
    gates = {
        "disjoint_prompt_samples": not (left_lines & right_lines),
        "all_matrix_cosines_at_least_0_90": all(
            cell["frobenius_cosine"] >= 0.90 for cell in stability_report.values()
        ),
        "selected_future_mrr_gain_at_least_25_percent": (
            selected["future"]["language_full"]["mrr"]
            >= 1.25 * selected["immediate"]["language_full"]["mrr"]
        ),
        "selected_future_top10_gain_at_least_0_10": (
            selected["future"]["language_full"]["top10"]
            - selected["immediate"]["language_full"]["top10"] >= 0.10
        ),
    }
    result = {
        "audit": "shohin_future_jacobian_semantic_readout_v1",
        "base": os.path.realpath(args.base),
        "base_sha256": sha256_file(args.base),
        "tokenizer": os.path.realpath(args.tokenizer),
        "tokenizer_sha256": sha256_file(args.tokenizer),
        "lens_a": os.path.realpath(args.lens_a),
        "lens_a_sha256": sha256_file(args.lens_a),
        "lens_b": os.path.realpath(args.lens_b),
        "lens_b_sha256": sha256_file(args.lens_b),
        "data": os.path.realpath(args.data),
        "data_sha256": sha256_file(args.data),
        "cases": len(examples),
        "records": records,
        "stability": stability_report,
        "summary": summary,
        "selected_layer": selected_layer,
        "gates": gates,
        "advance_to_causal_swap": all(gates.values()),
        "claim_boundary": (
            "A pass establishes stable future-Jacobian semantic readout over an immediate-logit "
            "control. It does not establish causal workspace use or reasoning without intervention."
        ),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("[jacobian-readout] " + json.dumps({
        "selected_layer": selected_layer, "gates": gates, "summary": summary,
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
