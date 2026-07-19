#!/usr/bin/env python3
"""No-fit probe of ordered token-set relations for composed referent identity."""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from model import GPTConfig
from referential_gather_delete_executor import normalized_sigmoid_weights
from referential_literal_pointer_compiler import compile_row, make_batches, pad_batch, sha256_file
from train_referential_gather_delete_executor import load_frozen_compiler


METHODS = (
    "lexical_mean_t1",
    "token_mass_cosine_t05",
    "ordered_sequence_t1",
    "ordered_sequence_t05",
    "ordered_sequence_t025",
    "gold_ordered_sequence",
)


def role_weights(logits, valid, temperature):
    return normalized_sigmoid_weights(logits, valid, temperature=temperature)


def gold_weights(examples, label, length, device):
    weights = torch.zeros((len(examples), length), device=device)
    for row, example in enumerate(examples):
        positions = tuple(map(int, example.target_positions[label]))
        weights[row, list(positions)] = 1.0 / len(positions)
    return weights


def gather(memory, weights):
    return torch.einsum("bl,bld->bd", weights.to(memory.dtype), memory).float()


def lexical_mean_scores(memory, initial_weights, operation_weights):
    initial = torch.stack([
        gather(memory, weights) for weights in initial_weights
    ], dim=1)
    operation = gather(memory, operation_weights)
    return torch.einsum(
        "bd,bid->bi", F.normalize(operation, dim=-1), F.normalize(initial, dim=-1),
    )


def token_mass(ids, weights, vocab_size):
    mass = torch.zeros(
        (ids.shape[0], len(weights), vocab_size), device=ids.device, dtype=torch.float32,
    )
    index = ids.unsqueeze(1).expand(-1, len(weights), -1)
    return mass.scatter_add_(2, index, torch.stack(weights, dim=1).float())


def token_mass_cosine_scores(ids, initial_weights, operation_weights, vocab_size):
    initial = token_mass(ids, initial_weights, vocab_size)
    operation = token_mass(ids, [operation_weights], vocab_size)[:, 0]
    return torch.einsum(
        "bv,biv->bi", F.normalize(operation, dim=-1), F.normalize(initial, dim=-1),
    )


def ordered_sequence_scores(ids, initial_weights, operation_weights):
    """Translation-invariant exact-token sequence kernel over soft role spans."""
    length = ids.shape[1]
    initial = torch.stack(initial_weights, dim=1).float()
    operation = operation_weights.float()
    equality = ids[:, None, :, None].eq(ids[:, None, None, :])
    weighted = (
        operation[:, None, :, None]
        * initial[:, :, None, :]
        * equality.float()
    )
    diagonal_scores = []
    for offset in range(-length + 1, length):
        diagonal_scores.append(weighted.diagonal(offset=offset, dim1=-2, dim2=-1).sum(-1))
    numerator = torch.stack(diagonal_scores, dim=-1).amax(-1)
    left_norm = operation.square().sum(-1, keepdim=True).sqrt()
    right_norm = initial.square().sum(-1).sqrt()
    return numerator / (left_norm * right_norm).clamp_min(1e-8)


def load_board(path):
    rows = [json.loads(line) for line in open(path) if line.strip()]
    if not rows or any(row.get("split") != "development_relational" for row in rows):
        raise ValueError("invalid relational-development board")
    return rows


def summarize(correct, total):
    return {
        "correct": int(correct),
        "total": int(total),
        "accuracy": float(correct / total) if total else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--compiler", required=True)
    parser.add_argument("--board", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("relational identity probe requires CUDA")
    if Path(args.out).exists():
        raise SystemExit("refusing existing relational identity output")
    report = json.load(open(args.report))
    if not report.get("all_gates_pass") or report.get("confirmation_access") != 0:
        raise SystemExit("relational-development report is not admitted")
    if report["artifact"]["sha256"] != sha256_file(args.board):
        raise SystemExit("report does not bind relational-development board")

    checkpoint, compiler, compiler_metadata = load_frozen_compiler(
        args.base, args.compiler, args.tokenizer, "cuda",
    )
    tokenizer = Tokenizer.from_file(args.tokenizer)
    cfg = GPTConfig(**checkpoint["cfg"])
    rows = load_board(args.board)
    examples = []
    metadata = []
    for row in rows:
        for chunk in row["chunks"]:
            examples.append(compile_row(chunk, tokenizer, keep_evidence=True))
            metadata.append({
                "active": int(chunk["active_operations"]),
                "depth": int(row["depth"]),
                "surface": row["surface_type"],
            })

    correct = collections.Counter()
    total = collections.Counter()
    by_depth = {method: collections.Counter() for method in METHODS}
    by_surface = {method: collections.Counter() for method in METHODS}
    batches = make_batches(examples, args.batch_size, seed=0, shuffle=False)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for batch_number, indices in enumerate(batches, 1):
            selected, ids, valid = pad_batch(examples, indices, "cuda")
            if ids.shape[1] > cfg.seq_len:
                raise ValueError("relational-development chunk exceeds sequence length")
            outputs = compiler(ids, valid)
            lexical = compiler.model.tok(ids).detach()
            logits = outputs["pointer_logits"]
            weights = {}
            for temperature in (1.0, 0.5, 0.25):
                key = str(temperature)
                weights[key] = {
                    label: role_weights(logits[label], valid, temperature)
                    for label in (
                        "intro.entity0", "intro.entity1", "intro.entity2",
                        "op0.entity", "op1.entity",
                    )
                }
            gold = {
                label: gold_weights(selected, label, ids.shape[1], ids.device)
                for label in (
                    "intro.entity0", "intro.entity1", "intro.entity2",
                    "op0.entity", "op1.entity",
                )
            }
            score_maps = []
            for operation_index in range(2):
                op_label = "op{}.entity".format(operation_index)
                score_maps.append({
                    "lexical_mean_t1": lexical_mean_scores(
                        lexical,
                        [weights["1.0"]["intro.entity{}".format(i)] for i in range(3)],
                        weights["1.0"][op_label],
                    ),
                    "token_mass_cosine_t05": token_mass_cosine_scores(
                        ids,
                        [weights["0.5"]["intro.entity{}".format(i)] for i in range(3)],
                        weights["0.5"][op_label],
                        tokenizer.get_vocab_size(),
                    ),
                    "ordered_sequence_t1": ordered_sequence_scores(
                        ids,
                        [weights["1.0"]["intro.entity{}".format(i)] for i in range(3)],
                        weights["1.0"][op_label],
                    ),
                    "ordered_sequence_t05": ordered_sequence_scores(
                        ids,
                        [weights["0.5"]["intro.entity{}".format(i)] for i in range(3)],
                        weights["0.5"][op_label],
                    ),
                    "ordered_sequence_t025": ordered_sequence_scores(
                        ids,
                        [weights["0.25"]["intro.entity{}".format(i)] for i in range(3)],
                        weights["0.25"][op_label],
                    ),
                    "gold_ordered_sequence": ordered_sequence_scores(
                        ids,
                        [gold["intro.entity{}".format(i)] for i in range(3)],
                        gold[op_label],
                    ),
                })
            for local, global_index in enumerate(indices):
                example = selected[local]
                item = metadata[global_index]
                for operation_index in range(item["active"]):
                    target = example.initial_order.index(example.program[operation_index][1])
                    for method, scores in score_maps[operation_index].items():
                        hit = int(scores[local].argmax(-1).item()) == int(target)
                        correct[method] += int(hit)
                        total[method] += 1
                        by_depth[method]["{}_correct".format(item["depth"])] += int(hit)
                        by_depth[method]["{}_total".format(item["depth"])] += 1
                        by_surface[method]["{}_correct".format(item["surface"])] += int(hit)
                        by_surface[method]["{}_total".format(item["surface"])] += 1
            if batch_number % 25 == 0:
                print("[rgde-relational] {}/{} batches".format(
                    batch_number, len(batches),
                ), flush=True)

    method_results = {}
    depths = sorted({int(row["depth"]) for row in rows})
    surfaces = sorted({row["surface_type"] for row in rows})
    for method in METHODS:
        method_results[method] = summarize(correct[method], total[method])
        method_results[method]["by_depth"] = {
            str(depth): summarize(
                by_depth[method]["{}_correct".format(depth)],
                by_depth[method]["{}_total".format(depth)],
            ) for depth in depths
        }
        method_results[method]["by_surface"] = {
            surface: summarize(
                by_surface[method]["{}_correct".format(surface)],
                by_surface[method]["{}_total".format(surface)],
            ) for surface in surfaces
        }
    primary = method_results["ordered_sequence_t05"]
    baseline = method_results["lexical_mean_t1"]
    bag = method_results["token_mass_cosine_t05"]
    gates = {
        "primary_overall_at_least_99pct": primary["accuracy"] >= 0.99,
        "primary_each_depth_at_least_985pct": all(
            value["accuracy"] >= 0.985 for value in primary["by_depth"].values()
        ),
        "primary_each_surface_at_least_985pct": all(
            value["accuracy"] >= 0.985 for value in primary["by_surface"].values()
        ),
        "primary_beats_mean_by_15_points": (
            primary["accuracy"] - baseline["accuracy"] >= 0.15
        ),
        "primary_not_worse_than_unordered_bag": primary["accuracy"] >= bag["accuracy"],
        "gold_ceiling_at_least_999pct": (
            method_results["gold_ordered_sequence"]["accuracy"] >= 0.999
        ),
    }
    result = {
        "schema": "r12_rgde_relational_identity_probe_v1",
        "base_sha256": sha256_file(args.base),
        "compiler_sha256": sha256_file(args.compiler),
        "compiler_adapter_sha256": compiler_metadata["final_adapter_sha256"],
        "board_sha256": sha256_file(args.board),
        "report_sha256": sha256_file(args.report),
        "tokenizer_sha256": sha256_file(args.tokenizer),
        "probe_sha256": sha256_file(__file__),
        "rows": len(rows),
        "chunks": len(examples),
        "entity_references": sum(int(item["active"]) for item in metadata),
        "methods": method_results,
        "gates": gates,
        "all_gates_pass": all(gates.values()),
        "fit_updates": 0,
        "confirmation_access": 0,
        "claim_boundary": (
            "No-fit public compiler-interface diagnostic. No executor, sealed confirmation, "
            "autonomous reasoning, or novelty claim."
        ),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "all_gates_pass": result["all_gates_pass"],
        "gates": gates,
        "methods": {name: value["accuracy"] for name, value in method_results.items()},
        "out": str(Path(args.out).resolve()),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
