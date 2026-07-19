#!/usr/bin/env python3
"""Evaluate S8 whole-source graph compilation on the sole development board."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from tokenizers import Tokenizer

from model import GPT, GPTConfig
from s7_learned_cayley_generator import LearnedCayleyGenerator, PRIMARY_MODULI
from s8_nil_linked_graph_compiler import (
    NilLinkedGraphCompiler,
    decode_graph,
    gold_graph,
    load_examples,
    load_adapter_state,
    pad_batch,
    recode_operation_ids,
    reindex_graph,
    semantic_graph_key,
    sha256_file,
)
from s8_nil_linked_law_graph import (
    derange_cards,
    execute_graph,
    one_witness_unit_completion,
    rewire_path,
)


def _state_hash(states: list[tuple[int, ...] | None]) -> str:
    payload = json.dumps(states, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _new_score() -> dict[str, object]:
    return {
        "state_correct": 0,
        "answer_correct": 0,
        "total": 0,
        "depth_state": {},
        "predicted_states": [],
    }


def _add_score(score, depth, predicted, expected_state, expected_answer):
    state_ok = predicted is not None and predicted[0] == expected_state
    answer_ok = predicted is not None and predicted[1] == expected_answer
    score["state_correct"] += int(state_ok)
    score["answer_correct"] += int(answer_ok)
    score["total"] += 1
    cell = score["depth_state"].setdefault(
        str(depth), {"correct": 0, "total": 0}
    )
    cell["correct"] += int(state_ok)
    cell["total"] += 1
    score["predicted_states"].append(predicted[0] if predicted is not None else None)


def _finish_score(score):
    result = {
        "state_correct": score["state_correct"],
        "answer_correct": score["answer_correct"],
        "total": score["total"],
        "state_accuracy": score["state_correct"] / score["total"],
        "answer_accuracy": score["answer_correct"] / score["total"],
        "depth_state": {
            depth: {
                **cell,
                "accuracy": cell["correct"] / cell["total"],
            }
            for depth, cell in sorted(score["depth_state"].items())
        },
        "predicted_state_sha256": _state_hash(score["predicted_states"]),
    }
    return result


def _instantiate(base, checkpoint, state_key, device):
    cfg = GPTConfig(**base["cfg"])
    model = GPT(cfg).to(device).eval()
    model.load_state_dict(base["model"])
    architecture = checkpoint["architecture"]
    compiler = NilLinkedGraphCompiler(
        model,
        layer=architecture["layer"],
        width=architecture["width"],
        heads=architecture["heads"],
        encoder_layers=architecture["encoder_layers"],
        ff=architecture["ff"],
    ).to(device)
    load_adapter_state(compiler, checkpoint[state_key])
    return compiler


def _decode_all(compiler, examples, batch_size, recoded=False):
    decoded = []
    compiler.eval()
    source = [recode_operation_ids(example) for example in examples] if recoded else examples
    with torch.no_grad():
        for start in range(0, len(source), batch_size):
            indices = list(range(start, min(len(source), start + batch_size)))
            selected, ids, valid, _, _ = pad_batch(source, indices, "cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                outputs = compiler(ids, valid)
            for row, example in enumerate(selected):
                try:
                    decoded.append(
                        decode_graph(
                            example,
                            outputs["role_logits"][row],
                            outputs["rank_logits"][row],
                        )
                    )
                except (ValueError, IndexError) as error:
                    decoded.append({"error": str(error)})
    return decoded


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing existing S8 evaluation: {args.out}")
    if not torch.cuda.is_available():
        raise SystemExit("S8 evaluation requires CUDA")
    torch.set_float32_matmul_precision("high")

    report_path = args.data_dir / "report.json"
    report = json.loads(report_path.read_text())
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("schema") != "r12_s8_nil_linked_law_graph_checkpoint_v1":
        raise SystemExit("unexpected S8 checkpoint schema")
    if checkpoint["board_report_sha256"] != sha256_file(report_path):
        raise SystemExit("S8 checkpoint/board mismatch")
    if checkpoint["base_sha256"] != sha256_file(args.base):
        raise SystemExit("S8 checkpoint/base mismatch")
    if checkpoint["tokenizer_sha256"] != sha256_file(args.tokenizer):
        raise SystemExit("S8 checkpoint/tokenizer mismatch")
    if report["audit"]["development_accesses"] != 0 or report["audit"]["confirmation_accesses"] != 0:
        raise SystemExit("S8 board access counters are not zero")
    development_path = args.data_dir / "development.jsonl"
    if sha256_file(development_path) != report["files"]["development.jsonl"]["sha256"]:
        raise SystemExit("S8 development hash mismatch")
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    base = torch.load(args.base, map_location="cpu", weights_only=False)
    cfg = GPTConfig(**base["cfg"])
    examples = load_examples(
        development_path,
        tokenizer,
        "s8_nil_graph_development",
        cfg.seq_len,
    )
    if len(examples) != 2048:
        raise SystemExit("S8 v1 requires 2,048 development rows")

    treatment_compiler = _instantiate(
        base, checkpoint, "treatment_adapter_state", "cuda"
    )
    treatment_decoded = _decode_all(
        treatment_compiler, examples, args.batch_size
    )
    recoded_decoded = _decode_all(
        treatment_compiler, examples, args.batch_size, recoded=True
    )
    del treatment_compiler
    torch.cuda.empty_cache()
    shuffled_compiler = _instantiate(
        base, checkpoint, "shuffled_adapter_state", "cuda"
    )
    shuffled_decoded = _decode_all(
        shuffled_compiler, examples, args.batch_size
    )
    del shuffled_compiler
    torch.cuda.empty_cache()

    generator = LearnedCayleyGenerator().to("cuda")
    generator.load_state_dict(checkpoint["generator_state"])
    successors = {
        modulus: generator.discrete_successor(modulus) for modulus in PRIMARY_MODULI
    }
    zeros = {
        modulus: generator.discrete_zero(modulus) for modulus in PRIMARY_MODULI
    }
    arms = {
        name: _new_score() for name in (
            "gold_graph",
            "treatment",
            "ordinary_sequence_parser",
            "storage_order_shortcut",
            "reversed_links",
            "deranged_cards",
            "one_witness",
            "state_reset",
            "early_nil",
        )
    }
    graph_valid = 0
    graph_exact = 0
    count_halt_exact = 0
    shuffled_graph_exact = 0
    reindex_invariant = 0
    reindex_eligible = 0
    nonce_invariant = 0
    nonce_eligible = 0
    samples = []
    for index, example in enumerate(examples):
        row = example.row
        modulus = int(row["modulus"])
        depth = int(row["depth"])
        expected_state = tuple(int(value) for value in row["final_state"])
        expected_answer = int(row["answer"])
        gold = gold_graph(example)
        gold_output = execute_graph(gold, successors[modulus], zeros[modulus])
        _add_score(
            arms["gold_graph"], depth, gold_output, expected_state, expected_answer
        )

        decoded = treatment_decoded[index]
        predictions = {name: None for name in arms if name != "gold_graph"}
        treatment_output = None
        if "graph" in decoded:
            graph = decoded["graph"]
            runtime_modulus = graph.modulus
            predictions["storage_order_shortcut"] = execute_graph(
                graph,
                successors[runtime_modulus],
                zeros[runtime_modulus],
                storage_order=True,
            )
            if decoded["treatment_path"] is not None:
                graph_valid += 1
                count_halt_exact += int(
                    len(graph.nodes) == depth
                    and len(decoded["treatment_path"]) == depth
                )
                graph_exact += int(
                    semantic_graph_key(graph) == semantic_graph_key(gold)
                )
                treatment_output = execute_graph(
                    graph, successors[runtime_modulus], zeros[runtime_modulus]
                )
                predictions["treatment"] = treatment_output
                predictions["reversed_links"] = execute_graph(
                    rewire_path(
                        graph, tuple(reversed(decoded["treatment_path"]))
                    ),
                    successors[runtime_modulus],
                    zeros[runtime_modulus],
                )
                predictions["deranged_cards"] = execute_graph(
                    derange_cards(graph),
                    successors[runtime_modulus],
                    zeros[runtime_modulus],
                )
                predictions["one_witness"] = execute_graph(
                    one_witness_unit_completion(
                        graph,
                        successors[runtime_modulus],
                        zeros[runtime_modulus],
                    ),
                    successors[runtime_modulus],
                    zeros[runtime_modulus],
                )
                predictions["state_reset"] = execute_graph(
                    graph,
                    successors[runtime_modulus],
                    zeros[runtime_modulus],
                    reset_state=True,
                )
                predictions["early_nil"] = execute_graph(
                    graph,
                    successors[runtime_modulus],
                    zeros[runtime_modulus],
                    halt_after=1,
                )
                permutation = tuple(reversed(range(len(graph.nodes))))
                renamed_output = execute_graph(
                    reindex_graph(graph, permutation),
                    successors[runtime_modulus],
                    zeros[runtime_modulus],
                )
                reindex_eligible += 1
                reindex_invariant += int(renamed_output[:2] == treatment_output[:2])
            if decoded.get("ordinary_graph") is not None:
                predictions["ordinary_sequence_parser"] = execute_graph(
                    decoded["ordinary_graph"],
                    successors[runtime_modulus],
                    zeros[runtime_modulus],
                )
        recoded = recoded_decoded[index]
        if (
            treatment_output is not None
            and "graph" in recoded
            and recoded["treatment_path"] is not None
        ):
            recoded_modulus = recoded["graph"].modulus
            recoded_output = execute_graph(
                recoded["graph"],
                successors[recoded_modulus],
                zeros[recoded_modulus],
            )
            nonce_eligible += 1
            nonce_invariant += int(recoded_output[:2] == treatment_output[:2])
        shuffled = shuffled_decoded[index]
        if "graph" in shuffled and shuffled["treatment_path"] is not None:
            shuffled_graph_exact += int(
                semantic_graph_key(shuffled["graph"]) == semantic_graph_key(gold)
            )
        for name, predicted in predictions.items():
            _add_score(
                arms[name], depth, predicted, expected_state, expected_answer
            )
        if len(samples) < 16:
            samples.append({
                "id": row["id"],
                "depth": depth,
                "renderer": row["renderer"],
                "graph_valid": decoded.get("treatment_path") is not None,
                "graph_exact": (
                    "graph" in decoded
                    and semantic_graph_key(decoded["graph"]) == semantic_graph_key(gold)
                ),
                "expected_state": list(expected_state),
                "predicted_state": (
                    list(treatment_output[0]) if treatment_output is not None else None
                ),
                "error": decoded.get("error"),
            })

    evaluation = {
        "schema": "r12_s8_nil_linked_law_graph_development_evaluation_v1",
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "board_report_sha256": sha256_file(report_path),
        "development_sha256": sha256_file(development_path),
        "parameters": checkpoint["parameters"],
        "training_contract": checkpoint["training_contract"],
        "fit": {
            "treatment": checkpoint["treatment_fit"],
            "shuffled": checkpoint["shuffled_fit"],
            "generator": checkpoint["generator_fit"],
        },
        "rows": len(examples),
        "graph": {
            "valid": graph_valid,
            "valid_accuracy": graph_valid / len(examples),
            "exact": graph_exact,
            "exact_accuracy": graph_exact / len(examples),
            "count_halt_exact": count_halt_exact,
            "count_halt_accuracy": count_halt_exact / len(examples),
            "shuffled_exact": shuffled_graph_exact,
            "shuffled_exact_accuracy": shuffled_graph_exact / len(examples),
        },
        "arms": {name: _finish_score(score) for name, score in arms.items()},
        "invariance": {
            "graph_reindex_identical": reindex_invariant,
            "graph_reindex_eligible": reindex_eligible,
            "graph_reindex_accuracy": (
                reindex_invariant / reindex_eligible if reindex_eligible else 0.0
            ),
            "operation_nonce_identical": nonce_invariant,
            "operation_nonce_eligible": nonce_eligible,
            "operation_nonce_accuracy": (
                nonce_invariant / nonce_eligible if nonce_eligible else 0.0
            ),
        },
        "samples": samples,
        "development_accesses": 1,
        "confirmation_accesses": 0,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(evaluation, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "out": str(args.out),
        "graph_exact": evaluation["graph"]["exact_accuracy"],
        "state": evaluation["arms"]["treatment"]["state_accuracy"],
        "answer": evaluation["arms"]["treatment"]["answer_accuracy"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
