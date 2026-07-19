#!/usr/bin/env python3
"""Evaluate the sole S9 occurrence-quotient development board."""

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
    compile_row as compile_s8_row,
    recode_operation_ids,
)
from s8_nil_linked_law_graph import (
    derange_cards,
    execute_graph,
    linked_path,
    one_witness_unit_completion,
    rewire_path,
)
from s9_occurrence_quotient import (
    compile_quotient,
    permute_relation_storage,
    quotient_from_emitted_spans,
    reindex_classes,
)
from s9_occurrence_quotient_compiler import (
    OccurrenceQuotientCompiler,
    compile_row as compile_s9_row,
    emitted_spans_from_logits,
    load_adapter_state,
    load_examples,
    pad_batch,
    sha256_file,
)
from s9_occurrence_quotient_falsifier import expected_graph, semantic_key


def _instantiate(base, checkpoint, state_key):
    model = GPT(GPTConfig(**base["cfg"])).to("cuda").eval()
    model.load_state_dict(base["model"])
    architecture = checkpoint["architecture"]
    compiler = OccurrenceQuotientCompiler(
        model,
        layer=architecture["layer"],
        width=architecture["width"],
        heads=architecture["heads"],
        encoder_layers=architecture["encoder_layers"],
        ff=architecture["ff"],
    ).to("cuda")
    load_adapter_state(compiler, checkpoint[state_key])
    return compiler


def _decode_all(compiler, examples, batch_size, class_messages):
    result = []
    compiler.eval()
    with torch.no_grad():
        for start in range(0, len(examples), batch_size):
            indices = list(range(start, min(len(examples), start + batch_size)))
            selected, candidates, ids, valid, tensors = pad_batch(
                examples, indices, "cuda", negative_limit=None, seed=0
            )
            with torch.autocast("cuda", dtype=torch.bfloat16):
                outputs = compiler(ids, valid, tensors, class_messages=class_messages)
            cursor = 0
            for example, row_candidates in zip(selected, candidates, strict=True):
                width = len(row_candidates)
                logits = outputs["role_logits"][cursor:cursor + width]
                cursor += width
                try:
                    spans = emitted_spans_from_logits(example, row_candidates, logits)
                    quotient = quotient_from_emitted_spans(
                        str(example.row["question"]), spans
                    )
                    graph = compile_quotient(quotient)
                    result.append({"graph": graph, "spans": spans, "quotient": quotient})
                except (ValueError, IndexError) as error:
                    result.append({"error": str(error)})
    return result


def _new_score():
    return {"state": 0, "answer": 0, "total": 0, "depth": {}}


def _add(score, depth, output, expected_state, expected_answer):
    state_ok = output is not None and output[0] == expected_state
    answer_ok = output is not None and output[1] == expected_answer
    score["state"] += int(state_ok)
    score["answer"] += int(answer_ok)
    score["total"] += 1
    cell = score["depth"].setdefault(str(depth), {"correct": 0, "total": 0})
    cell["correct"] += int(state_ok)
    cell["total"] += 1


def _finish(score):
    return {
        **score,
        "state_accuracy": score["state"] / score["total"],
        "answer_accuracy": score["answer"] / score["total"],
        "depth": {
            key: {**value, "accuracy": value["correct"] / value["total"]}
            for key, value in sorted(score["depth"].items())
        },
    }


def _coarse(label: str) -> str:
    if label.startswith("entity.roster."):
        return "entity.roster"
    if label.startswith("position.roster."):
        return "position.roster"
    if label.startswith("state.entity."):
        return "state.entity"
    if label.startswith("card.") or label.startswith("event."):
        return label.rsplit(".", 1)[-1] if label.startswith("card.") else "event." + label.rsplit(".", 1)[-1]
    return label


def _span_set(spans):
    return {
        (int(value["start"]), int(value["end"]), _coarse(str(label)))
        for label, value in spans.items()
    }


def _state_hash(values):
    return hashlib.sha256(json.dumps(values, separators=(",", ":")).encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing existing S9 evaluation: {args.out}")
    if not torch.cuda.is_available():
        raise SystemExit("S9 evaluation requires CUDA")
    torch.set_float32_matmul_precision("high")
    report_path = args.data_dir / "report.json"
    report = json.loads(report_path.read_text())
    if report["audit"]["development_accesses"] or report["audit"]["confirmation_accesses"]:
        raise SystemExit("S9 board access counters are not zero")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("schema") != "r12_s9_occurrence_quotient_checkpoint_v1":
        raise SystemExit("unexpected S9 checkpoint schema")
    if checkpoint["board_report_sha256"] != sha256_file(report_path):
        raise SystemExit("S9 checkpoint/board mismatch")
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    base = torch.load(args.base, map_location="cpu", weights_only=False)
    development_path = args.data_dir / "development.jsonl"
    if sha256_file(development_path) != report["files"]["development.jsonl"]["sha256"]:
        raise SystemExit("S9 development hash mismatch")
    examples = load_examples(
        development_path, tokenizer, "s8_nil_graph_development", base["cfg"]["seq_len"]
    )
    if len(examples) != 2048:
        raise SystemExit("S9 requires 2,048 development rows")

    treatment_model = _instantiate(base, checkpoint, "treatment_adapter_state")
    treatment = _decode_all(treatment_model, examples, args.batch_size, True)
    recoded_examples = [
        compile_s9_row(
            recode_operation_ids(compile_s8_row(example.row, tokenizer), tokenizer).row,
            tokenizer,
        )
        for example in examples
    ]
    recoded = _decode_all(treatment_model, recoded_examples, args.batch_size, True)
    del treatment_model
    torch.cuda.empty_cache()
    no_class_model = _instantiate(base, checkpoint, "no_class_adapter_state")
    no_class = _decode_all(no_class_model, examples, args.batch_size, False)
    del no_class_model
    torch.cuda.empty_cache()
    shuffled_model = _instantiate(base, checkpoint, "shuffled_adapter_state")
    shuffled = _decode_all(shuffled_model, examples, args.batch_size, True)
    del shuffled_model
    torch.cuda.empty_cache()

    generator = LearnedCayleyGenerator().to("cuda")
    generator.load_state_dict(checkpoint["generator_state"])
    successors = {m: generator.discrete_successor(m) for m in PRIMARY_MODULI}
    zeros = {m: generator.discrete_zero(m) for m in PRIMARY_MODULI}
    arms = {name: _new_score() for name in (
        "gold_graph", "treatment", "no_class_message", "reversed_links",
        "deranged_cards", "one_witness", "state_reset", "early_nil",
    )}
    graph = {"valid": 0, "exact": 0, "no_class_exact": 0, "shuffled_exact": 0}
    spans = {"tp": 0, "predicted": 0, "gold": 0, "row_exact": 0, "class_exact": 0}
    invariance = {
        "class_reindex": 0, "relation_storage_reindex": 0,
        "eligible": 0, "nonce_identical": 0, "nonce_eligible": 0,
    }
    sample = []
    treatment_states = []
    for index, example in enumerate(examples):
        row = example.row
        modulus, depth = int(row["modulus"]), int(row["depth"])
        expected_state = tuple(int(value) for value in row["final_state"])
        expected_answer = int(row["answer"])
        expected = expected_graph(row)
        gold_output = execute_graph(expected, successors[modulus], zeros[modulus])
        _add(arms["gold_graph"], depth, gold_output, expected_state, expected_answer)
        decoded = treatment[index]
        output = None
        if "graph" in decoded:
            predicted = decoded["graph"]
            try:
                path = tuple(linked_path(predicted))
                graph["valid"] += 1
                graph["exact"] += int(semantic_key(predicted) == semantic_key(expected))
                output = execute_graph(predicted, successors[modulus], zeros[modulus])
                controls = {
                    "reversed_links": rewire_path(predicted, tuple(reversed(path))),
                    "deranged_cards": derange_cards(predicted),
                    "one_witness": one_witness_unit_completion(
                        predicted, successors[modulus], zeros[modulus]
                    ),
                }
                for name, control in controls.items():
                    _add(
                        arms[name], depth,
                        execute_graph(control, successors[modulus], zeros[modulus]),
                        expected_state, expected_answer,
                    )
                _add(
                    arms["state_reset"], depth,
                    execute_graph(predicted, successors[modulus], zeros[modulus], reset_state=True),
                    expected_state, expected_answer,
                )
                _add(
                    arms["early_nil"], depth,
                    execute_graph(predicted, successors[modulus], zeros[modulus], halt_after=1),
                    expected_state, expected_answer,
                )
                quotient = decoded["quotient"]
                class_perm = tuple(reversed(range(len(quotient.classes))))
                relation_perm = tuple(reversed(range(len(quotient.relations))))
                invariance["eligible"] += 1
                invariance["class_reindex"] += int(
                    semantic_key(compile_quotient(reindex_classes(quotient, class_perm)))
                    == semantic_key(predicted)
                )
                invariance["relation_storage_reindex"] += int(
                    semantic_key(compile_quotient(permute_relation_storage(quotient, relation_perm)))
                    == semantic_key(predicted)
                )
            except (ValueError, IndexError):
                pass
            predicted_spans = _span_set(decoded["spans"])
            gold_spans = _span_set(row["spans"])
            spans["tp"] += len(predicted_spans & gold_spans)
            spans["predicted"] += len(predicted_spans)
            spans["gold"] += len(gold_spans)
            spans["row_exact"] += int(predicted_spans == gold_spans)
            if {value[:2] for value in predicted_spans} == {value[:2] for value in gold_spans}:
                spans["class_exact"] += 1
        _add(arms["treatment"], depth, output, expected_state, expected_answer)
        treatment_states.append(list(output[0]) if output is not None else None)
        for name in ("reversed_links", "deranged_cards", "one_witness", "state_reset", "early_nil"):
            if arms[name]["total"] <= index:
                _add(arms[name], depth, None, expected_state, expected_answer)
        for decoded_arm, name in ((no_class[index], "no_class_message"),):
            arm_output = None
            if "graph" in decoded_arm:
                graph["no_class_exact"] += int(
                    semantic_key(decoded_arm["graph"]) == semantic_key(expected)
                )
                try:
                    arm_output = execute_graph(
                        decoded_arm["graph"], successors[modulus], zeros[modulus]
                    )
                except (ValueError, IndexError):
                    pass
            _add(arms[name], depth, arm_output, expected_state, expected_answer)
        if "graph" in shuffled[index]:
            graph["shuffled_exact"] += int(
                semantic_key(shuffled[index]["graph"]) == semantic_key(expected)
            )
        if output is not None and "graph" in recoded[index]:
            try:
                recoded_output = execute_graph(
                    recoded[index]["graph"], successors[modulus], zeros[modulus]
                )
                invariance["nonce_eligible"] += 1
                invariance["nonce_identical"] += int(recoded_output[:2] == output[:2])
            except (ValueError, IndexError):
                pass
        if len(sample) < 16:
            sample.append({
                "id": row["id"], "renderer": row["renderer"], "depth": depth,
                "valid": "graph" in decoded, "exact": (
                    "graph" in decoded and semantic_key(decoded["graph"]) == semantic_key(expected)
                ), "error": decoded.get("error"),
            })
    finished = {name: _finish(value) for name, value in arms.items()}
    precision = spans["tp"] / max(1, spans["predicted"])
    recall = spans["tp"] / max(1, spans["gold"])
    spans.update({
        "precision": precision, "recall": recall,
        "f1": 2 * precision * recall / max(1e-12, precision + recall),
        "row_exact_accuracy": spans["row_exact"] / len(examples),
        "class_exact_accuracy": spans["class_exact"] / len(examples),
    })
    evaluation = {
        "schema": "r12_s9_occurrence_quotient_development_evaluation_v1",
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "board_report_sha256": sha256_file(report_path),
        "development_sha256": sha256_file(development_path),
        "parameters": checkpoint["parameters"],
        "training_contract": checkpoint["training_contract"],
        "fit": {
            "generator": checkpoint["generator_fit"],
            "treatment": checkpoint["treatment_fit"],
            "no_class": checkpoint["no_class_fit"],
            "shuffled": checkpoint["shuffled_fit"],
        },
        "rows": len(examples),
        "span": spans,
        "graph": {**graph, **{key + "_accuracy": value / len(examples) for key, value in graph.items()}},
        "arms": finished,
        "invariance": invariance,
        "predicted_state_sha256": _state_hash(treatment_states),
        "samples": sample,
        "development_accesses": 1,
        "confirmation_accesses": 0,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(evaluation, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "out": str(args.out), "graph": evaluation["graph"],
        "span": spans, "treatment": finished["treatment"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
