#!/usr/bin/env python3
"""Contracts for the frozen R5 fresh-board comparator."""

import copy
import json
import subprocess
import tempfile
from pathlib import Path


def report(argument_graph, answers, programs):
    summary = {}
    cases = {"fit_iid": 256, "depth_ood": 192, "language_ood": 256, "full_ood": 192}
    for regime, count in cases.items():
        correct = answers[regime]
        exact = programs[regime]
        summary[regime] = {
            "cases": count, "answer_correct": correct, "answer_accuracy": correct / count,
            "program_exact": exact, "program_exact_accuracy": exact / count,
        }
    summary["all"] = {
        "cases": 896, "answer_correct": sum(answers.values()),
        "answer_accuracy": sum(answers.values()) / 896,
        "program_exact": sum(programs.values()),
        "program_exact_accuracy": sum(programs.values()) / 896,
    }
    graph = {"enabled": argument_graph}
    if argument_graph:
        for regime, operations in (("language_ood", 800), ("full_ood", 1000)):
            graph[regime] = {
                "operations": operations, "arity_correct": operations,
                "arity_accuracy": 1.0, "target_operation_kinds": list(range(5)),
                "target_query_kinds": list(range(3)),
            }
    return {
        "audit": (
            "referential_argument_graph_eval_v5" if argument_graph
            else "referential_slot_microcode_eval_v4"
        ),
        "base_sha256": "b", "adapter_sha256": "p", "data_sha256": "d",
        "admission_sha256": "a", "label_admission_sha256": "l",
        "evaluation_label_admission_sha256": "e",
        "adapter_metadata": {"role_mode": "pointer", "protocol": "v4"},
        "summary": summary, "gates": {"absolute": True},
        "argument_graph": {"enabled": argument_graph, "threshold": 0.80 if argument_graph else None},
        "argument_graph_diagnostics": graph,
    }


def run(root, raw_path, candidate_path, out):
    subprocess.run([
        "python3", str(root / "compare_referential_argument_graph.py"),
        "--raw", str(raw_path), "--argument-graph", str(candidate_path), "--out", str(out),
    ], check=True, capture_output=True, text=True)
    return json.loads(out.read_text())


def main():
    root = Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory() as directory:
        directory = Path(directory)
        raw = report(False, {
            "fit_iid": 240, "depth_ood": 180, "language_ood": 100, "full_ood": 40,
        }, {"fit_iid": 220, "depth_ood": 150, "language_ood": 70, "full_ood": 20})
        candidate = report(True, {
            "fit_iid": 235, "depth_ood": 170, "language_ood": 205, "full_ood": 130,
        }, {"fit_iid": 215, "depth_ood": 145, "language_ood": 170, "full_ood": 100})
        raw_path, candidate_path, out = directory / "raw.json", directory / "graph.json", directory / "out.json"
        raw_path.write_text(json.dumps(raw))
        candidate_path.write_text(json.dumps(candidate))
        result = run(root, raw_path, candidate_path, out)
        assert result["advance_to_future_effect_operator_fit"]
        assert result["decision"] == "advance_argument_graph_r5_to_operator_fit"

        failed = copy.deepcopy(candidate)
        failed["argument_graph"]["threshold"] = 0.79
        candidate_path.write_text(json.dumps(failed))
        out.unlink()
        assert not run(root, raw_path, candidate_path, out)["advance_to_future_effect_operator_fit"]
    print("referential argument graph comparator tests passed")


if __name__ == "__main__":
    main()
