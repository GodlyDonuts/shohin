#!/usr/bin/env python3
"""Apply frozen recurrent-depth confirmation gates without model access."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(path):
    with open(path) as handle:
        return json.load(handle)


def metric(result, name):
    return float(result["overall"][name])


def depth_gate(result, depths, answer, final, transitions):
    rows = {
        depth: {
            "answer": float(result["by_depth"][str(depth)]["answer_accuracy"]),
            "final": float(result["by_depth"][str(depth)]["final_assignment_exact"]),
            "transitions": float(result["by_depth"][str(depth)]["all_transitions_exact"]),
        }
        for depth in depths
    }
    return {
        "depths": rows,
        "thresholds": {"answer": answer, "final": final, "transitions": transitions},
        "pass": all(
            row["answer"] >= answer
            and row["final"] >= final
            and row["transitions"] >= transitions
            for row in rows.values()
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-dir", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--sacct", required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    root = Path(args.evaluation_dir)
    paths = {
        "predicted": root / "predicted.json",
        "gold": root / "gold_packet.json",
        "operation": root / "operation_derangement.json",
        "query": root / "query_derangement.json",
        "report": Path(args.report),
        "sacct": Path(args.sacct),
    }
    out = Path(args.out)
    if out.exists():
        raise SystemExit("refusing existing depth assessment")
    if any(not path.is_file() for path in paths.values()):
        raise SystemExit("missing depth assessment input")
    predicted, gold, operation, query, report = (
        load(paths[name]) for name in ("predicted", "gold", "operation", "query", "report")
    )
    job = None
    for line in paths["sacct"].read_text().splitlines():
        fields = line.split("|")
        if fields and fields[0] == args.job_id:
            job = {
                "job_id": fields[0], "name": fields[1], "state": fields[2],
                "exit_code": fields[3], "elapsed": fields[4], "node": fields[5],
            }
    if job is None:
        raise SystemExit("expected depth job absent from sacct")
    answer = metric(predicted, "answer_accuracy")
    final = metric(predicted, "final_assignment_exact")
    gates = {
        "g1_board_custody": {
            "all_board_gates": report.get("all_gates_pass"),
            "old_confirmation_access": report.get("old_confirmation_access"),
            "pass": report.get("all_gates_pass") is True
            and report.get("old_confirmation_access") == 0,
        },
        "g2_depths_3_4": depth_gate(predicted, (3, 4), 0.95, 0.95, 0.90),
        "g3_depths_5_6": depth_gate(predicted, (5, 6), 0.90, 0.90, 0.80),
        "g4_depths_7_8": depth_gate(predicted, (7, 8), 0.85, 0.85, 0.70),
        "g5_surfaces_quartets": {
            "minimum_surface_answer": min(
                float(row["answer_accuracy"]) for row in predicted["by_surface"].values()
            ),
            "all_four_answers": predicted["group_summary"]["all_four_answers_correct"],
            "pass": (
                min(float(row["answer_accuracy"])
                    for row in predicted["by_surface"].values()) >= 0.85
                and predicted["group_summary"]["all_four_answers_correct"] >= 400
            ),
        },
        "g6_gold_gap": {
            "answer_gap": metric(gold, "answer_accuracy") - answer,
            "final_gap": metric(gold, "final_assignment_exact") - final,
            "pass": (
                abs(metric(gold, "answer_accuracy") - answer) <= 0.03
                and abs(metric(gold, "final_assignment_exact") - final) <= 0.03
            ),
        },
        "g7_entity_amount": {
            "entity": metric(predicted, "entity_match_accuracy"),
            "amount": metric(predicted, "amount_accuracy"),
            "pass": metric(predicted, "entity_match_accuracy") >= 0.98
            and metric(predicted, "amount_accuracy") >= 0.98,
        },
        "g8_operation_causality": {
            "answer": metric(operation, "answer_accuracy"),
            "final": metric(operation, "final_assignment_exact"),
            "answer_drop": answer - metric(operation, "answer_accuracy"),
            "final_drop": final - metric(operation, "final_assignment_exact"),
            "pass": (
                metric(operation, "answer_accuracy") <= 0.40
                and metric(operation, "final_assignment_exact") <= 0.40
                and answer - metric(operation, "answer_accuracy") >= 0.50
                and final - metric(operation, "final_assignment_exact") >= 0.50
            ),
        },
        "g9_query_causality": {
            "answer": metric(query, "answer_accuracy"),
            "answer_drop": answer - metric(query, "answer_accuracy"),
            "final_change": abs(final - metric(query, "final_assignment_exact")),
            "pass": (
                metric(query, "answer_accuracy") <= 0.05
                and answer - metric(query, "answer_accuracy") >= 0.90
                and abs(final - metric(query, "final_assignment_exact")) <= 0.01
            ),
        },
        "g10_receipts": {
            "job": job,
            "operation_interventions": operation.get("intervention_rows"),
            "query_interventions": query.get("intervention_rows"),
            "old_confirmation_access": [
                row.get("old_confirmation_access")
                for row in (predicted, gold, operation, query)
            ],
            "pass": (
                job["state"] == "COMPLETED"
                and job["exit_code"] == "0:0"
                and operation.get("intervention_rows") == predicted.get("rows")
                and query.get("intervention_rows") == predicted.get("rows")
                and all(row.get("old_confirmation_access") == 0
                        for row in (predicted, gold, operation, query))
                and len({
                    row.get("board_sha256") for row in (predicted, gold, operation, query)
                }) == 1
                and len({
                    row.get("executor_sha256") for row in (predicted, gold, operation, query)
                }) == 1
            ),
        },
    }
    all_pass = all(row["pass"] for row in gates.values())
    result = {
        "schema": "r12_rgde_depth_confirmation_assessment_v1",
        "decision": (
            "confirm_rgde_recurrent_execution_through_depth_8"
            if all_pass else
            "reject_rgde_depth_confirmation"
        ),
        "all_gates_pass": all_pass,
        "gates": gates,
        "input_sha256": {name: sha256_file(path) for name, path in paths.items()},
        "claim_boundary": (
            "Source-deleted packet-stream depth confirmation with external operation count "
            "and halt; not autonomous language reasoning."
        ),
    }
    canonical = json.dumps(result, sort_keys=True, separators=(",", ":")).encode()
    result["assessment_sha256"] = hashlib.sha256(canonical).hexdigest()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "decision": result["decision"],
        "assessment_sha256": result["assessment_sha256"],
        "out": str(out.resolve()),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
