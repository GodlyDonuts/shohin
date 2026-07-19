#!/usr/bin/env python3
"""Mechanically assess the one-shot lexical closed-S3 confirmation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from referential_literal_pointer_compiler import sha256_file


def load(path, kind_protocol):
    value = json.load(open(path))
    if value.get("schema") != "r12_s3_lexical_confirmation_eval_v1":
        raise SystemExit("invalid S3 lexical confirmation result")
    if value.get("kind_protocol", "training_lexicon_v1") != kind_protocol:
        raise SystemExit("S3 lexical confirmation kind protocol mismatch")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True)
    parser.add_argument("--ordered", required=True)
    parser.add_argument("--mean", required=True)
    parser.add_argument("--gold", required=True)
    parser.add_argument("--operation", required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--sacct", required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--kind-protocol",
        choices=("training_lexicon_v1", "training_lexicon_pointer_anchor_v1"),
        default="training_lexicon_v1",
    )
    args = parser.parse_args()
    if Path(args.out).exists():
        raise SystemExit("refusing existing S3 lexical confirmation assessment")
    paths = {name: getattr(args, name) for name in (
        "report", "ordered", "mean", "gold", "operation", "query", "sacct",
    )}
    report = json.load(open(args.report))
    values = {
        name: load(path, args.kind_protocol)
        for name, path in paths.items() if name not in {"report", "sacct"}
    }
    ordered = values["ordered"]
    mean = values["mean"]
    gold = values["gold"]
    operation = values["operation"]
    query = values["query"]
    job = None
    for line in Path(args.sacct).read_text().splitlines():
        fields = line.split("|")
        if fields and fields[0] == args.job_id:
            job = fields
    if job is None:
        raise SystemExit("confirmation job absent from sacct")
    primary = ordered["overall"]
    gates = {
        "board_custody": (
            report.get("all_gates_pass") is True
            and report.get("old_confirmation_access") == 0
        ),
        "ordered_overall_answer_state_chains": (
            primary["answer_accuracy"] >= 0.97
            and primary["final_assignment_exact"] >= 0.97
            and primary["all_transitions_exact"] >= 0.95
        ),
        "ordered_every_depth_answer_state_chains": all(
            row["answer_accuracy"] >= 0.95
            and row["final_assignment_exact"] >= 0.97
            and row["all_transitions_exact"] >= 0.93
            for row in ordered["by_depth"].values()
        ),
        "mean_overall_answer_state_chains": (
            mean["overall"]["answer_accuracy"] >= 0.90
            and mean["overall"]["final_assignment_exact"] >= 0.90
            and mean["overall"]["all_transitions_exact"] >= 0.82
        ),
        "gold_exact_execution": (
            gold["overall"]["answer_accuracy"] >= 0.98
            and gold["overall"]["final_assignment_exact"] == 1.0
            and gold["overall"]["all_transitions_exact"] == 1.0
            and gold["overall"]["kind_accuracy"] == 1.0
            and gold["overall"]["amount_accuracy"] == 1.0
        ),
        "operation_causality": (
            operation["intervention_rows"] == ordered["rows"]
            and operation["overall"]["answer_accuracy"] <= 0.45
            and operation["overall"]["final_assignment_exact"] <= 0.35
            and primary["answer_accuracy"] - operation["overall"]["answer_accuracy"] >= 0.50
            and primary["final_assignment_exact"] - operation["overall"]["final_assignment_exact"] >= 0.60
        ),
        "query_causality": (
            query["intervention_rows"] == ordered["rows"]
            and query["overall"]["answer_accuracy"] <= 0.05
            and primary["answer_accuracy"] - query["overall"]["answer_accuracy"] >= 0.90
            and abs(primary["final_assignment_exact"] - query["overall"]["final_assignment_exact"]) <= 0.001
        ),
        "receipts_and_hashes": (
            job[2] == "COMPLETED" and job[3] == "0:0"
            and all(row.get("fit_updates") == 0 for row in values.values())
            and all(row.get("old_confirmation_access") == 0 for row in values.values())
            and len({row["board_sha256"] for row in values.values()}) == 1
            and len({row["executor_sha256"] for row in values.values()}) == 1
            and len({row["kind_lexicon_sha256"] for row in values.values()}) == 1
        ),
    }
    candidate = (
        "pointer_anchor_s3_v1_4"
        if args.kind_protocol == "training_lexicon_pointer_anchor_v1"
        else "lexical_closed_s3_v1_3"
    )
    decision = (
        "confirm_{}_execution_through_depth_8".format(candidate)
        if all(gates.values()) else
        "reject_{}_confirmation".format(candidate)
    )
    result = {
        "schema": "r12_s3_lexical_confirmation_assessment_v1",
        "decision": decision,
        "all_gates_pass": all(gates.values()),
        "gates": gates,
        "scores": {name: value["overall"] for name, value in values.items()},
        "input_sha256": {name: sha256_file(path) for name, path in paths.items()},
        "job": {"id": job[0], "name": job[1], "state": job[2], "exit": job[3], "elapsed": job[4], "node": job[5]},
        "fit_updates": 0,
        "old_confirmation_access": 0,
        "claim_boundary": (
            "Confirmed known-atom source-deleted S3 execution with external schedule/halt only; "
            "not unseen-phrase generalization, autonomous planning, or learned halt."
        ),
    }
    Path(args.out).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"decision": decision, "gates": gates}, sort_keys=True))


if __name__ == "__main__":
    main()
