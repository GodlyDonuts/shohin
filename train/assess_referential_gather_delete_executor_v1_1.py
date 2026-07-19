#!/usr/bin/env python3
"""Apply the frozen RGDE v1.1 development gates to completed artifacts."""

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


def score(result, name):
    return float(result["overall"][name])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--carrier", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    root = Path(args.results_dir)
    out = Path(args.out)
    if out.exists():
        raise SystemExit("refusing existing assessment")
    paths = {
        "tied": root / "tied/development_predicted.json",
        "gold_rescore": root / "tied/development_gold_packet.json",
        "operation_derangement": (
            root / "tied/development_operation_semantic_derangement.json"
        ),
        "query_derangement": root / "tied/development_query_semantic_derangement.json",
        "untied": root / "untied/development_predicted.json",
        "tied_gold": root / "tied_gold/development_gold_packet.json",
        "tied_composed": root / "tied_composed/development_predicted.json",
        "carrier": Path(args.carrier),
        "sacct": root / "sacct.txt",
    }
    if any(not path.is_file() for path in paths.values()):
        missing = [str(path) for path in paths.values() if not path.is_file()]
        raise SystemExit("missing assessment inputs: {}".format(missing))
    tied = load(paths["tied"])
    gold_rescore = load(paths["gold_rescore"])
    operation = load(paths["operation_derangement"])
    query = load(paths["query_derangement"])
    untied = load(paths["untied"])
    tied_gold = load(paths["tied_gold"])
    composed = load(paths["tied_composed"])
    carrier = load(paths["carrier"])
    all_results = (tied, gold_rescore, operation, query, untied, tied_gold, composed)
    parent_jobs = {}
    for line in paths["sacct"].read_text().splitlines():
        fields = line.split("|")
        if len(fields) == 6 and fields[0].isdigit():
            parent_jobs[fields[0]] = {
                "name": fields[1], "state": fields[2], "exit_code": fields[3],
                "elapsed": fields[4], "node": fields[5],
            }
    jobs_clean = set(parent_jobs) == {"693118", "693119", "693120", "693121", "693122"}
    jobs_clean = jobs_clean and all(
        row["state"] == "COMPLETED" and row["exit_code"] == "0:0"
        for row in parent_jobs.values()
    )
    surfaces = tied["by_surface"]
    tied_answers = score(tied, "answer_accuracy")
    tied_final = score(tied, "final_assignment_exact")
    gates = {
        "g1_identity_carrier_at_least_99pct": {
            "value": carrier["methods"]["lexical_sigmoid_span"]["accuracy"],
            "pass": carrier["methods"]["lexical_sigmoid_span"]["accuracy"] >= 0.99,
        },
        "g2_composed_supervision_ceiling": {
            "answer": score(composed, "answer_accuracy"),
            "final": score(composed, "final_assignment_exact"),
            "transitions": score(composed, "all_transitions_exact"),
            "pass": (
                score(composed, "answer_accuracy") >= 0.99
                and score(composed, "final_assignment_exact") >= 0.98
                and score(composed, "all_transitions_exact") >= 0.98
            ),
        },
        "g3_gold_atomic": {
            "answer": score(tied_gold, "answer_accuracy"),
            "final": score(tied_gold, "final_assignment_exact"),
            "transitions": score(tied_gold, "all_transitions_exact"),
            "query": score(tied_gold, "query_accuracy"),
            "pass": (
                score(tied_gold, "answer_accuracy") >= 0.98
                and score(tied_gold, "final_assignment_exact") >= 0.98
                and score(tied_gold, "all_transitions_exact") >= 0.95
                and score(tied_gold, "query_accuracy") >= 0.99
            ),
        },
        "g4_tied_predicted_atomic": {
            "answer": tied_answers,
            "final": tied_final,
            "transitions": score(tied, "all_transitions_exact"),
            "query": score(tied, "query_accuracy"),
            "pass": (
                tied_answers >= 0.95
                and tied_final >= 0.95
                and score(tied, "all_transitions_exact") >= 0.90
                and score(tied, "query_accuracy") >= 0.99
            ),
        },
        "g5_surfaces_and_quartets": {
            "minimum_surface_answer": min(
                float(row["answer_accuracy"]) for row in surfaces.values()
            ),
            "all_four_answers": tied["group_summary"]["all_four_answers_correct"],
            "pass": (
                min(float(row["answer_accuracy"]) for row in surfaces.values()) >= 0.90
                and tied["group_summary"]["all_four_answers_correct"] >= 450
            ),
        },
        "g6_gold_rescore_gap": {
            "answer_gap": score(gold_rescore, "answer_accuracy") - tied_answers,
            "final_gap": score(gold_rescore, "final_assignment_exact") - tied_final,
            "pass": (
                score(gold_rescore, "answer_accuracy") - tied_answers <= 0.02
                and score(gold_rescore, "final_assignment_exact") - tied_final <= 0.02
            ),
        },
        "g7_operation_causality": {
            "answer": score(operation, "answer_accuracy"),
            "final": score(operation, "final_assignment_exact"),
            "answer_drop": tied_answers - score(operation, "answer_accuracy"),
            "final_drop": tied_final - score(operation, "final_assignment_exact"),
            "pass": (
                score(operation, "answer_accuracy") <= 0.40
                and score(operation, "final_assignment_exact") <= 0.40
                and tied_answers - score(operation, "answer_accuracy") >= 0.50
                and tied_final - score(operation, "final_assignment_exact") >= 0.50
            ),
        },
        "g8_query_causality": {
            "answer": score(query, "answer_accuracy"),
            "answer_drop": tied_answers - score(query, "answer_accuracy"),
            "pass": (
                score(query, "answer_accuracy") <= 0.45
                and tied_answers - score(query, "answer_accuracy") >= 0.45
            ),
        },
        "g9_tied_vs_untied": {
            "answer_advantage": tied_answers - score(untied, "answer_accuracy"),
            "final_advantage": tied_final - score(untied, "final_assignment_exact"),
            "pass": (
                abs(tied_answers - score(untied, "answer_accuracy")) <= 0.02
                and abs(tied_final - score(untied, "final_assignment_exact")) <= 0.02
            ),
        },
        "g10_custody_and_interventions": {
            "jobs_clean": jobs_clean,
            "operation_interventions": operation["intervention_rows"],
            "query_interventions": query["intervention_rows"],
            "confirmation_access": [row["confirmation_access"] for row in all_results]
            + [carrier["confirmation_access"]],
            "pass": (
                jobs_clean
                and operation["intervention_rows"] == operation["overall"]["rows"]
                and query["intervention_rows"] == query["overall"]["rows"]
                and all(row["confirmation_access"] == 0 for row in all_results)
                and carrier["confirmation_access"] == 0
            ),
        },
    }
    all_pass = all(row["pass"] for row in gates.values())
    result = {
        "schema": "r12_referential_gather_delete_executor_v1_1_assessment",
        "decision": (
            "qualify_rgde_v1_1_for_fresh_depth_confirmation"
            if all_pass else
            "reject_rgde_v1_1_development"
        ),
        "all_gates_pass": all_pass,
        "gates": gates,
        "jobs": parent_jobs,
        "input_sha256": {name: sha256_file(path) for name, path in paths.items()},
        "claim_boundary": (
            "Development-only source-deleted list executor. Qualification permits a fresh "
            "longer-depth board, not a broad language-reasoning or novelty claim."
        ),
    }
    canonical = json.dumps(result, sort_keys=True, separators=(",", ":")).encode()
    result["assessment_sha256"] = hashlib.sha256(canonical).hexdigest()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "out": str(out.resolve()),
        "decision": result["decision"],
        "assessment_sha256": result["assessment_sha256"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
