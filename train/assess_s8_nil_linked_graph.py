#!/usr/bin/env python3
"""Apply the frozen S8 development gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def assess(evaluation: dict[str, object]) -> dict[str, object]:
    graph = evaluation["graph"]
    arms = evaluation["arms"]
    treatment = arms["treatment"]
    gold = arms["gold_graph"]
    ordinary = arms["ordinary_sequence_parser"]
    gates = {
        "graph_valid_at_least_95pct": graph["valid_accuracy"] >= 0.95,
        "exact_graph_at_least_90pct": graph["exact_accuracy"] >= 0.90,
        "count_and_nil_at_least_98pct": graph["count_halt_accuracy"] >= 0.98,
        "state_at_least_85pct": treatment["state_accuracy"] >= 0.85,
        "answer_at_least_90pct": treatment["answer_accuracy"] >= 0.90,
        "every_depth_state_at_least_75pct": all(
            cell["accuracy"] >= 0.75
            for cell in treatment["depth_state"].values()
        ),
        "gold_graph_state_exact": gold["state_accuracy"] == 1.0,
        "gold_graph_answer_exact": gold["answer_accuracy"] == 1.0,
        "treatment_within_10pp_of_gold": (
            gold["state_accuracy"] - treatment["state_accuracy"] <= 0.10
        ),
        "treatment_within_3pp_of_ordinary": (
            ordinary["state_accuracy"] - treatment["state_accuracy"] <= 0.03
        ),
        "storage_order_below_40pct": (
            arms["storage_order_shortcut"]["state_accuracy"] < 0.40
        ),
        "reversed_links_drop_at_least_40pp": (
            treatment["state_accuracy"]
            - arms["reversed_links"]["state_accuracy"]
            >= 0.40
        ),
        "card_derangement_drop_at_least_50pp": (
            treatment["state_accuracy"]
            - arms["deranged_cards"]["state_accuracy"]
            >= 0.50
        ),
        "one_witness_drop_at_least_30pp": (
            treatment["state_accuracy"]
            - arms["one_witness"]["state_accuracy"]
            >= 0.30
        ),
        "state_reset_drop_at_least_20pp": (
            treatment["state_accuracy"]
            - arms["state_reset"]["state_accuracy"]
            >= 0.20
        ),
        "early_nil_drop_at_least_30pp": (
            treatment["state_accuracy"]
            - arms["early_nil"]["state_accuracy"]
            >= 0.30
        ),
        "shuffled_exact_graph_below_10pct": (
            graph["shuffled_exact_accuracy"] < 0.10
        ),
        "graph_reindex_bit_identical": (
            evaluation["invariance"]["graph_reindex_eligible"]
            == graph["valid"]
            and evaluation["invariance"]["graph_reindex_accuracy"] == 1.0
        ),
        "operation_nonce_bit_identical": (
            evaluation["invariance"]["operation_nonce_eligible"]
            == graph["valid"]
            and evaluation["invariance"]["operation_nonce_accuracy"] == 1.0
        ),
        "graph_compiler_below_16m": (
            evaluation["parameters"]["graph_compiler"] <= 16_000_000
        ),
        "complete_system_below_150m": (
            evaluation["parameters"]["complete_system"] < 150_000_000
        ),
        "graph_only_training_contract": (
            "zero final-state, answer, recurrent" in evaluation["training_contract"]
        ),
        "one_development_zero_confirmation_access": (
            evaluation["development_accesses"] == 1
            and evaluation["confirmation_accesses"] == 0
        ),
    }
    return {
        "schema": "r12_s8_nil_linked_law_graph_development_assessment_v1",
        "decision": (
            "qualify_s8_nil_linked_law_graph_for_fresh_confirmation"
            if all(gates.values())
            else "reject_s8_nil_linked_law_graph_v1"
        ),
        "gates": gates,
        "scores": {
            "graph": graph,
            "arms": arms,
            "invariance": evaluation["invariance"],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing existing S8 assessment: {args.out}")
    evaluation = json.loads(args.evaluation.read_text())
    if evaluation.get("schema") != "r12_s8_nil_linked_law_graph_development_evaluation_v1":
        raise SystemExit("unexpected S8 evaluation schema")
    result = assess(evaluation)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"decision": result["decision"], "out": str(args.out)}, sort_keys=True))


if __name__ == "__main__":
    main()
