#!/usr/bin/env python3
"""Apply frozen S9.2 global-anchor development gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from s9_occurrence_quotient_compiler import sha256_file


CLOSED_S9_EXACT = 1941 / 2048
FROZEN_ROWS = 2048
MINIMUM_EXACT = 2031
FIT_ARMS = ("treatment", "positive_only", "no_class", "shuffled", "layout")


def _is_lower_hex(value, width):
    return (
        isinstance(value, str)
        and len(value) == width
        and all(character in "0123456789abcdef" for character in value)
    )


def _frozen_contract(evaluation):
    expected_architecture = {
        "compiler_class": "OccurrenceQuotientCompiler",
        "layer": 19,
        "width": 384,
        "heads": 8,
        "encoder_layers": 5,
        "ff": 1408,
        "max_span_width": 4,
        "negative_candidates_per_view": 128,
        "hard_negative_top_k": 8,
        "added_trainable_parameters": 0,
    }
    contract = evaluation["training_contract"]
    if evaluation.get("architecture") != expected_architecture:
        return False
    if any(
        contract.get(key) != value
        for key, value in {
            "unique_sources_per_arm": 24_000,
            "charged_views_per_arm": 48_000,
            "batch_size": 64,
            "pair_batch_size": 32,
            "updates_per_arm": 750,
            "negative_candidates_per_view": 128,
            "hard_negative_top_k": 8,
            "orbit_weight": 0.25,
            "learning_rate": 1e-3,
            "warmup_updates": 50,
            "gradient_clip": 1.0,
        }.items()
    ):
        return False
    expected_arms = {
        "treatment": ("full", True, False, 8),
        "positive_only": ("positive", True, False, None),
        "no_class": ("full", False, False, 8),
        "shuffled": ("full", True, False, 8),
        "layout": ("full", False, True, 8),
    }
    for name, (mode, classes, masked, top_k) in expected_arms.items():
        fit = evaluation["fit"].get(name, {})
        if any(
            (
                fit.get("unique_sources") != 24_000,
                fit.get("charged_views") != 48_000,
                fit.get("batch_size") != 64,
                fit.get("updates") != 750,
                fit.get("negative_candidates_per_view") != 128,
                fit.get("orbit_mode") != mode,
                fit.get("class_messages") is not classes,
                fit.get("mask_gold_tokens") is not masked,
                fit.get("hard_negative_top_k") != top_k,
                fit.get("orbit_weight") != 0.25,
                fit.get("learning_rate") != 1e-3,
                fit.get("warmup_updates") != 50,
                fit.get("gradient_clip") != 1.0,
            )
        ):
            return False
    return True


def _inherited_gates(evaluation):
    rows = int(evaluation["rows"])
    graph = evaluation["graph"]
    arms = evaluation["arms"]
    span = evaluation["span"]
    invariance = evaluation["invariance"]
    treatment = arms["treatment"]
    return {
        "span_f1_at_least_98pct": span["f1"] >= 0.98,
        "class_membership_exact_at_least_95pct": (span["class_exact_accuracy"] >= 0.95),
        "complete_relation_tuple_at_least_90pct": graph["exact_accuracy"] >= 0.90,
        "valid_graph_at_least_90pct": graph["valid_accuracy"] >= 0.90,
        "exact_graph_at_least_85pct": graph["exact_accuracy"] >= 0.85,
        "structured_exact_graph_at_least_95pct": graph["exact_accuracy"] >= 0.95,
        "structured_exact_graph_no_regression_from_s9": (
            graph["exact_accuracy"] >= CLOSED_S9_EXACT
        ),
        "state_at_least_80pct": treatment["state_accuracy"] >= 0.80,
        "answer_at_least_85pct": treatment["answer_accuracy"] >= 0.85,
        "every_depth_state_at_least_70pct": min(
            value["accuracy"] for value in treatment["depth"].values()
        )
        >= 0.70,
        "exact_graph_plus_20pp_over_s8_1": graph["exact_accuracy"] >= 0.4509765625,
        "exact_graph_plus_5pp_over_no_class": (
            graph["exact_accuracy"] - graph["no_class_exact_accuracy"] >= 0.05
        ),
        "shuffled_exact_below_10pct": graph["shuffled_exact_accuracy"] < 0.10,
        "uniform_exact_zero": graph["uniform_exact"] == 0,
        "source_free_exact_below_10pct": graph["source_free_exact_accuracy"] < 0.10,
        "unconstrained_ablation_reported": "unconstrained_exact_accuracy" in graph,
        "class_reindex_bit_identical": (
            invariance["eligible"] > 0
            and invariance["class_reindex"] == invariance["eligible"]
        ),
        "relation_storage_reindex_bit_identical": (
            invariance["eligible"] > 0
            and invariance["relation_storage_reindex"] == invariance["eligible"]
        ),
        "operation_nonce_all_valid_eligible": (
            invariance["nonce_eligible"] == graph["valid"]
        ),
        "operation_nonce_graph_bit_identical": (
            invariance["nonce_eligible"] > 0
            and invariance["nonce_graph_identical"] == invariance["nonce_eligible"]
        ),
        "operation_nonce_state_bit_identical": (
            invariance["nonce_state_identical"] == invariance["nonce_eligible"]
        ),
        "operation_nonce_answer_bit_identical": (
            invariance["nonce_answer_identical"] == invariance["nonce_eligible"]
        ),
        "reversed_links_drop_at_least_40pp": (
            treatment["state_accuracy"] - arms["reversed_links"]["state_accuracy"]
            >= 0.40
        ),
        "swapped_cards_drop_at_least_50pp": (
            treatment["state_accuracy"] - arms["deranged_cards"]["state_accuracy"]
            >= 0.50
        ),
        "one_witness_drop_at_least_30pp": (
            treatment["state_accuracy"] - arms["one_witness"]["state_accuracy"] >= 0.30
        ),
        "state_reset_drop_at_least_20pp": (
            treatment["state_accuracy"] - arms["state_reset"]["state_accuracy"] >= 0.20
        ),
        "early_nil_drop_at_least_30pp": (
            treatment["state_accuracy"] - arms["early_nil"]["state_accuracy"] >= 0.30
        ),
        "complete_system_below_150m": (
            evaluation["parameters"]["complete_system"] < 150_000_000
        ),
        "equal_budget_48k_views_750_updates": all(
            evaluation["fit"][name]["charged_views"] == 48_000
            and evaluation["fit"][name]["updates"] == 750
            for name in ("treatment", "no_class", "shuffled")
        ),
        "one_development_zero_confirmation_access": (
            evaluation["development_accesses"] == 1
            and evaluation["confirmation_accesses"] == 0
            and _is_lower_hex(evaluation.get("access_ledger_sha256"), 64)
            and _is_lower_hex(evaluation.get("base_sha256"), 64)
            and _is_lower_hex(evaluation.get("tokenizer_sha256"), 64)
            and _is_lower_hex(evaluation.get("source_commit"), 40)
        ),
        "all_rows_scored": treatment["total"] == rows == FROZEN_ROWS,
    }


def _s9_2_gates(evaluation):
    graph = evaluation["graph"]
    treatment = evaluation["arms"]["treatment"]
    root = evaluation["root"]["treatment"]
    invariance = evaluation["invariance"]
    return {
        "exact_graph_state_answer_at_least_2031": (
            graph["exact"] >= MINIMUM_EXACT
            and treatment["state"] >= MINIMUM_EXACT
            and treatment["answer"] >= MINIMUM_EXACT
        ),
        "every_valid_emitted_graph_exact": graph["valid"] == graph["exact"],
        "global_strictly_beats_same_logit_local_root": (
            graph["exact"] > graph["local_root_exact"]
        ),
        "root_spans_at_least_99pct_exact": root["span_exact_accuracy"] >= 0.99,
        "root_counts_at_least_99pct_exact": root["count_exact_accuracy"] >= 0.99,
        "positive_only_arm_reported": (
            "positive_only_exact_accuracy" in graph
            and "positive_only" in evaluation["arms"]
            and "positive_only" in evaluation["fit"]
        ),
        "layout_exact_below_10pct": graph["layout_exact_accuracy"] < 0.10,
        "operation_nonce_has_zero_graph_failures": (
            invariance["nonce_eligible"] == graph["valid"]
            and invariance["nonce_graph_identical"] == graph["valid"]
            and invariance["nonce_state_identical"] == graph["valid"]
            and invariance["nonce_answer_identical"] == graph["valid"]
        ),
        "operation_nonce_root_decisions_identical": (
            invariance["nonce_root_eligible"] == graph["valid"]
            and invariance["nonce_root_identical"] == graph["valid"]
        ),
        "operation_nonce_counts_identical": (
            invariance["nonce_root_eligible"] == graph["valid"]
            and invariance["nonce_count_identical"] == graph["valid"]
        ),
        "equal_five_arm_budget": _frozen_contract(evaluation),
        "complete_system_exactly_134580264": (
            evaluation["parameters"]["complete_system"] == 134_580_264
        ),
    }


def assess(evaluation):
    inherited = _inherited_gates(evaluation)
    if len(inherited) != 31:
        raise AssertionError("S9.2 must preserve exactly 31 inherited gates")
    added = _s9_2_gates(evaluation)
    gates = {**inherited, **added}
    return {
        "schema": "r12_s9_2_global_anchor_development_assessment_v1",
        "decision": (
            "qualify_s9_2_global_anchor_for_fresh_confirmation"
            if all(gates.values())
            else "reject_s9_2_global_anchor_v1"
        ),
        "gate_summary": {
            "inherited_passed": sum(inherited.values()),
            "inherited_total": len(inherited),
            "s9_2_passed": sum(added.values()),
            "s9_2_total": len(added),
            "passed": sum(gates.values()),
            "total": len(gates),
        },
        "inherited_gates": inherited,
        "s9_2_gates": added,
        "gates": gates,
        "scores": {
            "span": evaluation["span"],
            "root": evaluation["root"],
            "graph": evaluation["graph"],
            "treatment": evaluation["arms"]["treatment"],
            "positive_only": evaluation["arms"]["positive_only"],
            "no_class": evaluation["arms"]["no_class_message"],
            "layout": evaluation["arms"]["layout"],
            "invariance": evaluation["invariance"],
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing existing S9.2 assessment: {args.out}")
    evaluation = json.loads(args.evaluation.read_text())
    expected = "r12_s9_2_global_anchor_development_evaluation_v1"
    if evaluation.get("schema") != expected:
        raise SystemExit("unexpected S9.2 evaluation schema")
    result = assess(evaluation)
    result["evaluation_sha256"] = sha256_file(args.evaluation)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "decision": result["decision"],
                "gate_summary": result["gate_summary"],
                "out": str(args.out),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
