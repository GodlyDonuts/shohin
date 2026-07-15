#!/usr/bin/env python3
"""Fixtures for the frozen R8 curvature gate."""

from evaluate_counterfactual_curvature_r8 import R7_SHA256, evaluate


def report(curvature=60, random_pairs=40, direct=46, shuffled_curvature=25):
    opcodes = {
        "add_0": 6, "add_1": 6, "sub_0": 6, "sub_1": 6, "move_0_1": 6,
        "move_1_0": 6, "merge_0_1": 8, "merge_1_0": 8, "swap": 8,
    }
    return {
        "protocol": "counterfactual_curvature_binding_canary_r8",
        "layers": [5, 11, 17, 23, 29],
        "pair_budget": 2,
        "limit_per_opcode": 12,
        "regimes": ["language_ood", "full_ood"],
        "r7_report_sha256": R7_SHA256,
        "r7_reference": {"active": 32, "direct": 46},
        "selection_balance": {key: 12 for key in opcodes},
        "summary": {
            "curvature": {
                "correct": curvature, "total": 108, "by_opcode_correct": opcodes,
                "by_group_correct": {"numeric": 36, "structural": 24},
            },
            "random_pairs": {"correct": random_pairs, "total": 108},
            "direct": {"correct": direct, "total": 108},
            "shuffled_curvature": {"correct": shuffled_curvature, "total": 108},
        },
    }


def main():
    reasons, metrics = evaluate(report())
    assert not reasons, reasons
    assert all(metrics["checks"].values())
    weak = report(random_pairs=45, shuffled_curvature=40)
    weak["summary"]["curvature"].update({
        "correct": 48,
        "by_opcode_correct": {
            "add_0": 4, "add_1": 4, "sub_0": 4, "sub_1": 4, "move_0_1": 4,
            "move_1_0": 4, "merge_0_1": 8, "merge_1_0": 8, "swap": 8,
        },
        "by_group_correct": {"numeric": 24, "structural": 24},
    })
    reasons, _ = evaluate(weak)
    assert "curvature_accuracy_at_least_50pct" in reasons
    assert "curvature_beats_random_pairs_by_10pp" in reasons
    assert "curvature_beats_shuffled_by_20pp" in reasons
    print("R8 counterfactual curvature gate: passed")


if __name__ == "__main__":
    main()
