#!/usr/bin/env python3
"""Fixtures for the frozen R7 ISQ canary gate."""

from evaluate_counterfactual_influence_r7 import evaluate


def report(active=60, random=35, direct=50, shuffled=20):
    opcodes = {
        "add_0": 7, "add_1": 7, "sub_0": 7, "sub_1": 7, "move_0_1": 7,
        "move_1_0": 7, "merge_0_1": 6, "merge_1_0": 6, "swap": 6,
    }
    return {
        "protocol": "interventional_semantic_quotient_canary_r7",
        "layers": [5, 11, 17, 23, 29],
        "intervention_budget": 2,
        "limit_per_opcode": 12,
        "regimes": ["language_ood", "full_ood"],
        "selection_balance": {key: 12 for key in opcodes},
        "summary": {
            "active": {"correct": active, "total": 108, "by_opcode_correct": opcodes},
            "random": {"correct": random, "total": 108, "by_opcode_correct": {}},
            "direct": {"correct": direct, "total": 108, "by_opcode_correct": {}},
            "shuffled": {"correct": shuffled, "total": 108, "by_opcode_correct": {}},
        },
    }


def main():
    reasons, metrics = evaluate(report())
    assert not reasons, reasons
    assert all(metrics["checks"].values())
    reasons, _ = evaluate(report(active=45, random=44, direct=43, shuffled=42))
    assert "active_accuracy_at_least_45pct" in reasons
    assert "active_beats_random_by_5pp" in reasons
    assert "active_beats_direct_by_5pp" in reasons
    assert "active_beats_shuffled_by_15pp" in reasons
    print("R7 interventional semantic quotient gate: passed")


if __name__ == "__main__":
    main()
