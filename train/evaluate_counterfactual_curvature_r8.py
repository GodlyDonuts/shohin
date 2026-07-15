#!/usr/bin/env python3
"""Frozen advancement rule for R8 counterfactual curvature binding."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


POLICIES = ("curvature", "random_pairs", "direct", "shuffled_curvature")
R7_SHA256 = "2531c6f5b0166feab75a02ac4061fb96e1f773e072c4a690a72436d8a106cfbd"


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evaluate(report):
    if report.get("protocol") != "counterfactual_curvature_binding_canary_r8":
        raise ValueError("invalid R8 protocol")
    if report.get("layers") != [5, 11, 17, 23, 29]:
        raise ValueError("R8 layer contract changed")
    if report.get("pair_budget") != 2 or report.get("limit_per_opcode") != 12:
        raise ValueError("R8 budget changed")
    if report.get("regimes") != ["language_ood", "full_ood"]:
        raise ValueError("R8 regimes changed")
    if report.get("r7_report_sha256") != R7_SHA256 or report.get("r7_reference") != {"active": 32, "direct": 46}:
        raise ValueError("R8 R7 reference changed")
    summary = report["summary"]
    if set(summary) != set(POLICIES):
        raise ValueError("R8 policy contract changed")
    total = summary["curvature"]["total"]
    if total != 108 or any(summary[policy]["total"] != total for policy in POLICIES):
        raise ValueError("R8 canary must contain exactly 108 events")
    accuracy = {policy: summary[policy]["correct"] / total for policy in POLICIES}
    opcodes = summary["curvature"]["by_opcode_correct"]
    groups = summary["curvature"]["by_group_correct"]
    if sum(opcodes.values()) != summary["curvature"]["correct"]:
        raise ValueError("R8 opcode summary is inconsistent")
    if sum(groups.values()) != summary["curvature"]["correct"]:
        raise ValueError("R8 group summary is inconsistent")
    opcode_floor = sum(opcodes.get(opcode, 0) >= 5 for opcode in report["selection_balance"])
    numeric_accuracy = groups.get("numeric", 0) / 72
    structural_accuracy = groups.get("structural", 0) / 36
    checks = {
        "curvature_accuracy_at_least_50pct": accuracy["curvature"] >= 0.50,
        "curvature_beats_frozen_r7_direct_by_7pp": accuracy["curvature"] - (46 / 108) >= 0.07,
        "curvature_beats_frozen_r7_active_by_15pp": accuracy["curvature"] - (32 / 108) >= 0.15,
        "curvature_beats_random_pairs_by_10pp": accuracy["curvature"] - accuracy["random_pairs"] >= 0.10,
        "curvature_beats_shuffled_by_20pp": accuracy["curvature"] - accuracy["shuffled_curvature"] >= 0.20,
        "numeric_accuracy_at_least_40pct": numeric_accuracy >= 0.40,
        "structural_accuracy_at_least_65pct": structural_accuracy >= 0.65,
        "at_least_seven_opcodes_reach_5_of_12": opcode_floor >= 7,
    }
    reasons = [name for name, passed in checks.items() if not passed]
    return reasons, {
        "accuracy": accuracy,
        "numeric_accuracy": numeric_accuracy,
        "structural_accuracy": structural_accuracy,
        "curvature_opcode_families_at_floor": opcode_floor,
        "checks": checks,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if Path(args.out).exists():
        raise SystemExit("refusing existing R8 decision output")
    report = json.load(open(args.report))
    reasons, metrics = evaluate(report)
    decision = {
        "audit": "counterfactual_curvature_binding_development_gate_r8",
        "report": str(Path(args.report).resolve()),
        "report_sha256": sha256_file(args.report),
        "advance_to_untouched_confirmation": not reasons,
        "decision": "advance_r8_to_untouched_confirmation" if not reasons else "reject_r8_curvature_canary",
        "reasons": reasons,
        "metrics": metrics,
        "claim_boundary": (
            "A pass authorizes one untouched confirmation board only. It does not authorize "
            "training, execution, reasoning, source deletion, or context scaling."
        ),
    }
    Path(args.out).write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n")
    print(json.dumps(decision, sort_keys=True))


if __name__ == "__main__":
    main()
