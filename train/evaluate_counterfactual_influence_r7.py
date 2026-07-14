#!/usr/bin/env python3
"""Frozen advancement rule for the 108-event R7 ISQ development canary."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


POLICIES = ("active", "random", "direct", "shuffled")


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evaluate(report):
    if report.get("protocol") != "interventional_semantic_quotient_canary_r7":
        raise ValueError("invalid R7 canary protocol")
    if report.get("layers") != [5, 11, 17, 23, 29]:
        raise ValueError("R7 layer contract changed")
    if report.get("intervention_budget") != 2 or report.get("limit_per_opcode") != 12:
        raise ValueError("R7 canary budget changed")
    if report.get("regimes") != ["language_ood", "full_ood"]:
        raise ValueError("R7 canary regimes changed")
    summary = report["summary"]
    if set(summary) != set(POLICIES):
        raise ValueError("R7 policy contract changed")
    total = summary["active"]["total"]
    if total != 108 or any(summary[policy]["total"] != total for policy in POLICIES):
        raise ValueError("R7 canary must contain exactly 108 events")
    accuracy = {policy: summary[policy]["correct"] / total for policy in POLICIES}
    by_opcode = summary["active"]["by_opcode_correct"]
    opcode_floor = sum(by_opcode.get(opcode, 0) >= 4 for opcode in report["selection_balance"])
    checks = {
        "active_accuracy_at_least_45pct": accuracy["active"] >= 0.45,
        "active_beats_random_by_5pp": accuracy["active"] - accuracy["random"] >= 0.05,
        "active_beats_direct_by_5pp": accuracy["active"] - accuracy["direct"] >= 0.05,
        "active_beats_shuffled_by_15pp": accuracy["active"] - accuracy["shuffled"] >= 0.15,
        "at_least_seven_opcodes_reach_4_of_12": opcode_floor >= 7,
    }
    reasons = [name for name, passed in checks.items() if not passed]
    return reasons, {
        "accuracy": accuracy,
        "active_opcode_families_at_floor": opcode_floor,
        "checks": checks,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if Path(args.out).exists():
        raise SystemExit("refusing existing R7 decision output")
    report = json.load(open(args.report))
    reasons, metrics = evaluate(report)
    decision = {
        "audit": "interventional_semantic_quotient_development_gate_r7",
        "report": str(Path(args.report).resolve()),
        "report_sha256": sha256_file(args.report),
        "advance_to_full_old_board": not reasons,
        "decision": "advance_r7_to_full_old_board" if not reasons else "reject_r7_isq_canary",
        "reasons": reasons,
        "metrics": metrics,
        "claim_boundary": (
            "A pass authorizes only a full evaluation on the already-used R5 board. It does not "
            "authorize fresh data, training, reasoning, or source deletion."
        ),
    }
    Path(args.out).write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n")
    print(json.dumps(decision, sort_keys=True))


if __name__ == "__main__":
    main()
