#!/usr/bin/env python3
"""Freeze the development decision gate for active counterfactual distinction."""

import argparse
import hashlib
import json
import math
from pathlib import Path


POLICIES = ("active", "random", "zero", "shuffled", "oracle")
RAW_R5_ANSWERS = 196
RAW_R5_EXACT = 174


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fraction(value, total):
    return value / total if total else 0.0


def combine_regimes(summary, names):
    combined = {"cases": sum(summary[name]["cases"] for name in names), "policies": {}}
    operation_total = sum(
        summary[name]["policies"]["active"]["operations"] for name in names
    )
    for policy in POLICIES:
        combined["policies"][policy] = {
            key: sum(summary[name]["policies"][policy][key] for name in names)
            for key in ("cases", "answers_correct", "exact_programs", "operations_correct", "operations")
        }
    combined["query_correct"] = sum(summary[name]["query_correct"] for name in names)
    combined["train_probe_mse"] = sum(
        summary[name]["train_probe_mse"]
        * summary[name]["policies"]["active"]["operations"]
        for name in names
    ) / operation_total
    combined["heldout_probe_mse"] = sum(
        summary[name]["heldout_probe_mse"]
        * summary[name]["policies"]["active"]["operations"]
        for name in names
    ) / operation_total
    return combined


def evaluate(report):
    reasons = []
    if report.get("protocol") != "active_counterfactual_distinction_eval_r6":
        reasons.append("wrong_protocol")
    if report.get("latent_steps") != 3:
        reasons.append("wrong_latent_step_budget")
    if report.get("hypotheses") != 597:
        reasons.append("wrong_hypothesis_count")
    if tuple(report.get("policies", ())) != POLICIES:
        reasons.append("wrong_policy_set_or_order")
    summary = report.get("summary", {})
    for regime in ("all", "fit", "depth", "language", "full"):
        if regime not in summary:
            reasons.append("missing_regime_{}".format(regime))
    if reasons:
        return reasons, {}

    all_summary = summary["all"]
    if all_summary.get("cases") != 896:
        reasons.append("wrong_case_count")
    fresh = combine_regimes(summary, ("language", "full"))
    if fresh["cases"] != 448:
        reasons.append("wrong_fresh_case_count")
    metrics = {}
    for policy in POLICIES:
        item = fresh["policies"][policy]
        metrics[policy] = {
            "answer_accuracy": fraction(item["answers_correct"], item["cases"]),
            "exact_program_accuracy": fraction(item["exact_programs"], item["cases"]),
            "operation_accuracy": fraction(item["operations_correct"], item["operations"]),
        }
    active = metrics["active"]
    random = metrics["random"]
    control_answer = max(metrics["zero"]["answer_accuracy"], metrics["shuffled"]["answer_accuracy"])
    control_exact = max(
        metrics["zero"]["exact_program_accuracy"],
        metrics["shuffled"]["exact_program_accuracy"],
    )
    language = summary["language"]["policies"]["active"]
    full = summary["full"]["policies"]["active"]
    query_accuracy = fraction(fresh["query_correct"], fresh["cases"])
    train_mse = float(fresh["train_probe_mse"])
    heldout_mse = float(fresh["heldout_probe_mse"])
    metrics["derived"] = {
        "active_over_random_answers": active["answer_accuracy"] - random["answer_accuracy"],
        "active_over_random_exact": active["exact_program_accuracy"] - random["exact_program_accuracy"],
        "active_over_random_operations": active["operation_accuracy"] - random["operation_accuracy"],
        "active_over_best_control_answers": active["answer_accuracy"] - control_answer,
        "active_over_best_control_exact": active["exact_program_accuracy"] - control_exact,
        "language_answer_accuracy": fraction(language["answers_correct"], language["cases"]),
        "full_answer_accuracy": fraction(full["answers_correct"], full["cases"]),
        "fit_answer_accuracy": fraction(
            summary["fit"]["policies"]["active"]["answers_correct"], summary["fit"]["cases"],
        ),
        "depth_answer_accuracy": fraction(
            summary["depth"]["policies"]["active"]["answers_correct"], summary["depth"]["cases"],
        ),
        "query_accuracy": query_accuracy,
        "train_probe_mse": train_mse,
        "heldout_probe_mse": heldout_mse,
    }

    checks = {
        "active_all_answers_at_least_55pct": active["answer_accuracy"] >= 0.55,
        "active_all_exact_at_least_50pct": active["exact_program_accuracy"] >= 0.50,
        "active_operations_at_least_65pct": active["operation_accuracy"] >= 0.65,
        "active_language_answers_at_least_60pct": metrics["derived"]["language_answer_accuracy"] >= 0.60,
        "active_full_answers_at_least_40pct": metrics["derived"]["full_answer_accuracy"] >= 0.40,
        "active_fit_answers_at_least_80pct": metrics["derived"]["fit_answer_accuracy"] >= 0.80,
        "active_depth_answers_at_least_60pct": metrics["derived"]["depth_answer_accuracy"] >= 0.60,
        "active_beats_raw_r5_answers_by_10pp": fresh["policies"]["active"]["answers_correct"] >= RAW_R5_ANSWERS + 45,
        "active_beats_raw_r5_exact_by_10pp": fresh["policies"]["active"]["exact_programs"] >= RAW_R5_EXACT + 45,
        "active_beats_random_answers_by_5pp": metrics["derived"]["active_over_random_answers"] >= 0.05,
        "active_beats_random_exact_by_5pp": metrics["derived"]["active_over_random_exact"] >= 0.05,
        "active_beats_random_operations_by_10pp": metrics["derived"]["active_over_random_operations"] >= 0.10,
        "active_beats_controls_answers_by_10pp": metrics["derived"]["active_over_best_control_answers"] >= 0.10,
        "active_beats_controls_exact_by_10pp": metrics["derived"]["active_over_best_control_exact"] >= 0.10,
        "oracle_answers_at_least_80pct": metrics["oracle"]["answer_accuracy"] >= 0.80,
        "oracle_exact_at_least_80pct": metrics["oracle"]["exact_program_accuracy"] >= 0.80,
        "query_accuracy_at_least_95pct": query_accuracy >= 0.95,
        "heldout_probe_mse_calibrated": (
            math.isfinite(train_mse)
            and math.isfinite(heldout_mse)
            and heldout_mse <= max(2.0 * train_mse, 1.0)
        ),
    }
    reasons.extend(name for name, passed in checks.items() if not passed)
    metrics["checks"] = checks
    return reasons, metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if Path(args.out).exists():
        raise SystemExit("refusing existing output")
    report = json.load(open(args.report))
    reasons, metrics = evaluate(report)
    decision = {
        "audit": "active_counterfactual_distinction_development_gate_r6",
        "report": str(Path(args.report).resolve()),
        "report_sha256": sha256_file(args.report),
        "advance_to_fresh_board": not reasons,
        "decision": "authorize_fresh_r6_board" if not reasons else "reject_r6_before_fresh_board",
        "reasons": reasons,
        "metrics": metrics,
        "claim_boundary": (
            "A pass authorizes generation of one untouched R6 board only. It does not establish "
            "reasoning, transfer, source-dropped context scaling, or broad capability."
        ),
    }
    Path(args.out).write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n")
    print(json.dumps(decision, sort_keys=True))


if __name__ == "__main__":
    main()
