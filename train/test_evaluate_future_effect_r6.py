#!/usr/bin/env python3
"""Tests for the frozen R6 development promotion rule."""

from evaluate_future_effect_r6 import POLICIES, evaluate


def policy(cases, answers, exact, operations=896, operations_correct=0):
    return {
        "cases": cases,
        "answers_correct": answers,
        "exact_programs": exact,
        "operations": operations,
        "operations_correct": operations_correct,
    }


def passing_report():
    regimes = {}
    answers = {"all": 650, "fit": 230, "depth": 140, "language": 180, "full": 100}
    exact = {"all": 570, "fit": 215, "depth": 120, "language": 160, "full": 90}
    for regime, cases in (("all", 896), ("fit", 256), ("depth", 192), ("language", 256), ("full", 192)):
        operations = 896 if regime == "all" else max(cases * 2, 1)
        active_answers = answers[regime]
        active_exact = exact[regime]
        regimes[regime] = {
            "cases": cases,
            "query_correct": cases,
            "train_probe_mse": 0.2,
            "heldout_probe_mse": 0.3,
            "policies": {
                "active": policy(cases, active_answers, active_exact, operations, int(operations * 0.80)),
                "random": policy(cases, max(active_answers - 50, 0), max(active_exact - 50, 0), operations, int(operations * 0.60)),
                "zero": policy(cases, max(active_answers - 100, 0), max(active_exact - 100, 0), operations, int(operations * 0.30)),
                "shuffled": policy(cases, max(active_answers - 90, 0), max(active_exact - 90, 0), operations, int(operations * 0.35)),
                "oracle": policy(cases, cases, cases, operations, operations),
            },
        }
    return {
        "protocol": "active_counterfactual_distinction_eval_r6",
        "latent_steps": 3,
        "hypotheses": 597,
        "policies": list(POLICIES),
        "summary": regimes,
    }


def main():
    report = passing_report()
    reasons, metrics = evaluate(report)
    assert not reasons, reasons
    assert all(metrics["checks"].values())

    report = passing_report()
    for regime in ("language", "full"):
        report["summary"][regime]["policies"]["random"] = (
            report["summary"][regime]["policies"]["active"].copy()
        )
    reasons, _ = evaluate(report)
    assert "active_beats_random_answers_by_5pp" in reasons
    assert "active_beats_random_exact_by_5pp" in reasons
    assert "active_beats_random_operations_by_10pp" in reasons

    report = passing_report()
    report["summary"]["language"]["heldout_probe_mse"] = 9.0
    report["summary"]["full"]["heldout_probe_mse"] = 9.0
    reasons, _ = evaluate(report)
    assert "heldout_probe_mse_calibrated" in reasons
    print("R6 active-distinction development gate: passed")


if __name__ == "__main__":
    main()
