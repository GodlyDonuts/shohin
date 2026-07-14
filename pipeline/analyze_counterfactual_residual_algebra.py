#!/usr/bin/env python3
"""Classify a completed CRA arm without converting diagnostics into a score.

This consumes only frozen evaluation reports.  Its role is to distinguish a
primitive that was never learned from one that is template-bound, loses the
counterfactual sign, or bypasses the source-free tape.  The report never
promotes a mechanism and deliberately preserves raw counts beside all rates.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


BEHAVIOR_KEYS = (
    "normal_correct",
    "paraphrase_correct",
    "counterfactual_correct",
    "zero_recreates_normal",
    "shuffle_recreates_normal",
    "strict_causal",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_behavior(path: Path) -> dict:
    report = json.loads(path.read_text())
    rows, summary = int(report.get("rows", 0)), report.get("summary", {})
    if rows <= 0 or any(key not in summary for key in BEHAVIOR_KEYS):
        raise ValueError("{} is not a complete CRA behavior report".format(path))
    counts = {key: int(summary[key]) for key in BEHAVIOR_KEYS}
    if any(value < 0 or value > rows for value in counts.values()):
        raise ValueError("{} has invalid CRA behavior counts".format(path))
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "rows": rows,
        "counts": counts,
        "rates": {key: value / rows for key, value in counts.items()},
    }


def load_nll(path: Path) -> dict:
    report = json.loads(path.read_text())
    rows, summary = int(report.get("rows", 0)), report.get("summary", {})
    needed = ("normal_margin", "counterfactual_margin", "paraphrase_margin", "zero_margin", "shuffle_margin")
    if rows <= 0 or any(key not in summary for key in needed):
        raise ValueError("{} is not a complete CRA NLL report".format(path))
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "rows": rows,
        "margins": {key: float(summary[key]) for key in needed},
        "paired_directional": int(report.get("paired_directional", 0)),
        "strict_directional": int(report.get("strict_directional", 0)),
    }


def classify(combined: dict, train: dict, nll: dict, factors: dict) -> tuple[list[str], str]:
    held, fitted = combined["rates"], train["rates"]
    diagnoses = []
    learned_in_distribution = fitted["strict_causal"] >= 0.20
    directional = nll["margins"]["counterfactual_margin"] > 0 and held["counterfactual_correct"] >= 0.20
    controls_clean = held["zero_recreates_normal"] <= 0.05 and held["shuffle_recreates_normal"] <= 0.05
    generalizes = held["strict_causal"] >= 0.25

    if not learned_in_distribution:
        diagnoses.append("not_learned_in_distribution")
    if held["normal_correct"] >= 0.20 and not directional:
        diagnoses.append("counterfactual_sign_not_functional")
    if held["normal_correct"] >= 0.20 and not controls_clean:
        diagnoses.append("source_free_control_failure")
    if learned_in_distribution and not generalizes:
        diagnoses.append("cross_split_generalization_failure")
    factor_strict = {name: item["rates"]["strict_causal"] for name, item in factors.items()}
    if learned_in_distribution and factor_strict and factor_strict.get("language", 1.0) < min(
        factor_strict.get("values", 1.0), factor_strict.get("delta", 1.0), factor_strict.get("query", 1.0),
    ):
        diagnoses.append("language_chart_binding_suspected")
    if not diagnoses:
        diagnoses.append("mixed_or_partial_signal_requires_row_level_review")

    if not learned_in_distribution:
        recommendation = "close_residual_algebra_at_this_capacity"
    elif not directional and controls_clean:
        recommendation = "paired_sign_discrimination_is_the_only_justified_follow_up"
    elif "language_chart_binding_suspected" in diagnoses and directional and controls_clean:
        recommendation = "counterfactual_chart_closure_is_conditionally_justified"
    else:
        recommendation = "inspect_row_level_failures_before_any_new_arm"
    return diagnoses, recommendation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--combined", required=True)
    parser.add_argument("--nll", required=True)
    parser.add_argument("--train", required=True)
    parser.add_argument("--language", required=True)
    parser.add_argument("--values", required=True)
    parser.add_argument("--delta", required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--two-edit", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = Path(args.out)
    if out.exists():
        raise SystemExit("refusing existing output: {}".format(out))
    combined = load_behavior(Path(args.combined))
    train = load_behavior(Path(args.train))
    nll = load_nll(Path(args.nll))
    factors = {
        name: load_behavior(Path(value))
        for name, value in {
            "language": args.language,
            "values": args.values,
            "delta": args.delta,
            "query": args.query,
            "two_edit": args.two_edit,
        }.items()
    }
    diagnoses, recommendation = classify(combined, train, nll, factors)
    report = {
        "audit": "counterfactual_residual_algebra_failure_taxonomy_v1",
        "claim_boundary": "Diagnostic classification only; no output authorizes a reasoning, workspace, or benchmark claim.",
        "combined": combined,
        "train": train,
        "nll": nll,
        "factors": factors,
        "diagnoses": diagnoses,
        "recommendation": recommendation,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"diagnoses": diagnoses, "recommendation": recommendation}, sort_keys=True))


if __name__ == "__main__":
    main()
