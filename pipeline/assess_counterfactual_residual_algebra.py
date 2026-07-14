#!/usr/bin/env python3
"""Aggregate pre-registered CRA evidence without changing any model artifact."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


BEHAVIORAL_GATES = {
    "strict_causal": 300,
    "normal_correct": 350,
    "paraphrase_correct": 350,
    "counterfactual_correct": 350,
    "zero_recreates_normal_max": 25,
    "shuffle_recreates_normal_max": 25,
}


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_report(label, path):
    report = json.load(open(path))
    if not isinstance(report, dict) or not isinstance(report.get("summary"), dict):
        raise SystemExit("{} is not a report with a summary: {}".format(label, path))
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "rows": report.get("rows"),
        "summary": report["summary"],
        "strict_causal": report["summary"].get("strict_causal"),
        "claim_boundary": report.get("claim_boundary"),
    }


def check_combined(summary):
    checks = {
        "strict_causal": summary.get("strict_causal", 0) >= BEHAVIORAL_GATES["strict_causal"],
        "normal_correct": summary.get("normal_correct", 0) >= BEHAVIORAL_GATES["normal_correct"],
        "paraphrase_correct": summary.get("paraphrase_correct", 0) >= BEHAVIORAL_GATES["paraphrase_correct"],
        "counterfactual_correct": summary.get("counterfactual_correct", 0) >= BEHAVIORAL_GATES["counterfactual_correct"],
        "zero_recreates_normal": summary.get("zero_recreates_normal", 10 ** 9) <= BEHAVIORAL_GATES["zero_recreates_normal_max"],
        "shuffle_recreates_normal": summary.get("shuffle_recreates_normal", 10 ** 9) <= BEHAVIORAL_GATES["shuffle_recreates_normal_max"],
    }
    return checks, all(checks.values())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("raw", "raw_nll", "combined", "combined_nll", "train", "language", "values", "delta", "query", "two_edit"):
        parser.add_argument("--" + name.replace("_", "-"), required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = Path(args.out)
    if out.exists():
        raise SystemExit("refusing existing output: {}".format(out))
    reports = {}
    for name in ("raw", "raw_nll", "combined", "combined_nll", "train", "language", "values", "delta", "query", "two_edit"):
        reports[name] = load_report(name, getattr(args, name))
    combined_checks, combined_pass = check_combined(reports["combined"]["summary"])
    report = {
        "audit": "counterfactual_residual_algebra_gate_v1",
        "claim_boundary": "This artifact aggregates evidence; only behavioral factor results can support a narrow CRA primitive claim.",
        "behavioral_gates": BEHAVIORAL_GATES,
        "combined_checks": combined_checks,
        "combined_gate_pass": combined_pass,
        "reports": reports,
        "decision": (
            "factor_review_required_before_any_extension"
            if combined_pass else
            "reject_cra_l19_r1_no_recurrence_or_public_benchmark"
        ),
        "factor_rule": "A combined pass does not advance alone. Review every pre-constructed language/value/delta/query/two-edit behavioral report before any extension.",
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print("[cra-gate] " + json.dumps({
        "combined_gate_pass": combined_pass,
        "decision": report["decision"],
        "combined_checks": combined_checks,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
