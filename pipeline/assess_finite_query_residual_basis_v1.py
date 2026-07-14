#!/usr/bin/env python3
"""Aggregate hash-bound FQRB evidence without promoting a reasoning claim.

FQRB can at most establish a bounded, source-free numeric basis. This assessor
requires group-level causal controls and reports manual interaction separately,
so a finite-label result cannot be relabeled as general reasoning.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


QUERY_KINDS = ("ones", "tens", "sign", "parity", "relation")
DIRECT_FIELDS = ("normal_correct", "paraphrase_correct", "counterfactual_correct")
CONTROL_FIELDS = ("any_zero_recreates_normal", "any_shuffle_recreates_normal", "any_wrong_query_recreates_normal")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def evaluate_report(report: dict) -> dict:
    if report.get("audit") != "finite_query_residual_basis_v1":
        raise ValueError("not an FQRB evaluation report")
    basis, consumers = report.get("basis_summary", {}), report.get("consumer_summary", {})
    if basis.get("groups") != 500 or set(consumers) != set(QUERY_KINDS):
        raise ValueError("FQRB report does not cover the frozen 500-group, five-consumer suite")
    per_consumer = {
        kind: {field: int(consumers[kind].get(field, -1)) for field in DIRECT_FIELDS}
        for kind in QUERY_KINDS
    }
    controls = {field: int(basis.get(field, -1)) for field in CONTROL_FIELDS}
    direct_pass = all(value >= 350 for kind in per_consumer.values() for value in kind.values())
    joint_strict = int(basis.get("joint_strict", -1))
    control_pass = all(value <= 25 for value in controls.values())
    return {
        "joint_strict": joint_strict,
        "per_consumer": per_consumer,
        "controls": controls,
        "direct_pass": direct_pass,
        "joint_pass": joint_strict >= 300,
        "control_pass": control_pass,
        "bounded_basis_gate": bool(direct_pass and joint_strict >= 300 and control_pass),
    }


def manual_summary(report: dict, fqrb_checkpoint: str) -> dict:
    if report.get("audit") != "manual_capability_probe_v1":
        raise ValueError("not a manual capability report")
    models = report.get("models", [])
    if len(models) != 2:
        raise ValueError("manual report must compare raw and FQRB checkpoints")
    raw, candidate = models
    if candidate.get("checkpoint") != fqrb_checkpoint:
        raise ValueError("manual report candidate does not bind the requested FQRB checkpoint")
    fields = ("initial", "review", "verified_fact", "state_reuse")
    raw_summary = {field: int(raw.get("summary", {}).get(field, -1)) for field in fields}
    fqrb_summary = {field: int(candidate.get("summary", {}).get(field, -1)) for field in fields}
    return {
        "raw": raw_summary,
        "fqrb": fqrb_summary,
        # This is deliberately a non-regression guard, not a reasoning score. A
        # carrier arm that emits its tiny answer alphabet on ordinary prompts is
        # unusable as the parent of a direct-reasoning continuation.
        "direct_decode_preserved": bool(
            fqrb_summary["initial"] >= raw_summary["initial"]
            and fqrb_summary["verified_fact"] >= raw_summary["verified_fact"]
        ),
        "claim_boundary": "Seven hand-authored cases are diagnostic only; this non-regression guard blocks automatic continuation but cannot establish general reasoning.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--combined", required=True)
    parser.add_argument("--core", required=True)
    parser.add_argument("--magnitude", required=True)
    parser.add_argument("--manual", required=True)
    parser.add_argument("--fqrb-checkpoint", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = Path(args.out)
    if out.exists():
        raise SystemExit("refusing existing assessment output")
    paths = {key: Path(value) for key, value in {
        "combined": args.combined, "core": args.core, "magnitude": args.magnitude, "manual": args.manual,
    }.items()}
    if any(not path.is_file() for path in paths.values()):
        raise SystemExit("all FQRB evidence reports must exist")
    reports = {key: json.loads(path.read_text()) for key, path in paths.items()}
    combined, core, magnitude = (evaluate_report(reports[key]) for key in ("combined", "core", "magnitude"))
    manual = manual_summary(reports["manual"], args.fqrb_checkpoint)
    if combined["bounded_basis_gate"] and core["bounded_basis_gate"] and manual["direct_decode_preserved"]:
        decision = "bounded_fqrb_basis_candidate_magnitude_and_interaction_still_required"
    elif combined["bounded_basis_gate"] and core["bounded_basis_gate"]:
        decision = "reject_fqrb_due_to_direct_decode_regression"
    else:
        decision = "reject_fqrb_as_reusable_source_free_numeric_basis"
    report = {
        "audit": "finite_query_residual_basis_v1_assessment",
        "evidence": {key: {"path": str(path), "sha256": sha256_file(path)} for key, path in paths.items()},
        "combined": combined, "core": core, "magnitude": magnitude, "manual": manual,
        "decision": decision,
        "claim_boundary": "No outcome of this assessor authorizes a general-reasoning claim. A positive result is only a bounded numeric-basis candidate whose direct decoding did not regress on the diagnostic interview.",
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"decision": decision, "combined": combined["bounded_basis_gate"], "core": core["bounded_basis_gate"], "magnitude": magnitude["bounded_basis_gate"]}, sort_keys=True))


if __name__ == "__main__":
    main()
