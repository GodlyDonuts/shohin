#!/usr/bin/env python3
"""Classify FQRB failure modes from frozen read-only reports.

This never promotes a model.  It turns a failed bounded-basis gate into a
specific next diagnostic and prevents a generic re-run from being mistaken for
research progress.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


QUERY_KINDS = ("ones", "tens", "sign", "parity", "relation")
DIRECT = ("normal_correct", "paraphrase_correct", "counterfactual_correct")
CONTROLS = ("any_zero_recreates_normal", "any_shuffle_recreates_normal", "any_wrong_query_recreates_normal")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path) -> dict:
    if not path.is_file() or not path.stat().st_size:
        raise ValueError("missing report: {}".format(path))
    report = json.loads(path.read_text())
    if report.get("audit") != "finite_query_residual_basis_v1" or report.get("groups") != 500:
        raise ValueError("not a frozen 500-group FQRB report: {}".format(path))
    if set(report.get("consumer_summary", ())) != set(QUERY_KINDS):
        raise ValueError("incomplete FQRB consumers: {}".format(path))
    return report


def gate(report: dict) -> dict:
    consumer = report["consumer_summary"]
    direct = all(consumer[kind].get(field, -1) >= 350 for kind in QUERY_KINDS for field in DIRECT)
    basis = report["basis_summary"]
    controls = all(basis.get(field, 501) <= 25 for field in CONTROLS)
    joint = basis.get("joint_strict", -1) >= 300
    return {"direct": direct, "joint": joint, "controls": controls, "pass": bool(direct and joint and controls)}


def analyze(train: dict, combined: dict, core: dict, magnitude: dict) -> dict:
    gates = {name: gate(report) for name, report in {
        "train": train, "combined": combined, "core": core, "magnitude": magnitude,
    }.items()}
    reasons: list[str] = []
    recommendation: str
    if not gates["train"]["pass"]:
        reasons.append("no_in_distribution_multi_reader_primitive")
        recommendation = "close this residual-basis shape; inspect row-level decoding before any successor"
    elif not gates["core"]["controls"] or not gates["combined"]["controls"]:
        reasons.append("source_free_control_leakage")
        recommendation = "do not add query complexity; inspect zero/shuffle readout leakage"
    elif not gates["core"]["pass"]:
        reasons.append("unseen_source_tuple_transport_failure")
        recommendation = "inspect source-state representation, not output vocabulary or codebook binding"
    elif not gates["combined"]["pass"]:
        reasons.append("combined_language_or_joint_surface_failure")
        recommendation = "add a wording-isolated factor before any semantic bridge or reflection arm"
    elif not gates["magnitude"]["pass"]:
        reasons.append("primary_magnitude_transport_failure")
        recommendation = "keep output alphabet fixed and diagnose numeric-scale representation before ECLI"
    else:
        reasons.append("bounded_fqrb_gate_passed")
        recommendation = "ECLI may be generated and trained as a bounded late-binding successor"
    weak = {
        name: {
            kind: [field for field in DIRECT if report["consumer_summary"][kind].get(field, -1) < 350]
            for kind in QUERY_KINDS
        }
        for name, report in {"train": train, "combined": combined, "core": core, "magnitude": magnitude}.items()
    }
    return {"assessment": "finite_query_residual_basis_v1_failure_taxonomy", "gates": gates, "reasons": reasons,
            "weak_consumers": weak, "recommendation": recommendation,
            "claim_boundary": "This is a diagnostic classification, not a reasoning result or training authorization."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True)
    parser.add_argument("--combined", required=True)
    parser.add_argument("--core", required=True)
    parser.add_argument("--magnitude", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    paths = {name: Path(value) for name, value in {
        "train": args.train, "combined": args.combined, "core": args.core, "magnitude": args.magnitude,
    }.items()}
    out = Path(args.out)
    if out.exists():
        raise SystemExit("refusing to overwrite {}".format(out))
    reports = {name: load(path) for name, path in paths.items()}
    result = analyze(**reports)
    result["evidence"] = {name: {"path": str(path), "sha256": sha256_file(path)} for name, path in paths.items()}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"reasons": result["reasons"], "recommendation": result["recommendation"]}, sort_keys=True))


if __name__ == "__main__":
    main()
