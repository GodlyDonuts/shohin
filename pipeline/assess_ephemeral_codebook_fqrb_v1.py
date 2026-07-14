#!/usr/bin/env python3
"""Pre-registered decision gate for Ephemeral-Codebook Latent Interrogation.

The assessor is deliberately conservative.  It accepts a report only when
all five finite readers, the late-binding table intervention, and the source
controls agree.  A pass is recorded as a bounded latent-interrogation result;
it never authorizes a general-reasoning claim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


QUERY_KINDS = ("ones", "tens", "sign", "parity", "relation")
DIRECT_PATHS = ("normal_correct", "paraphrase_correct", "counterfactual_correct", "codebook_swap_correct")
CONTROL_PATHS = (
    "any_zero_recreates_normal", "any_shuffle_recreates_normal",
    "any_wrong_query_recreates_normal", "any_codebook_swap_recreates_normal",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    if not path.is_file() or not path.stat().st_size:
        raise ValueError("missing or empty JSON artifact: {}".format(path))
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("JSON artifact is not an object: {}".format(path))
    return value


def assess(report: dict, audit: dict) -> dict:
    reasons: list[str] = []
    if audit.get("audit") != "ephemeral_codebook_fqrb_v1" or audit.get("mechanism") != "ephemeral_codebook_fqrb_v1":
        reasons.append("audit_mechanism_mismatch")
    if report.get("groups") != 500 or report.get("rows") != 2500:
        reasons.append("unexpected_evaluation_cardinality")
    if audit.get("heldout_rows") != 2500 or audit.get("train_rows") != 60000:
        reasons.append("unexpected_frozen_corpus_cardinality")
    if report.get("data_sha256") != audit.get("heldout_sha256"):
        reasons.append("evaluation_data_hash_mismatch")
    metadata = report.get("checkpoint_metadata")
    if not isinstance(metadata, dict) or metadata.get("source_present_at_suffix") is not False:
        reasons.append("checkpoint_does_not_certify_source_free_suffix")
    if metadata.get("extra_trainable_parameters") != 0 or metadata.get("composition") != "donor + edited - base":
        reasons.append("checkpoint_composition_contract_failed")
    consumer = report.get("consumer_summary")
    if not isinstance(consumer, dict):
        reasons.append("missing_consumer_summary")
        consumer = {}
    for kind in QUERY_KINDS:
        metrics = consumer.get(kind, {})
        if not isinstance(metrics, dict):
            reasons.append("missing_consumer_{}".format(kind))
            continue
        for path in DIRECT_PATHS:
            if metrics.get(path, -1) < 350:
                reasons.append("{}_{}_below_350".format(kind, path))
    basis = report.get("basis_summary")
    if not isinstance(basis, dict):
        reasons.append("missing_basis_summary")
        basis = {}
    if basis.get("joint_strict", -1) < 300:
        reasons.append("joint_strict_below_300")
    if basis.get("joint_codebook_swap", -1) < 300:
        reasons.append("joint_codebook_swap_below_300")
    for path in CONTROL_PATHS:
        if basis.get(path, 501) > 25:
            reasons.append("{}_above_25".format(path))
    decision = "bounded_ecli_late_binding_candidate" if not reasons else "reject_ecli_late_binding"
    return {
        "assessment": "ephemeral_codebook_fqrb_v1",
        "decision": decision,
        "reasons": reasons,
        "thresholds": {
            "groups": 500, "rows": 2500, "consumer_direct_min": 350,
            "joint_strict_min": 300, "joint_codebook_swap_min": 300, "control_max": 25,
        },
        "claim_boundary": "A pass establishes only bounded late-bound latent interrogation. It is not evidence of general reasoning or a general workspace.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True)
    parser.add_argument("--audit", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    report_path, audit_path, out_path = (Path(value) for value in (args.report, args.audit, args.out))
    if out_path.exists():
        raise SystemExit("refusing to overwrite {}".format(out_path))
    report, audit = load_json(report_path), load_json(audit_path)
    result = assess(report, audit)
    result.update({
        "report": str(report_path), "report_sha256": sha256_file(report_path),
        "audit": str(audit_path), "audit_sha256": sha256_file(audit_path),
        "checkpoint": report.get("checkpoint"), "data": report.get("data"), "data_sha256": report.get("data_sha256"),
    })
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"decision": result["decision"], "reasons": result["reasons"]}, sort_keys=True))


if __name__ == "__main__":
    main()
