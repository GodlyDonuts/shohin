#!/usr/bin/env python3
"""Independently assess ER-CST Witness Equality Bus v1.1 confirmation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping

from assess_er_cst_witness_equality import (
    ARMS,
    ASSESSMENT_SCHEMA as DEVELOPMENT_ASSESSMENT_SCHEMA,
    EXPECTED_PARAMETERS,
    compute_gates as development_compute_gates,
    load_json,
    load_torch,
    metric_equal,
    recompute_arm,
)
from build_er_cst_fresh_board import CONFIRMATION_SPLIT
from build_er_cst_witness_equality_board import PROTOCOL
from er_cst_witness_equality import TRAINING_CONTRACT
from pilot_er_cst_witness_equality import (
    ACCESS_SCHEMA as DEVELOPMENT_ACCESS_SCHEMA,
    BOARD_REPORT_SHA256,
    BOARD_SOURCE_COMMIT,
    CHECKPOINT_SCHEMA,
    REPORT_SCHEMA as DEVELOPMENT_REPORT_SCHEMA,
    THRESHOLDS,
)
from pilot_sd_cst_byte_addressed import sha256_file
from confirm_er_cst_witness_equality import (
    ACCESS_SCHEMA,
    AUTHORIZATION_SCHEMA,
    CHECKPOINT_SHA256,
    CONFIRMATION_SHA256,
    DEVELOPMENT_ASSESSMENT_SHA256,
    DEVELOPMENT_EVIDENCE_SHA256,
    DEVELOPMENT_LEDGER_SHA256,
    DEVELOPMENT_REPORT_SHA256,
    EVIDENCE_SCHEMA,
    REPORT_SCHEMA,
    SCIENTIFIC_SOURCE_COMMIT,
    TRAINING_SEED,
)


ASSESSMENT_SCHEMA = "r12_er_cst_witness_equality_confirmation_assessment_v1_1"


class ConfirmationAssessmentError(RuntimeError):
    pass


def confirmation_gates(
    metrics: Mapping[str, Mapping[str, object]], checkpoint: Mapping[str, object]
) -> dict[str, bool]:
    gates = development_compute_gates(metrics, checkpoint)
    if gates.pop("development_one_confirmation_zero", None) is not True:
        raise ConfirmationAssessmentError("development custody gate differs")
    gates["development_one_confirmation_one"] = True
    return gates


def assess_confirmation(
    *,
    report: Mapping[str, object],
    checkpoint: Mapping[str, object],
    development_report: Mapping[str, object],
    development_assessment: Mapping[str, object],
    development_ledger: Mapping[str, object],
    authorization: Mapping[str, object],
    evidence: Mapping[str, object],
    ledger: Mapping[str, object],
    hashes: Mapping[str, str],
) -> dict[str, object]:
    if (
        report.get("schema") != REPORT_SCHEMA
        or checkpoint.get("schema") != CHECKPOINT_SCHEMA
        or development_report.get("schema") != DEVELOPMENT_REPORT_SCHEMA
        or development_assessment.get("schema") != DEVELOPMENT_ASSESSMENT_SCHEMA
        or development_ledger.get("schema") != DEVELOPMENT_ACCESS_SCHEMA
        or authorization.get("schema") != AUTHORIZATION_SCHEMA
        or evidence.get("schema") != EVIDENCE_SCHEMA
        or ledger.get("schema") != ACCESS_SCHEMA
        or any(
            value.get("protocol") != PROTOCOL
            for value in (
                report,
                checkpoint,
                development_report,
                development_assessment,
                development_ledger,
                authorization,
                evidence,
                ledger,
            )
        )
    ):
        raise ConfirmationAssessmentError("ER-CST witness confirmation identity differs")
    if report.get("thresholds") != THRESHOLDS:
        raise ConfirmationAssessmentError("ER-CST witness confirmation thresholds differ")
    if checkpoint.get("training_contract") != TRAINING_CONTRACT:
        raise ConfirmationAssessmentError("ER-CST witness training contract differs")
    if (
        hashes["checkpoint"] != CHECKPOINT_SHA256
        or hashes["development_evidence"] != DEVELOPMENT_EVIDENCE_SHA256
        or hashes["development_report"] != DEVELOPMENT_REPORT_SHA256
        or hashes["development_assessment"] != DEVELOPMENT_ASSESSMENT_SHA256
        or hashes["development_ledger"] != DEVELOPMENT_LEDGER_SHA256
        or development_report.get("decision") != "authorize_one_sealed_confirmation"
        or development_report.get("all_gates_pass") is not True
        or development_assessment.get("decision") != "authorize_one_sealed_confirmation"
        or development_assessment.get("all_gates_pass") is not True
        or development_assessment.get("custody")
        != {"development_accesses": 1, "confirmation_accesses": 0}
        or development_ledger.get("split") != "er_cst_development"
        or development_ledger.get("access_number") != 1
        or development_ledger.get("scientific_source_commit")
        != SCIENTIFIC_SOURCE_COMMIT
    ):
        raise ConfirmationAssessmentError("ER-CST witness development authorization differs")
    if (
        checkpoint.get("scientific_source_commit") != SCIENTIFIC_SOURCE_COMMIT
        or checkpoint.get("training_seed") != TRAINING_SEED
        or checkpoint.get("board_source_commit") != BOARD_SOURCE_COMMIT
        or checkpoint.get("board_report_sha256") != BOARD_REPORT_SHA256
        or evidence.get("scientific_source_commit") != SCIENTIFIC_SOURCE_COMMIT
        or evidence.get("board_report_sha256") != BOARD_REPORT_SHA256
        or evidence.get("confirmation_sha256") != CONFIRMATION_SHA256
    ):
        raise ConfirmationAssessmentError("ER-CST witness scientific binding differs")
    if (
        ledger.get("split") != CONFIRMATION_SPLIT
        or ledger.get("split_sha256") != CONFIRMATION_SHA256
        or ledger.get("access_number") != 1
        or ledger.get("scientific_source_commit") != SCIENTIFIC_SOURCE_COMMIT
    ):
        raise ConfirmationAssessmentError("ER-CST witness confirmation ledger differs")
    evidence_arms = evidence.get("arms")
    if not isinstance(evidence_arms, Mapping) or set(evidence_arms) != ARMS:
        raise ConfirmationAssessmentError("ER-CST witness confirmation arms differ")
    checkpoint_arms = checkpoint.get("arms")
    if not isinstance(checkpoint_arms, Mapping) or set(checkpoint_arms) != ARMS:
        raise ConfirmationAssessmentError("ER-CST witness checkpoint arms differ")
    metrics = {name: recompute_arm(evidence_arms[name]) for name in sorted(ARMS)}
    gates = confirmation_gates(metrics, checkpoint)
    parameters = checkpoint.get("parameters")
    parameter_exact = isinstance(parameters, Mapping) and all(
        int(parameters.get(name, -1)) == expected
        for name, expected in EXPECTED_PARAMETERS.items()
    )
    evaluator_source = authorization.get("evaluator_source")
    authorization_exact = (
        authorization.get("scientific_source", {}).get("commit")
        == SCIENTIFIC_SOURCE_COMMIT
        and isinstance(evaluator_source, Mapping)
        and evaluator_source.get("commit") == report.get("evaluator_source_commit")
        and evidence.get("evaluator_source_commit") == report.get("evaluator_source_commit")
        and ledger.get("evaluator_source_commit") == report.get("evaluator_source_commit")
        and authorization.get("board_report_sha256") == BOARD_REPORT_SHA256
        and authorization.get("checkpoint_sha256") == CHECKPOINT_SHA256
        and authorization.get("development_evidence_sha256")
        == DEVELOPMENT_EVIDENCE_SHA256
        and authorization.get("development_report_sha256")
        == DEVELOPMENT_REPORT_SHA256
        and authorization.get("development_assessment_sha256")
        == DEVELOPMENT_ASSESSMENT_SHA256
        and authorization.get("development_ledger_sha256")
        == DEVELOPMENT_LEDGER_SHA256
        and authorization.get("development_decision")
        == "authorize_one_sealed_confirmation"
        and authorization.get("development_accesses") == 1
        and authorization.get("confirmation_accesses") == 0
    )
    artifact_exact = (
        report.get("artifacts", {}).get("checkpoint_sha256") == hashes["checkpoint"]
        and report.get("artifacts", {}).get("authorization_sha256")
        == hashes["authorization"]
        and report.get("artifacts", {}).get("evidence_sha256") == hashes["evidence"]
        and report.get("artifacts", {}).get("development_assessment_sha256")
        == hashes["development_assessment"]
        and evidence.get("checkpoint_sha256") == hashes["checkpoint"]
        and report.get("custody", {}).get("confirmation_ledger", {}).get("sha256")
        == hashes["ledger"]
    )
    custody_exact = (
        checkpoint.get("development_accesses") == 0
        and checkpoint.get("confirmation_accesses") == 0
        and evidence.get("development_accesses") == 1
        and evidence.get("confirmation_accesses") == 1
        and report.get("custody", {}).get("development_accesses") == 1
        and report.get("custody", {}).get("confirmation_accesses") == 1
    )
    assessor_gates = {
        "artifact_hashes_match": artifact_exact,
        "parameter_certificate_exact": parameter_exact,
        "development_authorization_exact": authorization_exact,
        "pilot_metrics_match_independent_recomputation": metric_equal(
            report.get("metrics"), metrics
        ),
        "pilot_gate_vector_matches_independent_recomputation": report.get("gates")
        == gates,
        "confirmation_custody_exact": custody_exact,
    }
    all_pass = all(gates.values()) and all(assessor_gates.values())
    return {
        "schema": ASSESSMENT_SCHEMA,
        "protocol": PROTOCOL,
        "decision": (
            "confirm_er_cst_witness_equality_v1_1"
            if all_pass
            else "reject_er_cst_witness_equality_confirmation_v1_1"
        ),
        "all_gates_pass": all_pass,
        "core_gates": gates,
        "assessor_gates": assessor_gates,
        "thresholds": THRESHOLDS,
        "parameters": dict(parameters) if isinstance(parameters, Mapping) else None,
        "metrics": metrics,
        "artifact_sha256": dict(hashes),
        "custody": {"development_accesses": 1, "confirmation_accesses": 1},
        "claim_boundary": report.get("claim_boundary"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "report",
        "checkpoint",
        "development_evidence",
        "development_report",
        "development_assessment",
        "development_ledger",
        "authorization",
        "evidence",
        "ledger",
        "output",
    ):
        parser.add_argument(f"--{name.replace('_', '-')}", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing existing ER-CST witness confirmation assessment: {args.output}")
    paths = {
        "report": args.report,
        "checkpoint": args.checkpoint,
        "development_evidence": args.development_evidence,
        "development_report": args.development_report,
        "development_assessment": args.development_assessment,
        "development_ledger": args.development_ledger,
        "authorization": args.authorization,
        "evidence": args.evidence,
        "ledger": args.ledger,
    }
    value = assess_confirmation(
        report=load_json(args.report),
        checkpoint=load_torch(args.checkpoint),
        development_report=load_json(args.development_report),
        development_assessment=load_json(args.development_assessment),
        development_ledger=load_json(args.development_ledger),
        authorization=load_json(args.authorization),
        evidence=load_torch(args.evidence),
        ledger=load_json(args.ledger),
        hashes={name: sha256_file(path) for name, path in paths.items()},
    )
    args.output.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    args.output.chmod(0o444)
    print(json.dumps({"decision": value["decision"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
