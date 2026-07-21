#!/usr/bin/env python3
"""Independently assess ordinal-route fresh-development evidence."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Mapping

import torch

from assess_er_relation_tensor import (
    AssessmentError,
    load_json,
    load_torch,
    metric_equal,
    recompute_arm,
)
from build_er_relation_tensor_board import DEVELOPMENT_SPLIT, PROTOCOL
from er_dual_stream_fresh_scoring import (
    SCORING_ARMS,
    invariance_metrics,
)
from pilot_er_dual_stream_fresh import (
    ACCESS_SCHEMA,
    BOARD_REPORT_SHA256,
    BOARD_SOURCE_COMMIT,
    BOARD_VARIANT,
    CANARY_CHECKPOINT_SHA256,
    CHECKPOINT_SCHEMA,
    EVIDENCE_SCHEMA,
    EXPECTED_PARAMETERS,
    FROZEN_SOURCE_PATHS,
    REPORT_SCHEMA,
    THRESHOLDS,
    compute_gates,
)
from pilot_sd_cst_byte_addressed import sha256_file


ASSESSMENT_SCHEMA = "r12_er_dual_stream_fresh_development_assessment_v1"
ROWS = 2_048
EVALUATED_ARMS = set(SCORING_ARMS) | {"source_free"}


def _tensor_mapping(value: object, name: str) -> dict[str, torch.Tensor]:
    if not isinstance(value, Mapping) or not value:
        raise AssessmentError(f"fresh dual-stream {name} mapping is absent")
    output = {}
    for key, tensor in value.items():
        if (
            not isinstance(key, str)
            or not isinstance(tensor, torch.Tensor)
            or tensor.shape[0] != ROWS
        ):
            raise AssessmentError(f"fresh dual-stream {name} tensor differs: {key}")
        output[key] = tensor
    return output


def recompute_invariance(raw: Mapping[str, object]) -> dict[str, object]:
    expected = {
        "canonical_hard",
        "alpha_hard",
        "distractor_hard",
        "canonical_semantic",
        "rule_reindex",
        "physical_reindex",
    }
    if set(raw) != expected:
        raise AssessmentError("fresh dual-stream invariance arms differ")
    values = {name: _tensor_mapping(raw[name], name) for name in sorted(expected)}
    return invariance_metrics(
        values["canonical_hard"],
        values["alpha_hard"],
        values["distractor_hard"],
        values["canonical_semantic"],
        values["rule_reindex"],
        values["physical_reindex"],
    )


def assess(
    report: Mapping[str, object],
    checkpoint: Mapping[str, object],
    evidence: Mapping[str, object],
    ledger: Mapping[str, object],
    hashes: Mapping[str, str],
) -> dict[str, object]:
    if (
        report.get("schema") != REPORT_SCHEMA
        or checkpoint.get("schema") != CHECKPOINT_SCHEMA
        or evidence.get("schema") != EVIDENCE_SCHEMA
        or ledger.get("schema") != ACCESS_SCHEMA
        or any(
            value.get("protocol") != PROTOCOL
            or value.get("board_variant") != BOARD_VARIANT
            for value in (report, checkpoint, evidence, ledger)
        )
    ):
        raise AssessmentError("fresh dual-stream artifact identity differs")
    if report.get("thresholds") != THRESHOLDS:
        raise AssessmentError("fresh dual-stream thresholds differ")
    if checkpoint.get("training_contract") != report.get("training_contract"):
        raise AssessmentError("fresh dual-stream training contract differs")
    if (
        checkpoint.get("board_source_commit") != BOARD_SOURCE_COMMIT
        or checkpoint.get("board_report_sha256") != BOARD_REPORT_SHA256
        or evidence.get("board_report_sha256") != BOARD_REPORT_SHA256
        or ledger.get("board_report_sha256") != BOARD_REPORT_SHA256
        or checkpoint.get("qualified_canary_checkpoint_sha256")
        != CANARY_CHECKPOINT_SHA256
    ):
        raise AssessmentError("fresh dual-stream provenance differs")
    if (
        hashes["checkpoint"] != report["artifacts"]["checkpoint_sha256"]
        or hashes["checkpoint"] != evidence.get("checkpoint_sha256")
        or hashes["evidence"] != report["artifacts"]["evidence_sha256"]
        or hashes["ledger"] != report["artifacts"]["development_ledger_sha256"]
    ):
        raise AssessmentError("fresh dual-stream artifact hashes differ")
    source_manifest = checkpoint.get("source_manifest")
    if (
        not isinstance(source_manifest, Mapping)
        or source_manifest.get("commit") != checkpoint.get("scientific_source_commit")
        or set(source_manifest.get("files", {})) != set(FROZEN_SOURCE_PATHS)
    ):
        raise AssessmentError("fresh dual-stream source manifest differs")
    if (
        checkpoint.get("parameters") != EXPECTED_PARAMETERS
        or report.get("parameters") != EXPECTED_PARAMETERS
    ):
        raise AssessmentError("fresh dual-stream parameter certificate differs")

    arms = checkpoint.get("arms")
    raw_arms = evidence.get("arms")
    if (
        not isinstance(arms, Mapping)
        or set(arms) != set(SCORING_ARMS)
        or not isinstance(raw_arms, Mapping)
        or set(raw_arms) != EVALUATED_ARMS
    ):
        raise AssessmentError("fresh dual-stream arm identity differs")
    expected_names = set(checkpoint["parent_receipt"]["trainable_names"])
    for name, arm in arms.items():
        if (
            arm["fit"]["arm"] != name
            or arm["fit"]["updates"] != 3_000
            or arm["fit"]["frozen_parent_unchanged"] is not True
            or arm["fit"]["motor_parameters"] != 0
            or arm["fit"]["reader_parameters"] != 0
            or arm["initial_state_sha256"]
            != checkpoint["shared_initial_state_sha256"]
            or set(arm["compiler_trainable_state"]) != expected_names
        ):
            raise AssessmentError(f"fresh dual-stream arm receipt differs: {name}")

    recomputed = {
        name: recompute_arm(raw_arms[name], treatment=False)
        for name in sorted(EVALUATED_ARMS)
    }
    invariant_raw = evidence.get("invariance")
    if not isinstance(invariant_raw, Mapping):
        raise AssessmentError("fresh dual-stream invariant evidence is absent")
    recomputed["treatment"]["invariance"] = recompute_invariance(invariant_raw)
    if not metric_equal(recomputed, report.get("metrics")):
        raise AssessmentError("fresh dual-stream metrics differ from raw evidence")
    gates = compute_gates(recomputed, checkpoint)
    if (
        gates != report.get("gates")
        or all(gates.values()) != report.get("all_gates_pass")
    ):
        raise AssessmentError("fresh dual-stream gates differ")
    expected_decision = (
        "authorize_one_sealed_confirmation"
        if all(gates.values())
        else "reject_er_dual_stream_fresh_v1"
    )
    if report.get("decision") != expected_decision:
        raise AssessmentError("fresh dual-stream decision differs")
    if (
        checkpoint.get("development_accesses") != 0
        or checkpoint.get("confirmation_accesses") != 0
        or evidence.get("development_accesses") != 1
        or evidence.get("confirmation_accesses") != 0
        or ledger.get("split") != DEVELOPMENT_SPLIT
        or ledger.get("access_number") != 1
        or report.get("custody", {}).get("development_accesses") != 1
        or report.get("custody", {}).get("confirmation_accesses") != 0
    ):
        raise AssessmentError("fresh dual-stream custody differs")
    return {
        "schema": ASSESSMENT_SCHEMA,
        "protocol": PROTOCOL,
        "board_variant": BOARD_VARIANT,
        "scientific_source_commit": checkpoint["scientific_source_commit"],
        "decision": expected_decision,
        "all_gates_pass": all(gates.values()),
        "gates": gates,
        "metrics": recomputed,
        "parameters": EXPECTED_PARAMETERS,
        "artifacts": dict(hashes),
        "custody": {"development_accesses": 1, "confirmation_accesses": 0},
        "independent_metric_recomputation": True,
        "independent_list_executor": True,
        "independent_invariance_recomputation": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--access-ledger", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing existing fresh assessment: {args.output}")
    report = load_json(args.report)
    checkpoint = load_torch(args.checkpoint)
    evidence = load_torch(args.evidence)
    ledger = load_json(args.access_ledger)
    hashes = {
        "report": sha256_file(args.report),
        "checkpoint": sha256_file(args.checkpoint),
        "evidence": sha256_file(args.evidence),
        "ledger": sha256_file(args.access_ledger),
    }
    assessment = assess(report, checkpoint, evidence, ledger, hashes)
    payload = (json.dumps(assessment, indent=2, sort_keys=True) + "\n").encode()
    descriptor = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        os.write(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    args.output.chmod(0o444)
    print(
        json.dumps(
            {"decision": assessment["decision"], "sha256": sha256_file(args.output)},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
