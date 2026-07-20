from __future__ import annotations

import json
from pathlib import Path

import pytest

import assess_er_cst_witness_equality as development
import assess_er_cst_witness_equality_confirmation as confirmation
from assess_er_cst_witness_equality import EXPECTED_PARAMETERS
from build_er_cst_fresh_board import CONFIRMATION_SPLIT
from build_er_cst_witness_equality_board import PROTOCOL
from er_cst_witness_equality import TRAINING_CONTRACT
from pilot_er_cst_witness_equality import BOARD_REPORT_SHA256, BOARD_SOURCE_COMMIT, CHECKPOINT_SCHEMA, THRESHOLDS
from confirm_er_cst_witness_equality import (
    ACCESS_SCHEMA,
    AUTHORIZATION_SCHEMA,
    CHECKPOINT_SHA256,
    CONFIRMATION_SHA256,
    DEVELOPMENT_ASSESSMENT_SCHEMA,
    DEVELOPMENT_ASSESSMENT_SHA256,
    DEVELOPMENT_EVIDENCE_SHA256,
    DEVELOPMENT_LEDGER_SHA256,
    DEVELOPMENT_REPORT_SHA256,
    EVIDENCE_SCHEMA,
    REPORT_SCHEMA,
    SCIENTIFIC_SOURCE_COMMIT,
    TRAINING_SEED,
    consume_confirmation_access,
)


def _summary(rate: float) -> dict[str, object]:
    correct = int(round(rate * 2_048))
    return {"correct": correct, "rows": 2_048, "rate": correct / 2_048}


def _metrics(rate: float) -> dict[str, object]:
    fields = (
        "initial", "cards", "events", "halt", "query", "line_pointer",
        "binding_pointer", "initial_pointer", "witness_pointer", "query_pointer",
        "state", "answer", "packet", "joint",
    )
    grouped = {
        str(index): {name: _summary(rate) for name in ("packet", "state", "answer", "joint")}
        for index in range(1, 9)
    }
    renderers = {
        name: {field: _summary(rate) for field in ("packet", "state", "answer", "joint")}
        for name in ("a", "b", "c", "d")
    }
    return {
        "overall": {name: _summary(rate) for name in fields},
        "by_depth": grouped,
        "by_renderer": renderers,
    }


def test_confirmation_assessor_accepts_exact_capsule(monkeypatch) -> None:
    metrics = {
        "treatment": _metrics(1.0),
        "family_deranged": _metrics(0.0),
        "equality_ablated": _metrics(0.0),
    }
    monkeypatch.setattr(
        confirmation, "recompute_arm", lambda raw: metrics[str(raw["name"])]
    )
    monkeypatch.setattr(development, "certificate_exact", lambda _arm: True)
    checkpoint = {
        "schema": CHECKPOINT_SCHEMA,
        "protocol": PROTOCOL,
        "scientific_source_commit": SCIENTIFIC_SOURCE_COMMIT,
        "training_seed": TRAINING_SEED,
        "training_contract": dict(TRAINING_CONTRACT),
        "board_source_commit": BOARD_SOURCE_COMMIT,
        "board_report_sha256": BOARD_REPORT_SHA256,
        "parameters": dict(EXPECTED_PARAMETERS),
        "arms": {
            name: {"fit": {"frozen_parent_unchanged": True}}
            for name in metrics
        },
        "development_accesses": 0,
        "confirmation_accesses": 0,
    }
    gates = confirmation.confirmation_gates(metrics, checkpoint)
    evaluator_commit = "f" * 40
    hashes = {
        "report": "report-sha",
        "checkpoint": CHECKPOINT_SHA256,
        "development_evidence": DEVELOPMENT_EVIDENCE_SHA256,
        "development_report": DEVELOPMENT_REPORT_SHA256,
        "development_assessment": DEVELOPMENT_ASSESSMENT_SHA256,
        "development_ledger": DEVELOPMENT_LEDGER_SHA256,
        "authorization": "authorization-sha",
        "evidence": "evidence-sha",
        "ledger": "ledger-sha",
    }
    report = {
        "schema": REPORT_SCHEMA,
        "protocol": PROTOCOL,
        "thresholds": dict(THRESHOLDS),
        "evaluator_source_commit": evaluator_commit,
        "metrics": metrics,
        "gates": gates,
        "artifacts": {
            "checkpoint_sha256": hashes["checkpoint"],
            "authorization_sha256": hashes["authorization"],
            "evidence_sha256": hashes["evidence"],
            "development_assessment_sha256": hashes["development_assessment"],
        },
        "custody": {
            "development_accesses": 1,
            "confirmation_accesses": 1,
            "confirmation_ledger": {"sha256": hashes["ledger"]},
        },
        "claim_boundary": "bounded claim",
    }
    development_assessment = {
        "schema": DEVELOPMENT_ASSESSMENT_SCHEMA,
        "protocol": PROTOCOL,
        "decision": "authorize_one_sealed_confirmation",
        "all_gates_pass": True,
        "custody": {"development_accesses": 1, "confirmation_accesses": 0},
    }
    development_report = {
        "schema": "r12_er_cst_witness_equality_development_report_v1_1",
        "protocol": PROTOCOL,
        "decision": "authorize_one_sealed_confirmation",
        "all_gates_pass": True,
    }
    development_ledger = {
        "schema": "r12_er_cst_witness_equality_development_access_v1_1",
        "protocol": PROTOCOL,
        "split": "er_cst_development",
        "access_number": 1,
        "scientific_source_commit": SCIENTIFIC_SOURCE_COMMIT,
    }
    authorization = {
        "schema": AUTHORIZATION_SCHEMA,
        "protocol": PROTOCOL,
        "scientific_source": {"commit": SCIENTIFIC_SOURCE_COMMIT},
        "evaluator_source": {"commit": evaluator_commit},
        "board_report_sha256": BOARD_REPORT_SHA256,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "development_evidence_sha256": DEVELOPMENT_EVIDENCE_SHA256,
        "development_report_sha256": DEVELOPMENT_REPORT_SHA256,
        "development_assessment_sha256": DEVELOPMENT_ASSESSMENT_SHA256,
        "development_ledger_sha256": DEVELOPMENT_LEDGER_SHA256,
        "development_decision": "authorize_one_sealed_confirmation",
        "development_accesses": 1,
        "confirmation_accesses": 0,
    }
    evidence = {
        "schema": EVIDENCE_SCHEMA,
        "protocol": PROTOCOL,
        "scientific_source_commit": SCIENTIFIC_SOURCE_COMMIT,
        "evaluator_source_commit": evaluator_commit,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "board_report_sha256": BOARD_REPORT_SHA256,
        "confirmation_sha256": CONFIRMATION_SHA256,
        "arms": {name: {"name": name} for name in metrics},
        "development_accesses": 1,
        "confirmation_accesses": 1,
    }
    ledger = {
        "schema": ACCESS_SCHEMA,
        "protocol": PROTOCOL,
        "split": CONFIRMATION_SPLIT,
        "split_sha256": CONFIRMATION_SHA256,
        "scientific_source_commit": SCIENTIFIC_SOURCE_COMMIT,
        "evaluator_source_commit": evaluator_commit,
        "access_number": 1,
    }
    result = confirmation.assess_confirmation(
        report=report,
        checkpoint=checkpoint,
        development_report=development_report,
        development_assessment=development_assessment,
        development_ledger=development_ledger,
        authorization=authorization,
        evidence=evidence,
        ledger=ledger,
        hashes=hashes,
    )
    assert result["all_gates_pass"] is True
    assert result["decision"] == "confirm_er_cst_witness_equality_v1_1"
    assert result["custody"] == {"development_accesses": 1, "confirmation_accesses": 1}


def test_confirmation_access_is_exclusive(tmp_path: Path) -> None:
    (tmp_path / "access").mkdir()
    first = consume_confirmation_access(tmp_path, "e" * 40)
    assert Path(first["path"]).stat().st_mode & 0o777 == 0o444
    payload = json.loads(Path(first["path"]).read_text())
    assert payload["access_number"] == 1
    assert payload["split_sha256"] == CONFIRMATION_SHA256
    with pytest.raises(FileExistsError):
        consume_confirmation_access(tmp_path, "e" * 40)
