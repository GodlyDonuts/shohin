from __future__ import annotations

import json

import pytest

import assess_ctaa_evidence as assessment
from assess_ctaa_evidence import assess
from commit_ctaa_raw_evidence import RAW_EVIDENCE_RECEIPT_SCHEMA, RAW_EVIDENCE_SCHEMA
from ctaa_assessment import score_evidence
from ctaa_evaluation_io import sha256_file, write_json_once, write_jsonl_once


def _oracle(family_id: str = "f0") -> dict[str, object]:
    schedule = [0, 1, 4] + [3] * 38
    prefix = [[0, 1, 2] for _ in range(42)]
    return {
        "family_id": family_id,
        "partition": "development",
        "factorial_cell": "hhh",
        "program_class": "stable_rank_two",
        "depth": 2,
        "action_cards": [[0, 1, 2]] * 4,
        "initial_state": [0, 1, 2],
        "schedule": schedule,
        "query_position": 1,
        "prefix_states": prefix,
        "terminal_state": [0, 1, 2],
        "answer": 1,
        "map_deletion_depth": 1,
        "state_deletion_depth": 1,
        "answer_deletion_depth": 1,
        "shortest_equivalent_length": 1,
        "max_run_length": 1,
        "normalized_event_entropy": 1.0,
        "renderer": 7,
    }


def _evidence(family_id: str = "f0", *, valid: bool = True) -> dict[str, object]:
    schedule = [0, 1, 4] + [3] * 38
    prefix = [[0, 1, 2] for _ in range(42)]
    halted = [False, False, False] + [True] * 39
    return {
        "schema": RAW_EVIDENCE_SCHEMA,
        "family_id": family_id,
        "source_index": 0,
        "packet_valid": valid,
        "predicted_action_cards": [[0, 1, 2]] * 4,
        "predicted_initial_state": [0, 1, 2],
        "predicted_schedule": schedule,
        "predicted_query_position": 1 if valid else None,
        "state_route": prefix if valid else None,
        "halted": halted if valid else None,
        "composed_states": prefix if valid else None,
        "route_agreement": valid,
        "answer": 1 if valid else None,
    }


def _evidence_dir(
    path,
    rows,
    *,
    program_sha: str = "p" * 64,
    query_sha: str = "q" * 64,
) -> None:
    path.mkdir()
    count, digest = write_jsonl_once(path / "evidence.jsonl", rows)
    write_json_once(
        path / "receipt.json",
        {
            "schema": RAW_EVIDENCE_RECEIPT_SCHEMA,
            "rows": count,
            "evidence_sha256": digest,
            "program_source_sha256": program_sha,
            "query_source_sha256": query_sha,
            "core_training": {
                "training_seed": 1,
                "training_arm": "ctaa_closure",
            },
        },
    )


def test_scorer_counts_invalid_rows_as_failures_and_reports_marginals() -> None:
    first = _evidence()
    second = _evidence("f1", valid=False)
    second["source_index"] = 1
    report = score_evidence([first, second], [_oracle(), _oracle("f1")])
    assert report["overall"]["rows"] == 2
    assert report["overall"]["packet_valid"] == 0.5
    assert report["overall"]["prefix_exact"] == 0.5
    assert report["overall"]["answer_exact"] == 0.5
    assert set(report["by_action_active_prefix_accuracy"]) == {"0", "1"}
    assert set(report["by_semantic_action_active_prefix_accuracy"]) == {
        "[0,1,2]"
    }
    assert set(report["by_action_rank_active_prefix_accuracy"]) == {"3"}
    assert set(report["by_step_quartile_active_prefix_accuracy"]) == {"1", "3"}
    assert report["factorial_main_effects"]["semantic"]["held_out"] == 0.5
    assert report["family_scores"][0]["cluster_family_id"] == "f0"


def test_assessor_spends_access_before_oracle_score_and_refuses_reuse(
    tmp_path,
    monkeypatch,
) -> None:
    board = tmp_path / "board"
    board.mkdir()
    oracle = board / "development_oracle.jsonl"
    _, oracle_sha = write_jsonl_once(oracle, [_oracle()], mode=0o600)
    program = board / "development_program.jsonl"
    _, program_sha = write_jsonl_once(
        program,
        [{"family_id": "f0", "program_source": "program"}],
    )
    query = board / "development_query.jsonl"
    _, query_sha = write_jsonl_once(
        query,
        [{"family_id": "f0", "query_source": "query"}],
        mode=0o600,
    )
    ledger = board / "access_ledger.json"
    write_json_once(
        ledger,
        {
            "schema": "r12_ctaa_v2_access_ledger_v1",
            "development_access": 0,
            "confirmation_access": 0,
        },
        mode=0o600,
    )
    manifest = board / "manifest.json"
    write_json_once(
        manifest,
        {
            "schema": "r12_ctaa_v2_manifest_v1",
            "seed": 1,
            "files": {
                oracle.name: oracle_sha,
                program.name: program_sha,
                query.name: query_sha,
                ledger.name: sha256_file(ledger),
            },
        },
    )
    evidence = tmp_path / "evidence"
    _evidence_dir(
        evidence,
        [_evidence()],
        program_sha=program_sha,
        query_sha=query_sha,
    )
    with pytest.raises(ValueError, match="core checkpoint"):
        assess(
            manifest_path=manifest,
            ledger_path=ledger,
            partition="development",
            runs=[("arm", evidence, oracle)],
            output_path=tmp_path / "mislabeled.json",
            run_metadata={
                "arm": {"seed": 2, "arm": "ctaa_closure", "dataset": "base"}
            },
        )
    assert json.loads(ledger.read_text())["development_access"] == 0
    original_verify = assessment._verify_board_file

    def verify_after_spend(path, manifest_path, manifest):
        assert json.loads(ledger.read_text())["development_access"] == 1
        return original_verify(path, manifest_path, manifest)

    monkeypatch.setattr(assessment, "_verify_board_file", verify_after_spend)
    report = assess(
        manifest_path=manifest,
        ledger_path=ledger,
        partition="development",
        runs=[("arm", evidence, oracle)],
        output_path=tmp_path / "report.json",
        run_metadata={
            "arm": {"seed": 1, "arm": "ctaa_closure", "dataset": "base"}
        },
    )
    assert report["runs"]["arm"]["scores"]["overall"]["answer_exact"] == 1.0
    assert json.loads(ledger.read_text())["development_access"] == 1
    with pytest.raises(ValueError, match="ledger lineage"):
        assess(
            manifest_path=manifest,
            ledger_path=ledger,
            partition="development",
            runs=[("arm", evidence, oracle)],
            output_path=tmp_path / "second.json",
            run_metadata={
                "arm": {"seed": 1, "arm": "ctaa_closure", "dataset": "base"}
            },
        )


def test_family_set_mismatch_is_fatal() -> None:
    with pytest.raises(ValueError, match="family set"):
        score_evidence([_evidence()], [_oracle("other")])


def test_source_commitment_mismatch_fails_before_access_is_spent(tmp_path) -> None:
    board = tmp_path / "board"
    board.mkdir()
    oracle = board / "development_oracle.jsonl"
    _, oracle_sha = write_jsonl_once(oracle, [_oracle()], mode=0o600)
    program = board / "development_program.jsonl"
    _, program_sha = write_jsonl_once(
        program,
        [{"family_id": "f0", "program_source": "program"}],
    )
    query = board / "development_query.jsonl"
    _, query_sha = write_jsonl_once(
        query,
        [{"family_id": "f0", "query_source": "query"}],
        mode=0o600,
    )
    ledger = board / "access_ledger.json"
    write_json_once(
        ledger,
        {
            "schema": "r12_ctaa_v2_access_ledger_v1",
            "development_access": 0,
            "confirmation_access": 0,
        },
        mode=0o600,
    )
    manifest = board / "manifest.json"
    write_json_once(
        manifest,
        {
            "schema": "r12_ctaa_v2_manifest_v1",
            "seed": 1,
            "files": {
                oracle.name: oracle_sha,
                program.name: program_sha,
                query.name: query_sha,
                ledger.name: sha256_file(ledger),
            },
        },
    )
    evidence = tmp_path / "evidence"
    _evidence_dir(
        evidence,
        [_evidence()],
        program_sha="0" * 64,
        query_sha=query_sha,
    )
    with pytest.raises(ValueError, match="not bound"):
        assess(
            manifest_path=manifest,
            ledger_path=ledger,
            partition="development",
            runs=[("arm", evidence, oracle)],
            output_path=tmp_path / "report.json",
            run_metadata={
                "arm": {"seed": 1, "arm": "ctaa_closure", "dataset": "base"}
            },
        )
    assert json.loads(ledger.read_text())["development_access"] == 0


def test_intervention_relation_cannot_credit_invalid_or_jointly_wrong_rows() -> None:
    child_oracle = {
        **_oracle("child"),
        "parent_family_id": "parent",
        "relation": "post_stop_poison",
        "invariant_terminal": True,
        "invariant_trace": True,
    }
    invalid = score_evidence(
        [_evidence("child", valid=False)],
        [child_oracle],
        parent_evidence_rows=[_evidence("parent", valid=False)],
        parent_oracle_rows=[_oracle("parent")],
    )
    assert invalid["intervention_relation_correct"]["post_stop_poison"] == 0.0

    valid = score_evidence(
        [_evidence("child")],
        [child_oracle],
        parent_evidence_rows=[_evidence("parent")],
        parent_oracle_rows=[_oracle("parent")],
    )
    assert valid["intervention_relation_correct"]["post_stop_poison"] == 1.0
    assert valid["family_scores"][0]["cluster_family_id"] == "parent"
