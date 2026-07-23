from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
import pytest

import assess_ctaa_evidence as assessment
from assess_ctaa_evidence import assess
from commit_ctaa_raw_evidence import RAW_EVIDENCE_RECEIPT_SCHEMA, RAW_EVIDENCE_SCHEMA
import ctaa_access_registry as access_registry
from ctaa_assessment import load_committed_evidence_bundle, load_oracle, score_evidence
from ctaa_evaluation_io import sha256_file, write_json_once, write_jsonl_once
from ctaa_statistical_gate_spec import (
    StatisticalGateBindings,
    write_signed_statistical_gate_spec,
)


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
        "opcode_to_card": [0, 1, 2, 3],
        "initial_state": [0, 1, 2],
        "opcode_schedule": schedule,
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
        "predicted_opcode_to_card": [0, 1, 2, 3],
        "predicted_initial_state": [0, 1, 2],
        "predicted_opcode_schedule": schedule,
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


def _signed_access(
    tmp_path,
    monkeypatch,
    *,
    manifest,
    evidence,
    oracle,
) -> tuple[dict[str, object], Ed25519PrivateKey, Path, list[str]]:
    receipt = json.loads((evidence / "receipt.json").read_text())
    contract = {
        "partition": "development",
        "manifest_sha256": sha256_file(manifest),
        "board_sha256": "b" * 64,
        "run_plan_sha256": "f" * 64,
        "run_contract_sha256": "c" * 64,
        "bootstrap_seed_receipt_sha256": "d" * 64,
        "bootstrap_seed": 123456789,
        "training_seeds": [1, 2, 3, 4, 5],
        "oracle_files": {
            "base": {"filename": oracle.name, "sha256": sha256_file(oracle)}
        },
        "runs": [
            {
                "run_id": "arm",
                "seed": 1,
                "arm": "ctaa_closure",
                "dataset": "base",
                "raw_evidence_receipt_sha256": sha256_file(evidence / "receipt.json"),
                "compiler_sha256": receipt.get("compiler_sha256"),
                "evidence_artifacts": {"evidence_sha256": receipt["evidence_sha256"]},
                "core_training": receipt["core_training"],
                "sealed_sources": {"oracle_sha256": sha256_file(oracle)},
            }
        ],
    }
    monkeypatch.setattr(assessment, "validate_run_contract", lambda **_kwargs: contract)
    runtime_bundle_path = tmp_path / "runtime-bundle.json"
    write_json_once(runtime_bundle_path, {"fixture": "runtime-bundle"})
    custody_events: list[str] = []

    def fake_runtime_bundle(path, **kwargs):
        custody_events.append("query_bundle_replay")
        return (
            {
                "partition": kwargs["run_contract"]["partition"],
                "bundle_sha256": "5" * 64,
            },
            hashlib.sha256(
                assessment._read_immutable_bytes(path, "runtime bundle")
            ).hexdigest(),
        )

    monkeypatch.setattr(
        assessment,
        "_load_runtime_bundle_with_replay_and_sha",
        fake_runtime_bundle,
    )
    key = Ed25519PrivateKey.from_private_bytes(b"k" * 32)
    statistical_gate_spec_path = tmp_path / "statistical-gate-spec.json"
    statistical_gate_spec_file_sha256 = write_signed_statistical_gate_spec(
        statistical_gate_spec_path,
        bindings=StatisticalGateBindings(
            manifest_sha256=contract["manifest_sha256"],
            board_sha256=contract["board_sha256"],
            run_plan_sha256=contract["run_plan_sha256"],
            run_contract_sha256=contract["run_contract_sha256"],
            runtime_bundle_file_sha256=sha256_file(runtime_bundle_path),
            runtime_bundle_sha256="5" * 64,
            runtime_execution_set_file_sha256="4" * 64,
            runtime_execution_set_sha256="3" * 64,
            assessment_source_bundle_sha256="6" * 64,
            assessment_source_manifest_sha256="7" * 64,
            bootstrap_seed_receipt_sha256=contract[
                "bootstrap_seed_receipt_sha256"
            ],
            bootstrap_seed=contract["bootstrap_seed"],
            training_seeds=(1, 2, 3, 4, 5),
        ),
        signing_key=key,
    )
    gate_spec_sha256 = json.loads(
        statistical_gate_spec_path.read_text()
    )["gate_spec_sha256"]
    registry_path = tmp_path / "access.jsonl"
    spend_receipt = access_registry.append_access_spend(
        registry_path,
        signing_key=key,
        registry_id="assessment-test-registry",
        event_id="development-spend",
        access_id="development-access",
        partition="development",
        manifest_sha256=contract["manifest_sha256"],
        board_sha256=contract["board_sha256"],
        run_contract_sha256=contract["run_contract_sha256"],
        runtime_bundle_sha256=sha256_file(runtime_bundle_path),
        assessment_claim_sha256="e" * 64,
        bootstrap_seed_receipt_sha256=contract["bootstrap_seed_receipt_sha256"],
        bootstrap_seed=contract["bootstrap_seed"],
        statistical_gate_spec_file_sha256=statistical_gate_spec_file_sha256,
        gate_spec_sha256=gate_spec_sha256,
        expected_previous_hash=access_registry.GENESIS_PREVIOUS_HASH,
    )
    head_path = tmp_path / "spend-head.json"
    write_json_once(head_path, spend_receipt)
    public_key = key.public_key().public_bytes(
        serialization.Encoding.Raw,
        serialization.PublicFormat.Raw,
    )
    runtime_execution_set_path = tmp_path / "runtime-execution-set.json"
    write_json_once(runtime_execution_set_path, {"fixture": "execution-set"})
    execution_set = {
        "partition": "development",
        "run_contract_sha256": contract["run_contract_sha256"],
        "runtime_bundle_file_sha256": sha256_file(runtime_bundle_path),
        "execution_set_sha256": "3" * 64,
    }

    def fake_execution_set(
        path, *, runtime_bundle_path, run_contract, verification_key
    ):
        assert Path(path) == runtime_execution_set_path
        assert Path(runtime_bundle_path) == runtime_bundle_path
        assert run_contract is contract
        assert verification_key == public_key
        custody_events.append("execution_set")
        return execution_set, "4" * 64

    monkeypatch.setattr(
        assessment,
        "read_runtime_execution_set_with_replay",
        fake_execution_set,
    )
    kwargs = {
        "run_plan_path": tmp_path / "run-plan.json",
        "run_contract_path": tmp_path / "run-contract.json",
        "bootstrap_seed_receipt_path": tmp_path / "bootstrap.json",
        "runtime_bundle_path": runtime_bundle_path,
        "runtime_program_source_path": next(
            manifest.parent / name
            for name in json.loads(manifest.read_text())["files"]
            if "program" in name
        ),
        "runtime_query_source_path": next(
            manifest.parent / name
            for name in json.loads(manifest.read_text())["files"]
            if "query" in name
        ),
        "runtime_tokenizer_path": runtime_bundle_path,
        "runtime_execution_set_path": runtime_execution_set_path,
        "statistical_gate_spec_path": statistical_gate_spec_path,
        "access_registry_path": registry_path,
        "access_head_receipt_path": head_path,
        "registry_verification_key": public_key,
    }
    return kwargs, key, registry_path, custody_events


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
    assert set(report["by_semantic_action_active_prefix_accuracy"]) == {"[0,1,2]"}
    assert set(report["by_action_rank_active_prefix_accuracy"]) == {"3"}
    assert set(report["by_step_quartile_active_prefix_accuracy"]) == {"1", "3"}
    assert report["factorial_main_effects"]["semantic"]["held_out"] == 0.5
    assert report["family_scores"][0]["cluster_family_id"] == "f0"


def test_scorer_independently_resolves_nonidentity_opcode_binding() -> None:
    binding = [2, 0, 3, 1]
    opcode_schedule = [0, 1, 4] + [3] * 38
    resolved_schedule = [2, 0, 4] + [1] * 38
    oracle = {
        **_oracle(),
        "opcode_to_card": binding,
        "opcode_schedule": opcode_schedule,
        "schedule": resolved_schedule,
    }
    evidence = {
        **_evidence(),
        "predicted_opcode_to_card": binding,
        "predicted_opcode_schedule": opcode_schedule,
        "predicted_schedule": resolved_schedule,
    }
    report = score_evidence([evidence], [oracle])
    assert report["overall"]["independent_binding_exact"] == 1.0
    assert report["overall"]["opcode_schedule_exact"] == 1.0
    assert report["overall"]["schedule_exact"] == 1.0
    assert set(report["by_action_active_prefix_accuracy"]) == {"0", "1"}


def test_scorer_rejects_forged_resolved_schedule() -> None:
    evidence = _evidence()
    evidence["predicted_schedule"] = [1, 0, 4] + [3] * 38
    with pytest.raises(ValueError, match="committed resolved schedule"):
        score_evidence([evidence], [_oracle()])


def test_binding_error_cannot_hide_behind_matching_cards() -> None:
    oracle = _oracle()
    oracle["opcode_to_card"] = [2, 0, 3, 1]
    oracle["schedule"] = [2, 0, 4] + [1] * 38
    evidence = _evidence()
    report = score_evidence([evidence], [oracle])
    assert report["overall"]["cards_exact"] == 1.0
    assert report["overall"]["independent_binding_exact"] == 0.0
    assert report["overall"]["schedule_exact"] == 0.0
    assert report["overall"]["program_exact"] == 0.0


def test_assessor_requires_signed_access_before_oracle_and_refuses_closed_reuse(
    tmp_path,
    monkeypatch,
) -> None:
    board = tmp_path / "board"
    board.mkdir()
    oracle = board / "development_oracle.jsonl"
    _, oracle_sha = write_jsonl_once(oracle, [_oracle()], mode=0o400)
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
    manifest = board / "manifest.json"
    write_json_once(
        manifest,
        {
            "schema": "r12_ctaa_v2_manifest_v2",
            "seed": 1,
            "files": {
                oracle.name: oracle_sha,
                program.name: program_sha,
                query.name: query_sha,
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
    custody, signing_key, registry_path, custody_events = _signed_access(
        tmp_path,
        monkeypatch,
        manifest=manifest,
        evidence=evidence,
        oracle=oracle,
    )
    with pytest.raises(ValueError, match="core checkpoint"):
        assess(
            manifest_path=manifest,
            **custody,
            partition="development",
            runs=[("arm", evidence, oracle)],
            output_path=tmp_path / "mislabeled.json",
            run_metadata={"arm": {"seed": 2, "arm": "ctaa_closure", "dataset": "base"}},
        )
    custody_events.clear()
    original_load_oracle = assessment.load_oracle

    def load_after_spend(path, partition, *, expected_sha256=None):
        custody_events.append("oracle_read")
        state = access_registry.verify_registry(
            registry_path,
            custody["registry_verification_key"],
            expected_head_receipt=json.loads(
                custody["access_head_receipt_path"].read_text()
            ),
        )
        assert state.open_access_id == "development-access"
        return original_load_oracle(path, partition, expected_sha256=expected_sha256)

    monkeypatch.setattr(assessment, "load_oracle", load_after_spend)
    immutable_reads: dict[Path, int] = {}
    original_immutable_read = assessment._read_immutable_bytes

    def count_immutable_reads(path, label):
        resolved = Path(path)
        immutable_reads[resolved] = immutable_reads.get(resolved, 0) + 1
        return original_immutable_read(resolved, label)

    monkeypatch.setattr(assessment, "_read_immutable_bytes", count_immutable_reads)
    report = assess(
        manifest_path=manifest,
        **custody,
        partition="development",
        runs=[("arm", evidence, oracle)],
        output_path=tmp_path / "report.json",
        run_metadata={"arm": {"seed": 1, "arm": "ctaa_closure", "dataset": "base"}},
    )
    assert report["runs"]["arm"]["scores"]["overall"]["answer_exact"] == 1.0
    assert custody_events[:3] == [
        "execution_set",
        "query_bundle_replay",
        "oracle_read",
    ]
    assert report["access"]["schema"] == "r12_ctaa_v2_assessment_access_v7"
    assert report["access"]["execution_set_file_sha256"] == "4" * 64
    assert report["access"]["execution_set_sha256"] == "3" * 64
    assert report["access"]["statistical_gate_spec_file_sha256"] == sha256_file(
        custody["statistical_gate_spec_path"]
    )
    assert report["access"]["gate_spec_sha256"] == json.loads(
        custody["statistical_gate_spec_path"].read_text()
    )["gate_spec_sha256"]
    for removed in (
        "execution_receipt_sha256",
        "execution_aggregate_sha256",
        "execution_sha256",
        "execution_projection_sha256",
    ):
        assert removed not in report["access"]
    assert immutable_reads[custody["runtime_bundle_path"]] == 1
    assert immutable_reads[custody["access_head_receipt_path"]] == 1
    assert immutable_reads[evidence / "receipt.json"] == 1
    spend_head = json.loads(custody["access_head_receipt_path"].read_text())
    access_registry.append_assessment_commit(
        registry_path,
        signing_key=signing_key,
        registry_id="assessment-test-registry",
        event_id="development-assessment",
        access_id="development-access",
        assessment_sha256=report["report_sha256"],
        statistical_gate_spec_file_sha256=report["access"][
            "statistical_gate_spec_file_sha256"
        ],
        gate_spec_sha256=report["access"]["gate_spec_sha256"],
        expected_previous_hash=spend_head["payload"]["entry_hash"],
        expected_head_receipt=spend_head,
    )
    with pytest.raises(ValueError, match="retained receipt|signed access"):
        assess(
            manifest_path=manifest,
            **custody,
            partition="development",
            runs=[("arm", evidence, oracle)],
            output_path=tmp_path / "second.json",
            run_metadata={"arm": {"seed": 1, "arm": "ctaa_closure", "dataset": "base"}},
        )


def test_family_set_mismatch_is_fatal() -> None:
    with pytest.raises(ValueError, match="family set"):
        score_evidence([_evidence()], [_oracle("other")])


def test_committed_evidence_rejects_writable_or_symlinked_inputs(tmp_path) -> None:
    evidence = tmp_path / "evidence"
    _evidence_dir(evidence, [_evidence()])
    evidence_path = evidence / "evidence.jsonl"
    evidence_path.chmod(0o644)
    with pytest.raises(ValueError, match="single-link immutable"):
        load_committed_evidence_bundle(evidence)
    evidence_path.chmod(0o444)
    backing = evidence / "evidence-backing.jsonl"
    evidence_path.rename(backing)
    evidence_path.symlink_to(backing.name)
    with pytest.raises(ValueError, match="single-link immutable"):
        load_committed_evidence_bundle(evidence)


def test_oracle_rejects_hardlinks(tmp_path) -> None:
    oracle = tmp_path / "oracle.jsonl"
    write_jsonl_once(oracle, [_oracle()], mode=0o600)
    oracle.chmod(0o444)
    oracle.with_name("oracle-alias.jsonl").hardlink_to(oracle)
    with pytest.raises(ValueError, match="single-link immutable"):
        load_oracle(oracle, "development")


def test_immutable_reader_rejects_symlinked_parent_directory(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    path = real / "input.json"
    path.write_text("{}\n")
    path.chmod(0o444)
    linked = tmp_path / "linked"
    linked.symlink_to(real, target_is_directory=True)
    with pytest.raises(ValueError, match="parent is missing or symlinked"):
        load_oracle(linked / path.name, "development")


def test_immutable_reader_rejects_in_place_mutation(tmp_path, monkeypatch) -> None:
    path = tmp_path / "large-receipt.json"
    path.write_bytes(b"x" * (2 * 1024 * 1024))
    path.chmod(0o400)
    original_read = os.read
    mutated = False

    def mutate_after_first_chunk(descriptor, size):
        nonlocal mutated
        chunk = original_read(descriptor, size)
        if chunk and not mutated:
            mutated = True
            path.chmod(0o600)
            path.chmod(0o400)
        return chunk

    monkeypatch.setattr(assessment.os, "read", mutate_after_first_chunk)
    with pytest.raises(ValueError, match="changed while being read"):
        assessment._read_immutable_bytes(path, "adversarial receipt")


def test_runtime_bundle_digest_and_content_share_one_snapshot(
    tmp_path, monkeypatch
) -> None:
    path = tmp_path / "runtime-bundle.json"
    original = {"entries": [], "partition": "development"}
    path.write_bytes(assessment._canonical_json_bytes(original))
    path.chmod(0o400)
    expected_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    reads = 0
    original_read = assessment._read_immutable_bytes

    def count_reads(candidate, label):
        nonlocal reads
        if Path(candidate) == path:
            reads += 1
        return original_read(candidate, label)

    monkeypatch.setattr(assessment, "_read_immutable_bytes", count_reads)
    monkeypatch.setattr(
        assessment,
        "validate_runtime_bundle",
        lambda value, **_kwargs: dict(value),
    )
    monkeypatch.setattr(
        assessment,
        "make_runtime_bundle",
        lambda **_kwargs: dict(original),
    )
    value, digest = assessment._load_runtime_bundle_with_replay_and_sha(
        path,
        run_contract={},
        program_path=tmp_path / "program.jsonl",
        query_path=tmp_path / "query.jsonl",
        tokenizer_path=tmp_path / "tokenizer.json",
    )
    assert value == original
    assert digest == expected_sha256
    assert reads == 1


def test_failed_execution_set_prevents_query_replay_and_oracle_access(
    tmp_path, monkeypatch
) -> None:
    events: list[str] = []
    contract = {
        "partition": "development",
        "run_contract_sha256": "c" * 64,
    }
    monkeypatch.setattr(
        assessment,
        "validate_run_contract",
        lambda **_kwargs: contract,
    )

    def fail_execution_set(*_args, **_kwargs):
        events.append("execution_set")
        raise assessment.RuntimeExecutionSetError("forged member")

    def forbidden_query_replay(*_args, **_kwargs):
        events.append("query_bundle_replay")
        raise AssertionError("query replay occurred after failed execution set")

    def forbidden_oracle(*_args, **_kwargs):
        events.append("oracle_read")
        raise AssertionError("oracle read occurred after failed execution set")

    monkeypatch.setattr(
        assessment,
        "read_runtime_execution_set_with_replay",
        fail_execution_set,
    )
    monkeypatch.setattr(
        assessment,
        "_load_runtime_bundle_with_replay_and_sha",
        forbidden_query_replay,
    )
    monkeypatch.setattr(assessment, "load_oracle", forbidden_oracle)

    with pytest.raises(ValueError, match="execution set differs"):
        assessment._validate_preaccess_custody(
            manifest_path=tmp_path / "manifest.json",
            run_plan_path=tmp_path / "run-plan.json",
            run_contract_path=tmp_path / "run-contract.json",
            bootstrap_seed_receipt_path=tmp_path / "bootstrap.json",
            runtime_bundle_path=tmp_path / "runtime-bundle.json",
            runtime_program_source_path=tmp_path / "program.jsonl",
            runtime_query_source_path=tmp_path / "query.jsonl",
                runtime_tokenizer_path=tmp_path / "tokenizer.json",
                runtime_execution_set_path=tmp_path / "execution-set.json",
                statistical_gate_spec_path=tmp_path / "statistical-gate-spec.json",
            access_registry_path=tmp_path / "access.jsonl",
            access_head_receipt_path=tmp_path / "head.json",
            registry_verification_key=b"k" * 32,
            partition="development",
        )
    assert events == ["execution_set"]


def test_execution_set_barrier_precedes_committed_evidence_open(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        assessment,
        "_validate_preaccess_custody",
        lambda **_kwargs: (_ for _ in ()).throw(
            ValueError("execution set differs")
        ),
    )
    monkeypatch.setattr(
        assessment,
        "_load_committed_evidence_bundle_with_receipt_sha",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("committed evidence opened before execution-set barrier")
        ),
    )
    with pytest.raises(ValueError, match="execution set differs"):
        assess(
            manifest_path=tmp_path / "manifest.json",
            run_plan_path=tmp_path / "missing-plan",
            run_contract_path=tmp_path / "missing-contract",
            bootstrap_seed_receipt_path=tmp_path / "missing-bootstrap",
            runtime_bundle_path=tmp_path / "missing-runtime-bundle",
            runtime_program_source_path=tmp_path / "missing-runtime-program",
            runtime_query_source_path=tmp_path / "missing-runtime-query",
            runtime_tokenizer_path=tmp_path / "missing-runtime-tokenizer",
            runtime_execution_set_path=tmp_path / "missing-execution-set",
            statistical_gate_spec_path=tmp_path / "missing-statistical-gate-spec",
            access_registry_path=tmp_path / "missing-registry",
            access_head_receipt_path=tmp_path / "missing-head",
            registry_verification_key=b"x" * 32,
            partition="development",
            runs=[
                (
                    "arm",
                    tmp_path / "evidence",
                    tmp_path / "development_oracle.jsonl",
                )
            ],
            output_path=tmp_path / "report.json",
            run_metadata={"arm": {"seed": 1, "arm": "ctaa_closure", "dataset": "base"}},
        )


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
