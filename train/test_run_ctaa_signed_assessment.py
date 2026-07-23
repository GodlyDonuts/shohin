from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

import ctaa_access_registry as registry
import run_ctaa_signed_assessment as authority


def _hash(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _write_read_only(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    path.chmod(0o444)


def _write_json(path: Path, value: object) -> None:
    _write_read_only(path, (json.dumps(value, sort_keys=True) + "\n").encode())


def _public(key: Ed25519PrivateKey) -> bytes:
    return key.public_key().public_bytes(
        serialization.Encoding.Raw,
        serialization.PublicFormat.Raw,
    )


def _fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    board = tmp_path / "board"
    board.mkdir()
    manifest = board / "manifest.json"
    manifest_value = {
        "schema": "r12_ctaa_v2_manifest_v1",
        "seed": 11,
        "files": {
            "development_program.jsonl": _hash("program"),
            "development_query.jsonl": _hash("query"),
            "development_oracle.jsonl": _hash("oracle"),
        },
    }
    _write_json(manifest, manifest_value)

    evidence = tmp_path / "evidence" / "base"
    evidence.mkdir(parents=True)
    receipt = evidence / "receipt.json"
    _write_json(receipt, {"fixture": True})
    evidence.chmod(0o555)
    plan = tmp_path / "custody" / "run-plan.json"
    plan_value = {
        "schema": "r12_ctaa_v2_run_plan_v1",
        "partition": "development",
        "expected_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "expected_board_sha256": _hash("board"),
        "runs": [
            {
                "schema": "r12_ctaa_v2_run_input_v1",
                "seed": 1,
                "arm": "ctaa_closure",
                "dataset": "base",
                "evidence_receipt_path": str(receipt),
                "parent_evidence_receipt_path": None,
                "core_training": {},
            }
        ],
    }
    _write_json(plan, plan_value)
    bootstrap = tmp_path / "custody" / "bootstrap.json"
    _write_json(bootstrap, {"fixture": "bootstrap"})
    runtime_bundle = tmp_path / "custody" / "runtime-bundle.json"
    _write_json(runtime_bundle, {"fixture": "runtime-bundle"})
    run_id = "seed-1:ctaa_closure:base"
    contract = {
        "schema": "r12_ctaa_v2_run_contract_v1",
        "partition": "development",
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "board_sha256": _hash("board"),
        "run_plan_sha256": hashlib.sha256(plan.read_bytes()).hexdigest(),
        "bootstrap_seed_receipt_sha256": hashlib.sha256(
            bootstrap.read_bytes()
        ).hexdigest(),
        "bootstrap_seed": 123456789,
        "training_seeds": [1, 2, 3, 4, 5],
        "arms": ["ctaa_closure"],
        "datasets": ["base"],
        "run_count": 1,
        "oracle_files": {
            "base": {
                "filename": "development_oracle.jsonl",
                "sha256": _hash("oracle"),
            }
        },
        "runs": [
            {
                "run_id": run_id,
                "seed": 1,
                "arm": "ctaa_closure",
                "dataset": "base",
                "sealed_sources": {
                    "oracle_filename": "development_oracle.jsonl",
                },
            }
        ],
        "run_contract_sha256": _hash("contract"),
    }
    contract_path = tmp_path / "custody" / "run-contract.json"
    _write_json(contract_path, contract)
    custody_events: list[str] = []
    monkeypatch.setattr(authority, "validate_run_contract", lambda **_kwargs: contract)

    def fake_runtime_bundle(_path, **kwargs):
        custody_events.append("query_bundle_replay")
        return {
            "partition": kwargs["run_contract"]["partition"],
            "bundle_sha256": _hash("logical-runtime-bundle"),
        }

    monkeypatch.setattr(
        authority,
        "read_runtime_bundle_with_replay",
        fake_runtime_bundle,
    )
    runtime_program = board / "runtime-program.jsonl"
    runtime_query = board / "runtime-query.jsonl"
    runtime_tokenizer = tmp_path / "custody" / "runtime-tokenizer.json"
    _write_read_only(runtime_program, b"{}\n")
    _write_read_only(runtime_query, b"{}\n")
    _write_read_only(runtime_tokenizer, b"{}\n")
    runtime_execution_set = tmp_path / "custody" / "runtime-execution-set.json"
    assessment_source_bundle = tmp_path / "custody" / "assessor.pyz"
    assessment_source_manifest = tmp_path / "custody" / "assessor.manifest.json"
    python_executable = tmp_path / "custody" / "python"
    bwrap_executable = tmp_path / "custody" / "bwrap"
    for path, value in (
        (runtime_execution_set, {"fixture": "execution-set"}),
        (assessment_source_manifest, {"fixture": "source-manifest"}),
    ):
        _write_json(path, value)
    _write_read_only(assessment_source_bundle, b"fixture-assessor\n")
    _write_read_only(python_executable, b"fixture-python\n")
    _write_read_only(bwrap_executable, b"fixture-bwrap\n")
    python_executable.chmod(0o555)
    bwrap_executable.chmod(0o555)

    key = Ed25519PrivateKey.from_private_bytes(b"q" * 32)
    private_key = tmp_path / "external-secrets" / "registry.key"
    public_key = tmp_path / "custody" / "registry.pub"
    _write_read_only(private_key, b"q" * 32)
    _write_read_only(public_key, _public(key))
    writable = tmp_path / "assessment-output"
    writable.mkdir()
    custody = tmp_path / "authority"
    custody.mkdir()
    config = authority.AssessmentAuthorityConfig(
        manifest_path=manifest,
        run_plan_path=plan,
        run_contract_path=contract_path,
        bootstrap_seed_receipt_path=bootstrap,
        runtime_bundle_path=runtime_bundle,
        runtime_program_source_path=runtime_program,
        runtime_query_source_path=runtime_query,
        runtime_tokenizer_path=runtime_tokenizer,
        runtime_execution_set_path=runtime_execution_set,
        assessment_source_bundle_path=assessment_source_bundle,
        assessment_source_manifest_path=assessment_source_manifest,
        statistical_gate_spec_path=custody / "statistical-gate-spec.json",
        python_executable_path=python_executable,
        bwrap_executable_path=bwrap_executable,
        registry_path=custody / "access.jsonl",
        registry_public_key_path=public_key,
        registry_private_key_path=private_key,
        previous_head_receipt_path=None,
        claim_path=custody / "claim.json",
        spend_head_receipt_path=custody / "spend-head.json",
        commit_head_receipt_path=custody / "commit-head.json",
        assessment_output_path=writable / "assessment.json",
        writable_root=writable,
        board_root=board,
        registry_id="test-registry",
        access_id="development-access-1",
        spend_event_id="development-spend-1",
        commit_event_id="development-assessment-1",
        partition="development",
        timeout_seconds=30,
    )
    sandbox_calls: list[tuple[list[str], Path, Path]] = []

    def fake_hidden(
        command: list[str], *, writable_root: Path, board_root: Path
    ) -> list[str]:
        sandbox_calls.append((list(command), writable_root, board_root))
        return [str(bwrap_executable), "--", *command]

    monkeypatch.setattr(authority, "hidden_board_command", fake_hidden)
    execution_set = {
        "partition": "development",
        "run_contract_sha256": contract["run_contract_sha256"],
        "runtime_bundle_file_sha256": hashlib.sha256(
            runtime_bundle.read_bytes()
        ).hexdigest(),
        "execution_set_sha256": _hash("execution-set"),
    }

    def fake_execution_set(
        path, *, runtime_bundle_path, run_contract, verification_key
    ):
        assert Path(path) == runtime_execution_set
        assert Path(runtime_bundle_path) == runtime_bundle
        assert run_contract is contract
        assert verification_key == _public(key)
        custody_events.append("execution_set")
        return execution_set, _hash("execution-set-file")

    monkeypatch.setattr(
        authority, "read_runtime_execution_set_with_replay", fake_execution_set
    )
    source_manifest = {
        "bundle_sha256": _hash("assessment-source-bundle"),
        "python_interpreter": {"sha256": _hash("python")},
        "bwrap_executable": {"sha256": _hash("bwrap")},
    }

    def fake_load_assessor(_config):
        descriptor = os.open(assessment_source_bundle, os.O_RDONLY)
        os.set_inheritable(descriptor, True)
        return descriptor, source_manifest, _hash("assessment-source-manifest")

    monkeypatch.setattr(authority, "_load_sealed_assessor", fake_load_assessor)
    return {
        "config": config,
        "contract": contract,
        "key": key,
        "public": _public(key),
        "run_id": run_id,
        "sandbox_calls": sandbox_calls,
        "custody_events": custody_events,
        "execution_set": execution_set,
    }


def _valid_report(paths: dict[str, object]) -> dict[str, object]:
    config = paths["config"]
    assert isinstance(config, authority.AssessmentAuthorityConfig)
    contract = paths["contract"]
    assert isinstance(contract, dict)
    public = paths["public"]
    assert isinstance(public, bytes)
    spend_receipt = json.loads(config.spend_head_receipt_path.read_text())
    state = registry.verify_registry(
        config.registry_path,
        public,
        expected_head_receipt=spend_receipt,
    )
    event = registry.verify_registry_events(
        config.registry_path,
        public,
        expected_head_receipt=spend_receipt,
    )[-1]
    run_id = str(paths["run_id"])
    return {
        "schema": authority.ASSESSMENT_SCHEMA,
        "partition": "development",
        "manifest_sha256": contract["manifest_sha256"],
        "access": {
            "schema": authority.ASSESSMENT_ACCESS_SCHEMA,
            "registry_id": config.registry_id,
            "registry_head_receipt_sha256": hashlib.sha256(
                config.spend_head_receipt_path.read_bytes()
            ).hexdigest(),
            "registry_head_entry_hash": state.head_hash,
            "access_event_payload_sha256": hashlib.sha256(
                event.canonical_payload
            ).hexdigest(),
            "access_id": config.access_id,
            "partition": config.partition,
            "manifest_sha256": contract["manifest_sha256"],
            "board_sha256": contract["board_sha256"],
            "run_contract_sha256": contract["run_contract_sha256"],
            "runtime_bundle_sha256": hashlib.sha256(
                config.runtime_bundle_path.read_bytes()
            ).hexdigest(),
            "assessment_claim_sha256": hashlib.sha256(
                config.claim_path.read_bytes()
            ).hexdigest(),
            "execution_set_file_sha256": _hash("execution-set-file"),
            "execution_set_sha256": _hash("execution-set"),
            "statistical_gate_spec_file_sha256": hashlib.sha256(
                config.statistical_gate_spec_path.read_bytes()
            ).hexdigest(),
            "gate_spec_sha256": json.loads(
                config.statistical_gate_spec_path.read_text()
            )["gate_spec_sha256"],
            "bootstrap_seed_receipt_sha256": contract["bootstrap_seed_receipt_sha256"],
            "bootstrap_seed": contract["bootstrap_seed"],
            "access": 1,
        },
        "oracle_sha256": {run_id: _hash("oracle")},
        "runs": {run_id: {"scores": {}}},
        "capability_gate_computed": False,
    }


def _install_child(
    paths: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
    *,
    returncode: int = 0,
    mutate_report=None,
    writable_output: bool = False,
) -> list[tuple[list[str], dict[str, object]]]:
    calls: list[tuple[list[str], dict[str, object]]] = []

    def fake_run(command: list[str], **kwargs: object) -> SimpleNamespace:
        calls.append((list(command), dict(kwargs)))
        config = paths["config"]
        assert isinstance(config, authority.AssessmentAuthorityConfig)
        child_command = command[command.index("--") + 1 :]
        assert str(config.registry_private_key_path) not in child_command
        assert b"q" * 32 not in [item.encode() for item in child_command]
        assert kwargs.get("stdin") is authority.subprocess.DEVNULL
        assert kwargs.get("close_fds") is True
        pass_fds = kwargs.get("pass_fds")
        assert isinstance(pass_fds, tuple) and len(pass_fds) == 1
        assert f"/proc/self/fd/{pass_fds[0]}" in child_command
        assert kwargs.get("env") == {
            "PATH": os.environ.get("PATH", ""),
            "LANG": "C",
            "LC_ALL": "C",
            "PYTHONHASHSEED": "0",
        }
        if returncode == 0 or mutate_report is not None:
            report = _valid_report(paths)
            if mutate_report is not None:
                mutate_report(report)
            config.assessment_output_path.write_text(
                json.dumps(report, sort_keys=True, indent=2) + "\n"
            )
            config.assessment_output_path.chmod(0o644 if writable_output else 0o444)
        return SimpleNamespace(returncode=returncode, stdout=b"", stderr=b"")

    monkeypatch.setattr(authority.subprocess, "run", fake_run)
    return calls


def test_success_spends_runs_once_and_hash_binds_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    calls = _install_child(paths, monkeypatch)
    config = paths["config"]
    assert isinstance(config, authority.AssessmentAuthorityConfig)

    result = authority.run_signed_assessment(config)

    assert paths["custody_events"] == [
        "execution_set",
        "query_bundle_replay",
    ]
    assert len(calls) == 1
    assert len(paths["sandbox_calls"]) == 1
    command, kwargs = calls[0]
    assert "shell" not in kwargs
    separator = command.index("--")
    preamble = command[:separator]
    assert str(config.manifest_path) in preamble
    assert str(config.board_root / "development_oracle.jsonl") in preamble
    private_index = preamble.index(str(config.registry_private_key_path))
    assert preamble[private_index - 2 : private_index] == ["--ro-bind", "/dev/null"]
    child = command[separator + 1 :]
    assert child[0] == str(config.python_executable_path)
    assert child[1].startswith("/proc/self/fd/")
    assert child[child.index("--runtime-execution-set") + 1] == str(
        config.runtime_execution_set_path
    )
    assert child.count("--runtime-execution-set") == 1
    assert child[child.index("--statistical-gate-spec") + 1] == str(
        config.statistical_gate_spec_path
    )
    assert child.count("--statistical-gate-spec") == 1
    for removed in (
        "--runtime-intervention-plan",
        "--execution-projection",
        "--execution-aggregate",
        "--execution-artifact-directory",
        "--execution-receipt",
    ):
        assert removed not in child
    assert child.count("--output") == 1
    assert str(config.registry_private_key_path) not in child

    public = paths["public"]
    assert isinstance(public, bytes)
    commit_receipt = json.loads(config.commit_head_receipt_path.read_text())
    final = registry.verify_registry(
        config.registry_path,
        public,
        expected_head_receipt=commit_receipt,
    )
    event = registry.verify_registry_events(
        config.registry_path,
        public,
        expected_head_receipt=commit_receipt,
    )[-1]
    assert final.open_access_id is None
    assert final.head_event_type == registry.ASSESSMENT_COMMIT
    assert (
        event.payload["assessment_sha256"]
        == hashlib.sha256(config.assessment_output_path.read_bytes()).hexdigest()
    )
    assert result.assessment_sha256 == event.payload["assessment_sha256"]
    for path in (
        config.claim_path,
        config.spend_head_receipt_path,
        config.commit_head_receipt_path,
        config.statistical_gate_spec_path,
        config.assessment_output_path,
    ):
        assert path.stat().st_mode & 0o222 == 0
        assert path.stat().st_nlink == 1
    claim = json.loads(config.claim_path.read_text())
    assert set(claim) == {"payload", "signature"}
    assert str(config.registry_private_key_path) not in config.claim_path.read_text()
    assert (b"q" * 32).hex() not in config.claim_path.read_text()
    key = Ed25519PrivateKey.from_private_bytes(b"q" * 32).public_key()
    key.verify(
        bytes.fromhex(claim["signature"]),
        registry.canonical_json_bytes(claim["payload"]),
    )
    assert claim["payload"]["schema"] == "r12_ctaa_signed_assessment_claim_v5"
    assert claim["payload"]["execution_set_file_sha256"] == _hash("execution-set-file")
    assert claim["payload"]["execution_set_sha256"] == _hash("execution-set")
    gate_spec = json.loads(config.statistical_gate_spec_path.read_text())
    gate_spec_file_sha256 = hashlib.sha256(
        config.statistical_gate_spec_path.read_bytes()
    ).hexdigest()
    assert (
        claim["payload"]["statistical_gate_spec_file_sha256"]
        == gate_spec_file_sha256
    )
    assert claim["payload"]["gate_spec_sha256"] == gate_spec["gate_spec_sha256"]
    events = registry.verify_registry_events(
        config.registry_path,
        paths["public"],
        expected_head_receipt=commit_receipt,
    )
    for bound_event in events[-2:]:
        assert (
            bound_event.payload["statistical_gate_spec_file_sha256"]
            == gate_spec_file_sha256
        )
        assert (
            bound_event.payload["gate_spec_sha256"]
            == gate_spec["gate_spec_sha256"]
        )
    for removed in (
        "execution_receipt_sha256",
        "execution_aggregate_sha256",
        "execution_sha256",
        "execution_projection_sha256",
    ):
        assert removed not in claim["payload"]
    assert claim["payload"]["assessment_source_bundle_sha256"] == _hash(
        "assessment-source-bundle"
    )
    assert claim["payload"]["python_interpreter_sha256"] == _hash("python")
    assert claim["payload"]["bwrap_executable_sha256"] == _hash("bwrap")


def test_statistical_gate_spec_is_published_before_access_spend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    _install_child(paths, monkeypatch)
    config = paths["config"]
    assert isinstance(config, authority.AssessmentAuthorityConfig)
    original_append = authority.append_access_spend
    observed: list[str] = []

    def checked_append(*args, **kwargs):
        assert config.statistical_gate_spec_path.exists()
        assert config.statistical_gate_spec_path.stat().st_mode & 0o222 == 0
        record = json.loads(config.statistical_gate_spec_path.read_text())
        assert (
            hashlib.sha256(config.statistical_gate_spec_path.read_bytes()).hexdigest()
            == kwargs["statistical_gate_spec_file_sha256"]
        )
        assert record["gate_spec_sha256"] == kwargs["gate_spec_sha256"]
        observed.append("spec-before-spend")
        return original_append(*args, **kwargs)

    monkeypatch.setattr(authority, "append_access_spend", checked_append)
    authority.run_signed_assessment(config)
    assert observed == ["spec-before-spend"]


def test_failed_execution_set_blocks_query_oracle_claim_and_access_spend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    config = paths["config"]
    assert isinstance(config, authority.AssessmentAuthorityConfig)

    child_calls: list[object] = []

    def fail_set(*_args, **_kwargs):
        paths["custody_events"].append("execution_set")
        raise authority.RuntimeExecutionSetError("forged")

    def forbidden_child(*_args, **_kwargs):
        child_calls.append((_args, _kwargs))
        raise AssertionError("oracle-bearing assessor launched after failed set")

    monkeypatch.setattr(authority, "read_runtime_execution_set_with_replay", fail_set)
    monkeypatch.setattr(authority.subprocess, "run", forbidden_child)
    with pytest.raises(authority.SignedAssessmentError, match="before query release"):
        authority.run_signed_assessment(config)
    assert paths["custody_events"] == ["execution_set"]
    assert child_calls == []
    assert not config.claim_path.exists()
    assert not config.registry_path.exists()
    assert not config.spend_head_receipt_path.exists()


def test_execution_set_runtime_bundle_substitution_fails_before_query_or_spend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    config = paths["config"]
    execution_set = paths["execution_set"]
    assert isinstance(config, authority.AssessmentAuthorityConfig)
    assert isinstance(execution_set, dict)
    execution_set["runtime_bundle_file_sha256"] = _hash("substituted-bundle")

    with pytest.raises(authority.SignedAssessmentError, match="file binding differs"):
        authority.run_signed_assessment(config)
    assert paths["custody_events"] == ["execution_set"]
    assert not config.claim_path.exists()
    assert not config.registry_path.exists()


def test_child_failure_leaves_open_access_and_claim_blocks_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    calls = _install_child(paths, monkeypatch, returncode=17)
    config = paths["config"]
    assert isinstance(config, authority.AssessmentAuthorityConfig)

    with pytest.raises(authority.ChildAssessmentError, match="remains open"):
        authority.run_signed_assessment(config)

    assert len(calls) == 1
    public = paths["public"]
    assert isinstance(public, bytes)
    spend_receipt = json.loads(config.spend_head_receipt_path.read_text())
    state = registry.verify_registry(
        config.registry_path,
        public,
        expected_head_receipt=spend_receipt,
    )
    assert state.open_access_id == config.access_id
    assert not config.commit_head_receipt_path.exists()
    assert not config.assessment_output_path.exists()
    with pytest.raises(FileExistsError, match="authority output"):
        authority.run_signed_assessment(config)
    assert len(calls) == 1


def test_unsuccessful_child_output_never_creates_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    _install_child(paths, monkeypatch, returncode=3, mutate_report=lambda _report: None)
    config = paths["config"]
    assert isinstance(config, authority.AssessmentAuthorityConfig)
    with pytest.raises(authority.ChildAssessmentError):
        authority.run_signed_assessment(config)
    public = paths["public"]
    assert isinstance(public, bytes)
    assert (
        registry.verify_registry(config.registry_path, public).open_access_id
        == config.access_id
    )
    assert config.assessment_output_path.exists()
    assert not config.commit_head_receipt_path.exists()


@pytest.mark.parametrize(
    "mutation",
    (
        lambda report: report["access"].__setitem__(
            "registry_head_receipt_sha256", _hash("wrong-head")
        ),
        lambda report: report["access"].__setitem__(
            "run_contract_sha256", _hash("wrong-contract")
        ),
        lambda report: report["oracle_sha256"].__setitem__(
            "seed-1:ctaa_closure:base", _hash("wrong-oracle")
        ),
    ),
)
def test_mismatched_assessment_bindings_leave_registry_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation,
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    _install_child(paths, monkeypatch, mutate_report=mutation)
    config = paths["config"]
    assert isinstance(config, authority.AssessmentAuthorityConfig)
    with pytest.raises(authority.SignedAssessmentError, match="differs"):
        authority.run_signed_assessment(config)
    public = paths["public"]
    assert isinstance(public, bytes)
    assert (
        registry.verify_registry(config.registry_path, public).open_access_id
        == config.access_id
    )
    assert not config.commit_head_receipt_path.exists()


def test_writable_assessment_is_rejected_without_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    _install_child(paths, monkeypatch, writable_output=True)
    config = paths["config"]
    assert isinstance(config, authority.AssessmentAuthorityConfig)
    with pytest.raises(authority.SignedAssessmentError, match="read-only"):
        authority.run_signed_assessment(config)
    assert not config.commit_head_receipt_path.exists()


def test_preexisting_output_rejected_before_claim_or_spend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    config = paths["config"]
    assert isinstance(config, authority.AssessmentAuthorityConfig)
    config.assessment_output_path.write_text("occupied")
    with pytest.raises(FileExistsError, match="authority output"):
        authority.run_signed_assessment(config)
    assert not config.claim_path.exists()
    assert not config.registry_path.exists()


@pytest.mark.parametrize(
    "target", ("run_contract", "bootstrap", "public_key", "private_key")
)
def test_writable_custody_inputs_are_rejected_before_spend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, target: str
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    config = paths["config"]
    assert isinstance(config, authority.AssessmentAuthorityConfig)
    path = {
        "run_contract": config.run_contract_path,
        "bootstrap": config.bootstrap_seed_receipt_path,
        "public_key": config.registry_public_key_path,
        "private_key": config.registry_private_key_path,
    }[target]
    path.chmod(0o644)
    with pytest.raises(authority.SignedAssessmentError, match="read-only"):
        authority.run_signed_assessment(config)
    assert not config.claim_path.exists()
    assert not config.registry_path.exists()


def test_symlink_and_hardlink_contracts_are_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    config = paths["config"]
    assert isinstance(config, authority.AssessmentAuthorityConfig)
    symlink = tmp_path / "custody" / "contract-link.json"
    symlink.symlink_to(config.run_contract_path)
    with pytest.raises(authority.SignedAssessmentError, match="symlink"):
        authority.run_signed_assessment(replace(config, run_contract_path=symlink))

    hardlink = tmp_path / "custody" / "contract-hard.json"
    os.link(config.run_contract_path, hardlink)
    with pytest.raises(authority.SignedAssessmentError, match="single-link"):
        authority.run_signed_assessment(replace(config, run_contract_path=hardlink))
    assert not config.claim_path.exists()
    assert not config.registry_path.exists()


def test_runtime_bundle_cannot_be_hidden_inside_sealed_board(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    config = paths["config"]
    assert isinstance(config, authority.AssessmentAuthorityConfig)
    hidden = config.board_root / "runtime-bundle.json"
    _write_json(hidden, {"fixture": "runtime-bundle"})
    with pytest.raises(authority.SignedAssessmentError, match="sealed board"):
        authority.run_signed_assessment(replace(config, runtime_bundle_path=hidden))
    assert not config.claim_path.exists()
    assert not config.registry_path.exists()


def test_mismatched_retained_head_is_rejected_before_new_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    config = paths["config"]
    key = paths["key"]
    public = paths["public"]
    assert isinstance(config, authority.AssessmentAuthorityConfig)
    assert isinstance(key, Ed25519PrivateKey)
    assert isinstance(public, bytes)
    contract = paths["contract"]
    assert isinstance(contract, dict)
    stale = registry.append_access_spend(
        config.registry_path,
        signing_key=key,
        registry_id=config.registry_id,
        event_id="old-spend",
        access_id="old-access",
        partition="development",
        manifest_sha256=str(contract["manifest_sha256"]),
        board_sha256=str(contract["board_sha256"]),
        run_contract_sha256=str(contract["run_contract_sha256"]),
        runtime_bundle_sha256=hashlib.sha256(
            config.runtime_bundle_path.read_bytes()
        ).hexdigest(),
        assessment_claim_sha256="e" * 64,
        bootstrap_seed_receipt_sha256=str(contract["bootstrap_seed_receipt_sha256"]),
        bootstrap_seed=int(contract["bootstrap_seed"]),
        statistical_gate_spec_file_sha256=_hash("old-statistical-gate-spec-file"),
        gate_spec_sha256=_hash("old-statistical-gate-spec"),
        expected_previous_hash=registry.GENESIS_PREVIOUS_HASH,
    )
    spend_state = registry.verify_registry(config.registry_path, public)
    registry.append_assessment_commit(
        config.registry_path,
        signing_key=key,
        registry_id=config.registry_id,
        event_id="old-assessment",
        access_id="old-access",
        assessment_sha256=_hash("old-assessment"),
        statistical_gate_spec_file_sha256=_hash("old-statistical-gate-spec-file"),
        gate_spec_sha256=_hash("old-statistical-gate-spec"),
        expected_previous_hash=spend_state.head_hash,
        expected_head_receipt=stale,
    )
    stale_path = tmp_path / "authority" / "stale-head.json"
    _write_read_only(stale_path, registry.serialize_head_receipt(stale))
    with pytest.raises(registry.RegistryVerificationError, match="head differs"):
        authority.run_signed_assessment(
            replace(config, previous_head_receipt_path=stale_path)
        )
    assert not config.claim_path.exists()


def test_command_injection_identifier_is_rejected_without_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    calls = _install_child(paths, monkeypatch)
    config = paths["config"]
    assert isinstance(config, authority.AssessmentAuthorityConfig)
    marker = tmp_path / "injected"
    malicious = f"access;touch-{marker.name}"
    with pytest.raises(authority.SignedAssessmentError, match="unsafe access_id"):
        authority.run_signed_assessment(replace(config, access_id=malicious))
    assert calls == []
    assert not marker.exists()
    assert not config.claim_path.exists()


def test_signing_key_must_be_external_and_match_public_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    config = paths["config"]
    assert isinstance(config, authority.AssessmentAuthorityConfig)
    wrong = tmp_path / "external-secrets" / "wrong.key"
    _write_read_only(wrong, b"z" * 32)
    with pytest.raises(authority.SignedAssessmentError, match="keys differ"):
        authority.run_signed_assessment(
            replace(config, registry_private_key_path=wrong)
        )
    with pytest.raises(authority.SignedAssessmentError, match="absolute"):
        authority.run_signed_assessment(
            replace(config, registry_private_key_path=Path("relative.key"))
        )
    assert not config.claim_path.exists()


def test_selective_disclosure_rejects_non_bubblewrap_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    config = paths["config"]
    assert isinstance(config, authority.AssessmentAuthorityConfig)
    monkeypatch.setattr(
        authority,
        "hidden_board_command",
        lambda command, **_kwargs: ["sandbox-exec", "--", *command],
    )
    with pytest.raises(authority.SignedAssessmentError, match="bubblewrap"):
        authority.run_signed_assessment(config)
    assert not config.claim_path.exists()
    assert not config.registry_path.exists()
