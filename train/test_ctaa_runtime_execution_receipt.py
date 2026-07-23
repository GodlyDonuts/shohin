from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

import ctaa_runtime_execution_receipt as receipt
from ctaa_runtime_execution_artifact import RuntimeExecutionArtifactIndex
from ctaa_runtime_execution_projection import (
    make_execution_projection,
    write_execution_projection,
)
from test_ctaa_intervention_protocol import valid_plan


def _key(seed: int = 17) -> Ed25519PrivateKey:
    return Ed25519PrivateKey.from_private_bytes(bytes([seed]) * 32)


def _public(key: Ed25519PrivateKey) -> bytes:
    return key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


@pytest.fixture(scope="module")
def plan():
    return valid_plan()


@pytest.fixture(scope="module")
def projection(plan):
    return make_execution_projection(plan)


@pytest.fixture(scope="module")
def projection_path(tmp_path_factory: pytest.TempPathFactory, plan) -> Path:
    path = tmp_path_factory.mktemp("execution-receipt") / "projection.json"
    write_execution_projection(path, plan)
    return path


@pytest.fixture(scope="module")
def outputs(projection):
    return tuple(
        {
            "attempt_id": row["attempt_id"],
            "status": "failure" if index % 997 == 0 else "success",
            "raw_output_artifact_sha256": _hash(
                f"{row['attempt_id']}:{'failure' if index % 997 == 0 else 'success'}"
            ),
        }
        for index, row in enumerate(projection["attempts"])
    )


@pytest.fixture(scope="module", autouse=True)
def replayed_artifacts(projection, outputs):
    records = tuple(
        SimpleNamespace(
            attempt_index=row["attempt_index"],
            attempt_id=row["attempt_id"],
            operation=row["operation"],
            status=output["status"],
        )
        for row, output in zip(projection["attempts"], outputs)
    )
    result = SimpleNamespace(
        attempts=records,
        execution_sha256=_hash("execution"),
    )
    index = RuntimeExecutionArtifactIndex(
        _hash("aggregate"),
        (),
        tuple(row["raw_output_artifact_sha256"] for row in outputs),
        outputs,
    )
    original = receipt._load_execution_artifacts
    receipt._load_execution_artifacts = lambda **_: (result, index)
    try:
        yield result, index
    finally:
        receipt._load_execution_artifacts = original


def _artifact_kwargs() -> dict[str, object]:
    return {
        "execution_aggregate_path": Path("/artifact/aggregate.json"),
        "execution_artifact_directory": Path("/artifact/objects"),
        "execution_aggregate_sha256": _hash("aggregate"),
    }


@pytest.fixture(scope="module")
def signed(projection_path: Path, plan):
    return receipt.make_runtime_execution_receipt(
        execution_projection_path=projection_path,
        plan=plan,
        **_artifact_kwargs(),
        signing_key=_key(),
    )


def _resign(value: dict[str, object], key: Ed25519PrivateKey | None = None):
    signing_key = key or _key()
    payload = value["payload"]
    assert isinstance(payload, dict)
    signature = signing_key.sign(receipt.canonical_json_bytes(payload)).hex()
    return {
        "payload": payload,
        "signature": signature,
        "receipt_sha256": receipt._record_hash(payload, signature),
    }


def _validate(value, projection_path: Path, plan, key: Ed25519PrivateKey | None = None):
    return receipt.validate_runtime_execution_receipt(
        value,
        execution_projection_path=projection_path,
        plan=plan,
        **_artifact_kwargs(),
        verification_key=_public(key or _key()),
    )


def _immutable_raw(path: Path, raw: bytes) -> None:
    path.write_bytes(raw)
    path.chmod(0o400)


def test_complete_signed_receipt_is_strictly_pre_query(
    signed, projection, projection_path: Path, plan
):
    verified = _validate(signed, projection_path, plan)
    payload = verified["payload"]
    assert payload["oracle_access_count"] == 0
    assert payload["execution_attempt_count"] == len(projection["attempts"])
    assert any(row["status"] == "failure" for row in payload["attempts"])
    assert len(payload["source_deletion_probe_artifact_sha256s"]) == 864
    assert len(payload["query_isolation_probe_artifact_sha256s"]) == 864
    serialized = receipt.canonical_json_bytes(verified).decode("ascii").lower()
    for forbidden in (
        "answer",
        "query_position",
        "query_source",
        "late_query_swap",
    ):
        assert forbidden not in serialized
    assert "scientific_pass" not in serialized
    assert "advancement" not in serialized


def test_envelope_authentication_opens_no_projection_or_execution_artifact(
    signed: dict[str, object],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        receipt,
        "_load_projection",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("projection opened during envelope authentication")
        ),
    )
    monkeypatch.setattr(
        receipt,
        "_load_execution_artifacts",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("artifact opened during envelope authentication")
        ),
    )
    path = (tmp_path / "receipt.json").absolute()
    raw = receipt.canonical_json_bytes(signed) + b"\n"
    _immutable_raw(path, raw)
    verified, file_sha = receipt.read_runtime_execution_receipt_envelope_with_sha(
        path, verification_key=_public(_key())
    )
    assert verified == signed
    assert file_sha == hashlib.sha256(raw).hexdigest()


def test_round_trip_is_canonical_immutable_and_single_link(
    tmp_path: Path, projection_path: Path, plan, outputs
):
    path = tmp_path / "receipt.json"
    digest = receipt.write_runtime_execution_receipt(
        path,
        execution_projection_path=projection_path,
        plan=plan,
        **_artifact_kwargs(),
        signing_key=_key(),
    )
    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()
    assert path.stat().st_nlink == 1
    assert not path.stat().st_mode & 0o222
    verified = receipt.read_runtime_execution_receipt(
        path,
        execution_projection_path=projection_path,
        plan=plan,
        **_artifact_kwargs(),
        verification_key=_public(_key()),
    )
    assert verified["payload"]["execution_attempt_count"] == len(outputs)


@pytest.mark.parametrize(
    "field",
    [
        "execution_projection_file_sha256",
        "execution_projection_sha256",
        "plan_sha256",
        "board_manifest_sha256",
        "board_tree_sha256",
        "run_contract_sha256",
        "compiler_sha256",
        "core_sha256",
        "tokenizer_sha256",
        "base_checkpoint_sha256",
        "base_raw_evidence_receipt_sha256",
        "runtime_implementation_sha256",
        "selection_seed_receipt_sha256",
        "anchor_panel_sha256",
        "donor_registry_sha256",
        "batch_order_sha256",
    ],
)
def test_resigned_hash_substitutions_fail_projection_binding(
    signed, projection_path: Path, plan, field: str
):
    changed = deepcopy(signed)
    changed["payload"][field] = "f" * 64
    with pytest.raises(receipt.ExecutionReceiptError, match="projection contract"):
        _validate(_resign(changed), projection_path, plan)


@pytest.mark.parametrize("field", ["selection_seed", "training_seed"])
def test_resigned_seed_substitution_fails_projection_binding(
    signed, projection_path: Path, plan, field: str
):
    changed = deepcopy(signed)
    changed["payload"][field] += 1
    with pytest.raises(receipt.ExecutionReceiptError, match="projection contract"):
        _validate(_resign(changed), projection_path, plan)


@pytest.mark.parametrize(
    "leak",
    [
        {"answer": 3},
        {"query_position": 2},
        {"query_source_sha256": "a" * 64},
        {"late_query_swap": "enabled"},
    ],
)
def test_query_fields_cannot_be_smuggled_anywhere(
    signed, projection_path: Path, plan, leak
):
    changed = deepcopy(signed)
    changed["payload"]["attempts"][0].update(leak)
    with pytest.raises(receipt.ExecutionReceiptError, match="leaks"):
        _validate(_resign(changed), projection_path, plan)


def test_late_query_value_cannot_be_smuggled(signed, projection_path: Path, plan):
    changed = deepcopy(signed)
    changed["payload"]["attempts"][0]["operation"] = "late_query_swap"
    with pytest.raises(receipt.ExecutionReceiptError, match="leaks"):
        _validate(_resign(changed), projection_path, plan)


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "reordered"])
def test_no_missing_duplicate_or_reordered_attempts(
    signed, projection_path: Path, plan, mutation: str
):
    changed = deepcopy(signed)
    attempts = changed["payload"]["attempts"]
    if mutation == "missing":
        attempts.pop()
    elif mutation == "duplicate":
        attempts[-1] = deepcopy(attempts[-2])
        attempts[-1]["attempt_index"] = len(attempts) - 1
    else:
        attempts[0], attempts[1] = attempts[1], attempts[0]
        attempts[0]["attempt_index"] = 0
        attempts[1]["attempt_index"] = 1
    changed["payload"]["execution_attempt_count"] = len(attempts)
    changed["payload"]["attempts_sha256"] = receipt._sha256_json(attempts)
    changed["payload"]["execution_attempt_ids_sha256"] = receipt._sha256_json(
        [row["attempt_id"] for row in attempts]
    )
    with pytest.raises(receipt.ExecutionReceiptError):
        _validate(_resign(changed), projection_path, plan)


def test_unsigned_output_artifact_substitution_fails_signature(
    signed, projection_path: Path, plan
):
    changed = deepcopy(signed)
    changed["payload"]["attempts"][0]["raw_output_artifact_sha256"] = "a" * 64
    with pytest.raises(receipt.ExecutionReceiptError, match="signature verification"):
        _validate(changed, projection_path, plan)


def test_probe_hash_substitution_fails_signature(signed, projection_path: Path, plan):
    changed = deepcopy(signed)
    changed["payload"]["source_deletion_probe_artifact_sha256s"][0] = "b" * 64
    with pytest.raises(receipt.ExecutionReceiptError, match="signature verification"):
        _validate(changed, projection_path, plan)


def test_signature_mutation_is_rejected(signed, projection_path: Path, plan):
    changed = deepcopy(signed)
    changed["signature"] = ("0" if changed["signature"][0] != "0" else "1") + changed[
        "signature"
    ][1:]
    with pytest.raises(receipt.ExecutionReceiptError, match="signature verification"):
        _validate(changed, projection_path, plan)


def test_wrong_verification_key_is_rejected(signed, projection_path: Path, plan):
    with pytest.raises(receipt.ExecutionReceiptError, match="wrong signing key"):
        _validate(signed, projection_path, plan, _key(19))


def test_resigned_public_key_substitution_is_rejected(
    signed, projection_path: Path, plan
):
    attacker = _key(19)
    changed = deepcopy(signed)
    changed["payload"]["signing_public_key"] = _public(attacker).hex()
    with pytest.raises(receipt.ExecutionReceiptError, match="wrong signing key"):
        _validate(_resign(changed, attacker), projection_path, plan)


def test_non_prime_order_public_key_is_rejected(signed, projection_path: Path, plan):
    with pytest.raises(receipt.ExecutionReceiptError, match="prime-order"):
        receipt.validate_runtime_execution_receipt(
            signed,
            execution_projection_path=projection_path,
            plan=plan,
            **_artifact_kwargs(),
            verification_key=b"\x01" + b"\x00" * 31,
        )


def test_invalid_status_is_rejected_even_when_resigned(
    signed, projection_path: Path, plan
):
    changed = deepcopy(signed)
    changed["payload"]["attempts"][0]["status"] = "passed"
    with pytest.raises(receipt.ExecutionReceiptError, match="status"):
        _validate(_resign(changed), projection_path, plan)


def test_uppercase_hash_is_rejected_even_when_resigned(
    signed, projection_path: Path, plan
):
    changed = deepcopy(signed)
    changed["payload"]["attempts"][0]["raw_output_artifact_sha256"] = "A" * 64
    with pytest.raises(receipt.ExecutionReceiptError, match="raw output"):
        _validate(_resign(changed), projection_path, plan)


def test_scientific_pass_bit_is_not_in_schema(signed, projection_path: Path, plan):
    changed = deepcopy(signed)
    changed["payload"]["all_advancement_gates_pass"] = True
    with pytest.raises(receipt.ExecutionReceiptError, match="payload schema"):
        _validate(_resign(changed), projection_path, plan)


def test_reader_rejects_duplicate_keys(
    tmp_path: Path, signed, projection_path: Path, plan
):
    raw = receipt.canonical_json_bytes(signed) + b"\n"
    malformed = raw.replace(b'"payload":{', b'"payload":{},"payload":{', 1)
    path = tmp_path / "duplicate.json"
    _immutable_raw(path, malformed)
    with pytest.raises(receipt.ExecutionReceiptError, match="duplicate"):
        receipt.read_runtime_execution_receipt(
            path,
            execution_projection_path=projection_path,
            plan=plan,
            **_artifact_kwargs(),
            verification_key=_public(_key()),
        )


def test_reader_rejects_nonfinite_constant(
    tmp_path: Path, signed, projection_path: Path, plan
):
    raw = receipt.canonical_json_bytes(signed) + b"\n"
    malformed = raw.replace(b'"oracle_access_count":0', b'"oracle_access_count":NaN')
    path = tmp_path / "nan.json"
    _immutable_raw(path, malformed)
    with pytest.raises(receipt.ExecutionReceiptError, match="non-finite"):
        receipt.read_runtime_execution_receipt(
            path,
            execution_projection_path=projection_path,
            plan=plan,
            **_artifact_kwargs(),
            verification_key=_public(_key()),
        )


def test_relative_and_unnormalized_paths_are_rejected(
    tmp_path: Path, projection_path: Path, plan, outputs
):
    with pytest.raises(receipt.ExecutionReceiptError, match="absolute and normalized"):
        receipt.write_runtime_execution_receipt(
            Path("relative.json"),
            execution_projection_path=projection_path,
            plan=plan,
            **_artifact_kwargs(),
            signing_key=_key(),
        )
    unsafe = Path(f"{tmp_path}/missing/../receipt.json")
    with pytest.raises(receipt.ExecutionReceiptError, match="absolute and normalized"):
        receipt.write_runtime_execution_receipt(
            unsafe,
            execution_projection_path=projection_path,
            plan=plan,
            **_artifact_kwargs(),
            signing_key=_key(),
        )


def test_symlink_parent_is_rejected(
    tmp_path: Path, projection_path: Path, plan, outputs
):
    actual = tmp_path / "actual"
    actual.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(actual, target_is_directory=True)
    with pytest.raises(receipt.ExecutionReceiptError, match="parent"):
        receipt.write_runtime_execution_receipt(
            alias / "receipt.json",
            execution_projection_path=projection_path,
            plan=plan,
            **_artifact_kwargs(),
            signing_key=_key(),
        )


def test_symlink_receipt_is_rejected(
    tmp_path: Path, signed, projection_path: Path, plan
):
    target = tmp_path / "target.json"
    _immutable_raw(target, receipt.canonical_json_bytes(signed) + b"\n")
    alias = tmp_path / "alias.json"
    alias.symlink_to(target)
    with pytest.raises(receipt.ExecutionReceiptError, match="symlinked"):
        receipt.read_runtime_execution_receipt(
            alias,
            execution_projection_path=projection_path,
            plan=plan,
            **_artifact_kwargs(),
            verification_key=_public(_key()),
        )


def test_hardlinked_receipt_is_rejected(
    tmp_path: Path, signed, projection_path: Path, plan
):
    target = tmp_path / "target.json"
    _immutable_raw(target, receipt.canonical_json_bytes(signed) + b"\n")
    alias = tmp_path / "alias.json"
    os.link(target, alias)
    with pytest.raises(receipt.ExecutionReceiptError, match="single-link"):
        receipt.read_runtime_execution_receipt(
            alias,
            execution_projection_path=projection_path,
            plan=plan,
            **_artifact_kwargs(),
            verification_key=_public(_key()),
        )


def test_writable_receipt_is_rejected(
    tmp_path: Path, signed, projection_path: Path, plan
):
    path = tmp_path / "writable.json"
    path.write_bytes(receipt.canonical_json_bytes(signed) + b"\n")
    path.chmod(0o600)
    with pytest.raises(receipt.ExecutionReceiptError, match="immutable"):
        receipt.read_runtime_execution_receipt(
            path,
            execution_projection_path=projection_path,
            plan=plan,
            **_artifact_kwargs(),
            verification_key=_public(_key()),
        )


def test_projection_symlink_is_rejected(
    tmp_path: Path, projection_path: Path, plan, outputs
):
    alias = tmp_path / "projection-link.json"
    alias.symlink_to(projection_path)
    with pytest.raises(receipt.ExecutionReceiptError, match="symlinked"):
        receipt.make_runtime_execution_receipt(
            execution_projection_path=alias,
            plan=plan,
            **_artifact_kwargs(),
            signing_key=_key(),
        )


def test_writer_refuses_existing_path(
    tmp_path: Path, projection_path: Path, plan, outputs
):
    path = tmp_path / "receipt.json"
    path.write_text("occupied", encoding="ascii")
    with pytest.raises(FileExistsError):
        receipt.write_runtime_execution_receipt(
            path,
            execution_projection_path=projection_path,
            plan=plan,
            **_artifact_kwargs(),
            signing_key=_key(),
        )


def test_failed_publication_removes_new_receipt(
    tmp_path: Path, projection_path: Path, plan, outputs, monkeypatch
):
    path = tmp_path / "receipt.json"
    real_fsync = os.fsync
    calls = 0

    def fail_parent_fsync(descriptor: int) -> None:
        nonlocal calls
        calls += 1
        if calls == 3:
            raise OSError("simulated parent fsync failure")
        real_fsync(descriptor)

    monkeypatch.setattr(os, "fsync", fail_parent_fsync)
    with pytest.raises(OSError, match="simulated parent fsync failure"):
        receipt.write_runtime_execution_receipt(
            path,
            execution_projection_path=projection_path,
            plan=plan,
            **_artifact_kwargs(),
            signing_key=_key(),
        )
    assert not path.exists()


def test_artifact_index_requires_exact_attempt_schema(
    projection_path: Path, plan, replayed_artifacts, monkeypatch
):
    result, index = replayed_artifacts
    changed = [dict(row) for row in index.attempt_outputs]
    changed[0]["error_message"] = "not allowed"
    changed_index = RuntimeExecutionArtifactIndex(
        index.aggregate_sha256,
        index.parent_artifact_sha256s,
        index.attempt_artifact_sha256s,
        tuple(changed),
    )
    monkeypatch.setattr(
        receipt,
        "_load_execution_artifacts",
        lambda **_: (result, changed_index),
    )
    with pytest.raises(receipt.ExecutionReceiptError, match="schema"):
        receipt.make_runtime_execution_receipt(
            execution_projection_path=projection_path,
            plan=plan,
            **_artifact_kwargs(),
            signing_key=_key(),
        )


def test_receipt_file_is_plain_canonical_json(
    tmp_path: Path, projection_path: Path, plan, outputs
):
    path = tmp_path / "receipt.json"
    receipt.write_runtime_execution_receipt(
        path,
        execution_projection_path=projection_path,
        plan=plan,
        **_artifact_kwargs(),
        signing_key=_key(),
    )
    decoded = json.loads(path.read_text(encoding="ascii"))
    assert path.read_bytes() == receipt.canonical_json_bytes(decoded) + b"\n"
