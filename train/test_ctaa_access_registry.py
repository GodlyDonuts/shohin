from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

import ctaa_access_registry as registry


REGISTRY_ID = "ctaa-evaluation-001"


def _key(seed: int) -> Ed25519PrivateKey:
    # Deterministic, non-operational fixtures; no production key material.
    return Ed25519PrivateKey.from_private_bytes(bytes([seed]) * 32)


def _public(key: Ed25519PrivateKey) -> bytes:
    return key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )


def _hash(character: str) -> str:
    return character * 64


def _head(path: Path, key: Ed25519PrivateKey) -> registry.RegistryState:
    return registry.verify_registry(path, _public(key))


def _spend(
    path: Path,
    key: Ed25519PrivateKey,
    *,
    expected: str,
    event_id: str = "event-development-spend-1",
    access_id: str = "development-access-1",
    partition: str = "development",
    manifest: str = "1",
    board: str = "2",
    run_contract: str = "3",
    runtime_bundle: str = "7",
    bootstrap_receipt: str = "6",
    bootstrap_seed: int = 123456789,
    receipt: object | None = None,
) -> dict[str, object]:
    return registry.append_access_spend(
        path,
        signing_key=key,
        registry_id=REGISTRY_ID,
        event_id=event_id,
        access_id=access_id,
        partition=partition,
        manifest_sha256=_hash(manifest),
        board_sha256=_hash(board),
        run_contract_sha256=_hash(run_contract),
        runtime_bundle_sha256=_hash(runtime_bundle),
        assessment_claim_sha256=_hash("8"),
        bootstrap_seed_receipt_sha256=_hash(bootstrap_receipt),
        bootstrap_seed=bootstrap_seed,
        statistical_gate_spec_file_sha256=_hash("9"),
        gate_spec_sha256=_hash("a"),
        expected_previous_hash=expected,
        expected_head_receipt=receipt,
    )


def _assess(
    path: Path,
    key: Ed25519PrivateKey,
    *,
    expected: str,
    event_id: str = "event-development-assessment-1",
    access_id: str = "development-access-1",
    assessment: str = "4",
    receipt: object | None = None,
) -> dict[str, object]:
    return registry.append_assessment_commit(
        path,
        signing_key=key,
        registry_id=REGISTRY_ID,
        event_id=event_id,
        access_id=access_id,
        assessment_sha256=_hash(assessment),
        statistical_gate_spec_file_sha256=_hash("9"),
        gate_spec_sha256=_hash("a"),
        expected_previous_hash=expected,
        expected_head_receipt=receipt,
    )


def _gate(
    path: Path,
    key: Ed25519PrivateKey,
    *,
    expected: str,
    event_id: str = "event-development-gate-1",
    development_access_id: str = "development-access-1",
    gate_receipt: str = "5",
    receipt: object | None = None,
) -> dict[str, object]:
    return registry.append_development_gate_commit(
        path,
        signing_key=key,
        registry_id=REGISTRY_ID,
        event_id=event_id,
        development_access_id=development_access_id,
        development_gate_receipt_sha256=_hash(gate_receipt),
        expected_previous_hash=expected,
        expected_head_receipt=receipt,
    )


def _close_development(
    path: Path, key: Ed25519PrivateKey
) -> tuple[dict[str, object], dict[str, object]]:
    spend = _spend(path, key, expected=registry.GENESIS_PREVIOUS_HASH)
    assessment = _assess(
        path,
        key,
        expected=_head(path, key).head_hash,
        receipt=spend,
    )
    return spend, assessment


def _rewrite_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_bytes(
        b"".join(registry.canonical_json_bytes(row) + b"\n" for row in rows)
    )


def _next_signed_entry(
    key: Ed25519PrivateKey,
    previous: dict[str, object],
    *,
    event_type: str,
    event_id: str,
    fields: dict[str, object],
    **common_overrides: object,
) -> dict[str, object]:
    previous_payload = previous["payload"]
    payload = {
        "schema": registry.ENTRY_SCHEMA,
        "registry_id": previous_payload["registry_id"],
        "sequence": int(previous_payload["sequence"]) + 1,
        "previous_hash": previous["entry_hash"],
        "event_type": event_type,
        "event_id": event_id,
        "signing_public_key": previous_payload["signing_public_key"],
        **fields,
        **common_overrides,
    }
    return registry._make_signed_entry(payload, key)


def test_full_development_gate_confirmation_state_machine(tmp_path: Path) -> None:
    path = tmp_path / "access.jsonl"
    key = _key(7)

    development_spend = _spend(path, key, expected=registry.GENESIS_PREVIOUS_HASH)
    spend_row = json.loads(path.read_text())
    assert spend_row["payload"]["event_type"] == registry.ACCESS_SPEND
    assert "assessment_sha256" not in spend_row["payload"]
    assert spend_row["payload"]["manifest_sha256"] == _hash("1")
    assert spend_row["payload"]["board_sha256"] == _hash("2")
    assert spend_row["payload"]["run_contract_sha256"] == _hash("3")
    assert spend_row["payload"]["runtime_bundle_sha256"] == _hash("7")
    spend_state = registry.verify_registry(
        path, _public(key), expected_head_receipt=development_spend
    )
    assert spend_state.open_access_id == "development-access-1"

    development_assessment = _assess(
        path,
        key,
        expected=spend_state.head_hash,
        receipt=development_spend,
    )
    assessment_state = registry.verify_registry(
        path, _public(key), expected_head_receipt=development_assessment
    )
    assert assessment_state.open_access_id is None
    assert assessment_state.head_event_type == registry.ASSESSMENT_COMMIT

    development_gate = _gate(
        path,
        key,
        expected=assessment_state.head_hash,
        receipt=development_assessment,
    )
    gate_state = registry.verify_registry(
        path, _public(key), expected_head_receipt=development_gate
    )
    assert gate_state.development_gate_access_id == "development-access-1"

    confirmation_spend = _spend(
        path,
        key,
        expected=gate_state.head_hash,
        event_id="event-confirmation-spend-1",
        access_id="confirmation-access-1",
        partition="confirmation",
        manifest="6",
        board="7",
        run_contract="8",
        receipt=development_gate,
    )
    confirmation_state = registry.verify_registry(
        path, _public(key), expected_head_receipt=confirmation_spend
    )
    assert confirmation_state.confirmation_started is True
    assert confirmation_state.open_access_id == "confirmation-access-1"

    confirmation_assessment = _assess(
        path,
        key,
        expected=confirmation_state.head_hash,
        event_id="event-confirmation-assessment-1",
        access_id="confirmation-access-1",
        assessment="9",
        receipt=confirmation_spend,
    )
    final = registry.verify_registry(
        path, _public(key), expected_head_receipt=confirmation_assessment
    )
    assert final.entry_count == 5
    assert final.open_access_id is None
    assert final.confirmation_started is True
    assert (
        registry.verify_registry(
            path,
            _public(key),
            expected_head_receipt=development_spend,
            allow_extensions=True,
        ).head_hash
        == final.head_hash
    )


def test_access_spend_schema_rejects_assessment_hash_even_when_signed(
    tmp_path: Path,
) -> None:
    path = tmp_path / "access.jsonl"
    key = _key(8)
    _spend(path, key, expected=registry.GENESIS_PREVIOUS_HASH)
    row = json.loads(path.read_text())
    row["payload"]["assessment_sha256"] = _hash("a")
    signed = registry._make_signed_entry(row["payload"], key)
    _rewrite_rows(path, [signed])
    with pytest.raises(registry.RegistryVerificationError, match="schema"):
        registry.verify_registry(path, _public(key))


def test_access_spend_requires_exact_lowercase_runtime_bundle_hash(
    tmp_path: Path,
) -> None:
    path = tmp_path / "access.jsonl"
    key = _key(26)
    assert registry.ENTRY_SCHEMA == "r12_ctaa_access_registry_event_v5"
    with pytest.raises(
        registry.RegistryVerificationError, match="runtime_bundle_sha256"
    ):
        registry.append_access_spend(
            path,
            signing_key=key,
            registry_id=REGISTRY_ID,
            event_id="event-development-spend-1",
            access_id="development-access-1",
            partition="development",
            manifest_sha256=_hash("1"),
            board_sha256=_hash("2"),
            run_contract_sha256=_hash("3"),
            runtime_bundle_sha256="A" * 64,
            assessment_claim_sha256=_hash("8"),
            bootstrap_seed_receipt_sha256=_hash("6"),
            bootstrap_seed=123456789,
            statistical_gate_spec_file_sha256=_hash("9"),
            gate_spec_sha256=_hash("a"),
            expected_previous_hash=registry.GENESIS_PREVIOUS_HASH,
        )
    assert not path.exists()

    valid = _spend(path, key, expected=registry.GENESIS_PREVIOUS_HASH)
    original = json.loads(path.read_text())
    row = json.loads(path.read_text())
    row["payload"]["runtime_bundle_sha256"] = "A" * 64
    resigned = registry._make_signed_entry(row["payload"], key)
    _rewrite_rows(path, [resigned])
    with pytest.raises(
        registry.RegistryVerificationError, match="runtime_bundle_sha256"
    ):
        registry.verify_registry(path, _public(key))

    row = original
    del row["payload"]["runtime_bundle_sha256"]
    resigned = registry._make_signed_entry(row["payload"], key)
    _rewrite_rows(path, [resigned])
    with pytest.raises(registry.RegistryVerificationError, match="schema"):
        registry.verify_registry(path, _public(key))
    del valid


def test_runtime_bundle_mutation_and_signed_substitution_are_rejected(
    tmp_path: Path,
) -> None:
    path = tmp_path / "access.jsonl"
    key = _key(27)
    retained = _spend(path, key, expected=registry.GENESIS_PREVIOUS_HASH)
    original = json.loads(path.read_text())

    mutated = json.loads(path.read_text())
    mutated["payload"]["runtime_bundle_sha256"] = _hash("a")
    _rewrite_rows(path, [mutated])
    with pytest.raises(registry.RegistryVerificationError, match="signature"):
        registry.verify_registry(path, _public(key))

    substituted_payload = {
        **original["payload"],
        "runtime_bundle_sha256": _hash("b"),
    }
    substituted = registry._make_signed_entry(substituted_payload, key)
    _rewrite_rows(path, [substituted])
    with pytest.raises(registry.RegistryVerificationError, match="retained head"):
        registry.verify_registry(
            path,
            _public(key),
            expected_head_receipt=retained,
        )


def test_generic_registry_verifier_rejects_malformed_claim_hash(
    tmp_path: Path,
) -> None:
    path = tmp_path / "access.jsonl"
    key = _key(28)
    _spend(path, key, expected=registry.GENESIS_PREVIOUS_HASH)
    row = json.loads(path.read_text())
    row["payload"]["assessment_claim_sha256"] = "A" * 64
    _rewrite_rows(path, [registry._make_signed_entry(row["payload"], key)])
    with pytest.raises(
        registry.RegistryVerificationError, match="assessment_claim_sha256"
    ):
        registry.verify_registry(path, _public(key))


def test_second_access_and_gate_are_forbidden_while_access_is_open(
    tmp_path: Path,
) -> None:
    path = tmp_path / "access.jsonl"
    key = _key(9)
    receipt = _spend(path, key, expected=registry.GENESIS_PREVIOUS_HASH)
    state = _head(path, key)
    before = path.read_bytes()
    with pytest.raises(registry.RegistryVerificationError, match="immediately closed"):
        _spend(
            path,
            key,
            expected=state.head_hash,
            event_id="event-development-spend-2",
            access_id="development-access-2",
            manifest="4",
            board="5",
            run_contract="6",
            receipt=receipt,
        )
    with pytest.raises(registry.RegistryVerificationError, match="cannot precede"):
        _gate(
            path,
            key,
            expected=state.head_hash,
            receipt=receipt,
        )
    assert path.read_bytes() == before


def test_assessment_must_immediately_close_matching_open_access(
    tmp_path: Path,
) -> None:
    key = _key(10)
    missing = tmp_path / "missing-open.jsonl"
    with pytest.raises(registry.RegistryVerificationError, match="no open"):
        _assess(
            missing,
            key,
            expected=registry.GENESIS_PREVIOUS_HASH,
        )

    path = tmp_path / "access.jsonl"
    spend = _spend(path, key, expected=registry.GENESIS_PREVIOUS_HASH)
    state = _head(path, key)
    with pytest.raises(registry.RegistryVerificationError, match="open access_id"):
        _assess(
            path,
            key,
            expected=state.head_hash,
            access_id="different-access",
            receipt=spend,
        )
    assessment = _assess(
        path,
        key,
        expected=state.head_hash,
        receipt=spend,
    )
    closed = _head(path, key)
    with pytest.raises(registry.RegistryVerificationError, match="no open"):
        _assess(
            path,
            key,
            expected=closed.head_hash,
            event_id="event-development-assessment-2",
            assessment="5",
            receipt=assessment,
        )


def test_confirmation_requires_gate_and_gate_requires_closed_development(
    tmp_path: Path,
) -> None:
    key = _key(11)
    with pytest.raises(
        registry.RegistryVerificationError, match="requires development_gate"
    ):
        _spend(
            tmp_path / "confirmation-first.jsonl",
            key,
            expected=registry.GENESIS_PREVIOUS_HASH,
            partition="confirmation",
            event_id="event-confirmation-spend-1",
            access_id="confirmation-access-1",
        )
    with pytest.raises(registry.RegistryVerificationError, match="closed development"):
        _gate(
            tmp_path / "gate-first.jsonl",
            key,
            expected=registry.GENESIS_PREVIOUS_HASH,
        )

    path = tmp_path / "access.jsonl"
    spend, assessment = _close_development(path, key)
    del spend
    closed = _head(path, key)
    with pytest.raises(registry.RegistryVerificationError, match="closed development"):
        _gate(
            path,
            key,
            expected=closed.head_hash,
            development_access_id="unknown-development-access",
            receipt=assessment,
        )


def test_gate_must_bind_latest_closed_development_and_seals_development(
    tmp_path: Path,
) -> None:
    path = tmp_path / "access.jsonl"
    key = _key(12)
    _, first_assessment = _close_development(path, key)
    state = _head(path, key)
    second_spend = _spend(
        path,
        key,
        expected=state.head_hash,
        event_id="event-development-spend-2",
        access_id="development-access-2",
        manifest="6",
        board="7",
        run_contract="8",
        receipt=first_assessment,
    )
    state = _head(path, key)
    second_assessment = _assess(
        path,
        key,
        expected=state.head_hash,
        event_id="event-development-assessment-2",
        access_id="development-access-2",
        assessment="9",
        receipt=second_spend,
    )
    state = _head(path, key)
    with pytest.raises(registry.RegistryVerificationError, match="latest closed"):
        _gate(
            path,
            key,
            expected=state.head_hash,
            development_access_id="development-access-1",
            receipt=second_assessment,
        )
    gate = _gate(
        path,
        key,
        expected=state.head_hash,
        development_access_id="development-access-2",
        receipt=second_assessment,
    )
    state = _head(path, key)
    with pytest.raises(registry.RegistryVerificationError, match="cannot follow"):
        _spend(
            path,
            key,
            expected=state.head_hash,
            event_id="event-development-spend-3",
            access_id="development-access-3",
            manifest="a",
            board="b",
            run_contract="c",
            receipt=gate,
        )


def test_duplicate_event_id_access_id_and_spend_binding_are_rejected(
    tmp_path: Path,
) -> None:
    path = tmp_path / "access.jsonl"
    key = _key(13)
    spend, assessment = _close_development(path, key)
    del spend
    state = _head(path, key)
    with pytest.raises(registry.RegistryVerificationError, match="duplicate event_id"):
        _spend(
            path,
            key,
            expected=state.head_hash,
            event_id="event-development-spend-1",
            access_id="development-access-2",
            manifest="6",
            board="7",
            run_contract="8",
            receipt=assessment,
        )
    with pytest.raises(registry.RegistryVerificationError, match="duplicate access_id"):
        _spend(
            path,
            key,
            expected=state.head_hash,
            event_id="different-event-id",
            access_id="development-access-1",
            manifest="6",
            board="7",
            run_contract="8",
            receipt=assessment,
        )
    with pytest.raises(registry.RegistryVerificationError, match="duplicate event"):
        _spend(
            path,
            key,
            expected=state.head_hash,
            event_id="renamed-duplicate-event",
            access_id="development-access-1",
            receipt=assessment,
        )
    with pytest.raises(
        registry.RegistryVerificationError, match="duplicate access_spend binding"
    ):
        _spend(
            path,
            key,
            expected=state.head_hash,
            event_id="replayed-binding-event",
            access_id="replayed-binding-access",
            receipt=assessment,
        )
    _spend(
        path,
        key,
        expected=state.head_hash,
        event_id="distinct-runtime-bundle-event",
        access_id="distinct-runtime-bundle-access",
        runtime_bundle="8",
        receipt=assessment,
    )
    last_row = json.loads(path.read_text().splitlines()[-1])
    assert last_row["payload"]["runtime_bundle_sha256"] == _hash("8")


def test_stale_expected_hash_and_wrong_external_receipt_do_not_modify_file(
    tmp_path: Path,
) -> None:
    path = tmp_path / "access.jsonl"
    key = _key(14)
    spend = _spend(path, key, expected=registry.GENESIS_PREVIOUS_HASH)
    before = path.read_bytes()
    with pytest.raises(registry.ConcurrentAppendError):
        _assess(
            path,
            key,
            expected=_hash("f"),
            receipt=spend,
        )
    assert path.read_bytes() == before

    tampered_receipt = json.loads(registry.serialize_head_receipt(spend))
    tampered_receipt["payload"]["event_payload_sha256"] = _hash("a")
    with pytest.raises(registry.RegistryVerificationError, match="signature"):
        _assess(
            path,
            key,
            expected=_head(path, key).head_hash,
            receipt=tampered_receipt,
        )
    assert path.read_bytes() == before


def test_existing_registry_append_requires_external_head_receipt(
    tmp_path: Path,
) -> None:
    path = tmp_path / "access.jsonl"
    key = _key(23)
    _spend(path, key, expected=registry.GENESIS_PREVIOUS_HASH)
    state = _head(path, key)
    before = path.read_bytes()
    with pytest.raises(registry.RegistryVerificationError, match="requires externally"):
        _assess(
            path,
            key,
            expected=state.head_hash,
            receipt=None,
        )
    assert path.read_bytes() == before


def test_canonical_signatures_wrong_keys_and_small_order_keys(tmp_path: Path) -> None:
    path = tmp_path / "access.jsonl"
    key = _key(15)
    receipt = _spend(path, key, expected=registry.GENESIS_PREVIOUS_HASH)
    row = json.loads(path.read_text())
    assert path.read_bytes() == registry.canonical_json_bytes(row) + b"\n"
    registry.verify_head_receipt(receipt, _public(key))

    path.write_text(json.dumps(row, separators=(", ", ": ")) + "\n")
    with pytest.raises(registry.RegistryVerificationError, match="canonical"):
        registry.verify_registry(path, _public(key))

    _rewrite_rows(path, [row])
    with pytest.raises(registry.RegistryVerificationError, match="wrong signing key"):
        registry.verify_registry(path, _public(_key(16)))
    with pytest.raises(registry.RegistryVerificationError, match="32 bytes"):
        registry.verify_registry(path, b"short")
    with pytest.raises(registry.RegistryVerificationError, match="prime-order"):
        registry.verify_registry(path, b"\x00" * 32)
    with pytest.raises(registry.RegistryVerificationError, match="prime-order"):
        registry.verify_registry(path, (1).to_bytes(32, "little"))
    with pytest.raises(registry.RegistryVerificationError, match="non-canonical"):
        registry.verify_registry(path, b"\xff" * 32)


def test_tamper_and_validly_resigned_substitution_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "access.jsonl"
    key = _key(17)
    retained = _spend(path, key, expected=registry.GENESIS_PREVIOUS_HASH)
    original = json.loads(path.read_text())
    tampered = json.loads(path.read_text())
    tampered["payload"]["board_sha256"] = _hash("a")
    _rewrite_rows(path, [tampered])
    with pytest.raises(registry.RegistryVerificationError, match="signature"):
        registry.verify_registry(path, _public(key))

    substituted_payload = {
        **original["payload"],
        "board_sha256": _hash("b"),
    }
    substituted = registry._make_signed_entry(substituted_payload, key)
    _rewrite_rows(path, [substituted])
    with pytest.raises(registry.RegistryVerificationError, match="retained head"):
        registry.verify_registry(path, _public(key), expected_head_receipt=retained)


def test_truncation_reset_and_same_uid_replacement_are_rejected(
    tmp_path: Path,
) -> None:
    path = tmp_path / "access.jsonl"
    key = _key(18)
    spend, assessment = _close_development(path, key)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    _rewrite_rows(path, rows[:1])
    with pytest.raises(registry.RegistryVerificationError, match="truncated"):
        registry.verify_registry(path, _public(key), expected_head_receipt=assessment)

    replacement = tmp_path / "replacement.jsonl"
    registry.append_access_spend(
        replacement,
        signing_key=key,
        registry_id="same-uid-reset",
        event_id="replacement-event-1",
        access_id="replacement-access-1",
        partition="development",
        manifest_sha256=_hash("a"),
        board_sha256=_hash("b"),
        run_contract_sha256=_hash("c"),
        runtime_bundle_sha256=_hash("e"),
        assessment_claim_sha256=_hash("f"),
        bootstrap_seed_receipt_sha256=_hash("d"),
        bootstrap_seed=123456789,
        statistical_gate_spec_file_sha256=_hash("9"),
        gate_spec_sha256=_hash("a"),
        expected_previous_hash=registry.GENESIS_PREVIOUS_HASH,
    )
    os.replace(replacement, path)
    with pytest.raises(registry.RegistryVerificationError, match="retained head"):
        registry.verify_registry(path, _public(key), expected_head_receipt=spend)


def test_verifier_rejects_signed_sequence_chain_and_temporal_attacks(
    tmp_path: Path,
) -> None:
    path = tmp_path / "access.jsonl"
    key = _key(19)
    _spend(path, key, expected=registry.GENESIS_PREVIOUS_HASH)
    first = json.loads(path.read_text())

    skipped = _next_signed_entry(
        key,
        first,
        event_type=registry.ASSESSMENT_COMMIT,
        event_id="event-assessment-1",
        fields={
            "access_id": "development-access-1",
            "assessment_sha256": _hash("4"),
            "statistical_gate_spec_file_sha256": _hash("9"),
            "gate_spec_sha256": _hash("a"),
        },
        sequence=2,
    )
    _rewrite_rows(path, [first, skipped])
    with pytest.raises(registry.RegistryVerificationError, match="monotonic"):
        registry.verify_registry(path, _public(key))

    broken = _next_signed_entry(
        key,
        first,
        event_type=registry.ASSESSMENT_COMMIT,
        event_id="event-assessment-1",
        fields={
            "access_id": "development-access-1",
            "assessment_sha256": _hash("4"),
            "statistical_gate_spec_file_sha256": _hash("9"),
            "gate_spec_sha256": _hash("a"),
        },
        previous_hash=_hash("f"),
    )
    _rewrite_rows(path, [first, broken])
    with pytest.raises(registry.RegistryVerificationError, match="hash chain"):
        registry.verify_registry(path, _public(key))

    second_spend = _next_signed_entry(
        key,
        first,
        event_type=registry.ACCESS_SPEND,
        event_id="event-development-spend-2",
        fields={
            "access_id": "development-access-2",
            "partition": "development",
            "manifest_sha256": _hash("5"),
            "board_sha256": _hash("6"),
            "run_contract_sha256": _hash("7"),
            "runtime_bundle_sha256": _hash("9"),
            "assessment_claim_sha256": _hash("a"),
            "bootstrap_seed_receipt_sha256": _hash("8"),
            "bootstrap_seed": 987654321,
            "statistical_gate_spec_file_sha256": _hash("b"),
            "gate_spec_sha256": _hash("c"),
        },
    )
    _rewrite_rows(path, [first, second_spend])
    with pytest.raises(registry.RegistryVerificationError, match="immediately closed"):
        registry.verify_registry(path, _public(key))


def test_external_receipt_exact_head_policy_and_binding(tmp_path: Path) -> None:
    path = tmp_path / "access.jsonl"
    key = _key(20)
    spend, assessment = _close_development(path, key)
    with pytest.raises(registry.RegistryVerificationError, match="head differs"):
        registry.verify_registry(path, _public(key), expected_head_receipt=spend)
    registry.verify_registry(
        path,
        _public(key),
        expected_head_receipt=spend,
        allow_extensions=True,
    )
    registry.verify_registry(path, _public(key), expected_head_receipt=assessment)
    with pytest.raises(registry.RegistryVerificationError, match="wrong signing key"):
        registry.verify_head_receipt(assessment, _public(_key(21)))
    assert registry.serialize_head_receipt(assessment).endswith(b"\n")


def test_public_verified_event_view_exposes_exact_immutable_bindings(
    tmp_path: Path,
) -> None:
    path = tmp_path / "access.jsonl"
    key = _key(24)
    spend, assessment = _close_development(path, key)
    state = _head(path, key)
    gate = _gate(
        path,
        key,
        expected=state.head_hash,
        receipt=assessment,
    )
    events = registry.verify_registry_events(
        path,
        _public(key),
        expected_head_receipt=gate,
    )
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(events) == 3
    assert [dict(event.payload) for event in events] == [row["payload"] for row in rows]
    assert [event.canonical_payload for event in events] == [
        registry.canonical_json_bytes(row["payload"]) for row in rows
    ]
    assert events[0].payload["manifest_sha256"] == _hash("1")
    assert events[0].payload["board_sha256"] == _hash("2")
    assert events[0].payload["run_contract_sha256"] == _hash("3")
    assert events[0].payload["runtime_bundle_sha256"] == _hash("7")
    assert "assessment_sha256" not in events[0].payload
    assert events[1].payload["assessment_sha256"] == _hash("4")
    assert events[2].payload["development_gate_receipt_sha256"] == _hash("5")
    assert [event.signature for event in events] == [row["signature"] for row in rows]
    assert [event.entry_hash for event in events] == [row["entry_hash"] for row in rows]
    with pytest.raises(TypeError):
        events[0].payload["manifest_sha256"] = _hash("a")
    with pytest.raises(AttributeError):
        events.append(events[0])
    del spend


def test_public_event_view_returns_nothing_from_tampered_chain(tmp_path: Path) -> None:
    path = tmp_path / "access.jsonl"
    key = _key(25)
    spend, assessment = _close_development(path, key)
    del spend
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["payload"]["manifest_sha256"] = _hash("a")
    _rewrite_rows(path, rows)
    with pytest.raises(registry.RegistryVerificationError, match="signature"):
        registry.verify_registry_events(
            path,
            _public(key),
            expected_head_receipt=assessment,
        )


def test_incomplete_symlink_and_missing_registry_are_rejected(tmp_path: Path) -> None:
    key = _key(22)
    with pytest.raises(registry.RegistryVerificationError, match="does not exist"):
        registry.verify_registry(tmp_path / "missing.jsonl", _public(key))
    incomplete = tmp_path / "incomplete.jsonl"
    incomplete.write_bytes(b"{}")
    with pytest.raises(registry.RegistryVerificationError, match="incomplete"):
        registry.verify_registry(incomplete, _public(key))
    target = tmp_path / "target.jsonl"
    _spend(target, key, expected=registry.GENESIS_PREVIOUS_HASH)
    symlink = tmp_path / "symlink.jsonl"
    symlink.symlink_to(target)
    with pytest.raises(registry.RegistryVerificationError, match="regular file"):
        registry.verify_registry(symlink, _public(key))


def test_registry_reader_rejects_symlinked_intermediate_parent(tmp_path: Path) -> None:
    key = _key(26)
    real = tmp_path / "real" / "nested"
    path = real / "access.jsonl"
    receipt = _spend(path, key, expected=registry.GENESIS_PREVIOUS_HASH)
    alias = tmp_path / "alias"
    alias.symlink_to(tmp_path / "real", target_is_directory=True)

    with pytest.raises(registry.RegistryVerificationError, match="parent.*safely"):
        registry.verify_registry(
            alias / "nested" / path.name,
            _public(key),
            expected_head_receipt=receipt,
        )


def test_registry_reader_holds_parent_across_symlink_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    key = _key(27)
    visible = tmp_path / "visible"
    authentic = visible / "nested" / "access.jsonl"
    receipt = _spend(authentic, key, expected=registry.GENESIS_PREVIOUS_HASH)

    decoy = tmp_path / "decoy"
    decoy_path = decoy / "nested" / authentic.name
    _spend(decoy_path, _key(28), expected=registry.GENESIS_PREVIOUS_HASH)
    held = tmp_path / "held"
    original = registry.os.open
    mutated = False

    def mutate_after_open(path, flags, *args, **kwargs):
        nonlocal mutated
        descriptor = original(path, flags, *args, **kwargs)
        if path == visible.name and kwargs.get("dir_fd") is not None and not mutated:
            visible.rename(held)
            visible.symlink_to(decoy, target_is_directory=True)
            mutated = True
        return descriptor

    monkeypatch.setattr(registry.os, "open", mutate_after_open)
    state = registry.verify_registry(
        authentic,
        _public(key),
        expected_head_receipt=receipt,
    )
    assert mutated
    assert state.head_hash == receipt["payload"]["entry_hash"]
    assert authentic.read_bytes() == decoy_path.read_bytes()
