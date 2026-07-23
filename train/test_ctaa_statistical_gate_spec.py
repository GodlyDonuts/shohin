"""Focused custody and semantic tests for the signed CTAA gate specification."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
import pytest

import ctaa_statistical_gate_spec as gate_spec


def _key(seed: int = 31) -> Ed25519PrivateKey:
    return Ed25519PrivateKey.from_private_bytes(bytes([seed]) * 32)


def _public(key: Ed25519PrivateKey) -> bytes:
    return key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )


def _hash(character: str) -> str:
    return character * 64


def _bindings() -> gate_spec.StatisticalGateBindings:
    return gate_spec.StatisticalGateBindings(
        manifest_sha256=_hash("1"),
        board_sha256=_hash("2"),
        run_plan_sha256=_hash("3"),
        run_contract_sha256=_hash("4"),
        runtime_bundle_file_sha256=_hash("5"),
        runtime_bundle_sha256=_hash("6"),
        runtime_execution_set_file_sha256=_hash("7"),
        runtime_execution_set_sha256=_hash("8"),
        assessment_source_bundle_sha256=_hash("9"),
        assessment_source_manifest_sha256=_hash("a"),
        bootstrap_seed_receipt_sha256=_hash("b"),
        bootstrap_seed=8_765_432_101,
        training_seeds=(101, 202, 303, 404, 505),
    )


def _record(
    key: Ed25519PrivateKey | None = None,
) -> tuple[dict[str, object], Ed25519PrivateKey]:
    signing_key = key or _key()
    return (
        gate_spec.make_signed_statistical_gate_spec(
            bindings=_bindings(), signing_key=signing_key
        ),
        signing_key,
    )


def _clone(value: object) -> object:
    return json.loads(json.dumps(value))


def _resign(record: dict[str, object], key: Ed25519PrivateKey) -> None:
    payload = record["payload"]
    assert isinstance(payload, dict)
    encoded = gate_spec.canonical_json_bytes(payload)
    record["signature"] = key.sign(gate_spec.SIGNATURE_DOMAIN + encoded).hex()
    record["gate_spec_sha256"] = hashlib.sha256(encoded).hexdigest()


def _immutable(path: Path, record: dict[str, object]) -> None:
    path.write_bytes(gate_spec.canonical_json_bytes(record) + b"\n")
    path.chmod(0o400)


def _validate(
    record: dict[str, object],
    key: Ed25519PrivateKey,
    *,
    expected_bindings: gate_spec.StatisticalGateBindings | None = None,
) -> dict[str, object]:
    return gate_spec.validate_signed_statistical_gate_spec(
        record,
        verification_key=_public(key),
        expected_bindings=expected_bindings or _bindings(),
    )


def test_valid_spec_freezes_every_required_statistical_boundary(tmp_path: Path) -> None:
    record, key = _record()
    path = tmp_path / "gate-spec.json"
    file_sha256 = gate_spec.write_signed_statistical_gate_spec(
        path, bindings=_bindings(), signing_key=key
    )
    verified, observed_file_sha256 = (
        gate_spec.read_signed_statistical_gate_spec_with_sha(
            path,
            verification_key=_public(key),
            expected_bindings=_bindings(),
        )
    )

    assert verified == record
    assert file_sha256 == observed_file_sha256
    assert path.stat().st_mode & 0o222 == 0
    assert path.stat().st_nlink == 1

    payload = verified["payload"]
    assert isinstance(payload, dict)
    assert payload["end_to_end_family_metric"]["definition"] == (
        "prefix_exact AND terminal_exact AND answer_exact"
    )
    assert payload["end_to_end_family_metric"]["all_failures_retained_in_denominator"]
    assert [item["name"] for item in payload["absolute_gates"]["strata"]] == [
        "training_seed",
        "factorial_cell",
        "depth",
    ]
    assert payload["compiler_frontend_gates"]["binding_metric"]["name"] == (
        "independent_binding_exact"
    )
    finite = payload["finite_core_audits"]
    assert finite["receipt_count"] == 20
    assert [axis["name"] for axis in finite["semantic_axes"]] == [
        "train",
        "development",
        "confirmation",
    ]
    assert all(axis["required_atomic_cases"] == 243 for axis in finite["semantic_axes"])
    assert all(
        axis["required_two_action_cases"] == 2_187 for axis in finite["semantic_axes"]
    )
    assert len(payload["runtime_intervention_gates"]["operations"]) == len(
        gate_spec.MANDATORY_OPERATIONS
    )
    assert all(
        item["anchor_count"] == 864
        for item in payload["runtime_intervention_gates"]["operations"]
    )
    assert payload["bootstrap"]["draws"] == 100_000
    assert payload["bootstrap"]["dtype"] == "float64"
    assert payload["bootstrap"]["numpy_bit_generator"] == "numpy.random.PCG64"
    assert payload["multiple_testing"]["primary_claim"]["method"] == (
        "intersection_union"
    )
    assert {
        item["method"] for item in payload["multiple_testing"]["secondary_families"]
    } == {"holm"}
    assert payload["scope"]["authorizes_capability_claim"] is False


def test_writer_refuses_to_replace_an_existing_specification(tmp_path: Path) -> None:
    path = tmp_path / "gate-spec.json"
    gate_spec.write_signed_statistical_gate_spec(
        path, bindings=_bindings(), signing_key=_key()
    )
    with pytest.raises(FileExistsError):
        gate_spec.write_signed_statistical_gate_spec(
            path, bindings=_bindings(), signing_key=_key()
        )


@pytest.mark.parametrize("target", ["record", "payload", "binding_metric"])
def test_unknown_keys_are_rejected_even_when_resigned(target: str) -> None:
    record, key = _record()
    changed = _clone(record)
    assert isinstance(changed, dict)
    payload = changed["payload"]
    assert isinstance(payload, dict)
    if target == "record":
        changed["unexpected"] = False
    elif target == "payload":
        payload["unexpected"] = False
        _resign(changed, key)
    else:
        payload["compiler_frontend_gates"]["binding_metric"]["unexpected"] = False
        _resign(changed, key)
    with pytest.raises(gate_spec.StatisticalGateSpecError, match="schema|policy"):
        _validate(changed, key)


def test_invalid_hash_is_rejected_even_when_resigned() -> None:
    record, key = _record()
    changed = _clone(record)
    assert isinstance(changed, dict)
    changed["payload"]["bindings"]["manifest_sha256"] = "not-a-hash"
    _resign(changed, key)
    with pytest.raises(gate_spec.StatisticalGateSpecError, match="manifest.*SHA-256"):
        _validate(changed, key)


@pytest.mark.parametrize(
    "seeds",
    [
        [101, 202, 303, 404],
        [101, 202, 303, 404, 404],
        [505, 404, 303, 202, 101],
        [101, 202, 303, 404, -1],
    ],
)
def test_seed_count_uniqueness_range_and_order_are_exact(seeds: list[int]) -> None:
    record, key = _record()
    changed = _clone(record)
    assert isinstance(changed, dict)
    changed["payload"]["bindings"]["training_seeds"] = seeds
    _resign(changed, key)
    with pytest.raises(
        gate_spec.StatisticalGateSpecError, match="exactly five.*ordered"
    ):
        _validate(changed, key)


def test_validly_resigned_binding_substitution_is_rejected() -> None:
    record, key = _record()
    changed = _clone(record)
    assert isinstance(changed, dict)
    changed["payload"]["bindings"]["runtime_execution_set_sha256"] = _hash("c")
    _resign(changed, key)
    with pytest.raises(gate_spec.StatisticalGateSpecError, match="substitution"):
        _validate(changed, key)


def test_bootstrap_seed_substitution_is_rejected() -> None:
    record, key = _record()
    changed = _clone(record)
    assert isinstance(changed, dict)
    changed["payload"]["bindings"]["bootstrap_seed"] += 1
    _resign(changed, key)
    with pytest.raises(gate_spec.StatisticalGateSpecError, match="substitution"):
        _validate(changed, key)


def test_signature_corruption_and_wrong_key_are_rejected() -> None:
    record, key = _record()
    changed = _clone(record)
    assert isinstance(changed, dict)
    changed["signature"] = "0" * 128
    with pytest.raises(gate_spec.StatisticalGateSpecError, match="signature"):
        _validate(changed, key)
    with pytest.raises(gate_spec.StatisticalGateSpecError, match="signing key"):
        gate_spec.validate_signed_statistical_gate_spec(
            record,
            verification_key=_public(_key(32)),
            expected_bindings=_bindings(),
        )


def test_impossible_999_cp_lower_at_864_is_rejected_even_when_signed() -> None:
    record, key = _record()
    changed = _clone(record)
    assert isinstance(changed, dict)
    operations = changed["payload"]["runtime_intervention_gates"]["operations"]
    criterion = next(
        criterion
        for operation in operations
        for criterion in operation["criteria"]
        if criterion["kind"] == "one_sided_clopper_pearson"
    )
    criterion["minimum_lower_bound"] = 0.999
    _resign(changed, key)
    with pytest.raises(gate_spec.StatisticalGateSpecError, match="infeasible.*864"):
        _validate(changed, key)


def test_runtime_operations_must_each_use_exactly_864_anchors() -> None:
    record, key = _record()
    changed = _clone(record)
    assert isinstance(changed, dict)
    changed["payload"]["runtime_intervention_gates"]["operations"][0][
        "anchor_count"
    ] = 863
    _resign(changed, key)
    with pytest.raises(gate_spec.StatisticalGateSpecError, match="exactly 864"):
        _validate(changed, key)


@pytest.mark.parametrize("mutation", ["claim_mode", "absolute_stratum"])
def test_renderer_level_claims_are_rejected_as_underpowered(mutation: str) -> None:
    record, key = _record()
    changed = _clone(record)
    assert isinstance(changed, dict)
    if mutation == "claim_mode":
        changed["payload"]["feasibility"]["renderer_level_claim_mode"] = "primary"
    else:
        changed["payload"]["absolute_gates"]["strata"].append(
            {
                "name": "renderer",
                "minimum_rate": 0.95,
                "minimum_one_sided_cp_lower": 0.90,
                "alpha": 0.05,
            }
        )
    _resign(changed, key)
    with pytest.raises(gate_spec.StatisticalGateSpecError, match="underpowered.*108"):
        _validate(changed, key)


def test_cards_exact_cannot_pose_as_the_binding_metric() -> None:
    record, key = _record()
    changed = _clone(record)
    assert isinstance(changed, dict)
    frontend = changed["payload"]["compiler_frontend_gates"]
    frontend["components"][1] = "cards_exact"
    frontend["binding_metric"]["name"] = "cards_exact"
    frontend["binding_metric"]["cards_exact_is_binding_metric"] = True
    _resign(changed, key)
    with pytest.raises(gate_spec.StatisticalGateSpecError, match="not an independent"):
        _validate(changed, key)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("method", "independent_seed_bootstrap"),
        ("independent_within_seed_family_resampling", True),
        ("dtype", "float32"),
        ("draws", 10_000),
        ("numpy_bit_generator", "numpy.random.MT19937"),
    ],
)
def test_wrong_crossed_bootstrap_semantics_are_rejected(
    field: str, value: object
) -> None:
    record, key = _record()
    changed = _clone(record)
    assert isinstance(changed, dict)
    changed["payload"]["bootstrap"][field] = value
    _resign(changed, key)
    with pytest.raises(gate_spec.StatisticalGateSpecError, match="crossed seed"):
        _validate(changed, key)


def test_family_metric_requires_conjunction_and_retains_every_failure() -> None:
    record, key = _record()
    for field, value in (
        ("definition", "answer_exact"),
        ("all_failures_retained_in_denominator", False),
    ):
        changed = _clone(record)
        assert isinstance(changed, dict)
        changed["payload"]["end_to_end_family_metric"][field] = value
        _resign(changed, key)
        with pytest.raises(gate_spec.StatisticalGateSpecError, match="metric policy"):
            _validate(changed, key)


def test_development_must_precede_confirmation_under_the_same_spec() -> None:
    record, key = _record()
    changed = _clone(record)
    assert isinstance(changed, dict)
    changed["payload"]["partition_policy"]["ordered_partitions"].reverse()
    _resign(changed, key)
    with pytest.raises(gate_spec.StatisticalGateSpecError, match="partition_policy"):
        _validate(changed, key)


def test_primary_claim_cannot_use_holm_and_secondary_families_must() -> None:
    record, key = _record()
    changed = _clone(record)
    assert isinstance(changed, dict)
    changed["payload"]["multiple_testing"]["primary_claim"]["method"] = "holm"
    _resign(changed, key)
    with pytest.raises(gate_spec.StatisticalGateSpecError, match="multiple_testing"):
        _validate(changed, key)


def test_writable_input_is_rejected(tmp_path: Path) -> None:
    record, key = _record()
    path = tmp_path / "writable.json"
    path.write_bytes(gate_spec.canonical_json_bytes(record) + b"\n")
    path.chmod(0o600)
    with pytest.raises(gate_spec.StatisticalGateSpecError, match="read-only"):
        gate_spec.read_signed_statistical_gate_spec(
            path,
            verification_key=_public(key),
            expected_bindings=_bindings(),
        )


def test_symlink_input_is_rejected(tmp_path: Path) -> None:
    record, key = _record()
    target = tmp_path / "target.json"
    link = tmp_path / "link.json"
    _immutable(target, record)
    link.symlink_to(target)
    with pytest.raises(gate_spec.StatisticalGateSpecError, match="symlinked"):
        gate_spec.read_signed_statistical_gate_spec(
            link,
            verification_key=_public(key),
            expected_bindings=_bindings(),
        )


def test_hardlink_input_is_rejected(tmp_path: Path) -> None:
    record, key = _record()
    target = tmp_path / "target.json"
    hardlink = tmp_path / "hardlink.json"
    _immutable(target, record)
    os.link(target, hardlink)
    with pytest.raises(gate_spec.StatisticalGateSpecError, match="single-link"):
        gate_spec.read_signed_statistical_gate_spec(
            target,
            verification_key=_public(key),
            expected_bindings=_bindings(),
        )


def test_noncanonical_file_is_rejected(tmp_path: Path) -> None:
    record, key = _record()
    path = tmp_path / "pretty.json"
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="ascii")
    path.chmod(0o400)
    with pytest.raises(gate_spec.StatisticalGateSpecError, match="not canonical"):
        gate_spec.read_signed_statistical_gate_spec(
            path,
            verification_key=_public(key),
            expected_bindings=_bindings(),
        )


def test_duplicate_json_key_is_rejected(tmp_path: Path) -> None:
    record, key = _record()
    raw = gate_spec.canonical_json_bytes(record)
    assert raw.startswith(b'{"gate_spec_sha256"')
    corrupt = b'{"gate_spec_sha256":"' + _hash("0").encode() + b'",' + raw[1:]
    path = tmp_path / "duplicate.json"
    path.write_bytes(corrupt + b"\n")
    path.chmod(0o400)
    with pytest.raises(gate_spec.StatisticalGateSpecError, match="duplicate"):
        gate_spec.read_signed_statistical_gate_spec(
            path,
            verification_key=_public(key),
            expected_bindings=_bindings(),
        )
