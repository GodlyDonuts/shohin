"""Hostile CPU contracts for the SCERT board and custody builder."""

from __future__ import annotations

import base64
import copy
import os
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

import generate_scert_boards as boards


@pytest.fixture(scope="module")
def registry_and_request():
    return boards.build_local_registry()


def _signed_receipt(request, private_key, custodian_id="external-custodian-test"):
    hidden = {}
    for index, (name, spec) in enumerate(sorted(boards.HIDDEN_BOARD_SPECS.items())):
        hidden[name] = {
            "spec_sha256": boards.sha256_bytes(boards.canonical_json_bytes(spec)),
            "declared_spec": copy.deepcopy(spec),
            "ciphertext_bytes": 1000 + index,
            "ciphertext_sha256": f"{index + 1:064x}",
            "exact_key_set_commitment_sha256": f"{index + 11:064x}",
            "semantic_key_set_commitment_sha256": f"{index + 21:064x}",
        }
    private_commitments = {
        name: {
            "exact": hidden[name]["exact_key_set_commitment_sha256"],
            "semantic": hidden[name]["semantic_key_set_commitment_sha256"],
        }
        for name in sorted(hidden)
    }
    unsigned = {
        "schema": "r12-scert-external-custody-receipt-v1",
        "protocol": boards.PROTOCOL_ID,
        "custodian_id": custodian_id,
        "request_sha256": boards.sha256_bytes(boards.canonical_json_bytes(request)),
        "theory_sha256": boards.THEORY_SHA256,
        "hidden_boards": hidden,
        "zero_intersection_certificate": {
            "local_key_set_commitment_sha256": request[
                "local_key_set_commitment_sha256"
            ],
            "private_key_set_commitments_sha256": boards.sha256_bytes(
                boards.canonical_json_bytes(private_commitments)
            ),
            "pairwise_exact_intersections": 0,
            "pairwise_semantic_intersections": 0,
        },
        "fit_reveal_gate": {
            "required_stage1_checkpoints": 3,
            "required_true_heads": 3,
            "required_shuffled_heads": 3,
            "revealed": False,
        },
    }
    signature = private_key.sign(boards.canonical_json_bytes(unsigned))
    return {**unsigned, "signature_base64": base64.b64encode(signature).decode("ascii")}


def _resign(receipt, private_key):
    unsigned = dict(receipt)
    unsigned.pop("signature_base64", None)
    return {
        **unsigned,
        "signature_base64": base64.b64encode(
            private_key.sign(boards.canonical_json_bytes(unsigned))
        ).decode("ascii"),
    }


def _trust_anchor(tmp_path, private_key):
    raw = private_key.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw
    )
    path = tmp_path / f"custodian-{boards.sha256_bytes(raw)[:12]}.ed25519.pub"
    path.write_bytes(raw)
    path.chmod(0o444)
    return path


def test_local_registry_exact_counts_and_no_hidden_content(registry_and_request):
    registry, request = registry_and_request
    assert {name: len(rows) for name, rows in registry["boards"].items()} == {
        "D_12": 12,
        "T_14": 2048,
        "T_15": 2048,
        "T_16": 2048,
        "D_256": 256,
    }
    assert registry["hidden_content_present"] is False
    assert "hidden_boards" not in registry
    assert request["hidden_board_specs"] == boards.HIDDEN_BOARD_SPECS
    assert request["required_custody"]["domain_keys_withheld"] is True
    hb = request["hidden_board_specs"]["H_B"]
    assert hb["episodes"] == 384
    assert hb["rows"] == 2688
    assert hb["action_row_counts"] == {"COMMIT": 2304, "HALT": 384}
    assert sum(hb["width_row_counts"].values()) == 2688


def test_scaffold_and_frozen_ids(registry_and_request):
    registry, _ = registry_and_request
    scaffold = registry["scaffold"]
    assert len(scaffold["g_l_ids"]) == 70
    assert len(scaffold["g_r_ids"]) == 3
    assert scaffold["dummy_id"] == 0
    assert scaffold["neutral_id"] == 233
    assert (scaffold["v0"], scaffold["v1"]) == (28, 29)


def test_training_balance_and_seed_disjointness(registry_and_request):
    registry, _ = registry_and_request
    all_semantic = set()
    for name in ("T_14", "T_15", "T_16"):
        rows = registry["boards"][name]
        cells = {}
        for row in rows:
            key = (row["operation"], row["pattern"])
            cells[key] = cells.get(key, 0) + 1
            assert row["width"] == 4
            assert len(row["lanes"]) == 5
            assert len(row["transition_key_sha256"]) == 5
            assert [lane["action"] for lane in row["lanes"]] == [
                "COMMIT",
                "COMMIT",
                "COMMIT",
                "COMMIT",
                "HALT",
            ]
            assert row["semantic_episode_key"] not in all_semantic
            all_semantic.add(row["semantic_episode_key"])
        assert set(cells.values()) == {128}
        assert len(cells) == 16
    assert registry["local_disjointness"]["all_pairwise_intersections"] == 0


def test_public_d12_order_and_balance(registry_and_request):
    registry, _ = registry_and_request
    rows = registry["boards"]["D_12"]
    assert tuple(row["id"] for row in rows) == boards.D12_IDS
    cells = {
        (width, operation): 0 for width in (4, 6, 8) for operation in boards.OPERATIONS
    }
    for row in rows:
        cells[(row["width"], row["operation"])] += 1
    assert set(cells.values()) == {2}


def test_registry_is_deterministic(registry_and_request):
    registry, request = registry_and_request
    second_registry, second_request = boards.build_local_registry()
    assert boards.canonical_json_bytes(second_registry) == boards.canonical_json_bytes(
        registry
    )
    assert boards.canonical_json_bytes(second_request) == boards.canonical_json_bytes(
        request
    )


def test_builder_has_no_private_key_or_hidden_generation_api():
    source = Path(boards.__file__).read_text()
    assert "Ed25519PrivateKey" not in source
    assert "generate_hidden" not in source
    assert "--private-key" not in source
    assert "--sign" not in source


def test_valid_external_receipt_verifies(registry_and_request, tmp_path):
    _, request = registry_and_request
    private_key = Ed25519PrivateKey.generate()
    receipt = _signed_receipt(request, private_key)
    verified = boards.verify_external_custody_receipt(
        receipt, request, _trust_anchor(tmp_path, private_key)
    )
    assert verified["verified"] is True
    assert verified["hidden_content_opened"] is False


def test_self_attestation_is_rejected(registry_and_request, tmp_path):
    _, request = registry_and_request
    private_key = Ed25519PrivateKey.generate()
    receipt = _signed_receipt(request, private_key, request["builder_id"])
    with pytest.raises(boards.ContractError, match="self-attested"):
        boards.verify_external_custody_receipt(
            receipt, request, _trust_anchor(tmp_path, private_key)
        )


@pytest.mark.parametrize(
    "mutation",
    ("signature", "request", "spec", "intersection", "plaintext", "reveal"),
)
def test_hostile_custody_mutations_fail(registry_and_request, tmp_path, mutation):
    _, request = registry_and_request
    private_key = Ed25519PrivateKey.generate()
    receipt = _signed_receipt(request, private_key)
    if mutation == "signature":
        receipt["signature_base64"] = base64.b64encode(b"x" * 64).decode("ascii")
    elif mutation == "request":
        receipt["request_sha256"] = "0" * 64
        receipt = _resign(receipt, private_key)
    elif mutation == "spec":
        receipt["hidden_boards"]["H_B"]["spec_sha256"] = "0" * 64
        receipt = _resign(receipt, private_key)
    elif mutation == "intersection":
        receipt["zero_intersection_certificate"]["pairwise_semantic_intersections"] = 1
        receipt = _resign(receipt, private_key)
    elif mutation == "plaintext":
        receipt["hidden_boards"]["H_B"]["plaintext"] = ["forbidden"]
        receipt = _resign(receipt, private_key)
    else:
        receipt["fit_reveal_gate"]["revealed"] = True
        receipt = _resign(receipt, private_key)
    with pytest.raises(boards.ContractError):
        boards.verify_external_custody_receipt(
            receipt, request, _trust_anchor(tmp_path, private_key)
        )


def test_non_hex_commitment_and_count_substitution_fail(registry_and_request, tmp_path):
    _, request = registry_and_request
    private_key = Ed25519PrivateKey.generate()
    anchor = _trust_anchor(tmp_path, private_key)
    receipt = _signed_receipt(request, private_key)
    receipt["hidden_boards"]["H_A"]["ciphertext_sha256"] = "z" * 64
    with pytest.raises(boards.ContractError, match="commitment"):
        boards.verify_external_custody_receipt(
            _resign(receipt, private_key), request, anchor
        )
    receipt = _signed_receipt(request, private_key)
    receipt["hidden_boards"]["H_B"]["declared_spec"]["rows"] = 2687
    with pytest.raises(boards.ContractError, match="specification"):
        boards.verify_external_custody_receipt(
            _resign(receipt, private_key), request, anchor
        )


def test_signed_request_count_substitution_still_fails(registry_and_request, tmp_path):
    _, request = registry_and_request
    private_key = Ed25519PrivateKey.generate()
    hostile_request = copy.deepcopy(request)
    hostile_request["local_board_receipts"]["D_12"]["rows"] = 83
    receipt = _signed_receipt(hostile_request, private_key)
    with pytest.raises(boards.ContractError, match="local board receipt"):
        boards.verify_external_custody_receipt(
            receipt, hostile_request, _trust_anchor(tmp_path, private_key)
        )


def test_wrong_trust_anchor_and_hard_link_alias_fail(registry_and_request, tmp_path):
    _, request = registry_and_request
    signer = Ed25519PrivateKey.generate()
    other = Ed25519PrivateKey.generate()
    receipt = _signed_receipt(request, signer)
    with pytest.raises(boards.ContractError, match="signature"):
        boards.verify_external_custody_receipt(
            receipt, request, _trust_anchor(tmp_path, other)
        )
    anchor = _trust_anchor(tmp_path, signer)
    alias = tmp_path / "alias.pub"
    os.link(anchor, alias)
    with pytest.raises(boards.ContractError, match="single-link"):
        boards.verify_external_custody_receipt(receipt, request, alias)


def test_writable_trust_anchor_is_rejected(registry_and_request, tmp_path):
    _, request = registry_and_request
    private_key = Ed25519PrivateKey.generate()
    receipt = _signed_receipt(request, private_key)
    anchor = _trust_anchor(tmp_path, private_key)
    anchor.chmod(0o644)
    with pytest.raises(boards.ContractError, match="read-only"):
        boards.verify_external_custody_receipt(receipt, request, anchor)


def test_atomic_publication_refuses_overwrite_and_symlink(tmp_path):
    target = tmp_path / "report.json"
    value = {"schema": "unit-schema", "value": 1}
    receipt = boards.atomic_publish_json(target, value, "unit-schema")
    assert receipt["sha256"] == boards.sha256_file(target)
    assert target.stat().st_mode & 0o777 == 0o444
    with pytest.raises(FileExistsError):
        boards.atomic_publish_json(target, value, "unit-schema")
    linked = tmp_path / "linked.json"
    linked.symlink_to(target)
    with pytest.raises(FileExistsError):
        boards.atomic_publish_json(linked, value, "unit-schema")
    alias_source = tmp_path / "alias-source.json"
    alias_source.write_text("{}\n")
    hard_link = tmp_path / "hard-link.json"
    os.link(alias_source, hard_link)
    with pytest.raises(FileExistsError):
        boards.atomic_publish_json(hard_link, value, "unit-schema")


def test_publication_rejects_wrong_schema_without_final(tmp_path):
    target = tmp_path / "wrong.json"
    with pytest.raises(boards.ContractError, match="schema"):
        boards.atomic_publish_json(target, {"schema": "wrong"}, "expected")
    assert not target.exists()


def test_reference_is_separate_and_deterministic():
    first = boards.independent_toy_reference((2, 3), (20, 30), True)
    second = boards.independent_toy_reference((2, 3), (20, 30), True)
    assert first["next_token"] == second["next_token"]
    assert first["next_logits"].equal(second["next_logits"])
    assert first["probe"].equal(second["probe"])
    assert len(boards.independent_toy_board()) == 256
