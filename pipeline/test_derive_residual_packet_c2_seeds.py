#!/usr/bin/env python3
"""Adversarial toy-only tests for the RSP-C2 seed custody contract."""

from __future__ import annotations

import ast
import copy
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import derive_residual_packet_c2_seeds as derive


FREEZE_TIME = "2030-01-01T00:00:00.000Z"
FREEZE_OBSERVED_TIME = "2030-01-01T00:02:00.000Z"
BEACON_TIME = "2030-01-01T01:00:30.000Z"


def raw_json(payload: dict) -> bytes:
    return derive.canonical_json_bytes(payload)


def toy_prerequisite() -> dict:
    return {
        "advance_to_internalization": True,
        "all_locked_gates_pass": True,
        "confirmation_contract_sha256": "1" * 64,
        "confirmation_result_sha256": "2" * 64,
        "independent_recomputation_complete": True,
        "independent_score_receipt_sha256": "6" * 64,
        "independent_scorer_sha256": "4" * 64,
        "primary_score_receipt_sha256": "5" * 64,
        "primary_scorer_sha256": "3" * 64,
        "result_immutable": True,
        "schema": derive.PREREQUISITE_SCHEMA,
        "scorers_agree": True,
    }


def toy_freeze_push() -> dict:
    return {
        "branch": "main",
        "commit_oid": "a" * 40,
        "observed_at": FREEZE_OBSERVED_TIME,
        "observer_id": "freeze-observer-a",
        "observer_implementation_sha256": "7" * 64,
        "pushed_at": FREEZE_TIME,
        "ref": "refs/heads/main",
        "remote_ref_evidence_sha256": "8" * 64,
        "remote_ref_oid": "a" * 40,
        "remote_url": "https://github.com/example/shohin.git",
        "schema": derive.FREEZE_PUSH_SCHEMA,
    }


def toy_beacon() -> dict:
    return {
        "pulse": {
            "certificateId": "toy-certificate",
            "chainIndex": 7,
            "outputValue": "AB" * 64,
            "period": 60_000,
            "pulseIndex": 123,
            "signatureValue": "cd" * 96,
            "statusCode": 0,
            "timeStamp": BEACON_TIME,
        }
    }


def toy_validator(
    identifier: str, implementation: str, evidence: str, raw_hash: str
) -> dict:
    return {
        "certificate_valid": True,
        "chain_valid": True,
        "evidence_sha256": evidence * 64,
        "implementation_sha256": implementation * 64,
        "raw_beacon_sha256": raw_hash,
        "signature_valid": True,
        "validated_at": "2030-01-01T01:01:00.000Z",
        "validator_id": identifier,
    }


def toy_beacon_verification(beacon: dict, beacon_raw: bytes) -> dict:
    pulse = beacon["pulse"]
    raw_hash = derive.sha256_bytes(beacon_raw)
    timestamp_ms = derive.parse_utc_timestamp_ms(pulse["timeStamp"], "toy pulse")
    return {
        "beacon_raw_sha256": raw_hash,
        "certificate_id": pulse["certificateId"],
        "certificate_sha256": "9" * 64,
        "chain_index": pulse["chainIndex"],
        "output_value_sha256": derive.sha256_bytes(bytes.fromhex(pulse["outputValue"])),
        "period_ms": pulse["period"],
        "pulse_index": pulse["pulseIndex"],
        "schema": derive.BEACON_VERIFICATION_SCHEMA,
        "signature_value_sha256": derive.sha256_bytes(
            pulse["signatureValue"].encode("utf-8")
        ),
        "status_code": pulse["statusCode"],
        "time_stamp": derive.canonical_utc_timestamp(timestamp_ms),
        "time_stamp_ms": timestamp_ms,
        "validators": [
            toy_validator("nist-validator-a", "a", "b", raw_hash),
            toy_validator("nist-validator-b", "c", "d", raw_hash),
        ],
        "validators_agree": True,
    }


def derive_toy(
    *,
    freeze: dict | None = None,
    prerequisite: dict | None = None,
    beacon: dict | None = None,
    verification: dict | None = None,
    **overrides,
) -> dict:
    freeze = toy_freeze_push() if freeze is None else freeze
    prerequisite = toy_prerequisite() if prerequisite is None else prerequisite
    beacon = toy_beacon() if beacon is None else beacon
    beacon_raw = raw_json(beacon)
    verification = (
        toy_beacon_verification(beacon, beacon_raw)
        if verification is None
        else verification
    )
    arguments = {
        "freeze_push_payload": freeze,
        "freeze_push_raw": raw_json(freeze),
        "prerequisite_payload": prerequisite,
        "prerequisite_raw": raw_json(prerequisite),
        "beacon_payload": beacon,
        "beacon_raw": beacon_raw,
        "beacon_verification_payload": verification,
        "beacon_verification_raw": raw_json(verification),
    }
    arguments.update(overrides)
    return derive.derive_seed_receipt(**arguments)


class ResidualPacketC2SeedTests(unittest.TestCase):
    def test_known_answer_vector_and_domain_separation(self):
        receipt = derive_toy()
        self.assertEqual(receipt["schema"], derive.RECEIPT_SCHEMA)
        self.assertEqual(receipt["seed_labels"], list(derive.SEED_LABELS))
        self.assertEqual(
            receipt["base_commitment_sha256"],
            "0c39b93579ca87dbd7417250fdf3f41b0b671b0a060381d7668d84a40de25ace",
        )
        self.assertEqual(
            receipt["seeds"]["board"]["sha256"],
            "3812d6b4eafb3bdd6b90211f69bb0e5835284660c75add554c51d233798c632b",
        )
        self.assertEqual(
            receipt["seeds"]["fit-b"]["sha256"],
            "15c555c0c9f05915c730247865a1121b96f9b74dfd59c1fc6e39f1a54bc3cf52",
        )
        digests = [receipt["seeds"][label]["sha256"] for label in derive.SEED_LABELS]
        self.assertEqual(len(digests), len(set(digests)))
        for label in derive.SEED_LABELS:
            seed = receipt["seeds"][label]
            self.assertEqual(int(seed["integer_decimal"]), int(seed["sha256"], 16))

    def test_derivation_is_byte_deterministic(self):
        first = derive_toy()
        second = derive_toy()
        self.assertEqual(first, second)
        self.assertEqual(
            derive.canonical_json_bytes(first), derive.canonical_json_bytes(second)
        )

    def test_every_receipt_and_beacon_binding_changes_commitment(self):
        baseline = derive_toy()["base_commitment_sha256"]
        freeze = toy_freeze_push()
        freeze["remote_ref_evidence_sha256"] = "e" * 64
        prerequisite = toy_prerequisite()
        prerequisite["confirmation_result_sha256"] = "f" * 64
        beacon = toy_beacon()
        beacon["pulse"]["outputValue"] = "AC" * 64
        verification_beacon = toy_beacon()
        verification_raw = raw_json(verification_beacon)
        verification = toy_beacon_verification(verification_beacon, verification_raw)
        verification["validators"][0]["evidence_sha256"] = "e" * 64
        variants = (
            derive_toy(freeze=freeze),
            derive_toy(prerequisite=prerequisite),
            derive_toy(beacon=beacon),
            derive_toy(beacon=verification_beacon, verification=verification),
        )
        self.assertTrue(
            all(variant["base_commitment_sha256"] != baseline for variant in variants)
        )

    def test_rejects_any_incomplete_prerequisite_gate(self):
        for field in (
            "advance_to_internalization",
            "all_locked_gates_pass",
            "independent_recomputation_complete",
            "result_immutable",
            "scorers_agree",
        ):
            prerequisite = toy_prerequisite()
            prerequisite[field] = False
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, field):
                derive_toy(prerequisite=prerequisite)

    def test_rejects_nonindependent_prerequisite_scorers(self):
        prerequisite = toy_prerequisite()
        prerequisite["independent_scorer_sha256"] = prerequisite[
            "primary_scorer_sha256"
        ]
        with self.assertRaisesRegex(ValueError, "implementations must differ"):
            derive_toy(prerequisite=prerequisite)
        prerequisite = toy_prerequisite()
        prerequisite["independent_score_receipt_sha256"] = prerequisite[
            "primary_score_receipt_sha256"
        ]
        with self.assertRaisesRegex(ValueError, "receipts must differ"):
            derive_toy(prerequisite=prerequisite)

    def test_rejects_hash_only_or_extra_freeze_receipt(self):
        hash_only = {"schema": derive.FREEZE_PUSH_SCHEMA, "raw_sha256": "1" * 64}
        with self.assertRaisesRegex(ValueError, "freeze push receipt keys differ"):
            derive_toy(freeze=hash_only)
        freeze = toy_freeze_push()
        freeze["unexpected"] = True
        with self.assertRaisesRegex(ValueError, r"extra=\['unexpected'\]"):
            derive_toy(freeze=freeze)

    def test_rejects_forged_freeze_ref_oid_or_timestamp(self):
        mutations = []
        freeze = toy_freeze_push()
        freeze["ref"] = "refs/heads/other"
        mutations.append((freeze, "ref does not match branch"))
        freeze = toy_freeze_push()
        freeze["remote_ref_oid"] = "b" * 40
        mutations.append((freeze, "does not match remotely observed ref OID"))
        freeze = toy_freeze_push()
        freeze["observed_at"] = "2029-12-31T23:59:59.999Z"
        mutations.append((freeze, "within five minutes"))
        freeze = toy_freeze_push()
        freeze["observed_at"] = "2030-01-01T00:05:00.001Z"
        mutations.append((freeze, "within five minutes"))
        freeze = toy_freeze_push()
        freeze["remote_url"] = "https://evil.example/example/shohin.git"
        mutations.append((freeze, "canonical HTTPS GitHub"))
        for freeze, message in mutations:
            with (
                self.subTest(message=message),
                self.assertRaisesRegex(ValueError, message),
            ):
                derive_toy(freeze=freeze)

    def test_rejects_noncanonical_freeze_receipt_bytes(self):
        freeze = toy_freeze_push()
        noncanonical = json.dumps(freeze, indent=2, sort_keys=True).encode("ascii")
        with self.assertRaisesRegex(ValueError, "canonical JSON bytes"):
            derive_toy(freeze=freeze, freeze_push_raw=noncanonical)

    def test_accepts_first_pulse_at_target_and_rejects_alternates(self):
        beacon = toy_beacon()
        beacon["pulse"]["timeStamp"] = "2030-01-01T01:00:00.000Z"
        derive_toy(beacon=beacon)
        beacon = toy_beacon()
        beacon["pulse"]["timeStamp"] = "2029-12-31T23:59:59.999Z"
        with self.assertRaisesRegex(ValueError, "first pulse at or after"):
            derive_toy(beacon=beacon)
        beacon = toy_beacon()
        beacon["pulse"]["timeStamp"] = "2030-01-01T01:01:00.000Z"
        with self.assertRaisesRegex(ValueError, "first pulse at or after"):
            derive_toy(beacon=beacon)

    def test_rejects_invalid_raw_pulse_fields(self):
        mutations = (
            ("period", 30_000, "period"),
            ("statusCode", 1, "statusCode"),
            ("outputValue", "ab" * 63, "512 bits"),
            ("signatureValue", "", "signatureValue"),
            ("certificateId", "", "certificateId"),
            ("chainIndex", 0, "chainIndex"),
            ("pulseIndex", True, "pulseIndex"),
        )
        for field, value, message in mutations:
            beacon = toy_beacon()
            beacon["pulse"][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, message):
                derive_toy(beacon=beacon)

    def test_rejects_beacon_receipt_mismatches(self):
        beacon = toy_beacon()
        beacon_raw = raw_json(beacon)
        baseline = toy_beacon_verification(beacon, beacon_raw)
        mutations = (
            ("beacon_raw_sha256", "0" * 64),
            ("chain_index", baseline["chain_index"] + 1),
            ("pulse_index", baseline["pulse_index"] + 1),
            ("time_stamp_ms", baseline["time_stamp_ms"] + 1),
            ("period_ms", 30_000),
            ("status_code", 1),
            ("output_value_sha256", "0" * 64),
            ("signature_value_sha256", "0" * 64),
            ("certificate_id", "other-certificate"),
        )
        for field, value in mutations:
            verification = copy.deepcopy(baseline)
            verification[field] = value
            with (
                self.subTest(field=field),
                self.assertRaisesRegex(ValueError, f"{field} does not match raw pulse"),
            ):
                derive_toy(beacon=beacon, verification=verification)

    def test_rejects_duplicate_beacon_validators(self):
        for field in ("validator_id", "implementation_sha256", "evidence_sha256"):
            beacon = toy_beacon()
            beacon_raw = raw_json(beacon)
            verification = toy_beacon_verification(beacon, beacon_raw)
            verification["validators"][1][field] = verification["validators"][0][field]
            with (
                self.subTest(field=field),
                self.assertRaisesRegex(ValueError, f"distinct {field}"),
            ):
                derive_toy(beacon=beacon, verification=verification)

    def test_rejects_validation_booleans_without_evidence(self):
        beacon = toy_beacon()
        beacon_raw = raw_json(beacon)
        verification = toy_beacon_verification(beacon, beacon_raw)
        del verification["validators"][0]["evidence_sha256"]
        with self.assertRaisesRegex(ValueError, "beacon validator keys differ"):
            derive_toy(beacon=beacon, verification=verification)
        verification = toy_beacon_verification(beacon, beacon_raw)
        verification["validators"][0]["signature_valid"] = False
        with self.assertRaisesRegex(ValueError, "signature_valid=true"):
            derive_toy(beacon=beacon, verification=verification)
        verification = toy_beacon_verification(beacon, beacon_raw)
        verification["validators_agree"] = False
        with self.assertRaisesRegex(ValueError, "validators_agree=true"):
            derive_toy(beacon=beacon, verification=verification)

    def test_rejects_noncanonical_or_extra_beacon_verification_receipt(self):
        beacon = toy_beacon()
        beacon_raw = raw_json(beacon)
        verification = toy_beacon_verification(beacon, beacon_raw)
        with self.assertRaisesRegex(ValueError, "canonical JSON bytes"):
            derive_toy(
                beacon=beacon,
                verification=verification,
                beacon_verification_raw=json.dumps(verification, indent=2).encode(),
            )
        verification["unexpected"] = True
        with self.assertRaisesRegex(ValueError, r"extra=\['unexpected'\]"):
            derive_toy(beacon=beacon, verification=verification)

    def test_json_parser_rejects_duplicate_keys_and_nonfinite_values(self):
        with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
            derive.parse_json_bytes(b'{"pulse":{},"pulse":{}}', "duplicate")
        with self.assertRaisesRegex(ValueError, "non-finite"):
            derive.parse_json_bytes(b'{"value":NaN}', "nonfinite")

    def test_immutable_reader_rejects_writable_symlink_and_noncanonical_receipts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            writable = root / "writable.json"
            writable.write_bytes(raw_json(toy_freeze_push()))
            with self.assertRaisesRegex(PermissionError, "read-only"):
                derive.read_immutable_json(writable, "writable", require_canonical=True)
            os.chmod(writable, 0o444)
            payload, raw = derive.read_immutable_json(
                writable, "read-only", require_canonical=True
            )
            self.assertEqual(payload, toy_freeze_push())
            self.assertEqual(raw, raw_json(toy_freeze_push()))
            symlink = root / "link.json"
            symlink.symlink_to(writable)
            with self.assertRaisesRegex(ValueError, "regular file"):
                derive.read_immutable_json(symlink, "symlink", require_canonical=True)
            os.chmod(writable, 0o600)
            writable.write_text(json.dumps(toy_freeze_push(), indent=2))
            os.chmod(writable, 0o444)
            with self.assertRaisesRegex(ValueError, "canonical JSON bytes"):
                derive.read_immutable_json(writable, "pretty", require_canonical=True)
            os.chmod(writable, 0o600)

    def test_exclusive_writer_is_read_only_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "receipt.json"
            payload = derive.canonical_json_bytes(derive_toy())
            digest = derive.write_exclusive_readonly(output, payload)
            self.assertEqual(digest, derive.sha256_bytes(payload))
            self.assertEqual(output.read_bytes(), payload)
            self.assertEqual(output.stat().st_mode & 0o222, 0)
            with self.assertRaises(FileExistsError):
                derive.write_exclusive_readonly(output, payload)
            os.chmod(output, 0o600)

    def test_offline_cli_round_trip_uses_only_supplied_toy_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixtures = {
                "freeze.json": toy_freeze_push(),
                "prerequisite.json": toy_prerequisite(),
                "beacon.json": toy_beacon(),
            }
            beacon_raw = raw_json(fixtures["beacon.json"])
            fixtures["verification.json"] = toy_beacon_verification(
                fixtures["beacon.json"], beacon_raw
            )
            paths = {}
            for name, payload in fixtures.items():
                path = root / name
                path.write_bytes(raw_json(payload))
                os.chmod(path, 0o444)
                paths[name] = path
            output_path = root / "seeds.json"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(Path(derive.__file__)),
                    "--freeze-push-receipt",
                    str(paths["freeze.json"]),
                    "--prerequisite-pass-receipt",
                    str(paths["prerequisite.json"]),
                    "--beacon-json",
                    str(paths["beacon.json"]),
                    "--beacon-verification-receipt",
                    str(paths["verification.json"]),
                    "--out",
                    str(output_path),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            summary = json.loads(completed.stdout)
            self.assertEqual(summary["schema"], derive.RECEIPT_SCHEMA)
            self.assertEqual(
                summary["receipt_sha256"], derive.sha256_bytes(output_path.read_bytes())
            )
            self.assertEqual(
                derive.parse_json_bytes(output_path.read_bytes(), "output"),
                derive_toy(),
            )
            self.assertEqual(output_path.stat().st_mode & 0o222, 0)
            for path in (*paths.values(), output_path):
                os.chmod(path, 0o600)

    def test_module_has_no_network_import_path(self):
        source_path = Path(derive.__file__)
        tree = ast.parse(source_path.read_text())
        imported_roots = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_roots.update(
                    alias.name.split(".", 1)[0] for alias in node.names
                )
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_roots.add(node.module.split(".", 1)[0])
        forbidden = {"http", "requests", "socket", "urllib", "urllib3"}
        self.assertFalse(imported_roots & forbidden)


if __name__ == "__main__":
    unittest.main()
