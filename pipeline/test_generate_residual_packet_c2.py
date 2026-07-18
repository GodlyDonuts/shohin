#!/usr/bin/env python3
"""Toy-only tests for deferred-seed RSP-C2 materialization custody."""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import os
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import derive_residual_packet_c2_seeds as derive
import generate_residual_packet_c2 as generate


TOY_LABEL = "TOY_ONLY_NEVER_PRODUCTION_GENERATOR_FIXTURE_A"
TOY_FREEZE_COMMIT = "a" * 40
TOY_FREEZE_TIME = "2035-02-03T04:05:06.000Z"
TOY_FREEZE_OBSERVED_TIME = "2035-02-03T04:07:06.000Z"


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
        "schema": generate.PREREQUISITE_SCHEMA,
        "scorers_agree": True,
    }


def toy_beacon() -> dict:
    return {
        "pulse": {
            "certificateId": f"{TOY_LABEL}_CERTIFICATE",
            "chainIndex": 17,
            "outputValue": "AB" * 64,
            "period": 60_000,
            "pulseIndex": 29,
            "signatureValue": f"{TOY_LABEL}_SIGNATURE",
            "statusCode": 0,
            "timeStamp": "2035-02-03T05:05:36.000Z",
        }
    }


def compact_json(payload: dict) -> bytes:
    return (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()


def toy_freeze_push() -> dict:
    return {
        "branch": "toy-c2-freeze",
        "commit_oid": TOY_FREEZE_COMMIT,
        "observed_at": TOY_FREEZE_OBSERVED_TIME,
        "observer_id": "toy-freeze-observer",
        "observer_implementation_sha256": "7" * 64,
        "pushed_at": TOY_FREEZE_TIME,
        "ref": "refs/heads/toy-c2-freeze",
        "remote_ref_evidence_sha256": "8" * 64,
        "remote_ref_oid": TOY_FREEZE_COMMIT,
        "remote_url": "https://github.com/example/toy-shohin.git",
        "schema": derive.FREEZE_PUSH_SCHEMA,
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
        "validated_at": "2035-02-03T05:06:00.000Z",
        "validator_id": identifier,
    }


def toy_beacon_verification(beacon: dict, beacon_raw: bytes) -> dict:
    pulse = beacon["pulse"]
    raw_hash = hashlib.sha256(beacon_raw).hexdigest()
    timestamp_ms = derive.parse_utc_timestamp_ms(pulse["timeStamp"], "toy pulse")
    return {
        "beacon_raw_sha256": raw_hash,
        "certificate_id": pulse["certificateId"],
        "certificate_sha256": "9" * 64,
        "chain_index": pulse["chainIndex"],
        "output_value_sha256": hashlib.sha256(
            bytes.fromhex(pulse["outputValue"])
        ).hexdigest(),
        "period_ms": pulse["period"],
        "pulse_index": pulse["pulseIndex"],
        "schema": derive.BEACON_VERIFICATION_SCHEMA,
        "signature_value_sha256": hashlib.sha256(
            pulse["signatureValue"].encode()
        ).hexdigest(),
        "status_code": pulse["statusCode"],
        "time_stamp": derive.canonical_utc_timestamp(timestamp_ms),
        "time_stamp_ms": timestamp_ms,
        "validators": [
            toy_validator("toy-validator-a", "a", "b", raw_hash),
            toy_validator("toy-validator-b", "c", "d", raw_hash),
        ],
        "validators_agree": True,
    }


def toy_seed_receipt() -> tuple[dict, bytes, bytes]:
    prerequisite = toy_prerequisite()
    prerequisite_raw = compact_json(prerequisite)
    freeze = toy_freeze_push()
    freeze_raw = compact_json(freeze)
    beacon = toy_beacon()
    beacon_raw = compact_json(beacon)
    verification = toy_beacon_verification(beacon, beacon_raw)
    receipt = derive.derive_seed_receipt(
        freeze_push_payload=freeze,
        freeze_push_raw=freeze_raw,
        prerequisite_payload=prerequisite,
        prerequisite_raw=prerequisite_raw,
        beacon_payload=beacon,
        beacon_raw=beacon_raw,
        beacon_verification_payload=verification,
        beacon_verification_raw=compact_json(verification),
    )
    return receipt, derive.canonical_json_bytes(receipt), prerequisite_raw


def compiler_rows(rows: list[dict]) -> list[dict]:
    return [row for row in rows if row["kind"] == "compiler"]


class ResidualPacketC2GeneratorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = generate.build_toy_fixture_bundle(TOY_LABEL)

    def test_toy_bundle_is_deterministic_and_unmistakably_nonproduction(self):
        second = generate.build_toy_fixture_bundle(TOY_LABEL)
        self.assertEqual(self.bundle, second)
        self.assertEqual(self.bundle["schema"], "toy_only_rsp_c2_bundle_fixture_v1")
        board = self.bundle["board"]
        self.assertEqual(board["schema"], "toy_only_rsp_c2_board_fixture_v1")
        self.assertEqual(board["case_count"], 16)
        self.assertLess(board["case_count"], generate.PER_STRATUM)
        self.assertTrue(all(row["id"].startswith(TOY_LABEL) for row in board["rows"]))
        self.assertNotEqual(board["profile"], generate.PRODUCTION_PROFILE)
        self.assertEqual(self.bundle["manifest"]["program_count"], 16)

    def test_toy_board_obeys_geometry_and_uniqueness(self):
        rows = self.bundle["board"]["rows"]
        self.assertEqual(
            Counter(row["stratum"] for row in rows),
            Counter({name: 4 for name in generate.STRATUM_ORDER}),
        )
        signatures = []
        sources = []
        packets = []
        trajectories = []
        answers = []
        for row in rows:
            states = generate.trajectory(row["initial_state"], row["operations"])
            self.assertGreater(min(states), 0)
            self.assertEqual(row["trajectory"], list(states))
            self.assertEqual(row["answer"], states[-1])
            self.assertEqual(
                row["source"],
                generate.render_source(
                    row["initial_state"], row["operations"], row["template_id"]
                ),
            )
            self.assertEqual(
                row["packet"],
                generate.render_packet(row["initial_state"], row["operations"]),
            )
            signatures.append(
                generate.semantic_signature(row["initial_state"], row["operations"])
            )
            sources.append(row["source"])
            packets.append(row["packet"])
            trajectories.append(tuple(row["trajectory"]))
            answers.append(row["answer"])
        for values in (signatures, sources, packets, trajectories, answers):
            self.assertEqual(len(values), len(set(values)))

    def test_treatment_and_sham_hold_every_locked_token_parity_field(self):
        treatment = self.bundle["treatment"]
        sham = self.bundle["sham"]
        manifest = self.bundle["manifest"]
        self.assertEqual(len(treatment), 64)
        self.assertEqual(len(treatment), len(sham))
        self.assertEqual(
            [row["completion_prompt"] for row in treatment],
            [row["completion_prompt"] for row in sham],
        )
        self.assertEqual(
            [row for row in treatment if row["kind"] == "updater"],
            [row for row in sham if row["kind"] == "updater"],
        )
        self.assertEqual(
            manifest["token_accounting"]["treatment"],
            manifest["token_accounting"]["sham"],
        )
        treatment_compilers = compiler_rows(treatment)
        sham_compilers = compiler_rows(sham)
        treatment_ids = [row["response_program_id"] for row in treatment_compilers]
        sham_ids = [row["response_program_id"] for row in sham_compilers]
        self.assertCountEqual(treatment_ids, sham_ids)
        self.assertTrue(
            all(
                treatment_row["program_id"] != sham_row["response_program_id"]
                for treatment_row, sham_row in zip(treatment_compilers, sham_compilers)
            )
        )

    def test_updater_observations_are_wrong_and_exclude_source_values(self):
        board_answers = {row["answer"] for row in self.bundle["board"]["rows"]}
        for row in self.bundle["treatment"]:
            if row["kind"] != "updater":
                continue
            prompt = row["completion_prompt"]
            packet = prompt.split("Packet:\n", 1)[1].split("\nObservation:", 1)[0]
            observed = int(prompt.split("\nObservation: ", 1)[1].split("\n", 1)[0])
            state_text = packet.split("|S=", 1)[1].split("|", 1)[0]
            plan_text = packet.split("|R=", 1)[1][:-1]
            opcode, operand_text = plan_text.split(",", 1)[0].split(":", 1)
            kind = {"ADD": "add", "MUL": "multiply", "SUB": "subtract"}[opcode]
            state = int(state_text)
            self.assertNotEqual(
                observed, generate.apply_operation(state, [kind, int(operand_text)])
            )
            self.assertNotIn(state, board_answers)
            self.assertNotIn(observed, board_answers)

    def test_seed_receipt_validation_rejects_any_seed_substitution(self):
        receipt, _, _ = toy_seed_receipt()
        integers, commitments = generate.validate_seed_receipt(receipt)
        self.assertEqual(set(integers), set(generate.SEED_LABELS))
        self.assertEqual(len(set(commitments.values())), len(generate.SEED_LABELS))
        tampered = copy.deepcopy(receipt)
        tampered["seeds"]["board"]["integer_decimal"] = str(
            int(tampered["seeds"]["board"]["integer_decimal"]) + 1
        )
        with self.assertRaisesRegex(ValueError, "board seed"):
            generate.validate_seed_receipt(tampered)
        tampered = copy.deepcopy(receipt)
        tampered["freeze"]["remote_url"] = "https://github.com/example/other-toy.git"
        with self.assertRaisesRegex(ValueError, "does not replay"):
            generate.validate_seed_receipt(tampered)
        tampered = copy.deepcopy(receipt)
        tampered["beacon"]["verification"]["validators"][0]["evidence_sha256"] = (
            "e" * 64
        )
        with self.assertRaisesRegex(ValueError, "verification raw SHA-256"):
            generate.validate_seed_receipt(tampered)

    def test_custody_requires_exact_read_only_hash_bound_receipts(self):
        receipt, seed_raw, prerequisite_raw = toy_seed_receipt()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prerequisite_path = root / f"{TOY_LABEL}_prerequisite.json"
            seed_path = root / f"{TOY_LABEL}_seed_receipt.json"
            tokenizer_path = root / f"{TOY_LABEL}_tokenizer.json"
            runtime_path = root / f"{TOY_LABEL}_runtime.txt"
            provenance_path = root / f"{TOY_LABEL}_provenance.json"
            prerequisite_path.write_bytes(prerequisite_raw)
            seed_path.write_bytes(seed_raw)
            tokenizer_path.write_bytes(f"{TOY_LABEL}_TOKENIZER_BYTES".encode())
            runtime_path.write_bytes(f"{TOY_LABEL}_RUNTIME_BYTES".encode())
            prerequisite_hash = hashlib.sha256(prerequisite_raw).hexdigest()
            runtime_hash = hashlib.sha256(runtime_path.read_bytes()).hexdigest()
            provenance = {
                "auditor_sha256": generate.sha256_file(generate.AUDITOR_PATH),
                "c1_closure_sha256": generate.FROZEN_C1_CLOSURE_SHA256,
                "freeze_commit": TOY_FREEZE_COMMIT,
                "freeze_push_receipt_sha256": hashlib.sha256(
                    compact_json(toy_freeze_push())
                ).hexdigest(),
                "generator_sha256": generate.sha256_file(generate.__file__),
                "prerequisite_receipt_sha256": prerequisite_hash,
                "preregistration_sha256": generate.FROZEN_PREREGISTRATION_SHA256,
                "runtime_receipt_sha256": runtime_hash,
                "schema": generate.PROVENANCE_SCHEMA,
                "seed_derivation_sha256": generate.sha256_file(
                    generate.SEED_DERIVATION_PATH
                ),
                "tokenizer_sha256": hashlib.sha256(
                    tokenizer_path.read_bytes()
                ).hexdigest(),
            }
            provenance_raw = generate.canonical_json_bytes(provenance)
            provenance_path.write_bytes(provenance_raw)
            for path in (
                prerequisite_path,
                seed_path,
                tokenizer_path,
                runtime_path,
                provenance_path,
            ):
                os.chmod(path, 0o444)
            custody = generate.load_production_custody(
                prerequisite_path=prerequisite_path,
                prerequisite_sha256=prerequisite_hash,
                seed_receipt_path=seed_path,
                seed_receipt_sha256=hashlib.sha256(seed_raw).hexdigest(),
                provenance_path=provenance_path,
                provenance_sha256=hashlib.sha256(provenance_raw).hexdigest(),
                tokenizer_path=tokenizer_path,
                runtime_receipt_path=runtime_path,
            )
            self.assertEqual(custody.seed_receipt, receipt)
            with self.assertRaisesRegex(ValueError, "prerequisite receipt SHA-256"):
                generate.load_production_custody(
                    prerequisite_path=prerequisite_path,
                    prerequisite_sha256="0" * 64,
                    seed_receipt_path=seed_path,
                    seed_receipt_sha256=hashlib.sha256(seed_raw).hexdigest(),
                    provenance_path=provenance_path,
                    provenance_sha256=hashlib.sha256(provenance_raw).hexdigest(),
                    tokenizer_path=tokenizer_path,
                    runtime_receipt_path=runtime_path,
                )

    def test_exclusive_writer_is_read_only_and_never_overwrites(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / f"{TOY_LABEL}_immutable.json"
            payload = generate.canonical_json_bytes(
                {"fixture_label": TOY_LABEL, "schema": "toy_only_write_fixture_v1"}
            )
            observed = generate._exclusive_immutable_write(destination, payload)
            self.assertEqual(observed, hashlib.sha256(payload).hexdigest())
            self.assertEqual(destination.stat().st_mode & 0o222, 0)
            with self.assertRaises(FileExistsError):
                generate._exclusive_immutable_write(destination, payload)

    def test_production_cli_has_receipt_only_seed_custody(self):
        parser = generate.build_parser()
        subparsers = next(
            action
            for action in parser._actions
            if hasattr(action, "choices") and action.choices
        )
        for command in ("board", "data"):
            options = {
                option
                for action in subparsers.choices[command]._actions
                for option in action.option_strings
            }
            self.assertIn("--seed-receipt", options)
            self.assertIn("--seed-receipt-sha256", options)
            self.assertNotIn("--seed", options)
            self.assertNotIn("--board-seed", options)
            self.assertNotIn("--training-seed", options)

    def test_generator_has_no_c1_executable_import(self):
        tree = ast.parse(Path(generate.__file__).read_text())
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
        self.assertFalse(any("residual_packet_v1" in name for name in imported))
        self.assertFalse(
            {name.split(".", 1)[0] for name in imported}
            & {"http", "requests", "socket", "urllib", "urllib3"}
        )


if __name__ == "__main__":
    unittest.main()
