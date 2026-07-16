#!/usr/bin/env python3
"""Unit and process-interface tests for R12 packet transport v3."""

from __future__ import annotations

import ast
import hashlib
import inspect
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from pipeline import post_commit_interface_falsifier as v1
from pipeline import post_commit_packet_transport_falsifier as v2


class RoleUnitTests(unittest.TestCase):
    def setUp(self) -> None:
        self.sources = b"".join(
            v2.canonical_json_bytes({"source": list(source)})
            for source in ((1, 2, 3, 4), (5, 6, 7, 8))
        )
        self.writer_header = {"protocol_id": v2.PROTOCOL_ID, "role": "writer"}

    def test_writer_packets_are_source_free_and_equal_width(self) -> None:
        payload = v2._stream(self.writer_header, self.sources)
        state = v2.run_role("writer", payload, "--arm", "state")
        motor = v2.run_role("writer", payload, "--arm", "motor")
        self.assertEqual(state.exit_code, 0)
        self.assertEqual(motor.exit_code, 0)
        self.assertEqual(state.command[1], "-p")
        self.assertEqual(
            v2.sha256_bytes(state.command[2].encode("utf-8")),
            state.sandbox_profile_sha256,
        )
        self.assertIn("<ROLE_CWD>", state.command[2])
        self.assertEqual(Path(state.command[3]), v2.ROLE_PYTHON)
        self.assertEqual(state.command[4:6], ("-I", "-S"))
        self.assertEqual(Path(state.command[6]), v2.ROLE_SCRIPT)
        self.assertTrue(state.sandbox_enforced)
        self.assertEqual(state.sandbox_launcher, str(v2.SANDBOX_EXEC))
        state_rows = v2._json_lines(state.output)
        motor_rows = v2._json_lines(motor.output)
        self.assertEqual(state_rows, [{"values": [1, 2, 3, 4]}, {"values": [5, 6, 7, 8]}])
        self.assertEqual(motor_rows, [{"values": [1, 2, 0, 0]}, {"values": [5, 6, 0, 0]}])
        self.assertTrue(all(set(row) == {"values"} for row in state_rows + motor_rows))

    def test_updater_receives_one_event_and_matches_exact_application(self) -> None:
        payload = v2._stream(self.writer_header, self.sources)
        packets = v2.require_success(v2.run_role("writer", payload, "--arm", "state"))
        update = v1.AffineUpdate(v1.identity_matrix(), (1, 2, 3, 4))
        updater_input = v2._stream(
            {"protocol_id": v2.PROTOCOL_ID, "role": "updater", "update": update.serialized()},
            b"",
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            v2._immutable_write(root / "input.jsonl", packets)
            updated = v2.run_role(
                "updater",
                updater_input,
                "--packet-in",
                "input.jsonl",
                "--packet-out",
                "output.jsonl",
                cwd=root,
                file_inputs=("input.jsonl",),
                file_outputs=("output.jsonl",),
            )
            self.assertEqual(updated.exit_code, 0)
            self.assertEqual(updated.output, b"")
            self.assertEqual(
                v2._json_lines((root / "output.jsonl").read_bytes()),
                [{"values": [2, 4, 6, 8]}, {"values": [6, 8, 10, 12]}],
            )
            self.assertEqual(updated.file_inputs[0]["mode"], "0444")
            self.assertEqual(updated.file_outputs[0]["mode"], "0444")
            self.assertEqual(updated.cwd_regular_files_before, ("input.jsonl",))
            self.assertEqual(
                updated.cwd_regular_files_after, ("input.jsonl", "output.jsonl")
            )

    def test_reader_emits_recoded_symbol_not_raw_answer(self) -> None:
        packets = v2.canonical_json_bytes({"values": [1, 2, 3, 4]})
        permutation = list(range(1, 17)) + [0]
        reader_input = v2._stream(
            {
                "protocol_id": v2.PROTOCOL_ID,
                "role": "reader",
                "consumer": [1, 0, 0, 0],
                "output_permutation": permutation,
            },
            b"",
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            v2._immutable_write(root / "terminal.jsonl", packets)
            run = v2.run_role(
                "reader",
                reader_input,
                "--packet-in",
                "terminal.jsonl",
                cwd=root,
                file_inputs=("terminal.jsonl",),
            )
            self.assertEqual(run.exit_code, 0)
            self.assertEqual(v2._json_lines(run.output), [{"symbol": 2}])
            self.assertEqual(run.cwd_regular_files_before, ("terminal.jsonl",))
            self.assertEqual(run.cwd_regular_files_after, ("terminal.jsonl",))

    def test_role_executable_has_no_orchestrator_seed_or_v1_import(self) -> None:
        source = v2.ROLE_SCRIPT.read_bytes()
        self.assertNotIn(b"CHALLENGE_SEED", source)
        self.assertNotIn(b"post_commit_interface_falsifier", source)
        self.assertFalse(hasattr(v2, "CHALLENGE_SEED"))

    def test_role_headers_reject_leakage_fields(self) -> None:
        bad_writer = v2.run_role(
            "writer",
            v2._stream(
                {
                    "protocol_id": v2.PROTOCOL_ID,
                    "role": "writer",
                    "consumer": [1, 0, 0, 0],
                },
                self.sources,
            ),
            "--arm",
            "state",
        )
        self.assertNotEqual(bad_writer.exit_code, 0)

    def test_every_transport_and_reader_call_binds_the_run_root(self) -> None:
        tree = ast.parse(v2.SCRIPT.read_text(encoding="utf-8"))
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in {"_transport", "_run_reader"}
        ]
        self.assertGreater(len(calls), 0)
        for call in calls:
            self.assertIn(
                "run_root",
                {keyword.arg for keyword in call.keywords},
                f"line {call.lineno} does not bind run_root",
            )


class AlgebraAndChallengeTests(unittest.TestCase):
    def test_os_sandbox_blocks_parent_and_repository_access(self) -> None:
        with tempfile.TemporaryDirectory(dir="/private/tmp") as directory:
            proof = v2.run_sandbox_probe(Path(directory))
        self.assertEqual(proof["exit_code"], 0)
        self.assertTrue(all(proof["checks"].values()))
        self.assertEqual(set(proof["checks"]), v2.SANDBOX_PROBE_CHECK_NAMES)
        self.assertEqual(
            proof["sandbox_forbidden_roots"],
            list(v2.SANDBOX_SCIENTIFIC_FORBIDDEN_ROOTS),
        )

    def test_role_file_declarations_reject_path_traversal(self) -> None:
        for value in ("../secret", "/tmp/secret", ".", ""):
            with self.subTest(value=value), self.assertRaises(v2.TransportError):
                v2._validate_role_filename(value)

    def test_seed_independence_uses_manifest_hashes_after_packet_cleanup(self) -> None:
        state_payload = v2.canonical_json_bytes({"values": [1, 2, 3, 4]})
        motor_payload = v2.canonical_json_bytes({"values": [1, 2, 0, 0]})
        manifest = {
            "state_packets_sha256": v2.sha256_bytes(state_payload),
            "motor_packets_sha256": v2.sha256_bytes(motor_payload),
        }
        with tempfile.TemporaryDirectory() as directory:
            state_path = Path(directory) / "state.jsonl"
            motor_path = Path(directory) / "motor.jsonl"
            state_path.write_bytes(state_payload)
            motor_path.write_bytes(motor_payload)
        self.assertFalse(state_path.exists())
        self.assertFalse(motor_path.exists())
        self.assertTrue(
            v2.phase_one_seed_independence(
                state_payload,
                motor_payload,
                manifest,
                "a" * 64,
                "b" * 64,
            )
        )
        self.assertFalse(
            v2.phase_one_seed_independence(
                state_payload + b"\n",
                motor_payload,
                manifest,
                "a" * 64,
                "b" * 64,
            )
        )

    def test_v3_challenges_are_deterministic_and_cell_recoded(self) -> None:
        nonce = "17" * 32
        primary_seed = v2._derive_post_commit_seed(
            "a" * 64, nonce, v2.PRIMARY_CHALLENGE_DOMAIN
        )
        alternate_seed = v2._derive_post_commit_seed(
            "a" * 64, nonce, v2.ALTERNATE_CHALLENGE_DOMAIN
        )
        first = v2.generate_challenges(primary_seed)
        second = v2.generate_challenges(primary_seed)
        alternate = v2.generate_challenges(alternate_seed)
        self.assertEqual(first["serialized"], second["serialized"])
        self.assertNotEqual(first["payload_sha256"], alternate["payload_sha256"])
        self.assertEqual(
            tuple((item.challenge_id, item.kind, item.depth) for item in first["public"]),
            v2.EXPECTED_PUBLIC_LAYOUT,
        )
        self.assertEqual(
            tuple((item.challenge_id, item.kind, item.depth) for item in first["decisive"]),
            v2.EXPECTED_DECISIVE_LAYOUT,
        )
        for challenge in list(first["public"]) + list(first["decisive"]):
            self.assertEqual(sorted(challenge.output_permutation), list(range(17)))
            self.assertTrue(
                any(index != value for index, value in enumerate(challenge.output_permutation))
            )

    def test_post_commit_seed_uses_frozen_concatenation_contract(self) -> None:
        manifest = "a" * 64
        nonce = "17" * 32
        expected = int.from_bytes(
            hashlib.sha256(
                v2.PRIMARY_CHALLENGE_DOMAIN.encode("ascii")
                + manifest.encode("ascii")
                + bytes.fromhex(nonce)
            ).digest()[:8],
            "big",
        )
        self.assertEqual(
            v2._derive_post_commit_seed(
                manifest, nonce, v2.PRIMARY_CHALLENGE_DOMAIN
            ),
            expected,
        )
        with self.assertRaises(v2.TransportError):
            v2._derive_post_commit_seed(
                manifest, "not-a-256-bit-nonce", v2.PRIMARY_CHALLENGE_DOMAIN
            )

    def test_primary_post_commit_generator_accepts_no_caller_nonce(self) -> None:
        signature = inspect.signature(v2.generate_challenges_after_commit)
        self.assertEqual(tuple(signature.parameters), ("phase_one", "workdir"))
        self.assertEqual(tuple(inspect.signature(v2._build_core).parameters), ())
        tree = ast.parse(inspect.getsource(v2.build_report))
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
        build_core = next(
            node
            for node in calls
            if isinstance(node.func, ast.Name) and node.func.id == "_build_core"
        )
        self.assertEqual(build_core.args, [])
        self.assertEqual(build_core.keywords, [])

    def test_challenge_round_trip(self) -> None:
        challenge = v2.generate_challenges(12345)["decisive"][0]
        self.assertEqual(v2.challenge_from_mapping(challenge.serialized()), challenge)


class ReportSchemaTests(unittest.TestCase):
    def test_canonical_publication_is_delegated_to_independent_process(self) -> None:
        report = {
            "deterministic_replay": {"first_core_payload_sha256": "a" * 64},
            "scientific_identity": {
                "scientific_commit": "b" * 40,
                "scientific_paths": {
                    "pipeline/audit_post_commit_packet_transport_v3.py": "c" * 64
                },
            },
            "payload_sha256": "d" * 64,
        }
        completed = mock.Mock(
            returncode=1, stdout=b"", stderr=b"FAIL: mutation detected\n"
        )
        with tempfile.TemporaryDirectory(), mock.patch.object(
            v2, "verify_report"
        ), mock.patch.object(v2.subprocess, "run", return_value=completed) as run:
            artifact = Path("artifacts/r12/post_commit_packet_transport_v3.json")
            receipt = Path(
                "artifacts/r12/post_commit_packet_transport_v3.receipt.json"
            )
            with self.assertRaises(v2.TransportError):
                v2.run_independent_publish(report, artifact, receipt)
            command = run.call_args.args[0]
            self.assertIn("publish", command)
            self.assertIn(str(artifact.resolve()), command)
            self.assertIn(str(receipt.resolve()), command)
            self.assertNotIn(str(artifact), command)
            self.assertNotIn(str(receipt), command)

    def test_parent_has_no_receipt_acceptor_or_canonical_writer(self) -> None:
        self.assertFalse(hasattr(v2, "_validate_independent_receipt"))
        self.assertFalse(hasattr(v2, "write_immutable_report"))
        with self.assertRaises(v2.TransportError):
            v2._immutable_write(v2.CANONICAL_ARTIFACT, b"forged\n")
        with self.assertRaises(v2.TransportError):
            v2._immutable_write(v2.CANONICAL_RECEIPT, b"forged\n")

    def test_symbol_parser_rejects_raw_reader_output(self) -> None:
        with self.assertRaises(v2.TransportError):
            v2._symbol_rows(v2.canonical_json_bytes({"raw": 3}))

    def test_symbol_scorer_requires_byte_canonical_equality(self) -> None:
        canonical = v2.canonical_json_bytes({"symbol": 3})
        with self.assertRaises(v2.TransportError):
            v2._score_symbols(b'{"symbol": 3}\n', canonical)

    def test_payload_hash_verifier_binds_scientific_identity(self) -> None:
        identity = {
            "scientific_commit": "a" * 40,
            "scientific_source_tree_sha256": "b" * 64,
        }
        core = {
            "audit": "post_commit_packet_transport_v3",
            "protocol_id": v2.PROTOCOL_ID,
            "schema_version": v2.SCHEMA_VERSION,
            "code_sha256": v2.sha256_bytes(v2.SCRIPT.read_bytes()),
            "scientific_identity": identity,
            "config": {},
            "phase_one": {
                "scientific_commit": identity["scientific_commit"],
                "scientific_source_tree_sha256": identity[
                    "scientific_source_tree_sha256"
                ],
            },
            "phase_two": {},
            "custody_events": [],
            "public_results": [],
            "decisive_results": [],
            "horizon_decoy_results": [],
            "executed_decoys": {},
            "role_invocations": [],
            "role_invocation_counts": {},
            "sandbox_probe": {},
            "deterministic_replay": v2.pending_replay_record(),
            "gates": {name: True for name in v2.FROZEN_GATE_NAMES},
            "pass": False,
            "claim_boundary": "test",
        }
        core["gates"]["full_deterministic_replay_byte_identical"] = False
        core["payload_sha256"] = v2.sha256_bytes(v2.canonical_json_bytes(core))
        report = v2._finalize_deterministic_replay(
            core, json.loads(json.dumps(core))
        )
        with mock.patch.object(v2, "scientific_identity", return_value=identity):
            with self.assertRaises(v2.TransportError):
                v2.verify_report(report)
            with mock.patch.object(v2, "verify_evidence_shape"):
                v2.verify_report(report)
                mutated = json.loads(json.dumps(report))
                mutated["pass"] = False
                with self.assertRaises(v2.TransportError):
                    v2.verify_report(mutated)
                wrong_identity = json.loads(json.dumps(report))
                wrong_identity["scientific_identity"] = {
                    **identity,
                    "scientific_commit": "c" * 40,
                }
                wrong_identity["payload_sha256"] = v2.sha256_bytes(
                    v2.canonical_json_bytes(
                        {
                            key: value
                            for key, value in wrong_identity.items()
                            if key != "payload_sha256"
                        }
                    )
                )
                with self.assertRaises(v2.TransportError):
                    v2.verify_report(wrong_identity)

    def test_verifier_rejects_incomplete_gate_schema(self) -> None:
        report = {"gates": {"x": True}, "pass": True}
        report["payload_sha256"] = v2.sha256_bytes(v2.canonical_json_bytes(report))
        with self.assertRaises(v2.TransportError):
            v2.verify_report(report)

    def test_full_replay_gate_requires_two_byte_identical_core_reports(self) -> None:
        core = {
            "gates": {name: True for name in v2.FROZEN_GATE_NAMES},
            "deterministic_replay": v2.pending_replay_record(),
            "pass": False,
        }
        core["gates"]["full_deterministic_replay_byte_identical"] = False
        core["payload_sha256"] = v2.sha256_bytes(v2.canonical_json_bytes(core))
        finalized = v2._finalize_deterministic_replay(
            core, json.loads(json.dumps(core))
        )
        self.assertTrue(finalized["pass"])
        self.assertTrue(finalized["gates"]["full_deterministic_replay_byte_identical"])
        self.assertEqual(
            finalized["deterministic_replay"]["status"],
            "confirmed_byte_identical",
        )
        changed = json.loads(json.dumps(core))
        changed["extra"] = True
        with self.assertRaises(v2.TransportError):
            v2._finalize_deterministic_replay(core, changed)


if __name__ == "__main__":
    unittest.main()
