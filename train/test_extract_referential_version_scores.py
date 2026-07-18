#!/usr/bin/env python3
"""CPU-only gate, provenance, and probability tests for the R10 extractor."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

import evaluate_version_space_workspace as evaluator
import extract_referential_version_scores as extractor
from categorical_microcode import OPCODES, QUERIES


ROOT = Path(__file__).resolve().parents[1]
EXTRACTOR_PATH = Path(extractor.__file__).resolve()
EVALUATOR_PATH = Path(evaluator.__file__).resolve()


def write_bytes(path, payload):
    path = Path(path).resolve()
    path.write_bytes(payload)
    return path


def write_json(path, payload):
    path = Path(path).resolve()
    path.write_text(
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def build_preflight_fixture(root):
    root = Path(root).resolve()
    base = write_bytes(root / "base.pt", b"base\n")
    pointer = write_bytes(root / "pointer.pt", b"pointer\n")
    adapter = write_bytes(root / "adapter.pt", b"adapter\n")
    data = write_bytes(root / "calibration.jsonl", b"{}\n")
    tokenizer = write_bytes(root / "tokenizer.json", b"{}\n")
    output = root / "scores.json"

    hashes = {
        "adapter": extractor.sha256_file(adapter),
        "data": extractor.sha256_file(data),
        "tokenizer": extractor.sha256_file(tokenizer),
    }
    training_sha256 = "8" * 64
    structural = write_json(
        root / "structural.json",
        {
            "audit": extractor.STRUCTURAL_ADMISSION_AUDIT,
            "all_checks_pass": True,
            "eval_sha256": hashes["data"],
            "train_sha256": training_sha256,
            "tokenizer_sha256": hashes["tokenizer"],
        },
    )
    labels = write_json(
        root / "labels.json",
        {
            "audit": extractor.LABEL_ADMISSION_AUDIT,
            "all_checks_pass": True,
            "tokenizer_sha256": hashes["tokenizer"],
            "datasets": {
                "eval": {"all_checks_pass": True, "sha256": hashes["data"]},
                "train": {
                    "all_checks_pass": True,
                    "sha256": training_sha256,
                },
            },
        },
    )
    hashes["structural"] = extractor.sha256_file(structural)
    hashes["labels"] = extractor.sha256_file(labels)

    repo_root, git_revision = extractor._discover_repo_context(EXTRACTOR_PATH, ROOT)
    if git_revision is None:
        raise AssertionError("focused preflight tests require the live Git checkout")
    source_files = {
        EVALUATOR_PATH.relative_to(repo_root).as_posix(): extractor.sha256_file(
            EVALUATOR_PATH
        ),
        EXTRACTOR_PATH.relative_to(repo_root).as_posix(): extractor.sha256_file(
            EXTRACTOR_PATH
        ),
    }
    runtime = extractor.current_runtime_identity()
    code_identity = {
        "git_revision": git_revision,
        "files": source_files,
        "aggregate_sha256": extractor.code_identity_aggregate_sha256(
            git_revision, source_files, runtime
        ),
        "runtime": runtime,
    }

    other = {
        "data": "7" * 64,
        "structural": "9" * 64,
        "labels": "a" * 64,
    }
    board_hashes = {
        "calibration": {
            "data": hashes["data"],
            "structural": hashes["structural"],
            "labels": hashes["labels"],
        },
        "confirmation": other,
    }
    admission_boards = {
        board_name: {
            "sha256": bindings["data"],
            "all_checks_pass": True,
        }
        for board_name, bindings in board_hashes.items()
    }
    compatibility_boards = {
        board_name: {
            "structural": {"sha256": bindings["structural"]},
            "referential_labels": {"sha256": bindings["labels"]},
            "all_checks_pass": True,
        }
        for board_name, bindings in board_hashes.items()
    }
    build_path = str((root / "build.json").resolve())
    build_sha256 = "b" * 64
    gate_admission = write_json(
        root / "gate_admission.json",
        {
            "audit": extractor.GATE_ADMISSION_AUDIT,
            "schema": extractor.BOARD_SCHEMA,
            "cpu_only": True,
            "score_outputs_read": False,
            "score_artifacts": [],
            "all_checks_pass": True,
            "r10_score_run_precondition_satisfied": True,
            "code_identity_aggregate_sha256": code_identity["aggregate_sha256"],
            "build_manifest": {
                "path": build_path,
                "sha256": build_sha256,
                "all_checks_pass": True,
            },
            "boards": admission_boards,
            "extractor_compatibility_admissions": {
                "enabled": True,
                "all_checks_pass": True,
                "boards": compatibility_boards,
            },
        },
    )
    gate_admission_sha256 = extractor.sha256_file(gate_admission)
    gate_boards = {
        board_name: {
            "sha256": bindings["data"],
            "structural_admission": {"sha256": bindings["structural"]},
            "referential_label_admission": {"sha256": bindings["labels"]},
        }
        for board_name, bindings in board_hashes.items()
    }
    gate_manifest = write_json(
        root / "gate.json",
        {
            "manifest": extractor.FROZEN_GATE_MANIFEST,
            "schema": extractor.BOARD_SCHEMA,
            "frozen_before_scores": True,
            "required_before_any_r10_score_run": True,
            "score_outputs_read": False,
            "score_artifacts": [],
            "board_gate_satisfied": True,
            "code_identity": code_identity,
            "admission_report": {
                "audit": extractor.GATE_ADMISSION_AUDIT,
                "path": str(gate_admission),
                "sha256": gate_admission_sha256,
            },
            "build_manifest": {
                "build": extractor.GATE_MANIFEST_BUILD,
                "path": build_path,
                "sha256": build_sha256,
            },
            "boards": gate_boards,
            "implementations": {
                "evaluator": {
                    "identifier": extractor.EVALUATOR_AUDIT,
                    "path": str(EVALUATOR_PATH),
                    "sha256": source_files[
                        EVALUATOR_PATH.relative_to(repo_root).as_posix()
                    ],
                },
                "extractor": {
                    "identifier": extractor.SCORE_AUDIT,
                    "path": str(EXTRACTOR_PATH),
                    "sha256": source_files[
                        EXTRACTOR_PATH.relative_to(repo_root).as_posix()
                    ],
                    "expected_seed": evaluator.EXPECTED_EXTRACTOR_SEED,
                },
                "expected_adapter_sha256": hashes["adapter"],
            },
        },
    )

    args = SimpleNamespace(
        board_name="calibration",
        base=str(base),
        pointer_adapter=str(pointer),
        adapter=str(adapter),
        adapter_sha256=hashes["adapter"],
        data=str(data),
        data_sha256=hashes["data"],
        tokenizer=str(tokenizer),
        admission=str(structural),
        admission_sha256=hashes["structural"],
        label_admission=str(labels),
        label_admission_sha256=hashes["labels"],
        gate_manifest=str(gate_manifest),
        gate_manifest_sha256=extractor.sha256_file(gate_manifest),
        gate_admission=str(gate_admission),
        gate_admission_sha256=gate_admission_sha256,
        evaluator=str(EVALUATOR_PATH),
        evaluator_sha256=extractor.sha256_file(EVALUATOR_PATH),
        out=str(output),
        batch_size=2,
        seed=evaluator.EXPECTED_EXTRACTOR_SEED,
        code_revision=git_revision,
        extractor_sha256=extractor.sha256_file(EXTRACTOR_PATH),
    )
    return SimpleNamespace(
        args=args,
        code_identity=code_identity,
        gate_manifest=gate_manifest,
    )


class GatePreflightTests(unittest.TestCase):
    def assert_sensitive_operations_not_called(self, args):
        with (
            mock.patch.object(extractor.torch.cuda, "is_available") as cuda_check,
            mock.patch.object(extractor.torch, "load") as checkpoint_load,
            mock.patch.object(extractor, "categorical_probabilities") as probabilities,
        ):
            with self.assertRaises(SystemExit):
                extractor.run(args)
        cuda_check.assert_not_called()
        checkpoint_load.assert_not_called()
        probabilities.assert_not_called()

    def test_missing_gate_fails_before_cuda_model_or_probability_access(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = build_preflight_fixture(temporary)
            Path(fixture.args.gate_manifest).unlink()
            self.assert_sensitive_operations_not_called(fixture.args)

    def test_hash_valid_gate_with_wrong_board_fails_before_sensitive_access(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = build_preflight_fixture(temporary)
            gate = json.loads(fixture.gate_manifest.read_text(encoding="utf-8"))
            gate["boards"]["calibration"]["sha256"] = "f" * 64
            write_json(fixture.gate_manifest, gate)
            fixture.args.gate_manifest_sha256 = extractor.sha256_file(
                fixture.gate_manifest
            )
            self.assert_sensitive_operations_not_called(fixture.args)

    def test_valid_preflight_builds_evaluator_compatible_metadata_without_cuda(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = build_preflight_fixture(temporary)
            with (
                mock.patch.object(extractor.torch.cuda, "is_available") as cuda_check,
                mock.patch.object(extractor.torch, "load") as checkpoint_load,
                mock.patch.object(
                    extractor, "categorical_probabilities"
                ) as probabilities,
            ):
                preflight = extractor.validate_preflight(fixture.args)
                metadata = extractor.score_binding_metadata(preflight)
            cuda_check.assert_not_called()
            checkpoint_load.assert_not_called()
            probabilities.assert_not_called()
            self.assertEqual(metadata["audit"], evaluator.SCORE_AUDIT)
            self.assertEqual(metadata["schema_version"], evaluator.SCORE_SCHEMA_VERSION)
            self.assertEqual(metadata["board_name"], "calibration")
            self.assertEqual(metadata["code_identity"], fixture.code_identity)
            self.assertEqual(
                fixture.code_identity["aggregate_sha256"],
                evaluator.code_identity_aggregate(
                    fixture.code_identity["git_revision"],
                    fixture.code_identity["files"],
                    fixture.code_identity["runtime"],
                ),
            )
            self.assertEqual(
                metadata["code_identity_aggregate_sha256"],
                fixture.code_identity["aggregate_sha256"],
            )
            expected_bindings = {
                "evaluator_sha256": fixture.args.evaluator_sha256,
                "extractor_sha256": fixture.args.extractor_sha256,
                "gate_manifest_sha256": fixture.args.gate_manifest_sha256,
                "gate_admission_sha256": fixture.args.gate_admission_sha256,
            }
            self.assertTrue(
                all(metadata[key] == value for key, value in expected_bindings.items())
            )
            expected_paths = {
                "evaluator": fixture.args.evaluator,
                "extractor": str(EXTRACTOR_PATH),
                "gate_manifest": fixture.args.gate_manifest,
                "gate_admission": fixture.args.gate_admission,
            }
            self.assertTrue(
                all(metadata[key] == value for key, value in expected_paths.items())
            )


class ExistingExtractionContractTests(unittest.TestCase):
    def test_hash_bindings(self):
        hashes = {
            "base": "1" * 64,
            "pointer_adapter": "2" * 64,
            "adapter": "3" * 64,
            "data": "4" * 64,
            "tokenizer": "5" * 64,
            "structural_admission": "6" * 64,
            "referential_label_admission": "7" * 64,
        }
        training_sha256 = "8" * 64
        metadata = {
            "protocol": extractor.R9C_PROTOCOL,
            "arm": "no_syndrome",
            "arm_config": extractor.NO_SYNDROME_CONFIG,
            "pointer_protocol": extractor.POINTER_PROTOCOL,
            "pointer_parameters_trainable": 0,
            "rounds": 3,
            "base_sha256": hashes["base"],
            "pointer_adapter_sha256": hashes["pointer_adapter"],
            "data_sha256": training_sha256,
            "tokenizer_sha256": hashes["tokenizer"],
            "admission_sha256": hashes["structural_admission"],
            "label_admission_sha256": hashes["referential_label_admission"],
            "final_adapter_sha256": "9" * 64,
        }
        admission = {
            "audit": extractor.STRUCTURAL_ADMISSION_AUDIT,
            "all_checks_pass": True,
            "eval_sha256": hashes["data"],
            "train_sha256": training_sha256,
            "tokenizer_sha256": hashes["tokenizer"],
        }
        label_admission = {
            "audit": extractor.LABEL_ADMISSION_AUDIT,
            "all_checks_pass": True,
            "tokenizer_sha256": hashes["tokenizer"],
            "datasets": {
                "eval": {
                    "all_checks_pass": True,
                    "sha256": hashes["data"],
                },
                "train": {
                    "all_checks_pass": True,
                    "sha256": training_sha256,
                },
            },
        }
        extractor.validate_hash_bindings(metadata, hashes, admission, label_admission)

        corrupted = copy.deepcopy(metadata)
        corrupted["base_sha256"] = "a" * 64
        with self.assertRaises(SystemExit):
            extractor.validate_hash_bindings(
                corrupted, hashes, admission, label_admission
            )

    def test_categorical_probabilities(self):
        generator = torch.Generator().manual_seed(20260714)
        forward = torch.randn(
            2, 3, len(OPCODES), generator=generator, dtype=torch.float64
        )
        backward = torch.randn(
            2, 3, len(OPCODES), generator=generator, dtype=torch.float64
        )
        query = torch.randn(2, len(QUERIES), generator=generator, dtype=torch.float64)
        probabilities = extractor.categorical_probabilities(forward, backward, query)

        self.assertEqual(set(probabilities), {"joint", "forward", "backward", "query"})
        self.assertEqual(probabilities["joint"].shape, (2, 3, len(OPCODES)))
        self.assertEqual(probabilities["query"].shape, (2, len(QUERIES)))
        self.assertTrue(
            all(value.dtype == torch.float32 for value in probabilities.values())
        )
        torch.testing.assert_close(
            probabilities["joint"],
            (0.5 * (forward.float() + backward.float())).softmax(dim=-1),
        )
        torch.testing.assert_close(
            probabilities["forward"], forward.float().softmax(dim=-1)
        )
        torch.testing.assert_close(
            probabilities["backward"], backward.float().softmax(dim=-1)
        )
        torch.testing.assert_close(
            probabilities["query"], query.float().softmax(dim=-1)
        )


if __name__ == "__main__":
    unittest.main()
