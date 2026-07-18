#!/usr/bin/env python3
"""Small-fixture rejection tests for the R10 workspace board pipeline."""

from __future__ import annotations

import copy
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

import audit_r10_workspace_boards as auditor
import generate_r10_workspace_boards as generator


def write_jsonl(path: Path, rows) -> None:
    path.write_text("".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows
    ))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


class WorkspaceBoardPipelineTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.training = self.root / "training.jsonl"
        self.r5 = self.root / "r5.jsonl"
        self.tokenizer = self.root / "tokenizer.json"
        self.calibration = self.root / "calibration.jsonl"
        self.confirmation = self.root / "confirmation.jsonl"
        self.manifest = self.root / "build.json"
        write_jsonl(self.training, [{
            "question": (
                "Unrelated archived fixture words remain deliberately remote from every synthetic "
                "ledger sentence used by either newly generated evaluation board in this test."
            ),
        }])
        write_jsonl(self.r5, [{
            "question": (
                "Separate historical fixture language contains no matching domains operations "
                "queries quantities templates or long lexical windows from the new boards."
            ),
        }])
        self.fixture_r5_sha256 = auditor.sha256_file(self.r5)
        tokenizer = Tokenizer(WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        tokenizer.save(str(self.tokenizer))
        with mock.patch.object(
            generator,
            "CANONICAL_R5_NOVELTY_BOARD_SHA256",
            self.fixture_r5_sha256,
        ):
            generator.build_bundle(
                training_data=[self.training],
                r5_board=self.r5,
                tokenizer_path=self.tokenizer,
                calibration_out=self.calibration,
                confirmation_out=self.confirmation,
                manifest_out=self.manifest,
                calibration_count=80,
                confirmation_count=80,
                calibration_seed=generator.CANONICAL_GENERATOR_SEEDS["calibration"],
                confirmation_seed=generator.CANONICAL_GENERATOR_SEEDS["confirmation"],
                max_tokens=2048,
                minimum_calibration=80,
                minimum_confirmation=80,
                require_confirmation_capacity=False,
            )

    def tearDown(self):
        self.temporary.cleanup()

    def audit(self, *, minimum_calibration=80, minimum_confirmation=80):
        with mock.patch.object(
            auditor,
            "CANONICAL_R5_NOVELTY_BOARD_SHA256",
            self.fixture_r5_sha256,
        ):
            return auditor.audit_bundle(
                training_data=[self.training],
                r5_board=self.r5,
                tokenizer_path=self.tokenizer,
                calibration_path=self.calibration,
                confirmation_path=self.confirmation,
                build_manifest_path=self.manifest,
                max_tokens=2048,
                minimum_calibration=minimum_calibration,
                minimum_confirmation=minimum_confirmation,
                require_confirmation_capacity=False,
            )

    def refresh_manifest_hashes(self):
        manifest = json.loads(self.manifest.read_text())
        outputs = {
            str(self.calibration.resolve()): self.calibration,
            str(self.confirmation.resolve()): self.confirmation,
        }
        for record in manifest["outputs"].values():
            path = outputs[record["path"]]
            record["sha256"] = auditor.sha256_file(path)
            record["rows"] = len(read_jsonl(path))
        inputs = {
            str(self.training.resolve()): self.training,
            str(self.r5.resolve()): self.r5,
        }
        for record in manifest["inputs"]:
            path = inputs[record["path"]]
            record["sha256"] = auditor.sha256_file(path)
            record["rows_scanned"] = len(read_jsonl(path))
        self.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    def test_clean_small_fixture_passes(self):
        report = self.audit()
        self.assertTrue(report["all_checks_pass"])
        self.assertTrue(report["boards"]["calibration"]["checks"]["queries_exactly_balanced"])
        confirmation = report["boards"]["confirmation"]
        self.assertTrue(confirmation["checks"]["regime_depth_query_family_cells_exact"])
        self.assertEqual(confirmation["expected_cell_count"], 80)
        self.assertEqual(confirmation["rows_per_exact_cell"], 1)
        quota = report["confirmation_empirical_quota"]
        self.assertTrue(quota["passes"])
        self.assertEqual(set(quota["partitions"]), {"language_ood", "full_ood"})
        self.assertTrue(all(
            partition["exact_cells"] == 40
            and partition["rows_per_exact_cell"] == 1
            and partition["minimum_accepted_per_exact_cell"] == 1
            and partition["minimum_accepted"] == 40
            and partition["maximum_false_certificates"] == 0
            for partition in quota["partitions"].values()
        ))

    def test_generation_is_byte_deterministic_with_distinct_seeds(self):
        calibration_seed = generator.CANONICAL_GENERATOR_SEEDS["calibration"]
        confirmation_seed = generator.CANONICAL_GENERATOR_SEEDS["confirmation"]
        first = generator.serialize_jsonl(
            generator.build_board("calibration", 80, calibration_seed)
        )
        second = generator.serialize_jsonl(
            generator.build_board("calibration", 80, calibration_seed)
        )
        confirmation = generator.serialize_jsonl(
            generator.build_board("confirmation", 80, confirmation_seed)
        )
        self.assertEqual(first, second)
        self.assertNotEqual(generator.sha256_bytes(first), generator.sha256_bytes(confirmation))

    def test_wrong_seed_is_rejected_without_generation(self):
        for name, seed in generator.CANONICAL_GENERATOR_SEEDS.items():
            with self.subTest(board=name):
                with self.assertRaisesRegex(ValueError, "frozen canonical seed"):
                    generator.build_board(name, 80, seed + 1)

    def test_wrong_seed_board_is_rejected_by_independent_auditor(self):
        rows = read_jsonl(self.calibration)
        wrong_seed = generator.CANONICAL_GENERATOR_SEEDS["calibration"] + 1
        for row in rows:
            row["generation_seed"] = wrong_seed
        write_jsonl(self.calibration, rows)
        self.refresh_manifest_hashes()
        manifest = json.loads(self.manifest.read_text())
        manifest["outputs"]["calibration"]["seed"] = wrong_seed
        self.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        report = self.audit()
        self.assertFalse(
            report["boards"]["calibration"]["checks"]["canonical_generation_seed"]
        )
        self.assertFalse(report["canonical_generator_seeds"])
        self.assertFalse(report["all_checks_pass"])

    def test_wrong_r5_bytes_are_rejected_by_generator_and_auditor(self):
        write_jsonl(self.r5, [{"question": "tampered R5 novelty source"}])
        with mock.patch.object(
            generator,
            "CANONICAL_R5_NOVELTY_BOARD_SHA256",
            self.fixture_r5_sha256,
        ):
            with self.assertRaisesRegex(ValueError, "R5 novelty board SHA256"):
                generator.build_bundle(
                    training_data=[self.training],
                    r5_board=self.r5,
                    tokenizer_path=self.tokenizer,
                    calibration_out=self.root / "wrong-r5-calibration.jsonl",
                    confirmation_out=self.root / "wrong-r5-confirmation.jsonl",
                    manifest_out=self.root / "wrong-r5-build.json",
                    calibration_count=80,
                    confirmation_count=80,
                    minimum_calibration=80,
                    minimum_confirmation=80,
                    require_confirmation_capacity=False,
                )
        with self.assertRaisesRegex(ValueError, "R5 novelty board SHA256"):
            self.audit()

    def test_tampered_build_manifest_content_is_rejected(self):
        manifest = json.loads(self.manifest.read_text())
        manifest["forged_all_checks_pass"] = True
        self.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        report = self.audit()
        self.assertFalse(report["build_manifest"]["checks"]["exact_fields"])
        self.assertFalse(report["build_manifest"]["all_checks_pass"])
        self.assertFalse(report["all_checks_pass"])

    def test_source_overlap_is_rejected_after_rebinding_hash(self):
        calibration = read_jsonl(self.calibration)
        write_jsonl(self.training, [{"question": calibration[0]["question"]}])
        self.refresh_manifest_hashes()
        report = self.audit()
        scan = report["hard_scan"]["source_reports"][0]["boards"]["calibration"]
        self.assertFalse(report["all_checks_pass"])
        self.assertGreater(scan["exact_prompt_rows"], 0)
        self.assertGreater(scan["ngram13_rows"], 0)

    def test_valid_query_retarget_breaks_exact_balance(self):
        rows = read_jsonl(self.confirmation)
        index = next(
            index for index, row in enumerate(rows)
            if generator.query_opcode(row["query"], row["keys"]) == "read_0"
        )
        row = copy.deepcopy(rows[index])
        spec = generator.BOARD_SPECS["confirmation"]
        domain = next(domain for domain in spec.domains if domain.family == row["family"])
        values = dict(row["initial"])
        for operation in row["operations"]:
            values = generator.apply_operation(values, operation)
        query = generator.make_query("read_1", values, domain.keys)
        query_template = row["surface"]["query_template"]
        query["text"] = generator.render_query(query, domain, spec, query_template)
        row["query"] = query
        row["answer"] = str(query["answer"])
        row["response"] = "The answer is {}.".format(query["answer"])
        row["question"] = generator.render_question(
            spec,
            domain,
            row["initial"],
            row["operations"],
            query,
            row["surface"]["intro_template"],
            row["surface"]["operation_templates"],
            query_template,
        )
        row["cell_id"] = generator.cell_id(
            row["eval_regime"], row["depth"], "read_1", row["family"],
        )
        row["prompt_sha256"] = generator.sha256_bytes(row["question"].encode())
        row["program_sha256"] = generator.canonical_hash(generator.program_signature(row))
        rows[index] = row
        write_jsonl(self.confirmation, rows)
        self.refresh_manifest_hashes()
        report = self.audit()
        checks = report["boards"]["confirmation"]["checks"]
        self.assertTrue(checks["all_rows_structurally_valid"])
        self.assertFalse(checks["queries_exactly_balanced"])
        self.assertFalse(checks["regime_depth_query_family_cells_exact"])
        self.assertFalse(report["all_checks_pass"])

    def test_missing_exact_cell_is_rejected(self):
        rows = read_jsonl(self.confirmation)
        removed_cell = rows[-1]["cell_id"]
        write_jsonl(self.confirmation, rows[:-1])
        self.refresh_manifest_hashes()
        report = self.audit()
        board = report["boards"]["confirmation"]
        self.assertFalse(board["checks"]["regime_depth_query_family_cells_exact"])
        self.assertIn(removed_cell, board["cell_failures"]["missing"])
        self.assertFalse(report["all_checks_pass"])

    def test_present_cells_below_frozen_size_are_rejected(self):
        report = self.audit(minimum_confirmation=160)
        board = report["boards"]["confirmation"]
        self.assertEqual(board["cell_failures"]["missing"], [])
        self.assertEqual(len(board["cell_failures"]["undersized"]), 80)
        self.assertTrue(all(
            item["actual"] == 1 and item["expected"] == 2
            for item in board["cell_failures"]["undersized"]
        ))
        self.assertFalse(board["checks"]["regime_depth_query_family_cells_exact"])
        self.assertFalse(report["all_checks_pass"])

    def test_structured_surface_disagreement_is_rejected(self):
        rows = read_jsonl(self.calibration)
        row_index, operation_index = next(
            (row_index, operation_index)
            for row_index, row in enumerate(rows)
            for operation_index, operation in enumerate(row["operations"])
            if operation["kind"] in {"add", "sub"}
        )
        row = copy.deepcopy(rows[row_index])
        operation = row["operations"][operation_index]
        operation["target"] = next(key for key in row["keys"] if key != operation["target"])
        row["program_sha256"] = generator.canonical_hash(generator.program_signature(row))
        rows[row_index] = row
        write_jsonl(self.calibration, rows)
        self.refresh_manifest_hashes()
        report = self.audit()
        errors = report["boards"]["calibration"]["errors"]
        self.assertFalse(report["all_checks_pass"])
        self.assertTrue(any(error["category"] == "structured_semantics" for error in errors))

    def test_existing_outputs_are_never_overwritten(self):
        with self.assertRaises(FileExistsError):
            generator.build_bundle(
                training_data=[self.training],
                r5_board=self.r5,
                tokenizer_path=self.tokenizer,
                calibration_out=self.calibration,
                confirmation_out=self.confirmation,
                manifest_out=self.manifest,
                calibration_count=80,
                confirmation_count=80,
                minimum_calibration=80,
                minimum_confirmation=80,
                require_confirmation_capacity=False,
            )


def run_git(repo: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *arguments],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return result.stdout.strip()


def initialize_clean_code_repo(root: Path) -> tuple[Path, str]:
    repo = root / "code"
    repo.mkdir()
    required = {
        *auditor.CODE_IDENTITY_PIPELINE_FILES,
        *auditor.CODE_IDENTITY_R10_JOB_FILES,
        auditor.EVALUATOR_REPO_PATH,
        auditor.EXTRACTOR_REPO_PATH,
    }
    for relative in sorted(required):
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if relative == auditor.EVALUATOR_REPO_PATH:
            content = "\n".join((
                'AUDIT = "{}"'.format(auditor.EVALUATOR_IDENTIFIER),
                'GATE_ADMISSION_AUDIT = "{}"'.format(auditor.ADMISSION_AUDIT),
                'BOARD_SCHEMA = "{}"'.format(auditor.SCHEMA),
                'GATE_MANIFEST_BUILD = "r10_workspace_boards_v2"',
                "",
            ))
        elif relative == auditor.EXTRACTOR_REPO_PATH:
            content = 'SCORE_AUDIT = "{}"\n'.format(auditor.EXTRACTOR_IDENTIFIER)
        else:
            content = "# committed custody fixture\n"
        path.write_text(content)
    run_git(repo, "init", "--quiet")
    run_git(repo, "config", "user.name", "R10 Custody Test")
    run_git(repo, "config", "user.email", "r10-custody@example.invalid")
    run_git(repo, "add", ".")
    run_git(repo, "commit", "--quiet", "-m", "fixture")
    return repo, run_git(repo, "rev-parse", "HEAD")


def build_frozen_gate_fixture(root: Path, code_repo: Path, revision: str):
    training = root / "training.jsonl"
    write_jsonl(training, [{"question": "independent training fixture"}])
    r5 = auditor.ROOT / "artifacts/evals/referential_argument_graph_v5_fresh.jsonl"
    tokenizer = root / "tokenizer.json"
    tokenizer.write_text("{}\n")
    tokenizer_binding = {
        "path": str(tokenizer.resolve()),
        "sha256": auditor.sha256_file(tokenizer),
        "max_tokens": 2048,
    }

    boards = {}
    outputs = {}
    compatible_boards = {}
    for name, rows in auditor.DEFAULT_BOARD_ROWS.items():
        rows_per_cell = auditor.DEFAULT_CELL_ROWS[name]
        path = root / "{}.jsonl".format(name)
        path.write_text("{}\n" * rows)
        regimes = {
            regime: rows // len(auditor.SPECS[name].regimes)
            for regime in auditor.SPECS[name].regimes
        }
        exact_cells = {
            auditor.cell_id(*key): rows_per_cell
            for key in auditor.expected_cell_keys(auditor.SPECS[name])
        }
        output = {
            "path": str(path.resolve()),
            "sha256": auditor.sha256_file(path),
            "rows": rows,
            "regimes": regimes,
            "expected_cell_count": 80,
            "rows_per_exact_cell": rows_per_cell,
            "exact_cells": exact_cells,
            "seed": auditor.CANONICAL_GENERATOR_SEEDS[name],
            "r5_novelty_board_sha256": auditor.CANONICAL_R5_NOVELTY_BOARD_SHA256,
        }
        outputs[name] = output
        boards[name] = {
            **{key: value for key, value in output.items() if key not in {
                "seed", "r5_novelty_board_sha256",
            }},
            "generation_seeds": [auditor.CANONICAL_GENERATOR_SEEDS[name]],
            "generation_seed": auditor.CANONICAL_GENERATOR_SEEDS[name],
            "r5_novelty_board_sha256": auditor.CANONICAL_R5_NOVELTY_BOARD_SHA256,
            "checks": {key: True for key in auditor.BOARD_ADMISSION_CHECK_FIELDS},
            "all_checks_pass": True,
        }
        structural = root / "{}.structural.json".format(name)
        labels = root / "{}.labels.json".format(name)
        structural.write_text("{}\n")
        labels.write_text("{}\n")
        compatible_boards[name] = {
            "structural": {
                "path": str(structural.resolve()),
                "sha256": auditor.sha256_file(structural),
            },
            "referential_labels": {
                "path": str(labels.resolve()),
                "sha256": auditor.sha256_file(labels),
            },
            "checks": {
                key: True for key in auditor.COMPATIBILITY_ADMISSION_CHECK_FIELDS
            },
            "all_checks_pass": True,
        }

    build_content = {
        "build": "r10_workspace_boards_v2",
        "schema": auditor.SCHEMA,
        "cpu_only": True,
        "score_outputs_read": False,
        "score_artifacts": [],
        "ready_for_r10_score_run": False,
        "ngram_width": auditor.NGRAM_WIDTH,
        "executor_width": auditor.EXECUTOR_WIDTH,
        "generation_contract": auditor.canonical_generation_contract(),
        "schedule_contract": auditor.expected_schedule_contract(),
        "tokenizer": tokenizer_binding,
        "inputs": [
            {
                "role": "training_data",
                "path": str(training.resolve()),
                "sha256": auditor.sha256_file(training),
                "rows_scanned": 1,
            },
            {
                "role": "r5_fresh_board",
                "path": str(r5.resolve()),
                "sha256": auditor.CANONICAL_R5_NOVELTY_BOARD_SHA256,
                "rows_scanned": sum(1 for line in r5.read_bytes().splitlines() if line.strip()),
            },
        ],
        "outputs": outputs,
        "cross_board_scan": {
            "exact_prompt_hits": 0,
            "ngram13_hits": 0,
            "program_hits": 0,
        },
        "claim_boundary": auditor.BUILD_MANIFEST_CLAIM_BOUNDARY,
    }
    build_path = root / "build.json"
    build_bytes = (json.dumps(build_content, indent=2, sort_keys=True) + "\n").encode()
    build_path.write_bytes(build_bytes)
    build_checks = {
        key: True for key in auditor.expected_build_manifest_check_fields(build_content)
    }
    build_binding = {
        "build": "r10_workspace_boards_v2",
        "path": str(build_path.resolve()),
        "sha256": auditor.sha256_bytes(build_bytes),
        "byte_length": len(build_bytes),
        "bytes_base64": auditor.base64.b64encode(build_bytes).decode("ascii"),
        "content": build_content,
        "checks": build_checks,
        "all_checks_pass": True,
    }
    identity = auditor.capture_clean_committed_code_identity(revision, code_repo)
    report = {
        "audit": auditor.ADMISSION_AUDIT,
        "schema": auditor.SCHEMA,
        "cpu_only": True,
        "score_outputs_read": False,
        "score_artifacts": [],
        "tokenizer": tokenizer_binding,
        "build_manifest": build_binding,
        "extractor_compatibility_admissions": {
            "enabled": True,
            "boards": compatible_boards,
            "all_checks_pass": True,
        },
        "boards": boards,
        "hard_scan": {
            "all_source_scans_zero": True,
            "cross_board_zero": True,
        },
        "deterministic_distinct_seeds": True,
        "canonical_generator_seeds": True,
        "generation_contract": auditor.canonical_generation_contract(),
        "calibration_confirmation_regimes_disjoint": True,
        "confirmation_empirical_quota": {
            "passes": True,
            "scope": "frozen confirmation rows only",
        },
        "all_checks_pass": True,
        "r10_score_run_precondition_satisfied": True,
        "code_identity": identity,
        "code_identity_aggregate_sha256": identity["aggregate_sha256"],
    }
    admission = root / "admission.json"
    admission.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    manifest = auditor.build_frozen_gate_manifest(
        report=report,
        admission_report_path=admission,
        evaluator_path=code_repo / auditor.EVALUATOR_REPO_PATH,
        extractor_path=code_repo / auditor.EXTRACTOR_REPO_PATH,
        code_revision=revision,
        repo_root=code_repo,
    )
    return report, admission, build_path, manifest


class CheckoutCustodyTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.repo, self.revision = initialize_clean_code_repo(self.root)

    def tearDown(self):
        self.temporary.cleanup()

    def test_dirty_tracked_source_is_rejected(self):
        source = self.repo / auditor.EVALUATOR_REPO_PATH
        source.write_text(source.read_text() + "# mutation\n")
        with self.assertRaisesRegex(ValueError, "not clean and committed"):
            auditor.capture_clean_committed_code_identity(self.revision, self.repo)

    def test_untracked_source_is_rejected(self):
        untracked = self.repo / "train/injected_source.py"
        untracked.write_text("# untracked\n")
        with self.assertRaisesRegex(ValueError, "not clean and committed"):
            auditor.capture_clean_committed_code_identity(self.revision, self.repo)

    def test_source_mutation_during_admission_is_rejected(self):
        source = self.repo / auditor.EVALUATOR_REPO_PATH

        def mutate_source():
            source.write_text(source.read_text() + "# mutation during audit\n")
            return {"all_checks_pass": True}

        with self.assertRaisesRegex(RuntimeError, "changed during score-blind admission"):
            auditor.run_admission_with_code_custody(
                code_revision=self.revision,
                admission_work=mutate_source,
                repo_root=self.repo,
            )


class FrozenGateContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.code_repo, cls.revision = initialize_clean_code_repo(cls.root)
        cls.report, cls.admission, cls.build_path, cls.manifest = build_frozen_gate_fixture(
            cls.root, cls.code_repo, cls.revision,
        )

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def validate(self, manifest):
        auditor.validate_frozen_gate_manifest(
            manifest,
            repo_root=self.code_repo,
            expected_code_revision=self.revision,
        )

    def test_frozen_gate_uses_exact_finite_board_quotas(self):
        self.validate(self.manifest)
        partitions = self.manifest["partitions"]["confirmation"]
        self.assertEqual(set(partitions), {"language_ood", "full_ood"})
        self.assertTrue(all(
            partition["rows"] == 920
            and partition["exact_cells"] == 40
            and partition["rows_per_exact_cell"] == 23
            and partition["minimum_accepted_per_exact_cell"] == 10
            and partition["minimum_accepted"] == 400
            and partition["maximum_false_certificates"] == 0
            for partition in partitions.values()
        ))
        thresholds = self.manifest["confirmation_thresholds"]
        self.assertEqual(thresholds["minimum_accepted_each_exact_cell"], 10)
        self.assertEqual(thresholds["minimum_accepted_each_partition"], 400)
        self.assertEqual(thresholds["maximum_false_certificates_each_partition"], 0)
        self.assertTrue(thresholds["extrapolation_beyond_frozen_board_forbidden"])

    def test_gate_embeds_independently_verifiable_exact_build_manifest(self):
        content = auditor.validate_gate_build_manifest_contract(self.manifest)
        self.assertEqual(content, json.loads(self.build_path.read_bytes()))
        self.assertEqual(
            self.manifest["build_manifest"]["sha256"],
            auditor.sha256_file(self.build_path),
        )

    def test_tampered_embedded_build_manifest_is_rejected(self):
        manifest = copy.deepcopy(self.manifest)
        encoded = manifest["build_manifest"]["bytes_base64"]
        manifest["build_manifest"]["bytes_base64"] = ("A" if encoded[0] != "A" else "B") + encoded[1:]
        with self.assertRaisesRegex(ValueError, "exact byte hash differs"):
            self.validate(manifest)

    def test_forged_top_level_all_checks_pass_is_rejected(self):
        report = copy.deepcopy(self.report)
        report["boards"]["calibration"]["checks"]["zero_oracle_errors"] = False
        report["boards"]["calibration"]["all_checks_pass"] = True
        report["all_checks_pass"] = True
        admission = self.root / "forged-admission.json"
        admission.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        with self.assertRaisesRegex(ValueError, "board admission checks did not all pass"):
            auditor.build_frozen_gate_manifest(
                report=report,
                admission_report_path=admission,
                evaluator_path=self.code_repo / auditor.EVALUATOR_REPO_PATH,
                extractor_path=self.code_repo / auditor.EXTRACTOR_REPO_PATH,
                code_revision=self.revision,
                repo_root=self.code_repo,
            )

    def test_tampered_external_build_manifest_is_rejected_at_freeze(self):
        report = copy.deepcopy(self.report)
        tampered = self.root / "tampered-build.json"
        tampered.write_bytes(self.build_path.read_bytes() + b" ")
        report["build_manifest"]["path"] = str(tampered.resolve())
        admission = self.root / "tampered-build-admission.json"
        admission.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        with self.assertRaisesRegex(ValueError, "external build manifest bytes differ"):
            auditor.build_frozen_gate_manifest(
                report=report,
                admission_report_path=admission,
                evaluator_path=self.code_repo / auditor.EVALUATOR_REPO_PATH,
                extractor_path=self.code_repo / auditor.EXTRACTOR_REPO_PATH,
                code_revision=self.revision,
                repo_root=self.code_repo,
            )

    def test_code_identity_covers_the_exact_source_closure(self):
        identity = self.manifest["code_identity"]
        auditor.validate_code_identity(
            identity,
            repo_root=self.code_repo,
            expected_revision=self.revision,
        )
        self.assertEqual(
            list(identity["files"]),
            list(auditor.code_identity_relative_paths(self.code_repo)),
        )
        self.assertEqual(
            identity["aggregate_sha256"],
            auditor.code_identity_aggregate(
                identity["git_revision"], identity["files"], identity["runtime"],
            ),
        )
        self.assertEqual(set(identity["runtime"]), {"python", "torch", "tokenizers"})

    def test_missing_code_identity_is_rejected(self):
        manifest = copy.deepcopy(self.manifest)
        del manifest["code_identity"]
        with self.assertRaises(ValueError):
            self.validate(manifest)

    def test_tampered_code_identity_is_rejected(self):
        manifest = copy.deepcopy(self.manifest)
        manifest["code_identity"]["files"][auditor.EVALUATOR_REPO_PATH] = "f" * 64
        with self.assertRaises(ValueError):
            self.validate(manifest)

    def test_legacy_confidence_fields_are_rejected(self):
        cases = {
            "top_level": ("confidence_scope", {}),
            "clopper_pearson": ("clopper_pearson_lower_bound", 0.99),
            "simultaneous": ("simultaneous_confidence", 0.95),
            "old_probability": ("target_success_probability", 0.99),
            "population": ("population_extrapolation", False),
        }
        for name, (field, value) in cases.items():
            with self.subTest(name=name):
                manifest = copy.deepcopy(self.manifest)
                manifest["confirmation_thresholds"][field] = value
                with self.assertRaises(ValueError):
                    self.validate(manifest)

    def test_invalid_or_abbreviated_code_revision_is_rejected(self):
        for revision in ("", "a" * 39, "A" * 40, "g" * 40):
            with self.subTest(revision=revision):
                with self.assertRaises(ValueError):
                    auditor.validate_git_revision(revision)


if __name__ == "__main__":
    unittest.main()
