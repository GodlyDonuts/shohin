#!/usr/bin/env python3
"""CPU-only contract tests for the frozen R10 v2 evaluator."""

from __future__ import annotations

import hashlib
import json
import math
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

import evaluate_version_space_workspace as evaluator
from categorical_microcode import OPCODES, QUERIES
from evaluate_version_space_workspace import (
    BOARD_SCHEMA,
    CALIBRATION_REGIMES,
    CONFIRMATION_REGIMES,
    EXPECTED_DEPTHS,
    EXPECTED_FAMILIES,
    FROZEN_GATE_MANIFEST,
    GATE_ADMISSION_AUDIT,
    GATE_MANIFEST_BUILD,
    MAX_AFFINE_AMBIGUITY_RANK,
    SCORE_AUDIT,
    SCORE_SCHEMA_VERSION,
    BoundScoreReport,
    CalibrationResult,
    CaseEvaluation,
    EvaluationContractError,
    ScoreRecord,
    _expected_summary,
    _matched_coverage_baselines,
    _partition_empirical_gates,
    analyze_candidate_program,
    assess_static_confirmation,
    atomic_write_json_no_overwrite,
    calibrate_program_threshold,
    canonical_sha256,
    code_identity_aggregate,
    current_runtime_identity,
    main,
    sha256_file,
    validate_gate_bundle,
    validate_record_geometry,
    validate_score_report,
)

CODE_REVISION = evaluator._live_git_revision(evaluator.REPO_ROOT)
EVALUATOR_PATH = Path(evaluator.__file__).resolve()
EXTRACTOR_PATH = EVALUATOR_PATH.with_name("extract_referential_version_scores.py")


def frozen_code_identity():
    files = {
        EVALUATOR_PATH.relative_to(evaluator.REPO_ROOT).as_posix(): sha256_file(
            EVALUATOR_PATH
        ),
        EXTRACTOR_PATH.relative_to(evaluator.REPO_ROOT).as_posix(): sha256_file(
            EXTRACTOR_PATH
        ),
    }
    runtime = current_runtime_identity()
    return {
        "git_revision": CODE_REVISION,
        "files": files,
        "aggregate_sha256": code_identity_aggregate(CODE_REVISION, files, runtime),
        "runtime": runtime,
    }


def probability_row(width, assignments):
    values = [0.0] * width
    for index, value in assignments.items():
        values[index] = value
    unset = [index for index in range(width) if index not in assignments]
    remaining = 1.0 - sum(values)
    for index in unset:
        values[index] = remaining / len(unset)
    return tuple(values)


def geometry_record(index, reference, regime, depth, query_target, family):
    return ScoreRecord(
        index=index,
        reference=reference,
        regime=regime,
        family=family,
        operation_targets=(0,) * depth,
        operation_values=(1,) * depth,
        initial_state=(2, 5),
        query_target=query_target,
        answer=0,
        joint_probabilities=(),
        forward_probabilities=(),
        backward_probabilities=(),
        query_probabilities=(),
        source_record_sha256="0" * 64,
        event_source_payloads=(),
    )


def valid_record(
    index,
    reference,
    regime,
    *,
    depth=1,
    query_target=0,
    family="binding hall",
    true_probability=0.40,
):
    operation_target = 0
    operation_row = probability_row(
        len(OPCODES), {operation_target: true_probability, 1: 0.38}
    )
    query_row = probability_row(
        len(QUERIES), {query_target: true_probability, (query_target + 1) % 5: 0.38}
    )
    operation_targets = (operation_target,) * depth
    operation_values = (1,) * depth
    answer = evaluator._program_transform(operation_targets, operation_values).answer(
        (2, 5), query_target
    )
    payloads = tuple(
        {
            "event_index": offset,
            "operation": {"kind": "add", "target": "A", "value": 1},
            "reference": reference,
            "text": "Event {}: synthetic add.".format(offset + 1),
        }
        for offset in range(depth)
    )
    return ScoreRecord(
        index=index,
        reference=reference,
        regime=regime,
        family=family,
        operation_targets=operation_targets,
        operation_values=operation_values,
        initial_state=(2, 5),
        query_target=query_target,
        answer=answer,
        joint_probabilities=(operation_row,) * depth,
        forward_probabilities=(operation_row,) * depth,
        backward_probabilities=(operation_row,) * depth,
        query_probabilities=query_row,
        source_record_sha256=canonical_sha256({"reference": reference}),
        event_source_payloads=payloads,
    )


def frozen_geometry(board_name):
    regimes = (
        CALIBRATION_REGIMES if board_name == "calibration" else CONFIRMATION_REGIMES
    )
    records = []
    for regime in regimes:
        for depth in EXPECTED_DEPTHS[regime]:
            for query_target in range(len(QUERIES)):
                for family in EXPECTED_FAMILIES[board_name]:
                    for cell_index in range(evaluator.EXPECTED_CELL_ROWS[board_name]):
                        index = len(records)
                        records.append(
                            geometry_record(
                                index,
                                "{}-{:06d}".format(board_name, index),
                                regime,
                                depth,
                                query_target,
                                family,
                            )
                        )
    return tuple(records)


def write_json(path, payload):
    Path(path).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def frozen_partitions():
    return {
        "calibration": {
            "fit_iid": {
                "rows": 400,
                "depths": [4, 8],
                "numeric_profile": "in_range",
                "exact_cells": 40,
                "rows_per_cell": 10,
                "rows_per_exact_cell": 10,
            },
            "depth_ood": {
                "rows": 400,
                "depths": [16, 32],
                "numeric_profile": "shifted",
                "exact_cells": 40,
                "rows_per_cell": 10,
                "rows_per_exact_cell": 10,
            },
        },
        "confirmation": {
            "language_ood": {
                "rows": 920,
                "depths": [4, 8],
                "numeric_profile": "in_range",
                "exact_cells": 40,
                "rows_per_cell": 23,
                "rows_per_exact_cell": 23,
                "minimum_accepted_per_exact_cell": 10,
                "minimum_accepted": 400,
                "maximum_false_certificates": 0,
            },
            "full_ood": {
                "rows": 920,
                "depths": [16, 32],
                "numeric_profile": "shifted",
                "exact_cells": 40,
                "rows_per_cell": 23,
                "rows_per_exact_cell": 23,
                "minimum_accepted_per_exact_cell": 10,
                "minimum_accepted": 400,
                "maximum_false_certificates": 0,
            },
        },
    }


def build_gate_fixture(root):
    root = Path(root)
    evaluator_hash = sha256_file(EVALUATOR_PATH)
    extractor_hash = sha256_file(EXTRACTOR_PATH)
    adapter_hash = "3" * 64
    code_identity = frozen_code_identity()
    bindings = {
        "calibration": {
            "data_sha256": "4" * 64,
            "structural_admission_sha256": "5" * 64,
            "label_admission_sha256": "6" * 64,
        },
        "confirmation": {
            "data_sha256": "7" * 64,
            "structural_admission_sha256": "8" * 64,
            "label_admission_sha256": "9" * 64,
        },
    }
    build_path = str((root / "build.json").resolve())
    build_hash = "b" * 64
    admission_path = (root / "admission.json").resolve()
    admission_boards = {}
    compatibility_boards = {}
    for board_name in ("calibration", "confirmation"):
        admission_boards[board_name] = {
            **_expected_summary(board_name),
            "sha256": bindings[board_name]["data_sha256"],
            "all_checks_pass": True,
        }
        compatibility_boards[board_name] = {
            "structural": {
                "sha256": bindings[board_name]["structural_admission_sha256"]
            },
            "referential_labels": {
                "sha256": bindings[board_name]["label_admission_sha256"]
            },
            "all_checks_pass": True,
        }
    admission = {
        "audit": GATE_ADMISSION_AUDIT,
        "schema": BOARD_SCHEMA,
        "cpu_only": True,
        "score_outputs_read": False,
        "score_artifacts": [],
        "all_checks_pass": True,
        "r10_score_run_precondition_satisfied": True,
        "code_identity_aggregate_sha256": code_identity["aggregate_sha256"],
        "build_manifest": {
            "path": build_path,
            "sha256": build_hash,
            "all_checks_pass": True,
        },
        "boards": admission_boards,
        "extractor_compatibility_admissions": {
            "enabled": True,
            "all_checks_pass": True,
            "boards": compatibility_boards,
        },
    }
    write_json(admission_path, admission)
    admission_hash = sha256_file(admission_path)

    gate_boards = {}
    for board_name in ("calibration", "confirmation"):
        summary = _expected_summary(board_name)
        gate_boards[board_name] = {
            "sha256": bindings[board_name]["data_sha256"],
            "rows": summary["rows"],
            "regimes": summary["regimes"],
            "expected_cell_count": summary["expected_cell_count"],
            "rows_per_exact_cell": summary["rows_per_exact_cell"],
            "structural_admission": {
                "sha256": bindings[board_name]["structural_admission_sha256"]
            },
            "referential_label_admission": {
                "sha256": bindings[board_name]["label_admission_sha256"]
            },
        }
    gate = {
        "manifest": FROZEN_GATE_MANIFEST,
        "schema": BOARD_SCHEMA,
        "frozen_before_scores": True,
        "required_before_any_r10_score_run": True,
        "score_outputs_read": False,
        "score_artifacts": [],
        "board_gate_satisfied": True,
        "code_identity": code_identity,
        "admission_report": {
            "audit": GATE_ADMISSION_AUDIT,
            "path": str(admission_path),
            "sha256": admission_hash,
        },
        "build_manifest": {
            "build": GATE_MANIFEST_BUILD,
            "path": build_path,
            "sha256": build_hash,
        },
        "boards": gate_boards,
        "partitions": frozen_partitions(),
        "calibration_threshold": {
            "scope": "frozen calibration rows only",
            "partitions_pooled_only_for_calibration": list(CALIBRATION_REGIMES),
            "quantile": 0.97,
            "threshold_count": 1,
            "confirmation_rows_must_not_influence_threshold": True,
            "post_score_tuning_forbidden": True,
            "extrapolation_beyond_frozen_board_forbidden": True,
        },
        "confirmation_thresholds": {
            "scope": "frozen confirmation rows only",
            "minimum_selective_coverage_each_partition": 0.40,
            "acceptance_quota_fraction_per_exact_cell": 0.40,
            "acceptance_quota_rounding": "ceil(0.40 * 23) = 10",
            "exact_cells_each_partition": 40,
            "rows_each_exact_cell": 23,
            "minimum_accepted_each_partition": 400,
            "minimum_accepted_each_exact_cell": 10,
            "maximum_false_certificates_each_exact_cell": 0,
            "maximum_false_certificates_each_partition": 0,
            "minimum_empirical_selective_accuracy_each_partition": 0.99,
            "all_exact_cells_required": True,
            "pooled_partition_substitution_forbidden": True,
            "extrapolation_beyond_frozen_board_forbidden": True,
        },
        "implementations": {
            "evaluator": {
                "identifier": evaluator.AUDIT,
                "path": str(EVALUATOR_PATH),
                "sha256": evaluator_hash,
            },
            "extractor": {
                "identifier": SCORE_AUDIT,
                "path": str(EXTRACTOR_PATH),
                "sha256": extractor_hash,
                "expected_seed": evaluator.EXPECTED_EXTRACTOR_SEED,
            },
            "expected_adapter_sha256": adapter_hash,
        },
    }
    gate_path = (root / "gate.json").resolve()
    write_json(gate_path, gate)
    return {
        "gate_path": gate_path,
        "gate_sha256": sha256_file(gate_path),
        "admission_path": admission_path,
        "admission_sha256": admission_hash,
        "bindings": bindings,
        "evaluator_sha256": evaluator_hash,
        "extractor_sha256": extractor_hash,
        "adapter_sha256": adapter_hash,
        "code_identity": code_identity,
    }


def passing_partition_evidence():
    evidence = {}
    for partition in CONFIRMATION_REGIMES:
        cells = {}
        for depth in EXPECTED_DEPTHS[partition]:
            for query in QUERIES:
                for family in EXPECTED_FAMILIES["confirmation"]:
                    key = evaluator._cell_key(partition, depth, query, family)
                    cells[key] = {
                        "accepted": 10,
                        "correct": 10,
                        "false_certificates": 0,
                        "rows": 23,
                        "regime": partition,
                        "depth": depth,
                        "query": query,
                        "family": family,
                    }
        evidence[partition] = {
            "accepted": 400,
            "false_certificates": 0,
            "empirical_selective_accuracy": 1.0,
            "empirical_selective_coverage": 400 / 920,
            "exact_cells": cells,
            "candidate_coverage": {
                "complete_program": 0.95,
                "event": 0.97,
                "query": 0.97,
            },
            "retrieval_backed_hot_removal": 0.75,
            "acaw_over_best_stratified_matched_coverage_baseline": 0.01,
        }
    return evidence


class FrozenGeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.calibration = frozen_geometry("calibration")
        cls.confirmation = frozen_geometry("confirmation")

    def test_exact_v2_geometry_requires_all_queries_families_and_cells(self):
        calibration = validate_record_geometry(self.calibration, "calibration")
        confirmation = validate_record_geometry(self.confirmation, "confirmation")
        self.assertEqual(calibration["rows"], 800)
        self.assertEqual(confirmation["rows"], 1840)
        self.assertEqual(set(calibration["queries"]), set(QUERIES))
        self.assertEqual(
            set(confirmation["families"]), set(EXPECTED_FAMILIES["confirmation"])
        )
        self.assertTrue(calibration["all_cells_exact"])
        self.assertTrue(confirmation["all_cells_exact"])

    def test_missing_depth_and_missing_cell_fail_closed(self):
        wrong_depth = list(self.calibration)
        first = wrong_depth[0]
        wrong_depth[0] = replace(
            first,
            operation_targets=(0,) * 16,
            operation_values=(1,) * 16,
        )
        with self.assertRaisesRegex(EvaluationContractError, "forbidden depth"):
            validate_record_geometry(wrong_depth, "calibration")

        missing_cell = list(self.confirmation)
        first = missing_cell[0]
        replacement_family = next(
            family
            for family in EXPECTED_FAMILIES["confirmation"]
            if family != first.family
        )
        missing_cell[0] = replace(first, family=replacement_family)
        with self.assertRaisesRegex(EvaluationContractError, "undersized frozen cells"):
            validate_record_geometry(missing_cell, "confirmation")

    def test_arbitrary_confirmation_partitions_are_rejected(self):
        empty = BoundScoreReport("/synthetic", "0" * 64, {}, ())
        calibration = CalibrationResult(0.0, 0.97, 0, 0, 0, "0" * 64)
        with self.assertRaisesRegex(EvaluationContractError, "must be exactly"):
            assess_static_confirmation(
                empty, empty, calibration, ("language_ood", "arbitrary")
            )


class EmpiricalGateTests(unittest.TestCase):
    def test_98_percent_partition_accuracy_is_rejected(self):
        evidence = passing_partition_evidence()
        evidence["language_ood"] = {
            **evidence["language_ood"],
            "accepted": 400,
            "false_certificates": 8,
            "empirical_selective_accuracy": 0.98,
        }
        gates = _partition_empirical_gates(evidence)
        self.assertFalse(
            gates["acaw_empirical_selective_accuracy_at_least_99pct_each_partition"]
        )
        self.assertFalse(gates["acaw_zero_false_certificates_each_partition"])

    def test_insufficient_accepted_count_is_rejected(self):
        evidence = passing_partition_evidence()
        evidence["full_ood"] = {
            **evidence["full_ood"],
            "accepted": 399,
            "empirical_selective_coverage": 399 / 920,
        }
        gates = _partition_empirical_gates(evidence)
        self.assertFalse(gates["acaw_at_least_400_certificates_each_partition"])

    def test_family_collapsed_quota_is_rejected(self):
        evidence = passing_partition_evidence()
        cells = evidence["language_ood"]["exact_cells"]
        keys = list(cells)
        cells[keys[0]] = {**cells[keys[0]], "accepted": 9, "correct": 9}
        cells[keys[1]] = {**cells[keys[1]], "accepted": 11, "correct": 11}
        gates = _partition_empirical_gates(evidence)
        self.assertTrue(gates["acaw_at_least_400_certificates_each_partition"])
        self.assertFalse(gates["acaw_at_least_10_certificates_each_exact_cell"])

    def test_no_population_confidence_claim_or_bound_remains(self):
        self.assertFalse(hasattr(evaluator, "clopper_pearson_lower_bound"))
        serialized = json.dumps(passing_partition_evidence(), sort_keys=True).lower()
        for invalid in (
            "clopper-pearson",
            "simultaneous_95pct",
            "finite-sample conformal",
            "guarantee_scope",
        ):
            self.assertNotIn(invalid, serialized)

    def test_calibration_is_program_level_and_uses_only_source_regimes(self):
        records = tuple(
            valid_record(
                index,
                "cal-{:03d}".format(index),
                CALIBRATION_REGIMES[index % 2],
                true_probability=0.39,
            )
            for index in range(34)
        )
        result = calibrate_program_threshold(records)
        self.assertEqual(result.programs, 34)
        self.assertEqual(result.order_statistic, 34)
        self.assertEqual(result.quantile, 0.97)
        self.assertAlmostEqual(result.threshold, -math.log(0.39))


class GateBindingTests(unittest.TestCase):
    def test_frozen_gate_and_admission_bind_exact_v2_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = build_gate_fixture(temporary)
            bundle = validate_gate_bundle(
                fixture["gate_path"],
                expected_manifest_sha256=fixture["gate_sha256"],
                admission_path=fixture["admission_path"],
                expected_admission_sha256=fixture["admission_sha256"],
                expected_board_bindings=fixture["bindings"],
                expected_evaluator_sha256=fixture["evaluator_sha256"],
                expected_extractor_sha256=fixture["extractor_sha256"],
                evaluator_path=EVALUATOR_PATH,
                extractor_path=EXTRACTOR_PATH,
                expected_code_revision=CODE_REVISION,
                expected_adapter_sha256=fixture["adapter_sha256"],
                expected_seed=evaluator.EXPECTED_EXTRACTOR_SEED,
            )
            self.assertEqual(bundle.manifest_sha256, fixture["gate_sha256"])
            self.assertEqual(bundle.admission_sha256, fixture["admission_sha256"])

    def test_legacy_confidence_field_is_rejected_anywhere_in_gate(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = build_gate_fixture(temporary)
            gate = json.loads(fixture["gate_path"].read_text())
            gate["confirmation_thresholds"]["confidence_scope"] = {}
            write_json(fixture["gate_path"], gate)
            with self.assertRaisesRegex(EvaluationContractError, "forbidden"):
                validate_gate_bundle(
                    fixture["gate_path"],
                    expected_manifest_sha256=sha256_file(fixture["gate_path"]),
                    admission_path=fixture["admission_path"],
                    expected_admission_sha256=fixture["admission_sha256"],
                    expected_board_bindings=fixture["bindings"],
                    expected_evaluator_sha256=fixture["evaluator_sha256"],
                    expected_extractor_sha256=fixture["extractor_sha256"],
                    evaluator_path=EVALUATOR_PATH,
                    extractor_path=EXTRACTOR_PATH,
                    expected_code_revision=CODE_REVISION,
                    expected_adapter_sha256=fixture["adapter_sha256"],
                    expected_seed=evaluator.EXPECTED_EXTRACTOR_SEED,
                )

    def test_code_identity_source_runtime_aggregate_and_revision_tampering_fails(self):
        def source(identity):
            key = next(iter(identity["files"]))
            identity["files"][key] = "f" * 64
            identity["aggregate_sha256"] = code_identity_aggregate(
                identity["git_revision"], identity["files"], identity["runtime"]
            )

        def runtime(identity):
            identity["runtime"]["python"] = "0.0.0"
            identity["aggregate_sha256"] = code_identity_aggregate(
                identity["git_revision"], identity["files"], identity["runtime"]
            )

        def aggregate(identity):
            identity["aggregate_sha256"] = "0" * 64

        def revision(identity):
            identity["git_revision"] = "b" * 40
            identity["aggregate_sha256"] = code_identity_aggregate(
                identity["git_revision"], identity["files"], identity["runtime"]
            )

        for name, mutate in (
            ("source", source),
            ("runtime", runtime),
            ("aggregate", aggregate),
            ("revision", revision),
        ):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                fixture = build_gate_fixture(temporary)
                gate = json.loads(fixture["gate_path"].read_text())
                mutate(gate["code_identity"])
                write_json(fixture["gate_path"], gate)
                with self.assertRaises(EvaluationContractError):
                    validate_gate_bundle(
                        fixture["gate_path"],
                        expected_manifest_sha256=sha256_file(fixture["gate_path"]),
                        admission_path=fixture["admission_path"],
                        expected_admission_sha256=fixture["admission_sha256"],
                        expected_board_bindings=fixture["bindings"],
                        expected_evaluator_sha256=fixture["evaluator_sha256"],
                        expected_extractor_sha256=fixture["extractor_sha256"],
                        evaluator_path=EVALUATOR_PATH,
                        extractor_path=EXTRACTOR_PATH,
                        expected_code_revision=CODE_REVISION,
                        expected_adapter_sha256=fixture["adapter_sha256"],
                        expected_seed=evaluator.EXPECTED_EXTRACTOR_SEED,
                    )

    def test_gate_admission_with_missing_exact_cell_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = build_gate_fixture(temporary)
            admission = json.loads(fixture["admission_path"].read_text())
            admission["boards"]["confirmation"]["exact_cells"].popitem()
            write_json(fixture["admission_path"], admission)
            admission_hash = sha256_file(fixture["admission_path"])
            gate = json.loads(fixture["gate_path"].read_text())
            gate["admission_report"]["sha256"] = admission_hash
            write_json(fixture["gate_path"], gate)
            with self.assertRaisesRegex(EvaluationContractError, "exact_cells"):
                validate_gate_bundle(
                    fixture["gate_path"],
                    expected_manifest_sha256=sha256_file(fixture["gate_path"]),
                    admission_path=fixture["admission_path"],
                    expected_admission_sha256=admission_hash,
                    expected_board_bindings=fixture["bindings"],
                    expected_evaluator_sha256=fixture["evaluator_sha256"],
                    expected_extractor_sha256=fixture["extractor_sha256"],
                    evaluator_path=EVALUATOR_PATH,
                    extractor_path=EXTRACTOR_PATH,
                    expected_code_revision=CODE_REVISION,
                    expected_adapter_sha256=fixture["adapter_sha256"],
                    expected_seed=evaluator.EXPECTED_EXTRACTOR_SEED,
                )

    def test_score_report_gate_bindings_precede_probability_access(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = build_gate_fixture(root)
            bundle = validate_gate_bundle(
                fixture["gate_path"],
                expected_manifest_sha256=fixture["gate_sha256"],
                admission_path=fixture["admission_path"],
                expected_admission_sha256=fixture["admission_sha256"],
                expected_board_bindings=fixture["bindings"],
                expected_evaluator_sha256=fixture["evaluator_sha256"],
                expected_extractor_sha256=fixture["extractor_sha256"],
                evaluator_path=EVALUATOR_PATH,
                extractor_path=EXTRACTOR_PATH,
                expected_code_revision=CODE_REVISION,
                expected_adapter_sha256=fixture["adapter_sha256"],
                expected_seed=evaluator.EXPECTED_EXTRACTOR_SEED,
            )
            base = {
                "audit": SCORE_AUDIT,
                "schema_version": SCORE_SCHEMA_VERSION,
                "board_name": "calibration",
                "code_revision": CODE_REVISION,
                "code_identity_aggregate_sha256": bundle.code_identity[
                    "aggregate_sha256"
                ],
                "evaluator": str(EVALUATOR_PATH),
                "evaluator_sha256": fixture["evaluator_sha256"],
                "extractor": str(EXTRACTOR_PATH),
                "extractor_sha256": fixture["extractor_sha256"],
                "gate_manifest": bundle.manifest_path,
                "gate_manifest_sha256": bundle.manifest_sha256,
                "gate_admission": bundle.admission_path,
                "gate_admission_sha256": bundle.admission_sha256,
                "code_identity": bundle.code_identity,
            }
            for missing in ("gate_manifest_sha256", "gate_admission_sha256"):
                payload = dict(base)
                del payload[missing]
                score = root / "missing_{}.json".format(missing)
                write_json(score, payload)
                with mock.patch.object(
                    evaluator, "_validate_probability_row"
                ) as opened:
                    with self.assertRaisesRegex(
                        EvaluationContractError, "gate bindings"
                    ):
                        validate_score_report(
                            score,
                            board_name="calibration",
                            expected_report_sha256=sha256_file(score),
                            expected_data_sha256=fixture["bindings"]["calibration"][
                                "data_sha256"
                            ],
                            expected_structural_admission_sha256=fixture["bindings"][
                                "calibration"
                            ]["structural_admission_sha256"],
                            expected_label_admission_sha256=fixture["bindings"][
                                "calibration"
                            ]["label_admission_sha256"],
                            extractor_path=EXTRACTOR_PATH,
                            expected_extractor_sha256=fixture["extractor_sha256"],
                            expected_evaluator_sha256=fixture["evaluator_sha256"],
                            gate_bundle=bundle,
                            expected_code_revision=CODE_REVISION,
                            expected_adapter_sha256=fixture["adapter_sha256"],
                        )
                    opened.assert_not_called()

            payload = json.loads(json.dumps(base))
            payload["code_identity"]["runtime"]["python"] = "0.0.0"
            score = root / "tampered_code_identity.json"
            write_json(score, payload)
            with mock.patch.object(evaluator, "_validate_probability_row") as opened:
                with self.assertRaisesRegex(
                    EvaluationContractError, "code_identity mismatch"
                ):
                    validate_score_report(
                        score,
                        board_name="calibration",
                        expected_report_sha256=sha256_file(score),
                        expected_data_sha256=fixture["bindings"]["calibration"][
                            "data_sha256"
                        ],
                        expected_structural_admission_sha256=fixture["bindings"][
                            "calibration"
                        ]["structural_admission_sha256"],
                        expected_label_admission_sha256=fixture["bindings"][
                            "calibration"
                        ]["label_admission_sha256"],
                        extractor_path=EXTRACTOR_PATH,
                        expected_extractor_sha256=fixture["extractor_sha256"],
                        expected_evaluator_sha256=fixture["evaluator_sha256"],
                        gate_bundle=bundle,
                        expected_code_revision=CODE_REVISION,
                        expected_adapter_sha256=fixture["adapter_sha256"],
                    )
            opened.assert_not_called()


class BaselineAndMechanicsTests(unittest.TestCase):
    def test_baselines_match_inside_cells_with_reference_hash_tiebreak(self):
        cases = []
        references_by_cell = {}
        index = 0
        for partition in CONFIRMATION_REGIMES:
            for depth in EXPECTED_DEPTHS[partition]:
                for query_target, query_name in enumerate(QUERIES):
                    for family in EXPECTED_FAMILIES["confirmation"]:
                        cell_name = evaluator._cell_key(
                            partition, depth, query_name, family
                        )
                        references = []
                        for offset in range(23):
                            reference = "{}-{}-{}-{}-{:03d}".format(
                                partition, depth, query_name, family, offset
                            )
                            references.append(reference)
                            record = geometry_record(
                                index,
                                reference,
                                partition,
                                depth,
                                query_target,
                                family,
                            )
                            cases.append(
                                CaseEvaluation(
                                    record=record,
                                    operation_candidates=(),
                                    query_candidates=(),
                                    top1={
                                        "answer_correct": offset % 2 == 0,
                                        "selection_scores": {
                                            "max_probability": 0.5,
                                            "minimum_top1_margin": 0.5,
                                            "maximum_entropy": 0.5,
                                        },
                                    },
                                    analysis={
                                        "acaw": {
                                            "query_certificate": offset == 0,
                                            "query_certificate_correct": offset == 0,
                                        }
                                    },
                                )
                            )
                            index += 1
                        references_by_cell[cell_name] = references
        result = _matched_coverage_baselines(cases, CONFIRMATION_REGIMES)
        family = EXPECTED_FAMILIES["confirmation"][0]
        cell_name = evaluator._cell_key("language_ood", 4, "read_0", family)
        cell = result["baselines"]["max_probability"]["partitions"]["language_ood"][
            "cells"
        ][cell_name]
        expected = min(
            hashlib.sha256(reference.encode()).hexdigest()
            for reference in references_by_cell[cell_name]
        )
        self.assertEqual(cell["selected_reference_sha256"], [expected])
        self.assertEqual(cell["family"], family)
        self.assertIn("Family pooling is forbidden", result["matching_rule"])
        with self.assertRaisesRegex(EvaluationContractError, "expected 23"):
            _matched_coverage_baselines(cases[:-1], CONFIRMATION_REGIMES)

    def test_mechanics_are_observed_counters_and_affine_cap_remains_six(self):
        record = valid_record(0, "mechanics", "language_ood")
        analyzed = analyze_candidate_program(
            record,
            ((record.operation_targets[0],),),
            (record.query_target,),
            score_report_sha256="1" * 64,
            board_sha256="2" * 64,
        )
        self.assertEqual(MAX_AFFINE_AMBIGUITY_RANK, 6)
        self.assertEqual(analyzed["mechanics"]["ambiguity_rank_excess"], 0)
        self.assertEqual(
            analyzed["mechanics"]["exact_transforms_outside_acaw_hulls"], 0
        )
        storage = analyzed["acaw"]["storage"]
        self.assertEqual(storage["unbound_evicted_source_events"], 0)
        self.assertEqual(
            storage["retrieval_bound_source_events"], storage["evicted_source_events"]
        )


class PublicationTests(unittest.TestCase):
    def cli_argv(self, output):
        return [
            "evaluate_version_space_workspace.py",
            "--calibration-scores",
            "calibration.json",
            "--calibration-scores-sha256",
            "1" * 64,
            "--calibration-data-sha256",
            "2" * 64,
            "--calibration-structural-admission-sha256",
            "3" * 64,
            "--calibration-label-admission-sha256",
            "4" * 64,
            "--test-scores",
            "confirmation.json",
            "--test-scores-sha256",
            "5" * 64,
            "--test-data-sha256",
            "6" * 64,
            "--test-structural-admission-sha256",
            "7" * 64,
            "--test-label-admission-sha256",
            "8" * 64,
            "--gate-manifest",
            "gate.json",
            "--gate-manifest-sha256",
            "9" * 64,
            "--gate-admission",
            "admission.json",
            "--gate-admission-sha256",
            "a" * 64,
            "--extractor-sha256",
            "b" * 64,
            "--evaluator-sha256",
            "c" * 64,
            "--code-revision",
            CODE_REVISION,
            "--out",
            str(output),
        ]

    def test_rejection_is_atomically_published_then_exits_one(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "rejection.json"
            result = {
                "advance_r10_static_path": False,
                "decision": "reject_r10_static_confirmation",
                "gates": {"empirical_audit": False},
            }
            with mock.patch.object(sys, "argv", self.cli_argv(output)):
                with mock.patch.object(
                    evaluator, "run_evaluation", return_value=result
                ):
                    with self.assertRaises(SystemExit) as raised:
                        main()
            self.assertEqual(raised.exception.code, 1)
            self.assertEqual(json.loads(output.read_text()), result)
            original = output.read_bytes()
            with self.assertRaises(FileExistsError):
                atomic_write_json_no_overwrite({"changed": True}, output)
            self.assertEqual(output.read_bytes(), original)

    def test_contract_error_exits_two_without_report(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "contract.json"
            with mock.patch.object(sys, "argv", self.cli_argv(output)):
                with mock.patch.object(
                    evaluator,
                    "run_evaluation",
                    side_effect=EvaluationContractError("contract failed"),
                ):
                    with self.assertRaises(SystemExit) as raised:
                        main()
            self.assertEqual(raised.exception.code, 2)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
