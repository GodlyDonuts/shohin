#!/usr/bin/env python3
"""Tests for the immutable DRS constant-bias audit."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path

import audit_drs_carry_constant_bias as audit


class CarryConstantBiasAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.raw = audit.DEFAULT_SOURCE.read_bytes()
        cls.payload = json.loads(cls.raw)

    def test_frozen_probe_replays_known_favorable_null(self):
        self.assertEqual(
            hashlib.sha256(self.raw).hexdigest(), audit.FROZEN_SOURCE_SHA256
        )
        report = audit.audit_payload(self.payload, audit.FROZEN_SOURCE_SHA256)
        self.assertEqual(report["raw_delta_zero"]["correct"], 32)
        self.assertEqual(report["raw_delta_zero"]["target_0_correct"], 13)
        self.assertEqual(report["raw_delta_zero"]["target_1_correct"], 19)
        self.assertEqual(report["optimal_total_correct"], 35)
        self.assertFalse(report["perfect_constant_feasibility"]["feasible"])
        self.assertEqual(len(report["records"]), 40)
        self.assertEqual(len(report["optimal_intervals"]), 2)
        selected = report["selected_favorable_null"]
        self.assertEqual(selected["metrics"]["correct"], 35)
        self.assertEqual(selected["metrics"]["target_0_correct"], 18)
        self.assertEqual(selected["metrics"]["target_1_correct"], 17)

        nuisance = report["fit_only_nuisance_nulls"]
        self.assertEqual(nuisance["global"]["fit_metrics"]["correct"], 15)
        self.assertEqual(nuisance["global"]["full_board_metrics"]["correct"], 35)
        self.assertEqual(
            nuisance["global"]["binary_margin_cross_entropy_fit"]["full_board_metrics"][
                "correct"
            ],
            33,
        )
        self.assertEqual(
            nuisance["global"]["fit_optimal_value_ood_score_range"][
                "minimum_eval_correct"
            ],
            11,
        )
        self.assertEqual(
            nuisance["global"]["fit_optimal_value_ood_score_range"][
                "maximum_eval_correct"
            ],
            13,
        )
        self.assertEqual(nuisance["operation_only"]["fit_metrics"]["correct"], 15)
        self.assertEqual(
            nuisance["operation_only"]["full_board_metrics"]["correct"], 35
        )
        self.assertEqual(
            nuisance["operation_only"]["binary_margin_cross_entropy_fit"][
                "full_board_metrics"
            ]["correct"],
            32,
        )
        self.assertEqual(
            nuisance["operation_only"]["fit_optimal_value_ood_score_range"],
            {
                "maximum_value_ood_correct": 13,
                "minimum_value_ood_correct": 11,
                "value_ood_total": 16,
            },
        )
        self.assertEqual(nuisance["operation_width"]["fit_metrics"]["correct"], 16)
        self.assertEqual(
            nuisance["operation_width"]["held_out_value_metrics"]["correct"],
            15,
        )
        self.assertEqual(
            nuisance["operation_width"]["binary_margin_cross_entropy_fit"][
                "held_out_value_metrics"
            ]["correct"],
            14,
        )
        self.assertEqual(
            nuisance["operation_width"]["fit_optimal_value_ood_score_range"],
            {
                "maximum_value_ood_correct": 16,
                "minimum_value_ood_correct": 11,
                "value_ood_total": 16,
            },
        )
        self.assertEqual(
            nuisance["operation_width"]["eligible_seen_width_metrics"]["correct"],
            31,
        )
        self.assertEqual(
            nuisance["operation_width"]["excluded_regimes"], ["width_ood_w8"]
        )

    def test_fit_only_nuisance_selection_ignores_ood_logits(self):
        tampered = copy.deepcopy(self.payload)
        for record in tampered["records"]:
            if record["field"] != "carry" or record["regime"].startswith("fit_"):
                continue
            layer = next(result for result in record["layers"] if result["layer"] == 29)
            for direction in ("a_to_b", "b_to_a"):
                baseline = layer[direction]["baseline"]
                baseline["own_logit"] += 100.0
                baseline["toward_other_logodds"] = (
                    baseline["other_logit"] - baseline["own_logit"]
                )
        original = audit.audit_payload(self.payload, audit.FROZEN_SOURCE_SHA256)[
            "fit_only_nuisance_nulls"
        ]
        changed = audit.audit_payload(tampered, audit.FROZEN_SOURCE_SHA256)[
            "fit_only_nuisance_nulls"
        ]
        self.assertEqual(original["global"]["delta"], changed["global"]["delta"])
        self.assertEqual(
            original["operation_only"]["deltas"],
            changed["operation_only"]["deltas"],
        )
        self.assertEqual(
            original["operation_width"]["deltas"],
            changed["operation_width"]["deltas"],
        )
        self.assertEqual(
            original["global"]["binary_margin_cross_entropy_fit"]["delta"],
            changed["global"]["binary_margin_cross_entropy_fit"]["delta"],
        )
        self.assertEqual(
            original["operation_only"]["binary_margin_cross_entropy_fit"]["deltas"],
            changed["operation_only"]["binary_margin_cross_entropy_fit"]["deltas"],
        )
        self.assertEqual(
            original["operation_width"]["binary_margin_cross_entropy_fit"]["deltas"],
            changed["operation_width"]["binary_margin_cross_entropy_fit"]["deltas"],
        )

    def test_record_order_does_not_change_report(self):
        shuffled = copy.deepcopy(self.payload)
        shuffled["records"] = list(reversed(shuffled["records"]))
        left = audit.audit_payload(self.payload, audit.FROZEN_SOURCE_SHA256)
        right = audit.audit_payload(shuffled, audit.FROZEN_SOURCE_SHA256)
        self.assertEqual(
            audit.canonical_json_bytes(left), audit.canonical_json_bytes(right)
        )

    def test_rejects_duplicate_state_identity(self):
        tampered = copy.deepcopy(self.payload)
        carry = [record for record in tampered["records"] if record["field"] == "carry"]
        carry[1]["a"]["id"] = carry[0]["a"]["id"]
        with self.assertRaisesRegex(ValueError, "duplicate state id"):
            audit.extract_rows(tampered)

    def test_rejects_nonbinary_or_nonstring_carry_target(self):
        for value in (0, "2", True):
            with self.subTest(value=value):
                tampered = copy.deepcopy(self.payload)
                carry = next(
                    record
                    for record in tampered["records"]
                    if record["field"] == "carry"
                )
                carry["a"]["target"] = value
                with self.assertRaisesRegex(ValueError, "carry target"):
                    audit.extract_rows(tampered)

    def test_rejects_state_metadata_mismatch(self):
        tampered = copy.deepcopy(self.payload)
        carry = next(
            record for record in tampered["records"] if record["field"] == "carry"
        )
        carry["a"]["state"] = carry["a"]["state"].replace(";w=4;", ";w=5;")
        with self.assertRaisesRegex(ValueError, "width changes|regime width"):
            audit.extract_rows(tampered)

    def test_rejects_logodds_that_do_not_replay(self):
        tampered = copy.deepcopy(self.payload)
        carry = next(
            record for record in tampered["records"] if record["field"] == "carry"
        )
        layer = next(result for result in carry["layers"] if result["layer"] == 29)
        layer["a_to_b"]["baseline"]["toward_other_logodds"] += 0.1
        with self.assertRaisesRegex(ValueError, "logodds do not replay"):
            audit.extract_rows(tampered)

    def test_rejects_nonfinite_and_boolean_logits(self):
        for value in (float("nan"), True):
            with self.subTest(value=value):
                tampered = copy.deepcopy(self.payload)
                carry = next(
                    record
                    for record in tampered["records"]
                    if record["field"] == "carry"
                )
                layer = next(
                    result for result in carry["layers"] if result["layer"] == 29
                )
                layer["a_to_b"]["baseline"]["own_logit"] = value
                with self.assertRaises((TypeError, ValueError)):
                    audit.extract_rows(tampered)

    def test_json_decoder_rejects_duplicate_keys_and_nonfinite_constants(self):
        with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
            audit.decode_json(b'{"x":1,"x":2}\n', "fixture")
        with self.assertRaisesRegex(ValueError, "non-finite"):
            audit.decode_json(b'{"x":NaN}\n', "fixture")

    def test_exclusive_report_is_read_only_and_not_reusable(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "report.json"
            payload = audit.canonical_json_bytes({"schema": "fixture"})
            observed = audit.exclusive_write(destination, payload)
            self.assertEqual(observed, hashlib.sha256(payload).hexdigest())
            self.assertEqual(destination.stat().st_mode & 0o222, 0)
            with self.assertRaises(FileExistsError):
                audit.exclusive_write(destination, payload)

    def test_read_immutable_rejects_writable_source(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.json"
            source.write_text("{}\n", encoding="ascii")
            os.chmod(source, 0o644)
            with self.assertRaisesRegex(PermissionError, "read-only"):
                audit.read_immutable(source)

    def test_canonical_renderer_rejects_nan(self):
        with self.assertRaises(ValueError):
            audit.canonical_json_bytes({"x": float("nan")})


if __name__ == "__main__":
    unittest.main()
