#!/usr/bin/env python3
"""Focused contract tests for the frozen cursor-action canary loader."""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path


TRAIN = Path(__file__).resolve().parent
ROOT = TRAIN.parent
PIPELINE = ROOT / "pipeline"
sys.path.insert(0, str(TRAIN))
sys.path.insert(0, str(PIPELINE))

import audit_counterfactual_cursor_action_canary as audit  # noqa: E402
import generate_counterfactual_cursor_action_canary as generate  # noqa: E402
from counterfactual_cursor_action_data import (  # noqa: E402
    ALL_ARMS,
    CanaryArm,
    ModelInputMode,
    SidecarModelInput,
    TextControlModelInput,
    load_canary,
    validate_model_input_fields,
)


TOKENIZER = ROOT / "artifacts/shohin-tok-32k.json"
EVALGRAMS = ROOT / "artifacts/evals/evalgrams.pkl"


def _write_read_only_json(path: Path, value: object) -> None:
    path.write_bytes(json.dumps(value, indent=2, sort_keys=True).encode("ascii") + b"\n")
    path.chmod(0o444)


class CounterfactualCursorActionDataTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.document = generate.generate_document(TOKENIZER, bind_identity=False)
        cls.canary_bytes = json.dumps(
            cls.document, indent=2, sort_keys=True,
        ).encode("ascii") + b"\n"
        cls.canary_sha256 = hashlib.sha256(cls.canary_bytes).hexdigest()
        cls.report = audit.audit_document(
            cls.document,
            TOKENIZER,
            EVALGRAMS,
            canary_file_sha256=cls.canary_sha256,
        )

    def _artifacts(self) -> tuple[tempfile.TemporaryDirectory[str], Path, Path, Path]:
        directory = tempfile.TemporaryDirectory()
        root = Path(directory.name)
        canary_path = root / "canary.json"
        audit_path = root / "audit.json"
        tokenizer_path = root / "tokenizer.json"
        canary_path.write_bytes(self.canary_bytes)
        canary_path.chmod(0o444)
        _write_read_only_json(audit_path, self.report)
        shutil.copyfile(TOKENIZER, tokenizer_path)
        tokenizer_path.chmod(0o444)
        return directory, canary_path, audit_path, tokenizer_path

    def test_stable_split_api_and_gold_isolation(self) -> None:
        directory, canary_path, audit_path, tokenizer_path = self._artifacts()
        with directory:
            dataset = load_canary(canary_path, audit_path, tokenizer_path)

        train = dataset.split("train")
        self.assertEqual(train.counts.sources, 1152)
        self.assertEqual(train.counts.cells, 5760)
        self.assertEqual(len(train.sources), train.counts.sources)
        self.assertEqual(len(train.cells), train.counts.cells)
        self.assertEqual(len(train.training_units), train.counts.training_units)
        self.assertIs(train.source_by_id[train.sources[0].source_id], train.sources[0])
        self.assertIs(
            train.cell_by_key[(train.cells[0].source_id, train.cells[0].cursor)],
            train.cells[0],
        )

        unit = train.training_units[0]
        self.assertEqual([pair.renderer_id for pair in unit.adjacent_pairs], [0, 1, 2, 3, 4, 5])
        for pair in unit.adjacent_pairs:
            self.assertEqual([cell.cursor for cell in pair.left_cells], [0, 1, 2, 3, 4])
            self.assertEqual([cell.cursor for cell in pair.right_cells], [0, 1, 2, 3, 4])
            self.assertTrue(pair.left_source_id.startswith("train-r"))
            self.assertTrue(pair.right_source_id.startswith("train-r"))

        sidecar = train.sidecar_examples((0, 1))
        text = train.text_examples((0, 1))
        self.assertIsInstance(sidecar[0].model_input, SidecarModelInput)
        self.assertIsInstance(text[0].model_input, TextControlModelInput)
        self.assertEqual(
            tuple(field.name for field in dataclasses.fields(SidecarModelInput)),
            ("prompt_token_ids", "cursor"),
        )
        self.assertEqual(
            tuple(field.name for field in dataclasses.fields(TextControlModelInput)),
            ("text_prompt_token_ids",),
        )
        self.assertTrue(SidecarModelInput.__dataclass_params__.frozen)
        self.assertTrue(TextControlModelInput.__dataclass_params__.frozen)
        self.assertFalse(hasattr(sidecar[0].model_input, "source_id"))
        self.assertFalse(hasattr(sidecar[0].model_input, "target_index"))
        self.assertFalse(hasattr(text[0].model_input, "cursor"))
        self.assertFalse(hasattr(train.relations, "sidecar_examples"))

    def test_all_six_arm_maps_and_batches_are_deterministic(self) -> None:
        directory, canary_path, audit_path, tokenizer_path = self._artifacts()
        with directory:
            split = load_canary(canary_path, audit_path, tokenizer_path).split("development")

        expected_rows = tuple(range(split.counts.cells))
        for arm in ALL_ARMS:
            index_map = split.arm_index_maps[arm]
            self.assertEqual(index_map.row_indices, expected_rows)
            if arm is CanaryArm.TEXT_CURSOR_LORA:
                self.assertEqual(index_map.input_mode, ModelInputMode.TEXT_CONTROL)
            else:
                self.assertEqual(index_map.input_mode, ModelInputMode.SIDECAR)
            batches = split.batches(arm, 127)
            self.assertEqual(
                tuple(index for batch in batches for index in batch.row_indices), expected_rows,
            )
            self.assertEqual(
                tuple(label for batch in batches for label in batch.labels),
                tuple(cell.label for cell in split.cells),
            )

        normal = split.examples_for_arm(CanaryArm.TREATMENT, (0, 1, 2))
        source_only = split.examples_for_arm(CanaryArm.SOURCE_ONLY, (0, 1, 2))
        self.assertEqual([item.model_input.cursor for item in normal], [0, 1, 2])
        self.assertEqual([item.model_input.cursor for item in source_only], [0, 0, 0])
        self.assertEqual([item.label for item in normal], [item.label for item in source_only])

    def test_rejects_forbidden_model_field_requests(self) -> None:
        directory, canary_path, audit_path, tokenizer_path = self._artifacts()
        with directory:
            with self.assertRaisesRegex(ValueError, "forbidden model input fields"):
                load_canary(
                    canary_path,
                    audit_path,
                    tokenizer_path,
                    requested_model_fields={
                        "sidecar": ("prompt_token_ids", "cursor", "target_index"),
                    },
                )
        with self.assertRaisesRegex(ValueError, "forbidden model input fields"):
            validate_model_input_fields("text_control", ("text_prompt_token_ids", "source_id"))

    def test_rejects_tampered_payload_hash_and_audit_bindings(self) -> None:
        directory, canary_path, audit_path, tokenizer_path = self._artifacts()
        with directory:
            tampered = copy.deepcopy(self.document)
            tampered["payload_sha256"] = "0" * 64
            canary_path.chmod(0o644)
            _write_read_only_json(canary_path, tampered)
            with self.assertRaisesRegex(ValueError, "canary payload hash mismatch"):
                load_canary(canary_path, audit_path, tokenizer_path)

        directory, canary_path, audit_path, tokenizer_path = self._artifacts()
        with directory:
            tampered_report = copy.deepcopy(self.report)
            tampered_report["all_checks_pass"] = False
            audit_path.chmod(0o644)
            _write_read_only_json(audit_path, tampered_report)
            with self.assertRaisesRegex(ValueError, "all_checks_pass"):
                load_canary(canary_path, audit_path, tokenizer_path)

        directory, canary_path, audit_path, tokenizer_path = self._artifacts()
        with directory:
            tampered_report = copy.deepcopy(self.report)
            tampered_report["canary_file_sha256"] = "0" * 64
            audit_path.chmod(0o644)
            _write_read_only_json(audit_path, tampered_report)
            with self.assertRaisesRegex(ValueError, "audit canary file hash mismatch"):
                load_canary(canary_path, audit_path, tokenizer_path)

    def test_rejects_duplicate_json_and_unsafe_input_files(self) -> None:
        directory, canary_path, audit_path, tokenizer_path = self._artifacts()
        with directory:
            canary_path.chmod(0o644)
            canary_path.write_text('{"schema": 1, "schema": 2}\n', encoding="ascii")
            canary_path.chmod(0o444)
            with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
                load_canary(canary_path, audit_path, tokenizer_path)

        directory, canary_path, audit_path, tokenizer_path = self._artifacts()
        with directory:
            link_path = canary_path.with_name("canary-link.json")
            link_path.symlink_to(canary_path)
            with self.assertRaisesRegex(ValueError, "may not be a symlink"):
                load_canary(link_path, audit_path, tokenizer_path)

        directory, canary_path, audit_path, tokenizer_path = self._artifacts()
        with directory:
            tokenizer_path.chmod(0o644)
            with self.assertRaisesRegex(ValueError, "tokenizer input must be read-only"):
                load_canary(canary_path, audit_path, tokenizer_path)


if __name__ == "__main__":
    unittest.main()
