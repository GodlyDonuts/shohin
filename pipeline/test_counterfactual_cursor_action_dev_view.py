import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "train"))

from audit_counterfactual_cursor_action_dev_view import project
from counterfactual_cursor_action_dev_view import _pairs, load_development_view


ROOT = Path(__file__).resolve().parents[1]
VIEW_SHA256 = "24abd93737be57c6792a1d44c8f2e3a28d7c5fbc1666b083383350f410ce6ec9"
AUDIT_SHA256 = "33fb4792ed0a8027d49de157c295cb9ba651cdd9c59ab5cfa04a71e99af8ea25"


class DevelopmentViewTests(unittest.TestCase):
    def test_projection_keeps_only_declared_fields(self):
        source = {
            "splits": {
                "train": {
                    "sources": [{
                        "source_id": "train-r00-k00-p00",
                        "renderer_id": 0,
                        "pack_id": 0,
                        "permutation_id": 0,
                        "prompt_token_ids": [1, 2],
                        "operation_order": ["add"],
                    }],
                    "cells": [{
                        "source_id": "train-r00-k00-p00",
                        "cursor": 0,
                        "target_index": 0,
                        "target_token_id": 820,
                        "target_action": "add",
                    }],
                }
            }
        }
        observed = project(source, "train")
        self.assertEqual(
            set(observed["sources"][0]),
            {"source_id", "renderer_id", "pack_id", "permutation_id", "prompt_token_ids"},
        )
        self.assertEqual(
            set(observed["cells"][0]),
            {"source_id", "cursor", "target_index", "target_token_id"},
        )

    def test_duplicate_json_keys_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
            json.loads('{"a": 1, "a": 2}', object_pairs_hook=_pairs)

    def test_frozen_view_loads_without_confirmation(self):
        view_path = ROOT / "artifacts/r12/counterfactual_cursor_action_dev_view_v1.json"
        audit_path = ROOT / "artifacts/r12/counterfactual_cursor_action_dev_view_v1.audit.json"
        dataset = load_development_view(
            view_path,
            audit_path,
            expected_view_sha256=VIEW_SHA256,
            expected_audit_sha256=AUDIT_SHA256,
        )
        self.assertEqual([split.name for split in dataset.splits], ["train", "development"])
        self.assertNotIn(b'"confirmation"', view_path.read_bytes())
        self.assertEqual(len(dataset.split("train").cells), 5760)
        self.assertEqual(len(dataset.split("development").cells), 960)


if __name__ == "__main__":
    unittest.main()
