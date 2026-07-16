import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "train"))

from audit_counterfactual_cursor_action_dev_view import project
from counterfactual_cursor_action_dev_view import _pairs


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


if __name__ == "__main__":
    unittest.main()
