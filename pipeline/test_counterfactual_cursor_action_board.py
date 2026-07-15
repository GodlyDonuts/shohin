import copy
import json
import tempfile
import unittest
from pathlib import Path

import audit_counterfactual_cursor_action_board as audit
import generate_counterfactual_cursor_action_board as generate


class CounterfactualCursorActionBoardTest(unittest.TestCase):
    def setUp(self):
        self.document = generate.generate_document()

    def test_exact_geometry_and_scores(self):
        report = audit.audit_document(self.document)
        self.assertTrue(report["all_passed"])
        self.assertEqual(report["counts"]["rows"], 600)
        self.assertEqual(report["counts"]["sources"], 120)
        self.assertEqual(report["counts"]["adjacent_order_pairs"], 180)
        self.assertEqual(report["collapse"]["event_fsm_states"], 12)
        self.assertEqual(report["collapse"]["event_transition_assertions"], 96)
        self.assertEqual(report["collapse"]["query_projection_assertions"], 320)
        self.assertEqual(report["collapse"]["valid_event_trace"][-1], ["HALT", "HALT"])
        self.assertEqual(report["symbolic_scores"], {
            "oracle_source_cursor": 600,
            "global_constant": 120,
            "best_source_only": 120,
            "best_renderer_only": 120,
            "best_cursor_only": 240,
            "best_renderer_cursor": 240,
            "oracle_cursor_clamped_zero": 120,
            "oracle_cursor_five_cycle": 0,
        })

    def rehash(self, document):
        document["rows_sha256"] = generate.sha256_bytes(generate.canonical_json(document["rows"]))

    def assert_rejected(self, document, message):
        with self.assertRaisesRegex(ValueError, message):
            audit.audit_document(document)

    def test_rejects_target_tamper_even_after_rehash(self):
        document = copy.deepcopy(self.document)
        document["rows"][0]["target_action"] = "subtract"
        document["rows"][0]["target_index"] = 1
        self.rehash(document)
        self.assert_rejected(document, "target action mismatch")

    def test_rejects_source_tamper_even_after_rehash(self):
        document = copy.deepcopy(self.document)
        document["rows"][0]["source"] += " extra"
        self.rehash(document)
        self.assert_rejected(document, "recover exactly one")

    def test_rejects_cursor_leak_or_duplicate(self):
        leaked = copy.deepcopy(self.document)
        leaked["rows"][1]["source"] += " cursor one"
        self.rehash(leaked)
        self.assert_rejected(leaked, "recover exactly one")

        duplicate = copy.deepcopy(self.document)
        duplicate["rows"][1] = copy.deepcopy(duplicate["rows"][0])
        self.rehash(duplicate)
        self.assert_rejected(duplicate, "duplicate row ID")

    def test_rejects_pair_map_and_hash_tamper(self):
        pairs = copy.deepcopy(self.document)
        pairs["adjacent_order_pairs"][0]["swap_index"] = 2
        self.assert_rejected(pairs, "pair map mismatch")

        hashed = copy.deepcopy(self.document)
        hashed["rows_sha256"] = "0" * 64
        self.assert_rejected(hashed, "row hash mismatch")

    def test_rejects_reordering_boolean_integers_and_exposure_tamper(self):
        reordered = copy.deepcopy(self.document)
        reordered["rows"][0], reordered["rows"][1] = reordered["rows"][1], reordered["rows"][0]
        self.rehash(reordered)
        self.assert_rejected(reordered, "canonical row ordering mismatch")

        boolean_id = copy.deepcopy(self.document)
        boolean_id["rows"][0]["permutation_id"] = False
        boolean_id["rows"][0]["target_index"] = False
        self.rehash(boolean_id)
        self.assert_rejected(boolean_id, "permutation ID is not an integer")

        exposed = copy.deepcopy(self.document)
        exposed["exposure_contract"]["selector_model_visible_row_fields"].append("target_action")
        self.assert_rejected(exposed, "model exposure contract mismatch")

    def test_strict_json_rejects_duplicate_keys(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "duplicate.json"
            path.write_text('{"schema": 1, "schema": 2}\n')
            with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
                audit.load_json_strict(path)

    def test_exclusive_read_only_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            board_path = Path(directory) / "board.json"
            report_path = Path(directory) / "report.json"
            generate.write_exclusive_read_only(board_path, self.document)
            self.assertEqual(board_path.stat().st_mode & 0o777, 0o444)
            with self.assertRaises(FileExistsError):
                generate.write_exclusive_read_only(board_path, self.document)
            board = json.loads(board_path.read_text())
            report = audit.audit_document(board, board_file_sha256=audit.file_sha256(board_path))
            self.assertEqual(report["board_file_sha256"], audit.file_sha256(board_path))
            audit.write_exclusive_read_only(report_path, report)
            self.assertEqual(report_path.stat().st_mode & 0o777, 0o444)


if __name__ == "__main__":
    unittest.main()
