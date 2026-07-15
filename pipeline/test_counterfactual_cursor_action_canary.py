import copy
import pickle
import sys
import tempfile
import unittest
from pathlib import Path

import audit_counterfactual_cursor_action_canary as audit
import generate_counterfactual_cursor_action_canary as generate


ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "train"
if str(TRAIN) not in sys.path:
    sys.path.insert(0, str(TRAIN))

from counterfactual_cursor_action_data import (  # noqa: E402
    IMPLEMENTATION_PATHS as LOADER_IMPLEMENTATION_PATHS,
)

TOKENIZER = ROOT / "artifacts/shohin-tok-32k.json"
EVALGRAMS = ROOT / "artifacts/evals/evalgrams.pkl"


class CounterfactualCursorActionCanaryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.document = generate.generate_document(TOKENIZER, bind_identity=False)

    def setUp(self):
        self.candidate = copy.deepcopy(self.document)

    def test_implementation_ledgers_are_identical(self):
        self.assertEqual(generate.IMPLEMENTATION_PATHS, audit.IMPLEMENTATION_PATHS)
        self.assertEqual(generate.IMPLEMENTATION_PATHS, LOADER_IMPLEMENTATION_PATHS)

    def rehash(self, split=None):
        if split is not None:
            payload = self.candidate["splits"][split]
            payload["sources_sha256"] = generate.sha256_bytes(
                generate.canonical_json(payload["sources"])
            )
            payload["cells_sha256"] = generate.sha256_bytes(
                generate.canonical_json(payload["cells"])
            )
        self.candidate["payload_sha256"] = generate.sha256_bytes(generate.canonical_json({
            key: value for key, value in self.candidate.items() if key != "payload_sha256"
        }))

    def audit(self):
        return audit.audit_document(self.candidate, TOKENIZER, EVALGRAMS)

    def assert_rejected(self, message):
        with self.assertRaisesRegex(ValueError, message):
            self.audit()

    def test_exact_geometry_and_shortcut_ceilings(self):
        report = self.audit()
        self.assertTrue(report["all_checks_pass"])
        self.assertEqual(report["split_summary"]["train"]["cells"], 5760)
        self.assertEqual(report["split_summary"]["development"]["cells"], 960)
        self.assertEqual(report["split_summary"]["confirmation"]["cells"], 4800)
        self.assertEqual(report["split_summary"]["train"]["training_units"], 288)
        self.assertEqual(report["split_summary"]["confirmation"]["cursor_only_ceiling"], 1920)
        self.assertEqual(report["split_summary"]["confirmation"]["public_evalgram_hits"], 0)
        self.assertEqual(
            report["pretraining_corpus_overlap"],
            audit.PRETRAINING_OVERLAP_STATUS,
        )
        self.assertFalse(report["pretraining_corpus_overlap"]["claim_authorized"])

    def test_rejects_target_tamper_after_full_rehash(self):
        cell = self.candidate["splits"]["confirmation"]["cells"][0]
        cell["target_action"] = "subtract"
        cell["target_index"] = 1
        cell["target_token_id"] = 5498
        self.rehash("confirmation")
        self.assert_rejected("target action mismatch")

    def test_rejects_source_and_token_tamper_after_full_rehash(self):
        source = self.candidate["splits"]["train"]["sources"][0]
        source["source_text"] += " altered"
        self.rehash("train")
        self.assert_rejected("source text mismatch")

        self.candidate = copy.deepcopy(self.document)
        source = self.candidate["splits"]["train"]["sources"][0]
        source["prompt_token_ids"][-1] += 1
        self.rehash("train")
        self.assert_rejected("prompt tokenization mismatch")

    def test_rejects_reordering_boolean_and_pair_tamper(self):
        sources = self.candidate["splits"]["development"]["sources"]
        sources[0], sources[1] = sources[1], sources[0]
        self.rehash("development")
        self.assert_rejected("canonical source ordering mismatch")

        self.candidate = copy.deepcopy(self.document)
        self.candidate["splits"]["train"]["sources"][0]["pack_id"] = False
        self.rehash("train")
        self.assert_rejected("invalid source pack")

        self.candidate = copy.deepcopy(self.document)
        self.candidate["splits"]["confirmation"]["adjacent_pairs"][0]["swap_index"] = 2
        self.rehash()
        self.assert_rejected("adjacent pair map mismatch")

    def test_rejects_exposure_payload_and_evalgram_identity_tamper(self):
        self.candidate["exposure_contract"]["sidecar_model_row_inputs"].append("target_index")
        self.rehash()
        self.assert_rejected("exposure contract mismatch")

        self.candidate = copy.deepcopy(self.document)
        self.candidate["payload_sha256"] = "0" * 64
        self.assert_rejected("payload hash mismatch")

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "evalgrams.pkl"
            with EVALGRAMS.open("rb") as source:
                value = pickle.load(source)
            value["grams"] = set(value["grams"])
            value["grams"].add("definitely not the frozen set")
            with path.open("wb") as target:
                pickle.dump(value, target)
            with self.assertRaisesRegex(ValueError, "evalgrams hash mismatch"):
                audit.audit_document(self.candidate, TOKENIZER, path)

    def test_latin_operand_balance_rejects_operation_magnitude_leak(self):
        contract = generate.load_contract()
        for split_name, split in contract["splits"].items():
            audit.audit_latin_packs(split["packs"], split_name)

        leaked = copy.deepcopy(contract["splits"]["train"]["packs"])
        leaked[0]["add"] = 999
        with self.assertRaisesRegex(ValueError, "operand marginals leak operation identity"):
            audit.audit_latin_packs(leaked, "train")

    def test_strict_json_and_immutable_publication(self):
        with tempfile.TemporaryDirectory() as directory:
            duplicate = Path(directory) / "duplicate.json"
            duplicate.write_text('{"schema": 1, "schema": 2}\n', encoding="ascii")
            with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
                audit.load_json_strict(duplicate)

            canary_path = Path(directory) / "canary.json"
            report_path = Path(directory) / "audit.json"
            generate.write_exclusive_read_only(canary_path, self.candidate)
            self.assertEqual(canary_path.stat().st_mode & 0o777, 0o444)
            audit.require_regular_read_only(canary_path)
            report = audit.audit_document(
                audit.load_json_strict(canary_path), TOKENIZER, EVALGRAMS,
                canary_file_sha256=audit.file_sha256(canary_path),
            )
            audit.write_exclusive_read_only(report_path, report)
            self.assertEqual(report_path.stat().st_mode & 0o777, 0o444)
            with self.assertRaises(FileExistsError):
                generate.write_exclusive_read_only(canary_path, self.candidate)


if __name__ == "__main__":
    unittest.main()
