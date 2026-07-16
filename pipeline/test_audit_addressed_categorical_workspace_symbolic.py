import copy
import inspect
import unittest

from pipeline.addressed_categorical_workspace import (
    payload_sha256,
    run_symbolic_falsifier,
)
from pipeline.audit_addressed_categorical_workspace_symbolic import verify_report


class IndependentSymbolicAuditTests(unittest.TestCase):
    def setUp(self):
        self.report = run_symbolic_falsifier(
            coefficient_values=(0, 1, 16), bind_identity=False,
        )

    def test_independent_reconstruction_accepts_valid_report(self):
        result = verify_report(self.report, allow_unbound=True)
        self.assertTrue(result["evidence_valid"])
        self.assertFalse(result["pass"])
        self.assertEqual(
            result["updates_reconstructed"],
            sum(item["updates_checked"] for item in self.report["dimensions"]),
        )

    def test_self_hash_consistent_stream_forgery_is_rejected(self):
        forged = copy.deepcopy(self.report)
        forged["dimensions"][1]["exact_update_stream_sha256"] = "0" * 64
        forged["payload_sha256"] = payload_sha256(forged)
        with self.assertRaises(ValueError):
            verify_report(forged, allow_unbound=True)

    def test_unbound_report_is_rejected_by_default(self):
        with self.assertRaises(ValueError):
            verify_report(self.report)

    def test_auditor_does_not_import_candidate(self):
        import pipeline.audit_addressed_categorical_workspace_symbolic as auditor

        source = inspect.getsource(auditor)
        self.assertNotIn("import pipeline.addressed_categorical_workspace", source)
        self.assertNotIn("from pipeline.addressed_categorical_workspace", source)


if __name__ == "__main__":
    unittest.main()
