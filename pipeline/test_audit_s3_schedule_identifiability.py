#!/usr/bin/env python3

import unittest

from audit_s3_schedule_identifiability import audit


def operation(kind, entity, amount):
    return {"kind": kind, "entity": entity, "amount": amount}


def row(row_id, active, second):
    initial = ["a", "b", "c"]
    true_program = [operation("right", "b", 1)]
    if active == 2:
        true_program.append(second)
    return {
        "id": row_id,
        "initial_order": initial,
        "program": true_program,
        "query": {"position": 0},
        "chunks": [{
            "active_operations": active,
            "program": [operation("right", "b", 1), second],
        }],
    }


class ScheduleIdentifiabilityAuditTest(unittest.TestCase):
    def test_detects_semantically_identical_real_and_padding_operations(self):
        filler = operation("left", "a", 1)
        report = audit([
            row("padding", 1, filler),
            row("legitimate", 2, filler),
            row("other", 2, operation("right", "c", 2)),
        ])
        self.assertTrue(report["all_gates_pass"])
        self.assertEqual(report["filler_signature_label_histogram"], {"1": 1, "2": 1})
        self.assertEqual(report["minimum_equivariant_signature_classifier_errors"], 1)
        self.assertEqual(report["policies"]["oracle"]["exact_programs"], 3)
        self.assertEqual(report["policies"]["drop_filler_signature"]["exact_programs"], 2)

    def test_rejects_a_corpus_without_a_semantic_collision(self):
        report = audit([
            row("padding", 1, operation("left", "a", 1)),
            row("legitimate", 2, operation("right", "c", 2)),
        ])
        self.assertFalse(report["all_gates_pass"])
        self.assertEqual(report["minimum_equivariant_signature_classifier_errors"], 0)


if __name__ == "__main__":
    unittest.main()
