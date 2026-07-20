from __future__ import annotations

from copy import deepcopy
import unittest

from assess_s9_2_global_anchor import assess


def passing_evaluation():
    depth = {str(value): {"accuracy": 0.995} for value in range(3, 9)}
    arm = {
        "state": 2040,
        "answer": 2040,
        "state_accuracy": 2040 / 2048,
        "answer_accuracy": 2040 / 2048,
        "depth": depth,
        "total": 2048,
    }
    low = {
        "state": 100,
        "answer": 100,
        "state_accuracy": 100 / 2048,
        "answer_accuracy": 100 / 2048,
        "depth": depth,
        "total": 2048,
    }
    fit = {
        "unique_sources": 24_000,
        "charged_views": 48_000,
        "batch_size": 64,
        "updates": 750,
        "negative_candidates_per_view": 128,
        "orbit_weight": 0.25,
        "learning_rate": 1e-3,
        "warmup_updates": 50,
        "gradient_clip": 1.0,
    }
    root = {
        "span_exact_accuracy": 0.995,
        "count_exact_accuracy": 0.995,
    }
    return _complete_fit_contract({
        "rows": 2048,
        "span": {"f1": 0.995, "class_exact_accuracy": 0.995},
        "root": {"treatment": root},
        "graph": {
            "valid": 2040,
            "exact": 2040,
            "valid_accuracy": 2040 / 2048,
            "exact_accuracy": 2040 / 2048,
            "positive_only_exact_accuracy": 2035 / 2048,
            "no_class_exact_accuracy": 0.80,
            "shuffled_exact_accuracy": 0.0,
            "layout_exact_accuracy": 0.05,
            "uniform_exact": 0,
            "source_free_exact_accuracy": 0.0,
            "local_root_exact": 2025,
            "unconstrained_exact_accuracy": 0.94,
        },
        "arms": {
            "treatment": arm,
            "positive_only": arm,
            "no_class_message": arm,
            "layout": low,
            "reversed_links": low,
            "deranged_cards": low,
            "one_witness": low,
            "state_reset": low,
            "early_nil": low,
        },
        "invariance": {
            "eligible": 2040,
            "class_reindex": 2040,
            "relation_storage_reindex": 2040,
            "nonce_eligible": 2040,
            "nonce_graph_identical": 2040,
            "nonce_state_identical": 2040,
            "nonce_answer_identical": 2040,
            "nonce_root_eligible": 2040,
            "nonce_root_identical": 2040,
            "nonce_count_identical": 2040,
        },
        "parameters": {"complete_system": 134_580_264},
        "architecture": {
            "compiler_class": "OccurrenceQuotientCompiler",
            "layer": 19,
            "width": 384,
            "heads": 8,
            "encoder_layers": 5,
            "ff": 1408,
            "max_span_width": 4,
            "negative_candidates_per_view": 128,
            "hard_negative_top_k": 8,
            "added_trainable_parameters": 0,
        },
        "training_contract": {
            "unique_sources_per_arm": 24_000,
            "charged_views_per_arm": 48_000,
            "batch_size": 64,
            "pair_batch_size": 32,
            "updates_per_arm": 750,
            "negative_candidates_per_view": 128,
            "hard_negative_top_k": 8,
            "orbit_weight": 0.25,
            "learning_rate": 1e-3,
            "warmup_updates": 50,
            "gradient_clip": 1.0,
        },
        "fit": {
            name: deepcopy(fit)
            for name in (
                "treatment",
                "positive_only",
                "no_class",
                "shuffled",
                "layout",
            )
        },
        "development_accesses": 1,
        "confirmation_accesses": 0,
        "access_ledger_sha256": "a" * 64,
        "base_sha256": "b" * 64,
        "tokenizer_sha256": "c" * 64,
        "source_commit": "d" * 40,
    })


def _complete_fit_contract(evaluation):
    values = {
        "treatment": ("full", True, False, 8),
        "positive_only": ("positive", True, False, None),
        "no_class": ("full", False, False, 8),
        "shuffled": ("full", True, False, 8),
        "layout": ("full", False, True, 8),
    }
    for name, (mode, classes, masked, top_k) in values.items():
        evaluation["fit"][name].update(
            orbit_mode=mode,
            class_messages=classes,
            mask_gold_tokens=masked,
            hard_negative_top_k=top_k,
        )
    return evaluation


class S92AssessmentTest(unittest.TestCase):
    def test_passing_fixture_qualifies(self):
        result = assess(passing_evaluation())
        self.assertTrue(all(result["gates"].values()))
        self.assertEqual(result["gate_summary"]["inherited_total"], 31)
        self.assertEqual(
            result["decision"],
            "qualify_s9_2_global_anchor_for_fresh_confirmation",
        )

    def test_2031_threshold_includes_graph_state_and_answer(self):
        for field, container in (
            ("exact", "graph"),
            ("state", "treatment"),
            ("answer", "treatment"),
        ):
            evaluation = passing_evaluation()
            if container == "graph":
                evaluation["graph"][field] = 2030
            else:
                evaluation["arms"][container][field] = 2030
            result = assess(evaluation)
            self.assertFalse(
                result["s9_2_gates"]["exact_graph_state_answer_at_least_2031"]
            )

    def test_global_must_strictly_beat_local_root(self):
        evaluation = passing_evaluation()
        evaluation["graph"]["local_root_exact"] = evaluation["graph"]["exact"]
        result = assess(evaluation)
        self.assertFalse(
            result["s9_2_gates"]["global_strictly_beats_same_logit_local_root"]
        )

    def test_root_and_count_gates_are_independent(self):
        evaluation = passing_evaluation()
        evaluation["root"]["treatment"]["span_exact_accuracy"] = 0.989
        result = assess(evaluation)
        self.assertFalse(result["s9_2_gates"]["root_spans_at_least_99pct_exact"])
        self.assertTrue(result["s9_2_gates"]["root_counts_at_least_99pct_exact"])

    def test_layout_and_five_arm_budget_are_frozen(self):
        evaluation = passing_evaluation()
        evaluation["graph"]["layout_exact_accuracy"] = 0.10
        evaluation["fit"]["positive_only"]["updates"] = 749
        result = assess(evaluation)
        self.assertFalse(result["s9_2_gates"]["layout_exact_below_10pct"])
        self.assertFalse(result["s9_2_gates"]["equal_five_arm_budget"])

    def test_same_parameter_architecture_and_arm_changes_fail_contract(self):
        evaluation = passing_evaluation()
        evaluation["architecture"]["layer"] = 18
        self.assertFalse(
            assess(evaluation)["s9_2_gates"]["equal_five_arm_budget"]
        )
        evaluation = passing_evaluation()
        evaluation["fit"]["layout"]["mask_gold_tokens"] = False
        self.assertFalse(
            assess(evaluation)["s9_2_gates"]["equal_five_arm_budget"]
        )

    def test_access_gate_requires_persistent_hash_receipts(self):
        evaluation = passing_evaluation()
        evaluation.pop("access_ledger_sha256")
        self.assertFalse(
            assess(evaluation)["inherited_gates"][
                "one_development_zero_confirmation_access"
            ]
        )

    def test_parameter_count_must_remain_exact(self):
        evaluation = passing_evaluation()
        evaluation["parameters"]["complete_system"] += 1
        result = assess(evaluation)
        self.assertFalse(
            result["s9_2_gates"]["complete_system_exactly_134580264"]
        )

    def test_recode_requires_every_originally_valid_row(self):
        evaluation = passing_evaluation()
        evaluation["invariance"]["nonce_root_eligible"] -= 1
        result = assess(evaluation)
        self.assertFalse(
            result["s9_2_gates"]["operation_nonce_root_decisions_identical"]
        )
        self.assertFalse(result["s9_2_gates"]["operation_nonce_counts_identical"])


if __name__ == "__main__":
    unittest.main()
