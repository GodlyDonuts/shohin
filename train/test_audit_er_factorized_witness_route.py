from __future__ import annotations

from copy import deepcopy

import pytest
import torch

from audit_er_factorized_witness_route import (
    EXPECTED_PARAMETERS,
    EXPECTED_SHA256,
    SEED,
    SOURCE_COMMIT,
    _grammar_diagnostic,
    _metric_rate,
    _validate_exact_receipt,
    _validate_metric_schema,
)


def test_factorized_audit_freezes_exact_run_identity() -> None:
    assert SOURCE_COMMIT == "4643d1a51defe53397f9bed481051621d85c0b11"
    assert SEED == 6_769_631_927_967_421_693
    assert EXPECTED_PARAMETERS["complete_system"] == 185_534_660
    assert EXPECTED_SHA256 == {
        "compiler.pt": "e93bb4cff5f316616c7a02bce272112acf454f42f56f8b4ea07ffac6074318a2",
        "train_probe_evidence.pt": "11d931b37ad854de9976015fc1ff38522da0812776ad67ea7d408d43820889c6",
        "train_probe_report.json": "87ea12a28cfaf82c4556f1730da6778cf63df9e5c2ad3df8b09a86613786ccca",
    }


def test_factorized_grammar_diagnostic_uses_effective_gate_and_frozen_targets() -> None:
    table = torch.zeros(14, 12, 14)
    gate = torch.ones(12)
    for cardinality in range(3, 7):
        count = 1 + 2 * cardinality
        for position in range(cardinality):
            table[count, position, position + 1] = 10.0
            table[count, 6 + position, cardinality + position + 1] = 10.0
    perfect = _grammar_diagnostic(
        {
            "er_fw_witness_address_bias": table,
            "er_fw_witness_gate": gate,
        }
    )
    assert perfect["correct"] == 36
    assert perfect["rows"] == 36

    disabled = _grammar_diagnostic(
        {
            "er_fw_witness_address_bias": table,
            "er_fw_witness_gate": torch.zeros(12),
        }
    )
    assert disabled["correct"] == 0


def test_factorized_metric_rate_is_independently_recounted() -> None:
    assert _metric_rate({"correct": 1, "rows": 2, "rate": 0.5}) == 0.5
    with pytest.raises(AssertionError, match="rate differs"):
        _metric_rate({"correct": 1, "rows": 2, "rate": 0.75})
    with pytest.raises(AssertionError, match="exact-rate"):
        _validate_exact_receipt({"exact": 0, "rows": 8_000, "rate": 1.0})


def test_factorized_metric_schema_kills_missing_leaf() -> None:
    overall_names = {
        "answer",
        "binding_pointer",
        "cardinality",
        "events",
        "halt",
        "initial_pointer",
        "initial_rows",
        "joint",
        "line_pointer",
        "packet",
        "query",
        "query_pointer",
        "relation_rows",
        "rule_active",
        "state",
        "witness_pointer",
    }
    aggregate_names = {"answer", "joint", "packet", "state"}

    def leaf(rows):
        return {"correct": rows, "rows": rows, "rate": 1.0}

    overall = {name: leaf(24) for name in overall_names}
    groups = {
        "by_cardinality": {
            str(index): {name: leaf(6) for name in aggregate_names}
            for index in range(3, 7)
        },
        "by_depth": {
            str(index): {name: leaf(2) for name in aggregate_names}
            for index in range(1, 13)
        },
        "by_renderer": {
            name: {metric: leaf(6) for metric in aggregate_names}
            for name in (
                "er-tt-d0w0e0q0-v1",
                "er-tt-d0w0e1q1-v1",
                "er-tt-d1w1e0q0-v1",
                "er-tt-d1w1e1q1-v1",
            )
        },
    }
    metrics = {
        "overall": overall,
        **groups,
        "non_bijective": {name: leaf(24) for name in aggregate_names},
    }
    assert _validate_metric_schema(metrics) == 100
    mutated = deepcopy(metrics)
    del mutated["overall"]["query"]
    with pytest.raises(AssertionError, match="overall metric schema"):
        _validate_metric_schema(mutated)
