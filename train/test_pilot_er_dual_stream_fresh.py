from __future__ import annotations

from pilot_er_dual_stream_fresh import (
    CANARY_CHECKPOINT_SHA256,
    EXPECTED_PARAMETERS,
    SCORING_ARMS,
    THRESHOLDS,
    compute_gates,
)


def _summary(rate: float) -> dict[str, object]:
    return {"correct": int(rate * 2_048), "rows": 2_048, "rate": rate}


def _metrics(rate: float) -> dict[str, object]:
    fields = {
        name: _summary(rate)
        for name in (
            "cardinality",
            "initial_rows",
            "relation_rows",
            "rule_active",
            "events",
            "halt",
            "query",
            "line_pointer",
            "binding_pointer",
            "initial_pointer",
            "witness_pointer",
            "query_pointer",
            "packet",
            "state",
            "answer",
            "joint",
        )
    }
    grouped = {"x": {name: _summary(rate) for name in ("packet", "state", "answer", "joint")}}
    return {
        "overall": fields,
        "by_cardinality": grouped,
        "by_depth": grouped,
        "by_renderer": grouped,
        "non_bijective": {name: _summary(rate) for name in ("packet", "state", "answer", "joint")},
        "interventions": {
            name: {
                "eligible": 100,
                "sensitive": 80,
                "exact_on_eligible": 100,
                "changed_on_sensitive": 80,
            }
            for name in ("relation_derangement", "cardinality_mask", "state_reset", "query_swap")
        },
    }


def test_frozen_score_contract_is_strict_and_parameter_bounded() -> None:
    assert SCORING_ARMS == ("treatment", "family_deranged", "equality_ablated")
    assert EXPECTED_PARAMETERS["complete_system"] == 185_532_296
    assert EXPECTED_PARAMETERS["complete_system"] < 200_000_000
    assert THRESHOLDS["joint_overall"] == 0.95
    assert THRESHOLDS["source_free_joint_max"] == 0.10
    assert len(CANARY_CHECKPOINT_SHA256) == 64


def test_metric_fixture_represents_passing_and_collapsed_controls() -> None:
    treatment = _metrics(1.0)
    treatment["invariance"] = {
        name: {"exact": 2_048, "rows": 2_048, "rate": 1.0}
        for name in (
            "alpha",
            "distractor_rotation",
            "rule_storage_reindex",
            "physical_record_reindex",
        )
    }
    collapsed = _metrics(0.0)
    metrics = {
        "treatment": treatment,
        "family_deranged": collapsed,
        "equality_ablated": collapsed,
        "source_free": collapsed,
    }
    initialization = "a" * 64
    checkpoint = {
        "qualified_canary_checkpoint_sha256": CANARY_CHECKPOINT_SHA256,
        "shared_initial_state_sha256": initialization,
        "parameters": EXPECTED_PARAMETERS,
        "arms": {
            name: {
                "initial_state_sha256": initialization,
                "fit": {
                    "frozen_parent_unchanged": True,
                    "motor_parameters": 0,
                    "reader_parameters": 0,
                },
            }
            for name in SCORING_ARMS
        },
    }
    gates = compute_gates(metrics, checkpoint)
    assert gates and all(gates.values())
