from __future__ import annotations

import inspect
import json

from pipeline.episode_functor_independent_world import (
    GENERATOR_SCHEMA,
    generate_independent_world,
)


KWARGS = {
    "protocol_root": "12" * 32,
    "beacon_round": 10_000,
    "beacon_value": "consumed-independent-world",
    "state_count": 5,
    "action_count": 3,
    "observer_count": 2,
    "answer_count": 5,
    "renderer_count": 1,
}


def test_independent_world_is_deterministic_complete_and_query_free() -> None:
    first = generate_independent_world(**KWARGS)
    second = generate_independent_world(**KWARGS)
    assert first == second
    assert first.admissibility_receipt["admitted"]
    assert first.admissibility_receipt["query_fields_seen"] == 0
    assert first.admissibility_receipt["generator"] == GENERATOR_SCHEMA
    assert all(first.admissibility_receipt["checks"].values())
    assert len(first.transitions) == 3
    assert all(sorted(row) == list(range(5)) for row in first.transitions)
    evidence = json.loads(first.evidence)
    assert len(evidence["demonstrations"]) == 15
    assert set(evidence) == {
        "demonstrations",
        "observations",
        "renderer_choice",
        "schema",
    }
    assert evidence["schema"] == "efc-raw-world-evidence-v2"
    assert len(evidence["observations"]) == 10
    assert (
        first.admissibility_receipt["empty_observer_class_count"]
        < 5
    )
    assert first.admissibility_receipt["future_behavior_class_count"] == 5
    commitment_names = {
        name for name, _ in first.stream_commitments
    }
    assert any(
        name.startswith("accepted-mechanics-candidate-")
        for name in commitment_names
    )
    assert {
        "mechanics/transition-action-0",
        "mechanics/transition-action-1",
        "mechanics/transition-action-2",
        "mechanics/observer-values-0",
        "mechanics/observer-values-1",
        "world/state-keys",
        "world/action-keys",
        "world/observer-keys",
        "world/demonstration-order",
        "world/observation-order",
        "world/renderer-choice",
    }.issubset(commitment_names)

    parameters = inspect.signature(generate_independent_world).parameters
    assert "query" not in parameters
    assert "challenge" not in parameters
    assert "prediction" not in parameters
    assert "answer" not in parameters


def test_beacon_or_protocol_change_changes_world_without_reroll_api() -> None:
    baseline = generate_independent_world(**KWARGS)
    changed_beacon = generate_independent_world(
        **{**KWARGS, "beacon_value": "different-consumed-beacon"}
    )
    changed_protocol = generate_independent_world(
        **{**KWARGS, "protocol_root": "34" * 32}
    )
    assert baseline.evidence != changed_beacon.evidence
    assert baseline.evidence != changed_protocol.evidence
    assert baseline.world_seed_commitment != (
        changed_beacon.world_seed_commitment
    )
    assert baseline.world_seed_commitment != (
        changed_protocol.world_seed_commitment
    )
