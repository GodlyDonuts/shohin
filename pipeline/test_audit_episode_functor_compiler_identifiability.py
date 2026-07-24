from __future__ import annotations

from pipeline.audit_episode_functor_compiler_identifiability import (
    EXPECTED_WORLDS_SHA256,
    FrozenWorldMachine,
    _machine_bit_receipt,
    audit_bundle,
    parse_query,
    parse_world,
)
from pipeline.episode_workspace_custody import (
    DEFAULT_CUSTODY_BUNDLE,
    read_jsonl_verified,
)


def test_frozen_world_compiles_to_complete_categorical_machine() -> None:
    row = read_jsonl_verified(
        DEFAULT_CUSTODY_BUNDLE / "development_worlds.jsonl",
        EXPECTED_WORLDS_SHA256,
    )[0]
    machine = parse_world(row)
    assert isinstance(machine, FrozenWorldMachine)
    assert len(machine.state_tokens) == 8
    assert len(machine.action_tokens) == 3
    assert len(machine.transitions) == 3
    assert all(len(row) == 8 for row in machine.transitions)


def test_query_parser_uses_only_late_query_and_retained_keys() -> None:
    world_rows = read_jsonl_verified(
        DEFAULT_CUSTODY_BUNDLE / "development_worlds.jsonl",
        EXPECTED_WORLDS_SHA256,
    )
    machine = parse_world(world_rows[0])
    query = {
        "schema": "episode_workspace_development_query_v1",
        "packet_sha256": "a" * 64,
        "world_id": machine.world_id,
        "query_tokens": [
            5,
            machine.state_tokens[3],
            machine.action_tokens[2],
            6,
            machine.action_tokens[0],
            7,
            8,
        ],
    }
    assert parse_query(query, machine) == (3, (2, 0))


def test_resource_gate_makes_full_depth_six_cache_expensive() -> None:
    receipt = _machine_bit_receipt(6)
    assert receipt["min_depth"] == 1
    assert receipt["query_count_all_starts_through_depth"] == 8_736
    assert receipt["full_answer_cache_bits"] == 26_208
    assert receipt["conservative_machine_semantic_bits"] == 276
    assert receipt["cache_to_machine_bit_ratio"] > 94.9


def test_frozen_bundle_audit_narrows_the_two_answer_no_go() -> None:
    report = audit_bundle(DEFAULT_CUSTODY_BUNDLE, max_depth=6)
    assert report["realized_machine_exact"] == {
        "correct": 384,
        "total": 384,
        "rate": 1.0,
    }
    assert report["world_transition_coverage"]["complete_worlds"] == 192
    assert report["causal_quotient"]["classes_per_world"] == 8
    assert report["exhaustive_execution"]["queries_per_world"] == 8_736
    assert report["exhaustive_execution"]["total_queries"] == 1_677_312
    assert (
        report["finite_query_theorem"][
            "two_realized_answers_are_source_derivable"
        ]
        == "not established"
    )
    assert report["finite_query_theorem"]["old_board_decisive_no_go_as_stated"] is False
    assert report["reasoning_promotion_authorized"] is False
    assert report["continuation_pretraining_authorized"] is False
    assert report["pretraining_started"] is False
