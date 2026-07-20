from __future__ import annotations

import torch

from assess_er_relation_tensor import recompute_interventions, recompute_invariance
from build_er_relation_tensor_board import (
    CONFIRMATION_SPLIT,
    DEVELOPMENT_SPLIT,
    TRAIN_SPLIT,
    build_board,
)
from er_relation_tensor_training import _targets, parse_row


def _perfect_packet() -> dict[str, torch.Tensor]:
    splits, _ = build_board(
        seed=188_221,
        families={TRAIN_SPLIT: 12, DEVELOPMENT_SPLIT: 12, CONFIRMATION_SPLIT: 12},
    )
    rows = [parse_row(row, DEVELOPMENT_SPLIT) for row in splits[DEVELOPMENT_SPLIT]][:32]
    target = _targets(rows, torch.device("cpu"))
    packet = {
        "cardinality": target["cardinality"],
        "initial": target["initial"],
        "relations": target["relation"],
        "rule_active": target["rule_active"].long(),
        "events": target["events"],
        "halt": target["halt"],
        "query": target["query"],
    }
    return {name: value.repeat((64,) + (1,) * (value.ndim - 1)) for name, value in packet.items()}


def test_independent_list_interventions_match_a_perfect_packet() -> None:
    packet = _perfect_packet()
    result = recompute_interventions(packet, packet)
    for value in result.values():
        assert value["eligible"] == 2_048
        assert value["exact_on_eligible"] == 2_048
        assert value["sensitive"] > 0
        assert value["changed_on_sensitive"] == value["sensitive"]


def test_independent_invariance_recomputation_is_exact() -> None:
    packet = _perfect_packet()
    packet["state"] = packet["initial"].clone()
    packet["answer"] = packet["query"].clone()
    packet["valid"] = torch.ones(2_048, dtype=torch.long)
    raw: dict[str, object] = {}
    for key, value in packet.items():
        raw[f"invariance_canonical_{key}"] = value.to(torch.int16)
    for name in (
        "rule_storage_reindex",
        "physical_record_reindex",
        "witness_alpha_rename",
        "opcode_alpha_rename",
        "post_halt_suffix",
    ):
        for key, value in packet.items():
            raw[f"invariance_{name}_{key}"] = value.to(torch.int16)
    result = recompute_invariance(raw)
    assert set(result) == {
        "rule_storage_reindex",
        "physical_record_reindex",
        "witness_alpha_rename",
        "opcode_alpha_rename",
        "post_halt_suffix",
        "source_poison_after_seal",
    }
    assert all(value == {"exact": 2_048, "rows": 2_048} for value in result.values())
