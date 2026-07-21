#!/usr/bin/env python3
"""Independent seedless audit for the CTAA revision-2 family contract."""

from __future__ import annotations

from collections import Counter
import hashlib
from itertools import product
import json
import math
from typing import Sequence

from pipeline.ctaa_board_v2 import (
    FACTORIAL_BITS,
    INITIAL_STATES,
    CTAAProgramFamilyV2,
    board_contract_counts,
    build_long_families,
    factorial_cells,
    train_closed_pairs,
)


SCHEMA = "r12_ctaa_v2_seedless_independent_audit_v1"
EXPECTED_SPLITS = {
    "train": {
        (0, 0, 0), (0, 1, 1), (0, 1, 2), (1, 0, 1), (1, 2, 0),
        (1, 2, 2), (2, 0, 2), (2, 1, 1), (2, 2, 0),
    },
    "development": {
        (0, 0, 2), (0, 1, 0), (0, 2, 1), (1, 0, 0), (1, 1, 0),
        (1, 1, 2), (2, 0, 1), (2, 2, 1), (2, 2, 2),
    },
    "confirmation": {
        (0, 0, 1), (0, 2, 0), (0, 2, 2), (1, 0, 2), (1, 1, 1),
        (1, 2, 1), (2, 0, 0), (2, 1, 0), (2, 1, 2),
    },
}


class V2AuditFailure(AssertionError):
    pass


def _require(value: bool, message: str) -> None:
    if not value:
        raise V2AuditFailure(message)


def _apply(action: tuple[int, int, int], state: tuple[int, int, int]):
    return (state[action[0]], state[action[1]], state[action[2]])


def _compose(after: tuple[int, int, int], before: tuple[int, int, int]):
    return (before[after[0]], before[after[1]], before[after[2]])


def _compose_events(cards: Sequence[tuple[int, int, int]], events: Sequence[int]):
    result = (0, 1, 2)
    for event in events:
        result = _compose(cards[event], result)
    return result


def _execute(family: CTAAProgramFamilyV2):
    state = family.initial_state
    states = [state]
    halted = False
    for event in family.schedule:
        if event == 4:
            halted = True
        elif not halted:
            state = _apply(family.action_cards[event], state)
        states.append(state)
    return tuple(states)


def _entropy(events: Sequence[int]) -> float:
    counts = Counter(events)
    total = len(events)
    return -sum((count / total) * math.log(count / total) for count in counts.values()) / math.log(len(counts))


def _max_run(events: Sequence[int]) -> int:
    longest = current = 0
    previous = None
    for event in events:
        current = current + 1 if event == previous else 1
        previous = event
        longest = max(longest, current)
    return longest


def audit_long_families(
    families: Sequence[CTAAProgramFamilyV2],
    *,
    partition: str,
    per_class_depth_cell: int,
) -> dict[str, object]:
    expected = len(FACTORIAL_BITS) * 3 * 2 * per_class_depth_cell
    _require(len(families) == expected, "v2 long family count differs")
    _require(
        len({family.canonical_key for family in families}) == len(families),
        "v2 canonical families repeat",
    )
    cells = {
        (cell.semantic_axis, cell.renderer_axis, cell.lexical_axis)
        for cell in factorial_cells(partition)  # type: ignore[arg-type]
    }
    _require(len(cells) == 8, "v2 factorial cells differ")
    strata: Counter[tuple[object, ...]] = Counter()
    query_initial: Counter[tuple[object, ...]] = Counter()
    oracle_checks = 0
    for family in families:
        _require(family.partition == partition, "v2 partition differs")
        _require(set(family.action_cards) <= EXPECTED_SPLITS[family.cell.semantic_axis], "v2 semantic-axis action leak")
        _require(family.initial_state in INITIAL_STATES, "v2 initial symbols are not distinct")
        _require(family.schedule.count(4) == 1, "v2 STOP count differs")
        _require(len(set(family.active)) >= 3, "v2 schedule uses fewer than three cards")
        _require(_max_run(family.active) <= 3, "v2 schedule run shortcut admitted")
        _require(_entropy(family.active) >= 0.75, "v2 schedule entropy differs")
        composite = _compose_events(family.action_cards, family.active)
        expected_rank = 2 if family.program_class == "stable_rank_two" else 1
        _require(len(set(composite)) == expected_rank, "v2 class rank differs")
        _require(_execute(family) == family.execute(), "v2 independent execution differs")
        _require(family.terminal_state == _apply(composite, family.initial_state), "v2 terminal state differs")
        _require(family.answer == family.terminal_state[family.query_position], "v2 answer differs")
        _require(family.map_deletion_depth >= family.depth // 4, "v2 deletion depth differs")
        strata[(family.cell.tag, family.program_class, family.depth)] += 1
        query_initial[(family.cell.tag, family.program_class, family.depth, family.query_position, family.initial_state)] += 1
        oracle_checks += 1
    _require(set(strata.values()) == {per_class_depth_cell}, "v2 class/depth/cell balance differs")
    expected_qi = per_class_depth_cell // 18
    _require(set(query_initial.values()) == {expected_qi}, "v2 query/initial balance differs")
    return {
        "families": len(families),
        "canonical_unique": len({family.canonical_key for family in families}),
        "factorial_cells": len(cells),
        "class_depth_cell_strata": len(strata),
        "query_initial_strata": len(query_initial),
        "independent_oracle_traces": oracle_checks,
    }


def audit_v2(*, dry_seed: int = 611_953, per_class_depth_cell: int = 288) -> dict[str, object]:
    _require(per_class_depth_cell % 288 == 0, "v2 dry count balance differs")
    finite_actions = tuple(product(range(3), repeat=3))
    atomic = 0
    closure = 0
    for action in finite_actions:
        for state in finite_actions:
            _apply(action, state)
            atomic += 1
        for before in finite_actions:
            card = _compose(action, before)
            for state in finite_actions:
                _require(_apply(card, state) == _apply(action, _apply(before, state)), "v2 algebra differs")
                closure += 1
    _require(len(train_closed_pairs()) == 35, "v2 train closure count differs")
    development = build_long_families(dry_seed, "development", per_class_depth_cell=per_class_depth_cell)
    confirmation = build_long_families(dry_seed, "confirmation", per_class_depth_cell=per_class_depth_cell)
    report: dict[str, object] = {
        "schema": SCHEMA,
        "status": "pass",
        "finite_contract": board_contract_counts(),
        "atomic_oracle_checks": atomic,
        "closure_execution_checks": closure,
        "development": audit_long_families(
            development,
            partition="development",
            per_class_depth_cell=per_class_depth_cell,
        ),
        "confirmation": audit_long_families(
            confirmation,
            partition="confirmation",
            per_class_depth_cell=per_class_depth_cell,
        ),
        "production_seed_generated": False,
        "board_artifact_written": False,
        "jobs_launched": False,
    }
    encoded = json.dumps(report, sort_keys=True, separators=(",", ":")).encode()
    report["report_sha256"] = hashlib.sha256(encoded).hexdigest()
    return report


def main() -> None:
    print(json.dumps(audit_v2(), sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
