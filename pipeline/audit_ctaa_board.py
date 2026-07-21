#!/usr/bin/env python3
"""Independent, read-only audit of the seedless CTAA board mechanics.

The audit imports the candidate mechanics as a subject under test but carries
its own fixed semantic split, copy-map oracle, renderer equations, and family
fixtures. It does not choose a production seed, persist rows, or launch work.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass, replace
import hashlib
from itertools import combinations, product
import json
from typing import Callable, Mapping, Sequence

try:
    from pipeline import generate_ctaa_board as board
except ModuleNotFoundError:  # Direct execution from pipeline/.
    import generate_ctaa_board as board  # type: ignore[no-redef]


SCHEMA = "r12_ctaa_seedless_board_cpu_audit_v1"
SPLITS = ("train", "development", "confirmation")
WIDTH = 3
ACTION_COUNT = 4
MAX_STEPS = 41
STOP_ID = 4
AUDIT_RENDER_SENTINEL = 0
CopyMap = tuple[int, int, int]
State = tuple[int, int, int]

EXPECTED_SEMANTIC_SPLITS: dict[str, tuple[CopyMap, ...]] = {
    "train": (
        (0, 0, 0),
        (0, 1, 1),
        (0, 1, 2),
        (1, 0, 1),
        (1, 2, 0),
        (1, 2, 2),
        (2, 0, 2),
        (2, 1, 1),
        (2, 2, 0),
    ),
    "development": (
        (0, 0, 2),
        (0, 1, 0),
        (0, 2, 1),
        (1, 0, 0),
        (1, 1, 0),
        (1, 1, 2),
        (2, 0, 1),
        (2, 2, 1),
        (2, 2, 2),
    ),
    "confirmation": (
        (0, 0, 1),
        (0, 2, 0),
        (0, 2, 2),
        (1, 0, 2),
        (1, 1, 1),
        (1, 2, 1),
        (2, 0, 0),
        (2, 1, 0),
        (2, 1, 2),
    ),
}
EXPECTED_TRAINING_KEYS = {
    "family_id",
    "view",
    "program_source",
    "query_source",
    "action_cards",
    "initial_state",
    "schedule",
    "query_position",
}


class AuditFailure(AssertionError):
    """Raised when the candidate board mechanics differ from the frozen contract."""


@dataclass(frozen=True)
class AuditSubject:
    """Injectable candidate interface used by the audit and mutation tests."""

    width: int
    action_count: int
    max_steps: int
    stop_id: int
    semantic_splits: Callable[[], Mapping[str, Sequence[CopyMap]]]
    renderers: Mapping[str, Sequence[int]]
    family_type: type
    apply_copy: Callable[[CopyMap, State], State]
    compose_maps: Callable[[CopyMap, CopyMap], CopyMap]
    execute_family: Callable[[object, State], Sequence[State]]
    causal_depth: Callable[[object], int]
    training_record: Callable[[object], Mapping[str, object]]
    render_row: Callable[..., object]


def candidate_subject() -> AuditSubject:
    """Snapshot the current candidate mechanics without mutating their module."""

    return AuditSubject(
        width=board.WIDTH,
        action_count=board.ACTION_COUNT,
        max_steps=board.MAX_STEPS,
        stop_id=board.STOP_ID,
        semantic_splits=board.semantic_splits,
        renderers=board.RENDERERS,
        family_type=board.CTAAFamily,
        apply_copy=board.apply_copy,
        compose_maps=board.compose_maps,
        execute_family=lambda family, initial: family.execute(initial),
        causal_depth=lambda family: family.causal_depth,
        training_record=lambda row: row.training_record(),
        render_row=board.render_row,
    )


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditFailure(message)


def _oracle_apply(action: CopyMap, state: State) -> State:
    _require(len(action) == WIDTH and len(state) == WIDTH, "oracle geometry differs")
    _require(
        all(0 <= value < WIDTH for value in (*action, *state)),
        "oracle domain differs",
    )
    return (state[action[0]], state[action[1]], state[action[2]])


def _oracle_compose(after: CopyMap, before: CopyMap) -> CopyMap:
    return (before[after[0]], before[after[1]], before[after[2]])


def _oracle_compose_events(cards: Sequence[CopyMap], events: Sequence[int]) -> CopyMap:
    composite: CopyMap = (0, 1, 2)
    for event in events:
        _require(0 <= event < len(cards), "oracle active event leaves card domain")
        composite = _oracle_compose(cards[event], composite)
    return composite


def _oracle_execute(
    cards: Sequence[CopyMap], schedule: Sequence[int], initial: State
) -> tuple[State, ...]:
    state = initial
    states = [state]
    halted = False
    for event in schedule:
        if event == STOP_ID:
            halted = True
        elif not halted:
            _require(0 <= event < len(cards), "oracle event leaves card domain")
            state = _oracle_apply(cards[event], state)
        states.append(state)
    return tuple(states)


def _normalize_semantics(
    values: Mapping[str, Sequence[CopyMap]],
) -> dict[str, tuple[CopyMap, ...]]:
    _require(set(values) == set(SPLITS), "semantic split names differ")
    return {
        split: tuple(tuple(action) for action in values[split])  # type: ignore[misc]
        for split in SPLITS
    }


def _audit_fixed_semantics(subject: AuditSubject) -> dict[str, object]:
    _require(subject.width == WIDTH, "board width differs")
    _require(subject.action_count == ACTION_COUNT, "action-card count differs")
    _require(subject.max_steps == MAX_STEPS, "schedule width differs")
    _require(subject.stop_id == STOP_ID, "STOP identity differs")
    observed = _normalize_semantics(subject.semantic_splits())
    _require(observed == EXPECTED_SEMANTIC_SPLITS, "fixed semantic split differs")
    return _audit_coordinate_balance(observed)


def _audit_coordinate_balance(
    semantics: Mapping[str, Sequence[CopyMap]],
) -> dict[str, object]:
    union: set[CopyMap] = set()
    rank_counts: dict[str, dict[str, int]] = {}
    coordinate_checks = 0
    for split in SPLITS:
        actions = tuple(semantics[split])
        _require(len(actions) == 9, f"{split} semantic count differs")
        _require(len(set(actions)) == 9, f"{split} semantic cards repeat")
        _require(union.isdisjoint(actions), f"{split} semantic custody overlaps")
        union.update(actions)
        ranks = Counter(len(set(action)) for action in actions)
        _require(ranks == {1: 1, 2: 6, 3: 2}, f"{split} rank balance differs")
        rank_counts[split] = {str(rank): ranks[rank] for rank in (1, 2, 3)}
        for coordinate in range(WIDTH):
            counts = Counter(action[coordinate] for action in actions)
            _require(
                counts == {0: 3, 1: 3, 2: 3},
                f"{split} coordinate {coordinate} balance differs",
            )
            coordinate_checks += 1
    all_maps = set(product(range(WIDTH), repeat=WIDTH))
    _require(union == all_maps, "semantic split is not exhaustive")
    return {
        "split_sizes": {split: len(semantics[split]) for split in SPLITS},
        "rank_counts": rank_counts,
        "coordinate_balance_checks": coordinate_checks,
        "exhaustive_maps": len(union),
    }


def _audit_algebra(subject: AuditSubject) -> dict[str, int]:
    actions = tuple(product(range(WIDTH), repeat=WIDTH))
    states = tuple(product(range(WIDTH), repeat=WIDTH))
    atomic_checks = 0
    composition_checks = 0
    closure_execution_checks = 0
    for action in actions:
        for state in states:
            observed = subject.apply_copy(action, state)
            expected = _oracle_apply(action, state)
            _require(
                observed == expected, "copy executor differs from independent oracle"
            )
            atomic_checks += 1
    for before in actions:
        for after in actions:
            observed_card = subject.compose_maps(after, before)
            expected_card = _oracle_compose(after, before)
            _require(
                observed_card == expected_card,
                "copy composition differs from independent oracle",
            )
            composition_checks += 1
            for state in states:
                sequential = _oracle_apply(after, _oracle_apply(before, state))
                observed = subject.apply_copy(observed_card, state)
                _require(observed == sequential, "composed execution differs")
                closure_execution_checks += 1
    return {
        "atomic_execution_checks": atomic_checks,
        "composition_card_checks": composition_checks,
        "closure_execution_checks": closure_execution_checks,
    }


def _make_schedule(active: Sequence[int]) -> tuple[int, ...]:
    _require(len(active) < MAX_STEPS, "audit active schedule is too long")
    suffix_length = MAX_STEPS - len(active) - 1
    suffix = tuple((index + 1) % ACTION_COUNT for index in range(suffix_length))
    return (*active, STOP_ID, *suffix)


def _make_family(
    subject: AuditSubject,
    *,
    split: str,
    family_id: str,
    cards: Sequence[CopyMap],
    active: Sequence[int],
    query_position: int,
    program_class: str,
) -> object:
    return subject.family_type(
        split=split,
        family_id=family_id,
        action_cards=tuple(cards),
        initial_state=(2, 0, 1),
        schedule=_make_schedule(active),
        query_position=query_position,
        program_class=program_class,
    )


def _audit_stop_and_suffix(subject: AuditSubject) -> dict[str, int]:
    cards: tuple[CopyMap, ...] = (
        (1, 2, 0),
        (0, 0, 1),
        (0, 1, 1),
        (0, 0, 0),
    )
    states = tuple(product(range(WIDTH), repeat=WIDTH))
    trace_checks = 0
    post_stop_event_checks = 0
    for halt_at in range(MAX_STEPS):
        active = tuple(index % ACTION_COUNT for index in range(halt_at))
        family = _make_family(
            subject,
            split="development",
            family_id=f"AUDIT_STOP_{halt_at:02d}",
            cards=cards,
            active=active,
            query_position=0,
            program_class="absorbing",
        )
        schedule = tuple(family.schedule)
        _require(schedule.count(STOP_ID) == 1, "audit family STOP count differs")
        _require(schedule.index(STOP_ID) == halt_at, "audit STOP boundary differs")
        for initial in states:
            observed = tuple(subject.execute_family(family, initial))
            expected = _oracle_execute(cards, schedule, initial)
            _require(observed == expected, "STOP/suffix execution differs")
            committed = expected[halt_at]
            _require(
                all(state == committed for state in observed[halt_at:]),
                "post-STOP suffix changes committed state",
            )
            trace_checks += 1
            post_stop_event_checks += MAX_STEPS - halt_at
    return {
        "halt_boundaries": MAX_STEPS,
        "halt_trace_checks": trace_checks,
        "post_stop_event_checks": post_stop_event_checks,
    }


def _causal_depth(cards: Sequence[CopyMap], active: Sequence[int]) -> int:
    full = _oracle_compose_events(cards, active)
    return sum(
        _oracle_compose_events(cards, (*active[:index], *active[index + 1 :])) != full
        for index in range(len(active))
    )


def _class_fixtures(depth: int) -> dict[str, tuple[int, ...]]:
    return {
        "persistent": (0,) * depth,
        "mixed_copy": (1,) + (0,) * (depth - 1),
        "absorbing": (0,) * (depth - 2) + (3, 2),
    }


def _audit_causal_depth_classes(subject: AuditSubject) -> dict[str, object]:
    cards: tuple[CopyMap, ...] = (
        (1, 2, 0),
        (0, 0, 1),
        (0, 1, 1),
        (0, 0, 0),
    )
    results: dict[str, dict[str, dict[str, int]]] = {}
    checks = 0
    for depth in (16, 32):
        depth_results: dict[str, dict[str, int]] = {}
        for program_class, active in _class_fixtures(depth).items():
            family = _make_family(
                subject,
                split="development",
                family_id=f"AUDIT_{program_class}_{depth}",
                cards=cards,
                active=active,
                query_position=1,
                program_class=program_class,
            )
            expected_composite = _oracle_compose_events(cards, active)
            expected_causal = _causal_depth(cards, active)
            observed_composite = tuple(family.composite)
            observed_causal = subject.causal_depth(family)
            _require(
                observed_composite == expected_composite, "family composite differs"
            )
            _require(observed_causal == expected_causal, "family causal depth differs")
            rank = len(set(expected_composite))
            if program_class == "persistent":
                _require(
                    rank == 3 and observed_causal == depth, "persistent class differs"
                )
            elif program_class == "mixed_copy":
                _require(
                    rank == 2 and observed_causal == depth, "mixed-copy class differs"
                )
            else:
                _require(
                    rank == 1 and observed_causal < depth, "absorbing class differs"
                )
            depth_results[program_class] = {
                "raw_depth": depth,
                "causal_depth": observed_causal,
                "terminal_rank": rank,
            }
            checks += 1
        results[str(depth)] = depth_results
    return {"class_checks": checks, "fixtures": results}


def _renderer_bits(renderer: int) -> tuple[int, ...]:
    return tuple((renderer >> index) & 1 for index in range(6))


def _renderer_syndrome(renderer: int) -> tuple[int, int]:
    bits = _renderer_bits(renderer)
    return (
        bits[0] ^ bits[1] ^ bits[2] ^ bits[3],
        bits[2] ^ bits[3] ^ bits[4] ^ bits[5],
    )


def _expected_renderer_cosets() -> dict[tuple[int, int], tuple[int, ...]]:
    return {
        syndrome: tuple(
            renderer
            for renderer in range(64)
            if _renderer_syndrome(renderer) == syndrome
        )
        for syndrome in product(range(2), repeat=2)
    }


def _audit_renderer_cosets(subject: AuditSubject) -> dict[str, object]:
    expected = _expected_renderer_cosets()
    split_syndromes = {
        "train": (0, 0),
        "development": (0, 1),
        "confirmation": (1, 0),
    }
    observed = {
        split: tuple(subject.renderers[split])
        for split in SPLITS
        if split in subject.renderers
    }
    _require(set(observed) == set(SPLITS), "renderer split names differ")
    for split, syndrome in split_syndromes.items():
        _require(
            observed[split] == expected[syndrome],
            f"{split} renderer coset differs",
        )
    used = set().union(*(set(observed[split]) for split in SPLITS))
    _require(
        sum(len(observed[split]) for split in SPLITS) == len(used),
        "renderer cosets overlap",
    )
    reserved = set(expected[(1, 1)])
    _require(used.isdisjoint(reserved), "reserved renderer coset leaked")
    _require(used | reserved == set(range(64)), "renderer cosets are not exhaustive")

    marginal_checks = 0
    for syndrome, renderers in expected.items():
        rows = tuple(_renderer_bits(renderer) for renderer in renderers)
        _require(len(rows) == 16, f"renderer coset {syndrome} size differs")
        for order in (1, 2, 3):
            expected_count = len(rows) // (2**order)
            for coordinates in combinations(range(6), order):
                counts = Counter(
                    tuple(bits[index] for index in coordinates) for bits in rows
                )
                _require(
                    set(counts.values()) == {expected_count}
                    and len(counts) == 2**order,
                    f"renderer coset {syndrome} marginal differs",
                )
                marginal_checks += 1
    return {
        "split_sizes": {split: len(observed[split]) for split in SPLITS},
        "reserved_size": len(reserved),
        "low_order_marginal_checks": marginal_checks,
    }


def _render_fixture_family(
    subject: AuditSubject, split: str, query_position: int
) -> object:
    cards = tuple(EXPECTED_SEMANTIC_SPLITS[split][:ACTION_COUNT])
    return _make_family(
        subject,
        split=split,
        family_id=f"AUDIT_RENDER_{split.upper()}",
        cards=cards,
        active=(2,),
        query_position=query_position,
        program_class="persistent",
    )


def _audit_training_custody_and_query(subject: AuditSubject) -> dict[str, int]:
    train_family = _render_fixture_family(subject, "train", 0)
    training_rows = 0
    for view, renderer in enumerate(subject.renderers["train"]):
        row = subject.render_row(
            AUDIT_RENDER_SENTINEL,
            train_family,
            view,
            force_renderer=renderer,
        )
        _require(row.terminal_state is None, "training terminal state is present")
        _require(row.answer is None, "training answer is present")
        record = dict(subject.training_record(row))
        _require(set(record) == EXPECTED_TRAINING_KEYS, "training record fields differ")
        _require("terminal_state" not in record, "training terminal state leaked")
        _require("answer" not in record, "training answer leaked")
        _require(row.query_source, "query source is empty")
        _require(
            row.query_source not in row.program_source,
            "query source leaked into program",
        )
        _require(
            "READ THE" not in row.program_source, "query grammar leaked into program"
        )
        _require(
            "REPORT VALUE" not in row.program_source,
            "query grammar leaked into program",
        )
        training_rows += 1

    development_family = _render_fixture_family(subject, "development", 0)
    scored = subject.render_row(
        AUDIT_RENDER_SENTINEL,
        development_family,
        0,
        force_renderer=subject.renderers["development"][0],
    )
    _require(scored.terminal_state is not None, "scored terminal state is absent")
    _require(scored.answer is not None, "scored answer is absent")
    try:
        subject.training_record(scored)
    except ValueError:
        pass
    else:
        raise AuditFailure("scored row was admitted as a training record")

    query_zero = development_family
    query_two = _render_fixture_family(subject, "development", 2)
    first = subject.render_row(
        AUDIT_RENDER_SENTINEL,
        query_zero,
        0,
        force_renderer=subject.renderers["development"][0],
    )
    second = subject.render_row(
        AUDIT_RENDER_SENTINEL,
        query_two,
        0,
        force_renderer=subject.renderers["development"][0],
    )
    _require(
        first.program_source == second.program_source, "query changes program source"
    )
    _require(
        first.query_source != second.query_source, "query positions do not separate"
    )
    _require(
        first.query_source not in first.program_source,
        "first query leaked into program",
    )
    _require(
        second.query_source not in second.program_source,
        "second query leaked into program",
    )
    return {
        "training_renderer_rows": training_rows,
        "scored_training_rejections": 1,
        "late_query_pair_checks": 1,
    }


def _positive_audit(subject: AuditSubject) -> dict[str, object]:
    return {
        "semantic_split": _audit_fixed_semantics(subject),
        "copy_algebra": _audit_algebra(subject),
        "stop_suffix": _audit_stop_and_suffix(subject),
        "causal_depth": _audit_causal_depth_classes(subject),
        "renderer_cosets": _audit_renderer_cosets(subject),
        "training_and_query": _audit_training_custody_and_query(subject),
    }


def _expect_mutation_kill(name: str, check: Callable[[], object]) -> dict[str, object]:
    try:
        check()
    except AuditFailure as error:
        return {"killed": True, "reason": str(error)}
    raise AuditFailure(f"mutation survived: {name}")


def run_mutation_kills(subject: AuditSubject | None = None) -> dict[str, object]:
    """Exercise independent mutants without changing the candidate module."""

    subject = candidate_subject() if subject is None else subject

    swapped = {
        split: list(values) for split, values in EXPECTED_SEMANTIC_SPLITS.items()
    }
    swapped["train"][0], swapped["development"][0] = (
        swapped["development"][0],
        swapped["train"][0],
    )
    imbalanced = {
        split: list(values) for split, values in EXPECTED_SEMANTIC_SPLITS.items()
    }
    imbalanced["train"][1] = (0, 0, 1)

    renderer_leak = {
        split: tuple(values) for split, values in subject.renderers.items()
    }
    renderer_leak["development"] = (
        subject.renderers["train"][0],
        *subject.renderers["development"][1:],
    )

    def execute_after_stop(family: object, initial: State) -> tuple[State, ...]:
        state = initial
        states = [state]
        for event in family.schedule:
            if event != STOP_ID:
                state = _oracle_apply(family.action_cards[event], state)
            states.append(state)
        return tuple(states)

    def leaking_training_record(row: object) -> Mapping[str, object]:
        record = dict(row.training_record())
        record["terminal_state"] = (0, 0, 0)
        record["answer"] = 0
        return record

    def leaking_render(*args: object, **kwargs: object) -> object:
        row = subject.render_row(*args, **kwargs)
        return replace(row, program_source=row.program_source + row.query_source)

    mutations = {
        "fixed_semantic_card_swap": lambda: _audit_fixed_semantics(
            replace(subject, semantic_splits=lambda: swapped)
        ),
        "coordinate_imbalance": lambda: _audit_coordinate_balance(imbalanced),
        "identity_copy_executor": lambda: _audit_algebra(
            replace(subject, apply_copy=lambda _action, state: state)
        ),
        "reversed_composition": lambda: _audit_algebra(
            replace(
                subject,
                compose_maps=lambda after, before: _oracle_compose(before, after),
            )
        ),
        "nonabsorbing_stop": lambda: _audit_stop_and_suffix(
            replace(subject, execute_family=execute_after_stop)
        ),
        "raw_depth_substitution": lambda: _audit_causal_depth_classes(
            replace(subject, causal_depth=lambda family: family.depth)
        ),
        "renderer_coset_leak": lambda: _audit_renderer_cosets(
            replace(subject, renderers=renderer_leak)
        ),
        "training_outcome_leak": lambda: _audit_training_custody_and_query(
            replace(subject, training_record=leaking_training_record)
        ),
        "query_in_program": lambda: _audit_training_custody_and_query(
            replace(subject, render_row=leaking_render)
        ),
    }
    return {
        name: _expect_mutation_kill(name, check) for name, check in mutations.items()
    }


def audit_board(
    subject: AuditSubject | None = None,
    *,
    include_mutation_kills: bool = True,
) -> dict[str, object]:
    """Run the independent mechanics audit and return an in-memory receipt."""

    subject = candidate_subject() if subject is None else subject
    report: dict[str, object] = {
        "schema": SCHEMA,
        "status": "pass",
        "scope": {
            "production_seed_generated": False,
            "board_artifact_written": False,
            "jobs_launched": False,
            "audit_render_sentinel": AUDIT_RENDER_SENTINEL,
            "audit_render_sentinel_is_production_seed": False,
        },
        "checks": _positive_audit(subject),
    }
    if include_mutation_kills:
        report["mutation_kills"] = run_mutation_kills(subject)
    digest = hashlib.sha256(canonical_json(report).encode()).hexdigest()
    report["report_sha256"] = digest
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretty", action="store_true", help="indent JSON output")
    args = parser.parse_args()
    report = audit_board()
    print(json.dumps(report, sort_keys=True, indent=2 if args.pretty else None))


if __name__ == "__main__":
    main()
