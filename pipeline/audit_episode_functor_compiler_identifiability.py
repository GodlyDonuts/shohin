#!/usr/bin/env python3
"""Audit EPISODE query-cache and categorical-machine identifiability."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
from itertools import product
import json
import math
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from pipeline.episode_workspace_custody import (
    DEFAULT_CUSTODY_BUNDLE,
    file_sha256,
    read_jsonl_verified,
    write_json_fsync,
)


WORLD_SCHEMA = "episode_workspace_development_world_v1"
QUERY_SCHEMA = "episode_workspace_development_query_v1"
ASSESSOR_SCHEMA = "episode_workspace_assessor_row_v1"
REPORT_SCHEMA = "episode_functor_compiler_identifiability_audit_v1"
EXPECTED_WORLDS_SHA256 = (
    "e448d0585427c7314b3f21c567b231b6e8ebe08dbd4acf4e103151a3af56851e"
)
EXPECTED_QUERIES_SHA256 = (
    "456b9bcef3762a9922323a5928078a8998f51c8d6c9d358e5f6adae69aa1bfba"
)
EXPECTED_ASSESSOR_SHA256 = (
    "0bf99717d677b5807c47873cf365fc408230e2eef16218e9f768bb9489294235"
)

BOS = 1
DEMO = 2
YIELDS = 3
SEP = 4
QUERY = 5
THEN = 6
ANSWER = 7
EOS = 8
STATE_COUNT = 8
ACTION_COUNT = 3
VOCAB_SIZE = 32_768


class FunctorCompilerAuditError(ValueError):
    """The frozen board does not satisfy one exact audit invariant."""


@dataclass(frozen=True)
class FrozenWorldMachine:
    world_id: str
    state_tokens: tuple[int, ...]
    action_tokens: tuple[int, ...]
    transitions: tuple[tuple[int, ...], ...]

    def execute(self, start: int, word: Sequence[int]) -> int:
        if start not in range(STATE_COUNT):
            raise FunctorCompilerAuditError("start state is outside the machine")
        state = start
        for action in word:
            if action not in range(ACTION_COUNT):
                raise FunctorCompilerAuditError("action is outside the machine")
            state = self.transitions[action][state]
        return state


def _strict_integer_list(value: object, label: str) -> list[int]:
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, int) or isinstance(item, bool) for item in value)
    ):
        raise FunctorCompilerAuditError(f"{label} is not an integer list")
    return value


def parse_world(row: Mapping[str, object]) -> FrozenWorldMachine:
    """Compile one world-only row into an exact anonymous transition machine."""

    if set(row) != {"schema", "world_id", "world_tokens"}:
        raise FunctorCompilerAuditError("world row fields differ")
    world_id = row["world_id"]
    if (
        row["schema"] != WORLD_SCHEMA
        or not isinstance(world_id, str)
        or len(world_id) != 64
    ):
        raise FunctorCompilerAuditError("world row identity is invalid")
    tokens = _strict_integer_list(row["world_tokens"], "world tokens")
    if tokens[0] != BOS or len(tokens) != 1 + STATE_COUNT * ACTION_COUNT * 6:
        raise FunctorCompilerAuditError("world token grammar or length differs")

    raw_transitions: dict[tuple[int, int], int] = {}
    state_tokens: set[int] = set()
    action_tokens: set[int] = set()
    for cursor in range(1, len(tokens), 6):
        marker, source, action, yields, target, separator = tokens[
            cursor : cursor + 6
        ]
        if (marker, yields, separator) != (DEMO, YIELDS, SEP):
            raise FunctorCompilerAuditError("world demonstration markers differ")
        key = (source, action)
        if key in raw_transitions:
            raise FunctorCompilerAuditError("world repeats a state-action pair")
        raw_transitions[key] = target
        state_tokens.update((source, target))
        action_tokens.add(action)
    if len(state_tokens) != STATE_COUNT or len(action_tokens) != ACTION_COUNT:
        raise FunctorCompilerAuditError("world state/action cardinality differs")

    ordered_states = tuple(sorted(state_tokens))
    ordered_actions = tuple(sorted(action_tokens))
    state_index = {token: index for index, token in enumerate(ordered_states)}
    transitions = tuple(
        tuple(
            state_index[raw_transitions[(state_token, action_token)]]
            for state_token in ordered_states
        )
        for action_token in ordered_actions
    )
    return FrozenWorldMachine(
        world_id=world_id,
        state_tokens=ordered_states,
        action_tokens=ordered_actions,
        transitions=transitions,
    )


def parse_query(
    row: Mapping[str, object],
    machine: FrozenWorldMachine,
) -> tuple[int, tuple[int, ...]]:
    """Map one late raw query into anonymous machine coordinates."""

    if set(row) != {"schema", "packet_sha256", "world_id", "query_tokens"}:
        raise FunctorCompilerAuditError("query row fields differ")
    if (
        row["schema"] != QUERY_SCHEMA
        or row["world_id"] != machine.world_id
        or not isinstance(row["packet_sha256"], str)
        or len(row["packet_sha256"]) != 64
    ):
        raise FunctorCompilerAuditError("query row identity is invalid")
    tokens = _strict_integer_list(row["query_tokens"], "query tokens")
    if len(tokens) < 5 or tokens[0] != QUERY or tokens[-2:] != [ANSWER, EOS]:
        raise FunctorCompilerAuditError("query grammar differs")
    try:
        start = machine.state_tokens.index(tokens[1])
    except ValueError as exc:
        raise FunctorCompilerAuditError("query start is absent from the machine") from exc
    actions: list[int] = []
    expect_action = True
    for token in tokens[2:-2]:
        if expect_action:
            try:
                actions.append(machine.action_tokens.index(token))
            except ValueError as exc:
                raise FunctorCompilerAuditError(
                    "query action is absent from the machine"
                ) from exc
            expect_action = False
        elif token == THEN:
            expect_action = True
        else:
            raise FunctorCompilerAuditError("query action separators differ")
    if expect_action or not actions:
        raise FunctorCompilerAuditError("query action word is incomplete")
    return start, tuple(actions)


def _all_words(
    max_depth: int,
    *,
    min_depth: int = 1,
) -> Iterable[tuple[int, ...]]:
    if min_depth < 1 or max_depth < min_depth:
        raise FunctorCompilerAuditError("current EPISODE requires a nonempty word")
    for depth in range(min_depth, max_depth + 1):
        yield from product(range(ACTION_COUNT), repeat=depth)


def _machine_bit_receipt(
    max_depth: int,
    *,
    min_depth: int = 1,
) -> dict[str, int | float]:
    state_bits = math.ceil(math.log2(STATE_COUNT))
    action_bits = math.ceil(math.log2(ACTION_COUNT))
    token_bits = math.ceil(math.log2(VOCAB_SIZE))
    words = sum(
        ACTION_COUNT**depth for depth in range(min_depth, max_depth + 1)
    )
    queries = STATE_COUNT * words
    answer_cache_bits = queries * state_bits
    transition_bits = ACTION_COUNT * STATE_COUNT * state_bits
    retained_key_bits = (STATE_COUNT + ACTION_COUNT) * token_bits
    active_mask_bits = STATE_COUNT + ACTION_COUNT + 1
    observer_bits = STATE_COUNT * state_bits
    initial_state_bits = state_bits
    machine_bits = (
        transition_bits
        + retained_key_bits
        + active_mask_bits
        + observer_bits
        + initial_state_bits
    )
    return {
        "max_depth": max_depth,
        "min_depth": min_depth,
        "state_index_bits": state_bits,
        "action_index_bits": action_bits,
        "retained_token_bits": token_bits,
        "word_count_through_depth": words,
        "query_count_all_starts_through_depth": queries,
        "full_answer_cache_bits": answer_cache_bits,
        "transition_table_bits": transition_bits,
        "retained_state_action_key_bits": retained_key_bits,
        "active_mask_bits": active_mask_bits,
        "observer_map_bits": observer_bits,
        "initial_state_bits": initial_state_bits,
        "conservative_machine_semantic_bits": machine_bits,
        "cache_to_machine_bit_ratio": answer_cache_bits / machine_bits,
    }


def audit_bundle(
    custody: Path,
    *,
    max_depth: int = 6,
) -> dict[str, object]:
    """Run the exact old-board cache and machine-capacity audit."""

    if max_depth < 1 or max_depth > 10:
        raise FunctorCompilerAuditError("max depth must be in [1, 10]")
    world_rows = read_jsonl_verified(
        custody / "development_worlds.jsonl",
        EXPECTED_WORLDS_SHA256,
    )
    query_rows = read_jsonl_verified(
        custody / "development_queries.jsonl",
        EXPECTED_QUERIES_SHA256,
    )
    assessor_rows = read_jsonl_verified(
        custody / "development_assessor.jsonl",
        EXPECTED_ASSESSOR_SHA256,
    )
    machines = {machine.world_id: machine for machine in map(parse_world, world_rows)}
    if len(machines) != 192:
        raise FunctorCompilerAuditError("world count differs")
    assessor_by_packet = {}
    for row in assessor_rows:
        if (
            set(row)
            != {
                "schema",
                "packet_sha256",
                "target_token",
                "state_tokens",
                "cluster_id",
                "cluster_index",
                "query_variant",
                "binding_shift",
                "query_depth",
                "world_id",
            }
            or row["schema"] != ASSESSOR_SCHEMA
            or row["packet_sha256"] in assessor_by_packet
        ):
            raise FunctorCompilerAuditError("assessor row fields differ")
        assessor_by_packet[row["packet_sha256"]] = row

    queries_by_world: dict[str, list[tuple[int, tuple[int, ...]]]] = defaultdict(list)
    realized_exact = 0
    depths: Counter[int] = Counter()
    for row in query_rows:
        world_id = row.get("world_id")
        machine = machines.get(str(world_id))
        if machine is None:
            raise FunctorCompilerAuditError("query references an unknown world")
        start, word = parse_query(row, machine)
        assessor = assessor_by_packet.get(row["packet_sha256"])
        if assessor is None or assessor["world_id"] != world_id:
            raise FunctorCompilerAuditError("query/assessor join differs")
        predicted = machine.execute(start, word)
        target_token = assessor["target_token"]
        if machine.state_tokens[predicted] == target_token:
            realized_exact += 1
        queries_by_world[machine.world_id].append((start, word))
        depths[len(word)] += 1

    if realized_exact != len(query_rows) or set(queries_by_world) != set(machines):
        raise FunctorCompilerAuditError("world machine fails a realized query")
    for queries in queries_by_world.values():
        if len(queries) != 2:
            raise FunctorCompilerAuditError("realized queries per world differ")
        starts = {start for start, _ in queries}
        action_bags = {tuple(sorted(Counter(word).items())) for _, word in queries}
        if len(starts) != 1 or len(action_bags) != 1 or queries[0] == queries[1]:
            raise FunctorCompilerAuditError("realized order-twin invariant differs")

    execution_digest = hashlib.sha256()
    words = tuple(_all_words(max_depth, min_depth=1))
    for machine in sorted(machines.values(), key=lambda value: value.world_id):
        execution_digest.update(machine.world_id.encode("ascii"))
        for start in range(STATE_COUNT):
            for word in words:
                result = machine.execute(start, word)
                execution_digest.update(bytes((start, len(word), *word, result)))

    bits = _machine_bit_receipt(max_depth)
    realized_answer_bits_per_world = 2 * int(bits["state_index_bits"])
    return {
        "schema": REPORT_SCHEMA,
        "claim_scope": (
            "CPU protocol and finite-machine audit only; no neural fit, "
            "reasoning result, or continuation-pretraining authorization"
        ),
        "custody_path": str(custody.absolute()),
        "input_sha256": {
            "development_worlds.jsonl": EXPECTED_WORLDS_SHA256,
            "development_queries.jsonl": EXPECTED_QUERIES_SHA256,
            "development_assessor.jsonl": EXPECTED_ASSESSOR_SHA256,
        },
        "worlds": len(machines),
        "realized_queries": len(query_rows),
        "realized_queries_per_world": 2,
        "realized_query_depths": dict(sorted(depths.items())),
        "realized_machine_exact": {
            "correct": realized_exact,
            "total": len(query_rows),
            "rate": realized_exact / len(query_rows),
        },
        "world_transition_coverage": {
            "state_count": STATE_COUNT,
            "action_count": ACTION_COUNT,
            "transitions_per_world": STATE_COUNT * ACTION_COUNT,
            "complete_worlds": len(machines),
        },
        "causal_quotient": {
            "observer": "identity state-token observer",
            "classes_per_world": STATE_COUNT,
            "separated_at_empty_continuation": True,
        },
        "resource_receipt": {
            **bits,
            "realized_two_answer_cache_bits_per_world_ignoring_routing": (
                realized_answer_bits_per_world
            ),
        },
        "exhaustive_execution": {
            "queries_per_world": int(
                bits["query_count_all_starts_through_depth"]
            ),
            "total_queries": len(machines)
            * int(bits["query_count_all_starts_through_depth"]),
            "sha256": execution_digest.hexdigest(),
        },
        "finite_query_theorem": {
            "theorem_valid": True,
            "two_realized_answers_are_source_derivable": "not established",
            "reason": (
                "start state and action word are late-query fields absent from "
                "the compiler-visible world row"
            ),
            "old_board_decisive_no_go_as_stated": False,
            "post_seal_independent_challenge_required": True,
        },
        "next_action": (
            "build independent post-seal challenge generator and two "
            "independent categorical-machine runtimes"
        ),
        "reasoning_promotion_authorized": False,
        "continuation_pretraining_authorized": False,
        "pretraining_started": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--custody", type=Path, default=DEFAULT_CUSTODY_BUNDLE)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--output", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = audit_bundle(args.custody, max_depth=args.max_depth)
    if args.output is not None:
        if args.output.exists() or args.output.is_symlink():
            raise FileExistsError(f"refusing to replace {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        write_json_fsync(args.output, report)
        report = {
            **report,
            "output": str(args.output.absolute()),
            "output_sha256": file_sha256(args.output),
        }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
