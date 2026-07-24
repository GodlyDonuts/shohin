#!/usr/bin/env python3
"""CPU falsifiers for the proposed EPISODE Functor Compiler.

This module is mechanics-only.  It does not instantiate a neural model, open a
sealed confirmation split, authorize a source freeze, or start pretraining.

The audits deliberately distinguish:

* the number of query rows sampled for one frozen corpus world; and
* the query support available to the post-seal executor.

Those are not interchangeable.  A two-entry cache can be exact if it is
preloaded with the two hidden query identities and answers, but that preload
crosses the current compiler/executor/assessor custody boundary.
"""

from __future__ import annotations

import argparse
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, replace
from hashlib import sha256
from itertools import product
import json
import math
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from pipeline.episode_action_binding_board import (
    ACTION_COUNT,
    ANSWER,
    BOS,
    DEMO,
    EOS,
    MAX_QUERY_DEPTH,
    QUERY,
    SEP,
    STATE_COUNT,
    THEN,
    YIELDS,
    ModelPacket,
    parse_episode,
    split_world_and_query,
    world_commitment,
)
from pipeline.episode_workspace_custody import DEFAULT_CUSTODY_BUNDLE


DEFAULT_CORPUS = Path(
    "artifacts/r12/episode_action_binding_corpus_v1_1532fe2_seed2026072309"
)


class FunctorFalsifierError(ValueError):
    """A mechanics, custody, quotient, or resource invariant failed."""


@dataclass(frozen=True)
class CategoricalEpisodeMachine:
    """Anonymous categorical machine reconstructed from one visible world.

    ``state_keys`` and ``action_keys`` are retained opaque names.  The current
    EPISODE late query supplies a start-state key, so a lawful committed schema
    needs state keys (or an explicit redesign that moves the initial state into
    the source).  An ``initial_state`` field alone cannot execute this board.
    """

    state_keys: tuple[int, ...]
    action_keys: tuple[int, ...]
    action_next: tuple[tuple[int, ...], ...]
    observer_answer: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        state_count = len(self.state_keys)
        if state_count == 0 or len(set(self.state_keys)) != state_count:
            raise FunctorFalsifierError("state keys must be nonempty and unique")
        if not self.action_keys or len(set(self.action_keys)) != len(self.action_keys):
            raise FunctorFalsifierError("action keys must be nonempty and unique")
        if len(self.action_next) != len(self.action_keys):
            raise FunctorFalsifierError("one transition table is required per action")
        for row in self.action_next:
            if len(row) != state_count or any(
                destination not in range(state_count) for destination in row
            ):
                raise FunctorFalsifierError("transition table is malformed")
        if not self.observer_answer:
            raise FunctorFalsifierError("at least one observer is required")
        if any(len(row) != state_count for row in self.observer_answer):
            raise FunctorFalsifierError("observer table is malformed")

    @property
    def state_count(self) -> int:
        return len(self.state_keys)

    @property
    def action_count(self) -> int:
        return len(self.action_keys)

    def execute_indices(
        self,
        start_state: int,
        action_path: Sequence[int],
        *,
        observer: int = 0,
    ) -> int:
        if start_state not in range(self.state_count):
            raise FunctorFalsifierError("start state is outside the active machine")
        if observer not in range(len(self.observer_answer)):
            raise FunctorFalsifierError("observer is outside the active machine")
        state = start_state
        for action in action_path:
            if action not in range(self.action_count):
                raise FunctorFalsifierError("action is outside the active machine")
            state = self.action_next[action][state]
        return self.observer_answer[observer][state]

    def execute_keys(
        self,
        start_key: int,
        action_word: Sequence[int],
        *,
        observer: int = 0,
    ) -> int:
        try:
            start_state = self.state_keys.index(start_key)
            action_path = tuple(self.action_keys.index(key) for key in action_word)
        except ValueError as exc:
            raise FunctorFalsifierError("query contains an unknown opaque key") from exc
        answer_state = self.execute_indices(
            start_state,
            action_path,
            observer=observer,
        )
        if answer_state not in range(self.state_count):
            raise FunctorFalsifierError(
                "EPISODE identity observer returned a non-state answer"
            )
        return self.state_keys[answer_state]

    def permute_action_keys(
        self,
        permutation: Sequence[int],
    ) -> "CategoricalEpisodeMachine":
        checked = _checked_permutation(permutation, self.action_count)
        return replace(
            self,
            action_keys=tuple(self.action_keys[index] for index in checked),
        )

    def permute_action_operators(
        self,
        permutation: Sequence[int],
    ) -> "CategoricalEpisodeMachine":
        checked = _checked_permutation(permutation, self.action_count)
        return replace(
            self,
            action_next=tuple(self.action_next[index] for index in checked),
        )

    def compensated_action_permutation(
        self,
        permutation: Sequence[int],
    ) -> "CategoricalEpisodeMachine":
        checked = _checked_permutation(permutation, self.action_count)
        return replace(
            self,
            action_keys=tuple(self.action_keys[index] for index in checked),
            action_next=tuple(self.action_next[index] for index in checked),
        )

    def transplant_transition_row(
        self,
        *,
        action: int,
        state: int,
        destination: int,
    ) -> "CategoricalEpisodeMachine":
        if action not in range(self.action_count):
            raise FunctorFalsifierError("transplant action is outside the machine")
        if state not in range(self.state_count):
            raise FunctorFalsifierError("transplant state is outside the machine")
        if destination not in range(self.state_count):
            raise FunctorFalsifierError("transplant destination is outside the machine")
        tables = [list(row) for row in self.action_next]
        tables[action][state] = destination
        return replace(self, action_next=tuple(tuple(row) for row in tables))


@dataclass(frozen=True)
class TwoAnswerCache:
    """A fixed two-entry query-indexed answer cache."""

    entries: tuple[tuple[tuple[int, ...], int], ...]

    def __post_init__(self) -> None:
        if len(self.entries) != 2:
            raise FunctorFalsifierError("the cache must contain exactly two entries")
        keys = [key for key, _ in self.entries]
        if len(set(keys)) != len(keys):
            raise FunctorFalsifierError("cache query keys must be distinct")

    def lookup(self, query_tokens: Sequence[int]) -> int | None:
        query = tuple(query_tokens)
        for key, answer in self.entries:
            if key == query:
                return answer
        return None


@dataclass(frozen=True)
class ResourceReceipt:
    states: int
    actions: int
    observers: int
    maximum_depth: int
    query_count: int
    answer_bits_per_query: int
    exhaustive_answer_bits: int
    transition_bits: int
    observer_bits: int
    opaque_state_key_bits: int
    opaque_action_key_bits: int
    machine_semantic_bits: int


def _checked_permutation(
    permutation: Sequence[int],
    size: int,
) -> tuple[int, ...]:
    result = tuple(permutation)
    if sorted(result) != list(range(size)):
        raise FunctorFalsifierError("action intervention is not a permutation")
    return result


def parse_world_machine(world_tokens: Sequence[int]) -> CategoricalEpisodeMachine:
    """Compile a complete current-EPISODE world into a categorical machine."""

    tokens = tuple(world_tokens)
    if not tokens or tokens[0] != BOS:
        raise FunctorFalsifierError("world must begin with BOS")
    cursor = 1
    demonstrations: list[tuple[int, int, int]] = []
    while cursor < len(tokens):
        if cursor + 5 >= len(tokens):
            raise FunctorFalsifierError("truncated world demonstration")
        marker, source, action, yields, target, separator = tokens[cursor : cursor + 6]
        if marker != DEMO or yields != YIELDS or separator != SEP:
            raise FunctorFalsifierError("malformed world demonstration")
        demonstrations.append((source, action, target))
        cursor += 6
    if not demonstrations:
        raise FunctorFalsifierError("world has no demonstrations")

    state_keys = tuple(
        sorted({value for source, _, target in demonstrations for value in (source, target)})
    )
    action_keys = tuple(sorted({action for _, action, _ in demonstrations}))
    if len(state_keys) != STATE_COUNT or len(action_keys) != ACTION_COUNT:
        raise FunctorFalsifierError("world cardinality differs from current EPISODE")

    state_index = {key: index for index, key in enumerate(state_keys)}
    action_index = {key: index for index, key in enumerate(action_keys)}
    tables: list[list[int | None]] = [
        [None] * len(state_keys) for _ in range(len(action_keys))
    ]
    for source, action, target in demonstrations:
        row = tables[action_index[action]]
        source_index = state_index[source]
        destination = state_index[target]
        if row[source_index] is not None:
            raise FunctorFalsifierError("world repeats a state/action transition")
        row[source_index] = destination
    if any(destination is None for row in tables for destination in row):
        raise FunctorFalsifierError("world does not identify the full transition table")
    action_next = tuple(
        tuple(int(destination) for destination in row) for row in tables
    )
    return CategoricalEpisodeMachine(
        state_keys=state_keys,
        action_keys=action_keys,
        action_next=action_next,
        observer_answer=(tuple(range(len(state_keys))),),
    )


def parse_query_tokens(
    query_tokens: Sequence[int],
) -> tuple[int, tuple[int, ...]]:
    tokens = tuple(query_tokens)
    if len(tokens) < 5 or tokens[0] != QUERY or tokens[-2:] != (ANSWER, EOS):
        raise FunctorFalsifierError("malformed late query")
    start_key = tokens[1]
    cursor = 2
    actions: list[int] = []
    expect_action = True
    while cursor < len(tokens) - 2:
        token = tokens[cursor]
        if expect_action:
            if token == THEN:
                raise FunctorFalsifierError("late query is missing an action")
            actions.append(token)
            expect_action = False
        else:
            if token != THEN:
                raise FunctorFalsifierError("late query is missing THEN")
            expect_action = True
        cursor += 1
    if expect_action or not actions:
        raise FunctorFalsifierError("late query is empty or truncated")
    return start_key, tuple(actions)


def execute_query_tokens(
    machine: CategoricalEpisodeMachine,
    query_tokens: Sequence[int],
) -> int:
    start_key, action_word = parse_query_tokens(query_tokens)
    return machine.execute_keys(start_key, action_word)


def enumerate_action_words(
    action_count: int,
    maximum_depth: int,
    *,
    minimum_depth: int = 1,
) -> Iterable[tuple[int, ...]]:
    if action_count <= 0:
        raise FunctorFalsifierError("action count must be positive")
    if minimum_depth < 0 or maximum_depth < minimum_depth:
        raise FunctorFalsifierError("invalid action-word depth interval")
    for depth in range(minimum_depth, maximum_depth + 1):
        yield from product(range(action_count), repeat=depth)


def canonical_query_universe(
    machine: CategoricalEpisodeMachine,
    *,
    maximum_depth: int = MAX_QUERY_DEPTH,
) -> tuple[tuple[int, ...], ...]:
    queries: list[tuple[int, ...]] = []
    for start_key in machine.state_keys:
        for word in enumerate_action_words(machine.action_count, maximum_depth):
            action_keys = tuple(machine.action_keys[index] for index in word)
            tokens: list[int] = [QUERY, start_key]
            for position, action_key in enumerate(action_keys):
                if position:
                    tokens.append(THEN)
                tokens.append(action_key)
            tokens.extend((ANSWER, EOS))
            queries.append(tuple(tokens))
    return tuple(queries)


def compile_canonical_two_answer_cache(
    machine: CategoricalEpisodeMachine,
) -> TwoAnswerCache:
    """Build a lawful world-only cache for two fixed canonical late queries."""

    queries = canonical_query_universe(machine)[:2]
    return TwoAnswerCache(
        entries=tuple(
            (query, execute_query_tokens(machine, query)) for query in queries
        )
    )


def compile_leaky_two_answer_cache(
    machine: CategoricalEpisodeMachine,
    queries: Sequence[Sequence[int]],
    answers: Sequence[int],
) -> TwoAnswerCache:
    """Build the theorem's cache after receiving forbidden query/answer data."""

    if len(queries) != 2 or len(answers) != 2:
        raise FunctorFalsifierError("leaky cache construction requires two rows")
    for query, answer in zip(queries, answers, strict=True):
        if execute_query_tokens(machine, query) != answer:
            raise FunctorFalsifierError("leaky cache answer disagrees with the world")
    return TwoAnswerCache(
        entries=tuple(
            (tuple(query), int(answer))
            for query, answer in zip(queries, answers, strict=True)
        )
    )


def causal_quotient(
    action_next: Sequence[Sequence[int]],
    observer_answer: Sequence[Sequence[int]],
) -> tuple[int, ...]:
    """Compute the exact Moore-machine future-equivalence partition."""

    transitions = tuple(tuple(row) for row in action_next)
    observers = tuple(tuple(row) for row in observer_answer)
    if not transitions or not observers:
        raise FunctorFalsifierError("quotient requires actions and observers")
    state_count = len(transitions[0])
    if state_count == 0:
        raise FunctorFalsifierError("quotient state set is empty")
    if any(len(row) != state_count for row in transitions + observers):
        raise FunctorFalsifierError("quotient tables have inconsistent state counts")
    if any(
        destination not in range(state_count)
        for row in transitions
        for destination in row
    ):
        raise FunctorFalsifierError("quotient transition leaves the state set")

    classes = _class_ids(
        tuple(tuple(observer[state] for observer in observers) for state in range(state_count))
    )
    while True:
        signatures = tuple(
            (
                tuple(observer[state] for observer in observers),
                tuple(classes[action[state]] for action in transitions),
            )
            for state in range(state_count)
        )
        refined = _class_ids(signatures)
        if refined == classes:
            return classes
        classes = refined


def _class_ids(signatures: Sequence[object]) -> tuple[int, ...]:
    identifiers: dict[object, int] = {}
    result: list[int] = []
    for signature in signatures:
        if signature not in identifiers:
            identifiers[signature] = len(identifiers)
        result.append(identifiers[signature])
    return tuple(result)


def shortest_separating_words(
    machine: CategoricalEpisodeMachine,
) -> Mapping[tuple[int, int], tuple[int, ...]]:
    """Return one shortest action word separating every distinguishable pair."""

    results: dict[tuple[int, int], tuple[int, ...]] = {}
    for left in range(machine.state_count):
        for right in range(left + 1, machine.state_count):
            queue: deque[tuple[int, int, tuple[int, ...]]] = deque(
                [(left, right, ())]
            )
            seen = {(left, right)}
            found: tuple[int, ...] | None = None
            while queue:
                current_left, current_right, word = queue.popleft()
                if any(
                    observer[current_left] != observer[current_right]
                    for observer in machine.observer_answer
                ):
                    found = word
                    break
                for action, transition in enumerate(machine.action_next):
                    next_pair = (
                        transition[current_left],
                        transition[current_right],
                    )
                    if next_pair in seen:
                        continue
                    seen.add(next_pair)
                    queue.append((*next_pair, (*word, action)))
            if found is not None:
                results[(left, right)] = found
    return results


def resource_receipt(
    machine: CategoricalEpisodeMachine,
    *,
    maximum_depth: int = MAX_QUERY_DEPTH,
    vocabulary_size: int = 32_768,
) -> ResourceReceipt:
    if vocabulary_size <= max((*machine.state_keys, *machine.action_keys)):
        raise FunctorFalsifierError("vocabulary size does not contain opaque keys")
    state_bits = max(1, math.ceil(math.log2(machine.state_count)))
    answer_alphabet = max(
        2,
        1
        + max(answer for observer in machine.observer_answer for answer in observer),
    )
    answer_bits = math.ceil(math.log2(answer_alphabet))
    query_count = machine.state_count * sum(
        machine.action_count**depth for depth in range(1, maximum_depth + 1)
    )
    transition_bits = machine.action_count * machine.state_count * state_bits
    observer_bits = (
        len(machine.observer_answer) * machine.state_count * answer_bits
    )
    key_bits = math.ceil(math.log2(vocabulary_size))
    state_key_bits = machine.state_count * key_bits
    action_key_bits = machine.action_count * key_bits
    return ResourceReceipt(
        states=machine.state_count,
        actions=machine.action_count,
        observers=len(machine.observer_answer),
        maximum_depth=maximum_depth,
        query_count=query_count,
        answer_bits_per_query=answer_bits,
        exhaustive_answer_bits=query_count * answer_bits,
        transition_bits=transition_bits,
        observer_bits=observer_bits,
        opaque_state_key_bits=state_key_bits,
        opaque_action_key_bits=action_key_bits,
        machine_semantic_bits=(
            transition_bits + observer_bits + state_key_bits + action_key_bits
        ),
    )


def audit_interventions(
    machine: CategoricalEpisodeMachine,
    *,
    maximum_depth: int = 4,
) -> Mapping[str, int | bool]:
    permutation = tuple(range(1, machine.action_count)) + (0,)
    key_only = machine.permute_action_keys(permutation)
    operator_only = machine.permute_action_operators(permutation)
    compensated = machine.compensated_action_permutation(permutation)
    checked = 0
    key_changed = 0
    operator_changed = 0
    compensated_changed = 0
    for start in range(machine.state_count):
        for word in enumerate_action_words(machine.action_count, maximum_depth):
            query_keys = tuple(machine.action_keys[index] for index in word)
            original = machine.execute_keys(machine.state_keys[start], query_keys)
            key_answer = key_only.execute_keys(machine.state_keys[start], query_keys)
            operator_answer = operator_only.execute_keys(
                machine.state_keys[start],
                query_keys,
            )
            compensated_answer = compensated.execute_keys(
                machine.state_keys[start],
                query_keys,
            )
            checked += 1
            key_changed += int(key_answer != original)
            operator_changed += int(operator_answer != original)
            compensated_changed += int(compensated_answer != original)
    return {
        "queries_checked": checked,
        "key_only_changed": key_changed,
        "operator_only_changed": operator_changed,
        "compensated_changed": compensated_changed,
        "key_intervention_nontrivial": key_changed > 0,
        "operator_intervention_nontrivial": operator_changed > 0,
        "compensated_invariance": compensated_changed == 0,
    }


def audit_frozen_corpus(
    corpus: Path = DEFAULT_CORPUS,
    custody: Path = DEFAULT_CUSTODY_BUNDLE,
) -> dict[str, object]:
    """Run the complete CPU audit against the already-consumed EPISODE corpus."""

    bundle_manifest = _read_json_object(corpus / "bundle_manifest.json")
    if bundle_manifest.get("schema") != "episode_action_binding_bundle_v1":
        raise FunctorFalsifierError("corpus bundle manifest schema differs")
    corpus_files = _digest_map(
        bundle_manifest.get("artifacts"),
        required={
            "manifest.json",
            "model_packets.jsonl",
            "offline_ledger.jsonl",
            "target_labels.jsonl",
        },
        label="corpus bundle",
    )
    packet_rows = _read_jsonl(
        corpus / "model_packets.jsonl",
        corpus_files["model_packets.jsonl"],
    )
    target_rows = _read_jsonl(
        corpus / "target_labels.jsonl",
        corpus_files["target_labels.jsonl"],
    )
    targets = {
        _required_string(row, "packet_sha256"): _required_int(row, "target_token")
        for row in target_rows
    }
    machines: dict[str, CategoricalEpisodeMachine] = {}
    behavior_exact = 0
    for row in packet_rows:
        packet_digest = _required_string(row, "packet_sha256")
        tokens = _required_int_tuple(row, "tokens")
        mask = _required_int_tuple(row, "attention_mask")
        packet = ModelPacket(tokens=tokens, attention_mask=mask)
        parsed = parse_episode(packet)
        world, query = split_world_and_query(packet)
        commitment = world_commitment(packet)
        machine = machines.setdefault(commitment, parse_world_machine(world))
        predicted = machine.execute_keys(parsed.query_start, parsed.query_actions)
        if predicted == targets[packet_digest]:
            behavior_exact += 1
        if predicted != execute_query_tokens(machine, query):
            raise FunctorFalsifierError("packet and split-query execution differ")

    quotient_class_counts = {
        len(set(causal_quotient(machine.action_next, machine.observer_answer)))
        for machine in machines.values()
    }
    if len(quotient_class_counts) != 1:
        raise FunctorFalsifierError("frozen worlds have inconsistent quotient sizes")
    first_machine = next(iter(machines.values()))
    separators = shortest_separating_words(first_machine)
    receipt = resource_receipt(first_machine)
    intervention = audit_interventions(first_machine)
    cache = audit_two_answer_cache_custody(custody)
    report = {
        "schema": "episode_functor_compiler_cpu_falsifier_v1",
        "status": "MECHANICS_ONLY_NO_NEURAL_AUTHORIZATION",
        "corpus": str(corpus),
        "custody": str(custody),
        "counts": {
            "packets": len(packet_rows),
            "unique_committed_worlds": len(machines),
            "exact_machine_executions": behavior_exact,
            "causal_quotient_classes": next(iter(quotient_class_counts)),
            "separated_state_pairs": len(separators),
            "maximum_shortest_separator_depth": max(
                (len(word) for word in separators.values()),
                default=0,
            ),
        },
        "resource_receipt": asdict(receipt),
        "intervention_audit": dict(intervention),
        "two_answer_cache_audit": cache,
        "findings": {
            "old_board_has_two_sampled_queries_per_committed_world": True,
            "old_board_post_seal_query_support_is_two": False,
            "two_entry_cache_theorem_application_is_lawful_under_current_custody": False,
            "current_query_requires_opaque_start_state_binding": True,
            "draft_initial_state_without_state_keys_is_sufficient": False,
            "categorical_machine_executes_current_corpus": (
                behavior_exact == len(packet_rows)
            ),
            "no_neural_fit_authorized": True,
            "no_source_freeze_authorized": True,
            "continuation_pretraining_authorized": False,
        },
    }
    report["report_sha256"] = _json_sha256(report)
    return report


def audit_two_answer_cache_custody(
    custody: Path = DEFAULT_CUSTODY_BUNDLE,
) -> dict[str, object]:
    manifest = _read_json_object(custody / "custody_manifest.json")
    if manifest.get("schema") != "episode_workspace_custody_v1":
        raise FunctorFalsifierError("custody manifest schema differs")
    custody_files = _digest_map(
        manifest.get("files"),
        required={
            "development_worlds.jsonl",
            "development_queries.jsonl",
            "development_assessor.jsonl",
            "train_true_groups.jsonl",
            "train_shuffled_groups.jsonl",
        },
        label="custody",
    )
    if manifest.get("compiler_visible_files") != ["development_worlds.jsonl"]:
        raise FunctorFalsifierError("compiler custody visibility differs")
    if manifest.get("executor_visible_files") != ["development_queries.jsonl"]:
        raise FunctorFalsifierError("executor custody visibility differs")
    if manifest.get("assessor_visible_files") != ["development_assessor.jsonl"]:
        raise FunctorFalsifierError("assessor custody visibility differs")
    worlds = _read_jsonl(
        custody / "development_worlds.jsonl",
        custody_files["development_worlds.jsonl"],
    )
    queries = _read_jsonl(
        custody / "development_queries.jsonl",
        custody_files["development_queries.jsonl"],
    )
    assessor = _read_jsonl(
        custody / "development_assessor.jsonl",
        custody_files["development_assessor.jsonl"],
    )
    world_by_id = {
        _required_string(row, "world_id"): parse_world_machine(
            _required_int_tuple(row, "world_tokens")
        )
        for row in worlds
    }
    queries_by_world: dict[str, list[tuple[str, tuple[int, ...]]]] = defaultdict(list)
    for row in queries:
        queries_by_world[_required_string(row, "world_id")].append(
            (
                _required_string(row, "packet_sha256"),
                _required_int_tuple(row, "query_tokens"),
            )
        )
    targets = {
        _required_string(row, "packet_sha256"): _required_int(row, "target_token")
        for row in assessor
    }
    if set(world_by_id) != set(queries_by_world):
        raise FunctorFalsifierError("custody world/query identities differ")

    canonical_coverage = 0
    leaky_exact = 0
    sampled_rows = 0
    for world_id, machine in world_by_id.items():
        rows = queries_by_world[world_id]
        if len(rows) != 2:
            raise FunctorFalsifierError("custody world does not have two sampled rows")
        canonical = compile_canonical_two_answer_cache(machine)
        leaky = compile_leaky_two_answer_cache(
            machine,
            [query for _, query in rows],
            [targets[packet_digest] for packet_digest, _ in rows],
        )
        for packet_digest, query in rows:
            sampled_rows += 1
            canonical_coverage += int(canonical.lookup(query) is not None)
            leaky_exact += int(leaky.lookup(query) == targets[packet_digest])

    receipt = resource_receipt(next(iter(world_by_id.values())))
    universe = receipt.query_count
    pair_probability = 1.0 / math.comb(universe, 2)
    return {
        "compiler_visible_files": manifest["compiler_visible_files"],
        "executor_visible_files": manifest["executor_visible_files"],
        "assessor_visible_files": manifest["assessor_visible_files"],
        "committed_worlds": len(world_by_id),
        "sampled_query_rows": sampled_rows,
        "sampled_queries_per_world": 2,
        "post_seal_query_support_per_world": universe,
        "canonical_world_only_two_entry_coverage": canonical_coverage,
        "leaky_hidden_query_and_target_cache_exact": leaky_exact,
        "probability_fixed_two_keys_match_random_hidden_pair": pair_probability,
        "leaky_cache_requires_executor_and_assessor_inputs": True,
        "lawful_world_only_theorem_construction_available": False,
    }


def _required_string(row: Mapping[str, object], key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str):
        raise FunctorFalsifierError(f"{key} is not a string")
    return value


def _required_int(row: Mapping[str, object], key: str) -> int:
    value = row.get(key)
    if type(value) is not int:
        raise FunctorFalsifierError(f"{key} is not an integer")
    return value


def _required_int_tuple(row: Mapping[str, object], key: str) -> tuple[int, ...]:
    value = row.get(key)
    if not isinstance(value, list) or any(type(item) is not int for item in value):
        raise FunctorFalsifierError(f"{key} is not an integer list")
    return tuple(value)


def _read_json_object(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FunctorFalsifierError(f"{path} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise FunctorFalsifierError(f"{path} is not a JSON object")
    return value


def _digest_map(
    value: object,
    *,
    required: set[str],
    label: str,
) -> dict[str, str]:
    if not isinstance(value, dict) or set(value) != required:
        raise FunctorFalsifierError(f"{label} digest fields differ")
    result: dict[str, str] = {}
    for name, digest in value.items():
        if (
            not isinstance(name, str)
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise FunctorFalsifierError(f"{label} digest entry is invalid")
        result[name] = digest
    return result


def _read_jsonl(
    path: Path,
    expected_sha256: str | None = None,
) -> list[dict[str, object]]:
    raw = path.read_bytes()
    if (
        expected_sha256 is not None
        and sha256(raw).hexdigest() != expected_sha256
    ):
        raise FunctorFalsifierError(f"{path.name} hash differs from its manifest")
    rows: list[dict[str, object]] = []
    for line_number, line in enumerate(raw.splitlines(), start=1):
        try:
            value = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise FunctorFalsifierError(
                f"{path}:{line_number} is invalid JSON"
            ) from exc
        if not isinstance(value, dict):
            raise FunctorFalsifierError(f"{path}:{line_number} is not an object")
        rows.append(value)
    return rows


def _json_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return sha256(payload).hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--custody", type=Path, default=DEFAULT_CUSTODY_BUNDLE)
    parser.add_argument(
        "--output",
        type=Path,
        help="Write the deterministic JSON report instead of printing it.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    payload = json.dumps(
        audit_frozen_corpus(args.corpus, args.custody),
        indent=2,
        sort_keys=True,
    )
    if args.output is None:
        print(payload)
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(f"{payload}\n")


if __name__ == "__main__":
    main()
