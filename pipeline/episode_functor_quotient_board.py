#!/usr/bin/env python3
"""Deterministic CPU mechanics for nontrivial observed Moore quotients.

This module is deliberately disjoint from neural training and sealed-board
custody.  It creates an in-memory exploratory fixture, proves its finite-state
properties with exact CPU algorithms, and writes nothing.

The three quotient oracles are intentionally independent:

* iterative Moore partition refinement;
* pair-product reachability (a distinguishing-word search); and
* exhaustive future traces through the finite-state distinguishing bound.
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import asdict, dataclass, replace
from hashlib import sha256
from itertools import permutations, product
import json
import random
from typing import Sequence


PHYSICAL_STATE_COUNT = 8
ACTION_NAMES = ("act_a", "act_b", "act_c")
OBSERVER_NAMES = ("obs_left", "obs_right")
QUOTIENT_SIZES = tuple(range(3, PHYSICAL_STATE_COUNT + 1))
DEFAULT_FIXTURE_SEED = 20_260_723_41
EXHAUSTIVE_DEPTH = PHYSICAL_STATE_COUNT - 1


class QuotientBoardError(ValueError):
    """A finite-machine construction or exact-audit invariant failed."""


Word = tuple[int, ...]


@dataclass(frozen=True)
class QuotientMachine:
    """Eight-key Moore machine with an asserted physical-to-causal quotient."""

    split: str
    name: str
    physical_keys: tuple[str, ...]
    action_names: tuple[str, ...]
    observer_names: tuple[str, ...]
    transitions: tuple[tuple[int, ...], ...]
    observations: tuple[tuple[int, ...], ...]
    key_to_quotient_class: tuple[int, ...]
    generation_nonce: int

    def __post_init__(self) -> None:
        if self.split not in {"train", "development", "intervention", "gauge"}:
            raise QuotientBoardError(f"unsupported split {self.split!r}")
        if len(self.physical_keys) != PHYSICAL_STATE_COUNT:
            raise QuotientBoardError("machine must expose exactly eight physical keys")
        if len(set(self.physical_keys)) != PHYSICAL_STATE_COUNT:
            raise QuotientBoardError("physical keys must be unique")
        if self.action_names != ACTION_NAMES:
            raise QuotientBoardError("machine must expose the fixed three actions")
        if self.observer_names != OBSERVER_NAMES:
            raise QuotientBoardError("machine must expose the fixed two observers")
        if len(self.transitions) != len(ACTION_NAMES):
            raise QuotientBoardError("one transition row is required per action")
        if any(len(row) != PHYSICAL_STATE_COUNT for row in self.transitions):
            raise QuotientBoardError("transition rows must cover all physical keys")
        if any(
            destination not in range(PHYSICAL_STATE_COUNT)
            for row in self.transitions
            for destination in row
        ):
            raise QuotientBoardError("transition leaves the physical state set")
        if len(self.observations) != len(OBSERVER_NAMES):
            raise QuotientBoardError("both observers are required")
        if any(len(row) != PHYSICAL_STATE_COUNT for row in self.observations):
            raise QuotientBoardError("observer rows must cover all physical keys")
        if len(self.key_to_quotient_class) != PHYSICAL_STATE_COUNT:
            raise QuotientBoardError("key-to-class map must cover all physical keys")
        classes = set(self.key_to_quotient_class)
        if classes != set(range(len(classes))):
            raise QuotientBoardError("quotient class identifiers must be contiguous")
        _validate_descends_to_asserted_quotient(self)

    @property
    def quotient_size(self) -> int:
        return len(set(self.key_to_quotient_class))

    def step(self, state: int, action: int) -> int:
        if state not in range(PHYSICAL_STATE_COUNT):
            raise QuotientBoardError("start state is outside the physical machine")
        if action not in range(len(ACTION_NAMES)):
            raise QuotientBoardError("action is outside the action alphabet")
        return self.transitions[action][state]

    def execute(self, start: int, word: Sequence[int]) -> int:
        state = start
        for action in word:
            state = self.step(state, action)
        return state

    def observe(self, state: int) -> tuple[int, ...]:
        if state not in range(PHYSICAL_STATE_COUNT):
            raise QuotientBoardError("observation state is outside the machine")
        return tuple(observer[state] for observer in self.observations)

    def response(self, start: int, word: Sequence[int]) -> tuple[int, ...]:
        return self.observe(self.execute(start, word))


@dataclass(frozen=True)
class QuotientCPUFixture:
    """Unsealed in-memory fixture consumed only by exact CPU tests."""

    schema: str
    status: str
    seed: int
    train: tuple[QuotientMachine, ...]
    development: tuple[QuotientMachine, ...]

    def __post_init__(self) -> None:
        if self.schema != "episode_functor_quotient_cpu_fixture_v1":
            raise QuotientBoardError("unexpected fixture schema")
        if self.status != "exploratory_cpu_only_not_official_not_sealed":
            raise QuotientBoardError("fixture status must remain explicitly unsealed")
        if not self.train or not self.development:
            raise QuotientBoardError(
                "both train and development fixture cells are needed"
            )
        if any(machine.split != "train" for machine in self.train):
            raise QuotientBoardError("train fixture contains a non-train machine")
        if any(machine.split != "development" for machine in self.development):
            raise QuotientBoardError(
                "development fixture contains a non-development machine"
            )


def _validate_descends_to_asserted_quotient(machine: QuotientMachine) -> None:
    """Check that observations and transitions are class-representative invariant."""

    class_map = machine.key_to_quotient_class
    for left in range(PHYSICAL_STATE_COUNT):
        for right in range(left + 1, PHYSICAL_STATE_COUNT):
            if class_map[left] != class_map[right]:
                continue
            if machine.observe(left) != machine.observe(right):
                raise QuotientBoardError(
                    "asserted class contains unequal empty observations"
                )
            for transition in machine.transitions:
                if class_map[transition[left]] != class_map[transition[right]]:
                    raise QuotientBoardError(
                        "asserted class is not closed under an action"
                    )


def _class_ids(signatures: Sequence[object]) -> tuple[int, ...]:
    identifiers: dict[object, int] = {}
    result: list[int] = []
    for signature in signatures:
        if signature not in identifiers:
            identifiers[signature] = len(identifiers)
        result.append(identifiers[signature])
    return tuple(result)


def equivalence_relation(class_ids: Sequence[int]) -> tuple[tuple[bool, ...], ...]:
    """Return the label-gauge-invariant equivalence relation."""

    return tuple(tuple(left == right for right in class_ids) for left in class_ids)


def partition_refinement(machine: QuotientMachine) -> tuple[int, ...]:
    """Compute Moore equivalence by iterative observation/action refinement."""

    classes = _class_ids(
        tuple(machine.observe(state) for state in range(PHYSICAL_STATE_COUNT))
    )
    while True:
        signatures = tuple(
            (
                machine.observe(state),
                tuple(classes[transition[state]] for transition in machine.transitions),
            )
            for state in range(PHYSICAL_STATE_COUNT)
        )
        refined = _class_ids(signatures)
        if equivalence_relation(refined) == equivalence_relation(classes):
            return refined
        classes = refined


def shortest_separator(
    machine: QuotientMachine,
    left: int,
    right: int,
) -> Word | None:
    """Find a shortest distinguishing word by BFS in the pair product.

    ``None`` means the two physical states are behaviorally equivalent.
    """

    if left not in range(PHYSICAL_STATE_COUNT) or right not in range(
        PHYSICAL_STATE_COUNT
    ):
        raise QuotientBoardError("separator endpoints leave the physical state set")
    queue: deque[tuple[int, int, Word]] = deque([(left, right, ())])
    seen = {(left, right)}
    while queue:
        current_left, current_right, word = queue.popleft()
        if machine.observe(current_left) != machine.observe(current_right):
            return word
        for action in range(len(ACTION_NAMES)):
            pair = (
                machine.transitions[action][current_left],
                machine.transitions[action][current_right],
            )
            if pair in seen:
                continue
            seen.add(pair)
            queue.append((pair[0], pair[1], (*word, action)))
    return None


def product_automaton_partition(machine: QuotientMachine) -> tuple[int, ...]:
    """Compute equivalence from pair-product non-reachability.

    This implementation does not call or share signatures with partition
    refinement.  It joins a pair exactly when no unequal-output product state
    is reachable.
    """

    parent = list(range(PHYSICAL_STATE_COUNT))

    def find(item: int) -> int:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for left in range(PHYSICAL_STATE_COUNT):
        for right in range(left + 1, PHYSICAL_STATE_COUNT):
            if shortest_separator(machine, left, right) is None:
                union(left, right)
    return _class_ids(tuple(find(state) for state in range(PHYSICAL_STATE_COUNT)))


def words_through_depth(maximum_depth: int) -> tuple[Word, ...]:
    if maximum_depth < 0:
        raise QuotientBoardError("word depth cannot be negative")
    return tuple(
        word
        for depth in range(maximum_depth + 1)
        for word in product(range(len(ACTION_NAMES)), repeat=depth)
    )


def exhaustive_future_partition(
    machine: QuotientMachine,
    *,
    maximum_depth: int = EXHAUSTIVE_DEPTH,
) -> tuple[int, ...]:
    """Group states by every observed future through the finite-state bound."""

    words = words_through_depth(maximum_depth)
    traces = tuple(
        tuple(machine.response(state, word) for word in words)
        for state in range(PHYSICAL_STATE_COUNT)
    )
    return _class_ids(traces)


def _quotient_tables(
    machine: QuotientMachine,
) -> tuple[tuple[tuple[int, ...], ...], tuple[tuple[int, ...], ...]]:
    representatives = tuple(
        machine.key_to_quotient_class.index(class_id)
        for class_id in range(machine.quotient_size)
    )
    transitions = tuple(
        tuple(
            machine.key_to_quotient_class[transition[representative]]
            for representative in representatives
        )
        for transition in machine.transitions
    )
    observations = tuple(
        tuple(observer[representative] for representative in representatives)
        for observer in machine.observations
    )
    return transitions, observations


def quotient_word_transform(
    machine: QuotientMachine, word: Sequence[int]
) -> tuple[int, ...]:
    transitions, _ = _quotient_tables(machine)
    result: list[int] = []
    for start in range(machine.quotient_size):
        state = start
        for action in word:
            if action not in range(len(ACTION_NAMES)):
                raise QuotientBoardError("word contains an unknown action")
            state = transitions[action][state]
        result.append(state)
    return tuple(result)


def equivalent_word_witness(
    machine: QuotientMachine,
    *,
    maximum_depth: int = 4,
) -> dict[str, object]:
    """Return two distinct nonempty words with the same quotient transformation."""

    seen: dict[tuple[int, ...], Word] = {}
    for word in words_through_depth(maximum_depth):
        if not word:
            continue
        transform = quotient_word_transform(machine, word)
        previous = seen.get(transform)
        if previous is not None and previous != word:
            return {
                "left": previous,
                "right": word,
                "quotient_transform": transform,
            }
        seen[transform] = word
    raise QuotientBoardError("machine has no bounded equivalent-word witness")


def noncommuting_witness(machine: QuotientMachine) -> dict[str, object]:
    """Return ordered actions whose quotient effects differ at one start class."""

    representatives = tuple(
        machine.key_to_quotient_class.index(class_id)
        for class_id in range(machine.quotient_size)
    )
    for start_class, representative in enumerate(representatives):
        for left_action in range(len(ACTION_NAMES)):
            for right_action in range(left_action + 1, len(ACTION_NAMES)):
                left_word = (left_action, right_action)
                right_word = (right_action, left_action)
                left_state = machine.execute(representative, left_word)
                right_state = machine.execute(representative, right_word)
                left_class = machine.key_to_quotient_class[left_state]
                right_class = machine.key_to_quotient_class[right_state]
                if left_class == right_class:
                    continue
                suffix = shortest_separator(machine, left_state, right_state)
                if suffix is None:
                    raise QuotientBoardError(
                        "distinct asserted classes lack a product separator"
                    )
                return {
                    "start_class": start_class,
                    "left_word": left_word,
                    "right_word": right_word,
                    "left_class": left_class,
                    "right_class": right_class,
                    "distinguishing_suffix": suffix,
                }
    raise QuotientBoardError("machine has no noncommuting action witness")


def empty_merge_future_separator(machine: QuotientMachine) -> dict[str, object]:
    """Find a pair merged now but separated by a nonempty future word."""

    for left in range(PHYSICAL_STATE_COUNT):
        for right in range(left + 1, PHYSICAL_STATE_COUNT):
            if machine.observe(left) != machine.observe(right):
                continue
            if (
                machine.key_to_quotient_class[left]
                == machine.key_to_quotient_class[right]
            ):
                continue
            word = shortest_separator(machine, left, right)
            if not word:
                continue
            return {
                "left_key": machine.physical_keys[left],
                "right_key": machine.physical_keys[right],
                "left_state": left,
                "right_state": right,
                "left_class": machine.key_to_quotient_class[left],
                "right_class": machine.key_to_quotient_class[right],
                "separator": word,
            }
    raise QuotientBoardError(
        "machine lacks a pair merged at empty observation and split in the future"
    )


def _canonicalize_values(values: Sequence[int]) -> tuple[int, ...]:
    return _class_ids(tuple(values))


def structural_signature(machine: QuotientMachine) -> str:
    """Canonical quotient structure under state and observer-value gauges."""

    transitions, observations = _quotient_tables(machine)
    size = machine.quotient_size
    candidates: list[tuple[int, ...]] = []
    for order in permutations(range(size)):
        old_to_new = [0] * size
        for new, old in enumerate(order):
            old_to_new[old] = new
        encoded: list[int] = [size, len(ACTION_NAMES), len(OBSERVER_NAMES)]
        for observer in observations:
            encoded.extend(_canonicalize_values(tuple(observer[old] for old in order)))
        for transition in transitions:
            encoded.extend(old_to_new[transition[old]] for old in order)
        candidates.append(tuple(encoded))
    canonical = min(candidates)
    payload = json.dumps(canonical, separators=(",", ":")).encode("ascii")
    return sha256(payload).hexdigest()


def conjugate_state_gauge(
    machine: QuotientMachine,
    old_to_new: Sequence[int],
) -> QuotientMachine:
    """Conjugate all state-indexed tensors by a physical-state permutation."""

    permutation = tuple(old_to_new)
    if sorted(permutation) != list(range(PHYSICAL_STATE_COUNT)):
        raise QuotientBoardError("state gauge must be a permutation of eight states")
    transitions = [[0] * PHYSICAL_STATE_COUNT for _ in ACTION_NAMES]
    observations = [[0] * PHYSICAL_STATE_COUNT for _ in OBSERVER_NAMES]
    class_map = [0] * PHYSICAL_STATE_COUNT
    keys = [""] * PHYSICAL_STATE_COUNT
    for old, new in enumerate(permutation):
        keys[new] = f"gauge_{new}_{machine.physical_keys[old]}"
        class_map[new] = machine.key_to_quotient_class[old]
        for observer_index, observer in enumerate(machine.observations):
            observations[observer_index][new] = observer[old]
        for action_index, transition in enumerate(machine.transitions):
            transitions[action_index][new] = permutation[transition[old]]
    return QuotientMachine(
        split="gauge",
        name=f"{machine.name}_state_gauge",
        physical_keys=tuple(keys),
        action_names=machine.action_names,
        observer_names=machine.observer_names,
        transitions=tuple(tuple(row) for row in transitions),
        observations=tuple(tuple(row) for row in observations),
        key_to_quotient_class=tuple(class_map),
        generation_nonce=machine.generation_nonce,
    )


def state_gauge_audit(machine: QuotientMachine) -> dict[str, object]:
    digest = sha256(f"{machine.name}:state-gauge".encode("ascii")).digest()
    order = sorted(
        range(PHYSICAL_STATE_COUNT),
        key=lambda state: (digest[state], state),
    )
    old_to_new = [0] * PHYSICAL_STATE_COUNT
    for new, old in enumerate(order):
        old_to_new[old] = new
    conjugate = conjugate_state_gauge(machine, old_to_new)
    words = words_through_depth(4)
    behavior_preserved = all(
        conjugate.response(old_to_new[state], word) == machine.response(state, word)
        and conjugate.execute(old_to_new[state], word)
        == old_to_new[machine.execute(state, word)]
        for state in range(PHYSICAL_STATE_COUNT)
        for word in words
    )
    class_transport_preserved = all(
        conjugate.key_to_quotient_class[old_to_new[state]]
        == machine.key_to_quotient_class[state]
        for state in range(PHYSICAL_STATE_COUNT)
    )
    return {
        "old_to_new": tuple(old_to_new),
        "behavior_preserved": behavior_preserved,
        "class_transport_preserved": class_transport_preserved,
        "structural_signature_preserved": (
            structural_signature(conjugate) == structural_signature(machine)
        ),
    }


def observer_intervention(
    machine: QuotientMachine,
) -> tuple[QuotientMachine, dict[str, object]]:
    """Change one observer on one whole causal class, freezing all actions."""

    witness = empty_merge_future_separator(machine)
    target_class = int(witness["right_class"])
    observer_index = 0
    fresh_value = max(machine.observations[observer_index]) + 1
    rows = [list(row) for row in machine.observations]
    changed_states = tuple(
        state
        for state, class_id in enumerate(machine.key_to_quotient_class)
        if class_id == target_class
    )
    for state in changed_states:
        rows[observer_index][state] = fresh_value
    intervened = replace(
        machine,
        split="intervention",
        name=f"{machine.name}_observer_do",
        observations=tuple(tuple(row) for row in rows),
    )
    return intervened, {
        "observer": observer_index,
        "target_class": target_class,
        "changed_states": changed_states,
        "fresh_value": fresh_value,
        "probe_left": int(witness["left_state"]),
        "probe_right": int(witness["right_state"]),
    }


def action_intervention(
    machine: QuotientMachine,
) -> tuple[QuotientMachine, dict[str, object]]:
    """Transplant one quotient action row, freezing every observation."""

    representatives = tuple(
        machine.key_to_quotient_class.index(class_id)
        for class_id in range(machine.quotient_size)
    )
    for action, transition in enumerate(machine.transitions):
        for source_class, source_representative in enumerate(representatives):
            original_state = transition[source_representative]
            original_observation = machine.observe(original_state)
            donor_class = next(
                (
                    class_id
                    for class_id, representative in enumerate(representatives)
                    if machine.observe(representative) != original_observation
                ),
                None,
            )
            if donor_class is None:
                continue
            donor_state = representatives[donor_class]
            rows = [list(row) for row in machine.transitions]
            changed_states = tuple(
                state
                for state, class_id in enumerate(machine.key_to_quotient_class)
                if class_id == source_class
            )
            for state in changed_states:
                rows[action][state] = donor_state
            intervened = replace(
                machine,
                split="intervention",
                name=f"{machine.name}_action_do",
                transitions=tuple(tuple(row) for row in rows),
            )
            return intervened, {
                "action": action,
                "source_class": source_class,
                "changed_states": changed_states,
                "original_target_class": machine.key_to_quotient_class[original_state],
                "donor_target_class": donor_class,
                "probe_state": source_representative,
            }
    raise QuotientBoardError("no behavior-changing action intervention exists")


def _observation_palette(class_count: int) -> tuple[tuple[int, int], ...]:
    if class_count not in QUOTIENT_SIZES:
        raise QuotientBoardError("unsupported quotient size")
    palette = (
        (0, 0),
        (1, 1),
        (2, 0),
        (0, 1),
        (1, 2),
        (2, 2),
        (0, 2),
    )
    symbols = tuple(range(class_count - 1)) + (0,)
    return tuple(palette[symbol] for symbol in symbols)


def _stable_seed(*parts: object) -> int:
    payload = ":".join(map(str, parts)).encode("ascii")
    return int.from_bytes(sha256(payload).digest()[:8], "big")


def _candidate_quotient_tables(
    *,
    fixture_seed: int,
    split: str,
    class_count: int,
    nonce: int,
) -> tuple[tuple[int, ...], ...]:
    rng = random.Random(_stable_seed(fixture_seed, split, class_count, nonce))
    first = tuple(rng.randrange(class_count) for _ in range(class_count))
    second = tuple(rng.randrange(class_count) for _ in range(class_count))

    fixed_count = 2 if class_count < 6 else 3
    fixed_points = tuple(sorted(rng.sample(range(class_count), fixed_count)))
    third_list = [rng.choice(fixed_points) for _ in range(class_count)]
    for fixed in fixed_points:
        third_list[fixed] = fixed
    third = tuple(third_list)
    return first, second, third


def _abstract_machine_for_validation(
    *,
    split: str,
    class_count: int,
    nonce: int,
    transitions: tuple[tuple[int, ...], ...],
) -> QuotientMachine:
    outputs = _observation_palette(class_count)
    class_map = tuple(range(class_count)) + tuple(
        duplicate % class_count
        for duplicate in range(PHYSICAL_STATE_COUNT - class_count)
    )
    members = {
        class_id: tuple(
            state for state, value in enumerate(class_map) if value == class_id
        )
        for class_id in range(class_count)
    }
    physical_transitions = tuple(
        tuple(
            members[row[class_map[state]]][0] for state in range(PHYSICAL_STATE_COUNT)
        )
        for row in transitions
    )
    observations = tuple(
        tuple(
            outputs[class_map[state]][observer] for state in range(PHYSICAL_STATE_COUNT)
        )
        for observer in range(len(OBSERVER_NAMES))
    )
    return QuotientMachine(
        split=split,
        name=f"{split}_q{class_count}_abstract",
        physical_keys=tuple(
            f"abstract_{state}" for state in range(PHYSICAL_STATE_COUNT)
        ),
        action_names=ACTION_NAMES,
        observer_names=OBSERVER_NAMES,
        transitions=physical_transitions,
        observations=observations,
        key_to_quotient_class=class_map,
        generation_nonce=nonce,
    )


def _lift_candidate(
    *,
    fixture_seed: int,
    split: str,
    class_count: int,
    nonce: int,
    quotient_transitions: tuple[tuple[int, ...], ...],
) -> QuotientMachine:
    rng = random.Random(
        _stable_seed(fixture_seed, split, class_count, nonce, "physical-lift")
    )
    class_map = list(range(class_count))
    class_map.extend(rng.randrange(class_count) for _ in range(8 - class_count))
    rng.shuffle(class_map)
    members = {
        class_id: tuple(
            state for state, value in enumerate(class_map) if value == class_id
        )
        for class_id in range(class_count)
    }
    transitions = tuple(
        tuple(
            rng.choice(members[row[class_map[state]]])
            for state in range(PHYSICAL_STATE_COUNT)
        )
        for row in quotient_transitions
    )
    outputs = _observation_palette(class_count)
    observations = tuple(
        tuple(
            outputs[class_map[state]][observer] for state in range(PHYSICAL_STATE_COUNT)
        )
        for observer in range(len(OBSERVER_NAMES))
    )
    keys = tuple(
        f"{split[0]}{class_count}_{rng.getrandbits(48):012x}"
        for _ in range(PHYSICAL_STATE_COUNT)
    )
    return QuotientMachine(
        split=split,
        name=f"{split}_nontrivial_observer_q{class_count}",
        physical_keys=keys,
        action_names=ACTION_NAMES,
        observer_names=OBSERVER_NAMES,
        transitions=transitions,
        observations=observations,
        key_to_quotient_class=tuple(class_map),
        generation_nonce=nonce,
    )


def _generate_machine(
    *,
    fixture_seed: int,
    split: str,
    class_count: int,
    forbidden_signatures: set[str],
) -> QuotientMachine:
    for nonce in range(10_000):
        quotient_transitions = _candidate_quotient_tables(
            fixture_seed=fixture_seed,
            split=split,
            class_count=class_count,
            nonce=nonce,
        )
        abstract = _abstract_machine_for_validation(
            split=split,
            class_count=class_count,
            nonce=nonce,
            transitions=quotient_transitions,
        )
        expected = equivalence_relation(abstract.key_to_quotient_class)
        if equivalence_relation(partition_refinement(abstract)) != expected:
            continue
        try:
            empty_merge_future_separator(abstract)
            equivalent_word_witness(abstract)
            noncommuting_witness(abstract)
        except QuotientBoardError:
            continue
        signature = structural_signature(abstract)
        if signature in forbidden_signatures:
            continue
        machine = _lift_candidate(
            fixture_seed=fixture_seed,
            split=split,
            class_count=class_count,
            nonce=nonce,
            quotient_transitions=quotient_transitions,
        )
        if structural_signature(machine) != signature:
            raise QuotientBoardError("physical lift changed quotient structure")
        return machine
    raise QuotientBoardError(
        f"failed to generate {split} quotient-size-{class_count} machine"
    )


def build_cpu_fixture(seed: int = DEFAULT_FIXTURE_SEED) -> QuotientCPUFixture:
    """Build the deterministic unsealed train/development mechanics fixture."""

    train: list[QuotientMachine] = []
    train_signatures: set[str] = set()
    for class_count in QUOTIENT_SIZES:
        machine = _generate_machine(
            fixture_seed=seed,
            split="train",
            class_count=class_count,
            forbidden_signatures=set(),
        )
        train.append(machine)
        train_signatures.add(structural_signature(machine))

    development = tuple(
        _generate_machine(
            fixture_seed=seed,
            split="development",
            class_count=class_count,
            forbidden_signatures=train_signatures,
        )
        for class_count in QUOTIENT_SIZES
    )
    return QuotientCPUFixture(
        schema="episode_functor_quotient_cpu_fixture_v1",
        status="exploratory_cpu_only_not_official_not_sealed",
        seed=seed,
        train=tuple(train),
        development=development,
    )


def _verify_shortest_separator(
    machine: QuotientMachine,
    left: int,
    right: int,
    separator: Word,
) -> bool:
    if machine.response(left, separator) == machine.response(right, separator):
        return False
    if not separator:
        return True
    return all(
        machine.response(left, word) == machine.response(right, word)
        for word in words_through_depth(len(separator) - 1)
    )


def audit_machine(machine: QuotientMachine) -> dict[str, object]:
    """Consume one fixture machine and return exact, JSON-safe mechanics evidence."""

    asserted = tuple(machine.key_to_quotient_class)
    refined = partition_refinement(machine)
    product_partition = product_automaton_partition(machine)
    exhaustive = exhaustive_future_partition(machine)
    asserted_relation = equivalence_relation(asserted)
    oracle_agreement = (
        equivalence_relation(refined)
        == equivalence_relation(product_partition)
        == equivalence_relation(exhaustive)
        == asserted_relation
    )
    if not oracle_agreement:
        raise QuotientBoardError(
            f"{machine.name}: independent quotient oracles disagree"
        )

    separators: dict[str, list[int]] = {}
    for left in range(PHYSICAL_STATE_COUNT):
        for right in range(left + 1, PHYSICAL_STATE_COUNT):
            separator = shortest_separator(machine, left, right)
            same_class = asserted[left] == asserted[right]
            if same_class and separator is not None:
                raise QuotientBoardError(
                    f"{machine.name}: equivalent states received a separator"
                )
            if not same_class:
                if separator is None or not _verify_shortest_separator(
                    machine, left, right, separator
                ):
                    raise QuotientBoardError(
                        f"{machine.name}: missing or nonminimal separator"
                    )
                separators[f"{left}:{right}"] = list(separator)

    observer_profiles = tuple(
        {
            "distinct_values": len(set(observer)),
            "nonconstant": len(set(observer)) > 1,
            "noninjective": len(set(observer)) < PHYSICAL_STATE_COUNT,
        }
        for observer in machine.observations
    )
    if not all(
        profile["nonconstant"] and profile["noninjective"]
        for profile in observer_profiles
    ):
        raise QuotientBoardError(
            f"{machine.name}: observers must each be nonconstant and noninjective"
        )

    empty_merge = empty_merge_future_separator(machine)
    equivalent = equivalent_word_witness(machine)
    noncommuting = noncommuting_witness(machine)
    gauge = state_gauge_audit(machine)
    if not all(
        bool(gauge[key])
        for key in (
            "behavior_preserved",
            "class_transport_preserved",
            "structural_signature_preserved",
        )
    ):
        raise QuotientBoardError(f"{machine.name}: state-gauge conjugacy failed")

    observer_do, observer_receipt = observer_intervention(machine)
    observer_left = int(observer_receipt["probe_left"])
    observer_right = int(observer_receipt["probe_right"])
    observer_isolated = (
        observer_do.transitions == machine.transitions
        and shortest_separator(observer_do, observer_left, observer_right) == ()
    )

    action_do, action_receipt = action_intervention(machine)
    probe_state = int(action_receipt["probe_state"])
    action = int(action_receipt["action"])
    action_isolated = (
        action_do.observations == machine.observations
        and action_do.response(probe_state, (action,))
        != machine.response(probe_state, (action,))
    )
    if not observer_isolated or not action_isolated:
        raise QuotientBoardError(f"{machine.name}: intervention isolation failed")

    return {
        "name": machine.name,
        "split": machine.split,
        "quotient_size": machine.quotient_size,
        "key_to_quotient_class": {
            key: class_id
            for key, class_id in zip(
                machine.physical_keys,
                machine.key_to_quotient_class,
                strict=True,
            )
        },
        "structural_signature": structural_signature(machine),
        "observer_profiles": observer_profiles,
        "oracle_agreement": oracle_agreement,
        "partition_refinement": refined,
        "product_automaton": product_partition,
        "exhaustive_future": exhaustive,
        "shortest_separators": separators,
        "maximum_shortest_separator_length": max(map(len, separators.values())),
        "empty_merge_future_separator": empty_merge,
        "equivalent_word_witness": equivalent,
        "noncommuting_witness": noncommuting,
        "state_gauge": gauge,
        "observer_intervention": {
            **observer_receipt,
            "isolated_and_causal": observer_isolated,
        },
        "action_intervention": {
            **action_receipt,
            "isolated_and_causal": action_isolated,
        },
    }


def _machine_payload(machine: QuotientMachine) -> dict[str, object]:
    payload = asdict(machine)
    payload["structural_signature"] = structural_signature(machine)
    return payload


def fixture_digest(fixture: QuotientCPUFixture) -> str:
    payload = {
        "schema": fixture.schema,
        "status": fixture.status,
        "seed": fixture.seed,
        "train": [_machine_payload(machine) for machine in fixture.train],
        "development": [_machine_payload(machine) for machine in fixture.development],
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    return sha256(encoded).hexdigest()


def consume_cpu_fixture(
    fixture: QuotientCPUFixture | None = None,
) -> dict[str, object]:
    """Run every exact gate over the in-memory CPU fixture."""

    active = fixture if fixture is not None else build_cpu_fixture()
    train_audits = tuple(audit_machine(machine) for machine in active.train)
    development_audits = tuple(audit_machine(machine) for machine in active.development)
    train_signatures = {str(report["structural_signature"]) for report in train_audits}
    development_signatures = {
        str(report["structural_signature"]) for report in development_audits
    }
    overlap = sorted(train_signatures & development_signatures)
    quotient_coverage = {
        split: sorted(int(report["quotient_size"]) for report in reports)
        for split, reports in (
            ("train", train_audits),
            ("development", development_audits),
        )
    }
    gates = {
        "cpu_only_unsealed_fixture": (
            active.status == "exploratory_cpu_only_not_official_not_sealed"
        ),
        "eight_physical_keys": all(
            len(machine.physical_keys) == PHYSICAL_STATE_COUNT
            for machine in (*active.train, *active.development)
        ),
        "three_actions": all(
            len(machine.action_names) == 3
            for machine in (*active.train, *active.development)
        ),
        "two_noninjective_observers": all(
            all(profile["noninjective"] for profile in report["observer_profiles"])
            for report in (*train_audits, *development_audits)
        ),
        "quotient_sizes_3_through_8": all(
            sizes == list(QUOTIENT_SIZES) for sizes in quotient_coverage.values()
        ),
        "independent_oracles_agree": all(
            bool(report["oracle_agreement"])
            for report in (*train_audits, *development_audits)
        ),
        "nonempty_future_separators": all(
            len(report["empty_merge_future_separator"]["separator"]) > 0
            for report in (*train_audits, *development_audits)
        ),
        "state_gauge_conjugacy": all(
            bool(report["state_gauge"]["behavior_preserved"])
            and bool(report["state_gauge"]["class_transport_preserved"])
            and bool(report["state_gauge"]["structural_signature_preserved"])
            for report in (*train_audits, *development_audits)
        ),
        "observer_interventions": all(
            bool(report["observer_intervention"]["isolated_and_causal"])
            for report in (*train_audits, *development_audits)
        ),
        "action_interventions": all(
            bool(report["action_intervention"]["isolated_and_causal"])
            for report in (*train_audits, *development_audits)
        ),
        "equivalent_word_witnesses": all(
            report["equivalent_word_witness"]["left"]
            != report["equivalent_word_witness"]["right"]
            for report in (*train_audits, *development_audits)
        ),
        "noncommuting_witnesses": all(
            report["noncommuting_witness"]["left_class"]
            != report["noncommuting_witness"]["right_class"]
            for report in (*train_audits, *development_audits)
        ),
        "zero_train_development_structural_overlap": not overlap,
    }
    failed = sorted(name for name, passed in gates.items() if not passed)
    if failed:
        raise QuotientBoardError(f"CPU fixture failed gates: {failed}")
    return {
        "schema": "episode_functor_quotient_cpu_fixture_audit_v1",
        "status": active.status,
        "fixture_seed": active.seed,
        "fixture_sha256": fixture_digest(active),
        "counts": {
            "train_machines": len(active.train),
            "development_machines": len(active.development),
            "physical_keys_per_machine": PHYSICAL_STATE_COUNT,
            "actions_per_machine": len(ACTION_NAMES),
            "observers_per_machine": len(OBSERVER_NAMES),
        },
        "quotient_coverage": quotient_coverage,
        "train_development_structural_signature_overlap": overlap,
        "gates": gates,
        "train": train_audits,
        "development": development_audits,
        "claims_excluded": (
            "neural_fit",
            "model_capability",
            "gpu_execution",
            "official_sealed_board",
            "pretraining",
        ),
    }


def _json_default(value: object) -> object:
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(f"cannot encode {type(value).__name__}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="consume the exploratory CPU-only quotient mechanics fixture"
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_FIXTURE_SEED)
    parser.add_argument("--compact", action="store_true")
    args = parser.parse_args(argv)
    report = consume_cpu_fixture(build_cpu_fixture(args.seed))
    print(
        json.dumps(
            report,
            default=_json_default,
            indent=None if args.compact else 2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
