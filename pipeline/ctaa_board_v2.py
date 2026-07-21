"""Factorial, shortcut-resistant CTAA board mechanics.

This module defines typed rows and deterministic family generation. It does not
choose a production seed or write artifacts. Long-program outcomes are exposed
only by scored-record methods; compiler-training records contain packet labels
but no recurrent trajectory, terminal state, or answer.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, replace
import hashlib
from itertools import permutations, product
import math
import random
from typing import Iterable, Iterator, Literal, Mapping, Sequence

from pipeline.generate_ctaa_board import (
    ACTION_COUNT,
    MAX_STEPS,
    RENDERERS,
    SCORED_DEPTHS,
    STOP_ID,
    TRAIN_DEPTHS,
    WIDTH,
    CopyMap,
    State,
    apply_copy,
    compose_events,
    compose_maps,
    derive_seed,
    semantic_splits,
)


Axis = Literal["train", "development", "confirmation"]
ProgramClass = Literal[
    "stable_rank_two",
    "implicit_final_collapse",
    "explicit_final_collapse",
]
PROGRAM_CLASSES: tuple[ProgramClass, ...] = (
    "stable_rank_two",
    "implicit_final_collapse",
    "explicit_final_collapse",
)
INITIAL_STATES: tuple[State, ...] = tuple(permutations(range(WIDTH)))  # type: ignore[assignment]
FACTORIAL_BITS = tuple(product((0, 1), repeat=3))
LONG_PER_CLASS_DEPTH_CELL = 576


@dataclass(frozen=True)
class FactorialCell:
    semantic_axis: Axis
    renderer_axis: Axis
    lexical_axis: Axis

    @property
    def tag(self) -> str:
        return "".join(
            "h" if value != "train" else "i"
            for value in (self.semantic_axis, self.renderer_axis, self.lexical_axis)
        )


@dataclass(frozen=True)
class CTAAProgramFamilyV2:
    partition: Axis
    cell: FactorialCell
    family_id: str
    action_cards: tuple[CopyMap, ...]
    initial_state: State
    schedule: tuple[int, ...]
    query_position: int
    program_class: ProgramClass

    def __post_init__(self) -> None:
        if len(self.action_cards) != ACTION_COUNT or len(set(self.action_cards)) != ACTION_COUNT:
            raise ValueError("CTAA v2 action-card geometry differs")
        if len(self.schedule) != MAX_STEPS or self.schedule.count(STOP_ID) != 1:
            raise ValueError("CTAA v2 schedule geometry differs")
        if not 0 < self.depth < MAX_STEPS - 1:
            raise ValueError("CTAA v2 STOP boundary differs")
        if any(event < 0 or event > STOP_ID for event in self.schedule):
            raise ValueError("CTAA v2 schedule leaves event domain")
        if self.program_class not in PROGRAM_CLASSES:
            raise ValueError("CTAA v2 program class differs")
        if self.initial_state not in INITIAL_STATES:
            raise ValueError("CTAA v2 initial state must contain distinct symbols")
        if not 0 <= self.query_position < WIDTH:
            raise ValueError("CTAA v2 query position differs")

    @property
    def depth(self) -> int:
        return self.schedule.index(STOP_ID)

    @property
    def active(self) -> tuple[int, ...]:
        return self.schedule[: self.depth]

    @property
    def composite(self) -> CopyMap:
        return compose_events(self.action_cards, self.active)

    def execute(self, initial: State | None = None) -> tuple[State, ...]:
        state = self.initial_state if initial is None else initial
        states = [state]
        halted = False
        for event in self.schedule:
            if event == STOP_ID:
                halted = True
            elif not halted:
                state = apply_copy(self.action_cards[event], state)
            states.append(state)
        return tuple(states)

    @property
    def terminal_state(self) -> State:
        return self.execute()[-1]

    @property
    def answer(self) -> int:
        return self.terminal_state[self.query_position]

    @property
    def map_deletion_depth(self) -> int:
        return deletion_depth(self.action_cards, self.active, target="map")

    @property
    def state_deletion_depth(self) -> int:
        return deletion_depth(
            self.action_cards,
            self.active,
            target="state",
            initial=self.initial_state,
        )

    @property
    def answer_deletion_depth(self) -> int:
        return deletion_depth(
            self.action_cards,
            self.active,
            target="answer",
            initial=self.initial_state,
            query_position=self.query_position,
        )

    @property
    def shortest_equivalent_length(self) -> int:
        return shortest_word_length(self.action_cards, self.composite)

    @property
    def max_run_length(self) -> int:
        return max_run_length(self.active)

    @property
    def normalized_event_entropy(self) -> float:
        return normalized_event_entropy(self.active)

    @property
    def canonical_key(self) -> tuple[object, ...]:
        return (
            self.action_cards,
            self.initial_state,
            self.active,
            self.query_position,
            self.program_class,
        )


@dataclass(frozen=True)
class CTAASurfaceRowV2:
    family: CTAAProgramFamilyV2
    renderer: int
    program_source: str
    query_source: str

    def compiler_record(self) -> dict[str, object]:
        if self.family.partition != "train":
            raise ValueError("CTAA v2 compiler record partition differs")
        return {
            "family_id": self.family.family_id,
            "program_source": self.program_source,
            "query_source": self.query_source,
            "action_cards": self.family.action_cards,
            "initial_state": self.family.initial_state,
            "schedule": self.family.schedule,
            "query_position": self.family.query_position,
        }

    def scored_record(self) -> dict[str, object]:
        if self.family.partition == "train":
            raise ValueError("CTAA v2 scored record partition differs")
        trace = self.family.execute()
        return {
            "family_id": self.family.family_id,
            "partition": self.family.partition,
            "factorial_cell": self.family.cell.tag,
            "program_class": self.family.program_class,
            "depth": self.family.depth,
            "program_source": self.program_source,
            "query_source": self.query_source,
            "action_cards": self.family.action_cards,
            "initial_state": self.family.initial_state,
            "schedule": self.family.schedule,
            "query_position": self.family.query_position,
            "prefix_states": trace,
            "terminal_state": self.family.terminal_state,
            "answer": self.family.answer,
            "map_deletion_depth": self.family.map_deletion_depth,
            "state_deletion_depth": self.family.state_deletion_depth,
            "answer_deletion_depth": self.family.answer_deletion_depth,
            "shortest_equivalent_length": self.family.shortest_equivalent_length,
            "max_run_length": self.family.max_run_length,
            "normalized_event_entropy": self.family.normalized_event_entropy,
        }


@dataclass(frozen=True)
class AtomicExposure:
    action: CopyMap
    state: State
    context: int
    output: State


@dataclass(frozen=True)
class ClosureExposure:
    first: CopyMap
    second: CopyMap
    state: State
    context: int
    composed: CopyMap
    output: State


@dataclass(frozen=True)
class CTAATwinV2:
    relation: Literal[
        "order_contrast",
        "equivalent_composite",
        "prefix_contrast",
        "card_reindex",
        "post_stop_poison",
        "stop_relocation",
    ]
    parent: CTAAProgramFamilyV2
    child: CTAAProgramFamilyV2


def factorial_cells(partition: Axis) -> tuple[FactorialCell, ...]:
    if partition == "train":
        return (FactorialCell("train", "train", "train"),)
    return tuple(
        FactorialCell(
            partition if semantic else "train",
            partition if renderer else "train",
            partition if lexical else "train",
        )
        for semantic, renderer, lexical in FACTORIAL_BITS
    )


def iter_atomic_exposures(axis: Axis, contexts: int = 64) -> Iterator[AtomicExposure]:
    if contexts < 1:
        raise ValueError("CTAA v2 atomic context count differs")
    for action in semantic_splits()[axis]:
        for state in product(range(WIDTH), repeat=WIDTH):
            typed_state: State = tuple(state)  # type: ignore[assignment]
            for context in range(contexts):
                yield AtomicExposure(action, typed_state, context, apply_copy(action, typed_state))


def train_closed_pairs() -> tuple[tuple[CopyMap, CopyMap], ...]:
    actions = semantic_splits()["train"]
    allowed = set(actions)
    return tuple(
        (first, second)
        for first in actions
        for second in actions
        if compose_maps(second, first) in allowed
    )


def iter_closure_exposures(contexts: int = 64) -> Iterator[ClosureExposure]:
    if contexts < 1:
        raise ValueError("CTAA v2 closure context count differs")
    for first, second in train_closed_pairs():
        composed = compose_maps(second, first)
        for state in product(range(WIDTH), repeat=WIDTH):
            typed_state: State = tuple(state)  # type: ignore[assignment]
            for context in range(contexts):
                yield ClosureExposure(
                    first,
                    second,
                    typed_state,
                    context,
                    composed,
                    apply_copy(composed, typed_state),
                )


def deletion_depth(
    cards: tuple[CopyMap, ...],
    active: tuple[int, ...],
    *,
    target: Literal["map", "state", "answer"],
    initial: State | None = None,
    query_position: int | None = None,
) -> int:
    full_map = compose_events(cards, active)
    if target == "map":
        full: object = full_map
    else:
        if initial is None:
            raise ValueError("CTAA v2 deletion state is missing")
        full_state = apply_copy(full_map, initial)
        full = full_state if target == "state" else full_state[query_position]  # type: ignore[index]
    changed = 0
    for index in range(len(active)):
        reduced_map = compose_events(cards, active[:index] + active[index + 1 :])
        if target == "map":
            reduced: object = reduced_map
        else:
            reduced_state = apply_copy(reduced_map, initial)  # type: ignore[arg-type]
            reduced = reduced_state if target == "state" else reduced_state[query_position]  # type: ignore[index]
        changed += reduced != full
    return changed


def shortest_word_length(cards: tuple[CopyMap, ...], target: CopyMap) -> int:
    identity: CopyMap = (0, 1, 2)
    if target == identity:
        return 0
    queue: deque[tuple[CopyMap, int]] = deque([(identity, 0)])
    seen = {identity}
    while queue:
        current, depth = queue.popleft()
        for card in cards:
            candidate = compose_maps(card, current)
            if candidate == target:
                return depth + 1
            if candidate not in seen:
                seen.add(candidate)
                queue.append((candidate, depth + 1))
    raise ValueError("CTAA v2 target is unreachable from its own cards")


def max_run_length(events: Sequence[int]) -> int:
    longest = current = 0
    previous = None
    for event in events:
        current = current + 1 if event == previous else 1
        previous = event
        longest = max(longest, current)
    return longest


def normalized_event_entropy(events: Sequence[int]) -> float:
    counts = Counter(events)
    if len(counts) <= 1:
        return 0.0
    total = len(events)
    entropy = -sum((count / total) * math.log(count / total) for count in counts.values())
    return entropy / math.log(len(counts))


def _draw_without_long_runs(rng: random.Random, slots: Sequence[int], length: int) -> list[int]:
    events: list[int] = []
    for _ in range(length):
        candidates = list(slots)
        if len(events) >= 3 and len(set(events[-3:])) == 1:
            candidates = [slot for slot in candidates if slot != events[-1]]
        events.append(rng.choice(candidates))
    return events


def _make_active(
    cards: tuple[CopyMap, ...],
    depth: int,
    program_class: ProgramClass,
    rng: random.Random,
) -> tuple[int, ...] | None:
    ranks = [len(set(card)) for card in cards]
    nonconstant = [index for index, rank in enumerate(ranks) if rank > 1]
    constant = [index for index, rank in enumerate(ranks) if rank == 1]
    if depth < 2 or len(nonconstant) < 2:
        return None
    if program_class == "explicit_final_collapse":
        if not constant:
            return None
        prefix = _draw_without_long_runs(rng, nonconstant, depth - 1)
        active = (*prefix, rng.choice(constant))
    else:
        if constant:
            return None
        prefix = _draw_without_long_runs(rng, nonconstant, depth - 1)
        prefix_map = compose_events(cards, prefix)
        final_candidates = [
            slot
            for slot in nonconstant
            if len(set(compose_maps(cards[slot], prefix_map)))
            == (2 if program_class == "stable_rank_two" else 1)
        ]
        if not final_candidates:
            return None
        active = (*prefix, rng.choice(final_candidates))
    composite_rank = len(set(compose_events(cards, active)))
    target_rank = 2 if program_class == "stable_rank_two" else 1
    if composite_rank != target_rank or len(set(active)) < 3:
        return None
    if max_run_length(active) > 3 or normalized_event_entropy(active) < 0.75:
        return None
    map_depth = deletion_depth(cards, active, target="map")
    minimum = max(2, depth // 4)
    if map_depth < minimum:
        return None
    return tuple(active)


def _make_compiler_active(
    cards: tuple[CopyMap, ...],
    depth: int,
    program_class: ProgramClass,
    rng: random.Random,
) -> tuple[int, ...] | None:
    ranks = [len(set(card)) for card in cards]
    nonconstant = [index for index, rank in enumerate(ranks) if rank > 1]
    constants = [index for index, rank in enumerate(ranks) if rank == 1]
    if depth == 1:
        if program_class == "stable_rank_two":
            candidates = [index for index, rank in enumerate(ranks) if rank == 2]
        elif program_class == "explicit_final_collapse":
            candidates = constants
        else:
            return None
        return (rng.choice(candidates),) if candidates else None
    if program_class == "explicit_final_collapse":
        if not constants or not nonconstant:
            return None
        return (*_draw_without_long_runs(rng, nonconstant, depth - 1), rng.choice(constants))
    if constants or len(nonconstant) < 2:
        return None
    for _ in range(256):
        active = tuple(_draw_without_long_runs(rng, nonconstant, depth))
        rank = len(set(compose_events(cards, active)))
        target = 2 if program_class == "stable_rank_two" else 1
        if rank == target:
            return active
    return None


def make_family_v2(
    seed: int,
    partition: Axis,
    cell: FactorialCell,
    family_index: int,
    *,
    program_class: ProgramClass,
    depth: int,
    query_position: int,
    initial_state: State,
) -> CTAAProgramFamilyV2:
    actions = semantic_splits()[cell.semantic_axis]
    for attempt in range(20_000):
        rng = random.Random(
            derive_seed(seed, partition, cell.tag, family_index, program_class, depth, attempt)
        )
        if program_class == "explicit_final_collapse":
            constant = [action for action in actions if len(set(action)) == 1]
            nonconstant = [action for action in actions if len(set(action)) > 1]
            if not constant:
                raise ValueError("CTAA v2 semantic split lacks a constant map")
            cards_list = [constant[0], *rng.sample(nonconstant, 3)]
        else:
            nonconstant = [action for action in actions if len(set(action)) > 1]
            cards_list = rng.sample(nonconstant, ACTION_COUNT)
        rng.shuffle(cards_list)
        cards = tuple(cards_list)
        active = _make_active(cards, depth, program_class, rng)
        if active is None:
            continue
        suffix = tuple(rng.randrange(ACTION_COUNT) for _ in range(MAX_STEPS - depth - 1))
        schedule = (*active, STOP_ID, *suffix)
        return CTAAProgramFamilyV2(
            partition=partition,
            cell=cell,
            family_id=f"{partition[0].upper()}{cell.tag}{family_index:08d}",
            action_cards=cards,
            initial_state=initial_state,
            schedule=schedule,
            query_position=query_position,
            program_class=program_class,
        )
    raise RuntimeError("CTAA v2 constrained family search exhausted")


def make_compiler_family_v2(
    seed: int,
    family_index: int,
    *,
    depth: int,
    program_class: ProgramClass,
    query_position: int,
    initial_state: State,
) -> CTAAProgramFamilyV2:
    cell = FactorialCell("train", "train", "train")
    actions = semantic_splits()["train"]
    for attempt in range(20_000):
        rng = random.Random(
            derive_seed(seed, "compiler", family_index, depth, program_class, attempt)
        )
        if program_class == "explicit_final_collapse":
            constants = [action for action in actions if len(set(action)) == 1]
            others = [action for action in actions if len(set(action)) > 1]
            cards_list = [constants[0], *rng.sample(others, 3)]
        else:
            others = [action for action in actions if len(set(action)) > 1]
            cards_list = rng.sample(others, ACTION_COUNT)
        rng.shuffle(cards_list)
        cards = tuple(cards_list)
        active = _make_compiler_active(cards, depth, program_class, rng)
        if active is None:
            continue
        suffix = tuple(rng.randrange(ACTION_COUNT) for _ in range(MAX_STEPS - depth - 1))
        return CTAAProgramFamilyV2(
            partition="train",
            cell=cell,
            family_id=f"TC{family_index:08d}",
            action_cards=cards,
            initial_state=initial_state,
            schedule=(*active, STOP_ID, *suffix),
            query_position=query_position,
            program_class=program_class,
        )
    raise RuntimeError("CTAA v2 compiler-family search exhausted")


def build_compiler_families(
    seed: int,
    *,
    per_depth: int = 4096,
) -> tuple[CTAAProgramFamilyV2, ...]:
    if per_depth < 1:
        raise ValueError("CTAA v2 compiler family count differs")
    result: list[CTAAProgramFamilyV2] = []
    seen: set[tuple[object, ...]] = set()
    serial = 0
    for depth in TRAIN_DEPTHS:
        classes = (
            ("stable_rank_two", "explicit_final_collapse")
            if depth == 1
            else PROGRAM_CLASSES
        )
        accepted = 0
        candidate = 0
        while accepted < per_depth:
            program_class = classes[accepted % len(classes)]
            initial = INITIAL_STATES[accepted % len(INITIAL_STATES)]
            query = (accepted // len(INITIAL_STATES)) % WIDTH
            family = make_compiler_family_v2(
                seed,
                serial + candidate,
                depth=depth,
                program_class=program_class,  # type: ignore[arg-type]
                query_position=query,
                initial_state=initial,
            )
            candidate += 1
            if family.canonical_key in seen:
                continue
            seen.add(family.canonical_key)
            result.append(family)
            serial += 1
            accepted += 1
    return tuple(result)


def build_long_families(
    seed: int,
    partition: Literal["development", "confirmation"],
    *,
    per_class_depth_cell: int = LONG_PER_CLASS_DEPTH_CELL,
) -> tuple[CTAAProgramFamilyV2, ...]:
    if per_class_depth_cell < 1 or per_class_depth_cell % 288:
        raise ValueError(
            "CTAA v2 long count must jointly balance 16 renderers and 18 query/state cells"
        )
    result: list[CTAAProgramFamilyV2] = []
    seen: set[tuple[object, ...]] = set()
    serial = 0
    for cell in factorial_cells(partition):
        for program_class in PROGRAM_CLASSES:
            for depth in SCORED_DEPTHS:
                accepted = 0
                candidate = 0
                while accepted < per_class_depth_cell:
                    balance_index = accepted
                    initial = INITIAL_STATES[balance_index % len(INITIAL_STATES)]
                    query = (balance_index // len(INITIAL_STATES)) % WIDTH
                    family = make_family_v2(
                        seed,
                        partition,
                        cell,
                        serial + candidate,
                        program_class=program_class,
                        depth=depth,
                        query_position=query,
                        initial_state=initial,
                    )
                    candidate += 1
                    if family.canonical_key in seen:
                        continue
                    seen.add(family.canonical_key)
                    result.append(family)
                    serial += 1
                    accepted += 1
    return tuple(result)


def balanced_renderer_index(family_index: int, per_class_depth_cell: int) -> int:
    """Cross every renderer with every query/initial cell equally per stratum."""
    if (
        family_index < 0
        or per_class_depth_cell < 288
        or per_class_depth_cell % 288
    ):
        raise ValueError("CTAA v2 renderer cross-balance geometry differs")
    within_stratum = family_index % per_class_depth_cell
    return (within_stratum // 18) % 16


def _with_schedule(
    family: CTAAProgramFamilyV2,
    schedule: tuple[int, ...],
    suffix: str,
) -> CTAAProgramFamilyV2:
    return replace(
        family,
        family_id=f"{family.family_id}-{suffix}",
        schedule=schedule,
    )


def make_order_contrast_twin(family: CTAAProgramFamilyV2) -> CTAATwinV2:
    active = family.active
    for index in range(len(active) - 1):
        if active[index] == active[index + 1]:
            continue
        candidate = list(active)
        candidate[index], candidate[index + 1] = candidate[index + 1], candidate[index]
        typed = tuple(candidate)
        if compose_events(family.action_cards, typed) == family.composite:
            continue
        schedule = (*typed, STOP_ID, *family.schedule[family.depth + 1 :])
        child = _with_schedule(family, schedule, "order")
        return CTAATwinV2("order_contrast", family, child)
    raise ValueError("CTAA v2 family has no noncommuting order contrast")


def make_equivalent_composite_twin(
    family: CTAAProgramFamilyV2,
    *,
    seed: int,
) -> CTAATwinV2:
    rng = random.Random(derive_seed(seed, family.family_id, "equivalent"))
    for _ in range(20_000):
        candidate = tuple(_draw_without_long_runs(rng, range(ACTION_COUNT), family.depth))
        if candidate == family.active or len(set(candidate)) < 3:
            continue
        if max_run_length(candidate) > 3 or normalized_event_entropy(candidate) < 0.75:
            continue
        if compose_events(family.action_cards, candidate) != family.composite:
            continue
        schedule = (*candidate, STOP_ID, *family.schedule[family.depth + 1 :])
        child = _with_schedule(family, schedule, "equiv")
        if child.execute() != family.execute():
            return CTAATwinV2("equivalent_composite", family, child)
    raise ValueError("CTAA v2 family has no sampled equivalent-composite twin")


def make_prefix_contrast_twin(family: CTAAProgramFamilyV2) -> CTAATwinV2:
    active = family.active
    boundary = family.depth // 2
    for replacement in range(ACTION_COUNT):
        if replacement == active[boundary]:
            continue
        candidate = (*active[:boundary], replacement, *active[boundary + 1 :])
        if compose_events(family.action_cards, candidate) == family.composite:
            continue
        schedule = (*candidate, STOP_ID, *family.schedule[family.depth + 1 :])
        child = _with_schedule(family, schedule, "prefix")
        parent_trace = family.execute()
        child_trace = child.execute()
        if parent_trace[: boundary + 1] == child_trace[: boundary + 1]:
            return CTAATwinV2("prefix_contrast", family, child)
    raise ValueError("CTAA v2 family has no prefix contrast")


def make_card_reindex_twin(
    family: CTAAProgramFamilyV2,
    permutation: tuple[int, int, int, int] = (2, 0, 3, 1),
) -> CTAATwinV2:
    if set(permutation) != set(range(ACTION_COUNT)):
        raise ValueError("CTAA v2 card permutation differs")
    inverse = {old: new for new, old in enumerate(permutation)}
    cards = tuple(family.action_cards[old] for old in permutation)
    schedule = tuple(
        STOP_ID if event == STOP_ID else inverse[event]
        for event in family.schedule
    )
    child = replace(
        family,
        family_id=f"{family.family_id}-reindex",
        action_cards=cards,
        schedule=schedule,
    )
    if child.execute() != family.execute():
        raise AssertionError("CTAA v2 card reindex changed execution")
    return CTAATwinV2("card_reindex", family, child)


def make_post_stop_poison_twin(family: CTAAProgramFamilyV2) -> CTAATwinV2:
    suffix = tuple(
        (event + 1) % ACTION_COUNT
        for event in family.schedule[family.depth + 1 :]
    )
    if suffix == family.schedule[family.depth + 1 :]:
        raise AssertionError("CTAA v2 poison suffix did not change")
    schedule = (*family.active, STOP_ID, *suffix)
    child = _with_schedule(family, schedule, "poison")
    if child.execute() != family.execute():
        raise AssertionError("CTAA v2 post-STOP poison changed execution")
    return CTAATwinV2("post_stop_poison", family, child)


def make_stop_relocation_twin(family: CTAAProgramFamilyV2) -> CTAATwinV2:
    if family.depth < 2:
        raise ValueError("CTAA v2 STOP relocation requires depth at least two")
    for boundary in range(family.depth - 1, 0, -1):
        dropped = family.active[boundary:]
        suffix = (*dropped, *family.schedule[family.depth + 1 :])
        schedule = (*family.active[:boundary], STOP_ID, *suffix)
        child = _with_schedule(family, schedule, f"stop-{boundary}")
        if child.terminal_state != family.terminal_state:
            return CTAATwinV2("stop_relocation", family, child)
    raise ValueError("CTAA v2 family has no state-sensitive STOP relocation")


_HEADER = (
    "SYMBOL ORDER :: {symbols}",
    "REGISTER ALPHABET = {symbols}",
)
_RULE = (
    "CARD {address}; CODE {opcode}; BEFORE {before}; AFTER {after}",
    "{address} binds {opcode}: {before} => {after}",
)
_START = ("INITIAL STATE :: {state}", "LOAD REGISTERS = {state}")
_TAPE = ("EVENT TAPE :: {events}", "RUN QUEUE = {events}")
_QUERY = ("READ THE {position} CELL.", "REPORT VALUE AT {position}.")
_POSITION = (("FIRST", "SECOND", "THIRD"), ("LEFT", "MIDDLE", "RIGHT"))


def _format_tuple(values: Iterable[str], style: int) -> str:
    values = tuple(values)
    return "[" + ",".join(values) + "]" if style == 0 else "(" + " | ".join(values) + ")"


def render_family_v2(
    seed: int,
    family: CTAAProgramFamilyV2,
    name_pools: Mapping[str, tuple[str, ...]],
    *,
    renderer_index: int,
    reverse_rule_storage: bool = False,
    surface_key: str | None = None,
) -> CTAASurfaceRowV2:
    if set(name_pools) != {"train", "development", "confirmation"}:
        raise ValueError("CTAA v2 production name pools differ")
    renderers = RENDERERS[family.cell.renderer_axis]
    renderer = renderers[renderer_index % len(renderers)]
    pool = name_pools[family.cell.lexical_axis]
    if len(pool) < WIDTH + ACTION_COUNT:
        raise ValueError("CTAA v2 lexical pool is too small")
    rng = random.Random(
        derive_seed(seed, surface_key or family.family_id, renderer_index, "surface")
    )
    selected = rng.sample(pool, WIDTH + ACTION_COUNT)
    symbols = tuple(selected[:WIDTH])
    opcodes = tuple(selected[WIDTH:])
    bits = tuple((renderer >> index) & 1 for index in range(6))
    lines = [_HEADER[bits[0]].format(symbols=_format_tuple(symbols, bits[0]))]
    order = list(range(ACTION_COUNT))
    rng.shuffle(order)
    if reverse_rule_storage:
        order.reverse()
    before = _format_tuple(symbols, bits[1])
    for slot in order:
        action = family.action_cards[slot]
        after = _format_tuple((symbols[index] for index in action), bits[1])
        lines.append(
            _RULE[bits[1]].format(
                address=f"W{slot + 1}",
                opcode=opcodes[slot],
                before=before,
                after=after,
            )
        )
    state = _format_tuple((symbols[index] for index in family.initial_state), bits[2])
    lines.append(_START[bits[2]].format(state=state))
    stop = "STOP" if bits[4] == 0 else "HALT_NOW"
    events = tuple(stop if event == STOP_ID else opcodes[event] for event in family.schedule)
    separator = " ; " if bits[4] == 0 else " / "
    lines.append(_TAPE[bits[3]].format(events=separator.join(events)))
    query = _QUERY[bits[5]].format(position=_POSITION[bits[5]][family.query_position])
    return CTAASurfaceRowV2(
        family=family,
        renderer=renderer,
        program_source="\n".join(lines) + "\n",
        query_source=query + "\n",
    )


def board_contract_counts() -> dict[str, int]:
    return {
        "atomic_optimization_exposures": 9 * 27 * 64,
        "atomic_unique_finite_cases": 9 * 27,
        "closure_optimization_exposures": len(train_closed_pairs()) * 27 * 64,
        "closure_unique_finite_cases": len(train_closed_pairs()) * 27,
        "compiler_schedule_rows": 4096 * len(TRAIN_DEPTHS),
        "long_scored_families_per_partition": (
            len(FACTORIAL_BITS)
            * len(PROGRAM_CLASSES)
            * len(SCORED_DEPTHS)
            * LONG_PER_CLASS_DEPTH_CELL
        ),
    }


def canonical_family_sha256(family: CTAAProgramFamilyV2) -> str:
    return hashlib.sha256(repr(family.canonical_key).encode()).hexdigest()
