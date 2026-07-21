"""Seeded mechanics for a future CTAA neural falsifier board.

This module is seed-capable but does not choose a production seed or write a
board by itself. Training rows expose compiler fields only, never terminal
states or answers.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from itertools import product
import random
from typing import Iterable, Literal


WIDTH = 3
ACTION_COUNT = 4
MAX_STEPS = 41
STOP_ID = ACTION_COUNT
TRAIN_DEPTHS = tuple(range(1, 9))
SCORED_DEPTHS = (16, 32)
Split = Literal["train", "development", "confirmation"]
CopyMap = tuple[int, int, int]
State = tuple[int, int, int]

SPLIT_COUNTS = {split: {1: 1, 2: 6, 3: 2} for split in (
    "train",
    "development",
    "confirmation",
)}
_SEMANTIC_SPLITS: dict[Split, tuple[CopyMap, ...]] = {
    "train": (
        (0, 0, 0), (0, 1, 1), (0, 1, 2), (1, 0, 1), (1, 2, 0),
        (1, 2, 2), (2, 0, 2), (2, 1, 1), (2, 2, 0),
    ),
    "development": (
        (0, 0, 2), (0, 1, 0), (0, 2, 1), (1, 0, 0), (1, 1, 0),
        (1, 1, 2), (2, 0, 1), (2, 2, 1), (2, 2, 2),
    ),
    "confirmation": (
        (0, 0, 1), (0, 2, 0), (0, 2, 2), (1, 0, 2), (1, 1, 1),
        (1, 2, 1), (2, 0, 0), (2, 1, 0), (2, 1, 2),
    ),
}


def _renderer_syndrome(renderer: int) -> tuple[int, int]:
    bits = tuple((renderer >> index) & 1 for index in range(6))
    return (
        bits[0] ^ bits[1] ^ bits[2] ^ bits[3],
        bits[2] ^ bits[3] ^ bits[4] ^ bits[5],
    )


RENDERERS = {
    "train": tuple(value for value in range(64) if _renderer_syndrome(value) == (0, 0)),
    "development": tuple(
        value for value in range(64) if _renderer_syndrome(value) == (0, 1)
    ),
    "confirmation": tuple(
        value for value in range(64) if _renderer_syndrome(value) == (1, 0)
    ),
}


def derive_seed(master: int, *tags: object) -> int:
    payload = "|".join((str(master), *(str(tag) for tag in tags))).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def apply_copy(action: CopyMap, state: State) -> State:
    if len(action) != WIDTH or len(state) != WIDTH:
        raise ValueError("CTAA board tuple geometry differs")
    if any(value < 0 or value >= WIDTH for value in (*action, *state)):
        raise ValueError("CTAA board tuple leaves domain")
    return tuple(state[index] for index in action)  # type: ignore[return-value]


def all_actions() -> tuple[CopyMap, ...]:
    return tuple(product(range(WIDTH), repeat=WIDTH))  # type: ignore[return-value]


def semantic_splits() -> dict[Split, tuple[CopyMap, ...]]:
    result = dict(_SEMANTIC_SPLITS)
    if set(result["train"]) & set(result["development"]):
        raise AssertionError("CTAA semantic train/development overlap")
    if set(result["train"]) & set(result["confirmation"]):
        raise AssertionError("CTAA semantic train/confirmation overlap")
    if set(result["development"]) & set(result["confirmation"]):
        raise AssertionError("CTAA semantic scored-split overlap")
    if set().union(*map(set, result.values())) != set(all_actions()):
        raise AssertionError("CTAA semantic split is not exhaustive")
    return result


def _opaque_name(seed: int, namespace: str, index: int) -> str:
    digest = hashlib.sha256(f"{seed}|{namespace}|{index}".encode()).hexdigest()
    tag = namespace.upper()
    return f"{tag}{digest[:10].upper()}{tag}"


@dataclass(frozen=True)
class CTAAFamily:
    split: Split
    family_id: str
    action_cards: tuple[CopyMap, ...]
    initial_state: State
    schedule: tuple[int, ...]
    query_position: int
    program_class: str

    def __post_init__(self) -> None:
        if len(self.action_cards) != ACTION_COUNT:
            raise ValueError("CTAA family action count differs")
        if len(set(self.action_cards)) != ACTION_COUNT:
            raise ValueError("CTAA family actions are not distinct")
        if len(self.schedule) != MAX_STEPS:
            raise ValueError("CTAA family schedule length differs")
        if self.schedule.count(STOP_ID) != 1:
            raise ValueError("CTAA family requires exactly one STOP")
        if any(event < 0 or event > STOP_ID for event in self.schedule):
            raise ValueError("CTAA family schedule leaves event domain")
        if not 0 <= self.query_position < WIDTH:
            raise ValueError("CTAA family query differs")
        if self.program_class not in {"persistent", "mixed_copy", "absorbing"}:
            raise ValueError("CTAA family program class differs")

    @property
    def depth(self) -> int:
        return self.schedule.index(STOP_ID)

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
    def composite(self) -> CopyMap:
        return compose_events(self.action_cards, self.schedule[: self.depth])

    @property
    def causal_depth(self) -> int:
        active = self.schedule[: self.depth]
        full = compose_events(self.action_cards, active)
        return sum(
            compose_events(self.action_cards, active[:index] + active[index + 1 :])
            != full
            for index in range(len(active))
        )


@dataclass(frozen=True)
class CTAARow:
    split: Split
    family_id: str
    view: int
    renderer: int
    program_source: str
    query_source: str
    action_cards: tuple[CopyMap, ...]
    initial_state: State
    schedule: tuple[int, ...]
    query_position: int
    program_class: str
    causal_depth: int
    terminal_state: State | None
    answer: int | None

    def training_record(self) -> dict[str, object]:
        if self.split != "train" or self.terminal_state is not None or self.answer is not None:
            raise ValueError("CTAA training-record custody differs")
        return {
            "family_id": self.family_id,
            "view": self.view,
            "program_source": self.program_source,
            "query_source": self.query_source,
            "action_cards": self.action_cards,
            "initial_state": self.initial_state,
            "schedule": self.schedule,
            "query_position": self.query_position,
        }


def compose_maps(after: CopyMap, before: CopyMap) -> CopyMap:
    return tuple(before[index] for index in after)  # type: ignore[return-value]


def compose_events(
    cards: tuple[CopyMap, ...],
    events: Iterable[int],
) -> CopyMap:
    composite: CopyMap = (0, 1, 2)
    for event in events:
        if event < 0 or event >= len(cards):
            raise ValueError("CTAA active event leaves card domain")
        composite = compose_maps(cards[event], composite)
    return composite


def _persistent_events(
    cards: tuple[CopyMap, ...],
    depth: int,
    rng: random.Random,
) -> list[int]:
    candidates = [
        index
        for index, action in enumerate(cards)
        if len(set(action)) == WIDTH and action != (0, 1, 2)
    ]
    if not candidates:
        raise ValueError("CTAA persistent family lacks a nonidentity permutation")
    selected = rng.choice(candidates)
    events = [selected] * depth
    full = compose_events(cards, events)
    if not all(
        compose_events(cards, events[:index] + events[index + 1 :]) != full
        for index in range(depth)
    ):
        raise AssertionError("CTAA persistent construction is not deletion-sensitive")
    return events


def _mixed_events(
    cards: tuple[CopyMap, ...],
    depth: int,
    rng: random.Random,
) -> list[int]:
    rank_two = [
        index for index, action in enumerate(cards) if len(set(action)) == 2
    ]
    permutations = [
        index
        for index, action in enumerate(cards)
        if len(set(action)) == WIDTH and action != (0, 1, 2)
    ]
    if len(rank_two) != 2 or not permutations:
        raise ValueError("CTAA mixed family cards differ")
    candidates = [(copy, permutation) for copy in rank_two for permutation in permutations]
    rng.shuffle(candidates)
    for copy, permutation in candidates:
        events = [copy, *([permutation] * (depth - 1))]
        prefixes = [compose_events(cards, events[:end]) for end in range(1, depth + 1)]
        full = prefixes[-1]
        sensitive = sum(
            compose_events(cards, events[:index] + events[index + 1 :]) != full
            for index in range(depth)
        )
        if all(len(set(value)) == 2 for value in prefixes) and sensitive == depth:
            return events
    raise RuntimeError("CTAA mixed schedule construction has no sensitive pair")


def make_family(seed: int, split: Split, family_index: int) -> CTAAFamily:
    if split not in SPLIT_COUNTS:
        raise ValueError("CTAA board split differs")
    rng = random.Random(derive_seed(seed, split, family_index, "family"))
    actions = semantic_splits()[split]
    by_rank = {
        rank: [action for action in actions if len(set(action)) == rank]
        for rank in (1, 2, 3)
    }
    program_class = ("persistent", "mixed_copy", "absorbing")[family_index % 3]
    if program_class == "persistent":
        cards_list = [*by_rank[3], *rng.sample(by_rank[2] + by_rank[1], 2)]
    elif program_class == "mixed_copy":
        cards_list = [*by_rank[3], *rng.sample(by_rank[2], 2)]
    else:
        cards_list = [by_rank[1][0], *rng.sample(by_rank[2] + by_rank[3], 3)]
    rng.shuffle(cards_list)
    cards = tuple(cards_list)
    initial = tuple(rng.randrange(WIDTH) for _ in range(WIDTH))
    if split == "train":
        depth = TRAIN_DEPTHS[family_index % len(TRAIN_DEPTHS)]
    else:
        depth = SCORED_DEPTHS[family_index % len(SCORED_DEPTHS)]
    if program_class == "persistent":
        active = _persistent_events(cards, depth, rng)
    elif program_class == "mixed_copy":
        active = _mixed_events(cards, depth, rng)
    else:
        constant_slot = next(
            index for index, action in enumerate(cards) if len(set(action)) == 1
        )
        nonconstant = [index for index in range(ACTION_COUNT) if index != constant_slot]
        active = [rng.choice(nonconstant) for _ in range(max(0, depth - 2))]
        if depth == 1:
            active = [constant_slot]
        else:
            active.extend((constant_slot, rng.choice(nonconstant)))
    suffix = [rng.randrange(ACTION_COUNT) for _ in range(MAX_STEPS - depth - 1)]
    schedule = tuple((*active, STOP_ID, *suffix))
    return CTAAFamily(
        split=split,
        family_id=f"{split[:1].upper()}{family_index:08d}",
        action_cards=cards,
        initial_state=initial,  # type: ignore[arg-type]
        schedule=schedule,
        query_position=family_index % WIDTH,
        program_class=program_class,
    )


_HEADER = (
    "SYMBOL ORDER :: {symbols}",
    "REGISTER ALPHABET = {symbols}",
)
_RULE = (
    "CARD {address}; CODE {opcode}; BEFORE {before}; AFTER {after}",
    "{address} binds {opcode}: {before} => {after}",
)
_START = (
    "INITIAL STATE :: {state}",
    "LOAD REGISTERS = {state}",
)
_TAPE = (
    "EVENT TAPE :: {events}",
    "RUN QUEUE = {events}",
)
_QUERY = (
    "READ THE {position} CELL.",
    "REPORT VALUE AT {position}.",
)
_POSITION = (
    ("FIRST", "SECOND", "THIRD"),
    ("LEFT", "MIDDLE", "RIGHT"),
)


def _format_tuple(values: Iterable[str], style: int) -> str:
    values = tuple(values)
    if style == 0:
        return "[" + ",".join(values) + "]"
    if style == 1:
        return "(" + " | ".join(values) + ")"
    raise ValueError("CTAA tuple renderer differs")


def render_row(
    seed: int,
    family: CTAAFamily,
    view: int,
    *,
    force_renderer: int | None = None,
    reverse_rule_storage: bool = False,
    name_pools: dict[str, tuple[str, ...]] | None = None,
) -> CTAARow:
    renderers = RENDERERS[family.split]
    renderer_offset = derive_seed(seed, family.family_id, "renderer") % len(renderers)
    renderer = (
        renderers[(renderer_offset + view) % len(renderers)]
        if force_renderer is None
        else force_renderer
    )
    if renderer not in renderers:
        raise ValueError("CTAA renderer crosses split custody")
    name_seed = derive_seed(seed, family.family_id, view, "names")
    if name_pools is None:
        symbols = tuple(
            _opaque_name(name_seed, f"{family.split}v", index)
            for index in range(WIDTH)
        )
        opcodes = tuple(
            _opaque_name(name_seed, f"{family.split}op", index)
            for index in range(ACTION_COUNT)
        )
    else:
        if set(name_pools) != {"train", "development", "confirmation"}:
            raise ValueError("CTAA production name-pool splits differ")
        pool = name_pools[family.split]
        if len(pool) < WIDTH + ACTION_COUNT:
            raise ValueError("CTAA production name pool is too small")
        name_rng = random.Random(derive_seed(name_seed, "pool-assignment"))
        selected = name_rng.sample(pool, WIDTH + ACTION_COUNT)
        symbols = tuple(selected[:WIDTH])
        opcodes = tuple(selected[WIDTH:])
    bits = tuple((renderer >> index) & 1 for index in range(6))
    lines = [_HEADER[bits[0]].format(symbols=_format_tuple(symbols, bits[0]))]
    rule_order = list(range(ACTION_COUNT))
    random.Random(derive_seed(name_seed, "storage")).shuffle(rule_order)
    if reverse_rule_storage:
        rule_order.reverse()
    before = _format_tuple(symbols, bits[1])
    for slot in rule_order:
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
    stop_name = "STOP" if bits[4] == 0 else "HALT_NOW"
    event_names = tuple(
        stop_name if event == STOP_ID else opcodes[event]
        for event in family.schedule
    )
    separator = " ; " if bits[4] == 0 else " / "
    lines.append(_TAPE[bits[3]].format(events=separator.join(event_names)))
    query_source = _QUERY[bits[5]].format(
        position=_POSITION[bits[5]][family.query_position]
    )
    scored = family.split != "train"
    return CTAARow(
        split=family.split,
        family_id=family.family_id,
        view=view,
        renderer=renderer,
        program_source="\n".join(lines) + "\n",
        query_source=query_source + "\n",
        action_cards=family.action_cards,
        initial_state=family.initial_state,
        schedule=family.schedule,
        query_position=family.query_position,
        program_class=family.program_class,
        causal_depth=family.causal_depth,
        terminal_state=family.terminal_state if scored else None,
        answer=family.answer if scored else None,
    )


def build_rows(
    seed: int,
    split: Split,
    families: int,
    *,
    views: int = 4,
    name_pools: dict[str, tuple[str, ...]] | None = None,
) -> tuple[CTAARow, ...]:
    if families < 1 or views < 1:
        raise ValueError("CTAA board size differs")
    return tuple(
        render_row(
            seed,
            make_family(seed, split, family_index),
            view,
            name_pools=name_pools,
        )
        for family_index in range(families)
        for view in range(views)
    )
