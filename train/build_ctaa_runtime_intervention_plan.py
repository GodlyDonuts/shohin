#!/usr/bin/env python3
"""Build the frozen, oracle-blind CTAA runtime-intervention sidecar plan."""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
from typing import Mapping, Sequence

from tokenizers import Tokenizer

from ctaa_bootstrap_seed_receipt import validate_receipt as validate_seed_receipt
from ctaa_intervention_protocol import (
    LOCKED_SCORED_ROW_COUNT,
    OPERATION_SPECS,
    RUNTIME_PANEL_SIZE,
    AnchorBinding,
    DonorConstraint,
    DonorPair,
    GateFamily,
    InterventionFamily,
    Partition,
    PlanBindings,
    RuntimeInterventionPlan,
    anchor_to_dict,
    make_anchor_operation_commitment,
    make_donor_derangement,
    make_runtime_intervention_plan,
    operation_semantics_sha256,
    plan_to_dict,
    validate_runtime_intervention_plan,
)
from ctaa_run_contract import (
    ARMS,
    BOARD_TREE_SCHEMA,
    RUN_CONTRACT_SCHEMA,
    _validate_contract_shape,
    canonical_json,
    canonical_sha256,
)
from pipeline.ctaa_board_v2 import INITIAL_STATES, PROGRAM_CLASSES
from pipeline.generate_ctaa_board import (
    ACTION_COUNT,
    MAX_STEPS,
    RENDERERS,
    SCORED_DEPTHS,
    STOP_ID,
    compose_events,
)


MANIFEST_SCHEMA = "r12_ctaa_v2_manifest_v2"
MASK_SCHEMA = "r12_ctaa_v2_fixed_right_padding_mask_v1"
MIDPOINT_SUFFIX_SCHEMA = "r12_ctaa_v2_midpoint_suffix_geometry_v1"
RUNTIME_IMPLEMENTATION_SCHEMA = "r12_ctaa_v2_runtime_implementation_v2"
RUNTIME_IMPLEMENTATION_SOURCES = (
    "build_ctaa_runtime_intervention_plan.py",
    "ctaa_intervention_protocol.py",
    "ctaa_runtime_interventions.py",
    "ctaa_trunk_compiler.py",
    "ctaa_neural_core.py",
    "ctaa_packet_io.py",
    "ctaa_runtime_plan_replay.py",
    "ctaa_runtime_execution_projection.py",
    "ctaa_runtime_program_artifacts.py",
    "ctaa_runtime_execution_engine.py",
    "ctaa_runtime_execution_artifact.py",
    "ctaa_runtime_execution_receipt.py",
    "ctaa_runtime_evidence_finalizer.py",
    "ctaa_runtime_execution_set.py",
    "ctaa_runtime_evidence.py",
    "ctaa_runtime_bundle.py",
)
SEQUENCE_LIMIT = 2048
ROWS_PER_RENDERER_QUERY_CELL = 2
ROWS_PER_CLASS_DEPTH = 576
HHH_ROWS = len(PROGRAM_CLASSES) * len(SCORED_DEPTHS) * ROWS_PER_CLASS_DEPTH
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
FAMILY_ID = re.compile(r"(?P<prefix>[DC])(?P<cell>[ih]{3})(?P<serial>[0-9]{8})\Z")


class RuntimePlanBuildError(ValueError):
    """A frozen runtime-plan input or deterministic construction is invalid."""


@dataclass(frozen=True)
class ParsedSource:
    family_id: str
    program_source: str
    query_source: str
    class_id: str
    depth: int
    renderer_index: int
    renderer_value: int
    query_state_cell_id: str
    query_position: int
    symbols: tuple[str, str, str]
    opcodes: tuple[str, str, str, str]
    cards: tuple[tuple[int, int, int], ...]
    opcode_to_card: tuple[int, int, int, int]
    initial_state: tuple[int, int, int]
    opcode_schedule: tuple[int, ...]
    schedule: tuple[int, ...]
    rule_order: tuple[int, ...]
    packet_bytes: bytes
    token_count: int
    midpoint_suffix_bytes: bytes
    midpoint_state_bytes: bytes
    midpoint_action_bytes: bytes


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _tagged_sha256(schema: str, value: object) -> str:
    return _sha256_bytes(
        canonical_json({"schema": schema, "value": value}).encode("ascii")
    )


def _runtime_implementation_sha256() -> str:
    root = Path(__file__).resolve().parent
    sources = {}
    for name in RUNTIME_IMPLEMENTATION_SOURCES:
        sources[name] = _sha256_bytes((root / name).read_bytes())
    return canonical_sha256(
        {"schema": RUNTIME_IMPLEMENTATION_SCHEMA, "sources": sources}
    )


def _hash_order(seed: int, *values: object) -> bytes:
    encoded = canonical_json([seed, *values]).encode("ascii")
    return hashlib.sha256(b"SHOHIN-CTAA-RUNTIME-PLAN-v1\0" + encoded).digest()


def _permutation(
    seed: int, operation: str, anchor_id: str, size: int
) -> tuple[int, ...]:
    ordered = tuple(
        sorted(
            range(size),
            key=lambda index: _hash_order(
                seed, "permutation", operation, anchor_id, index
            ),
        )
    )
    if ordered == tuple(range(size)):
        ordered = ordered[1:] + ordered[:1]
    return ordered


def _three_cycle(
    seed: int,
    operation: str,
    anchor_id: str,
    required_moved_slot: int | None = None,
) -> tuple[int, int, int, int]:
    """Return a deterministic non-involutive old-slot to new-slot map."""

    selected = _permutation(seed, operation, anchor_id, ACTION_COUNT)
    if required_moved_slot is None:
        first, second, third = selected[:3]
    else:
        if not 0 <= required_moved_slot < ACTION_COUNT:
            raise RuntimePlanBuildError("CTAA required three-cycle slot differs")
        first = required_moved_slot
        second, third = tuple(
            slot for slot in selected if slot != required_moved_slot
        )[:2]
    old_to_new = list(range(ACTION_COUNT))
    old_to_new[first] = second
    old_to_new[second] = third
    old_to_new[third] = first
    return tuple(old_to_new)  # type: ignore[return-value]


def _opaque_names(
    seed: int, operation: str, anchor_id: str, prefix: str, size: int
) -> tuple[str, ...]:
    return tuple(
        f"{prefix}{_hash_order(seed, operation, anchor_id, index).hex()[:12]}"
        for index in range(size)
    )


def _format_tuple(values: Sequence[str], style: int) -> str:
    if style == 0:
        return "[" + ",".join(values) + "]"
    return "(" + " | ".join(values) + ")"


def _render_program(
    row: ParsedSource,
    *,
    symbols: tuple[str, str, str] | None = None,
    opcodes: tuple[str, str, str, str] | None = None,
    addresses: tuple[str, str, str, str] | None = None,
    cards: tuple[tuple[int, int, int], ...] | None = None,
    schedule: tuple[int, ...] | None = None,
    renderer_value: int | None = None,
    rule_order: tuple[int, ...] | None = None,
) -> str:
    symbols = row.symbols if symbols is None else symbols
    opcodes = row.opcodes if opcodes is None else opcodes
    addresses = ("W1", "W2", "W3", "W4") if addresses is None else addresses
    cards = row.cards if cards is None else cards
    schedule = row.schedule if schedule is None else schedule
    renderer = row.renderer_value if renderer_value is None else renderer_value
    order = row.rule_order if rule_order is None else rule_order
    bits = tuple((renderer >> index) & 1 for index in range(6))
    lines = [
        ("SYMBOL ORDER :: " if bits[0] == 0 else "REGISTER ALPHABET = ")
        + _format_tuple(symbols, bits[0])
    ]
    before = _format_tuple(symbols, bits[1])
    for slot in order:
        after = _format_tuple(tuple(symbols[index] for index in cards[slot]), bits[1])
        if bits[1] == 0:
            lines.append(
                f"CARD {addresses[slot]}; CODE {opcodes[slot]}; BEFORE {before}; AFTER {after}"
            )
        else:
            lines.append(
                f"{addresses[slot]} binds {opcodes[slot]}: {before} => {after}"
            )
    initial = tuple(symbols[index] for index in row.initial_state)
    lines.append(
        ("INITIAL STATE :: " if bits[2] == 0 else "LOAD REGISTERS = ")
        + _format_tuple(initial, bits[2])
    )
    stop_name = "STOP" if bits[4] == 0 else "HALT_NOW"
    names = [stop_name if event == STOP_ID else opcodes[event] for event in schedule]
    separator = " ; " if bits[4] == 0 else " / "
    lines.append(
        ("EVENT TAPE :: " if bits[3] == 0 else "RUN QUEUE = ") + separator.join(names)
    )
    return "\n".join(lines) + "\n"


def _packet_bytes(
    cards: Sequence[Sequence[int]],
    initial: Sequence[int],
    schedule: Sequence[int],
    opcode_to_card: Sequence[int],
) -> bytes:
    binding = tuple(int(value) for value in opcode_to_card)
    if sorted(binding) != list(range(ACTION_COUNT)):
        raise RuntimePlanBuildError("CTAA packet binding is not a permutation")
    inverse = {card: opcode for opcode, card in enumerate(binding)}
    opcode_schedule = tuple(
        STOP_ID if event == STOP_ID else inverse[int(event)]
        for event in schedule
    )
    return bytes(
        [value for card in cards for value in card]
        + list(binding)
        + list(initial)
        + list(opcode_schedule)
    )


def _packet_bytes_from_local(
    cards: Sequence[Sequence[int]],
    opcode_to_card: Sequence[int],
    initial: Sequence[int],
    opcode_schedule: Sequence[int],
) -> bytes:
    binding = tuple(int(value) for value in opcode_to_card)
    local_tape = tuple(int(value) for value in opcode_schedule)
    if sorted(binding) != list(range(ACTION_COUNT)):
        raise RuntimePlanBuildError("CTAA packet binding is not a permutation")
    if len(local_tape) != MAX_STEPS or any(
        value < 0 or value > STOP_ID for value in local_tape
    ):
        raise RuntimePlanBuildError("CTAA packet local tape differs")
    return bytes(
        [value for card in cards for value in card]
        + list(binding)
        + list(initial)
        + list(local_tape)
    )


def _json_object(pairs: Sequence[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise RuntimePlanBuildError(f"CTAA JSON contains duplicate key {key!r}")
        result[key] = value
    return result


def _decode_json(data: bytes, label: str) -> dict[str, object]:
    try:
        value = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_json_object,
            parse_constant=lambda item: (_ for _ in ()).throw(
                RuntimePlanBuildError(f"CTAA {label} contains non-finite {item}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimePlanBuildError(f"CTAA {label} JSON differs") from error
    if not isinstance(value, dict):
        raise RuntimePlanBuildError(f"CTAA {label} JSON root differs")
    return value


def _read_locked_bytes(path: Path, label: str) -> bytes:
    """Read a single-link read-only regular file once without following symlinks."""

    try:
        metadata = path.lstat()
    except OSError as error:
        raise RuntimePlanBuildError(f"CTAA {label} is unavailable") from error
    if (
        not stat.S_ISREG(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or metadata.st_mode & 0o222
        or metadata.st_nlink != 1
    ):
        raise RuntimePlanBuildError(
            f"CTAA {label} is not a single-link read-only regular file"
        )
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise RuntimePlanBuildError(f"CTAA {label} cannot be opened safely") from error
    try:
        before = os.fstat(descriptor)
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if (
        (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        != (
            metadata.st_dev,
            metadata.st_ino,
            metadata.st_size,
            metadata.st_mtime_ns,
            metadata.st_ctime_ns,
        )
        or (after.st_size, after.st_mtime_ns, after.st_ctime_ns)
        != (before.st_size, before.st_mtime_ns, before.st_ctime_ns)
        or after.st_mode & 0o222
        or after.st_nlink != 1
    ):
        raise RuntimePlanBuildError(f"CTAA {label} changed while being read")
    return b"".join(chunks)


def _read_jsonl(
    data: bytes, label: str, keys: frozenset[str]
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line_number, line in enumerate(data.splitlines(), 1):
        if not line.strip():
            raise RuntimePlanBuildError(f"CTAA {label} row {line_number} is empty")
        row = _decode_json(line, f"{label} row {line_number}")
        if set(row) != keys:
            raise RuntimePlanBuildError(
                f"CTAA {label} row {line_number} schema differs"
            )
        family_id = row.get("family_id")
        source_key = next(key for key in keys if key != "family_id")
        if (
            not isinstance(family_id, str)
            or not family_id
            or not isinstance(row.get(source_key), str)
            or not row[source_key]
        ):
            raise RuntimePlanBuildError(f"CTAA {label} row {line_number} value differs")
        rows.append(row)
    if not rows or len({str(row["family_id"]) for row in rows}) != len(rows):
        raise RuntimePlanBuildError(f"CTAA {label} family identities differ")
    return rows


def _require_hash(value: object, label: str) -> str:
    if not isinstance(value, str) or HEX64.fullmatch(value) is None:
        raise RuntimePlanBuildError(f"CTAA {label} is not a canonical SHA-256")
    return value


def _safe_name(value: object) -> str:
    if not isinstance(value, str) or not value or len(value) > 255:
        raise RuntimePlanBuildError("CTAA manifest filename differs")
    pure = PurePosixPath(value)
    if pure.name != value or value in {".", ".."} or "\x00" in value:
        raise RuntimePlanBuildError("CTAA manifest filename is unsafe")
    return value


def _manifest_bindings(
    raw: bytes,
) -> tuple[dict[str, object], str, str, dict[str, str]]:
    manifest = _decode_json(raw, "board manifest")
    if set(manifest) != {"schema", "seed", "files"}:
        raise RuntimePlanBuildError("CTAA board manifest schema differs")
    if (
        manifest.get("schema") != MANIFEST_SCHEMA
        or type(manifest.get("seed")) is not int
        or int(manifest["seed"]) < 0
        or not isinstance(manifest.get("files"), dict)
    ):
        raise RuntimePlanBuildError("CTAA board manifest identity differs")
    files: dict[str, str] = {}
    for name, digest in manifest["files"].items():  # type: ignore[union-attr]
        safe = _safe_name(name)
        files[safe] = _require_hash(digest, f"manifest file {safe}")
    if not files or len(files) != len(manifest["files"]):  # type: ignore[arg-type]
        raise RuntimePlanBuildError("CTAA board manifest file map differs")
    tree = {
        "schema": BOARD_TREE_SCHEMA,
        "files": [{"name": name, "sha256": files[name]} for name in sorted(files)],
    }
    return manifest, _sha256_bytes(raw), canonical_sha256(tree), files


def _tuple_values(value: str) -> tuple[str, str, str]:
    value = value.strip()
    if value.startswith("[") and value.endswith("]"):
        parts = value[1:-1].split(",")
    elif value.startswith("(") and value.endswith(")"):
        parts = value[1:-1].split("|")
    else:
        raise RuntimePlanBuildError("CTAA source tuple syntax differs")
    normalized = tuple(part.strip() for part in parts)
    if len(normalized) != 3 or any(not item for item in normalized):
        raise RuntimePlanBuildError("CTAA source tuple geometry differs")
    return normalized  # type: ignore[return-value]


def _split_once(line: str, options: Sequence[str], label: str) -> tuple[int, str]:
    matches = [
        (index, line[len(prefix) :])
        for index, prefix in enumerate(options)
        if line.startswith(prefix)
    ]
    if len(matches) != 1:
        raise RuntimePlanBuildError(f"CTAA {label} syntax differs")
    return matches[0]


def _parse_program(
    family_id: str,
    program_source: str,
    query_source: str,
    tokenizer: Tokenizer,
    partition: str,
) -> ParsedSource:
    match = FAMILY_ID.fullmatch(family_id)
    expected_prefix = "D" if partition == "development" else "C"
    if match is None or match.group("prefix") != expected_prefix:
        raise RuntimePlanBuildError("CTAA source family identity differs")
    lines = program_source.splitlines()
    if len(lines) != 7 or not program_source.endswith("\n"):
        raise RuntimePlanBuildError("CTAA program source line geometry differs")

    header_bit, raw_symbols = _split_once(
        lines[0], ("SYMBOL ORDER :: ", "REGISTER ALPHABET = "), "source header"
    )
    symbols = _tuple_values(raw_symbols)
    if len(set(symbols)) != 3:
        raise RuntimePlanBuildError("CTAA source symbol namespace differs")
    symbol_index = {symbol: index for index, symbol in enumerate(symbols)}

    cards: list[tuple[int, int, int] | None] = [None] * ACTION_COUNT
    opcodes: list[str | None] = [None] * ACTION_COUNT
    rule_order: list[int] = []
    rule_bits: set[int] = set()
    for line in lines[1:5]:
        if line.startswith("CARD "):
            rule_bits.add(0)
            found = re.fullmatch(
                r"CARD W([1-4]); CODE ([^;\s]+); BEFORE (.+); AFTER (.+)", line
            )
        else:
            rule_bits.add(1)
            found = re.fullmatch(r"W([1-4]) binds ([^:\s]+): (.+) => (.+)", line)
        if found is None:
            raise RuntimePlanBuildError("CTAA source rule syntax differs")
        slot = int(found.group(1)) - 1
        opcode = found.group(2)
        before = _tuple_values(found.group(3))
        after = _tuple_values(found.group(4))
        if before != symbols or any(item not in symbol_index for item in after):
            raise RuntimePlanBuildError("CTAA source rule binding differs")
        if cards[slot] is not None or opcode in opcodes:
            raise RuntimePlanBuildError("CTAA source rule identity repeats")
        cards[slot] = tuple(symbol_index[item] for item in after)  # type: ignore[assignment]
        opcodes[slot] = opcode
        rule_order.append(slot)
    if len(rule_bits) != 1 or any(item is None for item in cards + opcodes):
        raise RuntimePlanBuildError("CTAA source rule set differs")

    start_bit, raw_initial = _split_once(
        lines[5], ("INITIAL STATE :: ", "LOAD REGISTERS = "), "initial state"
    )
    initial_names = _tuple_values(raw_initial)
    if any(item not in symbol_index for item in initial_names):
        raise RuntimePlanBuildError("CTAA source initial state differs")
    initial = tuple(symbol_index[item] for item in initial_names)
    if initial not in INITIAL_STATES:
        raise RuntimePlanBuildError("CTAA source initial state is not a permutation")

    tape_bit, raw_tape = _split_once(
        lines[6], ("EVENT TAPE :: ", "RUN QUEUE = "), "event tape"
    )
    if " ; " in raw_tape and " / " not in raw_tape:
        stop_bit = 0
        separator = " ; "
    elif " / " in raw_tape and " ; " not in raw_tape:
        stop_bit = 1
        separator = " / "
    else:
        raise RuntimePlanBuildError("CTAA source event separator differs")
    event_names = raw_tape.split(separator)
    local_opcodes = tuple(str(opcodes[card]) for card in rule_order)
    opcode_index = {opcode: index for index, opcode in enumerate(local_opcodes)}
    stop_names = [name for name in event_names if name in {"STOP", "HALT_NOW"}]
    if len(event_names) != MAX_STEPS or len(stop_names) != 1:
        raise RuntimePlanBuildError("CTAA source event geometry differs")
    stop_name = stop_names[0]
    observed_stop_bit = 0 if stop_name == "STOP" else 1
    if stop_bit != observed_stop_bit:
        raise RuntimePlanBuildError("CTAA source STOP renderer bit differs")
    opcode_schedule = tuple(
        STOP_ID if name == stop_name else opcode_index.get(name, -1)
        for name in event_names
    )
    if any(event < 0 for event in opcode_schedule):
        raise RuntimePlanBuildError("CTAA source event opcode differs")
    binding = tuple(rule_order)
    schedule = tuple(
        STOP_ID if event == STOP_ID else binding[event]
        for event in opcode_schedule
    )
    depth = schedule.index(STOP_ID)
    if depth not in SCORED_DEPTHS:
        raise RuntimePlanBuildError("CTAA source scored depth differs")

    query_lines = query_source.splitlines()
    if len(query_lines) != 1 or not query_source.endswith("\n"):
        raise RuntimePlanBuildError("CTAA query source line geometry differs")
    query_bit, raw_position = _split_once(
        query_lines[0], ("READ THE ", "REPORT VALUE AT "), "query"
    )
    suffix = " CELL." if query_bit == 0 else "."
    if not raw_position.endswith(suffix):
        raise RuntimePlanBuildError("CTAA query source syntax differs")
    position_name = raw_position[: -len(suffix)]
    positions = (
        ("FIRST", "SECOND", "THIRD") if query_bit == 0 else ("LEFT", "MIDDLE", "RIGHT")
    )
    if position_name not in positions:
        raise RuntimePlanBuildError("CTAA query position differs")
    query_position = positions.index(position_name)

    renderer_value = (
        header_bit
        | (next(iter(rule_bits)) << 1)
        | (start_bit << 2)
        | (tape_bit << 3)
        | (stop_bit << 4)
        | (query_bit << 5)
    )
    try:
        renderer_index = RENDERERS[partition].index(renderer_value)  # type: ignore[index]
    except ValueError as error:
        raise RuntimePlanBuildError(
            "CTAA source renderer is outside partition"
        ) from error

    typed_cards = tuple(cards)  # type: ignore[arg-type]
    active = schedule[:depth]
    composite_rank = len(set(compose_events(typed_cards, active)))
    if len(set(typed_cards[active[-1]])) == 1:
        class_id = "explicit_final_collapse"
    elif composite_rank == 1:
        class_id = "implicit_final_collapse"
    elif composite_rank == 2:
        class_id = "stable_rank_two"
    else:
        raise RuntimePlanBuildError("CTAA source program class differs")

    initial_index = INITIAL_STATES.index(initial)  # type: ignore[arg-type]
    query_state_cell = f"q{query_position}-s{initial_index}"
    token_ids = tuple(tokenizer.encode(program_source).ids)
    if not token_ids or len(token_ids) > SEQUENCE_LIMIT:
        raise RuntimePlanBuildError("CTAA source tokenizer geometry differs")
    packet_bytes = _packet_bytes(typed_cards, initial, schedule, binding)
    midpoint = depth // 2
    state = initial
    for event in active[:midpoint]:
        card = typed_cards[event]
        state = tuple(state[index] for index in card)
    suffix_geometry = [
        0 if event == STOP_ID else (1 if index < depth else 2)
        for index, event in enumerate(schedule[midpoint:], midpoint)
    ]
    return ParsedSource(
        family_id=family_id,
        program_source=program_source,
        query_source=query_source,
        class_id=class_id,
        depth=depth,
        renderer_index=renderer_index,
        renderer_value=renderer_value,
        query_state_cell_id=query_state_cell,
        query_position=query_position,
        symbols=symbols,
        opcodes=tuple(opcodes),  # type: ignore[arg-type]
        cards=typed_cards,
        opcode_to_card=binding,  # type: ignore[arg-type]
        initial_state=initial,  # type: ignore[arg-type]
        opcode_schedule=opcode_schedule,
        schedule=schedule,
        rule_order=tuple(rule_order),
        packet_bytes=packet_bytes,
        token_count=len(token_ids),
        midpoint_suffix_bytes=canonical_json(
            {
                "schema": MIDPOINT_SUFFIX_SCHEMA,
                "midpoint": midpoint,
                "suffix_geometry": suffix_geometry,
            }
        ).encode("ascii"),
        midpoint_state_bytes=bytes(state),
        midpoint_action_bytes=bytes(typed_cards[active[midpoint]]),
    )


def _select_panel(
    candidates: Sequence[ParsedSource], seed: int, partition: Partition
) -> tuple[AnchorBinding, ...]:
    by_stratum: dict[tuple[str, int], list[ParsedSource]] = {}
    for row in candidates:
        by_stratum.setdefault((row.class_id, row.depth), []).append(row)
    expected_strata = {
        (class_id, depth) for class_id in PROGRAM_CLASSES for depth in SCORED_DEPTHS
    }
    if set(by_stratum) != expected_strata:
        raise RuntimePlanBuildError("CTAA hhh class-depth strata differ")

    for attempt in range(4096):
        selected: list[ParsedSource] = []
        valid_geometry = True
        for class_id, depth in sorted(expected_strata):
            rows = by_stratum[(class_id, depth)]
            if len(rows) != ROWS_PER_CLASS_DEPTH:
                raise RuntimePlanBuildError("CTAA hhh class-depth row count differs")
            cells: dict[tuple[int, str], list[ParsedSource]] = {}
            for row in rows:
                cells.setdefault(
                    (row.renderer_index, row.query_state_cell_id), []
                ).append(row)
            if len(cells) != 16 * 18 or any(
                len(values) != ROWS_PER_RENDERER_QUERY_CELL for values in cells.values()
            ):
                raise RuntimePlanBuildError(
                    "CTAA hhh renderer/query cross geometry differs"
                )
            query_cells = sorted({cell for _, cell in cells})
            renderers = sorted({renderer for renderer, _ in cells})
            renderer_order = sorted(
                renderers,
                key=lambda value: _hash_order(
                    seed, "renderer", attempt, class_id, depth, value
                ),
            )
            cell_order = sorted(
                query_cells,
                key=lambda value: _hash_order(
                    seed, "query-cell", attempt, class_id, depth, value
                ),
            )
            renderer_rank = {value: index for index, value in enumerate(renderer_order)}
            cell_rank = {value: index for index, value in enumerate(cell_order)}
            chosen_cells = [
                key
                for key in cells
                if (renderer_rank[key[0]] + cell_rank[key[1]]) % 2 == 0
            ]
            if len(chosen_cells) != 144:
                raise AssertionError("CTAA biregular panel construction differs")
            for renderer, cell in chosen_cells:
                options = sorted(
                    cells[(renderer, cell)],
                    key=lambda row: _hash_order(
                        seed,
                        "cell-choice",
                        attempt,
                        class_id,
                        depth,
                        renderer,
                        cell,
                        row.family_id,
                    ),
                )
                selected.append(options[0])
        anchors = tuple(
            sorted(
                (
                    AnchorBinding(
                        anchor_id=f"anchor-{row.family_id}",
                        family_id=row.family_id,
                        class_id=row.class_id,
                        depth=row.depth,
                        shift_cell="hhh",
                        renderer_index=row.renderer_index,
                        query_state_cell_id=row.query_state_cell_id,
                        query_position=row.query_position,
                        partition=partition,
                        program_source_sha256=_sha256_bytes(
                            row.program_source.encode("utf-8")
                        ),
                        query_source_sha256=_sha256_bytes(
                            row.query_source.encode("utf-8")
                        ),
                        packet_sha256=_sha256_bytes(row.packet_bytes),
                        padding_mask_sha256=_tagged_sha256(
                            MASK_SCHEMA,
                            [1] * row.token_count
                            + [0] * (SEQUENCE_LIMIT - row.token_count),
                        ),
                        midpoint_suffix_sha256=_sha256_bytes(row.midpoint_suffix_bytes),
                        midpoint_state_sha256=_sha256_bytes(row.midpoint_state_bytes),
                        midpoint_action_sha256=_sha256_bytes(row.midpoint_action_bytes),
                        action_card_sha256s=tuple(
                            _sha256_bytes(row.packet_bytes[index : index + 3])
                            for index in range(0, ACTION_COUNT * 3, 3)
                        ),
                    )
                    for row in selected
                ),
                key=lambda row: (
                    row.class_id,
                    row.depth,
                    row.renderer_index,
                    row.query_state_cell_id,
                    row.anchor_id,
                ),
            )
        )
        if valid_geometry and _all_donor_matchings_exist(anchors, seed):
            return anchors
    raise RuntimePlanBuildError(
        "CTAA deterministic balanced panel has no donor-feasible draw"
    )


def _pair_allowed(
    constraint: DonorConstraint, parent: AnchorBinding, donor: AnchorBinding
) -> bool:
    if parent.anchor_id == donor.anchor_id:
        return False
    midpoint_constraint = constraint in {
        DonorConstraint.MIDPOINT_STATE_DIFFERENT_MATCHED_SUFFIX,
        DonorConstraint.MIDPOINT_ACTION_DIFFERENT_MATCHED_SUFFIX,
    }
    if parent.depth != donor.depth or (
        not midpoint_constraint and parent.class_id != donor.class_id
    ):
        return False
    if constraint is DonorConstraint.RESIDUAL_EXACT_MASK:
        return (
            parent.padding_mask_sha256 == donor.padding_mask_sha256
            and parent.packet_sha256 != donor.packet_sha256
        )
    if constraint is DonorConstraint.LATE_QUERY_DIFFERENT:
        return (
            parent.query_source_sha256 != donor.query_source_sha256
            and parent.query_position != donor.query_position
        )
    if constraint is DonorConstraint.MIDPOINT_STATE_DIFFERENT_MATCHED_SUFFIX:
        return (
            parent.midpoint_suffix_sha256 == donor.midpoint_suffix_sha256
            and parent.midpoint_state_sha256 != donor.midpoint_state_sha256
        )
    if constraint is DonorConstraint.MIDPOINT_ACTION_DIFFERENT_MATCHED_SUFFIX:
        return parent.midpoint_suffix_sha256 == donor.midpoint_suffix_sha256 and any(
            item != parent.midpoint_action_sha256 for item in donor.action_card_sha256s
        )
    if constraint is DonorConstraint.PACKET_DIFFERENT:
        return parent.packet_sha256 != donor.packet_sha256
    return False


def _perfect_derangement(
    anchors: Sequence[AnchorBinding],
    constraint: DonorConstraint,
    seed: int,
    operation: str,
) -> dict[str, str] | None:
    parents = sorted(anchors, key=lambda row: row.anchor_id)
    parent_by_id = {row.anchor_id: row for row in parents}
    parent_ids = sorted(
        parent_by_id,
        key=lambda parent_id: _hash_order(seed, "parent", operation, parent_id),
    )
    adjacency = {
        parent_id: tuple(
            donor.anchor_id
            for donor in sorted(
                parents,
                key=lambda donor: _hash_order(
                    seed, "donor", operation, parent_id, donor.anchor_id
                ),
            )
            if _pair_allowed(constraint, parent_by_id[parent_id], donor)
        )
        for parent_id in parent_ids
    }
    if any(not values for values in adjacency.values()):
        return None

    left_to_right: dict[str, str] = {}
    right_to_left: dict[str, str] = {}
    distance: dict[str, int] = {}
    infinity = len(parent_ids) + 1

    def breadth_first() -> bool:
        queue: deque[str] = deque()
        shortest = infinity
        for parent_id in parent_ids:
            if parent_id not in left_to_right:
                distance[parent_id] = 0
                queue.append(parent_id)
            else:
                distance[parent_id] = infinity
        while queue:
            parent_id = queue.popleft()
            if distance[parent_id] >= shortest:
                continue
            for donor_id in adjacency[parent_id]:
                prior = right_to_left.get(donor_id)
                if prior is None:
                    shortest = distance[parent_id] + 1
                elif distance[prior] == infinity:
                    distance[prior] = distance[parent_id] + 1
                    queue.append(prior)
        return shortest != infinity

    def depth_first(parent_id: str) -> bool:
        for donor_id in adjacency[parent_id]:
            prior = right_to_left.get(donor_id)
            if prior is None or (
                distance.get(prior) == distance[parent_id] + 1 and depth_first(prior)
            ):
                left_to_right[parent_id] = donor_id
                right_to_left[donor_id] = parent_id
                return True
        distance[parent_id] = infinity
        return False

    while breadth_first():
        for parent_id in parent_ids:
            if parent_id not in left_to_right:
                depth_first(parent_id)
    return left_to_right if len(left_to_right) == len(parent_ids) else None


def _donor_maps(anchors: tuple[AnchorBinding, ...], seed: int) -> tuple[object, ...]:
    by_stratum: dict[tuple[str, int], tuple[AnchorBinding, ...]] = {}
    for class_id in PROGRAM_CLASSES:
        for depth in SCORED_DEPTHS:
            by_stratum[(class_id, depth)] = tuple(
                row for row in anchors if (row.class_id, row.depth) == (class_id, depth)
            )
    result = []
    for spec in OPERATION_SPECS.values():
        constraint = spec.expected.donor_constraint
        if constraint is DonorConstraint.NONE:
            continue
        strata = (
            {
                ("all-classes", depth): tuple(
                    row for row in anchors if row.depth == depth
                )
                for depth in SCORED_DEPTHS
            }
            if constraint
            in {
                DonorConstraint.MIDPOINT_STATE_DIFFERENT_MATCHED_SUFFIX,
                DonorConstraint.MIDPOINT_ACTION_DIFFERENT_MATCHED_SUFFIX,
            }
            else by_stratum
        )
        complete: dict[str, str] = {}
        for stratum in sorted(strata):
            mapping = _perfect_derangement(
                strata[stratum], constraint, seed, spec.operation
            )
            if mapping is None:
                raise RuntimePlanBuildError(
                    f"CTAA donor derangement is infeasible for {spec.operation}"
                )
            complete.update(mapping)
        pairs = tuple(
            DonorPair(anchor.anchor_id, complete[anchor.anchor_id])
            for anchor in sorted(anchors, key=lambda row: row.anchor_id)
        )
        result.append(make_donor_derangement(operation=spec.operation, pairs=pairs))
    return tuple(result)


def _all_donor_matchings_exist(anchors: tuple[AnchorBinding, ...], seed: int) -> bool:
    try:
        _donor_maps(anchors, seed)
    except RuntimePlanBuildError:
        return False
    return True


def _build_attempt_commitments(
    *,
    anchors: tuple[AnchorBinding, ...],
    rows_by_family: Mapping[str, ParsedSource],
    donor_derangements: Sequence[object],
    selection_seed: int,
    batch_order: Sequence[str],
    partition: str,
) -> tuple[object, ...]:
    anchor_by_id = {item.anchor_id: item for item in anchors}
    row_by_anchor = {item.anchor_id: rows_by_family[item.family_id] for item in anchors}
    donor_objects = {item.operation: item for item in donor_derangements}  # type: ignore[attr-defined]
    donor_maps = {
        operation: {pair.anchor_id: pair.donor_anchor_id for pair in value.pairs}
        for operation, value in donor_objects.items()
    }
    panel_sha = canonical_sha256([anchor_to_dict(item) for item in anchors])
    operation_hashes = {
        operation: operation_semantics_sha256(
            spec,
            panel_sha,
            (
                donor_objects[operation].derangement_sha256
                if operation in donor_objects
                else None
            ),
        )
        for operation, spec in OPERATION_SPECS.items()
    }
    attempts = []
    attempt_index = 0
    for operation, spec in OPERATION_SPECS.items():
        for anchor_id in batch_order:
            anchor = anchor_by_id[anchor_id]
            row = row_by_anchor[anchor_id]
            donor_id = donor_maps.get(operation, {}).get(anchor_id)
            donor = row_by_anchor[donor_id] if donor_id is not None else None
            payload: dict[str, object] = {
                "schema": "r12_ctaa_v2_concrete_mutation_v2",
                "operation": operation,
                "anchor_id": anchor_id,
                "timing": spec.timing,
            }
            program_sha = None
            query_sha = None
            packet_sha = None
            if operation in {
                InterventionFamily.H19_ZERO.value,
                InterventionFamily.H29_ZERO.value,
                InterventionFamily.H19_BATCH_ROTATE.value,
                InterventionFamily.H19_DONOR_TRANSPLANT.value,
                InterventionFamily.H29_BATCH_ROTATE.value,
                InterventionFamily.H29_DONOR_TRANSPLANT.value,
            }:
                payload.update(
                    {
                        "residual_layer": 19 if operation.startswith("h19_") else 29,
                        "token_start": 0,
                        "token_stop": row.token_count,
                        "channel_start": 0,
                        "channel_stop": "model_width",
                        "padding_mask_sha256": anchor.padding_mask_sha256,
                        "donor_anchor_id": donor_id,
                    }
                )
            elif operation == InterventionFamily.ENTITY_RECODE.value:
                names = _opaque_names(selection_seed, operation, anchor_id, "E", 3)
                source = _render_program(row, symbols=names)
                payload["old_to_new"] = dict(zip(row.symbols, names, strict=True))
                program_sha = _sha256_bytes(source.encode("utf-8"))
            elif operation == InterventionFamily.WITNESS_RECODE.value:
                names = _opaque_names(selection_seed, operation, anchor_id, "WQ", 4)
                source = _render_program(row, addresses=names)
                payload["old_to_new"] = dict(
                    zip(("W1", "W2", "W3", "W4"), names, strict=True)
                )
                program_sha = _sha256_bytes(source.encode("utf-8"))
            elif operation == InterventionFamily.OPCODE_RECODE.value:
                names = _opaque_names(selection_seed, operation, anchor_id, "OPQ", 4)
                source = _render_program(row, opcodes=names)
                payload["old_to_new"] = dict(zip(row.opcodes, names, strict=True))
                program_sha = _sha256_bytes(source.encode("utf-8"))
            elif operation == InterventionFamily.RENDERER_SUBSTITUTION.value:
                candidates = tuple(
                    value
                    for value in RENDERERS[partition]
                    if value != row.renderer_value
                    and ((value >> 5) & 1) == ((row.renderer_value >> 5) & 1)
                )
                target = candidates[
                    int.from_bytes(
                        _hash_order(selection_seed, operation, anchor_id)[:8], "big"
                    )
                    % len(candidates)
                ]
                source = _render_program(row, renderer_value=target)
                payload.update(
                    {
                        "parent_renderer": row.renderer_value,
                        "target_renderer": target,
                    }
                )
                program_sha = _sha256_bytes(source.encode("utf-8"))
            elif operation == InterventionFamily.RULE_LINE_SHUFFLE.value:
                order = _permutation(selection_seed, operation, anchor_id, ACTION_COUNT)
                if order == row.rule_order:
                    order = order[1:] + order[:1]
                source = _render_program(row, rule_order=order)
                payload["rule_order"] = list(order)
                program_sha = _sha256_bytes(source.encode("utf-8"))
                packet_sha = _sha256_bytes(
                    _packet_bytes(
                        row.cards,
                        row.initial_state,
                        row.schedule,
                        order,
                    )
                )
            elif operation == InterventionFamily.CARD_ONLY_COUNTERFACTUAL.value:
                card_address = row.schedule[0]
                coordinate = int.from_bytes(
                    _hash_order(selection_seed, operation, anchor_id, "coordinate")[
                        :8
                    ],
                    "big",
                ) % 3
                cards_list = [list(card) for card in row.cards]
                before = cards_list[card_address][coordinate]
                cards_list[card_address][coordinate] = (before + 1) % 3
                mutated = _packet_bytes_from_local(
                    tuple(tuple(card) for card in cards_list),
                    row.opcode_to_card,
                    row.initial_state,
                    row.opcode_schedule,
                )
                payload.update(
                    {
                        "card_address": card_address,
                        "coordinate": coordinate,
                        "before": before,
                        "after": cards_list[card_address][coordinate],
                    }
                )
                packet_sha = _sha256_bytes(mutated)
            elif operation == InterventionFamily.BINDING_ONLY_COUNTERFACTUAL.value:
                old_to_new = _three_cycle(
                    selection_seed,
                    operation,
                    anchor_id,
                    row.opcode_schedule[0],
                )
                new_to_old = [0] * ACTION_COUNT
                for old_slot, new_slot in enumerate(old_to_new):
                    new_to_old[new_slot] = old_slot
                binding = tuple(
                    row.opcode_to_card[old_slot] for old_slot in new_to_old
                )
                mutated = _packet_bytes_from_local(
                    row.cards,
                    binding,
                    row.initial_state,
                    row.opcode_schedule,
                )
                payload.update(
                    {
                        "old_to_new_opcode": list(old_to_new),
                        "new_to_old_opcode": new_to_old,
                    }
                )
                packet_sha = _sha256_bytes(mutated)
            elif operation == InterventionFamily.COMPENSATED_OPCODE_RELABEL.value:
                old_to_new = _three_cycle(selection_seed, operation, anchor_id)
                binding = [0] * ACTION_COUNT
                for old_opcode, new_opcode in enumerate(old_to_new):
                    binding[new_opcode] = row.opcode_to_card[old_opcode]
                opcode_schedule = tuple(
                    STOP_ID if opcode == STOP_ID else old_to_new[opcode]
                    for opcode in row.opcode_schedule
                )
                mutated = _packet_bytes_from_local(
                    row.cards,
                    binding,
                    row.initial_state,
                    opcode_schedule,
                )
                payload["old_to_new_opcode"] = list(old_to_new)
                packet_sha = _sha256_bytes(mutated)
            elif operation == InterventionFamily.CARD_STORAGE_REINDEX.value:
                order = _permutation(selection_seed, operation, anchor_id, ACTION_COUNT)
                inverse = [0] * ACTION_COUNT
                for new_slot, old_slot in enumerate(order):
                    inverse[old_slot] = new_slot
                cards = tuple(row.cards[slot] for slot in order)
                schedule = tuple(
                    event if event == STOP_ID else inverse[event]
                    for event in row.schedule
                )
                binding = tuple(inverse[card] for card in row.opcode_to_card)
                mutated = _packet_bytes(
                    cards,
                    row.initial_state,
                    schedule,
                    binding,
                )
                payload.update({"storage_order": list(order), "inverse": inverse})
                packet_sha = _sha256_bytes(mutated)
            elif operation == InterventionFamily.WITNESS_CORRUPTION.value:
                slot = (
                    int.from_bytes(
                        _hash_order(selection_seed, operation, anchor_id, "slot")[:8],
                        "big",
                    )
                    % ACTION_COUNT
                )
                position = (
                    int.from_bytes(
                        _hash_order(selection_seed, operation, anchor_id, "position")[
                            :8
                        ],
                        "big",
                    )
                    % 3
                )
                cards_list = [list(card) for card in row.cards]
                before = cards_list[slot][position]
                cards_list[slot][position] = (before + 1) % 3
                cards = tuple(tuple(card) for card in cards_list)
                source = _render_program(row, cards=cards)
                mutated = _packet_bytes(
                    cards,
                    row.initial_state,
                    row.schedule,
                    row.opcode_to_card,
                )
                payload.update(
                    {
                        "slot": slot,
                        "position": position,
                        "before": before,
                        "after": cards_list[slot][position],
                    }
                )
                program_sha = _sha256_bytes(source.encode("utf-8"))
                packet_sha = _sha256_bytes(mutated)
            elif operation == InterventionFamily.PAIRED_SHUFFLED_LAW.value:
                order = _permutation(selection_seed, operation, anchor_id, ACTION_COUNT)
                cards = tuple(row.cards[slot] for slot in order)
                source = _render_program(row, cards=cards)
                mutated = _packet_bytes(
                    cards,
                    row.initial_state,
                    row.schedule,
                    row.opcode_to_card,
                )
                payload["law_order"] = list(order)
                program_sha = _sha256_bytes(source.encode("utf-8"))
                packet_sha = _sha256_bytes(mutated)
            elif operation == InterventionFamily.SCHEDULE_ORDER_TWIN.value:
                pairs = [
                    (left, right)
                    for left in range(row.depth)
                    for right in range(left + 1, row.depth)
                    if row.schedule[left] != row.schedule[right]
                ]
                pairs.sort(
                    key=lambda pair: _hash_order(
                        selection_seed, operation, anchor_id, *pair
                    )
                )
                left, right = pairs[0]
                schedule_list = list(row.schedule)
                schedule_list[left], schedule_list[right] = (
                    schedule_list[right],
                    schedule_list[left],
                )
                schedule = tuple(schedule_list)
                source = _render_program(row, schedule=schedule)
                payload["swapped_active_slots"] = [left, right]
                program_sha = _sha256_bytes(source.encode("utf-8"))
                packet_sha = _sha256_bytes(
                    _packet_bytes(
                        row.cards,
                        row.initial_state,
                        schedule,
                        row.opcode_to_card,
                    )
                )
            elif operation == InterventionFamily.SOURCE_POISON.value:
                poison = b"SHOHIN-CTAA-SOURCE-POISON-v1\0" + _hash_order(
                    selection_seed, operation, anchor_id
                )
                payload.update(
                    {
                        "replacement_offset": 0,
                        "replacement_length": len(row.program_source.encode("utf-8")),
                        "poison_bytes_hex": poison.hex(),
                        "poison_bytes_sha256": _sha256_bytes(poison),
                    }
                )
                program_sha = _sha256_bytes(poison)
            elif operation == InterventionFamily.FUTURE_MASK.value:
                boundary = 1 + int.from_bytes(
                    _hash_order(selection_seed, operation, anchor_id)[:8], "big"
                ) % (row.depth - 1)
                opcode_schedule = tuple(
                    ((opcode + 1) % ACTION_COUNT)
                    if boundary <= index < row.depth
                    else opcode
                    for index, opcode in enumerate(row.opcode_schedule)
                )
                schedule = tuple(
                    STOP_ID
                    if opcode == STOP_ID
                    else row.opcode_to_card[opcode]
                    for opcode in opcode_schedule
                )
                payload.update(
                    {
                        "first_exposure_step": boundary,
                        "changed_slots": list(range(boundary, row.depth)),
                    }
                )
                packet_sha = _sha256_bytes(
                    _packet_bytes(
                        row.cards,
                        row.initial_state,
                        schedule,
                        row.opcode_to_card,
                    )
                )
            elif operation == InterventionFamily.STOP_RELOCATION.value:
                target = (
                    int.from_bytes(
                        _hash_order(selection_seed, operation, anchor_id)[:8], "big"
                    )
                    % row.depth
                )
                schedule_list = list(row.schedule)
                schedule_list[target], schedule_list[row.depth] = (
                    schedule_list[row.depth],
                    schedule_list[target],
                )
                schedule = tuple(schedule_list)
                source = _render_program(row, schedule=schedule)
                payload.update(
                    {
                        "old_stop_index": row.depth,
                        "new_stop_index": target,
                        "displaced_event": schedule_list[row.depth],
                    }
                )
                program_sha = _sha256_bytes(source.encode("utf-8"))
                packet_sha = _sha256_bytes(
                    _packet_bytes(
                        row.cards,
                        row.initial_state,
                        schedule,
                        row.opcode_to_card,
                    )
                )
            elif operation == InterventionFamily.LATE_QUERY_SWAP.value:
                assert donor is not None
                payload.update(
                    {
                        "parent_query_position": row.query_position,
                        "donor_query_position": donor.query_position,
                        "execution_policy": "reuse_immutable_parent_execution",
                    }
                )
                query_sha = _sha256_bytes(donor.query_source.encode("utf-8"))
            elif operation == InterventionFamily.POST_STOP_POISON.value:
                opcode_schedule = tuple(
                    ((opcode + 1) % ACTION_COUNT) if index > row.depth else opcode
                    for index, opcode in enumerate(row.opcode_schedule)
                )
                schedule = tuple(
                    STOP_ID
                    if opcode == STOP_ID
                    else row.opcode_to_card[opcode]
                    for opcode in opcode_schedule
                )
                payload.update(
                    {
                        "stop_index": row.depth,
                        "changed_slots": list(range(row.depth + 1, MAX_STEPS)),
                        "replacement_rule": "(event+1)%4",
                    }
                )
                packet_sha = _sha256_bytes(
                    _packet_bytes(
                        row.cards,
                        row.initial_state,
                        schedule,
                        row.opcode_to_card,
                    )
                )
            elif operation == InterventionFamily.MIDPOINT_DONOR_STATE.value:
                assert donor is not None
                payload.update(
                    {
                        "midpoint_step": row.depth // 2,
                        "donor_state_sha256": _sha256_bytes(donor.midpoint_state_bytes),
                    }
                )
            elif operation == InterventionFamily.MIDPOINT_DONOR_ACTION.value:
                assert donor is not None
                order = _permutation(selection_seed, operation, anchor_id, ACTION_COUNT)
                slot = next(
                    slot
                    for slot in order
                    if _sha256_bytes(bytes(donor.cards[slot]))
                    != anchor.midpoint_action_sha256
                )
                payload.update(
                    {
                        "midpoint_step": row.depth // 2,
                        "donor_card_slot": slot,
                        "donor_action_sha256": _sha256_bytes(bytes(donor.cards[slot])),
                    }
                )
            elif operation == InterventionFamily.PACKET_TRANSPLANT.value:
                assert donor is not None
                payload["literal_donor_packet_sha256"] = _sha256_bytes(
                    donor.packet_bytes
                )
                packet_sha = _sha256_bytes(donor.packet_bytes)
            elif operation == GateFamily.SOURCE_DELETION.value:
                payload.update(
                    {
                        "probe_stage": "source_blind_packet_executor",
                        "probe_targets": ["program_source", "board_root"],
                        "required_result": "all_open_attempts_denied",
                        "allowed_errno": ["EACCES", "ENOENT"],
                    }
                )
            elif operation == GateFamily.QUERY_ISOLATION.value:
                payload.update(
                    {
                        "probe_stage": "execution_before_receipt_commit",
                        "probe_target": "query_source",
                        "required_result": "all_open_attempts_denied",
                        "disclosure_after": "validated_immutable_execution_receipt",
                    }
                )
            elif operation == GateFamily.ROUTE_AGREEMENT.value:
                payload.update(
                    {
                        "positions": list(range(MAX_STEPS + 1)),
                        "comparison": "exact_uint8_state_route_equals_composed_route",
                        "required_tensor_shape": [MAX_STEPS + 1, 3],
                    }
                )
            else:  # pragma: no cover - exhaustive enum contract
                raise RuntimePlanBuildError("CTAA concrete mutation operation differs")
            attempts.append(
                make_anchor_operation_commitment(
                    attempt_index=attempt_index,
                    operation=operation,
                    operation_sha256=operation_hashes[operation],
                    anchor_id=anchor_id,
                    donor_anchor_id=donor_id,
                    mutation_payload=payload,
                    resulting_program_source_sha256=program_sha,
                    resulting_query_source_sha256=query_sha,
                    resulting_packet_sha256=packet_sha,
                )
            )
            attempt_index += 1
    return tuple(attempts)


def _write_once(path: Path, value: Mapping[str, object]) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing existing CTAA runtime plan: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(temporary, flags, 0o600)
    try:
        payload = (canonical_json(dict(value)) + "\n").encode("ascii")
        offset = 0
        while offset < len(payload):
            offset += os.write(descriptor, payload[offset:])
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o444)
        os.close(descriptor)
        descriptor = -1
        os.link(temporary, path, follow_symlinks=False)
        temporary.unlink()
        directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary.exists():
            temporary.chmod(0o600)
            temporary.unlink()


def build_runtime_intervention_plan(
    *,
    manifest_path: Path,
    program_path: Path,
    query_path: Path,
    tokenizer_path: Path,
    run_contract_path: Path,
    selection_seed_receipt_path: Path,
    output_path: Path,
    compiler_sha256: str,
    base_checkpoint_sha256: str,
    arm_id: str,
    training_seed: int,
    core_sha256: str,
    core_kind: str,
    base_raw_evidence_receipt_sha256: str,
    partition: str,
    custody_root_public_key_hex: str | None = None,
) -> RuntimeInterventionPlan:
    """Build and publish one deterministic, source-only intervention plan."""

    if output_path.exists() or output_path.is_symlink():
        raise FileExistsError(f"refusing existing CTAA runtime plan: {output_path}")
    if partition not in {item.value for item in Partition}:
        raise RuntimePlanBuildError("CTAA runtime-plan partition differs")
    if arm_id not in ARMS:
        raise RuntimePlanBuildError("CTAA runtime-plan arm differs")
    compiler_sha256 = _require_hash(compiler_sha256, "compiler")
    base_checkpoint_sha256 = _require_hash(base_checkpoint_sha256, "base checkpoint")
    core_sha256 = _require_hash(core_sha256, "core")
    base_raw_evidence_receipt_sha256 = _require_hash(
        base_raw_evidence_receipt_sha256, "base evidence receipt"
    )
    if type(training_seed) is not int or training_seed < 0:
        raise RuntimePlanBuildError("CTAA runtime-plan training seed differs")

    manifest_raw = _read_locked_bytes(manifest_path, "board manifest")
    _, manifest_sha, board_sha, manifest_files = _manifest_bindings(manifest_raw)
    expected_program = f"{partition}_program.jsonl"
    expected_query = f"{partition}_query.jsonl"
    if program_path.name != expected_program or query_path.name != expected_query:
        raise RuntimePlanBuildError("CTAA runtime-plan source filename differs")
    program_raw = _read_locked_bytes(program_path, "program source")
    query_raw = _read_locked_bytes(query_path, "query source")
    if manifest_files.get(expected_program) != _sha256_bytes(
        program_raw
    ) or manifest_files.get(expected_query) != _sha256_bytes(query_raw):
        raise RuntimePlanBuildError("CTAA runtime-plan source manifest binding differs")

    tokenizer_raw = _read_locked_bytes(tokenizer_path, "tokenizer")
    _decode_json(tokenizer_raw, "tokenizer")
    try:
        tokenizer = Tokenizer.from_str(tokenizer_raw.decode("utf-8"))
    except Exception as error:
        raise RuntimePlanBuildError("CTAA tokenizer payload differs") from error
    tokenizer_sha = _sha256_bytes(tokenizer_raw)

    contract_raw = _read_locked_bytes(run_contract_path, "run contract")
    contract_value = _decode_json(contract_raw, "run contract")
    try:
        contract = _validate_contract_shape(contract_value)
    except ValueError as error:
        raise RuntimePlanBuildError("CTAA run contract differs") from error
    if contract_raw != (canonical_json(contract) + "\n").encode("ascii"):
        raise RuntimePlanBuildError("CTAA run contract is not canonical")
    if (
        contract.get("schema") != RUN_CONTRACT_SCHEMA
        or contract.get("partition") != partition
        or contract.get("manifest_sha256") != manifest_sha
        or contract.get("board_sha256") != board_sha
    ):
        raise RuntimePlanBuildError("CTAA run contract source binding differs")
    matching_runs = [
        row
        for row in contract["runs"]  # type: ignore[union-attr]
        if row["arm"] == arm_id
        and row["dataset"] == "base"
        and row["seed"] == training_seed
    ]
    if (
        len(matching_runs) != 1
        or compiler_sha256 != matching_runs[0]["compiler_sha256"]
        or core_sha256 != matching_runs[0]["core_training"]["core_sha256"]
        or core_kind != matching_runs[0]["core_training"]["core_kind"]
        or base_raw_evidence_receipt_sha256
        != matching_runs[0]["raw_evidence_receipt_sha256"]
        or any(
            row["sealed_sources"]["program_sha256"] != _sha256_bytes(program_raw)
            or row["sealed_sources"]["query_sha256"] != _sha256_bytes(query_raw)
            for row in matching_runs
        )
    ):
        raise RuntimePlanBuildError(
            "CTAA run contract arm/source/compiler binding differs"
        )

    seed_receipt_raw = _read_locked_bytes(
        selection_seed_receipt_path, "selection-seed receipt"
    )
    seed_receipt = _decode_json(seed_receipt_raw, "selection-seed receipt")
    try:
        checked_seed = validate_seed_receipt(
            seed_receipt,
            manifest_sha256=manifest_sha,
            gate_source_sha256=str(seed_receipt.get("gate_source_sha256")),
            statistics_source_sha256=str(seed_receipt.get("statistics_source_sha256")),
            custody_root_public_key_hex=custody_root_public_key_hex,
        )
    except ValueError as error:
        raise RuntimePlanBuildError("CTAA selection-seed receipt differs") from error
    selection_seed = checked_seed.get("bootstrap_seed")
    if type(selection_seed) is not int or selection_seed < 0:
        raise RuntimePlanBuildError("CTAA selection seed differs")
    if (
        contract.get("bootstrap_seed_receipt_sha256") != _sha256_bytes(seed_receipt_raw)
        or contract.get("bootstrap_seed") != selection_seed
    ):
        raise RuntimePlanBuildError("CTAA run contract selection-seed binding differs")

    program_rows = _read_jsonl(
        program_raw, "program source", frozenset({"family_id", "program_source"})
    )
    query_rows = _read_jsonl(
        query_raw, "query source", frozenset({"family_id", "query_source"})
    )
    query_by_id = {
        str(row["family_id"]): str(row["query_source"]) for row in query_rows
    }
    program_ids = {str(row["family_id"]) for row in program_rows}
    if set(query_by_id) != program_ids:
        raise RuntimePlanBuildError("CTAA program/query family set differs")
    parsed = [
        _parse_program(
            str(row["family_id"]),
            str(row["program_source"]),
            query_by_id[str(row["family_id"])],
            tokenizer,
            partition,
        )
        for row in program_rows
        if (match := FAMILY_ID.fullmatch(str(row["family_id"]))) is not None
        and match.group("cell") == "hhh"
    ]
    if len(parsed) != HHH_ROWS:
        raise RuntimePlanBuildError("CTAA hhh source count differs")
    partition_value = Partition(partition)
    anchors = _select_panel(parsed, selection_seed, partition_value)
    batch_order = [
        row.anchor_id
        for row in sorted(
            anchors,
            key=lambda row: _hash_order(selection_seed, "batch-order", row.anchor_id),
        )
    ]
    bindings = PlanBindings(
        board_manifest_sha256=manifest_sha,
        board_tree_sha256=board_sha,
        compiler_sha256=compiler_sha256,
        tokenizer_sha256=tokenizer_sha,
        base_checkpoint_sha256=base_checkpoint_sha256,
        run_contract_sha256=str(contract["run_contract_sha256"]),
        selection_seed=selection_seed,
        selection_seed_receipt_sha256=_sha256_bytes(seed_receipt_raw),
        arm_id=arm_id,
        training_seed=training_seed,
        core_sha256=core_sha256,
        core_kind=core_kind,
        base_raw_evidence_receipt_sha256=base_raw_evidence_receipt_sha256,
        runtime_implementation_sha256=_runtime_implementation_sha256(),
        partition=partition_value,
        batch_order=tuple(batch_order),
        batch_order_sha256=_tagged_sha256(
            "r12_ctaa_v2_runtime_batch_order_v1", batch_order
        ),
        scored_row_count=LOCKED_SCORED_ROW_COUNT,
        runtime_panel_size=RUNTIME_PANEL_SIZE,
        runtime_attempts_affect_scored_denominator=False,
    )
    donor_derangements = _donor_maps(anchors, selection_seed)
    rows_by_family = {row.family_id: row for row in parsed}
    attempts = _build_attempt_commitments(
        anchors=anchors,
        rows_by_family=rows_by_family,
        donor_derangements=donor_derangements,
        selection_seed=selection_seed,
        batch_order=batch_order,
        partition=partition,
    )
    plan = make_runtime_intervention_plan(
        bindings=bindings,
        anchors=anchors,
        donor_derangements=donor_derangements,
        attempts=attempts,
    )
    validated = validate_runtime_intervention_plan(plan)
    payload = plan_to_dict(validated)
    _write_once(output_path, payload)
    published_raw = _read_locked_bytes(output_path, "published runtime plan")
    published_value = _decode_json(published_raw, "published runtime plan")
    if published_raw != (canonical_json(published_value) + "\n").encode("ascii"):
        raise RuntimePlanBuildError("CTAA published runtime plan is not canonical")
    published = validate_runtime_intervention_plan(published_value)
    if published != validated:
        raise RuntimePlanBuildError("CTAA published runtime plan differs")
    return published


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--program", type=Path, required=True)
    parser.add_argument("--query", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--run-contract", type=Path, required=True)
    parser.add_argument("--selection-seed-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compiler-sha256", required=True)
    parser.add_argument("--base-checkpoint-sha256", required=True)
    parser.add_argument("--arm", required=True, choices=ARMS)
    parser.add_argument("--training-seed", required=True, type=int)
    parser.add_argument("--core-sha256", required=True)
    parser.add_argument(
        "--core-kind",
        required=True,
        choices=("closure_feature", "outer_product_control"),
    )
    parser.add_argument("--base-raw-evidence-receipt-sha256", required=True)
    parser.add_argument(
        "--partition", required=True, choices=tuple(item.value for item in Partition)
    )
    args = parser.parse_args()
    result = build_runtime_intervention_plan(
        manifest_path=args.manifest,
        program_path=args.program,
        query_path=args.query,
        tokenizer_path=args.tokenizer,
        run_contract_path=args.run_contract,
        selection_seed_receipt_path=args.selection_seed_receipt,
        output_path=args.output,
        compiler_sha256=args.compiler_sha256,
        base_checkpoint_sha256=args.base_checkpoint_sha256,
        arm_id=args.arm,
        training_seed=args.training_seed,
        core_sha256=args.core_sha256,
        core_kind=args.core_kind,
        base_raw_evidence_receipt_sha256=args.base_raw_evidence_receipt_sha256,
        partition=args.partition,
    )
    print(
        canonical_json(
            {"plan_sha256": result.plan_sha256, "anchors": len(result.anchors)}
        )
    )


if __name__ == "__main__":
    main()
