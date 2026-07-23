"""Strict query-blind loader for CTAA runtime program artifacts.

The loader treats source bytes as the only program authority.  Projection
anchors contribute only opaque anchor IDs plus committed program and packet
hashes.  Source-intervention programs are reconstructed from committed
payloads, and token IDs are always generated here from immutable tokenizer
bytes.  No family, class, renderer, partition, query, answer, oracle, or
caller-supplied token sequence is accepted as execution authority.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import stat
from types import MappingProxyType
from typing import Mapping, Sequence

from tokenizers import Tokenizer

from build_ctaa_runtime_intervention_plan import (
    SEQUENCE_LIMIT,
    _packet_bytes,
    _render_program,
    _split_once,
    _tuple_values,
)
from ctaa_intervention_protocol import AnchorOperationCommitment, InterventionFamily
from ctaa_neural_core import CTAA_ACTION_COUNT, CTAA_MAX_STEPS, CTAA_WIDTH
from ctaa_run_contract import canonical_json
from ctaa_runtime_execution_engine import PROGRAM_ARTIFACT_SCHEMA, ProgramArtifact
from ctaa_runtime_execution_projection import (
    ExecutionProjectionError,
    validate_execution_projection_standalone,
)
from ctaa_runtime_plan_replay import (
    ReplayValidationError,
    _payload as _validate_replay_payload,
)


_MAX_PROGRAM_BYTES = 64 * 1024 * 1024
_MAX_TOKENIZER_BYTES = 64 * 1024 * 1024
_LEGACY_SOURCE_ROW_KEYS = frozenset({"family_id", "program_source"})
_HASHED_SOURCE_ROW_KEYS = frozenset({"program_source_sha256", "program_source"})
_HEX = frozenset("0123456789abcdef")
_SOURCE_OPERATIONS = frozenset(
    {
        InterventionFamily.ENTITY_RECODE.value,
        InterventionFamily.WITNESS_RECODE.value,
        InterventionFamily.OPCODE_RECODE.value,
        InterventionFamily.RENDERER_SUBSTITUTION.value,
        InterventionFamily.RULE_LINE_SHUFFLE.value,
        InterventionFamily.WITNESS_CORRUPTION.value,
        InterventionFamily.PAIRED_SHUFFLED_LAW.value,
        InterventionFamily.SCHEDULE_ORDER_TWIN.value,
        InterventionFamily.STOP_RELOCATION.value,
    }
)
_SOURCE_PACKET_OPERATIONS = frozenset(
    {
        InterventionFamily.WITNESS_CORRUPTION.value,
        InterventionFamily.PAIRED_SHUFFLED_LAW.value,
        InterventionFamily.SCHEDULE_ORDER_TWIN.value,
        InterventionFamily.STOP_RELOCATION.value,
    }
)


class ProgramArtifactLoadError(ValueError):
    """Frozen CTAA source material does not reproduce the projected registry."""


@dataclass(frozen=True)
class _ProgramShape:
    """Query-free structure required to render committed source mutations."""

    program_source: str
    symbols: tuple[str, str, str]
    opcodes: tuple[str, str, str, str]
    cards: tuple[tuple[int, int, int], ...]
    initial_state: tuple[int, int, int]
    schedule: tuple[int, ...]
    rule_order: tuple[int, ...]
    renderer_value: int
    depth: int


def _is_hash(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= _HEX


def _path_text(path: Path) -> str:
    try:
        raw = os.fspath(path)
    except TypeError as error:
        raise ProgramArtifactLoadError("artifact input path differs") from error
    if (
        not isinstance(raw, str)
        or "\x00" in raw
        or not os.path.isabs(raw)
        or os.path.normpath(raw) != raw
        or raw == "/"
        or any(part in ("", ".", "..") for part in raw.split("/")[1:])
    ):
        raise ProgramArtifactLoadError("artifact input path is unsafe")
    return raw


def _open_parent(path: Path) -> tuple[int, str]:
    if not hasattr(os, "O_NOFOLLOW"):
        raise ProgramArtifactLoadError("O_NOFOLLOW is required")
    components = _path_text(path).split("/")[1:]
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open("/", flags)
    try:
        for component in components[:-1]:
            try:
                child = os.open(component, flags, dir_fd=descriptor)
            except OSError as error:
                raise ProgramArtifactLoadError(
                    "artifact input parent is missing or symlinked"
                ) from error
            metadata = os.fstat(child)
            if not stat.S_ISDIR(metadata.st_mode):
                os.close(child)
                raise ProgramArtifactLoadError(
                    "artifact input parent is not a directory"
                )
            os.close(descriptor)
            descriptor = child
        return descriptor, components[-1]
    except Exception:
        os.close(descriptor)
        raise


def _identity(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _read_immutable(path: Path, *, label: str, maximum: int) -> bytes:
    parent, name = _open_parent(path)
    descriptor = -1
    try:
        flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
        try:
            descriptor = os.open(name, flags, dir_fd=parent)
        except OSError as error:
            raise ProgramArtifactLoadError(
                f"{label} is missing or symlinked"
            ) from error
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_mode & 0o222
            or before.st_size <= 0
            or before.st_size > maximum
        ):
            raise ProgramArtifactLoadError(
                f"{label} must be immutable, regular, and single-link"
            )
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                raise ProgramArtifactLoadError(f"{label} changed during read")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise ProgramArtifactLoadError(f"{label} grew during read")
        after = os.fstat(descriptor)
        if _identity(before) != _identity(after):
            raise ProgramArtifactLoadError(f"{label} changed during read")
        return b"".join(chunks)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        os.close(parent)


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ProgramArtifactLoadError("program source contains duplicate keys")
        result[key] = value
    return result


def _reject_constant(value: str) -> object:
    raise ProgramArtifactLoadError(f"program source contains non-finite {value}")


def _read_source_rows(raw: bytes) -> list[bytes]:
    if not raw.endswith(b"\n"):
        raise ProgramArtifactLoadError("program source JSONL is not canonical")
    sources: list[bytes] = []
    schema: frozenset[str] | None = None
    for line_number, line in enumerate(raw.splitlines(), 1):
        if not line:
            raise ProgramArtifactLoadError(f"program source row {line_number} is empty")
        try:
            value = json.loads(
                line.decode("ascii"),
                object_pairs_hook=_strict_object,
                parse_constant=_reject_constant,
            )
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ProgramArtifactLoadError(
                f"program source row {line_number} is malformed"
            ) from error
        row_schema = frozenset(value) if isinstance(value, dict) else frozenset()
        if row_schema not in (_LEGACY_SOURCE_ROW_KEYS, _HASHED_SOURCE_ROW_KEYS):
            raise ProgramArtifactLoadError(
                f"program source row {line_number} schema differs"
            )
        if schema is None:
            schema = row_schema
        elif row_schema != schema:
            raise ProgramArtifactLoadError("program source row schemas are mixed")
        if canonical_json(value).encode("ascii") != line:
            raise ProgramArtifactLoadError(
                f"program source row {line_number} is not canonical"
            )
        source = value["program_source"]
        if not isinstance(source, str) or not source:
            raise ProgramArtifactLoadError(
                f"program source row {line_number} value differs"
            )
        source_bytes = source.encode("utf-8")
        if row_schema == _HASHED_SOURCE_ROW_KEYS:
            committed = value["program_source_sha256"]
            if (
                not _is_hash(committed)
                or hashlib.sha256(source_bytes).hexdigest() != committed
            ):
                raise ProgramArtifactLoadError(
                    f"program source row {line_number} hash differs"
                )
        else:
            legacy_id = value["family_id"]
            if not isinstance(legacy_id, str) or not legacy_id:
                raise ProgramArtifactLoadError(
                    f"program source row {line_number} value differs"
                )
        sources.append(source_bytes)
    if not sources:
        raise ProgramArtifactLoadError("program source JSONL is empty")
    return sources


def _load_tokenizer(raw: bytes, expected_sha256: object) -> Tokenizer:
    if (
        not _is_hash(expected_sha256)
        or hashlib.sha256(raw).hexdigest() != expected_sha256
    ):
        raise ProgramArtifactLoadError("tokenizer source hash differs")
    try:
        tokenizer = Tokenizer.from_str(raw.decode("utf-8"))
    except Exception as error:  # noqa: BLE001 - tokenizers uses generic exceptions
        raise ProgramArtifactLoadError("tokenizer source is malformed") from error
    return tokenizer


def _token_ids(tokenizer: Tokenizer, source: bytes) -> tuple[int, ...]:
    try:
        text = source.decode("utf-8")
        first = tuple(tokenizer.encode(text).ids)
        second = tuple(tokenizer.encode(text).ids)
    except Exception as error:  # noqa: BLE001 - tokenizers uses generic exceptions
        raise ProgramArtifactLoadError("program tokenization failed") from error
    if first != second:
        raise ProgramArtifactLoadError("program tokenization is nondeterministic")
    if (
        not first
        or len(first) > SEQUENCE_LIMIT
        or any(type(token) is not int or token < 0 for token in first)
    ):
        raise ProgramArtifactLoadError("program token geometry differs")
    return first


def _artifact(tokenizer: Tokenizer, source: bytes) -> ProgramArtifact:
    return ProgramArtifact(
        PROGRAM_ARTIFACT_SCHEMA, source, _token_ids(tokenizer, source)
    )


def _parse_program_source(source: bytes) -> _ProgramShape:
    try:
        text = source.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ProgramArtifactLoadError("parent program source is not UTF-8") from error
    lines = text.splitlines()
    if len(lines) != 7 or not text.endswith("\n"):
        raise ProgramArtifactLoadError("parent program line geometry differs")
    try:
        header_bit, raw_symbols = _split_once(
            lines[0], ("SYMBOL ORDER :: ", "REGISTER ALPHABET = "), "source header"
        )
        symbols = _tuple_values(raw_symbols)
        if len(symbols) != CTAA_WIDTH or len(set(symbols)) != CTAA_WIDTH:
            raise ProgramArtifactLoadError("parent symbol namespace differs")
        symbol_index = {symbol: index for index, symbol in enumerate(symbols)}

        cards: list[tuple[int, int, int] | None] = [None] * CTAA_ACTION_COUNT
        opcodes: list[str | None] = [None] * CTAA_ACTION_COUNT
        rule_order: list[int] = []
        rule_bits: set[int] = set()
        for line in lines[1:5]:
            if line.startswith("CARD "):
                rule_bits.add(0)
                found = re.fullmatch(
                    r"CARD W([1-4]); CODE ([^;\s]+); BEFORE (.+); AFTER (.+)",
                    line,
                )
            else:
                rule_bits.add(1)
                found = re.fullmatch(r"W([1-4]) binds ([^:\s]+): (.+) => (.+)", line)
            if found is None:
                raise ProgramArtifactLoadError("parent rule syntax differs")
            slot = int(found.group(1)) - 1
            opcode = found.group(2)
            before = _tuple_values(found.group(3))
            after = _tuple_values(found.group(4))
            if (
                before != symbols
                or len(after) != CTAA_WIDTH
                or any(item not in symbol_index for item in after)
                or cards[slot] is not None
                or opcode in opcodes
            ):
                raise ProgramArtifactLoadError("parent rule binding differs")
            cards[slot] = tuple(symbol_index[item] for item in after)
            opcodes[slot] = opcode
            rule_order.append(slot)
        if len(rule_bits) != 1 or any(item is None for item in cards + opcodes):
            raise ProgramArtifactLoadError("parent rule set differs")

        start_bit, raw_initial = _split_once(
            lines[5], ("INITIAL STATE :: ", "LOAD REGISTERS = "), "initial state"
        )
        initial_names = _tuple_values(raw_initial)
        if (
            len(initial_names) != CTAA_WIDTH
            or len(set(initial_names)) != CTAA_WIDTH
            or any(item not in symbol_index for item in initial_names)
        ):
            raise ProgramArtifactLoadError("parent initial state differs")
        initial = tuple(symbol_index[item] for item in initial_names)

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
            raise ProgramArtifactLoadError("parent event separator differs")
        event_names = raw_tape.split(separator)
        typed_opcodes = tuple(str(item) for item in opcodes)
        opcode_index = {opcode: index for index, opcode in enumerate(typed_opcodes)}
        stop_names = [name for name in event_names if name in {"STOP", "HALT_NOW"}]
        if len(event_names) != CTAA_MAX_STEPS or len(stop_names) != 1:
            raise ProgramArtifactLoadError("parent event geometry differs")
        stop_name = stop_names[0]
        if stop_bit != (0 if stop_name == "STOP" else 1):
            raise ProgramArtifactLoadError("parent STOP renderer differs")
        schedule = tuple(
            CTAA_ACTION_COUNT if name == stop_name else opcode_index.get(name, -1)
            for name in event_names
        )
        if any(event < 0 for event in schedule):
            raise ProgramArtifactLoadError("parent event opcode differs")
        depth = schedule.index(CTAA_ACTION_COUNT)
        if depth <= 0 or depth >= CTAA_MAX_STEPS:
            raise ProgramArtifactLoadError("parent STOP position differs")
    except ProgramArtifactLoadError:
        raise
    except Exception as error:  # builder parsing helpers expose ValueError subclasses
        raise ProgramArtifactLoadError("parent program source differs") from error

    renderer = (
        header_bit
        | (next(iter(rule_bits)) << 1)
        | (start_bit << 2)
        | (tape_bit << 3)
        | (stop_bit << 4)
    )
    return _ProgramShape(
        program_source=text,
        symbols=tuple(symbols),  # type: ignore[arg-type]
        opcodes=typed_opcodes,  # type: ignore[arg-type]
        cards=tuple(cards),  # type: ignore[arg-type]
        initial_state=initial,  # type: ignore[arg-type]
        schedule=schedule,
        rule_order=tuple(rule_order),
        renderer_value=renderer,
        depth=depth,
    )


def _attempt_payload(row: Mapping[str, object]) -> dict[str, object]:
    try:
        commitment = AnchorOperationCommitment(
            schema="r12_ctaa_anchor_operation_commitment_v1",
            attempt_index=int(row["attempt_index"]),
            attempt_id=str(row["attempt_id"]),
            operation=str(row["operation"]),
            operation_sha256=str(row["operation_sha256"]),
            anchor_id=str(row["anchor_id"]),
            donor_anchor_id=(
                None if row["donor_anchor_id"] is None else str(row["donor_anchor_id"])
            ),
            mutation_payload_json=str(row["mutation_payload_json"]),
            mutation_payload_sha256=str(row["mutation_payload_sha256"]),
            resulting_program_source_sha256=str(row["resulting_program_source_sha256"]),
            resulting_query_source_sha256=None,
            resulting_packet_sha256=(
                None
                if row["resulting_packet_sha256"] is None
                else str(row["resulting_packet_sha256"])
            ),
            attempt_plan_sha256=str(row["attempt_plan_sha256"]),
        )
        return _validate_replay_payload(commitment)
    except (KeyError, TypeError, ValueError, ReplayValidationError) as error:
        raise ProgramArtifactLoadError("source mutation payload differs") from error


def _integer(value: object, label: str, *, minimum: int, maximum: int) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ProgramArtifactLoadError(f"{label} differs")
    return value


def _permutation_payload(value: object, label: str) -> tuple[int, ...]:
    if (
        not isinstance(value, list)
        or any(type(item) is not int for item in value)
        or sorted(value) != list(range(CTAA_ACTION_COUNT))
    ):
        raise ProgramArtifactLoadError(f"{label} differs")
    return tuple(value)


def _renaming(value: object, keys: Sequence[str], label: str) -> tuple[str, ...]:
    if not isinstance(value, dict) or set(value) != set(keys):
        raise ProgramArtifactLoadError(f"{label} schema differs")
    result = tuple(value[key] for key in keys)
    if (
        any(not isinstance(item, str) or not item for item in result)
        or len(set(result)) != len(result)
        or all(old == new for old, new in zip(keys, result, strict=True))
    ):
        raise ProgramArtifactLoadError(f"{label} differs")
    return result  # type: ignore[return-value]


def _packet_sha(
    cards: Sequence[Sequence[int]], initial: Sequence[int], schedule: Sequence[int]
) -> str:
    return hashlib.sha256(_packet_bytes(cards, initial, schedule)).hexdigest()


def _source_intervention(
    *, attempt: Mapping[str, object], parent: _ProgramShape
) -> bytes:
    operation = str(attempt["operation"])
    payload = _attempt_payload(attempt)
    packet_sha: str | None = None

    if operation == InterventionFamily.ENTITY_RECODE.value:
        names = _renaming(payload["old_to_new"], parent.symbols, "entity recode")
        rendered = _render_program(parent, symbols=names)
    elif operation == InterventionFamily.WITNESS_RECODE.value:
        names = _renaming(
            payload["old_to_new"], ("W1", "W2", "W3", "W4"), "witness recode"
        )
        rendered = _render_program(parent, addresses=names)
    elif operation == InterventionFamily.OPCODE_RECODE.value:
        names = _renaming(payload["old_to_new"], parent.opcodes, "opcode recode")
        rendered = _render_program(parent, opcodes=names)
    elif operation == InterventionFamily.RENDERER_SUBSTITUTION.value:
        parent_renderer = _integer(
            payload["parent_renderer"], "parent renderer", minimum=0, maximum=63
        )
        target = _integer(
            payload["target_renderer"], "target renderer", minimum=0, maximum=63
        )
        if (
            parent_renderer & 31 != parent.renderer_value
            or target & 31 == parent.renderer_value
        ):
            raise ProgramArtifactLoadError("renderer substitution differs")
        rendered = _render_program(parent, renderer_value=target & 31)
    elif operation == InterventionFamily.RULE_LINE_SHUFFLE.value:
        order = _permutation_payload(payload["rule_order"], "rule-line order")
        if order == parent.rule_order:
            raise ProgramArtifactLoadError("rule-line shuffle is a no-op")
        rendered = _render_program(parent, rule_order=order)
    elif operation == InterventionFamily.WITNESS_CORRUPTION.value:
        slot = _integer(
            payload["slot"], "witness-corruption slot", minimum=0, maximum=3
        )
        position = _integer(
            payload["position"], "witness-corruption position", minimum=0, maximum=2
        )
        before = _integer(
            payload["before"], "witness-corruption before", minimum=0, maximum=2
        )
        after = _integer(
            payload["after"], "witness-corruption after", minimum=0, maximum=2
        )
        if parent.cards[slot][position] != before or after == before:
            raise ProgramArtifactLoadError("witness corruption differs")
        cards_list = [list(card) for card in parent.cards]
        cards_list[slot][position] = after
        cards = tuple(tuple(card) for card in cards_list)
        rendered = _render_program(parent, cards=cards)
        packet_sha = _packet_sha(cards, parent.initial_state, parent.schedule)
    elif operation == InterventionFamily.PAIRED_SHUFFLED_LAW.value:
        order = _permutation_payload(payload["law_order"], "shuffled-law order")
        cards = tuple(parent.cards[slot] for slot in order)
        if cards == parent.cards:
            raise ProgramArtifactLoadError("shuffled-law order is a no-op")
        rendered = _render_program(parent, cards=cards)
        packet_sha = _packet_sha(cards, parent.initial_state, parent.schedule)
    elif operation == InterventionFamily.SCHEDULE_ORDER_TWIN.value:
        slots = payload["swapped_active_slots"]
        if (
            not isinstance(slots, list)
            or len(slots) != 2
            or any(type(item) is not int for item in slots)
        ):
            raise ProgramArtifactLoadError("schedule swap differs")
        left, right = slots
        if (
            not 0 <= left < right < parent.depth
            or parent.schedule[left] == parent.schedule[right]
        ):
            raise ProgramArtifactLoadError("schedule swap is invalid")
        schedule_list = list(parent.schedule)
        schedule_list[left], schedule_list[right] = (
            schedule_list[right],
            schedule_list[left],
        )
        schedule = tuple(schedule_list)
        rendered = _render_program(parent, schedule=schedule)
        packet_sha = _packet_sha(parent.cards, parent.initial_state, schedule)
    elif operation == InterventionFamily.STOP_RELOCATION.value:
        old_stop = _integer(
            payload["old_stop_index"],
            "old STOP index",
            minimum=0,
            maximum=CTAA_MAX_STEPS - 1,
        )
        target = _integer(
            payload["new_stop_index"],
            "new STOP index",
            minimum=0,
            maximum=CTAA_MAX_STEPS - 1,
        )
        displaced = _integer(
            payload["displaced_event"],
            "displaced event",
            minimum=0,
            maximum=CTAA_ACTION_COUNT - 1,
        )
        if (
            old_stop != parent.depth
            or target >= parent.depth
            or parent.schedule[target] != displaced
        ):
            raise ProgramArtifactLoadError("STOP relocation differs")
        schedule_list = list(parent.schedule)
        schedule_list[target], schedule_list[parent.depth] = (
            schedule_list[parent.depth],
            schedule_list[target],
        )
        schedule = tuple(schedule_list)
        rendered = _render_program(parent, schedule=schedule)
        packet_sha = _packet_sha(parent.cards, parent.initial_state, schedule)
    else:  # pragma: no cover - closed source-operation set above
        raise ProgramArtifactLoadError("source operation differs")

    source = rendered.encode("utf-8")
    committed = attempt["resulting_program_source_sha256"]
    if (
        source == parent.program_source.encode("utf-8")
        or not _is_hash(committed)
        or hashlib.sha256(source).hexdigest() != committed
    ):
        raise ProgramArtifactLoadError("source intervention hash differs")
    committed_packet = attempt["resulting_packet_sha256"]
    if operation in _SOURCE_PACKET_OPERATIONS:
        if not _is_hash(committed_packet) or packet_sha != committed_packet:
            raise ProgramArtifactLoadError("source intervention packet hash differs")
    elif committed_packet is not None:
        raise ProgramArtifactLoadError("source intervention packet presence differs")
    return source


def load_runtime_program_artifacts(
    *,
    projection: Mapping[str, object],
    program_source_path: Path,
    tokenizer_path: Path,
) -> Mapping[str, ProgramArtifact]:
    """Reconstruct the exact query-blind program registry required by a projection."""

    try:
        snapshot = json.loads(canonical_json(projection))
        frozen = validate_execution_projection_standalone(snapshot)
    except (TypeError, ValueError, ExecutionProjectionError) as error:
        raise ProgramArtifactLoadError("execution projection differs") from error

    source_raw = _read_immutable(
        program_source_path,
        label="program source",
        maximum=_MAX_PROGRAM_BYTES,
    )
    tokenizer_raw = _read_immutable(
        tokenizer_path,
        label="tokenizer source",
        maximum=_MAX_TOKENIZER_BYTES,
    )
    source_rows = _read_source_rows(source_raw)
    tokenizer = _load_tokenizer(tokenizer_raw, frozen["tokenizer_sha256"])

    anchors = frozen["anchors"]
    attempts = frozen["attempts"]
    if not isinstance(anchors, list) or not isinstance(attempts, list):
        raise ProgramArtifactLoadError("projection program registry differs")

    expected_parent_hashes: list[str] = []
    anchor_ids: set[str] = set()
    for anchor in anchors:
        if not isinstance(anchor, Mapping):
            raise ProgramArtifactLoadError("projection parent differs")
        anchor_id = anchor.get("anchor_id")
        source_sha = anchor.get("program_source_sha256")
        packet_sha = anchor.get("packet_sha256")
        if (
            not isinstance(anchor_id, str)
            or not anchor_id
            or anchor_id in anchor_ids
            or not _is_hash(source_sha)
            or not _is_hash(packet_sha)
        ):
            raise ProgramArtifactLoadError("projection parent binding differs")
        anchor_ids.add(anchor_id)
        expected_parent_hashes.append(str(source_sha))

    observed_parent_hashes = [
        hashlib.sha256(source).hexdigest() for source in source_rows
    ]
    if Counter(observed_parent_hashes) != Counter(expected_parent_hashes):
        raise ProgramArtifactLoadError("program source hash coverage differs")
    source_by_hash: dict[str, bytes] = {}
    for digest, source in zip(observed_parent_hashes, source_rows, strict=True):
        prior = source_by_hash.setdefault(digest, source)
        if prior != source:
            raise ProgramArtifactLoadError("program source hash aliases bytes")

    parents: dict[str, _ProgramShape] = {}
    ordered_sources: list[tuple[str, bytes]] = []
    for anchor in anchors:
        assert isinstance(anchor, Mapping)
        anchor_id = str(anchor["anchor_id"])
        digest = str(anchor["program_source_sha256"])
        source = source_by_hash[digest]
        parsed = _parse_program_source(source)
        if (
            _packet_sha(parsed.cards, parsed.initial_state, parsed.schedule)
            != anchor["packet_sha256"]
        ):
            raise ProgramArtifactLoadError("parent packet source hash differs")
        parents[anchor_id] = parsed
        ordered_sources.append((digest, source))

    for attempt in attempts:
        if not isinstance(attempt, Mapping):
            raise ProgramArtifactLoadError("projection source attempt differs")
        if attempt.get("operation") not in _SOURCE_OPERATIONS:
            continue
        parent = parents.get(str(attempt.get("anchor_id")))
        if parent is None:
            raise ProgramArtifactLoadError("source attempt parent differs")
        source = _source_intervention(attempt=attempt, parent=parent)
        ordered_sources.append(
            (str(attempt["resulting_program_source_sha256"]), source)
        )

    required: list[str] = []
    registry: dict[str, ProgramArtifact] = {}
    for digest, source in ordered_sources:
        if digest not in required:
            required.append(digest)
        prior = registry.get(digest)
        if prior is not None:
            if prior.source != source:
                raise ProgramArtifactLoadError("program digest aliases source bytes")
            continue
        artifact = _artifact(tokenizer, source)
        if artifact.source_sha256 != digest:
            raise ProgramArtifactLoadError("program artifact source hash differs")
        registry[digest] = artifact
    if list(registry) != required or set(registry) != set(required):
        raise ProgramArtifactLoadError("program artifact registry coverage differs")
    return MappingProxyType(registry)
