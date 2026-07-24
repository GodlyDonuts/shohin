#!/usr/bin/env python3
"""Standalone source-blind EFC candidate compiler role.

This script intentionally imports only the Python standard library. A custody
launcher gives it one public evidence file and write access only to its fresh
invocation directory. The assessor machine and latent semantics are never
mounted into this process.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import struct
import sys


ROLE_SCHEMA = "efc-candidate-compiler-role-v1"
JSON_SCHEMA = "efc-raw-world-evidence-v2"
LINE_MAGIC = "EFC-RAW-LINES-V1"
CYCLE_MAGIC = "EFC-CYCLE-PROGRAM-V1"
MACHINE_MAGIC = b"EFCMACH\0"
MACHINE_SIZE = 1_536
MACHINE_HASH_OFFSET = 1_504
FORMAT_VERSION = 1
MAX_STATES = 16
MAX_ACTIONS = 8
STATE_COUNT = 5
ACTION_COUNT = 3
OBSERVER_COUNT = 2
ANSWER_COUNT = 5
HEX_U64 = re.compile(r"[0-9a-f]{16}")
STATE_TOKEN = re.compile(r"s#([0-9a-f]{16})")
ACTION_LINE = re.compile(r"  a#([0-9a-f]{16}) := (.+);")
OBSERVER_LINE = re.compile(r"  o#([0-9a-f]{16}) := (.+);")
CYCLE_CLAUSE = re.compile(r"cycle\[(.+)\]")
CLASS_CLAUSE = re.compile(r"class\[(0|[1-9][0-9]{0,19})\]\{(.+)\}")
STATES_LINE = re.compile(r"states = \[(.+)\];")


class CandidateRoleError(RuntimeError):
    """The candidate input, custody boundary, or output is invalid."""


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("ascii")


def _plain_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise CandidateRoleError(f"{field} must be a plain integer")
    return value


def _validate_filename(value: str) -> str:
    path = Path(value)
    if path.is_absolute() or len(path.parts) != 1 or path.name in {"", ".", ".."}:
        raise CandidateRoleError("role filenames must be one relative component")
    return value


def _regular_files() -> tuple[str, ...]:
    return tuple(sorted(path.name for path in Path.cwd().iterdir() if path.is_file()))


def _decode_json(payload: bytes) -> dict[str, object]:
    try:
        row = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise CandidateRoleError("JSON evidence is malformed") from exc
    if _canonical_json_bytes(row) != payload:
        raise CandidateRoleError("JSON evidence is not canonical")
    if not isinstance(row, dict):
        raise CandidateRoleError("JSON evidence root must be an object")
    return row


def _decode_lines(payload: bytes) -> dict[str, object]:
    try:
        text = payload.decode("ascii")
    except UnicodeDecodeError as exc:
        raise CandidateRoleError("line evidence is not ASCII") from exc
    if not text.endswith("\n") or "\r" in text:
        raise CandidateRoleError("line evidence has noncanonical newlines")
    lines = text[:-1].split("\n")
    if not lines or lines[0] != f"{LINE_MAGIC}\t1":
        raise CandidateRoleError("line evidence header is invalid")
    demonstrations: list[dict[str, int]] = []
    observations: list[dict[str, int]] = []
    observation_phase = False
    for line in lines[1:]:
        fields = line.split("\t")
        if len(fields) != 4:
            raise CandidateRoleError("line evidence field count is invalid")
        if fields[0] == "D":
            if observation_phase or not all(
                HEX_U64.fullmatch(field) for field in fields[1:]
            ):
                raise CandidateRoleError("line transition is noncanonical")
            demonstrations.append(
                {
                    "action_key": int(fields[1], 16),
                    "source_key": int(fields[2], 16),
                    "target_key": int(fields[3], 16),
                }
            )
        elif fields[0] == "O":
            observation_phase = True
            if (
                not HEX_U64.fullmatch(fields[1])
                or not HEX_U64.fullmatch(fields[2])
                or not fields[3].isdigit()
                or (len(fields[3]) > 1 and fields[3].startswith("0"))
                or len(fields[3]) > 20
            ):
                raise CandidateRoleError("line observation is noncanonical")
            observations.append(
                {
                    "answer": int(fields[3]),
                    "observer_key": int(fields[1], 16),
                    "state_key": int(fields[2], 16),
                }
            )
        else:
            raise CandidateRoleError("line evidence kind is invalid")
    return {
        "demonstrations": demonstrations,
        "observations": observations,
        "renderer_choice": 1,
        "schema": JSON_SCHEMA,
    }


def _cycle_state_list(source: str) -> tuple[int, ...]:
    values: list[int] = []
    for token in source.split(","):
        match = STATE_TOKEN.fullmatch(token)
        if match is None:
            raise CandidateRoleError("cycle state spelling is invalid")
        value = int(match.group(1), 16)
        if value == 0:
            raise CandidateRoleError("cycle state key is zero")
        values.append(value)
    if not values or len(values) > MAX_STATES or len(set(values)) != len(values):
        raise CandidateRoleError("cycle state list is invalid")
    return tuple(values)


def _decode_cycle_program(payload: bytes) -> dict[str, object]:
    try:
        text = payload.decode("ascii")
    except UnicodeDecodeError as exc:
        raise CandidateRoleError("cycle program is not ASCII") from exc
    if (
        not text.endswith("\n")
        or "\r" in text
        or "\t" in text
        or "\x00" in text
        or len(payload) > 65_568
    ):
        raise CandidateRoleError("cycle program framing is invalid")
    lines = text[:-1].split("\n")
    if (
        len(lines) < 9
        or len(lines) > 32
        or any(not line or len(line.encode("ascii")) > 2_048 for line in lines)
        or lines[0] != CYCLE_MAGIC
    ):
        raise CandidateRoleError("cycle program line structure is invalid")
    state_match = STATES_LINE.fullmatch(lines[1])
    if state_match is None:
        raise CandidateRoleError("cycle state declaration is invalid")
    state_keys = _cycle_state_list(state_match.group(1))
    if state_keys != tuple(sorted(state_keys)) or len(state_keys) != STATE_COUNT:
        raise CandidateRoleError("cycle state declaration is noncanonical")
    if lines[2] != "actions = {":
        raise CandidateRoleError("cycle action section is invalid")
    try:
        action_end = lines.index("};", 3)
    except ValueError as exc:
        raise CandidateRoleError("cycle action section is unterminated") from exc
    if action_end + 1 >= len(lines) or lines[action_end + 1] != "observers = {":
        raise CandidateRoleError("cycle observer section is invalid")
    try:
        observer_end = lines.index("};", action_end + 2)
    except ValueError as exc:
        raise CandidateRoleError("cycle observer section is unterminated") from exc
    if observer_end != len(lines) - 2 or lines[-1] != "halt.":
        raise CandidateRoleError("cycle program trailer is invalid")
    action_lines = lines[3:action_end]
    observer_lines = lines[action_end + 2 : observer_end]
    if len(action_lines) != ACTION_COUNT or len(observer_lines) != OBSERVER_COUNT:
        raise CandidateRoleError("cycle declaration counts differ")

    demonstrations: list[dict[str, int]] = []
    action_keys: list[int] = []
    known_states = set(state_keys)
    for line in action_lines:
        match = ACTION_LINE.fullmatch(line)
        if match is None:
            raise CandidateRoleError("cycle action declaration is malformed")
        action = int(match.group(1), 16)
        if action == 0 or action in action_keys:
            raise CandidateRoleError("cycle action key is zero or duplicated")
        action_keys.append(action)
        clauses = match.group(2).split(" * ")
        destination: dict[int, int] = {}
        cycle_starts: list[int] = []
        for clause in clauses:
            cycle_match = CYCLE_CLAUSE.fullmatch(clause)
            if cycle_match is None:
                raise CandidateRoleError("cycle clause is malformed")
            cycle = _cycle_state_list(cycle_match.group(1))
            if any(state not in known_states for state in cycle) or cycle[0] != min(
                cycle
            ):
                raise CandidateRoleError("cycle clause is noncanonical")
            cycle_starts.append(cycle[0])
            for index, source in enumerate(cycle):
                if source in destination:
                    raise CandidateRoleError("cycle clauses overlap")
                destination[source] = cycle[(index + 1) % len(cycle)]
        if cycle_starts != sorted(cycle_starts) or set(destination) != known_states:
            raise CandidateRoleError("cycle clauses are incomplete or unordered")
        demonstrations.extend(
            {
                "action_key": action,
                "source_key": state,
                "target_key": destination[state],
            }
            for state in state_keys
        )
    if action_keys != sorted(action_keys):
        raise CandidateRoleError("cycle action declarations are unordered")

    observations: list[dict[str, int]] = []
    observer_keys: list[int] = []
    for line in observer_lines:
        match = OBSERVER_LINE.fullmatch(line)
        if match is None:
            raise CandidateRoleError("cycle observer declaration is malformed")
        observer = int(match.group(1), 16)
        if observer == 0 or observer in observer_keys:
            raise CandidateRoleError("cycle observer key is zero or duplicated")
        observer_keys.append(observer)
        answer_for_state: dict[int, int] = {}
        answers: list[int] = []
        for clause in match.group(2).split(" + "):
            class_match = CLASS_CLAUSE.fullmatch(clause)
            if class_match is None:
                raise CandidateRoleError("cycle observer class is malformed")
            answer = int(class_match.group(1))
            members = _cycle_state_list(class_match.group(2))
            if (
                answer >= ANSWER_COUNT
                or answer in answers
                or members != tuple(sorted(members))
                or any(state not in known_states for state in members)
            ):
                raise CandidateRoleError("cycle observer class is noncanonical")
            answers.append(answer)
            for state in members:
                if state in answer_for_state:
                    raise CandidateRoleError("cycle observer classes overlap")
                answer_for_state[state] = answer
        if answers != sorted(answers) or set(answer_for_state) != known_states:
            raise CandidateRoleError("cycle observer partition is invalid")
        observations.extend(
            {
                "answer": answer_for_state[state],
                "observer_key": observer,
                "state_key": state,
            }
            for state in state_keys
        )
    if observer_keys != sorted(observer_keys):
        raise CandidateRoleError("cycle observer declarations are unordered")
    return {
        "demonstrations": demonstrations,
        "observations": observations,
        "renderer_choice": 2,
        "schema": JSON_SCHEMA,
    }


def _parse_evidence(
    payload: bytes,
) -> tuple[
    tuple[int, ...],
    tuple[int, ...],
    tuple[int, ...],
    tuple[tuple[int, ...], ...],
    tuple[tuple[int, ...], ...],
]:
    if payload.startswith((LINE_MAGIC + "\t").encode("ascii")):
        row = _decode_lines(payload)
        renderer = 1
    elif payload.startswith((CYCLE_MAGIC + "\n").encode("ascii")):
        row = _decode_cycle_program(payload)
        renderer = 2
    else:
        row = _decode_json(payload)
        renderer = 0
    if set(row) != {
        "demonstrations",
        "observations",
        "renderer_choice",
        "schema",
    }:
        raise CandidateRoleError("evidence fields differ from the frozen schema")
    if (
        row["schema"] != JSON_SCHEMA
        or _plain_int(row["renderer_choice"], "renderer_choice") != renderer
    ):
        raise CandidateRoleError("evidence schema or renderer differs")

    raw_demonstrations = row["demonstrations"]
    if not isinstance(raw_demonstrations, list):
        raise CandidateRoleError("demonstrations must be a list")
    demonstrations: list[tuple[int, int, int]] = []
    states: set[int] = set()
    actions: set[int] = set()
    for event in raw_demonstrations:
        if not isinstance(event, dict) or set(event) != {
            "action_key",
            "source_key",
            "target_key",
        }:
            raise CandidateRoleError("transition event is malformed")
        action = _plain_int(event["action_key"], "action_key")
        source = _plain_int(event["source_key"], "source_key")
        target = _plain_int(event["target_key"], "target_key")
        if min(action, source, target) <= 0 or max(action, source, target) >= 1 << 64:
            raise CandidateRoleError("transition key is not nonzero uint64")
        actions.add(action)
        states.update((source, target))
        demonstrations.append((action, source, target))
    if len(states) != STATE_COUNT or len(actions) != ACTION_COUNT:
        raise CandidateRoleError("inferred state or action count differs")
    state_keys = tuple(sorted(states))
    action_keys = tuple(sorted(actions))
    state_index = {key: index for index, key in enumerate(state_keys)}
    action_index = {key: index for index, key in enumerate(action_keys)}
    transition_rows: list[list[int | None]] = [
        [None] * STATE_COUNT for _ in range(ACTION_COUNT)
    ]
    for action_key, source_key, target_key in demonstrations:
        action = action_index[action_key]
        source = state_index[source_key]
        if transition_rows[action][source] is not None:
            raise CandidateRoleError("duplicate transition event")
        transition_rows[action][source] = state_index[target_key]
    if any(cell is None for row_value in transition_rows for cell in row_value):
        raise CandidateRoleError("transition table is incomplete")

    raw_observations = row["observations"]
    if not isinstance(raw_observations, list):
        raise CandidateRoleError("observations must be a list")
    observations: list[tuple[int, int, int]] = []
    observer_keys_set: set[int] = set()
    for event in raw_observations:
        if not isinstance(event, dict) or set(event) != {
            "answer",
            "observer_key",
            "state_key",
        }:
            raise CandidateRoleError("observation event is malformed")
        observer = _plain_int(event["observer_key"], "observer_key")
        state = _plain_int(event["state_key"], "state_key")
        answer = _plain_int(event["answer"], "answer")
        if observer <= 0 or observer >= 1 << 64 or state not in state_index:
            raise CandidateRoleError("observation contains an invalid key")
        if answer < 0 or answer >= ANSWER_COUNT:
            raise CandidateRoleError("observation answer is outside the alphabet")
        observer_keys_set.add(observer)
        observations.append((observer, state, answer))
    if len(observer_keys_set) != OBSERVER_COUNT:
        raise CandidateRoleError("inferred observer count differs")
    observer_keys = tuple(sorted(observer_keys_set))
    observer_index = {key: index for index, key in enumerate(observer_keys)}
    observer_rows: list[list[int | None]] = [
        [None] * STATE_COUNT for _ in range(OBSERVER_COUNT)
    ]
    for observer_key, state_key, answer in observations:
        observer = observer_index[observer_key]
        state = state_index[state_key]
        if observer_rows[observer][state] is not None:
            raise CandidateRoleError("duplicate observation event")
        observer_rows[observer][state] = answer
    if any(cell is None for row_value in observer_rows for cell in row_value):
        raise CandidateRoleError("observer table is incomplete")
    return (
        state_keys,
        action_keys,
        observer_keys,
        tuple(tuple(int(cell) for cell in row_value) for row_value in transition_rows),
        tuple(tuple(int(cell) for cell in row_value) for row_value in observer_rows),
    )


def _put_u16(buffer: bytearray, offset: int, value: int) -> None:
    struct.pack_into("<H", buffer, offset, value)


def _put_u32(buffer: bytearray, offset: int, value: int) -> None:
    struct.pack_into("<I", buffer, offset, value)


def _put_u64(buffer: bytearray, offset: int, value: int) -> None:
    struct.pack_into("<Q", buffer, offset, value)


def _compile_machine(evidence: bytes) -> bytes:
    state_keys, action_keys, observer_keys, transitions, observations = _parse_evidence(
        evidence
    )
    machine = bytearray(MACHINE_SIZE)
    machine[:8] = MACHINE_MAGIC
    _put_u32(machine, 8, FORMAT_VERSION)
    _put_u32(machine, 12, 64)
    _put_u32(machine, 16, MACHINE_SIZE)
    _put_u16(machine, 24, STATE_COUNT)
    _put_u16(machine, 26, ACTION_COUNT)
    _put_u16(machine, 28, OBSERVER_COUNT)
    _put_u64(machine, 32, (1 << STATE_COUNT) - 1)
    _put_u64(machine, 40, (1 << ACTION_COUNT) - 1)
    _put_u64(machine, 48, (1 << OBSERVER_COUNT) - 1)
    for slot, key in enumerate(state_keys):
        _put_u64(machine, 64 + slot * 8, key)
    for slot, key in enumerate(action_keys):
        _put_u64(machine, 192 + slot * 8, key)
    for slot, key in enumerate(observer_keys):
        _put_u64(machine, 256 + slot * 8, key)
    for action, relation in enumerate(transitions):
        for state, destination in enumerate(relation):
            machine[320 + action * MAX_STATES + state] = destination
    for observer, row in enumerate(observations):
        for state, answer in enumerate(row):
            _put_u64(
                machine,
                448 + (observer * MAX_STATES + state) * 8,
                answer,
            )
    machine[MACHINE_HASH_OFFSET:] = hashlib.sha256(
        machine[:MACHINE_HASH_OFFSET]
    ).digest()
    return bytes(machine)


def _write_immutable(filename: str, payload: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(filename, flags, 0o400)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise CandidateRoleError("output write made no progress")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _sandbox_receipt() -> tuple[str, str]:
    if (
        os.environ.get("SHOHIN_EFC_SANDBOX_ENFORCED") == "1"
        and os.environ.get("SHOHIN_EFC_SANDBOX_STAGE")
        and os.environ.get("SHOHIN_EFC_SANDBOX_POLICY_SHA256")
    ):
        return (
            os.environ["SHOHIN_EFC_SANDBOX_STAGE"],
            os.environ["SHOHIN_EFC_SANDBOX_POLICY_SHA256"],
        )
    if (
        os.environ.get("SHOHIN_LANDLOCK_ENFORCED") == "1"
        and os.environ.get("SHOHIN_LANDLOCK_STAGE")
        and os.environ.get("SHOHIN_LANDLOCK_POLICY_SHA256")
    ):
        return (
            os.environ["SHOHIN_LANDLOCK_STAGE"],
            os.environ["SHOHIN_LANDLOCK_POLICY_SHA256"],
        )
    raise CandidateRoleError("candidate role is not inside the frozen sandbox")


def main(argv: list[str] | None = None) -> int:
    arguments = sys.argv[1:] if argv is None else argv
    if len(arguments) != 3:
        raise CandidateRoleError("usage: candidate-role EVIDENCE MACHINE RECEIPT")
    evidence_name, machine_name, receipt_name = (
        _validate_filename(value) for value in arguments
    )
    if len({evidence_name, machine_name, receipt_name}) != 3:
        raise CandidateRoleError("role filenames must be distinct")
    sandbox_stage, sandbox_policy_sha256 = _sandbox_receipt()
    if sandbox_stage != "candidate-compiler":
        raise CandidateRoleError("candidate sandbox stage differs")
    before = _regular_files()
    if before != (evidence_name,):
        raise CandidateRoleError("candidate invocation contains undeclared files")
    evidence_path = Path(evidence_name)
    if evidence_path.stat().st_size > 1024 * 1024:
        raise CandidateRoleError("candidate evidence exceeds one MiB")
    evidence = evidence_path.read_bytes()
    machine = _compile_machine(evidence)
    _write_immutable(machine_name, machine)
    receipt = {
        "candidate_source_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "declared_input_files": [evidence_name],
        "declared_output_files": [machine_name, receipt_name],
        "evidence_sha256": hashlib.sha256(evidence).hexdigest(),
        "machine_bytes": len(machine),
        "machine_sha256": hashlib.sha256(machine).hexdigest(),
        "regular_files_before": list(before),
        "sandbox_enforced": True,
        "sandbox_policy_sha256": sandbox_policy_sha256,
        "sandbox_stage": sandbox_stage,
        "schema": ROLE_SCHEMA,
    }
    _write_immutable(receipt_name, _canonical_json_bytes(receipt))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (CandidateRoleError, OSError) as exc:
        print(f"efc-candidate-role: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(125) from exc
