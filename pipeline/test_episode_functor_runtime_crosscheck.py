from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
from pathlib import Path
import shutil
import struct
import subprocess
from typing import Callable

import pytest


ROOT = Path(__file__).resolve().parents[1]
C_SOURCE = ROOT / "tools" / "episode_functor_runtime_c.c"
RUST_SOURCE = ROOT / "tools" / "episode_functor_runtime_rust.rs"

MACHINE_SIZE = 1536
MACHINE_HASH_OFFSET = 1504
QUERY_HEADER_SIZE = 64
QUERY_RECORD_SIZE = 320
TRANSCRIPT_HEADER_SIZE = 96
TRANSCRIPT_RECORD_SIZE = 32
HASH_SIZE = 32
MAX_STATES = 16
MAX_ACTIONS = 8
MAX_OBSERVERS = 8
MAX_WORD = 32

MACHINE_MAGIC = b"EFCMACH\0"
QUERY_MAGIC = b"EFCQRY\0\0"
TRANSCRIPT_MAGIC = b"EFCOUT\0\0"

# This file deliberately constructs the wire format from bytes using only the
# Python standard library. It imports neither Shohin code nor either runtime.


@dataclass(frozen=True)
class MachineSpec:
    initial_slot: int
    state_keys: dict[int, int]
    action_keys: dict[int, int]
    observer_keys: dict[int, int]
    transitions: dict[int, dict[int, int]]
    observations: dict[int, dict[int, int]]


QuerySpec = tuple[int, int, int, tuple[int, ...]]
TranscriptRecord = tuple[int, int, int, int, int, int, int]


def _put_u16(buffer: bytearray, offset: int, value: int) -> None:
    struct.pack_into("<H", buffer, offset, value)


def _put_u32(buffer: bytearray, offset: int, value: int) -> None:
    struct.pack_into("<I", buffer, offset, value)


def _put_u64(buffer: bytearray, offset: int, value: int) -> None:
    struct.pack_into("<Q", buffer, offset, value)


def _mask(slots: dict[int, int]) -> int:
    return sum(1 << slot for slot in slots)


def seal_machine(buffer: bytearray) -> bytes:
    assert len(buffer) == MACHINE_SIZE
    buffer[MACHINE_HASH_OFFSET:] = hashlib.sha256(
        buffer[:MACHINE_HASH_OFFSET]
    ).digest()
    return bytes(buffer)


def seal_queries(buffer: bytearray) -> bytes:
    buffer[-HASH_SIZE:] = hashlib.sha256(buffer[:-HASH_SIZE]).digest()
    return bytes(buffer)


def encode_machine(spec: MachineSpec) -> bytes:
    machine = bytearray(MACHINE_SIZE)
    machine[:8] = MACHINE_MAGIC
    _put_u32(machine, 8, 1)
    _put_u32(machine, 12, 64)
    _put_u32(machine, 16, MACHINE_SIZE)
    _put_u16(machine, 24, len(spec.state_keys))
    _put_u16(machine, 26, len(spec.action_keys))
    _put_u16(machine, 28, len(spec.observer_keys))
    _put_u64(machine, 32, _mask(spec.state_keys))
    _put_u64(machine, 40, _mask(spec.action_keys))
    _put_u64(machine, 48, _mask(spec.observer_keys))
    machine[56] = spec.initial_slot

    for slot, key in spec.state_keys.items():
        _put_u64(machine, 64 + slot * 8, key)
    for slot, key in spec.action_keys.items():
        _put_u64(machine, 192 + slot * 8, key)
    for slot, key in spec.observer_keys.items():
        _put_u64(machine, 256 + slot * 8, key)
    for action_slot, relation in spec.transitions.items():
        for state_slot, destination in relation.items():
            machine[
                320 + action_slot * MAX_STATES + state_slot
            ] = destination
    for observer_slot, row in spec.observations.items():
        for state_slot, answer in row.items():
            _put_u64(
                machine,
                448 + (observer_slot * MAX_STATES + state_slot) * 8,
                answer,
            )
    return seal_machine(machine)


def encode_queries(
    machine: bytes,
    records: tuple[QuerySpec, ...],
) -> bytes:
    queries = bytearray(
        QUERY_HEADER_SIZE + len(records) * QUERY_RECORD_SIZE + HASH_SIZE
    )
    queries[:8] = QUERY_MAGIC
    _put_u32(queries, 8, 1)
    _put_u32(queries, 12, QUERY_HEADER_SIZE)
    _put_u32(queries, 16, QUERY_RECORD_SIZE)
    _put_u32(queries, 20, len(records))
    queries[24:56] = machine[MACHINE_HASH_OFFSET:]
    for index, (challenge, start, observer, word) in enumerate(records):
        offset = QUERY_HEADER_SIZE + index * QUERY_RECORD_SIZE
        _put_u64(queries, offset, challenge)
        _put_u64(queries, offset + 8, start)
        _put_u64(queries, offset + 16, observer)
        _put_u16(queries, offset + 24, len(word))
        for word_index, action in enumerate(word):
            _put_u64(queries, offset + 32 + word_index * 8, action)
    return seal_queries(queries)


def base_spec() -> MachineSpec:
    states = (0, 2, 5, 9)
    action_a = {0: 2, 2: 5, 5: 9, 9: 0}
    action_b = {0: 2, 2: 0, 5: 5, 9: 9}
    identity = {state: state for state in states}
    return MachineSpec(
        initial_slot=0,
        state_keys={0: 101, 2: 303, 5: 606, 9: 1000},
        action_keys={0: 1001, 3: 4004, 6: 7007},
        observer_keys={1: 9009, 5: 5005},
        transitions={0: action_a, 3: action_b, 6: identity},
        observations={
            1: {0: 700, 2: 800, 5: 900, 9: 1000},
            5: {0: 17, 2: 19, 5: 23, 9: 29},
        },
    )


def valid_queries() -> tuple[QuerySpec, ...]:
    a, b, identity = 1001, 4004, 7007
    return (
        (11, 101, 9009, ()),
        (12, 101, 9009, (a, a, a, a)),
        (13, 101, 9009, (b, b)),
        (14, 101, 9009, (a, b)),
        (15, 101, 9009, (b, a)),
        (16, 303, 5005, (a, identity, b)),
        (17, 606, 5005, (identity,) * MAX_WORD),
        (18, 1000, 9009, (a, b, a, b, identity, a)),
    )


def execute_oracle(
    spec: MachineSpec,
    records: tuple[QuerySpec, ...],
) -> tuple[TranscriptRecord, ...]:
    state_by_key = {key: slot for slot, key in spec.state_keys.items()}
    action_by_key = {key: slot for slot, key in spec.action_keys.items()}
    observer_by_key = {
        key: slot for slot, key in spec.observer_keys.items()
    }
    output: list[TranscriptRecord] = []
    for challenge, start_key, observer_key, word in records:
        state = state_by_key[start_key]
        for action_key in word:
            state = spec.transitions[action_by_key[action_key]][state]
        observer = observer_by_key[observer_key]
        output.append(
            (
                challenge,
                spec.state_keys[state],
                spec.observations[observer][state],
                state,
                0,
                len(word),
                0,
            )
        )
    return tuple(output)


def decode_transcript(transcript: bytes) -> tuple[TranscriptRecord, ...]:
    assert transcript[:8] == TRANSCRIPT_MAGIC
    count = struct.unpack_from("<I", transcript, 20)[0]
    return tuple(
        struct.unpack_from(
            "<QQQHHHH",
            transcript,
            TRANSCRIPT_HEADER_SIZE + index * TRANSCRIPT_RECORD_SIZE,
        )
        for index in range(count)
    )


@pytest.fixture(scope="session")
def runtimes(
    tmp_path_factory: pytest.TempPathFactory,
) -> dict[str, Path]:
    cc = shutil.which("cc")
    rustc = shutil.which("rustc")
    if cc is None or rustc is None:
        pytest.skip("strict C and Rust compilers are required")
    root = tmp_path_factory.mktemp("episode_functor_crosscheck")
    c_runtime = root / "runtime_c"
    rust_runtime = root / "runtime_rust"
    subprocess.run(
        [
            cc,
            "-std=c11",
            "-O2",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-pedantic",
            str(C_SOURCE),
            "-o",
            str(c_runtime),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [
            rustc,
            "--edition=2021",
            "-C",
            "opt-level=2",
            "-D",
            "warnings",
            str(RUST_SOURCE),
            "-o",
            str(rust_runtime),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return {"c": c_runtime, "rust": rust_runtime}


def run_one(
    runtime: Path,
    root: Path,
    machine: bytes,
    queries: bytes,
) -> tuple[subprocess.CompletedProcess[str], Path]:
    root.mkdir(parents=True)
    machine_path = root / "machine.bin"
    query_path = root / "queries.bin"
    transcript_path = root / "transcript.bin"
    machine_path.write_bytes(machine)
    query_path.write_bytes(queries)
    completed = subprocess.run(
        [runtime, machine_path, query_path, transcript_path],
        capture_output=True,
        text=True,
        check=False,
    )
    return completed, transcript_path


def run_both(
    runtimes: dict[str, Path],
    root: Path,
    machine: bytes,
    queries: bytes,
) -> tuple[bytes, bytes]:
    transcripts: dict[str, bytes] = {}
    for name, runtime in runtimes.items():
        completed, transcript_path = run_one(
            runtime,
            root / name,
            machine,
            queries,
        )
        assert completed.returncode == 0, (
            f"{name} rejected valid input: {completed.stderr}"
        )
        transcripts[name] = transcript_path.read_bytes()
    return transcripts["c"], transcripts["rust"]


def reject_both(
    runtimes: dict[str, Path],
    root: Path,
    machine: bytes,
    queries: bytes,
) -> None:
    for name, runtime in runtimes.items():
        completed, transcript_path = run_one(
            runtime,
            root / name,
            machine,
            queries,
        )
        assert completed.returncode != 0, f"{name} accepted malformed input"
        assert completed.stderr
        assert not transcript_path.exists()


def assert_valid_transcript_bindings(
    transcript: bytes,
    machine: bytes,
    queries: bytes,
) -> None:
    count = struct.unpack_from("<I", queries, 20)[0]
    assert len(transcript) == (
        TRANSCRIPT_HEADER_SIZE + count * TRANSCRIPT_RECORD_SIZE + HASH_SIZE
    )
    assert struct.unpack_from("<III", transcript, 8) == (
        1,
        TRANSCRIPT_HEADER_SIZE,
        TRANSCRIPT_RECORD_SIZE,
    )
    assert transcript[24:56] == machine[MACHINE_HASH_OFFSET:]
    assert transcript[56:88] == queries[-HASH_SIZE:]
    assert transcript[88:96] == bytes(8)
    assert transcript[-HASH_SIZE:] == hashlib.sha256(
        transcript[:-HASH_SIZE]
    ).digest()


def test_byte_identical_valid_transcripts_and_independent_oracle(
    runtimes: dict[str, Path],
    tmp_path: Path,
) -> None:
    spec = base_spec()
    records = valid_queries()
    machine = encode_machine(spec)
    queries = encode_queries(machine, records)
    c_transcript, rust_transcript = run_both(
        runtimes, tmp_path, machine, queries
    )

    assert c_transcript == rust_transcript
    assert_valid_transcript_bindings(c_transcript, machine, queries)
    assert decode_transcript(c_transcript) == execute_oracle(spec, records)

    decoded = decode_transcript(c_transcript)
    assert decoded[0][1:3] == decoded[1][1:3] == decoded[2][1:3]
    assert decoded[3][1:3] != decoded[4][1:3]


def conjugate_states(
    spec: MachineSpec,
    gauge: dict[int, int],
) -> MachineSpec:
    return MachineSpec(
        initial_slot=gauge[spec.initial_slot],
        state_keys={
            gauge[slot]: key for slot, key in spec.state_keys.items()
        },
        action_keys=dict(spec.action_keys),
        observer_keys=dict(spec.observer_keys),
        transitions={
            action: {
                gauge[source]: gauge[destination]
                for source, destination in relation.items()
            }
            for action, relation in spec.transitions.items()
        },
        observations={
            observer: {
                gauge[state]: answer for state, answer in row.items()
            }
            for observer, row in spec.observations.items()
        },
    )


def test_state_gauge_conjugacy(
    runtimes: dict[str, Path],
    tmp_path: Path,
) -> None:
    original = base_spec()
    gauge = {0: 1, 2: 4, 5: 7, 9: 12}
    conjugated = conjugate_states(original, gauge)
    records = valid_queries()

    original_machine = encode_machine(original)
    original_queries = encode_queries(original_machine, records)
    original_c, original_rust = run_both(
        runtimes,
        tmp_path / "original",
        original_machine,
        original_queries,
    )
    conjugated_machine = encode_machine(conjugated)
    conjugated_queries = encode_queries(conjugated_machine, records)
    conjugated_c, conjugated_rust = run_both(
        runtimes,
        tmp_path / "conjugated",
        conjugated_machine,
        conjugated_queries,
    )
    assert original_c == original_rust
    assert conjugated_c == conjugated_rust

    left = decode_transcript(original_c)
    right = decode_transcript(conjugated_c)
    for left_record, right_record in zip(left, right, strict=True):
        assert left_record[:3] == right_record[:3]
        assert right_record[3] == gauge[left_record[3]]
        assert left_record[4:] == right_record[4:]


def swap_action_rows(
    spec: MachineSpec,
    *,
    keys: bool,
    operators: bool,
) -> MachineSpec:
    action_keys = dict(spec.action_keys)
    transitions = {
        slot: dict(row) for slot, row in spec.transitions.items()
    }
    if keys:
        action_keys[0], action_keys[3] = action_keys[3], action_keys[0]
    if operators:
        transitions[0], transitions[3] = (
            transitions[3],
            transitions[0],
        )
    return replace(
        spec,
        action_keys=action_keys,
        transitions=transitions,
    )


def _semantic_records(
    transcript: bytes,
) -> tuple[tuple[int, int, int, int, int, int], ...]:
    return tuple(
        record[:3] + record[4:] for record in decode_transcript(transcript)
    )


def test_key_operator_and_compensated_interventions(
    runtimes: dict[str, Path],
    tmp_path: Path,
) -> None:
    records: tuple[QuerySpec, ...] = (
        (91, 101, 9009, (1001, 4004)),
        (92, 101, 9009, (4004, 1001)),
        (93, 303, 5005, (1001,)),
    )
    arms = {
        "base": base_spec(),
        "key_only": swap_action_rows(
            base_spec(), keys=True, operators=False
        ),
        "operator_only": swap_action_rows(
            base_spec(), keys=False, operators=True
        ),
        "compensated": swap_action_rows(
            base_spec(), keys=True, operators=True
        ),
    }
    results: dict[str, bytes] = {}
    for name, spec in arms.items():
        machine = encode_machine(spec)
        queries = encode_queries(machine, records)
        c_transcript, rust_transcript = run_both(
            runtimes,
            tmp_path / name,
            machine,
            queries,
        )
        assert c_transcript == rust_transcript
        results[name] = c_transcript

    base = _semantic_records(results["base"])
    key_only = _semantic_records(results["key_only"])
    operator_only = _semantic_records(results["operator_only"])
    compensated = _semantic_records(results["compensated"])
    assert key_only == operator_only
    assert key_only != base
    assert compensated == base
    assert results["compensated"] != results["base"]


def test_equivalent_words_noncommuting_reversals_and_query_hash_binding(
    runtimes: dict[str, Path],
    tmp_path: Path,
) -> None:
    spec = base_spec()
    machine = encode_machine(spec)
    a, b, identity = 1001, 4004, 7007
    left_records: tuple[QuerySpec, ...] = (
        (301, 101, 9009, (identity, identity)),
        (302, 101, 9009, (a, b)),
        (303, 101, 9009, (b, a)),
    )
    right_records: tuple[QuerySpec, ...] = (
        (301, 101, 9009, (b, b)),
        (302, 101, 9009, (a, b)),
        (303, 101, 9009, (b, a)),
    )
    left_queries = encode_queries(machine, left_records)
    right_queries = encode_queries(machine, right_records)
    left_c, left_rust = run_both(
        runtimes, tmp_path / "left", machine, left_queries
    )
    right_c, right_rust = run_both(
        runtimes, tmp_path / "right", machine, right_queries
    )
    assert left_c == left_rust
    assert right_c == right_rust

    left_decoded = decode_transcript(left_c)
    right_decoded = decode_transcript(right_c)
    assert left_decoded == right_decoded
    assert left_decoded[1][1:3] != left_decoded[2][1:3]
    assert left_c[56:88] == left_queries[-HASH_SIZE:]
    assert right_c[56:88] == right_queries[-HASH_SIZE:]
    assert left_c[56:88] != right_c[56:88]
    assert left_c != right_c


MachineMutation = Callable[[bytearray], None]
QueryMutation = Callable[[bytearray], None]


def _set_machine_padding(data: bytearray) -> None:
    data[1472] = 1


def _set_inactive_state_key(data: bytearray) -> None:
    _put_u64(data, 64 + 1 * 8, 777)


def _zero_active_action_key(data: bytearray) -> None:
    _put_u64(data, 192, 0)


def _duplicate_observer_key(data: bytearray) -> None:
    _put_u64(data, 256 + 5 * 8, 9009)


def _set_transition_padding(data: bytearray) -> None:
    data[320 + 1 * MAX_STATES] = 2


def _set_invalid_destination(data: bytearray) -> None:
    data[320] = 1


def _set_observer_padding(data: bytearray) -> None:
    _put_u64(data, 448, 55)


@pytest.mark.parametrize(
    "mutate",
    (
        _set_machine_padding,
        _set_inactive_state_key,
        _zero_active_action_key,
        _duplicate_observer_key,
        _set_transition_padding,
        _set_invalid_destination,
        _set_observer_padding,
    ),
    ids=(
        "reserved-padding",
        "inactive-key",
        "zero-key",
        "duplicate-key",
        "transition-padding",
        "invalid-destination",
        "observer-padding",
    ),
)
def test_both_reject_resealed_malformed_machines(
    runtimes: dict[str, Path],
    tmp_path: Path,
    mutate: MachineMutation,
) -> None:
    machine = bytearray(encode_machine(base_spec()))
    mutate(machine)
    malformed = seal_machine(machine)
    queries = encode_queries(malformed, valid_queries())
    reject_both(runtimes, tmp_path, malformed, queries)


def test_both_reject_machine_length_and_stale_hash(
    runtimes: dict[str, Path],
    tmp_path: Path,
) -> None:
    machine = encode_machine(base_spec())
    queries = encode_queries(machine, valid_queries())
    reject_both(
        runtimes,
        tmp_path / "short",
        machine[:-1],
        queries,
    )
    stale = bytearray(machine)
    stale[448 + (1 * MAX_STATES + 0) * 8] ^= 1
    reject_both(
        runtimes,
        tmp_path / "hash",
        bytes(stale),
        queries,
    )


def _set_query_header_padding(data: bytearray) -> None:
    _put_u32(data, 60, 1)


def _set_query_record_padding(data: bytearray) -> None:
    data[QUERY_HEADER_SIZE + 288] = 1


def _set_query_action_padding(data: bytearray) -> None:
    first_length = struct.unpack_from("<H", data, QUERY_HEADER_SIZE + 24)[0]
    _put_u64(
        data,
        QUERY_HEADER_SIZE + 32 + first_length * 8,
        1001,
    )


def _set_unknown_state_key(data: bytearray) -> None:
    _put_u64(data, QUERY_HEADER_SIZE + 8, 123_456)


def _set_unknown_observer_key(data: bytearray) -> None:
    _put_u64(data, QUERY_HEADER_SIZE + 16, 123_456)


def _set_unknown_action_key(data: bytearray) -> None:
    offset = QUERY_HEADER_SIZE + QUERY_RECORD_SIZE + 32
    _put_u64(data, offset, 123_456)


def _set_zero_challenge(data: bytearray) -> None:
    _put_u64(data, QUERY_HEADER_SIZE, 0)


def _set_duplicate_challenge(data: bytearray) -> None:
    first = struct.unpack_from("<Q", data, QUERY_HEADER_SIZE)[0]
    _put_u64(data, QUERY_HEADER_SIZE + QUERY_RECORD_SIZE, first)


def _set_wrong_machine_binding(data: bytearray) -> None:
    data[24] ^= 1


@pytest.mark.parametrize(
    "mutate",
    (
        _set_query_header_padding,
        _set_query_record_padding,
        _set_query_action_padding,
        _set_unknown_state_key,
        _set_unknown_observer_key,
        _set_unknown_action_key,
        _set_zero_challenge,
        _set_duplicate_challenge,
        _set_wrong_machine_binding,
    ),
    ids=(
        "header-padding",
        "record-padding",
        "action-padding",
        "unknown-state-key",
        "unknown-observer-key",
        "unknown-action-key",
        "zero-challenge",
        "duplicate-challenge",
        "machine-hash-binding",
    ),
)
def test_both_reject_resealed_malformed_queries(
    runtimes: dict[str, Path],
    tmp_path: Path,
    mutate: QueryMutation,
) -> None:
    machine = encode_machine(base_spec())
    queries = bytearray(encode_queries(machine, valid_queries()))
    mutate(queries)
    reject_both(runtimes, tmp_path, machine, seal_queries(queries))


def test_both_reject_query_length_and_stale_exact_hash(
    runtimes: dict[str, Path],
    tmp_path: Path,
) -> None:
    machine = encode_machine(base_spec())
    queries = encode_queries(machine, valid_queries())
    reject_both(
        runtimes,
        tmp_path / "short",
        machine,
        queries[:-1],
    )
    stale = bytearray(queries)
    stale[QUERY_HEADER_SIZE + 32] ^= 1
    reject_both(
        runtimes,
        tmp_path / "hash",
        machine,
        bytes(stale),
    )


def test_both_reject_extra_source_argument(
    runtimes: dict[str, Path],
    tmp_path: Path,
) -> None:
    machine = encode_machine(base_spec())
    queries = encode_queries(machine, valid_queries())
    for name, runtime in runtimes.items():
        root = tmp_path / name
        root.mkdir()
        machine_path = root / "machine.bin"
        query_path = root / "queries.bin"
        output_path = root / "transcript.bin"
        machine_path.write_bytes(machine)
        query_path.write_bytes(queries)
        completed = subprocess.run(
            [
                runtime,
                machine_path,
                query_path,
                output_path,
                root / "forbidden-source.jsonl",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        assert completed.returncode == 2
        assert "usage:" in completed.stderr
        assert not output_path.exists()
