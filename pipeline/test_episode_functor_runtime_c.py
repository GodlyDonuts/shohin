from __future__ import annotations

import hashlib
from pathlib import Path
import shutil
import struct
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "tools" / "episode_functor_runtime_c.c"

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

# The fixture builders intentionally duplicate the C wire specification and
# import no Shohin serializer or oracle:
#
# MACHINE.bin is one fixed 1536-byte little-endian record:
#   64-byte header; 16/8/8 uint64 state/action/observer key slots;
#   8*16 uint8 flat transitions; 8*16 uint64 observer values;
#   32 zero bytes; SHA-256 of bytes 0:1504.
#
# QUERIES.bin is a 64-byte header, fixed 320-byte records, and a trailing
# SHA-256. Each record contains challenge/start/observer keys, uint16 word
# length, 32 uint64 action-key slots, and strict zero padding.
#
# TRANSCRIPT.bin is a 96-byte header, fixed 32-byte records, and a trailing
# SHA-256. The header binds both the machine and query payload SHA-256 values.
# Records contain challenge ID, final state key, answer, final state slot,
# status, step count, and reserved zero bytes.


def _put_u16(buffer: bytearray, offset: int, value: int) -> None:
    struct.pack_into("<H", buffer, offset, value)


def _put_u32(buffer: bytearray, offset: int, value: int) -> None:
    struct.pack_into("<I", buffer, offset, value)


def _put_u64(buffer: bytearray, offset: int, value: int) -> None:
    struct.pack_into("<Q", buffer, offset, value)


def _seal_machine(buffer: bytearray) -> bytes:
    assert len(buffer) == MACHINE_SIZE
    buffer[MACHINE_HASH_OFFSET:] = hashlib.sha256(
        buffer[:MACHINE_HASH_OFFSET]
    ).digest()
    return bytes(buffer)


def _seal_queries(buffer: bytearray) -> bytes:
    buffer[-HASH_SIZE:] = hashlib.sha256(buffer[:-HASH_SIZE]).digest()
    return bytes(buffer)


def build_machine() -> bytes:
    machine = bytearray(MACHINE_SIZE)
    machine[:8] = MACHINE_MAGIC
    _put_u32(machine, 8, 1)
    _put_u32(machine, 12, 64)
    _put_u32(machine, 16, MACHINE_SIZE)
    _put_u16(machine, 24, 3)
    _put_u16(machine, 26, 2)
    _put_u16(machine, 28, 1)

    state_mask = (1 << 0) | (1 << 2) | (1 << 5)
    action_mask = (1 << 0) | (1 << 3)
    observer_mask = 1 << 1
    _put_u64(machine, 32, state_mask)
    _put_u64(machine, 40, action_mask)
    _put_u64(machine, 48, observer_mask)
    machine[56] = 0

    state_keys = {0: 101, 2: 303, 5: 606}
    action_keys = {0: 1001, 3: 4004}
    observer_keys = {1: 9009}
    for slot, key in state_keys.items():
        _put_u64(machine, 64 + slot * 8, key)
    for slot, key in action_keys.items():
        _put_u64(machine, 192 + slot * 8, key)
    for slot, key in observer_keys.items():
        _put_u64(machine, 256 + slot * 8, key)

    transitions = {
        (0, 0): 2,
        (0, 2): 5,
        (0, 5): 0,
        (3, 0): 5,
        (3, 2): 0,
        (3, 5): 2,
    }
    for (action, state), destination in transitions.items():
        machine[320 + action * MAX_STATES + state] = destination

    for state, answer in {0: 700, 2: 800, 5: 900}.items():
        _put_u64(
            machine,
            448 + (1 * MAX_STATES + state) * 8,
            answer,
        )
    return _seal_machine(machine)


def build_queries(
    machine: bytes,
    records: tuple[tuple[int, int, int, tuple[int, ...]], ...] | None = None,
) -> bytes:
    if records is None:
        records = (
            (11, 101, 9009, (1001, 1001)),
            (12, 303, 9009, (4004, 1001, 4004)),
            (13, 606, 9009, ()),
        )
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
    return _seal_queries(queries)


@pytest.fixture(scope="session")
def runtime(tmp_path_factory: pytest.TempPathFactory) -> Path:
    compiler = shutil.which("cc")
    if compiler is None:
        pytest.skip("a C compiler is required")
    executable = tmp_path_factory.mktemp("episode_runtime_c") / "runtime"
    subprocess.run(
        [
            compiler,
            "-std=c11",
            "-O2",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-pedantic",
            str(SOURCE),
            "-o",
            str(executable),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return executable


def run_runtime(
    runtime: Path,
    tmp_path: Path,
    machine: bytes,
    queries: bytes,
) -> tuple[subprocess.CompletedProcess[str], Path]:
    machine_path = tmp_path / "machine.bin"
    query_path = tmp_path / "queries.bin"
    transcript_path = tmp_path / "transcript.bin"
    machine_path.write_bytes(machine)
    query_path.write_bytes(queries)
    completed = subprocess.run(
        [runtime, machine_path, query_path, transcript_path],
        capture_output=True,
        text=True,
        check=False,
    )
    return completed, transcript_path


def assert_rejected(
    runtime: Path,
    tmp_path: Path,
    machine: bytes,
    queries: bytes,
    message: str,
) -> None:
    completed, transcript_path = run_runtime(
        runtime, tmp_path, machine, queries
    )
    assert completed.returncode != 0
    assert message in completed.stderr
    assert not transcript_path.exists()


def test_executes_flat_machine_and_emits_hashed_fixed_records(
    runtime: Path,
    tmp_path: Path,
) -> None:
    machine = build_machine()
    queries = build_queries(machine)
    completed, transcript_path = run_runtime(
        runtime, tmp_path, machine, queries
    )
    assert completed.returncode == 0, completed.stderr

    transcript = transcript_path.read_bytes()
    assert len(transcript) == (
        TRANSCRIPT_HEADER_SIZE + 3 * TRANSCRIPT_RECORD_SIZE + HASH_SIZE
    )
    assert transcript[:8] == TRANSCRIPT_MAGIC
    assert struct.unpack_from("<III", transcript, 8) == (
        1,
        TRANSCRIPT_HEADER_SIZE,
        TRANSCRIPT_RECORD_SIZE,
    )
    assert struct.unpack_from("<I", transcript, 20)[0] == 3
    assert transcript[24:56] == machine[MACHINE_HASH_OFFSET:]
    assert transcript[56:88] == queries[-HASH_SIZE:]
    assert transcript[88:96] == bytes(8)
    assert transcript[-HASH_SIZE:] == hashlib.sha256(
        transcript[:-HASH_SIZE]
    ).digest()

    expected = (
        (11, 606, 900, 5, 0, 2, 0),
        (12, 101, 700, 0, 0, 3, 0),
        (13, 606, 900, 5, 0, 0, 0),
    )
    actual = tuple(
        struct.unpack_from(
            "<QQQHHHH",
            transcript,
            TRANSCRIPT_HEADER_SIZE + index * TRANSCRIPT_RECORD_SIZE,
        )
        for index in range(3)
    )
    assert actual == expected


def test_transcript_binds_exact_query_bytes_even_when_records_match(
    runtime: Path,
    tmp_path: Path,
) -> None:
    machine = build_machine()
    left_queries = build_queries(
        machine,
        records=((77, 101, 9009, (1001, 1001)),),
    )
    right_queries = build_queries(
        machine,
        records=((77, 303, 9009, (4004, 4004)),),
    )
    left_root = tmp_path / "left"
    right_root = tmp_path / "right"
    left_root.mkdir()
    right_root.mkdir()

    left_completed, left_path = run_runtime(
        runtime, left_root, machine, left_queries
    )
    right_completed, right_path = run_runtime(
        runtime, right_root, machine, right_queries
    )
    assert left_completed.returncode == 0, left_completed.stderr
    assert right_completed.returncode == 0, right_completed.stderr

    left = left_path.read_bytes()
    right = right_path.read_bytes()
    left_record = left[
        TRANSCRIPT_HEADER_SIZE:
        TRANSCRIPT_HEADER_SIZE + TRANSCRIPT_RECORD_SIZE
    ]
    right_record = right[
        TRANSCRIPT_HEADER_SIZE:
        TRANSCRIPT_HEADER_SIZE + TRANSCRIPT_RECORD_SIZE
    ]
    assert left_record == right_record
    assert left[24:56] == right[24:56] == machine[MACHINE_HASH_OFFSET:]
    assert left[56:88] == left_queries[-HASH_SIZE:]
    assert right[56:88] == right_queries[-HASH_SIZE:]
    assert left[56:88] != right[56:88]
    assert left != right


@pytest.mark.parametrize(
    ("name", "mutate", "message"),
    (
        (
            "version",
            lambda data: _put_u32(data, 8, 2),
            "machine version",
        ),
        (
            "zero active key",
            lambda data: _put_u64(data, 64, 0),
            "machine keys",
        ),
        (
            "duplicate active key",
            lambda data: _put_u64(data, 64 + 2 * 8, 101),
            "machine keys",
        ),
        (
            "mask outside capacity",
            lambda data: _put_u64(data, 32, struct.unpack_from("<Q", data, 32)[0] | (1 << 20)),
            "machine active masks",
        ),
        (
            "inactive key padding",
            lambda data: _put_u64(data, 64 + 1 * 8, 777),
            "machine keys",
        ),
        (
            "inactive destination",
            lambda data: data.__setitem__(320, 1),
            "machine transition destination",
        ),
        (
            "transition padding",
            lambda data: data.__setitem__(320 + 1 * MAX_STATES, 2),
            "machine transition padding",
        ),
        (
            "observer padding",
            lambda data: _put_u64(data, 448, 99),
            "machine observer padding",
        ),
        (
            "reserved padding",
            lambda data: data.__setitem__(1472, 1),
            "machine flags or padding",
        ),
    ),
)
def test_rejects_malformed_machine_fields(
    runtime: Path,
    tmp_path: Path,
    name: str,
    mutate,
    message: str,
) -> None:
    del name
    valid = build_machine()
    machine = bytearray(valid)
    mutate(machine)
    malformed = _seal_machine(machine)
    assert_rejected(
        runtime,
        tmp_path,
        malformed,
        build_queries(malformed),
        message,
    )


def test_rejects_machine_length_and_hash(
    runtime: Path,
    tmp_path: Path,
) -> None:
    machine = build_machine()
    assert_rejected(
        runtime,
        tmp_path,
        machine[:-1],
        build_queries(machine),
        "machine length",
    )

    corrupt = bytearray(machine)
    corrupt[448 + (1 * MAX_STATES + 0) * 8] ^= 1
    assert_rejected(
        runtime,
        tmp_path,
        bytes(corrupt),
        build_queries(machine),
        "machine hash mismatch",
    )


@pytest.mark.parametrize(
    ("name", "mutate", "message"),
    (
        (
            "version",
            lambda data: _put_u32(data, 8, 2),
            "query version",
        ),
        (
            "unknown state key",
            lambda data: _put_u64(data, QUERY_HEADER_SIZE + 8, 123456),
            "query state key",
        ),
        (
            "unknown observer key",
            lambda data: _put_u64(data, QUERY_HEADER_SIZE + 16, 123456),
            "query observer key",
        ),
        (
            "unknown action key",
            lambda data: _put_u64(data, QUERY_HEADER_SIZE + 32, 123456),
            "query action key",
        ),
        (
            "oversize word",
            lambda data: _put_u16(data, QUERY_HEADER_SIZE + 24, MAX_WORD + 1),
            "query word length",
        ),
        (
            "action padding",
            lambda data: _put_u64(data, QUERY_HEADER_SIZE + 32 + 2 * 8, 1001),
            "query action padding",
        ),
        (
            "record padding",
            lambda data: data.__setitem__(QUERY_HEADER_SIZE + 288, 1),
            "query record flags or padding",
        ),
        (
            "header padding",
            lambda data: _put_u32(data, 60, 1),
            "query flags or header padding",
        ),
        (
            "machine binding",
            lambda data: data.__setitem__(24, data[24] ^ 1),
            "query machine hash",
        ),
        (
            "duplicate challenge",
            lambda data: _put_u64(
                data,
                QUERY_HEADER_SIZE + QUERY_RECORD_SIZE,
                struct.unpack_from("<Q", data, QUERY_HEADER_SIZE)[0],
            ),
            "query challenge IDs are duplicate",
        ),
    ),
)
def test_rejects_malformed_query_fields(
    runtime: Path,
    tmp_path: Path,
    name: str,
    mutate,
    message: str,
) -> None:
    del name
    machine = build_machine()
    queries = bytearray(build_queries(machine))
    mutate(queries)
    malformed = _seal_queries(queries)
    assert_rejected(runtime, tmp_path, machine, malformed, message)


def test_rejects_query_length_and_hash(
    runtime: Path,
    tmp_path: Path,
) -> None:
    machine = build_machine()
    queries = build_queries(machine)
    assert_rejected(
        runtime,
        tmp_path,
        machine,
        queries[:-1],
        "query file length",
    )

    corrupt = bytearray(queries)
    corrupt[QUERY_HEADER_SIZE + 32] ^= 1
    assert_rejected(
        runtime,
        tmp_path,
        machine,
        bytes(corrupt),
        "query hash mismatch",
    )


def test_cli_accepts_no_source_path(
    runtime: Path,
    tmp_path: Path,
) -> None:
    machine_path = tmp_path / "machine.bin"
    query_path = tmp_path / "queries.bin"
    output_path = tmp_path / "transcript.bin"
    machine = build_machine()
    machine_path.write_bytes(machine)
    query_path.write_bytes(build_queries(machine))
    completed = subprocess.run(
        [
            runtime,
            machine_path,
            query_path,
            output_path,
            tmp_path / "forbidden-source.txt",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 2
    assert "usage:" in completed.stderr
    assert not output_path.exists()
