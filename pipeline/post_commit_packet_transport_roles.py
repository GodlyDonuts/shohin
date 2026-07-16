#!/usr/bin/env python3
"""Seed-free subprocess roles for the R12 packet-transport falsifier."""

from __future__ import annotations

import argparse
import json
import os
import stat
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


PROTOCOL_ID = "R12-PCPT-F17x4-v2"
MODULUS = 17
DIMENSION = 4


class RoleError(ValueError):
    """Raised when a role invocation violates its allowlisted interface."""


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("ascii")


def _strict_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise RoleError(
            f"{label} fields must be {sorted(expected)}, got {sorted(value)}"
        )


def _json_lines(payload: bytes) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, raw in enumerate(payload.splitlines(), start=1):
        if not raw:
            raise RoleError(f"blank JSONL row at line {line_number}")
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise RoleError(f"JSONL row {line_number} must be an object")
        rows.append(value)
    return rows


def _split_stream(payload: bytes) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = _json_lines(payload)
    if not rows:
        raise RoleError("role input requires a header row")
    return rows[0], rows[1:]


def _source(row: Mapping[str, Any]) -> tuple[int, int, int, int]:
    _strict_keys(row, {"source"}, "source row")
    raw = row["source"]
    if (
        not isinstance(raw, list)
        or len(raw) != DIMENSION
        or any(not isinstance(value, int) or not 0 <= value < MODULUS for value in raw)
    ):
        raise RoleError("source must contain four F_17 elements")
    return tuple(raw)  # type: ignore[return-value]


def _packet(row: Mapping[str, Any]) -> tuple[int, int, int, int]:
    _strict_keys(row, {"values"}, "packet")
    raw = row["values"]
    if (
        not isinstance(raw, list)
        or len(raw) != DIMENSION
        or any(not isinstance(value, int) or not 0 <= value < MODULUS for value in raw)
    ):
        raise RoleError("packet must contain four F_17 elements")
    return tuple(raw)  # type: ignore[return-value]


def _invertible(matrix: Sequence[Sequence[int]]) -> bool:
    work = [[int(value) % MODULUS for value in row] for row in matrix]
    rank = 0
    for column in range(DIMENSION):
        pivot = next(
            (row for row in range(rank, DIMENSION) if work[row][column] != 0),
            None,
        )
        if pivot is None:
            continue
        work[rank], work[pivot] = work[pivot], work[rank]
        inverse = pow(work[rank][column], -1, MODULUS)
        work[rank] = [(value * inverse) % MODULUS for value in work[rank]]
        for row in range(DIMENSION):
            if row == rank:
                continue
            factor = work[row][column]
            work[row] = [
                (left - factor * right) % MODULUS
                for left, right in zip(work[row], work[rank], strict=True)
            ]
        rank += 1
    return rank == DIMENSION


def _update(value: Mapping[str, Any]) -> tuple[tuple[tuple[int, ...], ...], tuple[int, ...]]:
    _strict_keys(value, {"matrix", "offset"}, "update")
    matrix = value["matrix"]
    offset = value["offset"]
    if (
        not isinstance(matrix, list)
        or len(matrix) != DIMENSION
        or any(not isinstance(row, list) or len(row) != DIMENSION for row in matrix)
        or not isinstance(offset, list)
        or len(offset) != DIMENSION
    ):
        raise RoleError("update must be a four-dimensional affine map")
    flat = [item for row in matrix for item in row] + offset
    if any(not isinstance(item, int) or not 0 <= item < MODULUS for item in flat):
        raise RoleError("update value is outside F_17")
    if not _invertible(matrix):
        raise RoleError("update matrix must be invertible")
    return tuple(tuple(row) for row in matrix), tuple(offset)


def _apply(
    update: tuple[tuple[tuple[int, ...], ...], tuple[int, ...]],
    vector: Sequence[int],
) -> tuple[int, int, int, int]:
    matrix, offset = update
    return tuple(
        (sum(left * right for left, right in zip(row, vector, strict=True)) + bias)
        % MODULUS
        for row, bias in zip(matrix, offset, strict=True)
    )  # type: ignore[return-value]


def _consumer(value: Any) -> tuple[int, int, int, int]:
    if (
        not isinstance(value, list)
        or len(value) != DIMENSION
        or any(not isinstance(item, int) or not 0 <= item < MODULUS for item in value)
    ):
        raise RoleError("consumer must contain four F_17 elements")
    return tuple(value)  # type: ignore[return-value]


def _permutation(value: Any) -> tuple[int, ...]:
    if (
        not isinstance(value, list)
        or len(value) != MODULUS
        or any(not isinstance(item, int) for item in value)
        or sorted(value) != list(range(MODULUS))
        or all(index == item for index, item in enumerate(value))
    ):
        raise RoleError("output permutation must be a nonidentity permutation")
    return tuple(value)


def _challenge(value: Mapping[str, Any]) -> dict[str, Any]:
    _strict_keys(
        value,
        {
            "challenge_id",
            "kind",
            "depth",
            "updates",
            "consumer",
            "output_permutation",
        },
        "challenge",
    )
    if not isinstance(value["challenge_id"], str) or not isinstance(value["kind"], str):
        raise RoleError("challenge identifiers must be strings")
    if not isinstance(value["depth"], int) or value["depth"] < 1:
        raise RoleError("challenge depth must be positive")
    if not isinstance(value["updates"], list):
        raise RoleError("challenge updates must be a list")
    updates = tuple(_update(item) for item in value["updates"])
    if len(updates) != value["depth"]:
        raise RoleError("challenge depth does not match update count")
    return {
        "updates": updates,
        "consumer": _consumer(value["consumer"]),
        "output_permutation": _permutation(value["output_permutation"]),
    }


def _dot(left: Sequence[int], right: Sequence[int]) -> int:
    return sum(a * b for a, b in zip(left, right, strict=True)) % MODULUS


def _packet_path(raw: str, *, must_exist: bool) -> Path:
    path = Path(raw)
    if path.is_absolute() or len(path.parts) != 1 or path.name in {"", ".", ".."}:
        raise RoleError("packet path must be one relative filename")
    if must_exist:
        if not path.is_file():
            raise RoleError("packet input does not exist")
        if stat.S_IMODE(path.stat().st_mode) != 0o444:
            raise RoleError("packet input must be read-only")
    elif path.exists():
        raise RoleError("packet output already exists")
    return path


def _immutable_write(path: Path, payload: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(descriptor)
    path.chmod(0o444)


def _write_rows(rows: Iterable[Mapping[str, Any]]) -> None:
    for row in rows:
        sys.stdout.buffer.write(canonical_json_bytes(dict(row)))


def role_writer(arm: str, payload: bytes) -> None:
    header, rows = _split_stream(payload)
    _strict_keys(header, {"protocol_id", "role"}, "writer header")
    if header != {"protocol_id": PROTOCOL_ID, "role": "writer"}:
        raise RoleError("invalid writer header")
    if arm not in {"state", "motor"}:
        raise RoleError("writer arm must be state or motor")

    def output_rows() -> Iterable[dict[str, Any]]:
        for row in rows:
            source = _source(row)
            values = source if arm == "state" else (source[0], source[1], 0, 0)
            yield {"values": list(values)}

    _write_rows(output_rows())


def role_updater(payload: bytes, packet_in: str, packet_out: str) -> None:
    header, rows = _split_stream(payload)
    _strict_keys(header, {"protocol_id", "role", "update"}, "updater header")
    if header["protocol_id"] != PROTOCOL_ID or header["role"] != "updater":
        raise RoleError("invalid updater header")
    if rows:
        raise RoleError("updater stdin may contain only its one-event header")
    update = _update(header["update"])
    input_path = _packet_path(packet_in, must_exist=True)
    output_path = _packet_path(packet_out, must_exist=False)
    output = b"".join(
        canonical_json_bytes({"values": list(_apply(update, _packet(row)))})
        for row in _json_lines(input_path.read_bytes())
    )
    _immutable_write(output_path, output)


def role_reader(payload: bytes, packet_in: str, *, raw: bool = False) -> None:
    header, rows = _split_stream(payload)
    expected_role = "raw_reader" if raw else "reader"
    _strict_keys(
        header,
        {"protocol_id", "role", "consumer", "output_permutation"},
        "reader header",
    )
    if header["protocol_id"] != PROTOCOL_ID or header["role"] != expected_role:
        raise RoleError("invalid reader header")
    if rows:
        raise RoleError("reader stdin may contain only its late-query header")
    consumer = _consumer(header["consumer"])
    permutation = _permutation(header["output_permutation"])
    input_path = _packet_path(packet_in, must_exist=True)

    def output_rows() -> Iterable[dict[str, Any]]:
        for row in _json_lines(input_path.read_bytes()):
            answer = _dot(consumer, _packet(row))
            yield {"raw": answer} if raw else {"symbol": permutation[answer]}

    _write_rows(output_rows())


def role_oracle(payload: bytes) -> None:
    header, rows = _split_stream(payload)
    _strict_keys(header, {"protocol_id", "role", "challenge", "emit"}, "oracle header")
    if header["protocol_id"] != PROTOCOL_ID or header["role"] != "oracle":
        raise RoleError("invalid oracle header")
    if header["emit"] not in {"symbols", "packets"}:
        raise RoleError("oracle emit must be symbols or packets")
    challenge = _challenge(header["challenge"])

    def output_rows() -> Iterable[dict[str, Any]]:
        for row in rows:
            state = _source(row)
            for update in challenge["updates"]:
                state = _apply(update, state)
            if header["emit"] == "packets":
                yield {"values": list(state)}
            else:
                answer = _dot(challenge["consumer"], state)
                yield {"symbol": challenge["output_permutation"][answer]}

    _write_rows(output_rows())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("role", choices=("writer", "updater", "reader", "raw_reader", "oracle"))
    parser.add_argument("--arm", choices=("state", "motor"))
    parser.add_argument("--packet-in")
    parser.add_argument("--packet-out")
    args = parser.parse_args()
    payload = sys.stdin.buffer.read()
    try:
        if args.role == "writer":
            if args.arm is None or args.packet_in is not None or args.packet_out is not None:
                raise RoleError("writer requires only --arm")
            role_writer(args.arm, payload)
        elif args.arm is not None:
            raise RoleError("--arm is valid only for writer")
        elif args.role == "updater":
            if args.packet_in is None or args.packet_out is None:
                raise RoleError("updater requires --packet-in and --packet-out")
            role_updater(payload, args.packet_in, args.packet_out)
        elif args.role == "reader":
            if args.packet_in is None or args.packet_out is not None:
                raise RoleError("reader requires only --packet-in")
            role_reader(payload, args.packet_in)
        elif args.role == "raw_reader":
            if args.packet_in is None or args.packet_out is not None:
                raise RoleError("raw reader requires only --packet-in")
            role_reader(payload, args.packet_in, raw=True)
        else:
            if args.packet_in is not None or args.packet_out is not None:
                raise RoleError("oracle cannot receive packet paths")
            role_oracle(payload)
    except Exception as exc:
        sys.stderr.write(f"{type(exc).__name__}:{exc}\n")
        raise SystemExit(2)


if __name__ == "__main__":
    main()
