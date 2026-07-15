#!/usr/bin/env python3
"""Independent fail-closed audit for the frozen Q-LIFT v1 CPU board.

The auditor intentionally does not import the generator.  It independently
checks finite-field closure, quotient answers, archive reversibility, complete
bit accounting, adversarial controls, canonical encoding, and frozen digests.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
import stat
from pathlib import Path
from typing import Any, Sequence


BOARD_SCHEMA = "q_lift_board_v1"
AUDIT_SCHEMA = "q_lift_admission_audit_v1"
FIELD = 17
WORLD_DIM = 6
STATE_DIM = 2
CASE_LENGTHS = (4, 8, 16, 32)
CASES_PER_LENGTH = 8
PAIR_COUNT = 8
COPY_COORDINATES = 2
INDEX_BITS = 8

EXPECTED_SEEDS = {
    "case": 2026071521,
    "copy": 2026071524,
    "index": 2026071526,
    "merge": 2026071522,
    "split": 2026071523,
    "swap": 2026071525,
}
EXPECTED_CONTENT_SHA256 = "b08ab33faabe15aa09fad0b6abfa1cc94e423c3bd6447de55f547d1312d02165"
EXPECTED_BOARD_SHA256 = "06ea09988dd2b1f84d5cc2ee5baa6e0a8bc1ea0102c3ba325d371af1929dc376"


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("ascii")


def pretty_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("ascii")


def digest_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _mod(value: int) -> int:
    return int(value) % FIELD


def _validate_vector(vector: Any, size: int, label: str) -> list[int]:
    if not isinstance(vector, list) or len(vector) != size:
        raise ValueError(f"{label} has wrong dimension")
    if any(not isinstance(value, int) or not 0 <= value < FIELD for value in vector):
        raise ValueError(f"{label} is not a GF({FIELD}) vector")
    return vector


def _validate_matrix(matrix: Any, height: int, width: int, label: str) -> list[list[int]]:
    if not isinstance(matrix, list) or len(matrix) != height:
        raise ValueError(f"{label} has wrong height")
    for index, row in enumerate(matrix):
        _validate_vector(row, width, f"{label}[{index}]")
    return matrix


def vector_add(left: Sequence[int], right: Sequence[int]) -> list[int]:
    if len(left) != len(right):
        raise ValueError("vector dimensions differ")
    return [_mod(a + b) for a, b in zip(left, right)]


def dot(left: Sequence[int], right: Sequence[int]) -> int:
    if len(left) != len(right):
        raise ValueError("dot-product dimensions differ")
    return _mod(sum(a * b for a, b in zip(left, right)))


def matrix_vector(matrix: Sequence[Sequence[int]], vector: Sequence[int]) -> list[int]:
    if any(len(row) != len(vector) for row in matrix):
        raise ValueError("matrix-vector dimensions differ")
    return [dot(row, vector) for row in matrix]


def matrix_multiply(
    left: Sequence[Sequence[int]], right: Sequence[Sequence[int]]
) -> list[list[int]]:
    if not left or not right or len(left[0]) != len(right):
        raise ValueError("matrix dimensions differ")
    width = len(right[0])
    if any(len(row) != len(left[0]) for row in left) or any(
        len(row) != width for row in right
    ):
        raise ValueError("ragged matrix")
    columns = [[right[i][j] for i in range(len(right))] for j in range(width)]
    return [[dot(row, column) for column in columns] for row in left]


def matrix_rank(matrix: Sequence[Sequence[int]]) -> int:
    rows = [[_mod(value) for value in row] for row in matrix]
    if not rows:
        return 0
    width = len(rows[0])
    rank = 0
    for column in range(width):
        pivot = next((i for i in range(rank, len(rows)) if rows[i][column]), None)
        if pivot is None:
            continue
        rows[rank], rows[pivot] = rows[pivot], rows[rank]
        inverse = pow(rows[rank][column], -1, FIELD)
        rows[rank] = [_mod(value * inverse) for value in rows[rank]]
        for index in range(len(rows)):
            if index == rank or not rows[index][column]:
                continue
            factor = rows[index][column]
            rows[index] = [
                _mod(value - factor * pivot_value)
                for value, pivot_value in zip(rows[index], rows[rank])
            ]
        rank += 1
        if rank == len(rows):
            break
    return rank


def affine_apply(matrix: Sequence[Sequence[int]], bias: Sequence[int], vector: Sequence[int]) -> list[int]:
    return vector_add(matrix_vector(matrix, vector), bias)


def final_world(context: dict[str, Any]) -> list[int]:
    world = list(context["initial"])
    for event in context["events"]:
        world = affine_apply(event["matrix"], event["bias"], world)
    return world


def folded_state(projection: Sequence[Sequence[int]], context: dict[str, Any]) -> list[int]:
    state = matrix_vector(projection, context["initial"])
    for event in context["events"]:
        state = affine_apply(event["state_matrix"], event["state_bias"], state)
    return state


def state_code(state: Sequence[int]) -> int:
    code = 0
    for value in state:
        code = code * FIELD + value
    return code


def expected_state_bits() -> int:
    return math.ceil(math.log2(FIELD**STATE_DIM))


def rowspace_contains(projection: Sequence[Sequence[int]], vector: Sequence[int]) -> bool:
    return matrix_rank(list(projection) + [list(vector)]) == matrix_rank(projection)


def _validate_event(event: dict[str, Any], projection: Sequence[Sequence[int]], label: str) -> None:
    if set(event) != {"bias", "matrix", "state_bias", "state_matrix"}:
        raise ValueError(f"{label} has unexpected fields")
    matrix = _validate_matrix(event["matrix"], WORLD_DIM, WORLD_DIM, f"{label}.matrix")
    bias = _validate_vector(event["bias"], WORLD_DIM, f"{label}.bias")
    state_matrix = _validate_matrix(
        event["state_matrix"], STATE_DIM, STATE_DIM, f"{label}.state_matrix"
    )
    state_bias = _validate_vector(event["state_bias"], STATE_DIM, f"{label}.state_bias")
    if matrix_multiply(projection, matrix) != matrix_multiply(state_matrix, projection):
        raise ValueError(f"{label} does not preserve the task quotient")
    if matrix_vector(projection, bias) != state_bias:
        raise ValueError(f"{label} bias does not preserve the task quotient")


def _validate_context(context: Any, projection: Sequence[Sequence[int]], length: int, label: str) -> None:
    if not isinstance(context, dict) or set(context) != {"events", "initial"}:
        raise ValueError(f"{label} has unexpected fields")
    _validate_vector(context["initial"], WORLD_DIM, f"{label}.initial")
    if not isinstance(context["events"], list) or len(context["events"]) != length:
        raise ValueError(f"{label} has wrong event count")
    for index, event in enumerate(context["events"]):
        if not isinstance(event, dict):
            raise ValueError(f"{label}.events[{index}] is not an object")
        _validate_event(event, projection, f"{label}.events[{index}]")


def restore_archive(archive: Any) -> dict[str, Any]:
    if not isinstance(archive, dict) or set(archive) != {
        "address_bits_per_read",
        "codec",
        "full_retrieval_bits",
        "packet_count",
        "packets",
        "payload_bits",
    }:
        raise ValueError("archive has unexpected fields")
    if archive["codec"] != "canonical-json-base64-packets-v1":
        raise ValueError("archive codec changed")
    packets = archive["packets"]
    if not isinstance(packets, list) or not packets or archive["packet_count"] != len(packets):
        raise ValueError("archive packet count mismatch")
    address_bits = max(1, math.ceil(math.log2(len(packets))))
    if archive["address_bits_per_read"] != address_bits:
        raise ValueError("archive address accounting mismatch")
    decoded = []
    payload_bits = 0
    for expected_address, packet in enumerate(packets):
        if not isinstance(packet, dict) or set(packet) != {
            "address",
            "payload_b64",
            "payload_bytes",
            "sha256",
        }:
            raise ValueError("archive packet has unexpected fields")
        if packet["address"] != expected_address:
            raise ValueError("archive addresses are not canonical")
        raw = base64.b64decode(packet["payload_b64"], validate=True)
        if packet["payload_bytes"] != len(raw) or packet["sha256"] != digest_bytes(raw):
            raise ValueError("archive payload accounting mismatch")
        value = json.loads(raw)
        if raw != canonical_json_bytes(value):
            raise ValueError("archive payload is not canonical JSON")
        decoded.append(value)
        payload_bits += len(raw) * 8
    if archive["payload_bits"] != payload_bits:
        raise ValueError("archive payload-bit count mismatch")
    if archive["full_retrieval_bits"] != payload_bits + address_bits * len(packets):
        raise ValueError("archive total retrieval-bit count mismatch")
    if decoded[0].get("kind") != "initial" or set(decoded[0]) != {"kind", "value"}:
        raise ValueError("archive initial packet malformed")
    events = []
    for index, value in enumerate(decoded[1:]):
        if set(value) != {"index", "kind", "value"} or value["kind"] != "event" or value["index"] != index:
            raise ValueError("archive event packet malformed")
        events.append(value["value"])
    return {"events": events, "initial": decoded[0]["value"]}


def answer_in_family(state: Sequence[int], query: dict[str, Any]) -> int:
    event = query["future_event"]
    next_state = affine_apply(event["state_matrix"], event["state_bias"], state)
    return dot(query["coefficients"], next_state)


def answer_out_of_family(archive: dict[str, Any], query: dict[str, Any]) -> int:
    return dot(query["coefficients"], final_world(restore_archive(archive)))


def _validate_case(case: Any, expected_id: str, expected_length: int) -> dict[str, int]:
    if not isinstance(case, dict):
        raise ValueError("case is not an object")
    required = {
        "accounting",
        "archive",
        "context",
        "final_world",
        "id",
        "in_family_query",
        "length",
        "out_of_family_query",
        "sham_state",
        "state",
        "state_code",
        "task",
    }
    if set(case) != required or case["id"] != expected_id or case["length"] != expected_length:
        raise ValueError(f"case contract mismatch for {expected_id}")
    task = case["task"]
    if set(task) != {"basis", "projection", "sham_projection"}:
        raise ValueError("task has unexpected fields")
    basis = _validate_matrix(task["basis"], WORLD_DIM, WORLD_DIM, "task.basis")
    projection = _validate_matrix(task["projection"], STATE_DIM, WORLD_DIM, "task.projection")
    sham = _validate_matrix(task["sham_projection"], STATE_DIM, WORLD_DIM, "task.sham_projection")
    if matrix_rank(basis) != WORLD_DIM or basis[:STATE_DIM] != projection or basis[STATE_DIM:2*STATE_DIM] != sham:
        raise ValueError("task basis/projection relation changed")
    _validate_context(case["context"], projection, expected_length, "context")
    world = final_world(case["context"])
    state = folded_state(projection, case["context"])
    if world != case["final_world"] or state != case["state"] or state != matrix_vector(projection, world):
        raise ValueError("case world or quotient state is wrong")
    if case["sham_state"] != matrix_vector(sham, world) or case["state_code"] != state_code(state):
        raise ValueError("case state encoding is wrong")
    if restore_archive(case["archive"]) != case["context"]:
        raise ValueError("archive does not reconstruct the exact context")

    in_query = case["in_family_query"]
    if set(in_query) != {"answer", "coefficients", "future_event"}:
        raise ValueError("in-family query has unexpected fields")
    coefficients = _validate_vector(in_query["coefficients"], STATE_DIM, "in coefficients")
    if not any(coefficients):
        raise ValueError("in-family query is vacuous")
    _validate_event(in_query["future_event"], projection, "future event")
    if answer_in_family(state, in_query) != in_query["answer"]:
        raise ValueError("in-family answer is wrong")
    future_world = affine_apply(
        in_query["future_event"]["matrix"], in_query["future_event"]["bias"], world
    )
    if in_query["answer"] != dot(coefficients, matrix_vector(projection, future_world)):
        raise ValueError("state and full-world answers disagree")

    out_query = case["out_of_family_query"]
    if set(out_query) != {"answer", "coefficients", "requires_archive"} or out_query["requires_archive"] is not True:
        raise ValueError("out-of-family query has unexpected fields")
    out_coefficients = _validate_vector(out_query["coefficients"], WORLD_DIM, "out coefficients")
    if rowspace_contains(projection, out_coefficients):
        raise ValueError("out-of-family query lies in the task rowspace")
    if answer_out_of_family(case["archive"], out_query) != out_query["answer"]:
        raise ValueError("out-of-family answer is wrong")

    accounting = case["accounting"]
    if accounting != {
        "active_state_bits": expected_state_bits(),
        "archive_full_retrieval_bits": case["archive"]["full_retrieval_bits"],
        "archive_payload_bits": case["archive"]["payload_bits"],
        "in_family_retrieval_bits": 0,
    }:
        raise ValueError("case accounting is wrong")
    return {
        "archive_bits": case["archive"]["full_retrieval_bits"],
        "copy_correct": int(
            answer_in_family(
                matrix_vector(
                    projection,
                    case["context"]["initial"][:COPY_COORDINATES]
                    + [0] * (WORLD_DIM - COPY_COORDINATES),
                ),
                in_query,
            )
            == in_query["answer"]
        ),
        "sham_correct": int(answer_in_family(case["sham_state"], in_query) == in_query["answer"]),
    }


def _control_projection(control: dict[str, Any], label: str) -> list[list[int]]:
    task = control.get("task")
    if not isinstance(task, dict) or set(task) != {"basis", "projection"}:
        raise ValueError(f"{label} task has unexpected fields")
    basis = _validate_matrix(task["basis"], WORLD_DIM, WORLD_DIM, f"{label}.basis")
    projection = _validate_matrix(
        task["projection"], STATE_DIM, WORLD_DIM, f"{label}.projection"
    )
    if matrix_rank(basis) != WORLD_DIM or basis[:STATE_DIM] != projection:
        raise ValueError(f"{label} basis/projection relation changed")
    return projection


def _audit_controls(controls: Any) -> dict[str, int]:
    if not isinstance(controls, dict) or set(controls) != {"copy", "index", "merge", "split", "swap"}:
        raise ValueError("control families changed")
    if any(len(controls[name]) != PAIR_COUNT for name in ("copy", "merge", "split", "swap")):
        raise ValueError("paired-control counts changed")

    merge_exact = 0
    for control in controls["merge"]:
        projection = _control_projection(control, control["id"])
        left_length = len(control["left_context"]["events"])
        _validate_context(control["left_context"], projection, left_length, "merge.left")
        _validate_context(control["right_context"], projection, left_length, "merge.right")
        if control["left_context"] == control["right_context"]:
            raise ValueError("merge pair did not vary nuisance context")
        if restore_archive(control["left_archive"]) != control["left_context"] or restore_archive(control["right_archive"]) != control["right_context"]:
            raise ValueError("merge archive is not reversible")
        if control["left_state"] != control["right_state"]:
            raise ValueError("merge pair has different quotient states")
        if folded_state(projection, control["left_context"]) != control["left_state"] or folded_state(projection, control["right_context"]) != control["right_state"]:
            raise ValueError("merge states do not follow their contexts")
        if dot(control["coefficients"], control["left_state"]) != control["shared_answer"] or dot(control["coefficients"], control["right_state"]) != control["shared_answer"]:
            raise ValueError("merge pair is behaviorally distinguishable")
        merge_exact += 1

    split_exact = 0
    for control in controls["split"]:
        projection = _control_projection(control, control["id"])
        _validate_context(control["left_context"], projection, 0, "split.left")
        _validate_context(control["right_context"], projection, 0, "split.right")
        if restore_archive(control["left_archive"]) != control["left_context"] or restore_archive(control["right_archive"]) != control["right_context"]:
            raise ValueError("split archive is not reversible")
        if matrix_vector(projection, control["left_context"]["initial"]) != control["left_state"] or matrix_vector(projection, control["right_context"]["initial"]) != control["right_state"]:
            raise ValueError("split states do not follow their contexts")
        if control["left_state"] == control["right_state"] or control["left_answer"] == control["right_answer"]:
            raise ValueError("split pair is not separated")
        if dot(control["coefficients"], control["left_state"]) != control["left_answer"] or dot(control["coefficients"], control["right_state"]) != control["right_answer"]:
            raise ValueError("split answers are wrong")
        split_exact += 1

    copy_exact = 0
    for control in controls["copy"]:
        projection = _control_projection(control, control["id"])
        _validate_context(control["left_context"], projection, 0, "copy.left")
        _validate_context(control["right_context"], projection, 0, "copy.right")
        left = control["left_context"]["initial"]
        right = control["right_context"]["initial"]
        if left[:COPY_COORDINATES] != right[:COPY_COORDINATES] or control["copied_prefix"] != left[:COPY_COORDINATES]:
            raise ValueError("copy witness does not collide")
        if control["copied_prefix_bits"] != expected_state_bits():
            raise ValueError("copy sham is not capacity matched")
        if control["left_answer"] == control["right_answer"]:
            raise ValueError("copy collision has no separating answer")
        if matrix_vector(projection, left) != control["left_state"] or matrix_vector(projection, right) != control["right_state"]:
            raise ValueError("copy states do not follow their contexts")
        if dot(control["coefficients"], control["left_state"]) != control["left_answer"] or dot(control["coefficients"], control["right_state"]) != control["right_answer"]:
            raise ValueError("copy separating answers are wrong")
        copy_exact += 1

    swap_exact = 0
    for control in controls["swap"]:
        projection = _control_projection(control, control["id"])
        _validate_context(control["left_context"], projection, 0, "swap.left")
        _validate_context(control["right_context"], projection, 0, "swap.right")
        left_context = restore_archive(control["left_archive"])
        right_context = restore_archive(control["right_archive"])
        if left_context != control["left_context"] or right_context != control["right_context"]:
            raise ValueError("swap archive is not reversible")
        if dot(control["coefficients"], control["left_state"]) != control["left_answer"] or dot(control["coefficients"], control["right_state"]) != control["right_answer"]:
            raise ValueError("state-swap behavior is wrong")
        if matrix_vector(projection, left_context["initial"]) != control["left_state"] or matrix_vector(projection, right_context["initial"]) != control["right_state"]:
            raise ValueError("swap states do not follow their contexts")
        if rowspace_contains(projection, control["out_coefficients"]):
            raise ValueError("swap out-of-family query lies in the task rowspace")
        out_query = {"coefficients": control["out_coefficients"]}
        if answer_out_of_family(control["left_archive"], out_query) != control["left_out_answer"] or answer_out_of_family(control["right_archive"], out_query) != control["right_out_answer"]:
            raise ValueError("archive-swap behavior is wrong")
        if control["left_answer"] == control["right_answer"] or control["left_out_answer"] == control["right_out_answer"]:
            raise ValueError("swap pair lacks a causal separator")
        swap_exact += 1

    index = controls["index"]
    if index["n"] != INDEX_BITS or index["exact_fixed_state_lower_bound_bits"] != INDEX_BITS or index["underbudget_bits"] != INDEX_BITS - 1:
        raise ValueError("INDEX lower-bound contract changed")
    expected_errors = {
        str(bits): (INDEX_BITS - bits) / (2 * INDEX_BITS)
        for bits in range(INDEX_BITS + 1)
    }
    if index["prefix_baseline_average_error"] != expected_errors or len(index["witnesses"]) != 1 << (INDEX_BITS - 1):
        raise ValueError("INDEX baseline contract changed")
    prefixes = set()
    for witness in index["witnesses"]:
        left, right, query = witness["left"], witness["right"], witness["query"]
        if query != INDEX_BITS - 1 or left[:-1] != right[:-1] or left[query] == right[query]:
            raise ValueError("INDEX witness is not a prefix collision")
        prefixes.add(tuple(left[:-1]))
    if len(prefixes) != 1 << (INDEX_BITS - 1):
        raise ValueError("INDEX witnesses do not exhaust all underbudget states")
    return {
        "copy_exact": copy_exact,
        "index_exact": len(prefixes),
        "merge_exact": merge_exact,
        "split_exact": split_exact,
        "swap_exact": swap_exact,
    }


def _expected_metrics(board: dict[str, Any], case_stats: Sequence[dict[str, int]], control_stats: dict[str, int]) -> dict[str, Any]:
    cases = board["cases"]
    retrieval_bits = sum(statistic["archive_bits"] for statistic in case_stats)
    return {
        "analytic_quotient": {"correct": len(cases), "retrieval_bits": 0, "total": len(cases)},
        "copy_prefix_zero_fill": {"correct": sum(item["copy_correct"] for item in case_stats), "total": len(cases)},
        "copy_witness_collisions": control_stats["copy_exact"],
        "index_underbudget_witnesses": control_stats["index_exact"],
        "merge_exact": control_stats["merge_exact"],
        "retrieval_only": {"correct": len(cases), "retrieval_bits": retrieval_bits, "total": len(cases)},
        "sham_projection": {"correct": sum(item["sham_correct"] for item in case_stats), "total": len(cases)},
        "split_exact": control_stats["split_exact"],
        "swap_exact": control_stats["swap_exact"],
    }


def read_immutable_board(path: str | Path) -> tuple[dict[str, Any], bytes]:
    source = Path(path)
    info = source.lstat()
    if not stat.S_ISREG(info.st_mode):
        raise ValueError("board is not a regular file")
    if info.st_mode & 0o222:
        raise PermissionError("board is writable")
    raw = source.read_bytes()
    value = json.loads(raw)
    if not isinstance(value, dict) or raw != pretty_json_bytes(value):
        raise ValueError("board is not canonical pretty JSON")
    return value, raw


def audit_board(path: str | Path) -> dict[str, Any]:
    board, raw = read_immutable_board(path)
    board_sha = digest_bytes(raw)
    if EXPECTED_BOARD_SHA256 != "TO_BE_FROZEN" and board_sha != EXPECTED_BOARD_SHA256:
        raise ValueError(f"board SHA-256 mismatch: {board_sha} != {EXPECTED_BOARD_SHA256}")
    if board.get("schema") != BOARD_SCHEMA or board.get("seeds") != EXPECTED_SEEDS:
        raise ValueError("board schema or seeds changed")
    expected_config = {
        "case_lengths": list(CASE_LENGTHS),
        "cases_per_length": CASES_PER_LENGTH,
        "copy_coordinates": COPY_COORDINATES,
        "field": FIELD,
        "index_bits": INDEX_BITS,
        "pair_count": PAIR_COUNT,
        "quotient_classes": FIELD**STATE_DIM,
        "state_dim": STATE_DIM,
        "state_fixed_bits": expected_state_bits(),
        "world_dim": WORLD_DIM,
    }
    if board.get("config") != expected_config or board.get("claim_status") != "cpu_falsifier_only_no_capability_or_novelty_claim":
        raise ValueError("board configuration or claim status changed")
    content = {"cases": board.get("cases"), "controls": board.get("controls")}
    content_sha = digest_bytes(canonical_json_bytes(content))
    if board.get("content_sha256") != content_sha:
        raise ValueError("board content digest is internally inconsistent")
    if EXPECTED_CONTENT_SHA256 != "TO_BE_FROZEN" and content_sha != EXPECTED_CONTENT_SHA256:
        raise ValueError("board content SHA-256 changed")

    cases = board["cases"]
    if not isinstance(cases, list) or len(cases) != len(CASE_LENGTHS) * CASES_PER_LENGTH:
        raise ValueError("case count changed")
    case_stats = []
    offset = 0
    for length in CASE_LENGTHS:
        for index in range(CASES_PER_LENGTH):
            case_stats.append(_validate_case(cases[offset], f"q-{length:02d}-{index:03d}", length))
            offset += 1
    control_stats = _audit_controls(board["controls"])
    expected_metrics = _expected_metrics(board, case_stats, control_stats)
    if board.get("reference_metrics") != expected_metrics:
        raise ValueError("reference metrics do not independently recompute")
    return {
        "admitted": True,
        "board_sha256": board_sha,
        "case_count": len(cases),
        "checks": {
            "archive_reversibility": len(cases) + PAIR_COUNT * 6,
            "copy_witnesses": control_stats["copy_exact"],
            "finite_field_closure": sum(case["length"] + 1 for case in cases),
            "index_witnesses": control_stats["index_exact"],
            "merge_witnesses": control_stats["merge_exact"],
            "split_witnesses": control_stats["split_exact"],
            "swap_witnesses": control_stats["swap_exact"],
        },
        "claim_status": board["claim_status"],
        "content_sha256": content_sha,
        "gpu_used": False,
        "model_fit": False,
        "reference_metrics": expected_metrics,
        "schema": AUDIT_SCHEMA,
    }


def _exclusive_immutable_write(path: str | Path, payload: bytes) -> str:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = None
    created = False
    try:
        descriptor = os.open(destination, flags, 0o444)
        created = True
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short write while creating immutable audit")
            view = view[written:]
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
    except BaseException:
        if descriptor is not None:
            os.close(descriptor)
            descriptor = None
        if created:
            try:
                destination.unlink()
            except OSError:
                pass
        raise
    finally:
        if descriptor is not None:
            os.close(descriptor)
    info = destination.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_mode & 0o222:
        raise PermissionError("immutable audit contract failed")
    return digest_bytes(payload)


def write_audit(board_path: str | Path, output_path: str | Path) -> dict[str, Any]:
    if Path(board_path).resolve() == Path(output_path).resolve():
        raise ValueError("board and audit paths must differ")
    report = audit_board(board_path)
    report_sha = _exclusive_immutable_write(output_path, pretty_json_bytes(report))
    return {"admitted": True, "audit_sha256": report_sha, "board_sha256": report["board_sha256"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--board", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    print(json.dumps(write_audit(args.board, args.out), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
