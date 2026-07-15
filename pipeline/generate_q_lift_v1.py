#!/usr/bin/env python3
"""Generate the frozen Q-LIFT v1 CPU falsifier board.

This is a deterministic, model-free finite-field reference package.  It does
not fit a model, invoke a GPU, or claim that Shohin has learned a sufficient
state.  Outputs are canonical JSON created with O_EXCL, fsynced, and made
read-only.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
import random
import stat
from pathlib import Path
from typing import Any, Sequence


SCHEMA = "q_lift_board_v1"
FIELD = 17
WORLD_DIM = 6
STATE_DIM = 2
CASE_LENGTHS = (4, 8, 16, 32)
CASES_PER_LENGTH = 8
PAIR_COUNT = 8
COPY_COORDINATES = 2
INDEX_BITS = 8

CASE_SEED = 2026071521
MERGE_SEED = 2026071522
SPLIT_SEED = 2026071523
COPY_SEED = 2026071524
SWAP_SEED = 2026071525
INDEX_SEED = 2026071526

# Filled only after the board contract is final.  These constants are not
# serialized into the board, so freezing them does not perturb its digest.
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


def vector_add(left: Sequence[int], right: Sequence[int]) -> list[int]:
    if len(left) != len(right):
        raise ValueError("vector dimensions differ")
    return [_mod(a + b) for a, b in zip(left, right)]


def dot(left: Sequence[int], right: Sequence[int]) -> int:
    if len(left) != len(right):
        raise ValueError("dot-product dimensions differ")
    return _mod(sum(a * b for a, b in zip(left, right)))


def matrix_vector(matrix: Sequence[Sequence[int]], vector: Sequence[int]) -> list[int]:
    if not matrix:
        return []
    if any(len(row) != len(vector) for row in matrix):
        raise ValueError("matrix-vector dimensions differ")
    return [dot(row, vector) for row in matrix]


def matrix_multiply(
    left: Sequence[Sequence[int]], right: Sequence[Sequence[int]]
) -> list[list[int]]:
    if not left or not right:
        raise ValueError("empty matrices are not supported")
    inner = len(left[0])
    if any(len(row) != inner for row in left) or len(right) != inner:
        raise ValueError("matrix dimensions differ")
    width = len(right[0])
    if any(len(row) != width for row in right):
        raise ValueError("ragged matrix")
    columns = [[right[i][j] for i in range(inner)] for j in range(width)]
    return [[dot(row, column) for column in columns] for row in left]


def matrix_rank(matrix: Sequence[Sequence[int]]) -> int:
    rows = [[_mod(value) for value in row] for row in matrix]
    if not rows:
        return 0
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise ValueError("ragged matrix")
    rank = 0
    for column in range(width):
        pivot = next((i for i in range(rank, len(rows)) if rows[i][column]), None)
        if pivot is None:
            continue
        rows[rank], rows[pivot] = rows[pivot], rows[rank]
        inverse = pow(rows[rank][column], -1, FIELD)
        rows[rank] = [_mod(value * inverse) for value in rows[rank]]
        for index, row in enumerate(rows):
            if index == rank or not row[column]:
                continue
            factor = row[column]
            rows[index] = [
                _mod(value - factor * pivot_value)
                for value, pivot_value in zip(row, rows[rank])
            ]
        rank += 1
        if rank == len(rows):
            break
    return rank


def matrix_inverse(matrix: Sequence[Sequence[int]]) -> list[list[int]]:
    size = len(matrix)
    if size == 0 or any(len(row) != size for row in matrix):
        raise ValueError("inverse requires a nonempty square matrix")
    rows = [
        [_mod(value) for value in row]
        + [1 if i == j else 0 for j in range(size)]
        for i, row in enumerate(matrix)
    ]
    for column in range(size):
        pivot = next((i for i in range(column, size) if rows[i][column]), None)
        if pivot is None:
            raise ValueError("matrix is singular")
        rows[column], rows[pivot] = rows[pivot], rows[column]
        inverse = pow(rows[column][column], -1, FIELD)
        rows[column] = [_mod(value * inverse) for value in rows[column]]
        for index in range(size):
            if index == column or not rows[index][column]:
                continue
            factor = rows[index][column]
            rows[index] = [
                _mod(value - factor * pivot_value)
                for value, pivot_value in zip(rows[index], rows[column])
            ]
    return [row[size:] for row in rows]


def random_vector(rng: random.Random, size: int) -> list[int]:
    return [rng.randrange(FIELD) for _ in range(size)]


def random_matrix(rng: random.Random, height: int, width: int) -> list[list[int]]:
    return [random_vector(rng, width) for _ in range(height)]


def random_invertible_matrix(rng: random.Random, size: int) -> list[list[int]]:
    for _ in range(10_000):
        candidate = random_matrix(rng, size, size)
        if matrix_rank(candidate) == size:
            return candidate
    raise RuntimeError("failed to draw an invertible matrix")


def affine_apply(
    matrix: Sequence[Sequence[int]], bias: Sequence[int], vector: Sequence[int]
) -> list[int]:
    return vector_add(matrix_vector(matrix, vector), bias)


def induced_event(
    rng: random.Random, basis: Sequence[Sequence[int]]
) -> dict[str, Any]:
    inverse = matrix_inverse(basis)
    state_matrix = random_matrix(rng, STATE_DIM, STATE_DIM)
    state_bias = random_vector(rng, STATE_DIM)
    kernel_dim = WORLD_DIM - STATE_DIM
    transformed = []
    for row in state_matrix:
        transformed.append(list(row) + [0] * kernel_dim)
    transformed.extend(
        [
            random_vector(rng, STATE_DIM) + random_vector(rng, kernel_dim)
            for _ in range(kernel_dim)
        ]
    )
    transformed_bias = state_bias + random_vector(rng, kernel_dim)
    world_matrix = matrix_multiply(matrix_multiply(inverse, transformed), basis)
    world_bias = matrix_vector(inverse, transformed_bias)
    return {
        "bias": world_bias,
        "matrix": world_matrix,
        "state_bias": state_bias,
        "state_matrix": state_matrix,
    }


def final_world(context: dict[str, Any]) -> list[int]:
    world = list(context["initial"])
    for event in context["events"]:
        world = affine_apply(event["matrix"], event["bias"], world)
    return world


def fold_state(
    projection: Sequence[Sequence[int]], context: dict[str, Any]
) -> list[int]:
    state = matrix_vector(projection, context["initial"])
    for event in context["events"]:
        state = affine_apply(event["state_matrix"], event["state_bias"], state)
    return state


def state_code(state: Sequence[int]) -> int:
    if len(state) != STATE_DIM or any(not 0 <= value < FIELD for value in state):
        raise ValueError("invalid quotient state")
    code = 0
    for value in state:
        code = code * FIELD + value
    return code


def state_bits() -> int:
    return math.ceil(math.log2(FIELD**STATE_DIM))


def rowspace_contains(projection: Sequence[Sequence[int]], vector: Sequence[int]) -> bool:
    return matrix_rank(list(projection) + [list(vector)]) == matrix_rank(projection)


def choose_out_vector(
    rng: random.Random, projection: Sequence[Sequence[int]]
) -> list[int]:
    for _ in range(10_000):
        candidate = random_vector(rng, WORLD_DIM)
        if any(candidate) and not rowspace_contains(projection, candidate):
            return candidate
    raise RuntimeError("failed to draw an out-of-family query")


def choose_separating_out_vector(
    rng: random.Random,
    projection: Sequence[Sequence[int]],
    left: Sequence[int],
    right: Sequence[int],
) -> list[int]:
    for _ in range(10_000):
        candidate = choose_out_vector(rng, projection)
        if dot(candidate, left) != dot(candidate, right):
            return candidate
    raise RuntimeError("failed to draw a separating out-of-family query")


def _packet(address: int, value: dict[str, Any]) -> dict[str, Any]:
    raw = canonical_json_bytes(value)
    return {
        "address": address,
        "payload_b64": base64.b64encode(raw).decode("ascii"),
        "payload_bytes": len(raw),
        "sha256": digest_bytes(raw),
    }


def archive_context(context: dict[str, Any]) -> dict[str, Any]:
    packets = [_packet(0, {"kind": "initial", "value": context["initial"]})]
    packets.extend(
        _packet(index + 1, {"index": index, "kind": "event", "value": event})
        for index, event in enumerate(context["events"])
    )
    address_bits = max(1, math.ceil(math.log2(len(packets))))
    payload_bits = sum(packet["payload_bytes"] * 8 for packet in packets)
    return {
        "address_bits_per_read": address_bits,
        "codec": "canonical-json-base64-packets-v1",
        "full_retrieval_bits": payload_bits + address_bits * len(packets),
        "packet_count": len(packets),
        "packets": packets,
        "payload_bits": payload_bits,
    }


def restore_context(archive: dict[str, Any]) -> dict[str, Any]:
    packets = archive.get("packets")
    if not isinstance(packets, list) or not packets:
        raise ValueError("archive has no packets")
    decoded: list[dict[str, Any]] = []
    for expected_address, packet in enumerate(packets):
        if packet.get("address") != expected_address:
            raise ValueError("archive address order changed")
        raw = base64.b64decode(packet["payload_b64"], validate=True)
        if len(raw) != packet["payload_bytes"] or digest_bytes(raw) != packet["sha256"]:
            raise ValueError("archive packet accounting mismatch")
        value = json.loads(raw)
        if raw != canonical_json_bytes(value):
            raise ValueError("archive packet is not canonical")
        decoded.append(value)
    if decoded[0] != {"kind": "initial", "value": decoded[0].get("value")}:
        raise ValueError("archive initial packet malformed")
    events = []
    for index, value in enumerate(decoded[1:]):
        if value.get("kind") != "event" or value.get("index") != index:
            raise ValueError("archive event packet malformed")
        events.append(value["value"])
    context = {"events": events, "initial": decoded[0]["value"]}
    expected = archive_context(context)
    if expected != archive:
        raise ValueError("archive is not the canonical reversible encoding")
    return context


def make_context(
    rng: random.Random, basis: Sequence[Sequence[int]], length: int
) -> dict[str, Any]:
    return {
        "events": [induced_event(rng, basis) for _ in range(length)],
        "initial": random_vector(rng, WORLD_DIM),
    }


def make_case(rng: random.Random, identifier: str, length: int) -> dict[str, Any]:
    basis = random_invertible_matrix(rng, WORLD_DIM)
    projection = basis[:STATE_DIM]
    sham_projection = basis[STATE_DIM : 2 * STATE_DIM]
    context = make_context(rng, basis, length)
    world = final_world(context)
    state = fold_state(projection, context)
    if state != matrix_vector(projection, world):
        raise RuntimeError("constructed event does not close on the quotient")

    future = induced_event(rng, basis)
    future_state = affine_apply(future["state_matrix"], future["state_bias"], state)
    query_vector = random_vector(rng, STATE_DIM)
    while not any(query_vector):
        query_vector = random_vector(rng, STATE_DIM)
    out_vector = choose_out_vector(rng, projection)
    archive = archive_context(context)

    return {
        "accounting": {
            "active_state_bits": state_bits(),
            "archive_full_retrieval_bits": archive["full_retrieval_bits"],
            "archive_payload_bits": archive["payload_bits"],
            "in_family_retrieval_bits": 0,
        },
        "archive": archive,
        "context": context,
        "final_world": world,
        "id": identifier,
        "in_family_query": {
            "answer": dot(query_vector, future_state),
            "coefficients": query_vector,
            "future_event": future,
        },
        "length": length,
        "out_of_family_query": {
            "answer": dot(out_vector, world),
            "coefficients": out_vector,
            "requires_archive": True,
        },
        "sham_state": matrix_vector(sham_projection, world),
        "state": state,
        "state_code": state_code(state),
        "task": {
            "basis": basis,
            "projection": projection,
            "sham_projection": sham_projection,
        },
    }


def answer_in_family(state: Sequence[int], query: dict[str, Any]) -> int:
    event = query["future_event"]
    next_state = affine_apply(event["state_matrix"], event["state_bias"], state)
    return dot(query["coefficients"], next_state)


def answer_out_of_family(archive: dict[str, Any], query: dict[str, Any]) -> int:
    world = final_world(restore_context(archive))
    return dot(query["coefficients"], world)


def _context_with_initial(context: dict[str, Any], initial: Sequence[int]) -> dict[str, Any]:
    return {"events": context["events"], "initial": list(initial)}


def _kernel_delta(
    rng: random.Random, basis: Sequence[Sequence[int]]
) -> list[int]:
    transformed = [0] * STATE_DIM + random_vector(rng, WORLD_DIM - STATE_DIM)
    while not any(transformed[STATE_DIM:]):
        transformed = [0] * STATE_DIM + random_vector(rng, WORLD_DIM - STATE_DIM)
    return matrix_vector(matrix_inverse(basis), transformed)


def make_merge_controls() -> list[dict[str, Any]]:
    rng = random.Random(MERGE_SEED)
    controls = []
    for index in range(PAIR_COUNT):
        basis = random_invertible_matrix(rng, WORLD_DIM)
        projection = basis[:STATE_DIM]
        context = make_context(rng, basis, 4 + index)
        right = _context_with_initial(
            context, vector_add(context["initial"], _kernel_delta(rng, basis))
        )
        left_state = fold_state(projection, context)
        right_state = fold_state(projection, right)
        if left_state != right_state or context == right:
            raise RuntimeError("merge witness construction failed")
        coefficients = [1, 0]
        controls.append(
            {
                "coefficients": coefficients,
                "id": f"merge-{index:03d}",
                "left_archive": archive_context(context),
                "left_context": context,
                "left_state": left_state,
                "right_archive": archive_context(right),
                "right_context": right,
                "right_state": right_state,
                "shared_answer": dot(coefficients, left_state),
                "task": {"basis": basis, "projection": projection},
            }
        )
    return controls


def _state_delta(basis: Sequence[Sequence[int]]) -> list[int]:
    transformed = [1] + [0] * (WORLD_DIM - 1)
    return matrix_vector(matrix_inverse(basis), transformed)


def make_split_controls(seed: int = SPLIT_SEED, label: str = "split") -> list[dict[str, Any]]:
    rng = random.Random(seed)
    controls = []
    for index in range(PAIR_COUNT):
        basis = random_invertible_matrix(rng, WORLD_DIM)
        projection = basis[:STATE_DIM]
        left_initial = random_vector(rng, WORLD_DIM)
        right_initial = vector_add(left_initial, _state_delta(basis))
        left_context = {"events": [], "initial": left_initial}
        right_context = {"events": [], "initial": right_initial}
        left_state = matrix_vector(projection, left_initial)
        right_state = matrix_vector(projection, right_initial)
        coefficients = [1, 0]
        left_answer = dot(coefficients, left_state)
        right_answer = dot(coefficients, right_state)
        if left_answer == right_answer:
            raise RuntimeError("split witness construction failed")
        out_coefficients = choose_separating_out_vector(
            rng, projection, left_initial, right_initial
        )
        controls.append(
            {
                "coefficients": coefficients,
                "id": f"{label}-{index:03d}",
                "left_answer": left_answer,
                "left_archive": archive_context(left_context),
                "left_context": left_context,
                "left_out_answer": dot(out_coefficients, left_initial),
                "left_state": left_state,
                "out_coefficients": out_coefficients,
                "right_answer": right_answer,
                "right_archive": archive_context(right_context),
                "right_context": right_context,
                "right_out_answer": dot(out_coefficients, right_initial),
                "right_state": right_state,
                "task": {"basis": basis, "projection": projection},
            }
        )
    return controls


def make_copy_controls() -> list[dict[str, Any]]:
    rng = random.Random(COPY_SEED)
    controls = []
    for index in range(PAIR_COUNT):
        for _ in range(10_000):
            basis = random_invertible_matrix(rng, WORLD_DIM)
            projection = basis[:STATE_DIM]
            delta = [0] * COPY_COORDINATES + random_vector(
                rng, WORLD_DIM - COPY_COORDINATES
            )
            state_delta = matrix_vector(projection, delta)
            if any(delta) and any(state_delta):
                break
        else:
            raise RuntimeError("copy witness construction failed")
        left = random_vector(rng, WORLD_DIM)
        right = vector_add(left, delta)
        component = next(i for i, value in enumerate(state_delta) if value)
        coefficients = [1 if i == component else 0 for i in range(STATE_DIM)]
        left_state = matrix_vector(projection, left)
        right_state = matrix_vector(projection, right)
        controls.append(
            {
                "coefficients": coefficients,
                "copied_prefix": left[:COPY_COORDINATES],
                "copied_prefix_bits": state_bits(),
                "id": f"copy-{index:03d}",
                "left_answer": dot(coefficients, left_state),
                "left_context": {"events": [], "initial": left},
                "left_state": left_state,
                "right_answer": dot(coefficients, right_state),
                "right_context": {"events": [], "initial": right},
                "right_state": right_state,
                "task": {"basis": basis, "projection": projection},
            }
        )
    return controls


def make_index_control() -> dict[str, Any]:
    rng = random.Random(INDEX_SEED)
    witnesses = []
    prefixes = list(range(1 << (INDEX_BITS - 1)))
    rng.shuffle(prefixes)
    for prefix in prefixes:
        left = [(prefix >> shift) & 1 for shift in reversed(range(INDEX_BITS - 1))] + [0]
        right = left[:-1] + [1]
        witnesses.append(
            {
                "copied_prefix": left[:-1],
                "left": left,
                "query": INDEX_BITS - 1,
                "right": right,
            }
        )
    prefix_error = {
        str(bits): (INDEX_BITS - bits) / (2 * INDEX_BITS)
        for bits in range(INDEX_BITS + 1)
    }
    return {
        "exact_fixed_state_lower_bound_bits": INDEX_BITS,
        "n": INDEX_BITS,
        "prefix_baseline_average_error": prefix_error,
        "underbudget_bits": INDEX_BITS - 1,
        "witnesses": witnesses,
    }


def reference_metrics(cases: Sequence[dict[str, Any]], controls: dict[str, Any]) -> dict[str, Any]:
    analytic_correct = 0
    sham_correct = 0
    copy_correct = 0
    retrieval_correct = 0
    retrieval_bits = 0
    for case in cases:
        query = case["in_family_query"]
        analytic_correct += answer_in_family(case["state"], query) == query["answer"]
        sham_correct += answer_in_family(case["sham_state"], query) == query["answer"]

        copied = case["context"]["initial"][:COPY_COORDINATES]
        copied_world = copied + [0] * (WORLD_DIM - COPY_COORDINATES)
        copied_state = matrix_vector(case["task"]["projection"], copied_world)
        copy_correct += answer_in_family(copied_state, query) == query["answer"]

        out_query = case["out_of_family_query"]
        retrieval_correct += answer_out_of_family(case["archive"], out_query) == out_query["answer"]
        retrieval_bits += case["archive"]["full_retrieval_bits"]

    merge_exact = sum(
        control["left_state"] == control["right_state"]
        and dot(control["coefficients"], control["left_state"])
        == control["shared_answer"]
        == dot(control["coefficients"], control["right_state"])
        for control in controls["merge"]
    )
    split_exact = sum(
        control["left_answer"] != control["right_answer"]
        for control in controls["split"]
    )
    copy_collisions = sum(
        control["left_context"]["initial"][:COPY_COORDINATES]
        == control["right_context"]["initial"][:COPY_COORDINATES]
        and control["left_answer"] != control["right_answer"]
        for control in controls["copy"]
    )
    swap_exact = sum(
        dot(control["coefficients"], control["left_state"])
        == control["left_answer"]
        and dot(control["coefficients"], control["right_state"])
        == control["right_answer"]
        and restore_context(control["left_archive"]) == control["left_context"]
        and restore_context(control["right_archive"]) == control["right_context"]
        and answer_out_of_family(
            control["right_archive"],
            {"coefficients": control["out_coefficients"]},
        )
        == control["right_out_answer"]
        and answer_out_of_family(
            control["left_archive"],
            {"coefficients": control["out_coefficients"]},
        )
        == control["left_out_answer"]
        and control["left_out_answer"] != control["right_out_answer"]
        for control in controls["swap"]
    )
    index = controls["index"]
    index_witnesses = sum(
        witness["left"][:-1] == witness["right"][:-1]
        and witness["left"][witness["query"]] != witness["right"][witness["query"]]
        for witness in index["witnesses"]
    )
    return {
        "analytic_quotient": {
            "correct": analytic_correct,
            "retrieval_bits": 0,
            "total": len(cases),
        },
        "copy_prefix_zero_fill": {"correct": copy_correct, "total": len(cases)},
        "copy_witness_collisions": copy_collisions,
        "index_underbudget_witnesses": index_witnesses,
        "merge_exact": merge_exact,
        "retrieval_only": {
            "correct": retrieval_correct,
            "retrieval_bits": retrieval_bits,
            "total": len(cases),
        },
        "sham_projection": {"correct": sham_correct, "total": len(cases)},
        "split_exact": split_exact,
        "swap_exact": swap_exact,
    }


def board_payload() -> dict[str, Any]:
    rng = random.Random(CASE_SEED)
    cases = []
    for length in CASE_LENGTHS:
        for index in range(CASES_PER_LENGTH):
            cases.append(make_case(rng, f"q-{length:02d}-{index:03d}", length))
    controls = {
        "copy": make_copy_controls(),
        "index": make_index_control(),
        "merge": make_merge_controls(),
        "split": make_split_controls(),
        "swap": make_split_controls(SWAP_SEED, "swap"),
    }
    content = {"cases": cases, "controls": controls}
    payload = {
        "claim_status": "cpu_falsifier_only_no_capability_or_novelty_claim",
        "config": {
            "case_lengths": list(CASE_LENGTHS),
            "cases_per_length": CASES_PER_LENGTH,
            "copy_coordinates": COPY_COORDINATES,
            "field": FIELD,
            "index_bits": INDEX_BITS,
            "pair_count": PAIR_COUNT,
            "quotient_classes": FIELD**STATE_DIM,
            "state_dim": STATE_DIM,
            "state_fixed_bits": state_bits(),
            "world_dim": WORLD_DIM,
        },
        "content_sha256": digest_bytes(canonical_json_bytes(content)),
        "cases": cases,
        "controls": controls,
        "reference_metrics": reference_metrics(cases, controls),
        "schema": SCHEMA,
        "seeds": {
            "case": CASE_SEED,
            "copy": COPY_SEED,
            "index": INDEX_SEED,
            "merge": MERGE_SEED,
            "split": SPLIT_SEED,
            "swap": SWAP_SEED,
        },
    }
    return payload


def assert_frozen_board(payload: dict[str, Any]) -> None:
    content = {"cases": payload["cases"], "controls": payload["controls"]}
    observed_content = digest_bytes(canonical_json_bytes(content))
    observed_board = digest_bytes(pretty_json_bytes(payload))
    if payload["content_sha256"] != observed_content:
        raise RuntimeError("Q-LIFT content digest is internally inconsistent")
    if EXPECTED_CONTENT_SHA256 != "TO_BE_FROZEN" and observed_content != EXPECTED_CONTENT_SHA256:
        raise RuntimeError(
            f"Q-LIFT content SHA-256 mismatch: {observed_content} != {EXPECTED_CONTENT_SHA256}"
        )
    if EXPECTED_BOARD_SHA256 != "TO_BE_FROZEN" and observed_board != EXPECTED_BOARD_SHA256:
        raise RuntimeError(
            f"Q-LIFT board SHA-256 mismatch: {observed_board} != {EXPECTED_BOARD_SHA256}"
        )


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
                raise OSError("short write while creating immutable output")
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
        raise PermissionError("immutable output contract failed")
    return digest_bytes(payload)


def write_board(path: str | Path) -> dict[str, Any]:
    payload = board_payload()
    assert_frozen_board(payload)
    digest = _exclusive_immutable_write(path, pretty_json_bytes(payload))
    return {
        "board_sha256": digest,
        "case_count": len(payload["cases"]),
        "content_sha256": payload["content_sha256"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="exclusive output JSON path")
    args = parser.parse_args()
    print(json.dumps(write_board(args.out), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
