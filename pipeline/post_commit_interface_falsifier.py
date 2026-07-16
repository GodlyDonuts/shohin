#!/usr/bin/env python3
"""Exact CPU falsifier for post-commit state-versus-motor interfaces.

This module is model-free. It verifies that a frozen evaluator separates a
complete four-element state packet from an equal-width packet that contains
only answers for a public consumer subspace. It does not train Shohin or prove
that every finite motor table is impossible.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import random
import stat
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


PROTOCOL_ID = "R12-PCIF-F17x4-v1"
SCHEMA_VERSION = 1
MODULUS = 17
DIMENSION = 4
PUBLIC_DIMENSION = 2
PACKET_WIDTH = 4
SOURCE_SEED = 2026071504
CHALLENGE_SEED = 2026071505
ALTERNATE_CHALLENGE_SEED = 2026071506
DEPTHS = (1, 2, 4, 8, 9)
DECISIVE_KINDS = ("hidden_consumer", "hidden_update", "joint")


class AuditError(ValueError):
    """Raised when a packet or generated report violates the frozen contract."""


Vector = tuple[int, int, int, int]
Matrix = tuple[Vector, Vector, Vector, Vector]


@dataclass(frozen=True)
class SealedPacket:
    """The complete object available after the source process exits."""

    values: Vector

    def __post_init__(self) -> None:
        if len(self.values) != PACKET_WIDTH:
            raise AuditError(f"packet width must be {PACKET_WIDTH}")
        if any(not 0 <= int(value) < MODULUS for value in self.values):
            raise AuditError("packet value is outside F_17")

    def serialized(self) -> dict[str, list[int]]:
        return {"values": [int(value) for value in self.values]}


@dataclass(frozen=True)
class AffineUpdate:
    matrix: Matrix
    offset: Vector

    def apply(self, value: Vector) -> Vector:
        return vec_add(mat_vec(self.matrix, value), self.offset)

    def serialized(self) -> dict[str, Any]:
        return {
            "matrix": [list(row) for row in self.matrix],
            "offset": list(self.offset),
        }


@dataclass(frozen=True)
class Challenge:
    challenge_id: str
    kind: str
    depth: int
    updates: tuple[AffineUpdate, ...]
    consumer: Vector
    output_permutation: tuple[int, ...]

    def serialized(self) -> dict[str, Any]:
        return {
            "challenge_id": self.challenge_id,
            "kind": self.kind,
            "depth": self.depth,
            "updates": [update.serialized() for update in self.updates],
            "consumer": list(self.consumer),
            "output_permutation": list(self.output_permutation),
        }


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("ascii")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def mod(value: int) -> int:
    return int(value) % MODULUS


def enumerate_states() -> Iterable[Vector]:
    return itertools.product(range(MODULUS), repeat=DIMENSION)


def dot(left: Sequence[int], right: Sequence[int]) -> int:
    if len(left) != DIMENSION or len(right) != DIMENSION:
        raise AuditError("dot operands must have dimension four")
    return mod(sum(int(a) * int(b) for a, b in zip(left, right, strict=True)))


def vec_add(left: Sequence[int], right: Sequence[int]) -> Vector:
    if len(left) != DIMENSION or len(right) != DIMENSION:
        raise AuditError("vector operands must have dimension four")
    return tuple(
        mod(int(a) + int(b)) for a, b in zip(left, right, strict=True)
    )  # type: ignore[return-value]


def mat_vec(matrix: Sequence[Sequence[int]], vector: Sequence[int]) -> Vector:
    if len(matrix) != DIMENSION or any(len(row) != DIMENSION for row in matrix):
        raise AuditError("matrix must be four by four")
    return tuple(dot(row, vector) for row in matrix)  # type: ignore[return-value]


def mat_mul(left: Sequence[Sequence[int]], right: Sequence[Sequence[int]]) -> Matrix:
    if len(left) != DIMENSION or len(right) != DIMENSION:
        raise AuditError("matrices must be four by four")
    columns = tuple(tuple(right[row][column] for row in range(DIMENSION)) for column in range(DIMENSION))
    return tuple(
        tuple(dot(row, column) for column in columns) for row in left
    )  # type: ignore[return-value]


def transpose(matrix: Sequence[Sequence[int]]) -> Matrix:
    return tuple(
        tuple(int(matrix[row][column]) for row in range(DIMENSION))
        for column in range(DIMENSION)
    )  # type: ignore[return-value]


def identity_matrix() -> Matrix:
    return tuple(
        tuple(1 if row == column else 0 for column in range(DIMENSION))
        for row in range(DIMENSION)
    )  # type: ignore[return-value]


def is_invertible(matrix: Sequence[Sequence[int]]) -> bool:
    work = [[mod(value) for value in row] for row in matrix]
    rank = 0
    for column in range(DIMENSION):
        pivot = next(
            (row for row in range(rank, DIMENSION) if work[row][column]), None
        )
        if pivot is None:
            continue
        work[rank], work[pivot] = work[pivot], work[rank]
        inverse = pow(work[rank][column], -1, MODULUS)
        work[rank] = [mod(value * inverse) for value in work[rank]]
        for row in range(DIMENSION):
            if row == rank:
                continue
            factor = work[row][column]
            if factor:
                work[row] = [
                    mod(value - factor * pivot_value)
                    for value, pivot_value in zip(work[row], work[rank], strict=True)
                ]
        rank += 1
    return rank == DIMENSION


def state_packet(state: Vector) -> SealedPacket:
    return SealedPacket(tuple(mod(value) for value in state))  # type: ignore[arg-type]


def motor_packet(state: Vector) -> SealedPacket:
    return SealedPacket((mod(state[0]), mod(state[1]), 0, 0))


def validate_serialized_packet(value: Mapping[str, Any]) -> SealedPacket:
    if set(value) != {"values"}:
        raise AuditError("sealed packet may contain only the values field")
    raw = value["values"]
    if not isinstance(raw, list) or len(raw) != PACKET_WIDTH:
        raise AuditError(f"serialized packet must contain {PACKET_WIDTH} values")
    if any(not isinstance(item, int) for item in raw):
        raise AuditError("serialized packet values must be integers")
    return SealedPacket(tuple(raw))  # type: ignore[arg-type]


def packet_reader(packet: SealedPacket, coefficient: Vector, constant: int) -> int:
    """Read an affine answer using only the sealed packet and late interface."""

    return mod(dot(coefficient, packet.values) + constant)


def sealed_packet_has_no_source_fields() -> bool:
    return {field.name for field in fields(SealedPacket)} == {"values"}


def phase_one_hashes() -> dict[str, Any]:
    state_hash = hashlib.sha256()
    motor_hash = hashlib.sha256()
    paired_hash = hashlib.sha256()
    count = 0
    for source in enumerate_states():
        complete = state_packet(source).serialized()
        motor = motor_packet(source).serialized()
        state_payload = canonical_json_bytes(complete)
        motor_payload = canonical_json_bytes(motor)
        state_hash.update(state_payload)
        motor_hash.update(motor_payload)
        paired_hash.update(canonical_json_bytes({"state": complete, "motor": motor}))
        count += 1
    return {
        "source_seed": SOURCE_SEED,
        "source_count": count,
        "packet_width_field_elements": PACKET_WIDTH,
        "state_packets_sha256": state_hash.hexdigest(),
        "motor_packets_sha256": motor_hash.hexdigest(),
        "paired_packets_sha256": paired_hash.hexdigest(),
    }


def _random_vector(rng: random.Random) -> Vector:
    return tuple(rng.randrange(MODULUS) for _ in range(DIMENSION))  # type: ignore[return-value]


def _random_matrix(rng: random.Random) -> Matrix:
    while True:
        matrix = tuple(
            tuple(rng.randrange(MODULUS) for _ in range(DIMENSION))
            for _ in range(DIMENSION)
        )
        if is_invertible(matrix):
            return matrix  # type: ignore[return-value]


def _random_invertible_two(rng: random.Random) -> tuple[tuple[int, int], tuple[int, int]]:
    while True:
        matrix = (
            (rng.randrange(MODULUS), rng.randrange(MODULUS)),
            (rng.randrange(MODULUS), rng.randrange(MODULUS)),
        )
        determinant = mod(matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0])
        if determinant:
            return matrix


def _public_matrix(rng: random.Random) -> Matrix:
    upper = _random_invertible_two(rng)
    lower = _random_invertible_two(rng)
    coupling = (
        (rng.randrange(MODULUS), rng.randrange(MODULUS)),
        (rng.randrange(MODULUS), rng.randrange(MODULUS)),
    )
    matrix = (
        (upper[0][0], upper[0][1], 0, 0),
        (upper[1][0], upper[1][1], 0, 0),
        (coupling[0][0], coupling[0][1], lower[0][0], lower[0][1]),
        (coupling[1][0], coupling[1][1], lower[1][0], lower[1][1]),
    )
    if not is_invertible(matrix):
        raise AssertionError("block-preserving construction must be invertible")
    return matrix


def _updates(rng: random.Random, depth: int, *, public: bool) -> tuple[AffineUpdate, ...]:
    factory = _public_matrix if public else _random_matrix
    return tuple(
        AffineUpdate(factory(rng), _random_vector(rng)) for _ in range(depth)
    )


def compose_affine(updates: Sequence[AffineUpdate]) -> AffineUpdate:
    matrix = identity_matrix()
    offset: Vector = (0, 0, 0, 0)
    for update in updates:
        offset = vec_add(mat_vec(update.matrix, offset), update.offset)
        matrix = mat_mul(update.matrix, matrix)
    return AffineUpdate(matrix, offset)


def effective_functional(challenge: Challenge) -> tuple[Vector, int]:
    total = compose_affine(challenge.updates)
    coefficient = mat_vec(transpose(total.matrix), challenge.consumer)
    constant = dot(challenge.consumer, total.offset)
    return coefficient, constant


def _public_consumer(rng: random.Random) -> Vector:
    while True:
        result: Vector = (rng.randrange(MODULUS), rng.randrange(MODULUS), 0, 0)
        if any(result):
            return result


def _hidden_consumer(rng: random.Random) -> Vector:
    while True:
        result = _random_vector(rng)
        if any(result[PUBLIC_DIMENSION:]):
            return result


def _derangement(rng: random.Random) -> tuple[int, ...]:
    shift = rng.randrange(1, MODULUS)
    result = tuple((value + shift) % MODULUS for value in range(MODULUS))
    if any(result[value] == value for value in range(MODULUS)):
        raise AssertionError("nonzero cyclic shift must be a derangement")
    return result


def _make_decisive_challenge(
    rng: random.Random,
    kind: str,
    depth: int,
    output_permutation: tuple[int, ...],
) -> Challenge:
    if kind not in DECISIVE_KINDS:
        raise ValueError(f"unknown decisive kind: {kind}")
    for attempt in range(10_000):
        if kind == "hidden_consumer":
            updates = _updates(rng, depth, public=True)
            consumer = _hidden_consumer(rng)
        elif kind == "hidden_update":
            updates = _updates(rng, depth, public=False)
            consumer = _public_consumer(rng)
        else:
            updates = _updates(rng, depth, public=False)
            consumer = _hidden_consumer(rng)
        challenge = Challenge(
            challenge_id=f"{kind}-d{depth}-a{attempt}",
            kind=kind,
            depth=depth,
            updates=updates,
            consumer=consumer,
            output_permutation=output_permutation,
        )
        coefficient, _ = effective_functional(challenge)
        if any(coefficient[PUBLIC_DIMENSION:]):
            return challenge
    raise AuditError("failed to generate a decisive challenge")


def generate_challenges(challenge_seed: int = CHALLENGE_SEED) -> dict[str, Any]:
    rng = random.Random(int(challenge_seed))
    output_permutation = _derangement(rng)
    public: list[Challenge] = []
    decisive: list[Challenge] = []
    for depth in DEPTHS:
        public.append(
            Challenge(
                challenge_id=f"public-d{depth}",
                kind="public",
                depth=depth,
                updates=_updates(rng, depth, public=True),
                consumer=_public_consumer(rng),
                output_permutation=output_permutation,
            )
        )
        for kind in DECISIVE_KINDS:
            decisive.append(
                _make_decisive_challenge(rng, kind, depth, output_permutation)
            )
    serialized = {
        "challenge_seed": int(challenge_seed),
        "output_permutation": list(output_permutation),
        "public": [item.serialized() for item in public],
        "decisive": [item.serialized() for item in decisive],
    }
    return {
        "challenge_seed": int(challenge_seed),
        "output_permutation": output_permutation,
        "public": tuple(public),
        "decisive": tuple(decisive),
        "challenge_payload_sha256": sha256_bytes(canonical_json_bytes(serialized)),
    }


def _collision_witness(challenge: Challenge) -> dict[str, Any]:
    coefficient, constant = effective_functional(challenge)
    hidden_coordinate = next(
        index
        for index in range(PUBLIC_DIMENSION, DIMENSION)
        if coefficient[index]
    )
    left: Vector = (0, 0, 0, 0)
    right_values = [0, 0, 0, 0]
    right_values[hidden_coordinate] = 1
    right: Vector = tuple(right_values)  # type: ignore[assignment]
    left_packet = motor_packet(left)
    right_packet = motor_packet(right)
    if left_packet != right_packet:
        raise AssertionError("collision sources must share a motor packet")
    left_answer = mod(dot(coefficient, left) + constant)
    right_answer = mod(dot(coefficient, right) + constant)
    if left_answer == right_answer:
        raise AssertionError("collision witness must have different answers")
    permutation = challenge.output_permutation
    return {
        "left_source": list(left),
        "right_source": list(right),
        "shared_motor_packet": left_packet.serialized(),
        "distinguishing_hidden_coordinate": hidden_coordinate,
        "left_answer": left_answer,
        "right_answer": right_answer,
        "left_recoded": permutation[left_answer],
        "right_recoded": permutation[right_answer],
    }


def score_challenge(challenge: Challenge) -> dict[str, Any]:
    coefficient, constant = effective_functional(challenge)
    permutation = challenge.output_permutation
    state_correct = 0
    state_recoded_correct = 0
    motor_correct = 0
    motor_recoded_correct = 0
    total = 0
    for source in enumerate_states():
        truth = mod(dot(coefficient, source) + constant)
        complete = validate_serialized_packet(state_packet(source).serialized())
        motor = validate_serialized_packet(motor_packet(source).serialized())
        state_prediction = packet_reader(complete, coefficient, constant)
        motor_prediction = packet_reader(motor, coefficient, constant)
        state_correct += int(state_prediction == truth)
        motor_correct += int(motor_prediction == truth)
        state_recoded_correct += int(permutation[state_prediction] == permutation[truth])
        motor_recoded_correct += int(permutation[motor_prediction] == permutation[truth])
        total += 1
    decisive = any(coefficient[PUBLIC_DIMENSION:])
    result = {
        "challenge_id": challenge.challenge_id,
        "kind": challenge.kind,
        "depth": challenge.depth,
        "effective_functional": list(coefficient),
        "effective_constant": constant,
        "decisive_outside_public_span": decisive,
        "total_sources": total,
        "state_correct": state_correct,
        "state_recoded_correct": state_recoded_correct,
        "motor_correct": motor_correct,
        "motor_recoded_correct": motor_recoded_correct,
        "state_accuracy": f"{state_correct}/{total}",
        "motor_accuracy": f"{motor_correct}/{total}",
    }
    if decisive:
        result["collision_witness"] = _collision_witness(challenge)
    return result


def source_pointer_decoy_audit() -> dict[str, Any]:
    rejected = False
    reason = ""
    try:
        validate_serialized_packet(
            {"values": [0, 0, 0, 0], "source_id": "forbidden-pointer"}
        )
    except AuditError as exc:
        rejected = True
        reason = str(exc)
    return {"rejected": rejected, "reason": reason}


def score_horizon_decoy(challenge: Challenge, max_depth: int = 8) -> dict[str, Any]:
    """Replay a source-free reader that deliberately breaks after max_depth."""

    coefficient, constant = effective_functional(challenge)
    correct = 0
    total = 0
    for source in enumerate_states():
        truth = mod(dot(coefficient, source) + constant)
        packet = validate_serialized_packet(state_packet(source).serialized())
        prediction = packet_reader(packet, coefficient, constant)
        if challenge.depth > max_depth:
            prediction = mod(prediction + 1)
        correct += int(prediction == truth)
        total += 1
    return {
        "challenge_id": challenge.challenge_id,
        "depth": challenge.depth,
        "max_depth": int(max_depth),
        "correct": correct,
        "total_sources": total,
        "accuracy": f"{correct}/{total}",
    }


def horizon_decoy_audit(challenges: Sequence[Challenge]) -> dict[str, Any]:
    rows = [score_horizon_decoy(challenge) for challenge in challenges]
    return {
        "declared_horizon": 8,
        "rows": rows,
        "passes_through_8": all(
            row["correct"] == row["total_sources"]
            for row in rows
            if row["depth"] <= 8
        ),
        "rejected_at_9": all(
            row["correct"] < row["total_sources"]
            for row in rows
            if row["depth"] == 9
        ),
    }


def code_sha256() -> str:
    return sha256_bytes(Path(__file__).resolve().read_bytes())


def build_report(challenge_seed: int = CHALLENGE_SEED) -> dict[str, Any]:
    phase_one = phase_one_hashes()
    challenge_bundle = generate_challenges(challenge_seed)
    public_results = [score_challenge(item) for item in challenge_bundle["public"]]
    decisive_results = [score_challenge(item) for item in challenge_bundle["decisive"]]
    source_pointer = source_pointer_decoy_audit()
    horizon = horizon_decoy_audit(challenge_bundle["decisive"])
    expected_total = MODULUS**DIMENSION
    expected_motor = MODULUS ** (DIMENSION - 1)
    gates = {
        "equal_packet_width": PACKET_WIDTH == DIMENSION,
        "packet_schema_has_no_source_fields": sealed_packet_has_no_source_fields(),
        "public_state_all_exact": all(row["state_correct"] == expected_total for row in public_results),
        "public_motor_all_exact": all(row["motor_correct"] == expected_total for row in public_results),
        "decisive_state_all_exact": all(row["state_correct"] == expected_total for row in decisive_results),
        "decisive_state_recoded_all_exact": all(row["state_recoded_correct"] == expected_total for row in decisive_results),
        "decisive_motor_exactly_one_over_17": all(row["motor_correct"] == expected_motor for row in decisive_results),
        "decisive_motor_recoded_exactly_one_over_17": all(row["motor_recoded_correct"] == expected_motor for row in decisive_results),
        "all_decisive_cells_have_collision_witness": all("collision_witness" in row for row in decisive_results),
        "source_pointer_decoy_rejected": source_pointer["rejected"],
        "horizon_decoy_passes_through_8": horizon["passes_through_8"],
        "horizon_decoy_rejected_at_9": horizon["rejected_at_9"],
    }
    report: dict[str, Any] = {
        "audit": "post_commit_interface_falsifier_v1",
        "protocol_id": PROTOCOL_ID,
        "schema_version": SCHEMA_VERSION,
        "config": {
            "field_modulus": MODULUS,
            "state_dimension": DIMENSION,
            "public_consumer_dimension": PUBLIC_DIMENSION,
            "packet_width_field_elements": PACKET_WIDTH,
            "depths": list(DEPTHS),
            "decisive_kinds": list(DECISIVE_KINDS),
            "expected_source_count": expected_total,
            "expected_motor_correct_per_decisive_cell": expected_motor,
        },
        "seeds": {"source": SOURCE_SEED, "challenge": int(challenge_seed)},
        "code_sha256": code_sha256(),
        "phase_one": phase_one,
        "phase_two": {
            "challenge_payload_sha256": challenge_bundle["challenge_payload_sha256"],
            "output_permutation": list(challenge_bundle["output_permutation"]),
            "public_challenges": len(public_results),
            "decisive_challenges": len(decisive_results),
        },
        "resource_vector": {
            "state_packet_field_elements": PACKET_WIDTH,
            "motor_packet_field_elements": PACKET_WIDTH,
            "state_packet_bits_upper_bound": PACKET_WIDTH * 5,
            "motor_packet_bits_upper_bound": PACKET_WIDTH * 5,
            "post_commit_source_fields": 0,
            "training_examples": 0,
            "trainable_parameters": 0,
            "external_execution": "exact_cpu_positive_and_control_only",
        },
        "public_results": public_results,
        "decisive_results": decisive_results,
        "source_pointer_decoy": source_pointer,
        "horizon_decoy": horizon,
        "gates": gates,
        "pass": all(gates.values()),
        "claim_boundary": (
            "A pass proves only that this bounded exact scorer separates the declared "
            "four-element state packet from the equal-width public-answer motor packet. "
            "It does not exclude unlimited finite tables or establish learned reasoning."
        ),
    }
    report["payload_sha256"] = sha256_bytes(canonical_json_bytes(report))
    return report


def verify_report(report: Mapping[str, Any]) -> None:
    copy = dict(report)
    claimed = copy.pop("payload_sha256", None)
    actual = sha256_bytes(canonical_json_bytes(copy))
    if claimed != actual:
        raise AuditError(f"payload hash mismatch: expected {claimed}, got {actual}")
    if not report.get("pass"):
        raise AuditError("falsifier report did not pass every frozen gate")
    if not all(report.get("gates", {}).values()):
        raise AuditError("one or more frozen gates are false")


def report_bytes(challenge_seed: int = CHALLENGE_SEED) -> bytes:
    report = build_report(challenge_seed)
    verify_report(report)
    return canonical_json_bytes(report)


def write_immutable_report(path: Path, report: Mapping[str, Any]) -> None:
    verify_report(report)
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        payload = canonical_json_bytes(report)
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(descriptor)
    path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--challenge-seed", type=int, default=CHALLENGE_SEED)
    args = parser.parse_args()
    report = build_report(args.challenge_seed)
    verify_report(report)
    write_immutable_report(args.out, report)
    print(
        "[pcif] pass={} public={} decisive={} payload_sha256={}".format(
            report["pass"],
            report["phase_two"]["public_challenges"],
            report["phase_two"]["decisive_challenges"],
            report["payload_sha256"],
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
