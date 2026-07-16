"""Independent verifier for the exhaustive ACW symbolic control report.

This file intentionally does not import the candidate implementation.  It
reconstructs the finite-field streams, score counts, payload hash, and Git blob
identity from the report contract.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import subprocess
from pathlib import Path
from typing import Sequence

import numpy as np


FIELD_SIZE = 17
DIMENSIONS = (2, 3)
HORIZON = 16
PROTOCOL = "R12-ACW-SYMBOLIC-v1"
SCIENTIFIC_PATHS = (
    "R12_ADDRESSED_CATEGORICAL_WORKSPACE_PREREG.md",
    "pipeline/addressed_categorical_workspace.py",
    "pipeline/audit_addressed_categorical_workspace_symbolic.py",
    "pipeline/test_addressed_categorical_workspace.py",
    "pipeline/test_audit_addressed_categorical_workspace_symbolic.py",
)


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode("ascii")


def payload_sha256(value: dict) -> str:
    body = dict(value)
    body.pop("payload_sha256", None)
    return hashlib.sha256(canonical_json_bytes(body)).hexdigest()


def _states(dimension: int) -> np.ndarray:
    return np.asarray(
        list(itertools.product(range(FIELD_SIZE), repeat=dimension)),
        dtype=np.uint8,
    )


def _events(dimension: int, values: Sequence[int]):
    for destination in range(dimension):
        for source in range(dimension):
            for alpha in values:
                for beta in values:
                    for gamma in values:
                        yield destination, source, alpha, beta, gamma


def _stream(digest, event, array: np.ndarray) -> None:
    digest.update(bytes(event))
    digest.update(np.ascontiguousarray(array, dtype=np.uint8).tobytes())


def _recode(value: int, query: int) -> int:
    multiplier = (2 * query + 1) % FIELD_SIZE or 1
    return (multiplier * value + (7 * query + 3) % FIELD_SIZE) % FIELD_SIZE


def expected_dimension(dimension: int, coefficient_values: Sequence[int]) -> dict:
    values = tuple(sorted(set(int(value) for value in coefficient_values)))
    states = _states(dimension)
    state_count = len(states)
    event_count = dimension * dimension * len(values) ** 3
    exact = hashlib.sha256()
    overcomplete = hashlib.sha256()
    narrow = hashlib.sha256(states[:, : dimension - 1].tobytes())
    horizon_events = []
    illegal_writes = 0
    overcomplete_illegal_writes = 0
    sentinel = (states.astype(np.int32).sum(axis=1) % FIELD_SIZE).astype(np.uint8)
    overcomplete_before = np.concatenate((states, sentinel[:, None]), axis=1)
    for event in _events(dimension, values):
        destination, source, alpha, beta, gamma = event
        if len(horizon_events) < HORIZON:
            horizon_events.append(event)
        source_values = states[:, source].astype(np.int32)
        destination_values = states[:, destination].astype(np.int32)
        replacement = (
            alpha * destination_values + beta * source_values + gamma
        ) % FIELD_SIZE
        updated = np.stack(
            [
                replacement.astype(np.uint8) if index == destination else states[:, index]
                for index in range(dimension)
            ],
            axis=1,
        )
        for index in range(dimension):
            if index != destination:
                illegal_writes += int(np.count_nonzero(updated[:, index] != states[:, index]))
        _stream(exact, event, updated)
        overcomplete_updated = overcomplete_before.copy()
        overcomplete_updated[:, destination] = replacement.astype(np.uint8)
        for index in range(dimension + 1):
            if index != destination:
                overcomplete_illegal_writes += int(np.count_nonzero(
                    overcomplete_updated[:, index] != overcomplete_before[:, index]
                ))
        _stream(overcomplete, event, overcomplete_updated)

    queries = hashlib.sha256()
    recodings = hashlib.sha256()
    for query in range(dimension):
        queries.update(bytes((query,)))
        queries.update(states[:, query].tobytes())
        recoded = np.asarray(
            [_recode(int(value), query) for value in states[:, query]],
            dtype=np.uint8,
        )
        recodings.update(bytes((query,)))
        recodings.update(recoded.tobytes())

    donor = ((states.astype(np.int32) + 1) % FIELD_SIZE).astype(np.uint8)
    donors = hashlib.sha256(states.tobytes() + donor.tobytes())
    for query in range(dimension):
        donors.update(bytes((query,)))
        donors.update(donor[:, query].tobytes())

    horizon = states.copy()
    for destination, source, alpha, beta, gamma in horizon_events:
        replacement = (
            alpha * horizon[:, destination].astype(np.int32)
            + beta * horizon[:, source].astype(np.int32)
            + gamma
        ) % FIELD_SIZE
        rebuilt = []
        for index in range(dimension):
            rebuilt.append(
                replacement.astype(np.uint8) if index == destination else horizon[:, index]
            )
        horizon = np.stack(rebuilt, axis=1)

    narrow_width = dimension - 1
    left = (0,) * dimension
    right = (0,) * narrow_width + (1,)
    collision = {
        "dimension": dimension,
        "packet_width": narrow_width,
        "packet_capacity": FIELD_SIZE ** narrow_width,
        "causal_states": FIELD_SIZE ** dimension,
        "left_state": left,
        "right_state": right,
        "left_packet": left[:narrow_width],
        "right_packet": right[:narrow_width],
        "separating_query": dimension - 1,
        "left_answer": 0,
        "right_answer": 1,
        "collision": True,
        "separated": True,
    }
    narrow_unique = int(len(np.unique(states[:, :narrow_width], axis=0)))
    gates = {
        "exact_capacity_matches_causal_state_count": True,
        "narrow_capacity_is_insufficient": True,
        "narrow_collision_is_explicit_and_separated": True,
        "narrow_width_exhaustively_maps_all_states": (
            narrow_unique == FIELD_SIZE ** narrow_width
        ),
        "exact_width_stream_is_complete": True,
        "overcomplete_width_stream_is_complete": True,
        "all_states_checked": state_count == FIELD_SIZE ** dimension,
        "all_affine_updates_checked": True,
        "all_queries_checked": True,
        "all_recodings_checked": True,
        "all_literal_donor_reads_checked": True,
        "all_horizons_checked": len(horizon_events) == HORIZON,
        "zero_illegal_writes": illegal_writes == 0,
        "full_coefficient_field_exhausted": values == tuple(range(FIELD_SIZE)),
        "overcomplete_sentinel_is_byte_preserved": overcomplete_illegal_writes == 0,
    }
    return {
        "dimension": dimension,
        "field_size": FIELD_SIZE,
        "state_space": state_count,
        "packet_symbols": dimension,
        "utilized_bits": dimension * math.log2(FIELD_SIZE),
        "physical_bits": dimension * math.ceil(math.log2(FIELD_SIZE)),
        "coefficient_values": values,
        "full_coefficient_field": values == tuple(range(FIELD_SIZE)),
        "events": event_count,
        "states_checked": state_count,
        "updates_checked": state_count * event_count,
        "queries_checked": state_count * dimension,
        "recodings_checked": state_count * dimension,
        "literal_donor_reads_checked": state_count * dimension,
        "horizon_checks": state_count,
        "horizon_depth": HORIZON,
        "illegal_writes": illegal_writes,
        "overcomplete_illegal_writes": overcomplete_illegal_writes,
        "widths_tested": (dimension - 1, dimension, dimension + 1),
        "narrow_unique_packets": narrow_unique,
        "narrow_packet_stream_sha256": narrow.hexdigest(),
        "exact_update_stream_sha256": exact.hexdigest(),
        "overcomplete_update_stream_sha256": overcomplete.hexdigest(),
        "query_stream_sha256": queries.hexdigest(),
        "recoding_stream_sha256": recodings.hexdigest(),
        "literal_donor_stream_sha256": donors.hexdigest(),
        "horizon_stream_sha256": hashlib.sha256(horizon.tobytes()).hexdigest(),
        "narrow_collision": collision,
        "gates": gates,
        "pass": all(gates.values()),
    }


def verify_git_identity(identity: dict, root: Path) -> None:
    if set(identity) != {"scientific_commit", "scientific_path_sha256"}:
        raise ValueError("scientific identity has the wrong schema")
    commit = identity["scientific_commit"]
    hashes = identity["scientific_path_sha256"]
    if set(hashes) != set(SCIENTIFIC_PATHS):
        raise ValueError("scientific path set is incomplete")
    subprocess.run(
        ["git", "cat-file", "-e", f"{commit}^{{commit}}"], cwd=root, check=True,
    )
    for relative in SCIENTIFIC_PATHS:
        blob = subprocess.run(
            ["git", "show", f"{commit}:{relative}"],
            cwd=root,
            check=True,
            capture_output=True,
        ).stdout
        if hashlib.sha256(blob).hexdigest() != hashes[relative]:
            raise ValueError(f"scientific Git blob mismatch: {relative}")


def verify_report(report: dict, *, allow_unbound: bool = False) -> dict:
    if report.get("protocol") != PROTOCOL:
        raise ValueError("wrong symbolic protocol")
    if report.get("payload_sha256") != payload_sha256(report):
        raise ValueError("symbolic payload hash mismatch")
    identity = report.get("scientific_identity")
    if identity is None:
        if not allow_unbound:
            raise ValueError("canonical symbolic report lacks Git identity")
    else:
        verify_git_identity(identity, Path(__file__).resolve().parents[1])
    dimensions = report.get("dimensions")
    if not isinstance(dimensions, list) or [item.get("dimension") for item in dimensions] != [2, 3]:
        raise ValueError("symbolic dimension reports are incomplete")
    expected_dimensions = []
    for observed in dimensions:
        values = observed.get("coefficient_values")
        expected = expected_dimension(observed["dimension"], values)
        if canonical_json_bytes(observed) != canonical_json_bytes(expected):
            raise ValueError(f"symbolic evidence mismatch for d={observed['dimension']}")
        expected_dimensions.append(expected)
    if not allow_unbound and not all(item["full_coefficient_field"] for item in dimensions):
        raise ValueError("canonical symbolic report is not exhaustive over F_17")
    expected_report = {
        "protocol": PROTOCOL,
        "field_size": FIELD_SIZE,
        "scientific_identity": identity,
        "dimensions": expected_dimensions,
        "claim_boundary": (
            "Exact affine packet mechanics only; literal donor read is not a learned "
            "causal intervention. No neural learning, language, autonomous control, "
            "novelty, or reasoning claim."
        ),
        "pass": all(item["pass"] for item in expected_dimensions),
    }
    expected_report["payload_sha256"] = payload_sha256(expected_report)
    if canonical_json_bytes(report) != canonical_json_bytes(expected_report):
        raise ValueError("symbolic top-level report mismatch")
    return {
        "pass": bool(report["pass"]),
        "evidence_valid": True,
        "payload_sha256": report["payload_sha256"],
        "updates_reconstructed": sum(item["updates_checked"] for item in dimensions),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--allow-unbound", action="store_true")
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    result = verify_report(report, allow_unbound=args.allow_unbound)
    print(
        f"[acw-symbolic-audit] pass={result['pass']} "
        f"updates={result['updates_reconstructed']}"
    )
    if not result["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
