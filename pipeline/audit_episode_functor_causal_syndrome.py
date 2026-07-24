#!/usr/bin/env python3
"""Audit single-cell causal-syndrome identifiability on the frozen EFC board.

This is an exact CPU mechanics test. It does not fit a neural model or show
that noisy learned signatures can be corrected.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from functools import lru_cache
from hashlib import sha256
import json
from pathlib import Path
import sys
from typing import cast

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.audit_episode_functor_identifiable_board import (  # noqa: E402
    DEFAULT_COUNTS,
)
from pipeline.episode_functor_hankel_shift import (  # noqa: E402
    HankelCodebook,
    build_hankel_codebook,
)
from pipeline.episode_functor_identifiable_board import (  # noqa: E402
    ACTION_COUNT,
    ANSWER_COUNT,
    IdentifiableMachine,
    OBSERVER_COUNT,
    STATE_COUNT,
    generate_pilot_rows,
)


AUDIT_SCHEMA = "efc-adjoint-causal-syndrome-single-swap-audit/v1"
DEFAULT_SEED = "efc-identifiable-pilot-v1"


class CausalSyndromeAuditError(ValueError):
    """The finite fault-localization audit failed closed."""


@dataclass(frozen=True, slots=True)
class _MachineTables:
    transitions: tuple[tuple[int, ...], ...]
    observations: tuple[tuple[int, ...], ...]


@dataclass(frozen=True, slots=True)
class WorldSyndromeReceipt:
    world_id: str
    fault_count: int
    unique_fingerprint_count: int
    zero_fingerprint_count: int
    collision_count: int
    minimum_changed_coordinates: int
    maximum_changed_coordinates: int
    fingerprint_set_sha256: str


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _flatten_codebook(codebook: HankelCodebook) -> tuple[int, ...]:
    return (
        tuple(
            answer
            for signature in codebook.base
            for word in signature
            for answer in word
        )
        + tuple(
            answer
            for action in codebook.derivative
            for signature in action
            for word in signature
            for answer in word
        )
    )


def _syndrome_fingerprint(
    reference: tuple[int, ...],
    mutated: tuple[int, ...],
) -> tuple[tuple[int, int, int], ...]:
    if len(reference) != len(mutated):
        raise CausalSyndromeAuditError("codebook sizes differ")
    return tuple(
        (index, before, after)
        for index, (before, after) in enumerate(
            zip(reference, mutated, strict=True)
        )
        if before != after
    )


def _codebook_for_tables(
    transitions: tuple[tuple[int, ...], ...],
    observations: tuple[tuple[int, ...], ...],
) -> tuple[int, ...]:
    tables = _MachineTables(
        transitions=transitions,
        observations=observations,
    )
    # The exact Hankel mechanics consume only these two immutable tables.
    machine = cast(IdentifiableMachine, tables)
    return _flatten_codebook(
        build_hankel_codebook(machine, max_depth=3)
    )


def _transition_swaps(
    machine: IdentifiableMachine,
) -> tuple[tuple[str, tuple[tuple[int, ...], ...], tuple[tuple[int, ...], ...]], ...]:
    mutations = []
    for action in range(ACTION_COUNT):
        for left in range(STATE_COUNT):
            for right in range(left + 1, STATE_COUNT):
                transitions = [
                    list(row) for row in machine.transitions
                ]
                transitions[action][left], transitions[action][right] = (
                    transitions[action][right],
                    transitions[action][left],
                )
                mutations.append(
                    (
                        f"transition:{action}:{left}:{right}",
                        tuple(tuple(row) for row in transitions),
                        machine.observations,
                    )
                )
    return tuple(mutations)


def _observer_swaps(
    machine: IdentifiableMachine,
) -> tuple[tuple[str, tuple[tuple[int, ...], ...], tuple[tuple[int, ...], ...]], ...]:
    mutations = []
    for observer in range(OBSERVER_COUNT):
        for left in range(STATE_COUNT):
            for right in range(left + 1, STATE_COUNT):
                if (
                    machine.observations[observer][left]
                    == machine.observations[observer][right]
                ):
                    continue
                observations = [
                    list(row) for row in machine.observations
                ]
                observations[observer][left], observations[observer][right] = (
                    observations[observer][right],
                    observations[observer][left],
                )
                mutations.append(
                    (
                        f"observer:{observer}:{left}:{right}",
                        machine.transitions,
                        tuple(tuple(row) for row in observations),
                    )
                )
    return tuple(mutations)


def audit_world(
    world_id: str,
    machine: IdentifiableMachine,
) -> WorldSyndromeReceipt:
    reference = _codebook_for_tables(
        machine.transitions,
        machine.observations,
    )
    seen: dict[tuple[tuple[int, int, int], ...], str] = {}
    collisions = 0
    zeros = 0
    changed_counts: list[int] = []
    mutations = _transition_swaps(machine) + _observer_swaps(machine)
    expected_faults = (
        ACTION_COUNT * STATE_COUNT * (STATE_COUNT - 1) // 2
        + OBSERVER_COUNT
        * (
            STATE_COUNT * (STATE_COUNT - 1) // 2
            - ANSWER_COUNT
        )
    )
    if len(mutations) != expected_faults:
        raise CausalSyndromeAuditError(
            "single-swap fault inventory differs"
        )
    for name, transitions, observations in mutations:
        mutated = _codebook_for_tables(transitions, observations)
        fingerprint = _syndrome_fingerprint(reference, mutated)
        if not fingerprint:
            zeros += 1
        if fingerprint in seen:
            collisions += 1
        else:
            seen[fingerprint] = name
        changed_counts.append(len(fingerprint))
    fingerprint_rows = tuple(
        {
            "fault": fault,
            "fingerprint": fingerprint,
        }
        for fingerprint, fault in sorted(
            seen.items(),
            key=lambda item: item[1],
        )
    )
    return WorldSyndromeReceipt(
        world_id=world_id,
        fault_count=len(mutations),
        unique_fingerprint_count=len(seen),
        zero_fingerprint_count=zeros,
        collision_count=collisions,
        minimum_changed_coordinates=min(changed_counts),
        maximum_changed_coordinates=max(changed_counts),
        fingerprint_set_sha256=sha256(
            _canonical_json_bytes(fingerprint_rows)
        ).hexdigest(),
    )


@lru_cache(maxsize=None)
def audit_frozen_board(
    *,
    seed: str = DEFAULT_SEED,
) -> dict[str, object]:
    rows = generate_pilot_rows(
        seed=seed,
        counts=DEFAULT_COUNTS,
    )
    machines = {
        row.world_id: row.machine
        for row in rows
    }
    receipts = tuple(
        audit_world(world_id, machines[world_id])
        for world_id in sorted(machines)
    )
    total_faults = sum(row.fault_count for row in receipts)
    total_unique = sum(row.unique_fingerprint_count for row in receipts)
    total_zero = sum(row.zero_fingerprint_count for row in receipts)
    total_collisions = sum(row.collision_count for row in receipts)
    report = {
        "schema": AUDIT_SCHEMA,
        "seed": seed,
        "world_count": len(receipts),
        "faults_per_world": sorted(
            {row.fault_count for row in receipts}
        ),
        "total_faults": total_faults,
        "total_unique_within_world_fingerprints": total_unique,
        "zero_fingerprint_count": total_zero,
        "within_world_collision_count": total_collisions,
        "minimum_changed_coordinates": min(
            row.minimum_changed_coordinates for row in receipts
        ),
        "maximum_changed_coordinates": max(
            row.maximum_changed_coordinates for row in receipts
        ),
        "world_receipts": [asdict(row) for row in receipts],
        "decision": (
            "single_swap_syndromes_identifiable_mechanics_only"
            if total_faults == total_unique
            and total_zero == 0
            and total_collisions == 0
            else "single_swap_syndromes_not_identifiable"
        ),
        "claim_boundary": (
            "exact finite CPU fault localization only; no noisy-signature, "
            "neural-learnability, source-compilation, reasoning, or "
            "pretraining claim"
        ),
    }
    report["report_payload_sha256"] = sha256(
        _canonical_json_bytes(report)
    ).hexdigest()
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=DEFAULT_SEED)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    report = audit_frozen_board(seed=args.seed)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered, encoding="ascii")
    print(rendered, end="")


if __name__ == "__main__":
    main()
