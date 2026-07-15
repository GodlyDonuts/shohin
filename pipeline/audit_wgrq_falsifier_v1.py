#!/usr/bin/env python3
"""Fail-closed independent audit for the frozen WGRQ Stage-A acquisition.

The generator and residual-oracle modules are deliberately not imported.  The
auditor independently replays every source event and continuation, rebuilds
all ordinary READ answers and answer-derived relation targets, joins every
call-ledger row, checks the frozen strata and cancellation controls, and binds
the immutable transcript, ledger, and generation report by SHA-256.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Any, Iterable, Mapping, Sequence


TRANSCRIPT_SCHEMA = "wgrq_falsifier_v1"
LEDGER_SCHEMA = "wgrq_ordinary_read_call_v1"
GENERATION_REPORT_SCHEMA = "wgrq_falsifier_v1_report"
SYMBOLIC_AUDIT_SCHEMA = "wgrq_symbolic_audit_v1"
AUDIT_NAME = "wgrq_stage_a_independent_admission_v1"

TRAINING_SCALES = (4, 6, 8)
LENGTH_BANDS = ("le_2n", "le_8n")
LENGTH_MULTIPLIERS = {"le_2n": 2, "le_8n": 8}
SYMBOLIC_SCALES = (3, 6)
EPISODES_PER_CELL = 3_072
HISTORIES_PER_EPISODE = 4
PROBES_PER_HISTORY = 8
ORDINARY_CALLS_PER_EPISODE = HISTORIES_PER_EPISODE * PROBES_PER_HISTORY
FROZEN_BATCH_SIZE = 64
HISTORY_ROLES = ("equivalent_a", "equivalent_b", "different_a", "different_b")
EVENT_CODE_BITS = {"R": [1, 0], "F": [0, 1]}
FROZEN_PAIRED_SEEDS = (
    17011,
    27103,
    38119,
    49201,
    50311,
    61403,
    72503,
    83609,
    94709,
    105019,
    116027,
    127031,
)
FROZEN_PRF_SEED_ASCII = "R12-WGRQ-DWEPR-STAGE-A-v1"
FROZEN_PRF_SEED_HEX = FROZEN_PRF_SEED_ASCII.encode("ascii").hex()
PRF_FORMULA = "SHA256(seed || 0x00 || ASCII(domain) || uint64_be(counter))"
PRF_SELECTION = "unbiased finite-bank rejection sampling only"
PARITY_REASON = (
    "At shortest-witness depth n-2, the edge difference is supported on "
    "n-2 and n-1, so the physical difference has odd Hamming parity. "
    "Rotation preserves parity and each F toggles it; the two histories "
    "therefore cannot have equal F-event-count parity."
)
CLAIM_BOUNDARY = (
    "Stage-A data admission only; no model result, oracle advantage, language transfer, "
    "or Shohin training is authorized."
)

EPISODE_FIELDS = {
    "schema",
    "episode_id",
    "split",
    "global_episode_index",
    "batch_index",
    "batch_offset",
    "cell",
    "event_code_bits",
    "gadget",
    "probe_rotations",
    "histories",
    "equivalence_label_matrix",
    "pairs",
    "first_distinguishing_witness_mask",
    "uniform_probe_index",
    "uniform_probe_mask",
    "balance",
    "oracle_call_span",
}
CELL_FIELDS = {"n", "length_band", "source_length_ceiling", "cell_episode_index"}
GADGET_FIELDS = {
    "name",
    "events",
    "event_counts",
    "common_identity_padding_events",
    "padding_plan",
}
PADDING_PLAN_FIELDS = {"ff_blocks", "rn_blocks"}
HISTORY_FIELDS = {
    "history_index",
    "history_id",
    "role",
    "events",
    "source_length",
    "event_counts",
    "history_sha256",
    "probes",
    "canonical_edge_bits_from_public_answers",
}
PROBE_FIELDS = {
    "probe_index",
    "continuation_rotations",
    "continuation",
    "answer",
    "oracle_call_id",
}
PAIRS_FIELDS = {"equivalent", "non_equivalent"}
EQUIVALENT_PAIR_FIELDS = {"history_indices", "label"}
NON_EQUIVALENT_PAIR_FIELDS = {
    "history_indices",
    "label",
    "shortest_witness_depth",
    "first_distinguishing_probe_index",
    "first_distinguishing_witness_mask",
}
EPISODE_BALANCE_FIELDS = {
    "declared_pair_labels",
    "all_source_lengths_matched",
    "all_event_counts_matched",
    "maximum_depth_flip_parity_obstruction",
}
CALL_SPAN_FIELDS = {"first_call_id", "last_call_id", "ordinary_one_bit_read_calls"}
LEDGER_FIELDS = {
    "schema",
    "call_id",
    "episode_id",
    "global_episode_index",
    "history_index",
    "history_sha256",
    "probe_index",
    "continuation_rotations",
    "call_kind",
    "returned_bits",
    "answer",
}
GENERATION_REPORT_FIELDS = {
    "schema",
    "passed",
    "generation_contract",
    "symbolic_gates",
    "cells",
    "totals",
    "frozen_call_ledger",
    "balance",
    "prf_ledger",
    "artifacts",
    "hashes",
}
FORBIDDEN_PUBLIC_FIELDS = {
    "physical_state",
    "endpoint_state",
    "physical_endpoint",
    "residual_class_id",
    "hidden_state_id",
    "canonical_state_id",
    "source_state_id",
    "oracle_handle",
    "simulator_handle",
    "verifier_handle",
    "source_cache",
    "kv_cache",
}


class AuditError(ValueError):
    """One categorized, fail-closed contract violation."""

    def __init__(self, category: str, message: str):
        super().__init__(message)
        self.category = category


def require(condition: bool, category: str, message: str) -> None:
    if not condition:
        raise AuditError(category, message)


def exact_int(value: Any, name: str, *, minimum: int | None = None) -> int:
    require(
        isinstance(value, int) and not isinstance(value, bool),
        "structure",
        "{} must be an integer".format(name),
    )
    result = int(value)
    if minimum is not None:
        require(result >= minimum, "structure", "{} is below its minimum".format(name))
    return result


def exact_bit(value: Any, name: str) -> int:
    result = exact_int(value, name)
    require(result in (0, 1), "structure", "{} must be a bit".format(name))
    return result


def exact_bool(value: Any, name: str) -> bool:
    require(isinstance(value, bool), "structure", "{} must be boolean".format(name))
    return bool(value)


def exact_fields(value: Any, fields: set[str], name: str) -> dict[str, Any]:
    require(isinstance(value, dict), "structure", "{} must be an object".format(name))
    require(
        set(value) == fields,
        "public_fields",
        "{} fields differ from the frozen contract".format(name),
    )
    return value


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")


def canonical_jsonl_record(value: Any) -> bytes:
    return canonical_json_bytes(value) + b"\n"


def pretty_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    ).encode("ascii")


def canonical_hash(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def _reject_constant(value: str) -> None:
    raise AuditError("json", "non-finite JSON constant {} is forbidden".format(value))


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise AuditError("json", "duplicate JSON key {!r}".format(key))
        result[key] = value
    return result


def strict_json_loads(text: str) -> Any:
    try:
        return json.loads(
            text,
            object_pairs_hook=_object_without_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except AuditError:
        raise
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise AuditError("json", "invalid JSON: {}".format(error)) from error


def _scan_forbidden_public_fields(value: Any, location: str) -> None:
    if isinstance(value, dict):
        forbidden = FORBIDDEN_PUBLIC_FIELDS & set(value)
        require(
            not forbidden,
            "hidden_state",
            "{} contains forbidden fields {}".format(location, sorted(forbidden)),
        )
        for key, nested in value.items():
            _scan_forbidden_public_fields(nested, "{}.{}".format(location, key))
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            _scan_forbidden_public_fields(nested, "{}[{}]".format(location, index))


def rotate(state: int, n: int) -> int:
    return (state >> 1) | ((state & 1) << (n - 1))


def inverse_rotate(state: int, n: int) -> int:
    return ((state << 1) & ((1 << n) - 1)) | (state >> (n - 1))


def apply_event(state: int, event: str, n: int) -> int:
    if event == "R":
        return rotate(state, n)
    if event == "F":
        return state ^ 1
    raise AuditError("history", "event must be exactly 'R' or 'F'")


def replay_history(events: Iterable[str], n: int, initial: int = 0) -> int:
    state = initial
    for event in events:
        state = apply_event(state, event, n)
    return state


def read_sensor(state: int) -> int:
    return (state & 1) ^ ((state >> 1) & 1)


def read_after_rotations(state: int, rotations: int, n: int) -> int:
    branch = state
    for _ in range(rotations):
        branch = rotate(branch, n)
    return read_sensor(branch)


def residual_code(state: int, n: int) -> int:
    return state ^ rotate(state, n)


def serialized_canonical_code(state: int, n: int) -> bytes:
    width = (n - 1 + 7) // 8
    return (residual_code(state, n) & ((1 << (n - 1)) - 1)).to_bytes(width, "big")


def event_counts(events: Iterable[str]) -> dict[str, int]:
    counts = Counter(events)
    require(not (set(counts) - {"R", "F"}), "history", "unknown event in count ledger")
    return {"R": counts["R"], "F": counts["F"]}


def probe_rotation_bank(n: int) -> list[int]:
    probes: list[int] = []
    while len(probes) < PROBES_PER_HISTORY:
        probes.extend(range(n))
    return probes[:PROBES_PER_HISTORY]


def one_hot(index: int, size: int = PROBES_PER_HISTORY) -> list[int]:
    require(0 <= index < size, "structure", "one-hot index is outside its vector")
    return [int(position == index) for position in range(size)]


def episode_id(n: int, length_band: str, cell_episode_index: int) -> str:
    return "wgrq-train-n{:02d}-{}-{:06d}".format(
        n,
        length_band.replace("_", ""),
        cell_episode_index,
    )


def history_sha256(events: Sequence[str]) -> str:
    return sha256_bytes("".join(events).encode("ascii"))


def canonical_access_word(state: int, n: int) -> tuple[str, ...]:
    result: list[str] = []
    for index in range(n):
        if (state >> index) & 1:
            result.append("F")
        result.append("R")
    return tuple(result)


def cancellation_words(n: int) -> dict[str, tuple[str, ...]]:
    words = {
        "ff_rn_identity": tuple("FF" + "R" * n),
        "frf_rn_minus_1_nonidentity": tuple("FRF" + "R" * (n - 1)),
        "fr_order": tuple("FR"),
        "rf_order": tuple("RF"),
        "global_complement": tuple("FR" * n),
    }
    if n % 2 == 0:
        words["equal_count_identity"] = tuple("FF" * (n // 2) + "R" * n)
    return words


def generator_cancellation_report(n: int) -> dict[str, Any]:
    words = cancellation_words(n)
    states = tuple(range(1 << n))
    mask = (1 << n) - 1
    identity = words["ff_rn_identity"]
    nonidentity = words["frf_rn_minus_1_nonidentity"]
    fr = words["fr_order"]
    rf = words["rf_order"]
    complement = words["global_complement"]
    identity_ok = all(replay_history(identity, n, state) == state for state in states)
    nonidentity_ok = all(
        replay_history(nonidentity, n, state) != state for state in states
    )
    order_ok = all(
        replay_history(fr, n, state) != replay_history(rf, n, state) for state in states
    )
    complement_ok = all(
        replay_history(complement, n, state) == (state ^ mask)
        and residual_code(replay_history(complement, n, state), n)
        == residual_code(state, n)
        for state in states
    )
    same_counts = event_counts(identity) == event_counts(nonidentity)
    equal_identity: bool | None = None
    global_equal_counts: bool | None = None
    if "equal_count_identity" in words:
        equal_word = words["equal_count_identity"]
        equal_identity = all(
            replay_history(equal_word, n, state) == state for state in states
        )
        global_equal_counts = event_counts(complement) == event_counts(equal_word)
    passed = all((identity_ok, nonidentity_ok, order_ok, complement_ok, same_counts))
    if equal_identity is not None:
        passed = passed and equal_identity and bool(global_equal_counts)
    return {
        "n": n,
        "passed": bool(passed),
        "words": {name: "".join(word) for name, word in words.items()},
        "event_counts": {name: event_counts(word) for name, word in words.items()},
        "ff_rn_identity": identity_ok,
        "same_count_nonidentity": nonidentity_ok and same_counts,
        "fr_ne_rf": order_ok,
        "global_complement_observationally_null": complement_ok,
        "equal_count_identity": equal_identity,
        "global_complement_equal_counts": global_equal_counts,
    }


def symbolic_gate_audit(scales: Iterable[int] = SYMBOLIC_SCALES) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for n in scales:
        states = tuple(range(1 << n))
        mask = (1 << n) - 1
        reversible = True
        classes: dict[int, list[int]] = defaultdict(list)
        serial_to_edges: dict[bytes, set[int]] = defaultdict(set)
        for state in states:
            reversible = reversible and inverse_rotate(rotate(state, n), n) == state
            reversible = reversible and rotate(inverse_rotate(state, n), n) == state
            reversible = (
                reversible and apply_event(apply_event(state, "F", n), "F", n) == state
            )
            code = residual_code(state, n)
            classes[code].append(state)
            serial_to_edges[serialized_canonical_code(state, n)].add(code)

        determining_ok = True
        witness_depths: Counter[int] = Counter()
        determining_checks = 0
        for left in states:
            for right in states:
                equal = True
                first_depth: int | None = None
                for depth in range(n - 1):
                    same = read_after_rotations(left, depth, n) == read_after_rotations(
                        right, depth, n
                    )
                    equal = equal and same
                    determining_checks += 1
                    if not same and first_depth is None:
                        first_depth = depth
                expected_equal = right in {left, left ^ mask}
                determining_ok = determining_ok and equal == expected_equal
                if not expected_equal:
                    require(
                        first_depth is not None,
                        "symbolic",
                        "inequivalent pair lacks a witness",
                    )
                    witness_depths[int(first_depth)] += 1

        representative_independent = True
        for representatives in classes.values():
            for event in ("R", "F"):
                targets = {
                    residual_code(apply_event(state, event, n), n)
                    for state in representatives
                }
                representative_independent = (
                    representative_independent and len(targets) == 1
                )

        minimum_checks = (n - 1) * (1 << (2 * n)) + 3 * (1 << n)
        class_sizes = Counter(len(members) for members in classes.values())
        checks = {
            "physical_transitions_reversible": reversible,
            "determining_rotations_exact": determining_ok,
            "quotient_class_count": len(classes) == 1 << (n - 1),
            "quotient_classes_size_two": class_sizes == Counter({2: 1 << (n - 1)}),
            "representative_independent_transitions": representative_independent,
            "all_witness_depths_present": set(witness_depths) == set(range(n - 1)),
            "tight_witness_depth_present": witness_depths[n - 2] > 0,
            "canonical_serialization_exact": (
                len(serial_to_edges) == 1 << (n - 1)
                and all(len(edges) == 1 for edges in serial_to_edges.values())
            ),
            "minimum_check_count_met": determining_checks + 3 * len(states)
            == minimum_checks,
        }
        results[str(n)] = {
            "passed": all(checks.values()),
            "checks": checks,
            "check_count": minimum_checks,
            "minimum_check_count": minimum_checks,
            "quotient_classes": len(classes),
            "class_size": 2 if class_sizes == Counter({2: 1 << (n - 1)}) else None,
            "witness_depth_counts": {
                str(depth): witness_depths[depth] for depth in range(n - 1)
            },
            "maximum_witness_depth": max(witness_depths) if witness_depths else None,
        }
    return results


def scorer_symbolic_projection(
    symbolic: Mapping[str, Any],
    cancellation: Mapping[str, Any],
    generation_controls: Mapping[str, bool],
    *,
    passed: bool,
) -> dict[str, Any]:
    scales: dict[str, Any] = {}
    for n in SYMBOLIC_SCALES:
        detail = symbolic[str(n)]
        checks = detail["checks"]
        scales[str(n)] = {
            "check_count": detail["check_count"],
            "quotient_class_count": detail["quotient_classes"],
            "quotient_class_size": detail["class_size"],
            "maximum_shortest_witness_depth": detail["maximum_witness_depth"],
            "gates": {
                "physical_transitions_and_reversibility": checks[
                    "physical_transitions_reversible"
                ],
                "future_equivalence_exact": checks["determining_rotations_exact"],
                "quotient_cardinality_exact": checks["quotient_class_count"]
                and checks["quotient_classes_size_two"],
                "quotient_transitions_representative_independent": checks[
                    "representative_independent_transitions"
                ],
                "shortest_witness_depths_exact": checks["all_witness_depths_present"],
                "tight_maximum_witness_depth": checks["tight_witness_depth_present"],
                "canonical_serialization_exact": checks[
                    "canonical_serialization_exact"
                ],
            },
        }
    cancellation_gates = {
        "ff_rn_identity": all(item["ff_rn_identity"] for item in cancellation.values()),
        "frf_rn_minus_1_nonidentity": all(
            item["same_count_nonidentity"] for item in cancellation.values()
        ),
        "fr_noncommutes_rf": all(item["fr_ne_rf"] for item in cancellation.values()),
        "global_complement_observationally_null": all(
            item["global_complement_observationally_null"]
            for item in cancellation.values()
        ),
        "equal_count_identity": all(
            item["equal_count_identity"] is True for item in cancellation.values()
        ),
    }
    return {
        "schema": SYMBOLIC_AUDIT_SCHEMA,
        "passed": bool(
            passed
            and all(item["passed"] for item in symbolic.values())
            and all(cancellation_gates.values())
            and all(generation_controls.values())
        ),
        "scales": scales,
        "cancellation_controls": cancellation_gates,
        "generation_controls": dict(generation_controls),
    }


@dataclass(frozen=True)
class AuditContract:
    scales: tuple[int, ...] = TRAINING_SCALES
    length_bands: tuple[str, ...] = LENGTH_BANDS
    episodes_per_cell: int = EPISODES_PER_CELL
    batch_size: int = FROZEN_BATCH_SIZE

    @property
    def cells(self) -> tuple[tuple[int, str], ...]:
        return tuple((n, band) for n in self.scales for band in self.length_bands)

    @property
    def total_episodes(self) -> int:
        return len(self.cells) * self.episodes_per_cell

    @property
    def total_calls(self) -> int:
        return self.total_episodes * ORDINARY_CALLS_PER_EPISODE


@dataclass(frozen=True)
class EpisodeAudit:
    episode_id: str
    global_index: int
    cell: tuple[int, str]
    cell_episode_index: int
    witness_depth: int
    gadget_name: str
    source_lengths: tuple[int, ...]
    event_count_rows: tuple[tuple[int, int], ...]
    probe_rotations: tuple[int, ...]
    answers: tuple[tuple[int, ...], ...]
    length_matched: bool
    event_count_matched: bool
    parity_obstruction: bool
    expected_ledger: tuple[dict[str, Any], ...]


def expected_gadgets(n: int, length_band: str) -> dict[str, tuple[str, ...]]:
    if length_band == "le_2n":
        return {
            "canonical_access": (),
            "ff_local_identity": tuple("FF"),
            "fr_order": tuple("FR"),
            "rf_order": tuple("RF"),
        }
    require(length_band == "le_8n", "strata", "unknown source-length band")
    return cancellation_words(n)


def feasible_gadget_names(n: int, length_band: str, depth: int) -> tuple[str, ...]:
    capacity = LENGTH_MULTIPLIERS[length_band] * n
    base_ceiling = n + n // 2 + int(depth == n - 2)
    return tuple(
        name
        for name, word in expected_gadgets(n, length_band).items()
        if base_ceiling + len(word) <= capacity
    )


def _final_difference_for_depth(n: int, depth: int) -> int:
    if depth == n - 2:
        return 1 << (n - 1)
    return (1 << (depth + 1)) | (1 << (depth + 2))


def _inverse_rotate_by(state: int, rotations: int, n: int) -> int:
    for _ in range(rotations % n):
        state = inverse_rotate(state, n)
    return state


def _validate_event_list(value: Any, name: str) -> tuple[str, ...]:
    require(isinstance(value, list), "structure", "{} must be a list".format(name))
    events = tuple(value)
    require(
        all(isinstance(event, str) and event in {"R", "F"} for event in events),
        "history",
        "{} contains a non-event token".format(name),
    )
    return events


def _padding_matches_plan(
    events: tuple[str, ...], n: int, ff_blocks: int, rn_blocks: int
) -> bool:
    memo: dict[tuple[int, int, int], bool] = {}

    def visit(position: int, ff_left: int, rn_left: int) -> bool:
        key = (position, ff_left, rn_left)
        if key in memo:
            return memo[key]
        if position == len(events):
            result = ff_left == 0 and rn_left == 0
        else:
            result = False
            if ff_left and events[position : position + 2] == ("F", "F"):
                result = visit(position + 2, ff_left - 1, rn_left)
            if not result and rn_left and events[position : position + n] == ("R",) * n:
                result = visit(position + n, ff_left, rn_left - 1)
        memo[key] = result
        return result

    return visit(0, ff_blocks, rn_blocks)


def _derive_relation(
    left: Sequence[int], right: Sequence[int], n: int
) -> tuple[int, int | None, list[int]]:
    differences = [left[index] ^ right[index] for index in range(n - 1)]
    if not any(differences):
        return 1, None, [0] * PROBES_PER_HISTORY
    depth = differences.index(1)
    return 0, depth, one_hot(depth)


def audit_episode(
    value: Any,
    *,
    expected_global_index: int,
    contract: AuditContract = AuditContract(),
) -> EpisodeAudit:
    require(isinstance(value, dict), "structure", "episode must be an object")
    _scan_forbidden_public_fields(value, "episode")
    episode = exact_fields(value, EPISODE_FIELDS, "episode")
    require(
        episode["schema"] == TRANSCRIPT_SCHEMA, "structure", "transcript schema differs"
    )
    require(
        episode["split"] == "training", "structure", "episode split is not training"
    )
    global_index = exact_int(
        episode["global_episode_index"], "global_episode_index", minimum=0
    )
    require(
        global_index == expected_global_index,
        "order",
        "global episode index is discontinuous",
    )
    cell_number, cell_episode_index = divmod(global_index, contract.episodes_per_cell)
    require(
        cell_number < len(contract.cells), "strata", "episode exceeds the frozen cells"
    )
    expected_cell = contract.cells[cell_number]
    n, length_band = expected_cell

    cell = exact_fields(episode["cell"], CELL_FIELDS, "episode.cell")
    require(
        cell["n"] == n and cell["length_band"] == length_band,
        "strata",
        "cell order or identity differs",
    )
    require(
        cell["source_length_ceiling"] == LENGTH_MULTIPLIERS[length_band] * n,
        "strata",
        "source-length ceiling differs",
    )
    require(
        cell["cell_episode_index"] == cell_episode_index,
        "order",
        "cell episode index differs",
    )
    expected_episode_id = episode_id(n, length_band, cell_episode_index)
    require(
        episode["episode_id"] == expected_episode_id,
        "identity",
        "episode ID is not canonical",
    )
    require(
        episode["batch_index"] == global_index // contract.batch_size
        and episode["batch_offset"] == global_index % contract.batch_size,
        "batching",
        "frozen batch coordinates differ",
    )
    require(
        episode["event_code_bits"] == EVENT_CODE_BITS,
        "event_code",
        "public event code differs",
    )

    probes = episode["probe_rotations"]
    expected_probes = probe_rotation_bank(n)
    require(probes == expected_probes, "probe_bank", "eight-probe bank differs")

    gadget = exact_fields(episode["gadget"], GADGET_FIELDS, "episode.gadget")
    gadget_name = gadget["name"]
    allowed_gadgets = expected_gadgets(n, length_band)
    require(
        isinstance(gadget_name, str) and gadget_name in allowed_gadgets,
        "gadget",
        "unknown gadget",
    )
    gadget_events = _validate_event_list(gadget["events"], "gadget.events")
    require(
        gadget_events == allowed_gadgets[gadget_name], "gadget", "gadget events differ"
    )
    require(
        gadget["event_counts"] == event_counts(gadget_events),
        "gadget",
        "gadget event counts differ",
    )
    padding = _validate_event_list(
        gadget["common_identity_padding_events"],
        "gadget.common_identity_padding_events",
    )
    padding_plan = exact_fields(
        gadget["padding_plan"], PADDING_PLAN_FIELDS, "gadget.padding_plan"
    )
    ff_blocks = exact_int(padding_plan["ff_blocks"], "padding ff_blocks", minimum=0)
    rn_blocks = exact_int(padding_plan["rn_blocks"], "padding rn_blocks", minimum=0)
    require(
        _padding_matches_plan(padding, n, ff_blocks, rn_blocks),
        "gadget",
        "identity padding does not match its FF/R^n plan",
    )
    require(
        replay_history(padding, n) == 0,
        "gadget",
        "declared identity padding is nonidentity",
    )
    common_suffix = gadget_events + padding

    histories_value = episode["histories"]
    require(
        isinstance(histories_value, list)
        and len(histories_value) == HISTORIES_PER_EPISODE,
        "structure",
        "episode must contain exactly four histories",
    )
    source_words: list[tuple[str, ...]] = []
    source_lengths: list[int] = []
    count_rows: list[tuple[int, int]] = []
    answers_by_history: list[tuple[int, ...]] = []
    endpoints: list[int] = []
    access_endpoints: list[int] = []
    expected_ledger: list[dict[str, Any]] = []
    for history_index, history_value in enumerate(histories_value):
        history = exact_fields(history_value, HISTORY_FIELDS, "history")
        require(
            history["history_index"] == history_index, "order", "history index differs"
        )
        require(
            history["history_id"]
            == "{}:h{}".format(expected_episode_id, history_index),
            "identity",
            "history ID differs",
        )
        require(
            history["role"] == HISTORY_ROLES[history_index],
            "order",
            "history role order differs",
        )
        events = _validate_event_list(history["events"], "history.events")
        require(
            len(events) <= int(cell["source_length_ceiling"]),
            "length_band",
            "source history exceeds its ceiling",
        )
        require(
            history["source_length"] == len(events),
            "history",
            "source length ledger differs",
        )
        counts = event_counts(events)
        require(
            history["event_counts"] == counts, "history", "source event counts differ"
        )
        digest = history_sha256(events)
        require(
            history["history_sha256"] == digest, "hash", "source history hash differs"
        )
        if common_suffix:
            require(
                events[-len(common_suffix) :] == common_suffix,
                "gadget",
                "source lacks the declared common suffix",
            )
            access_word = events[: -len(common_suffix)]
        else:
            access_word = events
        access_endpoint = replay_history(access_word, n)
        require(
            access_word == canonical_access_word(access_endpoint, n),
            "history",
            "source prefix is not the canonical access word",
        )
        endpoint = replay_history(events, n)

        history_probes = history["probes"]
        require(
            isinstance(history_probes, list)
            and len(history_probes) == PROBES_PER_HISTORY,
            "structure",
            "history must contain exactly eight probes",
        )
        answer_row: list[int] = []
        for probe_index, probe_value in enumerate(history_probes):
            probe = exact_fields(probe_value, PROBE_FIELDS, "history.probe")
            rotations = expected_probes[probe_index]
            call_id = (
                global_index * ORDINARY_CALLS_PER_EPISODE
                + history_index * PROBES_PER_HISTORY
                + probe_index
            )
            expected_answer = read_after_rotations(endpoint, rotations, n)
            require(probe["probe_index"] == probe_index, "probe", "probe index differs")
            require(
                probe["continuation_rotations"] == rotations,
                "probe",
                "probe rotation count differs",
            )
            require(
                probe["continuation"] == ["R"] * rotations,
                "probe",
                "probe continuation differs",
            )
            require(
                exact_bit(probe["answer"], "probe answer") == expected_answer,
                "ordinary_answers",
                "ordinary answer does not replay",
            )
            require(
                probe["oracle_call_id"] == call_id,
                "call_ledger",
                "probe call ID differs",
            )
            answer_row.append(expected_answer)
            expected_ledger.append(
                {
                    "schema": LEDGER_SCHEMA,
                    "call_id": call_id,
                    "episode_id": expected_episode_id,
                    "global_episode_index": global_index,
                    "history_index": history_index,
                    "history_sha256": digest,
                    "probe_index": probe_index,
                    "continuation_rotations": rotations,
                    "call_kind": "READ",
                    "returned_bits": 1,
                    "answer": expected_answer,
                }
            )
        require(
            history["canonical_edge_bits_from_public_answers"] == answer_row[: n - 1],
            "relation",
            "canonical edge bits are not derived from ordinary answers",
        )
        source_words.append(events)
        source_lengths.append(len(events))
        count_rows.append((counts["R"], counts["F"]))
        answers_by_history.append(tuple(answer_row))
        endpoints.append(endpoint)
        access_endpoints.append(access_endpoint)

    require(
        len(set(source_words)) == 4, "duplicate", "episode repeats a source history"
    )
    require(
        len({history_sha256(word) for word in source_words}) == 4,
        "duplicate",
        "episode repeats a source hash",
    )
    if length_band == "le_8n":
        require(
            max(source_lengths) > 2 * n,
            "overlap",
            "long-band episode remains inside the short band",
        )

    signatures = [answers[: n - 1] for answers in answers_by_history]
    expected_matrix = [
        [int(left == right) for right in signatures] for left in signatures
    ]
    require(
        episode["equivalence_label_matrix"] == expected_matrix,
        "relation",
        "equivalence matrix is not answer-derived",
    )
    require(
        expected_matrix[0][1] == 1 and expected_matrix[2][3] == 0,
        "relation",
        "declared pair roles differ",
    )
    require(
        residual_code(endpoints[0], n) == residual_code(endpoints[1], n)
        and residual_code(endpoints[2], n) != residual_code(endpoints[3], n),
        "relation",
        "public answer relations disagree with replayed futures",
    )

    pairs = exact_fields(episode["pairs"], PAIRS_FIELDS, "episode.pairs")
    equivalent_pair = exact_fields(
        pairs["equivalent"], EQUIVALENT_PAIR_FIELDS, "equivalent pair"
    )
    non_equivalent_pair = exact_fields(
        pairs["non_equivalent"],
        NON_EQUIVALENT_PAIR_FIELDS,
        "non-equivalent pair",
    )
    require(
        equivalent_pair == {"history_indices": [0, 1], "label": 1},
        "relation",
        "equivalent pair declaration differs",
    )
    label, depth, witness_mask = _derive_relation(
        answers_by_history[2], answers_by_history[3], n
    )
    require(
        label == 0 and depth is not None,
        "witness",
        "non-equivalent pair lacks a public witness",
    )
    differing_probe_indices = [
        index
        for index in range(PROBES_PER_HISTORY)
        if answers_by_history[2][index] != answers_by_history[3][index]
    ]
    require(
        differing_probe_indices and differing_probe_indices[0] == depth,
        "witness",
        "first probe is not the shortest witness",
    )
    require(
        non_equivalent_pair
        == {
            "history_indices": [2, 3],
            "label": 0,
            "shortest_witness_depth": depth,
            "first_distinguishing_probe_index": depth,
            "first_distinguishing_witness_mask": witness_mask,
        },
        "witness",
        "non-equivalent pair metadata is not answer-derived",
    )

    half = n // 2
    state_mask = (1 << n) - 1
    require(
        len(set(access_endpoints)) == 4,
        "duplicate",
        "endpoint bank repeats a pre-gadget endpoint",
    )
    require(
        access_endpoints[1] == (access_endpoints[0] ^ state_mask)
        and access_endpoints[0] < access_endpoints[1],
        "endpoint_strata",
        "equivalent pre-gadget endpoints are not the canonical complement pair",
    )
    require(
        all(access_endpoints[index].bit_count() == half for index in (0, 1, 2)),
        "endpoint_strata",
        "balanced endpoint roles left the half-weight stratum",
    )
    gadget_rotations = event_counts(gadget_events)["R"] % n
    expected_base_difference = _inverse_rotate_by(
        _final_difference_for_depth(n, depth),
        gadget_rotations,
        n,
    )
    require(
        access_endpoints[2] ^ access_endpoints[3] == expected_base_difference,
        "endpoint_strata",
        "non-equivalent endpoint difference differs from its depth stratum",
    )
    if depth < n - 2:
        require(
            access_endpoints[3].bit_count() == half
            and access_endpoints[2] < access_endpoints[3],
            "endpoint_strata",
            "balanceable non-equivalent endpoints left their canonical stratum",
        )
    else:
        require(
            abs(access_endpoints[3].bit_count() - half) == 1,
            "endpoint_strata",
            "tight-depth endpoint does not realize the parity obstruction",
        )
    require(
        episode["first_distinguishing_witness_mask"] == witness_mask,
        "witness",
        "top-level witness mask differs",
    )
    uniform_index = exact_int(episode["uniform_probe_index"], "uniform_probe_index")
    require(
        0 <= uniform_index < PROBES_PER_HISTORY, "probe", "uniform probe index differs"
    )
    require(
        episode["uniform_probe_mask"] == one_hot(uniform_index),
        "probe",
        "uniform probe mask differs",
    )

    lengths_matched = len(set(source_lengths)) == 1
    counts_matched = len(set(count_rows)) == 1
    obstruction = depth == n - 2
    balance = exact_fields(
        episode["balance"], EPISODE_BALANCE_FIELDS, "episode.balance"
    )
    require(
        balance["declared_pair_labels"] == {"equivalent": 1, "non_equivalent": 1},
        "balance",
        "pair-label ledger differs",
    )
    require(
        balance["all_source_lengths_matched"] is lengths_matched,
        "balance",
        "length balance flag differs",
    )
    require(
        balance["all_event_counts_matched"] is counts_matched,
        "balance",
        "event-count balance flag differs",
    )
    require(
        balance["maximum_depth_flip_parity_obstruction"] is obstruction,
        "balance",
        "parity obstruction flag differs",
    )
    if obstruction:
        require(
            count_rows[0] == count_rows[1] == count_rows[2],
            "cancellation_balance",
            "avoidable max-depth count imbalance",
        )
        require(
            count_rows[2][0] == count_rows[3][0],
            "cancellation_balance",
            "max-depth rotations differ",
        )
        require(
            abs(count_rows[2][1] - count_rows[3][1]) == 1,
            "cancellation_balance",
            "max-depth F mismatch is not minimal",
        )
        require(
            abs(source_lengths[2] - source_lengths[3]) == 1,
            "cancellation_balance",
            "max-depth length mismatch is not minimal",
        )
        require(
            not counts_matched and not lengths_matched,
            "cancellation_balance",
            "max-depth parity obstruction disappeared",
        )
    else:
        require(
            counts_matched and lengths_matched,
            "cancellation_balance",
            "balanceable episode is count-confounded",
        )

    span = exact_fields(
        episode["oracle_call_span"], CALL_SPAN_FIELDS, "episode.oracle_call_span"
    )
    first_call = global_index * ORDINARY_CALLS_PER_EPISODE
    require(
        span
        == {
            "first_call_id": first_call,
            "last_call_id": first_call + ORDINARY_CALLS_PER_EPISODE - 1,
            "ordinary_one_bit_read_calls": ORDINARY_CALLS_PER_EPISODE,
        },
        "call_ledger",
        "episode call span differs",
    )
    require(
        gadget_name in feasible_gadget_names(n, length_band, depth),
        "gadget",
        "gadget is infeasible for its cell/depth",
    )

    return EpisodeAudit(
        episode_id=expected_episode_id,
        global_index=global_index,
        cell=expected_cell,
        cell_episode_index=cell_episode_index,
        witness_depth=depth,
        gadget_name=gadget_name,
        source_lengths=tuple(source_lengths),
        event_count_rows=tuple(count_rows),
        probe_rotations=tuple(expected_probes),
        answers=tuple(answers_by_history),
        length_matched=lengths_matched,
        event_count_matched=counts_matched,
        parity_obstruction=obstruction,
        expected_ledger=tuple(expected_ledger),
    )


def audit_ledger_row(value: Any, expected: Mapping[str, Any]) -> None:
    require(
        isinstance(value, dict),
        "structure",
        "ordinary call ledger row must be an object",
    )
    _scan_forbidden_public_fields(value, "ordinary_call_ledger")
    row = exact_fields(value, LEDGER_FIELDS, "ordinary call ledger row")
    require(
        row == expected,
        "call_ledger",
        "ordinary call ledger row differs from replayed probe",
    )


class StrictJsonlReader:
    """Streaming canonical JSONL reader with stable regular-file checks."""

    def __init__(self, path: str | Path, label: str):
        self.path = Path(path)
        self.label = label
        self.rows = 0
        self.byte_count = 0
        self.digest = hashlib.sha256()
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(self.path, flags)
        except OSError as error:
            raise AuditError(
                "artifact", "cannot open {}: {}".format(label, error)
            ) from error
        self._before = os.fstat(descriptor)
        if not stat.S_ISREG(self._before.st_mode):
            os.close(descriptor)
            raise AuditError("artifact", "{} is not a regular file".format(label))
        self._file = os.fdopen(descriptor, "rb")

    def read(self) -> Any | None:
        raw = self._file.readline()
        if not raw:
            return None
        self.rows += 1
        self.byte_count += len(raw)
        self.digest.update(raw)
        require(
            raw.endswith(b"\n") and raw != b"\n",
            "partial",
            "{} row {} is blank or unterminated".format(self.label, self.rows),
        )
        try:
            text = raw[:-1].decode("ascii")
        except UnicodeDecodeError as error:
            raise AuditError(
                "encoding", "{} row {} is not ASCII".format(self.label, self.rows)
            ) from error
        value = strict_json_loads(text)
        require(
            raw == canonical_jsonl_record(value),
            "canonical_bytes",
            "{} row {} is not canonical JSONL".format(self.label, self.rows),
        )
        return value

    @property
    def sha256(self) -> str:
        return self.digest.hexdigest()

    def close(self) -> None:
        if self._file.closed:
            return
        after = os.fstat(self._file.fileno())
        stable_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        changed = any(
            getattr(self._before, name) != getattr(after, name)
            for name in stable_fields
        )
        self._file.close()
        require(not changed, "artifact", "{} changed during audit".format(self.label))

    def __enter__(self) -> "StrictJsonlReader":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


def read_regular_bytes(path: str | Path, label: str) -> bytes:
    candidate = Path(path)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(candidate, flags)
    except OSError as error:
        raise AuditError(
            "artifact", "cannot open {}: {}".format(label, error)
        ) from error
    try:
        before = os.fstat(descriptor)
        require(
            stat.S_ISREG(before.st_mode),
            "artifact",
            "{} is not a regular file".format(label),
        )
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        stable_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        require(
            not any(
                getattr(before, name) != getattr(after, name) for name in stable_fields
            ),
            "artifact",
            "{} changed during audit".format(label),
        )
        return b"".join(chunks)
    finally:
        os.close(descriptor)


@dataclass
class CellStats:
    n: int
    length_band: str
    episodes: int = 0
    calls: int = 0
    witness_depths: Counter[int] = field(default_factory=Counter)
    gadgets: Counter[str] = field(default_factory=Counter)
    gadgets_by_depth: dict[int, Counter[str]] = field(
        default_factory=lambda: defaultdict(Counter)
    )
    source_lengths: Counter[int] = field(default_factory=Counter)
    probe_rotations: Counter[int] = field(default_factory=Counter)
    answers: Counter[int] = field(default_factory=Counter)
    event_count_matched: int = 0
    length_matched: int = 0
    parity_obstructions: int = 0

    def add(self, episode: EpisodeAudit) -> None:
        self.episodes += 1
        self.calls += ORDINARY_CALLS_PER_EPISODE
        self.witness_depths[episode.witness_depth] += 1
        self.gadgets[episode.gadget_name] += 1
        self.gadgets_by_depth[episode.witness_depth][episode.gadget_name] += 1
        self.source_lengths.update(episode.source_lengths)
        for rotation in episode.probe_rotations:
            self.probe_rotations[rotation] += HISTORIES_PER_EPISODE
        for answer_row in episode.answers:
            self.answers.update(answer_row)
        self.event_count_matched += int(episode.event_count_matched)
        self.length_matched += int(episode.length_matched)
        self.parity_obstructions += int(episode.parity_obstruction)

    def validate_balance(self, expected_episodes: int) -> None:
        require(
            self.episodes == expected_episodes, "partial", "cell episode count differs"
        )
        depth_counts = [self.witness_depths[depth] for depth in range(self.n - 1)]
        require(
            set(self.witness_depths) == set(range(self.n - 1)),
            "strata",
            "cell misses a witness depth",
        )
        require(
            max(depth_counts) - min(depth_counts) <= 1,
            "strata",
            "witness depths are not floor/ceiling balanced",
        )
        for depth in range(self.n - 1):
            names = feasible_gadget_names(self.n, self.length_band, depth)
            counts = self.gadgets_by_depth[depth]
            require(
                set(counts) == set(names),
                "strata",
                "depth misses a feasible cancellation gadget",
            )
            require(
                max(counts.values()) - min(counts.values()) <= 1,
                "strata",
                "gadgets are not balanced within witness depth",
            )
        require(
            self.parity_obstructions == self.witness_depths[self.n - 2],
            "cancellation_balance",
            "parity obstruction count differs from the tight-depth stratum",
        )

    def report_record(self) -> dict[str, Any]:
        require(bool(self.source_lengths), "partial", "cell has no source histories")
        return {
            "n": self.n,
            "length_band": self.length_band,
            "source_length_ceiling": LENGTH_MULTIPLIERS[self.length_band] * self.n,
            "episodes": self.episodes,
            "histories": self.episodes * HISTORIES_PER_EPISODE,
            "ordinary_one_bit_read_calls": self.calls,
            "pair_labels": {
                "equivalent": self.episodes,
                "non_equivalent": self.episodes,
            },
            "witness_depth_counts": {
                str(depth): self.witness_depths[depth]
                for depth in sorted(self.witness_depths)
            },
            "witness_depth_division_remainder": self.episodes % (self.n - 1),
            "gadget_counts": {
                name: self.gadgets[name] for name in sorted(self.gadgets)
            },
            "gadget_pair_label_counts": {
                name: {
                    "equivalent": self.gadgets[name],
                    "non_equivalent": self.gadgets[name],
                }
                for name in sorted(self.gadgets)
            },
            "source_length_counts": {
                str(length): self.source_lengths[length]
                for length in sorted(self.source_lengths)
            },
            "minimum_source_length": min(self.source_lengths),
            "maximum_source_length": max(self.source_lengths),
            "probe_rotation_call_counts": {
                str(rotation): self.probe_rotations[rotation]
                for rotation in sorted(self.probe_rotations)
            },
            "answer_counts": {
                str(answer): self.answers[answer] for answer in sorted(self.answers)
            },
            "event_count_matched_episodes": self.event_count_matched,
            "length_matched_episodes": self.length_matched,
            "parity_obstruction_episodes": self.parity_obstructions,
        }


def _validate_symbolic_generation_report(
    value: Any, independent: Mapping[str, Any]
) -> None:
    fields = {
        "schema",
        "passed",
        "scales",
        "total_minimum_check_count",
        "total_core_check_count",
        "total_check_count",
    }
    report = exact_fields(value, fields, "generation symbolic gates")
    require(
        report["schema"] == "dwepr_stage_a_symbolic_gates_v1",
        "symbolic",
        "symbolic report schema differs",
    )
    require(
        report["passed"] is True, "symbolic", "generator symbolic gates did not pass"
    )
    scales = report["scales"]
    require(
        isinstance(scales, list) and len(scales) == 2,
        "symbolic",
        "symbolic scale list differs",
    )
    total_minimum = 0
    total_core = 0
    total_checks = 0
    scale_fields = {
        "schema",
        "n",
        "passed",
        "physical_states",
        "residual_classes",
        "class_size",
        "determining_rotations",
        "shortest_witness_depth_counts",
        "maximum_shortest_witness_depth",
        "minimum_check_count",
        "core_check_count",
        "check_count",
        "cancellation_controls",
    }
    for position, n in enumerate(SYMBOLIC_SCALES):
        scale = exact_fields(scales[position], scale_fields, "symbolic scale")
        detail = independent[str(n)]
        require(
            scale["schema"] == "dwepr_symbolic_gate_v1"
            and scale["n"] == n
            and scale["passed"] is True,
            "symbolic",
            "symbolic scale identity differs",
        )
        require(
            scale["physical_states"] == 1 << n,
            "symbolic",
            "physical state count differs",
        )
        require(
            scale["residual_classes"] == detail["quotient_classes"]
            and scale["class_size"] == 2,
            "symbolic",
            "quotient cardinality differs",
        )
        require(
            scale["determining_rotations"] == list(range(n - 1)),
            "symbolic",
            "determining rotations differ",
        )
        require(
            scale["shortest_witness_depth_counts"] == detail["witness_depth_counts"],
            "symbolic",
            "witness depth counts differ",
        )
        require(
            scale["maximum_shortest_witness_depth"] == n - 2,
            "symbolic",
            "maximum witness depth differs",
        )
        require(
            scale["minimum_check_count"] == detail["minimum_check_count"],
            "symbolic",
            "minimum check count differs",
        )
        require(
            scale["core_check_count"] == detail["minimum_check_count"],
            "symbolic",
            "core check ledger differs",
        )
        check_count = exact_int(scale["check_count"], "symbolic check_count")
        require(
            check_count >= detail["minimum_check_count"],
            "symbolic",
            "symbolic check count is too small",
        )
        require(
            scale["cancellation_controls"] == generator_cancellation_report(n),
            "symbolic",
            "symbolic cancellation report differs",
        )
        total_minimum += int(scale["minimum_check_count"])
        total_core += int(scale["core_check_count"])
        total_checks += check_count
    require(
        report["total_minimum_check_count"] == total_minimum,
        "symbolic",
        "total minimum check count differs",
    )
    require(
        report["total_core_check_count"] == total_core,
        "symbolic",
        "total core check count differs",
    )
    require(
        report["total_check_count"] == total_checks,
        "symbolic",
        "total symbolic check count differs",
    )


def expected_generation_contract(
    symbolic_report: Mapping[str, Any],
    contract: AuditContract,
) -> dict[str, Any]:
    return {
        "schema": TRANSCRIPT_SCHEMA,
        "preregistration": "R12_WGRQ_CPU_PREREG.md",
        "prf": {
            "formula": PRF_FORMULA,
            "seed_ascii": FROZEN_PRF_SEED_ASCII,
            "seed_hex": FROZEN_PRF_SEED_HEX,
            "selection": PRF_SELECTION,
        },
        "training_scales": list(contract.scales),
        "length_bands": [
            {"name": band, "maximum_n_multiple": LENGTH_MULTIPLIERS[band]}
            for band in contract.length_bands
        ],
        "episodes_per_cell": contract.episodes_per_cell,
        "total_episodes": contract.total_episodes,
        "histories_per_episode": HISTORIES_PER_EPISODE,
        "probes_per_history": PROBES_PER_HISTORY,
        "ordinary_calls_per_episode": ORDINARY_CALLS_PER_EPISODE,
        "total_ordinary_one_bit_read_calls": contract.total_calls,
        "probe_rotation_banks": {
            str(n): probe_rotation_bank(n) for n in contract.scales
        },
        "history_role_order": list(HISTORY_ROLES),
        "batch_size": contract.batch_size,
        "paired_initialization_order_seeds": list(FROZEN_PAIRED_SEEDS),
        "cancellation_controls": {
            str(n): generator_cancellation_report(n) for n in contract.scales
        },
        "symbolic_gate_schema": symbolic_report["schema"],
        "symbolic_gate_scales": [scale["n"] for scale in symbolic_report["scales"]],
        "no_model_dependent_mining": True,
        "no_target_dependent_rejection": True,
        "no_reseeding_or_seed_search": True,
    }


def validate_generation_report(
    value: Any,
    *,
    transcript_rows: int,
    transcript_bytes: int,
    transcript_sha256: str,
    ledger_rows: int,
    ledger_bytes: int,
    ledger_sha256: str,
    cells: Sequence[CellStats],
    independent_symbolic: Mapping[str, Any],
    contract: AuditContract,
) -> dict[str, Any]:
    report = exact_fields(value, GENERATION_REPORT_FIELDS, "generation report")
    require(
        report["schema"] == GENERATION_REPORT_SCHEMA,
        "report",
        "generation report schema differs",
    )
    require(report["passed"] is True, "report", "generation report is not passing")
    _validate_symbolic_generation_report(report["symbolic_gates"], independent_symbolic)
    expected_contract = expected_generation_contract(report["symbolic_gates"], contract)
    require(
        report["generation_contract"] == expected_contract,
        "report",
        "generation contract differs",
    )

    expected_cells = [stats.report_record() for stats in cells]
    require(
        report["cells"] == expected_cells,
        "strata",
        "generation report cell summaries differ",
    )
    expected_totals = {
        "cells": len(contract.cells),
        "episodes": contract.total_episodes,
        "histories": contract.total_episodes * HISTORIES_PER_EPISODE,
        "ordinary_one_bit_read_calls": contract.total_calls,
        "returned_answer_bits": contract.total_calls,
        "batches": contract.total_episodes // contract.batch_size,
    }
    require(report["totals"] == expected_totals, "counts", "generation totals differ")
    expected_call_ledger = {
        "schema": LEDGER_SCHEMA,
        "first_call_id": 0,
        "last_call_id": contract.total_calls - 1,
        "rows": contract.total_calls,
        "one_bit_read_calls": contract.total_calls,
        "model_dependent_calls": 0,
        "equivalence_oracle_calls": 0,
        "counterexample_oracle_calls": 0,
    }
    require(
        report["frozen_call_ledger"] == expected_call_ledger,
        "call_ledger",
        "frozen call ledger summary differs",
    )
    parity_episodes = sum(stats.parity_obstructions for stats in cells)
    expected_balance = {
        "pair_labels_per_episode": {"equivalent": 1, "non_equivalent": 1},
        "depth_stratification_rule": "floor/ceiling balanced then fixed-PRF shuffled",
        "gadget_stratification_rule": "balanced within depth over every feasible gadget",
        "maximum_depth_flip_parity_obstruction": {
            "affected_episodes": parity_episodes,
            "reported_separately": True,
            "reason": PARITY_REASON,
        },
    }
    require(
        report["balance"] == expected_balance,
        "balance",
        "generation balance report differs",
    )

    prf = exact_fields(
        report["prf_ledger"],
        {
            "formula",
            "seed_ascii",
            "seed_hex",
            "blocks_used",
            "rejected_blocks",
            "domain_count",
            "counter_ledger_sha256",
        },
        "prf ledger",
    )
    require(prf["formula"] == PRF_FORMULA, "prf", "PRF formula differs")
    require(
        prf["seed_ascii"] == FROZEN_PRF_SEED_ASCII
        and prf["seed_hex"] == FROZEN_PRF_SEED_HEX,
        "prf",
        "PRF seed differs",
    )
    require(
        exact_int(prf["blocks_used"], "PRF blocks", minimum=1)
        >= exact_int(prf["domain_count"], "PRF domains", minimum=1),
        "prf",
        "PRF block/domain ledger differs",
    )
    require(
        exact_int(prf["rejected_blocks"], "PRF rejected blocks", minimum=0)
        <= prf["blocks_used"],
        "prf",
        "PRF rejection ledger differs",
    )
    counter_hash = prf["counter_ledger_sha256"]
    require(
        isinstance(counter_hash, str)
        and len(counter_hash) == 64
        and all(char in "0123456789abcdef" for char in counter_hash),
        "hash",
        "PRF counter ledger hash differs",
    )

    expected_artifacts = {
        "transcript": {
            "schema": TRANSCRIPT_SCHEMA,
            "rows": transcript_rows,
            "bytes": transcript_bytes,
            "sha256": transcript_sha256,
        },
        "ordinary_call_ledger": {
            "schema": LEDGER_SCHEMA,
            "rows": ledger_rows,
            "bytes": ledger_bytes,
            "sha256": ledger_sha256,
        },
    }
    require(
        report["artifacts"] == expected_artifacts,
        "hash",
        "artifact hash/size bindings differ",
    )
    expected_hashes = {
        "generation_contract_sha256": sha256_bytes(
            canonical_jsonl_record(expected_contract)
        ),
        "transcript_sha256": transcript_sha256,
        "ordinary_call_ledger_sha256": ledger_sha256,
    }
    require(report["hashes"] == expected_hashes, "hash", "generation hashes differ")
    return {
        "schema": GENERATION_REPORT_SCHEMA,
        "passed": True,
        "generation_contract_sha256": expected_hashes["generation_contract_sha256"],
    }


def _failure_report(error: Exception) -> dict[str, Any]:
    symbolic = symbolic_gate_audit()
    cancellation = {str(n): generator_cancellation_report(n) for n in TRAINING_SCALES}
    generation_controls = {
        "labels_balanced_within_declared_strata_where_possible": False,
        "unavoidable_parity_obstruction_reported": False,
    }
    projection = scorer_symbolic_projection(
        symbolic,
        cancellation,
        generation_controls,
        passed=False,
    )
    category = error.category if isinstance(error, AuditError) else "fatal"
    return {
        "audit": AUDIT_NAME,
        **projection,
        "all_checks_pass": False,
        "checks": {"independent_admission": False},
        "errors": [{"category": category, "message": str(error)}],
        "claim_boundary": CLAIM_BOUNDARY,
    }


def audit_bundle(
    *,
    transcript_path: str | Path,
    ledger_path: str | Path,
    generation_report_path: str | Path,
    contract: AuditContract = AuditContract(),
) -> dict[str, Any]:
    try:
        require(
            contract.scales == TRAINING_SCALES,
            "contract",
            "training scales are not the frozen set",
        )
        require(
            contract.length_bands == LENGTH_BANDS,
            "contract",
            "length bands are not the frozen set",
        )
        require(
            contract.episodes_per_cell > 0, "contract", "cell size must be positive"
        )
        require(
            contract.batch_size == FROZEN_BATCH_SIZE, "contract", "batch size differs"
        )
        require(
            contract.total_episodes % contract.batch_size == 0,
            "contract",
            "episode count does not fill complete batches",
        )

        cell_stats = {
            cell: CellStats(n=cell[0], length_band=cell[1]) for cell in contract.cells
        }
        episode_ids: set[str] = set()
        with (
            StrictJsonlReader(transcript_path, "training transcript") as transcript,
            StrictJsonlReader(
                ledger_path,
                "ordinary call ledger",
            ) as ledger,
        ):
            global_index = 0
            while True:
                episode_value = transcript.read()
                if episode_value is None:
                    break
                audit = audit_episode(
                    episode_value,
                    expected_global_index=global_index,
                    contract=contract,
                )
                require(
                    audit.episode_id not in episode_ids,
                    "duplicate",
                    "duplicate episode ID",
                )
                episode_ids.add(audit.episode_id)
                for expected_call in audit.expected_ledger:
                    call_value = ledger.read()
                    require(
                        call_value is not None,
                        "partial",
                        "ordinary call ledger ended inside an episode",
                    )
                    audit_ledger_row(call_value, expected_call)
                cell_stats[audit.cell].add(audit)
                global_index += 1
            require(
                ledger.read() is None,
                "overlap",
                "ordinary call ledger has trailing rows",
            )
            require(
                transcript.rows == contract.total_episodes,
                "partial",
                "training transcript episode count differs",
            )
            require(
                ledger.rows == contract.total_calls,
                "partial",
                "ordinary call ledger row count differs",
            )
            for stats in cell_stats.values():
                stats.validate_balance(contract.episodes_per_cell)
            transcript_rows = transcript.rows
            transcript_bytes = transcript.byte_count
            transcript_sha = transcript.sha256
            ledger_rows = ledger.rows
            ledger_bytes = ledger.byte_count
            ledger_sha = ledger.sha256

        report_bytes = read_regular_bytes(generation_report_path, "generation report")
        require(
            report_bytes.endswith(b"\n"),
            "partial",
            "generation report lacks final newline",
        )
        try:
            report_value = strict_json_loads(report_bytes.decode("ascii"))
        except UnicodeDecodeError as error:
            raise AuditError("encoding", "generation report is not ASCII") from error
        require(
            report_bytes == pretty_json_bytes(report_value),
            "canonical_bytes",
            "generation report is not canonical pretty JSON",
        )
        independent_symbolic = symbolic_gate_audit()
        ordered_stats = [cell_stats[cell] for cell in contract.cells]
        report_binding = validate_generation_report(
            report_value,
            transcript_rows=transcript_rows,
            transcript_bytes=transcript_bytes,
            transcript_sha256=transcript_sha,
            ledger_rows=ledger_rows,
            ledger_bytes=ledger_bytes,
            ledger_sha256=ledger_sha,
            cells=ordered_stats,
            independent_symbolic=independent_symbolic,
            contract=contract,
        )
        cancellation = {
            str(n): generator_cancellation_report(n) for n in contract.scales
        }
        generation_controls = {
            "labels_balanced_within_declared_strata_where_possible": True,
            "unavoidable_parity_obstruction_reported": True,
        }
        projection = scorer_symbolic_projection(
            independent_symbolic,
            cancellation,
            generation_controls,
            passed=True,
        )
        safe_cells = {
            "n={}|band={}".format(stats.n, stats.length_band): {
                "episodes": stats.episodes,
                "ordinary_one_bit_read_calls": stats.calls,
                "witness_depth_counts": {
                    str(depth): stats.witness_depths[depth]
                    for depth in range(stats.n - 1)
                },
                "gadget_counts": {
                    name: stats.gadgets[name] for name in sorted(stats.gadgets)
                },
                "minimum_source_length": min(stats.source_lengths),
                "maximum_source_length": max(stats.source_lengths),
                "event_count_matched_episodes": stats.event_count_matched,
                "length_matched_episodes": stats.length_matched,
                "parity_obstruction_episodes": stats.parity_obstructions,
            }
            for stats in ordered_stats
        }
        return {
            "audit": AUDIT_NAME,
            **projection,
            "all_checks_pass": True,
            "checks": {
                "independent_symbolic_gates": True,
                "all_source_histories_replayed": True,
                "all_probes_and_answers_replayed": True,
                "all_relations_answer_derived": True,
                "no_hidden_state_fields": True,
                "call_ledger_exact": True,
                "no_duplicate_or_overlap": True,
                "counts_and_strata_exact": True,
                "cancellation_balance_exact": True,
                "artifact_hashes_exact": True,
                "generation_report_exact": True,
            },
            "transcript": {
                "path": str(Path(transcript_path).resolve()),
                "rows": transcript_rows,
                "bytes": transcript_bytes,
                "sha256": transcript_sha,
            },
            "ordinary_call_ledger": {
                "path": str(Path(ledger_path).resolve()),
                "rows": ledger_rows,
                "bytes": ledger_bytes,
                "sha256": ledger_sha,
            },
            "generation_report": {
                "path": str(Path(generation_report_path).resolve()),
                "sha256": sha256_bytes(report_bytes),
                **report_binding,
            },
            "cells": safe_cells,
            "expected_ordinary_one_bit_answer_calls": contract.total_calls,
            "errors": [],
            "claim_boundary": CLAIM_BOUNDARY,
        }
    except (AuditError, OSError, TypeError, ValueError) as error:
        return _failure_report(error)


def exclusive_write(path: str | Path, payload: bytes) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as output:
        output.write(payload)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transcript", required=True)
    parser.add_argument("--ledger", required=True)
    parser.add_argument("--generation-report", required=True)
    parser.add_argument("--out", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if Path(args.out).exists():
        raise SystemExit("refusing existing audit output {}".format(args.out))
    report = audit_bundle(
        transcript_path=args.transcript,
        ledger_path=args.ledger,
        generation_report_path=args.generation_report,
    )
    exclusive_write(args.out, pretty_json_bytes(report))
    print(
        json.dumps(
            {
                "schema": SYMBOLIC_AUDIT_SCHEMA,
                "passed": report["passed"],
                "out": str(Path(args.out).resolve()),
            },
            sort_keys=True,
        )
    )
    return 0 if report["all_checks_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
