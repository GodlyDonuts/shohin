#!/usr/bin/env python3
"""Generate one arm of the equal-budget 2x2 DRS factorial curriculum.

TERM changes terminal-transition supervision allocation, not the episode
board.  Each TERM/control pair uses a byte-identical frozen solver-valid board.
Control arms reallocate terminal carry-out-one rows to predeclared nonterminal
controls and use extra carry-out-zero terminal rows to preserve every structural
and position budget.  WIDTH changes only the width allocation.

This is intentionally not another v3 Cartesian local-context basis: v4 keeps
the v2-sized structural budget while requiring the 400 position-independent
arithmetic classes as a shared marginal support contract.

The script is CPU-only, creates no heldout data, and publishes fresh JSONL
artifacts atomically.  Production is bound to the immutable DRS v2 heldout
board, whose base and counterfactual tape signatures are reserved before any
training episode is sampled.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import stat
import sys
import time
import types


ROOT = Path(__file__).resolve().parents[1]
if "_FROZEN_PROTOCOL_EXPORTS" in globals():
    globals().update(globals()["_FROZEN_PROTOCOL_EXPORTS"])
    globals().update(globals()["_FROZEN_RECURRENT_EXPORTS"])
elif __name__ != "__main__":
    sys.path.insert(0, str(ROOT / "train"))
    sys.path.insert(0, str(ROOT / "pipeline"))
    from digitwise_protocol import (  # noqa: E402
        apply_microstep,
        canonical_state,
        final_prompt,
        initial_state,
        microstep_prompt,
        parse_state,
        state_answer,
    )
    from generate_digitwise_recurrent_v1 import (  # noqa: E402
        episode_from_operands,
        normalized,
        rows_from_episode,
    )


SCHEMA = "shohin-digitwise-factorial-v4"
TRAINING_GROUP = "digitwise_factorial_v4"
ARMS = ("iid", "term", "width", "term_width")
ARM_SEEDS = {
    "iid": 202607170101,
    "term": 202607170211,
    "width": 202607170307,
    "term_width": 202607170409,
}
BOARD_SEEDS = {
    False: 202607170701,
    True: 202607170907,
}
BOARD_NAMES = {False: "narrow", True: "wide"}
PRODUCTION_ALLOCATIONS = {
    False: {4: 19_985, 6: 20_000},
    True: {3: 7_982, 4: 8_012, 5: 7_997, 6: 7_997, 7: 7_997},
}
TEST_ALLOCATIONS = {
    False: {4: 200, 6: 200},
    True: {3: 80, 4: 80, 5: 80, 6: 80, 7: 80},
}
# Output-zero classes lead the deterministic tie-break.  This keeps the shared
# board globally balanced while ensuring control allocations are feasible.
ADD_TERMINAL_CLASSES = ("00", "10", "01", "11")
SUB_TERMINAL_CLASSES = ("00", "10")
CONTROL_ADD_TERMINAL_CLASSES = ("00", "10")
ALLOCATION_SUFFIX = "allocation_slot={}"
SOURCE_BY_KIND = {
    "transition": "digitwise_factorial_transition_v4",
    "digit": "digitwise_factorial_readout_v4",
    "final": "digitwise_factorial_final_v4",
}
FROZEN_HELDOUT_SHA256 = (
    "89ce11b36ff2f56e83cda72a1f07b1a90f4a3dc3803c69db2779a27219712646"
)
FROZEN_HELDOUT_COUNTS = {
    "top_level_episodes": 1_500,
    "branches": 3_000,
    "counterfactual_pairs": 1_500,
    "controller_prompts": 19_800,
    "unique_signatures": 3_000,
    "unique_normalized_prompts": 19_800,
}
FROZEN_HELDOUT_REGIMES = {
    "fit_w4": 300,
    "fit_w6": 300,
    "value_ood_w4": 300,
    "value_ood_w6": 300,
    "width_ood_w8": 300,
}
HELDOUT_BRANCH_FIELDS = {
    "id",
    "split",
    "prompt_style",
    "operation",
    "width",
    "left",
    "right",
    "initial_state",
    "expected_states",
    "expected_answer",
}
GENERATOR_RUNTIME_SOURCE_PATHS = (
    "pipeline/generate_digitwise_factorial_v4.py",
    "pipeline/generate_digitwise_recurrent_v1.py",
    "train/digitwise_protocol.py",
)
GENERATOR_VERIFICATION_SOURCE_PATHS = (
    "pipeline/test_generate_digitwise_factorial_v4.py",
)
GENERATOR_SOURCE_PATHS = (
    *GENERATOR_RUNTIME_SOURCE_PATHS,
    *GENERATOR_VERIFICATION_SOURCE_PATHS,
)


def has_term_factor(arm: str) -> bool:
    return arm in ("term", "term_width")


def has_width_factor(arm: str) -> bool:
    return arm in ("width", "term_width")


def paired_arm(arm: str) -> str:
    if arm == "iid":
        return "term"
    if arm == "term":
        return "iid"
    if arm == "width":
        return "term_width"
    if arm == "term_width":
        return "width"
    raise ValueError("unknown factorial arm")


def validate_mode(mode: str, test_scale: int | None) -> None:
    if mode == "production" and test_scale is not None:
        raise SystemExit("production mode forbids --test-scale")
    if mode == "test" and test_scale is None:
        raise SystemExit("test mode requires --test-scale")


def allocations_for_arm(arm: str, test_scale: int | None = None) -> dict[int, int]:
    if arm not in ARMS:
        raise ValueError("unknown factorial arm")
    if test_scale is not None and test_scale <= 0:
        raise ValueError("test scale must be positive")
    base = TEST_ALLOCATIONS if test_scale is not None else PRODUCTION_ALLOCATIONS
    multiplier = test_scale if test_scale is not None else 1
    return {
        width: count * multiplier
        for width, count in base[has_width_factor(arm)].items()
    }


def structural_counts(allocations: dict[int, int]) -> dict[str, int]:
    episodes = sum(allocations.values())
    transitions = sum(width * count for width, count in allocations.items())
    return {
        "episodes": episodes,
        "transitions": transitions,
        "rows": 2 * transitions + episodes,
    }


def operation_allocations(
    allocations: dict[int, int],
) -> dict[int, dict[str, int]]:
    """Split every width nearly equally while balancing odd-width remainders."""
    result: dict[int, dict[str, int]] = {}
    next_extra = "add"
    for width, count in sorted(allocations.items()):
        result[width] = {"add": count // 2, "sub": count // 2}
        if count % 2:
            result[width][next_extra] += 1
            next_extra = "sub" if next_extra == "add" else "add"
    return result


def balanced_labels(labels: tuple[str, ...], count: int) -> list[str]:
    quotient, remainder = divmod(count, len(labels))
    result: list[str] = []
    for index, label in enumerate(labels):
        result.extend([label] * (quotient + int(index < remainder)))
    return result


def expected_terminal_counts(operation_counts: Counter) -> dict[str, Counter]:
    return {
        "add": Counter(balanced_labels(ADD_TERMINAL_CLASSES, operation_counts["add"])),
        "sub": Counter(balanced_labels(SUB_TERMINAL_CLASSES, operation_counts["sub"])),
    }


def stratified_terminal_counts(
    operations_by_width: dict[int, dict[str, int]],
) -> dict[int, dict[str, Counter]]:
    """Balance classes within widths while preserving exact global targets."""
    result = {
        width: {"add": Counter(), "sub": Counter()} for width in operations_by_width
    }
    for operation, labels in (
        ("add", ADD_TERMINAL_CLASSES),
        ("sub", SUB_TERMINAL_CLASSES),
    ):
        total = sum(counts[operation] for counts in operations_by_width.values())
        global_target = Counter(balanced_labels(labels, total))
        residual = Counter(global_target)
        remainders = []
        for width, counts in sorted(operations_by_width.items()):
            quotient, remainder = divmod(counts[operation], len(labels))
            result[width][operation].update({label: quotient for label in labels})
            residual.subtract({label: quotient for label in labels})
            remainders.append((width, remainder))
        for width, remainder in sorted(
            remainders, key=lambda item: (-item[1], item[0])
        ):
            chosen = sorted(
                labels,
                key=lambda label: (-residual[label], labels.index(label)),
            )[:remainder]
            if any(residual[label] <= 0 for label in chosen):
                raise AssertionError("terminal stratification is infeasible")
            result[width][operation].update(chosen)
            residual.subtract(chosen)
        if any(residual.values()):
            raise AssertionError("terminal stratification missed its global target")
    return result


def expected_control_terminal_counts(
    operations_by_width: dict[int, dict[str, int]],
    board_terminal_counts_by_width: dict[int, dict[str, Counter]],
) -> dict[int, dict[str, Counter]]:
    """Project control targets from the immutable stratified board allocation."""
    if set(board_terminal_counts_by_width) != set(operations_by_width):
        raise AssertionError("control target board widths do not match allocations")
    result = {}
    for width, counts in operations_by_width.items():
        board_operations = board_terminal_counts_by_width[width]
        if set(board_operations) != {"add", "sub"}:
            raise AssertionError("control target board operations are incomplete")
        board_sub = Counter(board_operations["sub"])
        if (
            set(board_sub) - set(SUB_TERMINAL_CLASSES)
            or sum(board_sub.values()) != counts["sub"]
        ):
            raise AssertionError("control subtraction target is not a board allocation")
        result[width] = {
            "add": Counter(
                balanced_labels(CONTROL_ADD_TERMINAL_CLASSES, counts["add"])
            ),
            "sub": board_sub,
        }
    return result


def arithmetic_classes(operation: str | None = None) -> set[tuple[str, int, int, int]]:
    operations = (operation,) if operation is not None else ("add", "sub")
    return {
        (candidate, carry, left, right)
        for candidate in operations
        for carry in (0, 1)
        for left in range(10)
        for right in range(10)
    }


def required_width_positions(widths: tuple[int, ...]) -> set[tuple[int, int]]:
    return {(width, position) for width in widths for position in range(width)}


def required_control_contexts(
    widths: tuple[int, ...],
) -> set[tuple[int, int, str, int]]:
    return {
        (width, position, operation, carry)
        for width in widths
        for position in range(width)
        for operation in ("add", "sub")
        for carry in ((0,) if position == 0 else (0, 1))
    }


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _inode_identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_size),
        int(value.st_mtime_ns),
        stat.S_IMODE(value.st_mode),
    )


@dataclass(frozen=True)
class ExactByteSnapshot:
    logical_path: str
    payload: bytes
    sha256: str
    binding: dict

    def verify(self) -> str:
        actual = sha256_bytes(self.payload)
        if actual != self.sha256:
            raise RuntimeError(
                "captured bytes changed for {}: expected {}, got {}".format(
                    self.logical_path, self.sha256, actual
                )
            )
        return actual


def capture_exact_file(
    path: Path, logical_path: str | None = None
) -> ExactByteSnapshot:
    """Capture one regular file once and bind every later use to returned bytes."""
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise SystemExit("input is not a regular non-symlink file: {}".format(path))
    resolved = path.resolve(strict=True)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(resolved, flags)
    try:
        before = os.fstat(descriptor)
        path_before = os.stat(resolved, follow_symlinks=False)
        if not stat.S_ISREG(before.st_mode) or _inode_identity(
            before
        ) != _inode_identity(path_before):
            raise SystemExit("input changed while opening: {}".format(path))
        chunks = []
        digest = hashlib.sha256()
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            chunks.append(block)
            digest.update(block)
        payload = b"".join(chunks)
        after = os.fstat(descriptor)
        path_after = os.stat(resolved, follow_symlinks=False)
        if _inode_identity(after) != _inode_identity(before):
            raise SystemExit("input changed while capturing: {}".format(path))
        if _inode_identity(path_after) != _inode_identity(before):
            raise SystemExit("input path changed while capturing: {}".format(path))
        if len(payload) != int(before.st_size):
            raise SystemExit("input size changed while capturing: {}".format(path))
        identity = {
            "device": int(before.st_dev),
            "inode": int(before.st_ino),
            "mtime_ns": int(before.st_mtime_ns),
            "mode": "{:04o}".format(stat.S_IMODE(before.st_mode)),
            "size": int(before.st_size),
        }
        snapshot = ExactByteSnapshot(
            logical_path=logical_path or str(path.resolve()),
            payload=bytes(payload),
            sha256=digest.hexdigest(),
            binding={
                "bytes": len(payload),
                "capture": "one_pass_stable_descriptor_to_immutable_bytes_v1",
                "fd_identity": identity,
                "path": logical_path or str(path.resolve()),
                "sha256": digest.hexdigest(),
            },
        )
        snapshot.verify()
        return snapshot
    finally:
        os.close(descriptor)


def capture_generator_sources() -> dict[str, ExactByteSnapshot]:
    bootstrap = globals().get("_BOOTSTRAP_SOURCE_SNAPSHOTS")
    if bootstrap is not None:
        snapshots = dict(bootstrap)
    else:
        snapshots = {
            relative: capture_exact_file(ROOT / relative, relative)
            for relative in GENERATOR_SOURCE_PATHS
        }
    if set(snapshots) != set(GENERATOR_SOURCE_PATHS):
        raise RuntimeError("generator scientific source snapshot set mismatch")
    for snapshot in snapshots.values():
        snapshot.verify()
    return {key: snapshots[key] for key in GENERATOR_SOURCE_PATHS}


def source_manifest(snapshots: dict[str, ExactByteSnapshot]) -> dict:
    return {
        "capture": "one_pass_exact_bytes_consumed_by_frozen_cli_v1",
        "runtime_sources": list(GENERATOR_RUNTIME_SOURCE_PATHS),
        "verification_sources": list(GENERATOR_VERIFICATION_SOURCE_PATHS),
        "sources": {
            relative: {
                "bytes": snapshots[relative].binding["bytes"],
                "sha256": snapshots[relative].sha256,
            }
            for relative in GENERATOR_SOURCE_PATHS
        },
    }


def heldout_branch_record(branch: dict) -> dict:
    if not isinstance(branch, dict) or set(branch) != HELDOUT_BRANCH_FIELDS:
        raise ValueError("invalid heldout branch fields")
    if branch["prompt_style"] != "heldout":
        raise ValueError("heldout branch does not use heldout prompt semantics")
    operation = branch["operation"]
    width, left, right = int(branch["width"]), int(branch["left"]), int(branch["right"])
    if operation not in ("add", "sub"):
        raise ValueError("invalid heldout operation")
    rebuilt = canonical_state(initial_state(operation, left, right, width))
    if branch["initial_state"] != rebuilt:
        raise ValueError("heldout initial state does not match operands")
    expected_lines = branch["expected_states"]
    if not isinstance(expected_lines, list) or len(expected_lines) != width:
        raise ValueError("invalid heldout transition count")
    state = parse_state(rebuilt)
    prompts = []
    for expected_line in expected_lines:
        if state is None:
            raise ValueError("invalid heldout state")
        prompts.append(microstep_prompt(state, style="heldout"))
        expected = apply_microstep(state)
        if expected_line != canonical_state(expected):
            raise ValueError("invalid heldout transition target")
        state = parse_state(expected_line)
    if state is None or not state["z"]:
        raise ValueError("heldout branch does not terminate")
    prompts.append(final_prompt(state, style="heldout"))
    validation_only_answer = int(branch["expected_answer"])
    if validation_only_answer != state_answer(state):
        raise ValueError("invalid heldout answer")
    return {
        "id": str(branch["id"]),
        "split": str(branch["split"]),
        "operation": operation,
        "width": width,
        "left": left,
        "right": right,
        "_validation_only_answer": validation_only_answer,
        "signature": (width, left, right),
        "prompts": prompts,
    }


def heldout_pair_records(document: dict) -> tuple[dict, dict]:
    if not isinstance(document, dict) or set(document) != (
        HELDOUT_BRANCH_FIELDS | {"counterfactual"}
    ):
        raise ValueError("invalid heldout top-level fields")
    base_payload = {field: document[field] for field in HELDOUT_BRANCH_FIELDS}
    base = heldout_branch_record(base_payload)
    counterfactual = heldout_branch_record(document["counterfactual"])
    if (
        base["split"] != counterfactual["split"]
        or base["operation"] != counterfactual["operation"]
        or base["width"] != counterfactual["width"]
    ):
        raise ValueError("heldout counterfactual changed protocol")
    changed = int(base["left"] != counterfactual["left"]) + int(
        base["right"] != counterfactual["right"]
    )
    if (
        changed != 1
        or base["_validation_only_answer"] == counterfactual["_validation_only_answer"]
    ):
        raise ValueError("invalid heldout counterfactual intervention")
    del base["_validation_only_answer"]
    del counterfactual["_validation_only_answer"]
    return base, counterfactual


def load_heldout_contract(
    source: ExactByteSnapshot | Path, test_scale: int | None
) -> tuple[set[tuple[int, int, int]], dict]:
    snapshot = (
        source
        if isinstance(source, ExactByteSnapshot)
        else capture_exact_file(source, str(Path(source).resolve()))
    )
    snapshot.verify()
    payload = snapshot.payload
    digest = snapshot.sha256
    if test_scale is None and digest != FROZEN_HELDOUT_SHA256:
        raise SystemExit(
            "production heldout SHA-256 does not match frozen DRS v2 board"
        )

    signatures = set()
    branch_ids = set()
    prompt_keys = set()
    regimes = Counter()
    branches = 0
    controller_prompts = 0
    top_level_episodes = 0
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SystemExit("heldout board is not UTF-8 JSONL") from exc
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        top_level_episodes += 1
        try:
            document = json.loads(line)
            base, counterfactual = heldout_pair_records(document)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise SystemExit(
                "invalid heldout episode at line {}: {}".format(line_number, exc)
            ) from exc
        regimes[base["split"]] += 1
        for branch in (base, counterfactual):
            if branch["id"] in branch_ids:
                raise SystemExit("duplicate heldout branch id")
            if branch["signature"] in signatures:
                raise SystemExit("duplicate heldout reserved signature")
            branch_ids.add(branch["id"])
            signatures.add(branch["signature"])
            branches += 1
            controller_prompts += len(branch["prompts"])
            for prompt in branch["prompts"]:
                key = normalized(prompt)
                if key in prompt_keys:
                    raise SystemExit("duplicate normalized heldout controller prompt")
                prompt_keys.add(key)

    summary = {
        "path": snapshot.logical_path,
        "bytes": len(payload),
        "capture": snapshot.binding["capture"],
        "sha256": digest,
        "frozen_sha256_required": (
            FROZEN_HELDOUT_SHA256 if test_scale is None else None
        ),
        "top_level_episodes": top_level_episodes,
        "branches": branches,
        "counterfactual_pairs": top_level_episodes,
        "controller_prompts": controller_prompts,
        "unique_signatures": len(signatures),
        "unique_normalized_prompts": len(prompt_keys),
        "regimes": dict(sorted(regimes.items())),
        "answer_boundary": {
            "answer_fields_read_for": [
                "solver_witness_validation",
                "counterfactual_answer_change_validation",
            ],
            "answer_values_retained_for_training": False,
            "training_constructor_receives": ["reserved_operand_signatures"],
        },
    }
    if not top_level_episodes or branches != 2 * top_level_episodes:
        raise SystemExit("heldout board has no complete counterfactual pairs")
    if test_scale is None:
        observed_counts = {key: summary[key] for key in FROZEN_HELDOUT_COUNTS}
        if observed_counts != FROZEN_HELDOUT_COUNTS:
            raise SystemExit(
                "production heldout counts do not match frozen DRS v2 board"
            )
        if summary["regimes"] != FROZEN_HELDOUT_REGIMES:
            raise SystemExit(
                "production heldout regimes do not match frozen DRS v2 board"
            )
    return signatures, summary


def local_outgoing_bit(context: tuple[str, int, int, int]) -> int:
    operation, carry, left, right = context
    if operation == "add":
        return (left + right + carry) // 10
    return int(left - right - carry < 0)


def build_slots(allocations: dict[int, int], rng: random.Random) -> list[dict]:
    operations_by_width = operation_allocations(allocations)
    terminal_by_width = stratified_terminal_counts(operations_by_width)
    slots = []
    for width, counts in operations_by_width.items():
        for operation in ("add", "sub"):
            labels = list(terminal_by_width[width][operation].elements())
            rng.shuffle(labels)
            if len(labels) != counts[operation]:
                raise AssertionError("terminal slot count mismatch")
            slots.extend(
                {"width": width, "operation": operation, "terminal_class": label}
                for label in labels
            )
    rng.shuffle(slots)
    assign_control_allocation(slots, operations_by_width, terminal_by_width, rng)
    return slots


def assign_control_allocation(
    slots: list[dict],
    operations_by_width: dict[int, dict[str, int]],
    board_terminal_counts_by_width: dict[int, dict[str, Counter]],
    rng: random.Random,
) -> None:
    """Predeclare the no-TERM terminal multiplicity and matched row budget."""
    targets = expected_control_terminal_counts(
        operations_by_width, board_terminal_counts_by_width
    )
    for slot in slots:
        slot["control_terminal_multiplicity"] = 1
        slot["budget_transition_positions"] = []

    for width, operation_counts in operations_by_width.items():
        add_indices = [
            index
            for index, slot in enumerate(slots)
            if slot["width"] == width and slot["operation"] == "add"
        ]
        zero_output_by_class = {
            terminal_class: [
                index
                for index in add_indices
                if slots[index]["terminal_class"] == terminal_class
            ]
            for terminal_class in CONTROL_ADD_TERMINAL_CLASSES
        }
        for index in add_indices:
            if str(slots[index]["terminal_class"])[1] == "1":
                slots[index]["control_terminal_multiplicity"] = 0

        for terminal_class, indices in zero_output_by_class.items():
            rng.shuffle(indices)
            if not indices:
                raise RuntimeError("control allocation lacks output-zero donors")
            quotient, remainder = divmod(
                targets[width]["add"][terminal_class], len(indices)
            )
            multiplicities = [
                quotient + int(i < remainder) for i in range(len(indices))
            ]
            rng.shuffle(multiplicities)
            for index, multiplicity in zip(indices, multiplicities, strict=True):
                if not 1 <= multiplicity <= int(slots[index]["width"]):
                    raise RuntimeError("control terminal multiplicity is infeasible")
                slots[index]["control_terminal_multiplicity"] = multiplicity

        deficits = [
            index
            for index in add_indices
            if slots[index]["control_terminal_multiplicity"] == 0
        ]
        extra_units = [
            index
            for index in add_indices
            for _ in range(slots[index]["control_terminal_multiplicity"] - 1)
        ]
        if len(deficits) != len(extra_units):
            raise AssertionError("control row budget does not balance by width")
        rng.shuffle(deficits)
        rng.shuffle(extra_units)
        used_positions: dict[int, set[int]] = {}
        for deficit_index, donor_index in zip(deficits, extra_units, strict=True):
            used = used_positions.setdefault(donor_index, set())
            candidates = [
                position
                for position in range(int(slots[donor_index]["width"]) - 1)
                if position not in used
            ]
            candidates.sort(key=lambda position: (position == 1, position))
            if not candidates:
                raise RuntimeError("control row budget has no nonterminal position")
            position = candidates[0]
            used.add(position)
            slots[donor_index]["budget_transition_positions"].append(position)
            slots[deficit_index]["budget_transition_positions"].append(position)

        observed = Counter()
        for index in add_indices:
            observed[str(slots[index]["terminal_class"])] += int(
                slots[index]["control_terminal_multiplicity"]
            )
        if observed != targets[width]["add"]:
            raise AssertionError("control terminal allocation missed its target")
        if sum(observed.values()) != operation_counts["add"]:
            raise AssertionError("control terminal position budget changed")

        observed_sub = Counter(
            {
                terminal_class: sum(
                    int(slots[index]["control_terminal_multiplicity"])
                    for index, slot in enumerate(slots)
                    if slot["width"] == width
                    and slot["operation"] == "sub"
                    and slot["terminal_class"] == terminal_class
                )
                for terminal_class in SUB_TERMINAL_CLASSES
            }
        )
        if observed_sub != targets[width]["sub"]:
            raise AssertionError(
                "control subtraction allocation diverged from its board target"
            )
        if sum(observed_sub.values()) != operation_counts["sub"]:
            raise AssertionError("control subtraction position budget changed")


def assign_designated_classes(
    slots: list[dict], rng: random.Random
) -> dict[int, tuple]:
    """Assign all 400 arithmetic classes without recreating the v3 Cartesian basis."""
    result: dict[int, tuple] = {}
    for operation in ("add", "sub"):
        remaining = sorted(arithmetic_classes(operation))
        rng.shuffle(remaining)
        non_w3 = [
            index
            for index, slot in enumerate(slots)
            if slot["operation"] == operation
            and slot["width"] >= 4
            and not (
                slot["control_terminal_multiplicity"] > 1
                and 1 in slot["budget_transition_positions"]
            )
        ]
        width3 = [
            index
            for index, slot in enumerate(slots)
            if slot["operation"] == operation
            and slot["width"] == 3
            and not (
                slot["control_terminal_multiplicity"] > 1
                and 1 in slot["budget_transition_positions"]
            )
        ]
        rng.shuffle(non_w3)
        rng.shuffle(width3)
        non_w3 = non_w3[: min(len(non_w3), len(remaining))]
        needed_w3 = len(remaining) - len(non_w3)
        if needed_w3 > len(width3):
            raise RuntimeError("insufficient episodes for arithmetic-class coverage")

        for slot_index in width3[:needed_w3]:
            terminal_input = int(slots[slot_index]["terminal_class"][0])
            match = next(
                (
                    index
                    for index, context in enumerate(remaining)
                    if local_outgoing_bit(context) == terminal_input
                ),
                None,
            )
            if match is None:
                raise RuntimeError("cannot match width-3 class to terminal input")
            result[slot_index] = remaining.pop(match)

        if len(non_w3) != len(remaining):
            raise AssertionError("coverage slot accounting mismatch")
        for slot_index, context in zip(non_w3, remaining, strict=True):
            result[slot_index] = context
    if set(result.values()) != arithmetic_classes():
        raise AssertionError("designated arithmetic classes are incomplete")
    return result


def digits_to_value(digits: list[int]) -> int:
    return sum(digit * (10**index) for index, digit in enumerate(digits))


def terminal_top_pair(
    rng: random.Random, operation: str, terminal_class: str
) -> tuple[int, int]:
    terminal_input, terminal_output = map(int, terminal_class)
    if operation == "add":
        candidates = [
            (left, right)
            for left in range(10)
            for right in range(10)
            if (left + right + terminal_input) // 10 == terminal_output
        ]
    else:
        if terminal_output != 0:
            raise ValueError("terminal subtraction must have borrow_out=0")
        candidates = [
            (left, right)
            for left in range(10)
            for right in range(10)
            if int(left - right - terminal_input < 0) == terminal_output
        ]
    return rng.choice(candidates)


def pair_for_outgoing_bit(
    rng: random.Random,
    operation: str,
    carry_in: int,
    carry_out: int,
) -> tuple[int, int]:
    if operation == "add":
        candidates = [
            (left, right)
            for left in range(10)
            for right in range(10)
            if (left + right + carry_in) // 10 == carry_out
        ]
    else:
        candidates = [
            (left, right)
            for left in range(10)
            for right in range(10)
            if int(left - right - carry_in < 0) == carry_out
        ]
    if not candidates:
        raise RuntimeError("no digit pair realizes requested control transition")
    return rng.choice(candidates)


def next_control(operation: str, carry_in: int, left: int, right: int) -> int:
    if operation == "add":
        return (left + right + carry_in) // 10
    return int(left - right - carry_in < 0)


def random_board_operands(
    rng: random.Random,
    width: int,
    operation: str,
    terminal_class: str,
) -> tuple[int, int]:
    left = [rng.randrange(10) for _ in range(width)]
    right = [rng.randrange(10) for _ in range(width)]
    terminal_input = int(terminal_class[0])
    carry = 0
    for position in range(width - 2):
        carry = next_control(operation, carry, left[position], right[position])
    left[-2], right[-2] = pair_for_outgoing_bit(rng, operation, carry, terminal_input)
    left[-1], right[-1] = terminal_top_pair(rng, operation, terminal_class)
    return digits_to_value(left), digits_to_value(right)


def coverage_operands(
    rng: random.Random,
    width: int,
    context: tuple[str, int, int, int],
    terminal_class: str,
) -> tuple[int, int]:
    operation, carry, left_digit, right_digit = context
    if width < 3:
        raise ValueError("coverage episodes require width at least three")
    left = [rng.randrange(10) for _ in range(width)]
    right = [rng.randrange(10) for _ in range(width)]

    left[0], right[0] = pair_for_outgoing_bit(rng, operation, 0, carry)
    left[1], right[1] = left_digit, right_digit

    terminal_input = int(terminal_class[0])
    if width == 3:
        if local_outgoing_bit(context) != terminal_input:
            raise ValueError("width-3 context misses requested terminal input")
    else:
        carry_to_penultimate = local_outgoing_bit(context)
        for position in range(2, width - 2):
            carry_to_penultimate = next_control(
                operation,
                carry_to_penultimate,
                left[position],
                right[position],
            )
        left[-2], right[-2] = pair_for_outgoing_bit(
            rng, operation, carry_to_penultimate, terminal_input
        )
    left[-1], right[-1] = terminal_top_pair(rng, operation, terminal_class)

    left_value, right_value = digits_to_value(left), digits_to_value(right)
    if operation == "sub" and left_value < right_value:
        raise AssertionError("coverage constructor produced negative subtraction")
    return left_value, right_value


def episode_terminal_bits(episode: dict) -> tuple[int, int]:
    state = parse_state(episode["initial_state"])
    if state is None:
        raise ValueError("invalid generated initial state")
    terminal_input = None
    for line in episode["expected_states"]:
        expected = parse_state(line)
        if expected is None:
            raise ValueError("invalid generated expected state")
        if state["p"] == state["w"] - 1:
            terminal_input = int(state["c"])
        state = expected
    if terminal_input is None or not state["z"]:
        raise ValueError("generated episode lacks a terminal transition")
    return terminal_input, int(state["c"])


def episode_contexts(episode: dict) -> set[tuple[str, int, int, int]]:
    state = parse_state(episode["initial_state"])
    if state is None:
        raise ValueError("invalid generated initial state")
    result = set()
    for line in episode["expected_states"]:
        position = int(state["p"])
        result.add(
            (
                str(state["op"]),
                int(state["c"]),
                int(state["a"][position]),
                int(state["b"][position]),
            )
        )
        state = parse_state(line)
        if state is None:
            raise ValueError("invalid generated expected state")
    return result


def add_v4_metadata(
    episode: dict,
    width_factor: bool,
    designated: tuple[str, int, int, int] | None,
    slot: dict,
) -> dict:
    terminal_input, terminal_output = episode_terminal_bits(episode)
    episode.update(
        {
            "schema": SCHEMA,
            "board": BOARD_NAMES[width_factor],
            "board_seed": BOARD_SEEDS[width_factor],
            "terminal_class": "{}{}".format(terminal_input, terminal_output),
            "terminal_input": terminal_input,
            "terminal_output": terminal_output,
            "control_terminal_multiplicity": int(slot["control_terminal_multiplicity"]),
            "budget_transition_positions": sorted(
                int(position) for position in slot["budget_transition_positions"]
            ),
            "designated_arithmetic_class": (
                list(designated) if designated is not None else None
            ),
        }
    )
    if designated is not None and designated not in episode_contexts(episode):
        raise AssertionError("episode missed its designated arithmetic class")
    return episode


def build_episodes(
    arm: str,
    test_scale: int | None = None,
    reserved_signatures: set[tuple[int, int, int]] | None = None,
) -> tuple[list[dict], dict]:
    allocations = allocations_for_arm(arm, test_scale)
    width_factor = has_width_factor(arm)
    rng = random.Random(BOARD_SEEDS[width_factor])
    slots = build_slots(allocations, rng)
    designated = assign_designated_classes(slots, rng)
    episodes = []
    reserved = set(reserved_signatures or ())
    occupied_signatures = set(reserved)
    train_signatures = set()
    reserved_draw_collisions = 0
    for index, slot in enumerate(slots):
        context = designated.get(index)
        for _attempt in range(10_000):
            if context is not None:
                left, right = coverage_operands(
                    rng,
                    int(slot["width"]),
                    context,
                    slot["terminal_class"],
                )
            else:
                left, right = random_board_operands(
                    rng,
                    int(slot["width"]),
                    str(slot["operation"]),
                    str(slot["terminal_class"]),
                )
            signature = (int(slot["width"]), left, right)
            if signature not in occupied_signatures:
                break
            reserved_draw_collisions += int(signature in reserved)
        else:
            raise RuntimeError("could not construct a unique complete episode")
        occupied_signatures.add(signature)
        train_signatures.add(signature)
        episode = episode_from_operands(
            "dfv4-{}-{:05d}".format(BOARD_NAMES[width_factor], index),
            "train",
            int(slot["width"]),
            str(slot["operation"]),
            left,
            right,
            "core",
        )
        add_v4_metadata(episode, width_factor, context, slot)
        if episode["terminal_class"] != slot["terminal_class"]:
            raise AssertionError("terminal constructor missed its class")
        episodes.append(episode)
    if train_signatures & reserved:
        raise RuntimeError("generated train episodes collide with heldout signatures")
    stats = verify_episode_population(episodes, allocations, width_factor)
    stats.update(
        {
            "heldout_reserved_signatures": len(reserved),
            "train_heldout_reserved_signature_hits": len(train_signatures & reserved),
            "reserved_signature_draw_collisions_rejected": reserved_draw_collisions,
        }
    )
    return episodes, stats


def pair_distribution_contract(counts: Counter) -> bool:
    total = sum(counts.values())
    if total <= 1:
        return False
    minimum_unique = min(8, max(2, total // 4))
    maximum_share_count = max(3, (total + 3) // 4)
    return len(counts) >= minimum_unique and max(counts.values()) <= maximum_share_count


def verify_episode_population(
    episodes: list[dict], allocations: dict[int, int], width_factor: bool
) -> dict:
    expected_structure = structural_counts(allocations)
    width_counts = Counter(int(episode["width"]) for episode in episodes)
    operation_counts = Counter(episode["operation"] for episode in episodes)
    terminal_counts = {"add": Counter(), "sub": Counter()}
    terminal_counts_by_width = {
        width: {"add": Counter(), "sub": Counter()} for width in allocations
    }
    control_terminal_counts = {"add": Counter(), "sub": Counter()}
    control_terminal_counts_by_width = {
        width: {"add": Counter(), "sub": Counter()} for width in allocations
    }
    control_position_counts: Counter = Counter()
    expected_position_counts: Counter = Counter()
    penultimate_pairs: dict[tuple, Counter] = {}
    terminal_pairs: dict[tuple, Counter] = {}
    observed_arithmetic = set()
    observed_width_positions = set()
    observed_controls = set()
    signatures = set()
    for episode in episodes:
        width = int(episode["width"])
        operation = str(episode["operation"])
        signatures.add((width, int(episode["left"]), int(episode["right"])))
        terminal_counts[operation][episode["terminal_class"]] += 1
        terminal_counts_by_width[width][operation][episode["terminal_class"]] += 1
        multiplicity = int(episode["control_terminal_multiplicity"])
        control_terminal_counts[operation][episode["terminal_class"]] += multiplicity
        control_terminal_counts_by_width[width][operation][
            episode["terminal_class"]
        ] += multiplicity
        if operation == "sub" and int(episode["terminal_output"]) != 0:
            raise RuntimeError("terminal subtraction borrowed")
        budget_positions = [
            int(position) for position in episode["budget_transition_positions"]
        ]
        expected_budget_count = 1 if multiplicity == 0 else max(0, multiplicity - 1)
        if len(budget_positions) != expected_budget_count or len(
            set(budget_positions)
        ) != len(budget_positions):
            raise RuntimeError("invalid predeclared control-row budget")
        for position in range(width):
            expected_position_counts[(width, operation, position)] += 1
            control_position_counts[(width, operation, position)] += 1
        control_position_counts[(width, operation, width - 1)] += multiplicity - 1
        adjustment = 1 if multiplicity == 0 else -1
        for position in budget_positions:
            control_position_counts[(width, operation, position)] += adjustment
        state = parse_state(episode["initial_state"])
        for line in episode["expected_states"]:
            if state is None:
                raise RuntimeError("generated population contains an invalid state")
            position = int(state["p"])
            observed_arithmetic.add(
                (
                    operation,
                    int(state["c"]),
                    int(state["a"][position]),
                    int(state["b"][position]),
                )
            )
            observed_width_positions.add((width, position))
            observed_controls.add((width, position, operation, int(state["c"])))
            if position == width - 2:
                key = (width, operation, int(episode["terminal_input"]))
                penultimate_pairs.setdefault(key, Counter())[
                    (int(state["a"][position]), int(state["b"][position]))
                ] += 1
            if position == width - 1:
                key = (width, operation, episode["terminal_class"])
                terminal_pairs.setdefault(key, Counter())[
                    (int(state["a"][position]), int(state["b"][position]))
                ] += 1
            state = parse_state(line)

    widths = tuple(sorted(allocations))
    required_arithmetic = arithmetic_classes()
    required_positions = required_width_positions(widths)
    required_controls = required_control_contexts(widths)
    expected_operations = Counter()
    for counts in operation_allocations(allocations).values():
        expected_operations.update(counts)
    if len(episodes) != expected_structure["episodes"]:
        raise RuntimeError("episode target mismatch")
    if len(signatures) != len(episodes):
        raise RuntimeError("duplicate complete episode signature")
    if width_counts != Counter(allocations):
        raise RuntimeError("per-width episode target mismatch")
    if operation_counts != expected_operations:
        raise RuntimeError("operation target mismatch")
    if not required_arithmetic <= observed_arithmetic:
        raise RuntimeError("missing position-independent arithmetic classes")
    if not required_positions <= observed_width_positions:
        raise RuntimeError("missing width/position contexts")
    if not required_controls <= observed_controls:
        raise RuntimeError("missing width/position/control contexts")
    expected_terminal = expected_terminal_counts(operation_counts)
    if terminal_counts != expected_terminal:
        raise RuntimeError("board terminal classes are not deterministically balanced")
    expected_by_width = stratified_terminal_counts(operation_allocations(allocations))
    if terminal_counts_by_width != expected_by_width:
        raise RuntimeError("board terminal classes are not width-stratified")
    expected_control_by_width = expected_control_terminal_counts(
        operation_allocations(allocations), expected_by_width
    )
    expected_control = {"add": Counter(), "sub": Counter()}
    for operations in expected_control_by_width.values():
        for operation, counts in operations.items():
            expected_control[operation].update(counts)
    if control_terminal_counts != expected_control:
        raise RuntimeError("control terminal allocation missed its global target")
    if control_terminal_counts_by_width != expected_control_by_width:
        raise RuntimeError("control terminal allocation missed a width target")
    if control_position_counts != expected_position_counts:
        raise RuntimeError("control allocation changed positional row counts")
    if not all(
        pair_distribution_contract(counts) for counts in penultimate_pairs.values()
    ):
        raise RuntimeError("penultimate digit-pair distribution is concentrated")
    if not all(
        pair_distribution_contract(counts) for counts in terminal_pairs.values()
    ):
        raise RuntimeError("terminal digit-pair distribution is concentrated")

    return {
        "episode_counts_by_width": dict(sorted(width_counts.items())),
        "transition_counts_by_width": {
            width: width * count for width, count in sorted(width_counts.items())
        },
        "row_counts_by_width": {
            width: (2 * width + 1) * count
            for width, count in sorted(width_counts.items())
        },
        "operation_counts": dict(sorted(operation_counts.items())),
        "board_terminal_classes": {
            operation: dict(sorted(counts.items()))
            for operation, counts in terminal_counts.items()
        },
        "board_terminal_classes_by_width": {
            width: {
                operation: dict(sorted(counts.items()))
                for operation, counts in operations.items()
            }
            for width, operations in sorted(terminal_counts_by_width.items())
        },
        "control_terminal_allocation": {
            operation: dict(
                sorted((key, value) for key, value in counts.items() if value)
            )
            for operation, counts in control_terminal_counts.items()
        },
        "control_terminal_allocation_by_width": {
            width: {
                operation: dict(
                    sorted((key, value) for key, value in counts.items() if value)
                )
                for operation, counts in operations.items()
            }
            for width, operations in sorted(control_terminal_counts_by_width.items())
        },
        "paired_position_budget_exact": True,
        "pair_distribution_contract": {
            "penultimate_cells": len(penultimate_pairs),
            "terminal_cells": len(terminal_pairs),
            "all_cells_pass": True,
        },
        "required_arithmetic_classes": len(required_arithmetic),
        "covered_arithmetic_classes": len(required_arithmetic & observed_arithmetic),
        "required_width_positions": len(required_positions),
        "covered_width_positions": len(required_positions & observed_width_positions),
        "required_width_position_controls": len(required_controls),
        "covered_width_position_controls": len(required_controls & observed_controls),
    }


def prompt_variant(row: dict, slot: int, role: str) -> dict:
    result = dict(row)
    prompt = "{}\n{}".format(row["completion_prompt"], ALLOCATION_SUFFIX.format(slot))
    result["question"] = prompt
    result["completion_prompt"] = prompt
    result["allocation_role"] = role
    result["allocation_slot"] = slot
    return result


def rows_for_episode(episode: dict, arm: str) -> list[dict]:
    canonical_rows = rows_from_episode(episode)
    transitions = {
        int(row["transition_index"]): dict(row)
        for row in canonical_rows
        if row["kind"] == "transition"
    }
    digits = {
        int(row["transition_index"]): dict(row)
        for row in canonical_rows
        if row["kind"] == "digit"
    }
    final = next(dict(row) for row in canonical_rows if row["kind"] == "final")
    selected: dict[int, list[dict]] = {index: [] for index in transitions}
    terminal_index = int(episode["width"]) - 1
    budget_positions = [int(value) for value in episode["budget_transition_positions"]]
    multiplicity = int(episode["control_terminal_multiplicity"])

    if has_term_factor(arm):
        variant_count = len(budget_positions)
        variant_positions = [terminal_index] + [
            position for position in range(terminal_index) if position != terminal_index
        ]
        variant_by_position = {
            position: slot
            for slot, position in enumerate(variant_positions[:variant_count], 1)
        }
        for index, row in transitions.items():
            if index in variant_by_position:
                selected[index].append(
                    prompt_variant(row, variant_by_position[index], "term_allocation")
                )
            else:
                row["allocation_role"] = "canonical"
                row["allocation_slot"] = 0
                selected[index].append(row)
    else:
        omitted = set(budget_positions) if multiplicity > 1 else set()
        for index, row in transitions.items():
            if index == terminal_index and multiplicity == 0:
                continue
            if index in omitted:
                continue
            row["allocation_role"] = "canonical"
            row["allocation_slot"] = 0
            selected[index].append(row)
        if multiplicity == 0:
            control_position = budget_positions[0]
            selected[control_position].append(
                prompt_variant(transitions[control_position], 1, "nonterminal_control")
            )
        elif multiplicity > 1:
            for slot in range(1, multiplicity):
                selected[terminal_index].append(
                    prompt_variant(
                        transitions[terminal_index], slot, "terminal_reallocation"
                    )
                )

    rows = []
    for index in range(int(episode["width"])):
        rows.extend(selected[index])
        digit = digits[index]
        digit["allocation_role"] = "canonical"
        digit["allocation_slot"] = 0
        rows.append(digit)
    final["allocation_role"] = "canonical"
    final["allocation_slot"] = 0
    rows.append(final)
    for row in rows:
        row.update(
            {
                "schema": SCHEMA,
                "arm": arm,
                "seed": ARM_SEEDS[arm],
                "split": "train",
                "term_factor": has_term_factor(arm),
                "width_factor": has_width_factor(arm),
                "board": episode["board"],
                "board_seed": episode["board_seed"],
                "terminal_class": episode["terminal_class"],
                "terminal_input": episode["terminal_input"],
                "terminal_output": episode["terminal_output"],
                "source": SOURCE_BY_KIND[row["kind"]],
                "training_group": TRAINING_GROUP,
            }
        )
    return rows


def partial_path(path: Path) -> Path:
    return path.with_name(path.name + ".partial")


def ensure_pairwise_artifact_paths(paths: tuple[Path, ...]) -> None:
    expanded = [candidate for path in paths for candidate in (path, partial_path(path))]
    resolved = [path.resolve(strict=False) for path in expanded]
    if len(set(resolved)) != len(resolved):
        raise SystemExit("final and .partial paths must be pairwise distinct")
    existing = [path for path in expanded if path.exists()]
    for index, left in enumerate(existing):
        for right in existing[index + 1 :]:
            if os.path.samefile(left, right):
                raise SystemExit("final and .partial paths must not alias")


def _fd_sha256(descriptor: int) -> str:
    digest = hashlib.sha256()
    offset = 0
    while True:
        block = os.pread(descriptor, 1024 * 1024, offset)
        if not block:
            return digest.hexdigest()
        digest.update(block)
        offset += len(block)


@dataclass
class PreparedArtifact:
    path: Path
    directory_fd: int
    temporary_name: str
    descriptor: int
    identity: tuple[int, int, int, int, int]
    sha256: str
    bytes: int
    published: bool = False

    def publish(self, before_link=None) -> None:
        if before_link is not None:
            before_link(self.path)
        os.link(
            self.temporary_name,
            self.path.name,
            src_dir_fd=self.directory_fd,
            dst_dir_fd=self.directory_fd,
            follow_symlinks=False,
        )
        os.fsync(self.directory_fd)
        published = os.stat(
            self.path.name, dir_fd=self.directory_fd, follow_symlinks=False
        )
        if _inode_identity(published) != self.identity:
            raise RuntimeError(
                "published artifact identity mismatch: {}".format(self.path)
            )
        if _fd_sha256(self.descriptor) != self.sha256:
            raise RuntimeError("published artifact bytes changed: {}".format(self.path))
        self.published = True

    def close(self) -> None:
        try:
            os.unlink(self.temporary_name, dir_fd=self.directory_fd)
        except FileNotFoundError:
            pass
        os.close(self.descriptor)
        os.close(self.directory_fd)


def prepare_artifact(path: Path, writer, *, binary: bool = False) -> PreparedArtifact:
    """Prepare one fsynced private inode; final publication is a no-replace link."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    directory_fd = os.open(
        path.parent,
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    parent_stat = os.fstat(directory_fd)
    path_parent_stat = os.stat(path.parent, follow_symlinks=False)
    if not stat.S_ISDIR(parent_stat.st_mode) or (
        parent_stat.st_dev,
        parent_stat.st_ino,
    ) != (path_parent_stat.st_dev, path_parent_stat.st_ino):
        os.close(directory_fd)
        raise RuntimeError(
            "artifact parent changed while opening: {}".format(path.parent)
        )
    temporary_name = ".{}.private.{}.{}".format(path.name, os.getpid(), time.time_ns())
    descriptor = -1
    try:
        descriptor = os.open(
            temporary_name,
            os.O_RDWR | os.O_CREAT | os.O_EXCL,
            0o600,
            dir_fd=directory_fd,
        )
        mode = "wb" if binary else "w"
        kwargs = {} if binary else {"encoding": "utf-8", "newline": "\n"}
        with os.fdopen(os.dup(descriptor), mode, **kwargs) as output:
            writer(output)
            output.flush()
            os.fsync(output.fileno())
        file_stat = os.fstat(descriptor)
        if file_stat.st_size <= 0:
            raise RuntimeError("refusing empty artifact: {}".format(path))
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
        return PreparedArtifact(
            path=path,
            directory_fd=directory_fd,
            temporary_name=temporary_name,
            descriptor=descriptor,
            identity=_inode_identity(os.fstat(descriptor)),
            sha256=_fd_sha256(descriptor),
            bytes=int(file_stat.st_size),
        )
    except BaseException:
        try:
            os.unlink(temporary_name, dir_fd=directory_fd)
        except FileNotFoundError:
            pass
        if descriptor >= 0:
            os.close(descriptor)
        os.close(directory_fd)
        raise


def publish_bytes_no_replace(path: Path, payload: bytes, before_link=None) -> None:
    artifact = prepare_artifact(path, lambda output: output.write(payload), binary=True)
    try:
        artifact.publish(before_link=before_link)
    finally:
        artifact.close()


def publish(
    data_path: Path,
    episodes_path: Path,
    report_path: Path,
    episodes: list[dict],
    arm: str,
    mode: str,
    test_scale: int | None,
    population_stats: dict,
    heldout_summary: dict,
    source_snapshots: dict[str, ExactByteSnapshot],
) -> dict:
    destinations = (data_path, episodes_path, report_path)
    ensure_pairwise_artifact_paths(destinations)
    prepared: list[PreparedArtifact] = []
    try:

        def write_episodes(output) -> None:
            for episode in episodes:
                output.write(json.dumps(episode, sort_keys=True) + "\n")

        episodes_artifact = prepare_artifact(episodes_path, write_episodes)
        prepared.append(episodes_artifact)

        seen_prompts = set()
        row_counts = Counter()
        terminal_row_counts = {"add": Counter(), "sub": Counter()}
        allocation_roles = Counter()
        visible_features = Counter()

        def write_data(output) -> None:
            for episode in episodes:
                for row in rows_for_episode(episode, arm):
                    prompt = normalized(row["completion_prompt"])
                    if prompt in seen_prompts:
                        raise RuntimeError("duplicate normalized training prompt")
                    seen_prompts.add(prompt)
                    row_counts[row["kind"]] += 1
                    allocation_roles[row["allocation_role"]] += 1
                    visible_features[
                        (
                            row["kind"],
                            int(episode["width"]),
                            str(episode["operation"]),
                            int(episode["left"]),
                            int(episode["right"]),
                        )
                    ] += 1
                    if (
                        row["kind"] == "transition"
                        and int(row["transition_index"]) == int(row["width"]) - 1
                    ):
                        terminal_row_counts[row["operation"]][
                            row["terminal_class"]
                        ] += 1
                    output.write(json.dumps(row, sort_keys=True) + "\n")

        data_artifact = prepare_artifact(data_path, write_data)
        prepared.append(data_artifact)

        allocations = allocations_for_arm(arm, test_scale)
        target = structural_counts(allocations)
        if row_counts != Counter(
            {
                "transition": target["transitions"],
                "digit": target["transitions"],
                "final": target["episodes"],
            }
        ):
            raise RuntimeError("row-kind structural target mismatch")
        if len(seen_prompts) != target["rows"]:
            raise RuntimeError("unique prompt target mismatch")
        operation_counts = Counter(episode["operation"] for episode in episodes)
        expected_rows = (
            expected_terminal_counts(operation_counts)
            if has_term_factor(arm)
            else {operation: Counter() for operation in ("add", "sub")}
        )
        if not has_term_factor(arm):
            board_terminal_counts_by_width = stratified_terminal_counts(
                operation_allocations(allocations)
            )
            for operations in expected_control_terminal_counts(
                operation_allocations(allocations), board_terminal_counts_by_width
            ).values():
                for operation, counts in operations.items():
                    expected_rows[operation].update(counts)
        if terminal_row_counts != expected_rows:
            raise RuntimeError("supervised terminal allocation target mismatch")

        board_ids = "".join(episode["id"] + "\n" for episode in episodes).encode(
            "ascii"
        )
        visible_projection = json.dumps(
            [
                [list(feature), count]
                for feature, count in sorted(visible_features.items())
            ],
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")

        report = {
            "schema": SCHEMA,
            "arm": arm,
            "seed": ARM_SEEDS[arm],
            "board": BOARD_NAMES[has_width_factor(arm)],
            "board_seed": BOARD_SEEDS[has_width_factor(arm)],
            "paired_arm": paired_arm(arm),
            "factors": {
                "term": has_term_factor(arm),
                "width": has_width_factor(arm),
            },
            "mode": mode,
            "production_contract": mode == "production",
            "production_eligible": mode == "production",
            "production_admission": False,
            "test_scale": test_scale,
            "structural_target": target,
            "row_kind_counts": dict(sorted(row_counts.items())),
            "unique_complete_episode_signatures": len(episodes),
            "unique_normalized_prompts": len(seen_prompts),
            "terminal_transition_classes": {
                operation: dict(
                    sorted((key, value) for key, value in counts.items() if value)
                )
                for operation, counts in terminal_row_counts.items()
            },
            "allocation_roles": dict(sorted(allocation_roles.items())),
            "heldout_binding": heldout_summary,
            **population_stats,
            "data": str(data_path.resolve()),
            "episodes": str(episodes_path.resolve()),
            "data_bytes": data_artifact.bytes,
            "data_sha256": data_artifact.sha256,
            "episodes_bytes": episodes_artifact.bytes,
            "episodes_sha256": episodes_artifact.sha256,
            "paired_board": {
                "literal_jsonl_sha256": episodes_artifact.sha256,
                "episode_ids_sha256": sha256_bytes(board_ids),
                "episode_count": len(episodes),
                "first_episode_id": episodes[0]["id"],
                "last_episode_id": episodes[-1]["id"],
                "full_visible_feature_multiset_sha256": sha256_bytes(
                    visible_projection
                ),
                "required_counterpart_arm": paired_arm(arm),
            },
            "scientific_source_manifest": source_manifest(source_snapshots),
            "runtime": {
                "python": platform.python_version(),
                "python_implementation": platform.python_implementation(),
            },
            "generator_sha256": source_snapshots[
                "pipeline/generate_digitwise_factorial_v4.py"
            ].sha256,
            "recurrent_generator_sha256": source_snapshots[
                "pipeline/generate_digitwise_recurrent_v1.py"
            ].sha256,
            "protocol_sha256": source_snapshots["train/digitwise_protocol.py"].sha256,
            "factorial_contract": (
                "TERM changes terminal-transition supervision allocation on a byte-identical "
                "paired episode board; WIDTH changes the width allocation; every arm retains "
                "an equal episode/transition/row and visible-tape budget."
            ),
            "iid_semantics": (
                "iid and width are no-TERM supervision controls, not low-magnitude episode "
                "boards. The former magnitude-band definition is incompatible with literal "
                "paired-board equality."
            ),
            "basis_v3_boundary": (
                "The 400 arithmetic classes are position-independent marginal support, not the v3 "
                "Cartesian enumeration of every width/position/digit context."
            ),
            "claim_boundary": (
                "CPU-only data construction evidence. Heldout answers are read only to validate "
                "solver witnesses and counterfactual changes; only reserved operand signatures "
                "reach training-board construction. No heldout answer constructs a training row."
            ),
            "residual_bundle_confound": (
                "Digit-readout and final-answer rows remain common and expose the completed "
                "terminal state. Current-position arithmetic distributions necessarily differ "
                "with the supervised terminal-class allocation. The allocation suffix has an "
                "exactly matched global frequency but is associated with different positions by "
                "design; only full operand/tape, width, operation, row-kind, format-count, and "
                "aggregate position budgets are paired exactly."
            ),
        }

        def write_report(output) -> None:
            output.write(json.dumps(report, indent=2, sort_keys=True) + "\n")

        report_artifact = prepare_artifact(report_path, write_report)
        prepared.append(report_artifact)
        for artifact in prepared:
            artifact.publish()
        return report
    finally:
        for artifact in prepared:
            artifact.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("production", "test"))
    parser.add_argument("--arm", required=True, choices=ARMS)
    parser.add_argument("--data-out", required=True, type=Path)
    parser.add_argument("--episodes-out", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--heldout", required=True, type=Path)
    parser.add_argument(
        "--test-scale",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()
    if args.test_scale is not None and args.test_scale <= 0:
        raise SystemExit("--test-scale must be positive")
    validate_mode(args.mode, args.test_scale)
    ensure_pairwise_artifact_paths(
        (args.heldout, args.data_out, args.episodes_out, args.report)
    )
    source_snapshots = capture_generator_sources()
    heldout_snapshot = capture_exact_file(args.heldout, str(args.heldout.resolve()))
    reserved_signatures, heldout_summary = load_heldout_contract(
        heldout_snapshot, args.test_scale
    )
    episodes, stats = build_episodes(
        args.arm,
        args.test_scale,
        reserved_signatures=reserved_signatures,
    )
    report = publish(
        args.data_out,
        args.episodes_out,
        args.report,
        episodes,
        args.arm,
        args.mode,
        args.test_scale,
        stats,
        heldout_summary,
        source_snapshots,
    )
    print(json.dumps(report, sort_keys=True))


def _frozen_module(name: str, snapshot: ExactByteSnapshot) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__file__ = str(ROOT / snapshot.logical_path)
    module.__package__ = ""
    module.__loader__ = None
    module.__spec__ = None
    sys.modules[name] = module
    exec(compile(snapshot.payload, module.__file__, "exec"), module.__dict__)
    return module


def _run_frozen_cli() -> None:
    """Execute all scientific generator code from one exact-byte source capture."""
    snapshots = {
        relative: capture_exact_file(ROOT / relative, relative)
        for relative in GENERATOR_SOURCE_PATHS
    }
    protocol = _frozen_module(
        "digitwise_protocol", snapshots["train/digitwise_protocol.py"]
    )
    recurrent = _frozen_module(
        "generate_digitwise_recurrent_v1",
        snapshots["pipeline/generate_digitwise_recurrent_v1.py"],
    )
    protocol_names = (
        "apply_microstep",
        "canonical_state",
        "final_prompt",
        "initial_state",
        "microstep_prompt",
        "parse_state",
        "state_answer",
    )
    recurrent_names = ("episode_from_operands", "normalized", "rows_from_episode")
    namespace = {
        "__name__": "__main__",
        "__file__": str(ROOT / "pipeline/generate_digitwise_factorial_v4.py"),
        "__package__": None,
        "_BOOTSTRAP_ACTIVE": True,
        "_BOOTSTRAP_SOURCE_SNAPSHOTS": snapshots,
        "_FROZEN_PROTOCOL_EXPORTS": {
            name: getattr(protocol, name) for name in protocol_names
        },
        "_FROZEN_RECURRENT_EXPORTS": {
            name: getattr(recurrent, name) for name in recurrent_names
        },
    }
    generator = snapshots["pipeline/generate_digitwise_factorial_v4.py"]
    exec(compile(generator.payload, namespace["__file__"], "exec"), namespace)


if __name__ == "__main__":
    if globals().get("_BOOTSTRAP_ACTIVE"):
        main()
    else:
        _run_frozen_cli()
