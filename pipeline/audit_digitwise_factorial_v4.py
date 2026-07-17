#!/usr/bin/env python3
"""Independently audit one arm of the equal-budget 2x2 DRS factorial.

The auditor does not import the v4 generator or consume its report.  It
redefines every arm target, recomputes every solver transition/readout/final,
checks exact episode-row correspondence, and rejects data whose TERM or WIDTH
contract differs from the declared arm.

Production additionally reopens the immutable DRS v2 heldout board and proves
signature, exact normalized prompt, and literal normalized-word 13-gram
separation using the same transition-plus-final controller surface as the
established v2 audit.
"""

from __future__ import annotations

import argparse
from array import array
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import re
import stat
import sys
import tempfile
import time
import types


ROOT = Path(__file__).resolve().parents[1]
if "_FROZEN_PROTOCOL_EXPORTS" in globals():
    globals().update(globals()["_FROZEN_PROTOCOL_EXPORTS"])
    globals().update(globals()["_FROZEN_ENCODING_EXPORTS"])
elif __name__ != "__main__":
    sys.path.insert(0, str(ROOT / "train"))
    from digitwise_protocol import (  # noqa: E402
        apply_microstep,
        canonical_state,
        digit_prompt,
        final_prompt,
        initial_state,
        microstep_prompt,
        parse_state,
        state_answer,
        state_digit,
    )
    from sft_encoding import encode_supervised_example  # noqa: E402


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
ADD_TERMINAL_CLASSES = ("00", "10", "01", "11")
SUB_TERMINAL_CLASSES = ("00", "10")
CONTROL_ADD_TERMINAL_CLASSES = ("00", "10")
ALLOCATION_SUFFIX = "allocation_slot={}"
SOURCE_BY_KIND = {
    "transition": "digitwise_factorial_transition_v4",
    "digit": "digitwise_factorial_readout_v4",
    "final": "digitwise_factorial_final_v4",
}
WORD = re.compile(r"\w+")

EPISODE_FIELDS = {
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
    "schema",
    "board",
    "board_seed",
    "terminal_class",
    "terminal_input",
    "terminal_output",
    "control_terminal_multiplicity",
    "budget_transition_positions",
    "designated_arithmetic_class",
}
COMMON_ROW_FIELDS = {
    "question",
    "completion_prompt",
    "response",
    "source",
    "training_group",
    "kind",
    "episode_id",
    "width",
    "operation",
    "transition_index",
    "state",
    "prompt_style",
    "schema",
    "arm",
    "seed",
    "split",
    "term_factor",
    "width_factor",
    "board",
    "board_seed",
    "terminal_class",
    "terminal_input",
    "terminal_output",
    "allocation_role",
    "allocation_slot",
}
ROW_FIELDS_BY_KIND = {
    "transition": COMMON_ROW_FIELDS | {"expected_state"},
    "digit": COMMON_ROW_FIELDS | {"digit_index", "expected_digit"},
    "final": COMMON_ROW_FIELDS | {"expected_answer"},
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
CANONICAL_EXACT_BUDGET = {
    "optimizer_updates": 1_560,
    "batch_size": 16,
    "pack_len": 2_048,
    "seed": 1_337,
}
CANONICAL_REQUIRED_PACKS = (
    CANONICAL_EXACT_BUDGET["optimizer_updates"] * CANONICAL_EXACT_BUDGET["batch_size"]
)
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
GENERATOR_SOURCE_PATHS = (
    "pipeline/generate_digitwise_factorial_v4.py",
    "pipeline/generate_digitwise_recurrent_v1.py",
    "pipeline/test_generate_digitwise_factorial_v4.py",
    "train/digitwise_protocol.py",
)
AUDITOR_RUNTIME_SOURCE_PATHS = (
    "pipeline/audit_digitwise_factorial_v4.py",
    "train/digitwise_protocol.py",
    "train/sft.py",
    "train/sft_encoding.py",
)
AUDITOR_VERIFICATION_SOURCE_PATHS = (
    "pipeline/test_audit_digitwise_factorial_v4.py",
    "train/test_sft_exact_budget.py",
    "train/jobs/sft_factorial.sbatch",
)
ADMISSION_SOURCE_PATHS = tuple(
    dict.fromkeys(
        (
            *GENERATOR_SOURCE_PATHS,
            *AUDITOR_RUNTIME_SOURCE_PATHS,
            *AUDITOR_VERIFICATION_SOURCE_PATHS,
        )
    )
)


class ContractError(ValueError):
    """A stable, reportable admission failure."""


def require(condition: bool, code: str) -> None:
    if not condition:
        raise ContractError(code)


def exact_int(value, code: str) -> int:
    require(isinstance(value, int) and not isinstance(value, bool), code)
    return int(value)


def normalized(text) -> str:
    return " ".join(WORD.findall(str(text).lower()))


def ngrams(text, width: int = 13) -> set[str]:
    words = normalized(text).split()
    return {
        " ".join(words[index : index + width])
        for index in range(max(0, len(words) - width + 1))
    }


def has_term_factor(arm: str) -> bool:
    return arm in ("term", "term_width")


def has_width_factor(arm: str) -> bool:
    return arm in ("width", "term_width")


def paired_arm(arm: str) -> str:
    pairs = {
        "iid": "term",
        "term": "iid",
        "width": "term_width",
        "term_width": "width",
    }
    try:
        return pairs[arm]
    except KeyError as exc:
        raise ValueError("unknown factorial arm") from exc


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
    result: dict[int, dict[str, int]] = {}
    next_extra = "add"
    for width, count in sorted(allocations.items()):
        result[width] = {"add": count // 2, "sub": count // 2}
        if count % 2:
            result[width][next_extra] += 1
            next_extra = "sub" if next_extra == "add" else "add"
    return result


def balanced_counter(labels: tuple[str, ...], count: int) -> Counter:
    quotient, remainder = divmod(count, len(labels))
    return Counter(
        {label: quotient + int(index < remainder) for index, label in enumerate(labels)}
    )


def expected_terminal_counts(
    expected_operations: Counter,
) -> dict[str, Counter]:
    return {
        "add": balanced_counter(ADD_TERMINAL_CLASSES, expected_operations["add"]),
        "sub": balanced_counter(SUB_TERMINAL_CLASSES, expected_operations["sub"]),
    }


def stratified_terminal_counts(
    operations_by_width: dict[int, dict[str, int]],
) -> dict[int, dict[str, Counter]]:
    """Independently derive exact global and within-width TERM balances."""
    result = {
        width: {"add": Counter(), "sub": Counter()} for width in operations_by_width
    }
    for operation, labels in (
        ("add", ADD_TERMINAL_CLASSES),
        ("sub", SUB_TERMINAL_CLASSES),
    ):
        total = sum(counts[operation] for counts in operations_by_width.values())
        global_target = balanced_counter(labels, total)
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
    """Independently project control targets from the stratified board contract."""
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
            "add": balanced_counter(CONTROL_ADD_TERMINAL_CLASSES, counts["add"]),
            "sub": board_sub,
        }
    return result


def pair_distribution_contract(counts: Counter) -> bool:
    total = sum(counts.values())
    if total <= 1:
        return False
    minimum_unique = min(8, max(2, total // 4))
    maximum_share_count = max(3, (total + 3) // 4)
    return len(counts) >= minimum_unique and max(counts.values()) <= maximum_share_count


def required_arithmetic_classes() -> set[tuple[str, int, int, int]]:
    return {
        (operation, carry, left, right)
        for operation in ("add", "sub")
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

    def text(self) -> io.StringIO:
        self.verify()
        try:
            return io.StringIO(self.payload.decode("utf-8", errors="strict"))
        except UnicodeDecodeError as exc:
            raise SystemExit(
                "input is not UTF-8 text: {}".format(self.logical_path)
            ) from exc


def capture_exact_file(
    path: Path, logical_path: str | None = None
) -> ExactByteSnapshot:
    """Capture one regular file once and sever all later audit path dependence."""
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
        snapshot = ExactByteSnapshot(
            logical_path=logical_path or str(path.resolve()),
            payload=bytes(payload),
            sha256=digest.hexdigest(),
            binding={
                "bytes": len(payload),
                "capture": "one_pass_stable_descriptor_to_immutable_bytes_v1",
                "fd_identity": {
                    "device": int(before.st_dev),
                    "inode": int(before.st_ino),
                    "mtime_ns": int(before.st_mtime_ns),
                    "mode": "{:04o}".format(stat.S_IMODE(before.st_mode)),
                    "size": int(before.st_size),
                },
                "path": logical_path or str(path.resolve()),
                "sha256": digest.hexdigest(),
            },
        )
        snapshot.verify()
        return snapshot
    finally:
        os.close(descriptor)


def capture_auditor_sources() -> dict[str, ExactByteSnapshot]:
    bootstrap = globals().get("_BOOTSTRAP_SOURCE_SNAPSHOTS")
    snapshots = (
        dict(bootstrap)
        if bootstrap is not None
        else {
            relative: capture_exact_file(ROOT / relative, relative)
            for relative in ADMISSION_SOURCE_PATHS
        }
    )
    if set(snapshots) != set(ADMISSION_SOURCE_PATHS):
        raise RuntimeError("auditor scientific source snapshot set mismatch")
    for snapshot in snapshots.values():
        snapshot.verify()
    return {key: snapshots[key] for key in ADMISSION_SOURCE_PATHS}


def manifest_for_paths(
    snapshots: dict[str, ExactByteSnapshot], paths: tuple[str, ...]
) -> dict:
    return {
        "capture": "one_pass_exact_bytes_consumed_by_frozen_cli_v1",
        "sources": {
            relative: {
                "bytes": snapshots[relative].binding["bytes"],
                "sha256": snapshots[relative].sha256,
            }
            for relative in paths
        },
    }


def validate_generator_report(
    report_snapshot: ExactByteSnapshot,
    data_snapshot: ExactByteSnapshot,
    episodes_snapshot: ExactByteSnapshot,
    heldout_snapshot: ExactByteSnapshot,
    arm: str,
    mode: str,
    source_snapshots: dict[str, ExactByteSnapshot],
) -> dict:
    """Validate generator evidence without importing generator implementation code."""
    try:
        report = json.loads(report_snapshot.payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContractError("generator_report_json") from exc
    require(isinstance(report, dict), "generator_report_object")
    require(report.get("schema") == SCHEMA, "generator_report_schema")
    require(report.get("arm") == arm, "generator_report_arm")
    require(report.get("mode") == mode, "generator_report_mode")
    require(report.get("data_sha256") == data_snapshot.sha256, "generator_report_data")
    require(
        report.get("episodes_sha256") == episodes_snapshot.sha256,
        "generator_report_episodes",
    )
    heldout = report.get("heldout_binding")
    require(isinstance(heldout, dict), "generator_report_heldout_binding")
    require(
        heldout.get("sha256") == heldout_snapshot.sha256,
        "generator_report_heldout_sha256",
    )
    manifest = report.get("scientific_source_manifest")
    require(isinstance(manifest, dict), "generator_report_source_manifest")
    expected_manifest = manifest_for_paths(source_snapshots, GENERATOR_SOURCE_PATHS)
    require(
        manifest.get("sources") == expected_manifest["sources"],
        "generator_report_source_bytes",
    )
    require(
        manifest.get("runtime_sources")
        == [
            "pipeline/generate_digitwise_factorial_v4.py",
            "pipeline/generate_digitwise_recurrent_v1.py",
            "train/digitwise_protocol.py",
        ],
        "generator_report_runtime_sources",
    )
    runtime = report.get("runtime")
    require(
        isinstance(runtime, dict)
        and isinstance(runtime.get("python"), str)
        and bool(runtime["python"])
        and isinstance(runtime.get("python_implementation"), str)
        and bool(runtime["python_implementation"]),
        "generator_report_runtime",
    )
    return {
        "bytes": len(report_snapshot.payload),
        "path": report_snapshot.logical_path,
        "sha256": report_snapshot.sha256,
        "runtime": runtime,
        "validated": True,
        "source_manifest": manifest,
    }


def record_failure(
    failures: Counter,
    examples: list[dict],
    category: str,
    line_number: int,
    error: BaseException,
) -> None:
    code = str(error) or error.__class__.__name__
    failures["{}:{}".format(category, code)] += 1
    if len(examples) < 16:
        examples.append({"category": category, "line": line_number, "error": code})


def audit_heldout_branch(branch: dict) -> dict:
    require(isinstance(branch, dict), "heldout_branch_not_object")
    require(set(branch) == HELDOUT_BRANCH_FIELDS, "heldout_branch_fields")
    require(branch["prompt_style"] == "heldout", "heldout_prompt_style")
    require(isinstance(branch["id"], str) and branch["id"], "heldout_id")
    require(isinstance(branch["split"], str) and branch["split"], "heldout_split")
    operation = branch["operation"]
    require(operation in ("add", "sub"), "heldout_operation")
    width = exact_int(branch["width"], "heldout_width")
    left = exact_int(branch["left"], "heldout_left")
    right = exact_int(branch["right"], "heldout_right")
    require(left >= 0 and right >= 0, "heldout_negative_operand")
    require(operation != "sub" or left >= right, "heldout_negative_subtraction")

    rebuilt = canonical_state(initial_state(operation, left, right, width))
    require(branch["initial_state"] == rebuilt, "heldout_initial_state")
    expected_lines = branch["expected_states"]
    require(isinstance(expected_lines, list), "heldout_expected_states_type")
    require(len(expected_lines) == width, "heldout_transition_count")
    state = parse_state(rebuilt)
    prompts = []
    for index, expected_line in enumerate(expected_lines):
        require(state is not None, "heldout_state_parse")
        require(int(state["p"]) == index, "heldout_position_sequence")
        prompts.append(microstep_prompt(state, style="heldout"))
        expected = apply_microstep(state)
        require(expected_line == canonical_state(expected), "heldout_transition_target")
        state = parse_state(expected_line)
    require(state is not None and bool(state["z"]), "heldout_terminal_state")
    prompts.append(final_prompt(state, style="heldout"))
    validation_only_answer = exact_int(branch["expected_answer"], "heldout_answer_type")
    require(validation_only_answer == state_answer(state), "heldout_answer")
    return {
        "id": branch["id"],
        "split": branch["split"],
        "operation": operation,
        "width": width,
        "left": left,
        "right": right,
        "_validation_only_answer": validation_only_answer,
        "signature": (width, left, right),
        "prompts": prompts,
    }


def audit_heldout_pair(document: dict) -> tuple[dict, dict]:
    require(isinstance(document, dict), "heldout_top_not_object")
    require(
        set(document) == HELDOUT_BRANCH_FIELDS | {"counterfactual"},
        "heldout_top_fields",
    )
    base_payload = {field: document[field] for field in HELDOUT_BRANCH_FIELDS}
    base = audit_heldout_branch(base_payload)
    counterfactual = audit_heldout_branch(document["counterfactual"])
    require(base["split"] == counterfactual["split"], "heldout_pair_split")
    require(
        base["operation"] == counterfactual["operation"],
        "heldout_pair_operation",
    )
    require(base["width"] == counterfactual["width"], "heldout_pair_width")
    changed = int(base["left"] != counterfactual["left"]) + int(
        base["right"] != counterfactual["right"]
    )
    require(changed == 1, "heldout_counterfactual_edit")
    require(
        base["_validation_only_answer"] != counterfactual["_validation_only_answer"],
        "heldout_pair_answer",
    )
    del base["_validation_only_answer"]
    del counterfactual["_validation_only_answer"]
    return base, counterfactual


def observe_heldout(
    snapshot: ExactByteSnapshot,
    test_scale: int | None,
    failures: Counter,
    examples: list[dict],
) -> dict:
    snapshot.verify()
    payload = snapshot.payload
    digest = snapshot.sha256
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SystemExit("heldout board is not UTF-8 JSONL") from exc

    raw_top_level = 0
    valid_top_level = 0
    branch_ids = set()
    signatures = set()
    prompt_keys = set()
    gram13 = set()
    regimes = Counter()
    controller_prompts = 0
    blank_lines = 0
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            blank_lines += 1
            continue
        raw_top_level += 1
        try:
            document = json.loads(line)
            base, counterfactual = audit_heldout_pair(document)
            records = (base, counterfactual)
            new_ids = {record["id"] for record in records}
            new_signatures = {record["signature"] for record in records}
            require(len(new_ids) == 2, "duplicate_heldout_pair_id")
            require(len(new_signatures) == 2, "duplicate_heldout_pair_signature")
            require(not (new_ids & branch_ids), "duplicate_heldout_branch_id")
            require(
                not (new_signatures & signatures),
                "duplicate_heldout_reserved_signature",
            )
            new_prompts = [prompt for record in records for prompt in record["prompts"]]
            new_prompt_keys = {normalized(prompt) for prompt in new_prompts}
            require(
                len(new_prompt_keys) == len(new_prompts),
                "duplicate_heldout_pair_prompt",
            )
            require(
                not (new_prompt_keys & prompt_keys),
                "duplicate_normalized_heldout_prompt",
            )
            branch_ids.update(new_ids)
            signatures.update(new_signatures)
            prompt_keys.update(new_prompt_keys)
            for prompt in new_prompts:
                gram13.update(ngrams(prompt))
            controller_prompts += len(new_prompts)
            regimes[base["split"]] += 1
            valid_top_level += 1
        except (
            ContractError,
            KeyError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
        ) as exc:
            record_failure(failures, examples, "heldout", line_number, exc)

    return {
        "path": snapshot.logical_path,
        "bytes": len(snapshot.payload),
        "capture": snapshot.binding["capture"],
        "sha256": digest,
        "frozen_sha256_required": (
            FROZEN_HELDOUT_SHA256 if test_scale is None else None
        ),
        "raw_top_level_episodes": raw_top_level,
        "top_level_episodes": valid_top_level,
        "branches": 2 * valid_top_level,
        "counterfactual_pairs": valid_top_level,
        "controller_prompts": controller_prompts,
        "unique_signatures": len(signatures),
        "unique_normalized_prompts": len(prompt_keys),
        "regimes": dict(sorted(regimes.items())),
        "blank_lines": blank_lines,
        "identity_digests": {
            "branch_ids_sha256": hashlib.sha256(
                "".join(value + "\n" for value in sorted(branch_ids)).encode("utf-8")
            ).hexdigest(),
            "normalized_prompts_sha256": hashlib.sha256(
                "".join(value + "\n" for value in sorted(prompt_keys)).encode("utf-8")
            ).hexdigest(),
            "reserved_signatures_sha256": hashlib.sha256(
                json.dumps(
                    [list(value) for value in sorted(signatures)],
                    separators=(",", ":"),
                ).encode("ascii")
            ).hexdigest(),
        },
        "answer_boundary": {
            "answer_fields_read_for": [
                "solver_witness_validation",
                "counterfactual_answer_change_validation",
                "contamination_exclusion_identity_validation",
            ],
            "answer_values_retained_for_training": False,
            "training_rows_constructed_by_auditor": False,
        },
        "signatures": signatures,
        "prompt_keys": prompt_keys,
        "gram13": gram13,
    }


def audit_episode(episode: dict, arm: str) -> dict:
    require(isinstance(episode, dict), "episode_not_object")
    require(set(episode) == EPISODE_FIELDS, "episode_field_contract")
    require(episode["schema"] == SCHEMA, "episode_schema")
    width_factor = has_width_factor(arm)
    require(episode["board"] == BOARD_NAMES[width_factor], "episode_board")
    require(episode["board_seed"] == BOARD_SEEDS[width_factor], "episode_board_seed")
    require(episode["split"] == "train", "episode_split")
    require(episode["prompt_style"] == "core", "episode_prompt_style")
    require(isinstance(episode["id"], str) and episode["id"], "episode_id")

    width = exact_int(episode["width"], "episode_width")
    left = exact_int(episode["left"], "episode_left")
    right = exact_int(episode["right"], "episode_right")
    operation = episode["operation"]
    require(operation in ("add", "sub"), "episode_operation")
    require(left >= 0 and right >= 0, "episode_negative_operand")
    require(operation != "sub" or left >= right, "episode_negative_subtraction")

    rebuilt = canonical_state(initial_state(operation, left, right, width))
    require(episode["initial_state"] == rebuilt, "episode_initial_state")
    state = parse_state(rebuilt)
    require(state is not None, "episode_initial_parse")
    expected_lines = episode["expected_states"]
    require(isinstance(expected_lines, list), "episode_expected_states_type")
    require(len(expected_lines) == width, "episode_transition_count")

    arithmetic = set()
    width_positions = set()
    controls = set()
    terminal_input = None
    for index, expected_line in enumerate(expected_lines):
        require(state is not None, "episode_state_parse")
        position = int(state["p"])
        require(position == index, "episode_position_sequence")
        arithmetic.add(
            (
                operation,
                int(state["c"]),
                int(state["a"][position]),
                int(state["b"][position]),
            )
        )
        width_positions.add((width, position))
        controls.add((width, position, operation, int(state["c"])))
        if position == width - 1:
            terminal_input = int(state["c"])
        expected = apply_microstep(state)
        require(expected_line == canonical_state(expected), "episode_transition_target")
        state = parse_state(expected_line)

    require(state is not None and bool(state["z"]), "episode_terminal_state")
    terminal_output = int(state["c"])
    require(terminal_input is not None, "episode_terminal_input")
    terminal_class = "{}{}".format(terminal_input, terminal_output)
    require(episode["terminal_input"] == terminal_input, "episode_terminal_input_label")
    require(
        episode["terminal_output"] == terminal_output, "episode_terminal_output_label"
    )
    require(episode["terminal_class"] == terminal_class, "episode_terminal_class_label")
    require(
        exact_int(episode["expected_answer"], "episode_expected_answer_type")
        == state_answer(state),
        "episode_expected_answer",
    )
    require(operation != "sub" or terminal_output == 0, "episode_terminal_borrow")

    multiplicity = exact_int(
        episode["control_terminal_multiplicity"],
        "episode_control_terminal_multiplicity",
    )
    budget_positions = episode["budget_transition_positions"]
    require(isinstance(budget_positions, list), "episode_budget_positions_type")
    budget_tuple = tuple(
        exact_int(value, "episode_budget_position") for value in budget_positions
    )
    require(
        tuple(sorted(budget_tuple)) == budget_tuple, "episode_budget_positions_order"
    )
    require(
        len(set(budget_tuple)) == len(budget_tuple), "episode_budget_positions_unique"
    )
    require(
        all(0 <= position < width - 1 for position in budget_tuple),
        "episode_budget_position_range",
    )
    expected_budget_count = 1 if multiplicity == 0 else max(0, multiplicity - 1)
    require(len(budget_tuple) == expected_budget_count, "episode_budget_position_count")
    if operation == "sub":
        require(multiplicity == 1, "episode_sub_control_multiplicity")
    elif terminal_output == 1:
        require(multiplicity == 0, "episode_add_output_one_control_multiplicity")
    else:
        require(1 <= multiplicity <= width, "episode_add_output_zero_multiplicity")

    designated = episode["designated_arithmetic_class"]
    if designated is not None:
        require(
            isinstance(designated, list) and len(designated) == 4,
            "episode_designated_class_shape",
        )
        designated_tuple = (
            designated[0],
            exact_int(designated[1], "episode_designated_carry"),
            exact_int(designated[2], "episode_designated_left"),
            exact_int(designated[3], "episode_designated_right"),
        )
        require(
            designated_tuple in required_arithmetic_classes(),
            "episode_designated_class_domain",
        )
        require(designated_tuple in arithmetic, "episode_designated_class_miss")
    else:
        designated_tuple = None

    return {
        "episode": episode,
        "id": episode["id"],
        "width": width,
        "operation": operation,
        "signature": (width, left, right),
        "complete_signature": (width, operation, left, right),
        "terminal_class": terminal_class,
        "terminal_input": terminal_input,
        "terminal_output": terminal_output,
        "control_terminal_multiplicity": multiplicity,
        "budget_transition_positions": budget_tuple,
        "arithmetic": arithmetic,
        "width_positions": width_positions,
        "controls": controls,
        "designated": designated_tuple,
    }


def expected_row_slots(episode_record: dict, arm: str) -> set[tuple]:
    width = episode_record["width"]
    terminal_index = width - 1
    multiplicity = episode_record["control_terminal_multiplicity"]
    budget_positions = episode_record["budget_transition_positions"]
    slots = {("digit", index, "canonical", 0) for index in range(width)} | {
        ("final", width, "canonical", 0)
    }
    if has_term_factor(arm):
        variant_positions = [terminal_index] + list(range(terminal_index))
        variant_by_position = {
            position: slot
            for slot, position in enumerate(
                variant_positions[: len(budget_positions)], 1
            )
        }
        for index in range(width):
            slot = variant_by_position.get(index, 0)
            role = "term_allocation" if slot else "canonical"
            slots.add(("transition", index, role, slot))
        return slots

    omitted = set(budget_positions) if multiplicity > 1 else set()
    for index in range(width):
        if index == terminal_index and multiplicity == 0:
            continue
        if index in omitted:
            continue
        slots.add(("transition", index, "canonical", 0))
    if multiplicity == 0:
        slots.add(("transition", budget_positions[0], "nonterminal_control", 1))
    elif multiplicity > 1:
        for slot in range(1, multiplicity):
            slots.add(("transition", terminal_index, "terminal_reallocation", slot))
    return slots


def audit_row(row: dict, arm: str, episode_record: dict) -> dict:
    require(isinstance(row, dict), "row_not_object")
    kind = row.get("kind")
    require(kind in ROW_FIELDS_BY_KIND, "row_kind")
    require(set(row) == ROW_FIELDS_BY_KIND[kind], "row_field_contract")
    require(row["schema"] == SCHEMA, "row_schema")
    require(row["arm"] == arm, "row_arm")
    require(row["seed"] == ARM_SEEDS[arm], "row_seed")
    require(row["term_factor"] is has_term_factor(arm), "row_term_factor")
    require(row["width_factor"] is has_width_factor(arm), "row_width_factor")
    require(row["board"] == episode_record["episode"]["board"], "row_board")
    require(
        row["board_seed"] == episode_record["episode"]["board_seed"],
        "row_board_seed",
    )
    require(row["split"] == "train", "row_split")
    require(row["prompt_style"] == "core", "row_prompt_style")
    require(row["training_group"] == TRAINING_GROUP, "row_training_group")
    require(row["source"] == SOURCE_BY_KIND[kind], "row_source")
    require(
        isinstance(row["question"], str)
        and row["question"] == row["completion_prompt"]
        and bool(row["question"])
        and isinstance(row["response"], str)
        and bool(row["response"]),
        "row_completion_boundary",
    )

    episode = episode_record["episode"]
    require(row["episode_id"] == episode["id"], "row_episode_id")
    width = exact_int(row["width"], "row_width")
    require(width == episode_record["width"], "row_episode_width")
    require(row["operation"] == episode_record["operation"], "row_episode_operation")
    require(row["terminal_class"] == episode["terminal_class"], "row_terminal_class")
    require(row["terminal_input"] == episode["terminal_input"], "row_terminal_input")
    require(row["terminal_output"] == episode["terminal_output"], "row_terminal_output")
    index = exact_int(row["transition_index"], "row_transition_index")
    allocation_role = row["allocation_role"]
    allocation_slot = exact_int(row["allocation_slot"], "row_allocation_slot")
    require(isinstance(allocation_role, str), "row_allocation_role")
    slot_key = (kind, index, allocation_role, allocation_slot)
    require(
        slot_key in expected_row_slots(episode_record, arm), "row_allocation_contract"
    )

    if kind == "transition":
        require(0 <= index < width, "transition_index_range")
        expected_input = (
            episode["initial_state"]
            if index == 0
            else episode["expected_states"][index - 1]
        )
        require(row["state"] == expected_input, "transition_episode_state")
        state = parse_state(row["state"])
        require(state is not None, "transition_state_parse")
        expected_line = canonical_state(apply_microstep(state))
        require(
            expected_line == episode["expected_states"][index],
            "transition_episode_target",
        )
        require(row["expected_state"] == expected_line, "transition_witness")
        require(row["response"] == expected_line, "transition_response")
        expected_prompt = microstep_prompt(state, style="core")
        if allocation_slot:
            expected_prompt = "{}\n{}".format(
                expected_prompt, ALLOCATION_SUFFIX.format(allocation_slot)
            )
        require(row["question"] == expected_prompt, "transition_prompt")
        position = int(state["p"])
        require(position == index, "transition_position")
        context = (
            str(state["op"]),
            int(state["c"]),
            int(state["a"][position]),
            int(state["b"][position]),
        )
        expected_state = parse_state(expected_line)
        require(expected_state is not None, "transition_expected_parse")
        terminal_class = None
        if index == width - 1:
            terminal_class = "{}{}".format(state["c"], expected_state["c"])
            require(
                terminal_class == row["terminal_class"], "transition_terminal_class"
            )
        return {
            "slot": slot_key,
            "kind": kind,
            "width": width,
            "context": context,
            "width_position": (width, position),
            "control": (width, position, state["op"], int(state["c"])),
            "terminal_class": terminal_class,
        }

    if kind == "digit":
        require(
            allocation_role == "canonical" and allocation_slot == 0, "digit_allocation"
        )
        digit_index = exact_int(row["digit_index"], "digit_index_type")
        expected_digit = exact_int(row["expected_digit"], "digit_expected_type")
        require(index == digit_index, "digit_transition_index")
        require(0 <= digit_index < width, "digit_index_range")
        expected_state_line = episode["expected_states"][digit_index]
        require(row["state"] == expected_state_line, "digit_episode_state")
        state = parse_state(row["state"])
        require(state is not None, "digit_state_parse")
        digit = state_digit(state, digit_index)
        require(expected_digit == digit, "digit_witness")
        require(row["response"] == "digit={}".format(digit), "digit_response")
        require(
            row["question"] == digit_prompt(state, digit_index, style="core"),
            "digit_prompt",
        )
        return {
            "slot": slot_key,
            "kind": kind,
            "width": width,
            "context": None,
            "width_position": None,
            "control": None,
            "terminal_class": None,
        }

    require(allocation_role == "canonical" and allocation_slot == 0, "final_allocation")
    require(index == width, "final_transition_index")
    require(row["state"] == episode["expected_states"][-1], "final_episode_state")
    state = parse_state(row["state"])
    require(state is not None and bool(state["z"]), "final_state_parse")
    answer = state_answer(state)
    require(
        exact_int(row["expected_answer"], "final_expected_type") == answer,
        "final_witness",
    )
    require(row["response"] == "answer={}".format(answer), "final_response")
    require(row["question"] == final_prompt(state, style="core"), "final_prompt")
    return {
        "slot": slot_key,
        "kind": kind,
        "width": width,
        "context": None,
        "width_position": None,
        "control": None,
        "terminal_class": None,
    }


class TokenAccounting:
    def __init__(self, tokenizer_snapshot: ExactByteSnapshot, pack_length: int):
        try:
            import tokenizers as tokenizers_package
            from tokenizers import Tokenizer
        except ImportError as exc:
            raise SystemExit(
                "tokenizer accounting requires the repo tokenizers package"
            ) from exc
        tokenizer_snapshot.verify()
        self.snapshot = tokenizer_snapshot
        self.tokenizers_version = tokenizers_package.__version__
        try:
            tokenizer_text = tokenizer_snapshot.payload.decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise SystemExit("tokenizer is not UTF-8 JSON") from exc
        self.tokenizer = Tokenizer.from_str(tokenizer_text)
        self.eos_id = self.tokenizer.token_to_id("<|endoftext|>")
        if self.eos_id is None:
            raise SystemExit("tokenizer has no <|endoftext|> token")
        self.pack_length = pack_length
        self.total = Counter()
        self.by_kind: defaultdict[str, Counter] = defaultdict(Counter)
        self.token_buffer: list[int] = []
        self.mask_buffer: list[int] = []
        self.x_spool = tempfile.TemporaryFile()
        self.y_spool = tempfile.TemporaryFile()
        self.packed_sequences = 0
        self.packed_supervised_tokens = 0
        self.supervised_per_pack: list[int] = []
        self.cached_report: dict | None = None
        self.blank_lines = 0

    @staticmethod
    def _int64_bytes(values: list[int]) -> bytes:
        payload = array("q", values)
        if payload.itemsize != 8:
            raise RuntimeError("platform has no 64-bit signed q array")
        if sys.byteorder != "little":
            payload.byteswap()
        return payload.tobytes()

    def _drain_packs(self) -> None:
        while len(self.token_buffer) >= self.pack_length + 1:
            x_values = self.token_buffer[: self.pack_length]
            y_values = [
                self.token_buffer[index + 1] if self.mask_buffer[index + 1] else -1
                for index in range(self.pack_length)
            ]
            self.x_spool.write(self._int64_bytes(x_values))
            self.y_spool.write(self._int64_bytes(y_values))
            supervised_tokens = sum(value != -1 for value in y_values)
            self.packed_sequences += 1
            self.packed_supervised_tokens += supervised_tokens
            self.supervised_per_pack.append(supervised_tokens)
            del self.token_buffer[: self.pack_length]
            del self.mask_buffer[: self.pack_length]

    def add(self, row: dict) -> None:
        if self.cached_report is not None:
            raise RuntimeError("token accounting was already finalized")
        prompt = "Question: {}\nAnswer:".format(row["question"])
        response = str(row["response"]).strip()
        separator = "" if prompt.endswith((" ", "\n", "\t")) else " "
        prompt_ids, token_ids, mask = encode_supervised_example(
            self.tokenizer, prompt, separator + response, self.eos_id
        )
        if not token_ids or token_ids[-1] != self.eos_id or mask[-1] != 1:
            raise RuntimeError("canonical completion/EOS supervision was not preserved")
        prompt_count = len(prompt_ids)
        response_count = len(token_ids) - prompt_count - 1
        full_count = len(token_ids)
        counters = (self.total, self.by_kind[row["kind"]])
        for counter in counters:
            counter["rows_seen"] += 1
            counter["prompt_tokens"] += prompt_count
            counter["response_tokens"] += response_count
            counter["supervised_tokens"] += response_count + 1
            counter["full_tokens"] += full_count
            counter["over_pack_length_rows"] += int(full_count > self.pack_length)
            counter["max_full_tokens"] = max(counter["max_full_tokens"], full_count)
        if full_count > self.pack_length:
            return
        for counter in counters:
            counter["accepted_rows"] += 1
            counter["accepted_tokens"] += full_count
            counter["accepted_supervised_tokens"] += sum(mask)
        self.token_buffer.extend(token_ids)
        self.mask_buffer.extend(mask)
        self._drain_packs()

    def add_blank_line(self) -> None:
        if self.cached_report is not None:
            raise RuntimeError("token accounting was already finalized")
        self.blank_lines += 1

    @staticmethod
    def _serialize(counter: Counter) -> dict[str, int]:
        return {
            "rows_seen": counter["rows_seen"],
            "accepted_rows": counter["accepted_rows"],
            "prompt_tokens": counter["prompt_tokens"],
            "response_tokens": counter["response_tokens"],
            "supervised_tokens": counter["supervised_tokens"],
            "full_tokens": counter["full_tokens"],
            "accepted_tokens": counter["accepted_tokens"],
            "accepted_supervised_tokens": counter["accepted_supervised_tokens"],
            "over_pack_length_rows": counter["over_pack_length_rows"],
            "max_full_tokens": counter["max_full_tokens"],
        }

    def _canonical_exact_budget_receipt(self) -> dict:
        contract = dict(CANONICAL_EXACT_BUDGET)
        available = len(self.supervised_per_pack)
        receipt = {
            **contract,
            "available_packs": available,
            "available_supervised_tokens": sum(self.supervised_per_pack),
            "consumed_packs": CANONICAL_REQUIRED_PACKS,
            "forward_token_positions": (
                CANONICAL_REQUIRED_PACKS * CANONICAL_EXACT_BUDGET["pack_len"]
            ),
            "selection_algorithm": (
                "sha256(seed_u64_le||pack_index_u64_le)_lexicographic_v1"
            ),
            "target_thinning": False,
            "completion_and_eos_supervision_intact": True,
            "admitted": False,
        }
        if (
            self.pack_length != CANONICAL_EXACT_BUDGET["pack_len"]
            or available < CANONICAL_REQUIRED_PACKS
        ):
            return receipt

        seed_bytes = CANONICAL_EXACT_BUDGET["seed"].to_bytes(8, "little", signed=False)
        order = sorted(
            range(available),
            key=lambda index: hashlib.sha256(
                seed_bytes + index.to_bytes(8, "little", signed=False)
            ).digest(),
        )
        consumed = order[:CANONICAL_REQUIRED_PACKS]
        batch_size = CANONICAL_EXACT_BUDGET["batch_size"]
        update_supervised = [
            sum(
                self.supervised_per_pack[index]
                for index in consumed[offset : offset + batch_size]
            )
            for offset in range(0, len(consumed), batch_size)
        ]
        receipt.update(
            {
                "admitted": bool(update_supervised) and min(update_supervised) > 0,
                "consumed_pack_order_sha256": hashlib.sha256(
                    self._int64_bytes(consumed)
                ).hexdigest(),
                "consumed_supervised_tokens": sum(update_supervised),
                "supervised_tokens_per_update_max": max(update_supervised),
                "supervised_tokens_per_update_min": min(update_supervised),
                "supervised_tokens_per_update_sha256": hashlib.sha256(
                    self._int64_bytes(update_supervised)
                ).hexdigest(),
            }
        )
        return receipt

    def report(self) -> dict:
        if self.cached_report is not None:
            return self.cached_report
        shape = (
            (self.packed_sequences, self.pack_length) if self.packed_sequences else (0,)
        )
        digest = hashlib.sha256()
        expected_bytes = self.packed_sequences * self.pack_length * 8
        spool_sizes = []
        for name, spool in (("X", self.x_spool), ("Y", self.y_spool)):
            digest.update(
                json.dumps(
                    {"name": name, "shape": shape, "dtype": "<i8"},
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("ascii")
            )
            spool.seek(0, os.SEEK_END)
            spool_sizes.append(spool.tell())
            spool.seek(0)
            for block in iter(lambda: spool.read(1024 * 1024), b""):
                digest.update(block)

        encoded_tokens = self.total["accepted_tokens"]
        encoded_supervised = self.total["accepted_supervised_tokens"]
        first_unused = self.packed_sequences * self.pack_length + 1
        group = {
            "encoded_tokens": encoded_tokens,
            "encoded_supervised_tokens": encoded_supervised,
            "packed_sequences": self.packed_sequences,
            "packed_forward_positions": self.packed_sequences * self.pack_length,
            "unpacked_tail_tokens": max(0, encoded_tokens - first_unused),
            "unpacked_tail_supervised_tokens": int(sum(self.mask_buffer[1:])),
        }
        production_stats = {
            "examples": self.total["accepted_rows"],
            "encoded_tokens": encoded_tokens,
            "encoded_supervised_tokens": encoded_supervised,
            "packed_sequences": self.packed_sequences,
            "packed_forward_positions": self.packed_sequences * self.pack_length,
            "packed_supervised_tokens": self.packed_supervised_tokens,
            "pack_len": self.pack_length,
            "packing_sha256": digest.hexdigest(),
            "skipped": {
                "blank_lines": self.blank_lines,
                "invalid_fields": 0,
                "too_long": self.total["over_pack_length_rows"],
            },
            "groups": {"default": group},
        }
        self.cached_report = {
            "tokenizer": self.snapshot.logical_path,
            "tokenizer_bytes": len(self.snapshot.payload),
            "tokenizer_capture": self.snapshot.binding["capture"],
            "tokenizer_sha256": self.snapshot.sha256,
            "tokenizers_version": self.tokenizers_version,
            "eos_id": self.eos_id,
            "pack_length": self.pack_length,
            "encoding_boundary": (
                "Exact canonical train/sft.py semantics without a prompt override: "
                "Question: {question}\\nAnswer: and stripped response are independently "
                "encoded with one inserted separator space; EOS is supervised."
            ),
            "production_build_packed": production_stats,
            "canonical_exact_budget": self._canonical_exact_budget_receipt(),
            "packing_invariants": {
                "x_spool_bytes": spool_sizes[0],
                "y_spool_bytes": spool_sizes[1],
                "expected_bytes_each": expected_bytes,
                "byte_lengths_match": spool_sizes == [expected_bytes, expected_bytes],
                "accepted_token_sum_matches": encoded_tokens
                == sum(counter["accepted_tokens"] for counter in self.by_kind.values()),
                "accepted_supervised_sum_matches": encoded_supervised
                == sum(
                    counter["accepted_supervised_tokens"]
                    for counter in self.by_kind.values()
                ),
            },
            "overall": self._serialize(self.total),
            "by_kind": {
                kind: self._serialize(counter)
                for kind, counter in sorted(self.by_kind.items())
            },
        }
        self.x_spool.close()
        self.y_spool.close()
        return self.cached_report


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


def publish_bytes_no_replace(path: Path, payload: bytes, before_link=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    directory = os.open(
        path.parent,
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    parent = os.fstat(directory)
    path_parent = os.stat(path.parent, follow_symlinks=False)
    if not stat.S_ISDIR(parent.st_mode) or (parent.st_dev, parent.st_ino) != (
        path_parent.st_dev,
        path_parent.st_ino,
    ):
        os.close(directory)
        raise RuntimeError("audit parent changed while opening")
    temporary = ".{}.private.{}.{}".format(path.name, os.getpid(), time.time_ns())
    descriptor = -1
    try:
        descriptor = os.open(
            temporary,
            os.O_RDWR | os.O_CREAT | os.O_EXCL,
            0o600,
            dir_fd=directory,
        )
        view = memoryview(payload)
        written = 0
        while written < len(view):
            written += os.write(descriptor, view[written:])
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
        identity = _inode_identity(os.fstat(descriptor))
        digest = sha256_bytes(payload)
        if before_link is not None:
            before_link(path)
        os.link(
            temporary,
            path.name,
            src_dir_fd=directory,
            dst_dir_fd=directory,
            follow_symlinks=False,
        )
        os.fsync(directory)
        published = os.stat(path.name, dir_fd=directory, follow_symlinks=False)
        require(_inode_identity(published) == identity, "audit_publication_identity")
        os.lseek(descriptor, 0, os.SEEK_SET)
        captured = b""
        while len(captured) < len(payload):
            block = os.read(descriptor, min(1024 * 1024, len(payload) - len(captured)))
            if not block:
                break
            captured += block
        require(sha256_bytes(captured) == digest, "audit_publication_sha256")
    finally:
        try:
            os.unlink(temporary, dir_fd=directory)
        except FileNotFoundError:
            pass
        if descriptor >= 0:
            os.close(descriptor)
        os.close(directory)


def write_report(path: Path, report: dict) -> bytes:
    payload = (
        json.dumps(report, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("ascii")
    publish_bytes_no_replace(path, payload)
    return payload


def expected_per_width_rows(
    allocations: dict[int, int],
) -> dict[int, dict[str, int]]:
    return {
        width: {
            "transition": width * count,
            "digit": width * count,
            "final": count,
            "rows": (2 * width + 1) * count,
        }
        for width, count in sorted(allocations.items())
    }


def counter_sha256(counter: Counter) -> str:
    payload = json.dumps(
        [[list(key), value] for key, value in sorted(counter.items())],
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def observe_episode_board(
    episodes_snapshot: ExactByteSnapshot,
    arm: str,
    heldout_signatures: set[tuple[int, int, int]],
    failures: Counter,
    examples: list[dict],
) -> dict:
    records: dict[str, dict] = {}
    raw = 0
    duplicate_ids = 0
    duplicate_signatures = 0
    duplicate_complete_signatures = 0
    signatures = set()
    complete_signatures = set()
    widths = Counter()
    operations = Counter()
    operations_by_width: defaultdict[int, Counter] = defaultdict(Counter)
    terminals = {"add": Counter(), "sub": Counter()}
    terminals_by_width: defaultdict[int, dict[str, Counter]] = defaultdict(
        lambda: {"add": Counter(), "sub": Counter()}
    )
    control_terminals = {"add": Counter(), "sub": Counter()}
    control_terminals_by_width: defaultdict[int, dict[str, Counter]] = defaultdict(
        lambda: {"add": Counter(), "sub": Counter()}
    )
    arithmetic = set()
    positions = set()
    controls = set()
    designated = Counter()
    heldout_hits = set()
    penultimate_pairs: defaultdict[tuple, Counter] = defaultdict(Counter)
    terminal_pairs: defaultdict[tuple, Counter] = defaultdict(Counter)
    all_position_pairs = Counter()
    episode_ids = []
    blank_lines = 0

    with episodes_snapshot.text() as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                blank_lines += 1
                continue
            raw += 1
            try:
                episode = json.loads(line)
                record = audit_episode(episode, arm)
                if record["id"] in records:
                    duplicate_ids += 1
                    raise ContractError("duplicate_episode_id")
                if record["signature"] in signatures:
                    duplicate_signatures += 1
                    raise ContractError("duplicate_episode_signature")
                if record["complete_signature"] in complete_signatures:
                    duplicate_complete_signatures += 1
                    raise ContractError("duplicate_complete_episode_signature")
                records[record["id"]] = record
                episode_ids.append(record["id"])
                signatures.add(record["signature"])
                complete_signatures.add(record["complete_signature"])
                if record["signature"] in heldout_signatures:
                    heldout_hits.add(record["signature"])
                width = record["width"]
                operation = record["operation"]
                terminal_class = record["terminal_class"]
                multiplicity = record["control_terminal_multiplicity"]
                widths[width] += 1
                operations[operation] += 1
                operations_by_width[width][operation] += 1
                terminals[operation][terminal_class] += 1
                terminals_by_width[width][operation][terminal_class] += 1
                control_terminals[operation][terminal_class] += multiplicity
                control_terminals_by_width[width][operation][terminal_class] += (
                    multiplicity
                )
                arithmetic.update(record["arithmetic"])
                positions.update(record["width_positions"])
                controls.update(record["controls"])
                if record["designated"] is not None:
                    designated[record["designated"]] += 1

                state = parse_state(episode["initial_state"])
                for expected_line in episode["expected_states"]:
                    require(state is not None, "episode_pair_state_parse")
                    position = int(state["p"])
                    pair = (int(state["a"][position]), int(state["b"][position]))
                    all_position_pairs[(width, position, operation, *pair)] += 1
                    if position == width - 2:
                        penultimate_pairs[(width, operation, record["terminal_input"])][
                            pair
                        ] += 1
                    if position == width - 1:
                        terminal_pairs[(width, operation, terminal_class)][pair] += 1
                    state = parse_state(expected_line)
            except (
                ContractError,
                KeyError,
                TypeError,
                ValueError,
                json.JSONDecodeError,
            ) as exc:
                record_failure(failures, examples, "episode", line_number, exc)

    return {
        "records": records,
        "raw": raw,
        "duplicate_ids": duplicate_ids,
        "duplicate_signatures": duplicate_signatures,
        "duplicate_complete_signatures": duplicate_complete_signatures,
        "signatures": signatures,
        "complete_signatures": complete_signatures,
        "widths": widths,
        "operations": operations,
        "operations_by_width": operations_by_width,
        "terminals": terminals,
        "terminals_by_width": terminals_by_width,
        "control_terminals": control_terminals,
        "control_terminals_by_width": control_terminals_by_width,
        "arithmetic": arithmetic,
        "positions": positions,
        "controls": controls,
        "designated": designated,
        "heldout_hits": heldout_hits,
        "penultimate_pairs": penultimate_pairs,
        "terminal_pairs": terminal_pairs,
        "all_position_pairs": all_position_pairs,
        "episode_ids": episode_ids,
        "blank_lines": blank_lines,
    }


def observe_rows(
    data_snapshot: ExactByteSnapshot,
    arm: str,
    episode_records: dict[str, dict],
    heldout: dict,
    failures: Counter,
    examples: list[dict],
    category: str,
    token_accounting: TokenAccounting | None = None,
) -> dict:
    raw = 0
    valid = 0
    duplicate_prompts = 0
    duplicate_slots = 0
    seen_prompts = set()
    row_slots: defaultdict[str, set[tuple]] = defaultdict(set)
    kind_counts = Counter()
    width_kind_counts: defaultdict[int, Counter] = defaultdict(Counter)
    terminal_counts = {"add": Counter(), "sub": Counter()}
    terminal_counts_by_width: defaultdict[int, dict[str, Counter]] = defaultdict(
        lambda: {"add": Counter(), "sub": Counter()}
    )
    arithmetic = set()
    positions = set()
    controls = set()
    position_counts = Counter()
    allocation_formats = Counter()
    visible_features = Counter()
    current_pair_contexts = Counter()
    observed_arms = Counter()
    exact_heldout_hits = set()
    gram13_hits = set()
    contamination_examples = []
    blank_lines = 0

    with data_snapshot.text() as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                blank_lines += 1
                if token_accounting is not None:
                    token_accounting.add_blank_line()
                continue
            raw += 1
            try:
                row = json.loads(line)
                prompt_key = normalized(row.get("completion_prompt", ""))
                require(bool(prompt_key), "empty_normalized_prompt")
                if prompt_key in heldout["prompt_keys"]:
                    exact_heldout_hits.add(prompt_key)
                    if len(contamination_examples) < 8:
                        contamination_examples.append(
                            {
                                "kind": "exact_normalized_prompt",
                                "line": line_number,
                                "prompt": prompt_key,
                            }
                        )
                shared_grams = (
                    ngrams(row.get("completion_prompt", "")) & heldout["gram13"]
                )
                gram13_hits.update(shared_grams)
                if shared_grams and len(contamination_examples) < 8:
                    contamination_examples.append(
                        {
                            "kind": "literal_normalized_word_13gram",
                            "line": line_number,
                            "gram": min(shared_grams),
                        }
                    )
                if prompt_key in seen_prompts:
                    duplicate_prompts += 1
                    raise ContractError("duplicate_normalized_prompt")
                seen_prompts.add(prompt_key)
                episode_id = row.get("episode_id")
                require(episode_id in episode_records, "row_unknown_episode")
                episode_record = episode_records[episode_id]
                result = audit_row(row, arm, episode_record)
                if result["slot"] in row_slots[episode_id]:
                    duplicate_slots += 1
                    raise ContractError("duplicate_episode_row_slot")
                row_slots[episode_id].add(result["slot"])
                valid += 1
                observed_arms[str(row["arm"])] += 1
                kind_counts[result["kind"]] += 1
                width_kind_counts[result["width"]][result["kind"]] += 1
                allocation_formats[row["allocation_slot"]] += 1
                episode = episode_record["episode"]
                visible_features[
                    (
                        result["kind"],
                        episode_record["width"],
                        episode_record["operation"],
                        int(episode["left"]),
                        int(episode["right"]),
                    )
                ] += 1
                if result["context"] is not None:
                    arithmetic.add(result["context"])
                    positions.add(result["width_position"])
                    controls.add(result["control"])
                    width, position = result["width_position"]
                    position_counts[(width, position, result["context"][0])] += 1
                    current_pair_contexts[(width, position, *result["context"])] += 1
                if result["terminal_class"] is not None:
                    operation = episode_record["operation"]
                    terminal_counts[operation][result["terminal_class"]] += 1
                    terminal_counts_by_width[result["width"]][operation][
                        result["terminal_class"]
                    ] += 1
                if token_accounting is not None:
                    token_accounting.add(row)
            except (
                ContractError,
                KeyError,
                TypeError,
                ValueError,
                json.JSONDecodeError,
            ) as exc:
                record_failure(failures, examples, category, line_number, exc)

    incomplete = 0
    for episode_id, record in episode_records.items():
        incomplete += int(row_slots[episode_id] != expected_row_slots(record, arm))
    return {
        "raw": raw,
        "valid": valid,
        "duplicate_prompts": duplicate_prompts,
        "duplicate_slots": duplicate_slots,
        "seen_prompts": seen_prompts,
        "kind_counts": kind_counts,
        "width_kind_counts": width_kind_counts,
        "terminal_counts": terminal_counts,
        "terminal_counts_by_width": terminal_counts_by_width,
        "arithmetic": arithmetic,
        "positions": positions,
        "controls": controls,
        "position_counts": position_counts,
        "allocation_formats": allocation_formats,
        "visible_features": visible_features,
        "current_pair_contexts": current_pair_contexts,
        "observed_arms": observed_arms,
        "exact_heldout_hits": exact_heldout_hits,
        "gram13_hits": gram13_hits,
        "contamination_examples": contamination_examples,
        "incomplete": incomplete,
        "blank_lines": blank_lines,
    }


def audit(
    data_path: Path,
    episodes_path: Path,
    paired_data_path: Path,
    paired_episodes_path: Path,
    heldout_path: Path,
    out_path: Path,
    arm: str,
    mode: str,
    test_scale: int | None = None,
    tokenizer_path: Path | None = None,
    pack_length: int = 2048,
    generator_report_path: Path | None = None,
    paired_generator_report_path: Path | None = None,
) -> dict:
    validate_mode(mode, test_scale)
    artifact_paths = (
        data_path,
        episodes_path,
        paired_data_path,
        paired_episodes_path,
        heldout_path,
        out_path,
    ) + ((tokenizer_path,) if tokenizer_path is not None else ())
    artifact_paths += (
        (generator_report_path,) if generator_report_path is not None else ()
    )
    artifact_paths += (
        (paired_generator_report_path,)
        if paired_generator_report_path is not None
        else ()
    )
    ensure_pairwise_artifact_paths(artifact_paths)
    inputs = (
        data_path,
        episodes_path,
        paired_data_path,
        paired_episodes_path,
        heldout_path,
    )
    if not all(path.is_file() for path in inputs):
        raise SystemExit(
            "missing data, paired data, episodes, paired episodes, or heldout"
        )
    if pack_length < 2:
        raise SystemExit("--pack-length must be at least 2")
    if mode == "production" and tokenizer_path is None:
        raise SystemExit("production audit requires --tokenizer accounting")
    if mode == "production" and (
        generator_report_path is None or paired_generator_report_path is None
    ):
        raise SystemExit("production audit requires both generator reports")

    source_snapshots = capture_auditor_sources()
    input_snapshots = {
        "data": capture_exact_file(data_path, str(data_path.resolve())),
        "episodes": capture_exact_file(episodes_path, str(episodes_path.resolve())),
        "paired_data": capture_exact_file(
            paired_data_path, str(paired_data_path.resolve())
        ),
        "paired_episodes": capture_exact_file(
            paired_episodes_path, str(paired_episodes_path.resolve())
        ),
        "heldout": capture_exact_file(heldout_path, str(heldout_path.resolve())),
    }
    if tokenizer_path is not None:
        input_snapshots["tokenizer"] = capture_exact_file(
            tokenizer_path, str(tokenizer_path.resolve())
        )
    generator_report_snapshots = {}
    if generator_report_path is not None:
        generator_report_snapshots["primary"] = capture_exact_file(
            generator_report_path, str(generator_report_path.resolve())
        )
    if paired_generator_report_path is not None:
        generator_report_snapshots["counterpart"] = capture_exact_file(
            paired_generator_report_path,
            str(paired_generator_report_path.resolve()),
        )

    allocations = allocations_for_arm(arm, test_scale)
    target = structural_counts(allocations)
    expected_operations_by_width = operation_allocations(allocations)
    expected_operations = Counter()
    for counts in expected_operations_by_width.values():
        expected_operations.update(counts)
    expected_board_terminals = expected_terminal_counts(expected_operations)
    expected_board_by_width = stratified_terminal_counts(expected_operations_by_width)
    expected_control_by_width = expected_control_terminal_counts(
        expected_operations_by_width, expected_board_by_width
    )
    expected_control = {"add": Counter(), "sub": Counter()}
    for operations in expected_control_by_width.values():
        for operation, counts in operations.items():
            expected_control[operation].update(counts)
    expected_row_terminals = (
        expected_board_terminals if has_term_factor(arm) else expected_control
    )
    expected_row_by_width = (
        expected_board_by_width if has_term_factor(arm) else expected_control_by_width
    )
    required_arithmetic = required_arithmetic_classes()
    widths = tuple(sorted(allocations))
    required_positions = required_width_positions(widths)
    required_controls = required_control_contexts(widths)
    expected_designated = Counter({context: 1 for context in required_arithmetic})
    expected_ids = {
        "dfv4-{}-{:05d}".format(BOARD_NAMES[has_width_factor(arm)], index)
        for index in range(target["episodes"])
    }
    expected_row_kinds = Counter(
        {
            "transition": target["transitions"],
            "digit": target["transitions"],
            "final": target["episodes"],
        }
    )
    expected_position_counts = Counter(
        {
            (width, position, operation): counts[operation]
            for width, counts in expected_operations_by_width.items()
            for position in range(width)
            for operation in ("add", "sub")
        }
    )

    failures: Counter = Counter()
    examples: list[dict] = []
    heldout = observe_heldout(
        input_snapshots["heldout"], test_scale, failures, examples
    )
    board = observe_episode_board(
        input_snapshots["episodes"],
        arm,
        heldout["signatures"],
        failures,
        examples,
    )
    token_accounting = (
        TokenAccounting(input_snapshots["tokenizer"], pack_length)
        if tokenizer_path is not None
        else None
    )
    primary = observe_rows(
        input_snapshots["data"],
        arm,
        board["records"],
        heldout,
        failures,
        examples,
        "row",
        token_accounting,
    )
    counterpart = paired_arm(arm)
    paired = observe_rows(
        input_snapshots["paired_data"],
        counterpart,
        board["records"],
        heldout,
        failures,
        examples,
        "paired_row",
    )
    token_report = token_accounting.report() if token_accounting is not None else None
    generator_evidence = {}
    if generator_report_snapshots:
        for role, admitted_arm, data_name, episodes_name in (
            ("primary", arm, "data", "episodes"),
            ("counterpart", counterpart, "paired_data", "paired_episodes"),
        ):
            try:
                generator_evidence[role] = validate_generator_report(
                    generator_report_snapshots[role],
                    input_snapshots[data_name],
                    input_snapshots[episodes_name],
                    input_snapshots["heldout"],
                    admitted_arm,
                    mode,
                    source_snapshots,
                )
            except (ContractError, KeyError, TypeError, ValueError) as exc:
                record_failure(failures, examples, "generator_report", 0, exc)
    generator_reports_valid = set(generator_evidence) == {"primary", "counterpart"}

    actual_per_width_rows = {
        width: {
            "transition": primary["width_kind_counts"][width]["transition"],
            "digit": primary["width_kind_counts"][width]["digit"],
            "final": primary["width_kind_counts"][width]["final"],
            "rows": sum(primary["width_kind_counts"][width].values()),
        }
        for width in sorted(set(allocations) | set(primary["width_kind_counts"]))
    }
    heldout_counts = {key: heldout[key] for key in FROZEN_HELDOUT_COUNTS}
    heldout_solver_valid = (
        heldout["raw_top_level_episodes"] > 0
        and heldout["top_level_episodes"] == heldout["raw_top_level_episodes"]
        and heldout["branches"] == 2 * heldout["top_level_episodes"]
        and heldout["counterfactual_pairs"] == heldout["top_level_episodes"]
        and heldout["unique_signatures"] == heldout["branches"]
        and heldout["unique_normalized_prompts"] == heldout["controller_prompts"]
    )
    paired_board_hash_equal = (
        input_snapshots["episodes"].payload
        == input_snapshots["paired_episodes"].payload
    )
    pair_diversity = all(
        pair_distribution_contract(counts)
        for counts in (
            list(board["penultimate_pairs"].values())
            + list(board["terminal_pairs"].values())
        )
    )
    token_checks = {
        "tokenizer_row_accounting": token_report is None
        or token_report["overall"]["rows_seen"] == target["rows"],
        "tokenizer_no_overlength_rows": token_report is None
        or token_report["production_build_packed"]["skipped"]["too_long"] == 0,
        "tokenizer_no_zero_pack": token_report is None
        or token_report["production_build_packed"]["packed_sequences"] > 0,
        "tokenizer_packing_invariants": token_report is None
        or all(token_report["packing_invariants"].values()),
        "tokenizer_all_rows_accepted": token_report is None
        or token_report["production_build_packed"]["examples"] == target["rows"],
        "tokenizer_blank_line_parity": token_report is None
        or token_report["production_build_packed"]["skipped"]["blank_lines"]
        == primary["blank_lines"],
        "tokenizer_no_blank_lines": token_report is None
        or token_report["production_build_packed"]["skipped"]["blank_lines"] == 0,
        "tokenizer_canonical_exact_budget": token_report is None
        or mode == "test"
        or token_report["canonical_exact_budget"]["admitted"],
    }
    checks = {
        "generator_reports_exact_and_current": generator_reports_valid,
        "heldout_solver_valid": heldout_solver_valid,
        "heldout_no_blank_lines": heldout["blank_lines"] == 0,
        "heldout_frozen_sha256": mode == "test"
        or heldout["sha256"] == FROZEN_HELDOUT_SHA256,
        "heldout_frozen_counts": mode == "test"
        or heldout_counts == FROZEN_HELDOUT_COUNTS,
        "heldout_frozen_regimes": mode == "test"
        or heldout["regimes"] == FROZEN_HELDOUT_REGIMES,
        "zero_train_heldout_reserved_signature_hits": not board["heldout_hits"],
        "zero_train_heldout_exact_normalized_prompt_hits": not primary[
            "exact_heldout_hits"
        ]
        and not paired["exact_heldout_hits"],
        "zero_train_heldout_literal_13gram_hits": not primary["gram13_hits"]
        and not paired["gram13_hits"],
        "raw_episode_count": board["raw"] == target["episodes"],
        "episode_no_blank_lines": board["blank_lines"] == 0,
        "valid_episode_count": len(board["records"]) == target["episodes"],
        "episode_ids": set(board["records"]) == expected_ids,
        "unique_episode_ids": board["duplicate_ids"] == 0,
        "unique_episode_signatures": board["duplicate_signatures"] == 0,
        "unique_complete_episode_signatures": board["duplicate_complete_signatures"]
        == 0,
        "episode_width_allocations": board["widths"] == Counter(allocations),
        "episode_operation_allocations": board["operations"] == expected_operations,
        "episode_width_operation_allocations": all(
            board["operations_by_width"][width] == Counter(counts)
            for width, counts in expected_operations_by_width.items()
        ),
        "board_terminal_class_contract": board["terminals"] == expected_board_terminals
        and all(
            board["terminals_by_width"][width] == operations
            for width, operations in expected_board_by_width.items()
        ),
        "predeclared_control_terminal_contract": board["control_terminals"]
        == expected_control
        and all(
            board["control_terminals_by_width"][width] == operations
            for width, operations in expected_control_by_width.items()
        ),
        "designated_arithmetic_classes": board["designated"] == expected_designated,
        "episode_arithmetic_class_coverage": required_arithmetic <= board["arithmetic"],
        "episode_width_position_coverage": required_positions <= board["positions"],
        "episode_width_position_control_coverage": required_controls
        <= board["controls"],
        "nonconcentrated_terminal_and_penultimate_pairs": pair_diversity,
        "raw_row_count": primary["raw"] == target["rows"],
        "row_no_blank_lines": primary["blank_lines"] == 0,
        "data_arm_identity": primary["observed_arms"] == Counter({arm: target["rows"]}),
        "valid_row_count": primary["valid"] == target["rows"],
        "row_kind_counts": primary["kind_counts"] == expected_row_kinds,
        "row_width_counts": actual_per_width_rows
        == expected_per_width_rows(allocations),
        "unique_normalized_prompts": primary["duplicate_prompts"] == 0
        and len(primary["seen_prompts"]) == target["rows"],
        "unique_episode_row_slots": primary["duplicate_slots"] == 0,
        "complete_episode_row_slots": primary["incomplete"] == 0,
        "transition_terminal_class_contract": primary["terminal_counts"]
        == expected_row_terminals
        and all(
            primary["terminal_counts_by_width"][width] == operations
            for width, operations in expected_row_by_width.items()
        ),
        "transition_arithmetic_class_coverage": required_arithmetic
        <= primary["arithmetic"],
        "transition_width_position_coverage": required_positions
        <= primary["positions"],
        "transition_width_position_control_coverage": required_controls
        <= primary["controls"],
        "transition_position_distribution": primary["position_counts"]
        == expected_position_counts,
        "paired_board_literal_bytes": paired_board_hash_equal,
        "paired_board_episode_ids": paired_board_hash_equal,
        "paired_full_visible_operand_tape_features": primary["visible_features"]
        == paired["visible_features"],
        "paired_operand_magnitude_distribution": primary["visible_features"]
        == paired["visible_features"],
        "paired_position_distribution": primary["position_counts"]
        == paired["position_counts"]
        == expected_position_counts,
        "paired_digit_pair_distribution": paired_board_hash_equal,
        "paired_allocation_format_distribution": primary["allocation_formats"]
        == paired["allocation_formats"],
        "paired_row_count": paired["raw"] == target["rows"]
        and paired["valid"] == target["rows"],
        "paired_row_no_blank_lines": paired["blank_lines"] == 0,
        "paired_data_arm_identity": paired["observed_arms"]
        == Counter({counterpart: target["rows"]}),
        "paired_unique_prompts_and_slots": paired["duplicate_prompts"] == 0
        and paired["duplicate_slots"] == 0
        and paired["incomplete"] == 0,
        **token_checks,
    }
    mechanical_pass = not failures and all(checks.values())
    admitted_arm = (
        next(iter(primary["observed_arms"]))
        if primary["observed_arms"] == Counter({arm: target["rows"]})
        else None
    )
    production_admission = (
        mode == "production" and mechanical_pass and admitted_arm == arm
    )
    local_keys = set(primary["current_pair_contexts"]) | set(
        paired["current_pair_contexts"]
    )
    local_l1 = sum(
        abs(
            primary["current_pair_contexts"][key] - paired["current_pair_contexts"][key]
        )
        for key in local_keys
    )
    episode_ids_payload = "".join(
        episode_id + "\n" for episode_id in board["episode_ids"]
    ).encode("ascii")
    report = {
        "audit": "digitwise_factorial_v4_admission",
        "receipt_schema": "shohin-factorial-v4-production-admission-v1",
        "schema": SCHEMA,
        "declared_arm": arm,
        "admitted_arm": admitted_arm,
        "paired_arm": counterpart,
        "declared_factors": {
            "term": has_term_factor(arm),
            "width": has_width_factor(arm),
        },
        "seed": ARM_SEEDS[arm],
        "board": BOARD_NAMES[has_width_factor(arm)],
        "board_seed": BOARD_SEEDS[has_width_factor(arm)],
        "mode": mode,
        "production_contract": mode == "production",
        "test_scale": test_scale,
        "mechanical_pass": mechanical_pass,
        "runtime": {
            "python": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "tokenizers": (
                token_report["tokenizers_version"] if token_report is not None else None
            ),
        },
        "test_mechanics_pass": mechanical_pass if mode == "test" else None,
        "production_admission": production_admission,
        "admission_pass": production_admission,
        "checks": dict(sorted(checks.items())),
        "failures": dict(sorted(failures.items())),
        "failure_examples": examples,
        "target": target,
        "data": input_snapshots["data"].logical_path,
        "data_sha256": input_snapshots["data"].sha256,
        "episodes": input_snapshots["episodes"].logical_path,
        "episodes_sha256": input_snapshots["episodes"].sha256,
        "inputs": {
            name: snapshot.binding for name, snapshot in input_snapshots.items()
        },
        "generator_reports": generator_evidence,
        "independent_audit": {
            "auditor_recomputed_solver_and_contract": True,
            "expected_receipt_sha256_must_be_supplied_to_training": True,
            "generator_implementation_imported": False,
            "generator_reports_used_only_as_bound_provenance": True,
            "remote_attestation": False,
        },
        "scientific_contract": {
            "heldout_splits": sorted(heldout["regimes"]),
            "row_sources": sorted(SOURCE_BY_KIND.values()),
            "schema": SCHEMA,
            "training_group": TRAINING_GROUP,
            "training_split": "train",
        },
        "source_manifests": {
            "generator": manifest_for_paths(source_snapshots, GENERATOR_SOURCE_PATHS),
            "auditor": {
                **manifest_for_paths(
                    source_snapshots,
                    tuple(
                        dict.fromkeys(
                            (
                                *AUDITOR_RUNTIME_SOURCE_PATHS,
                                *AUDITOR_VERIFICATION_SOURCE_PATHS,
                            )
                        )
                    ),
                ),
                "runtime_sources": list(AUDITOR_RUNTIME_SOURCE_PATHS),
                "verification_sources": list(AUDITOR_VERIFICATION_SOURCE_PATHS),
            },
        },
        "heldout": {
            key: value
            for key, value in heldout.items()
            if key not in {"signatures", "prompt_keys", "gram13"}
        },
        "contamination": {
            "train_heldout_reserved_signature_hits": len(board["heldout_hits"]),
            "train_heldout_exact_normalized_prompt_hits": len(
                primary["exact_heldout_hits"] | paired["exact_heldout_hits"]
            ),
            "train_heldout_literal_13gram_hits": len(
                primary["gram13_hits"] | paired["gram13_hits"]
            ),
            "examples": primary["contamination_examples"]
            + paired["contamination_examples"],
            "gates": {
                "exact_normalized_prompt_clear": checks[
                    "zero_train_heldout_exact_normalized_prompt_hits"
                ],
                "literal_normalized_word_13gram_clear": checks[
                    "zero_train_heldout_literal_13gram_hits"
                ],
                "reserved_signature_clear": checks[
                    "zero_train_heldout_reserved_signature_hits"
                ],
            },
            "heldout_answer_boundary": heldout["answer_boundary"],
            "semantics": (
                "Heldout prompts are solver-recomputed v2 transition-plus-final controller "
                "prompts; exact hits use normalized prompts and 13-grams are literal "
                "contiguous normalized word sequences. Heldout answers are read only for "
                "solver and exclusion-identity validation and never construct training rows."
            ),
        },
        "paired_board": {
            "primary_episodes_sha256": input_snapshots["episodes"].sha256,
            "counterpart_episodes_sha256": input_snapshots["paired_episodes"].sha256,
            "literal_jsonl_equal": paired_board_hash_equal,
            "episode_ids_sha256": hashlib.sha256(episode_ids_payload).hexdigest(),
            "episode_count": len(board["episode_ids"]),
            "first_episode_id": board["episode_ids"][0]
            if board["episode_ids"]
            else None,
            "last_episode_id": board["episode_ids"][-1]
            if board["episode_ids"]
            else None,
            "primary_visible_features_sha256": counter_sha256(
                primary["visible_features"]
            ),
            "counterpart_visible_features_sha256": counter_sha256(
                paired["visible_features"]
            ),
            "all_position_digit_pairs_sha256": counter_sha256(
                board["all_position_pairs"]
            ),
            "position_counts_equal": primary["position_counts"]
            == paired["position_counts"],
            "full_visible_feature_equality": primary["visible_features"]
            == paired["visible_features"],
            "current_position_context_l1_difference": local_l1,
        },
        "episodes_observed": {
            "blank_lines": board["blank_lines"],
            "raw": board["raw"],
            "valid": len(board["records"]),
            "unique_ids": len(board["records"]),
            "unique_signatures": len(board["signatures"]),
            "unique_complete_signatures": len(board["complete_signatures"]),
            "by_width": dict(sorted(board["widths"].items())),
            "by_operation": dict(sorted(board["operations"].items())),
        },
        "rows_observed": {
            "blank_lines": primary["blank_lines"],
            "raw": primary["raw"],
            "valid": primary["valid"],
            "unique_normalized_prompts": len(primary["seen_prompts"]),
            "by_kind": dict(sorted(primary["kind_counts"].items())),
            "by_width": actual_per_width_rows,
        },
        "coverage": {
            "required_position_independent_arithmetic_classes": len(
                required_arithmetic
            ),
            "episode_arithmetic_classes": len(
                required_arithmetic & board["arithmetic"]
            ),
            "transition_arithmetic_classes": len(
                required_arithmetic & primary["arithmetic"]
            ),
            "required_width_positions": len(required_positions),
            "episode_width_positions": len(required_positions & board["positions"]),
            "transition_width_positions": len(
                required_positions & primary["positions"]
            ),
            "required_width_position_controls": len(required_controls),
            "episode_width_position_controls": len(
                required_controls & board["controls"]
            ),
            "transition_width_position_controls": len(
                required_controls & primary["controls"]
            ),
            "designated_arithmetic_classes": len(board["designated"]),
        },
        "terminal_transition_classes": {
            "board": {
                operation: dict(sorted(counts.items()))
                for operation, counts in board["terminals"].items()
            },
            "rows": {
                operation: dict(
                    sorted((key, value) for key, value in counts.items() if value)
                )
                for operation, counts in primary["terminal_counts"].items()
            },
            "required_for_declared_arm": {
                operation: dict(sorted(counts.items()))
                for operation, counts in expected_row_terminals.items()
            },
        },
        "pair_distribution": {
            "penultimate_cells": len(board["penultimate_pairs"]),
            "terminal_cells": len(board["terminal_pairs"]),
            "all_cells_nonconcentrated": pair_diversity,
        },
        "tokenizer_accounting": token_report,
        "iid_semantics": (
            "iid and width are no-TERM supervision controls on the same board, not "
            "low-magnitude episode boards. Literal board equality and the former "
            "max-leading-digit-two contract cannot both hold."
        ),
        "residual_bundle_confound": (
            "Digit-readout and final-answer rows are common and expose the completed terminal "
            "state. Current-position arithmetic distributions necessarily differ with the "
            "terminal supervision allocation. The allocation suffix has matched global frequency "
            "but is associated with different positions by design; full operands/tapes, "
            "magnitude, row kind, width, operation, format count, and aggregate position budgets "
            "are equal."
        ),
        "claim_boundary": (
            "Independent CPU-only data admission from one-pass immutable byte snapshots. "
            "Heldout answers are read only to validate solver witnesses, counterfactual "
            "changes, and exclusion identities; they never construct training rows. This is "
            "local process evidence, not remote attestation."
        ),
        "auditor_sha256": source_snapshots[
            "pipeline/audit_digitwise_factorial_v4.py"
        ].sha256,
        "protocol_sha256": source_snapshots["train/digitwise_protocol.py"].sha256,
        "sft_packing_sha256": source_snapshots["train/sft.py"].sha256,
        "sft_encoding_sha256": source_snapshots["train/sft_encoding.py"].sha256,
    }
    write_report(out_path, report)
    print(json.dumps(report, sort_keys=True))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("production", "test"))
    parser.add_argument("--arm", required=True, choices=ARMS)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--episodes", required=True, type=Path)
    parser.add_argument("--paired-data", required=True, type=Path)
    parser.add_argument("--paired-episodes", required=True, type=Path)
    parser.add_argument("--heldout", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--generator-report", type=Path)
    parser.add_argument("--paired-generator-report", type=Path)
    parser.add_argument("--pack-length", type=int, default=2048)
    parser.add_argument(
        "--test-scale",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()
    if args.test_scale is not None and args.test_scale <= 0:
        raise SystemExit("--test-scale must be positive")
    report = audit(
        args.data,
        args.episodes,
        args.paired_data,
        args.paired_episodes,
        args.heldout,
        args.out,
        args.arm,
        args.mode,
        test_scale=args.test_scale,
        tokenizer_path=args.tokenizer,
        pack_length=args.pack_length,
        generator_report_path=args.generator_report,
        paired_generator_report_path=args.paired_generator_report,
    )
    if args.mode == "test":
        raise SystemExit("test-scale audit completed without production admission")
    if not report["production_admission"]:
        raise SystemExit("digitwise factorial v4 admission failed")


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
    """Execute auditor scientific code from one exact-byte source capture."""
    snapshots = {
        relative: capture_exact_file(ROOT / relative, relative)
        for relative in ADMISSION_SOURCE_PATHS
    }
    protocol = _frozen_module(
        "digitwise_protocol", snapshots["train/digitwise_protocol.py"]
    )
    encoding = _frozen_module("sft_encoding", snapshots["train/sft_encoding.py"])
    protocol_names = (
        "apply_microstep",
        "canonical_state",
        "digit_prompt",
        "final_prompt",
        "initial_state",
        "microstep_prompt",
        "parse_state",
        "state_answer",
        "state_digit",
    )
    namespace = {
        "__name__": "__main__",
        "__file__": str(ROOT / "pipeline/audit_digitwise_factorial_v4.py"),
        "__package__": None,
        "_BOOTSTRAP_ACTIVE": True,
        "_BOOTSTRAP_SOURCE_SNAPSHOTS": snapshots,
        "_FROZEN_PROTOCOL_EXPORTS": {
            name: getattr(protocol, name) for name in protocol_names
        },
        "_FROZEN_ENCODING_EXPORTS": {
            "encode_supervised_example": encoding.encode_supervised_example
        },
    }
    auditor = snapshots["pipeline/audit_digitwise_factorial_v4.py"]
    exec(compile(auditor.payload, namespace["__file__"], "exec"), namespace)


if __name__ == "__main__":
    if globals().get("_BOOTSTRAP_ACTIVE"):
        main()
    else:
        _run_frozen_cli()
