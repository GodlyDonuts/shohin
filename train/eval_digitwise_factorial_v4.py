#!/usr/bin/env python3
"""Frozen full-board evaluator for the completed digitwise factorial-v4 arms.

One invocation evaluates exactly one sealed SFT checkpoint.  The arm is selected
by a fixed contract, not inferred from a caller label.  Every held-out episode is
run in its normal and counterfactual form with greedy, model-authored closed-loop
state updates.  A malformed state terminates that branch; an incorrect but valid
state is forwarded unchanged.  No arithmetic solver, state repair, beam search,
or transcript selection is available in this evaluator.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import gc
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
from typing import Any, Callable, Iterable, Mapping

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train"))

from digitwise_protocol import (  # noqa: E402
    final_prompt,
    microstep_prompt,
    parse_answer,
    parse_state,
)


AUDIT = "shohin-digitwise-factorial-v4-full-eval-v1"
TRAINING_SOURCE_COMMIT = "8cafb9a1ff6c666721f7c5045b5bee2b7f550cc9"
TRAINING_SOURCE_MANIFEST_SHA256 = (
    "3e29f0d3fc653550400b71f353f159d93683a88db9fb381ff9e9d37451b81508"
)
TRAINING_SOURCE_BUNDLE_SHA256 = (
    "f0c31f9fa80324598650eb7e229aca31c9e92bfc7adac557ba22661730a58312"
)
TRAINING_WRAPPER_SHA256 = (
    "aa6055b5137c0bbb49ef79fc248b76540edfe9952520e993775c436a900dae38"
)
HELDOUT_SHA256 = "89ce11b36ff2f56e83cda72a1f07b1a90f4a3dc3803c69db2779a27219712646"
TOKENIZER_SHA256 = "87532df5c121753de3b29194e1f9e3de47986d3f5359548fdf93606773a233d4"
TOKENIZER_BYTES = 2_309_567
TOKENIZER_VOCAB_SIZE = 32_768
CHECKPOINT_FILES = frozenset(
    {
        "exact_budget_final.json",
        "exact_budget_preflight.json",
        "production_admission.json",
        "reviewed_source_manifest.json",
        "scientific_sources.json",
        "sft_ep1.pt",
    }
)
EVAL_SOURCE_PATHS = (
    "train/eval_digitwise_factorial_v4.py",
    "train/test_eval_digitwise_factorial_v4.py",
    "train/jobs/eval_digitwise_factorial_v4.sbatch",
    "train/digitwise_protocol.py",
    "train/model.py",
)
REGIME_BUDGETS = {
    "fit_w4": 300,
    "fit_w6": 300,
    "value_ood_w4": 300,
    "value_ood_w6": 300,
    "width_ood_w8": 300,
}
REGIME_WIDTHS = {
    "fit_w4": 4,
    "fit_w6": 6,
    "value_ood_w4": 4,
    "value_ood_w6": 6,
    "width_ood_w8": 8,
}
PAIR_COUNT = sum(REGIME_BUDGETS.values())
BRANCH_COUNT = 2 * PAIR_COUNT
TRANSITION_MAX_NEW = 96
FINAL_MAX_NEW = 48
PROMPT_STYLE = "heldout"
DECODE_SEED = 20260717
EXACT_UPDATES = 1_560
EXACT_PACKS = 24_960
EXACT_FORWARD_TOKEN_POSITIONS = 51_118_080
EXPECTED_CHECKPOINT_KEYS = frozenset(
    {
        "cfg",
        "model",
        "step",
        "factorial_arm",
        "production_admission_sha256",
        "exact_budget_preflight_sha256",
        "exact_budget_updates",
    }
)
BRANCH_FIELDS = frozenset(
    {
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
)
TOP_LEVEL_FIELDS = BRANCH_FIELDS | {"counterfactual"}
HEX64 = re.compile(r"^[0-9a-f]{64}$")
GIT_OBJECT = re.compile(r"^[0-9a-f]{40,64}$")


@dataclass(frozen=True)
class ArmContract:
    arm: str
    task_index: int
    training_job_id: int
    checkpoint_origin: str
    checkpoint_sha256: str
    checkpoint_bytes: int
    final_sha256: str
    preflight_sha256: str
    admission_sha256: str
    source_bundle_sha256: str
    supervised_tokens: int
    supervised_schedule_sha256: str


ARM_CONTRACTS = {
    "iid": ArmContract(
        arm="iid",
        task_index=0,
        training_job_id=692071,
        checkpoint_origin=(
            "/lustre/fs1/home/sa305415/shohin_factorial_source_c24e16e_v3/"
            "train/sft_factorial_runs/job_692071_attempt_0"
        ),
        checkpoint_sha256=(
            "e2cfa811da5e7150c093f160412d9358d603ec9563a9ee9c432193e4516b8846"
        ),
        checkpoint_bytes=500_447_130,
        final_sha256=(
            "c7d7de273855ef2bdd826947eab918dfaeaad9b77b9c2d49e9344b1ac8b8df2e"
        ),
        preflight_sha256=(
            "b6a590851e2d245c1758a9acacc7ce811b02ee37750845a0bcbe79f9eb5a8c8c"
        ),
        admission_sha256=(
            "cf31110a3a10fa477e4df72b8d73b5149e7a1a8eaa8e2f0ff5f2fa25c4171c9a"
        ),
        source_bundle_sha256=TRAINING_SOURCE_BUNDLE_SHA256,
        supervised_tokens=10_170_918,
        supervised_schedule_sha256=(
            "3269e3b607dd48ef8633773ff2d05b50839902f1b29cc70b08cb94cbc36449c0"
        ),
    ),
    "term": ArmContract(
        arm="term",
        task_index=1,
        training_job_id=692073,
        checkpoint_origin=(
            "/lustre/fs1/home/sa305415/shohin_factorial_source_c24e16e_v3/"
            "train/sft_factorial_runs/job_692073_attempt_0"
        ),
        checkpoint_sha256=(
            "1f7307705f632e9290e9f5c8580d8445981d5ffffc974a5f2854c5b93c4ad96d"
        ),
        checkpoint_bytes=500_447_130,
        final_sha256=(
            "50f3d9fd4c12b3c384cf0c63fe2558c6816dc0db804b667859150d83f400171c"
        ),
        preflight_sha256=(
            "14a0cce6c43e1a277ac6e3018e690752099b9d912473f4ba4272cacced846331"
        ),
        admission_sha256=(
            "4c57551b0fa53b0f6a0a7b040377c69b4483678c8b68293b037cae55112af9a3"
        ),
        source_bundle_sha256=TRAINING_SOURCE_BUNDLE_SHA256,
        supervised_tokens=10_171_364,
        supervised_schedule_sha256=(
            "f544e15136ee5895d08d8806f99cb581440e6e817c6d8c2b613c429e812af716"
        ),
    ),
    "width": ArmContract(
        arm="width",
        task_index=2,
        training_job_id=692075,
        checkpoint_origin=(
            "/lustre/fs1/home/sa305415/shohin_factorial_source_c24e16e_v3/"
            "train/sft_factorial_runs/job_692075_attempt_0"
        ),
        checkpoint_sha256=(
            "f8f9a90ce2e15b58a3e93e255e71f8f0abe517cfacea1e5e51d2cdd45bb8c847"
        ),
        checkpoint_bytes=500_447_130,
        final_sha256=(
            "6b0be8b9e0c208df4d4965462a0dcd6fe897240c7daddefcd9f97525fc005b2b"
        ),
        preflight_sha256=(
            "386419c4889a3e47d85dda13fc06f560eb9ed4e91cf0818516c1d9f5d6849c13"
        ),
        admission_sha256=(
            "8413d4904641d80b7974949f5cb680dd3251ef77d1df30afe47d7ebd6dc854e7"
        ),
        source_bundle_sha256=TRAINING_SOURCE_BUNDLE_SHA256,
        supervised_tokens=10_217_994,
        supervised_schedule_sha256=(
            "0d9b650a00914a3e6556c580cd6f87cf3f20bd8466f7be505bf836e721201041"
        ),
    ),
    "term_width": ArmContract(
        arm="term_width",
        task_index=3,
        training_job_id=692077,
        checkpoint_origin=(
            "/lustre/fs1/home/sa305415/shohin_factorial_source_c24e16e_v3/"
            "train/sft_factorial_runs/job_692077_attempt_0"
        ),
        checkpoint_sha256=(
            "06b5d9f2dc4a210fbc4fbe9f62fe036dca2a05598d5e463cf6a0e9133866779e"
        ),
        checkpoint_bytes=500_447_194,
        final_sha256=(
            "b86dfd3947ba702ea415408ba83ebcedb7bb35280161785edc40c546b303b636"
        ),
        preflight_sha256=(
            "1c50b14b34fee2c3c22e660c92d76d46d6865065c18e3140cfe3d5470a1b6c35"
        ),
        admission_sha256=(
            "d3132fe709d781553940ed890a8c1f7e425c74c278a04adbd5a06f2429ef2639"
        ),
        source_bundle_sha256=TRAINING_SOURCE_BUNDLE_SHA256,
        supervised_tokens=10_217_275,
        supervised_schedule_sha256=(
            "cbd0413f8cc3c6af0a70787415b024d2a9721dee5013bd8b3b0ab0192a180d20"
        ),
    ),
}
TASK_TO_ARM = {contract.task_index: arm for arm, contract in ARM_CONTRACTS.items()}


class ContractError(ValueError):
    """Raised when immutable evaluation evidence violates the frozen contract."""


@dataclass(frozen=True)
class FrozenFile:
    path: Path
    payload: bytes
    sha256: str
    size: int
    mode: int


@dataclass(frozen=True)
class Generation:
    text: str
    content_token_ids: tuple[int, ...]
    sampled_token_ids: tuple[int, ...]
    prompt_token_count: int
    stop_reason: str


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("ascii")


def _identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_size),
        int(value.st_mtime_ns),
        stat.S_IMODE(value.st_mode),
    )


def read_frozen_file(
    path: str | Path,
    expected_sha256: str,
    *,
    expected_size: int | None = None,
    expected_mode: int | None = None,
) -> FrozenFile:
    """Capture one regular file once and verify path/descriptor identity."""
    path = Path(path)
    if not HEX64.fullmatch(expected_sha256):
        raise ContractError("expected SHA-256 is malformed")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ContractError(f"cannot open required regular file: {path}") from error
    try:
        before = os.fstat(descriptor)
        path_before = os.stat(path, follow_symlinks=False)
        if not stat.S_ISREG(before.st_mode) or _identity(before) != _identity(
            path_before
        ):
            raise ContractError(f"input is not one stable regular file: {path}")
        blocks: list[bytes] = []
        digest = hashlib.sha256()
        while True:
            block = os.read(descriptor, 4 * 1024 * 1024)
            if not block:
                break
            blocks.append(block)
            digest.update(block)
        after = os.fstat(descriptor)
        path_after = os.stat(path, follow_symlinks=False)
        if _identity(before) != _identity(after) or _identity(before) != _identity(
            path_after
        ):
            raise ContractError(f"input changed during capture: {path}")
        payload = b"".join(blocks)
        actual_sha256 = digest.hexdigest()
        if actual_sha256 != expected_sha256:
            raise ContractError(
                f"SHA-256 mismatch for {path}: expected={expected_sha256} actual={actual_sha256}"
            )
        if expected_size is not None and len(payload) != expected_size:
            raise ContractError(
                f"byte-size mismatch for {path}: expected={expected_size} actual={len(payload)}"
            )
        mode = stat.S_IMODE(before.st_mode)
        if expected_mode is not None and mode != expected_mode:
            raise ContractError(
                f"mode mismatch for {path}: expected={expected_mode:04o} actual={mode:04o}"
            )
        return FrozenFile(path, payload, actual_sha256, len(payload), mode)
    finally:
        os.close(descriptor)


def load_canonical_json(frozen: FrozenFile) -> Any:
    try:
        value = json.loads(frozen.payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ContractError(f"invalid JSON: {frozen.path}") from error
    if frozen.payload != canonical_json_bytes(value):
        raise ContractError(f"JSON is not canonical: {frozen.path}")
    return value


def _require_exact_keys(
    value: Mapping[str, Any], expected: set[str] | frozenset[str], label: str
) -> None:
    if set(value) != set(expected):
        missing = sorted(set(expected) - set(value))
        extra = sorted(set(value) - set(expected))
        raise ContractError(f"{label} keys differ: missing={missing} extra={extra}")


def _lsf_value(digits: str) -> int:
    return sum(int(digit) * (10**index) for index, digit in enumerate(digits))


def terminal_class(branch: Mapping[str, Any]) -> str:
    expected = branch["expected_states"]
    before = parse_state(
        branch["initial_state"] if len(expected) == 1 else expected[-2]
    )
    after = parse_state(expected[-1])
    if before is None or after is None:
        raise ContractError("cannot derive terminal carry class")
    return f"{before['c']}{after['c']}"


def _validate_branch(branch: Mapping[str, Any], label: str) -> None:
    _require_exact_keys(branch, BRANCH_FIELDS, label)
    if not isinstance(branch["id"], str) or not branch["id"]:
        raise ContractError(f"{label} has invalid id")
    regime = branch["split"]
    if regime not in REGIME_BUDGETS:
        raise ContractError(f"{label} has unknown regime")
    width = branch["width"]
    if type(width) is not int or width != REGIME_WIDTHS[regime]:
        raise ContractError(f"{label} width/regime mismatch")
    operation = branch["operation"]
    if operation not in {"add", "sub"}:
        raise ContractError(f"{label} has invalid operation")
    if branch["prompt_style"] != PROMPT_STYLE:
        raise ContractError(f"{label} has noncanonical prompt style")
    for name in ("left", "right", "expected_answer"):
        if type(branch[name]) is not int:
            raise ContractError(f"{label} has non-integer {name}")
    if branch["left"] < 0 or branch["right"] < 0:
        raise ContractError(f"{label} has negative operand")
    if operation == "sub" and branch["left"] < branch["right"]:
        raise ContractError(f"{label} violates nonnegative subtraction contract")
    initial = parse_state(branch["initial_state"])
    if initial is None:
        raise ContractError(f"{label} has invalid initial state")
    if (
        initial["op"] != operation
        or initial["w"] != width
        or initial["p"] != 0
        or initial["c"] != 0
        or initial["z"] != 0
        or _lsf_value(initial["a"]) != branch["left"]
        or _lsf_value(initial["b"]) != branch["right"]
    ):
        raise ContractError(f"{label} initial state does not bind declared operands")
    expected_states = branch["expected_states"]
    if not isinstance(expected_states, list) or len(expected_states) != width:
        raise ContractError(f"{label} has wrong transition budget")
    previous = initial
    for position, line in enumerate(expected_states):
        if not isinstance(line, str):
            raise ContractError(f"{label} expected state is not text")
        state = parse_state(line)
        if state is None:
            raise ContractError(f"{label} has malformed expected state")
        if (
            state["op"] != operation
            or state["w"] != width
            or state["a"] != initial["a"]
            or state["b"] != initial["b"]
            or state["p"] != position + 1
            or state["z"] != int(position + 1 == width)
            or state["r"][:position] != previous["r"][:position]
        ):
            raise ContractError(f"{label} expected trajectory is structurally invalid")
        previous = state
    terminal = previous
    result = _lsf_value(terminal["r"])
    if operation == "add":
        result += int(terminal["c"]) * (10**width)
    elif terminal["c"]:
        raise ContractError(f"{label} terminal subtraction retains a borrow")
    if result != branch["expected_answer"]:
        raise ContractError(f"{label} terminal readout differs from expected answer")
    carry_class = terminal_class(branch)
    allowed = {"00", "10", "01", "11"} if operation == "add" else {"00", "10"}
    if carry_class not in allowed:
        raise ContractError(f"{label} has impossible terminal carry class")


def validate_heldout_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    expected_regime_counts: Mapping[str, int] = REGIME_BUDGETS,
) -> list[dict[str, Any]]:
    validated: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    ids: set[str] = set()
    for index, raw in enumerate(rows):
        if not isinstance(raw, dict):
            raise ContractError(f"heldout row {index} is not an object")
        _require_exact_keys(raw, TOP_LEVEL_FIELDS, f"heldout row {index}")
        counterfactual = raw["counterfactual"]
        if not isinstance(counterfactual, dict):
            raise ContractError(f"heldout row {index} counterfactual is not an object")
        normal = {key: raw[key] for key in BRANCH_FIELDS}
        _validate_branch(normal, f"heldout row {index} normal")
        _validate_branch(counterfactual, f"heldout row {index} counterfactual")
        if (
            normal["split"] != counterfactual["split"]
            or normal["prompt_style"] != counterfactual["prompt_style"]
            or normal["operation"] != counterfactual["operation"]
            or normal["width"] != counterfactual["width"]
            or counterfactual["id"] != normal["id"] + "-cf"
        ):
            raise ContractError(f"heldout row {index} pair identity mismatch")
        for branch_id in (normal["id"], counterfactual["id"]):
            if branch_id in ids:
                raise ContractError(f"duplicate heldout branch id: {branch_id}")
            ids.add(branch_id)
        counts[normal["split"]] += 1
        validated.append(dict(raw))
    if dict(sorted(counts.items())) != dict(sorted(expected_regime_counts.items())):
        raise ContractError(
            f"heldout regime counts differ: expected={dict(expected_regime_counts)} actual={dict(counts)}"
        )
    if len(validated) != sum(expected_regime_counts.values()):
        raise ContractError("heldout pair count differs from frozen budget")
    return validated


def load_heldout(frozen: FrozenFile) -> list[dict[str, Any]]:
    try:
        text = frozen.payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ContractError("heldout board is not UTF-8") from error
    if not text.endswith("\n"):
        raise ContractError("heldout board lacks a final newline")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line:
            raise ContractError(f"heldout board contains blank line {line_number}")
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ContractError(
                f"invalid heldout JSON at line {line_number}"
            ) from error
        rows.append(row)
    return validate_heldout_rows(rows)


def validate_training_bundle(
    checkpoint_dir: str | Path,
    contract: ArmContract,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Verify the sealed six-file SFT output and return checkpoint plus provenance."""
    checkpoint_dir = Path(checkpoint_dir)
    if checkpoint_dir.is_symlink() or not checkpoint_dir.is_dir():
        raise ContractError("checkpoint directory is not a regular directory")
    if stat.S_IMODE(checkpoint_dir.stat().st_mode) != 0o500:
        raise ContractError("checkpoint snapshot directory must be sealed mode 0500")
    if {entry.name for entry in checkpoint_dir.iterdir()} != CHECKPOINT_FILES:
        raise ContractError("checkpoint snapshot is not closed-world")
    expected_hashes = {
        "sft_ep1.pt": contract.checkpoint_sha256,
        "exact_budget_final.json": contract.final_sha256,
        "exact_budget_preflight.json": contract.preflight_sha256,
        "production_admission.json": contract.admission_sha256,
        "reviewed_source_manifest.json": TRAINING_SOURCE_MANIFEST_SHA256,
        "scientific_sources.json": contract.source_bundle_sha256,
    }
    frozen = {
        name: read_frozen_file(
            checkpoint_dir / name,
            digest,
            expected_size=contract.checkpoint_bytes if name == "sft_ep1.pt" else None,
            expected_mode=0o400,
        )
        for name, digest in expected_hashes.items()
    }
    final = load_canonical_json(frozen["exact_budget_final.json"])
    preflight = load_canonical_json(frozen["exact_budget_preflight.json"])
    admission = load_canonical_json(frozen["production_admission.json"])
    manifest = load_canonical_json(frozen["reviewed_source_manifest.json"])
    source_bundle = load_canonical_json(frozen["scientific_sources.json"])
    if final.get("audit") != "sft_exact_budget_v1" or final.get("phase") != "final":
        raise ContractError("final receipt has wrong protocol")
    if final.get("arm") != contract.arm or final.get("status") != "complete":
        raise ContractError("final receipt arm/status mismatch")
    checkpoint_binding = final.get("checkpoint", {})
    if (
        checkpoint_binding.get("path") != "sft_ep1.pt"
        or checkpoint_binding.get("sha256") != contract.checkpoint_sha256
        or checkpoint_binding.get("bytes") != contract.checkpoint_bytes
    ):
        raise ContractError("final receipt checkpoint binding mismatch")
    if final.get("preflight", {}).get("sha256") != contract.preflight_sha256:
        raise ContractError("final receipt preflight binding mismatch")
    if final.get("admission", {}).get("sha256") != contract.admission_sha256:
        raise ContractError("final receipt admission binding mismatch")
    if (
        final.get("reviewed_source_manifest", {}).get("sha256")
        != TRAINING_SOURCE_MANIFEST_SHA256
        or final.get("source_bundle", {}).get("sha256") != contract.source_bundle_sha256
    ):
        raise ContractError("final receipt source binding mismatch")
    actual = final.get("actual", {})
    planned = final.get("planned", {})
    for accounting in (actual, planned):
        if (
            accounting.get("optimizer_updates") != EXACT_UPDATES
            or accounting.get("packs") != EXACT_PACKS
            or accounting.get("forward_token_positions")
            != EXACT_FORWARD_TOKEN_POSITIONS
            or accounting.get("supervised_tokens") != contract.supervised_tokens
            or accounting.get("supervised_tokens_per_update_sha256")
            != contract.supervised_schedule_sha256
        ):
            raise ContractError("exact training budget binding mismatch")
    if preflight.get("arm") != contract.arm or preflight.get("phase") != "preflight":
        raise ContractError("preflight arm/phase mismatch")
    if (
        admission.get("admitted_arm") != contract.arm
        or admission.get("production_admission") is not True
        or admission.get("admission_pass") is not True
    ):
        raise ContractError("production admission mismatch")
    if (
        manifest.get("schema") != "shohin-factorial-v4-reviewed-source-manifest-v1"
        or manifest.get("review_status") != "approved"
        or manifest.get("reviewed_clean_commit") != TRAINING_SOURCE_COMMIT
        or manifest.get("clean_source_tree") is not True
        or manifest.get("remote_attestation") is not False
        or manifest.get("sources", {}).get("train/jobs/sft_factorial.sbatch")
        != TRAINING_WRAPPER_SHA256
    ):
        raise ContractError("reviewed training source manifest mismatch")
    if source_bundle.get("schema") != "shohin-factorial-v4-scientific-source-bundle-v1":
        raise ContractError("training source bundle schema mismatch")
    try:
        checkpoint = torch.load(
            io.BytesIO(frozen["sft_ep1.pt"].payload),
            map_location="cpu",
            weights_only=True,
        )
    except Exception as error:
        raise ContractError("checkpoint cannot be loaded safely") from error
    if not isinstance(checkpoint, dict) or set(checkpoint) != EXPECTED_CHECKPOINT_KEYS:
        raise ContractError("checkpoint top-level schema mismatch")
    if (
        checkpoint.get("step") != "sft_ep1"
        or checkpoint.get("factorial_arm") != contract.arm
        or checkpoint.get("production_admission_sha256") != contract.admission_sha256
        or checkpoint.get("exact_budget_preflight_sha256") != contract.preflight_sha256
        or checkpoint.get("exact_budget_updates") != EXACT_UPDATES
        or not isinstance(checkpoint.get("cfg"), dict)
        or not isinstance(checkpoint.get("model"), dict)
    ):
        raise ContractError("checkpoint metadata binding mismatch")
    provenance = {
        "arm": contract.arm,
        "training_job_id": contract.training_job_id,
        "checkpoint_origin": contract.checkpoint_origin,
        "checkpoint_sha256": contract.checkpoint_sha256,
        "checkpoint_bytes": contract.checkpoint_bytes,
        "final_receipt_sha256": contract.final_sha256,
        "preflight_receipt_sha256": contract.preflight_sha256,
        "production_admission_sha256": contract.admission_sha256,
        "reviewed_source_manifest_sha256": TRAINING_SOURCE_MANIFEST_SHA256,
        "reviewed_source_commit": TRAINING_SOURCE_COMMIT,
        "training_spooled_wrapper_sha256": TRAINING_WRAPPER_SHA256,
        "scientific_source_bundle_sha256": contract.source_bundle_sha256,
        "exact_budget": {
            "optimizer_updates": EXACT_UPDATES,
            "packs": EXACT_PACKS,
            "forward_token_positions": EXACT_FORWARD_TOKEN_POSITIONS,
            "supervised_tokens": contract.supervised_tokens,
            "supervised_tokens_per_update_sha256": contract.supervised_schedule_sha256,
        },
    }
    return checkpoint, provenance


def validate_checkpoint_metadata(
    checkpoint: Mapping[str, Any], contract: ArmContract
) -> None:
    """Small pure validation hook retained for corruption-focused CPU tests."""
    if set(checkpoint) != EXPECTED_CHECKPOINT_KEYS:
        raise ContractError("checkpoint top-level schema mismatch")
    if (
        checkpoint.get("step") != "sft_ep1"
        or checkpoint.get("factorial_arm") != contract.arm
        or checkpoint.get("production_admission_sha256") != contract.admission_sha256
        or checkpoint.get("exact_budget_preflight_sha256") != contract.preflight_sha256
        or checkpoint.get("exact_budget_updates") != EXACT_UPDATES
        or not isinstance(checkpoint.get("cfg"), dict)
        or not isinstance(checkpoint.get("model"), dict)
    ):
        raise ContractError("checkpoint metadata binding mismatch")


def validate_source_binding(
    binding_file: FrozenFile,
    source_root: str | Path,
    contract: ArmContract,
) -> dict[str, Any]:
    binding = load_canonical_json(binding_file)
    expected_keys = {
        "schema",
        "eval_commit",
        "clean_source_tree",
        "sources",
        "spooled_wrapper_sha256",
        "slurm",
        "scientific_inputs",
    }
    _require_exact_keys(binding, expected_keys, "evaluation source binding")
    if (
        binding["schema"] != "shohin-digitwise-factorial-v4-eval-source-binding-v1"
        or not GIT_OBJECT.fullmatch(str(binding["eval_commit"]))
        or binding["clean_source_tree"] is not True
        or not HEX64.fullmatch(str(binding["spooled_wrapper_sha256"]))
    ):
        raise ContractError("evaluation source binding header mismatch")
    sources = binding["sources"]
    if not isinstance(sources, dict) or set(sources) != set(EVAL_SOURCE_PATHS):
        raise ContractError("evaluation source set mismatch")
    if (
        sources["train/jobs/eval_digitwise_factorial_v4.sbatch"]
        != binding["spooled_wrapper_sha256"]
    ):
        raise ContractError("spooled evaluation wrapper is not source-bound")
    source_root = Path(source_root)
    for relative, expected_sha256 in sources.items():
        if not HEX64.fullmatch(str(expected_sha256)):
            raise ContractError(f"malformed source hash: {relative}")
        read_frozen_file(
            source_root / relative,
            expected_sha256,
            expected_mode=0o400,
        )
    slurm = binding["slurm"]
    if (
        not isinstance(slurm, dict)
        or slurm.get("restart_count") != 0
        or slurm.get("array_task_id") != contract.task_index
        or not isinstance(slurm.get("job_id"), int)
        or slurm["job_id"] <= 0
    ):
        raise ContractError("Slurm source binding mismatch")
    scientific = binding["scientific_inputs"]
    expected_scientific = {
        "arm": contract.arm,
        "checkpoint_origin": contract.checkpoint_origin,
        "checkpoint_sha256": contract.checkpoint_sha256,
        "heldout_sha256": HELDOUT_SHA256,
        "tokenizer_sha256": TOKENIZER_SHA256,
    }
    if scientific != expected_scientific:
        raise ContractError("scientific input source binding mismatch")
    return binding


def require_exact_h100() -> dict[str, Any]:
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ContractError(
            "canonical evaluation requires exactly one visible CUDA GPU"
        )
    name = torch.cuda.get_device_name(0)
    capability = tuple(torch.cuda.get_device_capability(0))
    properties = torch.cuda.get_device_properties(0)
    if "H100" not in name or capability != (9, 0):
        raise ContractError(
            f"canonical evaluation requires an H100 (name={name!r}, capability={capability})"
        )
    if properties.total_memory < 79 * 1024**3:
        raise ContractError(
            "canonical evaluation refuses a partitioned or undersized H100"
        )
    if not torch.cuda.is_bf16_supported():
        raise ContractError("canonical evaluation requires native BF16 support")
    probe = torch.empty((1024,), device="cuda", dtype=torch.bfloat16)
    probe.add_(1)
    torch.cuda.synchronize()
    del probe
    return {
        "device": "cuda:0",
        "name": name,
        "capability": list(capability),
        "total_memory_bytes": int(properties.total_memory),
        "bf16_supported": True,
        "visible_device_count": 1,
    }


def greedy_generate(
    model: Any,
    tokenizer: Any,
    prompt: str,
    device: str,
    *,
    max_new: int,
) -> Generation:
    """Run one frozen argmax decode with no answer-specific stop or repair."""
    if max_new not in {TRANSITION_MAX_NEW, FINAL_MAX_NEW}:
        raise ContractError("decode max_new is outside the frozen contract")
    encoded = tokenizer.encode(prompt).ids
    if not encoded:
        raise ContractError("tokenizer produced an empty prompt")
    sequence_limit = int(model.cfg.seq_len)
    if len(encoded) + max_new > sequence_limit:
        raise ContractError("prompt plus frozen decode budget exceeds model context")
    eos_id = tokenizer.token_to_id("<|endoftext|>")
    if eos_id is None:
        raise ContractError("tokenizer lacks the required end-of-text token")
    content: list[int] = []
    sampled: list[int] = []
    autocast = torch.autocast("cuda", dtype=torch.bfloat16)
    with torch.inference_mode(), autocast:
        logits, cache = model(
            torch.tensor([encoded], device=device, dtype=torch.long),
            return_cache=True,
            pos=0,
        )
        position = len(encoded)
        stop_reason = "max_new"
        for _ in range(max_new):
            token = int(logits[:, -1].argmax(dim=-1).item())
            sampled.append(token)
            if token == eos_id:
                stop_reason = "eos"
                break
            content.append(token)
            logits, cache = model(
                torch.tensor([[token]], device=device, dtype=torch.long),
                cache=cache,
                pos=position,
                return_cache=True,
            )
            position += 1
    text = tokenizer.decode(content, skip_special_tokens=False)
    return Generation(
        text=text,
        content_token_ids=tuple(content),
        sampled_token_ids=tuple(sampled),
        prompt_token_count=len(encoded),
        stop_reason=stop_reason,
    )


Ask = Callable[[str, int, str], Generation]


def _generation_record(value: Generation) -> dict[str, Any]:
    return {
        "text": value.text,
        "content_token_ids": list(value.content_token_ids),
        "sampled_token_ids": list(value.sampled_token_ids),
        "prompt_token_count": value.prompt_token_count,
        "stop_reason": value.stop_reason,
    }


def rollout_branch(branch: Mapping[str, Any], ask: Ask) -> dict[str, Any]:
    """Forward valid model states exactly; stop only when no state exists."""
    state = parse_state(branch["initial_state"])
    if state is None:
        raise ContractError("validated branch lost its initial state")
    rows: list[dict[str, Any]] = []
    first_failure_position: int | None = None
    first_failure_reason: str | None = None
    prefix_exact_length = 0
    emitted_tokens = 0
    for position, expected_line in enumerate(branch["expected_states"]):
        prompt = microstep_prompt(state, style=PROMPT_STYLE)
        generation = ask(prompt, TRANSITION_MAX_NEW, "transition")
        emitted_tokens += len(generation.sampled_token_ids)
        predicted = parse_state(generation.text)
        expected = parse_state(expected_line)
        if expected is None:
            raise ContractError("validated expected state became invalid")
        correct = predicted == expected
        if correct and first_failure_position is None:
            prefix_exact_length += 1
        elif first_failure_position is None:
            first_failure_position = position
            first_failure_reason = (
                "malformed_state" if predicted is None else "state_mismatch"
            )
        rows.append(
            {
                "position": position,
                "prompt": prompt,
                "input_state": dict(state),
                "expected_state": dict(expected),
                "predicted_state": None if predicted is None else dict(predicted),
                "correct": correct,
                "generation": _generation_record(generation),
            }
        )
        if predicted is None:
            state = None
            break
        state = predicted
    transition_budget = len(branch["expected_states"])
    fully_parseable = len(rows) == transition_budget and all(
        row["predicted_state"] is not None for row in rows
    )
    state_closed_loop_exact = len(rows) == transition_budget and all(
        row["correct"] for row in rows
    )
    terminal_reached = state is not None and bool(state["z"])
    final_generation: Generation | None = None
    final_prompt_text: str | None = None
    emitted_answer: int | None = None
    final_answer_correct = False
    if terminal_reached:
        final_prompt_text = final_prompt(state, style=PROMPT_STYLE)
        final_generation = ask(final_prompt_text, FINAL_MAX_NEW, "final")
        emitted_tokens += len(final_generation.sampled_token_ids)
        emitted_answer = parse_answer(final_generation.text)
        final_answer_correct = emitted_answer == branch["expected_answer"]
        if first_failure_position is None and not final_answer_correct:
            first_failure_position = transition_budget
            first_failure_reason = (
                "malformed_answer" if emitted_answer is None else "answer_mismatch"
            )
    elif first_failure_position is None:
        first_failure_position = transition_budget
        first_failure_reason = "nonterminal_after_budget"
    terminal_transition_exact = len(rows) == transition_budget and bool(
        rows[-1]["correct"]
    )
    success = state_closed_loop_exact and final_answer_correct
    if success:
        first_failure_position = None
        first_failure_reason = None
    return {
        "id": branch["id"],
        "regime": branch["split"],
        "operation": branch["operation"],
        "width": branch["width"],
        "terminal_carry_class": terminal_class(branch),
        "expected_answer": branch["expected_answer"],
        "initial_state": branch["initial_state"],
        "transition_budget": transition_budget,
        "transition_calls": len(rows),
        "prefix_exact_length": prefix_exact_length,
        "fully_parseable": fully_parseable,
        "state_closed_loop_exact": state_closed_loop_exact,
        "terminal_transition_exact": terminal_transition_exact,
        "terminal_reached": terminal_reached,
        "final_prompt_issued": final_generation is not None,
        "emitted_answer": emitted_answer,
        "final_answer_correct": final_answer_correct,
        "closed_loop_success": success,
        "first_failure_position": first_failure_position,
        "first_failure_reason": first_failure_reason,
        "emitted_token_count": emitted_tokens,
        "rows": rows,
        "final_prompt": final_prompt_text,
        "final_generation": (
            None if final_generation is None else _generation_record(final_generation)
        ),
    }


def _first_difference(left: list[Any], right: list[Any]) -> int | None:
    for index, (a, b) in enumerate(zip(left, right, strict=True)):
        if a != b:
            return index
    return None


def evaluate_pair(episode: Mapping[str, Any], ask: Ask) -> dict[str, Any]:
    normal_branch = {key: episode[key] for key in BRANCH_FIELDS}
    counterfactual_branch = episode["counterfactual"]
    normal = rollout_branch(normal_branch, ask)
    counterfactual = rollout_branch(counterfactual_branch, ask)
    expected_divergence = _first_difference(
        normal_branch["expected_states"], counterfactual_branch["expected_states"]
    )
    predicted_normal = [row["predicted_state"] for row in normal["rows"]]
    predicted_counterfactual = [
        row["predicted_state"] for row in counterfactual["rows"]
    ]
    shared = min(len(predicted_normal), len(predicted_counterfactual))
    predicted_divergence = _first_difference(
        predicted_normal[:shared], predicted_counterfactual[:shared]
    )
    expected_changed = (
        normal_branch["expected_answer"] != counterfactual_branch["expected_answer"]
    )
    answer_intervention_success = (
        expected_changed
        and normal["final_answer_correct"]
        and counterfactual["final_answer_correct"]
        and normal["emitted_answer"] != counterfactual["emitted_answer"]
    )
    state_intervention_at_expected_position = (
        expected_divergence is not None
        and expected_divergence < shared
        and predicted_normal[expected_divergence]
        != predicted_counterfactual[expected_divergence]
    )
    return {
        "id": normal_branch["id"],
        "regime": normal_branch["split"],
        "operation": normal_branch["operation"],
        "width": normal_branch["width"],
        "normal_terminal_carry_class": normal["terminal_carry_class"],
        "counterfactual_terminal_carry_class": counterfactual["terminal_carry_class"],
        "expected_answer_changed": expected_changed,
        "first_expected_state_divergence_position": expected_divergence,
        "first_predicted_state_divergence_position": predicted_divergence,
        "state_intervention_at_expected_position": (
            state_intervention_at_expected_position
        ),
        "both_state_closed_loop_exact": (
            normal["state_closed_loop_exact"]
            and counterfactual["state_closed_loop_exact"]
        ),
        "both_final_answers_correct": (
            normal["final_answer_correct"] and counterfactual["final_answer_correct"]
        ),
        "answer_intervention_success": answer_intervention_success,
        "both_closed_loop_success": (
            normal["closed_loop_success"] and counterfactual["closed_loop_success"]
        ),
        "normal": normal,
        "counterfactual": counterfactual,
    }


def _ratio(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator


def _summarize_branches(branches: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    rows = list(branches)
    transition_budget = sum(int(row["transition_budget"]) for row in rows)
    transition_calls = sum(int(row["transition_calls"]) for row in rows)
    exact_transitions = sum(
        int(step["correct"]) for row in rows for step in row["rows"]
    )
    parseable_transitions = sum(
        int(step["predicted_state"] is not None) for row in rows for step in row["rows"]
    )
    counts = {
        "branches": len(rows),
        "transition_budget": transition_budget,
        "transition_calls": transition_calls,
        "parseable_transitions": parseable_transitions,
        "exact_transitions": exact_transitions,
        "exact_prefix_steps": sum(int(row["prefix_exact_length"]) for row in rows),
        "fully_parseable": sum(int(row["fully_parseable"]) for row in rows),
        "state_closed_loop_exact": sum(
            int(row["state_closed_loop_exact"]) for row in rows
        ),
        "terminal_transition_exact": sum(
            int(row["terminal_transition_exact"]) for row in rows
        ),
        "terminal_reached": sum(int(row["terminal_reached"]) for row in rows),
        "final_prompt_issued": sum(int(row["final_prompt_issued"]) for row in rows),
        "final_answer_parseable": sum(
            int(row["emitted_answer"] is not None) for row in rows
        ),
        "final_answer_correct": sum(int(row["final_answer_correct"]) for row in rows),
        "closed_loop_success": sum(int(row["closed_loop_success"]) for row in rows),
        "emitted_tokens": sum(int(row["emitted_token_count"]) for row in rows),
    }
    denominator = counts["branches"]
    rates = {
        "parseable_transition_per_budget": _ratio(
            counts["parseable_transitions"], transition_budget
        ),
        "exact_transition_per_budget": _ratio(
            counts["exact_transitions"], transition_budget
        ),
        "exact_prefix_survival_per_budget": _ratio(
            counts["exact_prefix_steps"], transition_budget
        ),
        "fully_parseable": _ratio(counts["fully_parseable"], denominator),
        "state_closed_loop_exact": _ratio(
            counts["state_closed_loop_exact"], denominator
        ),
        "terminal_transition_exact": _ratio(
            counts["terminal_transition_exact"], denominator
        ),
        "terminal_reached": _ratio(counts["terminal_reached"], denominator),
        "final_answer_correct": _ratio(counts["final_answer_correct"], denominator),
        "closed_loop_success": _ratio(counts["closed_loop_success"], denominator),
    }
    failures = Counter(
        "success"
        if row["first_failure_position"] is None
        else f"p{row['first_failure_position']}:{row['first_failure_reason']}"
        for row in rows
    )
    return {
        "counts": counts,
        "rates": rates,
        "first_failure_distribution": dict(sorted(failures.items())),
    }


def _summarize_pairs(pairs: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    rows = list(pairs)
    counts = {
        "pairs": len(rows),
        "expected_answer_changed": sum(
            int(row["expected_answer_changed"]) for row in rows
        ),
        "state_intervention_at_expected_position": sum(
            int(row["state_intervention_at_expected_position"]) for row in rows
        ),
        "both_state_closed_loop_exact": sum(
            int(row["both_state_closed_loop_exact"]) for row in rows
        ),
        "both_final_answers_correct": sum(
            int(row["both_final_answers_correct"]) for row in rows
        ),
        "answer_intervention_success": sum(
            int(row["answer_intervention_success"]) for row in rows
        ),
        "both_closed_loop_success": sum(
            int(row["both_closed_loop_success"]) for row in rows
        ),
    }
    denominator = counts["pairs"]
    rates = {
        name: _ratio(value, denominator)
        for name, value in counts.items()
        if name != "pairs"
    }
    expected_divergence = Counter(
        "none"
        if row["first_expected_state_divergence_position"] is None
        else str(row["first_expected_state_divergence_position"])
        for row in rows
    )
    predicted_divergence = Counter(
        "none"
        if row["first_predicted_state_divergence_position"] is None
        else str(row["first_predicted_state_divergence_position"])
        for row in rows
    )
    return {
        "counts": counts,
        "rates": rates,
        "first_expected_state_divergence_distribution": dict(
            sorted(expected_divergence.items())
        ),
        "first_predicted_state_divergence_distribution": dict(
            sorted(predicted_divergence.items())
        ),
    }


def _grouped_summary(
    values: list[Mapping[str, Any]],
    key: Callable[[Mapping[str, Any]], str],
    summarize: Callable[[Iterable[Mapping[str, Any]]], dict[str, Any]],
) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for value in values:
        grouped[key(value)].append(value)
    return {name: summarize(grouped[name]) for name in sorted(grouped)}


def _width_8_survival(branches: list[Mapping[str, Any]]) -> dict[str, Any]:
    selected = [row for row in branches if row["width"] == 8]
    positions: list[dict[str, Any]] = []
    for position in range(8):
        attempted = 0
        parseable = 0
        exact = 0
        prefix_survived = 0
        for branch in selected:
            if position < len(branch["rows"]):
                attempted += 1
                step = branch["rows"][position]
                parseable += int(step["predicted_state"] is not None)
                exact += int(step["correct"])
            prefix_survived += int(branch["prefix_exact_length"] > position)
        positions.append(
            {
                "position": position,
                "branches": len(selected),
                "attempted": attempted,
                "parseable": parseable,
                "exact_transition": exact,
                "exact_prefix_survived": prefix_survived,
                "exact_prefix_survival_rate": _ratio(prefix_survived, len(selected)),
            }
        )
    return {
        "branches": len(selected),
        "positions": positions,
        "terminal": _summarize_branches(selected),
    }


def build_metrics(records: list[Mapping[str, Any]]) -> dict[str, Any]:
    branches: list[Mapping[str, Any]] = []
    tagged_branches: list[dict[str, Any]] = []
    for pair in records:
        for branch_name in ("normal", "counterfactual"):
            branch = pair[branch_name]
            branches.append(branch)
            tagged_branches.append({**branch, "branch": branch_name})
    return {
        "branches": {
            "overall": _summarize_branches(branches),
            "by_branch": _grouped_summary(
                tagged_branches, lambda row: str(row["branch"]), _summarize_branches
            ),
            "by_regime": _grouped_summary(
                branches, lambda row: str(row["regime"]), _summarize_branches
            ),
            "by_operation": _grouped_summary(
                branches, lambda row: str(row["operation"]), _summarize_branches
            ),
            "by_width": _grouped_summary(
                branches, lambda row: str(row["width"]), _summarize_branches
            ),
            "by_terminal_carry_class": _grouped_summary(
                branches,
                lambda row: f"{row['operation']}:{row['terminal_carry_class']}",
                _summarize_branches,
            ),
            "by_operation_width_terminal_carry_class": _grouped_summary(
                branches,
                lambda row: (
                    f"{row['operation']}|w{row['width']}|{row['terminal_carry_class']}"
                ),
                _summarize_branches,
            ),
            "width_8_survival": _width_8_survival(branches),
        },
        "pairs": {
            "overall": _summarize_pairs(records),
            "by_regime": _grouped_summary(
                records, lambda row: str(row["regime"]), _summarize_pairs
            ),
            "by_operation": _grouped_summary(
                records, lambda row: str(row["operation"]), _summarize_pairs
            ),
            "by_width": _grouped_summary(
                records, lambda row: str(row["width"]), _summarize_pairs
            ),
            "by_terminal_carry_pair": _grouped_summary(
                records,
                lambda row: (
                    f"{row['operation']}:"
                    f"{row['normal_terminal_carry_class']}->"
                    f"{row['counterfactual_terminal_carry_class']}"
                ),
                _summarize_pairs,
            ),
        },
    }


def validate_accounting(
    records: list[Mapping[str, Any]], metrics: Mapping[str, Any]
) -> dict[str, Any]:
    if len(records) != PAIR_COUNT:
        raise ContractError("evaluated pair count differs from frozen budget")
    regime_counts = Counter(str(row["regime"]) for row in records)
    if dict(sorted(regime_counts.items())) != REGIME_BUDGETS:
        raise ContractError("evaluated regime counts differ from frozen budget")
    branches = [row[name] for row in records for name in ("normal", "counterfactual")]
    if len(branches) != BRANCH_COUNT:
        raise ContractError("evaluated branch count differs from frozen budget")
    expected_transition_budget = sum(
        2 * REGIME_BUDGETS[regime] * width for regime, width in REGIME_WIDTHS.items()
    )
    overall = metrics["branches"]["overall"]["counts"]
    if (
        overall["branches"] != BRANCH_COUNT
        or overall["transition_budget"] != expected_transition_budget
        or overall["transition_calls"] > expected_transition_budget
        or overall["final_prompt_issued"] > BRANCH_COUNT
    ):
        raise ContractError("rollout accounting is impossible")
    actual_generation_calls = (
        overall["transition_calls"] + overall["final_prompt_issued"]
    )
    return {
        "pairs": PAIR_COUNT,
        "branches": BRANCH_COUNT,
        "by_regime": dict(sorted(regime_counts.items())),
        "transition_budget": expected_transition_budget,
        "transition_calls": overall["transition_calls"],
        "max_final_calls": BRANCH_COUNT,
        "actual_final_calls": overall["final_prompt_issued"],
        "max_generation_calls": expected_transition_budget + BRANCH_COUNT,
        "actual_generation_calls": actual_generation_calls,
        "early_termination_is_scored_failure_not_repaired": True,
    }


def publish_one_report(
    output_dir: str | Path, report: Mapping[str, Any]
) -> tuple[Path, str]:
    """Publish one canonical report into a fresh, then sealed, directory."""
    output_dir = Path(output_dir)
    if os.path.lexists(output_dir):
        raise ContractError(f"refusing existing output directory: {output_dir}")
    parent = output_dir.parent
    if parent.is_symlink() or not parent.is_dir():
        raise ContractError("output parent must be an existing regular directory")
    output_dir.mkdir(mode=0o700)
    target = output_dir / "report.json"
    payload = canonical_json_bytes(report)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".report.", suffix=".tmp", dir=output_dir
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as sink:
            sink.write(payload)
            sink.flush()
            os.fsync(sink.fileno())
        os.chmod(temporary, 0o400)
        os.link(temporary, target, follow_symlinks=False)
        temporary.unlink()
        directory_fd = os.open(output_dir, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        os.chmod(output_dir, 0o500)
    except Exception:
        try:
            temporary.unlink(missing_ok=True)
        finally:
            raise
    if {path.name for path in output_dir.iterdir()} != {"report.json"}:
        raise ContractError("one-purpose publication contains extra files")
    frozen = read_frozen_file(
        target,
        sha256_bytes(payload),
        expected_size=len(payload),
        expected_mode=0o400,
    )
    if stat.S_IMODE(output_dir.stat().st_mode) != 0o500:
        raise ContractError("published output directory lost its seal")
    return target, frozen.sha256


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=tuple(ARM_CONTRACTS), required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--episodes", required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--source-binding", required=True)
    parser.add_argument("--source-binding-sha256", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    contract = ARM_CONTRACTS[args.arm]
    runtime = require_exact_h100()
    source_binding_file = read_frozen_file(
        args.source_binding,
        args.source_binding_sha256,
        expected_mode=0o400,
    )
    source_binding = validate_source_binding(
        source_binding_file, args.source_root, contract
    )
    heldout_file = read_frozen_file(args.episodes, HELDOUT_SHA256, expected_mode=0o400)
    tokenizer_file = read_frozen_file(
        args.tokenizer,
        TOKENIZER_SHA256,
        expected_size=TOKENIZER_BYTES,
        expected_mode=0o400,
    )
    episodes = load_heldout(heldout_file)
    checkpoint, training_provenance = validate_training_bundle(
        args.checkpoint_dir, contract
    )

    try:
        from tokenizers import Tokenizer
        from tokenizers import __version__ as tokenizers_version
        from model import GPT, GPTConfig
    except ImportError as error:
        raise ContractError(
            "canonical model/tokenizer runtime is unavailable"
        ) from error

    tokenizer = Tokenizer.from_str(tokenizer_file.payload.decode("utf-8"))
    if (
        tokenizer.get_vocab_size() != TOKENIZER_VOCAB_SIZE
        or tokenizer.token_to_id("<|endoftext|>") is None
    ):
        raise ContractError("tokenizer vocabulary contract mismatch")
    try:
        model = GPT(GPTConfig(**checkpoint["cfg"]))
        model.load_state_dict(checkpoint["model"], strict=True)
    except Exception as error:
        raise ContractError("checkpoint model/config schema mismatch") from error
    del checkpoint
    gc.collect()
    model = model.to("cuda:0").eval()
    torch.manual_seed(DECODE_SEED)
    torch.cuda.manual_seed_all(DECODE_SEED)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.cuda.synchronize()

    def ask(prompt: str, max_new: int, _: str) -> Generation:
        return greedy_generate(model, tokenizer, prompt, "cuda:0", max_new=max_new)

    records: list[dict[str, Any]] = []
    for index, episode in enumerate(episodes, 1):
        records.append(evaluate_pair(episode, ask))
        if index % 25 == 0 or index == len(episodes):
            successes = sum(int(row["both_closed_loop_success"]) for row in records)
            print(
                f"[factorial-eval] arm={contract.arm} pairs={index}/{len(episodes)} "
                f"both_success={successes}",
                flush=True,
            )
    metrics = build_metrics(records)
    accounting = validate_accounting(records, metrics)
    transcript_sha256 = sha256_bytes(canonical_json_bytes(records))
    selected_ids_sha256 = sha256_bytes(
        canonical_json_bytes([episode["id"] for episode in episodes])
    )
    report = {
        "audit": AUDIT,
        "status": "complete",
        "arm": contract.arm,
        "training": training_provenance,
        "heldout": {
            "episodes_sha256": heldout_file.sha256,
            "tokenizer_sha256": tokenizer_file.sha256,
            "tokenizer_bytes": tokenizer_file.size,
            "tokenizer_vocab_size": TOKENIZER_VOCAB_SIZE,
            "pair_count": PAIR_COUNT,
            "branch_count": BRANCH_COUNT,
            "regime_budgets": REGIME_BUDGETS,
            "selected_ids_sha256": selected_ids_sha256,
            "selection": "all_rows_in_frozen_file_order_no_sampling",
        },
        "decode_contract": {
            "method": "greedy_argmax",
            "temperature": 0.0,
            "top_k": None,
            "transition_max_new": TRANSITION_MAX_NEW,
            "final_max_new": FINAL_MAX_NEW,
            "prompt_style": PROMPT_STYLE,
            "seed": DECODE_SEED,
            "normal_and_counterfactual": True,
            "incorrect_parseable_state_is_forwarded_unchanged": True,
            "malformed_state_terminates_branch": True,
            "solver_repair": False,
            "host_arithmetic_during_rollout": False,
            "sampling_or_rescoring": False,
        },
        "runtime": {
            **runtime,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "tokenizers": tokenizers_version,
            "cuda_runtime": torch.version.cuda,
            "source_binding_sha256": source_binding_file.sha256,
            "source_binding": source_binding,
        },
        "accounting": accounting,
        "metrics": metrics,
        "transcript_count": len(records),
        "transcripts_sha256": transcript_sha256,
        "transcripts": records,
        "claim_boundary": (
            "This report measures greedy model-authored execution of the frozen digitwise "
            "protocol on the exact normal/counterfactual held-out board. It does not establish "
            "language parsing, unrestricted arithmetic, broad reasoning, or owner-proof artifact "
            "immutability. Permission seals prevent accidental writes only."
        ),
    }
    output_path, output_sha256 = publish_one_report(args.output_dir, report)
    print(
        json.dumps(
            {
                "arm": contract.arm,
                "output": str(output_path),
                "output_sha256": output_sha256,
                "transcripts_sha256": transcript_sha256,
                "metrics": metrics["branches"]["overall"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
