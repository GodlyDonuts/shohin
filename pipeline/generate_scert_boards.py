#!/usr/bin/env python3
"""Build SCERT public/training boards and verify external hidden-board custody.

This module deliberately cannot generate or sign hidden confirmation boards.  It
emits the exact hidden-board specifications and verifies a detached Ed25519
receipt from a separately supplied trust anchor.  The hidden ciphertext and
domain keys remain outside this process.
"""

from __future__ import annotations

import argparse
import base64
import ctypes
import errno
import hashlib
import json
import math
import os
import platform
import stat
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
import torch
from tokenizers import Tokenizer


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train"))
from digitwise_protocol import (  # noqa: E402
    apply_microstep,
    canonical_state,
    initial_state,
    state_answer,
)


PROTOCOL_ID = "R12-SCERT-EXECUTION-CPU-v1"
THEORY_PATH = ROOT / "R12_SELF_CANONICALIZING_EPOCH_RETIREMENT_THEORY.md"
THEORY_SHA256 = "d315b107b6ce3d486a83e091168f027356007c41dd17dd0f8f2f9d7441281dc4"
TOKENIZER_PATH = ROOT / "artifacts/shohin-tok-32k.json"
TOKENIZER_SHA256 = "87532df5c121753de3b29194e1f9e3de47986d3f5359548fdf93606773a233d4"
PARENT_PATH = ROOT / "train/sft_digitwise_recurrent_v2_200k_r3/sft_ep1.pt"
PARENT_SHA256 = "d79e9df26caecb9801118d1bf68bd7b85381a06b256f23478acffe40a2108459"
PUBLIC_HELDOUT_PATH = ROOT / "artifacts/evals/digitwise_recurrent_v2_heldout.jsonl"
PUBLIC_HELDOUT_SHA256 = (
    "89ce11b36ff2f56e83cda72a1f07b1a90f4a3dc3803c69db2779a27219712646"
)
TRAINING_SEEDS = (2026071814, 2026071815, 2026071816)
OPERATIONS = ("add", "sub")
PATTERNS = tuple(range(8))
EOS_ID = 0
NEUTRAL_ID = 233

SCAFFOLD = (
    "<|system|>\n"
    "SCERT-DWS-v1.\n"
    "Current state is the only mutable state.\n"
    "If it is nonterminal, emit exactly one canonical successor state and end the response.\n"
    "If it is terminal, emit exactly answer=<integer> and end the response.\n"
    "Emit no explanation.\n"
    "<|user|>\n"
    "Current state:\n"
    "{STATE}"
    "\n<|assistant|>\n"
)
G_L_TEXT, G_R_TEXT = SCAFFOLD.split("{STATE}")

D12_IDS = (
    "fit_w4-00258",
    "value_ood_w4-00217",
    "fit_w4-00261",
    "fit_w4-00196",
    "fit_w6-00122",
    "value_ood_w6-00028",
    "value_ood_w6-00280",
    "value_ood_w6-00067",
    "width_ood_w8-00120",
    "width_ood_w8-00176",
    "width_ood_w8-00180",
    "width_ood_w8-00103",
)
D12_IDS_SHA256 = "1dc75ec7995e61a85f7bec9ae1fa62aa1adaf71bd46172e880aea901482396b9"

HIDDEN_BOARD_SPECS: dict[str, dict[str, Any]] = {
    "H_B": {
        "kind": "boundary_head_holdout",
        "episodes": 384,
        "rows": 2688,
        "width_episode_counts": {"4": 128, "6": 128, "8": 128},
        "width_row_counts": {"4": 640, "6": 896, "8": 1152},
        "action_row_counts": {"COMMIT": 2304, "HALT": 384},
        "cell_episode_count": 64,
        "cell_row_counts": {
            "4-add": {"COMMIT": 256, "HALT": 64, "rows": 320},
            "4-sub": {"COMMIT": 256, "HALT": 64, "rows": 320},
            "6-add": {"COMMIT": 384, "HALT": 64, "rows": 448},
            "6-sub": {"COMMIT": 384, "HALT": 64, "rows": 448},
            "8-add": {"COMMIT": 512, "HALT": 64, "rows": 576},
            "8-sub": {"COMMIT": 512, "HALT": 64, "rows": 576},
        },
    },
    "H_M": {
        "kind": "one_dispatch_mechanistic",
        "cases": 384,
        "cell_case_count": 64,
        "width_case_counts": {"4": 128, "6": 128, "8": 128},
    },
    "H_A": {
        "kind": "autonomous_and_target_switch",
        "episodes": 768,
        "cell_episode_count": 128,
        "width_episode_counts": {"4": 256, "6": 256, "8": 256},
    },
    "H_O": {
        "kind": "observational_candidates",
        "candidate_pairs": 432,
        "strata": 18,
        "pairs_per_stratum": 24,
        "minimum_admitted_total": 216,
        "minimum_admitted_per_stratum": 12,
    },
}


class ContractError(RuntimeError):
    """A fail-closed SCERT contract violation."""


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode(
        "ascii"
    )


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def is_sha256_hex(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def file_receipt(path: Path) -> dict[str, Any]:
    target = Path(path)
    info = target.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise ContractError(f"source is not a single-link regular file: {target}")
    resolved = target.resolve()
    try:
        rendered_path = str(resolved.relative_to(ROOT))
    except ValueError:
        rendered_path = str(resolved)
    return {
        "path": rendered_path,
        "mode": stat.S_IMODE(info.st_mode),
        "bytes": info.st_size,
        "sha256": sha256_file(target),
    }


def _rename_noreplace(source: Path, destination: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    source_b = os.fsencode(source)
    destination_b = os.fsencode(destination)
    system = platform.system()
    if system == "Linux" and hasattr(libc, "renameat2"):
        at_fdcwd = -100
        rename_noreplace = 1
        result = libc.renameat2(
            at_fdcwd,
            ctypes.c_char_p(source_b),
            at_fdcwd,
            ctypes.c_char_p(destination_b),
            rename_noreplace,
        )
    elif system == "Darwin" and hasattr(libc, "renamex_np"):
        rename_excl = 0x00000004
        result = libc.renamex_np(
            ctypes.c_char_p(source_b), ctypes.c_char_p(destination_b), rename_excl
        )
    else:
        raise ContractError("atomic no-replace rename is unavailable on this platform")
    if result != 0:
        error = ctypes.get_errno()
        if error in (errno.EEXIST, errno.ENOTEMPTY):
            raise FileExistsError(destination)
        raise OSError(error, os.strerror(error), str(destination))


def atomic_publish_json(path: Path, value: Any, expected_schema: str) -> dict[str, Any]:
    """Crash-atomically publish one immutable JSON artifact without overwrite."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    parent_info = target.parent.lstat()
    if (
        not stat.S_ISDIR(parent_info.st_mode)
        or parent_info.st_uid != os.getuid()
        or stat.S_IMODE(parent_info.st_mode) != 0o700
    ):
        raise ContractError(
            "output directory must be a current-UID-owned mode-0700 directory"
        )
    if target.exists() or target.is_symlink():
        raise FileExistsError(target)
    if not isinstance(value, dict) or value.get("schema") != expected_schema:
        raise ContractError("artifact schema differs before publication")
    payload = canonical_json_bytes(value)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".partial", dir=target.parent
    )
    temporary = Path(temporary_name)
    published = False
    before = None
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "wb", closefd=True) as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        before = temporary.lstat()
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise ContractError("temporary artifact is aliased or non-regular")
        _rename_noreplace(temporary, target)
        published = True
        directory_fd = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        verify_fd = os.open(target, flags)
        try:
            current = os.fstat(verify_fd)
            if (
                not stat.S_ISREG(current.st_mode)
                or current.st_nlink != 1
                or current.st_uid != os.getuid()
                or (current.st_dev, current.st_ino) != (before.st_dev, before.st_ino)
            ):
                raise ContractError("published artifact identity changed")
            reopened = b""
            while True:
                block = os.read(verify_fd, 1024 * 1024)
                if not block:
                    break
                reopened += block
        finally:
            os.close(verify_fd)
        if reopened != payload or sha256_bytes(reopened) != sha256_bytes(payload):
            raise ContractError("published artifact bytes changed")
        decoded = json.loads(reopened)
        if decoded.get("schema") != expected_schema:
            raise ContractError("published artifact schema changed")
        os.chmod(target, 0o444)
        directory_fd = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        return file_receipt(target)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        if published and target.exists():
            target.unlink()
            directory_fd = os.open(target.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        raise


def _domain_bytes(seed: int, domain: str, index: int, attempt: int) -> bytes:
    return hashlib.sha256(
        PROTOCOL_ID.encode("ascii")
        + b"\0"
        + str(seed).encode("ascii")
        + b"\0"
        + domain.encode("ascii")
        + b"\0"
        + str(index).encode("ascii")
        + b"\0"
        + str(attempt).encode("ascii")
    ).digest()


def _digit(raw: bytes, cursor: int, modulus: int) -> tuple[int, int]:
    if modulus <= 0:
        raise ContractError("invalid deterministic digit modulus")
    if cursor >= len(raw):
        raw = hashlib.sha256(raw).digest()
        cursor = 0
    return raw[cursor] % modulus, cursor + 1


def _desired_intermediate(pattern: int, position: int) -> int:
    return (int(pattern) >> (position % 3)) & 1


def _make_operands(
    seed: int,
    domain: str,
    operation: str,
    width: int,
    pattern: int,
    replicate: int,
    attempt: int,
) -> tuple[int, int]:
    raw = _domain_bytes(seed, domain, replicate, attempt)
    cursor = 0
    left_digits: list[int] = []
    right_digits: list[int] = []
    incoming = 0
    for position in range(width):
        desired = (
            _desired_intermediate(pattern, position)
            if position < width - 1
            else (raw[-1] & 1 if operation == "add" else 0)
        )
        if operation == "add":
            if desired == 0:
                left, cursor = _digit(raw, cursor, 10 - incoming)
                right, cursor = _digit(raw, cursor, 10 - left - incoming)
            else:
                left, cursor = _digit(raw, cursor, 10)
                if left + incoming == 0:
                    left = 1
                minimum = 10 - left - incoming
                right_offset, cursor = _digit(raw, cursor, 10 - minimum)
                right = minimum + right_offset
        elif operation == "sub":
            if desired == 0:
                right, cursor = _digit(raw, cursor, 10 - incoming)
                minimum = right + incoming
                left_offset, cursor = _digit(raw, cursor, 10 - minimum)
                left = minimum + left_offset
            else:
                if incoming:
                    right, cursor = _digit(raw, cursor, 10)
                else:
                    right_offset, cursor = _digit(raw, cursor, 9)
                    right = 1 + right_offset
                left, cursor = _digit(raw, cursor, right + incoming)
        else:
            raise ContractError("invalid operation")
        left_digits.append(left)
        right_digits.append(right)
        if operation == "add":
            incoming = int(left + right + incoming >= 10)
        else:
            incoming = int(left - right - incoming < 0)
        if incoming != desired:
            raise ContractError("operand constructor failed its carry pattern")
    left_value = sum(value * (10**index) for index, value in enumerate(left_digits))
    right_value = sum(value * (10**index) for index, value in enumerate(right_digits))
    if operation == "sub" and left_value < right_value:
        raise ContractError("subtraction constructor produced a negative result")
    return left_value, right_value


def _actual_pattern(operation: str, left: int, right: int, width: int) -> int:
    state = initial_state(operation, left, right, width)
    bits = 0
    for position in range(width - 1):
        state = apply_microstep(state)
        bits |= int(state["c"]) << (position % 3)
    return bits & 7


def semantic_episode_key(operation: str, width: int, left: int, right: int) -> str:
    left_text = f"{left:0{width}d}"
    right_text = f"{right:0{width}d}"
    if operation == "add" and right_text < left_text:
        left_text, right_text = right_text, left_text
    return f"SCERT-DWS-v1|{operation}|{width}|{left_text}|{right_text}"


def _lane_records(
    operation: str, width: int, left: int, right: int
) -> list[dict[str, Any]]:
    state = initial_state(operation, left, right, width)
    lanes: list[dict[str, Any]] = []
    for step_index in range(width):
        successor = apply_microstep(state)
        lanes.append(
            {
                "step_index": step_index,
                "P": canonical_state(state),
                "X": canonical_state(successor),
                "action": "COMMIT",
            }
        )
        state = successor
    lanes.append(
        {
            "step_index": width,
            "P": canonical_state(state),
            "X": f"answer={state_answer(state)}",
            "action": "HALT",
        }
    )
    return lanes


def _episode_record(
    board: str,
    episode_id: str,
    operation: str,
    width: int,
    left: int,
    right: int,
    pattern: int,
    tokenizer: Tokenizer,
) -> dict[str, Any]:
    if _actual_pattern(operation, left, right, width) != pattern:
        raise ContractError("episode pattern differs from requested pattern")
    lanes = _lane_records(operation, width, left, right)
    tokenized_lanes = []
    transition_keys = []
    for lane in lanes:
        p_ids = tokenizer.encode(lane["P"], add_special_tokens=False).ids
        x_ids = tokenizer.encode(lane["X"], add_special_tokens=False).ids
        if not p_ids or not x_ids or len(p_ids) > 512 or len(x_ids) > 512:
            raise ContractError("SCERT lane exceeds the frozen slot contract")
        lane_payload = {
            "step_index": lane["step_index"],
            "P_ids": p_ids,
            "X_ids": x_ids,
            "action": lane["action"],
        }
        lane_payload["token_row_sha256"] = sha256_bytes(
            canonical_json_bytes(lane_payload)
        )
        tokenized_lanes.append(lane_payload)
        transition_keys.append(
            sha256_bytes(
                canonical_json_bytes(
                    {
                        "episode": semantic_episode_key(operation, width, left, right),
                        "step_index": lane["step_index"],
                        "current": lane["P"],
                        "successor": lane["X"],
                    }
                )
            )
        )
    record = {
        "board": board,
        "id": episode_id,
        "operation": operation,
        "width": width,
        "left": left,
        "right": right,
        "pattern": pattern,
        "semantic_episode_key": semantic_episode_key(operation, width, left, right),
        "transition_key_sha256": transition_keys,
        "initial_state": canonical_state(initial_state(operation, left, right, width)),
        "lanes": lanes,
        "tokenized_lanes": tokenized_lanes,
    }
    exact_payload = dict(record)
    record["canonical_episode_sha256"] = sha256_bytes(
        canonical_json_bytes(exact_payload)
    )
    return record


def _load_d12(tokenizer: Tokenizer) -> list[dict[str, Any]]:
    if sha256_file(PUBLIC_HELDOUT_PATH) != PUBLIC_HELDOUT_SHA256:
        raise ContractError("public heldout source hash changed")
    rows = {
        json.loads(line)["id"]: json.loads(line)
        for line in PUBLIC_HELDOUT_PATH.read_text().splitlines()
        if line.strip()
    }
    selected = []
    for episode_id in D12_IDS:
        source = rows.get(episode_id)
        if source is None:
            raise ContractError("a frozen D_12 episode is missing")
        width = int(source["width"])
        operation = str(source["operation"])
        left = int(source["left"])
        right = int(source["right"])
        pattern = _actual_pattern(operation, left, right, width)
        selected.append(
            _episode_record(
                "D_12", episode_id, operation, width, left, right, pattern, tokenizer
            )
        )
    frozen_id_hash = sha256_bytes(("\n".join(D12_IDS) + "\n").encode("ascii"))
    if frozen_id_hash != D12_IDS_SHA256:
        raise ContractError("D_12 ordered identity changed")
    cells = Counter((row["width"], row["operation"]) for row in selected)
    if cells != Counter(
        {(width, operation): 2 for width in (4, 6, 8) for operation in OPERATIONS}
    ):
        raise ContractError("D_12 cell balance changed")
    return selected


def _build_balanced_board(
    board: str,
    seed: int,
    width: int,
    per_pattern: int,
    tokenizer: Tokenizer,
    reserved_semantic: set[str],
) -> list[dict[str, Any]]:
    episodes: list[dict[str, Any]] = []
    for operation in OPERATIONS:
        for pattern in PATTERNS:
            accepted = 0
            attempt = 0
            while accepted < per_pattern:
                left, right = _make_operands(
                    seed,
                    f"{board}|{operation}|{width}|{pattern}",
                    operation,
                    width,
                    pattern,
                    accepted,
                    attempt,
                )
                attempt += 1
                key = semantic_episode_key(operation, width, left, right)
                if key in reserved_semantic:
                    if attempt > per_pattern * 1000:
                        raise ContractError(
                            "could not find a disjoint deterministic episode"
                        )
                    continue
                reserved_semantic.add(key)
                episode_id = f"{board}-{operation}-p{pattern}-{accepted:04d}"
                episodes.append(
                    _episode_record(
                        board,
                        episode_id,
                        operation,
                        width,
                        left,
                        right,
                        pattern,
                        tokenizer,
                    )
                )
                accepted += 1
    return episodes


def _board_receipt(name: str, episodes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    exact = [str(row["canonical_episode_sha256"]) for row in episodes]
    token_rows = [
        str(lane["token_row_sha256"])
        for row in episodes
        for lane in row["tokenized_lanes"]
    ]
    semantic = [str(row["semantic_episode_key"]) for row in episodes]
    transitions = [
        str(value) for row in episodes for value in row["transition_key_sha256"]
    ]
    return {
        "name": name,
        "episodes": len(episodes),
        "rows": len(token_rows),
        "canonical_set_sha256": sha256_bytes(canonical_json_bytes(sorted(exact))),
        "token_row_set_sha256": sha256_bytes(canonical_json_bytes(sorted(token_rows))),
        "semantic_episode_set_sha256": sha256_bytes(
            canonical_json_bytes(sorted(semantic))
        ),
        "semantic_transition_set_sha256": sha256_bytes(
            canonical_json_bytes(sorted(transitions))
        ),
        "payload_sha256": sha256_bytes(canonical_json_bytes(list(episodes))),
    }


def _assert_local_disjoint(
    boards: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    for name, rows in boards.items():
        exact_values = [str(row["canonical_episode_sha256"]) for row in rows]
        token_values = [
            str(lane["token_row_sha256"])
            for row in rows
            for lane in row["tokenized_lanes"]
        ]
        episode_values = [str(row["semantic_episode_key"]) for row in rows]
        transition_values = [
            str(value) for row in rows for value in row["transition_key_sha256"]
        ]
        expected_transitions = sum(len(row["lanes"]) for row in rows)
        if (
            len(set(exact_values)) != len(exact_values)
            or len(set(token_values)) != len(token_values)
            or len(set(episode_values)) != len(episode_values)
            or len(transition_values) != expected_transitions
            or len(set(transition_values)) != len(transition_values)
        ):
            raise ContractError(f"board contains duplicate or missing keys: {name}")
    kinds = {
        name: {
            "exact": {str(row["canonical_episode_sha256"]) for row in rows},
            "tokens": {
                str(lane["token_row_sha256"])
                for row in rows
                for lane in row["tokenized_lanes"]
            },
            "episodes": {str(row["semantic_episode_key"]) for row in rows},
            "transitions": {
                str(value) for row in rows for value in row["transition_key_sha256"]
            },
        }
        for name, rows in boards.items()
    }
    names = sorted(kinds)
    comparisons = []
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            for key_type in ("exact", "tokens", "episodes", "transitions"):
                overlap = kinds[left][key_type] & kinds[right][key_type]
                if overlap:
                    raise ContractError(
                        f"local boards overlap: {left}/{right}/{key_type}"
                    )
            comparisons.append({"left": left, "right": right, "all_intersections": 0})
    commitment = {
        name: {
            key: sha256_bytes(canonical_json_bytes(sorted(values)))
            for key, values in sorted(payload.items())
        }
        for name, payload in sorted(kinds.items())
    }
    return {
        "schema": "r12-scert-local-disjointness-v1",
        "comparisons": comparisons,
        "set_commitments": commitment,
        "all_pairwise_intersections": 0,
    }


def build_local_registry() -> tuple[dict[str, Any], dict[str, Any]]:
    for path, expected in (
        (THEORY_PATH, THEORY_SHA256),
        (TOKENIZER_PATH, TOKENIZER_SHA256),
        (PARENT_PATH, PARENT_SHA256),
    ):
        if sha256_file(path) != expected:
            raise ContractError(f"frozen input hash changed: {path}")
    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
    g_l_ids = tokenizer.encode(G_L_TEXT, add_special_tokens=False).ids
    g_r_ids = tokenizer.encode(G_R_TEXT, add_special_tokens=False).ids
    zero_ids = tokenizer.encode("0", add_special_tokens=False).ids
    one_ids = tokenizer.encode("1", add_special_tokens=False).ids
    neutral_ids = tokenizer.encode(" ", add_special_tokens=False).ids
    if len(g_l_ids) != 70 or len(g_r_ids) != 3:
        raise ContractError("scaffold token lengths changed")
    if zero_ids != [28] or one_ids != [29] or neutral_ids != [NEUTRAL_ID]:
        raise ContractError("frozen digit or neutral token IDs changed")
    if EOS_ID in (zero_ids[0], one_ids[0]) or zero_ids[0] == one_ids[0]:
        raise ContractError("effective-logit token identities are invalid")

    d12 = _load_d12(tokenizer)
    reserved = {str(row["semantic_episode_key"]) for row in d12}
    boards: dict[str, list[dict[str, Any]]] = {"D_12": d12}
    for seed in TRAINING_SEEDS:
        name = f"T_{str(seed)[-2:]}"
        boards[name] = _build_balanced_board(name, seed, 4, 128, tokenizer, reserved)
        if len(boards[name]) != 2048:
            raise ContractError("training board size changed")
    boards["D_256"] = _build_balanced_board(
        "D_256", 2026071817, 4, 16, tokenizer, reserved
    )
    if len(boards["D_256"]) != 256:
        raise ContractError("development board size changed")

    disjointness = _assert_local_disjoint(boards)
    receipts = {
        name: _board_receipt(name, rows) for name, rows in sorted(boards.items())
    }
    registry = {
        "schema": "r12-scert-local-board-registry-v1",
        "protocol": PROTOCOL_ID,
        "claim_boundary": (
            "Public and training board freeze plus CPU mechanics only; no hidden board, "
            "model fit, capability score, confirmation result, or GPU authority."
        ),
        "theory": file_receipt(THEORY_PATH),
        "parent": file_receipt(PARENT_PATH),
        "tokenizer": file_receipt(TOKENIZER_PATH),
        "public_heldout": file_receipt(PUBLIC_HELDOUT_PATH),
        "scaffold": {
            "ascii_sha256": sha256_bytes(SCAFFOLD.encode("ascii")),
            "g_l_text_sha256": sha256_bytes(G_L_TEXT.encode("ascii")),
            "g_r_text_sha256": sha256_bytes(G_R_TEXT.encode("ascii")),
            "g_l_ids": g_l_ids,
            "g_r_ids": g_r_ids,
            "dummy_id": EOS_ID,
            "neutral_id": NEUTRAL_ID,
            "v0": zero_ids[0],
            "v1": one_ids[0],
        },
        "training_seeds": list(TRAINING_SEEDS),
        "boards": boards,
        "board_receipts": receipts,
        "local_disjointness": disjointness,
        "hidden_content_present": False,
    }
    registry_sha = sha256_bytes(canonical_json_bytes(registry))
    request = {
        "schema": "r12-scert-external-custody-request-v1",
        "protocol": PROTOCOL_ID,
        "theory_sha256": THEORY_SHA256,
        "builder_id": f"source:{sha256_file(Path(__file__))}",
        "local_registry_sha256": registry_sha,
        "local_board_receipts": receipts,
        "local_key_set_commitment_sha256": sha256_bytes(
            canonical_json_bytes(disjointness["set_commitments"])
        ),
        "hidden_board_specs": HIDDEN_BOARD_SPECS,
        "required_custody": {
            "encrypted_before_fit": True,
            "independent_ed25519_signature": True,
            "domain_keys_withheld": True,
            "zero_exact_and_semantic_intersections": True,
            "no_hidden_bytes_or_labels_in_receipt": True,
            "reveal_after_three_stage1_and_six_head_artifacts": True,
        },
        "claim_boundary": (
            "This request contains specifications and commitments only. It cannot create, "
            "sign, decrypt, reveal, or score a hidden board."
        ),
    }
    validate_custody_request(request)
    return registry, request


def _load_public_key(path: Path) -> Ed25519PublicKey:
    anchor = Path(path)
    info = anchor.lstat()
    if (
        not stat.S_ISREG(info.st_mode)
        or info.st_nlink != 1
        or stat.S_IMODE(info.st_mode) & 0o222
    ):
        raise ContractError("trust anchor must be a read-only single-link regular file")
    raw = anchor.read_bytes()
    try:
        key = serialization.load_pem_public_key(raw)
    except ValueError:
        if len(raw) == 32:
            key = Ed25519PublicKey.from_public_bytes(raw)
        else:
            raise ContractError(
                "trust anchor is not a valid Ed25519 public key"
            ) from None
    if not isinstance(key, Ed25519PublicKey):
        raise ContractError("trust anchor is not Ed25519")
    return key


def independent_toy_board() -> list[dict[str, Any]]:
    """Independently enumerate the frozen 2 x 8 x 16 mechanics board."""
    stale_pairs = tuple(
        zip(
            (2, 4, 6, 10, 12, 14, 16, 18),
            (3, 5, 11, 13, 15, 17, 19, 21),
            strict=True,
        )
    )
    latest_spans = tuple(range(20, 30)) + (0, 1, 22, 23, 24, 25)
    result: list[dict[str, Any]] = []
    for wrapper_id in range(2):
        for pair_id, (positive, negative) in enumerate(stale_pairs):
            for span_id, latest in enumerate(latest_spans):
                result.append(
                    {
                        "id": f"toy-{wrapper_id}-{pair_id}-{span_id}",
                        "wrapper": (0, wrapper_id + 1),
                        "p_positive": (positive, negative),
                        "p_negative": (negative, positive),
                        "x": (latest, 30),
                    }
                )
    if len(result) != 256:
        raise ContractError("independent toy board size changed")
    return result


def validate_custody_request(request: Mapping[str, Any]) -> None:
    base_fields = {
        "schema",
        "protocol",
        "theory_sha256",
        "builder_id",
        "local_registry_sha256",
        "local_board_receipts",
        "local_key_set_commitment_sha256",
        "hidden_board_specs",
        "required_custody",
        "claim_boundary",
    }
    if set(request) not in (base_fields, base_fields | {"published_registry_receipt"}):
        raise ContractError("custody request schema differs")
    if (
        request["schema"] != "r12-scert-external-custody-request-v1"
        or request["protocol"] != PROTOCOL_ID
        or request["theory_sha256"] != THEORY_SHA256
        or request["hidden_board_specs"] != HIDDEN_BOARD_SPECS
    ):
        raise ContractError("custody request protocol or board specification differs")
    builder_id = request["builder_id"]
    if (
        not isinstance(builder_id, str)
        or not builder_id.startswith("source:")
        or not is_sha256_hex(builder_id.removeprefix("source:"))
    ):
        raise ContractError("custody request builder identity is invalid")
    if not is_sha256_hex(request["local_registry_sha256"]) or not is_sha256_hex(
        request["local_key_set_commitment_sha256"]
    ):
        raise ContractError("custody request local commitment is invalid")
    expected_counts = {
        "D_12": (12, 84),
        "D_256": (256, 1280),
        "T_14": (2048, 10240),
        "T_15": (2048, 10240),
        "T_16": (2048, 10240),
    }
    receipts = request["local_board_receipts"]
    if not isinstance(receipts, dict) or set(receipts) != set(expected_counts):
        raise ContractError("custody request local board set differs")
    for name, (episodes, rows) in expected_counts.items():
        entry = receipts[name]
        if (
            not isinstance(entry, dict)
            or set(entry)
            != {
                "name",
                "episodes",
                "rows",
                "canonical_set_sha256",
                "token_row_set_sha256",
                "semantic_episode_set_sha256",
                "semantic_transition_set_sha256",
                "payload_sha256",
            }
            or entry["name"] != name
            or (entry["episodes"], entry["rows"]) != (episodes, rows)
            or any(
                not is_sha256_hex(entry[key])
                for key in (
                    "canonical_set_sha256",
                    "token_row_set_sha256",
                    "semantic_episode_set_sha256",
                    "semantic_transition_set_sha256",
                    "payload_sha256",
                )
            )
        ):
            raise ContractError("custody request local board receipt differs")
    if request["required_custody"] != {
        "encrypted_before_fit": True,
        "independent_ed25519_signature": True,
        "domain_keys_withheld": True,
        "zero_exact_and_semantic_intersections": True,
        "no_hidden_bytes_or_labels_in_receipt": True,
        "reveal_after_three_stage1_and_six_head_artifacts": True,
    }:
        raise ContractError("custody request reveal contract differs")
    if "published_registry_receipt" in request:
        published = request["published_registry_receipt"]
        if (
            not isinstance(published, dict)
            or set(published) != {"path", "mode", "bytes", "sha256"}
            or published["mode"] != 0o444
            or not isinstance(published["bytes"], int)
            or published["bytes"] <= 0
            or published["sha256"] != request["local_registry_sha256"]
        ):
            raise ContractError("published registry receipt differs")


def verify_external_custody_receipt(
    receipt: Mapping[str, Any], request: Mapping[str, Any], trust_anchor: Path
) -> dict[str, Any]:
    validate_custody_request(request)
    required = {
        "schema",
        "protocol",
        "custodian_id",
        "request_sha256",
        "theory_sha256",
        "hidden_boards",
        "zero_intersection_certificate",
        "fit_reveal_gate",
        "signature_base64",
    }
    if set(receipt) != required:
        raise ContractError("custody receipt schema differs")
    if receipt["schema"] != "r12-scert-external-custody-receipt-v1":
        raise ContractError("custody receipt version differs")
    if receipt["protocol"] != PROTOCOL_ID or receipt["theory_sha256"] != THEORY_SHA256:
        raise ContractError("custody receipt protocol or theory differs")
    if (
        not isinstance(receipt["custodian_id"], str)
        or not receipt["custodian_id"].strip()
        or receipt["custodian_id"] == request["builder_id"]
    ):
        raise ContractError("self-attested custody is forbidden")
    request_sha = sha256_bytes(canonical_json_bytes(request))
    if receipt["request_sha256"] != request_sha:
        raise ContractError("custody receipt binds another request")
    boards = receipt["hidden_boards"]
    if set(boards) != set(HIDDEN_BOARD_SPECS):
        raise ContractError("custody receipt hidden-board set differs")
    forbidden_content_keys = {
        "rows",
        "episodes",
        "labels",
        "plaintext",
        "domain_key",
        "secret",
    }
    for name, spec in HIDDEN_BOARD_SPECS.items():
        entry = boards[name]
        if not isinstance(entry, dict) or set(entry) != {
            "spec_sha256",
            "declared_spec",
            "ciphertext_bytes",
            "ciphertext_sha256",
            "exact_key_set_commitment_sha256",
            "semantic_key_set_commitment_sha256",
        }:
            raise ContractError("hidden-board custody entry differs")
        if forbidden_content_keys & set(entry):
            raise ContractError("custody receipt exposes hidden content")
        if entry["declared_spec"] != spec or entry["spec_sha256"] != sha256_bytes(
            canonical_json_bytes(spec)
        ):
            raise ContractError("hidden-board specification differs")
        if (
            not isinstance(entry["ciphertext_bytes"], int)
            or entry["ciphertext_bytes"] <= 0
        ):
            raise ContractError("hidden-board ciphertext length is invalid")
        for key in (
            "ciphertext_sha256",
            "exact_key_set_commitment_sha256",
            "semantic_key_set_commitment_sha256",
        ):
            value = entry[key]
            if not is_sha256_hex(value):
                raise ContractError("hidden-board commitment is invalid")
    certificate = receipt["zero_intersection_certificate"]
    if set(certificate) != {
        "local_key_set_commitment_sha256",
        "private_key_set_commitments_sha256",
        "pairwise_exact_intersections",
        "pairwise_semantic_intersections",
    }:
        raise ContractError("zero-intersection certificate schema differs")
    if (
        certificate["local_key_set_commitment_sha256"]
        != request["local_key_set_commitment_sha256"]
        or certificate["pairwise_exact_intersections"] != 0
        or certificate["pairwise_semantic_intersections"] != 0
    ):
        raise ContractError("zero-intersection certificate failed")
    private_commitment = {
        name: {
            "exact": boards[name]["exact_key_set_commitment_sha256"],
            "semantic": boards[name]["semantic_key_set_commitment_sha256"],
        }
        for name in sorted(boards)
    }
    if certificate["private_key_set_commitments_sha256"] != sha256_bytes(
        canonical_json_bytes(private_commitment)
    ):
        raise ContractError("private key-set commitment rollup differs")
    if receipt["fit_reveal_gate"] != {
        "required_stage1_checkpoints": 3,
        "required_true_heads": 3,
        "required_shuffled_heads": 3,
        "revealed": False,
    }:
        raise ContractError("custody reveal gate differs")
    try:
        signature = base64.b64decode(receipt["signature_base64"], validate=True)
    except (ValueError, TypeError):
        raise ContractError("custody signature encoding is invalid") from None
    unsigned = dict(receipt)
    del unsigned["signature_base64"]
    key = _load_public_key(trust_anchor)
    try:
        key.verify(signature, canonical_json_bytes(unsigned))
    except Exception as error:
        raise ContractError("custody signature verification failed") from error
    return {
        "schema": "r12-scert-custody-verification-v1",
        "protocol": PROTOCOL_ID,
        "request_sha256": request_sha,
        "receipt_sha256": sha256_bytes(canonical_json_bytes(dict(receipt))),
        "trust_anchor_sha256": sha256_file(Path(trust_anchor)),
        "hidden_content_opened": False,
        "verified": True,
    }


# This reference is intentionally implemented in the board/custody module rather
# than importing train.scert.  The runtime compares against these independently
# written float64 equations before it can publish a CPU mechanics receipt.
_REF_VOCAB = 32
_REF_WIDTH = 16
_REF_HEADS = 2
_REF_HEAD_DIM = 8


def _reference_toy_weights() -> dict[str, torch.Tensor]:
    dtype = torch.float64
    embedding = torch.zeros((_REF_VOCAB, _REF_WIDTH), dtype=dtype)
    embedding[:, 1] = 1.0
    embedding[[2, 4, 6, 8, 10, 12, 14, 16, 18], 0] = 1.0
    embedding[[3, 5, 9, 11, 13, 15, 17, 19, 21], 0] = -1.0
    positions = torch.zeros((8, _REF_WIDTH), dtype=dtype)
    positions[:, 2] = -8.0
    positions[2, 2] = 8.0
    positions[3, 2] = 4.0
    positions[4:6, 3] = 8.0
    values: dict[str, torch.Tensor] = {
        "embedding": embedding,
        "positions": positions,
    }
    for layer in range(2):
        for name in ("q", "k", "v", "o"):
            values[f"l{layer}_{name}"] = torch.zeros(
                (_REF_WIDTH, _REF_WIDTH), dtype=dtype
            )
    values["l0_q"][1, 0] = 1.0
    values["l0_k"][2, 0] = 1.0
    values["l0_v"][0, 0] = 1.0
    values["l0_o"][0, 4] = 1.0
    values["l1_q"][1, 0] = 1.0
    values["l1_k"][3, 0] = 1.0
    values["l1_v"][4, 0] = 1.0
    values["l1_o"][0, 5] = 1.0
    unembedding = torch.zeros((_REF_WIDTH, _REF_VOCAB), dtype=dtype)
    unembedding[1, 7] = 0.5
    unembedding[5, 8] = 8.0
    unembedding[5, 9] = -8.0
    values["unembedding"] = unembedding
    return values


def _reference_mask(clean: bool) -> tuple[torch.Tensor, tuple[int, ...]]:
    result = torch.zeros((7, 7), dtype=torch.bool)
    for query in range(7):
        for key in range(query + 1):
            if query < 2:
                include = key < 2
            elif query < 4:
                include = key < 4
            elif query < 6:
                include = key < 2 or 4 <= key <= query or (not clean and 2 <= key < 4)
            else:
                include = key < 2 or 4 <= key <= 6
            result[query, key] = include
    return result, (0, 1, 4, 5, 6)


def _reference_attend(
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    scores = (queries @ keys.T) / math.sqrt(_REF_HEAD_DIM)
    scores = torch.where(mask, scores, torch.full_like(scores, -torch.inf))
    output = torch.zeros_like(scores)
    valid_rows = mask.any(dim=1)
    output[valid_rows] = torch.softmax(scores[valid_rows], dim=1)
    return output @ values


def independent_toy_reference(
    source_ids: Sequence[int], latest_ids: Sequence[int], clean: bool
) -> dict[str, Any]:
    """Independent two-layer float64 reference for one toy reconstruction."""
    if len(source_ids) != 2 or len(latest_ids) != 2:
        raise ContractError("independent toy spans must both have length two")
    weights = _reference_toy_weights()
    mask, keep = _reference_mask(bool(clean))
    all_ids = torch.tensor((0, 1, *source_ids, *latest_ids, 1), dtype=torch.long)
    hidden = weights["embedding"][all_ids] + weights["positions"][:7]
    cache: list[tuple[torch.Tensor, torch.Tensor]] = []
    for layer in range(2):
        queries = hidden @ weights[f"l{layer}_q"]
        keys = hidden @ weights[f"l{layer}_k"]
        values = hidden @ weights[f"l{layer}_v"]
        heads = []
        for head in range(_REF_HEADS):
            lo = head * _REF_HEAD_DIM
            hi = lo + _REF_HEAD_DIM
            heads.append(
                _reference_attend(
                    queries[:, lo:hi], keys[:, lo:hi], values[:, lo:hi], mask
                )
            )
        hidden = hidden + torch.cat(heads, dim=1) @ weights[f"l{layer}_o"]
        cache.append((keys.clone(), values.clone()))
    current = weights["embedding"][[1]] + weights["positions"][[7]]
    for layer in range(2):
        queries = current @ weights[f"l{layer}_q"]
        current_k = current @ weights[f"l{layer}_k"]
        current_v = current @ weights[f"l{layer}_v"]
        past_k, past_v = cache[layer]
        keys = torch.cat((past_k[list(keep)], current_k), dim=0)
        values = torch.cat((past_v[list(keep)], current_v), dim=0)
        full = torch.ones((1, keys.shape[0]), dtype=torch.bool)
        heads = []
        for head in range(_REF_HEADS):
            lo = head * _REF_HEAD_DIM
            hi = lo + _REF_HEAD_DIM
            heads.append(
                _reference_attend(
                    queries[:, lo:hi], keys[:, lo:hi], values[:, lo:hi], full
                )
            )
        current = current + torch.cat(heads, dim=1) @ weights[f"l{layer}_o"]
    logits = (current @ weights["unembedding"])[0]
    maximum = logits.max()
    token = int(logits.eq(maximum).to(torch.int64).argmax().item())
    return {
        "probe": hidden[6].clone(),
        "kept_cache": tuple(
            (keys[list(keep)].clone(), values[list(keep)].clone())
            for keys, values in cache
        ),
        "next_logits": logits,
        "next_token": token,
        "mask": mask,
    }


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise ContractError("JSON artifact must be an object")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build-public")
    build.add_argument("--registry", type=Path, required=True)
    build.add_argument("--custody-request", type=Path, required=True)
    verify = subparsers.add_parser("verify-custody")
    verify.add_argument("--request", type=Path, required=True)
    verify.add_argument("--receipt", type=Path, required=True)
    verify.add_argument("--trust-anchor", type=Path, required=True)
    verify.add_argument("--verification", type=Path, required=True)
    args = parser.parse_args()

    if args.command == "build-public":
        registry, request = build_local_registry()
        registry_receipt = atomic_publish_json(
            args.registry, registry, "r12-scert-local-board-registry-v1"
        )
        request = dict(request)
        request["published_registry_receipt"] = registry_receipt
        validate_custody_request(request)
        request_receipt = atomic_publish_json(
            args.custody_request, request, "r12-scert-external-custody-request-v1"
        )
        print(
            json.dumps(
                {
                    "registry": registry_receipt,
                    "custody_request": request_receipt,
                    "hidden_content_generated": False,
                    "h100_authorized": False,
                },
                sort_keys=True,
            )
        )
        return
    request = _load_json(args.request)
    receipt = _load_json(args.receipt)
    verification = verify_external_custody_receipt(receipt, request, args.trust_anchor)
    published = atomic_publish_json(
        args.verification, verification, "r12-scert-custody-verification-v1"
    )
    print(
        json.dumps(
            {"verification": published, "h100_authorized": False}, sort_keys=True
        )
    )


if __name__ == "__main__":
    main()
