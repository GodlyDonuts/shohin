"""Strict write-once formats for blind CTAA evaluation stages."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping

import torch

from ctaa_neural_core import CTAA_ACTION_COUNT, CTAA_MAX_STEPS, CTAA_WIDTH


PROGRAM_PREDICTION_SCHEMA = "r12_ctaa_v2_program_predictions_v2"
PACKET_INDEX_SCHEMA = "r12_ctaa_v2_packet_index_v2"
QUERY_PREDICTION_SCHEMA = "r12_ctaa_v2_query_predictions_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def write_torch_once(path: Path, payload: Mapping[str, object]) -> str:
    if path.exists():
        raise FileExistsError(f"refusing existing CTAA evidence artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"refusing existing CTAA evidence temporary: {temporary}")
    try:
        torch.save(dict(payload), temporary)
        temporary.chmod(0o444)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.chmod(0o600)
            temporary.unlink()
    path.chmod(0o444)
    return sha256_file(path)


def write_json_once(path: Path, payload: Mapping[str, object], *, mode: int = 0o444) -> str:
    if path.exists():
        raise FileExistsError(f"refusing existing CTAA JSON artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"refusing existing CTAA JSON temporary: {temporary}")
    try:
        temporary.write_text(json.dumps(dict(payload), sort_keys=True, indent=2) + "\n")
        temporary.chmod(mode)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.chmod(0o600)
            temporary.unlink()
    path.chmod(mode)
    return sha256_file(path)


def write_jsonl_once(
    path: Path,
    rows: Iterable[Mapping[str, object]],
    *,
    mode: int = 0o444,
) -> tuple[int, str]:
    if path.exists():
        raise FileExistsError(f"refusing existing CTAA JSONL artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"refusing existing CTAA JSONL temporary: {temporary}")
    count = 0
    try:
        with temporary.open("x") as handle:
            for row in rows:
                handle.write(canonical_json(dict(row)) + "\n")
                count += 1
        temporary.chmod(mode)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.chmod(0o600)
            temporary.unlink()
    path.chmod(mode)
    return count, sha256_file(path)


def _validate_family_ids(value: object, rows: int) -> list[str]:
    if not isinstance(value, list) or len(value) != rows:
        raise ValueError("CTAA evidence family-id count differs")
    family_ids = [str(item) for item in value]
    if any(not item for item in family_ids) or len(set(family_ids)) != rows:
        raise ValueError("CTAA evidence family IDs differ")
    return family_ids


def resolve_opcode_schedule(
    opcode_to_card: torch.Tensor,
    opcode_schedule: torch.Tensor,
) -> torch.Tensor:
    if (
        opcode_to_card.ndim != 2
        or opcode_to_card.shape[1] != CTAA_ACTION_COUNT
        or opcode_to_card.dtype != torch.uint8
    ):
        raise ValueError("CTAA predicted opcode binding geometry differs")
    if (
        opcode_schedule.ndim != 2
        or opcode_schedule.shape
        != (opcode_to_card.shape[0], CTAA_MAX_STEPS)
        or opcode_schedule.dtype != torch.uint8
    ):
        raise ValueError("CTAA predicted opcode schedule geometry differs")
    if opcode_schedule.numel() and int(opcode_schedule.max()) > CTAA_ACTION_COUNT:
        raise ValueError("CTAA predicted opcode schedule leaves event domain")
    local = opcode_schedule.long()
    resolved = opcode_to_card.long().gather(
        1, local.clamp_max(CTAA_ACTION_COUNT - 1)
    )
    return torch.where(local.eq(CTAA_ACTION_COUNT), local, resolved).to(torch.uint8)


def packet_valid_mask(
    opcode_to_card: torch.Tensor,
    opcode_schedule: torch.Tensor,
) -> torch.Tensor:
    resolved = resolve_opcode_schedule(opcode_to_card, opcode_schedule)
    expected = torch.arange(
        CTAA_ACTION_COUNT,
        dtype=torch.uint8,
        device=opcode_to_card.device,
    )[None].expand_as(opcode_to_card)
    binding_valid = opcode_to_card.sort(1).values.eq(expected).all(1)
    stop = opcode_schedule.eq(CTAA_ACTION_COUNT)
    count = stop.sum(1)
    index = stop.long().argmax(1)
    return (
        binding_valid
        & count.eq(1)
        & index.gt(0)
        & index.lt(CTAA_MAX_STEPS - 1)
        & resolved.le(CTAA_ACTION_COUNT).all(1)
    )


def validate_program_predictions(value: object) -> dict[str, object]:
    keys = {
        "schema",
        "family_ids",
        "program_source_sha256",
        "compiler_sha256",
        "action_cards",
        "opcode_to_card",
        "initial_state",
        "opcode_schedule",
        "schedule",
        "packet_valid",
    }
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError("CTAA program-prediction schema differs")
    cards = value["action_cards"]
    binding = value["opcode_to_card"]
    initial = value["initial_state"]
    opcode_schedule = value["opcode_schedule"]
    schedule = value["schedule"]
    valid = value["packet_valid"]
    if value["schema"] != PROGRAM_PREDICTION_SCHEMA:
        raise ValueError("CTAA program-prediction version differs")
    if not isinstance(cards, torch.Tensor) or cards.dtype != torch.uint8 or cards.ndim != 3:
        raise ValueError("CTAA predicted card tensor differs")
    rows = cards.shape[0]
    if cards.shape != (rows, CTAA_ACTION_COUNT, CTAA_WIDTH):
        raise ValueError("CTAA predicted card geometry differs")
    if (
        not isinstance(binding, torch.Tensor)
        or binding.shape != (rows, CTAA_ACTION_COUNT)
        or binding.dtype != torch.uint8
    ):
        raise ValueError("CTAA predicted opcode binding differs")
    if not isinstance(initial, torch.Tensor) or initial.shape != (rows, CTAA_WIDTH):
        raise ValueError("CTAA predicted initial-state geometry differs")
    if initial.dtype != torch.uint8:
        raise ValueError("CTAA predicted initial-state dtype differs")
    if not isinstance(schedule, torch.Tensor) or schedule.shape != (rows, CTAA_MAX_STEPS):
        raise ValueError("CTAA predicted resolved schedule geometry differs")
    if schedule.dtype != torch.uint8:
        raise ValueError("CTAA predicted resolved schedule dtype differs")
    if (
        not isinstance(opcode_schedule, torch.Tensor)
        or opcode_schedule.shape != (rows, CTAA_MAX_STEPS)
        or opcode_schedule.dtype != torch.uint8
    ):
        raise ValueError("CTAA predicted opcode schedule differs")
    if not isinstance(valid, torch.Tensor) or valid.shape != (rows,) or valid.dtype != torch.bool:
        raise ValueError("CTAA predicted validity mask differs")
    if rows < 1 or int(cards.max()) >= CTAA_WIDTH or int(initial.max()) >= CTAA_WIDTH:
        raise ValueError("CTAA predicted tuple leaves categorical domain")
    resolved = resolve_opcode_schedule(binding, opcode_schedule)
    if not torch.equal(schedule.cpu(), resolved.cpu()):
        raise ValueError("CTAA predicted resolved schedule is not derived from binding")
    recomputed = packet_valid_mask(binding, opcode_schedule)
    if not torch.equal(valid.cpu(), recomputed.cpu()):
        raise ValueError("CTAA predicted validity mask is not derived from bytes")
    family_ids = _validate_family_ids(value["family_ids"], rows)
    for key in ("program_source_sha256", "compiler_sha256"):
        if not isinstance(value[key], str) or len(value[key]) != 64:
            raise ValueError(f"CTAA {key} differs")
    return {
        **value,
        "family_ids": family_ids,
        "action_cards": cards.cpu(),
        "opcode_to_card": binding.cpu(),
        "initial_state": initial.cpu(),
        "opcode_schedule": opcode_schedule.cpu(),
        "schedule": schedule.cpu(),
        "packet_valid": valid.cpu(),
    }


def read_program_predictions(path: Path) -> dict[str, object]:
    return validate_program_predictions(
        torch.load(path, map_location="cpu", weights_only=True)
    )


def validate_packet_index(value: object) -> dict[str, object]:
    keys = {
        "schema",
        "program_predictions_sha256",
        "packet_sha256",
        "valid_family_ids",
        "valid_source_indices",
        "invalid_family_ids",
    }
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError("CTAA packet-index schema differs")
    if value["schema"] != PACKET_INDEX_SCHEMA:
        raise ValueError("CTAA packet-index version differs")
    valid_ids = value["valid_family_ids"]
    invalid_ids = value["invalid_family_ids"]
    indices = value["valid_source_indices"]
    if (
        not isinstance(valid_ids, list)
        or not isinstance(invalid_ids, list)
        or not isinstance(indices, list)
        or len(valid_ids) != len(indices)
        or any(not isinstance(item, str) or not item for item in valid_ids + invalid_ids)
        or len(set(valid_ids + invalid_ids)) != len(valid_ids) + len(invalid_ids)
        or any(not isinstance(index, int) or index < 0 for index in indices)
        or indices != sorted(indices)
    ):
        raise ValueError("CTAA packet-index rows differ")
    if not isinstance(value["program_predictions_sha256"], str):
        raise ValueError("CTAA packet-index prediction hash differs")
    packet_sha = value["packet_sha256"]
    if (valid_ids and not isinstance(packet_sha, str)) or (not valid_ids and packet_sha is not None):
        raise ValueError("CTAA packet-index packet hash differs")
    return dict(value)


def read_packet_index(path: Path) -> dict[str, object]:
    return validate_packet_index(json.loads(path.read_text()))


def validate_query_predictions(value: object) -> dict[str, object]:
    keys = {
        "schema",
        "family_ids",
        "query_source_sha256",
        "compiler_sha256",
        "execution_sha256",
        "positions",
    }
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError("CTAA query-prediction schema differs")
    if value["schema"] != QUERY_PREDICTION_SCHEMA:
        raise ValueError("CTAA query-prediction version differs")
    positions = value["positions"]
    if not isinstance(positions, torch.Tensor) or positions.ndim != 1:
        raise ValueError("CTAA predicted query geometry differs")
    if positions.dtype != torch.uint8 or positions.numel() < 1:
        raise ValueError("CTAA predicted query dtype differs")
    if int(positions.max()) >= CTAA_WIDTH:
        raise ValueError("CTAA predicted query leaves position domain")
    family_ids = _validate_family_ids(value["family_ids"], positions.shape[0])
    for key in ("query_source_sha256", "compiler_sha256", "execution_sha256"):
        if not isinstance(value[key], str) or len(value[key]) != 64:
            raise ValueError(f"CTAA {key} differs")
    return {**value, "family_ids": family_ids, "positions": positions.cpu()}


def read_query_predictions(path: Path) -> dict[str, object]:
    return validate_query_predictions(
        torch.load(path, map_location="cpu", weights_only=True)
    )
