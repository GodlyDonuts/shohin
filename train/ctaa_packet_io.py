"""Binary custody format for source-deleted CTAA packets and late queries."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import struct

import torch

from ctaa_trunk_compiler import HardCTAAPacket, HardCTAAQuery
from ctaa_neural_core import CTAA_ACTION_COUNT, CTAA_MAX_STEPS, CTAA_WIDTH


PACKET_MAGIC = b"CTAAPKT1"
QUERY_MAGIC = b"CTAAQRY1"
HEADER_LENGTH = struct.Struct(">I")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_once(path: Path, payload: bytes) -> str:
    if path.exists():
        raise FileExistsError(f"refusing existing CTAA custody artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"refusing existing CTAA temporary artifact: {temporary}")
    try:
        temporary.write_bytes(payload)
        temporary.chmod(0o444)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.chmod(0o644)
            temporary.unlink()
    path.chmod(0o444)
    return sha256_file(path)


def _framed_payload(magic: bytes, header: dict[str, object], body: bytes) -> bytes:
    encoded = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
    return magic + HEADER_LENGTH.pack(len(encoded)) + encoded + body


def _read_frame(path: Path, magic: bytes) -> tuple[dict[str, object], bytes]:
    payload = path.read_bytes()
    prefix = len(magic) + HEADER_LENGTH.size
    if len(payload) < prefix or payload[: len(magic)] != magic:
        raise ValueError("CTAA custody artifact magic differs")
    header_length = HEADER_LENGTH.unpack(payload[len(magic) : prefix])[0]
    end = prefix + header_length
    if end > len(payload):
        raise ValueError("CTAA custody header length differs")
    header = json.loads(payload[prefix:end])
    if not isinstance(header, dict):
        raise ValueError("CTAA custody header differs")
    return header, payload[end:]


def packet_body(packet: HardCTAAPacket) -> bytes:
    batch, action_count, width = packet.action_cards.shape
    if packet.initial_state.shape != (batch, width):
        raise ValueError("CTAA packet initial geometry differs")
    if packet.schedule.ndim != 2 or packet.schedule.shape[0] != batch:
        raise ValueError("CTAA packet schedule geometry differs")
    rows = []
    for index in range(batch):
        rows.append(packet.action_cards[index].contiguous().cpu().numpy().tobytes())
        rows.append(packet.initial_state[index].contiguous().cpu().numpy().tobytes())
        rows.append(packet.schedule[index].contiguous().cpu().numpy().tobytes())
    body = b"".join(rows)
    if len(body) != batch * packet.bytes_per_row:
        raise AssertionError("CTAA packet byte count differs")
    return body


def write_packet_file(path: Path, packet: HardCTAAPacket) -> dict[str, object]:
    batch, action_count, width = packet.action_cards.shape
    header = {
        "schema": "ctaa_hard_packet_v1",
        "rows": batch,
        "action_count": action_count,
        "width": width,
        "max_steps": packet.schedule.shape[1],
        "bytes_per_row": packet.bytes_per_row,
    }
    digest = _write_once(path, _framed_payload(PACKET_MAGIC, header, packet_body(packet)))
    return {**header, "sha256": digest}


def read_packet_file(path: Path) -> HardCTAAPacket:
    header, body = _read_frame(path, PACKET_MAGIC)
    required = {
        "schema": "ctaa_hard_packet_v1",
        "rows": int(header.get("rows", -1)),
        "action_count": int(header.get("action_count", -1)),
        "width": int(header.get("width", -1)),
        "max_steps": int(header.get("max_steps", -1)),
        "bytes_per_row": int(header.get("bytes_per_row", -1)),
    }
    rows = required["rows"]
    action_count = required["action_count"]
    width = required["width"]
    max_steps = required["max_steps"]
    bytes_per_row = required["bytes_per_row"]
    if (
        header.get("schema") != required["schema"]
        or rows < 1
        or action_count != CTAA_ACTION_COUNT
        or width != CTAA_WIDTH
        or max_steps != CTAA_MAX_STEPS
        or bytes_per_row != action_count * width + width + max_steps
        or len(body) != rows * bytes_per_row
    ):
        raise ValueError("CTAA packet custody geometry differs")
    tensor = torch.frombuffer(bytearray(body), dtype=torch.uint8).reshape(
        rows,
        bytes_per_row,
    )
    card_end = action_count * width
    return HardCTAAPacket(
        action_cards=tensor[:, :card_end].reshape(rows, action_count, width).clone(),
        initial_state=tensor[:, card_end : card_end + width].clone(),
        schedule=tensor[:, card_end + width :].clone(),
    )


def write_query_file(path: Path, query: HardCTAAQuery) -> dict[str, object]:
    header = {
        "schema": "ctaa_late_query_v1",
        "rows": int(query.position.shape[0]),
        "bytes_per_row": 1,
    }
    body = query.position.contiguous().cpu().numpy().tobytes()
    digest = _write_once(path, _framed_payload(QUERY_MAGIC, header, body))
    return {**header, "sha256": digest}


def read_query_file(path: Path) -> HardCTAAQuery:
    header, body = _read_frame(path, QUERY_MAGIC)
    rows = int(header.get("rows", -1))
    if (
        header.get("schema") != "ctaa_late_query_v1"
        or header.get("bytes_per_row") != 1
        or rows < 1
        or len(body) != rows
    ):
        raise ValueError("CTAA late-query custody geometry differs")
    position = torch.frombuffer(bytearray(body), dtype=torch.uint8).clone()
    return HardCTAAQuery(position=position)
