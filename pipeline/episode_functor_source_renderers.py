#!/usr/bin/env python3
"""Byte-distinct source renderers for raw EFC world-event evidence.

This module has no dependency on the EFC compiler, generator, protocol, or
runtime. It converts canonical JSON raw-event evidence into a strict
line-oriented representation and independently decodes that representation
back to the normalized event object.
"""

from __future__ import annotations

import json
import re
from typing import Mapping


LINE_MAGIC = "EFC-RAW-LINES-V1"
JSON_SCHEMA = "efc-raw-world-evidence-v2"
RAW_FIELDS = frozenset(
    {
        "demonstrations",
        "observations",
        "renderer_choice",
        "schema",
    }
)
HEX_U64 = re.compile(r"[0-9a-f]{16}")


class SourceRendererError(ValueError):
    """A source renderer received or emitted a malformed event stream."""


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("ascii")


def _plain_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SourceRendererError(f"{field} must be a plain integer")
    return value


def _validate_normalized_row(
    row: object,
    *,
    expected_renderer: int,
) -> Mapping[str, object]:
    if not isinstance(row, dict) or set(row) != RAW_FIELDS:
        raise SourceRendererError("raw event object has incorrect fields")
    if row["schema"] != JSON_SCHEMA:
        raise SourceRendererError("raw event object has unknown schema")
    renderer = _plain_int(row["renderer_choice"], "renderer_choice")
    if renderer != expected_renderer:
        raise SourceRendererError(
            "renderer_choice does not match the source serialization"
        )
    demonstrations = row["demonstrations"]
    observations = row["observations"]
    if not isinstance(demonstrations, list) or not isinstance(observations, list):
        raise SourceRendererError("raw events must be lists")
    for event in demonstrations:
        if not isinstance(event, dict) or set(event) != {
            "action_key",
            "source_key",
            "target_key",
        }:
            raise SourceRendererError("malformed transition event")
        for field in ("action_key", "source_key", "target_key"):
            value = _plain_int(event[field], field)
            if value <= 0 or value >= 1 << 64:
                raise SourceRendererError("transition key is not nonzero uint64")
    for event in observations:
        if not isinstance(event, dict) or set(event) != {
            "answer",
            "observer_key",
            "state_key",
        }:
            raise SourceRendererError("malformed observation event")
        for field in ("observer_key", "state_key"):
            value = _plain_int(event[field], field)
            if value <= 0 or value >= 1 << 64:
                raise SourceRendererError("observation key is not nonzero uint64")
        answer = _plain_int(event["answer"], "answer")
        if answer < 0 or answer >= 1 << 64:
            raise SourceRendererError("answer does not fit uint64")
    return row


def decode_json_events(payload: bytes) -> dict[str, object]:
    try:
        row = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise SourceRendererError("JSON event evidence is malformed") from exc
    checked = _validate_normalized_row(row, expected_renderer=0)
    if _canonical_json_bytes(checked) != payload:
        raise SourceRendererError("JSON event evidence is not canonical")
    return dict(checked)


def _encode_line_row(row: object) -> bytes:
    checked = _validate_normalized_row(row, expected_renderer=1)
    lines = [f"{LINE_MAGIC}\t1"]
    demonstrations = checked["demonstrations"]
    observations = checked["observations"]
    assert isinstance(demonstrations, list)
    assert isinstance(observations, list)
    for event in demonstrations:
        assert isinstance(event, dict)
        lines.append(
            "D\t"
            f"{int(event['action_key']):016x}\t"
            f"{int(event['source_key']):016x}\t"
            f"{int(event['target_key']):016x}"
        )
    for event in observations:
        assert isinstance(event, dict)
        lines.append(
            "O\t"
            f"{int(event['observer_key']):016x}\t"
            f"{int(event['state_key']):016x}\t"
            f"{int(event['answer'])}"
        )
    return ("\n".join(lines) + "\n").encode("ascii")


def encode_line_events(json_payload: bytes) -> bytes:
    """Render canonical JSON events as strict line-oriented ASCII."""

    row = decode_json_events(json_payload)
    row["renderer_choice"] = 1
    return _encode_line_row(row)


def decode_line_events(payload: bytes) -> dict[str, object]:
    """Decode strict line-oriented evidence into the normalized event object."""

    try:
        text = payload.decode("ascii")
    except UnicodeDecodeError as exc:
        raise SourceRendererError("line event evidence is not ASCII") from exc
    if not text.endswith("\n") or "\r" in text:
        raise SourceRendererError("line event evidence has noncanonical newlines")
    lines = text[:-1].split("\n")
    if not lines or lines[0] != f"{LINE_MAGIC}\t1":
        raise SourceRendererError("line event header is invalid")
    demonstrations: list[dict[str, int]] = []
    observations: list[dict[str, int]] = []
    observation_phase = False
    for line in lines[1:]:
        fields = line.split("\t")
        if len(fields) != 4:
            raise SourceRendererError("line event has incorrect field count")
        if fields[0] == "D":
            if observation_phase:
                raise SourceRendererError(
                    "transition event appears after observation phase"
                )
            if not all(HEX_U64.fullmatch(field) for field in fields[1:]):
                raise SourceRendererError("transition key encoding is invalid")
            demonstrations.append(
                {
                    "action_key": int(fields[1], 16),
                    "source_key": int(fields[2], 16),
                    "target_key": int(fields[3], 16),
                }
            )
        elif fields[0] == "O":
            observation_phase = True
            if not HEX_U64.fullmatch(fields[1]) or not HEX_U64.fullmatch(fields[2]):
                raise SourceRendererError("observation key encoding is invalid")
            if (
                not fields[3].isdigit()
                or (len(fields[3]) > 1 and fields[3].startswith("0"))
                or len(fields[3]) > 20
            ):
                raise SourceRendererError("answer encoding is noncanonical")
            observations.append(
                {
                    "answer": int(fields[3]),
                    "observer_key": int(fields[1], 16),
                    "state_key": int(fields[2], 16),
                }
            )
        else:
            raise SourceRendererError("line event kind is unknown")
    row = {
        "demonstrations": demonstrations,
        "observations": observations,
        "renderer_choice": 1,
        "schema": JSON_SCHEMA,
    }
    _validate_normalized_row(row, expected_renderer=1)
    if _encode_line_row(row) != payload:
        raise SourceRendererError("line event evidence is not canonical")
    return row


__all__ = [
    "LINE_MAGIC",
    "SourceRendererError",
    "decode_json_events",
    "decode_line_events",
    "encode_line_events",
]
