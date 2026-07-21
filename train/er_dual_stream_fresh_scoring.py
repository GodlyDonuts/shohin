"""Frozen score mechanics for the ordinal-route neutral-distractor board."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import re
from typing import Mapping, Sequence

import torch

from er_relation_tensor_adapter import MAX_RULES
from er_relation_tensor_training import (
    RelationTensorRow,
    _family_derangement,
    compile_semantic_predictions,
    evaluate_arm,
    fit_arm,
)
from pilot_er_dual_stream_train_canary import alpha_predictions


NEUTRAL_PATTERN = re.compile(rb"(?<!\S)z[0-9a-z]{5}(?!\S)")
SCORING_ARMS = ("treatment", "family_deranged", "equality_ablated")
SEMANTIC_KEYS = (
    "cardinality",
    "initial",
    "relations",
    "rule_active",
    "events",
    "halt",
    "query",
    "state",
    "answer",
    "valid",
)


def _replace_neutral(
    value: tuple[int, ...], mapping: Mapping[bytes, bytes]
) -> tuple[int, ...]:
    payload = bytes(value)
    replaced = NEUTRAL_PATTERN.sub(
        lambda match: mapping.get(match.group(0), match.group(0)), payload
    )
    if len(replaced) != len(payload):
        raise ValueError("fresh neutral replacement changes source width")
    return tuple(replaced)


def alpha_recode_row(row: RelationTensorRow, salt: str) -> RelationTensorRow:
    """Bijectively recode every program and query name in one neutral namespace."""
    tokens = sorted(
        set(NEUTRAL_PATTERN.findall(bytes(row.program_bytes)))
        | set(NEUTRAL_PATTERN.findall(bytes(row.query_bytes)))
    )
    if not tokens:
        raise ValueError("fresh alpha recode found no neutral symbols")
    mapping: dict[bytes, bytes] = {}
    used: set[bytes] = set()
    for token in tokens:
        retry = 0
        while True:
            digest = hashlib.sha256(
                b":".join(
                    (
                        salt.encode(),
                        row.family_id.encode(),
                        token,
                        str(retry).encode(),
                    )
                )
            ).hexdigest()
            candidate = ("z" + digest[:5]).encode()
            if candidate not in used:
                break
            retry += 1
        mapping[token] = candidate
        used.add(candidate)
    return replace(
        row,
        program_bytes=_replace_neutral(row.program_bytes, mapping),
        query_bytes=_replace_neutral(row.query_bytes, mapping),
    )


def _program_semantic_spans(row: RelationTensorRow) -> set[tuple[int, int]]:
    """Recover semantic-name occurrences from public grammar plus train targets."""
    payload = bytes(row.program_bytes)
    semantic = set(row.binding_ranges) | set(row.initial_ranges)
    for rule in range(row.rule_count):
        semantic.update(row.witness_before_ranges[rule])
        semantic.update(row.witness_after_ranges[rule])

    opcodes: set[bytes] = set()
    for rule in range(row.rule_count):
        start, end = row.line_ranges[1 + rule]
        candidates = [
            match.span()
            for match in NEUTRAL_PATTERN.finditer(payload, start, end)
            if start <= match.start() and match.end() <= end
        ]
        opcode = [span for span in candidates if span not in semantic]
        if len(opcode) != 1:
            raise ValueError("fresh active rule does not expose one opcode candidate")
        semantic.add(opcode[0])
        opcodes.add(payload[slice(*opcode[0])])
    if len(opcodes) != row.rule_count:
        raise ValueError("fresh active opcodes are not unique")

    for event in range(len(row.event_cards)):
        start, end = row.line_ranges[1 + MAX_RULES + event]
        candidates = [
            match.span()
            for match in NEUTRAL_PATTERN.finditer(payload, start, end)
            if start <= match.start() and match.end() <= end
        ]
        event_semantic = [
            span for span in candidates if payload[slice(*span)] in opcodes
        ]
        expected = 0 if row.event_halt[event] else 1
        if len(event_semantic) != expected:
            raise ValueError("fresh event semantic opcode count differs")
        semantic.update(event_semantic)
    return semantic


def distractor_rotate_row(row: RelationTensorRow) -> RelationTensorRow:
    """Rotate only irrelevant neutral names while preserving every byte offset."""
    payload = bytes(row.program_bytes)
    semantic = _program_semantic_spans(row)
    program_occurrences = [match.span() for match in NEUTRAL_PATTERN.finditer(payload)]
    distractors = [payload[slice(*span)] for span in program_occurrences if span not in semantic]
    distractors.extend(NEUTRAL_PATTERN.findall(bytes(row.query_bytes)))
    if len(distractors) < 2 or len(set(distractors)) != len(distractors):
        raise ValueError("fresh distractor rotation requires unique irrelevant names")
    rotated = distractors[1:] + distractors[:1]
    mapping = dict(zip(distractors, rotated, strict=True))
    output = replace(
        row,
        program_bytes=_replace_neutral(row.program_bytes, mapping),
        query_bytes=_replace_neutral(row.query_bytes, mapping),
    )
    if any(
        bytes(output.program_bytes)[slice(*span)] != payload[slice(*span)]
        for span in semantic
    ):
        raise ValueError("fresh distractor rotation changed a semantic occurrence")
    return output


def source_free_row(row: RelationTensorRow) -> RelationTensorRow:
    """Destroy all name identity while retaining grammar, ordinals, and lengths."""
    mapping = {
        token: b"z00000"
        for token in set(NEUTRAL_PATTERN.findall(bytes(row.program_bytes)))
        | set(NEUTRAL_PATTERN.findall(bytes(row.query_bytes)))
    }
    if len(mapping) < 2:
        raise ValueError("fresh source-free control has too few names")
    return replace(
        row,
        program_bytes=_replace_neutral(row.program_bytes, mapping),
        query_bytes=_replace_neutral(row.query_bytes, mapping),
    )


def _fresh_equality_ablated_bytes(
    row: RelationTensorRow, seed: int
) -> tuple[int, ...]:
    payload = bytearray(row.program_bytes)
    used = set(NEUTRAL_PATTERN.findall(bytes(payload)))
    replacements: dict[tuple[int, int], bytes] = {}
    for rule in range(row.rule_count):
        for position, (start, end) in enumerate(row.witness_after_ranges[rule]):
            if end - start != 6:
                raise ValueError("fresh equality target is not six bytes")
            retry = 0
            while True:
                digest = hashlib.sha256(
                    f"{seed}:{row.family_id}:{rule}:{position}:{retry}".encode()
                ).hexdigest()
                candidate = ("z" + digest[:5]).encode()
                if candidate not in used:
                    break
                retry += 1
            used.add(candidate)
            replacements[(start, end)] = candidate
    for (start, end), candidate in replacements.items():
        payload[start:end] = candidate
    if len(payload) != len(row.program_bytes):
        raise ValueError("fresh equality ablation changes source width")
    return tuple(payload)


def arm_rows(
    rows: Sequence[RelationTensorRow], arm: str, seed: int
) -> list[RelationTensorRow]:
    if arm == "treatment":
        return list(rows)
    if arm == "family_deranged":
        return _family_derangement(rows, seed)
    if arm == "equality_ablated":
        return [
            replace(row, program_bytes=_fresh_equality_ablated_bytes(row, seed))
            for row in rows
        ]
    raise ValueError(f"unknown fresh dual-stream arm: {arm}")


def fit_fresh_arm(
    model: torch.nn.Module,
    rows: Sequence[RelationTensorRow],
    *,
    seed: int,
    arm: str,
    frozen_digest: str,
    digest_fn: object,
) -> dict[str, object]:
    transformed = arm_rows(rows, arm, seed)
    receipt = fit_arm(
        model,
        transformed,
        seed=seed,
        arm="treatment",
        frozen_digest=frozen_digest,
        digest_fn=digest_fn,  # type: ignore[arg-type]
    )
    receipt["arm"] = arm
    receipt["fresh_control_transform"] = arm
    return receipt


def record_reindex_row(row: RelationTensorRow, *, rule_only: bool) -> RelationTensorRow:
    lines = bytes(row.program_bytes).decode().splitlines()
    if len(lines) != len(row.line_ranges):
        raise ValueError("fresh record reindex line count differs")
    if rule_only:
        indices = [
            index
            for index, line in enumerate(lines)
            if re.match(r"^[RM][1-4] ", line) is not None
        ]
        if len(indices) != MAX_RULES:
            raise ValueError("fresh rule reindex did not find four rule records")
        values = [lines[index] for index in indices]
        values = values[1:] + values[:1]
        for index, value in zip(indices, values, strict=True):
            lines[index] = value
    else:
        lines = list(reversed(lines))
    return replace(row, program_bytes=tuple("\n".join(lines).encode()))


def _row_exact(
    left: Mapping[str, torch.Tensor], right: Mapping[str, torch.Tensor], keys: Sequence[str]
) -> torch.Tensor:
    if set(left) != set(right):
        raise ValueError("fresh invariant prediction fields differ")
    return torch.stack(
        [
            left[key].eq(right[key]).reshape(left[key].shape[0], -1).all(-1)
            for key in keys
        ]
    ).all(0)


def invariance_metrics(
    canonical_hard: Mapping[str, torch.Tensor],
    alpha_hard: Mapping[str, torch.Tensor],
    distractor_hard: Mapping[str, torch.Tensor],
    canonical_semantic: Mapping[str, torch.Tensor],
    rule_reindex: Mapping[str, torch.Tensor],
    physical_reindex: Mapping[str, torch.Tensor],
) -> dict[str, dict[str, int | float]]:
    hard_keys = tuple(sorted(canonical_hard))
    result = {}
    for name, exact in (
        ("alpha", _row_exact(canonical_hard, alpha_hard, hard_keys)),
        (
            "distractor_rotation",
            _row_exact(canonical_hard, distractor_hard, hard_keys),
        ),
        (
            "rule_storage_reindex",
            _row_exact(canonical_semantic, rule_reindex, SEMANTIC_KEYS),
        ),
        (
            "physical_record_reindex",
            _row_exact(canonical_semantic, physical_reindex, SEMANTIC_KEYS),
        ),
    ):
        result[name] = {
            "exact": int(exact.sum()),
            "rows": int(exact.numel()),
            "rate": float(exact.float().mean()),
        }
    return result


@torch.no_grad()
def evaluate_fresh_treatment(
    model: torch.nn.Module,
    rows: Sequence[RelationTensorRow],
    *,
    batch_size: int,
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    result = evaluate_arm(
        model,
        rows,
        batch_size=batch_size,
        include_raw=True,
        include_invariances=False,
    )
    raw = result.pop("raw")
    if not isinstance(raw, dict):
        raise RuntimeError("fresh treatment raw evidence is absent")

    alpha_rows = [alpha_recode_row(row, "fresh-alpha") for row in rows]
    distractor_rows = [distractor_rotate_row(row) for row in rows]
    rule_rows = [record_reindex_row(row, rule_only=True) for row in rows]
    physical_rows = [record_reindex_row(row, rule_only=False) for row in rows]
    canonical_hard = alpha_predictions(model, rows, batch_size=batch_size)
    alpha_hard = alpha_predictions(model, alpha_rows, batch_size=batch_size)
    distractor_hard = alpha_predictions(
        model, distractor_rows, batch_size=batch_size
    )
    canonical_semantic = compile_semantic_predictions(
        model, rows, batch_size=batch_size
    )
    rule_reindex = compile_semantic_predictions(
        model, rule_rows, batch_size=batch_size
    )
    physical_reindex = compile_semantic_predictions(
        model, physical_rows, batch_size=batch_size
    )
    invariance = invariance_metrics(
        canonical_hard,
        alpha_hard,
        distractor_hard,
        canonical_semantic,
        rule_reindex,
        physical_reindex,
    )
    result["invariance"] = invariance
    invariant_raw: dict[str, object] = {
        "canonical_hard": canonical_hard,
        "alpha_hard": alpha_hard,
        "distractor_hard": distractor_hard,
        "canonical_semantic": canonical_semantic,
        "rule_reindex": rule_reindex,
        "physical_reindex": physical_reindex,
    }
    return result, raw, invariant_raw


@torch.no_grad()
def evaluate_source_free(
    model: torch.nn.Module,
    rows: Sequence[RelationTensorRow],
    *,
    batch_size: int,
) -> tuple[dict[str, object], dict[str, object]]:
    result = evaluate_arm(
        model,
        [source_free_row(row) for row in rows],
        batch_size=batch_size,
        include_raw=True,
        include_invariances=False,
    )
    raw = result.pop("raw")
    if not isinstance(raw, dict):
        raise RuntimeError("fresh source-free raw evidence is absent")
    return result, raw
