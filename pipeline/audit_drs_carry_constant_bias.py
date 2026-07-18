#!/usr/bin/env python3
"""Audit grammar-gated carry-logit calibration and nuisance-only nulls.

The audit reads the immutable post-DRS residual-swap artifact and asks a narrow
question: how well can deltas added to logit(c=1)-logit(c=0) classify the
unpatched carry targets? Besides the all-board constant oracle, it selects
global, operation-only, and operation-by-width deltas from the fit regimes only
before scoring held-out regimes. It does not fit a hidden-state reader and does
not evaluate full-vocabulary or autonomous behavior.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import stat
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = (
    ROOT
    / "artifacts"
    / "eval_history"
    / "drs_workspace_probe_post_p4_20260718_mps.json"
)
FROZEN_SOURCE_SHA256 = (
    "c3c2d0b037852cb57d54e1f147d445d27093a8548b965c41466e81bcc1a27778"
)
EXPECTED_AUDIT = "digitwise_residual_patch_workspace_proxy_v1"
EXPECTED_REGIMES = (
    "fit_w4",
    "fit_w6",
    "value_ood_w4",
    "value_ood_w6",
    "width_ood_w8",
)
EXPECTED_LAYERS = (5, 9, 13, 17, 21, 25, 29)
AUDIT_LAYER = 29
SCHEMA = "drs_carry_constant_bias_audit_v4"
FIT_REGIMES = ("fit_w4", "fit_w6")
VALUE_OOD_REGIMES = ("value_ood_w4", "value_ood_w6")
WIDTH_BY_REGIME = {
    "fit_w4": 4,
    "fit_w6": 6,
    "value_ood_w4": 4,
    "value_ood_w6": 6,
    "width_ood_w8": 8,
}
STATE_PATTERN = re.compile(
    r"^dws:op=(add|sub);w=([1-9][0-9]*);p=([0-9]+);c=([01]);"
    r"a=([0-9]+);b=([0-9]+);r=([0-9]+);z=([01])$"
)


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode("ascii")


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _unique_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError(f"duplicate JSON key: {key}")
        output[key] = value
    return output


def decode_json(raw: bytes, label: str) -> Any:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"{label} is not UTF-8") from error
    try:
        return json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant: {value}")
            ),
        )
    except json.JSONDecodeError as error:
        raise ValueError(f"{label} is not valid JSON") from error


def read_immutable(path: Path) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("source is not a regular file")
        if before.st_mode & 0o222:
            raise PermissionError("source must be read-only")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        identity_before = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        identity_after = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        if identity_before != identity_after:
            raise RuntimeError("source changed during audit read")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _parse_state(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a string")
    match = STATE_PATTERN.fullmatch(value)
    if match is None:
        raise ValueError(f"{label} is not a canonical DWS state")
    operation, width, cursor, carry, operand_a, operand_b, result, terminal = (
        match.groups()
    )
    return {
        "a": operand_a,
        "b": operand_b,
        "carry": int(carry),
        "cursor": int(cursor),
        "operation": operation,
        "result": result,
        "terminal": int(terminal),
        "width": int(width),
    }


def extract_rows(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    if payload.get("audit") != EXPECTED_AUDIT:
        raise ValueError("unexpected probe audit schema")
    if payload.get("layers") != list(EXPECTED_LAYERS):
        raise ValueError("unexpected probe layers")
    if payload.get("pairs_per_regime") != 4:
        raise ValueError("unexpected pairs_per_regime")
    if payload.get("transition_index") != 2:
        raise ValueError("unexpected transition_index")
    records = payload.get("records")
    if not isinstance(records, list) or len(records) != 40:
        raise ValueError("probe must contain exactly 40 records")

    field_counts = Counter()
    carry_regimes = Counter()
    rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for record_index, record in enumerate(records):
        if not isinstance(record, dict):
            raise TypeError(f"record {record_index} must be an object")
        field = record.get("field")
        if field not in {"carry", "digit"}:
            raise ValueError(f"record {record_index} has unexpected field")
        field_counts[field] += 1
        if field != "carry":
            continue
        regime = record.get("regime")
        if regime not in EXPECTED_REGIMES:
            raise ValueError(f"record {record_index} has unexpected regime")
        carry_regimes[regime] += 1
        layers = record.get("layers")
        if not isinstance(layers, list):
            raise TypeError(f"record {record_index} layers must be a list")
        selected = [layer for layer in layers if layer.get("layer") == AUDIT_LAYER]
        if len(selected) != 1:
            raise ValueError(f"record {record_index} must have one layer-29 result")
        layer = selected[0]

        pair_targets: list[int] = []
        for side, direction in (("a", "a_to_b"), ("b", "b_to_a")):
            state = record.get(side)
            if not isinstance(state, dict):
                raise TypeError(f"record {record_index} side {side} must be an object")
            state_id = state.get("id")
            if not isinstance(state_id, str) or not state_id:
                raise ValueError(f"record {record_index} side {side} has invalid id")
            if state_id in seen_ids:
                raise ValueError(f"duplicate state id: {state_id}")
            seen_ids.add(state_id)
            target_raw = state.get("target")
            if target_raw not in {"0", "1"}:
                raise ValueError(f"carry target must be string 0 or 1: {state_id}")
            target = int(target_raw)
            pair_targets.append(target)
            current = _parse_state(state.get("state"), f"{state_id} state")
            following = _parse_state(state.get("next_state"), f"{state_id} next_state")
            if current["operation"] != following["operation"]:
                raise ValueError(f"operation changes across transition: {state_id}")
            if current["width"] != following["width"]:
                raise ValueError(f"width changes across transition: {state_id}")
            if current["cursor"] != 2 or following["cursor"] != 3:
                raise ValueError(f"unexpected cursor transition: {state_id}")
            if following["carry"] != target:
                raise ValueError(f"target does not match next carry: {state_id}")
            if current["width"] != WIDTH_BY_REGIME[regime]:
                raise ValueError(f"regime width mismatch: {state_id}")

            direction_payload = layer.get(direction)
            if not isinstance(direction_payload, dict):
                raise TypeError(f"missing {direction} payload")
            baseline = direction_payload.get("baseline")
            if not isinstance(baseline, dict):
                raise TypeError(f"missing {direction} baseline")
            own = _finite_number(baseline.get("own_logit"), f"{state_id} own_logit")
            other = _finite_number(
                baseline.get("other_logit"), f"{state_id} other_logit"
            )
            toward_other = _finite_number(
                baseline.get("toward_other_logodds"),
                f"{state_id} toward_other_logodds",
            )
            if not math.isclose(toward_other, other - own, abs_tol=1e-6, rel_tol=0.0):
                raise ValueError(f"baseline logodds do not replay: {state_id}")
            # margin is always logit(c=1)-logit(c=0), independent of target.
            margin = own - other if target == 1 else other - own
            rows.append(
                {
                    "id": state_id,
                    "margin_c1_minus_c0": margin,
                    "operation": current["operation"],
                    "regime": regime,
                    "target": target,
                    "width": current["width"],
                }
            )
        if sorted(pair_targets) != [0, 1]:
            raise ValueError(
                f"carry pair is not target-balanced: record {record_index}"
            )

    if field_counts != Counter({"carry": 20, "digit": 20}):
        raise ValueError(f"unexpected field counts: {dict(field_counts)}")
    if carry_regimes != Counter({regime: 4 for regime in EXPECTED_REGIMES}):
        raise ValueError(f"unexpected carry regime counts: {dict(carry_regimes)}")
    if len(rows) != 40 or Counter(row["target"] for row in rows) != Counter(
        {0: 20, 1: 20}
    ):
        raise ValueError("carry rows must contain 20 examples per target")
    return sorted(rows, key=lambda row: (row["regime"], row["id"]))


def _metrics_for_rule(
    rows: Sequence[Mapping[str, Any]], delta_for_row: Mapping[str, float]
) -> dict[str, Any]:
    adjusted: dict[str, float] = {}
    for row in rows:
        state_id = str(row["id"])
        if state_id not in delta_for_row:
            raise KeyError(f"missing delta for {state_id}")
        delta = _finite_number(delta_for_row[state_id], f"{state_id} delta")
        value = float(row["margin_c1_minus_c0"]) + delta
        if math.isclose(value, 0.0, abs_tol=1e-15):
            raise ValueError("delta lies on a decision boundary")
        adjusted[state_id] = value
    by_regime: dict[str, dict[str, Any]] = {}
    correct = 0
    target_correct = {0: 0, 1: 0}
    for regime in EXPECTED_REGIMES:
        cells: dict[str, Any] = {}
        for target in (0, 1):
            selected = [
                row
                for row in rows
                if row["regime"] == regime and row["target"] == target
            ]
            cell_correct = sum(
                (adjusted[str(row["id"])] > 0) == bool(target) for row in selected
            )
            cells[f"target_{target}"] = {
                "correct": cell_correct,
                "total": len(selected),
            }
            correct += cell_correct
            target_correct[target] += cell_correct
        by_regime[regime] = cells
    return {
        "by_regime_and_target": by_regime,
        "correct": correct,
        "minimum_cell_correct": min(
            cell["correct"]
            for regime in by_regime.values()
            for cell in regime.values()
            if cell["total"]
        ),
        "target_0_correct": target_correct[0],
        "target_1_correct": target_correct[1],
        "total": len(rows),
    }


def _metrics(rows: Sequence[Mapping[str, Any]], delta: float) -> dict[str, Any]:
    return _metrics_for_rule(rows, {str(row["id"]): delta for row in rows})


def _intervals(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    boundaries = sorted({-float(row["margin_c1_minus_c0"]) for row in rows})
    spans: list[tuple[float | None, float | None]] = [(None, boundaries[0])]
    spans.extend(zip(boundaries, boundaries[1:]))
    spans.append((boundaries[-1], None))
    output: list[dict[str, Any]] = []
    for lower, upper in spans:
        if lower is None:
            representative = float(upper) - 1.0
        elif upper is None:
            representative = float(lower) + 1.0
        else:
            representative = (lower + upper) / 2.0
        output.append(
            {
                "lower_open": lower,
                "metrics": _metrics(rows, representative),
                "representative_delta": representative,
                "upper_open": upper,
            }
        )
    return output


def _select_favorable_interval(
    rows: Sequence[Mapping[str, Any]], *, minimum_intervention: bool = False
) -> dict[str, Any]:
    intervals = _intervals(rows)
    best_correct = max(interval["metrics"]["correct"] for interval in intervals)
    best = [
        interval
        for interval in intervals
        if interval["metrics"]["correct"] == best_correct
    ]

    def minimum_intervention_representative(
        interval: Mapping[str, Any],
    ) -> float:
        lower = interval["lower_open"]
        upper = interval["upper_open"]
        if (lower is None or lower < 0.0) and (upper is None or upper > 0.0):
            return 0.0
        if upper is not None and upper <= 0.0:
            span = upper - lower if lower is not None else max(1.0, abs(float(upper)))
            return float(upper) - span * 1e-6
        if lower is not None and lower >= 0.0:
            span = upper - lower if upper is not None else max(1.0, abs(float(lower)))
            return float(lower) + span * 1e-6
        raise AssertionError("unreachable interval geometry")

    if minimum_intervention:
        candidates = [
            (minimum_intervention_representative(interval), interval)
            for interval in best
        ]
        representative, selected = min(
            candidates,
            key=lambda item: (abs(item[0]), item[0]),
        )
        selection_rule = (
            "maximum fit accuracy, then minimum absolute intervention; an open-boundary "
            "representative is placed 1e-6 of that interval width inside the interval"
        )
    else:
        selected = max(
            best,
            key=lambda interval: (
                interval["metrics"]["minimum_cell_correct"],
                min(
                    interval["metrics"]["target_0_correct"],
                    interval["metrics"]["target_1_correct"],
                ),
                -abs(interval["representative_delta"]),
            ),
        )
        representative = float(selected["representative_delta"])
        selection_rule = (
            "maximum all-board accuracy, then strongest minimum regime-target cell, "
            "target balance, and minimum representative magnitude"
        )
    output = dict(selected)
    output["metrics"] = _metrics(rows, representative)
    output["representative_delta"] = representative
    output["selection_rule"] = selection_rule
    return output


def _rule_metrics(
    rows: Sequence[Mapping[str, Any]],
    deltas: Mapping[str, float],
    key_fields: Sequence[str],
) -> dict[str, Any]:
    per_row: dict[str, float] = {}
    for row in rows:
        key = ":".join(str(row[field]) for field in key_fields)
        if key not in deltas:
            raise KeyError(f"missing nuisance delta for {key}")
        per_row[str(row["id"])] = deltas[key]
    return _metrics_for_rule(rows, per_row)


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        inverse = math.exp(-value)
        return 1.0 / (1.0 + inverse)
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def _cross_entropy_delta(rows: Sequence[Mapping[str, Any]]) -> float:
    if not rows or {int(row["target"]) for row in rows} != {0, 1}:
        raise ValueError("cross-entropy calibration requires both carry targets")
    lower = -100.0
    upper = 100.0
    for _ in range(256):
        midpoint = (lower + upper) / 2.0
        derivative = sum(
            _sigmoid(float(row["margin_c1_minus_c0"]) + midpoint) - int(row["target"])
            for row in rows
        )
        if derivative < 0.0:
            lower = midpoint
        else:
            upper = midpoint
    return (lower + upper) / 2.0


def _cross_entropy_loss(rows: Sequence[Mapping[str, Any]], delta: float) -> float:
    total = 0.0
    for row in rows:
        logit = float(row["margin_c1_minus_c0"]) + delta
        target = int(row["target"])
        total += max(logit, 0.0) - logit * target + math.log1p(math.exp(-abs(logit)))
    return total / len(rows)


def _fit_optimal_eval_score_range(
    fit_rows: Sequence[Mapping[str, Any]],
    eval_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    intervals = _intervals(fit_rows)
    best_fit = max(interval["metrics"]["correct"] for interval in intervals)
    optimal = [
        interval for interval in intervals if interval["metrics"]["correct"] == best_fit
    ]
    scores: list[int] = []
    for interval in optimal:
        lower = interval["lower_open"]
        upper = interval["upper_open"]
        boundaries = sorted(
            {
                -float(row["margin_c1_minus_c0"])
                for row in eval_rows
                if (lower is None or -float(row["margin_c1_minus_c0"]) > lower)
                and (upper is None or -float(row["margin_c1_minus_c0"]) < upper)
            }
        )
        cuts: list[float | None] = [lower, *boundaries, upper]
        for left, right in zip(cuts, cuts[1:]):
            if left is None:
                representative = float(right) - 1.0
            elif right is None:
                representative = float(left) + 1.0
            else:
                representative = (left + right) / 2.0
            scores.append(
                sum(
                    (float(row["margin_c1_minus_c0"]) + representative > 0.0)
                    == bool(row["target"])
                    for row in eval_rows
                )
            )
    return {
        "eval_total": len(eval_rows),
        "fit_correct": best_fit,
        "fit_total": len(fit_rows),
        "maximum_eval_correct": max(scores),
        "minimum_eval_correct": min(scores),
        "optimal_fit_interval_count": len(optimal),
    }


def _fit_only_nuisance_nulls(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    fit_rows = [row for row in rows if row["regime"] in FIT_REGIMES]
    value_ood_rows = [row for row in rows if row["regime"] in VALUE_OOD_REGIMES]
    seen_width_rows = [row for row in rows if row["width"] in {4, 6}]

    global_selected = _select_favorable_interval(fit_rows, minimum_intervention=True)
    global_delta = float(global_selected["representative_delta"])
    global_ce_delta = _cross_entropy_delta(fit_rows)
    global_range = _fit_optimal_eval_score_range(fit_rows, value_ood_rows)

    operation_deltas: dict[str, float] = {}
    operation_ce_deltas: dict[str, float] = {}
    operation_ranges: dict[str, Any] = {}
    operation_selection: dict[str, Any] = {}
    for operation in ("add", "sub"):
        selected_rows = [row for row in fit_rows if row["operation"] == operation]
        selected_value_ood = [
            row for row in value_ood_rows if row["operation"] == operation
        ]
        selected = _select_favorable_interval(selected_rows, minimum_intervention=True)
        operation_deltas[operation] = float(selected["representative_delta"])
        operation_ce_deltas[operation] = _cross_entropy_delta(selected_rows)
        operation_ranges[operation] = _fit_optimal_eval_score_range(
            selected_rows, selected_value_ood
        )
        operation_selection[operation] = selected

    operation_width_deltas: dict[str, float] = {}
    operation_width_ce_deltas: dict[str, float] = {}
    operation_width_ranges: dict[str, Any] = {}
    operation_width_selection: dict[str, Any] = {}
    for operation in ("add", "sub"):
        for width in (4, 6):
            key = f"{operation}:{width}"
            selected_rows = [
                row
                for row in fit_rows
                if row["operation"] == operation and row["width"] == width
            ]
            selected_value_ood = [
                row
                for row in value_ood_rows
                if row["operation"] == operation and row["width"] == width
            ]
            selected = _select_favorable_interval(
                selected_rows, minimum_intervention=True
            )
            operation_width_deltas[key] = float(selected["representative_delta"])
            operation_width_ce_deltas[key] = _cross_entropy_delta(selected_rows)
            operation_width_ranges[key] = _fit_optimal_eval_score_range(
                selected_rows, selected_value_ood
            )
            operation_width_selection[key] = selected

    operation_range = {
        "maximum_value_ood_correct": sum(
            value["maximum_eval_correct"] for value in operation_ranges.values()
        ),
        "minimum_value_ood_correct": sum(
            value["minimum_eval_correct"] for value in operation_ranges.values()
        ),
        "value_ood_total": len(value_ood_rows),
    }
    operation_width_range = {
        "maximum_value_ood_correct": sum(
            value["maximum_eval_correct"] for value in operation_width_ranges.values()
        ),
        "minimum_value_ood_correct": sum(
            value["minimum_eval_correct"] for value in operation_width_ranges.values()
        ),
        "value_ood_total": len(value_ood_rows),
    }

    def weighted_group_loss(
        deltas: Mapping[str, float], key_fields: Sequence[str]
    ) -> float:
        weighted = 0.0
        for key, delta in deltas.items():
            group = [
                row
                for row in fit_rows
                if ":".join(str(row[field]) for field in key_fields) == key
            ]
            weighted += _cross_entropy_loss(group, delta) * len(group)
        return weighted / len(fit_rows)

    return {
        "claim_boundary": (
            "Deltas are score-selected on fit_w4/fit_w6 only. Operation-by-width "
            "has no eligible width-8 rule; no extrapolator is inferred after OOD reveal. "
            "Fit-optimal interval ranges expose tie-break sensitivity. Binary margin "
            "cross-entropy solutions use only the two carry logits available in the probe; "
            "they are not the production full-vocabulary objective."
        ),
        "fit_regimes": list(FIT_REGIMES),
        "global": {
            "binary_margin_cross_entropy_fit": {
                "delta": global_ce_delta,
                "fit_loss": _cross_entropy_loss(fit_rows, global_ce_delta),
                "fit_metrics": _metrics(fit_rows, global_ce_delta),
                "full_board_metrics": _metrics(rows, global_ce_delta),
            },
            "delta": global_delta,
            "fit_metrics": _metrics(fit_rows, global_delta),
            "fit_optimal_value_ood_score_range": global_range,
            "full_board_metrics": _metrics(rows, global_delta),
            "selection": global_selected,
        },
        "operation_only": {
            "binary_margin_cross_entropy_fit": {
                "deltas": operation_ce_deltas,
                "fit_loss": weighted_group_loss(operation_ce_deltas, ("operation",)),
                "fit_metrics": _rule_metrics(
                    fit_rows, operation_ce_deltas, ("operation",)
                ),
                "full_board_metrics": _rule_metrics(
                    rows, operation_ce_deltas, ("operation",)
                ),
            },
            "deltas": operation_deltas,
            "fit_metrics": _rule_metrics(fit_rows, operation_deltas, ("operation",)),
            "fit_optimal_value_ood_score_range": operation_range,
            "full_board_metrics": _rule_metrics(rows, operation_deltas, ("operation",)),
            "selection_by_operation": operation_selection,
        },
        "operation_width": {
            "binary_margin_cross_entropy_fit": {
                "deltas": operation_width_ce_deltas,
                "fit_loss": weighted_group_loss(
                    operation_width_ce_deltas, ("operation", "width")
                ),
                "fit_metrics": _rule_metrics(
                    fit_rows,
                    operation_width_ce_deltas,
                    ("operation", "width"),
                ),
                "held_out_value_metrics": _rule_metrics(
                    value_ood_rows,
                    operation_width_ce_deltas,
                    ("operation", "width"),
                ),
            },
            "deltas": operation_width_deltas,
            "eligible_seen_width_metrics": _rule_metrics(
                seen_width_rows,
                operation_width_deltas,
                ("operation", "width"),
            ),
            "excluded_regimes": ["width_ood_w8"],
            "fit_metrics": _rule_metrics(
                fit_rows, operation_width_deltas, ("operation", "width")
            ),
            "fit_optimal_value_ood_score_range": operation_width_range,
            "held_out_value_metrics": _rule_metrics(
                value_ood_rows,
                operation_width_deltas,
                ("operation", "width"),
            ),
            "selection_by_operation_width": operation_width_selection,
        },
    }


def audit_payload(payload: Mapping[str, Any], source_sha256: str) -> dict[str, Any]:
    rows = extract_rows(payload)
    positive = [float(row["margin_c1_minus_c0"]) for row in rows if row["target"] == 1]
    negative = [float(row["margin_c1_minus_c0"]) for row in rows if row["target"] == 0]
    perfect_lower = -min(positive)
    perfect_upper = -max(negative)
    intervals = _intervals(rows)
    best_correct = max(interval["metrics"]["correct"] for interval in intervals)
    best = [
        interval
        for interval in intervals
        if interval["metrics"]["correct"] == best_correct
    ]
    selected = _select_favorable_interval(rows)
    return {
        "claim_boundary": (
            "Pairwise grammar-gated c0/c1 constant-calibration audit only; "
            "not full-vocabulary decoding, a hidden-state reader, autonomous execution, "
            "or reasoning."
        ),
        "frozen_source_sha256": FROZEN_SOURCE_SHA256,
        "fit_only_nuisance_nulls": _fit_only_nuisance_nulls(rows),
        "layer": AUDIT_LAYER,
        "margin_definition": "logit(c=1)-logit(c=0)",
        "optimal_intervals": best,
        "optimal_total_correct": best_correct,
        "perfect_constant_feasibility": {
            "feasible": perfect_lower < perfect_upper,
            "lower_open": perfect_lower,
            "upper_open": perfect_upper,
        },
        "raw_delta_zero": _metrics(rows, 0.0),
        "records": rows,
        "regimes": list(EXPECTED_REGIMES),
        "schema": SCHEMA,
        "selected_favorable_null": selected,
        "source_sha256": source_sha256,
        "treatment_gate": {
            "binary_margin_cross_entropy_fit_operation_width_value_ood_floor": 14,
            "deterministic_fit_only_operation_full_board_floor": 35,
            "deterministic_fit_only_operation_width_value_ood_floor": 15,
            "deterministic_fit_only_operation_width_value_ood_range": [11, 16],
            "deterministic_fit_only_operation_width_width_ood_is_ineligible": True,
            "must_include_fitted_no_residual_nuisance_arm": True,
            "must_exceed_constant_pairwise_correct": best_correct,
            "must_exceed_constant_on_every_regime_target_cell": True,
            "requires_separate_full_vocabulary_and_autonomous_gates": True,
        },
    }


def exclusive_write(path: Path, payload: bytes) -> str:
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(path, flags, 0o444)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short report write")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return sha256(payload)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument(
        "--source-sha256", default=FROZEN_SOURCE_SHA256, help="required source identity"
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    raw = read_immutable(args.source)
    observed = sha256(raw)
    if observed != args.source_sha256 or observed != FROZEN_SOURCE_SHA256:
        raise ValueError(
            f"source SHA-256 mismatch: observed={observed} required={FROZEN_SOURCE_SHA256}"
        )
    value = decode_json(raw, "source")
    if not isinstance(value, dict):
        raise TypeError("source root must be an object")
    report = audit_payload(value, observed)
    rendered = canonical_json_bytes(report)
    report_sha256 = exclusive_write(args.output, rendered)
    print(
        json.dumps(
            {
                "optimal_total_correct": report["optimal_total_correct"],
                "output": str(args.output),
                "report_sha256": report_sha256,
                "source_sha256": observed,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
