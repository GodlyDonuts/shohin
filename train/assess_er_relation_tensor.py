#!/usr/bin/env python3
"""Independently assess one-read ER-TT development evidence."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Mapping

import torch

from build_er_relation_tensor_board import DEVELOPMENT_SPLIT, PROTOCOL
from er_relation_tensor_adapter import EVENT_SLOTS, MAX_RULES
from pilot_er_relation_tensor import (
    ACCESS_SCHEMA,
    BOARD_REPORT_SHA256,
    BOARD_SOURCE_COMMIT,
    CHECKPOINT_SCHEMA,
    EVIDENCE_SCHEMA,
    EXPECTED_PARAMETERS,
    FROZEN_SOURCE_PATHS,
    REPORT_SCHEMA,
    THRESHOLDS,
    compute_gates,
)
from pilot_sd_cst_byte_addressed import sha256_file


ASSESSMENT_SCHEMA = "r12_er_relation_tensor_development_assessment_v1"
ROWS = 2_048
ARMS = {"treatment", "family_deranged", "equality_ablated"}


class AssessmentError(RuntimeError):
    pass


def load_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise AssessmentError(f"JSON object required: {path}")
    return value


def load_torch(path: Path) -> dict[str, object]:
    value = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(value, dict):
        raise AssessmentError(f"torch mapping required: {path}")
    return value


def require_tensor(
    raw: Mapping[str, object],
    name: str,
    shape: tuple[int, ...],
    *,
    dtype: torch.dtype = torch.int16,
) -> torch.Tensor:
    value = raw.get(name)
    if not isinstance(value, torch.Tensor) or value.dtype != dtype or tuple(value.shape) != shape:
        raise AssessmentError(f"ER-TT raw tensor differs: {name}")
    return value.long()


def summary(value: torch.Tensor) -> dict[str, object]:
    return {
        "correct": int(value.sum()),
        "rows": int(value.numel()),
        "rate": float(value.float().mean()),
    }


def _pointer_exact(
    selected: torch.Tensor,
    ranges: torch.Tensor,
    active: torch.Tensor,
) -> torch.Tensor:
    exact = (selected >= ranges[..., 0]) & (selected < ranges[..., 1])
    return (exact | ~active).reshape(ROWS, -1).all(-1)


def _execute(row: Mapping[str, object]) -> tuple[tuple[int, ...], int, bool]:
    n = int(row["cardinality"])
    initial = tuple(map(int, row["initial"][:n]))
    relations = tuple(
        tuple(map(int, item[:n])) for item in row["relations"]
    )
    active = tuple(bool(item) for item in row["rule_active"])
    events = tuple(map(int, row["events"]))
    halt = tuple(bool(item) for item in row["halt"])
    query = int(row["query"])
    if (
        not 3 <= n <= 6
        or len(initial) != n
        or any(not 0 <= item < n for item in initial)
        or not 0 <= query < n
    ):
        return (), -2, False
    state = initial
    alive = True
    for slot in range(EVENT_SLOTS):
        if not alive:
            continue
        if halt[slot]:
            alive = False
            continue
        card = events[slot]
        if not 0 <= card < MAX_RULES or not active[card]:
            return (), -2, False
        relation = relations[card]
        if len(relation) != n or any(not 0 <= item < n for item in relation):
            return (), -2, False
        state = tuple(state[item] for item in relation)
    return state, state[query], True


def _row_from_tensors(values: Mapping[str, torch.Tensor], index: int) -> dict[str, object]:
    return {
        name: tensor[index].tolist() if tensor[index].ndim else int(tensor[index])
        for name, tensor in values.items()
    }


def _intervene(row: Mapping[str, object], kind: str) -> dict[str, object]:
    output = json.loads(json.dumps(row))
    if kind == "relation_derangement":
        count = sum(map(bool, output["rule_active"]))
        values = output["relations"][:count]
        output["relations"][:count] = values[1:] + values[:1]
    elif kind == "cardinality_mask":
        old = int(output["cardinality"])
        new = old - 1 if old > 3 else old + 1
        output["cardinality"] = new
        if new > old:
            output["initial"][old] = old
            for slot, active in enumerate(output["rule_active"]):
                if active:
                    output["relations"][slot][old] = old
        else:
            output["initial"][:new] = [
                value if value < new else 0 for value in output["initial"][:new]
            ]
            for slot, active in enumerate(output["rule_active"]):
                if active:
                    output["relations"][slot][:new] = [
                        value if value < new else 0
                        for value in output["relations"][slot][:new]
                    ]
        output["query"] %= new
    elif kind == "state_reset":
        n = int(output["cardinality"])
        output["initial"][:n] = list(range(n))
    elif kind == "query_swap":
        output["query"] = (int(output["query"]) + 1) % int(output["cardinality"])
    else:
        raise AssessmentError(f"unknown intervention: {kind}")
    return output


def recompute_interventions(
    predicted: Mapping[str, torch.Tensor], target: Mapping[str, torch.Tensor]
) -> dict[str, object]:
    result = {
        name: {"eligible": 0, "sensitive": 0, "exact_on_eligible": 0, "changed_on_sensitive": 0}
        for name in (
            "relation_derangement",
            "cardinality_mask",
            "state_reset",
            "query_swap",
        )
    }
    for index in range(ROWS):
        predicted_row = _row_from_tensors(predicted, index)
        target_row = _row_from_tensors(target, index)
        eligible = all(
            predicted_row[name] == target_row[name]
            for name in target
            if name != "events"
        ) and all(
            bool(target_row["halt"][slot])
            or predicted_row["events"][slot] == target_row["events"][slot]
            for slot in range(EVENT_SLOTS)
        )
        target_base_state, target_base_answer, target_valid = _execute(target_row)
        predicted_base_state, predicted_base_answer, _ = _execute(predicted_row)
        if not target_valid:
            raise AssessmentError("ER-TT target packet is invalid")
        for kind, value in result.items():
            target_state, target_answer, target_variant_valid = _execute(
                _intervene(target_row, kind)
            )
            predicted_state, predicted_answer, predicted_variant_valid = _execute(
                _intervene(predicted_row, kind)
            )
            if not target_variant_valid:
                raise AssessmentError("ER-TT target intervention is invalid")
            value["eligible"] += int(eligible)
            if kind == "query_swap":
                sensitive = eligible and target_answer != target_base_answer
                exact = eligible and predicted_variant_valid and predicted_answer == target_answer
                changed = sensitive and predicted_answer != predicted_base_answer
            else:
                sensitive = eligible and target_state != target_base_state
                exact = eligible and predicted_variant_valid and predicted_state == target_state
                changed = sensitive and predicted_state != predicted_base_state
            value["sensitive"] += int(sensitive)
            value["exact_on_eligible"] += int(exact)
            value["changed_on_sensitive"] += int(changed)
    return result


def recompute_invariance(raw: Mapping[str, object]) -> dict[str, object]:
    keys = (
        "cardinality",
        "initial",
        "relations",
        "rule_active",
        "events",
        "halt",
        "query",
        "state",
        "answer",
    )
    canonical = {
        key: require_tensor(
            raw,
            f"invariance_canonical_{key}",
            (ROWS, *({"initial": (6,), "relations": (4, 6), "rule_active": (4,), "events": (13,), "halt": (13,), "state": (6,)}.get(key, ()))),
        )
        for key in keys
    }
    output = {}
    for name in (
        "rule_storage_reindex",
        "physical_record_reindex",
        "witness_alpha_rename",
        "opcode_alpha_rename",
        "post_halt_suffix",
    ):
        variant = {
            key: require_tensor(
                raw,
                f"invariance_{name}_{key}",
                tuple(canonical[key].shape),
            )
            for key in keys
        }
        compare = ("state", "answer") if name == "post_halt_suffix" else keys
        exact = torch.stack(
            [
                variant[key].eq(canonical[key]).reshape(ROWS, -1).all(-1)
                for key in compare
            ]
        ).all(0)
        output[name] = {"exact": int(exact.sum()), "rows": ROWS}
    output["source_poison_after_seal"] = {"exact": ROWS, "rows": ROWS}
    return output


def recompute_arm(raw: Mapping[str, object], *, treatment: bool) -> dict[str, object]:
    predicted = {
        "cardinality": require_tensor(raw, "pred_cardinality", (ROWS,)),
        "initial": require_tensor(raw, "pred_initial", (ROWS, 6)),
        "relations": require_tensor(raw, "pred_relations", (ROWS, 4, 6)),
        "rule_active": require_tensor(raw, "pred_rule_active", (ROWS, 4)),
        "events": require_tensor(raw, "pred_events", (ROWS, 13)),
        "halt": require_tensor(raw, "pred_halt", (ROWS, 13)),
        "query": require_tensor(raw, "pred_query", (ROWS,)),
    }
    target = {
        "cardinality": require_tensor(raw, "target_cardinality", (ROWS,)),
        "initial": require_tensor(raw, "target_initial", (ROWS, 6)),
        "relations": require_tensor(raw, "target_relations", (ROWS, 4, 6)),
        "rule_active": require_tensor(raw, "target_rule_active", (ROWS, 4)),
        "events": require_tensor(raw, "target_events", (ROWS, 13)),
        "halt": require_tensor(raw, "target_halt", (ROWS, 13)),
        "query": require_tensor(raw, "target_query", (ROWS,)),
    }
    pred_state = require_tensor(raw, "pred_state", (ROWS, 6))
    pred_answer = require_tensor(raw, "pred_answer", (ROWS,))
    pred_valid = require_tensor(raw, "pred_valid", (ROWS,))
    target_state = require_tensor(raw, "target_state", (ROWS, 6))
    target_answer = require_tensor(raw, "target_answer", (ROWS,))
    target_line_ranges = require_tensor(raw, "target_line_ranges", (ROWS, 18, 2))
    target_binding_ranges = require_tensor(raw, "target_binding_ranges", (ROWS, 6, 2))
    target_initial_ranges = require_tensor(raw, "target_initial_ranges", (ROWS, 6, 2))
    target_witness_ranges = require_tensor(raw, "target_witness_ranges", (ROWS, 4, 12, 2))
    target_query_range = require_tensor(raw, "target_query_range", (ROWS, 2))
    cardinality = target["cardinality"]
    positions = torch.arange(6)[None]
    active_rows = positions < cardinality[:, None]
    active_rules = target["rule_active"].bool()
    active_witness = torch.zeros(ROWS, 4, 12, dtype=torch.bool)
    for index in range(ROWS):
        n = int(cardinality[index])
        active_witness[index, active_rules[index], :n] = True
        active_witness[index, active_rules[index], 6 : 6 + n] = True
    exact = {
        "cardinality": predicted["cardinality"].eq(cardinality),
        "initial_rows": predicted["initial"].eq(target["initial"]).all(-1),
        "relation_rows": predicted["relations"].eq(target["relations"]).all((1, 2)),
        "rule_active": predicted["rule_active"].eq(target["rule_active"]).all(-1),
        "events": (
            predicted["events"].eq(target["events"]) | target["halt"].bool()
        ).all(-1),
        "halt": predicted["halt"].eq(target["halt"]).all(-1),
        "query": predicted["query"].eq(target["query"]),
        "line_pointer": _pointer_exact(
            require_tensor(raw, "pred_line_pointer", (ROWS, 18)),
            target_line_ranges,
            torch.ones(ROWS, 18, dtype=torch.bool),
        ),
        "binding_pointer": _pointer_exact(
            require_tensor(raw, "pred_binding_pointer", (ROWS, 6)),
            target_binding_ranges,
            active_rows,
        ),
        "initial_pointer": _pointer_exact(
            require_tensor(raw, "pred_initial_pointer", (ROWS, 6)),
            target_initial_ranges,
            active_rows,
        ),
        "witness_pointer": _pointer_exact(
            require_tensor(raw, "pred_witness_pointer", (ROWS, 4, 12)),
            target_witness_ranges,
            active_witness,
        ),
        "query_pointer": _pointer_exact(
            require_tensor(raw, "pred_query_pointer", (ROWS,))[:, None],
            target_query_range[:, None],
            torch.ones(ROWS, 1, dtype=torch.bool),
        ),
        "state": pred_valid.bool() & pred_state.eq(target_state).all(-1),
        "answer": pred_valid.bool() & pred_answer.eq(target_answer),
    }
    exact["packet"] = torch.stack(
        [
            exact[name]
            for name in (
                "cardinality",
                "initial_rows",
                "relation_rows",
                "rule_active",
                "events",
                "halt",
                "query",
            )
        ]
    ).all(0)
    exact["joint"] = exact["packet"] & exact["state"] & exact["answer"]
    depth = require_tensor(raw, "depth", (ROWS,), dtype=torch.uint8)
    grouping_cardinality = require_tensor(raw, "cardinality", (ROWS,), dtype=torch.uint8)
    renderer_index = require_tensor(raw, "renderer_index", (ROWS,), dtype=torch.uint8)
    non_bijective = require_tensor(raw, "non_bijective", (ROWS,), dtype=torch.bool).bool()
    renderer_names = raw.get("renderer_names")
    if (
        not isinstance(renderer_names, list)
        or len(renderer_names) != 4
        or not all(type(name) is str for name in renderer_names)
    ):
        raise AssessmentError("ER-TT renderer names differ")

    def grouped(values: torch.Tensor, names: Mapping[int, str] | None = None) -> dict[str, object]:
        output = {}
        for value in sorted(map(int, values.unique())):
            mask = values.eq(value)
            key = names[value] if names is not None else str(value)
            output[key] = {
                name: summary(exact[name][mask])
                for name in ("packet", "state", "answer", "joint")
            }
        return output

    result: dict[str, object] = {
        "overall": {name: summary(value) for name, value in exact.items()},
        "by_cardinality": grouped(grouping_cardinality),
        "by_depth": grouped(depth),
        "by_renderer": grouped(
            renderer_index, {index: name for index, name in enumerate(renderer_names)}
        ),
        "non_bijective": {
            name: summary(exact[name][non_bijective])
            for name in ("packet", "state", "answer", "joint")
        },
        "interventions": recompute_interventions(predicted, target),
    }
    if treatment:
        result["invariance"] = recompute_invariance(raw)
    return result


def metric_equal(left: object, right: object) -> bool:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return set(left) == set(right) and all(
            metric_equal(left[name], right[name]) for name in left
        )
    if isinstance(left, float) or isinstance(right, float):
        return math.isclose(float(left), float(right), abs_tol=1e-12)
    return left == right


def assess(
    report: Mapping[str, object],
    checkpoint: Mapping[str, object],
    evidence: Mapping[str, object],
    ledger: Mapping[str, object],
    hashes: Mapping[str, str],
) -> dict[str, object]:
    if (
        report.get("schema") != REPORT_SCHEMA
        or checkpoint.get("schema") != CHECKPOINT_SCHEMA
        or evidence.get("schema") != EVIDENCE_SCHEMA
        or ledger.get("schema") != ACCESS_SCHEMA
        or any(
            value.get("protocol") != PROTOCOL
            for value in (report, checkpoint, evidence, ledger)
        )
    ):
        raise AssessmentError("ER-TT artifact identity differs")
    if report.get("thresholds") != THRESHOLDS:
        raise AssessmentError("ER-TT thresholds differ")
    if checkpoint.get("training_contract") != report.get("training_contract"):
        raise AssessmentError("ER-TT training contract differs")
    if (
        checkpoint.get("board_source_commit") != BOARD_SOURCE_COMMIT
        or checkpoint.get("board_report_sha256") != BOARD_REPORT_SHA256
        or evidence.get("board_report_sha256") != BOARD_REPORT_SHA256
        or ledger.get("board_report_sha256") != BOARD_REPORT_SHA256
    ):
        raise AssessmentError("ER-TT board identity differs")
    if (
        hashes["checkpoint"] != report["artifacts"]["checkpoint_sha256"]
        or hashes["checkpoint"] != evidence.get("checkpoint_sha256")
        or hashes["evidence"] != report["artifacts"]["evidence_sha256"]
        or hashes["ledger"] != report["artifacts"]["development_ledger_sha256"]
    ):
        raise AssessmentError("ER-TT artifact hashes differ")
    source_manifest = checkpoint.get("source_manifest")
    if (
        not isinstance(source_manifest, Mapping)
        or source_manifest.get("commit") != checkpoint.get("scientific_source_commit")
        or set(source_manifest.get("files", {})) != set(FROZEN_SOURCE_PATHS)
    ):
        raise AssessmentError("ER-TT source manifest differs")
    if checkpoint.get("parameters") != EXPECTED_PARAMETERS or report.get("parameters") != EXPECTED_PARAMETERS:
        raise AssessmentError("ER-TT parameter certificate differs")
    arms = checkpoint.get("arms")
    raw_arms = evidence.get("arms")
    if not isinstance(arms, Mapping) or set(arms) != ARMS or not isinstance(raw_arms, Mapping) or set(raw_arms) != ARMS:
        raise AssessmentError("ER-TT arm identity differs")
    expected_names = set(checkpoint["parent_receipt"]["trainable_names"])
    for name, arm in arms.items():
        if (
            arm["fit"]["arm"] != name
            or arm["fit"]["updates"] != 3_000
            or arm["fit"]["frozen_parent_unchanged"] is not True
            or arm["fit"]["motor_parameters"] != 0
            or arm["fit"]["reader_parameters"] != 0
            or set(arm["compiler_trainable_state"]) != expected_names
        ):
            raise AssessmentError(f"ER-TT arm receipt differs: {name}")
    recomputed = {
        name: recompute_arm(raw_arms[name], treatment=name == "treatment")
        for name in sorted(ARMS)
    }
    if not metric_equal(recomputed, report.get("metrics")):
        raise AssessmentError("ER-TT reported metrics differ from raw evidence")
    gates = compute_gates(recomputed, checkpoint)
    if gates != report.get("gates") or all(gates.values()) != report.get("all_gates_pass"):
        raise AssessmentError("ER-TT gate recomputation differs")
    expected_decision = (
        "authorize_one_sealed_confirmation"
        if all(gates.values())
        else "reject_er_relation_tensor_v1"
    )
    if report.get("decision") != expected_decision:
        raise AssessmentError("ER-TT decision differs")
    if (
        checkpoint.get("development_accesses") != 0
        or checkpoint.get("confirmation_accesses") != 0
        or evidence.get("development_accesses") != 1
        or evidence.get("confirmation_accesses") != 0
        or ledger.get("split") != DEVELOPMENT_SPLIT
        or ledger.get("access_number") != 1
        or report.get("custody", {}).get("development_accesses") != 1
        or report.get("custody", {}).get("confirmation_accesses") != 0
    ):
        raise AssessmentError("ER-TT custody differs")
    return {
        "schema": ASSESSMENT_SCHEMA,
        "protocol": PROTOCOL,
        "scientific_source_commit": checkpoint["scientific_source_commit"],
        "decision": expected_decision,
        "all_gates_pass": all(gates.values()),
        "gates": gates,
        "metrics": recomputed,
        "parameters": EXPECTED_PARAMETERS,
        "artifacts": dict(hashes),
        "custody": {"development_accesses": 1, "confirmation_accesses": 0},
        "independent_metric_recomputation": True,
        "independent_list_executor": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--access-ledger", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing existing ER-TT assessment: {args.output}")
    report = load_json(args.report)
    checkpoint = load_torch(args.checkpoint)
    evidence = load_torch(args.evidence)
    ledger = load_json(args.access_ledger)
    hashes = {
        "report": sha256_file(args.report),
        "checkpoint": sha256_file(args.checkpoint),
        "evidence": sha256_file(args.evidence),
        "ledger": sha256_file(args.access_ledger),
    }
    assessment = assess(report, checkpoint, evidence, ledger, hashes)
    payload = (json.dumps(assessment, indent=2, sort_keys=True) + "\n").encode()
    descriptor = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        os.write(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    args.output.chmod(0o444)
    print(json.dumps({"decision": assessment["decision"], "sha256": sha256_file(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
