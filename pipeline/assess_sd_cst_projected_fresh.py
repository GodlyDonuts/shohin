#!/usr/bin/env python3
"""Independently recompute projected SD-CST fresh-board scores and gates."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from assess_sd_cst_projected_mechanics import (
    rotate_queries,
    semantic_rollout,
    shuffled_packet,
    swap_operand_suffix,
)
from build_sd_cst_projected_board import PROTOCOL
from projected_sd_cst_fresh import (
    PROJECTED_TRAINABLE_NAMES,
    STRUCTURED_KIND_DECODER,
    canonical_json,
    parse_projected_row,
)
from sd_cst import EVENT_STEPS, STOP_KIND, HardLateQuery, HardProgramTape
from train_eval_sd_cst_projected_fresh import (
    ASSESSMENT_SCHEMA,
    CONFIG_SCHEMA,
    CONFIRMATION_SPLIT,
    DEVELOPMENT_SPLIT,
    EVALUATION_SCHEMA,
    safe_mutate_first_active,
    safe_perturb_post_stop,
)


class AssessmentError(ValueError):
    pass


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value = {}
    for key, item in pairs:
        if key in value:
            raise AssessmentError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda item: (_ for _ in ()).throw(
                AssessmentError(f"non-finite JSON constant: {item}")
            ),
        )
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise AssessmentError(f"invalid JSON: {path}") from error
    if not isinstance(value, dict):
        raise AssessmentError("top-level JSON must be an object")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def row_ids_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    payload = "\n".join(sorted(str(row["row_id"]) for row in rows)) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _rate(correct: int, total: int) -> float:
    return correct / total if total else 0.0


def _gate_at_least(value: float, threshold: float) -> dict[str, Any]:
    return {
        "value": value,
        "threshold": threshold,
        "direction": "at_least",
        "pass": value >= threshold,
    }


def _gate_at_most(value: float, threshold: float) -> dict[str, Any]:
    return {
        "value": value,
        "threshold": threshold,
        "direction": "at_most",
        "pass": value <= threshold,
    }


def _tensor(
    value: Any,
    *,
    shape: tuple[int, ...],
    upper: int | None = None,
    dtype: torch.dtype = torch.uint8,
    label: str,
) -> torch.Tensor:
    integer_dtype = dtype is not torch.bool

    def validate(item: Any, dimensions: tuple[int, ...], location: str) -> None:
        if not dimensions:
            valid = type(item) is int if integer_dtype else type(item) is bool
            if not valid:
                expected = "integer" if integer_dtype else "boolean"
                raise AssessmentError(f"{location} is not an exact JSON {expected}")
            if integer_dtype and (item < 0 or (upper is not None and item >= upper)):
                raise AssessmentError(f"{location} is outside the categorical range")
            return
        if not isinstance(item, list) or len(item) != dimensions[0]:
            raise AssessmentError(f"{location} does not match shape {shape}")
        for index, child in enumerate(item):
            validate(child, dimensions[1:], f"{location}[{index}]")

    validate(value, shape, label)
    tensor = torch.tensor(value, dtype=dtype)
    return tensor


def _float_tensor(value: Any, *, shape: tuple[int, ...], label: str) -> torch.Tensor:
    def validate(item: Any, dimensions: tuple[int, ...], location: str) -> None:
        if not dimensions:
            if type(item) is not float or not math.isfinite(item):
                raise AssessmentError(f"{location} is not a finite JSON float")
            return
        if not isinstance(item, list) or len(item) != dimensions[0]:
            raise AssessmentError(f"{location} does not match shape {shape}")
        for index, child in enumerate(item):
            validate(child, dimensions[1:], f"{location}[{index}]")

    validate(value, shape, label)
    return torch.tensor(value, dtype=torch.float32)


def validate_kind_projection(
    value: Mapping[str, Any],
    tape: HardProgramTape,
    rows: int,
    label: str,
) -> dict[str, torch.Tensor]:
    if not isinstance(value, Mapping) or set(value) != {"decoder", "kind_logits"}:
        raise AssessmentError(f"{label} structured decoder evidence keys differ")
    if value.get("decoder") != STRUCTURED_KIND_DECODER:
        raise AssessmentError(f"{label} structured decoder identity differs")
    logits = _float_tensor(
        value["kind_logits"],
        shape=(rows, EVENT_STEPS, STOP_KIND + 1),
        label=f"{label}.kind_logits",
    )
    raw_kind = logits.argmax(-1).to(torch.uint8)
    non_stop_score, non_stop_kind = logits[..., :STOP_KIND].max(-1)
    stop_gain = logits[..., STOP_KIND] - non_stop_score
    selected_stop = stop_gain.argmax(-1)
    projected = non_stop_kind.to(torch.uint8)
    projected.scatter_(1, selected_stop[:, None], STOP_KIND)
    if not torch.equal(projected, tape.event_kind):
        raise AssessmentError(f"{label} packet differs from exact one-STOP MAP")
    return {
        "raw_kind": raw_kind,
        "raw_one_stop": raw_kind.eq(STOP_KIND).sum(-1).eq(1),
        "selected_stop": selected_stop,
    }


def parse_packet(
    value: Mapping[str, Any], rows: int, label: str
) -> tuple[HardProgramTape, HardLateQuery]:
    required = {"initial_state", "event_kind", "event_identity", "amount", "query"}
    if not required.issubset(value):
        raise AssessmentError(f"{label} lacks packet fields")
    tape = HardProgramTape(
        _tensor(
            value["initial_state"], shape=(rows,), upper=6, label=f"{label}.initial"
        ),
        _tensor(
            value["event_kind"],
            shape=(rows, EVENT_STEPS),
            upper=3,
            label=f"{label}.kind",
        ),
        _tensor(
            value["event_identity"],
            shape=(rows, EVENT_STEPS),
            upper=3,
            label=f"{label}.identity",
        ),
        _tensor(
            value["amount"], shape=(rows, EVENT_STEPS), upper=2, label=f"{label}.amount"
        ),
    )
    query = HardLateQuery(
        _tensor(value["query"], shape=(rows,), upper=3, label=f"{label}.query")
    )
    return tape, query


def parse_output(
    value: Mapping[str, Any], rows: int, label: str
) -> dict[str, torch.Tensor]:
    if set(value) != {"final_state", "answer", "state_trajectory", "alive_trajectory"}:
        raise AssessmentError(f"{label} output keys differ")
    return {
        "final_state": _tensor(
            value["final_state"], shape=(rows,), upper=6, label=f"{label}.state"
        ),
        "answer": _tensor(
            value["answer"],
            shape=(rows,),
            upper=3,
            dtype=torch.long,
            label=f"{label}.answer",
        ),
        "state_trajectory": _tensor(
            value["state_trajectory"],
            shape=(rows, EVENT_STEPS),
            upper=6,
            label=f"{label}.trajectory",
        ),
        "alive_trajectory": _tensor(
            value["alive_trajectory"],
            shape=(rows, EVENT_STEPS),
            dtype=torch.bool,
            label=f"{label}.alive",
        ),
    }


def packet_exact(
    prediction: tuple[HardProgramTape, HardLateQuery],
    gold: tuple[HardProgramTape, HardLateQuery],
) -> dict[str, torch.Tensor]:
    tape, query = prediction
    target, target_query = gold
    active = target.event_kind.ne(STOP_KIND)
    fields = {
        "initial": tape.initial_state.eq(target.initial_state),
        "kind": tape.event_kind.eq(target.event_kind).all(-1),
        "identity": (tape.event_identity.eq(target.event_identity) | ~active).all(-1),
        "amount": (tape.amount.eq(target.amount) | ~active).all(-1),
        "query": query.position.eq(target_query.position),
    }
    fields["packet"] = torch.stack(list(fields.values())).all(0)
    return fields


def grouped_rates(
    values: torch.Tensor,
    rows: Sequence[Mapping[str, Any]],
    field: str,
) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[str(row[field])].append(index)
    return {
        key: {
            "rows": len(indices),
            "correct": int(values[indices].sum()),
            "rate": _rate(int(values[indices].sum()), len(indices)),
        }
        for key, indices in sorted(groups.items())
    }


def pointer_exact(
    pointers: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, torch.Tensor]:
    count = len(rows)
    specs = {
        "line": (9, "pointer_ranges"),
        "binding": (3, "binding_ranges"),
        "initial_entity": (3, "initial_entity_ranges"),
        "event_entity": (8, "event_entity_ranges"),
    }
    output = {}
    for name, (slots, ranges_name) in specs.items():
        prediction = _tensor(
            pointers[name],
            shape=(count, slots),
            dtype=torch.long,
            label=f"pointers.{name}",
        )
        row_exact = torch.ones(count, dtype=torch.bool)
        for row_index, row in enumerate(rows):
            ranges = row[ranges_name]
            if len(ranges) != slots:
                raise AssessmentError(f"{ranges_name} slot count differs")
            for slot, pair in enumerate(ranges):
                start, end = (int(pair[0]), int(pair[1]))
                if end <= start:
                    if (
                        name == "event_entity"
                        and int(row["event_kind"][slot]) == STOP_KIND
                    ):
                        continue
                    raise AssessmentError(
                        f"{ranges_name} contains inactive required span"
                    )
                row_exact[row_index] &= start <= int(prediction[row_index, slot]) < end
        output[name] = row_exact
    return output


def _validate_rows(
    rows: Any, expected: Mapping[str, Any], registration: Mapping[str, Any]
) -> list[Mapping[str, Any]]:
    if not isinstance(rows, list) or len(rows) != int(expected["row_count"]):
        raise AssessmentError("evaluation row count differs")
    ids = set()
    family_variants: dict[str, set[str]] = defaultdict(set)
    depth_counts: dict[str, int] = defaultdict(int)
    variants = set(expected["variants"])
    for row in rows:
        if not isinstance(row, Mapping):
            raise AssessmentError("row evidence must be an object")
        row_id = str(row.get("row_id"))
        if row_id in ids:
            raise AssessmentError("duplicate row evidence")
        ids.add(row_id)
        variant = str(row.get("variant"))
        family = str(row.get("family_id"))
        depth = int(row.get("halt_after", -1))
        if variant not in variants or depth not in expected["depths"]:
            raise AssessmentError("row variant/depth outside registration")
        family_variants[family].add(variant)
        depth_counts[str(depth)] += 1
        if row.get("final_state") is None or row.get("answer_role") is None:
            raise AssessmentError("row lacks scored oracle")
        raw_text = row.get("raw_row_canonical_json")
        if not isinstance(raw_text, str):
            raise AssessmentError("row evidence lacks canonical raw source")
        try:
            raw = json.loads(
                raw_text,
                object_pairs_hook=_reject_duplicate_keys,
                parse_constant=lambda item: (_ for _ in ()).throw(
                    AssessmentError(f"non-finite row constant: {item}")
                ),
            )
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise AssessmentError("row canonical source is invalid JSON") from error
        if not isinstance(raw, dict) or canonical_json(raw) != raw_text:
            raise AssessmentError("row source is not canonical JSON")
        if hashlib.sha256(raw_text.encode("utf-8")).hexdigest() != row.get(
            "raw_row_sha256"
        ):
            raise AssessmentError("row canonical source digest differs")
        try:
            reparsed = asdict(parse_projected_row(raw, str(row.get("split"))))
        except (KeyError, TypeError, ValueError) as error:
            raise AssessmentError("row canonical source cannot be reparsed") from error
        reparsed.pop("program_bytes")
        reparsed.pop("query_bytes")
        if canonical_json(reparsed) != canonical_json(dict(row)):
            raise AssessmentError("row evidence differs from its canonical source")
    if len(family_variants) != int(expected["family_count"]):
        raise AssessmentError("family count differs")
    if any(value != variants for value in family_variants.values()):
        raise AssessmentError("family does not contain every registered variant")
    if row_ids_sha256(rows) != registration["row_ids_sha256"]:
        raise AssessmentError("row ID commitment differs")
    content_lines = []
    for row in sorted(rows, key=lambda item: str(item["row_id"])):
        digest = row.get("raw_row_sha256")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(value not in "0123456789abcdef" for value in digest)
        ):
            raise AssessmentError("row evidence lacks a canonical content digest")
        content_lines.append(f"{row['row_id']}:{digest}\n")
    content_sha = hashlib.sha256("".join(content_lines).encode("utf-8")).hexdigest()
    if content_sha != registration.get("row_content_sha256"):
        raise AssessmentError("row content commitment differs")
    if depth_counts != {
        str(key): int(value) for key, value in registration["depth_counts"].items()
    }:
        raise AssessmentError("depth registration differs")
    return rows


def _gold_from_rows(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[HardProgramTape, HardLateQuery, torch.Tensor, torch.Tensor]:
    count = len(rows)
    tape = HardProgramTape(
        _tensor(
            [row["initial_state"] for row in rows],
            shape=(count,),
            upper=6,
            label="rows.initial",
        ),
        _tensor(
            [row["event_kind"] for row in rows],
            shape=(count, 8),
            upper=3,
            label="rows.kind",
        ),
        _tensor(
            [row["event_identity"] for row in rows],
            shape=(count, 8),
            upper=3,
            label="rows.identity",
        ),
        _tensor(
            [row["amount"] for row in rows],
            shape=(count, 8),
            upper=2,
            label="rows.amount",
        ),
    )
    query = HardLateQuery(
        _tensor(
            [row["query_position"] for row in rows],
            shape=(count,),
            upper=3,
            label="rows.query",
        )
    )
    final_state = _tensor(
        [row["final_state"] for row in rows],
        shape=(count,),
        upper=6,
        label="rows.final",
    )
    answer = _tensor(
        [row["answer_role"] for row in rows],
        shape=(count,),
        upper=3,
        dtype=torch.long,
        label="rows.answer",
    )
    semantic_state, semantic_answer, trajectory, _ = semantic_rollout(tape, query)
    if not torch.equal(final_state, semantic_state) or not torch.equal(
        answer, semantic_answer
    ):
        raise AssessmentError("row oracle disagrees with independent packet simulation")
    for index, row in enumerate(rows):
        active = tuple(int(value) for value in row["active_state_trajectory"])
        stop = int(tape.event_kind[index].eq(STOP_KIND).nonzero()[0])
        simulated = (int(tape.initial_state[index]),) + tuple(
            int(value) for value in trajectory[index, :stop]
        )
        if active != simulated:
            raise AssessmentError("row active trajectory disagrees with simulation")
    return tape, query, final_state, answer


def _score_execution(
    output: Mapping[str, torch.Tensor],
    final: torch.Tensor,
    answer: torch.Tensor,
) -> dict[str, torch.Tensor]:
    state = output["final_state"].eq(final)
    answer_ok = output["answer"].eq(answer)
    return {"state": state, "answer": answer_ok, "joint": state & answer_ok}


def _same_packet(
    left: tuple[HardProgramTape, HardLateQuery],
    right: tuple[HardProgramTape, HardLateQuery],
) -> bool:
    return all(
        torch.equal(getattr(left[0], name), getattr(right[0], name))
        for name in (
            "initial_state",
            "event_kind",
            "event_identity",
            "amount",
        )
    ) and torch.equal(left[1].position, right[1].position)


def _expected_source_free(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[HardProgramTape, HardLateQuery]:
    count = len(rows)
    initial = torch.empty(count, dtype=torch.uint8)
    kind = torch.empty((count, EVENT_STEPS), dtype=torch.uint8)
    identity = torch.empty((count, EVENT_STEPS), dtype=torch.uint8)
    amount = torch.empty((count, EVENT_STEPS), dtype=torch.uint8)
    query = torch.empty(count, dtype=torch.uint8)
    for index, row in enumerate(rows):
        digest = hashlib.sha256(
            ("projected-source-free:" + str(row["row_id"])).encode()
        ).digest()
        initial[index] = digest[0] % 6
        stop = digest[1] % EVENT_STEPS
        for step in range(EVENT_STEPS):
            kind[index, step] = (
                STOP_KIND if step == stop else digest[2 + step] % STOP_KIND
            )
            identity[index, step] = digest[10 + step] % 3
            amount[index, step] = digest[18 + step] % 2
        query[index] = digest[26] % 3
    return HardProgramTape(initial, kind, identity, amount), HardLateQuery(query)


def validate_control_packets(
    packet_raw: Mapping[str, Mapping[str, Any]],
    packet_arms: Mapping[str, tuple[HardProgramTape, HardLateQuery]],
    rows: Sequence[Mapping[str, Any]],
) -> None:
    treatment = packet_arms["treatment"]
    count = len(rows)
    for name, value in packet_raw.items():
        control = str(value.get("control"))
        force_alive = value.get("force_alive")
        state_swap = value.get("state_swap")
        swap_after = value.get("swap_after_step")
        expected_control = (
            "reset" if name == "reset" else "freeze" if name == "freeze" else "normal"
        )
        expected_force = name == "force_alive_post_stop"
        expected_swap = (
            list(range(count - 1, -1, -1))
            if name == "state_swap_after_step_0"
            else None
        )
        if (
            control != expected_control
            or force_alive is not expected_force
            or state_swap != expected_swap
            or swap_after != 0
        ):
            raise AssessmentError(
                f"control settings differ from preregistration: {name}"
            )

    uniform_kind = torch.zeros((count, EVENT_STEPS), dtype=torch.uint8)
    uniform_kind[:, 0] = STOP_KIND
    expected = {
        "uniform": (
            HardProgramTape(
                torch.zeros(count, dtype=torch.uint8),
                uniform_kind,
                torch.zeros((count, EVENT_STEPS), dtype=torch.uint8),
                torch.zeros((count, EVENT_STEPS), dtype=torch.uint8),
            ),
            HardLateQuery(torch.zeros(count, dtype=torch.uint8)),
        ),
        "source_free_packet": _expected_source_free(rows),
        "shuffled_packet": (shuffled_packet(treatment[0]), treatment[1]),
        "reset": treatment,
        "freeze": treatment,
        "post_stop_perturbation": (
            safe_perturb_post_stop(treatment[0]),
            treatment[1],
        ),
        "force_alive_post_stop": (
            safe_perturb_post_stop(treatment[0]),
            treatment[1],
        ),
        "operand_suffix_swap": (
            swap_operand_suffix(treatment[0], 4),
            treatment[1],
        ),
        "query_rotation": (treatment[0], rotate_queries(treatment[1])),
        "state_swap_after_step_0": treatment,
        "initial_state_rotation": (
            safe_mutate_first_active(treatment[0], "initial_state"),
            treatment[1],
        ),
        "event_kind_flip": (
            safe_mutate_first_active(treatment[0], "kind"),
            treatment[1],
        ),
        "event_identity_rotation": (
            safe_mutate_first_active(treatment[0], "identity"),
            treatment[1],
        ),
        "event_amount_flip": (
            safe_mutate_first_active(treatment[0], "amount"),
            treatment[1],
        ),
    }
    for name, value in expected.items():
        if not _same_packet(packet_arms[name], value):
            raise AssessmentError(
                f"control packet differs from preregistration: {name}"
            )


def _metric_summary(values: Mapping[str, torch.Tensor]) -> dict[str, dict[str, Any]]:
    rows = next(iter(values.values())).numel()
    return {
        name: {
            "correct": int(value.sum()),
            "rows": rows,
            "rate": _rate(int(value.sum()), rows),
        }
        for name, value in values.items()
    }


def assess(
    eval_payload: Mapping[str, Any], config: Mapping[str, Any]
) -> dict[str, Any]:
    if config.get("schema") != CONFIG_SCHEMA or config.get("protocol") != PROTOCOL:
        raise AssessmentError("projected gate config schema/protocol mismatch")
    if (
        eval_payload.get("schema") != EVALUATION_SCHEMA
        or eval_payload.get("protocol") != PROTOCOL
    ):
        raise AssessmentError("projected evaluation schema/protocol mismatch")
    split = str(eval_payload.get("split"))
    if split not in (DEVELOPMENT_SPLIT, CONFIRMATION_SPLIT):
        raise AssessmentError("invalid projected scored split")
    expected = config["expected"]
    registration = config["registrations"][
        "development" if split == DEVELOPMENT_SPLIT else "confirmation"
    ]
    rows = _validate_rows(eval_payload.get("rows"), expected, registration)
    count = len(rows)
    custody = eval_payload.get("custody")
    expected_access = 0 if split == DEVELOPMENT_SPLIT else 1
    if (
        custody.get("development_accesses") != 1
        or custody.get("confirmation_accesses") != expected_access
        or custody.get("access_ledger", {}).get("sha256")
        != config["expected_ledger_sha256"][
            "development" if split == DEVELOPMENT_SPLIT else "confirmation"
        ]
    ):
        raise AssessmentError("projected scored access custody mismatch")
    authorization = custody.get("confirmation_authorization")
    if split == DEVELOPMENT_SPLIT:
        if authorization is not None:
            raise AssessmentError(
                "development evaluation contains confirmation authorization"
            )
    else:
        required_authorization = {
            "assessment_sha256",
            "development_evaluation_sha256",
            "development_packets_sha256",
            "development_executor_sha256",
        }
        if (
            not isinstance(authorization, Mapping)
            or set(authorization) != required_authorization
            or any(
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
                for value in authorization.values()
            )
        ):
            raise AssessmentError("confirmation authorization evidence differs")

    artifacts = eval_payload.get("artifact_hashes")
    for key in (
        "board",
        "checkpoint",
        "parent",
        "execution_core",
        "consumed_projected",
        "evaluator",
        "assessor",
    ):
        if artifacts.get(key) != config["artifact_hashes"][key]:
            raise AssessmentError(f"projected artifact hash mismatch: {key}")
    expected_split_hash = config["split_hashes"][
        "development" if split == DEVELOPMENT_SPLIT else "confirmation"
    ]
    if artifacts.get("split") != expected_split_hash:
        raise AssessmentError("projected scored split hash mismatch")
    if artifacts.get("gate_config") is None:
        raise AssessmentError("evaluation lacks gate-config commitment")
    if artifacts.get("source_manifest") != config["source"]["sha256"]:
        raise AssessmentError("projected source manifest mismatch")

    parameters = eval_payload.get("parameters")
    if (
        int(parameters.get("nominal_complete_system", -1)) != 146_057_595
        or int(parameters.get("trainable", -1)) != 6_748_897
        or int(parameters.get("nominal_complete_system", -1))
        >= int(config["parameter_caps"]["comparison"])
        or int(parameters.get("nominal_complete_system", -1))
        >= int(config["parameter_caps"]["global"])
        or set(parameters.get("trainable_names", [])) != set(PROJECTED_TRAINABLE_NAMES)
    ):
        raise AssessmentError("projected parameter composition mismatch")
    training = eval_payload.get("training")
    if training.get("contract") != config["training_contract"]:
        raise AssessmentError("projected training contract mismatch")
    fits = training.get("fits")
    if set(fits) != {"treatment", "row_shuffled_labels"}:
        raise AssessmentError("projected matched-arm fit evidence differs")
    if any(
        fit.get("updates") != 3_000 or fit.get("frozen_parent_unchanged") is not True
        for fit in fits.values()
    ):
        raise AssessmentError("projected fit did not preserve frozen contract")
    if (
        fits["treatment"]["initial_full_state_sha256"]
        != fits["row_shuffled_labels"]["initial_full_state_sha256"]
        or fits["treatment"]["minibatch_order_sha256"]
        != fits["row_shuffled_labels"]["minibatch_order_sha256"]
    ):
        raise AssessmentError("matched projected arms differ in initialization/order")
    mapping_sha = training.get("row_shuffled_label_mapping_sha256")
    if (
        not isinstance(mapping_sha, str)
        or len(mapping_sha) != 64
        or any(value not in "0123456789abcdef" for value in mapping_sha)
    ):
        raise AssessmentError("row-shuffled label mapping commitment differs")

    gold = _gold_from_rows(rows)
    declared_gold = parse_packet(eval_payload["gold_packet"], count, "gold_packet")
    if not all(
        torch.equal(getattr(gold[0], name), getattr(declared_gold[0], name))
        for name in (
            "initial_state",
            "event_kind",
            "event_identity",
            "amount",
        )
    ) or not torch.equal(gold[1].position, declared_gold[1].position):
        raise AssessmentError("declared gold packet differs from row evidence")

    expected_compiled = set(expected["arms"])
    compiled_raw = eval_payload.get("compiled")
    if set(compiled_raw) != expected_compiled:
        raise AssessmentError("compiled arm set differs")
    compiled = {}
    for name, value in compiled_raw.items():
        packet = parse_packet(value["packet"], count, f"compiled.{name}")
        compiled[name] = {
            "packet": packet,
            "pointers": pointer_exact(value["pointers"], rows),
            "kind_projection": validate_kind_projection(
                value["kind_projection"],
                packet[0],
                count,
                f"compiled.{name}",
            ),
            "source_poison": value.get("source_poison_bit_identical") is True,
        }

    packet_raw = eval_payload.get("packet_arms")
    required_packet_arms = expected_compiled | set(expected["control_arms"])
    if set(packet_raw) != required_packet_arms:
        raise AssessmentError("packet arm set differs from frozen config")
    packet_arms = {
        name: parse_packet(value, count, f"packet_arms.{name}")
        for name, value in packet_raw.items()
    }
    validate_control_packets(packet_raw, packet_arms, rows)
    for name in expected_compiled:
        left, right = compiled[name]["packet"], packet_arms[name]
        if not all(
            torch.equal(getattr(left[0], field), getattr(right[0], field))
            for field in (
                "initial_state",
                "event_kind",
                "event_identity",
                "amount",
            )
        ) or not torch.equal(left[1].position, right[1].position):
            raise AssessmentError(f"compiled and executed packets differ: {name}")

    executor_raw = eval_payload.get("executor_outputs")
    if set(executor_raw) != required_packet_arms:
        raise AssessmentError("executor arm set differs")
    executor = {
        name: parse_output(value, count, f"executor.{name}")
        for name, value in executor_raw.items()
    }
    gold_packet = (gold[0], gold[1])
    packet_scores = {
        name: packet_exact(value["packet"], gold_packet)
        for name, value in compiled.items()
    }
    execution_scores = {
        name: _score_execution(executor[name], gold[2], gold[3])
        for name in expected_compiled
    }
    treatment_packet = packet_scores["treatment"]
    treatment_execution = execution_scores["treatment"]
    pointer_scores = compiled["treatment"]["pointers"]
    thresholds = config["thresholds"]

    metrics: dict[str, Any] = {
        "compiler": {
            name: _metric_summary(values) for name, values in packet_scores.items()
        },
        "execution": {
            name: _metric_summary(values) for name, values in execution_scores.items()
        },
        "pointers": _metric_summary(pointer_scores),
        "treatment_by_variant": {
            "packet": grouped_rates(treatment_packet["packet"], rows, "variant"),
            "state": grouped_rates(treatment_execution["state"], rows, "variant"),
            "answer": grouped_rates(treatment_execution["answer"], rows, "variant"),
            "joint": grouped_rates(treatment_execution["joint"], rows, "variant"),
        },
        "treatment_by_depth": {
            "packet": grouped_rates(treatment_packet["packet"], rows, "halt_after"),
            "state": grouped_rates(treatment_execution["state"], rows, "halt_after"),
            "answer": grouped_rates(treatment_execution["answer"], rows, "halt_after"),
            "joint": grouped_rates(treatment_execution["joint"], rows, "halt_after"),
        },
        "pointer_by_variant": {
            name: grouped_rates(values, rows, "variant")
            for name, values in pointer_scores.items()
        },
        "raw_kind_diagnostics": {
            name: {
                "one_stop": _metric_summary(
                    {"one_stop": value["kind_projection"]["raw_one_stop"]}
                )["one_stop"],
                "exact_kind": _metric_summary(
                    {
                        "exact_kind": value["kind_projection"]["raw_kind"]
                        .eq(gold[0].event_kind)
                        .all(-1)
                    }
                )["exact_kind"],
            }
            for name, value in compiled.items()
        },
        "treatment_raw_one_stop_by_variant": grouped_rates(
            compiled["treatment"]["kind_projection"]["raw_one_stop"],
            rows,
            "variant",
        ),
    }

    exact_rows = treatment_packet["packet"]
    expected_state, expected_answer, expected_trajectory, expected_alive = (
        semantic_rollout(*compiled["treatment"]["packet"])
    )
    conditional = (
        executor["treatment"]["final_state"].eq(expected_state)
        & executor["treatment"]["answer"].eq(expected_answer)
        & executor["treatment"]["state_trajectory"].eq(expected_trajectory).all(-1)
        & executor["treatment"]["alive_trajectory"].eq(expected_alive).all(-1)
    )
    conditional_rate = _rate(int(conditional[exact_rows].sum()), int(exact_rows.sum()))
    stop_buckets = {}
    for position in range(EVENT_STEPS):
        members = exact_rows & compiled["treatment"]["packet"][0].event_kind[
            :, position
        ].eq(STOP_KIND)
        stop_buckets[str(position)] = {
            "rows": int(members.sum()),
            "correct": int(conditional[members].sum()),
            "rate": _rate(int(conditional[members].sum()), int(members.sum())),
        }
    metrics["conditional_execution"] = {
        "rows": int(exact_rows.sum()),
        "correct": int(conditional[exact_rows].sum()),
        "rate": conditional_rate,
        "stop_buckets": stop_buckets,
    }

    families: dict[str, dict[str, int]] = defaultdict(dict)
    for index, row in enumerate(rows):
        families[str(row["family_id"])][str(row["variant"])] = index
    pair_metrics = {}
    for variant in (
        "binding_recode",
        "paraphrase",
        "query_swap",
        "storage_order_shuffle",
        "post_halt_suffix",
    ):
        eligible = correct = 0
        for values in families.values():
            left, right = values["canonical"], values[variant]
            if not bool(exact_rows[left] and exact_rows[right]):
                continue
            eligible += 1
            left_output = executor["treatment"]
            if variant == "query_swap":
                ok = (
                    int(left_output["final_state"][left])
                    == int(left_output["final_state"][right])
                    and int(left_output["answer"][left]) == int(gold[3][left])
                    and int(left_output["answer"][right]) == int(gold[3][right])
                )
            else:
                ok = int(left_output["final_state"][left]) == int(
                    left_output["final_state"][right]
                ) and int(left_output["answer"][left]) == int(
                    left_output["answer"][right]
                )
            correct += int(ok)
        pair_metrics[variant] = {
            "eligible": eligible,
            "families": len(families),
            "eligibility_rate": _rate(eligible, len(families)),
            "correct": correct,
            "rate": _rate(correct, eligible),
        }
    metrics["paired"] = pair_metrics

    normal_expected = semantic_rollout(*packet_arms["treatment"])
    causal_metrics = {}
    causal_names = (
        "post_stop_perturbation",
        "force_alive_post_stop",
        "operand_suffix_swap",
        "query_rotation",
        "state_swap_after_step_0",
        "reset",
        "freeze",
        "initial_state_rotation",
        "event_kind_flip",
        "event_identity_rotation",
        "event_amount_flip",
    )
    for name in causal_names:
        settings = packet_raw[name]
        state_swap = settings.get("state_swap")
        swap_tensor = (
            torch.tensor(state_swap, dtype=torch.long)
            if state_swap is not None
            else None
        )
        expected_control = semantic_rollout(
            *packet_arms[name],
            control=str(settings.get("control", "normal")),
            state_swap=swap_tensor,
            swap_after_step=int(settings.get("swap_after_step", 0)),
            force_alive=bool(settings.get("force_alive", False)),
        )
        result = executor[name]
        exact = (
            result["final_state"].eq(expected_control[0])
            & result["answer"].eq(expected_control[1])
            & result["state_trajectory"].eq(expected_control[2]).all(-1)
            & result["alive_trajectory"].eq(expected_control[3]).all(-1)
        )
        opportunities = expected_control[0].ne(normal_expected[0]) | expected_control[
            1
        ].ne(normal_expected[1])
        causal_metrics[name] = {
            "rows": count,
            "exact": int(exact.sum()),
            "rate": _rate(int(exact.sum()), count),
            "opportunities": int(opportunities.sum()),
            "opportunity_rate": _rate(int(opportunities.sum()), count),
        }
    metrics["causal_controls"] = causal_metrics

    negative = {}
    for name in (
        "uniform",
        "source_free_packet",
        "shuffled_packet",
        "binding_source_free_compiler",
        "reset",
        "freeze",
    ):
        state = executor[name]["final_state"].eq(gold[2])
        answer = executor[name]["answer"].eq(gold[3])
        negative[name] = {
            "state_rate": _rate(int(state.sum()), count),
            "answer_rate": _rate(int(answer.sum()), count),
        }
    metrics["negative_controls"] = negative

    def minimum(group: Mapping[str, Mapping[str, Any]]) -> float:
        return min(float(value["rate"]) for value in group.values())

    gates = {
        "exact_packet_overall": _gate_at_least(
            metrics["compiler"]["treatment"]["packet"]["rate"],
            thresholds["exact_packet_overall"],
        ),
        "exact_packet_min_variant": _gate_at_least(
            minimum(metrics["treatment_by_variant"]["packet"]),
            thresholds["exact_packet_min_variant"],
        ),
        "exact_packet_min_depth": _gate_at_least(
            minimum(metrics["treatment_by_depth"]["packet"]),
            thresholds["exact_packet_min_depth"],
        ),
        "all_fields_overall": _gate_at_least(
            min(
                metrics["compiler"]["treatment"][name]["rate"]
                for name in (
                    "initial",
                    "kind",
                    "identity",
                    "amount",
                    "query",
                )
            ),
            thresholds["field_overall"],
        ),
        "all_pointers_overall": _gate_at_least(
            min(
                metrics["pointers"][name]["rate"]
                for name in (
                    "binding",
                    "initial_entity",
                    "event_entity",
                )
            ),
            thresholds["pointer_overall"],
        ),
        "all_pointers_min_variant": _gate_at_least(
            min(
                minimum(metrics["pointer_by_variant"][name])
                for name in ("binding", "initial_entity", "event_entity")
            ),
            thresholds["pointer_min_variant"],
        ),
        "autonomous_overall": _gate_at_least(
            min(
                metrics["execution"]["treatment"][name]["rate"]
                for name in (
                    "state",
                    "answer",
                    "joint",
                )
            ),
            thresholds["autonomous_overall"],
        ),
        "autonomous_min_variant": _gate_at_least(
            min(
                minimum(metrics["treatment_by_variant"][name])
                for name in (
                    "state",
                    "answer",
                    "joint",
                )
            ),
            thresholds["autonomous_min_variant"],
        ),
        "autonomous_min_depth": _gate_at_least(
            min(
                minimum(metrics["treatment_by_depth"][name])
                for name in (
                    "state",
                    "answer",
                    "joint",
                )
            ),
            thresholds["autonomous_min_depth"],
        ),
        "conditional_execution": _gate_at_least(
            conditional_rate, thresholds["conditional_execution"]
        ),
        "every_observed_stop_bucket_exact": {
            "value": all(
                item["rows"] == 0 or item["rate"] == 1.0
                for item in stop_buckets.values()
            ),
            "threshold": True,
            "direction": "equal",
            "pass": all(
                item["rows"] == 0 or item["rate"] == 1.0
                for item in stop_buckets.values()
            ),
        },
        "treatment_packet_advantage": _gate_at_least(
            metrics["compiler"]["treatment"]["packet"]["rate"]
            - metrics["compiler"]["row_shuffled_labels"]["packet"]["rate"],
            thresholds["treatment_advantage"],
        ),
        "treatment_joint_advantage": _gate_at_least(
            metrics["execution"]["treatment"]["joint"]["rate"]
            - metrics["execution"]["row_shuffled_labels"]["joint"]["rate"],
            thresholds["treatment_advantage"],
        ),
        "pair_eligibility": _gate_at_least(
            min(value["eligibility_rate"] for value in pair_metrics.values()),
            thresholds["pair_eligibility"],
        ),
        "paired_consistency": _gate_at_least(
            min(value["rate"] for value in pair_metrics.values()),
            thresholds["paired_consistency"],
        ),
        "every_causal_arm_matches_oracle": _gate_at_least(
            min(value["rate"] for value in causal_metrics.values()),
            thresholds["causal_oracle"],
        ),
        "causal_opportunities": _gate_at_least(
            min(
                value["opportunity_rate"]
                for name, value in causal_metrics.items()
                if name not in ("post_stop_perturbation", "query_rotation")
            ),
            thresholds["causal_min_opportunity"],
        ),
        "query_opportunities": _gate_at_least(
            causal_metrics["query_rotation"]["opportunity_rate"],
            thresholds["query_min_opportunity"],
        ),
        "post_stop_invariant": {
            "value": causal_metrics["post_stop_perturbation"]["opportunities"],
            "threshold": 0,
            "direction": "equal",
            "pass": causal_metrics["post_stop_perturbation"]["opportunities"] == 0,
        },
        "negative_state": _gate_at_most(
            max(
                negative[name]["state_rate"]
                for name in (
                    "uniform",
                    "source_free_packet",
                    "shuffled_packet",
                    "binding_source_free_compiler",
                )
            ),
            thresholds["negative_state_max"],
        ),
        "negative_answer": _gate_at_most(
            max(
                negative[name]["answer_rate"]
                for name in (
                    "uniform",
                    "source_free_packet",
                    "shuffled_packet",
                    "binding_source_free_compiler",
                )
            ),
            thresholds["negative_answer_max"],
        ),
        "reset_freeze_negative": _gate_at_most(
            max(
                negative[name][field]
                for name in ("reset", "freeze")
                for field in (
                    "state_rate",
                    "answer_rate",
                )
            ),
            thresholds["reset_freeze_max"],
        ),
        "source_deletion": {
            "value": (
                eval_payload.get("source_deletion")
                == {
                    "program_elements_per_row": 25,
                    "query_elements_per_row": 1,
                    "dtype": "torch.uint8",
                    "program_gpu_tensors_destroyed_before_query_compile": True,
                    "host_evaluator_retains_hash_bound_row_evidence": True,
                    "separate_typed_executor": True,
                    "structured_kind_decoder": STRUCTURED_KIND_DECODER,
                    "all_compiler_arms_source_poison_bit_identical": True,
                }
                and all(value["source_poison"] for value in compiled.values())
            ),
            "threshold": True,
            "direction": "equal",
            "pass": (
                eval_payload.get("source_deletion")
                == {
                    "program_elements_per_row": 25,
                    "query_elements_per_row": 1,
                    "dtype": "torch.uint8",
                    "program_gpu_tensors_destroyed_before_query_compile": True,
                    "host_evaluator_retains_hash_bound_row_evidence": True,
                    "separate_typed_executor": True,
                    "structured_kind_decoder": STRUCTURED_KIND_DECODER,
                    "all_compiler_arms_source_poison_bit_identical": True,
                }
                and all(value["source_poison"] for value in compiled.values())
            ),
        },
        "exact_map_one_stop_decoder_contract": {
            "value": True,
            "threshold": True,
            "direction": "equal",
            "pass": True,
        },
        "parameter_and_training_contract": {
            "value": True,
            "threshold": True,
            "direction": "equal",
            "pass": True,
        },
        "custody_and_hash_contract": {
            "value": True,
            "threshold": True,
            "direction": "equal",
            "pass": True,
        },
    }
    all_pass = all(bool(value["pass"]) for value in gates.values())
    return {
        "schema": ASSESSMENT_SCHEMA,
        "protocol": PROTOCOL,
        "split": split,
        "decision": (
            "authorize_one_sealed_confirmation"
            if split == DEVELOPMENT_SPLIT and all_pass
            else "accept_projected_fresh_confirmation"
            if split == CONFIRMATION_SPLIT and all_pass
            else "reject_projected_fresh_board"
        ),
        "confirmation_authorized": split == DEVELOPMENT_SPLIT and all_pass,
        "all_gates_pass": all_pass,
        "metrics": metrics,
        "gates": gates,
        "claim_boundary": (
            "Bounded fresh-board source-deleted state transport. The nominal Shohin "
            "trunk is inactive in the projected compiler forward path."
        ),
    }


def _pt_tensor(
    value: object,
    *,
    dtype: torch.dtype,
    shape: tuple[int, ...],
    label: str,
) -> torch.Tensor:
    if (
        not isinstance(value, torch.Tensor)
        or value.dtype != dtype
        or tuple(value.shape) != shape
        or value.layout != torch.strided
    ):
        raise AssessmentError(f"{label} tensor contract differs")
    return value


def _bind_auxiliary_artifacts(
    evaluation: Mapping[str, Any],
    packet_path: Path,
    executor_path: Path,
) -> None:
    artifacts = evaluation.get("artifact_hashes", {})
    if sha256_file(packet_path) != artifacts.get("hard_packets"):
        raise AssessmentError("hard-packet artifact hash differs")
    if sha256_file(executor_path) != artifacts.get("executor_outputs"):
        raise AssessmentError("executor-output artifact hash differs")
    try:
        packet_bundle = torch.load(packet_path, map_location="cpu", weights_only=True)
        output_bundle = torch.load(executor_path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise AssessmentError(
            "unsafe or malformed auxiliary tensor artifact"
        ) from error
    if (
        not isinstance(packet_bundle, Mapping)
        or set(packet_bundle) != {"schema", "arms"}
        or packet_bundle.get("schema") != "r12_sd_cst_hard_packet_bundle_v1"
        or not isinstance(packet_bundle.get("arms"), Mapping)
    ):
        raise AssessmentError("hard-packet bundle schema differs")
    packet_json: dict[str, Any] = {}
    expected_packet_json = evaluation.get("packet_arms", {})
    rows = len(evaluation.get("rows", []))
    arm_keys = {
        "initial_state",
        "event_kind",
        "event_identity",
        "amount",
        "query",
        "control",
        "force_alive",
        "state_swap",
        "swap_after_step",
    }
    if set(packet_bundle["arms"]) != set(expected_packet_json):
        raise AssessmentError("hard-packet arm set differs from evaluation")
    for name, arm in packet_bundle["arms"].items():
        if not isinstance(arm, Mapping) or set(arm) != arm_keys:
            raise AssessmentError(f"hard-packet arm schema differs: {name}")
        tensors = {
            "initial_state": _pt_tensor(
                arm["initial_state"],
                dtype=torch.uint8,
                shape=(rows,),
                label=f"packets.{name}.initial_state",
            ),
            "event_kind": _pt_tensor(
                arm["event_kind"],
                dtype=torch.uint8,
                shape=(rows, EVENT_STEPS),
                label=f"packets.{name}.event_kind",
            ),
            "event_identity": _pt_tensor(
                arm["event_identity"],
                dtype=torch.uint8,
                shape=(rows, EVENT_STEPS),
                label=f"packets.{name}.event_identity",
            ),
            "amount": _pt_tensor(
                arm["amount"],
                dtype=torch.uint8,
                shape=(rows, EVENT_STEPS),
                label=f"packets.{name}.amount",
            ),
            "query": _pt_tensor(
                arm["query"],
                dtype=torch.uint8,
                shape=(rows,),
                label=f"packets.{name}.query",
            ),
        }
        state_swap = arm["state_swap"]
        if state_swap is not None:
            state_swap = _pt_tensor(
                state_swap,
                dtype=torch.int64,
                shape=(rows,),
                label=f"packets.{name}.state_swap",
            ).tolist()
        if type(arm["control"]) is not str or type(arm["force_alive"]) is not bool:
            raise AssessmentError(f"hard-packet control types differ: {name}")
        if type(arm["swap_after_step"]) is not int:
            raise AssessmentError(f"hard-packet swap step type differs: {name}")
        packet_json[str(name)] = {
            key: tensor.tolist() for key, tensor in tensors.items()
        } | {
            "control": arm["control"],
            "force_alive": arm["force_alive"],
            "state_swap": state_swap,
            "swap_after_step": arm["swap_after_step"],
        }
    if packet_json != expected_packet_json:
        raise AssessmentError("hard-packet artifact content differs from evaluation")

    if (
        not isinstance(output_bundle, Mapping)
        or set(output_bundle) != {"schema", "outputs"}
        or output_bundle.get("schema") != "r12_sd_cst_hard_packet_outputs_v1"
        or not isinstance(output_bundle.get("outputs"), Mapping)
        or set(output_bundle["outputs"]) != set(expected_packet_json)
    ):
        raise AssessmentError("executor-output bundle schema differs")
    output_json: dict[str, Any] = {}
    output_specs = {
        "final_state": (torch.uint8, (rows,)),
        "answer": (torch.int64, (rows,)),
        "state_trajectory": (torch.uint8, (rows, EVENT_STEPS)),
        "alive_trajectory": (torch.bool, (rows, EVENT_STEPS)),
    }
    for name, output in output_bundle["outputs"].items():
        if not isinstance(output, Mapping) or set(output) != set(output_specs):
            raise AssessmentError(f"executor output schema differs: {name}")
        output_json[str(name)] = {
            field: _pt_tensor(
                output[field],
                dtype=dtype,
                shape=shape,
                label=f"executor.{name}.{field}",
            ).tolist()
            for field, (dtype, shape) in output_specs.items()
        }
    if output_json != evaluation.get("executor_outputs"):
        raise AssessmentError("executor artifact content differs from evaluation")


def assess_files(
    eval_path: Path,
    config_path: Path,
    packet_path: Path,
    executor_path: Path,
) -> dict[str, Any]:
    evaluation = load_json(eval_path)
    config = load_json(config_path)
    if sha256_file(Path(__file__)) != config.get("artifact_hashes", {}).get("assessor"):
        raise AssessmentError("runtime assessor differs from gate config")
    if evaluation.get("artifact_hashes", {}).get("gate_config") != sha256_file(
        config_path
    ):
        raise AssessmentError("evaluation gate-config hash differs")
    ledger = evaluation.get("custody", {}).get("access_ledger", {})
    ledger_path = Path(str(ledger.get("path", "")))
    canonical_access = Path(str(config.get("canonical_data_dir", ""))) / "access"
    try:
        if ledger_path.resolve().parent != canonical_access.resolve():
            raise AssessmentError("access ledger is outside the canonical board")
    except OSError as error:
        raise AssessmentError("access ledger path cannot be resolved") from error
    if not ledger_path.is_file() or sha256_file(ledger_path) != ledger.get("sha256"):
        raise AssessmentError("canonical access ledger bytes differ")
    _bind_auxiliary_artifacts(evaluation, packet_path, executor_path)
    result = assess(evaluation, config)
    result["gate_config_sha256"] = sha256_file(config_path)
    result["evaluation_sha256"] = sha256_file(eval_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--gate-config", type=Path, required=True)
    parser.add_argument("--hard-packets", type=Path, required=True)
    parser.add_argument("--executor-outputs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing existing projected assessment: {args.out}")
    result = assess_files(
        args.evaluation,
        args.gate_config,
        args.hard_packets,
        args.executor_outputs,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "saved": str(args.out.resolve()),
                "sha256": sha256_file(args.out),
                "decision": result["decision"],
                "all_gates_pass": result["all_gates_pass"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
