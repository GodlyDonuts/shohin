#!/usr/bin/env python3
"""Build the source-deleted counterfactual state-transport (SD-CST) board.

The board contains seven genuine operations in every source.  STOP identifies
the semantic boundary between an active prefix of depth one through six and a
nonempty, still-valid operation suffix.  Training rows expose compiler fields
only; final states, answers, and trajectories exist only on evaluation rows.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import random
import re
import subprocess
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from audit_sd_cst_board import audit_board


ENTITY_COUNT = 3
OPERATION_COUNT = 7
EVENT_COUNT = 8
DEPTHS = tuple(range(1, OPERATION_COUNT))
DIRECTIONS = ("left", "right")
AMOUNTS = (1, 2)
TRAIN_SPLIT = "sd_cst_train"
DEVELOPMENT_SPLIT = "sd_cst_development"
CONFIRMATION_SPLIT = "sd_cst_confirmation"
EVALUATION_SPLITS = (DEVELOPMENT_SPLIT, CONFIRMATION_SPLIT)
SURFACE_TYPES = (
    "canonical",
    "query_swap",
    "paraphrase",
    "binding_recode",
    "order_counterfactual",
    "stop_shift",
    "storage_order_shuffle",
    "post_halt_suffix",
)
WORD_RE = re.compile(r"[a-z0-9]+")
SPLIT_MARKERS = {
    TRAIN_SPLIT: "Talvek",
    DEVELOPMENT_SPLIT: "Mirexo",
    CONFIRMATION_SPLIT: "Quorin",
}
TEMPLATE_IDS = {
    TRAIN_SPLIT: {"direct": "train_talvek_direct_v1"},
    DEVELOPMENT_SPLIT: {
        "direct": "development_mirexo_direct_v1",
        "paraphrase": "development_mirexo_paraphrase_v1",
    },
    CONFIRMATION_SPLIT: {
        "direct": "confirmation_quorin_direct_v1",
        "paraphrase": "confirmation_quorin_paraphrase_v1",
    },
}
_SPLIT_SEED_MASKS = {
    TRAIN_SPLIT: 0x54A11E,
    DEVELOPMENT_SPLIT: 0xD3E10,
    CONFIRMATION_SPLIT: 0xC0F1A,
}
PERMUTATIONS = tuple(itertools.permutations(range(ENTITY_COUNT)))
PERMUTATION_TO_STATE = {value: index for index, value in enumerate(PERMUTATIONS)}


@dataclass(frozen=True)
class Operation:
    entity_role: int
    direction: str
    amount: int

    def as_dict(self, ordinal: int) -> dict[str, object]:
        return {
            "semantic_ordinal": int(ordinal),
            "entity_role": int(self.entity_role),
            "direction": self.direction,
            "amount": int(self.amount),
        }


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def normalized_prompt(text: str) -> str:
    return " ".join(WORD_RE.findall(text.lower()))


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def row_ids_sha256(rows: Sequence[dict[str, object]]) -> str:
    payload = "\n".join(sorted(str(row["id"]) for row in rows)) + "\n"
    return sha256_bytes(payload.encode("utf-8"))


def sequence_signature(program: Sequence[Operation]) -> str:
    return canonical_json(
        [(op.entity_role, op.direction, op.amount) for op in program]
    )


def operation_bag(program: Sequence[Operation]) -> Counter[tuple[int, str, int]]:
    return Counter((op.entity_role, op.direction, op.amount) for op in program)


def _destination(index: int, direction: str, amount: int) -> int:
    if direction == "left":
        return max(0, index - amount)
    if direction == "right":
        return min(ENTITY_COUNT - 1, index + amount)
    raise ValueError(f"unknown direction: {direction}")


def apply_operation_pop_insert(
    state: Sequence[int], operation: Operation
) -> tuple[int, ...]:
    """Primary simulator used by generation; the auditor uses adjacent swaps."""
    values = list(state)
    source = values.index(operation.entity_role)
    destination = _destination(source, operation.direction, operation.amount)
    value = values.pop(source)
    values.insert(destination, value)
    return tuple(values)


def simulate_pop_insert(
    initial_order: Sequence[int],
    program: Sequence[Operation],
    halt_after: int,
) -> tuple[tuple[int, ...], tuple[tuple[int, ...], ...]]:
    if halt_after not in range(0, OPERATION_COUNT + 1):
        raise ValueError("halt_after outside event tape")
    state = tuple(initial_order)
    trajectory = [state]
    for operation in program[:halt_after]:
        state = apply_operation_pop_insert(state, operation)
        trajectory.append(state)
    return state, tuple(trajectory)


def _operation_changes(state: Sequence[int], operation: Operation) -> bool:
    source = tuple(state).index(operation.entity_role)
    return _destination(source, operation.direction, operation.amount) != source


def _valid_operations(state: Sequence[int]) -> tuple[Operation, ...]:
    return tuple(
        operation
        for role in range(ENTITY_COUNT)
        for direction in DIRECTIONS
        for amount in AMOUNTS
        for operation in (Operation(role, direction, amount),)
        if _operation_changes(state, operation)
    )


def _sample_valid_continuation(
    rng: random.Random,
    initial_state: Sequence[int],
    length: int,
) -> tuple[Operation, ...]:
    state = tuple(initial_state)
    operations: list[Operation] = []
    for _ in range(length):
        operation = rng.choice(_valid_operations(state))
        operations.append(operation)
        state = apply_operation_pop_insert(state, operation)
    return tuple(operations)


def _program_is_fully_valid(
    initial_order: Sequence[int], program: Sequence[Operation]
) -> bool:
    state = tuple(initial_order)
    for operation in program:
        if not _operation_changes(state, operation):
            return False
        state = apply_operation_pop_insert(state, operation)
    return True


def _name_pool(split: str, seed: int, count: int) -> tuple[str, ...]:
    names: list[str] = []
    seen: set[str] = set()
    for index in range(count * 3):
        digest = hashlib.sha256(
            f"sd-cst:{split}:{seed}:{index}".encode("ascii")
        ).hexdigest()
        name = f"{digest[:4]}-{digest[4:12]}"
        if name not in seen:
            seen.add(name)
            names.append(name)
        if len(names) == count:
            return tuple(names)
    raise RuntimeError("could not construct opaque name pool")


def _lexical_direction(direction: str) -> str:
    return {"left": "west", "right": "east"}[direction]


def _lexical_amount(amount: int) -> str:
    return {1: "unit", 2: "pair"}[amount]


def render_program_text(
    *,
    split: str,
    template: str,
    names_by_role: Sequence[str],
    initial_order: Sequence[int],
    program: Sequence[Operation],
    halt_after: int,
    storage_order: Sequence[int],
) -> str:
    marker = SPLIT_MARKERS[split]
    initial_names = [names_by_role[role] for role in initial_order]
    if template == "direct":
        lines = [
            f"{marker} bindings: alpha {names_by_role[0]}; beta {names_by_role[1]}; "
            f"gamma {names_by_role[2]}; initial {', '.join(initial_names)}."
        ]
    elif template == "paraphrase":
        lines = [
            f"{marker} registry: {names_by_role[0]} is alpha; {names_by_role[1]} "
            f"is beta; {names_by_role[2]} is gamma; lineup {', '.join(initial_names)}."
        ]
    else:
        raise ValueError(f"unknown template: {template}")

    if len(program) != OPERATION_COUNT:
        raise ValueError("SD-CST requires exactly seven state-changing operations")
    stop_ordinal = halt_after + 1
    operation_iter = iter(program)
    by_ordinal: dict[int, Operation | None] = {}
    for ordinal in range(1, EVENT_COUNT + 1):
        by_ordinal[ordinal] = None if ordinal == stop_ordinal else next(operation_iter)
    for ordinal in storage_order:
        operation = by_ordinal[int(ordinal)]
        if operation is None:
            noun = "event" if template == "direct" else "action"
            lines.append(f"{marker} {noun} {ordinal}: STOP.")
            continue
        entity = names_by_role[operation.entity_role]
        direction = _lexical_direction(operation.direction)
        amount = _lexical_amount(operation.amount)
        if template == "direct":
            lines.append(
                f"{marker} event {ordinal}: move {entity} {direction} by {amount}."
            )
        else:
            lines.append(
                f"{marker} action {ordinal}: send {entity} {direction} for {amount}."
            )
    return "\n".join(lines)


def render_late_query_text(split: str, query_position: int) -> str:
    marker = SPLIT_MARKERS[split]
    position = query_position + 1
    if split == TRAIN_SPLIT:
        return f"{marker} asks which entity now occupies position {position}?"
    if split == DEVELOPMENT_SPLIT:
        return f"{marker} requests the entity currently in position {position}?"
    if split == CONFIRMATION_SPLIT:
        return f"{marker} seeks the entity presently at position {position}?"
    raise ValueError(f"unknown split: {split}")


def _compiler_targets(
    names_by_role: Sequence[str],
    initial_order: Sequence[int],
    program: Sequence[Operation],
    halt_after: int,
    storage_order: Sequence[int],
) -> dict[str, object]:
    stop_ordinal = halt_after + 1
    operation_iter = iter(program)
    event_slots: list[dict[str, object]] = []
    for ordinal in range(1, EVENT_COUNT + 1):
        if ordinal == stop_ordinal:
            event_slots.append({
                "semantic_ordinal": ordinal,
                "kind": "stop",
                "kind_id": 2,
                "entity_role": 0,
                "entity": None,
                "direction": None,
                "amount": 1,
                "amount_id": 0,
                "identity_and_amount_scored": False,
            })
            continue
        operation = next(operation_iter)
        event_slots.append(
            operation.as_dict(ordinal)
            | {
                "kind": operation.direction,
                "kind_id": {"left": 0, "right": 1}[operation.direction],
                "entity": names_by_role[operation.entity_role],
                "amount_id": operation.amount - 1,
                "identity_and_amount_scored": True,
            }
        )
    return {
        "entity_bindings": [
            {"entity_role": role, "entity": names_by_role[role]}
            for role in range(ENTITY_COUNT)
        ],
        "initial_order_roles": list(initial_order),
        "initial_state_id": PERMUTATION_TO_STATE[tuple(initial_order)],
        "event_slots": event_slots,
        "halt_after": int(halt_after),
        "storage_order": [int(value) for value in storage_order],
    }


def _oracle(
    names_by_role: Sequence[str],
    initial_order: Sequence[int],
    program: Sequence[Operation],
    halt_after: int,
    query_position: int,
) -> dict[str, object]:
    state, trajectory = simulate_pop_insert(initial_order, program, halt_after)
    answer_role = state[query_position]
    return {
        "final_state_roles": list(state),
        "answer_role": int(answer_role),
        "answer_entity": names_by_role[answer_role],
        "active_trajectory_roles": [list(item) for item in trajectory],
    }


def make_row(
    *,
    split: str,
    row_id: str,
    variant: str,
    names_by_role: Sequence[str],
    initial_order: Sequence[int],
    program: Sequence[Operation],
    halt_after: int,
    query_position: int,
    storage_order: Sequence[int],
    template: str,
    family_id: str | None = None,
) -> dict[str, object]:
    program_text = render_program_text(
        split=split,
        template=template,
        names_by_role=names_by_role,
        initial_order=initial_order,
        program=program,
        halt_after=halt_after,
        storage_order=storage_order,
    )
    late_query_text = render_late_query_text(split, query_position)
    combined_normalized_prompt = normalized_prompt(
        program_text + "\n" + late_query_text
    )
    row: dict[str, object] = {
        "id": row_id,
        "schema": "r12_sd_cst_row_v1",
        "split": split,
        "variant": variant,
        "template_id": TEMPLATE_IDS[split][template],
        "program_text": program_text,
        "late_query_text": late_query_text,
        "audit_only_combined_normalized_prompt": combined_normalized_prompt,
        "model_input_contract": {
            "compile_phase_input": "program_text",
            "delete_program_state_before": "late_query_text",
            "combined_prompt_is_model_input": False,
        },
        "source_shape": {
            "program_line_count": len(program_text.splitlines()),
            "late_query_line_count": len(late_query_text.splitlines()),
            "event_clause_count": EVENT_COUNT,
            "program_character_count": len(program_text),
            "program_word_count": len(program_text.replace("\n", " ").split()),
            "query_character_count": len(late_query_text),
            "query_word_count": len(late_query_text.split()),
        },
        "supervision": "compiler_fields_only" if split == TRAIN_SPLIT else "sealed_oracle",
        "compiler_targets": _compiler_targets(
            names_by_role,
            initial_order,
            program,
            halt_after,
            storage_order,
        ),
        "late_query_target": {"position": int(query_position)},
    }
    if family_id is not None:
        row["family_id"] = family_id
    if split in EVALUATION_SPLITS:
        row["oracle"] = _oracle(
            names_by_role,
            initial_order,
            program,
            halt_after,
            query_position,
        )
    return row


def _sample_unique_train_program(
    rng: random.Random,
    initial_order: Sequence[int],
    used_sequences: set[str],
) -> tuple[Operation, ...]:
    for _ in range(10000):
        program = _sample_valid_continuation(rng, initial_order, OPERATION_COUNT)
        signature = sequence_signature(program)
        if signature not in used_sequences:
            used_sequences.add(signature)
            return program
    raise RuntimeError("could not draw a fresh SD-CST training sequence")


def build_train(
    count: int,
    seed: int,
    used_sequences: set[str] | None = None,
) -> list[dict[str, object]]:
    if count < len(DEPTHS):
        raise ValueError("training board must cover all active depths")
    used = used_sequences if used_sequences is not None else set()
    rng = random.Random(seed)
    names = _name_pool(TRAIN_SPLIT, seed ^ _SPLIT_SEED_MASKS[TRAIN_SPLIT], 384)
    rows: list[dict[str, object]] = []
    for index in range(count):
        initial_order = tuple(rng.sample(range(ENTITY_COUNT), ENTITY_COUNT))
        program = _sample_unique_train_program(rng, initial_order, used)
        names_by_role = tuple(rng.sample(names, ENTITY_COUNT))
        depth = DEPTHS[index % len(DEPTHS)]
        query = rng.randrange(ENTITY_COUNT)
        storage_order = list(range(1, EVENT_COUNT + 1))
        rng.shuffle(storage_order)
        rows.append(
            make_row(
                split=TRAIN_SPLIT,
                row_id=f"SDCST-TRAIN-{index:06d}",
                variant="compiler_train",
                names_by_role=names_by_role,
                initial_order=initial_order,
                program=program,
                halt_after=depth,
                query_position=query,
                storage_order=tuple(storage_order),
                template="direct",
            )
        )
    return rows


def _state_at(
    initial_order: Sequence[int], program: Sequence[Operation], depth: int
) -> tuple[int, ...]:
    return simulate_pop_insert(initial_order, program, depth)[0]


def _storage_naive_state(
    initial_order: Sequence[int],
    program: Sequence[Operation],
    halt_after: int,
    storage_order: Sequence[int],
) -> tuple[int, ...]:
    stop_ordinal = halt_after + 1
    operation_iter = iter(program)
    tape: dict[int, Operation | None] = {}
    for ordinal in range(1, EVENT_COUNT + 1):
        tape[ordinal] = None if ordinal == stop_ordinal else next(operation_iter)
    state = tuple(initial_order)
    for ordinal in storage_order:
        operation = tape[int(ordinal)]
        if operation is None:
            break
        state = apply_operation_pop_insert(state, operation)
    return state


def _sample_post_halt_program(
    rng: random.Random,
    initial_order: Sequence[int],
    base_program: Sequence[Operation],
    halt_after: int,
    query_position: int,
    used_sequences: set[str],
) -> tuple[Operation, ...] | None:
    prefix = tuple(base_program[:halt_after])
    prefix_state = _state_at(initial_order, prefix, halt_after)
    for _ in range(500):
        suffix = _sample_valid_continuation(
            rng, prefix_state, OPERATION_COUNT - halt_after
        )
        candidate = prefix + suffix
        signature = sequence_signature(candidate)
        if candidate == tuple(base_program) or signature in used_sequences:
            continue
        full_state = _state_at(initial_order, candidate, OPERATION_COUNT)
        active_state = _state_at(initial_order, candidate, halt_after)
        if full_state[query_position] == active_state[query_position]:
            continue
        return candidate
    return None


def _sample_family_semantics(
    rng: random.Random,
    depth: int,
    target_query_position: int,
    target_answer_role: int,
    used_sequences: set[str],
) -> dict[str, object]:
    alternate_depth = OPERATION_COUNT - depth
    query = int(target_query_position)
    query_swap_position = (query + 1) % ENTITY_COUNT
    order_answer_role = (target_answer_role + 1) % ENTITY_COUNT
    stop_answer_role = (target_answer_role + 2) % ENTITY_COUNT
    for _ in range(20000):
        initial_order = tuple(rng.sample(range(ENTITY_COUNT), ENTITY_COUNT))
        base = _sample_valid_continuation(rng, initial_order, OPERATION_COUNT)
        base_signature = sequence_signature(base)
        if base_signature in used_sequences:
            continue
        base_state = _state_at(initial_order, base, depth)
        stop_state = _state_at(initial_order, base, alternate_depth)
        full_state = _state_at(initial_order, base, OPERATION_COUNT)
        if base_state[query] != target_answer_role:
            continue
        if base_state[query_swap_position] != order_answer_role:
            continue
        if stop_state[query] != stop_answer_role:
            continue
        if full_state[query] == target_answer_role:
            continue

        indices = list(range(OPERATION_COUNT))
        for _ in range(300):
            rng.shuffle(indices)
            permutation = tuple(indices)
            if permutation == tuple(range(OPERATION_COUNT)):
                continue
            order_program = tuple(base[index] for index in permutation)
            order_signature = sequence_signature(order_program)
            if order_signature in used_sequences or order_signature == base_signature:
                continue
            if not _program_is_fully_valid(initial_order, order_program):
                continue
            order_state = _state_at(initial_order, order_program, depth)
            if order_state[query] != order_answer_role:
                continue
            suffix_program = _sample_post_halt_program(
                rng,
                initial_order,
                base,
                depth,
                query,
                used_sequences | {base_signature, order_signature},
            )
            if suffix_program is None:
                continue
            suffix_signature = sequence_signature(suffix_program)
            used_sequences.update(
                {base_signature, order_signature, suffix_signature}
            )
            storage_indices = list(range(1, EVENT_COUNT + 1))
            for _ in range(100):
                rng.shuffle(storage_indices)
                if storage_indices == list(range(1, EVENT_COUNT + 1)):
                    continue
                naive_state = _storage_naive_state(
                    initial_order, base, depth, storage_indices,
                )
                if naive_state[query] != target_answer_role:
                    break
            else:
                continue
            return {
                "initial_order": initial_order,
                "base_program": base,
                "order_program": order_program,
                "suffix_program": suffix_program,
                "storage_order": tuple(storage_indices),
                "halt_after": depth,
                "alternate_halt_after": alternate_depth,
                "query_position": query,
                "query_swap_position": query_swap_position,
            }
    raise RuntimeError(f"could not sample separating depth-{depth} family")


def build_paired_split(
    split: str,
    families: int,
    seed: int,
    used_sequences: set[str] | None = None,
) -> list[dict[str, object]]:
    if split not in EVALUATION_SPLITS:
        raise ValueError("paired boards are development or confirmation only")
    if families < len(DEPTHS):
        raise ValueError("paired board must cover all active depths")
    used = used_sequences if used_sequences is not None else set()
    rng = random.Random(seed)
    names = _name_pool(split, seed ^ _SPLIT_SEED_MASKS[split], 768)
    natural_storage = tuple(range(1, EVENT_COUNT + 1))
    rows: list[dict[str, object]] = []
    split_tag = "DEV" if split == DEVELOPMENT_SPLIT else "CONF"

    for family_index in range(families):
        depth = DEPTHS[family_index % len(DEPTHS)]
        target_query_position = family_index % ENTITY_COUNT
        target_answer_role = (family_index // ENTITY_COUNT) % ENTITY_COUNT
        semantics = _sample_family_semantics(
            rng,
            depth,
            target_query_position,
            target_answer_role,
            used,
        )
        initial_order = semantics["initial_order"]
        base_program = semantics["base_program"]
        order_program = semantics["order_program"]
        suffix_program = semantics["suffix_program"]
        query = int(semantics["query_position"])
        query_swap = int(semantics["query_swap_position"])
        alternate_depth = int(semantics["alternate_halt_after"])
        storage_order = semantics["storage_order"]
        base_names = tuple(rng.sample(names, ENTITY_COUNT))
        recode_candidates = [name for name in names if name not in base_names]
        recoded_names = tuple(rng.sample(recode_candidates, ENTITY_COUNT))
        family_id = f"SDCST-{split_tag}-FAMILY-{family_index:06d}"

        definitions = (
            ("canonical", base_names, base_program, depth, natural_storage, "direct"),
            ("query_swap", base_names, base_program, depth, natural_storage, "direct"),
            ("paraphrase", base_names, base_program, depth, natural_storage, "paraphrase"),
            ("binding_recode", recoded_names, base_program, depth, natural_storage, "direct"),
            ("order_counterfactual", base_names, order_program, depth, natural_storage, "direct"),
            ("stop_shift", base_names, base_program, alternate_depth, natural_storage, "direct"),
            ("storage_order_shuffle", base_names, base_program, depth, storage_order, "direct"),
            ("post_halt_suffix", base_names, suffix_program, depth, natural_storage, "direct"),
        )
        for variant, row_names, program, row_depth, row_storage, template in definitions:
            row_query = query_swap if variant == "query_swap" else query
            rows.append(
                make_row(
                    split=split,
                    row_id=f"{family_id}-{variant}",
                    family_id=family_id,
                    variant=variant,
                    names_by_role=row_names,
                    initial_order=initial_order,
                    program=program,
                    halt_after=row_depth,
                    query_position=row_query,
                    storage_order=row_storage,
                    template=template,
                )
            )
    return rows


def build_all(
    *,
    train_rows: int,
    development_families: int,
    confirmation_families: int,
    seed: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    used_sequences: set[str] = set()
    train = build_train(
        train_rows, seed ^ _SPLIT_SEED_MASKS[TRAIN_SPLIT], used_sequences
    )
    development = build_paired_split(
        DEVELOPMENT_SPLIT,
        development_families,
        seed ^ _SPLIT_SEED_MASKS[DEVELOPMENT_SPLIT],
        used_sequences,
    )
    confirmation = build_paired_split(
        CONFIRMATION_SPLIT,
        confirmation_families,
        seed ^ _SPLIT_SEED_MASKS[CONFIRMATION_SPLIT],
        used_sequences,
    )
    return train, development, confirmation


def _jsonl(rows: Iterable[dict[str, object]]) -> bytes:
    return b"".join(
        (json.dumps(row, sort_keys=True) + "\n").encode("utf-8") for row in rows
    )


def _write_board(
    out_dir: Path,
    train: list[dict[str, object]],
    development: list[dict[str, object]],
    confirmation: list[dict[str, object]],
    report: dict[str, object],
) -> None:
    paths = {
        "train.jsonl": _jsonl(train),
        "development.jsonl": _jsonl(development),
        "confirmation.sealed.jsonl": _jsonl(confirmation),
    }
    report["files"] = {
        name: {"bytes": len(payload), "sha256": sha256_bytes(payload)}
        for name, payload in paths.items()
    }
    out_dir.mkdir(parents=True)
    for name, payload in paths.items():
        path = out_dir / name
        path.write_bytes(payload)
        if name == "confirmation.sealed.jsonl":
            path.chmod(0o600)
    (out_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )


def _verify_source_commit(source_commit: str) -> str:
    if re.fullmatch(r"[0-9a-f]{40}", source_commit) is None:
        raise ValueError("source commit must be a full lowercase Git SHA")
    root = Path(__file__).resolve().parents[1]
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if head != source_commit:
        raise ValueError(f"source commit {source_commit} is not current HEAD {head}")
    clean = subprocess.run(
        ["git", "diff", "--quiet", source_commit, "--"],
        cwd=root,
        check=False,
    )
    if clean.returncode != 0:
        raise ValueError("tracked worktree differs from frozen source commit")
    return source_commit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--train-rows", type=int, default=48000)
    parser.add_argument("--development-families", type=int, default=288)
    parser.add_argument("--confirmation-families", type=int, default=288)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--source-commit")
    source.add_argument("--test-only-unfrozen-source", action="store_true")
    args = parser.parse_args()
    if args.out_dir.exists():
        raise SystemExit(f"refusing existing SD-CST output: {args.out_dir}")
    train, development, confirmation = build_all(
        train_rows=args.train_rows,
        development_families=args.development_families,
        confirmation_families=args.confirmation_families,
        seed=args.seed,
    )
    report = audit_board(train, development, confirmation)
    report.update(
        {
            "schema": "r12_sd_cst_board_report_v1_1",
            "seed": args.seed,
            "confirmation_accesses": 0,
            "source_commit": (
                "UNFROZEN_TEST_ONLY"
                if args.test_only_unfrozen_source
                else _verify_source_commit(str(args.source_commit))
            ),
            "source_custody": (
                "test_only_unfrozen"
                if args.test_only_unfrozen_source
                else "clean_head_verified"
            ),
            "development_registration": {
                "protocol": "r12_sd_cst_v1_1",
                "row_count": len(development),
                "family_count": len(development) // len(SURFACE_TYPES),
                "family_size": len(SURFACE_TYPES),
                "row_ids_sha256": row_ids_sha256(development),
                "depth_counts": dict(sorted(Counter(
                    int(row["compiler_targets"]["halt_after"])
                    for row in development
                ).items())),
                "variants": list(SURFACE_TYPES),
            },
            "claim_boundary": (
                "Board mechanics only. Training labels are compiler fields; no "
                "neural or native-reasoning claim is implied."
            ),
        }
    )
    if not report["all_gates_pass"]:
        raise SystemExit(
            "SD-CST board rejected before write: "
            + canonical_json(report.get("violations", []))
        )
    _write_board(args.out_dir, train, development, confirmation, report)
    print(
        json.dumps(
            {
                "all_gates_pass": report["all_gates_pass"],
                "out_dir": str(args.out_dir),
                "rows": report["row_counts"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
