from __future__ import annotations

import ast
from pathlib import Path

import torch

from assess_sd_cst_projected_mechanics import (
    alpha_renamed_row,
    declaration_role_swap_row,
    event_counterfactual_row,
    expected_tape,
    relocated_event_lines_row,
    rotate_queries,
    semantic_rollout,
)
from pilot_sd_cst_binding_bus import parse_binding_row
from sd_cst import HardLateQuery


def _row() -> dict[str, object]:
    names = ["aaaa-bbbbbbbb", "cccc-dddddddd", "eeee-ffffffff"]
    storage = [4, 1, 8, 3, 6, 2, 7, 5]
    lines = [
        f"entities {names[0]}, {names[1]}, {names[2]}; "
        f"initial {names[2]}, {names[0]}, {names[1]}\n",
    ]
    slots = []
    for ordinal in range(1, 9):
        role = ordinal % 3
        kind = 2 if ordinal == 6 else ordinal % 2
        lines.append(
            f"step {ordinal}: {'stop' if kind == 2 else 'move'} "
            f"{names[role]} by {ordinal % 2}\n"
        )
        slots.append({
            "semantic_ordinal": ordinal,
            "kind_id": kind,
            "entity_role": role,
            "amount_id": ordinal % 2,
        })
    return {
        "id": "projected-mechanics-test-row",
        "split": "sd_cst_train",
        "program_text": "".join([lines[0]] + [lines[index] for index in storage]),
        "late_query_text": "which role is at position one?",
        "late_query_target": {"position": 1},
        "compiler_targets": {
            "initial_state_id": 4,
            "initial_order_roles": [2, 0, 1],
            "storage_order": storage,
            "entity_bindings": [
                {"entity_role": role, "entity": name}
                for role, name in enumerate(names)
            ],
            "event_slots": slots,
        },
    }


def test_source_level_controls_have_declared_semantics_and_fixed_width():
    row = parse_binding_row(_row())
    alpha = alpha_renamed_row(row)
    assert len(alpha.program_bytes) == len(row.program_bytes)
    assert alpha.initial_state == row.initial_state
    assert alpha.event_identity == row.event_identity
    assert bytes(alpha.program_bytes) != bytes(row.program_bytes)

    counterfactual = event_counterfactual_row(row)
    assert len(counterfactual.program_bytes) == len(row.program_bytes)
    changed = [
        index for index, (left, right) in enumerate(zip(
            row.event_identity, counterfactual.event_identity, strict=True,
        )) if left != right
    ]
    assert changed == [0]

    declaration = declaration_role_swap_row(row)
    assert len(declaration.program_bytes) == len(row.program_bytes)
    assert declaration.event_identity == tuple(
        {0: 1, 1: 0, 2: 2}[value] for value in row.event_identity
    )
    assert declaration.initial_state != row.initial_state

    relocated = relocated_event_lines_row(row)
    assert len(relocated.program_bytes) == len(row.program_bytes)
    assert bytes(relocated.program_bytes).splitlines()[0] == bytes(row.program_bytes).splitlines()[0]
    assert bytes(relocated.program_bytes).splitlines()[1:] == list(
        reversed(bytes(row.program_bytes).splitlines()[1:])
    )


def test_semantic_rollout_respects_halt_and_late_query_rotation():
    row = parse_binding_row(_row())
    tape = expected_tape([row])
    query = HardLateQuery(torch.tensor([row.query_position], dtype=torch.uint8))
    state, answer, states, alive = semantic_rollout(tape, query)
    assert state.shape == (1,)
    assert answer.shape == (1,)
    assert states.shape == (1, 8)
    assert alive.shape == (1, 8)
    stop = row.event_kind.index(2)
    assert not bool(alive[0, stop])
    assert torch.equal(states[0, stop:], states[0, stop].expand(8 - stop))
    rotated = rotate_queries(query)
    rotated_state, rotated_answer, _, _ = semantic_rollout(tape, rotated)
    assert torch.equal(rotated_state, state)
    assert not torch.equal(rotated_answer, answer)


def test_source_blind_executor_does_not_import_compiler_or_board_modules():
    source = Path(__file__).with_name("run_sd_cst_hard_packets.py").read_text()
    tree = ast.parse(source)
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported.add(node.module)
    assert not any(
        name.startswith(("pilot_", "sd_cst_binding_bus", "train_sd_cst"))
        for name in imported
    )
    assert "program_text" not in source
    assert "row_id" not in source
    assert "target" not in source
