from __future__ import annotations

import torch

from pilot_sd_cst_byte_addressed import (
    BASE_PARAMETERS,
    MAX_PARAMETERS,
    MOTOR_PARAMETERS,
    READER_PARAMETERS,
    parse_train_row,
)
from sd_cst_byte_addressed import ByteAddressedCompiler


def test_byte_addressed_compiler_shapes_and_parameter_cap():
    model = ByteAddressedCompiler(
        width=32, heads=4, encoder_layers=1, slot_layers=1,
        ff=64, slot_ff=64, max_bytes=64,
    )
    rows = [list(b"binding line\nevent one\n"), list(b"binding two\nevent two\n")]
    width = max(map(len, rows)) + 8
    ids = torch.tensor([row + [256] * (width - len(row)) for row in rows])
    valid = ids.ne(256)
    output = model.compile_program(ids, valid)
    query = model.compile_query(ids[:, :16], valid[:, :16])
    assert output.tape.initial_state.shape == (2, 6)
    assert output.tape.event_kind.shape == (2, 8, 3)
    assert output.pointer_logits.shape == (2, 9, ids.shape[1])
    assert query.logits.shape == (2, 3)
    production = ByteAddressedCompiler()
    assert (
        BASE_PARAMETERS + production.parameter_count()
        + MOTOR_PARAMETERS + READER_PARAMETERS
    ) < MAX_PARAMETERS


def test_training_pointer_ranges_follow_semantic_ordinals_not_storage_order():
    storage = [6, 8, 1, 3, 4, 2, 7, 5]
    lines = ["bindings and initial\n"] + [
        f"event {ordinal} payload\n" for ordinal in storage
    ]
    row = {
        "id": "pilot-row",
        "split": "sd_cst_train",
        "program_text": "".join(lines),
        "late_query_text": "which position?",
        "late_query_target": {"position": 1},
        "compiler_targets": {
            "initial_state_id": 0,
            "storage_order": storage,
            "event_slots": [
                {
                    "semantic_ordinal": ordinal,
                    "kind_id": 2 if ordinal == 2 else ordinal % 2,
                    "entity_role": ordinal % 3,
                    "amount_id": ordinal % 2,
                }
                for ordinal in range(1, 9)
            ],
        },
    }
    parsed = parse_train_row(row)
    source = bytes(parsed.program_bytes).decode("utf-8")
    assert source[slice(*parsed.pointer_ranges[0])] == lines[0]
    for ordinal in range(1, 9):
        assert f"event {ordinal} payload" in source[
            slice(*parsed.pointer_ranges[ordinal])
        ]
