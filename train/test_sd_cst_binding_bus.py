from __future__ import annotations

import torch

from pilot_sd_cst_binding_bus import parse_binding_row
from pilot_sd_cst_byte_addressed import (
    BASE_PARAMETERS,
    MAX_PARAMETERS,
    MOTOR_PARAMETERS,
    READER_PARAMETERS,
)
from pilot_sd_cst_hierarchical_binding import (
    PROJECTED_TRAINABLE_NAMES,
    TRAINABLE_NAMES,
    freeze_parent,
    load_parent_state,
)
from sd_cst_binding_bus import (
    BIGRAM_PAD,
    BindingBusCompiler,
    HierarchicalBindingBusCompiler,
    ProjectedHierarchicalBindingBusCompiler,
)
from sd_cst_byte_addressed import ByteAddressedCompiler


def _training_row() -> dict[str, object]:
    storage = [4, 1, 8, 3, 6, 2, 7, 5]
    names = ["zelda", "quorin", "mavik"]
    initial_roles = [2, 0, 1]
    lines = [
        "entities zelda, quorin, mavik; initial mavik, zelda, quorin\n",
    ]
    slots = []
    for ordinal in range(1, 9):
        role = ordinal % 3
        kind = 2 if ordinal == 8 else ordinal % 2
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
        "id": "binding-test-row",
        "split": "sd_cst_train",
        "program_text": "".join([lines[0]] + [lines[index] for index in storage]),
        "late_query_text": "where is zelda?",
        "late_query_target": {"position": 1},
        "compiler_targets": {
            "initial_state_id": 4,
            "initial_order_roles": initial_roles,
            "storage_order": storage,
            "entity_bindings": [
                {"entity_role": role, "entity": name}
                for role, name in enumerate(names)
            ],
            "event_slots": slots,
        },
    }


def test_binding_parser_locates_shared_entity_occurrences():
    parsed = parse_binding_row(_training_row())
    source = bytes(parsed.program_bytes)
    expected = [b"zelda", b"quorin", b"mavik"]
    assert [source[slice(*span)] for span in parsed.binding_ranges] == expected
    assert [source[slice(*span)] for span in parsed.initial_entity_ranges] == [
        b"mavik", b"zelda", b"quorin",
    ]
    for slot, span in enumerate(parsed.event_entity_ranges):
        if parsed.event_kind[slot] == 2:
            assert span == (0, 0)
        else:
            assert source[slice(*span)] == expected[parsed.event_identity[slot]]


def test_shared_bigram_fingerprints_match_identical_content():
    model = BindingBusCompiler(
        width=32,
        heads=4,
        encoder_layers=1,
        slot_layers=1,
        ff=64,
        slot_ff=64,
        max_bytes=64,
        fingerprint_width=16,
    )
    source = list(b"abc xyz?abc!qqq")
    ids = torch.tensor([source + [256] * (24 - len(source))])
    valid = ids.ne(256)
    logits = torch.full((1, 3, ids.shape[1]), -100.0)
    logits[0, 0, 0:3] = 0.0
    logits[0, 1, 8:11] = 0.0
    logits[0, 2, 4:7] = 0.0
    fingerprints = model._fingerprints(ids, valid, logits)
    repeated = torch.dot(fingerprints[0, 0], fingerprints[0, 1])
    different = torch.dot(fingerprints[0, 0], fingerprints[0, 2])
    assert torch.isclose(repeated, torch.tensor(1.0), atol=1e-6)
    assert repeated > different
    assert model._bigram_ids(ids, valid)[0, -1] == BIGRAM_PAD


def test_binding_bus_shapes_and_production_parameter_cap():
    model = BindingBusCompiler(
        width=32,
        heads=4,
        encoder_layers=1,
        slot_layers=1,
        ff=64,
        slot_ff=64,
        max_bytes=64,
        fingerprint_width=16,
    )
    rows = [list(b"names a b c; initial c a b\nmove a\n") for _ in range(2)]
    width = max(map(len, rows)) + 4
    ids = torch.tensor([row + [256] * (width - len(row)) for row in rows])
    valid = ids.ne(256)
    output = model.compile_program(ids, valid)
    assert output.tape.initial_state.shape == (2, 6)
    assert output.tape.event_kind.shape == (2, 8, 3)
    assert output.tape.event_identity.shape == (2, 8, 3)
    assert output.binding_pointer_logits.shape == (2, 3, width)
    assert output.initial_entity_pointer_logits.shape == (2, 3, width)
    assert output.event_entity_pointer_logits.shape == (2, 8, width)

    production = BindingBusCompiler()
    complete = (
        BASE_PARAMETERS
        + production.parameter_count()
        + MOTOR_PARAMETERS
        + READER_PARAMETERS
    )
    assert complete < MAX_PARAMETERS


def test_hierarchical_mask_uses_model_anchor_and_newline_syntax():
    ids = torch.tensor([list(b"alpha one\nbeta two\ngamma") + [256] * 4])
    valid = ids.ne(256)
    logits = torch.full((1, 3, ids.shape[1]), -100.0)
    logits[0, 0, 2] = 1.0
    logits[0, 1, 14] = 1.0
    logits[0, 2, 18] = 1.0
    mask = HierarchicalBindingBusCompiler._selected_line_mask(ids, valid, logits)
    source = bytes(value for value in ids[0].tolist() if value < 256)
    selected = [
        bytes(ids[0, slot_mask].tolist()) for slot_mask in mask[0]
    ]
    assert selected == [b"alpha one\n", b"beta two\n", b"beta two\n"]
    assert source == b"alpha one\nbeta two\ngamma"


def test_frozen_parent_load_exposes_only_binding_bus_parameters():
    kwargs = {
        "width": 32,
        "heads": 4,
        "encoder_layers": 1,
        "slot_layers": 1,
        "ff": 64,
        "slot_ff": 64,
        "max_bytes": 64,
    }
    parent = ByteAddressedCompiler(**kwargs)
    child = HierarchicalBindingBusCompiler(fingerprint_width=16, **kwargs)
    missing = load_parent_state(child, parent.state_dict())
    trainable = freeze_parent(child)
    assert "binding_queries" in missing
    assert set(trainable) == TRAINABLE_NAMES
    for name, parameter in child.named_parameters():
        assert parameter.requires_grad == (name in TRAINABLE_NAMES)


def test_dedicated_projection_is_the_only_additional_trainable_path():
    kwargs = {
        "width": 32,
        "heads": 4,
        "encoder_layers": 1,
        "slot_layers": 1,
        "ff": 64,
        "slot_ff": 64,
        "max_bytes": 64,
    }
    parent = ByteAddressedCompiler(**kwargs)
    child = ProjectedHierarchicalBindingBusCompiler(
        fingerprint_width=16, **kwargs,
    )
    load_parent_state(child, parent.state_dict())
    trainable = freeze_parent(child, PROJECTED_TRAINABLE_NAMES)
    assert set(trainable) == PROJECTED_TRAINABLE_NAMES
    assert child.parameter_count() > parent.parameter_count()
