from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline"))
sys.path.insert(0, str(ROOT / "train"))

from build_sd_cst_board import build_all  # noqa: E402
from projected_sd_cst_fresh import parse_projected_row  # noqa: E402
from sd_cst_byte_addressed import BYTE_PAD  # noqa: E402
from sd_cst_complete_physical_fresh import (  # noqa: E402
    fresh_trainable_names,
    group_rows,
    permute_family_labels,
)
from sd_cst_complete_physical_fresh_renderers import (  # noqa: E402
    TRAIN_RENDERERS,
    expand_rows,
)
from sd_cst_complete_physical_record_bus_v1_2 import (  # noqa: E402
    CompletePhysicalRecordBusCompilerV1_2,
)


def _rows():
    train, _, _ = build_all(
        train_rows=6,
        development_families=6,
        confirmation_families=6,
        seed=771,
    )
    rendered = expand_rows(train, TRAIN_RENDERERS)
    return [parse_projected_row(row, "sd_cst_train") for row in rendered]


def _small_model() -> CompletePhysicalRecordBusCompilerV1_2:
    return CompletePhysicalRecordBusCompilerV1_2(
        width=32,
        heads=4,
        encoder_layers=1,
        slot_layers=1,
        ff=64,
        slot_ff=64,
        max_bytes=256,
        fingerprint_width=16,
        orbit_width=32,
        orbit_heads=4,
        orbit_layers=1,
        orbit_ff=64,
        native_slot_layers=1,
        native_slot_heads=4,
        native_slot_ff=64,
        record_width=32,
        record_heads=4,
        record_layers=1,
        record_set_layers=1,
        record_ff=64,
        max_line_bytes=32,
        sinkhorn_steps=4,
        occurrence_ff=64,
    )


def _batch(text: bytes) -> tuple[torch.Tensor, torch.Tensor]:
    ids = torch.full((1, len(text)), BYTE_PAD, dtype=torch.long)
    ids[0] = torch.tensor(list(text), dtype=torch.long)
    return ids, torch.ones_like(ids, dtype=torch.bool)


def test_fresh_trainability_excludes_obsolete_queries() -> None:
    model = CompletePhysicalRecordBusCompilerV1_2()
    names = fresh_trainable_names(model)
    assert names
    assert all(not name.startswith("local_declaration_") for name in names)
    assert any(name.startswith("record_line_encoder.") for name in names)
    assert any(name.startswith("local_occurrence_") for name in names)
    assert any(name.startswith("local_query_") for name in names)
    assert len(names) == 102
    assert (
        sum(
            parameter.numel()
            for name, parameter in model.named_parameters()
            if name in names
        )
        == 12_152_855
    )
    assert model.parameter_count() == 67_027_474
    assert 125_081_664 + model.parameter_count() + 19_206 + 835 == 192_129_179


def test_family_grouping_and_control_are_view_consistent() -> None:
    rows = _rows()
    groups = group_rows(rows)
    assert len(groups) == 6
    assert all(len(group) == 4 for group in groups)
    control, digest = permute_family_labels(rows, 918)
    assert len(digest) == 64
    by_family = {}
    for original, changed in zip(rows, control, strict=True):
        value = (
            changed.initial_state,
            changed.event_identity,
        )
        prior = by_family.setdefault(original.family_id, value)
        assert prior == value
        assert json.loads(changed.raw_row_canonical_json) == json.loads(
            original.raw_row_canonical_json
        )
        assert changed.initial_state != original.initial_state


def test_obsolete_declaration_tensors_are_forward_dead_in_v1_2() -> None:
    torch.manual_seed(29)
    model = _small_model().eval()
    program = b"\n".join(f"record {index} zed".encode() for index in range(9))
    ids, valid = _batch(program)
    with torch.no_grad():
        before = model.compile_program(ids, valid)
        obsolete = [
            parameter
            for name, parameter in model.named_parameters()
            if name.startswith("local_declaration_")
        ]
        assert len(obsolete) == 2
        for parameter in obsolete:
            parameter.copy_(torch.randn_like(parameter).mul_(1000))
        after = model.compile_program(ids, valid)

    tensors = (
        "binding_pointer_logits",
        "initial_entity_pointer_logits",
        "event_entity_pointer_logits",
    )
    for name in tensors:
        assert torch.equal(getattr(before, name), getattr(after, name))
    for name in ("initial_state", "event_kind", "event_identity", "amount"):
        assert torch.equal(getattr(before.tape, name), getattr(after.tape, name))
