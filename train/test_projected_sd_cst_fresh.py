from __future__ import annotations

import itertools

from build_sd_cst_board import build_all
from projected_sd_cst_fresh import (
    PARENT_DERIVED_BUFFER_NAMES,
    PERMUTATIONS,
    PROJECTED_TRAINABLE_NAMES,
    as_binding_row,
    exact_one_stop_map,
    parse_projected_row,
    permute_training_labels,
    row_shuffled_permutation,
    validate_parent_missing_names,
)
from pilot_sd_cst_hierarchical_binding import freeze_parent, load_parent_state
from pilot_sd_cst_byte_addressed import byte_batch
from sd_cst import HardProgramTape
from sd_cst_byte_addressed import ByteAddressedCompiler
from sd_cst_binding_bus import ProjectedHierarchicalBindingBusCompiler
from train_eval_sd_cst_projected_fresh import (
    safe_mutate_first_active,
    safe_perturb_post_stop,
)
import pytest
import torch


def test_exact_one_stop_map_matches_exhaustive_constrained_argmax():
    generator = torch.Generator().manual_seed(44192)
    logits = torch.randn(5, 8, 3, generator=generator)
    decoded = exact_one_stop_map(logits)
    assert decoded.eq(2).sum(-1).eq(1).all()
    for row in range(len(logits)):
        best_score = None
        best_candidate = None
        for stop in range(8):
            active = [slot for slot in range(8) if slot != stop]
            for choices in itertools.product(range(2), repeat=7):
                candidate = torch.zeros(8, dtype=torch.long)
                candidate[stop] = 2
                candidate[active] = torch.tensor(choices)
                score = logits[row, torch.arange(8), candidate].sum()
                if best_score is None or bool(score > best_score):
                    best_score = score
                    best_candidate = candidate
        assert torch.equal(decoded[row].long(), best_candidate)


def test_exact_one_stop_map_preserves_an_already_valid_independent_argmax():
    logits = torch.full((2, 8, 3), -4.0)
    expected = torch.tensor([[0, 1, 0, 2, 1, 0, 1, 0], [1, 0, 2, 1, 0, 1, 0, 1]])
    logits.scatter_(2, expected[:, :, None], 4.0)
    assert torch.equal(exact_one_stop_map(logits).long(), expected)


def test_exact_one_stop_map_has_deterministic_ties_and_rejects_nonfinite():
    decoded = exact_one_stop_map(torch.zeros(3, 8, 3))
    assert decoded[:, 0].eq(2).all()
    assert decoded[:, 1:].eq(0).all()
    logits = torch.zeros(1, 8, 3)
    logits[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        exact_one_stop_map(logits)


def test_parent_contract_separates_derived_buffer_from_trainable_parameters():
    parent = ByteAddressedCompiler()
    model = ProjectedHierarchicalBindingBusCompiler()
    missing = set(load_parent_state(model, parent.state_dict()))
    trainable = set(freeze_parent(model, PROJECTED_TRAINABLE_NAMES))
    assert missing == set(PROJECTED_TRAINABLE_NAMES) | set(PARENT_DERIVED_BUFFER_NAMES)
    assert trainable == set(PROJECTED_TRAINABLE_NAMES)
    assert PARENT_DERIVED_BUFFER_NAMES.isdisjoint(trainable)
    assert torch.equal(
        model.permutations,
        torch.tensor(PERMUTATIONS, dtype=torch.long),
    )
    validate_parent_missing_names(tuple(sorted(missing)))


def test_parent_contract_rejects_any_unregistered_missing_state():
    missing = tuple(
        sorted(
            set(PROJECTED_TRAINABLE_NAMES)
            | set(PARENT_DERIVED_BUFFER_NAMES)
            | {"unregistered_buffer"}
        )
    )
    try:
        validate_parent_missing_names(missing)
    except ValueError as error:
        assert "derived-buffer contract" in str(error)
    else:
        raise AssertionError("unregistered parent state unexpectedly accepted")


def test_parser_handles_direct_paraphrase_and_storage_order():
    train, development, _ = build_all(
        train_rows=12,
        development_families=6,
        confirmation_families=6,
        seed=4412,
    )
    direct = parse_projected_row(train[0], "sd_cst_train")
    paraphrase_raw = next(row for row in development if row["variant"] == "paraphrase")
    paraphrase = parse_projected_row(paraphrase_raw, "sd_cst_development")
    assert len(direct.pointer_ranges) == 9
    assert len(paraphrase.pointer_ranges) == 9
    assert all(end > start for start, end in direct.binding_ranges)
    assert all(end > start for start, end in paraphrase.binding_ranges)
    assert sum(end > start for start, end in paraphrase.event_entity_ranges) == 7
    assert paraphrase.final_state is not None
    assert paraphrase.answer_role is not None
    assert paraphrase.active_state_trajectory is not None


def test_label_permutation_changes_semantics_but_not_source_or_occurrences():
    train, _, _ = build_all(
        train_rows=12,
        development_families=6,
        confirmation_families=6,
        seed=88173,
    )
    row = parse_projected_row(train[0], "sd_cst_train")
    permutation = (1, 2, 0)
    control = permute_training_labels(row, permutation)
    assert control.program_bytes == row.program_bytes
    assert control.query_bytes == row.query_bytes
    assert control.pointer_ranges == row.pointer_ranges
    assert control.initial_entity_ranges == row.initial_entity_ranges
    assert control.event_entity_ranges == row.event_entity_ranges
    assert control.binding_ranges != row.binding_ranges
    assert PERMUTATIONS[control.initial_state] == tuple(
        permutation[role] for role in PERMUTATIONS[row.initial_state]
    )
    assert control.event_identity == tuple(
        permutation[role] for role in row.event_identity
    )


def test_row_shuffled_labels_are_deterministic_and_not_one_global_relabeling():
    train, _, _ = build_all(
        train_rows=96,
        development_families=6,
        confirmation_families=6,
        seed=7219,
    )
    rows = [parse_projected_row(value, "sd_cst_train") for value in train]
    first = [row_shuffled_permutation(991, row.row_id) for row in rows]
    second = [row_shuffled_permutation(991, row.row_id) for row in rows]
    assert first == second
    assert len(set(first)) > 1
    assert all(sorted(value) == [0, 1, 2] for value in first)
    assert all(len(row.raw_row_sha256) == 64 for row in rows)


def test_binding_source_free_ablation_preserves_parent_fields():
    train, _, _ = build_all(
        train_rows=12,
        development_families=6,
        confirmation_families=6,
        seed=44119,
    )
    rows = [parse_projected_row(value, "sd_cst_train") for value in train[:4]]
    batch = [as_binding_row(row) for row in rows]
    ids, valid = byte_batch(batch, "program_bytes", torch.device("cpu"))
    model = ProjectedHierarchicalBindingBusCompiler().eval()
    with torch.no_grad():
        normal = model.compile_program(ids, valid)
        ablated = model.compile_program_source_free_binding(ids, valid)
        normal_again = model.compile_program(ids, valid)
    assert torch.equal(normal.tape.event_kind, normal_again.tape.event_kind)
    assert torch.equal(normal.tape.amount, normal_again.tape.amount)
    assert torch.equal(normal.tape.event_kind, ablated.tape.event_kind)
    assert torch.equal(normal.tape.amount, ablated.tape.amount)
    assert torch.equal(normal.line_pointer_logits, ablated.line_pointer_logits)
    assert not bool(ablated.tape.initial_state.any())
    assert not bool(ablated.tape.event_identity.any())


def test_control_mutations_preserve_the_predicted_stop_grammar():
    kinds = torch.zeros((1, 8), dtype=torch.uint8)
    kinds[0, 2] = 2
    tape = HardProgramTape(
        torch.tensor([0], dtype=torch.uint8),
        kinds,
        torch.zeros((1, 8), dtype=torch.uint8),
        torch.zeros((1, 8), dtype=torch.uint8),
    )
    post = safe_perturb_post_stop(tape)
    assert int(post.event_identity[0, 3]) == 1
    mutated = safe_mutate_first_active(tape, "identity")
    assert int(mutated.event_identity[0, 0]) == 1
    assert int(mutated.event_kind.eq(2).sum()) == 1
