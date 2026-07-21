from __future__ import annotations

from collections import Counter
from pathlib import Path

from pipeline.ctaa_board_v2 import (
    FACTORIAL_BITS,
    INITIAL_STATES,
    PROGRAM_CLASSES,
    FactorialCell,
    balanced_renderer_index,
    board_contract_counts,
    build_compiler_families,
    build_long_families,
    factorial_cells,
    iter_atomic_exposures,
    iter_closure_exposures,
    make_card_reindex_twin,
    make_equivalent_composite_twin,
    make_family_v2,
    make_order_contrast_twin,
    make_post_stop_poison_twin,
    make_prefix_contrast_twin,
    make_stop_relocation_twin,
    render_family_v2,
    train_closed_pairs,
)
from pipeline.ctaa_name_pool import build_name_pools


SEED = 941_711
TOKENIZER = Path(__file__).resolve().parents[1] / "artifacts/tokenizer/tokenizer.json"


def test_typed_finite_exposures_have_exact_unique_and_optimization_counts() -> None:
    counts = board_contract_counts()
    assert counts == {
        "atomic_optimization_exposures": 15_552,
        "atomic_unique_finite_cases": 243,
        "closure_optimization_exposures": 60_480,
        "closure_unique_finite_cases": 945,
        "compiler_schedule_rows": 32_768,
        "long_scored_families_per_partition": 27_648,
    }
    assert len(train_closed_pairs()) == 35
    assert len(tuple(iter_atomic_exposures("train", contexts=1))) == 243
    assert len(tuple(iter_closure_exposures(contexts=1))) == 945


def test_compiler_family_builder_covers_depth_one_without_claiming_implicit_collapse() -> None:
    families = build_compiler_families(SEED, per_depth=18)
    assert len(families) == 18 * 8
    assert len({family.canonical_key for family in families}) == len(families)
    depths = Counter(family.depth for family in families)
    assert depths == {depth: 18 for depth in range(1, 9)}
    depth_one = [family for family in families if family.depth == 1]
    assert {family.program_class for family in depth_one} == {
        "stable_rank_two",
        "explicit_final_collapse",
    }
    assert all(family.program_class != "implicit_final_collapse" for family in depth_one)


def test_factorial_cells_separate_semantic_renderer_and_lexical_axes() -> None:
    cells = factorial_cells("development")
    assert len(cells) == len(FACTORIAL_BITS) == 8
    assert len({(cell.semantic_axis, cell.renderer_axis, cell.lexical_axis) for cell in cells}) == 8
    assert FactorialCell("train", "train", "train") in cells
    assert FactorialCell("development", "development", "development") in cells


def test_new_program_classes_are_varied_and_report_distinct_causal_metrics() -> None:
    cell = FactorialCell("development", "development", "development")
    for class_index, program_class in enumerate(PROGRAM_CLASSES):
        family = make_family_v2(
            SEED,
            "development",
            cell,
            class_index,
            program_class=program_class,
            depth=32,
            query_position=class_index,
            initial_state=INITIAL_STATES[class_index],
        )
        assert len(set(family.active)) >= 3
        assert family.max_run_length <= 3
        assert family.normalized_event_entropy >= 0.75
        assert family.map_deletion_depth >= 8
        assert family.state_deletion_depth == family.map_deletion_depth
        assert family.answer_deletion_depth <= family.state_deletion_depth
        assert family.shortest_equivalent_length < family.depth
        target_rank = 2 if program_class == "stable_rank_two" else 1
        assert len(set(family.composite)) == target_rank
        assert family.execute()[family.depth + 1 :] == (family.terminal_state,) * (41 - family.depth)


def test_small_balanced_long_build_is_unique_and_crosses_query_initial() -> None:
    families = build_long_families(SEED, "development", per_class_depth_cell=288)
    assert len(families) == 8 * 3 * 2 * 288
    assert len({family.canonical_key for family in families}) == len(families)
    counts = Counter(
        (
            family.cell.tag,
            family.program_class,
            family.depth,
            family.query_position,
            family.initial_state,
        )
        for family in families
    )
    assert set(counts.values()) == {16}


def test_renderer_is_jointly_crossed_with_every_query_initial_cell() -> None:
    families = build_long_families(SEED, "development", per_class_depth_cell=288)
    first_stratum = families[:288]
    counts = Counter(
        (
            balanced_renderer_index(index, 288),
            family.query_position,
            family.initial_state,
        )
        for index, family in enumerate(first_stratum)
    )
    assert len(counts) == 16 * 18
    assert set(counts.values()) == {1}


def test_factorial_surface_uses_axis_specific_names_without_changing_packet() -> None:
    pools = build_name_pools(TOKENIZER, per_split=32)
    cell = FactorialCell("development", "train", "confirmation")
    family = make_family_v2(
        SEED,
        "development",
        cell,
        9,
        program_class="stable_rank_two",
        depth=16,
        query_position=2,
        initial_state=INITIAL_STATES[5],
    )
    first = render_family_v2(SEED, family, pools, renderer_index=0)
    second = render_family_v2(SEED, family, pools, renderer_index=1)
    reordered = render_family_v2(
        SEED,
        family,
        pools,
        renderer_index=0,
        reverse_rule_storage=True,
    )
    assert first.program_source != second.program_source
    assert first.program_source != reordered.program_source
    assert first.family.canonical_key == second.family.canonical_key == reordered.family.canonical_key
    assert all(name not in first.program_source for name in pools["development"])
    assert any(name in first.program_source for name in pools["confirmation"])


def test_training_and_scored_records_enforce_outcome_custody() -> None:
    pools = build_name_pools(TOKENIZER, per_split=16)
    train_cell = FactorialCell("train", "train", "train")
    train_family = make_family_v2(
        SEED,
        "train",
        train_cell,
        3,
        program_class="explicit_final_collapse",
        depth=8,
        query_position=0,
        initial_state=INITIAL_STATES[0],
    )
    train_row = render_family_v2(SEED, train_family, pools, renderer_index=0)
    record = train_row.compiler_record()
    assert "prefix_states" not in record and "terminal_state" not in record and "answer" not in record

    scored_cell = FactorialCell("confirmation", "confirmation", "confirmation")
    scored_family = make_family_v2(
        SEED,
        "confirmation",
        scored_cell,
        4,
        program_class="implicit_final_collapse",
        depth=16,
        query_position=1,
        initial_state=INITIAL_STATES[1],
    )
    scored = render_family_v2(SEED, scored_family, pools, renderer_index=0).scored_record()
    assert len(scored["prefix_states"]) == 42
    assert scored["answer"] == scored["terminal_state"][1]


def test_paired_twins_have_preregistered_semantic_relations() -> None:
    cell = FactorialCell("development", "development", "development")
    family = make_family_v2(
        SEED,
        "development",
        cell,
        77,
        program_class="stable_rank_two",
        depth=32,
        query_position=1,
        initial_state=INITIAL_STATES[2],
    )
    order = make_order_contrast_twin(family)
    assert order.child.composite != family.composite
    equivalent = make_equivalent_composite_twin(family, seed=SEED)
    assert equivalent.child.active != family.active
    assert equivalent.child.composite == family.composite
    assert equivalent.child.terminal_state == family.terminal_state
    assert equivalent.child.execute() != family.execute()
    prefix = make_prefix_contrast_twin(family)
    boundary = family.depth // 2
    assert prefix.child.execute()[: boundary + 1] == family.execute()[: boundary + 1]
    assert prefix.child.composite != family.composite
    reindex = make_card_reindex_twin(family)
    assert reindex.child.action_cards != family.action_cards
    assert reindex.child.execute() == family.execute()
    poison = make_post_stop_poison_twin(family)
    assert poison.child.schedule != family.schedule
    assert poison.child.execute() == family.execute()
    relocated = make_stop_relocation_twin(family)
    assert relocated.child.depth < family.depth
    assert relocated.child.terminal_state != family.terminal_state
