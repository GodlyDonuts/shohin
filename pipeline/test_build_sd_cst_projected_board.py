from __future__ import annotations

from build_sd_cst_board import build_all
from audit_sd_cst_board import _program_signature
from build_sd_cst_projected_board import projected_audit


def test_projected_audit_binds_names_renderers_and_balances():
    train, development, confirmation = build_all(
        train_rows=96,
        development_families=48,
        confirmation_families=48,
        seed=981273,
    )
    report = projected_audit(train, development, confirmation)
    # The production-only row-count gate intentionally fails on this small unit board.
    assert not report["all_gates_pass"]
    gates = report["projected_gates"]
    assert gates["names_disjoint_across_splits"]
    assert gates["all_opaque_names_fixed_13_bytes"]
    assert gates["no_name_prefix_or_substring_relationships"]
    assert gates["renderer_and_lexical_inventory_hashes_bound"]
    assert not gates["exact_row_counts"]


def test_projected_audit_rejects_cross_split_name_reuse():
    train, development, confirmation = build_all(
        train_rows=96,
        development_families=48,
        confirmation_families=48,
        seed=67531,
    )
    stolen = train[0]["compiler_targets"]["entity_bindings"][0]["entity"]
    original = development[0]["compiler_targets"]["entity_bindings"][0]["entity"]
    development[0]["compiler_targets"]["entity_bindings"][0]["entity"] = stolen
    development[0]["program_text"] = development[0]["program_text"].replace(
        original, stolen
    )
    report = projected_audit(train, development, confirmation)
    assert not report["projected_gates"]["names_disjoint_across_splits"]


def test_fresh_builder_reserves_every_inherited_training_sequence():
    parent_train, _, _ = build_all(
        train_rows=96,
        development_families=6,
        confirmation_families=6,
        seed=19443,
    )
    reserved = {_program_signature(row) for row in parent_train}
    train, development, confirmation = build_all(
        train_rows=96,
        development_families=6,
        confirmation_families=6,
        seed=19444,
        reserved_sequences=reserved,
    )
    fresh = train + development + confirmation
    assert not reserved.intersection(_program_signature(row) for row in fresh)


def test_projected_audit_detects_inherited_parent_leakage():
    train, development, confirmation = build_all(
        train_rows=96,
        development_families=6,
        confirmation_families=6,
        seed=5151,
    )
    report = projected_audit(train, development, confirmation, train)
    assert not report["projected_gates"]["zero_inherited_parent_instance_overlap"]
    assert report["inherited_parent_train_overlap"]["sd_cst_train"]["sequences"]
