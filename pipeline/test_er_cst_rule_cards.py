from __future__ import annotations

import pytest

from er_cst_rule_cards import (
    PERMUTATIONS,
    apply_position_permutation,
    execute_rule_program,
    infer_position_permutation,
    run_falsifier,
)


def test_every_permutation_is_identified_from_a_determining_witness() -> None:
    before = ("a", "b", "c")
    for permutation in PERMUTATIONS:
        after = tuple(before[index] for index in permutation)
        assert infer_position_permutation(before, after) == permutation


def test_malformed_or_ambiguous_witnesses_are_rejected() -> None:
    with pytest.raises(ValueError, match="distinct"):
        infer_position_permutation(("a", "a", "b"), ("a", "b", "a"))
    with pytest.raises(ValueError, match="permutation"):
        infer_position_permutation(("a", "b", "c"), ("a", "b", "d"))


def test_rule_program_uses_cards_in_order_and_halts_persistently() -> None:
    cards = {"turn": (1, 2, 0), "swap": (1, 0, 2)}
    final, trajectory = execute_rule_program(
        ("a", "b", "c"), cards, ("turn", "swap", "turn"), 2
    )
    assert trajectory == (
        ("a", "b", "c"),
        ("b", "c", "a"),
        ("c", "b", "a"),
    )
    assert final == ("c", "b", "a")
    assert execute_rule_program(
        ("a", "b", "c"), cards, ("turn", "swap", "turn", "turn"), 2
    ) == (final, trajectory)


def test_invalid_cards_programs_and_halt_are_rejected() -> None:
    with pytest.raises(ValueError, match="permutation"):
        apply_position_permutation(("a", "b", "c"), (0, 0, 1))
    with pytest.raises(ValueError, match="unknown"):
        execute_rule_program(("a", "b", "c"), {}, ("missing",), 1)
    with pytest.raises(ValueError, match="HALT"):
        execute_rule_program(("a", "b", "c"), {}, (), 1)


def test_cpu_falsifier_is_deterministic_and_passes() -> None:
    first = run_falsifier(seed=1729, trials=1_000)
    second = run_falsifier(seed=1729, trials=1_000)
    assert first == second
    assert first["all_gates_pass"] is True
    assert first["counts"]["execution_exact"] == 1_000
    assert first["rates"]["deranged_card_state_exact"] < 0.40
