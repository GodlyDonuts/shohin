from __future__ import annotations

import itertools

import pytest

from er_relation_tensor_transport import (
    apply_copy_relation,
    compose_relations,
    execute_relation_program,
    infer_copy_relation,
    run_falsifier,
)


def test_every_three_position_function_is_identified() -> None:
    before = ("a", "b", "c")
    for relation in itertools.product(range(3), repeat=3):
        after = tuple(before[index] for index in relation)
        assert infer_copy_relation(before, after) == relation


def test_non_bijective_relation_composes_and_halts() -> None:
    cards = {"copy": (0, 0, 2), "rotate": (1, 2, 0)}
    final, trajectory = execute_relation_program(
        ("a", "b", "c"), cards, ("copy", "rotate", "copy"), 2
    )
    assert trajectory == (
        ("a", "b", "c"),
        ("a", "a", "c"),
        ("a", "c", "a"),
    )
    assert final == ("a", "c", "a")
    composed = compose_relations(cards["rotate"], cards["copy"])
    assert apply_copy_relation(("a", "b", "c"), composed) == final


def test_malformed_relations_are_rejected() -> None:
    with pytest.raises(ValueError, match="distinct"):
        infer_copy_relation(("a", "a", "c"), ("a", "a", "c"))
    with pytest.raises(ValueError, match="unknown"):
        infer_copy_relation(("a", "b", "c"), ("a", "b", "x"))
    with pytest.raises(ValueError, match="invalid"):
        apply_copy_relation(("a", "b", "c"), (0, 1, 3))


def test_relation_falsifier_is_deterministic_and_passes() -> None:
    first = run_falsifier(seed=2718, trials=2_000)
    second = run_falsifier(seed=2718, trials=2_000)
    assert first == second
    assert first["all_gates_pass"] is True
    assert first["counts"]["execution_exact"] == 2_000
    assert first["rates"]["non_bijective_episode"] >= 0.90
