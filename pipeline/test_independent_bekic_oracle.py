from __future__ import annotations

import contrastive_bekic_program_orbits as board

from independent_bekic_oracle import (
    evaluate_nested_independently,
    evaluate_simultaneous_independently,
)


def test_independent_oracles_agree_across_every_development_cell() -> None:
    for index in range(15):
        row = board.generate_orbit(
            split="development",
            seed=2026072321,
            index=index,
        )
        for arm in ("p", "p_prime", "p_eq"):
            simultaneous = board.select_machine_input(
                row,
                arm=arm,
                form="simultaneous",
            )
            nested = board.select_machine_input(
                row,
                arm=arm,
                form="nested",
            )
            expected = board.evaluate_simultaneous(simultaneous)
            assert evaluate_simultaneous_independently(simultaneous) == expected
            assert evaluate_nested_independently(nested) == expected


def test_independent_oracle_detects_mutated_shared_compose(
    monkeypatch,
) -> None:
    packet = {
        "schema": "contrastive_bekic_machine_input_v1",
        "cardinality": 3,
        "constants": [
            {
                "id": "a",
                "relation": [[0, 1, 0], [0, 0, 0], [0, 0, 0]],
            },
            {
                "id": "b",
                "relation": [[0, 0, 0], [0, 0, 1], [0, 0, 0]],
            },
            *[
                {
                    "id": f"unused_{index}",
                    "relation": [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                }
                for index in range(6)
            ],
        ],
        "program": {
            "schema": "contrastive_monotone_program_v1",
            "form": "simultaneous_lfp",
            "variables": ["x", "y"],
            "equations": [
                {
                    "variable": "x",
                    "expression": {
                        "id": "compose",
                        "kind": "COMPOSE",
                        "children": [
                            {"id": "left", "kind": "CONSTANT", "constant": "a"},
                            {
                                "id": "right",
                                "kind": "CONSTANT",
                                "constant": "b",
                            },
                        ],
                    },
                },
                {
                    "variable": "y",
                    "expression": {"id": "identity", "kind": "IDENTITY"},
                },
            ],
        },
    }
    expected = evaluate_simultaneous_independently(packet)
    monkeypatch.setattr(
        board,
        "relation_compose",
        lambda left, _right: left,
    )
    assert board.evaluate_simultaneous(packet) != expected
