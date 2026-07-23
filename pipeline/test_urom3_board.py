from __future__ import annotations

from pathlib import Path

import torch

from build_urom3_board import build
from general_relational_object_machine import (
    HardDeletedRelationalProgram,
    HardDeletedRelationalQuery,
    rollout_hard_relational_program,
)
from urom3_board import (
    MAX_OBJECTS,
    MAX_RELATION_EDGES,
    MAX_RULES,
    UROMBoardError,
    axis_contract,
    compose_relations,
    generate_rows,
    relation_edges,
    split_contract,
    validate_row,
)


def test_independent_boolean_composition() -> None:
    left = (
        (1, 1, 0),
        (0, 0, 1),
        (1, 0, 0),
    )
    right = (
        (0, 1, 0),
        (0, 0, 1),
        (1, 0, 0),
    )
    assert compose_relations(left, right) == (
        (0, 1, 1),
        (1, 0, 0),
        (0, 1, 0),
    )


def test_split_axes_and_semantics_are_disjoint() -> None:
    generated = {
        split: generate_rows(split=split, count=96, seed=771)
        for split in ("train", "development", "confirmation")
    }
    semantics: dict[str, set[str]] = {}
    for split, rows in generated.items():
        for row in rows:
            validate_row(row)
        contract = split_contract(split)
        assert {row["axis_cell"] for row in rows} == set(
            contract["axis_cells"]
        )
        assert {row["family"] for row in rows} == set(contract["families"])
        assert {row["renderer"] for row in rows} == set(contract["renderers"])
        assert {
            row["compiler_targets"]["cardinality"]  # type: ignore[index]
            for row in rows
        } == set(contract["cardinalities"])
        semantics[split] = {str(row["semantic_sha256"]) for row in rows}
        assert len(semantics[split]) == len(rows)
        for row in rows:
            cell = axis_contract(
                split,
                row["axis_cell"],  # type: ignore[arg-type]
            )
            targets = row["compiler_targets"]
            stop = targets["event_kind"].index(1)  # type: ignore[index]
            assert row["family"] in cell["families"]
            assert row["renderer"] in cell["renderers"]
            assert targets["cardinality"] in cell["cardinalities"]  # type: ignore[index]
            assert sum(targets["rule_active"]) in cell["rule_counts"]  # type: ignore[index]
            assert cell["event_counts"][0] <= stop <= cell["event_counts"][1]  # type: ignore[index]
    assert semantics["train"].isdisjoint(semantics["development"])
    assert semantics["train"].isdisjoint(semantics["confirmation"])
    assert semantics["development"].isdisjoint(semantics["confirmation"])
    assert {
        (
            row["family"],
            row["compiler_targets"]["cardinality"],  # type: ignore[index]
        )
        for row in generated["train"]
    } == {
        (family, cardinality)
        for family in ("transport", "graph", "constraint")
        for cardinality in (4, 5, 6)
    }


def test_renderer_spans_are_real_ascii_source_occurrences() -> None:
    for split in ("train", "development", "confirmation"):
        for row in generate_rows(split=split, count=24, seed=90210):
            evidence = row["evidence_spans"]
            for source_name, text_name in (
                ("program", "program_text"),
                ("query", "query_text"),
            ):
                source = str(row[text_name]).encode("ascii")
                spans = evidence[source_name]  # type: ignore[index]
                for role_spans in spans.values():
                    for start, end in role_spans:
                        value = source[start:end]
                        assert len(value) == 6
                        assert value.isalnum()


def test_graph_and_constraint_rows_contain_many_to_many_relations() -> None:
    rows = generate_rows(split="train", count=120, seed=44)
    for family in ("graph", "constraint"):
        family_rows = [row for row in rows if row["family"] == family]
        assert family_rows
        found = False
        for row in family_rows:
            targets = row["compiler_targets"]
            cardinality = int(targets["cardinality"])  # type: ignore[index]
            for relation, active in zip(
                targets["rule_edges"],  # type: ignore[index]
                targets["rule_active"],  # type: ignore[index]
                strict=True,
            ):
                if not active:
                    continue
                cropped = tuple(
                    tuple(int(value) for value in values[:cardinality])
                    for values in relation[:cardinality]
                )
                assert len(relation_edges(cropped)) <= MAX_RELATION_EDGES
                if any(sum(values) > 1 for values in cropped):
                    found = True
        assert found


def test_neural_hard_executor_matches_independent_board_oracle() -> None:
    rows = [
        *generate_rows(split="train", count=18, seed=82),
        *generate_rows(split="development", count=18, seed=83),
        *generate_rows(split="confirmation", count=20, seed=84),
    ]
    for row in rows:
        targets = row["compiler_targets"]
        query = row["late_query_target"]
        oracle = row["oracle"]
        program = HardDeletedRelationalProgram(
            cardinality=torch.tensor(
                [targets["cardinality"]],  # type: ignore[index]
                dtype=torch.uint8,
            ),
            initial_edges=torch.tensor(
                [targets["initial_edges"]],  # type: ignore[index]
                dtype=torch.uint8,
            ),
            rule_edges=torch.tensor(
                [targets["rule_edges"]],  # type: ignore[index]
                dtype=torch.uint8,
            ),
            rule_active=torch.tensor(
                [targets["rule_active"]],  # type: ignore[index]
                dtype=torch.bool,
            ),
            event_rule=torch.tensor(
                [targets["event_rule"]],  # type: ignore[index]
                dtype=torch.uint8,
            ),
            event_kind=torch.tensor(
                [targets["event_kind"]],  # type: ignore[index]
                dtype=torch.uint8,
            ),
        )
        assert program.rule_edges.shape == (
            1,
            MAX_RULES,
            MAX_OBJECTS,
            MAX_OBJECTS,
        )
        result = rollout_hard_relational_program(
            program,
            HardDeletedRelationalQuery(
                torch.tensor(
                    [query["position"]],  # type: ignore[index]
                    dtype=torch.uint8,
                )
            ),
        )
        assert result.final_state[0].to(torch.uint8).tolist() == oracle[
            "terminal_state"
        ]
        assert result.answer_distribution[0].to(torch.uint8).tolist() == oracle[
            "answer_bits"
        ]


def test_validation_rejects_oracle_tampering() -> None:
    row = generate_rows(split="train", count=1, seed=9)[0]
    row["oracle"]["answer_bits"][0] ^= 1  # type: ignore[index]
    try:
        validate_row(row)
    except UROMBoardError:
        pass
    else:
        raise AssertionError("tampered UROM oracle was accepted")


def test_builder_is_write_once_and_keeps_confirmation_closed(tmp_path) -> None:
    tokenizer = Path(__file__).parents[1] / "artifacts/tokenizer/tokenizer.json"
    report = build(
        out_dir=tmp_path,
        seed=20260723,
        train_count=30,
        development_count=18,
        tokenizer_path=tokenizer,
    )
    assert report["confirmation_accesses"] == 0
    assert report["confirmation_generated"] is False
    assert report["overlap"] == {
        "semantic_worlds": 0,
        "row_hashes": 0,
    }
    assert (tmp_path / "urom3_train.jsonl").read_text().count("\n") == 30
    assert (tmp_path / "urom3_development.jsonl").read_text().count("\n") == 18
    assert (tmp_path / "manifest.json").is_file()
    assert report["tokenizer"]["maximum_tokens"] == 2_048
    assert (
        report["splits"]["development"]["token_lengths"]["program_maximum"]
        <= 2_048
    )
    try:
        build(
            out_dir=tmp_path,
            seed=20260723,
            train_count=30,
            development_count=18,
        )
    except FileExistsError:
        pass
    else:
        raise AssertionError("immutable UROM board was overwritten")
