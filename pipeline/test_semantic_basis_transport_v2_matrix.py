#!/usr/bin/env python3
"""Focused contracts for factorized semantic-transport evaluation data."""
import json
import tempfile
from pathlib import Path

from generate_semantic_basis_transport_v2 import build_split, write_jsonl
from generate_semantic_basis_transport_v2_matrix import FACTORS, build_factor, normalized_question


def question_set(rows):
    return {row["question"] for row in rows}


def main() -> None:
    train = build_split(30_000, 20260713, False)
    train_pairs = {(row["primary_value"], row["secondary_value"]) for row in train}
    forbidden_questions = {normalized_question(row["question"]) for row in train}
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        train_path = root / "train.jsonl"
        write_jsonl(train_path, train)
        for offset, name in enumerate(FACTORS, 1):
            rows = build_factor(name, 40, 20260713 + offset, train_pairs, forbidden_questions)
            assert len(rows) == 200
            assert all((row["primary_value"], row["secondary_value"]) not in train_pairs for row in rows)
            assert len(question_set(rows)) == len(rows)
            path = root / (name + ".jsonl")
            write_jsonl(path, rows)
            persisted = [json.loads(line) for line in path.read_text().splitlines()]
            assert {row["split"] for row in persisted} == {FACTORS[name]["split"]}
            forbidden_questions.update(normalized_question(row["question"]) for row in rows)
    print("semantic basis transport factor matrix checks: passed")


if __name__ == "__main__":
    main()
