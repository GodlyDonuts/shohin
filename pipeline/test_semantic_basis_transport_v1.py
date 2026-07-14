#!/usr/bin/env python3
"""Focused generation and independent-audit checks for semantic basis transport."""
import tempfile
from pathlib import Path

from audit_semantic_basis_transport_v1 import audit, clean
from generate_semantic_basis_transport_v1 import PHASES, build_split, ledger, write_jsonl


def main() -> None:
    train = build_split(30, seed=17, heldout=False)
    heldout = build_split(12, seed=18, heldout=True)
    assert len(train) == 30 * len(PHASES)
    assert len(heldout) == 12 * len(PHASES)
    for rows, expected_split in ((train, "train"), (heldout, "heldout")):
        by_episode = {}
        for row in rows:
            assert row["split"] == expected_split
            assert row["expected_ledger"] == ledger(row["primary_value"], row["secondary_value"])
            by_episode.setdefault(row["episode_id"], set()).add(row["phase"])
        assert all(phases == set(PHASES) for phases in by_episode.values())
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        train_path = root / "train.jsonl"
        heldout_path = root / "heldout.jsonl"
        write_jsonl(train_path, train)
        write_jsonl(heldout_path, heldout)
        report = audit(train_path, heldout_path)
    assert clean(report), report
    assert report["cross_split_exact_prompt_hits"] == 0
    assert report["cross_split_ngram13_hits"] == 0
    print("semantic basis transport generation and audit checks: passed")


if __name__ == "__main__":
    main()
