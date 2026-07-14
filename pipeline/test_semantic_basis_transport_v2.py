#!/usr/bin/env python3
"""Focused generator/auditor contracts for exact-carrier semantic transport."""
import tempfile
from pathlib import Path

from audit_semantic_basis_transport_v2 import audit, clean
from generate_semantic_basis_transport_v2 import PHASES, build_split, ledger, write_jsonl


def main() -> None:
    train, heldout = build_split(30, 17, False), build_split(12, 18, True)
    assert len(train) == 30 * len(PHASES) and len(heldout) == 12 * len(PHASES)
    for rows in (train, heldout):
        updated_ledgers = set()
        for row in rows:
            if row["phase"] in {"compile", "reflect"}:
                assert row["response"] == ledger(row["primary_value"], row["secondary_value"])
            if row["phase"] == "update":
                assert row["response"] == ledger(row["primary_value"] + row["delta"], row["secondary_value"])
            if row["phase"] == "difference":
                assert row["expected_next_ledger"] not in updated_ledgers
                updated_ledgers.add(row["expected_next_ledger"])
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        train_path, heldout_path = root / "train.jsonl", root / "heldout.jsonl"
        write_jsonl(train_path, train)
        write_jsonl(heldout_path, heldout)
        report = audit(train_path, heldout_path)
    assert clean(report), report
    assert report["cross_split_exact_prompt_hits"] == 0
    assert report["cross_split_ngram13_hits"] == 0
    try:
        build_split(40_000, 19, False)
    except ValueError as exc:
        assert "P/Q prompt capacity" in str(exc)
    else:
        raise AssertionError("capacity guard did not reject impossible request")
    print("semantic basis transport v2 generation/audit checks: passed")


if __name__ == "__main__":
    main()
