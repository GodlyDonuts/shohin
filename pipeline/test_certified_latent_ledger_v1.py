#!/usr/bin/env python3
"""Small deterministic generation checks for the certified latent ledger."""

import re
import tempfile
from pathlib import Path

from generate_certified_latent_ledger_v1 import build_rows, main, source_key
from generate_source_memory_packet_v1 import TRAIN_DOMAINS, TRAIN_STYLES


def run_main(argv):
    import sys

    old = sys.argv
    try:
        sys.argv = argv
        main()
    finally:
        sys.argv = old


def main_test():
    rows = build_rows(12, (2, 3, 4), TRAIN_DOMAINS, TRAIN_STYLES, (3, 55), False, 7)
    assert rows and len({source_key(row) for row in rows}) == len(rows)
    assert all("seal" in chunk for row in rows for chunk in row["chunks"])
    assert all("seal" not in row["query"] and "seal" not in row["response"] for row in rows)
    compact_rows = build_rows(
        12, (2, 3, 4), TRAIN_DOMAINS, TRAIN_STYLES, (3, 55), False, 7, tag_scheme="compact_v2",
    )
    assert all("seal" not in chunk for row in compact_rows for chunk in row["chunks"])
    assert all(row["protocol"] == "source_removed_readback_v2_compact_tags" for row in compact_rows)
    compact_tags = [
        re.findall(r"Reference ((?:[a-z] ?){12})\.", chunk)
        for row in compact_rows for chunk in row["chunks"]
    ]
    assert all(len(tags) == 1 and len(tags[0].split()) == 12 for tags in compact_tags)
    pairs = {}
    for row in rows:
        if row.get("counterfactual_id"):
            pairs.setdefault(row["counterfactual_id"], []).append(row)
    assert pairs and all(len(pair) == 2 and pair[0]["query"] == pair[1]["query"] and pair[0]["answer"] != pair[1]["answer"] for pair in pairs.values())
    with tempfile.TemporaryDirectory() as directory:
        train, evaluation = Path(directory) / "train.jsonl", Path(directory) / "eval.jsonl"
        run_main([
            "generate_certified_latent_ledger_v1.py", "--train-out", str(train), "--eval-out", str(evaluation),
            "--train-episodes", "12", "--eval-episodes-per-chunk", "1", "--seed", "8",
        ])
        assert train.exists() and evaluation.exists()
        compact_train, compact_evaluation = Path(directory) / "compact_train.jsonl", Path(directory) / "compact_eval.jsonl"
        run_main([
            "generate_certified_latent_ledger_v1.py", "--train-out", str(compact_train), "--eval-out", str(compact_evaluation),
            "--train-episodes", "12", "--eval-episodes-per-chunk", "1", "--seed", "8", "--tag-scheme", "compact_v2",
        ])
        assert compact_train.exists() and compact_evaluation.exists()
    print("certified latent ledger generator tests passed")


if __name__ == "__main__":
    main_test()
