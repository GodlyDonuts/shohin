#!/usr/bin/env python3
"""End-to-end small audit test for certified latent-ledger data."""

import json
import tempfile
from pathlib import Path

from audit_certified_latent_ledger_v1 import main as audit_main
from generate_certified_latent_ledger_v1 import main as generate_main


def invoke(main, argv):
    import sys

    old = sys.argv
    try:
        sys.argv = argv
        main()
    finally:
        sys.argv = old


def main():
    with tempfile.TemporaryDirectory() as directory:
        directory = Path(directory)
        train, evaluation, report = directory / "train.jsonl", directory / "eval.jsonl", directory / "audit.json"
        invoke(generate_main, [
            "generate", "--train-out", str(train), "--eval-out", str(evaluation),
            "--train-episodes", "20", "--eval-episodes-per-chunk", "1", "--seed", "17",
        ])
        invoke(audit_main, ["audit", "--train", str(train), "--eval", str(evaluation), "--out", str(report)])
        result = json.loads(report.read_text())
        assert result["train_eval_13gram_hits"] == 0
        assert result["invalid_counterfactual_train_pairs"] == 0
        assert result["invalid_counterfactual_eval_pairs"] == 0
        compact_train, compact_evaluation, compact_report = (
            directory / "compact_train.jsonl", directory / "compact_eval.jsonl", directory / "compact_audit.json",
        )
        invoke(generate_main, [
            "generate", "--train-out", str(compact_train), "--eval-out", str(compact_evaluation),
            "--train-episodes", "20", "--eval-episodes-per-chunk", "1", "--seed", "17", "--tag-scheme", "compact_v2",
        ])
        invoke(audit_main, ["audit", "--train", str(compact_train), "--eval", str(compact_evaluation), "--out", str(compact_report)])
        compact_result = json.loads(compact_report.read_text())
        assert compact_result["protocols"] == ["source_removed_readback_v2_compact_tags"]
        assert compact_result["train_eval_13gram_hits"] == 0
    print("certified latent ledger audit tests passed")


if __name__ == "__main__":
    main()
