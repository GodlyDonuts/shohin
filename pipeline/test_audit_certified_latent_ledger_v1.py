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
    print("certified latent ledger audit tests passed")


if __name__ == "__main__":
    main()
