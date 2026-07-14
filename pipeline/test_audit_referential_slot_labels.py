#!/usr/bin/env python3
"""Smoke the referential label auditor on train/eval/manual miniatures."""

import json
import itertools
import subprocess
import tempfile
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    train_rows = [
        json.loads(line) for line in itertools.islice(
            (root / "artifacts/sft/role_equivariant_microcode_v3.jsonl").open(), 6,
        )
    ]
    eval_rows = [json.loads(line) for line in itertools.islice(
        (root / "artifacts/evals/latent_operator_eval_slices_v2_64.jsonl").open(), 2,
    )]
    manual_rows = [json.loads(line) for line in itertools.islice(
        (root / "artifacts/evals/categorical_microcode_manual_v1.jsonl").open(), 2,
    )]
    with tempfile.TemporaryDirectory() as directory:
        directory = Path(directory)
        inputs = {}
        for name, rows in (("train", train_rows), ("eval", eval_rows), ("manual", manual_rows)):
            path = directory / (name + ".jsonl")
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            inputs[name] = path
        out = directory / "report.json"
        subprocess.run([
            "python3", str(root / "pipeline/audit_referential_slot_labels.py"),
            "--train", str(inputs["train"]), "--eval", str(inputs["eval"]),
            "--manual", str(inputs["manual"]),
            "--tokenizer", str(root / "artifacts/shohin-tok-32k.json"),
            "--out", str(out), "--train-rows", "6", "--eval-rows", "2", "--manual-rows", "2",
        ], check=True, capture_output=True, text=True)
        report = json.loads(out.read_text())
        assert report["all_checks_pass"]
        assert report["datasets"]["train"]["counts"]["intro_slot_mentions"] == 12
    print("referential slot label audit tests passed")


if __name__ == "__main__":
    main()
