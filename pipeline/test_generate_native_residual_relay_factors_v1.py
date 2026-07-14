#!/usr/bin/env python3
"""Small admission contract for factorized NRR held-out data."""
import json
import subprocess
import sys
import tempfile
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory() as directory:
        directory = Path(directory)
        train, heldout, audit = directory / "train.jsonl", directory / "heldout.jsonl", directory / "audit.json"
        subprocess.run([sys.executable, str(root / "generate_native_residual_relay_v1.py"),
                        "--train-out", str(train), "--heldout-out", str(heldout), "--report", str(audit),
                        "--train-count", "128", "--heldout-count", "32", "--seed", "17"], check=True,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        factor_dir, report = directory / "factors", directory / "factor_audit.json"
        subprocess.run([sys.executable, str(root / "generate_native_residual_relay_factors_v1.py"),
                        "--train", str(train), "--out-dir", str(factor_dir), "--report", str(report),
                        "--count", "32", "--seed", "19"], check=True, stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE, universal_newlines=True)
        value = json.loads(report.read_text())
        assert set(value["regimes"]) == {"language", "values", "delta", "combined"}
        assert not any(item["train_exact_prompt_hits"] or item["train_13gram_hits"] for item in value["regimes"].values())
        assert not any(value["cross_regime_exact_prompt_hits"].values())
    print("native residual relay factor generator checks: passed")


if __name__ == "__main__":
    main()
