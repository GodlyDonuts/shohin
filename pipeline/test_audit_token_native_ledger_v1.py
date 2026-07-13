#!/usr/bin/env python3
"""End-to-end admission and corruption rejection for token-native ledger data."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]


def run(args, check=True):
    return subprocess.run([sys.executable, *map(str, args)], check=check, capture_output=True, text=True)


with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    train, heldout, generated, audited = (root / name for name in ("train.jsonl", "heldout.jsonl", "generated.json", "audit.json"))
    run([
        ROOT / "pipeline" / "generate_token_native_ledger_v1.py",
        "--train-out", train, "--heldout-out", heldout, "--report", generated,
        "--variants", "1", "--heldout-per-regime", "2",
    ])
    run([ROOT / "pipeline" / "audit_token_native_ledger_v1.py", "--data", train, "--episodes", heldout, "--out", audited])
    report = json.loads(audited.read_text())
    assert report["valid_train_rows"] > 0
    assert report["covered_local_contexts"] == report["required_local_contexts"]
    assert report["train_heldout_exact_prompt_hits"] == report["train_heldout_13gram_hits"] == 0

    rows = [json.loads(line) for line in train.read_text().splitlines()]
    bad = next(row for row in rows if row["kind"] == "transition")
    bad["response"] = "<think>"  # wrong carrier length; independent audit must reject it.
    corrupt = root / "corrupt.jsonl"
    corrupt.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    rejected = root / "rejected.json"
    result = run([ROOT / "pipeline" / "audit_token_native_ledger_v1.py", "--data", corrupt, "--episodes", heldout, "--out", rejected], check=False)
    assert result.returncode != 0
    assert json.loads(rejected.read_text())["invalid_train_rows"] >= 1

print("token-native ledger audit checks: passed")
