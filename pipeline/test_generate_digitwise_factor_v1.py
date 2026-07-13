#!/usr/bin/env python3
"""End-to-end admission and corruption tests for factorized DRS data."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train"))
from digitwise_factor_protocol import local_context, parse_register, parse_tape


def command(*args):
    return [sys.executable, *map(str, args)]


with tempfile.TemporaryDirectory() as directory:
    directory = Path(directory)
    data, episodes, report, audit = (directory / name for name in ("train.jsonl", "heldout.jsonl", "report.json", "audit.json"))
    subprocess.run(command(
        ROOT / "pipeline" / "generate_digitwise_factor_v1.py",
        "--train-out", data,
        "--heldout-out", episodes,
        "--report", report,
        "--variants", 1,
        "--heldout-per-regime", 2,
        "--seed", 17,
    ), check=True, capture_output=True, text=True)
    subprocess.run(command(
        ROOT / "pipeline" / "audit_digitwise_factor_v1.py",
        "--data", data,
        "--episodes", episodes,
        "--out", audit,
    ), check=True, capture_output=True, text=True)
    verdict = json.loads(audit.read_text())
    assert verdict["required_local_contexts"] == verdict["covered_local_contexts"] == 3400
    assert verdict["invalid_train_rows"] == verdict["invalid_heldout_episodes"] == 0

    rows = [json.loads(line) for line in data.read_text().splitlines()]
    target = None
    for row in rows:
        if row["kind"] == "transition":
            tape = parse_tape(row["tape"])
            target = local_context(tape, parse_register(row["register"], tape))
            break
    assert target is not None
    corrupted = []
    for row in rows:
        if row["kind"] == "transition":
            tape = parse_tape(row["tape"])
            context = local_context(tape, parse_register(row["register"], tape))
            if context == target:
                continue
        corrupted.append(row)
    bad_data = directory / "bad.jsonl"
    bad_data.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in corrupted))
    bad_audit = directory / "bad_audit.json"
    failed = subprocess.run(command(
        ROOT / "pipeline" / "audit_digitwise_factor_v1.py",
        "--data", bad_data,
        "--episodes", episodes,
        "--out", bad_audit,
    ), capture_output=True, text=True)
    assert failed.returncode != 0
    assert json.loads(bad_audit.read_text())["missing_local_contexts"] > 0

print("digitwise factor v1 generator/audit checks: passed")
