#!/usr/bin/env python3
"""End-to-end small admission smoke for DCRD data and its independent audit."""
import json
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    train, heldout, generated, audited = (root / name for name in ("train.jsonl", "heldout.jsonl", "generated.json", "audit.json"))
    subprocess.run([
        sys.executable, str(ROOT / "pipeline" / "generate_dual_code_reversible_v1.py"),
        "--train-out", str(train), "--heldout-out", str(heldout), "--report", str(generated),
        "--train-episodes", "12", "--heldout-per-regime", "3",
    ], check=True, capture_output=True, text=True)
    subprocess.run([
        sys.executable, str(ROOT / "pipeline" / "audit_dual_code_reversible_v1.py"),
        "--train", str(train), "--heldout", str(heldout), "--report", str(audited),
    ], check=True, capture_output=True, text=True)
    report = json.loads(audited.read_text())
    assert report["invalid_train_rows"] == report["invalid_heldout_episodes"] == 0
    assert report["train_heldout_exact_prompt_hits"] == report["train_heldout_13gram_hits"] == 0
    assert set(report["train_kinds"]) == {"a_to_b", "b_to_a", "forward_a", "readout", "reverse_b"}

    rows = [json.loads(line) for line in train.read_text().splitlines()]
    rows[0]["response"] = "answer=not-a-state"
    corrupt_train = root / "corrupt-train.jsonl"
    corrupt_train.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    corrupt_train_report = root / "corrupt-train-audit.json"
    subprocess.run([
        sys.executable, str(ROOT / "pipeline" / "audit_dual_code_reversible_v1.py"),
        "--train", str(corrupt_train), "--heldout", str(heldout), "--report", str(corrupt_train_report),
    ], check=True, capture_output=True, text=True)
    assert json.loads(corrupt_train_report.read_text())["invalid_train_rows"] == 1

    style_rows = [json.loads(line) for line in train.read_text().splitlines()]
    style_rows[0]["prompt_style"] = "heldout"
    corrupt_style = root / "corrupt-style.jsonl"
    corrupt_style.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in style_rows))
    corrupt_style_report = root / "corrupt-style-audit.json"
    subprocess.run([
        sys.executable, str(ROOT / "pipeline" / "audit_dual_code_reversible_v1.py"),
        "--train", str(corrupt_style), "--heldout", str(heldout), "--report", str(corrupt_style_report),
    ], check=True, capture_output=True, text=True)
    assert json.loads(corrupt_style_report.read_text())["invalid_train_rows"] == 1

    episodes = [json.loads(line) for line in heldout.read_text().splitlines()]
    episodes[0]["counterfactual"]["expected_answer"] = episodes[0]["expected_answer"]
    corrupt_heldout = root / "corrupt-heldout.jsonl"
    corrupt_heldout.write_text("".join(json.dumps(episode, sort_keys=True) + "\n" for episode in episodes))
    corrupt_heldout_report = root / "corrupt-heldout-audit.json"
    subprocess.run([
        sys.executable, str(ROOT / "pipeline" / "audit_dual_code_reversible_v1.py"),
        "--train", str(train), "--heldout", str(corrupt_heldout), "--report", str(corrupt_heldout_report),
    ], check=True, capture_output=True, text=True)
    assert json.loads(corrupt_heldout_report.read_text())["invalid_heldout_episodes"] == 1
print("dual-code audit checks: passed")
