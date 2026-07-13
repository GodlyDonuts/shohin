#!/usr/bin/env python3
"""End-to-end CBC admission smoke, including semantic corruption detection."""
import json
import copy
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline"))
from generate_counterfactual_bisimulation_v1 import build_episodes, make_world, rows_for_episode

with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    train, heldout, generated, audited = (root / name for name in ("train.jsonl", "heldout.jsonl", "generated.json", "audit.json"))
    subprocess.run([
        sys.executable, str(ROOT / "pipeline" / "generate_counterfactual_bisimulation_v1.py"),
        "--train-out", str(train), "--heldout-out", str(heldout), "--report", str(generated),
        "--train-per-domain", "3", "--heldout-per-domain", "3",
    ], check=True, capture_output=True, text=True)
    subprocess.run([
        sys.executable, str(ROOT / "pipeline" / "audit_counterfactual_bisimulation_v1.py"),
        "--train", str(train), "--heldout", str(heldout), "--report", str(audited),
    ], check=True, capture_output=True, text=True)
    report = json.loads(audited.read_text())
    assert report["invalid_train_rows"] == report["invalid_heldout_episodes"] == 0
    assert report["train_heldout_exact_prompt_hits"] == report["train_heldout_13gram_hits"] == 0
    assert set(report["heldout_regimes"]) == {"cbc_len4", "cbc_len8", "cbc_len12"}

    rows = [json.loads(line) for line in train.read_text().splitlines()]
    rows[0]["response"] = "cbc:amber=999;brass=999"
    corrupt_train = root / "corrupt-train.jsonl"
    corrupt_train.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    corrupt_train_report = root / "corrupt-train-audit.json"
    subprocess.run([
        sys.executable, str(ROOT / "pipeline" / "audit_counterfactual_bisimulation_v1.py"),
        "--train", str(corrupt_train), "--heldout", str(heldout), "--report", str(corrupt_train_report),
    ], check=True, capture_output=True, text=True)
    assert json.loads(corrupt_train_report.read_text())["invalid_train_rows"] == 1

    episodes = [json.loads(line) for line in heldout.read_text().splitlines()]
    episodes[0]["counterfactual"] = copy.deepcopy(episodes[0]["normal"])
    corrupt_heldout = root / "corrupt-heldout.jsonl"
    corrupt_heldout.write_text("".join(json.dumps(episode, sort_keys=True) + "\n" for episode in episodes))
    corrupt_heldout_report = root / "corrupt-heldout-audit.json"
    subprocess.run([
        sys.executable, str(ROOT / "pipeline" / "audit_counterfactual_bisimulation_v1.py"),
        "--train", str(train), "--heldout", str(corrupt_heldout), "--report", str(corrupt_heldout_report),
    ], check=True, capture_output=True, text=True)
    assert json.loads(corrupt_heldout_report.read_text())["invalid_heldout_episodes"] == 1

    mismatch = copy.deepcopy(build_episodes(1, 20260713, heldout=False)[0])
    keys = tuple(mismatch["keys"])
    operations = [copy.deepcopy(step["operation"]) for step in mismatch["counterfactual"]["steps"]]
    operations[0] = {"kind": "add", "key": keys[0], "value": 23}
    mismatch["counterfactual"] = make_world(
        mismatch["counterfactual"]["initial_values"], operations, keys, mismatch["domain"], mismatch["item"],
        mismatch["reference"], "train",
    )
    mismatched_train = root / "mismatched-counterfactual-train.jsonl"
    mismatched_train.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows_for_episode(mismatch)))
    mismatched_train_report = root / "mismatched-counterfactual-train-audit.json"
    subprocess.run([
        sys.executable, str(ROOT / "pipeline" / "audit_counterfactual_bisimulation_v1.py"),
        "--train", str(mismatched_train), "--heldout", str(heldout), "--report", str(mismatched_train_report),
    ], check=True, capture_output=True, text=True)
    assert json.loads(mismatched_train_report.read_text())["invalid_train_rows"] == 1
print("counterfactual bisimulation audit checks: passed")
