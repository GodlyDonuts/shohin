#!/usr/bin/env python3
"""Regression tests for hash-bound shard admission approvals."""
import json
import subprocess
import sys
import tempfile
from pathlib import Path


def run(root, *args, ok=True):
    result = subprocess.run(
        [sys.executable, str(root / "pipeline" / "approve_shard_scan.py"), *args],
        text=True,
        capture_output=True,
    )
    if ok and result.returncode:
        raise AssertionError(result.stderr)
    if not ok and not result.returncode:
        raise AssertionError("expected shard approval to fail")
    return result


def main():
    root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory() as temporary:
        base = Path(temporary)
        shards = base / "shards"
        shards.mkdir()
        (shards / "manifest.json").write_text(json.dumps({"total_tokens": 250}))
        report = base / "scan.json"
        report.write_text(json.dumps({
            "shard_dirs": [str(shards.resolve())],
            "shards": [{"path": str(shards / "shard_00000.u16.zst"), "n": 250, "bytef": 0.0}],
        }))
        approval = base / "approval.json"
        run(root, "--shard-dir", str(shards), "--report", str(report), "--out", str(approval),
            "--review-note", "decoded samples and outlier report reviewed",
            "--min-total-tokens", "200", "--max-bytef", "0.01")
        payload = json.loads(approval.read_text())
        assert payload["total_tokens"] == 250
        assert payload["shards"] == 1
        bad = base / "bad.json"
        run(root, "--shard-dir", str(shards), "--report", str(report), "--out", str(bad),
            "--review-note", "bad floor", "--min-total-tokens", "251", ok=False)
        alias_shards = base / "alias_shards"
        alias_shards.mkdir()
        (alias_shards / "manifest.json").write_text(json.dumps({"tokens": 250}))
        alias_report = base / "alias_scan.json"
        alias_report.write_text(json.dumps({
            "shard_dirs": [str(alias_shards.resolve())],
            "shards": [{"path": str(alias_shards / "shard_00000.u16.zst"), "n": 250, "bytef": 0.0}],
        }))
        alias_approval = base / "alias_approval.json"
        run(root, "--shard-dir", str(alias_shards), "--report", str(alias_report), "--out", str(alias_approval),
            "--review-note", "tokenizer manifest alias verified",
            "--min-total-tokens", "200", "--max-bytef", "0.01")
        assert json.loads(alias_approval.read_text())["total_tokens"] == 250
    print("shard approval checks: passed")


if __name__ == "__main__":
    main()
