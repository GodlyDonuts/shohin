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
    print("shard approval checks: passed")


if __name__ == "__main__":
    main()
