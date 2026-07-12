#!/usr/bin/env python3
"""Regression test for machine-readable shard-scan reports."""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import zstandard as zstd


def write_shard(path, values):
    payload = np.asarray(values, dtype=np.uint16).tobytes()
    path.write_bytes(zstd.ZstdCompressor().compress(payload))


def main():
    root = Path(__file__).resolve().parents[1]
    tokenizer = root / "artifacts" / "shohin-tok-32k.json"
    with tempfile.TemporaryDirectory() as temporary:
        shards = Path(temporary) / "shards"
        shards.mkdir()
        write_shard(shards / "shard_00000.u16.zst", [1, 2, 3, 4] * 64)
        write_shard(shards / "shard_00001.u16.zst", [4, 5, 6, 7] * 64)
        report = Path(temporary) / "report.json"
        subprocess.run([
            sys.executable, str(root / "pipeline" / "scan_shards.py"),
            "--shard-dirs", str(shards), "--tokenizer", str(tokenizer),
            "--diverge-token", "0", "--out", str(report),
        ], check=True, stdout=subprocess.DEVNULL)
        payload = json.loads(report.read_text())
        assert len(payload["shards"]) == 2
        assert set(payload["metrics"]) == {"H", "top1f", "top5f", "bytef", "meanid"}
        assert set(payload["outliers"]) == {
            "entropy_high", "entropy_low", "bytef_high", "top1f_high",
        }
        assert all("robust_z" in row for row in payload["outliers"]["entropy_high"]["outliers"])
    print("shard scan report checks: passed")


if __name__ == "__main__":
    main()
