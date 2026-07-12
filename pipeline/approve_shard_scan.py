#!/usr/bin/env python3
"""Bind a reviewed shard scan to one immutable corpus manifest.

This is deliberately separate from ``scan_shards.py``: the scanner measures
outliers and samples decoded content, while this command records the explicit
admission decision after that report has been reviewed.  Flagship relaunch jobs
consume the resulting hash-bound approval record rather than trusting a manifest
file merely because it exists.
"""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-dir", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--review-note", required=True,
                        help="short evidence-based note about decoded content and outliers")
    parser.add_argument("--min-total-tokens", type=int, default=0)
    parser.add_argument("--max-bytef", type=float, default=0.01)
    args = parser.parse_args()

    shard_dir = Path(args.shard_dir).resolve()
    manifest_path = shard_dir / "manifest.json"
    report_path = Path(args.report).resolve()
    output = Path(args.out)
    if not args.review_note.strip():
        raise SystemExit("review-note must be nonblank")
    if not manifest_path.is_file() or not report_path.is_file():
        raise SystemExit("manifest or scan report is missing")

    manifest = json.loads(manifest_path.read_text())
    report = json.loads(report_path.read_text())
    report_dirs = {Path(path).resolve() for path in report.get("shard_dirs", [])}
    if shard_dir not in report_dirs:
        raise SystemExit("scan report does not describe the requested shard directory")
    shards = report.get("shards", [])
    if not shards or any(int(row.get("n", 0)) <= 0 for row in shards):
        raise SystemExit("scan report has no nonempty shard records")
    total_tokens = int(manifest.get("total_tokens", 0))
    if total_tokens < args.min_total_tokens:
        raise SystemExit(
            f"manifest token floor failed: {total_tokens} < {args.min_total_tokens}"
        )
    bytef = max(float(row.get("bytef", 1.0)) for row in shards)
    if bytef > args.max_bytef:
        raise SystemExit(f"byte-fallback gate failed: {bytef:.6f} > {args.max_bytef:.6f}")
    if output.exists():
        raise SystemExit(f"refusing to overwrite approval: {output}")

    approval = {
        "schema": "shohin-shard-scan-approval-v1",
        "approved_at_utc": datetime.now(timezone.utc).isoformat(),
        "review_note": args.review_note.strip(),
        "shard_dir": str(shard_dir),
        "manifest": str(manifest_path),
        "manifest_sha256": sha256(manifest_path),
        "scan_report": str(report_path),
        "scan_report_sha256": sha256(report_path),
        "total_tokens": total_tokens,
        "shards": len(shards),
        "max_bytef": bytef,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".partial")
    temporary.write_text(json.dumps(approval, indent=2, sort_keys=True) + "\n")
    temporary.replace(output)
    print(json.dumps(approval, sort_keys=True))


if __name__ == "__main__":
    main()
