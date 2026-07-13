#!/usr/bin/env python3
"""Independently validate every solver-derived prefix target before GPU use."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from generate_latent_operator_v1 import apply_operation


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prefix_states(row):
    keys = tuple(row["keys"])
    scale = int(row["state_scale"])
    if not keys or scale <= 0:
        raise ValueError("invalid keys or state scale")
    values = {key: int(row["initial"][key]) for key in keys}
    result = []
    for operation in row["operations"]:
        values = apply_operation(values, operation)
        result.append([float(values[key]) / float(scale) for key in keys])
    if len(result) != len(row["chunks"]):
        raise ValueError("prefix count does not match serialized chunks")
    final = [float(value) / float(scale) for value in row["state"]]
    if result[-1] != final:
        raise ValueError("final prefix target does not match serialized final state")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    data, output = Path(args.data), Path(args.out)
    if not data.is_file() or not data.stat().st_size:
        raise SystemExit("missing data: {}".format(data))
    if output.exists():
        raise SystemExit("refusing existing output: {}".format(output))
    counts, failures, examples = Counter(), Counter(), []
    rows = 0
    with data.open() as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            rows += 1
            try:
                row = json.loads(line)
                targets = prefix_states(row)
                counts["prefix_positions"] += len(targets)
                counts["chunk_count_{}".format(len(targets))] += 1
                counts["state_dim_{}".format(len(targets[-1]))] += 1
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                failures[str(exc)] += 1
                if len(examples) < 8:
                    examples.append({"line": line_number, "error": str(exc)})
    result = {
        "audit": "prefix_state_supervision_v1",
        "data": str(data.resolve()),
        "data_sha256": sha256(data),
        "rows": rows,
        "prefix_positions": counts.pop("prefix_positions", 0),
        "by_chunk_count": {key.removeprefix("chunk_count_"): value for key, value in sorted(counts.items()) if key.startswith("chunk_count_")},
        "by_state_dim": {key.removeprefix("state_dim_"): value for key, value in sorted(counts.items()) if key.startswith("state_dim_")},
        "invalid_rows": sum(failures.values()),
        "failures": dict(sorted(failures.items())),
        "examples": examples,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))
    if result["invalid_rows"]:
        raise SystemExit("prefix-state audit failed")


if __name__ == "__main__":
    main()
