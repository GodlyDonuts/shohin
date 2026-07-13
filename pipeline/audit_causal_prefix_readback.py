#!/usr/bin/env python3
"""Independently audit deterministic source-free prefix-readback derivation.

The readback targets are derived at trainer load time rather than serialized
into the LSA JSONL.  This audit deliberately recomputes that derivation from
the raw solver rows, binds it to a data SHA, and checks that every fresh decoder
query is canonical and contains no numerical answer leakage.
"""

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


def query(key):
    key = str(key)
    if not key or any(character.isspace() for character in key):
        raise ValueError("invalid readback key")
    return "After the updates received so far, what is the current value of {}? Return only the integer.".format(key)


def audit_row(row):
    keys = tuple(str(key) for key in row["keys"])
    values = {key: int(row["initial"][key]) for key in keys}
    operations, chunks = row["operations"], row["chunks"]
    if not keys or len(operations) != len(chunks):
        raise ValueError("invalid source/readback shape")
    targets = []
    for index, operation in enumerate(operations):
        values = apply_operation(values, operation)
        key = keys[index % len(keys)]
        target_query, answer = query(key), str(int(values[key]))
        if answer in target_query:
            raise ValueError("readback question leaks answer")
        targets.append((index, key, target_query, answer))
    if [int(values[key]) for key in keys] != [int(value) for value in row["state"]]:
        raise ValueError("final readback state does not match serialized state")
    return targets


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
                targets = audit_row(json.loads(line))
                counts["prefix_positions"] += len(targets)
                counts["chunk_count_{}".format(len(targets))] += 1
                for _, key, _, _ in targets:
                    counts["key_{}".format(key)] += 1
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                failures[str(exc)] += 1
                if len(examples) < 8:
                    examples.append({"line": line_number, "error": str(exc)})
    result = {
        "audit": "causal_prefix_readback_v1",
        "data": str(data.resolve()),
        "data_sha256": sha256(data),
        "rows": rows,
        "prefix_positions": counts.pop("prefix_positions", 0),
        "by_chunk_count": {key.removeprefix("chunk_count_"): value for key, value in sorted(counts.items()) if key.startswith("chunk_count_")},
        "by_key": {key.removeprefix("key_"): value for key, value in sorted(counts.items()) if key.startswith("key_")},
        "invalid_rows": sum(failures.values()),
        "failures": dict(sorted(failures.items())),
        "examples": examples,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))
    if result["invalid_rows"]:
        raise SystemExit("causal-prefix-readback audit failed")


if __name__ == "__main__":
    main()
