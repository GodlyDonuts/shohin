#!/usr/bin/env python3
"""Write a hash-bound token-cost report for source-packet training data."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from tokenizers import Tokenizer


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def percentile(values, quantile):
    return values[round((len(values) - 1) * quantile)]


def distribution(values):
    if not values:
        raise ValueError("cannot summarize an empty token distribution")
    values.sort()
    return {
        "count": len(values),
        "mean": round(sum(values) / len(values), 6),
        "min": values[0],
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": values[-1],
    }


def audit(path: Path, tokenizer: Tokenizer):
    chunks, sources, queries, answers = [], [], [], []
    protocols, tag_schemes = set(), set()
    rows = 0
    with path.open() as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            source_chunks = row.get("chunks")
            if not isinstance(source_chunks, list) or not source_chunks or not all(
                isinstance(chunk, str) for chunk in source_chunks
            ):
                raise ValueError("invalid chunks at {}:{}".format(path, line_number))
            if not isinstance(row.get("query"), str) or not isinstance(row.get("response"), str):
                raise ValueError("missing query/response at {}:{}".format(path, line_number))
            chunk_lengths = [len(tokenizer.encode(chunk).ids) for chunk in source_chunks]
            chunks.extend(chunk_lengths)
            sources.append(sum(chunk_lengths))
            queries.append(len(tokenizer.encode(row["query"]).ids))
            answers.append(len(tokenizer.encode(" " + row["response"].strip()).ids))
            protocols.add(row.get("protocol"))
            if row.get("tag_scheme") is not None:
                tag_schemes.add(row["tag_scheme"])
            rows += 1
    if not rows:
        raise ValueError("no rows in {}".format(path))
    return {
        "path": str(path.resolve()),
        "sha256": sha256(path),
        "rows": rows,
        "protocols": sorted(protocols, key=lambda value: "" if value is None else str(value)),
        "tag_schemes": sorted(tag_schemes),
        "chunk_tokens": distribution(chunks),
        "source_tokens_per_row": distribution(sources),
        "query_tokens": distribution(queries),
        "answer_tokens": distribution(answers),
    }


def parse_data(spec):
    try:
        label, raw_path = spec.split("=", 1)
    except ValueError as error:
        raise argparse.ArgumentTypeError("--data must be LABEL=PATH") from error
    if not label or not raw_path:
        raise argparse.ArgumentTypeError("--data must be LABEL=PATH")
    return label, Path(raw_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--data", action="append", required=True, type=parse_data, metavar="LABEL=PATH")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    labels = [label for label, _ in args.data]
    if len(labels) != len(set(labels)):
        raise SystemExit("data labels must be unique")
    output = Path(args.out)
    if output.exists():
        raise SystemExit("refusing to overwrite {}".format(output))
    tokenizer_path = Path(args.tokenizer)
    result = {
        "audit": "source_packet_token_cost_v1",
        "tokenizer": str(tokenizer_path.resolve()),
        "tokenizer_sha256": sha256(tokenizer_path),
        "data": {
            label: audit(path, Tokenizer.from_file(str(tokenizer_path)))
            for label, path in args.data
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
