#!/usr/bin/env python3
"""Hash-bound tokenizer accounting for the token-native ledger candidate."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from tokenizers import Tokenizer


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def summarize(data_path, tokenizer):
    by_kind = defaultdict(lambda: {
        "rows": 0,
        "prompt_tokens": 0,
        "response_tokens": 0,
        "response_length_counts": Counter(),
    })
    with open(data_path) as source:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row.get("completion_prompt"), str) or not isinstance(row.get("response"), str):
                raise ValueError("row lacks string completion prompt or response")
            kind = row.get("kind")
            if kind not in {"transition", "final"}:
                raise ValueError("unknown token-native ledger row kind")
            prompt_tokens = len(tokenizer.encode(row["completion_prompt"]).ids)
            response_tokens = len(tokenizer.encode(row["response"]).ids)
            stats = by_kind[kind]
            stats["rows"] += 1
            stats["prompt_tokens"] += prompt_tokens
            stats["response_tokens"] += response_tokens
            stats["response_length_counts"][response_tokens] += 1
    if not by_kind["transition"]["rows"] or not by_kind["final"]["rows"]:
        raise ValueError("token-native ledger data needs transition and final rows")
    if set(by_kind["transition"]["response_length_counts"]) != {3}:
        raise ValueError("token-native transition is not exactly three tokenizer tokens")
    result = {}
    for kind, stats in sorted(by_kind.items()):
        rows = stats["rows"]
        result[kind] = {
            "rows": rows,
            "mean_prompt_tokens": stats["prompt_tokens"] / rows,
            "mean_response_tokens": stats["response_tokens"] / rows,
            "mean_prompt_plus_response_tokens": (stats["prompt_tokens"] + stats["response_tokens"]) / rows,
            "response_length_counts": dict(sorted(stats["response_length_counts"].items())),
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = Path(args.out)
    if out.exists():
        raise SystemExit("refusing to overwrite token audit: {}".format(out))
    tokenizer = Tokenizer.from_file(args.tokenizer)
    by_kind = summarize(args.data, tokenizer)
    report = {
        "audit": "token_native_ledger_v1_token_accounting",
        "data": str(Path(args.data).resolve()),
        "data_sha256": sha256_file(args.data),
        "tokenizer": str(Path(args.tokenizer).resolve()),
        "tokenizer_sha256": sha256_file(args.tokenizer),
        "by_kind": by_kind,
        "claim_boundary": (
            "This report measures serialized token cost only. It neither compares matched SFT outcomes nor "
            "establishes reasoning, context scaling, or a workspace."
        ),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
