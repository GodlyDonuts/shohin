#!/usr/bin/env python3
"""Freeze a deterministic external text monitor for pretraining NLL trends.

The output is deliberately not a training shard and is rejected under
``artifacts/evals`` because that directory feeds training decontamination.  A
monitor may have natural-web overlap with a future corpus, so it measures
regression/trend rather than proving data-disjoint generalization.
"""

import argparse
import hashlib
import json
from pathlib import Path

from datasets import load_dataset
from tokenizers import Tokenizer


def monitor_paths(output):
    output = Path(output)
    if "artifacts/evals" in output.as_posix():
        raise ValueError("NLL monitors must stay outside artifacts/evals")
    return output, output.with_suffix(output.suffix + ".manifest.json")


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--split", required=True)
    parser.add_argument("--text-col", default="text")
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-tokens", type=int, default=1_000_000)
    parser.add_argument("--min-chars", type=int, default=200)
    args = parser.parse_args()
    if args.max_tokens < 1 or args.min_chars < 1:
        raise SystemExit("max tokens and min chars must be positive")

    output, manifest_path = monitor_paths(args.out)
    if output.exists() or manifest_path.exists():
        raise SystemExit(f"refusing to overwrite frozen monitor: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    tokenizer = Tokenizer.from_file(args.tokenizer)
    options = {"split": args.split, "streaming": True}
    if args.config:
        options["name"] = args.config
    dataset = load_dataset(args.dataset, **options)

    temporary = output.with_suffix(output.suffix + ".partial")
    seen = kept = token_total = 0
    with temporary.open("w") as target:
        for row in dataset:
            seen += 1
            text = row.get(args.text_col)
            if not isinstance(text, str) or len(text.strip()) < args.min_chars:
                continue
            tokens = len(tokenizer.encode(text).ids) + 1
            if token_total and token_total + tokens > args.max_tokens:
                break
            target.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            kept += 1
            token_total += tokens
            if token_total >= args.max_tokens:
                break
    if not kept:
        temporary.unlink(missing_ok=True)
        raise SystemExit("monitor selection kept zero documents")
    temporary.replace(output)
    manifest = {
        "schema": "shohin-pretrain-nll-monitor-v1",
        "dataset": args.dataset,
        "config": args.config,
        "split": args.split,
        "text_col": args.text_col,
        "max_tokens": args.max_tokens,
        "min_chars": args.min_chars,
        "seen": seen,
        "kept": kept,
        "tokens": token_total,
        "data_sha256": sha256(output),
        "monitor_role": "fixed external NLL trend monitor; not a training source or contamination gate",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
