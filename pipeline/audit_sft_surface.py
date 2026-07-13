#!/usr/bin/env python3
"""Audit the exact token surface seen by completion-only SFT without Torch.

This mirrors ``train/sft.py`` prompt and completion boundary construction but
does not materialize model tensors.  It is suitable for CPU-only data hosts
and reports length tails per immutable data field before a GPU SFT is allowed.
"""
from __future__ import annotations

import argparse
import collections
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


def percentile(values, percentile_value):
    if not values:
        return 0
    ordered = sorted(values)
    index = round((len(ordered) - 1) * percentile_value)
    return ordered[index]


def row_surface(row, tokenizer, eos_id, prompt_field, response_field):
    prompt = str(row.get(prompt_field) or "")
    answer = row.get(response_field)
    if not prompt or answer is None or not str(answer).strip():
        return None
    # Exact completion-prompt path used by DRS/ADL SFT. Prompt and continuation
    # are encoded independently to avoid a BPE merge across the decode boundary.
    answer = str(answer).rstrip()
    separator = "" if prompt.endswith((" ", "\n", "\t")) else " "
    prompt_tokens = tokenizer.encode(prompt).ids
    answer_tokens = tokenizer.encode(separator + answer).ids
    return {
        "prompt_tokens": len(prompt_tokens),
        "answer_tokens": len(answer_tokens) + 1,  # training EOS is supervised
        "total_tokens": len(prompt_tokens) + len(answer_tokens) + 1,
        "eos_id": eos_id,
    }


def summarize(rows, pack_len):
    prompt = [row["prompt_tokens"] for row in rows]
    answer = [row["answer_tokens"] for row in rows]
    total = [row["total_tokens"] for row in rows]
    fit = [value for value in total if value <= pack_len]
    # This matches build_packed's full-window requirement: a packed group needs
    # at least pack_len + 2 tokens to form one X/Y training window.
    predicted_packs = max(0, (sum(fit) - pack_len - 1) // pack_len + 1)
    return {
        "rows": len(rows),
        "fit_rows": len(fit),
        "over_pack_len_rows": len(rows) - len(fit),
        "fit_token_total": sum(fit),
        "predicted_full_packs_if_grouped": predicted_packs,
        "prompt_tokens": {"p50": percentile(prompt, 0.50), "p90": percentile(prompt, 0.90), "p99": percentile(prompt, 0.99), "max": max(prompt, default=0)},
        "answer_tokens": {"p50": percentile(answer, 0.50), "p90": percentile(answer, 0.90), "p99": percentile(answer, 0.99), "max": max(answer, default=0)},
        "total_tokens": {"p50": percentile(total, 0.50), "p90": percentile(total, 0.90), "p99": percentile(total, 0.99), "max": max(total, default=0)},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--pack-len", type=int, default=2048)
    parser.add_argument("--prompt-field", default="completion_prompt")
    parser.add_argument("--response-field", default="response")
    parser.add_argument("--by", nargs="*", default=("kind", "width", "prompt_style"),
                        help="immutable row fields used for separate length summaries")
    args = parser.parse_args()
    if args.pack_len < 2:
        raise SystemExit("--pack-len must be at least 2")
    source, out = Path(args.data), Path(args.out)
    if not source.is_file():
        raise SystemExit("missing data: {}".format(source))
    if out.exists():
        raise SystemExit("refusing to overwrite output: {}".format(out))
    tokenizer = Tokenizer.from_file(args.tokenizer)
    eos_id = tokenizer.token_to_id("<|endoftext|>")
    if eos_id is None:
        raise SystemExit("tokenizer has no <|endoftext|> token")

    valid, invalid, grouped = [], 0, collections.defaultdict(list)
    for line_number, line in enumerate(source.read_text().splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        surface = row_surface(row, tokenizer, eos_id, args.prompt_field, args.response_field)
        if surface is None:
            invalid += 1
            continue
        valid.append(surface)
        for field in args.by:
            grouped["{}={}".format(field, row.get(field, "<missing>"))].append(surface)

    report = {
        "audit": "sft_surface_v1",
        "data": str(source.resolve()),
        "data_sha256": sha256_file(source),
        "tokenizer": str(Path(args.tokenizer).resolve()),
        "pack_len": args.pack_len,
        "prompt_field": args.prompt_field,
        "response_field": args.response_field,
        "group_fields": args.by,
        "invalid_or_missing_rows": invalid,
        "overall": summarize(valid, args.pack_len),
        "by_field_value": {key: summarize(value, args.pack_len) for key, value in sorted(grouped.items())},
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"overall": report["overall"], "invalid_or_missing_rows": invalid}, sort_keys=True))


if __name__ == "__main__":
    main()
