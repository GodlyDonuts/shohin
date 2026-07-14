#!/usr/bin/env python3
"""Freeze prompt-only raw-logit retention contexts for isolated SFT runs.

The output contains prompts and provenance only, never supervised answers. A
candidate SFT model is regularized to keep the immutable raw model's next-token
distribution on these contexts while a separate completion-masked objective
teaches verified skills. Held-out evaluation prompts are excluded exactly.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from tokenizers import Tokenizer


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalized(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip().lower())


def prompt_for(row: dict) -> str:
    completion_prompt = row.get("completion_prompt")
    if isinstance(completion_prompt, str) and completion_prompt.strip():
        # Completion-form code prompts may require a trailing newline or body
        # indentation. Preserve their exact inference boundary.
        return completion_prompt
    for key in ("question", "problem", "instruction"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return "Question: {}\nAnswer:".format(value.strip())
    return ""


def excluded_prompts(paths: list[Path]) -> set[str]:
    excluded = set()
    for path in paths:
        with path.open() as source:
            for line in source:
                if not line.strip():
                    continue
                row = json.loads(line)
                prompt = prompt_for(row)
                if prompt:
                    excluded.add(normalized(prompt))
    return excluded


def select_round_robin(rows: list[dict], limit: int, seed: int) -> list[dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[str(row["source"])].append(row)
    rng = random.Random(seed)
    for group in groups.values():
        rng.shuffle(group)
    selected, names = [], sorted(groups)
    while names and len(selected) < limit:
        remaining = []
        for name in names:
            if groups[name] and len(selected) < limit:
                selected.append(groups[name].pop())
            if groups[name]:
                remaining.append(name)
        names = remaining
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", nargs="+", required=True)
    parser.add_argument("--exclude", nargs="*", default=[])
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--audit", required=True)
    parser.add_argument("--limit", type=int, default=4096)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260714)
    args = parser.parse_args()
    if args.limit <= 0 or args.max_tokens < 2:
        raise SystemExit("limit must be positive and max tokens at least two")
    source_paths = [Path(path) for path in args.source]
    exclude_paths = [Path(path) for path in args.exclude]
    out, audit_path = Path(args.out), Path(args.audit)
    if not all(path.is_file() for path in source_paths + exclude_paths):
        raise SystemExit("every source and exclude path must exist")
    if out.exists() or audit_path.exists():
        raise SystemExit("refusing existing replay output or audit")
    tokenizer = Tokenizer.from_file(args.tokenizer)
    forbidden = excluded_prompts(exclude_paths)
    rows, seen, skipped = [], set(), Counter()
    for source_path in source_paths:
        with source_path.open() as source:
            for line_number, line in enumerate(source, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                prompt = prompt_for(row)
                if not prompt:
                    skipped["missing_prompt"] += 1
                    continue
                key = normalized(prompt)
                if key in forbidden:
                    skipped["heldout_exact_prompt"] += 1
                    continue
                if key in seen:
                    skipped["duplicate_prompt"] += 1
                    continue
                ids = tokenizer.encode(prompt).ids[:args.max_tokens]
                if len(ids) < 2:
                    skipped["short"] += 1
                    continue
                seen.add(key)
                rows.append({
                    "prompt": prompt,
                    "source": row.get("source") or row.get("training_group") or source_path.stem,
                    "source_path": str(source_path),
                    "source_line": line_number,
                    "tokens": len(ids),
                })
    selected = select_round_robin(rows, args.limit, args.seed)
    if len(selected) != args.limit:
        raise SystemExit("insufficient valid replay prompts: {}".format(len(selected)))
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as sink:
        for index, row in enumerate(selected):
            sink.write(json.dumps({"replay_id": "prompt-replay-{:05d}".format(index), **row}, sort_keys=True) + "\n")
    report = {
        "audit": "prompt_replay_v1",
        "source": [str(path) for path in source_paths],
        "source_sha256": {str(path): sha256_file(path) for path in source_paths},
        "exclude": [str(path) for path in exclude_paths],
        "exclude_sha256": {str(path): sha256_file(path) for path in exclude_paths},
        "out": str(out),
        "out_sha256": sha256_file(out),
        "rows": len(selected),
        "max_tokens": args.max_tokens,
        "source_counts": dict(sorted(Counter(row["source"] for row in selected).items())),
        "skipped": dict(sorted(skipped.items())),
        "exact_heldout_prompt_hits": sum(normalized(row["prompt"]) in forbidden for row in selected),
        "claim_boundary": "Prompt-only raw-logit retention set; it supplies no supervised answer or reasoning target.",
    }
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
