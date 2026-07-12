#!/usr/bin/env python3
"""Freeze a CodeContests *test-split* Python monitor for NLL trends.

The project trains only on CodeContests train rows.  This builder selects a
syntax-valid Python 3 reference from the test split, removes task-evaluation
prompt overlap, and writes prompt-plus-code text outside every training path.
It is a fixed code-likelihood monitor, not SFT data or an execution benchmark.
"""

import argparse
import hashlib
import json
import re
from pathlib import Path

from datasets import load_dataset
from tokenizers import Tokenizer

from curate_apps import build_test_grams, has_eval_overlap
from curate_code_contests import python3_solutions


WORD = re.compile(r"\w+")


def normalized_question(text):
    return " ".join(WORD.findall(str(text).lower()))


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def monitor_paths(output):
    output = Path(output)
    if "artifacts/evals" in output.as_posix():
        raise ValueError("code monitors must stay outside artifacts/evals")
    return output, output.with_suffix(output.suffix + ".manifest.json")


def render_text(question, code):
    return f"{question.strip()}\n\n```python\n{code.strip()}\n```"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--evals", required=True)
    parser.add_argument("--max-tokens", type=int, default=500_000)
    parser.add_argument("--max-code-chars", type=int, default=20_000)
    parser.add_argument("--ngram", type=int, default=13)
    args = parser.parse_args()
    if args.max_tokens < 1:
        raise SystemExit("max tokens must be positive")

    output, manifest_path = monitor_paths(args.out)
    if output.exists() or manifest_path.exists():
        raise SystemExit(f"refusing to overwrite frozen monitor: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    tokenizer = Tokenizer.from_file(args.tokenizer)
    test_grams = build_test_grams(args.evals, args.ngram)
    stream = load_dataset("Imandra/code_contests", split="test", streaming=True)
    seen = kept = token_total = 0
    drops = {key: 0 for key in ("eval_overlap", "missing_python3", "duplicate_question", "over_budget")}
    questions = set()
    temporary = output.with_suffix(output.suffix + ".partial")
    with temporary.open("w") as target:
        for row in stream:
            seen += 1
            question = str(row.get("description") or "").strip()
            if not question or has_eval_overlap(question, test_grams, args.ngram):
                drops["eval_overlap"] += 1
                continue
            key = normalized_question(question)
            if key in questions:
                drops["duplicate_question"] += 1
                continue
            code = next(iter(python3_solutions(row, args.max_code_chars)), None)
            if code is None:
                drops["missing_python3"] += 1
                continue
            text = render_text(question, code)
            tokens = len(tokenizer.encode(text).ids) + 1
            if token_total and token_total + tokens > args.max_tokens:
                drops["over_budget"] += 1
                break
            target.write(json.dumps({"text": text, "source_id": row.get("name")}, ensure_ascii=False) + "\n")
            questions.add(key)
            kept += 1
            token_total += tokens
            if token_total >= args.max_tokens:
                break
    if not kept:
        temporary.unlink(missing_ok=True)
        raise SystemExit("code monitor selection kept zero rows")
    temporary.replace(output)
    manifest = {
        "schema": "shohin-codecontests-test-monitor-v1",
        "dataset": "Imandra/code_contests",
        "split": "test",
        "language": "Python 3",
        "seen": seen,
        "kept": kept,
        "tokens": token_total,
        "max_tokens": args.max_tokens,
        "drops": drops,
        "data_sha256": sha256(output),
        "monitor_role": "fixed external code NLL trend monitor; not training data or an execution score",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
