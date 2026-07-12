#!/usr/bin/env python3
"""Fetch the labelled GSM8K train split for isolated verifier experiments.

The output is deliberately separate from public evaluation data and is written
atomically.  It normalizes questions before rejecting any prompt that appears in
the evaluation file, so a verifier can be trained on student rollouts without
contaminating its held-out selection gate.
"""
import argparse
import json
import os
import re
from pathlib import Path

from datasets import load_dataset


def normalized_question(text):
    return re.sub(r"\s+", " ", str(text).strip().lower())


def read_eval_questions(path):
    questions = set()
    with open(path, errors="replace") as src:
        for line in src:
            if not line.strip():
                continue
            row = json.loads(line)
            question = normalized_question(row.get("question") or row.get("problem") or "")
            if question:
                questions.add(question)
    return questions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--eval", required=True)
    args = ap.parse_args()

    out = Path(args.out)
    tmp = out.with_suffix(out.suffix + ".partial")
    if out.exists() or tmp.exists():
        raise SystemExit(f"refusing to overwrite existing GSM8K train artifact: {out}")
    eval_questions = read_eval_questions(args.eval)
    dataset = load_dataset("openai/gsm8k", "main", split="train")

    rows = kept = overlaps = missing = 0
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp, "w") as dst:
        for item in dataset:
            rows += 1
            question = str(item.get("question") or "").strip()
            answer = str(item.get("answer") or "").strip()
            if not question or not answer:
                missing += 1
                continue
            if normalized_question(question) in eval_questions:
                overlaps += 1
                continue
            dst.write(json.dumps({
                "question": question,
                "answer": answer,
                "source": "gsm8k_train",
            }, ensure_ascii=False) + "\n")
            kept += 1
    if not kept:
        tmp.unlink(missing_ok=True)
        raise SystemExit("GSM8K train fetch produced no usable rows")
    os.replace(tmp, out)
    print(json.dumps({
        "out": str(out),
        "seen": rows,
        "kept": kept,
        "dropped_eval_overlap": overlaps,
        "dropped_missing": missing,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
