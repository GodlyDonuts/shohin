#!/usr/bin/env python3
"""Convert verifier-backed Reasoning-Gym execution traces into SFT examples.

``gen_reasoning_gym.py`` only emits a trace after the corresponding generator
verifies the answer. This adapter keeps that provenance explicit and produces
the completion format consumed by ``train/sft.py``. It never mutates a source
file and writes atomically to avoid partial JSONL snapshots.
"""
import argparse
import hashlib
import json
import os
import re
from pathlib import Path


WORD = re.compile(r"\w+")


def normalized_question_key(question):
    normalized = " ".join(WORD.findall(question.lower()))
    return hashlib.sha1(normalized.encode("utf-8", "ignore")).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out)
    tmp = out.with_suffix(out.suffix + ".partial")
    out.parent.mkdir(parents=True, exist_ok=True)
    kept = malformed = missing = duplicates = 0
    seen_questions = set()
    with open(args.input, errors="replace") as src, open(tmp, "w") as dst:
        for line in src:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1
                continue
            question = str(row.get("question") or "").strip()
            trace = str(row.get("trace") or "").strip()
            answer = str(row.get("answer") or "").strip()
            if not question or not trace or not answer:
                missing += 1
                continue
            key = normalized_question_key(question)
            if key in seen_questions:
                duplicates += 1
                continue
            seen_questions.add(key)
            response = f"<think>{trace}</think>\nThe answer is {answer}."
            dst.write(json.dumps({
                "question": question,
                "response": response,
                "answer": answer,
                "source": "reasoning_gym_trace",
                "family": row.get("family"),
            }, ensure_ascii=False) + "\n")
            kept += 1
    os.replace(tmp, out)
    print(f"[rg-sft] kept={kept:,} malformed={malformed:,} missing={missing:,} "
          f"duplicates={duplicates:,} -> {out}")


if __name__ == "__main__":
    main()
