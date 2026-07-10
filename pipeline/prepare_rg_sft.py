#!/usr/bin/env python3
"""Convert verifier-backed Reasoning-Gym execution traces into SFT examples.

``gen_reasoning_gym.py`` only emits a trace after the corresponding generator
verifies the answer. This adapter keeps that provenance explicit and produces
the completion format consumed by ``train/sft.py``. It never mutates a source
file and writes atomically to avoid partial JSONL snapshots.
"""
import argparse
import json
import os
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out)
    tmp = out.with_suffix(out.suffix + ".partial")
    out.parent.mkdir(parents=True, exist_ok=True)
    kept = malformed = missing = 0
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
    print(f"[rg-sft] kept={kept:,} malformed={malformed:,} missing={missing:,} -> {out}")


if __name__ == "__main__":
    main()
