#!/usr/bin/env python3
"""Convert answer-labeled student rollouts into a balanced verifier SFT corpus."""
import argparse
import hashlib
import json
import os
import random
from pathlib import Path


CORRECT = "<|correct|>"
INCORRECT = "<|incorrect|>"


def verifier_question(question, candidate):
    return (
        "Problem:\n"
        f"{question}\n\n"
        "Candidate solution:\n"
        f"{candidate}\n\n"
        "Is the candidate solution correct? Reply only <|correct|> or <|incorrect|>."
    )


def key(question, candidate):
    text = " ".join((str(question) + "\n" + str(candidate)).lower().split())
    return hashlib.sha1(text.encode("utf-8", "ignore")).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--negative-ratio", type=float, default=4.0)
    ap.add_argument("--balance-classes", action="store_true",
                    help="retain equal positive/negative counts instead of all positives plus negatives")
    ap.add_argument("--seed", type=int, default=20260712)
    args = ap.parse_args()
    if args.negative_ratio <= 0:
        raise SystemExit("negative-ratio must be positive")

    positives, negatives, seen = [], [], set()
    malformed = missing = duplicates = 0
    with open(args.input, errors="replace") as src:
        for line in src:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1
                continue
            question = str(row.get("question") or "").strip()
            candidate = str(row.get("candidate") or "").strip()
            if not question or not candidate or not isinstance(row.get("correct"), bool):
                missing += 1
                continue
            row_key = key(question, candidate)
            if row_key in seen:
                duplicates += 1
                continue
            seen.add(row_key)
            target = CORRECT if row["correct"] else INCORRECT
            clean = {
                "question": verifier_question(question, candidate),
                "response": target,
                "source": "student_rollout_verifier",
                "training_group": "verifier_correct" if row["correct"] else "verifier_incorrect",
            }
            (positives if row["correct"] else negatives).append(clean)

    rng = random.Random(args.seed)
    rng.shuffle(positives)
    rng.shuffle(negatives)
    if args.balance_classes:
        keep_positives = keep_negatives = min(len(positives), len(negatives))
        kept = positives[:keep_positives] + negatives[:keep_negatives]
    else:
        keep_positives = len(positives)
        keep_negatives = min(len(negatives), max(1, round(len(positives) * args.negative_ratio)))
        kept = positives + negatives[:keep_negatives]
    rng.shuffle(kept)
    if not positives or not negatives:
        raise SystemExit(f"need both positive and negative student rollouts, got +{len(positives)} -{len(negatives)}")

    out = Path(args.out)
    tmp = out.with_suffix(out.suffix + ".partial")
    if out.exists() or tmp.exists():
        raise SystemExit(f"refusing to overwrite verifier dataset: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp, "w") as dst:
        for row in kept:
            dst.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp, out)
    print(json.dumps({
        "out": str(out), "valid_rows": len(kept), "positive_rows": keep_positives,
        "negative_rows": keep_negatives, "available_negative_rows": len(negatives),
        "available_positive_rows": len(positives), "balance_classes": args.balance_classes,
        "malformed_rows": malformed, "missing_rows": missing, "duplicate_rows": duplicates,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
