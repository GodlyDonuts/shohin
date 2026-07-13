#!/usr/bin/env python3
"""Recompute latent-operator rows and enforce train/held-out separation."""

import argparse
import hashlib
import json
from pathlib import Path

from generate_latent_operator_v1 import apply_operation, ngrams, normalized


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def valid(row, heldout):
    required = ("question", "response", "answer", "initial", "keys", "operations", "query", "depth", "heldout")
    if any(field not in row for field in required) or bool(row["heldout"]) != heldout:
        return False
    keys = tuple(row["keys"])
    values = {key: int(row["initial"][key]) for key in keys}
    try:
        for operation in row["operations"]:
            values = apply_operation(values, operation)
    except (KeyError, TypeError, ValueError):
        return False
    query = row["query"]
    if query["kind"] == "read":
        expected = values[query["key"]]
    elif query["kind"] == "sum":
        expected = sum(values.values())
    elif query["kind"] == "difference":
        expected = values[query["high"]] - values[query["low"]]
    else:
        return False
    return int(query["answer"]) == expected == int(row["answer"]) and len(row["operations"]) == int(row["depth"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True)
    parser.add_argument("--heldout", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    train, heldout = load(args.train), load(args.heldout)
    train_questions = [normalized(row.get("question", "")) for row in train]
    heldout_questions = [normalized(row.get("question", "")) for row in heldout]
    train_grams = set().union(*(ngrams(question) for question in train_questions)) if train_questions else set()
    heldout_grams = set().union(*(ngrams(question) for question in heldout_questions)) if heldout_questions else set()
    result = {
        "audit": "latent_operator_v1",
        "train": str(Path(args.train).resolve()),
        "heldout": str(Path(args.heldout).resolve()),
        "train_sha256": sha256(args.train),
        "heldout_sha256": sha256(args.heldout),
        "train_rows": len(train),
        "heldout_rows": len(heldout),
        "invalid_train_rows": sum(not valid(row, False) for row in train),
        "invalid_heldout_rows": sum(not valid(row, True) for row in heldout),
        "duplicate_train_questions": len(train_questions) - len(set(train_questions)),
        "duplicate_heldout_questions": len(heldout_questions) - len(set(heldout_questions)),
        "overlap": {
            "exact_prompt_hits": len(set(train_questions) & set(heldout_questions)),
            "ngram13_hits": len(train_grams & heldout_grams),
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise SystemExit("refusing to overwrite {}".format(out))
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))
    if any(result[key] for key in ("invalid_train_rows", "invalid_heldout_rows", "duplicate_train_questions", "duplicate_heldout_questions")):
        raise SystemExit("latent operator structural audit failed")
    if result["overlap"]["exact_prompt_hits"] or result["overlap"]["ngram13_hits"]:
        raise SystemExit("latent operator overlap audit failed")


if __name__ == "__main__":
    main()
