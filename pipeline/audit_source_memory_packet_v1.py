#!/usr/bin/env python3
"""Independently verify source-memory packet rows and split separation."""

import argparse
import hashlib
import json
from pathlib import Path

from generate_latent_operator_v1 import apply_operation
from generate_source_memory_packet_v1 import source_key


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def valid(row, heldout):
    required = ("chunks", "query", "response", "answer", "initial", "keys", "operations", "query_spec", "chunk_count", "heldout")
    if any(field not in row for field in required) or bool(row["heldout"]) != heldout:
        return False
    if not isinstance(row["chunks"], list) or len(row["chunks"]) != len(row["operations"]) != int(row["chunk_count"]):
        return False
    keys = tuple(row["keys"])
    try:
        values = {key: int(row["initial"][key]) for key in keys}
        for operation in row["operations"]:
            values = apply_operation(values, operation)
    except (KeyError, TypeError, ValueError):
        return False
    query = row["query_spec"]
    if query.get("kind") == "read":
        expected = values.get(query.get("key"))
    elif query.get("kind") == "sum":
        expected = sum(values.values())
    elif query.get("kind") == "difference":
        expected = values.get(query.get("high"), 0) - values.get(query.get("low"), 0)
    else:
        return False
    return expected is not None and int(query.get("answer")) == expected == int(row["answer"]) and row["response"] == "The answer is {}.".format(expected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True)
    parser.add_argument("--eval", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    train, evaluation = load(args.train), load(args.eval)
    train_keys = [source_key(row) for row in train]
    eval_keys = [source_key(row) for row in evaluation]
    regimes = {}
    for row in evaluation:
        regime = row.get("eval_regime")
        regimes[regime] = regimes.get(regime, 0) + 1
    result = {
        "audit": "source_memory_packet_v1",
        "train": str(Path(args.train).resolve()),
        "eval": str(Path(args.eval).resolve()),
        "train_sha256": sha256(args.train),
        "eval_sha256": sha256(args.eval),
        "train_rows": len(train),
        "eval_rows": len(evaluation),
        "invalid_train_rows": sum(not valid(row, False) for row in train),
        "invalid_eval_rows": sum(not valid(row, True) for row in evaluation),
        "duplicate_train_prompts": len(train_keys) - len(set(train_keys)),
        "duplicate_eval_prompts": len(eval_keys) - len(set(eval_keys)),
        "train_eval_exact_prompt_hits": len(set(train_keys) & set(eval_keys)),
        "eval_regimes": dict(sorted(regimes.items())),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise SystemExit("refusing to overwrite {}".format(out))
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))
    failures = ("invalid_train_rows", "invalid_eval_rows", "duplicate_train_prompts", "duplicate_eval_prompts", "train_eval_exact_prompt_hits")
    if any(result[key] for key in failures) or set(regimes) != {"fit_iid", "length_ood", "language_ood", "full_ood"}:
        raise SystemExit("source-memory packet audit failed")


if __name__ == "__main__":
    main()
