#!/usr/bin/env python3
"""Independently audit certified latent-ledger data and its causal pair structure."""

import argparse
import hashlib
import json
import re
from array import array
from bisect import bisect_left
from pathlib import Path

from generate_latent_operator_v1 import apply_operation
from generate_certified_latent_ledger_v1 import query_for
from generate_source_memory_packet_v1 import source_key


WORD = re.compile(r"\w+")
PROBE_KINDS = {"read_left", "read_right", "sum", "difference"}
EXPECTED_REGIMES = {"fit_iid", "length_ood", "language_ood", "full_ood"}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def full_prompt(row):
    # Match the model input under the source-dropping protocol. Gold answers
    # are labels, not evaluation prompts, so they cannot constitute prompt
    # contamination.
    return "\n".join(row["chunks"] + [row["query"]])


def ngram_hashes(text, width=13):
    words = WORD.findall(text.lower())
    for index in range(max(0, len(words) - width + 1)):
        yield int.from_bytes(
            hashlib.blake2b(" ".join(words[index:index + width]).encode(), digest_size=8).digest(), "big"
        )


def ledger_valid(row, heldout):
    required = (
        "chunks", "query", "response", "answer", "initial", "keys", "item", "operations", "chunk_count",
        "heldout", "ledger_stage", "ledger_probe_kind", "ledger_record_id", "style",
    )
    if any(key not in row for key in required) or bool(row["heldout"]) != heldout:
        return False
    if not isinstance(row["chunks"], list) or len(row["chunks"]) != len(row["operations"]) != int(row["chunk_count"]):
        return False
    if row.get("protocol") != "source_removed_readback_v1":
        return False
    if int(row.get("ledger_stage", 0)) != int(row["chunk_count"]):
        return False
    if row.get("ledger_probe_kind") not in PROBE_KINDS:
        return False
    if not isinstance(row.get("ledger_record_id"), str):
        return False
    keys = tuple(row["keys"])
    try:
        values = {key: int(row["initial"][key]) for key in keys}
        for operation in row["operations"]:
            values = apply_operation(values, operation)
        expected_query, expected_answer = query_for(
            values, keys, row["item"], int(row["style"]), row["ledger_probe_kind"],
        )
    except (KeyError, TypeError, ValueError):
        return False
    if (
        row["query"] != expected_query
        or int(row["answer"]) != expected_answer
        or row["response"] != "The answer is {}.".format(expected_answer)
    ):
        return False
    counterfactual_id = row.get("counterfactual_id")
    return counterfactual_id is None or (isinstance(counterfactual_id, str) and row.get("counterfactual_variant") in {"a", "b"})


def counterfactual_failures(rows):
    groups = {}
    for row in rows:
        if row.get("counterfactual_id"):
            groups.setdefault(row["counterfactual_id"], []).append(row)
    invalid = 0
    for pair in groups.values():
        if len(pair) != 2 or {row.get("counterfactual_variant") for row in pair} != {"a", "b"}:
            invalid += 1
            continue
        first, second = pair
        if (
            first["query"] != second["query"]
            or first["chunks"][:-1] != second["chunks"][:-1]
            or first["chunks"][-1] == second["chunks"][-1]
            or first["answer"] == second["answer"]
        ):
            invalid += 1
    return len(groups), invalid


def sorted_train_ngrams(rows):
    hashes = array("Q")
    for row in rows:
        hashes.extend(ngram_hashes(full_prompt(row)))
    return array("Q", sorted(hashes))


def overlap_hits(sorted_hashes, rows):
    hits = 0
    for row in rows:
        for value in ngram_hashes(full_prompt(row)):
            index = bisect_left(sorted_hashes, value)
            hits += index < len(sorted_hashes) and sorted_hashes[index] == value
    return hits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True)
    parser.add_argument("--eval", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    train, evaluation = load(args.train), load(args.eval)
    train_keys = [source_key(row) for row in train]
    eval_keys = [source_key(row) for row in evaluation]
    train_pairs, invalid_train_pairs = counterfactual_failures(train)
    eval_pairs, invalid_eval_pairs = counterfactual_failures(evaluation)
    regimes = {}
    for row in evaluation:
        regimes[row.get("eval_regime")] = regimes.get(row.get("eval_regime"), 0) + 1
    result = {
        "audit": "certified_latent_ledger_v1",
        "train": str(Path(args.train).resolve()),
        "eval": str(Path(args.eval).resolve()),
        "train_sha256": sha256(args.train),
        "eval_sha256": sha256(args.eval),
        "train_rows": len(train),
        "eval_rows": len(evaluation),
        "invalid_train_rows": sum(not ledger_valid(row, False) for row in train),
        "invalid_eval_rows": sum(not ledger_valid(row, True) for row in evaluation),
        "duplicate_train_prompts": len(train_keys) - len(set(train_keys)),
        "duplicate_eval_prompts": len(eval_keys) - len(set(eval_keys)),
        "train_eval_exact_prompt_hits": len(set(train_keys) & set(eval_keys)),
        "train_eval_13gram_hits": overlap_hits(sorted_train_ngrams(train), evaluation),
        "counterfactual_train_pairs": train_pairs,
        "counterfactual_eval_pairs": eval_pairs,
        "invalid_counterfactual_train_pairs": invalid_train_pairs,
        "invalid_counterfactual_eval_pairs": invalid_eval_pairs,
        "eval_regimes": dict(sorted(regimes.items())),
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise SystemExit("refusing to overwrite {}".format(output))
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))
    failures = (
        "invalid_train_rows", "invalid_eval_rows", "duplicate_train_prompts", "duplicate_eval_prompts",
        "train_eval_exact_prompt_hits", "train_eval_13gram_hits", "invalid_counterfactual_train_pairs",
        "invalid_counterfactual_eval_pairs",
    )
    if any(result[key] for key in failures) or set(regimes) != EXPECTED_REGIMES:
        raise SystemExit("certified latent-ledger audit failed")


if __name__ == "__main__":
    main()
