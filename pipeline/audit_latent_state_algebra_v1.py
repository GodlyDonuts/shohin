#!/usr/bin/env python3
"""Independently audit latent-state-algebra pair data before any GPU use."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from array import array
from bisect import bisect_left
from collections import Counter, defaultdict
from pathlib import Path

from generate_certified_latent_ledger_v1 import query_for
from generate_latent_operator_v1 import apply_operation
from generate_latent_state_algebra_v1 import PAIR_KINDS, STATE_SCALE
from generate_source_memory_packet_v1 import source_key


WORD = re.compile(r"\w+")
EXPECTED_REGIMES = {"fit_iid", "length_ood", "language_ood", "full_ood"}


def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def prompt(row):
    return "\n".join(row["chunks"] + [row["query"]])


def ngram_hashes(text, width=13):
    words = WORD.findall(text.lower())
    for index in range(max(0, len(words) - width + 1)):
        yield int.from_bytes(
            hashlib.blake2b(" ".join(words[index:index + width]).encode(), digest_size=8).digest(), "big"
        )


def valid_row(row, heldout):
    required = (
        "chunks", "query", "response", "answer", "initial", "keys", "item", "operations", "chunk_count",
        "heldout", "style", "state", "state_scale", "pair_id", "pair_kind", "pair_member", "protocol",
        "query_kind",
    )
    if any(key not in row for key in required) or bool(row["heldout"]) != heldout:
        return False
    if (
        not isinstance(row["chunks"], list)
        or len(row["chunks"]) != len(row["operations"]) != int(row["chunk_count"])
        or row["pair_kind"] not in PAIR_KINDS
        or row["pair_member"] not in {"a", "b"}
        or row["query_kind"] not in {"read_left", "read_right", "sum", "difference"}
        or row["protocol"] != "source_removed_latent_state_algebra_v1"
        or int(row["state_scale"]) != STATE_SCALE
    ):
        return False
    keys = tuple(row["keys"])
    try:
        values = {key: int(row["initial"][key]) for key in keys}
        for operation in row["operations"]:
            values = apply_operation(values, operation)
        query_kind = "read_left" if "current {} count".format(keys[0]) in row["query"] else None
        # Reconstruct the fixed four-way query by exact text comparison.
        matched = []
        for kind in ("read_left", "read_right", "sum", "difference"):
            query, answer = query_for(values, keys, row["item"], int(row["style"]), kind)
            if query == row["query"]:
                matched.append((kind, answer))
    except (KeyError, TypeError, ValueError):
        return False
    if len(matched) != 1:
        return False
    kind, answer = matched[0]
    return (
        int(row["answer"]) == int(answer)
        and row["response"] == "The answer is {}.".format(answer)
        and row["query_kind"] == kind
        and [int(values[key]) for key in keys] == [int(value) for value in row["state"]]
    )


def pair_failures(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row.get("pair_id")].append(row)
    invalid = 0
    by_kind = Counter()
    for pair_id, pair in groups.items():
        if len(pair) != 2 or {row.get("pair_member") for row in pair} != {"a", "b"}:
            invalid += 1
            continue
        first, second = pair
        if first.get("pair_kind") != second.get("pair_kind") or first["pair_kind"] not in PAIR_KINDS:
            invalid += 1
            continue
        kind = first["pair_kind"]
        by_kind[kind] += 1
        shared = (
            first["initial"] == second["initial"]
            and first["keys"] == second["keys"]
            and first["query"] == second["query"]
            and first["chunks"] != second["chunks"]
            and first["operations"] != second["operations"]
        )
        if not shared:
            invalid += 1
            continue
        if kind == "equivalent":
            if first["state"] != second["state"] or first["answer"] != second["answer"]:
                invalid += 1
        else:
            if (
                first["state"] == second["state"]
                or first["answer"] == second["answer"]
                or first["operations"][:-1] != second["operations"][:-1]
                or first.get("counterfactual_id") != pair_id
                or second.get("counterfactual_id") != pair_id
                or {first.get("counterfactual_variant"), second.get("counterfactual_variant")} != {"a", "b"}
            ):
                invalid += 1
    return len(groups), invalid, dict(sorted(by_kind.items()))


def sorted_ngrams(rows):
    values = array("Q")
    for row in rows:
        values.extend(ngram_hashes(prompt(row)))
    return array("Q", sorted(values))


def overlap_hits(values, rows):
    hits = 0
    for row in rows:
        for value in ngram_hashes(prompt(row)):
            index = bisect_left(values, value)
            hits += index < len(values) and values[index] == value
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
    train_pairs, invalid_train_pairs, train_kinds = pair_failures(train)
    eval_pairs, invalid_eval_pairs, eval_kinds = pair_failures(evaluation)
    regimes = Counter(row.get("eval_regime") for row in evaluation)
    report = {
        "audit": "latent_state_algebra_v1",
        "train": str(Path(args.train).resolve()),
        "eval": str(Path(args.eval).resolve()),
        "train_sha256": sha256(args.train),
        "eval_sha256": sha256(args.eval),
        "train_rows": len(train),
        "eval_rows": len(evaluation),
        "invalid_train_rows": sum(not valid_row(row, False) for row in train),
        "invalid_eval_rows": sum(not valid_row(row, True) for row in evaluation),
        "duplicate_train_prompts": len(train_keys) - len(set(train_keys)),
        "duplicate_eval_prompts": len(eval_keys) - len(set(eval_keys)),
        "train_eval_exact_prompt_hits": len(set(train_keys) & set(eval_keys)),
        "train_eval_13gram_hits": overlap_hits(sorted_ngrams(train), evaluation),
        "train_pairs": train_pairs,
        "eval_pairs": eval_pairs,
        "invalid_train_pairs": invalid_train_pairs,
        "invalid_eval_pairs": invalid_eval_pairs,
        "train_pair_kinds": train_kinds,
        "eval_pair_kinds": eval_kinds,
        "eval_regimes": dict(sorted(regimes.items())),
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise SystemExit("refusing to overwrite {}".format(output))
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))
    failures = (
        "invalid_train_rows", "invalid_eval_rows", "duplicate_train_prompts", "duplicate_eval_prompts",
        "train_eval_exact_prompt_hits", "train_eval_13gram_hits", "invalid_train_pairs", "invalid_eval_pairs",
    )
    if any(report[key] for key in failures) or set(regimes) != EXPECTED_REGIMES or set(train_kinds) != set(PAIR_KINDS) or set(eval_kinds) != set(PAIR_KINDS):
        raise SystemExit("latent-state-algebra audit failed")


if __name__ == "__main__":
    main()
