#!/usr/bin/env python3
"""Build paired, solver-verified data for source-free latent state algebra.

The source is read only into continuous slots.  A later decoder receives those
slots and a fresh query, never source text.  Training-only labels provide the
two-register state after the source is read.  Equivalent pairs use different
event orderings that reach the same state; intervention pairs share a prefix
but finish at different verified states.  Neither state labels nor pair IDs
are supplied to the decoder at inference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path

from generate_certified_latent_ledger_v1 import (
    PROBE_KINDS,
    prompt_ngram_hashes,
    query_for,
    render_chunks,
)
from generate_latent_operator_v1 import apply_operation
from generate_source_memory_packet_v1 import (
    HELDOUT_DOMAINS,
    HELDOUT_STYLES,
    TRAIN_DOMAINS,
    TRAIN_STYLES,
    source_key,
)


STATE_SCALE = 256
WORD = re.compile(r"\w+")
PAIR_KINDS = ("equivalent", "intervention")
TAG_SCHEME = "compact_v2"


def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def copy_values(values):
    return {key: int(value) for key, value in values.items()}


def stable_operation(rng: random.Random, values, keys):
    """Sample bounded operations so the numeric state scale remains explicit."""
    left, right = keys
    kind = rng.choice(("add", "sub", "move", "swap"))
    if kind == "add":
        return {"kind": "add", "target": rng.choice(keys), "value": rng.randint(1, 9)}
    if kind == "sub":
        target = rng.choice(keys)
        return {"kind": "sub", "target": target, "value": rng.randint(1, max(1, values[target] // 3))}
    if kind == "move":
        source, target = (left, right) if rng.randrange(2) else (right, left)
        return {
            "kind": "move",
            "source": source,
            "target": target,
            "value": rng.randint(1, max(1, values[source] // 3)),
        }
    return {"kind": "swap", "left": left, "right": right}


def final_state(initial, operations):
    values = copy_values(initial)
    for operation in operations:
        values = apply_operation(values, operation)
    return values


def state_vector(values, keys):
    return [int(values[key]) for key in keys]


def make_row(
    *,
    domain,
    initial,
    operations,
    style,
    marker,
    heldout,
    pair_id,
    pair_kind,
    pair_member,
    query_kind,
    extra=None,
):
    _, keys, item = domain
    values = final_state(initial, operations)
    query, answer = query_for(values, keys, item, style, query_kind)
    payload = {
        "chunks": render_chunks(domain, initial, operations, style, marker, TAG_SCHEME),
        "query": query,
        "response": "The answer is {}.".format(answer),
        "answer": str(answer),
        "source": "latent_state_algebra_v1_{}".format("heldout" if heldout else "train"),
        "training_group": "latent_state_algebra",
        "family": domain[0],
        "item": item,
        "chunk_count": len(operations),
        "heldout": bool(heldout),
        "initial": copy_values(initial),
        "keys": list(keys),
        "operations": operations,
        "style": int(style),
        "state": state_vector(values, keys),
        "state_scale": STATE_SCALE,
        "pair_id": pair_id,
        "pair_kind": pair_kind,
        "pair_member": pair_member,
        "query_kind": query_kind,
        "protocol": "source_removed_latent_state_algebra_v1",
        "tag_scheme": TAG_SCHEME,
    }
    if extra:
        payload.update(extra)
    return payload


def equivalent_pair(index, rng, domain, depth, style, initial_range, heldout):
    """Create two distinct orderings with identical exact final state."""
    _, keys, _ = domain
    low, high = initial_range
    initial = {keys[0]: rng.randint(low, high), keys[1]: rng.randint(low, high)}
    target = rng.choice(keys)
    other = keys[1] if target == keys[0] else keys[0]
    first = {"kind": "add", "target": target, "value": rng.randint(1, 9)}
    second = {
        "kind": "move",
        "source": other,
        "target": target,
        "value": rng.randint(1, max(1, initial[other] // 3)),
    }
    # Both updates have fixed deltas, so the two orders commute exactly.
    operations_a = [first, second]
    operations_b = [second, first]
    values = final_state(initial, operations_a)
    for _ in range(depth - 2):
        operation = stable_operation(rng, values, keys)
        operations_a.append(operation)
        operations_b.append(dict(operation))
        values = apply_operation(values, operation)
    if final_state(initial, operations_a) != final_state(initial, operations_b):
        raise AssertionError("equivalent construction did not commute")
    pair_id = "{}-EQ-{:08d}".format("H" if heldout else "T", index)
    marker = "{}-eq-{:x}".format("h" if heldout else "t", index)
    kind = PROBE_KINDS[index % len(PROBE_KINDS)]
    return [
        make_row(
            domain=domain, initial=initial, operations=operations_a, style=style, marker=marker,
            heldout=heldout, pair_id=pair_id, pair_kind="equivalent", pair_member="a", query_kind=kind,
        ),
        make_row(
            domain=domain, initial=initial, operations=operations_b, style=style, marker=marker,
            heldout=heldout, pair_id=pair_id, pair_kind="equivalent", pair_member="b", query_kind=kind,
        ),
    ]


def intervention_pair(index, rng, domain, depth, style, initial_range, heldout):
    """Create a verified final-event intervention with an exact state delta."""
    _, keys, _ = domain
    low, high = initial_range
    initial = {keys[0]: rng.randint(low, high), keys[1]: rng.randint(low, high)}
    prefix, values = [], copy_values(initial)
    for _ in range(depth - 1):
        operation = stable_operation(rng, values, keys)
        prefix.append(operation)
        values = apply_operation(values, operation)
    target = keys[index % len(keys)]
    delta_a = rng.randint(1, 8)
    delta_b = delta_a + rng.randint(1, 8)
    operations_a = prefix + [{"kind": "add", "target": target, "value": delta_a}]
    operations_b = prefix + [{"kind": "add", "target": target, "value": delta_b}]
    pair_id = "{}-IV-{:08d}".format("H" if heldout else "T", index)
    marker = "{}-iv-{:x}".format("h" if heldout else "t", index)
    query_kind = "read_left" if target == keys[0] else "read_right"
    common = {"counterfactual_id": pair_id}
    return [
        make_row(
            domain=domain, initial=initial, operations=operations_a, style=style, marker=marker,
            heldout=heldout, pair_id=pair_id, pair_kind="intervention", pair_member="a",
            query_kind=query_kind, extra={**common, "counterfactual_variant": "a"},
        ),
        make_row(
            domain=domain, initial=initial, operations=operations_b, style=style, marker=marker,
            heldout=heldout, pair_id=pair_id, pair_kind="intervention", pair_member="b",
            query_kind=query_kind, extra={**common, "counterfactual_variant": "b"},
        ),
    ]


def build_pairs(pair_count, chunk_counts, domains, styles, initial_range, heldout, seed, forbidden=(), forbidden_ngrams=()):
    rng = random.Random(seed)
    rows, seen = [], set(forbidden)
    # Match the existing certified-ledger admission rule: n-grams only guard
    # the train/eval boundary.  Requiring every templated train prompt to have
    # globally unique 13-grams is not a decontamination requirement and caps
    # a valid large corpus long before the requested pair count.
    forbidden_ngrams = set(forbidden_ngrams)
    for index in range(pair_count):
        pair_kind = PAIR_KINDS[index % len(PAIR_KINDS)]
        domain = domains[index % len(domains)]
        depth = chunk_counts[index % len(chunk_counts)]
        style = styles[(index // len(domains)) % len(styles)]
        for attempt in range(10_000):
            pair_index = index * 10_000 + attempt
            block = (
                equivalent_pair(pair_index, rng, domain, depth, style, initial_range, heldout)
                if pair_kind == "equivalent"
                else intervention_pair(pair_index, rng, domain, depth, style, initial_range, heldout)
            )
            keys = [source_key(row) for row in block]
            ngrams = set().union(*(prompt_ngram_hashes(row) for row in block))
            if len(keys) == len(set(keys)) and not (set(keys) & seen) and not (ngrams & forbidden_ngrams):
                rows.extend(block)
                seen.update(keys)
                break
        else:
            raise RuntimeError("could not create unique pair {}".format(index))
    return rows


def write_jsonl(path, rows):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError("refusing to overwrite {}".format(target))
    target.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-out", required=True)
    parser.add_argument("--eval-out", required=True)
    parser.add_argument("--train-pairs", type=int, default=32_000)
    parser.add_argument("--eval-pairs-per-chunk", type=int, default=96)
    parser.add_argument("--seed", type=int, default=20260716)
    args = parser.parse_args()
    if args.train_pairs <= 0 or args.eval_pairs_per_chunk <= 0:
        raise SystemExit("pair counts must be positive")

    train = build_pairs(
        args.train_pairs, (2, 3, 4), TRAIN_DOMAINS, TRAIN_STYLES, (8, 55), False, args.seed,
    )
    train_keys = {source_key(row) for row in train}
    train_ngrams = set().union(*(prompt_ngram_hashes(row) for row in train))
    specs = (
        ("fit_iid", (2, 3, 4), TRAIN_DOMAINS, TRAIN_STYLES, (8, 55)),
        ("length_ood", (5, 6, 8), TRAIN_DOMAINS, TRAIN_STYLES, (8, 55)),
        ("language_ood", (2, 3, 4), HELDOUT_DOMAINS, HELDOUT_STYLES, (8, 55)),
        ("full_ood", (5, 6, 8), HELDOUT_DOMAINS, HELDOUT_STYLES, (56, 99)),
    )
    evaluation = []
    for offset, (regime, counts, domains, styles, value_range) in enumerate(specs):
        rows = build_pairs(
            args.eval_pairs_per_chunk * len(counts), counts, domains, styles, value_range, True,
            args.seed + 100 + offset, train_keys | {source_key(row) for row in evaluation}, train_ngrams,
        )
        for row in rows:
            row["eval_regime"] = regime
            row["pair_id"] = "{}-{}".format(regime, row["pair_id"])
            if row.get("counterfactual_id"):
                row["counterfactual_id"] = "{}-{}".format(regime, row["counterfactual_id"])
        evaluation.extend(rows)
    write_jsonl(args.train_out, train)
    write_jsonl(args.eval_out, evaluation)
    print(json.dumps({
        "schema": "latent_state_algebra_v1",
        "state_scale": STATE_SCALE,
        "train_pairs": args.train_pairs,
        "train_rows": len(train),
        "eval_rows": len(evaluation),
        "train_sha256": sha256(args.train_out),
        "eval_sha256": sha256(args.eval_out),
        "train_exact_eval_overlap": len(train_keys & {source_key(row) for row in evaluation}),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
