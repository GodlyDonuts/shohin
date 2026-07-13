#!/usr/bin/env python3
"""Generate solver-verified source-free readback data for a latent ledger.

Each row asks for a fact from a recurrent source packet after a specified
prefix of the record has been read. Final states receive several distinct
queries, and paired rows differ in one final event while retaining the same
prefix and query. The model must answer through continuous memory only; no
external state controller exists at inference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path

from generate_latent_operator_v1 import apply_operation, sample_operation
from generate_source_memory_packet_v1 import (
    HELDOUT_DOMAINS,
    HELDOUT_STYLES,
    TRAIN_DOMAINS,
    TRAIN_STYLES,
    render_initial,
    render_operation,
    source_key,
)


PROBE_KINDS = ("read_left", "read_right", "sum", "difference")
WORD = re.compile(r"\w+")


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prompt_ngram_hashes(payload, width=13):
    # The decoder never receives the reference source or gold response. The
    # contamination gate therefore covers exactly the serialized source and
    # query prompt, not the answer label.
    words = WORD.findall("\n".join(payload["chunks"] + [payload["query"]]).lower())
    return {
        int.from_bytes(
            hashlib.blake2b(" ".join(words[index:index + width]).encode(), digest_size=8).digest(), "big"
        )
        for index in range(max(0, len(words) - width + 1))
    }


def render_chunks(domain, initial, operations, style, marker):
    _, _, item = domain
    chunks = []
    for step, operation in enumerate(operations):
        event = "Update {}: {}".format(step + 1, render_operation(operation, item, style))
        opening = render_initial(domain, initial, style) + "\nCase marker {}.".format(marker)
        # Every chunk receives an inert, record-unique tag. This makes the
        # source portion of every 13-gram split-safe without exposing a tag to
        # the source-free query decoder or correlating it with the answer.
        compact_marker = re.sub(r"\\W+", "", marker)
        tag = "Reference " + " ".join(
            "seal{}{}{:02x}".format(compact_marker, step, part)
            for part in range(16)
        ) + "."
        chunks.append((opening + "\n" + event + "\n" + tag) if step == 0 else (event + "\n" + tag))
    return chunks


def query_for(values, keys, item, style, kind):
    left, right = keys
    if kind == "read_left":
        answer, text = values[left], "What is the current {} count?".format(left)
    elif kind == "read_right":
        answer, text = values[right], "What is the current {} count?".format(right)
    elif kind == "sum":
        answer, text = values[left] + values[right], "What is the combined number of {}?".format(item)
    elif kind == "difference":
        high, low = (left, right) if values[left] >= values[right] else (right, left)
        answer = values[high] - values[low]
        text = "How many more {} are in {} than {}?".format(item, high, low)
    else:
        raise ValueError("unknown ledger probe kind: {}".format(kind))
    if style >= 3:
        text = "After every recorded update, " + text[0].lower() + text[1:]
    return text, int(answer)


def row(chunks, query, answer, initial, keys, operations, domain, style, heldout, reference, stage, kind, **extra):
    payload = {
        "chunks": chunks,
        "query": query,
        "response": "The answer is {}.".format(answer),
        "answer": str(answer),
        "source": "certified_latent_ledger_v1_{}".format("heldout" if heldout else "train"),
        "training_group": "certified_latent_ledger",
        "family": domain[0],
        "item": domain[2],
        "chunk_count": len(chunks),
        "heldout": bool(heldout),
        "initial": dict(initial),
        "keys": list(keys),
        "operations": operations,
        "style": int(style),
        "reference": reference,
        "ledger_stage": int(stage),
        "ledger_probe_kind": kind,
        "protocol": "source_removed_readback_v1",
    }
    payload.update(extra)
    return payload


def episode_rows(index, rng, domain, chunk_count, style, initial_range, heldout):
    _, keys, item = domain
    low, high = initial_range
    initial = {keys[0]: rng.randint(low, high), keys[1]: rng.randint(low, high)}
    values, states, operations = dict(initial), [], []
    for _ in range(chunk_count):
        operation = sample_operation(rng, values, keys)
        operations.append(operation)
        values = apply_operation(values, operation)
        states.append(dict(values))
    record_id = "L-{}-{:08d}".format(domain[0], index)
    marker = "{}-{:x}".format("h" if heldout else "t", index)
    chunks = render_chunks(domain, initial, operations, style, marker)
    rows = []
    for stage, state in enumerate(states, 1):
        for kind in PROBE_KINDS:
            query, answer = query_for(state, keys, item, style, kind)
            rows.append(row(
                chunks[:stage], query, answer, initial, keys, operations[:stage], domain, style, heldout,
                "{}-s{}-{}".format(record_id, stage, kind), stage, kind,
                ledger_record_id=record_id,
                counterfactual_id=None,
            ))

    # A paired final-event distinction: same prefix and query, two different
    # source endings and consequently two verified answers.
    prefix_operations = operations[:-1]
    prefix_values = dict(initial)
    for operation in prefix_operations:
        prefix_values = apply_operation(prefix_values, operation)
    target = keys[index % len(keys)]
    delta_a = rng.randint(1, 7)
    delta_b = delta_a + rng.randint(1, 7)
    variants = (
        ("a", {"kind": "add", "target": target, "value": delta_a}),
        ("b", {"kind": "add", "target": target, "value": delta_b}),
    )
    counterfactual_id = "{}-cf".format(record_id)
    for variant, final_operation in variants:
        variant_operations = prefix_operations + [final_operation]
        variant_values = apply_operation(dict(prefix_values), final_operation)
        variant_chunks = render_chunks(domain, initial, variant_operations, style, marker)
        kind = "read_left" if target == keys[0] else "read_right"
        query, answer = query_for(variant_values, keys, item, style, kind)
        rows.append(row(
            variant_chunks, query, answer, initial, keys, variant_operations, domain, style, heldout,
            "{}-{}".format(counterfactual_id, variant), chunk_count, kind,
            ledger_record_id=record_id,
            counterfactual_id=counterfactual_id,
            counterfactual_variant=variant,
        ))
    return rows


def build_rows(episodes, chunk_counts, domains, styles, initial_range, heldout, seed, forbidden=(), forbidden_ngrams=()):
    rng = random.Random(seed)
    rows, seen = [], set(forbidden)
    forbidden_ngrams = set(forbidden_ngrams)
    for episode in range(episodes):
        for attempt in range(10_000):
            domain = domains[episode % len(domains)]
            chunk_count = chunk_counts[episode % len(chunk_counts)]
            style = styles[(episode // len(domains)) % len(styles)]
            block = episode_rows(episode * 10_000 + attempt, rng, domain, chunk_count, style, initial_range, heldout)
            keys = [source_key(item) for item in block]
            block_ngrams = set().union(*(prompt_ngram_hashes(item) for item in block))
            if len(keys) == len(set(keys)) and not (set(keys) & seen) and not (block_ngrams & forbidden_ngrams):
                rows.extend(block)
                seen.update(keys)
                break
        else:
            raise RuntimeError("could not build unique ledger episode {}".format(episode))
    return rows


def write_jsonl(path, rows):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError("refusing to overwrite {}".format(destination))
    destination.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-out", required=True)
    parser.add_argument("--eval-out", required=True)
    parser.add_argument("--train-episodes", type=int, default=16_000)
    parser.add_argument("--eval-episodes-per-chunk", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260715)
    args = parser.parse_args()
    if args.train_episodes <= 0 or args.eval_episodes_per_chunk <= 0:
        raise SystemExit("episode counts must be positive")
    train = build_rows(args.train_episodes, (2, 3, 4), TRAIN_DOMAINS, TRAIN_STYLES, (3, 55), False, args.seed)
    train_keys = {source_key(item) for item in train}
    train_ngrams = set().union(*(prompt_ngram_hashes(item) for item in train))
    specs = (
        ("fit_iid", (2, 3, 4), TRAIN_DOMAINS, TRAIN_STYLES, (3, 55)),
        ("length_ood", (5, 6, 8), TRAIN_DOMAINS, TRAIN_STYLES, (3, 55)),
        ("language_ood", (2, 3, 4), HELDOUT_DOMAINS, HELDOUT_STYLES, (3, 55)),
        ("full_ood", (5, 6, 8), HELDOUT_DOMAINS, HELDOUT_STYLES, (56, 99)),
    )
    evaluation = []
    for offset, (regime, counts, domains, styles, value_range) in enumerate(specs):
        rows = build_rows(
            args.eval_episodes_per_chunk * len(counts), counts, domains, styles, value_range, True,
            args.seed + 100 + offset, train_keys | {source_key(item) for item in evaluation},
            train_ngrams,
        )
        for item in rows:
            item["eval_regime"] = regime
            item["reference"] = "{}-{}".format(regime, item["reference"])
            item["ledger_record_id"] = "{}-{}".format(regime, item["ledger_record_id"])
            if item.get("counterfactual_id"):
                item["counterfactual_id"] = "{}-{}".format(regime, item["counterfactual_id"])
        evaluation.extend(rows)
    write_jsonl(args.train_out, train)
    write_jsonl(args.eval_out, evaluation)
    print(json.dumps({
        "schema": "certified_latent_ledger_v1",
        "train_rows": len(train),
        "eval_rows": len(evaluation),
        "train_sha256": sha256(args.train_out),
        "eval_sha256": sha256(args.eval_out),
        "train_exact_eval_overlap": len(train_keys & {source_key(item) for item in evaluation}),
        "counterfactual_train_pairs": sum(item.get("counterfactual_variant") == "a" for item in train),
        "counterfactual_eval_pairs": sum(item.get("counterfactual_variant") == "a" for item in evaluation),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
