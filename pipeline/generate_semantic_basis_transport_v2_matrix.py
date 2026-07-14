#!/usr/bin/env python3
"""Build source-pair-disjoint factorized OOD evaluations for semantic transport.

Each shard changes one held-out variable family relative to the V2 train split:
language surface, P/Q magnitude, or update delta. These are evaluation-only
artifacts. They are never mixed into SFT data.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from generate_semantic_basis_transport_v2 import (
    HELDOUT_DOMAINS,
    HELDOUT_LABELS,
    PHASES,
    TRAIN_DOMAINS,
    TRAIN_LABELS,
    make_row,
    normalized_question,
    write_jsonl,
)


FACTORS = {
    "language": {
        "split": "factor_language",
        "heldout_wording": True,
        "p": (10, 199),
        "q": (10, 199),
        "delta": (1, 9),
        "domains": HELDOUT_DOMAINS,
        "labels": HELDOUT_LABELS,
    },
    "values": {
        "split": "factor_values",
        "heldout_wording": False,
        "p": (201, 299),
        "q": (201, 299),
        "delta": (1, 9),
        "domains": TRAIN_DOMAINS,
        "labels": TRAIN_LABELS,
    },
    "delta": {
        "split": "factor_delta",
        "heldout_wording": False,
        "p": (10, 199),
        "q": (10, 199),
        "delta": (11, 29),
        "domains": TRAIN_DOMAINS,
        "labels": TRAIN_LABELS,
    },
}


def source_pairs(train_path: str | Path) -> set[tuple[int, int]]:
    pairs = set()
    with open(train_path) as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("schema") != "semantic_basis_transport_v2" or row.get("split") != "train":
                raise ValueError("invalid train row at line {}".format(line_number))
            pairs.add((int(row["primary_value"]), int(row["secondary_value"])))
    if not pairs:
        raise ValueError("train file contains no source pairs")
    return pairs


def build_factor(name: str, episodes: int, seed: int, train_pairs: set[tuple[int, int]],
                 forbidden_questions: set[str]) -> list[dict]:
    if name not in FACTORS:
        raise ValueError("unknown factor: {}".format(name))
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    cfg = FACTORS[name]
    rng = random.Random(seed)
    pairs = [
        (p, q)
        for p in range(cfg["p"][0], cfg["p"][1] + 1)
        for q in range(cfg["q"][0], cfg["q"][1] + 1)
        if (p, q) not in train_pairs
    ]
    if len(pairs) < episodes:
        raise ValueError("factor {} has only {} source-pair-disjoint candidates".format(name, len(pairs)))
    rng.shuffle(pairs)
    delta_by_q = {q: rng.randint(*cfg["delta"]) for q in range(cfg["q"][0], cfg["q"][1] + 1)}
    rows, prompts = [], set()
    for p, q in pairs:
        delta = delta_by_q[q]
        domain = rng.choice(cfg["domains"])
        labels = rng.choice(cfg["labels"])
        episode_id = "{}-{:06d}".format(cfg["split"], len(rows) // len(PHASES))
        candidate = [
            make_row(cfg["split"], episode_id, phase, p, q, delta, domain, labels, cfg["heldout_wording"])
            for phase in PHASES
        ]
        normalized = [normalized_question(row["question"]) for row in candidate]
        if len(set(normalized)) != len(normalized):
            raise RuntimeError("duplicate prompt within factor episode {}".format(name))
        if any(prompt in prompts or prompt in forbidden_questions for prompt in normalized):
            continue
        prompts.update(normalized)
        rows.extend(candidate)
        if len(rows) == episodes * len(PHASES):
            break
    if len(rows) != episodes * len(PHASES):
        raise ValueError("factor {} lacks {} prompt-disjoint episodes".format(name, episodes))
    rng.shuffle(rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True, help="immutable V2 training JSONL used only for pair exclusion")
    parser.add_argument("--language-out", required=True)
    parser.add_argument("--values-out", required=True)
    parser.add_argument("--delta-out", required=True)
    parser.add_argument("--episodes", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=20260713)
    args = parser.parse_args()
    pairs = source_pairs(args.train)
    with open(args.train) as source:
        forbidden_questions = {normalized_question(json.loads(line)["question"]) for line in source if line.strip()}
    outputs = {
        "language": args.language_out,
        "values": args.values_out,
        "delta": args.delta_out,
    }
    report = {"schema": "semantic_basis_transport_v2_factor_matrix", "episodes": args.episodes}
    for offset, (name, output) in enumerate(outputs.items(), 1):
        rows = build_factor(name, args.episodes, args.seed + offset, pairs, forbidden_questions)
        write_jsonl(output, rows)
        forbidden_questions.update(normalized_question(row["question"]) for row in rows)
        report[name] = {"rows": len(rows), "split": FACTORS[name]["split"]}
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
