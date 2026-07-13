#!/usr/bin/env python3
"""Build diagnostic slices for the continuous-latent operator experiment.

The original v1 held-out split changes several factors at once: composition
depth, language, labels, and numeric range. This generator decomposes those
changes so a failed full-OOD score is interpretable:

* ``fit_iid`` proves that answer-only training was learned at all.
* ``depth_ood`` changes only the number of composed state updates.
* ``language_ood`` changes surface language and labels, but not depth/range.
* ``full_ood`` changes all three factors.

Rows are evaluation-only. They are never included in a training mix and the
script refuses to overwrite an existing artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

from audit_latent_operator_v1 import valid
from generate_latent_operator_v1 import HELDOUT_DOMAINS, TRAIN_DOMAINS, make_row, ngrams, normalized


SLICES = (
    ("fit_iid", TRAIN_DOMAINS, False, (3, 29), (1, 2, 3, 4)),
    ("depth_ood", TRAIN_DOMAINS, False, (3, 29), (5, 6, 8)),
    ("language_ood", HELDOUT_DOMAINS, True, (3, 29), (1, 2, 3, 4)),
    ("full_ood", HELDOUT_DOMAINS, True, (37, 79), (5, 6, 8)),
)


def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_rows(path: str):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def build_slice(regime, domains, heldout_style, initial_range, depths, per_depth, seed, forbidden):
    """Create exact-prompt-disjoint rows while preserving requested factors."""
    rng = random.Random(seed)
    rows = []
    seen = set(forbidden)
    total = per_depth * len(depths)
    for index in range(total):
        depth = depths[index % len(depths)]
        domain = domains[index % len(domains)]
        for attempt in range(10_000):
            row = make_row(index * 10_000 + attempt, rng, domain, depth, heldout_style, initial_range)
            question = normalized(row["question"])
            if question not in seen:
                row["heldout"] = True
                row["eval_regime"] = regime
                row["source"] = "latent_operator_eval_slices_v2"
                rows.append(row)
                seen.add(question)
                break
        else:
            raise RuntimeError("could not make a unique {} row {}".format(regime, index))
    return rows


def build_slices(train_rows, per_depth, seed):
    train_questions = {normalized(row.get("question", "")) for row in train_rows}
    all_rows = []
    for offset, spec in enumerate(SLICES):
        regime, domains, heldout_style, initial_range, depths = spec
        all_rows.extend(build_slice(
            regime,
            domains,
            heldout_style,
            initial_range,
            depths,
            per_depth,
            seed + offset,
            train_questions | {normalized(row["question"]) for row in all_rows},
        ))
    return all_rows


def summarize(rows):
    summary = {}
    for regime, _, _, initial_range, depths in SLICES:
        matching = [row for row in rows if row["eval_regime"] == regime]
        summary[regime] = {
            "rows": len(matching),
            "depths": sorted({int(row["depth"]) for row in matching}),
            "initial_range": list(initial_range),
            "expected_depths": list(depths),
        }
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True, help="immutable v1 training JSONL used by the pilot")
    parser.add_argument("--out", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--per-depth", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260714)
    args = parser.parse_args()
    if args.per_depth <= 0:
        raise SystemExit("per-depth must be positive")
    if Path(args.out).exists() or Path(args.report).exists():
        raise SystemExit("refusing to overwrite a diagnostic artifact")

    train_rows = load_rows(args.train)
    rows = build_slices(train_rows, args.per_depth, args.seed)
    train_questions = {normalized(row.get("question", "")) for row in train_rows}
    questions = [normalized(row["question"]) for row in rows]
    train_grams = set().union(*(ngrams(question) for question in train_questions)) if train_questions else set()
    eval_grams = set().union(*(ngrams(question) for question in questions)) if questions else set()
    report = {
        "audit": "latent_operator_eval_slices_v2",
        "train": str(Path(args.train).resolve()),
        "train_sha256": sha256(args.train),
        "per_depth": args.per_depth,
        "seed": args.seed,
        "slices": summarize(rows),
        "rows": len(rows),
        "invalid_rows": sum(not valid(row, True) for row in rows),
        "duplicate_questions": len(questions) - len(set(questions)),
        "train_exact_prompt_hits": len(train_questions & set(questions)),
        # Same-surface slices deliberately share templates with train. Report
        # this without calling it leakage: these rows are evaluation-only.
        "train_ngram13_hits_report_only": len(train_grams & eval_grams),
    }
    if report["invalid_rows"] or report["duplicate_questions"] or report["train_exact_prompt_hits"]:
        raise SystemExit("diagnostic slice structural or exact-overlap failure: " + json.dumps(report, sort_keys=True))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    report["data_sha256"] = sha256(args.out)
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
