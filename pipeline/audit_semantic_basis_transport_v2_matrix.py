#!/usr/bin/env python3
"""Audit factorized semantic-transport evaluations against frozen V2 train data."""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

from audit_semantic_basis_transport_v2 import ANSWER, LEDGER, PHASES, expected_response, load, ngrams, normalized_question, sha256_file
from generate_semantic_basis_transport_v2_matrix import FACTORS


def audit_factor(path: str | Path, name: str, train_questions: set[str], train_ngrams: set[str], train_pairs: set[tuple[int, int]]) -> tuple[dict, set[str], set[str]]:
    cfg = FACTORS[name]
    rows = load(path)
    invalid, questions, episodes, factor_ngrams = [], Counter(), defaultdict(list), set()
    pairs = set()
    for line, row in enumerate(rows, 1):
        try:
            phase = row["phase"]
            p, q, delta = (int(row[key]) for key in ("primary_value", "secondary_value", "delta"))
            target, answer = expected_response(phase, p, q, delta)
            assert row.get("schema") == "semantic_basis_transport_v2"
            assert row.get("source") == "semantic_basis_transport_v2_candidate"
            assert row.get("training_group") == "semantic_basis_transport_exact_carrier"
            assert row.get("split") == cfg["split"] and phase in PHASES
            assert row.get("response") == target and str(row.get("answer")) == answer
            assert row.get("expected_ledger") == "ledger:P={};Q={}".format(p, q)
            assert row.get("expected_next_ledger") == "ledger:P={};Q={}".format(p + delta, q)
            assert cfg["p"][0] <= p <= cfg["p"][1]
            assert cfg["q"][0] <= q <= cfg["q"][1]
            assert cfg["delta"][0] <= delta <= cfg["delta"][1]
            assert row.get("domain") in cfg["domains"]
            assert (row.get("primary_label"), row.get("secondary_label")) in cfg["labels"]
            assert (p, q) not in train_pairs
            prompt = row["question"].lower()
            if phase in {"compile", "reflect"}:
                assert LEDGER.fullmatch(row["response"])
                assert row["domain"] in prompt
                assert row["primary_label"] in prompt and row["secondary_label"] in prompt
                assert "ledger:p=" not in prompt
            elif phase == "update":
                assert LEDGER.fullmatch(row["response"])
                assert row["expected_ledger"] in row["question"]
                assert row["expected_next_ledger"] not in row["question"]
            else:
                assert ANSWER.fullmatch(row["response"])
                assert row["expected_next_ledger"] in row["question"]
                assert row["expected_ledger"] not in row["question"]
        except (AssertionError, KeyError, TypeError, ValueError) as exc:
            invalid.append({"line": line, "error": str(exc)})
            continue
        question = normalized_question(row["question"])
        questions[question] += 1
        factor_ngrams.update(ngrams(row["question"]))
        episodes[row["episode_id"]].append(phase)
        pairs.add((p, q))
    bad_episodes = {key: sorted(value) for key, value in episodes.items() if set(value) != PHASES or len(value) != len(PHASES)}
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "rows": len(rows),
        "episodes": len(episodes),
        "invalid_rows": invalid,
        "duplicate_normalized_questions": sum(count - 1 for count in questions.values() if count > 1),
        "bad_episodes": bad_episodes,
        "cross_train_exact_prompt_hits": len(set(questions) & train_questions),
        "shared_scaffold_ngram13_with_train": len(factor_ngrams & train_ngrams),
        "source_pair_overlap_with_train": len(pairs & train_pairs),
    }, set(questions), factor_ngrams


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True)
    parser.add_argument("--language", required=True)
    parser.add_argument("--values", required=True)
    parser.add_argument("--delta", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    train = load(args.train)
    train_questions = {normalized_question(row["question"]) for row in train}
    train_ngrams = set().union(*(ngrams(row["question"]) for row in train))
    train_pairs = {(int(row["primary_value"]), int(row["secondary_value"])) for row in train}
    reports, seen_questions, seen_ngrams = {}, set(), set()
    for name, path in (("language", args.language), ("values", args.values), ("delta", args.delta)):
        report, questions, factor_ngrams = audit_factor(path, name, train_questions, train_ngrams, train_pairs)
        report["cross_factor_exact_prompt_hits"] = len(questions & seen_questions)
        report["shared_scaffold_ngram13_with_prior_factors"] = len(factor_ngrams & seen_ngrams)
        reports[name] = report
        seen_questions.update(questions)
        seen_ngrams.update(factor_ngrams)
    failed = [name for name, report in reports.items() if any((
        report["invalid_rows"], report["duplicate_normalized_questions"], report["bad_episodes"],
        report["cross_train_exact_prompt_hits"], report["source_pair_overlap_with_train"],
        report["cross_factor_exact_prompt_hits"],
    ))]
    result = {"schema": "semantic_basis_transport_v2_factor_matrix_audit", "train_sha256": sha256_file(args.train), "factors": reports}
    if failed:
        raise SystemExit("factor matrix audit failed: {}".format(json.dumps(result, sort_keys=True)))
    out = Path(args.out)
    if out.exists() or out.with_suffix(out.suffix + ".partial").exists():
        raise SystemExit("refusing to overwrite factor matrix audit: {}".format(out))
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_suffix(out.suffix + ".partial")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, out)
    print(json.dumps({"schema": result["schema"], "factors": {name: report["rows"] for name, report in reports.items()}}, sort_keys=True))


if __name__ == "__main__":
    main()
