#!/usr/bin/env python3
"""Independently audit semantic-basis transport candidate data.

This module intentionally recomputes every target without importing the
generator.  It checks the candidate's exact state contract, full episode
coverage, response/answer bindings, duplicates, and train/held-out prompt
leakage before any future SFT can be considered.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path


WORD = re.compile(r"\w+")
LEDGER = re.compile(r"ledger:P=(-?\d+);Q=(-?\d+)")
PHASES = {"compile", "reflect", "update", "difference", "sum"}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalized_question(text: str) -> str:
    return " ".join(WORD.findall(text.lower()))


def ngrams(text: str, width: int = 13) -> set[str]:
    words = WORD.findall(text.lower())
    return {" ".join(words[index:index + width]) for index in range(max(0, len(words) - width + 1))}


def expected(phase: str, p: int, q: int, delta: int) -> tuple[str, str]:
    state = "ledger:P={};Q={}".format(p, q)
    if phase in {"compile", "reflect"}:
        return "<think>Primary P={} and secondary Q={} are the retained values.</think>\n{}".format(p, q, state), state
    if phase == "update":
        updated = "ledger:P={};Q={}".format(p + delta, q)
        return "<think>Update P by {}: {}+{}={}; preserve Q={}. </think>\n{}".format(delta, p, delta, p + delta, q, updated), updated
    if phase == "difference":
        answer = p - q
        return "<think>Use P={} and Q={}: {}-{}={}. </think>\nThe answer is {}.".format(p, q, p, q, answer, answer), str(answer)
    if phase == "sum":
        answer = p + q
        return "<think>Use P={} and Q={}: {}+{}={}. </think>\nThe answer is {}.".format(p, q, p, q, answer, answer), str(answer)
    raise ValueError("unknown phase: {}".format(phase))


def load(path: str | Path) -> list[dict]:
    rows = []
    with open(path) as source:
        for number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError("invalid JSON at {}:{}".format(path, number)) from exc
    return rows


def audit_split(rows: list[dict], split: str) -> tuple[dict, set[str], set[str]]:
    invalid = []
    question_counts = Counter()
    episodes = defaultdict(list)
    prompt_ngrams: set[str] = set()
    for index, row in enumerate(rows, 1):
        try:
            phase = row["phase"]
            p, q, delta = (int(row[key]) for key in ("primary_value", "secondary_value", "delta"))
            expected_response, expected_answer = expected(phase, p, q, delta)
            assert row.get("schema") == "semantic_basis_transport_v1"
            assert row.get("source") == "semantic_basis_transport_v1_candidate"
            assert row.get("training_group") == "semantic_basis_transport"
            assert row.get("split") == split
            assert phase in PHASES
            assert row.get("response") == expected_response
            assert str(row.get("answer")) == expected_answer
            assert row.get("expected_ledger") == "ledger:P={};Q={}".format(p, q)
            assert row.get("episode_id")
            assert row.get("question")
            if split == "train":
                assert 10 <= p <= 199 and 10 <= q <= 199 and 1 <= delta <= 9
            else:
                assert 201 <= p <= 299 and 201 <= q <= 299 and 11 <= delta <= 29
            if phase in {"compile", "reflect", "update"}:
                found = LEDGER.findall(row["response"])
                assert found
                expected_p = p + delta if phase == "update" else p
                assert (str(expected_p), str(q)) in found
        except (AssertionError, KeyError, TypeError, ValueError) as exc:
            invalid.append({"line": index, "error": str(exc)})
            continue
        normalized = normalized_question(row["question"])
        question_counts[normalized] += 1
        prompt_ngrams.update(ngrams(row["question"]))
        episodes[row["episode_id"]].append(phase)
    bad_episodes = {
        episode_id: sorted(phases)
        for episode_id, phases in episodes.items()
        if set(phases) != PHASES or len(phases) != len(PHASES)
    }
    return {
        "rows": len(rows),
        "invalid_rows": invalid,
        "duplicate_normalized_questions": sum(count - 1 for count in question_counts.values() if count > 1),
        "episodes": len(episodes),
        "bad_episodes": bad_episodes,
        "phases": dict(sorted(Counter(row.get("phase") for row in rows).items())),
    }, set(question_counts), prompt_ngrams


def audit(train_path: str | Path, heldout_path: str | Path) -> dict:
    train = load(train_path)
    heldout = load(heldout_path)
    train_report, train_questions, train_ngrams = audit_split(train, "train")
    heldout_report, heldout_questions, heldout_ngrams = audit_split(heldout, "heldout")
    return {
        "schema": "semantic_basis_transport_v1_audit",
        "train": str(train_path),
        "heldout": str(heldout_path),
        "train_sha256": sha256_file(train_path),
        "heldout_sha256": sha256_file(heldout_path),
        "train_report": train_report,
        "heldout_report": heldout_report,
        "cross_split_exact_prompt_hits": len(train_questions & heldout_questions),
        "cross_split_ngram13_hits": len(train_ngrams & heldout_ngrams),
    }


def clean(report: dict) -> bool:
    return not any((
        report["train_report"]["invalid_rows"],
        report["heldout_report"]["invalid_rows"],
        report["train_report"]["duplicate_normalized_questions"],
        report["heldout_report"]["duplicate_normalized_questions"],
        report["train_report"]["bad_episodes"],
        report["heldout_report"]["bad_episodes"],
        report["cross_split_exact_prompt_hits"],
        report["cross_split_ngram13_hits"],
    ))


def write_report(path: str | Path, report: dict) -> None:
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".partial")
    if path.exists() or temporary.exists():
        raise SystemExit("refusing to overwrite audit artifact: {}".format(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True)
    parser.add_argument("--heldout", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    report = audit(args.train, args.heldout)
    if not clean(report):
        raise SystemExit("semantic-basis transport audit failed: {}".format(json.dumps(report, sort_keys=True)))
    write_report(args.out, report)
    print(json.dumps({"schema": report["schema"], "train_rows": report["train_report"]["rows"], "heldout_rows": report["heldout_report"]["rows"]}, sort_keys=True))


if __name__ == "__main__":
    main()
