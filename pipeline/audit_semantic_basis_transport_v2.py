#!/usr/bin/env python3
"""Independently audit exact-carrier semantic-basis transport data."""
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
ANSWER = re.compile(r"answer=(-?\d+)")
PHASES = {"compile", "reflect", "update", "difference", "sum"}
TRAIN_DOMAINS = {"workshop", "orchard", "foundry", "classroom", "warehouse", "greenhouse"}
HELDOUT_DOMAINS = {"harbor", "observatory", "clinic", "theater", "archive", "shipyard"}
TRAIN_LABELS = {("amber", "cobalt"), ("cedar", "flint"), ("violet", "ochre")}
HELDOUT_LABELS = {("north", "south"), ("silver", "basalt"), ("lilac", "ivory")}


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


def ledger(p: int, q: int) -> str:
    return "ledger:P={};Q={}".format(p, q)


def expected_response(phase: str, p: int, q: int, delta: int) -> tuple[str, str]:
    if phase in {"compile", "reflect"}:
        value = ledger(p, q)
        return value, value
    if phase == "update":
        value = ledger(p + delta, q)
        return value, value
    if phase == "difference":
        value = p + delta - q
        return "answer={}".format(value), str(value)
    if phase == "sum":
        value = p + delta + q
        return "answer={}".format(value), str(value)
    raise ValueError("unknown phase")


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
    invalid, questions, episodes, prompt_ngrams = [], Counter(), defaultdict(list), set()
    allowed_domains = TRAIN_DOMAINS if split == "train" else HELDOUT_DOMAINS
    allowed_labels = TRAIN_LABELS if split == "train" else HELDOUT_LABELS
    for line, row in enumerate(rows, 1):
        try:
            phase = row["phase"]
            p, q, delta = (int(row[key]) for key in ("primary_value", "secondary_value", "delta"))
            target, answer = expected_response(phase, p, q, delta)
            assert row.get("schema") == "semantic_basis_transport_v2"
            assert row.get("source") == "semantic_basis_transport_v2_candidate"
            assert row.get("training_group") == "semantic_basis_transport_exact_carrier"
            assert row.get("split") == split and phase in PHASES
            assert row.get("response") == target and str(row.get("answer")) == answer
            assert row.get("expected_ledger") == ledger(p, q)
            assert row.get("expected_next_ledger") == ledger(p + delta, q)
            assert row.get("domain") in allowed_domains
            assert (row.get("primary_label"), row.get("secondary_label")) in allowed_labels
            assert row.get("episode_id") and row.get("question")
            prompt_lower = row["question"].lower()
            if split == "train":
                assert 10 <= p <= 199 and 10 <= q <= 199 and 1 <= delta <= 9
            else:
                assert 201 <= p <= 299 and 201 <= q <= 299 and 11 <= delta <= 29
            if phase in {"compile", "reflect"}:
                assert LEDGER.fullmatch(row["response"])
                assert row["domain"] in prompt_lower
                assert row["primary_label"] in prompt_lower
                assert row["secondary_label"] in prompt_lower
                assert "ledger:P=" not in row["question"]
            elif phase == "update":
                assert LEDGER.fullmatch(row["response"])
                assert row["expected_ledger"] in row["question"]
                assert row["expected_next_ledger"] not in row["question"]
                assert row["domain"] not in prompt_lower
                assert row["primary_label"] not in prompt_lower
                assert row["secondary_label"] not in prompt_lower
            else:
                assert ANSWER.fullmatch(row["response"])
                assert row["expected_next_ledger"] in row["question"]
                assert row["expected_ledger"] not in row["question"]
                assert row["domain"] not in prompt_lower
                assert row["primary_label"] not in prompt_lower
                assert row["secondary_label"] not in prompt_lower
        except (AssertionError, KeyError, TypeError, ValueError) as exc:
            invalid.append({"line": line, "error": str(exc)})
            continue
        prompt = normalized_question(row["question"])
        questions[prompt] += 1
        prompt_ngrams.update(ngrams(row["question"]))
        episodes[row["episode_id"]].append(phase)
    bad_episodes = {key: sorted(values) for key, values in episodes.items() if set(values) != PHASES or len(values) != len(PHASES)}
    return {
        "rows": len(rows),
        "invalid_rows": invalid,
        "duplicate_normalized_questions": sum(count - 1 for count in questions.values() if count > 1),
        "episodes": len(episodes),
        "bad_episodes": bad_episodes,
        "phases": dict(sorted(Counter(row.get("phase") for row in rows).items())),
    }, set(questions), prompt_ngrams


def audit(train_path: str | Path, heldout_path: str | Path) -> dict:
    train, heldout = load(train_path), load(heldout_path)
    train_report, train_questions, train_ngrams = audit_split(train, "train")
    heldout_report, heldout_questions, heldout_ngrams = audit_split(heldout, "heldout")
    return {
        "schema": "semantic_basis_transport_v2_audit",
        "train": str(train_path), "heldout": str(heldout_path),
        "train_sha256": sha256_file(train_path), "heldout_sha256": sha256_file(heldout_path),
        "train_report": train_report, "heldout_report": heldout_report,
        "cross_split_exact_prompt_hits": len(train_questions & heldout_questions),
        "cross_split_ngram13_hits": len(train_ngrams & heldout_ngrams),
    }


def clean(report: dict) -> bool:
    return not any((
        report["train_report"]["invalid_rows"], report["heldout_report"]["invalid_rows"],
        report["train_report"]["duplicate_normalized_questions"], report["heldout_report"]["duplicate_normalized_questions"],
        report["train_report"]["bad_episodes"], report["heldout_report"]["bad_episodes"],
        report["cross_split_exact_prompt_hits"], report["cross_split_ngram13_hits"],
    ))


def write_report(path: str | Path, report: dict) -> None:
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".partial")
    if path.exists() or temporary.exists():
        raise SystemExit("refusing to overwrite semantic-basis v2 audit: {}".format(path))
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
        raise SystemExit("semantic-basis v2 audit failed: {}".format(json.dumps(report, sort_keys=True)))
    write_report(args.out, report)
    print(json.dumps({"schema": report["schema"], "train_rows": report["train_report"]["rows"], "heldout_rows": report["heldout_report"]["rows"]}, sort_keys=True))


if __name__ == "__main__":
    main()
