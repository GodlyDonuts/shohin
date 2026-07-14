#!/usr/bin/env python3
"""Build an exact-carrier semantic-basis transport curriculum.

This v2 candidate strengthens v1's state contract: compile, reflect, and
update responses are *only* ``ledger:P=<int>;Q=<int>``.  A later transport
controller can therefore forward an exact model emission without extracting or
canonicalizing a substring.  Difference and sum consume the updated ledger, so
one model-authored state must feed two independent downstream operations.

This is data generation only. It does not start training or grant any context
or reasoning claim.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
from collections import Counter
from pathlib import Path


WORD = re.compile(r"\w+")
PHASES = ("compile", "reflect", "update", "difference", "sum")
TRAIN_DOMAINS = ("workshop", "orchard", "foundry", "classroom", "warehouse", "greenhouse")
HELDOUT_DOMAINS = ("harbor", "observatory", "clinic", "theater", "archive", "shipyard")
TRAIN_LABELS = (("amber", "cobalt"), ("cedar", "flint"), ("violet", "ochre"))
HELDOUT_LABELS = (("north", "south"), ("silver", "basalt"), ("lilac", "ivory"))


def normalized_question(text: str) -> str:
    return " ".join(WORD.findall(text.lower()))


def ledger(p: int, q: int) -> str:
    return "ledger:P={};Q={}".format(p, q)


def settings(heldout: bool) -> dict:
    if heldout:
        return {
            "split": "heldout",
            "p": (201, 299),
            "q": (201, 299),
            "delta": (11, 29),
            "domains": HELDOUT_DOMAINS,
            "labels": HELDOUT_LABELS,
        }
    return {
        "split": "train",
        "p": (10, 199),
        "q": (10, 199),
        "delta": (1, 9),
        "domains": TRAIN_DOMAINS,
        "labels": TRAIN_LABELS,
    }


def question(phase: str, p: int, q: int, delta: int, domain: str, labels: tuple[str, str], heldout: bool) -> str:
    primary, secondary = labels
    initial = ledger(p, q)
    updated = ledger(p + delta, q)
    if phase == "compile":
        if heldout:
            return (
                "An {} inventory calls {} its operational first field and {} its operational second field. "
                "It records {} for {} and {} for {}. Return only the exact portable ledger."
            ).format(domain, primary, secondary, p, primary, q, secondary)
        return (
            "A {} report names {} the primary quantity and {} the secondary quantity. "
            "It records {}={} and {}={}. Return only the exact portable ledger."
        ).format(domain, primary, secondary, primary, p, secondary, q)
    if phase == "reflect":
        if heldout:
            return (
                "The source note from an {} will disappear. Its first role, {}, contains {}; "
                "its second role, {}, contains {}. Emit only the exact state needed later."
            ).format(domain, primary, p, secondary, q)
        return (
            "Before a {} record is discarded, its primary {} is {} and its secondary {} is {}. "
            "Emit only the exact retained state."
        ).format(domain, primary, p, secondary, q)
    if phase == "update":
        if heldout:
            return (
                "No source note remains. The exact transport string produced earlier is:\n{}\n"
                "Advance P by {} and leave Q unchanged. Emit only the replacement transport string."
            ).format(initial, delta)
        return (
            "The original report is unavailable. Exact previous model state:\n{}\n"
            "Increase P by {} while preserving Q. Return only the next exact ledger."
        ).format(initial, delta)
    if phase == "difference":
        if heldout:
            return (
                "The source has been erased. Received carry string:\n{}\n"
                "Emit only answer=<integer> containing P less Q."
            ).format(updated)
        return (
            "The original report is unavailable. Exact previous model state:\n{}\n"
            "Return only answer=<integer> for P minus Q."
        ).format(updated)
    if phase == "sum":
        if heldout:
            return (
                "The original description cannot be consulted. Received carry string:\n{}\n"
                "Emit only answer=<integer> containing P added to Q."
            ).format(updated)
        return (
            "The original report is unavailable. Exact previous model state:\n{}\n"
            "Return only answer=<integer> for P plus Q."
        ).format(updated)
    raise ValueError("unknown phase: {}".format(phase))


def response(phase: str, p: int, q: int, delta: int) -> tuple[str, str]:
    initial = ledger(p, q)
    updated = ledger(p + delta, q)
    if phase in {"compile", "reflect"}:
        return initial, initial
    if phase == "update":
        return updated, updated
    if phase == "difference":
        answer = p + delta - q
        return "answer={}".format(answer), str(answer)
    if phase == "sum":
        answer = p + delta + q
        return "answer={}".format(answer), str(answer)
    raise ValueError("unknown phase: {}".format(phase))


def make_row(split: str, episode_id: str, phase: str, p: int, q: int, delta: int,
             domain: str, labels: tuple[str, str], heldout: bool) -> dict:
    target, answer = response(phase, p, q, delta)
    return {
        "schema": "semantic_basis_transport_v2",
        "source": "semantic_basis_transport_v2_candidate",
        "training_group": "semantic_basis_transport_exact_carrier",
        "split": split,
        "episode_id": episode_id,
        "phase": phase,
        "question": question(phase, p, q, delta, domain, labels, heldout),
        "response": target,
        "answer": answer,
        "primary_value": p,
        "secondary_value": q,
        "delta": delta,
        "domain": domain,
        "primary_label": labels[0],
        "secondary_label": labels[1],
        "expected_ledger": ledger(p, q),
        "expected_next_ledger": ledger(p + delta, q),
    }


def build_split(episodes: int, seed: int, heldout: bool) -> list[dict]:
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    cfg = settings(heldout)
    p_values = cfg["p"][1] - cfg["p"][0] + 1
    q_values = cfg["q"][1] - cfg["q"][0] + 1
    if episodes > p_values * q_values:
        raise ValueError("requested episodes exceed unique P/Q prompt capacity")
    rng = random.Random(seed)
    pairs = [(p, q) for p in range(cfg["p"][0], cfg["p"][1] + 1) for q in range(cfg["q"][0], cfg["q"][1] + 1)]
    rng.shuffle(pairs)
    # A source P/Q pair being unique does not make P+delta/Q unique. Keeping
    # each Q's delta fixed makes the post-update carrier injective within Q,
    # while distinct Q values keep it injective across the whole split.
    delta_by_q = {q: rng.randint(*cfg["delta"]) for q in range(cfg["q"][0], cfg["q"][1] + 1)}
    rows, prompts = [], set()
    for index, (p, q) in enumerate(pairs[:episodes]):
        delta = delta_by_q[q]
        domain = rng.choice(cfg["domains"])
        labels = rng.choice(cfg["labels"])
        episode_id = "{}-{:06d}".format(cfg["split"], index)
        candidate = [make_row(cfg["split"], episode_id, phase, p, q, delta, domain, labels, heldout) for phase in PHASES]
        normalized = [normalized_question(item["question"]) for item in candidate]
        if len(set(normalized)) != len(normalized) or any(item in prompts for item in normalized):
            raise RuntimeError("unexpected duplicate semantic-basis v2 prompt")
        prompts.update(normalized)
        rows.extend(candidate)
    rng.shuffle(rows)
    return rows


def write_jsonl(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".partial")
    if path.exists() or temporary.exists():
        raise SystemExit("refusing to overwrite semantic-basis v2 artifact: {}".format(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("w") as output:
        for item in rows:
            output.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(temporary, path)


def summary(rows: list[dict]) -> dict:
    return {
        "rows": len(rows),
        "episodes": len({item["episode_id"] for item in rows}),
        "phases": dict(sorted(Counter(item["phase"] for item in rows).items())),
        "exact_carrier_rows": sum(item["phase"] in {"compile", "reflect", "update"} and item["response"].startswith("ledger:") for item in rows),
        "answer_rows": sum(item["phase"] in {"difference", "sum"} and item["response"].startswith("answer=") for item in rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-out", required=True)
    parser.add_argument("--heldout-out", required=True)
    parser.add_argument("--train-episodes", type=int, default=30_000)
    parser.add_argument("--heldout-episodes", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=20260713)
    args = parser.parse_args()
    train = build_split(args.train_episodes, args.seed, heldout=False)
    heldout = build_split(args.heldout_episodes, args.seed + 1, heldout=True)
    train_prompts = {normalized_question(item["question"]) for item in train}
    heldout_prompts = {normalized_question(item["question"]) for item in heldout}
    if train_prompts & heldout_prompts:
        raise RuntimeError("train/heldout normalized prompt overlap")
    write_jsonl(args.train_out, train)
    write_jsonl(args.heldout_out, heldout)
    print(json.dumps({"schema": "semantic_basis_transport_v2", "train": summary(train), "heldout": summary(heldout)}, sort_keys=True))


if __name__ == "__main__":
    main()
