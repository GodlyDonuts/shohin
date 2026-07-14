#!/usr/bin/env python3
"""Generate a conditional semantic-basis transport curriculum.

This candidate deliberately isolates a small language-to-state primitive from
the multiplication and base-conversion failures exposed by V10A.  A model must
compile a two-value record into an exact token ledger, update that ledger after
the source is gone, and answer two different readouts from the same state.  It
is data preparation only: no job or model path references this file.
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
TRAIN_PLACES = ("workshop", "orchard", "foundry", "classroom", "warehouse", "greenhouse")
HELDOUT_PLACES = ("harbor", "observatory", "clinic", "theater", "archive", "shipyard")
TRAIN_LABELS = (("amber", "cobalt"), ("cedar", "flint"), ("violet", "ochre"))
HELDOUT_LABELS = (("north", "south"), ("silver", "basalt"), ("lilac", "ivory"))


def normalized_question(text: str) -> str:
    return " ".join(WORD.findall(text.lower()))


def ledger(p: int, q: int) -> str:
    return "ledger:P={};Q={}".format(p, q)


def render_response(phase: str, p: int, q: int, delta: int) -> tuple[str, str]:
    """Render every supervised target from structured values, never a teacher."""
    state = ledger(p, q)
    if phase in ("compile", "reflect"):
        return "<think>Primary P={} and secondary Q={} are the retained values.</think>\n{}".format(p, q, state), state
    if phase == "update":
        updated = ledger(p + delta, q)
        return (
            "<think>Update P by {}: {}+{}={}; preserve Q={}. </think>\n{}".format(delta, p, delta, p + delta, q, updated),
            updated,
        )
    if phase == "difference":
        answer = p - q
        return "<think>Use P={} and Q={}: {}-{}={}. </think>\nThe answer is {}.".format(p, q, p, q, answer, answer), str(answer)
    if phase == "sum":
        answer = p + q
        return "<think>Use P={} and Q={}: {}+{}={}. </think>\nThe answer is {}.".format(p, q, p, q, answer, answer), str(answer)
    raise ValueError("unknown phase: {}".format(phase))


def render_question(
    phase: str,
    p: int,
    q: int,
    delta: int,
    place: str,
    labels: tuple[str, str],
    heldout: bool,
) -> str:
    primary, secondary = labels
    state = ledger(p, q)
    if not heldout:
        if phase == "compile":
            return (
                "A {} report calls {} the primary quantity and {} the secondary quantity. "
                "It records {}={} and {}={}. Write the retained state exactly as ledger:P=<primary>;Q=<secondary>."
            ).format(place, primary, secondary, primary, p, secondary, q)
        if phase == "reflect":
            return (
                "Before any calculation, a {} record says the primary {} has value {} and the secondary {} has value {}. "
                "If interrupted now, emit the compact ledger needed for a later update or comparison."
            ).format(place, primary, p, secondary, q)
        if phase == "update":
            return "Given {}. Increase P by {} while preserving Q. Emit only the updated ledger.".format(state, delta)
        if phase == "difference":
            return "Given {}. What is P minus Q? Return only the integer.".format(state)
        if phase == "sum":
            return "Given {}. What is P plus Q? Return only the integer.".format(state)
    else:
        if phase == "compile":
            return (
                "An {} inventory identifies {} as its operational first field and {} as its operational second field. "
                "Their recorded amounts are {} for {} and {} for {}. Produce the canonical two-field ledger."
            ).format(place, primary, secondary, p, primary, q, secondary)
        if phase == "reflect":
            return (
                "Imagine the source note from a {} will disappear. Its first role, {}, contains {}; its second role, {}, contains {}. "
                "State exactly the portable ledger that can answer future changes and queries."
            ).format(place, primary, p, secondary, q)
        if phase == "update":
            return "The source note is unavailable. Carry forward {} and add {} to its P field only. Return the next ledger.".format(state, delta)
        if phase == "difference":
            return "The only surviving record is {}. Report the signed excess of P over Q as an integer.".format(state)
        if phase == "sum":
            return "With no original record, use {} to report the combined quantity P+Q as an integer.".format(state)
    raise ValueError("unknown phase: {}".format(phase))


def row(
    split: str,
    episode_id: str,
    phase: str,
    p: int,
    q: int,
    delta: int,
    place: str,
    labels: tuple[str, str],
    heldout: bool,
) -> dict:
    response, answer = render_response(phase, p, q, delta)
    return {
        "schema": "semantic_basis_transport_v1",
        "source": "semantic_basis_transport_v1_candidate",
        "training_group": "semantic_basis_transport",
        "split": split,
        "episode_id": episode_id,
        "phase": phase,
        "question": render_question(phase, p, q, delta, place, labels, heldout),
        "response": response,
        "answer": answer,
        "primary_value": p,
        "secondary_value": q,
        "delta": delta,
        "expected_ledger": ledger(p, q),
    }


def settings(heldout: bool) -> dict:
    if heldout:
        return {
            "p": (201, 299),
            "q": (201, 299),
            "delta": (11, 29),
            "places": HELDOUT_PLACES,
            "labels": HELDOUT_LABELS,
            "split": "heldout",
        }
    return {
        # Difference and sum prompts only contain P and Q, so their value
        # space itself must support the full 30k unique-episode admission.
        "p": (10, 199),
        "q": (10, 199),
        "delta": (1, 9),
        "places": TRAIN_PLACES,
        "labels": TRAIN_LABELS,
        "split": "train",
    }


def build_split(episodes: int, seed: int, heldout: bool) -> list[dict]:
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    cfg = settings(heldout)
    p_values = cfg["p"][1] - cfg["p"][0] + 1
    q_values = cfg["q"][1] - cfg["q"][0] + 1
    # Difference and sum prompts depend only on P/Q. Each episode must own a
    # distinct pair or the independent prompt-identity audit would reject it.
    if episodes > p_values * q_values:
        raise ValueError("requested episodes exceed unique P/Q prompt capacity")
    rng = random.Random(seed)
    rows: list[dict] = []
    questions: set[str] = set()
    pairs = [(p, q) for p in range(cfg["p"][0], cfg["p"][1] + 1) for q in range(cfg["q"][0], cfg["q"][1] + 1)]
    rng.shuffle(pairs)
    for episode_index, (p, q) in enumerate(pairs[:episodes]):
        delta = rng.randint(*cfg["delta"])
        place = rng.choice(cfg["places"])
        labels = rng.choice(cfg["labels"])
        episode_id = "{}-{:06d}".format(cfg["split"], episode_index)
        candidate = [row(cfg["split"], episode_id, phase, p, q, delta, place, labels, heldout) for phase in PHASES]
        normalized = [normalized_question(item["question"]) for item in candidate]
        if len(set(normalized)) != len(normalized) or any(item in questions for item in normalized):
            raise RuntimeError("unexpected duplicate semantic-basis prompt")
        questions.update(normalized)
        rows.extend(candidate)
    rng.shuffle(rows)
    return rows


def write_jsonl(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".partial")
    if path.exists() or temporary.exists():
        raise SystemExit("refusing to overwrite semantic-basis artifact: {}".format(path))
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
        "all_have_think": all(item["response"].startswith("<think>") for item in rows),
        "all_groups": sorted(set(item["training_group"] for item in rows)),
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
    train_questions = {normalized_question(item["question"]) for item in train}
    heldout_questions = {normalized_question(item["question"]) for item in heldout}
    if train_questions & heldout_questions:
        raise RuntimeError("train/heldout normalized question overlap")
    write_jsonl(args.train_out, train)
    write_jsonl(args.heldout_out, heldout)
    print(json.dumps({"schema": "semantic_basis_transport_v1", "train": summary(train), "heldout": summary(heldout)}, sort_keys=True))


if __name__ == "__main__":
    main()
