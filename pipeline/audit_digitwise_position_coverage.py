#!/usr/bin/env python3
"""Report DRS digit-position and local-transition support without changing data.

This is a distribution diagnostic, not a quality score.  It identifies when a
held-out digit, position, or local arithmetic context was never present in the
training transition inputs, which otherwise makes an apparent execution OOD
failure ambiguous.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train"))
from digitwise_protocol import parse_state


def _add_state(state, target):
    if state is None or state["z"]:
        return
    width, position = int(state["w"]), int(state["p"])
    left, right = state["a"][position], state["b"][position]
    target["events"] += 1
    target["marginal"][(width, position, "a", left)] += 1
    target["marginal"][(width, position, "b", right)] += 1
    target["local"][(width, position, state["op"], int(state["c"]), left, right)] += 1


def _empty_counts():
    return {"events": 0, "marginal": Counter(), "local": Counter()}


def _states_from_episode(episode):
    yield parse_state(episode["initial_state"])
    for line in episode["expected_states"]:
        yield parse_state(line)


def _serialize_marginal(counter):
    return {
        "w{}p{}{}".format(width, position, tape): {str(digit): counter[(width, position, tape, str(digit))] for digit in range(10)}
        for width, position, tape in sorted({(key[0], key[1], key[2]) for key in counter})
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--episodes", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = Path(args.out)
    if out.exists():
        raise SystemExit("refusing to overwrite existing report: {}".format(out))

    train = _empty_counts()
    with open(args.data) as source:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("kind") == "transition":
                _add_state(parse_state(row.get("state", "")), train)

    heldout = defaultdict(_empty_counts)
    with open(args.episodes) as source:
        for line in source:
            if not line.strip():
                continue
            episode = json.loads(line)
            for branch in (episode, episode["counterfactual"]):
                for state in _states_from_episode(branch):
                    _add_state(state, heldout[episode["split"]])

    missing_train_digits = []
    train_positions = sorted({(width, position, tape) for width, position, tape, _digit in train["marginal"]})
    for width, position, tape in train_positions:
        absent = [digit for digit in range(10) if train["marginal"][(width, position, tape, str(digit))] == 0]
        if absent:
            missing_train_digits.append({"width": width, "position": position, "tape": tape, "digits": absent})

    heldout_summary = {}
    for split, counts in sorted(heldout.items()):
        unseen_digit_events, unseen_local_events = 0, 0
        for (width, position, tape, digit), count in counts["marginal"].items():
            if train["marginal"][(width, position, tape, digit)] == 0:
                unseen_digit_events += count
        for context, count in counts["local"].items():
            if train["local"][context] == 0:
                unseen_local_events += count
        heldout_summary[split] = {
            "transition_events": counts["events"],
            "unseen_train_digit_position_events": unseen_digit_events,
            "unseen_train_exact_local_context_events": unseen_local_events,
            "marginal_digit_counts": _serialize_marginal(counts["marginal"]),
        }

    report = {
        "audit": "digitwise_position_coverage_v1",
        "data": str(Path(args.data).resolve()),
        "episodes": str(Path(args.episodes).resolve()),
        "train_transition_events": train["events"],
        "train_missing_digit_position_cells": missing_train_digits,
        "train_marginal_digit_counts": _serialize_marginal(train["marginal"]),
        "heldout": heldout_summary,
        "claim_boundary": "Coverage diagnostic only; it neither proves nor disproves model reasoning.",
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "train_missing_digit_position_cells": len(missing_train_digits),
        "heldout": {
            split: {
                "unseen_train_digit_position_events": value["unseen_train_digit_position_events"],
                "unseen_train_exact_local_context_events": value["unseen_train_exact_local_context_events"],
            }
            for split, value in heldout_summary.items()
        },
    }, sort_keys=True))


if __name__ == "__main__":
    main()
