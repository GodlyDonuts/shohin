#!/usr/bin/env python3
"""Independently audit the digitwise recurrent curriculum and heldout split."""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train"))
from digitwise_protocol import (apply_microstep, canonical_state, digit_prompt, final_prompt,
                                initial_state, microstep_prompt, parse_state, state_answer, state_digit)
from generate_digitwise_recurrent_v1 import _episode_prompts, normalized


WORD = re.compile(r"\w+")


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ngrams(text, width=13):
    words = normalized(text).split()
    return {" ".join(words[index:index + width]) for index in range(max(0, len(words) - width + 1))}


def audit_train_row(row):
    required = {"question", "completion_prompt", "response", "training_group", "kind", "state", "prompt_style"}
    if any(not str(row.get(field, "")).strip() for field in required):
        raise ValueError("missing train field")
    if row["question"] != row["completion_prompt"] or row["training_group"] != "digitwise_recurrent":
        raise ValueError("invalid train row surface")
    state = parse_state(row["state"])
    if state is None:
        raise ValueError("invalid train state")
    style = row["prompt_style"]
    if row["kind"] == "transition":
        expected = apply_microstep(state)
        expected_line = canonical_state(expected)
        if row["question"] != microstep_prompt(state, style=style) or row["response"] != expected_line:
            raise ValueError("invalid transition completion")
        if row.get("expected_state") != expected_line:
            raise ValueError("missing transition witness")
        return
    if row["kind"] == "digit":
        position = int(row.get("digit_index", -1))
        expected = state_digit(state, position)
        if row["question"] != digit_prompt(state, position, style=style) or row["response"] != "digit={}".format(expected):
            raise ValueError("invalid digit completion")
        return
    if row["kind"] == "final":
        expected = state_answer(state)
        if row["question"] != final_prompt(state, style=style) or row["response"] != "answer={}".format(expected):
            raise ValueError("invalid final completion")
        return
    raise ValueError("unknown train row kind")


def _audit_episode_surface(episode):
    required = {"id", "split", "prompt_style", "operation", "width", "left", "right", "initial_state", "expected_states", "expected_answer"}
    if any(field not in episode for field in required):
        raise ValueError("missing episode field")
    state = parse_state(episode["initial_state"])
    if state is None or state["op"] != episode["operation"] or state["w"] != int(episode["width"]):
        raise ValueError("invalid episode initial state")
    if canonical_state(initial_state(episode["operation"], int(episode["left"]), int(episode["right"]), int(episode["width"]))) != episode["initial_state"]:
        raise ValueError("initial state does not match operands")
    if len(episode["expected_states"]) != state["w"]:
        raise ValueError("incorrect episode length")
    for expected_line in episode["expected_states"]:
        expected = parse_state(expected_line)
        if expected is None or expected != apply_microstep(state):
            raise ValueError("invalid episode transition")
        state = expected
    if not state["z"] or int(episode["expected_answer"]) != state_answer(state):
        raise ValueError("invalid episode answer")
    return _episode_prompts(episode)


def audit_episode(episode):
    prompts = _audit_episode_surface(episode)
    counterfactual = episode.get("counterfactual")
    if not isinstance(counterfactual, dict):
        raise ValueError("missing counterfactual episode")
    prompts.extend(_audit_episode_surface(counterfactual))
    if counterfactual["operation"] != episode["operation"] or counterfactual["width"] != episode["width"]:
        raise ValueError("counterfactual changed protocol")
    changed = int(counterfactual["left"] != episode["left"]) + int(counterfactual["right"] != episode["right"])
    if changed != 1 or int(counterfactual["expected_answer"]) == int(episode["expected_answer"]):
        raise ValueError("invalid counterfactual intervention")
    return prompts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--episodes", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    data, episodes_path, out = Path(args.data), Path(args.episodes), Path(args.out)
    if not data.is_file() or not episodes_path.is_file():
        raise SystemExit("missing data or episodes")
    if out.exists():
        raise SystemExit("refusing existing audit output")

    train_rows = [json.loads(line) for line in data.read_text().splitlines() if line.strip()]
    episodes = [json.loads(line) for line in episodes_path.read_text().splitlines() if line.strip()]
    failures, examples = collections.Counter(), []
    duplicate_questions, seen_questions = 0, set()
    for index, row in enumerate(train_rows, 1):
        question = normalized(row.get("question", ""))
        duplicate_questions += int(question in seen_questions)
        seen_questions.add(question)
        try:
            audit_train_row(row)
        except (KeyError, TypeError, ValueError) as exc:
            failures[str(exc)] += 1
            if len(examples) < 8:
                examples.append({"kind": "train", "line": index, "error": str(exc)})

    heldout_prompts, regimes = [], collections.Counter()
    for index, episode in enumerate(episodes, 1):
        regimes[episode.get("split", "missing")] += 1
        try:
            heldout_prompts.extend(audit_episode(episode))
        except (KeyError, TypeError, ValueError) as exc:
            failures[str(exc)] += 1
            if len(examples) < 8:
                examples.append({"kind": "heldout", "line": index, "error": str(exc)})

    train_prompts = {normalized(row.get("question", "")) for row in train_rows if row.get("question")}
    heldout_normalized = {normalized(prompt) for prompt in heldout_prompts}
    train_gram_rows = {}
    for index, row in enumerate(train_rows, 1):
        for gram in ngrams(row.get("question", "")):
            train_gram_rows.setdefault(gram, {"line": index, "kind": row.get("kind")})
    heldout_gram_rows = {}
    for index, prompt in enumerate(heldout_prompts, 1):
        for gram in ngrams(prompt):
            heldout_gram_rows.setdefault(gram, {"prompt_index": index})
    shared_grams = sorted(set(train_gram_rows) & set(heldout_gram_rows))
    result = {
        "audit": "digitwise_recurrent_v1_protocol_audit",
        "data": str(data.resolve()),
        "data_sha256": sha256_file(data),
        "episodes": str(episodes_path.resolve()),
        "episodes_sha256": sha256_file(episodes_path),
        "train_rows": len(train_rows),
        "heldout_episodes": len(episodes),
        "counterfactual_pairs": len(episodes),
        "heldout_controller_prompts": len(heldout_prompts),
        "invalid_rows_or_episodes": sum(failures.values()),
        "failures": dict(sorted(failures.items())),
        "examples": examples,
        "duplicate_normalized_train_questions": duplicate_questions,
        "regimes": dict(sorted(regimes.items())),
        "overlap": {
            "exact_prompt_hits": len(train_prompts & heldout_normalized),
            "ngram13_hits": len(shared_grams),
            "ngram13_examples": [
                {"ngram": gram, "train": train_gram_rows[gram], "heldout": heldout_gram_rows[gram]}
                for gram in shared_grams[:8]
            ],
        },
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))
    if result["invalid_rows_or_episodes"] or duplicate_questions or result["overlap"]["exact_prompt_hits"] or result["overlap"]["ngram13_hits"]:
        raise SystemExit("digitwise recurrent protocol audit failed")


if __name__ == "__main__":
    main()
