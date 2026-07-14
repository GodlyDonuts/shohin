#!/usr/bin/env python3
"""Controller-only tests for exact semantic-basis carrier forwarding."""
import re

from eval_semantic_basis_transport import evaluate_pair, model_prompt
from semantic_basis_transport_controller import rollout_episode

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))
from generate_semantic_basis_transport_v2 import build_split  # noqa: E402


LEDGER = re.compile(r"ledger:P=(-?\d+);Q=(-?\d+)")


def split_episodes(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["episode_id"], []).append(row)
    return list(grouped.values())


def make_oracle(episodes):
    source_answers = {}
    for rows in episodes:
        for row in rows:
            if row["phase"] in {"compile", "reflect"}:
                source_answers[row["question"]] = row["response"]
    prompts = []

    def ask(prompt, max_new):
        prompts.append(prompt)
        if prompt in source_answers:
            return source_answers[prompt]
        match = LEDGER.search(prompt)
        assert match, prompt
        p, q = map(int, match.groups())
        if "Advance P by" in prompt:
            delta = int(re.search(r"Advance P by (\d+)", prompt).group(1))
            return "ledger:P={};Q={}".format(p + delta, q)
        if "P less Q" in prompt or "P minus Q" in prompt:
            return "answer={}".format(p - q)
        if "P added to Q" in prompt or "P plus Q" in prompt:
            return "answer={}".format(p + q)
        raise AssertionError(prompt)

    return ask, prompts


def main():
    assert model_prompt("state", "qa") == "Question: state\nAnswer:"
    assert model_prompt("state", "direct") == "state"
    episodes = split_episodes(build_split(4, 29, True))
    ask, prompts = make_oracle(episodes)
    single = rollout_episode(episodes[0], ask)
    assert single["strict_success"], single
    assert single["compile"]["exact_response"] in single["update"]["prompt"]
    assert single["update"]["exact_response"] in single["consumers"]["difference"]["prompt"]
    assert single["update"]["exact_response"] in single["consumers"]["sum"]["prompt"]

    def malformed(prompt, max_new):
        if prompt == single["compile"]["prompt"]:
            return "prefix " + single["compile"]["expected"]
        return ask(prompt, max_new)

    rejected = rollout_episode(episodes[0], malformed)
    assert not rejected["strict_success"]
    assert rejected["update"] is None

    pair = evaluate_pair(episodes[0], episodes[1], ask)
    assert pair["normal_both_strict"], pair
    assert pair["model_authored_interchange_success"], pair
    assert not pair["zero_recreates_original"], pair
    assert pair["mismatch_success_and_rejects_original"], pair
    assert pair["strict_causal_pass"], pair
    assert pair["left_receives_right"]["carrier"] == pair["right_normal"]["update"]["exact_response"]
    assert pair["left_mismatch"]["carrier"] != pair["left_normal"]["update"]["exact_response"]
    assert prompts
    print("semantic basis transport exact-carrier controller checks: passed")


if __name__ == "__main__":
    main()
