#!/usr/bin/env python3
"""Independent integrity audit for the dual-code reversible curriculum."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train"))
from digitwise_protocol import apply_microstep, state_answer
from dual_code_reversible_protocol import (
    codebook_from_record, encode_state, invert_microstep, make_codebook, parse_state,
)


WORD = re.compile(r"\w+")
STATE_LINE = re.compile(r"dcr:(?:A\||B~)[^\n]+")
REQUIRED_ROW = {
    "question", "completion_prompt", "response", "source", "training_group", "kind", "episode_id",
    "split", "width", "operation", "transition_index", "code_seed", "codebook_vocabulary",
    "prompt_style",
}


def normalized(text: str) -> str:
    return " ".join(WORD.findall(str(text).lower()))


def grams(text: str, width: int = 13):
    tokens = normalized(text).split()
    return {tuple(tokens[index:index + width]) for index in range(max(0, len(tokens) - width + 1))}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def extract_state(text: str, book):
    matches = STATE_LINE.findall(str(text))
    lines = STATE_LINE.finditer(str(text))
    found = [match.group(0) for match in lines]
    if len(matches) != 1 or len(found) != 1:
        raise ValueError("expected exactly one coded state in prompt")
    state = parse_state(found[0], book)
    if state is None:
        raise ValueError("coded state does not parse under declared codebook")
    return state


def audit_row(row):
    if set(row) != REQUIRED_ROW:
        raise ValueError("invalid train row keys")
    if row["question"] != row["completion_prompt"] or not str(row["response"]):
        raise ValueError("malformed train prompt or response")
    vocabulary = str(row["codebook_vocabulary"])
    if str(row["prompt_style"]) != vocabulary:
        raise ValueError("row prompt style must be bound to its codebook vocabulary")
    a_book = make_codebook(str(row["code_seed"]), "A", vocabulary)
    b_book = make_codebook(str(row["code_seed"]), "B", vocabulary)
    kind = str(row["kind"])
    if kind == "forward_a":
        state = extract_state(row["question"], a_book)
        expected = encode_state(apply_microstep(state), a_book)
    elif kind == "a_to_b":
        state = extract_state(row["question"], a_book)
        expected = encode_state(state, b_book)
    elif kind == "reverse_b":
        state = extract_state(row["question"], b_book)
        expected = encode_state(invert_microstep(state), b_book)
    elif kind == "b_to_a":
        state = extract_state(row["question"], b_book)
        expected = encode_state(state, a_book)
    elif kind == "readout":
        state = extract_state(row["question"], a_book)
        expected = "answer={}".format(state_answer(state))
    else:
        raise ValueError("unknown DCRD task kind")
    if str(row["response"]) != expected:
        raise ValueError("row response disagrees with independently recomputed target")


BASE_EPISODE_KEYS = {
    "id", "split", "width", "operation", "left", "right", "code_seed", "codebooks", "initial_a",
    "expected_a_states", "expected_b_states", "expected_answer", "prompt_style",
}


def audit_single_episode(episode):
    if set(episode) != BASE_EPISODE_KEYS:
        raise ValueError("invalid single-episode keys")
    a_book = codebook_from_record(episode["codebooks"]["A"])
    b_book = codebook_from_record(episode["codebooks"]["B"])
    if a_book.vocabulary != "heldout" or b_book.vocabulary != "heldout":
        raise ValueError("heldout episode must use heldout-only codebook aliases")
    if episode["prompt_style"] != "heldout":
        raise ValueError("heldout episode must use heldout-only prompt style")
    if set(a_book.aliases) & set(make_codebook(episode["code_seed"], "A", "train").aliases):
        raise ValueError("heldout A aliases overlap train alias vocabulary")
    if set(b_book.aliases) & set(make_codebook(episode["code_seed"], "B", "train").aliases):
        raise ValueError("heldout B aliases overlap train alias vocabulary")
    current = parse_state(episode["initial_a"], a_book)
    if current is None:
        raise ValueError("invalid heldout initial A state")
    if len(episode["expected_a_states"]) != int(episode["width"]) or len(episode["expected_b_states"]) != int(episode["width"]):
        raise ValueError("heldout state sequence length does not match width")
    for a_line, b_line in zip(episode["expected_a_states"], episode["expected_b_states"]):
        expected = apply_microstep(current)
        a_state, b_state = parse_state(a_line, a_book), parse_state(b_line, b_book)
        if a_state != expected or b_state != expected:
            raise ValueError("heldout state does not match semantic transition")
        if invert_microstep(b_state) != current:
            raise ValueError("heldout B inverse target does not close")
        current = expected
    if state_answer(current) != int(episode["expected_answer"]):
        raise ValueError("heldout answer does not match terminal state")
    return current


def audit_episode(episode):
    if set(episode) != BASE_EPISODE_KEYS | {"counterfactual"}:
        raise ValueError("invalid heldout episode keys")
    normal = {key: episode[key] for key in BASE_EPISODE_KEYS}
    counterfactual = episode["counterfactual"]
    if not isinstance(counterfactual, dict):
        raise ValueError("heldout counterfactual must be an object")
    audit_single_episode(normal)
    audit_single_episode(counterfactual)
    if int(counterfactual["expected_answer"]) == int(episode["expected_answer"]):
        raise ValueError("heldout counterfactual does not change answer")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True)
    parser.add_argument("--heldout", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    report_path = Path(args.report)
    if report_path.exists():
        raise SystemExit("refusing to overwrite existing report: {}".format(report_path))

    train_prompts, heldout_prompts, heldout_grams = set(), set(), set()
    invalid_rows = 0
    kinds = Counter()
    with open(args.train) as source:
        for line in source:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                audit_row(row)
                prompt = normalized(row["completion_prompt"])
                if prompt in train_prompts:
                    raise ValueError("duplicate normalized train prompt")
                train_prompts.add(prompt)
                kinds[str(row["kind"])] += 1
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                invalid_rows += 1

    invalid_episodes = 0
    heldout_count = 0
    with open(args.heldout) as source:
        for line in source:
            if not line.strip():
                continue
            try:
                episode = json.loads(line)
                audit_episode(episode)
                heldout_count += 1
                from generate_dual_code_reversible_v1 import controller_prompts
                for prompt in controller_prompts(episode):
                    normalized_prompt = normalized(prompt)
                    if normalized_prompt in heldout_prompts:
                        raise ValueError("duplicate normalized heldout controller prompt")
                    heldout_prompts.add(normalized_prompt)
                    heldout_grams.update(grams(prompt))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                invalid_episodes += 1

    exact_hits = len(train_prompts & heldout_prompts)
    gram_hits = 0
    with open(args.train) as source:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            gram_hits += sum(1 for gram in grams(row["completion_prompt"]) if gram in heldout_grams)

    result = {
        "audit": "dual_code_reversible_v1",
        "train": args.train,
        "heldout": args.heldout,
        "train_sha256": sha256_file(args.train),
        "heldout_sha256": sha256_file(args.heldout),
        "valid_train_rows": len(train_prompts),
        "valid_heldout_episodes": heldout_count,
        "invalid_train_rows": invalid_rows,
        "invalid_heldout_episodes": invalid_episodes,
        "duplicate_train_prompts": 0,
        "duplicate_heldout_prompts": 0,
        "train_heldout_exact_prompt_hits": exact_hits,
        "train_heldout_13gram_hits": gram_hits,
        "train_kinds": dict(sorted(kinds.items())),
        "claim_boundary": "Data admission only; no model or reasoning result is implied.",
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
