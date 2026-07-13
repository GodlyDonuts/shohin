#!/usr/bin/env python3
"""Independent admission audit for token-native delta-ledger data."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline"))
sys.path.insert(0, str(ROOT / "train"))
from digitwise_factor_protocol import canonical_tape, initial_tape
from generate_token_native_ledger_v1 import ngrams, normalized, prompts_for_episode
from token_native_ledger_protocol import (
    canonical_delta,
    context_key,
    expected_answer,
    expected_delta,
    final_prompt,
    initial_delta,
    parse_delta,
    transition_prompt,
)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def required_contexts():
    result = set()
    for width in (4, 6):
        for operation in ("add", "sub"):
            for position in range(width):
                for carry in ((0,) if position == 0 else (0, 1)):
                    for left in range(10):
                        for right in range(10):
                            if operation == "sub" and position == width - 1:
                                if left < right or (left == right and carry):
                                    continue
                            result.add((width, operation, position, carry, left, right))
    return result


def tape_for(item):
    tape = initial_tape(item["operation"], int(item["left"]), int(item["right"]), int(item["width"]))
    if item.get("tape") != canonical_tape(tape):
        raise ValueError("tape mismatch")
    return tape


def audit_transition(row):
    tape = tape_for(row)
    prior, expected = parse_delta(row.get("prior_delta", "")), parse_delta(row.get("expected_delta", ""))
    if prior is None or expected is None or row.get("response") != canonical_delta(expected):
        raise ValueError("invalid transition carrier")
    position = int(prior["p"])
    if position < 0 or position >= int(tape["w"]):
        raise ValueError("prior position outside tape")
    if expected != expected_delta(tape, position, prior["c"]):
        raise ValueError("transition is not the legal next local delta")
    if int(row.get("transition_index")) != position:
        raise ValueError("transition index mismatch")
    if row.get("completion_prompt") != transition_prompt(tape, prior, style=row.get("prompt_style")):
        raise ValueError("transition prompt mismatch")
    return (int(tape["w"]), tape["op"], position, int(prior["c"]), int(tape["a"][position]), int(tape["b"][position]))


def audit_final(row):
    tape = tape_for(row)
    deltas = row.get("deltas")
    if not isinstance(deltas, list) or len(deltas) != int(tape["w"]):
        raise ValueError("invalid final ledger length")
    parsed = [parse_delta(item) for item in deltas]
    if any(item is None for item in parsed):
        raise ValueError("invalid final ledger carrier")
    carry = 0
    for position, item in enumerate(parsed):
        if item != expected_delta(tape, position, carry):
            raise ValueError("final ledger has incorrect delta")
        carry = item["c"]
    answer = expected_answer(tape)
    if row.get("response") != "answer={}".format(answer) or int(row.get("expected_answer")) != answer:
        raise ValueError("invalid final answer")
    if row.get("completion_prompt") != final_prompt(deltas, context_key(tape), style=row.get("prompt_style")):
        raise ValueError("final prompt mismatch")


def audit_row(row):
    required = {"question", "completion_prompt", "response", "training_group", "kind", "operation", "width", "left", "right", "tape", "expected_answer", "prompt_style"}
    if not required <= set(row) or row["question"] != row["completion_prompt"] or row["training_group"] != "token_native_ledger":
        raise ValueError("invalid common row fields")
    if row["kind"] == "transition":
        return audit_transition(row)
    if row["kind"] == "final":
        audit_final(row)
        return None
    raise ValueError("unknown row kind")


def audit_episode(episode):
    tape = tape_for(episode)
    if parse_delta(episode.get("initial_delta", "")) != initial_delta():
        raise ValueError("invalid initial delta")
    lines = episode.get("expected_deltas")
    if not isinstance(lines, list) or len(lines) != int(tape["w"]):
        raise ValueError("invalid episode ledger length")
    carry, prior, prompts = 0, initial_delta(), []
    for position, line in enumerate(lines):
        expected = expected_delta(tape, position, carry)
        if parse_delta(line) != expected:
            raise ValueError("invalid episode transition")
        prompts.append(transition_prompt(tape, prior, style=episode["prompt_style"]))
        prior, carry = expected, expected["c"]
    prompts.append(final_prompt(lines, context_key(tape), style=episode["prompt_style"]))
    if int(episode.get("expected_answer")) != expected_answer(tape):
        raise ValueError("invalid episode answer")
    return prompts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--episodes", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = Path(args.out)
    if out.exists():
        raise SystemExit("refusing to overwrite audit output")

    prompts, grams, observed = set(), set(), Counter()
    valid_rows, invalid_rows, duplicates = 0, 0, 0
    with open(args.data) as source:
        for line in source:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                context = audit_row(row)
                prompt = normalized(row["completion_prompt"])
                if prompt in prompts:
                    duplicates += 1
                prompts.add(prompt)
                grams.update(ngrams(row["completion_prompt"]))
                if context is not None:
                    observed[context] += 1
                valid_rows += 1
            except Exception:
                invalid_rows += 1

    heldout_prompts, heldout_grams, regimes = set(), set(), Counter()
    invalid_episodes, counterfactual_mismatches = 0, 0
    with open(args.episodes) as source:
        for line in source:
            if not line.strip():
                continue
            try:
                episode = json.loads(line)
                episode_prompts = audit_episode(episode)
                counterpart = episode.get("counterfactual")
                if not isinstance(counterpart, dict):
                    raise ValueError("missing paired counterfactual")
                cf_prompts = audit_episode(counterpart)
                if episode["split"] != counterpart["split"] or episode["operation"] != counterpart["operation"]:
                    counterfactual_mismatches += 1
                if int(episode["expected_answer"]) == int(counterpart["expected_answer"]):
                    counterfactual_mismatches += 1
                regimes[episode["split"]] += 1
                for prompt in episode_prompts + cf_prompts:
                    norm = normalized(prompt)
                    if norm in heldout_prompts:
                        raise ValueError("duplicate normalized heldout prompt")
                    heldout_prompts.add(norm)
                    heldout_grams.update(ngrams(prompt))
            except Exception:
                invalid_episodes += 1

    required = required_contexts()
    missing = sorted(required - set(observed))
    result = {
        "audit": "token_native_ledger_v1_admission",
        "data": str(Path(args.data).resolve()),
        "episodes": str(Path(args.episodes).resolve()),
        "data_sha256": sha256_file(args.data),
        "episodes_sha256": sha256_file(args.episodes),
        "valid_train_rows": valid_rows,
        "invalid_train_rows": invalid_rows,
        "duplicate_normalized_train_prompts": duplicates,
        "valid_heldout_episodes": sum(regimes.values()),
        "invalid_heldout_episodes": invalid_episodes,
        "counterfactual_mismatches": counterfactual_mismatches,
        "heldout_regimes": dict(sorted(regimes.items())),
        "required_local_contexts": len(required),
        "covered_local_contexts": len(required & set(observed)),
        "missing_local_contexts": len(missing),
        "train_heldout_exact_prompt_hits": len(prompts & heldout_prompts),
        "train_heldout_13gram_hits": len(grams & heldout_grams),
        "claim_boundary": (
            "Independent admission for a fixed three-token carrier. Passing this audit does not establish "
            "a model result, reasoning, or context scaling."
        ),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))
    if (
        invalid_rows or duplicates or invalid_episodes or counterfactual_mismatches or missing or
        result["train_heldout_exact_prompt_hits"] or result["train_heldout_13gram_hits"]
    ):
        raise SystemExit("token-native ledger admission failed")


if __name__ == "__main__":
    main()
