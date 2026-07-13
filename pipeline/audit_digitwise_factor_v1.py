#!/usr/bin/env python3
"""Independent admission audit for factorized static-tape DRS data."""
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
from digitwise_factor_protocol import (
    apply_microstep,
    canonical_register,
    canonical_tape,
    digit_prompt,
    final_prompt,
    local_context,
    microstep_prompt,
    parse_digit,
    parse_register,
    parse_tape,
    register_answer,
)


WORD = re.compile(r"\w+")


def normalized(text):
    return " ".join(WORD.findall(str(text).lower()))


def ngrams(text, width=13):
    words = normalized(text).split()
    return {tuple(words[index:index + width]) for index in range(max(0, len(words) - width + 1))}


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def required_contexts():
    """Independently enumerate legal decimal contexts rather than importing the generator basis."""
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


def audit_train_row(row):
    if row.get("question") != row.get("completion_prompt") or not row.get("response"):
        raise ValueError("invalid completion boundary")
    tape = parse_tape(row.get("tape", ""))
    if tape is None or row.get("tape") != canonical_tape(tape):
        raise ValueError("invalid static tape")
    register = parse_register(row.get("register", ""), tape)
    if register is None or row.get("register") != canonical_register(tape, register):
        raise ValueError("invalid dynamic register")
    style, index, kind = row.get("prompt_style"), int(row.get("transition_index")), row.get("kind")
    if kind == "transition":
        expected = parse_register(row.get("expected_register", ""), tape)
        if expected is None or row["response"] != canonical_register(tape, expected):
            raise ValueError("invalid transition target")
        if expected != apply_microstep(tape, register):
            raise ValueError("transition target is not a single legal step")
        if row["question"] != microstep_prompt(tape, register, style=style):
            raise ValueError("transition prompt mismatch")
        if index != int(register["p"]):
            raise ValueError("transition index mismatch")
    elif kind == "digit":
        digit_index = int(row.get("digit_index"))
        expected_digit = int(row.get("expected_digit"))
        if row["response"] != "digit={}".format(expected_digit) or parse_digit(row["response"]) != expected_digit:
            raise ValueError("invalid digit target")
        if digit_index < 0 or digit_index >= int(register["p"]) or int(register["r"][digit_index]) != expected_digit:
            raise ValueError("digit target is not visible in register")
        if row["question"] != digit_prompt(tape, register, digit_index, style=style):
            raise ValueError("digit prompt mismatch")
        if index != digit_index:
            raise ValueError("digit index mismatch")
    elif kind == "final":
        expected_answer = int(row.get("expected_answer"))
        if row["response"] != "answer={}".format(expected_answer) or register_answer(tape, register) != expected_answer:
            raise ValueError("invalid final target")
        if row["question"] != final_prompt(tape, register, style=style):
            raise ValueError("final prompt mismatch")
        if index != int(tape["w"]):
            raise ValueError("final index mismatch")
    else:
        raise ValueError("unknown row kind")
    if int(row.get("width")) != int(tape["w"]) or row.get("operation") != tape["op"]:
        raise ValueError("row metadata mismatch")


def audit_episode(episode):
    tape = parse_tape(episode.get("tape", ""))
    if tape is None or tape != parse_tape(canonical_tape(tape)):
        raise ValueError("invalid episode tape")
    register = parse_register(episode.get("initial_register", ""), tape)
    if register is None:
        raise ValueError("invalid episode initial register")
    if int(episode["left"]) != sum(int(digit) * (10 ** index) for index, digit in enumerate(tape["a"])):
        raise ValueError("episode left tape mismatch")
    if int(episode["right"]) != sum(int(digit) * (10 ** index) for index, digit in enumerate(tape["b"])):
        raise ValueError("episode right tape mismatch")
    if episode.get("operation") != tape["op"] or int(episode.get("width")) != int(tape["w"]):
        raise ValueError("episode metadata mismatch")
    prompts = []
    expected_lines = episode.get("expected_registers", [])
    if len(expected_lines) != int(tape["w"]):
        raise ValueError("wrong recurrent episode length")
    for index, line in enumerate(expected_lines):
        expected = parse_register(line, tape)
        if expected is None or expected != apply_microstep(tape, register):
            raise ValueError("episode register transition mismatch")
        prompts.append(microstep_prompt(tape, register, style=episode["prompt_style"]))
        register = expected
        prompts.append(digit_prompt(tape, register, index, style=episode["prompt_style"]))
    prompts.append(final_prompt(tape, register, style=episode["prompt_style"]))
    if register_answer(tape, register) != int(episode["expected_answer"]):
        raise ValueError("episode final answer mismatch")
    return prompts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--episodes", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = Path(args.out)
    if out.exists():
        raise SystemExit("refusing existing output: {}".format(out))

    train_prompts, train_grams, observed = set(), set(), Counter()
    invalid_rows, duplicate_rows = 0, 0
    with open(args.data) as source:
        for line in source:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                audit_train_row(row)
                prompt = normalized(row["completion_prompt"])
                if prompt in train_prompts:
                    duplicate_rows += 1
                train_prompts.add(prompt)
                train_grams.update(ngrams(row["completion_prompt"]))
                if row["kind"] == "transition":
                    tape = parse_tape(row["tape"])
                    register = parse_register(row["register"], tape)
                    context = local_context(tape, register)
                    if context is None:
                        raise ValueError("terminal register cannot train a transition")
                    observed[context] += 1
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                invalid_rows += 1

    heldout_prompts, heldout_grams, regimes = set(), set(), Counter()
    invalid_episodes, counterfactual_mismatches = 0, 0
    with open(args.episodes) as source:
        for line in source:
            if not line.strip():
                continue
            try:
                episode = json.loads(line)
                prompts = audit_episode(episode)
                counterfactual = episode.get("counterfactual")
                if not isinstance(counterfactual, dict):
                    raise ValueError("missing counterfactual episode")
                cf_prompts = audit_episode(counterfactual)
                if episode["split"] != counterfactual["split"] or episode["operation"] != counterfactual["operation"]:
                    counterfactual_mismatches += 1
                if int(episode["expected_answer"]) == int(counterfactual["expected_answer"]):
                    counterfactual_mismatches += 1
                regimes[episode["split"]] += 1
                for prompt in prompts + cf_prompts:
                    norm = normalized(prompt)
                    if norm in heldout_prompts:
                        raise ValueError("duplicate heldout controller prompt")
                    heldout_prompts.add(norm)
                    heldout_grams.update(ngrams(prompt))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                invalid_episodes += 1

    required = required_contexts()
    missing = sorted(required - set(observed))
    result = {
        "audit": "digitwise_factor_v1_admission",
        "data": str(Path(args.data).resolve()),
        "episodes": str(Path(args.episodes).resolve()),
        "data_sha256": sha256_file(args.data),
        "episodes_sha256": sha256_file(args.episodes),
        "valid_train_rows": len(train_prompts),
        "invalid_train_rows": invalid_rows,
        "duplicate_normalized_train_prompts": duplicate_rows,
        "valid_heldout_episodes": sum(regimes.values()),
        "invalid_heldout_episodes": invalid_episodes,
        "counterfactual_mismatches": counterfactual_mismatches,
        "heldout_regimes": dict(sorted(regimes.items())),
        "required_local_contexts": len(required),
        "covered_local_contexts": len(required & set(observed)),
        "missing_local_contexts": len(missing),
        "missing_local_context_examples": [list(context) for context in missing[:12]],
        "train_heldout_exact_prompt_hits": len(train_prompts & heldout_prompts),
        "train_heldout_13gram_hits": len(train_grams & heldout_grams),
        "claim_boundary": "Data admission only; factorization is a representation control, not a model result.",
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))
    if (
        invalid_rows or duplicate_rows or invalid_episodes or counterfactual_mismatches or missing or
        result["train_heldout_exact_prompt_hits"] or result["train_heldout_13gram_hits"]
    ):
        raise SystemExit("digitwise factor v1 admission failed")


if __name__ == "__main__":
    main()
