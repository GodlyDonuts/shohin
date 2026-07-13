#!/usr/bin/env python3
"""Generate a coverage-complete token-native delta-ledger candidate.

This is a CPU-only representation experiment.  The model is trained to emit
three existing special tokens per arithmetic transition, and to read a final
answer from only its own emitted token ledger.  No controller-generated
arithmetic appears in inference prompts.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import random
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline"))
sys.path.insert(0, str(ROOT / "train"))
from digitwise_basis_protocol import WIDTHS, context_label, reachable_contexts
from generate_digitwise_factor_v1 import operands_for_context
from digitwise_factor_protocol import canonical_tape, initial_tape
from token_native_ledger_protocol import (
    CODE_TOKENS,
    canonical_delta,
    context_key,
    expected_answer,
    expected_delta,
    final_prompt,
    initial_delta,
    parse_delta,
    transition_prompt,
)


WORD = re.compile(r"\w+")


def normalized(text):
    # Generic lexical normalization would turn `<think>` and `</think>` into
    # the same word.  Preserve each atomic carrier before stripping surface
    # punctuation so deduplication and contamination checks respect its code.
    text = str(text)
    for index, token in enumerate(CODE_TOKENS):
        text = text.replace(token, " tokennative{} ".format(index))
    return " ".join(WORD.findall(text.lower()))


def ngrams(text, width=13):
    words = normalized(text).split()
    return {tuple(words[index:index + width]) for index in range(max(0, len(words) - width + 1))}


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def episode_from_operands(episode_id, split, width, operation, left, right, prompt_style):
    tape = initial_tape(operation, int(left), int(right), int(width))
    carry, deltas = 0, []
    for position in range(int(width)):
        delta = expected_delta(tape, position, carry)
        deltas.append(canonical_delta(delta))
        carry = delta["c"]
    return {
        "id": str(episode_id),
        "split": str(split),
        "prompt_style": str(prompt_style),
        "operation": str(operation),
        "width": int(width),
        "left": int(left),
        "right": int(right),
        "tape": canonical_tape(tape),
        "initial_delta": canonical_delta(initial_delta()),
        "expected_deltas": deltas,
        "expected_answer": expected_answer(tape),
    }


def counterfactual_episode(episode):
    maximum = 10 ** int(episode["width"]) - 1
    left, right = int(episode["left"]), int(episode["right"])
    if episode["operation"] == "add":
        left = left + 1 if left < maximum else left - 1
    elif left < maximum:
        left += 1
    elif right > 0:
        right -= 1
    else:
        left -= 1
    result = episode_from_operands(
        episode["id"] + "-cf", episode["split"], episode["width"], episode["operation"],
        left, right, episode["prompt_style"],
    )
    if int(result["expected_answer"]) == int(episode["expected_answer"]):
        raise AssertionError("counterfactual did not change answer")
    return result


def signature(episode):
    # Reserve operands across operations because an operation token can be
    # outside a short lexical overlap window in a serialized tape.
    return int(episode["width"]), int(episode["left"]), int(episode["right"])


def prompts_for_episode(episode):
    tape = initial_tape(episode["operation"], int(episode["left"]), int(episode["right"]), int(episode["width"]))
    prior, prompts, deltas = initial_delta(), [], []
    for line in episode["expected_deltas"]:
        prompts.append(transition_prompt(tape, prior, style=episode["prompt_style"]))
        prior = expected = parse_delta(line)
        if expected is None:
            raise ValueError("invalid expected token-native delta")
        deltas.append(line)
    prompts.append(final_prompt(deltas, context_key(tape), style=episode["prompt_style"]))
    return prompts


def rows_from_episode(episode):
    tape = initial_tape(episode["operation"], int(episode["left"]), int(episode["right"]), int(episode["width"]))
    prior, rows, deltas = initial_delta(), [], []
    common = {key: episode[key] for key in (
        "id", "split", "prompt_style", "operation", "width", "left", "right", "tape", "expected_answer",
    )}
    for index, line in enumerate(episode["expected_deltas"]):
        expected = parse_delta(line)
        if expected is None:
            raise ValueError("invalid expected token-native delta")
        prompt = transition_prompt(tape, prior, style=episode["prompt_style"])
        rows.append({
            "question": prompt, "completion_prompt": prompt, "response": line,
            "source": "token_native_ledger_transition_v1", "training_group": "token_native_ledger",
            "kind": "transition", "transition_index": index,
            "prior_delta": canonical_delta(prior), "expected_delta": line, **common,
        })
        prior, deltas = expected, deltas + [line]
    prompt = final_prompt(deltas, context_key(tape), style=episode["prompt_style"])
    rows.append({
        "question": prompt, "completion_prompt": prompt, "response": "answer={}".format(episode["expected_answer"]),
        "source": "token_native_ledger_final_v1", "training_group": "token_native_ledger", "kind": "final",
        "deltas": list(deltas), **common,
    })
    return rows


def _write_jsonl(path, rows):
    path = Path(path)
    partial = path.with_suffix(path.suffix + ".partial")
    if path.exists() or partial.exists():
        raise SystemExit("refusing existing output: {}".format(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    with partial.open("w") as output:
        for row in rows:
            output.write(json.dumps(row, sort_keys=True) + "\n")
    os.replace(partial, path)


def _deduplicate(rows):
    result, seen, dropped = [], set(), 0
    for row in rows:
        key = normalized(row["completion_prompt"])
        if key in seen:
            dropped += 1
            continue
        seen.add(key)
        result.append(row)
    return result, dropped


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-out", required=True)
    parser.add_argument("--heldout-out", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--variants", type=int, default=8)
    parser.add_argument("--heldout-per-regime", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260713)
    args = parser.parse_args()
    if args.variants <= 0 or args.heldout_per_regime <= 0:
        raise SystemExit("variants and heldout-per-regime must be positive")
    destinations = tuple(Path(path) for path in (args.train_out, args.heldout_out, args.report))
    if any(path.exists() or path.with_suffix(path.suffix + ".partial").exists() for path in destinations):
        raise SystemExit("refusing to overwrite an atomic-ledger candidate")

    rng = random.Random(args.seed)
    required = sorted(reachable_contexts(WIDTHS))
    train_episodes = []
    for context in required:
        for variant in range(args.variants):
            left, right = operands_for_context(context, rng)
            width, operation = context[:2]
            train_episodes.append(episode_from_operands(
                "tnl-{}-v{:02d}".format(context_label(context), variant), "train", width, operation,
                left, right, "core",
            ))
    rows, duplicate_dropped = _deduplicate([row for item in train_episodes for row in rows_from_episode(item)])
    observed = Counter()
    for row in rows:
        if row["kind"] != "transition":
            continue
        tape = initial_tape(row["operation"], int(row["left"]), int(row["right"]), int(row["width"]))
        prior = parse_delta(row["prior_delta"])
        if prior is None:
            raise RuntimeError("generated invalid prior delta")
        position = int(prior["p"])
        if position < int(tape["w"]):
            observed[(int(tape["w"]), tape["op"], position, int(prior["c"]), int(tape["a"][position]), int(tape["b"][position]))] += 1
    missing = [context for context in required if observed[context] == 0]
    if missing:
        raise RuntimeError("token-native candidate lost local contexts after deduplication")

    reserved = {signature(episode) for episode in train_episodes}
    heldout_specs = (("recombine_w4", 4), ("recombine_w6", 6), ("width_ood_w8", 8))
    heldout = []
    for split, width in heldout_specs:
        maximum = 10 ** width - 1
        for index in range(args.heldout_per_regime):
            for _attempt in range(100000):
                operation = rng.choice(("add", "sub"))
                left, right = rng.randint(0, maximum), rng.randint(0, maximum)
                if operation == "sub" and left < right:
                    left, right = right, left
                episode = episode_from_operands("{}-{:05d}".format(split, index), split, width, operation, left, right, "heldout")
                counterpart = counterfactual_episode(episode)
                signatures = {signature(episode), signature(counterpart)}
                if signatures & reserved:
                    continue
                reserved.update(signatures)
                episode["counterfactual"] = counterpart
                heldout.append(episode)
                break
            else:
                raise RuntimeError("unable to sample disjoint token-native heldout episode")

    train_prompts = {normalized(row["completion_prompt"]) for row in rows}
    heldout_prompts = {normalized(prompt) for item in heldout for prompt in prompts_for_episode(item)}
    if train_prompts & heldout_prompts:
        raise RuntimeError("exact train/heldout token-native prompt overlap")
    train_grams = {gram for row in rows for gram in ngrams(row["completion_prompt"])}
    heldout_grams = {gram for item in heldout for prompt in prompts_for_episode(item) for gram in ngrams(prompt)}
    if train_grams & heldout_grams:
        raise RuntimeError("13-gram train/heldout token-native overlap")

    rng.shuffle(rows)
    _write_jsonl(args.train_out, rows)
    _write_jsonl(args.heldout_out, heldout)
    report = {
        "schema": "shohin-token-native-ledger-v1",
        "seed": args.seed,
        "variants": args.variants,
        "required_local_contexts": len(required),
        "covered_local_contexts": len(observed),
        "missing_local_contexts": len(missing),
        "train_episodes": len(train_episodes),
        "train_rows": len(rows),
        "duplicate_train_prompts_dropped": duplicate_dropped,
        "heldout_episodes": len(heldout),
        "heldout_by_regime": dict(sorted(Counter(item["split"] for item in heldout).items())),
        "train_sha256": sha256_file(args.train_out),
        "heldout_sha256": sha256_file(args.heldout_out),
        "claim_boundary": (
            "CPU-only fixed-token carrier candidate. Three-token state transport is a serialization-length "
            "control, not evidence of language reasoning, context scaling, or a global workspace."
        ),
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
