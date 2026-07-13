#!/usr/bin/env python3
"""Probe whether a checkpoint can verify local machine-state transitions.

This is a transcript-level feasibility check for proof-carrying deliberation.
The candidate state is always grammar-valid.  A negative is a local semantic
near-miss (wrong written digit, carry/borrow, or immutable operand tape), not
malformed text.  The model receives no answer candidates from a solver at
inference; this probe only measures whether *verification* is easier than
free-form execution before we consider an isolated verifier curriculum.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path

import torch
from tokenizers import Tokenizer

from digitwise_protocol import apply_microstep, canonical_state, initial_state, parse_state
from eval_suite import generate
from forced_choice_probe import candidate_score
from model import GPT, GPTConfig


VERDICT_RE = re.compile(r"\b(valid|invalid)\b", re.I)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verdict_prompt(state, candidate, style):
    source = canonical_state(state)
    proposed = canonical_state(candidate)
    if style == "core":
        return (
            "Verify one proposed decimal-machine transition. A valid next state preserves op, w, a, and b; "
            "writes exactly the current r[p] digit; advances p by one; and sets the exact carry or borrow.\n"
            "Current state: {}\nCandidate next state: {}\n"
            "Return only verdict=valid or verdict=invalid.\nAnswer:"
        ).format(source, proposed)
    if style == "heldout":
        return (
            "Act as a local checker for a retained decimal record. Decide whether the proposed successor is "
            "the one legal one-place rewrite: immutable tapes must remain fixed, one result cell changes, "
            "the program counter advances once, and c is the right carry/borrow.\n"
            "Machine record: {}\nProposed successor: {}\n"
            "Emit only verdict=valid or verdict=invalid.\nResult:"
        ).format(source, proposed)
    raise ValueError("unknown prompt style")


def parse_verdict(text):
    values = VERDICT_RE.findall(str(text))
    return values[-1].lower() if values else None


def mutate_candidate(state, expected, kind):
    """Return a grammar-valid but semantically wrong local successor."""
    candidate = dict(expected)
    position = int(state["p"])
    if kind == "digit":
        tape = list(candidate["r"])
        tape[position] = str((int(tape[position]) + 1) % 10)
        candidate["r"] = "".join(tape)
    elif kind == "carry":
        candidate["c"] = 1 - int(candidate["c"])
    elif kind == "operand":
        tape = list(candidate["a"])
        tape[position] = str((int(tape[position]) + 1) % 10)
        candidate["a"] = "".join(tape)
    else:
        raise ValueError("unknown near-miss kind")
    canonical_state(candidate)
    if candidate == expected:
        raise AssertionError("near miss did not change candidate")
    return candidate


def make_cases(count, seed):
    if count <= 0:
        raise ValueError("count must be positive")
    rng, rows, identifiers = random.Random(seed), [], set()
    styles = ("core", "heldout")
    mutations = ("digit", "carry", "operand")
    while len(rows) < count:
        width = 4 if len(rows) % 2 == 0 else 6
        operation = "add" if (len(rows) // 2) % 2 == 0 else "sub"
        maximum = 10 ** width - 1
        left, right = rng.randrange(maximum + 1), rng.randrange(maximum + 1)
        if operation == "sub" and left < right:
            left, right = right, left
        state = initial_state(operation, left, right, width)
        # Avoid terminal states: carry/borrow mutations remain syntactically valid and
        # semantically local, while the terminal arithmetic convention stays out of this probe.
        for _ in range(rng.randrange(0, width - 1)):
            state = apply_microstep(state)
        expected = apply_microstep(state)
        key = (canonical_state(state), canonical_state(expected))
        if key in identifiers:
            continue
        identifiers.add(key)
        valid = len(rows) % 2 == 0
        mutation = mutations[(len(rows) // 2) % len(mutations)]
        candidate = expected if valid else mutate_candidate(state, expected, mutation)
        style = styles[(len(rows) // 2) % len(styles)]
        rows.append({
            "id": "transition-verifier-{:03d}".format(len(rows)),
            "style": style,
            "near_miss": "none" if valid else mutation,
            "state": canonical_state(state),
            "expected": canonical_state(expected),
            "candidate": canonical_state(candidate),
            "label": "valid" if valid else "invalid",
            "prompt": verdict_prompt(state, candidate, style),
        })
    rng.shuffle(rows)
    return rows


def load_model(path, device):
    checkpoint = torch.load(path, map_location="cpu")
    model = GPT(GPTConfig(**checkpoint["cfg"])).to(device).eval()
    model.load_state_dict(checkpoint["model"])
    return checkpoint, model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--cases", type=int, default=48)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--max-new", type=int, default=16)
    args = parser.parse_args()
    if args.cases < 12 or args.cases % 2:
        raise SystemExit("--cases must be an even integer >= 12")
    output = Path(args.out)
    if output.exists():
        raise SystemExit("refusing to overwrite output: {}".format(output))

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    checkpoint, model = load_model(args.ckpt, device)
    tokenizer = Tokenizer.from_file(args.tokenizer)
    rows, totals, correct = make_cases(args.cases, args.seed), Counter(), Counter()
    for index, row in enumerate(rows, 1):
        response = generate(model, tokenizer, row["prompt"], device, max_new=args.max_new, temp=0.0)
        predicted = parse_verdict(response)
        likelihoods = {
            verdict: candidate_score(model, tokenizer, device, row["prompt"], "verdict=" + verdict)
            for verdict in ("valid", "invalid")
        }
        likelihood_predicted = max(
            likelihoods,
            key=lambda verdict: (likelihoods[verdict]["mean_logprob"], likelihoods[verdict]["total_logprob"]),
        )
        row["response"] = response
        row["predicted"] = predicted
        row["correct"] = predicted == row["label"]
        row["likelihoods"] = likelihoods
        row["likelihood_predicted"] = likelihood_predicted
        row["likelihood_correct"] = likelihood_predicted == row["label"]
        for key in ("all", "style=" + row["style"], "label=" + row["label"], "miss=" + row["near_miss"]):
            totals[key] += 1
            correct[key] += int(row["correct"])
        if index % 8 == 0 or index == len(rows):
            print("[transition-verifier] {}/{} correct={}".format(index, len(rows), correct["all"]), flush=True)
    result = {
        "audit": "transition_verifier_raw_v1",
        "checkpoint": args.ckpt,
        "checkpoint_sha256": sha256_file(args.ckpt),
        "step": checkpoint.get("step"),
        "device": device,
        "seed": args.seed,
        "cases": args.cases,
        "max_new": args.max_new,
        "accuracy": {key: {"correct": correct[key], "total": totals[key], "rate": correct[key] / totals[key]} for key in sorted(totals)},
        "likelihood_accuracy": {
            "correct": sum(int(row["likelihood_correct"]) for row in rows),
            "total": len(rows),
            "rate": sum(int(row["likelihood_correct"]) for row in rows) / len(rows),
        },
        "rows": rows,
        "claim_boundary": (
            "This is a raw feasibility probe for local candidate verification. It neither supplies a solver "
            "candidate at inference nor establishes broad reasoning or self-correction."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"free": result["accuracy"], "likelihood": result["likelihood_accuracy"]}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
