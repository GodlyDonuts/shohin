#!/usr/bin/env python3
"""Run model-generated semantic capsules through a bounded external controller.

The controller carries only model-emitted capsule values between events.  It
never executes, repairs, selects, or supplies an intermediate state.  A passed
episode therefore requires semantic state preservation across a context reset;
it is still a narrow context-scaling diagnostic, not general reasoning proof.
"""

import argparse
import collections
import hashlib
import json
import re
from pathlib import Path

import torch
from tokenizers import Tokenizer

from eval_suite import generate
from model import GPT, GPTConfig


CAPSULE = re.compile(r"(?mi)^\s*capsule:\s*([^\n]+)\s*$")
PAIR = re.compile(r"\s*([a-z][a-z0-9_]*)\s*=\s*(-?\d+)\s*", re.I)
FINAL = re.compile(r"the\s+answer\s+is\s*(-?\d+)\b", re.I)


def canonical_capsule(values, keys):
    return "capsule:" + ";".join("{}={}".format(key, int(values[key])) for key in keys)


def parse_capsule(response, keys):
    matches = CAPSULE.findall(str(response))
    if len(matches) != 1:
        return None
    values = {}
    for part in matches[0].split(";"):
        match = PAIR.fullmatch(part)
        if not match:
            return None
        key, value = match.group(1).lower(), int(match.group(2))
        if key in values:
            return None
        values[key] = value
    required = [key.lower() for key in keys]
    if set(values) != set(required):
        return None
    return {key: values[key.lower()] for key in keys}


def parse_final(response):
    values = FINAL.findall(str(response))
    return int(values[-1]) if values else None


def transition_prompt(values, keys, instruction, heldout=False, reference="", revision=0):
    if heldout:
        return (
            "Task: A previous record has been discarded after being compressed. Reference {} revision {} is not a quantity.\n"
            "Retained facts: {}\nNew record: {}\n"
            "Reason inside <think>, then emit only one final line in the form "
            "capsule:{}=<integer>;{}=<integer>.\nResult:"
        ).format(reference, revision, canonical_capsule(values, keys), instruction, keys[0], keys[1])
    return (
        "Question: Continue record {} revision {} from a compact semantic capsule; identifiers are not quantities.\n"
        "Capsule: {}\nEvent: {}\n"
        "Inside <think> apply the event while retaining both named facts. On the last line return only "
        "capsule:{}=<integer>;{}=<integer>.\nAnswer:"
    ).format(reference, revision, canonical_capsule(values, keys), instruction, keys[0], keys[1])


def query_prompt(values, keys, text, heldout=False, reference=""):
    if heldout:
        return (
            "Task: The original record is unavailable; reference {} is not a quantity. Use the retained facts below.\n"
            "Retained facts: {}\nRequest: {}\n"
            "Reason inside <think>, then finish with 'The answer is <integer>.'.\nResult:"
        ).format(reference, canonical_capsule(values, keys), text)
    return (
        "Question: Answer a query using record {} and only this compact semantic capsule. The record identifier is not a quantity.\n"
        "Capsule: {}\nQuery: {}\n"
        "Inside <think> use the named facts. Then end with 'The answer is <integer>.'.\nAnswer:"
    ).format(reference, canonical_capsule(values, keys), text)


def load_model(path, device):
    checkpoint = torch.load(path, map_location="cpu")
    model = GPT(GPTConfig(**checkpoint["cfg"])).to(device).eval()
    model.load_state_dict(checkpoint["model"])
    return checkpoint, model


def read_episodes(path, per_regime):
    rows, counts = [], collections.Counter()
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        episode = json.loads(line)
        required = ("keys", "initial", "operations", "query", "regime", "family")
        if any(field not in episode for field in required):
            raise ValueError("malformed capsule episode")
        if counts[episode["regime"]] >= per_regime:
            continue
        rows.append(episode)
        counts[episode["regime"]] += 1
    if not rows:
        raise ValueError("no semantic capsule episodes selected")
    return rows


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--episodes", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--per-regime", type=int, default=100)
    parser.add_argument("--max-new", type=int, default=128)
    args = parser.parse_args()
    if args.per_regime <= 0:
        raise SystemExit("--per-regime must be positive")
    out = Path(args.out)
    if out.exists():
        raise SystemExit("refusing to overwrite output: {}".format(out))

    episodes = read_episodes(args.episodes, args.per_regime)
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    tokenizer = Tokenizer.from_file(args.tokenizer)
    checkpoint, model = load_model(args.ckpt, device)

    totals, write_ok, transition_ok, transition_total, closed_loop, query_ok = (collections.Counter() for _ in range(6))
    rows = []
    for index, episode in enumerate(episodes, 1):
        regime = episode["regime"]
        keys = tuple(episode["keys"])
        totals[regime] += 1
        transition_total[regime] += len(episode["operations"])
        initial_response = generate(
            model, tokenizer, episode["initial"]["prompt"], device, max_new=args.max_new,
            temp=0.0, skip_special_tokens=False,
        )
        current = parse_capsule(initial_response, keys)
        initial_correct = current == episode["initial"]["values"]
        write_ok[regime] += int(initial_correct)
        transitions = []
        all_correct = initial_correct
        if current is not None:
            for step in episode["operations"]:
                prompt = transition_prompt(current, keys, step["instruction"], episode.get("heldout", False),
                                           episode.get("reference", ""), step.get("revision", 0))
                response = generate(
                    model, tokenizer, prompt, device, max_new=args.max_new,
                    temp=0.0, skip_special_tokens=False,
                )
                predicted = parse_capsule(response, keys)
                correct = predicted == step["expected"]
                transition_ok[regime] += int(correct)
                transitions.append({
                    "prompt": prompt, "response": response, "predicted": predicted,
                    "expected": step["expected"], "correct": correct,
                })
                if not correct:
                    all_correct = False
                    break
                current = predicted
        else:
            all_correct = False
        final_response = ""
        final_correct = False
        if all_correct:
            prompt = query_prompt(current, keys, episode["query"]["text"], episode.get("heldout", False),
                                  episode.get("reference", ""))
            final_response = generate(
                model, tokenizer, prompt, device, max_new=args.max_new,
                temp=0.0, skip_special_tokens=False,
            )
            final_correct = parse_final(final_response) == int(episode["query"]["answer"])
            query_ok[regime] += int(final_correct)
        closed_loop[regime] += int(all_correct and final_correct)
        rows.append({
            "id": episode["id"], "family": episode["family"], "regime": regime,
            "keys": list(keys), "initial_response": initial_response,
            "initial_expected": episode["initial"]["values"], "initial_correct": initial_correct,
            "transitions": transitions, "final_response": final_response,
            "final_expected": episode["query"]["answer"], "final_correct": final_correct,
            "closed_loop_correct": all_correct and final_correct,
        })
        if index % 20 == 0 or index == len(episodes):
            print("[semantic-capsule] {}/{} closed_loop={}".format(index, len(episodes), sum(closed_loop.values())), flush=True)

    by_regime = {}
    for regime in sorted(totals):
        by_regime[regime] = {
            "episodes": totals[regime],
            "write_correct": write_ok[regime],
            "transition_correct": transition_ok[regime],
            "transition_total": transition_total[regime],
            "closed_loop_correct": closed_loop[regime],
            "query_correct_given_closed_loop": query_ok[regime],
        }
    result = {
        "audit": "semantic_capsule_closed_loop_v1",
        "checkpoint": args.ckpt,
        "step": checkpoint.get("step"),
        "device": device,
        "episodes": args.episodes,
        "episodes_sha256": sha256_file(args.episodes),
        "per_regime": args.per_regime,
        "by_regime": by_regime,
        "rows": rows,
        "claim_boundary": (
            "Success requires model-generated factual capsules across context resets. It is a held-out "
            "context-scaling result, not proof of broad reasoning or autonomous long-horizon planning."
        ),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"step": result["step"], "by_regime": by_regime}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
