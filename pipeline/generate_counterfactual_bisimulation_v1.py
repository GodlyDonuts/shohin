#!/usr/bin/env python3
"""Generate an isolated Counterfactual Bisimulation Compiler curriculum.

Each world is described twice in unrelated natural-language forms.  The model
must compile either description into the same compact state, perform source-free
updates, identify each inverse delta, and answer from the retained state.  A
paired one-fact counterfactual receives the same operations but a changed final
sum, making state interchange falsifiable rather than cosmetic.
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
sys.path.insert(0, str(ROOT / "train"))
from bisimulation_compiler_protocol import (
    apply_operation, canonical_delta, canonical_state, compile_prompt, copy_values,
    delta_prompt, render_event, sum_query_prompt, update_prompt,
)


WORD = re.compile(r"\w+")
TRAIN_DOMAINS = (
    ("workshop", ("amber", "brass"), "parts"),
    ("bakery", ("trays", "racks"), "pastries"),
    ("library", ("folios", "returns"), "books"),
    ("garden", ("plots", "beds"), "seedlings"),
)
HELDOUT_DOMAINS = (
    ("harbor", ("crates", "lockers"), "lanterns"),
    ("observatory", ("cases", "drawers"), "lenses"),
    ("clinic", ("cabinets", "carts"), "bandages"),
)
ROW_FIELDS = {
    "question", "completion_prompt", "response", "source", "training_group", "kind", "episode_id",
    "world", "style", "keys", "reference", "domain", "item", "initial_values", "before_values",
    "after_values", "operation", "answer", "revision",
}


def normalized(text: str) -> str:
    return " ".join(WORD.findall(str(text).lower()))


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_operation(rng: random.Random, values, keys):
    left, right = keys
    kind = rng.choice(("add", "sub", "move", "swap"))
    if kind == "add":
        return {"kind": kind, "key": rng.choice(keys), "value": rng.randint(2, 19)}
    if kind == "sub":
        key = rng.choice(keys)
        return {"kind": kind, "key": key, "value": rng.randint(1, max(1, int(values[key]) // 4))}
    if kind == "move":
        source, target = (left, right) if rng.randrange(2) else (right, left)
        return {"kind": kind, "source": source, "target": target, "value": rng.randint(1, max(1, int(values[source]) // 4))}
    return {"kind": kind, "left": left, "right": right}


def make_world(values, operations, keys, domain, item, reference, style):
    values = copy_values(values, keys)
    state = canonical_state(values, keys)
    steps = []
    current = values
    for revision, operation in enumerate(operations, 1):
        after = apply_operation(current, operation, keys)
        event = render_event(operation, item, style)
        before_state, after_state = canonical_state(current, keys), canonical_state(after, keys)
        steps.append({
            "revision": revision,
            "operation": operation,
            "event": event,
            "before": current,
            "after": after,
            "before_state": before_state,
            "after_state": after_state,
            "delta": canonical_delta(operation, keys),
            "update_prompt": update_prompt(before_state, event, reference, revision, style),
            "delta_prompt": delta_prompt(before_state, after_state, reference, revision, style),
        })
        current = after
    return {
        "initial_values": values,
        "initial_state": state,
        "compile_prompts": {
            variant: compile_prompt(values, keys, domain, item, reference, style, variant)
            for variant in ("a", "b")
        },
        "steps": steps,
        "query": {
            "kind": "sum",
            "answer": int(current[keys[0]] + current[keys[1]]),
            "prompt": sum_query_prompt(canonical_state(current, keys), keys, reference, style),
        },
    }


def counterfactual_values(values, keys):
    result = copy_values(values, keys)
    result[keys[0]] += 1
    return result


def make_episode(episode_id: str, split: str, domain, index: int, rng: random.Random, heldout: bool):
    name, keys, item = domain
    style = "heldout" if heldout else "train"
    low, high = (101, 199) if heldout else (9, 79)
    initial = {keys[0]: rng.randint(low, high), keys[1]: rng.randint(low, high)}
    if heldout:
        regime = ("cbc_len4", "cbc_len8", "cbc_len12")[index % 3]
        length = {"cbc_len4": 4, "cbc_len8": 8, "cbc_len12": 12}[regime]
    else:
        regime = "train_len{}".format(1 + index % 4)
        length = 1 + index % 4
    operations, current = [], initial
    for _ in range(length):
        operation = make_operation(rng, current, keys)
        operations.append(operation)
        current = apply_operation(current, operation, keys)
    reference = "{}-{}-{:06d}".format("H" if heldout else "T", name, index)
    normal = make_world(initial, operations, keys, name, item, reference, style)
    counterfactual = make_world(counterfactual_values(initial, keys), operations, keys, name, item, reference, style)
    if normal["query"]["answer"] == counterfactual["query"]["answer"]:
        raise AssertionError("counterfactual must change the sum query")
    return {
        "schema": "counterfactual_bisimulation_v1",
        "id": episode_id,
        "split": split,
        "heldout": bool(heldout),
        "style": style,
        "domain": name,
        "item": item,
        "keys": list(keys),
        "reference": reference,
        "regime": regime,
        "normal": normal,
        "counterfactual": counterfactual,
    }


def row(episode, world_name, kind, prompt, response, before_values, after_values, operation, answer, revision):
    return {
        "question": prompt,
        "completion_prompt": prompt,
        "response": response,
        "source": "counterfactual_bisimulation_v1_train",
        "training_group": "counterfactual_bisimulation",
        "kind": kind,
        "episode_id": episode["id"],
        "world": world_name,
        "style": episode["style"],
        "keys": episode["keys"],
        "reference": episode["reference"],
        "domain": episode["domain"],
        "item": episode["item"],
        "initial_values": episode[world_name]["initial_values"],
        "before_values": before_values,
        "after_values": after_values,
        "operation": operation,
        "answer": answer,
        "revision": revision,
    }


def rows_for_episode(episode):
    keys = tuple(episode["keys"])
    rows = []
    for world_name in ("normal", "counterfactual"):
        world = episode[world_name]
        initial = copy_values(world["initial_values"], keys)
        for variant, prompt in world["compile_prompts"].items():
            rows.append(row(episode, world_name, "compile_{}".format(variant), prompt, world["initial_state"], initial, initial, None, None, 0))
        for step in world["steps"]:
            rows.append(row(episode, world_name, "update", step["update_prompt"], step["after_state"], step["before"], step["after"], step["operation"], None, step["revision"]))
            rows.append(row(episode, world_name, "inverse_delta", step["delta_prompt"], step["delta"], step["before"], step["after"], step["operation"], None, step["revision"]))
        rows.append(row(episode, world_name, "readout_sum", world["query"]["prompt"], "answer={}".format(world["query"]["answer"]), world["steps"][-1]["after"] if world["steps"] else initial, world["steps"][-1]["after"] if world["steps"] else initial, None, world["query"]["answer"], len(world["steps"]) + 1))
    return rows


def controller_prompts(episode):
    prompts = []
    for world_name in ("normal", "counterfactual"):
        world = episode[world_name]
        prompts.extend(world["compile_prompts"].values())
        for step in world["steps"]:
            prompts.extend((step["update_prompt"], step["delta_prompt"]))
        prompts.append(world["query"]["prompt"])
    return prompts


def deduplicate(rows):
    kept, seen, dropped = [], set(), 0
    for item in rows:
        key = normalized(item["completion_prompt"])
        if key in seen:
            dropped += 1
            continue
        seen.add(key)
        kept.append(item)
    return kept, dropped


def write_jsonl(path, rows):
    path = Path(path)
    partial = path.with_suffix(path.suffix + ".partial")
    if path.exists() or partial.exists():
        raise SystemExit("refusing to overwrite existing artifact: {}".format(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    with partial.open("w") as output:
        for item in rows:
            output.write(json.dumps(item, sort_keys=True) + "\n")
    os.replace(partial, path)


def build_episodes(per_domain: int, seed: int, heldout: bool):
    if per_domain <= 0:
        raise ValueError("per-domain must be positive")
    rng = random.Random(seed)
    domains = HELDOUT_DOMAINS if heldout else TRAIN_DOMAINS
    episodes, seen = [], set()
    for domain in domains:
        accepted, attempts = 0, 0
        while accepted < per_domain:
            attempts += 1
            if attempts > per_domain * 100:
                raise RuntimeError("unable to build unique CBC episodes")
            episode = make_episode(
                "{}-{}-{:06d}".format("heldout" if heldout else "train", domain[0], attempts),
                "heldout" if heldout else "train", domain, attempts, rng, heldout,
            )
            key = normalized(episode["normal"]["compile_prompts"]["a"])
            if key in seen:
                continue
            seen.add(key)
            episodes.append(episode)
            accepted += 1
    rng.shuffle(episodes)
    return episodes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-out", required=True)
    parser.add_argument("--heldout-out", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--train-per-domain", type=int, default=20000)
    parser.add_argument("--heldout-per-domain", type=int, default=400)
    parser.add_argument("--seed", type=int, default=20260713)
    args = parser.parse_args()
    if args.train_per_domain <= 0 or args.heldout_per_domain <= 0:
        raise SystemExit("episode counts must be positive")
    destinations = tuple(Path(path) for path in (args.train_out, args.heldout_out, args.report))
    if any(path.exists() for path in destinations):
        raise SystemExit("refusing to overwrite an existing CBC artifact")
    train_episodes = build_episodes(args.train_per_domain, args.seed, heldout=False)
    heldout_episodes = build_episodes(args.heldout_per_domain, args.seed + 1, heldout=True)
    rows, dropped = deduplicate([row for episode in train_episodes for row in rows_for_episode(episode)])
    train_prompts = {normalized(row["completion_prompt"]) for row in rows}
    heldout_prompts = {normalized(prompt) for episode in heldout_episodes for prompt in controller_prompts(episode)}
    if train_prompts & heldout_prompts:
        raise RuntimeError("exact CBC train/held-out prompt overlap")
    random.Random(args.seed + 2).shuffle(rows)
    write_jsonl(args.train_out, rows)
    write_jsonl(args.heldout_out, heldout_episodes)
    report = {
        "schema": "counterfactual_bisimulation_v1",
        "train_episodes": len(train_episodes),
        "heldout_episodes": len(heldout_episodes),
        "train_rows": len(rows),
        "duplicate_train_prompts_dropped": dropped,
        "train_by_kind": dict(sorted(Counter(row["kind"] for row in rows).items())),
        "heldout_by_regime": dict(sorted(Counter(episode["regime"] for episode in heldout_episodes).items())),
        "train_sha256": sha256_file(args.train_out),
        "heldout_sha256": sha256_file(args.heldout_out),
        "claim_boundary": "Data generation only. No model, reasoning, or context-scaling claim is implied.",
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
