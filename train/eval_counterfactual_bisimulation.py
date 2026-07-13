#!/usr/bin/env python3
"""Evaluate Counterfactual Bisimulation Compiler rollouts from model text only.

This evaluator never reconstructs a state, repairs a malformed state, selects
among candidates, or answers a query.  It delegates every generation to the
checkpoint and uses the transport-only controller to score source deletion,
same-world interchange, and a real counterfactual carrier swap afterward.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path

import torch
from tokenizers import Tokenizer

from bisimulation_compiler_controller import evaluate_pair
from eval_suite import generate
from model import GPT, GPTConfig


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_model(path, device):
    checkpoint = torch.load(path, map_location="cpu")
    model = GPT(GPTConfig(**checkpoint["cfg"])).to(device).eval()
    model.load_state_dict(checkpoint["model"])
    return checkpoint, model


def select_episodes(path, per_regime, seed):
    """Hash-select a stable, balanced held-out subset without prompt peeking."""
    if per_regime <= 0:
        raise ValueError("per_regime must be positive")
    grouped = collections.defaultdict(list)
    for line_number, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        episode = json.loads(line)
        if not episode.get("heldout") or not str(episode.get("id", "")):
            raise ValueError("invalid held-out CBC episode at line {}".format(line_number))
        grouped[str(episode["regime"])].append(episode)
    if not grouped:
        raise ValueError("CBC held-out corpus is empty")
    selected = []
    for regime, episodes in sorted(grouped.items()):
        if len(episodes) < per_regime:
            raise ValueError("{} has {} episodes, need {}".format(regime, len(episodes), per_regime))
        selected.extend(sorted(
            episodes,
            key=lambda item: hashlib.sha256((str(seed) + "\0" + item["id"]).encode()).hexdigest(),
        )[:per_regime])
    return selected


def _branch_counts(prefix, branch):
    rows = branch.get("rows", [])
    return {
        prefix + "_state_closed_loop": int(bool(branch.get("state_closed_loop"))),
        prefix + "_final_correct": int(bool(branch.get("final_correct"))),
        prefix + "_update_attempted": len(rows),
        prefix + "_update_correct": sum(int(bool(row.get("state_correct"))) for row in rows),
        prefix + "_inverse_delta_attempted": sum("inverse_delta_correct" in row for row in rows),
        prefix + "_inverse_delta_correct": sum(int(bool(row.get("inverse_delta_correct"))) for row in rows),
    }


def summarize_pair(pair):
    """Flatten one controller result without inventing a solver-side metric."""
    counts = collections.Counter()
    for world_name in ("normal", "counterfactual"):
        world = pair[world_name]
        for variant in ("a", "b"):
            counts["{}_compile_{}_correct".format(world_name, variant)] += int(
                bool(world["compilations"][variant]["correct"])
            )
        counts["{}_compile_equal".format(world_name)] += int(bool(world["compile_equal"]))
        for branch_name in ("primary", "interchange"):
            branch = world.get(branch_name)
            if branch is not None:
                counts.update(_branch_counts("{}_{}".format(world_name, branch_name), branch))
    for key in (
        "same_world_interchange_success",
        "counterfactual_interchange_success",
        "cross_world_counterfactual_success",
    ):
        counts[key] += int(bool(pair.get(key)))
    return dict(counts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--episodes", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--per-regime", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--max-new", type=int, default=48)
    parser.add_argument("--example-cap", type=int, default=4)
    args = parser.parse_args()
    if args.max_new <= 0 or args.example_cap < 0:
        raise SystemExit("max-new must be positive and example-cap non-negative")
    out = Path(args.out)
    if out.exists():
        raise SystemExit("refusing to overwrite output: {}".format(out))

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    checkpoint, model = load_model(args.ckpt, device)
    tokenizer = Tokenizer.from_file(args.tokenizer)
    episodes = select_episodes(args.episodes, args.per_regime, args.seed)

    def ask(prompt):
        return generate(model, tokenizer, prompt, device, max_new=args.max_new, temp=0.0,
                        skip_special_tokens=False)

    totals = collections.defaultdict(collections.Counter)
    examples = []
    for index, episode in enumerate(episodes, 1):
        pair = evaluate_pair(episode, ask)
        regime = str(episode["regime"])
        totals[regime]["episodes"] += 1
        totals[regime].update(summarize_pair(pair))
        if len(examples) < args.example_cap:
            examples.append({"id": episode["id"], "regime": regime, "result": pair})
        if index % 5 == 0 or index == len(episodes):
            print("[cbc-eval] {}/{} same_world={} cross_world={}".format(
                index, len(episodes),
                sum(values["same_world_interchange_success"] for values in totals.values()),
                sum(values["cross_world_counterfactual_success"] for values in totals.values()),
            ), flush=True)

    result = {
        "audit": "counterfactual_bisimulation_closed_loop_v1",
        "claim_boundary": (
            "A passing result establishes only source-deleted, model-authored state transport and a causal "
            "counterfactual carrier effect on this held-out protocol. It does not establish broad reasoning, "
            "general language understanding, or general context scaling."
        ),
        "checkpoint": args.ckpt,
        "step": checkpoint.get("step"),
        "device": device,
        "episodes": args.episodes,
        "episodes_sha256": sha256_file(args.episodes),
        "per_regime": args.per_regime,
        "seed": args.seed,
        "max_new": args.max_new,
        "by_regime": {name: dict(values) for name, values in sorted(totals.items())},
        "examples": examples,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("[cbc-eval] summary=" + json.dumps(result["by_regime"], sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
