#!/usr/bin/env python3
"""Evaluate self-authored append-only delta-ledger rollouts.

The evaluator starts from a held-out immutable operand tape, forwards only
model-emitted deltas and blocks, and separately executes the paired one-input
counterfactual. It never calculates, repairs, ranks, or replaces a ledger
record during a rollout.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path

import torch
from tokenizers import Tokenizer

from append_ledger_controller import rollout_episode
from append_ledger_protocol import PROMPT_STYLES
from eval_suite import generate
from model import GPT, GPTConfig


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_model(path, device):
    checkpoint = torch.load(path, map_location="cpu")
    model = GPT(GPTConfig(**checkpoint["cfg"])).to(device).eval()
    model.load_state_dict(checkpoint["model"])
    return checkpoint, model


def read_episodes(path, per_regime):
    selected, counts = [], collections.Counter()
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        episode = json.loads(line)
        if counts[episode["split"]] >= per_regime:
            continue
        if "counterfactual" not in episode:
            raise ValueError("heldout episode lacks counterfactual")
        selected.append(episode)
        counts[episode["split"]] += 1
    if not selected:
        raise ValueError("no heldout append-ledger episodes selected")
    return selected


def evaluate_pair(episode, ask, prompt_style=None):
    """Run normal and one-input-counterfactual branches without a fallback."""
    style = episode["prompt_style"] if prompt_style is None else prompt_style
    normal = rollout_episode(episode, ask, prompt_style=style)
    counterfactual = rollout_episode(episode["counterfactual"], ask, prompt_style=style)
    expected_changed = int(episode["expected_answer"]) != int(episode["counterfactual"]["expected_answer"])
    intervention = bool(expected_changed and normal["final_correct"] and counterfactual["final_correct"])
    return {
        "normal": normal,
        "counterfactual": counterfactual,
        "expected_changed": expected_changed,
        "intervention_success": intervention,
    }


def row_counts(rows):
    deltas = [row for row in rows if row["kind"] == "delta"]
    blocks = [row for row in rows if row["kind"] == "block"]
    return {
        "first_delta_correct": int(bool(deltas) and deltas[0]["correct"]),
        "delta_correct": sum(int(row["correct"]) for row in deltas),
        "delta_attempted_before_failure": len(deltas),
        "block_correct": sum(int(row["correct"]) for row in blocks),
        "block_attempted_before_failure": len(blocks),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--episodes", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--per-regime", type=int, default=100)
    parser.add_argument("--max-new", type=int, default=96)
    parser.add_argument("--prompt-style", choices=("auto",) + PROMPT_STYLES, default="auto")
    args = parser.parse_args()
    if args.per_regime <= 0 or args.max_new <= 0:
        raise SystemExit("--per-regime and --max-new must be positive")
    out = Path(args.out)
    if out.exists():
        raise SystemExit("refusing to overwrite output: {}".format(out))

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    checkpoint, model = load_model(args.ckpt, device)
    tokenizer = Tokenizer.from_file(args.tokenizer)
    episodes = read_episodes(args.episodes, args.per_regime)

    def ask(prompt):
        return generate(model, tokenizer, prompt, device, max_new=args.max_new, temp=0.0, skip_special_tokens=False)

    totals = collections.Counter()
    metrics = collections.defaultdict(collections.Counter)
    first_responses, examples = collections.Counter(), []
    for index, episode in enumerate(episodes, 1):
        regime = episode["split"]
        totals[regime] += 1
        pair = evaluate_pair(episode, ask, None if args.prompt_style == "auto" else args.prompt_style)
        normal, counterfactual = pair["normal"], pair["counterfactual"]
        current = row_counts(normal["rows"])
        metrics[regime].update(current)
        metrics[regime]["syntactic_closed_loop"] += int(normal["syntactic_closed_loop"])
        metrics[regime]["exact_chain"] += int(normal["exact_chain"])
        metrics[regime]["final_answer_correct"] += int(normal["final_correct"])
        metrics[regime]["counterfactual_final_correct"] += int(counterfactual["final_correct"])
        metrics[regime]["paired_intervention_correct_and_different"] += int(pair["intervention_success"])
        if normal["rows"]:
            first_responses[normal["rows"][0]["response"]] += 1
        if len(examples) < 12:
            examples.append({
                "id": episode["id"], "regime": regime,
                "expected_answer": episode["expected_answer"],
                "counterfactual_expected_answer": episode["counterfactual"]["expected_answer"],
                "normal": normal, "counterfactual": counterfactual,
                "intervention_success": pair["intervention_success"],
            })
        if index % 20 == 0 or index == len(episodes):
            print("[append-ledger] {}/{} final={}".format(
                index, len(episodes), sum(value["final_answer_correct"] for value in metrics.values())
            ), flush=True)

    by_regime = {regime: {"episodes": totals[regime], **dict(metrics[regime])} for regime in sorted(totals)}
    result = {
        "audit": "append_ledger_closed_loop_v1",
        "claim_boundary": (
            "A passing result establishes model-authored local digitwise execution and first-level block compaction "
            "from a fixed canonical tape only. It does not establish language parsing, broad reasoning, or general context scaling."
        ),
        "checkpoint": args.ckpt,
        "step": checkpoint.get("step"),
        "device": device,
        "episodes": args.episodes,
        "episodes_sha256": sha256_file(args.episodes),
        "per_regime": args.per_regime,
        "max_new": args.max_new,
        "prompt_style": args.prompt_style,
        "by_regime": by_regime,
        "first_response_unique": len(first_responses),
        "first_response_mode_count": max(first_responses.values(), default=0),
        "examples": examples,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"step": result["step"], "by_regime": by_regime}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
