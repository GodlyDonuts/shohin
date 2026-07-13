#!/usr/bin/env python3
"""Evaluate self-authored token-native delta-ledger rollouts.

The evaluator retains each episode's immutable tape only for prompt construction
and scoring. During a rollout it forwards an exact three-token carrier emitted
by the model; it never computes, repairs, chooses, or rewrites a transition.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path

import torch
from tokenizers import Tokenizer

from eval_suite import generate
from model import GPT, GPTConfig
from token_native_ledger_controller import rollout_episode
from token_native_ledger_protocol import PROMPT_STYLES


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
            raise ValueError("held-out token-native episode lacks a counterfactual")
        selected.append(episode)
        counts[episode["split"]] += 1
    if not selected:
        raise ValueError("no held-out token-native episodes selected")
    return selected


def evaluate_pair(episode, ask, prompt_style=None):
    """Run both branches; a different answer requires both exact rollouts."""
    style = episode["prompt_style"] if prompt_style is None else prompt_style
    normal = rollout_episode(episode, ask, prompt_style=style)
    counterfactual = rollout_episode(episode["counterfactual"], ask, prompt_style=style)
    expected_changed = int(episode["expected_answer"]) != int(episode["counterfactual"]["expected_answer"])
    intervention = bool(
        expected_changed and normal["success"] and counterfactual["success"] and
        normal["final_response"].strip() != counterfactual["final_response"].strip()
    )
    return {
        "normal": normal,
        "counterfactual": counterfactual,
        "expected_changed": expected_changed,
        "intervention_success": intervention,
    }


def failure_mode(rollout):
    """Report the first failure without changing the model-produced ledger."""
    if not rollout["syntactic_closed_loop"]:
        return "malformed_transition_{}".format(rollout["rows"][-1]["index"])
    for row in rollout["rows"]:
        if not row["correct"]:
            return "wrong_transition_{}".format(row["index"])
    if not rollout["final_correct"]:
        return "terminal_answer"
    return "success"


def transcript_record(episode, pair):
    return {
        "id": episode["id"],
        "regime": episode["split"],
        "expected_answer": episode["expected_answer"],
        "counterfactual_expected_answer": episode["counterfactual"]["expected_answer"],
        "normal": pair["normal"],
        "counterfactual": pair["counterfactual"],
        "intervention_success": pair["intervention_success"],
    }


def retain_regime_transcript(examples_by_regime, regime, record, succeeded, per_outcome):
    if per_outcome <= 0:
        return
    bucket = examples_by_regime[regime]
    key = "successes" if succeeded else "failures"
    if len(bucket[key]) < per_outcome:
        bucket[key].append(record)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--episodes", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--per-regime", type=int, default=100)
    parser.add_argument("--max-new-final", type=int, default=48)
    parser.add_argument("--examples-per-regime", type=int, default=0)
    parser.add_argument("--prompt-style", choices=("auto",) + PROMPT_STYLES, default="auto")
    args = parser.parse_args()
    if args.per_regime <= 0 or args.max_new_final <= 0 or args.examples_per_regime < 0:
        raise SystemExit("--per-regime and --max-new-final must be positive; --examples-per-regime must be nonnegative")
    out = Path(args.out)
    if out.exists():
        raise SystemExit("refusing to overwrite output: {}".format(out))

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    checkpoint, model = load_model(args.ckpt, device)
    tokenizer = Tokenizer.from_file(args.tokenizer)
    episodes = read_episodes(args.episodes, args.per_regime)

    def ask(prompt, max_new):
        # Special carrier tokens must remain literal for strict parsing.
        limit = args.max_new_final if max_new > 3 else max_new
        return generate(model, tokenizer, prompt, device, max_new=limit, temp=0.0, skip_special_tokens=False)

    totals, first_correct, transition_correct, transition_attempted = (collections.Counter() for _ in range(4))
    syntactic_loop, strict_loop, final_correct, counterfactual_final, paired_intervention = (
        collections.Counter() for _ in range(5)
    )
    failure_modes = collections.defaultdict(collections.Counter)
    examples, examples_by_regime = [], collections.defaultdict(lambda: {"successes": [], "failures": []})
    for index, episode in enumerate(episodes, 1):
        regime = episode["split"]
        totals[regime] += 1
        pair = evaluate_pair(episode, ask, None if args.prompt_style == "auto" else args.prompt_style)
        normal, counterfactual = pair["normal"], pair["counterfactual"]
        rows = normal["rows"]
        if rows:
            first_correct[regime] += int(rows[0]["correct"])
        transition_correct[regime] += sum(int(row["correct"]) for row in rows)
        transition_attempted[regime] += len(rows)
        syntactic_loop[regime] += int(normal["syntactic_closed_loop"])
        strict_loop[regime] += int(normal["strict_closed_loop"])
        final_correct[regime] += int(normal["success"])
        counterfactual_final[regime] += int(counterfactual["success"])
        paired_intervention[regime] += int(pair["intervention_success"])
        failure_modes[regime][failure_mode(normal)] += 1
        record = transcript_record(episode, pair)
        if len(examples) < 12:
            examples.append(record)
        retain_regime_transcript(examples_by_regime, regime, record, bool(normal["success"]), args.examples_per_regime)
        if index % 20 == 0 or index == len(episodes):
            print("[tnl] {}/{} final={}".format(index, len(episodes), sum(final_correct.values())), flush=True)

    by_regime = {
        regime: {
            "episodes": totals[regime],
            "first_transition_correct": first_correct[regime],
            "transition_correct": transition_correct[regime],
            "transition_attempted_before_failure": transition_attempted[regime],
            "syntactic_closed_loop": syntactic_loop[regime],
            "strict_closed_loop_correct": strict_loop[regime],
            "final_answer_correct": final_correct[regime],
            "counterfactual_final_correct": counterfactual_final[regime],
            "paired_intervention_correct_and_different": paired_intervention[regime],
            "normal_failure_modes": dict(sorted(failure_modes[regime].items())),
        }
        for regime in sorted(totals)
    }
    result = {
        "audit": "token_native_ledger_closed_loop_v1",
        "checkpoint": args.ckpt,
        "step": checkpoint.get("step"),
        "device": device,
        "episodes": args.episodes,
        "episodes_sha256": sha256_file(args.episodes),
        "per_regime": args.per_regime,
        "max_new_final": args.max_new_final,
        "examples_per_regime": args.examples_per_regime,
        "prompt_style": args.prompt_style,
        "by_regime": by_regime,
        "examples": examples,
        "examples_by_regime": dict(examples_by_regime),
        "claim_boundary": (
            "A pass establishes model-authored atomic-carrier transport from fixed supplied evidence only. "
            "It does not establish language reasoning, context scaling, or a global workspace."
        ),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"step": result["step"], "by_regime": by_regime}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
