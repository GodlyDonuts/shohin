#!/usr/bin/env python3
"""Evaluate self-authored digitwise recurrent scratchpad rollouts.

The evaluator starts from the canonical state supplied by a heldout episode,
then forwards only states emitted by the model.  It separately executes each
paired one-operand counterfactual.  It never runs the arithmetic solver during
rollout or repairs malformed output.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path

import torch
from tokenizers import Tokenizer

from digitwise_controller import rollout_episode
from digitwise_protocol import PROMPT_STYLES, parse_answer
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
        raise ValueError("no heldout digitwise episodes selected")
    return selected


def evaluate_pair(episode, ask, prompt_style=None):
    """Run normal and counterfactual branches without an arithmetic fallback."""
    style = episode["prompt_style"] if prompt_style is None else prompt_style
    normal = rollout_episode(episode, ask, prompt_style=style)
    counterfactual = rollout_episode(episode["counterfactual"], ask, prompt_style=style)
    expected_changed = int(episode["expected_answer"]) != int(episode["counterfactual"]["expected_answer"])
    emitted_normal, emitted_counterfactual = parse_answer(normal["final_response"]), parse_answer(counterfactual["final_response"])
    intervention = bool(
        expected_changed and normal["success"] and counterfactual["success"] and
        emitted_normal is not None and emitted_counterfactual is not None and emitted_normal != emitted_counterfactual
    )
    return {
        "normal": normal,
        "counterfactual": counterfactual,
        "expected_changed": expected_changed,
        "intervention_success": intervention,
    }


def transcript_record(episode, pair):
    """Keep rollout evidence without using it to alter the evaluation outcome."""
    return {
        "id": episode["id"],
        "regime": episode["split"],
        "expected_answer": episode["expected_answer"],
        "counterfactual_expected_answer": episode["counterfactual"]["expected_answer"],
        "normal": pair["normal"],
        "counterfactual": pair["counterfactual"],
        "intervention_success": pair["intervention_success"],
    }


def retain_regime_transcript(bucket, record, succeeded, per_outcome):
    """Retain early successes and failures separately so a regime is interpretable."""
    if per_outcome <= 0:
        return
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
    parser.add_argument("--max-new", type=int, default=96)
    parser.add_argument(
        "--examples-per-regime",
        type=int,
        default=0,
        help="retain this many successful and failed paired transcripts per regime (0 disables)",
    )
    parser.add_argument("--prompt-style", choices=("auto",) + PROMPT_STYLES, default="auto",
                        help="auto uses the episode's held-out wording; core isolates execution from wording transfer")
    args = parser.parse_args()
    if args.per_regime <= 0 or args.max_new <= 0 or args.examples_per_regime < 0:
        raise SystemExit("--per-regime and --max-new must be positive; --examples-per-regime must be nonnegative")
    out = Path(args.out)
    if out.exists():
        raise SystemExit("refusing to overwrite output: {}".format(out))

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    checkpoint, model = load_model(args.ckpt, device)
    tokenizer = Tokenizer.from_file(args.tokenizer)
    episodes = read_episodes(args.episodes, args.per_regime)

    def ask(prompt):
        return generate(model, tokenizer, prompt, device, max_new=args.max_new, temp=0.0, skip_special_tokens=False)

    totals, first_correct, transition_correct, transition_attempted = (collections.Counter() for _ in range(4))
    state_loop, final_correct, counterfactual_final, paired_intervention = (collections.Counter() for _ in range(4))
    first_responses, examples = collections.Counter(), []
    examples_by_regime = collections.defaultdict(lambda: {"successes": [], "failures": []})
    for index, episode in enumerate(episodes, 1):
        regime = episode["split"]
        totals[regime] += 1
        pair = evaluate_pair(episode, ask, None if args.prompt_style == "auto" else args.prompt_style)
        normal, counterfactual = pair["normal"], pair["counterfactual"]
        normal_rows = normal["rows"]
        if normal_rows:
            first_correct[regime] += int(normal_rows[0]["correct"])
            first_responses[normal_rows[0]["response"]] += 1
        transition_correct[regime] += sum(int(row["correct"]) for row in normal_rows)
        transition_attempted[regime] += len(normal_rows)
        state_loop[regime] += int(normal["state_closed_loop"])
        final_correct[regime] += int(normal["success"])
        counterfactual_final[regime] += int(counterfactual["success"])
        paired_intervention[regime] += int(pair["intervention_success"])
        record = transcript_record(episode, pair)
        if len(examples) < 12:
            examples.append(record)
        retain_regime_transcript(
            examples_by_regime,
            record,
            bool(normal["success"]),
            args.examples_per_regime,
        )
        if index % 20 == 0 or index == len(episodes):
            print("[digitwise] {}/{} final={}".format(index, len(episodes), sum(final_correct.values())), flush=True)

    by_regime = {
        regime: {
            "episodes": totals[regime],
            "first_transition_correct": first_correct[regime],
            "transition_correct": transition_correct[regime],
            "transition_attempted_before_failure": transition_attempted[regime],
            "state_closed_loop_correct": state_loop[regime],
            "final_answer_correct": final_correct[regime],
            "counterfactual_final_correct": counterfactual_final[regime],
            "paired_intervention_correct_and_different": paired_intervention[regime],
        }
        for regime in sorted(totals)
    }
    result = {
        "audit": "digitwise_recurrent_closed_loop_v1",
        "checkpoint": args.ckpt,
        "step": checkpoint.get("step"),
        "device": device,
        "episodes": args.episodes,
        "episodes_sha256": sha256_file(args.episodes),
        "per_regime": args.per_regime,
        "max_new": args.max_new,
        "examples_per_regime": args.examples_per_regime,
        "prompt_style": args.prompt_style,
        "by_regime": by_regime,
        "first_response_unique": len(first_responses),
        "first_response_mode_count": max(first_responses.values(), default=0),
        "examples": examples,
        "examples_by_regime": dict(examples_by_regime),
        "claim_boundary": (
            "A passing result establishes model-authored local digitwise execution from a fixed canonical state only. "
            "It does not establish language parsing, broad reasoning, or general context scaling."
        ),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"step": result["step"], "by_regime": by_regime}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
