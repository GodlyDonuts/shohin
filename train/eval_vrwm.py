#!/usr/bin/env python3
"""Evaluate VRWM one-step accuracy and model-generated closed-loop memory."""
import argparse
import collections
import json
from pathlib import Path

import torch
from tokenizers import Tokenizer

from eval_suite import generate
from model import GPT, GPTConfig
from vrwm_controller import rollout_episode
from vrwm_protocol import apply_operation, parse_answer, parse_memory, readout_prompt, transition_prompt


def load_model(path, device):
    checkpoint = torch.load(path, map_location="cpu")
    model = GPT(GPTConfig(**checkpoint["cfg"])).to(device).eval()
    model.load_state_dict(checkpoint["model"])
    return checkpoint, model


def read_episodes(path, limit, per_split):
    episodes = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if per_split:
        selected, counts = [], collections.Counter()
        for episode in episodes:
            if counts[episode["split"]] >= per_split:
                continue
            selected.append(episode)
            counts[episode["split"]] += 1
        episodes = selected
    elif limit:
        episodes = episodes[:limit]
    if not episodes:
        raise SystemExit("no VRWM evaluation episodes")
    return episodes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--episodes", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--per-split", type=int, default=0,
                        help="evaluate this many episodes from each regime; overrides --limit")
    parser.add_argument("--max-new", type=int, default=32)
    args = parser.parse_args()
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"refusing to overwrite output: {out}")
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    checkpoint, model = load_model(args.ckpt, device)
    tokenizer = Tokenizer.from_file(args.tokenizer)
    episodes = read_episodes(args.episodes, args.limit, args.per_split)

    def ask(prompt):
        return generate(model, tokenizer, prompt, device, max_new=args.max_new, temp=0.0)

    totals = collections.Counter()
    one_step = collections.Counter()
    loop_success = collections.Counter()
    readout_success = collections.Counter()
    examples = []
    for index, episode in enumerate(episodes):
        bucket = episode["split"]
        totals[bucket] += 1
        first_prompt = transition_prompt(episode["initial_memory"], episode["operations"][0])
        first_predicted = parse_memory(ask(first_prompt))
        first_expected = apply_operation(episode["initial_memory"], episode["operations"][0])
        one_step[bucket] += int(first_predicted == first_expected)
        rollout = rollout_episode(episode, ask)
        loop_success[bucket] += int(rollout["success"])
        readout_ok = False
        readout_response = ""
        if rollout["success"]:
            variable = episode["readout_variable"]
            readout_response = ask(readout_prompt(rollout["memory"], variable))
            readout_ok = parse_answer(readout_response) == rollout["memory"][variable]
            readout_success[bucket] += int(readout_ok)
        if len(examples) < 12:
            examples.append({
                "id": episode["id"], "split": bucket, "program_length": episode["program_length"],
                "first_step_correct": first_predicted == first_expected,
                "closed_loop_success": rollout["success"], "readout_correct": readout_ok,
                "rollout": rollout, "readout_response": readout_response,
            })
        if (index + 1) % 20 == 0 or index + 1 == len(episodes):
            print(f"[vrwm] {index + 1}/{len(episodes)} closed_loop={sum(loop_success.values())}", flush=True)
    by_split = {
        split: {
            "episodes": totals[split],
            "one_step_correct": one_step[split],
            "closed_loop_correct": loop_success[split],
            "readout_correct_given_closed_loop": readout_success[split],
            "one_step_accuracy": one_step[split] / totals[split],
            "closed_loop_accuracy": loop_success[split] / totals[split],
            "readout_accuracy_given_closed_loop": readout_success[split] / max(loop_success[split], 1),
        }
        for split in sorted(totals)
    }
    result = {
        "schema": "shohin-vrwm-eval-v1",
        "checkpoint": args.ckpt,
        "step": checkpoint.get("step"),
        "episodes": args.episodes,
        "device": device,
        "max_new": args.max_new,
        "by_split": by_split,
        "examples": examples,
        "claim_boundary": (
            "Closed-loop success means model-produced canonical states survived every transition; "
            "it is an algorithmic working-memory result, not general long-context reasoning."
        ),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"step": result["step"], "by_split": by_split}, sort_keys=True))


if __name__ == "__main__":
    main()
