#!/usr/bin/env python3
"""Generate an executable working-memory curriculum and OOD rollout episodes.

Unlike the retired typed-state curriculum, each supervised transition has one
stable interface and does not reveal the terminal answer.  Evaluation feeds the
model's own previous state back into the next turn, which exposes formatting
imitation and one-step errors immediately.
"""
import argparse
import hashlib
import json
import os
import random
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train"))
from vrwm_protocol import (PROMPT_STYLES, apply_operation, canonical_memory, readout_prompt,
                           repair_prompt, transition_prompt)


OP_KINDS = ("add_const", "sub_const", "add_var", "sub_var", "swap")


def make_operation(rng, const_limit, const_minimum=1):
    kind = rng.choice(OP_KINDS)
    target = rng.choice(("a", "b"))
    if kind in {"add_const", "sub_const"}:
        value = rng.randint(const_minimum, const_limit)
        return {"kind": kind, "target": target, "value": value}
    if kind in {"add_var", "sub_var"}:
        return {"kind": kind, "target": target, "source": "b" if target == "a" else "a"}
    return {"kind": "swap", "target": "a", "source": "b"}


def signed_value(rng, minimum, maximum):
    value = rng.randint(minimum, maximum)
    return value if rng.randrange(2) else -value


def make_episode(rng, episode_id, length, value_limit, const_limit, split,
                 value_minimum=0, const_minimum=1):
    memory = {
        "a": signed_value(rng, value_minimum, value_limit),
        "b": signed_value(rng, value_minimum, value_limit),
    }
    initial = dict(memory)
    operations, expected = [], []
    for _ in range(length):
        operation = make_operation(rng, const_limit, const_minimum)
        operations.append(operation)
        memory = apply_operation(memory, operation)
        expected.append(dict(memory))
    return {
        "id": episode_id,
        "split": split,
        "program_length": length,
        "initial_memory": initial,
        "operations": operations,
        "expected_memories": expected,
        "readout_variable": rng.choice(("a", "b")),
    }


def episode_signature(episode):
    return json.dumps({
        "initial_memory": episode["initial_memory"],
        "operations": episode["operations"],
        "readout_variable": episode["readout_variable"],
    }, sort_keys=True, separators=(",", ":"))


def scratch_transition(memory, operation, expected):
    """Render a deterministic one-line arithmetic check before the state."""
    kind, target = operation["kind"], operation["target"]
    before = memory[target]
    after = expected[target]
    if kind == "add_const":
        check = f"check: {target}={before}+{operation['value']}={after}"
    elif kind == "sub_const":
        check = f"check: {target}={before}-{operation['value']}={after}"
    elif kind == "add_var":
        source = operation["source"]
        check = f"check: {target}={before}+{memory[source]}={after}"
    elif kind == "sub_var":
        source = operation["source"]
        check = f"check: {target}={before}-{memory[source]}={after}"
    else:
        check = f"check: a,b={memory['b']},{memory['a']}"
    return f"{check}\n{canonical_memory(expected)}"


def repair_proposals(memory, operation, expected, count):
    """Create deterministic plausible drafts; only the model supplies repairs at inference."""
    if count <= 0:
        return []
    proposals = [dict(expected)]
    target = operation["target"]
    offsets = (1, -1, 10, -10, 100, -100)
    for index in range(1, count):
        proposal = dict(expected)
        if operation["kind"] == "swap":
            proposal = dict(memory)
        else:
            proposal[target] += offsets[(index - 1) % len(offsets)]
        if proposal == expected:
            proposal[target] += 1
        proposals.append(proposal)
    return proposals


def rows_from_episode(episode, prompt_style="default", response_mode="state", repair_examples=0):
    if response_mode not in {"state", "scratch"}:
        raise ValueError(f"unknown response mode: {response_mode!r}")
    if repair_examples < 0:
        raise ValueError("repair_examples must be non-negative")
    rows, memory = [], dict(episode["initial_memory"])
    for index, operation in enumerate(episode["operations"]):
        expected = episode["expected_memories"][index]
        prompt = transition_prompt(memory, operation, style=prompt_style)
        response = canonical_memory(expected)
        if response_mode == "scratch":
            response = scratch_transition(memory, operation, expected)
        rows.append({
            "question": prompt,
            "completion_prompt": prompt,
            "response": response,
            "source": "vrwm_transition_train" if response_mode == "state" else "vrwm_transition_scratch_train",
            "training_group": "vrwm",
            "episode_id": episode["id"],
            "transition_index": index,
            "program_length": episode["program_length"],
            "expected_memory": canonical_memory(expected),
            "prompt_style": prompt_style,
            "response_mode": response_mode,
        })
        for proposal in repair_proposals(memory, operation, expected, repair_examples):
            repair = repair_prompt(memory, operation, proposal, style=prompt_style)
            repair_response = canonical_memory(expected)
            if response_mode == "scratch":
                repair_response = scratch_transition(memory, operation, expected)
            rows.append({
                "question": repair,
                "completion_prompt": repair,
                "response": repair_response,
                "source": "vrwm_repair_train",
                "training_group": "vrwm",
                "episode_id": episode["id"],
                "transition_index": index,
                "program_length": episode["program_length"],
                "expected_memory": canonical_memory(expected),
                "proposal_memory": canonical_memory(proposal),
                "prompt_style": prompt_style,
                "response_mode": response_mode,
            })
        memory = expected
    variable = episode["readout_variable"]
    prompt = readout_prompt(memory, variable, style=prompt_style)
    rows.append({
        "question": prompt,
        "completion_prompt": prompt,
        "response": f"answer={memory[variable]}",
        "source": "vrwm_readout_train",
        "training_group": "vrwm",
        "episode_id": episode["id"],
        "transition_index": episode["program_length"],
        "program_length": episode["program_length"],
        "expected_memory": canonical_memory(memory),
        "prompt_style": prompt_style,
        "response_mode": response_mode,
    })
    return rows


def normalized_prompt(text):
    return " ".join(re.findall(r"\w+", str(text).lower()))


def episode_prompt_signatures(episode, prompt_style="default", include_repair=False):
    """Return every inference prompt used by this episode's transitions/readout."""
    return {
        normalized_prompt(row["completion_prompt"])
        for row in rows_from_episode(
            episode, prompt_style=prompt_style, repair_examples=1 if include_repair else 0
        )
    }


def deduplicate_rows(rows):
    """Keep one supervised completion per exact inference prompt."""
    unique, seen, dropped = [], set(), 0
    for row in rows:
        key = normalized_prompt(row["completion_prompt"])
        if key in seen:
            dropped += 1
            continue
        seen.add(key)
        unique.append(row)
    return unique, dropped


def write_jsonl(path, rows):
    path = Path(path)
    partial = path.with_suffix(path.suffix + ".partial")
    if path.exists() or partial.exists():
        raise SystemExit(f"refusing to overwrite existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with partial.open("w") as output:
        for row in rows:
            output.write(json.dumps(row, sort_keys=True) + "\n")
    os.replace(partial, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-out", required=True)
    parser.add_argument("--eval-out", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--train-episodes", type=int, default=20000)
    parser.add_argument("--train-max-steps", type=int, default=4)
    parser.add_argument("--train-value-limit", type=int, default=99)
    parser.add_argument("--train-const-limit", type=int, default=31)
    parser.add_argument("--train-styles", nargs="+", choices=PROMPT_STYLES, default=["default"],
                        help="prompt templates included in supervised rows")
    parser.add_argument("--eval-style", choices=PROMPT_STYLES, default="default",
                        help="template reserved for the generated held-out episodes")
    parser.add_argument("--response-mode", choices=("state", "scratch"), default="state",
                        help="state-only control or deterministic calculation-check completion")
    parser.add_argument("--repair-examples", type=int, default=0,
                        help="supervised proposed-state checks per transition (0 disables self-repair data)")
    parser.add_argument("--eval-per-length", type=int, default=80)
    parser.add_argument("--seed", type=int, default=20260712)
    args = parser.parse_args()
    if (args.train_episodes <= 0 or args.train_max_steps <= 0 or args.eval_per_length <= 0
            or args.train_value_limit <= 0 or args.train_const_limit <= 0 or args.repair_examples < 0):
        raise ValueError("episode and length arguments must be positive")
    if any(Path(path).exists() for path in (args.train_out, args.eval_out, args.report)):
        raise SystemExit("refusing to overwrite an existing VRWM artifact")

    rng = random.Random(args.seed)
    # Small values dominate early learning, while medium/large values prevent
    # an otherwise tiny prompt space from collapsing into repeated examples.
    train = []
    for index in range(args.train_episodes):
        fraction = index / max(args.train_episodes - 1, 1)
        if fraction < 0.10:
            value_limit, const_limit = 9, 5
        elif fraction < 0.40:
            value_limit, const_limit = min(49, args.train_value_limit), min(15, args.train_const_limit)
        else:
            value_limit, const_limit = args.train_value_limit, args.train_const_limit
        train.append(make_episode(
            rng, f"train-{index:06d}", rng.randint(1, args.train_max_steps),
            value_limit, const_limit, "train",
        ))
    eval_specs = (
        # The training prompt space is deliberately small at its first stage.
        # Reserve wider numeric bands for every eval episode so no purported
        # in-distribution score can be an accidental prompt replay.
        (4, 1000, 2000, 128, 255, "value_ood_len4"),
        (8, 1000, 2000, 128, 255, "value_and_length_ood_len8"),
        (16, 1000, 2000, 128, 255, "value_and_length_ood_len16"),
        (32, 1000, 2000, 128, 255, "value_and_length_ood_len32"),
        (8, 5000, 10000, 512, 1023, "wide_range_and_length_ood_len8"),
    )
    train_rows_all = [
        row
        for episode in train
        for style in args.train_styles
        for row in rows_from_episode(
            episode, prompt_style=style, response_mode=args.response_mode,
            repair_examples=args.repair_examples,
        )
    ]
    train_rows, duplicate_train_prompts = deduplicate_rows(train_rows_all)
    train_prompt_signatures = {normalized_prompt(row["completion_prompt"]) for row in train_rows}
    evaluation = []
    for length, value_minimum, value_limit, const_minimum, const_limit, split in eval_specs:
        attempts = 0
        while sum(row["split"] == split for row in evaluation) < args.eval_per_length:
            attempts += 1
            if attempts > args.eval_per_length * 100:
                raise RuntimeError(f"could not construct prompt-disjoint {split} evaluation")
            index = sum(row["split"] == split for row in evaluation)
            episode = make_episode(
                rng, f"{split}-{index:04d}", length, value_limit, const_limit, split,
                value_minimum=value_minimum, const_minimum=const_minimum,
            )
            if episode_prompt_signatures(
                episode, prompt_style=args.eval_style, include_repair=args.repair_examples > 0
            ) & train_prompt_signatures:
                continue
            evaluation.append(episode)
    train_signatures = {episode_signature(row) for row in train}
    eval_signatures = {episode_signature(row) for row in evaluation}
    if train_signatures & eval_signatures:
        raise RuntimeError("train/eval episode overlap")
    rng.shuffle(train_rows)
    for row in train_rows:
        if row["question"] != row["completion_prompt"] or not row["response"]:
            raise RuntimeError("malformed supervised VRWM row")
    write_jsonl(args.train_out, train_rows)
    write_jsonl(args.eval_out, evaluation)
    report = {
        "schema": "shohin-vrwm-v1",
        "seed": args.seed,
        "train_episodes": len(train),
        "train_rows": len(train_rows),
        "duplicate_train_prompts_dropped": duplicate_train_prompts,
        "train_max_steps": args.train_max_steps,
        "train_styles": args.train_styles,
        "eval_style": args.eval_style,
        "response_mode": args.response_mode,
        "repair_examples": args.repair_examples,
        "evaluation_episodes": len(evaluation),
        "evaluation_by_split": dict(sorted(Counter(row["split"] for row in evaluation).items())),
        "training_group": "vrwm",
        "train_sha256": hashlib.sha256(Path(args.train_out).read_bytes()).hexdigest(),
        "eval_sha256": hashlib.sha256(Path(args.eval_out).read_bytes()).hexdigest(),
        "protocol": "model emits one canonical state; controller forwards it without correction",
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
