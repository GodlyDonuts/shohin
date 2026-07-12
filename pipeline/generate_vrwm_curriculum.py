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
from vrwm_protocol import apply_operation, canonical_memory, readout_prompt, transition_prompt


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


def rows_from_episode(episode):
    rows, memory = [], dict(episode["initial_memory"])
    for index, operation in enumerate(episode["operations"]):
        expected = episode["expected_memories"][index]
        prompt = transition_prompt(memory, operation)
        rows.append({
            "question": prompt,
            "completion_prompt": prompt,
            "response": canonical_memory(expected),
            "source": "vrwm_transition_train",
            "training_group": "vrwm",
            "episode_id": episode["id"],
            "transition_index": index,
            "program_length": episode["program_length"],
            "expected_memory": canonical_memory(expected),
        })
        memory = expected
    variable = episode["readout_variable"]
    prompt = readout_prompt(memory, variable)
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
    })
    return rows


def normalized_prompt(text):
    return " ".join(re.findall(r"\w+", str(text).lower()))


def episode_prompt_signatures(episode):
    """Return every inference prompt used by this episode's transitions/readout."""
    return {normalized_prompt(row["completion_prompt"]) for row in rows_from_episode(episode)}


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
    parser.add_argument("--eval-per-length", type=int, default=80)
    parser.add_argument("--seed", type=int, default=20260712)
    args = parser.parse_args()
    if (args.train_episodes <= 0 or args.train_max_steps <= 0 or args.eval_per_length <= 0
            or args.train_value_limit <= 0 or args.train_const_limit <= 0):
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
    train_rows_all = [row for episode in train for row in rows_from_episode(episode)]
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
            if episode_prompt_signatures(episode) & train_prompt_signatures:
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
