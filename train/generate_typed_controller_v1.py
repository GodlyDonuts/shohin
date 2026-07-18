#!/usr/bin/env python3
"""Generate typed-controller SFT data from SSC-family solvers.

Produces atomic / rollout / resume rows with an exact completion_prompt boundary
compatible with train/sft.py --prompt-override-field completion_prompt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any


PROTOCOL = "R12-TYPED-CONTROLLER-v1"
FAMILIES = (
    "multiply_subtract",
    "base_conversion",
    "sequential_state",
    "modular_update",
)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_json(payload: object) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


def apply_op(state: int, op: str, arg: int) -> int:
    if op == "add":
        return state + arg
    if op == "subtract":
        return state - arg
    if op == "multiply":
        return state * arg
    if op == "remainder":
        return state % arg
    if op == "horner":
        # Horner step: state = state * base + digit; arg packs (base, digit)
        base, digit = divmod(arg, 1000)
        return state * base + digit
    raise ValueError(op)


def format_ops(schedule: list[tuple[str, int]]) -> str:
    parts = []
    for op, arg in schedule:
        if op == "horner":
            base, digit = divmod(arg, 1000)
            parts.append(f"horner {base} {digit}")
        else:
            parts.append(f"{op} {arg}")
    return " | ".join(parts)


def format_register(state: int, schedule: list[tuple[str, int]], cursor: int) -> str:
    return f"state={state}; ops={format_ops(schedule)}; cursor={cursor}"


def parse_op_token(op: str, arg: int) -> str:
    if op == "horner":
        base, digit = divmod(arg, 1000)
        return f"horner {base} {digit}"
    return f"{op} {arg}"


def step_line(op: str, arg: int, nxt: int, cursor: int, done: int) -> str:
    return f"{parse_op_token(op, arg)} -> {nxt}; cursor={cursor}; done={done}"


def sample_case(rng: random.Random, family: str) -> dict[str, Any]:
    if family == "multiply_subtract":
        a = rng.randint(20, 99)
        mult = rng.randint(2, 19)
        product = a * mult
        sub = rng.randint(1, min(50, product - 1))
        schedule = [("multiply", mult), ("subtract", sub)]
        start = a
        question = f"Compute (({a} * {mult}) - {sub})."
    elif family == "sequential_state":
        start = rng.randint(5, 50)
        addend = rng.randint(1, 25)
        mult = rng.randint(2, 7)
        mid = (start + addend) * mult
        sub = rng.randint(1, min(40, mid - 1))
        schedule = [("add", addend), ("multiply", mult), ("subtract", sub)]
        question = f"Start at {start}. Add {addend}, multiply by {mult}, subtract {sub}."
    elif family == "modular_update":
        a = rng.randint(10, 99)
        b = rng.randint(10, 99)
        mod = rng.randint(3, 25)
        schedule = [("add", b), ("remainder", mod)]
        start = a
        question = f"Compute ({a} + {b}) mod {mod}."
    elif family == "base_conversion":
        base = rng.randint(2, 12)
        digits = [rng.randint(0, base - 1) for _ in range(3)]
        if digits[0] == 0:
            digits[0] = rng.randint(1, base - 1)
        start = 0
        schedule = [("horner", base * 1000 + d) for d in digits]
        glyph = "".join(str(d) for d in digits)
        question = f"Convert base-{base} numeral {glyph} to decimal."
    else:
        raise ValueError(family)

    state = start
    states = [start]
    for op, arg in schedule:
        state = apply_op(state, op, arg)
        states.append(state)
    return {
        "family": family,
        "question": question,
        "initial_state": start,
        "schedule": schedule,
        "states": states,
        "final_answer": state,
    }


def rows_for_case(case: dict[str, Any], *, group: str) -> list[dict[str, Any]]:
    schedule: list[tuple[str, int]] = case["schedule"]
    states: list[int] = case["states"]
    rows: list[dict[str, Any]] = []

    # Atomic transitions
    for i, (op, arg) in enumerate(schedule):
        prompt = (
            f"Problem: {format_register(states[i], schedule, i)}\nWork:"
        )
        done = 1 if i + 1 == len(schedule) else 0
        completion = step_line(op, arg, states[i + 1], i + 1, done)
        if done:
            completion += f"\nanswer={states[-1]}"
        rows.append(
            {
                "training_group": "atomic",
                "family": case["family"],
                "split_group": group,
                "question": prompt,
                "completion_prompt": prompt,
                "response": completion,
                "final_answer": case["final_answer"],
            }
        )

    # Full rollout from cursor 0
    prompt = f"Problem: {format_register(states[0], schedule, 0)}\nWork:"
    lines = []
    for i, (op, arg) in enumerate(schedule):
        done = 1 if i + 1 == len(schedule) else 0
        lines.append(step_line(op, arg, states[i + 1], i + 1, done))
    lines.append(f"answer={states[-1]}")
    rows.append(
        {
            "training_group": "rollout",
            "family": case["family"],
            "split_group": group,
            "question": prompt,
            "completion_prompt": prompt,
            "response": "\n".join(lines),
            "final_answer": case["final_answer"],
        }
    )

    # Resume from a mid cursor when depth > 1
    if len(schedule) > 1:
        cursor = len(schedule) // 2
        prompt = f"Problem: {format_register(states[cursor], schedule, cursor)}\nWork:"
        lines = []
        for i in range(cursor, len(schedule)):
            op, arg = schedule[i]
            done = 1 if i + 1 == len(schedule) else 0
            lines.append(step_line(op, arg, states[i + 1], i + 1, done))
        lines.append(f"answer={states[-1]}")
        rows.append(
            {
                "training_group": "resume",
                "family": case["family"],
                "split_group": group,
                "question": prompt,
                "completion_prompt": prompt,
                "response": "\n".join(lines),
                "final_answer": case["final_answer"],
            }
        )

    return rows


def build_corpus(*, seed: int, n_train: int, n_heldout: int) -> tuple[list[dict], list[dict]]:
    # Keep confirmation seed 2026071502 out of training.
    if seed == 2026071502:
        raise ValueError("confirmation seed is reserved")
    rng = random.Random(seed)
    train: list[dict] = []
    held: list[dict] = []
    # Equal family coverage
    per_train = n_train // len(FAMILIES)
    per_held = n_heldout // len(FAMILIES)
    seen_q: set[str] = set()
    for family in FAMILIES:
        for _ in range(per_train):
            for _attempt in range(100):
                case = sample_case(rng, family)
                key = case["question"]
                if key in seen_q:
                    continue
                seen_q.add(key)
                train.extend(rows_for_case(case, group="train"))
                break
        for _ in range(per_held):
            for _attempt in range(100):
                case = sample_case(rng, family)
                key = case["question"]
                if key in seen_q:
                    continue
                seen_q.add(key)
                held.extend(rows_for_case(case, group="heldout"))
                break
    return train, held


def write_jsonl(path: Path, rows: list[dict]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    return _sha256_file(path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=2026071611)
    ap.add_argument("--n-train-cases", type=int, default=12_000)
    ap.add_argument("--n-heldout-cases", type=int, default=1_024)
    args = ap.parse_args()

    train, held = build_corpus(
        seed=args.seed, n_train=args.n_train_cases, n_heldout=args.n_heldout_cases
    )
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    train_path = out / "train.jsonl"
    held_path = out / "heldout.jsonl"
    train_sha = write_jsonl(train_path, train)
    held_sha = write_jsonl(held_path, held)

    # Overlap check on natural-language questions embedded in prompts is weak;
    # check exact completion_prompt collisions instead.
    train_prompts = {r["completion_prompt"] for r in train}
    held_prompts = {r["completion_prompt"] for r in held}
    overlap = sorted(train_prompts & held_prompts)

    audit = {
        "protocol": PROTOCOL,
        "seed": args.seed,
        "n_train_rows": len(train),
        "n_heldout_rows": len(held),
        "n_train_cases": args.n_train_cases,
        "n_heldout_cases": args.n_heldout_cases,
        "train_sha256": train_sha,
        "heldout_sha256": held_sha,
        "exact_prompt_overlap": len(overlap),
        "group_counts_train": {
            g: sum(1 for r in train if r["training_group"] == g)
            for g in ("atomic", "rollout", "resume")
        },
        "group_counts_heldout": {
            g: sum(1 for r in held if r["training_group"] == g)
            for g in ("atomic", "rollout", "resume")
        },
    }
    if overlap:
        raise SystemExit(f"exact prompt overlap: {len(overlap)}")
    audit_path = out / "audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    audit["audit_sha256"] = _sha256_file(audit_path)
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
