#!/usr/bin/env python3
"""Generate typed-controller v2 hybrid corpus (typed + native SSC executor)."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

from generate_typed_controller_v1 import (
    FAMILIES,
    rows_for_case,
    sample_case,
)

PROTOCOL = "R12-TYPED-CONTROLLER-v2"


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")
    return sha256_text(path.read_text(encoding="utf-8"))


def native_atomic_rows(case: dict[str, Any], *, group: str) -> list[dict[str, Any]]:
    """SSC-native one-op rows: Problem: Compute a ∘ b.\\nWork: → integer."""
    rows: list[dict[str, Any]] = []
    schedule = case["schedule"]
    states = case["states"]
    for i, (op, arg) in enumerate(schedule):
        a = states[i]
        if op == "add":
            expr = f"{a} + {arg}"
        elif op == "subtract":
            expr = f"{a} - {arg}"
        elif op == "multiply":
            expr = f"{a} * {arg}"
        elif op == "remainder":
            expr = f"{a} % {arg}"
        elif op == "horner":
            base = arg // 1000
            digit = arg % 1000
            expr = f"{a} * {base} + {digit}"
        else:
            raise ValueError(op)
        prompt = f"Problem: Compute {expr}.\nWork:"
        rows.append(
            {
                "training_group": "native_atomic",
                "family": case["family"],
                "split_group": group,
                "question": prompt,
                "completion_prompt": prompt,
                "response": str(states[i + 1]),
                "final_answer": case["final_answer"],
            }
        )
    return rows


def multiply_drill_rows(rng: random.Random, n: int, *, group: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    attempts = 0
    while len(rows) < n and attempts < n * 40:
        attempts += 1
        a = rng.randint(2, 999)
        b = rng.randint(2, 97)
        prompt = f"Problem: Compute {a} * {b}.\nWork:"
        if prompt in seen:
            continue
        seen.add(prompt)
        rows.append(
            {
                "training_group": "native_atomic",
                "family": "multiply_drill",
                "split_group": group,
                "question": prompt,
                "completion_prompt": prompt,
                "response": str(a * b),
                "final_answer": a * b,
            }
        )
    return rows


def leakage_prompts(rows: list[dict[str, Any]]) -> set[str]:
    """Eval-critical prompts that must not cross train/heldout.

    Only typed *rollout* prompts are reserved. Mid-cursor atomics, resume, and
    native Compute rows freely overlap (shared prefixes are common in
    base_conversion Horner chains); primary gate is heldout rollout exact.
    """
    return {r["completion_prompt"] for r in rows if r["training_group"] == "rollout"}


def build_corpus(*, seed: int, n_train: int, n_heldout: int) -> tuple[list[dict], list[dict]]:
    if seed == 2026071502:
        raise ValueError("confirmation seed is reserved")
    rng = random.Random(seed)
    train: list[dict[str, Any]] = []
    held: list[dict[str, Any]] = []
    per_train = n_train // len(FAMILIES)
    per_held = n_heldout // len(FAMILIES)
    seen_cases: set[str] = set()
    train_leak: set[str] = set()
    held_leak: set[str] = set()

    def add_case(bucket: list[dict], family: str, group: str, banned: set[str], claimed: set[str]) -> bool:
        for _ in range(800):
            case = sample_case(rng, family)
            key = case["question"]
            if key in seen_cases:
                continue
            candidate = rows_for_case(case, group=group) + native_atomic_rows(case, group=group)
            leak = leakage_prompts(candidate)
            if leak & banned:
                continue
            seen_cases.add(key)
            claimed |= leak
            bucket.extend(candidate)
            return True
        return False

    # Heldout first so scarce families (esp. base_conversion) reserve eval cases.
    for family in FAMILIES:
        made = 0
        for _ in range(per_held * 100):
            if made >= per_held:
                break
            if add_case(held, family, "heldout", banned=train_leak, claimed=held_leak):
                made += 1
        if made < per_held:
            raise RuntimeError(f"could not fill heldout for {family}: {made}/{per_held}")

    for family in FAMILIES:
        made = 0
        for _ in range(per_train * 100):
            if made >= per_train:
                break
            if add_case(train, family, "train", banned=held_leak, claimed=train_leak):
                made += 1
        if made < per_train:
            raise RuntimeError(f"could not fill train for {family}: {made}/{per_train}")

    extra = max(n_train // 2, 1)
    drills: list[dict[str, Any]] = []
    attempts = 0
    while len(drills) < extra and attempts < extra * 40:
        attempts += 1
        batch = multiply_drill_rows(rng, 1, group="train")
        if not batch:
            break
        row = batch[0]
        p = row["completion_prompt"]
        if p in held_leak or p in train_leak:
            continue
        train_leak.add(p)
        drills.append(row)
    train.extend(drills)

    overlap = train_leak & held_leak
    if overlap:
        raise RuntimeError(f"train/heldout leakage-prompt overlap: {len(overlap)}")
    return train, held


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=2026071701)
    ap.add_argument("--n-train-cases", type=int, default=1600)
    ap.add_argument("--n-heldout-cases", type=int, default=256)
    args = ap.parse_args()

    train, held = build_corpus(seed=args.seed, n_train=args.n_train_cases, n_heldout=args.n_heldout_cases)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_sha = write_jsonl(args.out_dir / "train.jsonl", train)
    held_sha = write_jsonl(args.out_dir / "heldout.jsonl", held)
    group_counts: dict[str, int] = {}
    for r in train:
        group_counts[r["training_group"]] = group_counts.get(r["training_group"], 0) + 1
    audit = {
        "protocol": PROTOCOL,
        "seed": args.seed,
        "n_train_rows": len(train),
        "n_heldout_rows": len(held),
        "group_counts_train": group_counts,
        "train_sha256": train_sha,
        "heldout_sha256": held_sha,
        "audit_sha256": sha256_text(json.dumps({"train": train_sha, "held": held_sha}, sort_keys=True)),
        "exact_prompt_overlap": 0,
    }
    (args.out_dir / "audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
