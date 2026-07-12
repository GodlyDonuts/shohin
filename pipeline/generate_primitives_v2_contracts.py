#!/usr/bin/env python3
"""Render verified primitive tasks under the exact contracts Shohin must learn.

V5 showed that a Q/A-only primitive curriculum teaches some routines, but its
transfer is sharply prompt-dependent. This generator takes disjoint, solver-
verified v1 source tasks and produces contract-diverse supervision for direct
answers, review, verified-state use, and compact-state reuse. The holdout is
rendered by the same code from the v1 heldout source and is never SFT input.
"""
import argparse
import collections
import hashlib
import json
import random
import re
from pathlib import Path


FAMILIES = (
    "arithmetic", "base_conversion", "state_update", "sort_unique",
    "string_insert", "syllogism", "correction",
)
CONTRACTS = ("qa", "direct", "cot", "review", "scaffold", "compact", "reuse")


def normalized(text):
    return " ".join(re.findall(r"\w+", str(text).lower()))


def reasoning(response):
    match = re.search(r"<think>(.*?)</think>", str(response), flags=re.S)
    text = match.group(1) if match else str(response)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        raise ValueError("source response has no reasoning text")
    return text


def wrong_answer(answer, family):
    value = str(answer).strip()
    if family == "syllogism":
        return "no" if value.lower() == "yes" else "yes"
    if family == "string_insert":
        return value[::-1]
    if family == "sort_unique":
        return value[::-1]
    try:
        return str(int(value) + 1)
    except ValueError:
        return "incorrect"


def completion(answer):
    return f"The answer is {answer}."


def render(source, contract):
    """Return one exact inference prompt and its verified continuation."""
    base = str(source["question"]).strip()
    answer = str(source["answer"]).strip()
    state = reasoning(source["response"])
    if contract == "qa":
        return f"Question: {base}\nAnswer:", str(source["response"]).strip()
    if contract == "direct":
        return (
            f"Solve this task and return only its final answer.\nTask: {base}\nFinal answer:",
            answer,
        )
    if contract == "cot":
        return (
            f"Question: {base}\nAnswer: Work step by step. End with "
            "'The answer is <final answer>.'.",
            str(source["response"]).strip(),
        )
    if contract == "review":
        return (
            f"Question: {base}\nPrevious answer: {wrong_answer(answer, source['family'])}\n\n"
            "Independently check the previous answer. If it is wrong, correct it. "
            "Return only the final answer.\nAnswer:",
            answer,
        )
    if contract == "scaffold":
        return (
            f"Question: {base}\nVerified intermediate fact: {state}\n"
            "Use that fact. Return only the final answer.\nAnswer:",
            answer,
        )
    if contract == "compact":
        return (
            f"Question: {base}\nWrite one short state line beginning with 'state=' "
            "containing the necessary intermediate values, then give the final answer.\nAnswer:",
            f"state={state}\n{completion(answer)}",
        )
    if contract == "reuse":
        return (
            f"Question: {base}\nThe previous compact state was:\nstate={state}\n\n"
            "Use that state to solve the original question. Return only the final answer.\nAnswer:",
            answer,
        )
    raise ValueError(f"unknown contract {contract}")


def load_source(path):
    grouped = collections.defaultdict(list)
    seen = set()
    with open(path, errors="replace") as source:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            family = row.get("family")
            question = str(row.get("question") or "").strip()
            if family not in FAMILIES or not question or not row.get("answer") or not row.get("response"):
                continue
            key = normalized(question)
            if key in seen:
                raise ValueError(f"duplicate normalized source prompt: {question[:80]!r}")
            seen.add(key)
            grouped[family].append(row)
    missing = [family for family in FAMILIES if not grouped[family]]
    if missing:
        raise ValueError(f"source is missing families: {', '.join(missing)}")
    return grouped


def build_rows(grouped, per_family, seed, split):
    rng = random.Random(seed)
    rows, prompts = [], set()
    for family in FAMILIES:
        source_rows = list(grouped[family])
        rng.shuffle(source_rows)
        if len(source_rows) < per_family:
            raise ValueError(f"{split} source has {len(source_rows)} {family} rows, need {per_family}")
        for index, source in enumerate(source_rows[:per_family]):
            source_question = str(source["question"]).strip()
            for contract in CONTRACTS:
                prompt, response = render(source, contract)
                key = normalized(prompt)
                if key in prompts:
                    raise ValueError(f"duplicate contract prompt: {prompt[:100]!r}")
                prompts.add(key)
                rows.append({
                    "question": prompt,
                    "completion_prompt": prompt,
                    "response": response,
                    "answer": str(source["answer"]).strip(),
                    "source_question": source_question,
                    "source": f"primitives_v2_{split}",
                    "training_group": "contracts",
                    "family": family,
                    "contract": contract,
                    "source_index": index,
                })
    rng.shuffle(rows)
    return rows


def write_jsonl(path, rows):
    path = Path(path)
    partial = path.with_suffix(path.suffix + ".partial")
    if path.exists() or partial.exists():
        raise SystemExit(f"refusing to overwrite existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with partial.open("w") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
    partial.replace(path)


def report(rows):
    by_family = collections.Counter(row["family"] for row in rows)
    by_contract = collections.Counter(row["contract"] for row in rows)
    return {
        "rows": len(rows),
        "families": dict(sorted(by_family.items())),
        "contracts": dict(sorted(by_contract.items())),
        "prompt_sha256": hashlib.sha256(
            "\n".join(sorted(row["completion_prompt"] for row in rows)).encode()
        ).hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-source", required=True)
    parser.add_argument("--eval-source", required=True)
    parser.add_argument("--train-out", required=True)
    parser.add_argument("--eval-out", required=True)
    parser.add_argument("--train-per-family", type=int, default=4285)
    parser.add_argument("--eval-per-family", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260713)
    args = parser.parse_args()
    if args.train_per_family <= 0 or args.eval_per_family <= 0:
        raise ValueError("per-family counts must be positive")

    train_source = load_source(args.train_source)
    eval_source = load_source(args.eval_source)
    train_base = {normalized(row["question"]) for rows in train_source.values() for row in rows}
    eval_base = {normalized(row["question"]) for rows in eval_source.values() for row in rows}
    if train_base & eval_base:
        raise ValueError("train and heldout source prompts overlap")

    train = build_rows(train_source, args.train_per_family, args.seed, "train")
    heldout = build_rows(eval_source, args.eval_per_family, args.seed + 1, "heldout")
    train_prompts = {normalized(row["completion_prompt"]) for row in train}
    heldout_prompts = {normalized(row["completion_prompt"]) for row in heldout}
    if train_prompts & heldout_prompts:
        raise ValueError("rendered train and heldout prompts overlap")
    write_jsonl(args.train_out, train)
    write_jsonl(args.eval_out, heldout)
    print(json.dumps({
        "train": report(train), "heldout": report(heldout),
        "train_heldout_prompt_overlap": 0,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
