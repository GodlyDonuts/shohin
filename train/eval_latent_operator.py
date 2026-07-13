#!/usr/bin/env python3
"""Held-out exact-answer evaluation for continuous latent-rollout experiments.

The generated operator set has disjoint held-out domains, labels, value bands,
event language, and longer depths (5/6/8) than the answer-only training set
(1--4).  A result remains narrow structured-reasoning/context evidence: it is
not a substitute for public benchmarks or direct model interaction.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
from pathlib import Path

import torch
from tokenizers import Tokenizer

from latent_rollout import generate_with_latents
from model import GPT, GPTConfig


FINAL = re.compile(r"the\s+answer\s+is\s*(-?\d+)\b", re.IGNORECASE)


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_rows(path: str):
    rows = []
    with open(path) as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            required = ("question", "response", "answer", "depth", "family", "heldout")
            if any(key not in row for key in required) or not row["heldout"]:
                raise ValueError("invalid held-out latent-operator row at line {}".format(line_number))
            rows.append(row)
    if not rows:
        raise ValueError("held-out latent operator data is empty")
    return rows


def select_rows(rows, per_depth: int, seed: int):
    if per_depth <= 0:
        raise ValueError("per_depth must be positive")
    grouped = collections.defaultdict(list)
    for row in rows:
        grouped[int(row["depth"])].append(row)
    selected = []
    for depth in sorted(grouped):
        candidates = grouped[depth]
        if len(candidates) < per_depth:
            raise ValueError("depth {} has {}, need {} rows".format(depth, len(candidates), per_depth))
        selected.extend(sorted(
            candidates,
            key=lambda row: hashlib.sha256((str(seed) + "\0" + row["question"]).encode()).hexdigest(),
        )[:per_depth])
    return selected


def final_answer(response: str):
    matches = FINAL.findall(str(response))
    return int(matches[-1]) if matches else None


def summarize(rows):
    by_steps = {}
    for latent_steps in sorted({int(row["latent_steps"]) for row in rows}):
        matching = [row for row in rows if int(row["latent_steps"]) == latent_steps]
        depths = {}
        for depth in sorted({int(row["depth"]) for row in matching}):
            depth_rows = [row for row in matching if int(row["depth"]) == depth]
            correct = sum(bool(row["correct"]) for row in depth_rows)
            depths[str(depth)] = {"cases": len(depth_rows), "correct": correct, "accuracy": correct / len(depth_rows)}
        correct = sum(bool(row["correct"]) for row in matching)
        by_steps[str(latent_steps)] = {
            "cases": len(matching),
            "correct": correct,
            "accuracy": correct / len(matching),
            "by_depth": depths,
        }
    return by_steps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--latent-steps", nargs="+", type=int, default=[0, 1, 2, 4, 8])
    parser.add_argument("--per-depth", type=int, default=200)
    parser.add_argument("--max-new", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--eos", default="<|endoftext|>")
    args = parser.parse_args()
    if any(step < 0 for step in args.latent_steps):
        raise SystemExit("latent steps must be non-negative")
    if args.max_new <= 0:
        raise SystemExit("max-new must be positive")
    out = Path(args.out)
    if out.exists():
        raise SystemExit("refusing existing output: {}".format(out))

    cases = select_rows(load_rows(args.data), args.per_depth, args.seed)
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    tokenizer = Tokenizer.from_file(args.tokenizer)
    eos_id = tokenizer.token_to_id(args.eos)
    if eos_id is None:
        raise SystemExit("tokenizer EOS token is missing")
    checkpoint = torch.load(args.ckpt, map_location="cpu")
    model = GPT(GPTConfig(**checkpoint["cfg"])).to(device).eval()
    model.load_state_dict(checkpoint["model"])
    rows = []
    ordered_steps = list(dict.fromkeys(args.latent_steps))
    for latent_steps in ordered_steps:
        for index, case in enumerate(cases, 1):
            prompt_ids = torch.tensor([tokenizer.encode(case["question"]).ids], dtype=torch.long, device=device)
            generated = generate_with_latents(model, prompt_ids, latent_steps, eos_id, args.max_new)
            response = tokenizer.decode(generated.tolist(), skip_special_tokens=False)
            prediction = final_answer(response)
            row = {
                "latent_steps": latent_steps,
                "depth": int(case["depth"]),
                "family": case["family"],
                "question": case["question"],
                "expected": int(case["answer"]),
                "response": response,
                "prediction": prediction,
                "correct": prediction == int(case["answer"]),
            }
            rows.append(row)
            if index % 25 == 0 or index == len(cases):
                correct = sum(bool(item["correct"]) for item in rows if item["latent_steps"] == latent_steps)
                print("[latent-eval] L={} {}/{} correct={}".format(latent_steps, index, len(cases), correct), flush=True)
    result = {
        "audit": "latent_operator_heldout_v1",
        "checkpoint": args.ckpt,
        "checkpoint_step": checkpoint.get("step"),
        "checkpoint_latent_metadata": checkpoint.get("latent_rollout"),
        "device": device,
        "data": args.data,
        "data_sha256": sha256_file(args.data),
        "per_depth": args.per_depth,
        "seed": args.seed,
        "latent_steps": ordered_steps,
        "summary": summarize(rows),
        "rows": rows,
        "claim_boundary": (
            "This is a held-out structured operator and context-length extrapolation test. "
            "It does not establish broad reasoning, benchmark performance, or general autonomous planning."
        ),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("[latent-eval] summary=" + json.dumps(result["summary"], sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
