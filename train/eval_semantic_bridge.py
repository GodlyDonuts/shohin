#!/usr/bin/env python3
"""Evaluate a checkpoint on the frozen semantic-bridge held-out corpus.

This is a narrow diagnostic for the solver-verified bridge curriculum, not a
public-benchmark substitute.  It uses only the held-out templates and value
ranges, samples each family deterministically, and preserves ``<think>`` tags
so format imitation and numerical accuracy are measured separately.
"""

import argparse
import collections
import hashlib
import json
import random
import re
from pathlib import Path

import torch
from tokenizers import Tokenizer

from eval_suite import generate
from model import GPT, GPTConfig


FINAL = re.compile(r"the\s+answer\s+is\s*(-?\d+)\b", re.IGNORECASE)
THINK = re.compile(r"<think>\s*(.*?)\s*</think>", re.IGNORECASE | re.DOTALL)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_rows(path):
    rows = []
    with open(path) as source:
        for line_number, line in enumerate(source, 1):
            item = json.loads(line)
            required = ("question", "answer", "family", "response")
            if any(not str(item.get(key, "")).strip() for key in required):
                raise ValueError("invalid semantic-bridge row at line {}".format(line_number))
            rows.append(item)
    if not rows:
        raise ValueError("semantic-bridge held-out corpus is empty")
    return rows


def select_rows(rows, per_family, seed):
    """Hash-sort every family so the fixed evaluation subset is reproducible."""
    if per_family <= 0:
        raise ValueError("per_family must be positive")
    grouped = collections.defaultdict(list)
    for row in rows:
        grouped[row["family"]].append(row)
    selected = []
    for family in sorted(grouped):
        candidates = grouped[family]
        if len(candidates) < per_family:
            raise ValueError("{} has {} rows, need {}".format(family, len(candidates), per_family))
        ranked = sorted(
            candidates,
            key=lambda row: hashlib.sha256((str(seed) + "\0" + row["question"]).encode()).hexdigest(),
        )
        selected.extend(ranked[:per_family])
    return selected


def score_response(answer, response):
    matches = FINAL.findall(response)
    predicted = int(matches[-1]) if matches else None
    trace = THINK.search(response)
    return {
        "final": predicted,
        "answer_correct": predicted == int(answer),
        "trace_present": trace is not None,
        "visible_answer_correct": trace is not None and predicted == int(answer),
    }


def load_model(path, device, n_loop=0):
    checkpoint = torch.load(path, map_location="cpu")
    cfg = GPTConfig(**checkpoint["cfg"])
    if n_loop:
        cfg.n_loop = n_loop
    model = GPT(cfg).to(device).eval()
    model.load_state_dict(checkpoint["model"])
    return checkpoint, model


def summarize(rows):
    totals = collections.Counter()
    by_family = collections.defaultdict(collections.Counter)
    for row in rows:
        totals["cases"] += 1
        family = by_family[row["family"]]
        family["cases"] += 1
        for key in ("answer_correct", "trace_present", "visible_answer_correct"):
            totals[key] += int(row[key])
            family[key] += int(row[key])
    return {
        **dict(totals),
        "by_family": {name: dict(values) for name, values in sorted(by_family.items())},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--per-family", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260712)
    parser.add_argument("--max-new", type=int, default=128)
    parser.add_argument("--n-loop", type=int, default=0,
                        help="test-only latent-depth override (0 preserves checkpoint config)")
    args = parser.parse_args()
    if args.n_loop < 0:
        raise SystemExit("--n-loop must be zero or positive")

    all_rows = load_rows(args.data)
    selected = select_rows(all_rows, args.per_family, args.seed)
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    tokenizer = Tokenizer.from_file(args.tokenizer)
    checkpoint, model = load_model(args.ckpt, device, n_loop=args.n_loop)
    rows = []
    for index, case in enumerate(selected, 1):
        prompt = "Question: {}\nAnswer:".format(case["question"])
        response = generate(
            model, tokenizer, prompt, device, max_new=args.max_new, temp=0.0,
            skip_special_tokens=False,
        )
        row = {
            "family": case["family"],
            "mode": case.get("mode"),
            "question": case["question"],
            "answer": case["answer"],
            "prompt": prompt,
            "response": response,
        }
        row.update(score_response(case["answer"], response))
        rows.append(row)
        print(
            "[semantic-bridge] {}/{} family={} answer={} trace={} visible={}".format(
                index, len(selected), row["family"], row["answer_correct"],
                row["trace_present"], row["visible_answer_correct"],
            ),
            flush=True,
        )

    result = {
        "audit": "semantic_bridge_heldout_v1",
        "checkpoint": args.ckpt,
        "step": checkpoint.get("step"),
        "device": device,
        "n_loop": model.cfg.n_loop,
        "data": args.data,
        "data_sha256": sha256_file(args.data),
        "per_family": args.per_family,
        "seed": args.seed,
        "summary": summarize(rows),
        "rows": rows,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("[semantic-bridge] summary=" + json.dumps(result["summary"], sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
