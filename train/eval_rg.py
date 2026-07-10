#!/usr/bin/env python3
"""Evaluate Shohin on held-out Reasoning-Gym questions.

The input is generated with families and seeds disjoint from the procedural SFT
corpus. This is a development gate for data and post-training experiments, not
a replacement for the public GSM8K/MATH/code board.
"""
import argparse
import collections
import json
import random
import re
import sys
from pathlib import Path

import torch
from tokenizers import Tokenizer

# Keep this runnable both as ``python train/eval_rg.py`` and from train/.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_suite import generate
from model import GPT, GPTConfig


def normalized(value):
    value = str(value or "").strip().lower()
    value = value.replace("$", "").replace(",", "")
    value = re.sub(r"\s+", " ", value)
    return value.rstrip(". ")


def extract(text):
    boxed = re.findall(r"\\boxed\{([^{}]+)\}", text)
    if boxed:
        return normalized(boxed[-1])
    answer = re.findall(r"(?:the )?answer is\s*([^\n<]+)", text, flags=re.I)
    if answer:
        return normalized(answer[-1])
    return normalized(text.split("\n", 1)[0])


def balanced_sample(rows, n, seed):
    """Round-robin families so generator-file ordering cannot skew a small eval."""
    groups = collections.defaultdict(list)
    for row in rows:
        groups[str(row.get("family") or "unknown")].append(row)
    rng = random.Random(seed)
    for group in groups.values():
        rng.shuffle(group)
    selected, index = [], 0
    families = sorted(groups)
    while len(selected) < n:
        emitted = False
        for family in families:
            group = groups[family]
            if index < len(group) and len(selected) < n:
                selected.append(group[index])
                emitted = True
        if not emitted:
            break
        index += 1
    return selected


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--max-new", type=int, default=128)
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.ckpt, map_location="cpu")
    model = GPT(GPTConfig(**ckpt["cfg"])).to(device).eval()
    model.load_state_dict(ckpt["model"])
    tok = Tokenizer.from_file(args.tokenizer)
    all_rows = [json.loads(line) for line in open(args.data) if line.strip()]
    rows = balanced_sample(all_rows, args.n, args.seed)

    by_family = collections.Counter()
    correct_by_family = collections.Counter()
    examples = []
    correct = 0
    for i, row in enumerate(rows):
        question = str(row.get("question") or "")
        family = str(row.get("family") or "unknown")
        gold = normalized(row.get("answer"))
        response = generate(model, tok, f"Question: {question}\nAnswer:", device, max_new=args.max_new)
        pred = extract(response)
        ok = bool(gold and pred == gold)
        correct += ok
        by_family[family] += 1
        correct_by_family[family] += ok
        if i < 5:
            examples.append({"family": family, "gold": gold, "pred": pred, "ok": ok})

    families = {
        family: {"correct": correct_by_family[family], "total": total,
                 "accuracy": correct_by_family[family] / total}
        for family, total in sorted(by_family.items())
    }
    result = {
        "checkpoint": args.ckpt,
        "step": ckpt.get("step"),
        "data": args.data,
        "n": len(rows),
        "sample_seed": args.seed,
        "correct": correct,
        "accuracy": correct / max(len(rows), 1),
        "families": families,
        "examples": examples,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(result, f, indent=2, sort_keys=True)
            f.write("\n")


if __name__ == "__main__":
    main()
