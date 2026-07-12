#!/usr/bin/env python3
"""Evaluate exact-prompt primitive contracts without collapsing them to Q/A."""
import argparse
import collections
import json
import re
from pathlib import Path

import torch
from tokenizers import Tokenizer

from eval_suite import generate
from model import GPT, GPTConfig


def normalized(value):
    return re.sub(r"\s+", "", str(value).strip().lower()).rstrip(".")


def predict(response, family):
    if family in {"arithmetic", "base_conversion", "state_update", "correction"}:
        answers = re.findall(r"(?:the\s+)?(?:final\s+)?answer\s+is\s*(-?\d+)", response, flags=re.I)
        values = answers or re.findall(r"-?\d+", response)
        return values[-1] if values else ""
    if family == "sort_unique":
        values = re.findall(r"\[[^\]\n]{1,200}\]", response)
        return normalized(values[-1]) if values else ""
    if family == "string_insert":
        answers = re.findall(r"(?:the\s+)?(?:final\s+)?answer\s+is\s*['\"]?([^\n'\"]+)", response, flags=re.I)
        if answers:
            return normalized(answers[-1])
        words = re.findall(r"[A-Za-z]+", response)
        return normalized(words[-1]) if words else ""
    if family == "syllogism":
        values = re.findall(r"\b(yes|no)\b", response, flags=re.I)
        return values[-1].lower() if values else ""
    raise ValueError(f"unknown family: {family}")


def load_model(path, device):
    checkpoint = torch.load(path, map_location="cpu")
    model = GPT(GPTConfig(**checkpoint["cfg"])).to(device).eval()
    model.load_state_dict(checkpoint["model"])
    return checkpoint, model


def read_rows(path, limit, per_contract_family):
    rows = []
    selected = collections.Counter()
    with open(path, errors="replace") as source:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            if not row.get("completion_prompt") or not row.get("answer"):
                continue
            key = (row.get("contract"), row.get("family"))
            if per_contract_family and selected[key] >= per_contract_family:
                continue
            rows.append(row)
            selected[key] += 1
            if limit and len(rows) >= limit:
                break
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--n", type=int, default=0, help="0 evaluates every valid row")
    parser.add_argument("--per-contract-family", type=int, default=0,
                        help="optional balanced quota for every (contract, family) bucket")
    parser.add_argument("--max-new", type=int, default=96)
    args = parser.parse_args()

    if args.n and args.per_contract_family:
        raise SystemExit("choose either --n or --per-contract-family")
    rows = read_rows(args.data, args.n, args.per_contract_family)
    if not rows:
        raise SystemExit("no valid contract rows")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint, model = load_model(args.ckpt, device)
    tokenizer = Tokenizer.from_file(args.tokenizer)
    totals, correct = collections.Counter(), collections.Counter()
    examples = []
    for index, row in enumerate(rows):
        response = generate(model, tokenizer, row["completion_prompt"], device, max_new=args.max_new, temp=0.0)
        prediction = predict(response, row["family"])
        ok = normalized(prediction) == normalized(row["answer"])
        key = (row["contract"], row["family"])
        totals[key] += 1
        correct[key] += int(ok)
        if len(examples) < 5:
            examples.append({
                "contract": row["contract"], "family": row["family"],
                "gold": row["answer"], "prediction": prediction, "correct": ok,
            })
        if (index + 1) % 100 == 0 or index + 1 == len(rows):
            print(f"[contract-eval] {index + 1}/{len(rows)} correct={sum(correct.values())}", flush=True)
    by_contract = {}
    for contract in sorted({row["contract"] for row in rows}):
        c_total = sum(totals[(contract, family)] for family in sorted({row["family"] for row in rows}))
        c_correct = sum(correct[(contract, family)] for family in sorted({row["family"] for row in rows}))
        by_contract[contract] = {
            "correct": c_correct, "total": c_total, "accuracy": c_correct / max(c_total, 1),
            "families": {
                family: {
                    "correct": correct[(contract, family)],
                    "total": totals[(contract, family)],
                    "accuracy": correct[(contract, family)] / max(totals[(contract, family)], 1),
                }
                for family in sorted({row["family"] for row in rows})
            },
        }
    result = {
        "checkpoint": args.ckpt, "step": checkpoint.get("step"), "data": args.data,
        "n": len(rows), "accuracy": sum(correct.values()) / max(len(rows), 1),
        "by_contract": by_contract, "examples": examples,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise SystemExit(f"refusing to overwrite output: {out}")
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
