#!/usr/bin/env python3
"""Evaluate exact typed-state generation and state reuse on held-out tasks."""
import argparse
import collections
import json
import re
from pathlib import Path

import torch
from tokenizers import Tokenizer

from eval_contract_primitives import normalized, predict
from eval_suite import generate
from model import GPT, GPTConfig


def state_from_response(response):
    match = re.search(r"^\s*(state=[^\n]+)", response, flags=re.I | re.M)
    return normalized(match.group(1)) if match else ""


def load_model(path, device):
    checkpoint = torch.load(path, map_location="cpu")
    model = GPT(GPTConfig(**checkpoint["cfg"])).to(device).eval()
    model.load_state_dict(checkpoint["model"])
    return checkpoint, model


def read_rows(path, per_contract_family):
    rows, selected = [], collections.Counter()
    with open(path, errors="replace") as source:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            required = ("completion_prompt", "answer", "expected_state", "contract", "family")
            if any(not row.get(field) for field in required):
                continue
            key = (row["contract"], row["family"])
            if selected[key] >= per_contract_family:
                continue
            rows.append(row)
            selected[key] += 1
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--per-contract-family", type=int, default=20)
    parser.add_argument("--max-new", type=int, default=96)
    args = parser.parse_args()
    if args.per_contract_family <= 0:
        raise SystemExit("per-contract-family must be positive")
    rows = read_rows(args.data, args.per_contract_family)
    if not rows:
        raise SystemExit("no valid state-protocol rows")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint, model = load_model(args.ckpt, device)
    tokenizer = Tokenizer.from_file(args.tokenizer)
    totals, answer_correct = collections.Counter(), collections.Counter()
    state_correct, state_total = collections.Counter(), collections.Counter()
    examples = []
    for index, row in enumerate(rows):
        response = generate(model, tokenizer, row["completion_prompt"], device, max_new=args.max_new, temp=0.0)
        answer_ok = normalized(predict(response, row["family"])) == normalized(row["answer"])
        expected_state = normalized(row["expected_state"])
        actual_state = state_from_response(response)
        state_required = row["contract"] in {"write", "repair"}
        state_ok = actual_state == expected_state if state_required else None
        key = (row["contract"], row["family"])
        totals[key] += 1
        answer_correct[key] += int(answer_ok)
        if state_required:
            state_total[key] += 1
            state_correct[key] += int(state_ok)
        if len(examples) < 8:
            examples.append({
                "contract": row["contract"], "family": row["family"],
                "answer_ok": answer_ok, "state_ok": state_ok,
                "expected_state": row["expected_state"], "actual_state": actual_state,
                "response": response,
            })
        if (index + 1) % 100 == 0 or index + 1 == len(rows):
            print(f"[state-protocol] {index + 1}/{len(rows)} answer={sum(answer_correct.values())} state={sum(state_correct.values())}/{sum(state_total.values())}", flush=True)
    families = sorted({row["family"] for row in rows})
    contracts = sorted({row["contract"] for row in rows})
    by_contract = {}
    for contract in contracts:
        total = sum(totals[(contract, family)] for family in families)
        answers = sum(answer_correct[(contract, family)] for family in families)
        states = sum(state_correct[(contract, family)] for family in families)
        state_rows = sum(state_total[(contract, family)] for family in families)
        by_contract[contract] = {
            "answer_correct": answers, "state_correct": states, "state_total": state_rows, "total": total,
            "answer_accuracy": answers / max(total, 1),
            "state_accuracy": states / state_rows if state_rows else None,
            "families": {
                family: {
                    "answer_correct": answer_correct[(contract, family)],
                    "state_correct": state_correct[(contract, family)],
                    "state_total": state_total[(contract, family)], "total": totals[(contract, family)],
                }
                for family in families
            },
        }
    total = len(rows)
    result = {
        "checkpoint": args.ckpt, "step": checkpoint.get("step"), "data": args.data,
        "n": total,
        "answer_accuracy": sum(answer_correct.values()) / total,
        "state_accuracy": sum(state_correct.values()) / max(sum(state_total.values()), 1),
        "by_contract": by_contract, "examples": examples,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise SystemExit(f"refusing to overwrite output: {out}")
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "examples"}, sort_keys=True))


if __name__ == "__main__":
    main()
