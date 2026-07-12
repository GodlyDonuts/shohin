#!/usr/bin/env python3
"""Sample a student on answer-labeled problems for verifier training or evaluation.

The output contains only the question, the student's candidate completion, and a
strict answer-check label. It is deliberately separate from SFT data: callers
must pass a train-only source for verifier training and a held-out source for
evaluation.
"""
import argparse
import json
import os
from pathlib import Path

import torch
from tokenizers import Tokenizer

from eval_suite import extract_gsm8k, generate, gold_gsm8k
from model import GPT, GPTConfig


def read_rows(path, limit):
    rows = []
    with open(path, errors="replace") as src:
        for line in src:
            if not line.strip():
                continue
            row = json.loads(line)
            question = str(row.get("question") or row.get("problem") or "").strip()
            gold = gold_gsm8k(row)
            if question and gold is not None:
                rows.append((question, gold))
            if limit and len(rows) >= limit:
                break
    return rows


def load_model(path, device):
    ckpt = torch.load(path, map_location="cpu")
    model = GPT(GPTConfig(**ckpt["cfg"])).to(device).eval()
    model.load_state_dict(ckpt["model"])
    return ckpt, model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--data", required=True, help="GSM8K-style labeled JSONL")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=0, help="0 means every valid row")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--seed", type=int, default=20260712)
    args = ap.parse_args()

    out = Path(args.out)
    tmp = out.with_suffix(out.suffix + ".partial")
    if out.exists() or tmp.exists():
        raise SystemExit(f"refusing to overwrite verifier rollout artifact: {out}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    rows = read_rows(args.data, args.n)
    if not rows:
        raise SystemExit("no valid GSM8K-style rows found")
    tokenizer = Tokenizer.from_file(args.tokenizer)
    ckpt, model = load_model(args.ckpt, device)
    out.parent.mkdir(parents=True, exist_ok=True)

    total = correct = 0
    with open(tmp, "w") as dst:
        for index, (question, gold) in enumerate(rows):
            prompt = f"Question: {question}\nAnswer:"
            for sample_index in range(args.k):
                candidate = generate(
                    model, tokenizer, prompt, device,
                    max_new=args.max_new, temp=args.temp,
                )
                prediction = extract_gsm8k(candidate)
                ok = prediction == gold
                dst.write(json.dumps({
                    "question": question,
                    "gold": gold,
                    "candidate": candidate,
                    "prediction": prediction,
                    "correct": ok,
                    "sample_index": sample_index,
                    "source_checkpoint": args.ckpt,
                }, ensure_ascii=False) + "\n")
                total += 1
                correct += int(ok)
            if (index + 1) % 50 == 0 or index + 1 == len(rows):
                print(f"[verifier-rollout] {index + 1}/{len(rows)} candidates={total} correct={correct}", flush=True)
    os.replace(tmp, out)
    print(json.dumps({
        "out": str(out), "checkpoint": args.ckpt, "rows": len(rows),
        "k": args.k, "candidates": total, "correct": correct,
        "accuracy": correct / max(total, 1), "step": ckpt.get("step"),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
