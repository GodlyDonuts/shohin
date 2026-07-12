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

from eval_rg import extract as extract_rg, normalized as normalized_rg
from eval_suite import extract_gsm8k, generate_batch, gold_gsm8k
from model import GPT, GPTConfig


def read_rows(path, limit, answer_mode, skip=0):
    rows = []
    skipped = 0
    with open(path, errors="replace") as src:
        for line in src:
            if not line.strip():
                continue
            row = json.loads(line)
            question = str(row.get("question") or row.get("problem") or "").strip()
            if answer_mode == "gsm8k":
                gold = gold_gsm8k(row)
            else:
                gold = normalized_rg(row.get("answer"))
            if question and gold is not None:
                if skipped < skip:
                    skipped += 1
                    continue
                rows.append({"question": question, "gold": gold, "family": row.get("family")})
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
    ap.add_argument("--skip", type=int, default=0,
                    help="skip this many valid source rows before taking --n")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--seed", type=int, default=20260712)
    ap.add_argument("--answer-mode", choices=("gsm8k", "rg"), default="gsm8k",
                    help="strict answer extractor for the labeled rollout source")
    args = ap.parse_args()

    out = Path(args.out)
    tmp = out.with_suffix(out.suffix + ".partial")
    if out.exists() or tmp.exists():
        raise SystemExit(f"refusing to overwrite verifier rollout artifact: {out}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    if args.skip < 0:
        raise SystemExit("skip must be non-negative")
    rows = read_rows(args.data, args.n, args.answer_mode, args.skip)
    if not rows:
        raise SystemExit("no valid GSM8K-style rows found")
    tokenizer = Tokenizer.from_file(args.tokenizer)
    ckpt, model = load_model(args.ckpt, device)
    out.parent.mkdir(parents=True, exist_ok=True)

    total = correct = 0
    with open(tmp, "w") as dst:
        for index, row in enumerate(rows):
            question, gold = row["question"], row["gold"]
            prompt = f"Question: {question}\nAnswer:"
            candidates = generate_batch(
                model, tokenizer, prompt, device, n=args.k,
                max_new=args.max_new, temp=args.temp,
            )
            for sample_index, candidate in enumerate(candidates):
                prediction = extract_gsm8k(candidate) if args.answer_mode == "gsm8k" else extract_rg(candidate)
                ok = prediction == gold if args.answer_mode == "gsm8k" else normalized_rg(prediction) == gold
                dst.write(json.dumps({
                    "question": question,
                    "gold": gold,
                    "candidate": candidate,
                    "prediction": prediction,
                    "correct": ok,
                    "sample_index": sample_index,
                    "family": row.get("family"),
                    "answer_mode": args.answer_mode,
                    "source_checkpoint": args.ckpt,
                }, ensure_ascii=False) + "\n")
                total += 1
                correct += int(ok)
            if (index + 1) % 50 == 0 or index + 1 == len(rows):
                print(f"[verifier-rollout] {index + 1}/{len(rows)} candidates={total} correct={correct}", flush=True)
    os.replace(tmp, out)
    print(json.dumps({
        "out": str(out), "checkpoint": args.ckpt, "rows": len(rows), "skip": args.skip,
        "k": args.k, "candidates": total, "correct": correct,
        "accuracy": correct / max(total, 1), "step": ckpt.get("step"),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
