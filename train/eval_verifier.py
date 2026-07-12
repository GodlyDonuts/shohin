#!/usr/bin/env python3
"""Measure whether a verifier improves best-of-N selection on held-out rollouts."""
import argparse
import collections
import json
import math
from pathlib import Path

import torch
from tokenizers import Tokenizer

from model import GPT, GPTConfig


def verifier_question(question, candidate):
    """Mirror the training prompt without making this script depend on PYTHONPATH."""
    return (
        "Problem:\n"
        f"{question}\n\n"
        "Candidate solution:\n"
        f"{candidate}\n\n"
        "Is the candidate solution correct? Reply only <|correct|> or <|incorrect|>."
    )


def load_model(path, device):
    ckpt = torch.load(path, map_location="cpu")
    model = GPT(GPTConfig(**ckpt["cfg"])).to(device).eval()
    model.load_state_dict(ckpt["model"])
    return ckpt, model


def forward_logits(model, token_ids):
    """Return logits from the GPT inference contract: ``(logits, auxiliary)``."""
    logits, _ = model(token_ids)
    return logits


@torch.no_grad()
def score_candidate(model, tokenizer, question, candidate, device, correct_id, incorrect_id):
    prompt = f"Question: {verifier_question(question, candidate)}\nAnswer:"
    ids = tokenizer.encode(prompt).ids[-model.cfg.seq_len:]
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
        logits = forward_logits(model, torch.tensor([ids], device=device))[0, -1].float()
    logp = torch.log_softmax(logits, dim=-1)
    return float(logp[correct_id] - logp[incorrect_id])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verifier", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--rollouts", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = Tokenizer.from_file(args.tokenizer)
    correct_id = tokenizer.token_to_id("<|correct|>")
    incorrect_id = tokenizer.token_to_id("<|incorrect|>")
    if correct_id is None or incorrect_id is None:
        raise SystemExit("verifier label tokens are absent from tokenizer")
    ckpt, model = load_model(args.verifier, device)

    grouped = collections.OrderedDict()
    for line in open(args.rollouts, errors="replace"):
        if not line.strip():
            continue
        row = json.loads(line)
        grouped.setdefault(str(row["question"]), []).append(row)

    rows, first_correct, oracle_correct, picked_correct = [], 0, 0, 0
    for question, candidates in grouped.items():
        scored = []
        for candidate in candidates:
            score = score_candidate(
                model, tokenizer, question, str(candidate["candidate"]), device,
                correct_id, incorrect_id,
            )
            scored.append({**candidate, "verifier_score": score})
        picked = max(scored, key=lambda row: row["verifier_score"])
        first_correct += int(bool(scored[0]["correct"]))
        oracle_correct += int(any(bool(row["correct"]) for row in scored))
        picked_correct += int(bool(picked["correct"]))
        rows.append({
            "question": question,
            "first_correct": bool(scored[0]["correct"]),
            "oracle_correct": any(bool(row["correct"]) for row in scored),
            "picked_correct": bool(picked["correct"]),
            "picked_score": picked["verifier_score"],
            "candidates": scored,
        })

    total = len(rows)
    result = {
        "verifier": args.verifier,
        "verifier_step": ckpt.get("step"),
        "rollouts": args.rollouts,
        "questions": total,
        "first_pass_at_1": first_correct / max(total, 1),
        "oracle_pass_at_k": oracle_correct / max(total, 1),
        "verifier_pass_at_k": picked_correct / max(total, 1),
        "rows": rows,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, sort_keys=True))


if __name__ == "__main__":
    main()
