#!/usr/bin/env python3
"""Diagnose answer recognition separately from free generation.

This is a small transcript-adjacent probe, not a public benchmark.  It scores
the correct completion against matched wrong completions under the same plain
question prompt.  A positive result with weak greedy generation would identify
an output-planning bottleneck; weak candidate ranking would instead show that
the answer representation is absent or unreliable.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from tokenizers import Tokenizer

from model import GPT, GPTConfig


CASES = [
    {
        "id": "arithmetic",
        "question": "Compute 37 times 14, then subtract 29. Return only the final integer.",
        "answer": "489",
        "candidates": ["489", "488", "491", "518"],
    },
    {
        "id": "base_conversion",
        "question": "What is the base-10 value of the base-7 number 356? Return only the integer.",
        "answer": "188",
        "candidates": ["188", "186", "182", "356"],
    },
    {
        "id": "state_update",
        "question": "Start with n = 11. Subtract 4, multiply the result by 6, then add 5. What is n? Return only the integer.",
        "answer": "47",
        "candidates": ["47", "35", "41", "71"],
    },
    {
        "id": "linear_equation",
        "question": "Solve 9x minus 5 equals 58. Return only the integer x.",
        "answer": "7",
        "candidates": ["7", "6", "8", "9"],
    },
    {
        "id": "sort_deduplicate",
        "question": "Sort [12, 4, 12, 7, 4, 1] ascending and remove duplicates. Return only the list.",
        "answer": "[1,4,7,12]",
        "candidates": ["[1,4,7,12]", "[1,4,12,7]", "[1,4,4,7,12]", "[12,7,4,1]"],
    },
    {
        "id": "string_insert",
        "question": "Insert 'ZX' after the first 2 characters of 'marble'. Return only the resulting string.",
        "answer": "maZXrble",
        "candidates": ["maZXrble", "marZXble", "ZXmarble", "marbleZX"],
    },
    {
        "id": "logic",
        "question": "Every plin is a nork. No nork is a ves. Can any plin be a ves? Answer yes or no only.",
        "answer": "no",
        "candidates": ["no", "yes"],
    },
]


def rank_candidates(scores):
    """Sort candidate dictionaries by mean then total log probability."""
    return sorted(
        scores,
        key=lambda row: (-row["mean_logprob"], -row["total_logprob"], row["candidate"]),
    )


def load_model(path, device):
    checkpoint = torch.load(path, map_location="cpu")
    model = GPT(GPTConfig(**checkpoint["cfg"])).to(device).eval()
    model.load_state_dict(checkpoint["model"])
    return checkpoint, model


@torch.no_grad()
def candidate_score(model, tokenizer, device, prompt, completion):
    """Return token-normalized and total log likelihood for one completion."""
    prompt_ids = tokenizer.encode(prompt).ids
    completion_ids = tokenizer.encode(" " + completion).ids
    if not completion_ids:
        raise ValueError("candidate encoded to no tokens")
    ids = prompt_ids + completion_ids
    if len(ids) > model.cfg.seq_len:
        raise ValueError("prompt plus candidate exceeds model context")
    tokens = torch.tensor([ids], device=device)
    ac = torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda")
    with ac:
        logits, _ = model(tokens)
    start = len(prompt_ids) - 1
    predicted = logits[0, start:start + len(completion_ids)].float()
    targets = tokens[0, len(prompt_ids):]
    values = torch.log_softmax(predicted, dim=-1).gather(1, targets[:, None]).squeeze(1)
    total = float(values.sum().item())
    return {"total_logprob": total, "mean_logprob": total / len(completion_ids), "tokens": len(completion_ids)}


def score_case(model, tokenizer, device, case):
    prompt = "Question: {}\nAnswer:".format(case["question"])
    scored = []
    for candidate in case["candidates"]:
        row = candidate_score(model, tokenizer, device, prompt, candidate)
        row["candidate"] = candidate
        row["correct"] = candidate == case["answer"]
        scored.append(row)
    ranked = rank_candidates(scored)
    correct_rank = next(index + 1 for index, row in enumerate(ranked) if row["correct"])
    return {
        "id": case["id"],
        "question": case["question"],
        "answer": case["answer"],
        "prompt": prompt,
        "ranked": ranked,
        "correct_rank": correct_rank,
        "correct_top1": correct_rank == 1,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = Path(args.out)
    if out.exists():
        raise SystemExit("refusing to overwrite output: {}".format(out))
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    checkpoint, model = load_model(args.ckpt, device)
    tokenizer = Tokenizer.from_file(args.tokenizer)
    rows = []
    for case in CASES:
        row = score_case(model, tokenizer, device, case)
        rows.append(row)
        print("[forced-choice] {} rank={}/{}".format(case["id"], row["correct_rank"], len(case["candidates"])), flush=True)
    result = {
        "audit": "forced_choice_probe_v1",
        "claim_boundary": (
            "This measures relative token likelihood for fixed candidates, not free generation, "
            "reasoning, or benchmark ability. It is only a diagnosis of recognition versus emission."
        ),
        "checkpoint": args.ckpt,
        "step": checkpoint.get("step"),
        "device": device,
        "cases": len(rows),
        "top1": sum(row["correct_top1"] for row in rows),
        "mean_correct_rank": sum(row["correct_rank"] for row in rows) / len(rows),
        "rows": rows,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(
        "[forced-choice] step={} top1={}/{} mean-rank={:.3f}".format(
            result["step"], result["top1"], result["cases"], result["mean_correct_rank"]
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
