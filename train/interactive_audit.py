#!/usr/bin/env python3
"""Run a small, hand-authored capability audit on one or more Shohin checkpoints.

This is deliberately separate from public benchmarks and generated procedural data.
The questions are fresh diagnostic interactions chosen to expose instruction following,
arithmetic invariants, symbolic transformations, state tracking, and minimal code
generation. It records verbatim model outputs; expected signals are context for human
review, not a benchmark score.
"""
import argparse
import json
from pathlib import Path

import torch
from tokenizers import Tokenizer

from eval_suite import generate
from model import GPT, GPTConfig


CASES = [
    {
        "id": "format_exact_word",
        "category": "instruction_following",
        "question": "Return exactly this word, with no explanation: saffron",
        "expected_signal": "saffron",
    },
    {
        "id": "arithmetic_product",
        "category": "arithmetic",
        "question": "Compute 19 multiplied by 17. Give the final integer.",
        "expected_signal": "323",
    },
    {
        "id": "equation_invariant",
        "category": "symbolic_reasoning",
        "question": "Solve for x: 4(x + 3) - 5 = 2x + 13.",
        "expected_signal": "3",
    },
    {
        "id": "base_conversion",
        "category": "symbolic_reasoning",
        "question": "What is the base-10 value of the base-6 number 254?",
        "expected_signal": "106",
    },
    {
        "id": "logic_syllogism",
        "category": "logic",
        "question": "Every fep is a tor. No tor is a lum. Can any fep be a lum? Answer yes or no.",
        "expected_signal": "no",
    },
    {
        "id": "string_insertion",
        "category": "algorithmic_transform",
        "question": "Insert the letters 'XY' after the third character of 'orchard'. Return the resulting string.",
        "expected_signal": "orcXYhard",
    },
    {
        "id": "sort_unique",
        "category": "algorithmic_transform",
        "question": "Sort [9, 2, 9, 4, 2] in ascending order and remove duplicates. Return the resulting list.",
        "expected_signal": "[2, 4, 9]",
    },
    {
        "id": "letter_count",
        "category": "algorithmic_transform",
        "question": "How many times does the letter 'a' occur in the word 'bananas'? Give the integer.",
        "expected_signal": "3",
    },
    {
        "id": "state_tracking",
        "category": "multi_step_control",
        "question": "Start with n = 7. Add 5, double the result, then subtract 4. What is n?",
        "expected_signal": "20",
    },
    {
        "id": "correct_false_claim",
        "category": "error_correction",
        "question": "A student says that 23 in base 5 equals 23 in base 10. Is the student correct? State the correct base-10 value.",
        "expected_signal": "13",
    },
    {
        "id": "minimal_python",
        "category": "code",
        "question": "Write only Python code for is_multiple_of_three(n), returning True exactly when n is divisible by 3.",
        "expected_signal": "n % 3",
    },
    {
        "id": "context_use",
        "category": "context_control",
        "question": "A previous calculation established r = 14. Using that value, what is 3r? Give the integer.",
        "expected_signal": "42",
    },
]


def load_model(path, device):
    checkpoint = torch.load(path, map_location="cpu")
    model = GPT(GPTConfig(**checkpoint["cfg"])).to(device).eval()
    model.load_state_dict(checkpoint["model"])
    return checkpoint, model


def run_checkpoint(path, tokenizer, device, max_new, temperature):
    checkpoint, model = load_model(path, device)
    transcripts = []
    for case in CASES:
        prompt = f"Question: {case['question']}\nAnswer:"
        response = generate(model, tokenizer, prompt, device, max_new=max_new, temp=temperature)
        row = dict(case)
        row["prompt"] = prompt
        row["response"] = response
        transcripts.append(row)
        print(f"\n===== {Path(path).name} :: {case['id']} =====", flush=True)
        print(f"Q: {case['question']}\nA:{response}", flush=True)
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return {
        "checkpoint": path,
        "step": checkpoint.get("step"),
        "transcripts": transcripts,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", nargs="+", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-new", type=int, default=160)
    parser.add_argument("--temp", type=float, default=0.0)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = Tokenizer.from_file(args.tokenizer)
    print(f"[interactive-audit] device={device} checkpoints={len(args.ckpt)}", flush=True)
    result = {
        "audit": "interactive_v1",
        "device": device,
        "temperature": args.temp,
        "max_new": args.max_new,
        "cases": len(CASES),
        "models": [run_checkpoint(path, tokenizer, device, args.max_new, args.temp) for path in args.ckpt],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"[interactive-audit] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
