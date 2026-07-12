#!/usr/bin/env python3
"""Generate auditable qualitative capability transcripts from a Shohin checkpoint.

The prompts are hand-authored and not part of any training or benchmark set. They
are intentionally small, diagnostic tasks spanning the model's stated reasoning
surface: arithmetic, symbolic manipulation, logic, algorithms, and code.
"""
import argparse
import json
from pathlib import Path

import torch
from tokenizers import Tokenizer

from eval_suite import generate
from model import GPT, GPTConfig


PROBES = [
    {
        "id": "arithmetic_discount",
        "question": "A jacket costs $84. It is discounted by 25%, then sales tax of 8% is added to the discounted price. What is the final price?",
    },
    {
        "id": "linear_equation",
        "question": "Solve for x: 3(x - 4) + 7 = 2x + 11.",
    },
    {
        "id": "fraction_reasoning",
        "question": "A recipe uses 3/4 cup of flour per batch. How many cups of flour are needed for 6 batches?",
    },
    {
        "id": "base_conversion",
        "question": "Convert the base-7 number 253 to base 10.",
    },
    {
        "id": "syllogism",
        "question": "All glims are plorks. No plorks are trens. Can any glim be a tren? Answer yes or no and explain briefly.",
    },
    {
        "id": "string_transform",
        "question": "Insert the letters 'xy' after the third character of the string 'planet'. What is the resulting string?",
    },
    {
        "id": "algorithm",
        "question": "Given the list [8, 3, 5, 3], what list results after sorting it in ascending order and removing duplicates?",
    },
    {
        "id": "python_function",
        "question": "Write a Python function named is_even that returns True when an integer n is even and False otherwise. Return only code.",
    },
]


def load_model(path, device):
    ckpt = torch.load(path, map_location="cpu")
    model = GPT(GPTConfig(**ckpt["cfg"])).to(device).eval()
    model.load_state_dict(ckpt["model"])
    return ckpt, model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--max-new", type=int, default=192)
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt, model = load_model(args.ckpt, device)
    tok = Tokenizer.from_file(args.tokenizer)
    transcripts = []
    for probe in PROBES:
        prompt = f"Question: {probe['question']}\nAnswer:"
        response = generate(model, tok, prompt, device, max_new=args.max_new, temp=args.temp)
        row = {
            "id": probe["id"],
            "question": probe["question"],
            "response": response,
        }
        transcripts.append(row)
        print(f"\n===== {row['id']} =====\nQ: {row['question']}\nA:{row['response']}\n", flush=True)

    result = {
        "checkpoint": args.ckpt,
        "step": ckpt.get("step"),
        "temperature": args.temp,
        "probes": transcripts,
    }
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
