#!/usr/bin/env python3
"""Probe direct interaction, correction, and scaffold use on a Shohin checkpoint.

This is a diagnostic conversation suite, deliberately separate from public
benchmarks and from any training corpus. It records each greedy response under
three conditions: an initial answer, an explicit request to correct that answer,
and a prompt with a verified intermediate fact. The goal is to distinguish a
model that has a calculation but stops badly from one that cannot execute the
underlying rule even when the state is supplied.
"""
import argparse
import json
import re
from pathlib import Path

import torch
from tokenizers import Tokenizer

from eval_suite import generate
from model import GPT, GPTConfig


CASES = [
    {
        "id": "arithmetic_correction",
        "kind": "number",
        "question": "Compute 27 times 14, then add 9. Return only the final integer.",
        "answer": "387",
        "hint": "The product 27 times 14 is 378.",
    },
    {
        "id": "base_conversion_correction",
        "kind": "number",
        "question": "What is the base-10 value of the base-7 number 356? Return only the integer.",
        "answer": "188",
        "hint": "Use place values 7 squared, 7, and 1.",
    },
    {
        "id": "state_update_correction",
        "kind": "number",
        "question": "Start with n = 11. Add 8, multiply by 3, then subtract 7. What is n? Return only the integer.",
        "answer": "50",
        "hint": "After the addition, n is 19.",
    },
    {
        "id": "set_transform_correction",
        "kind": "list",
        "question": "Sort [8, 3, 8, 1, 6, 3] ascending and remove duplicates. Return only the list.",
        "answer": "[1,3,6,8]",
        "hint": "The unique values are 8, 3, 1, and 6.",
    },
    {
        "id": "string_transform_correction",
        "kind": "string",
        "question": "Insert 'pq' after the first 4 characters of 'lantern'. Return only the resulting string.",
        "answer": "lantpqern",
        "hint": "The first four characters are 'lant'.",
    },
    {
        "id": "logic_correction",
        "kind": "yesno",
        "question": "Every mip is a zor. No zor is a tal. Can any mip be a tal? Answer yes or no only.",
        "answer": "no",
        "hint": "Any mip must be a zor, and no zor is a tal.",
    },
]


def normalized(text):
    return re.sub(r"\s+", "", str(text).strip().lower()).rstrip(".")


def extract(kind, text):
    if kind == "number":
        values = re.findall(r"-?\d+", text)
        return values[-1] if values else ""
    if kind == "yesno":
        values = re.findall(r"\b(?:yes|no)\b", text, flags=re.I)
        return values[-1].lower() if values else ""
    if kind == "list":
        values = re.findall(r"\[[^\]\n]{1,160}\]", text)
        return normalized(values[-1]) if values else ""
    if kind == "string":
        values = re.findall(r"[A-Za-z]+", text)
        return normalized(values[-1]) if values else ""
    raise ValueError(f"unknown kind: {kind}")


def score(case, response):
    return extract(case["kind"], response) == normalized(case["answer"])


def ask(model, tokenizer, device, prompt, max_new):
    return generate(model, tokenizer, prompt, device, max_new=max_new, temp=0.0)


def load_model(path, device):
    checkpoint = torch.load(path, map_location="cpu")
    model = GPT(GPTConfig(**checkpoint["cfg"])).to(device).eval()
    model.load_state_dict(checkpoint["model"])
    return checkpoint, model


def run(path, tokenizer, device, max_new):
    checkpoint, model = load_model(path, device)
    rows = []
    for case in CASES:
        initial_prompt = f"Question: {case['question']}\nAnswer:"
        initial = ask(model, tokenizer, device, initial_prompt, max_new)
        review_prompt = (
            f"Question: {case['question']}\nAnswer: {initial}\n\n"
            "Question: Check the previous answer independently. If it is wrong, correct it. "
            "Return only the final answer.\nAnswer:"
        )
        reviewed = ask(model, tokenizer, device, review_prompt, max_new)
        scaffold_prompt = (
            f"Question: {case['question']}\n"
            f"Verified intermediate fact: {case['hint']}\n"
            "Use that fact and return only the final answer.\nAnswer:"
        )
        scaffolded = ask(model, tokenizer, device, scaffold_prompt, max_new)
        row = {
            **case,
            "initial_prompt": initial_prompt,
            "initial_response": initial,
            "initial_correct": score(case, initial),
            "review_prompt": review_prompt,
            "review_response": reviewed,
            "review_correct": score(case, reviewed),
            "scaffold_prompt": scaffold_prompt,
            "scaffold_response": scaffolded,
            "scaffold_correct": score(case, scaffolded),
        }
        rows.append(row)
        print(
            f"[adaptive] {case['id']} initial={row['initial_correct']} "
            f"review={row['review_correct']} scaffold={row['scaffold_correct']}",
            flush=True,
        )
    summary = {
        condition: sum(row[f"{condition}_correct"] for row in rows)
        for condition in ("initial", "review", "scaffold")
    }
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return {"checkpoint": path, "step": checkpoint.get("step"), "summary": summary, "rows": rows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-new", type=int, default=96)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = Tokenizer.from_file(args.tokenizer)
    result = {
        "audit": "interactive_adaptive_v1",
        "device": device,
        "cases": len(CASES),
        "model": run(args.ckpt, tokenizer, device, args.max_new),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"[adaptive] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
