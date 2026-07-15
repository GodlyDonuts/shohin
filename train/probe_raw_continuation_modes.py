#!/usr/bin/env python3
"""Probe whether a raw Shohin checkpoint exposes procedures under pretrain-native formats.

This is transcript-first diagnostics only. It uses fresh hand-written cases, performs
no training, and reports both strict leading-answer and permissive terminal-answer
scores so verbose continuations cannot be mistaken for direct instruction following.
"""

import argparse
import hashlib
import json
import re
from pathlib import Path

import torch
from tokenizers import Tokenizer

from eval_suite import generate
from model import GPT, GPTConfig


CASES = [
    {
        "id": "multiply_subtract",
        "answer": 372,
        "question": "Compute 23 times 17, then subtract 19.",
        "expression": "23 * 17 - 19 =",
        "demos": [
            ("Compute 14 times 12, then subtract 9.", "14 * 12 = 168; 168 - 9 = 159", 159),
            ("Compute 18 times 15, then subtract 7.", "18 * 15 = 270; 270 - 7 = 263", 263),
        ],
        "trace": "23 * 17 = 391; 391 - 19 =",
        "decomposition": "The product is 391. Subtract 19 from 391. The final integer is",
    },
    {
        "id": "base_seven",
        "answer": 184,
        "question": "Convert the base-7 numeral 352 to base 10.",
        "expression": "int('352', 7) =",
        "demos": [
            ("Convert the base-5 numeral 243 to base 10.", "2*25 + 4*5 + 3 = 73", 73),
            ("Convert the base-6 numeral 314 to base 10.", "3*36 + 1*6 + 4 = 118", 118),
        ],
        "trace": "3 * 49 + 5 * 7 + 2 = 147 + 35 + 2 =",
        "decomposition": "The place-value terms are 147, 35, and 2. Their sum is",
    },
    {
        "id": "sequential_state",
        "answer": 59,
        "question": "Start at 11, add 7, multiply by 4, then subtract 13.",
        "expression": "((11 + 7) * 4) - 13 =",
        "demos": [
            ("Start at 9, add 5, multiply by 3, then subtract 8.", "9+5=14; 14*3=42; 42-8=34", 34),
            ("Start at 6, add 8, multiply by 2, then subtract 9.", "6+8=14; 14*2=28; 28-9=19", 19),
        ],
        "trace": "state=11 -> 18 -> 72 ->",
        "decomposition": "After add: 18. After multiply: 72. After subtracting 13:",
    },
    {
        "id": "modular_update",
        "answer": 5,
        "question": "Add 37 and 28, then give the remainder after division by 12.",
        "expression": "(37 + 28) % 12 =",
        "demos": [
            ("Add 23 and 16, then give the remainder after division by 7.", "23+16=39; 39 mod 7 = 4", 4),
            ("Add 31 and 24, then give the remainder after division by 9.", "31+24=55; 55 mod 9 = 1", 1),
        ],
        "trace": "37 + 28 = 65; 65 = 5 * 12 +",
        "decomposition": "The sum is 65. The largest multiple of 12 not exceeding 65 is 60. The remainder is",
    },
]


def prompts(case):
    demos = "\n\n".join(
        f"Problem: {question}\nWork: {work}\nAnswer: {answer}"
        for question, work, answer in case["demos"]
    )
    return {
        "direct_qa": f"Question: {case['question']} Return only the final integer.\nAnswer:",
        "bare_expression": case["expression"],
        "worked_completion": f"{demos}\n\nProblem: {case['question']}\nWork:",
        "algorithm_trace": case["trace"],
        "supplied_decomposition": case["decomposition"],
    }


def integer_scores(response, answer):
    values = [int(value) for value in re.findall(r"(?<![A-Za-z0-9_])-?\d+", response)]
    return {
        "integers": values,
        "leading_correct": bool(values) and values[0] == answer,
        "terminal_correct": bool(values) and values[-1] == answer,
        "contains_answer": answer in values,
    }


def load_model(path, device):
    checkpoint = torch.load(path, map_location="cpu")
    model = GPT(GPTConfig(**checkpoint["cfg"])).to(device).eval()
    model.load_state_dict(checkpoint["model"])
    return checkpoint, model


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-new", type=int, default=96)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    tokenizer = Tokenizer.from_file(args.tokenizer)
    checkpoint, model = load_model(args.ckpt, device)
    rows = []
    for case in CASES:
        mode_rows = {}
        for mode, prompt in prompts(case).items():
            print(f"[continuation-probe] case={case['id']} mode={mode}", flush=True)
            response = generate(model, tokenizer, prompt, device, max_new=args.max_new, temp=0.0)
            mode_rows[mode] = {
                "prompt": prompt,
                "response": response,
                **integer_scores(response, case["answer"]),
            }
        rows.append({
            "id": case["id"],
            "question": case["question"],
            "answer": case["answer"],
            "modes": mode_rows,
        })

    modes = list(prompts(CASES[0]))
    summary = {
        mode: {
            metric: sum(row["modes"][mode][metric] for row in rows)
            for metric in ("leading_correct", "terminal_correct", "contains_answer")
        }
        for mode in modes
    }
    result = {
        "audit": "raw_continuation_mode_probe_v1",
        "device": device,
        "checkpoint": args.ckpt,
        "checkpoint_step": checkpoint.get("step"),
        "checkpoint_sha256": file_sha256(args.ckpt),
        "tokenizer": args.tokenizer,
        "tokenizer_sha256": file_sha256(args.tokenizer),
        "case_count": len(CASES),
        "mode_count": len(modes),
        "max_new": args.max_new,
        "summary": summary,
        "rows": rows,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
