#!/usr/bin/env python3
"""Measure whether Shohin failures are prompt-contract or computation failures.

This is intentionally a small deterministic diagnostic, not a training set or a
public benchmark. It evaluates fresh generated tasks under several prompt
contracts and preserves transcripts for manual inspection.
"""
import argparse
import collections
import json
import random
import re
from pathlib import Path

import torch
from tokenizers import Tokenizer

from eval_suite import generate
from model import GPT, GPTConfig


def build_cases(n_per_family, seed):
    rng = random.Random(seed)
    cases = []
    for i in range(n_per_family):
        a, b, c = rng.randint(12, 39), rng.randint(7, 23), rng.randint(1, 19)
        cases.append({
            "id": f"arithmetic_{i:02d}", "family": "arithmetic", "kind": "number",
            "question": f"Compute {a} times {b}, then add {c}. Give only the final integer.",
            "answer": str(a * b + c),
            "example_question": "Compute 12 times 7, then add 5. Give only the final integer.",
            "example_answer": "89",
        })

        base = rng.choice([5, 6, 7, 8])
        digits = [rng.randrange(base) for _ in range(3)]
        digits[0] = max(1, digits[0])
        numeral = "".join(str(x) for x in digits)
        value = sum(digit * base ** power for power, digit in enumerate(reversed(digits)))
        cases.append({
            "id": f"base_conversion_{i:02d}", "family": "base_conversion", "kind": "number",
            "question": f"What is the base-10 value of the base-{base} number {numeral}? Give only the integer.",
            "answer": str(value),
            "example_question": "What is the base-10 value of the base-5 number 23? Give only the integer.",
            "example_answer": "13",
        })

        start, add, mult, sub = (rng.randint(3, 20), rng.randint(2, 12),
                                  rng.randint(2, 5), rng.randint(1, 15))
        value = (start + add) * mult - sub
        cases.append({
            "id": f"state_update_{i:02d}", "family": "state_update", "kind": "number",
            "question": (f"Start with n = {start}. Add {add}, multiply the result by {mult}, "
                         f"then subtract {sub}. What is n? Give only the integer."),
            "answer": str(value),
            "example_question": "Start with n = 4. Add 3, multiply the result by 2, then subtract 1. What is n? Give only the integer.",
            "example_answer": "13",
        })

        values = [rng.randint(0, 14) for _ in range(6)]
        answer = str(sorted(set(values))).replace(" ", "")
        cases.append({
            "id": f"sort_unique_{i:02d}", "family": "sort_unique", "kind": "list",
            "question": f"Sort {values} in ascending order and remove duplicates. Return only the resulting list.",
            "answer": answer,
            "example_question": "Sort [4, 1, 4, 2] in ascending order and remove duplicates. Return only the resulting list.",
            "example_answer": "[1,2,4]",
        })

        word = rng.choice(["harvest", "lantern", "mosaic", "thunder", "crayon", "violet"])
        insert = rng.choice(["XY", "pq", "ZZ", "mn"])
        position = rng.randint(2, len(word) - 2)
        answer = word[:position] + insert + word[position:]
        cases.append({
            "id": f"string_insert_{i:02d}", "family": "string_insert", "kind": "string",
            "question": (f"Insert the letters '{insert}' after the first {position} characters of '{word}'. "
                         "Return only the resulting string."),
            "answer": answer,
            "example_question": "Insert the letters 'XY' after the first 3 characters of 'planet'. Return only the resulting string.",
            "example_answer": "plaXYnet",
        })

        subject = f"dax{i}"
        middle = f"vop{i}"
        target = f"lum{i}"
        if i % 2:
            question = (f"Every {subject} is a {middle}. Every {middle} is a {target}. "
                        f"Can any {subject} be a {target}? Answer yes or no only.")
            answer = "yes"
            example_question = "Every glorp is a mib. Every mib is a zan. Can any glorp be a zan? Answer yes or no only."
            example_answer = "yes"
        else:
            question = (f"Every {subject} is a {middle}. No {middle} is a {target}. "
                        f"Can any {subject} be a {target}? Answer yes or no only.")
            answer = "no"
            example_question = "Every glorp is a mib. No mib is a zan. Can any glorp be a zan? Answer yes or no only."
            example_answer = "no"
        cases.append({
            "id": f"syllogism_{i:02d}", "family": "syllogism", "kind": "yesno",
            "question": question, "answer": answer,
            "example_question": example_question, "example_answer": example_answer,
        })
    return cases


def build_prompt(case, mode):
    question = case["question"]
    if mode == "qa":
        return f"Question: {question}\nAnswer:"
    if mode == "direct":
        return f"Solve this task and return only its final answer.\nTask: {question}\nFinal answer:"
    if mode == "cot":
        return f"Question: {question}\nAnswer: Work step by step. End with 'The answer is <final answer>.'."
    if mode == "one_shot":
        return (f"Question: {case['example_question']}\nAnswer: {case['example_answer']}\n\n"
                f"Question: {question}\nAnswer:")
    raise ValueError(f"unknown mode {mode}")


def normalize(value):
    return re.sub(r"\s+", "", str(value).strip().lower()).rstrip(".")


def score_response(case, response):
    expected = normalize(case["answer"])
    kind = case["kind"]
    if kind == "number":
        final = re.findall(r"(?:the\s+)?(?:final\s+)?answer\s+is\s*(-?\d+)", response, flags=re.I)
        candidates = final or re.findall(r"-?\d+", response)
        prediction = candidates[-1] if candidates else ""
    elif kind == "yesno":
        candidates = re.findall(r"\b(yes|no)\b", response, flags=re.I)
        prediction = candidates[0].lower() if candidates else ""
    elif kind == "list":
        candidates = re.findall(r"\[[^\]\n]{1,120}\]", response)
        prediction = normalize(candidates[-1]) if candidates else ""
    elif kind == "string":
        answer_lines = re.findall(r"(?:the\s+)?(?:final\s+)?answer\s+is\s*['\"]?([^\n'\"]+)", response, flags=re.I)
        if answer_lines:
            prediction = normalize(answer_lines[-1])
        else:
            tokens = re.findall(r"[A-Za-z]+", response)
            prediction = normalize(tokens[-1]) if tokens else ""
    else:
        raise ValueError(f"unknown kind {kind}")
    return prediction == expected, prediction


def load_model(path, device, n_loop=0):
    checkpoint = torch.load(path, map_location="cpu")
    cfg = GPTConfig(**checkpoint["cfg"])
    if n_loop:
        cfg.n_loop = n_loop
    model = GPT(cfg).to(device).eval()
    model.load_state_dict(checkpoint["model"])
    return checkpoint, model


def run_checkpoint(path, tokenizer, device, cases, modes, max_new, n_loop=0):
    checkpoint, model = load_model(path, device, n_loop=n_loop)
    used_n_loop = model.cfg.n_loop
    rows = []
    totals = collections.Counter()
    correct = collections.Counter()
    for mode in modes:
        for case in cases:
            prompt = build_prompt(case, mode)
            response = generate(model, tokenizer, prompt, device, max_new=max_new, temp=0.0)
            ok, prediction = score_response(case, response)
            key = (mode, case["family"])
            totals[key] += 1
            correct[key] += int(ok)
            rows.append({
                "id": case["id"], "family": case["family"], "kind": case["kind"],
                "mode": mode, "question": case["question"], "expected": case["answer"],
                "prediction": prediction, "correct": ok, "prompt": prompt, "response": response,
            })
    summary = {}
    for mode in modes:
        mode_total = sum(totals[(mode, family)] for family in sorted({c["family"] for c in cases}))
        mode_correct = sum(correct[(mode, family)] for family in sorted({c["family"] for c in cases}))
        summary[mode] = {
            "correct": mode_correct, "total": mode_total,
            "accuracy": mode_correct / max(1, mode_total),
            "families": {
                family: {
                    "correct": correct[(mode, family)], "total": totals[(mode, family)],
                    "accuracy": correct[(mode, family)] / max(1, totals[(mode, family)]),
                }
                for family in sorted({c["family"] for c in cases})
            },
        }
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return {
        "checkpoint": path,
        "step": checkpoint.get("step"),
        "n_loop": used_n_loop,
        "summary": summary,
        "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", nargs="+", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--n-per-family", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260712)
    parser.add_argument("--max-new", type=int, default=64)
    parser.add_argument("--modes", nargs="+", default=["qa", "direct", "cot", "one_shot"])
    parser.add_argument("--n-loop", type=int, default=0,
                        help="test-only latent-depth override (0 preserves checkpoint config)")
    args = parser.parse_args()
    if args.n_loop < 0:
        raise SystemExit("--n-loop must be zero or positive")

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    cases = build_cases(args.n_per_family, args.seed)
    tokenizer = Tokenizer.from_file(args.tokenizer)
    print(f"[capability-matrix] device={device} cases={len(cases)} modes={args.modes}", flush=True)
    result = {
        "audit": "capability_matrix_v1", "device": device, "seed": args.seed,
        "n_per_family": args.n_per_family, "modes": args.modes,
        "models": [
            run_checkpoint(path, tokenizer, device, cases, args.modes, args.max_new, args.n_loop)
            for path in args.ckpt
        ],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    for model in result["models"]:
        print(f"[capability-matrix] checkpoint={model['checkpoint']} step={model['step']}", flush=True)
        for mode, score in model["summary"].items():
            print(f"  {mode}: {score['correct']}/{score['total']} = {score['accuracy']:.1%}", flush=True)
    print(f"[capability-matrix] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
