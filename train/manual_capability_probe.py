#!/usr/bin/env python3
"""Record a small transcript-first capability probe for two Shohin checkpoints.

This is a diagnostic only.  It makes no training-data writes and intentionally
uses fresh hand-authored cases so an apparent score cannot hide a bad response
or an inability to use review and compact-state turns.
"""
import argparse
import ast
import json
import re
from pathlib import Path

import torch
from tokenizers import Tokenizer

from eval_suite import generate
from model import GPT, GPTConfig


CASES = [
    {
        "id": "arithmetic",
        "kind": "number",
        "question": "Compute 29 times 16, then subtract 37. Return only the final integer.",
        "answer": "427",
        "fact": "29 times 16 is 464.",
        "state": "Write the product as a one-line state, then subtract 37.",
    },
    {
        "id": "base_conversion",
        "kind": "number",
        "question": "What is the base-10 value of the base-6 number 425? Return only the integer.",
        "answer": "161",
        "fact": "Use 4 times 6 squared, plus 2 times 6, plus 5.",
        "state": "Write the three place-value terms as one compact state, then sum them.",
    },
    {
        "id": "state_transition",
        "kind": "number",
        "question": "Start with n = 14. Add 9, multiply the result by 3, then subtract 20. What is n? Return only the integer.",
        "answer": "49",
        "fact": "After the addition, n is 23.",
        "state": "Write n after each operation as a compact state, then give the final value.",
    },
    {
        "id": "sort_deduplicate",
        "kind": "list",
        "question": "Sort [13, 3, 13, 8, 1, 8] ascending and remove duplicates. Return only the list.",
        "answer": "[1,3,8,13]",
        "fact": "The distinct values are 13, 3, 8, and 1.",
        "state": "Write the distinct values as a compact state, then sort them.",
    },
    {
        "id": "string_insert",
        "kind": "string",
        "question": "Insert 'PQ' after the first 3 characters of 'lantern'. Return only the resulting string.",
        "answer": "lanPQtern",
        "fact": "The prefix is 'lan' and the remaining suffix is 'tern'.",
        "state": "Write prefix, insert, and suffix as a compact state, then concatenate them.",
    },
    {
        "id": "logic",
        "kind": "yesno",
        "question": "Every zol is a mar. No mar is a tiv. Can any zol be a tiv? Answer yes or no only.",
        "answer": "no",
        "fact": "A zol must be a mar, while no mar can be a tiv.",
        "state": "Write the two set constraints as a compact state, then answer the question.",
    },
    {
        "id": "python_predicate",
        "kind": "code",
        "question": "Write only Python code for is_multiple_of_three(n), returning True exactly when n is divisible by 3.",
        "answer": "is_multiple_of_three",
        "fact": "The required predicate is n modulo 3 equals 0.",
        "state": "State the predicate compactly, then write only the Python function.",
    },
]


PAIR_RESPONSE_MARKERS = {
    "problem_a": re.compile(r"\bproblem\s+a\s*:", re.IGNORECASE),
    "problem_b": re.compile(r"\bproblem\s+b\s*:", re.IGNORECASE),
    "answers_are_a": re.compile(r"\bthe\s+answers\s+are\s+a\s*=", re.IGNORECASE),
}


def normalized(value):
    return re.sub(r"\s+", "", str(value).strip().lower()).rstrip(".")


def response_mode(response):
    """Record the paired-answer grammar that should never leak into direct QA."""
    markers = [name for name, pattern in PAIR_RESPONSE_MARKERS.items() if pattern.search(response)]
    return {"paired_answer_mode": bool(markers), "paired_answer_markers": markers}


def score(case, response):
    if case["kind"] == "number":
        value = re.match(r"\s*(-?\d+)\b", response)
        return bool(value) and value.group(1) == case["answer"]
    if case["kind"] == "yesno":
        value = re.match(r"\s*(?:answer\s*:\s*)?(yes|no)\b", response, flags=re.I)
        return bool(value) and value.group(1).lower() == case["answer"]
    if case["kind"] == "list":
        value = re.match(r"\s*(\[[^\]\n]{1,160}\])", response)
        return bool(value) and normalized(value.group(1)) == normalized(case["answer"])
    if case["kind"] == "string":
        value = re.match(r"\s*['\"]?([A-Za-z]+)['\"]?", response)
        return bool(value) and normalized(value.group(1)) == normalized(case["answer"])
    if case["kind"] == "code":
        candidates = [response]
        candidates.extend(re.findall(r"```(?:[A-Za-z0-9_+-]+)?\n(.*?)```", response, flags=re.S))
        for candidate in candidates:
            try:
                tree = ast.parse(candidate)
            except SyntaxError:
                continue
            for node in tree.body:
                if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                if node.name != case["answer"] or len(node.args.args) != 1:
                    continue
                source = re.sub(r"\s+", "", ast.unparse(node).lower())
                if "n%3==0" in source and any(isinstance(item, ast.Return) for item in ast.walk(node)):
                    return True
        return False
    raise ValueError(f"unknown kind: {case['kind']}")


def load_model(path, device):
    checkpoint = torch.load(path, map_location="cpu")
    model = GPT(GPTConfig(**checkpoint["cfg"])).to(device).eval()
    model.load_state_dict(checkpoint["model"])
    return checkpoint, model


def ask(model, tokenizer, device, prompt, max_new):
    return generate(model, tokenizer, prompt, device, max_new=max_new, temp=0.0)


def probe(path, tokenizer, device, max_new):
    print(f"[manual-probe] loading checkpoint={path}", flush=True)
    checkpoint, model = load_model(path, device)
    print(
        f"[manual-probe] loaded checkpoint={path} step={checkpoint.get('step')} "
        f"cases={len(CASES)}",
        flush=True,
    )
    rows = []
    for case in CASES:
        initial_prompt = f"Question: {case['question']}\nAnswer:"
        print(f"[manual-probe] {case['id']} phase=initial", flush=True)
        initial = ask(model, tokenizer, device, initial_prompt, max_new)
        review_prompt = (
            f"Question: {case['question']}\nPrevious answer: {initial}\n\n"
            "Check the previous answer independently. If it is wrong, correct it. "
            "Return only the final answer.\nAnswer:"
        )
        print(f"[manual-probe] {case['id']} phase=review", flush=True)
        review = ask(model, tokenizer, device, review_prompt, max_new)
        fact_prompt = (
            f"Question: {case['question']}\nVerified intermediate fact: {case['fact']}\n"
            "Use that fact. Return only the final answer.\nAnswer:"
        )
        print(f"[manual-probe] {case['id']} phase=verified_fact", flush=True)
        fact = ask(model, tokenizer, device, fact_prompt, max_new)
        state_prompt = (
            f"Question: {case['question']}\n{case['state']} "
            "First write exactly one short line beginning with 'state=', then give the final answer.\nAnswer:"
        )
        print(f"[manual-probe] {case['id']} phase=compact_state", flush=True)
        state = ask(model, tokenizer, device, state_prompt, max_new)
        reuse_prompt = (
            f"Question: {case['question']}\nThe previous compact state was:\n{state}\n\n"
            "Use that state to solve the original question. Return only the final answer.\nAnswer:"
        )
        print(f"[manual-probe] {case['id']} phase=state_reuse", flush=True)
        reuse = ask(model, tokenizer, device, reuse_prompt, max_new)
        row = {
            "id": case["id"],
            "kind": case["kind"],
            "question": case["question"],
            "answer": case["answer"],
            "initial": {
                "prompt": initial_prompt, "response": initial, "correct": score(case, initial),
                "response_mode": response_mode(initial),
            },
            "review": {
                "prompt": review_prompt, "response": review, "correct": score(case, review),
                "response_mode": response_mode(review),
            },
            "verified_fact": {
                "prompt": fact_prompt, "response": fact, "correct": score(case, fact),
                "response_mode": response_mode(fact),
            },
            "compact_state": {
                "prompt": state_prompt, "response": state, "response_mode": response_mode(state),
            },
            "state_reuse": {
                "prompt": reuse_prompt, "response": reuse, "correct": score(case, reuse),
                "response_mode": response_mode(reuse),
            },
        }
        rows.append(row)
        print(
            f"[manual-probe] {case['id']} initial={row['initial']['correct']} "
            f"review={row['review']['correct']} fact={row['verified_fact']['correct']} "
            f"reuse={row['state_reuse']['correct']}",
            flush=True,
        )
    summary = {
        condition: sum(row[condition]["correct"] for row in rows)
        for condition in ("initial", "review", "verified_fact", "state_reuse")
    }
    summary["paired_answer_mode"] = {
        condition: sum(row[condition]["response_mode"]["paired_answer_mode"] for row in rows)
        for condition in ("initial", "review", "verified_fact", "compact_state", "state_reuse")
    }
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return {"checkpoint": path, "step": checkpoint.get("step"), "summary": summary, "rows": rows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", nargs="+", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-new", type=int, default=128)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    tokenizer = Tokenizer.from_file(args.tokenizer)
    result = {
        "audit": "manual_capability_probe_v1",
        "device": device,
        "cases": len(CASES),
        "models": [probe(path, tokenizer, device, args.max_new) for path in args.ckpt],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    for model in result["models"]:
        summary = model["summary"]
        print(
            f"[manual-probe] step={model['step']} initial={summary['initial']}/{len(CASES)} "
            f"review={summary['review']}/{len(CASES)} fact={summary['verified_fact']}/{len(CASES)} "
            f"reuse={summary['state_reuse']}/{len(CASES)}",
            flush=True,
        )


if __name__ == "__main__":
    main()
