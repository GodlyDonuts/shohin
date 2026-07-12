#!/usr/bin/env python3
"""Run a held-out, multi-turn capability interview against a checkpoint.

This is deliberately a diagnostic, not a training source or public benchmark.
It tests whether a model can solve a task, correct itself, use a verified
intermediate state, and create then reuse a compact working state.  The complete
verbatim conversation is retained so a score cannot hide a misleading response.
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
        "id": "arithmetic_invariant",
        "kind": "number",
        "question": "Compute 43 times 17, then subtract 29. Return only the final integer.",
        "answer": "702",
        "hint": "43 times 17 is 731.",
        "compact_request": "Write the product as a compact state, then subtract 29.",
    },
    {
        "id": "base8_conversion",
        "kind": "number",
        "question": "What is the base-10 value of the base-8 number 725? Return only the integer.",
        "answer": "469",
        "hint": "Use 7 times 8 squared, plus 2 times 8, plus 5.",
        "compact_request": "Write the three place-value terms as a compact state, then sum them.",
    },
    {
        "id": "state_transition",
        "kind": "number",
        "question": "Start with n = 12. Add 7, multiply by 4, then subtract 13. What is n? Return only the integer.",
        "answer": "63",
        "hint": "After the addition, n is 19.",
        "compact_request": "Write the current value after each operation as a compact state, then give the final value.",
    },
    {
        "id": "sort_deduplicate",
        "kind": "list",
        "question": "Sort [11, 4, 11, 2, 7, 4] ascending and remove duplicates. Return only the list.",
        "answer": "[2,4,7,11]",
        "hint": "The distinct values are 11, 4, 2, and 7.",
        "compact_request": "Write the distinct values as a compact state, then sort them.",
    },
    {
        "id": "string_splice",
        "kind": "string",
        "question": "Insert 'PQ' after the first 2 characters of 'mosaic'. Return only the resulting string.",
        "answer": "moPQsaic",
        "hint": "The first two characters are 'mo' and the remaining suffix is 'saic'.",
        "compact_request": "Write prefix, insert, and suffix as a compact state, then concatenate them.",
    },
    {
        "id": "logical_constraint",
        "kind": "yesno",
        "question": "Every vek is a nop. No nop is a rim. Can any vek be a rim? Answer yes or no only.",
        "answer": "no",
        "hint": "A vek must be a nop, and no nop can be a rim.",
        "compact_request": "Write the two set constraints as a compact state, then answer the question.",
    },
    {
        "id": "correction_with_counterexample",
        "kind": "number",
        "question": "A student says 18 divided by 3 plus 2 equals 3. Correct the student. Return only the correct integer.",
        "answer": "8",
        "hint": "Division is done before addition: 18 divided by 3 is 6.",
        "compact_request": "Write the precedence result as a compact state, then add 2.",
    },
    {
        "id": "minimal_python_contract",
        "kind": "code",
        "question": "Write only Python code for is_even(n), returning True exactly for even integers.",
        "answer": ["def is_even", "% 2", "== 0"],
        "hint": "The required predicate is n modulo 2 equals 0.",
        "compact_request": "State the predicate compactly, then write the Python function only.",
    },
]


def normalized(value):
    return re.sub(r"\s+", "", str(value).strip().lower()).rstrip(".")


def score(case, response):
    if case["kind"] == "number":
        values = re.findall(r"-?\d+", response)
        return bool(values) and values[-1] == case["answer"]
    if case["kind"] == "yesno":
        values = re.findall(r"\b(?:yes|no)\b", response, flags=re.I)
        return bool(values) and values[-1].lower() == case["answer"]
    if case["kind"] == "list":
        values = re.findall(r"\[[^\]\n]{1,160}\]", response)
        return bool(values) and normalized(values[-1]) == normalized(case["answer"])
    if case["kind"] == "string":
        values = re.findall(r"[A-Za-z]+", response)
        return bool(values) and normalized(values[-1]) == normalized(case["answer"])
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
                if node.name != "is_even" or len(node.args.args) != 1:
                    continue
                source = re.sub(r"\s+", "", ast.unparse(node).lower())
                if "n%2==0" in source and any(isinstance(item, ast.Return) for item in ast.walk(node)):
                    return True
        return False
    raise ValueError(f"unknown kind: {case['kind']}")


def ask(model, tokenizer, device, prompt, max_new):
    return generate(model, tokenizer, prompt, device, max_new=max_new, temp=0.0)


def load_model(path, device):
    checkpoint = torch.load(path, map_location="cpu")
    model = GPT(GPTConfig(**checkpoint["cfg"])).to(device).eval()
    model.load_state_dict(checkpoint["model"])
    return checkpoint, model


def interview(path, tokenizer, device, max_new):
    checkpoint, model = load_model(path, device)
    rows = []
    for case in CASES:
        initial_prompt = f"Question: {case['question']}\nAnswer:"
        initial = ask(model, tokenizer, device, initial_prompt, max_new)
        review_prompt = (
            f"Question: {case['question']}\nPrevious answer: {initial}\n\n"
            "Independently check the previous answer. If it is wrong, correct it. "
            "Return only the final answer.\nAnswer:"
        )
        reviewed = ask(model, tokenizer, device, review_prompt, max_new)
        scaffold_prompt = (
            f"Question: {case['question']}\nVerified intermediate fact: {case['hint']}\n"
            "Use that fact. Return only the final answer.\nAnswer:"
        )
        scaffolded = ask(model, tokenizer, device, scaffold_prompt, max_new)
        compact_prompt = (
            f"Question: {case['question']}\n{case['compact_request']} "
            "First write one short state line beginning with 'state=', then give the final answer.\nAnswer:"
        )
        compacted = ask(model, tokenizer, device, compact_prompt, max_new)
        reuse_prompt = (
            f"Question: {case['question']}\nThe previous compact state was:\n{compacted}\n\n"
            "Use that state to solve the original question. Return only the final answer.\nAnswer:"
        )
        reused = ask(model, tokenizer, device, reuse_prompt, max_new)
        row = {
            **case,
            "initial": {"prompt": initial_prompt, "response": initial, "correct": score(case, initial)},
            "review": {"prompt": review_prompt, "response": reviewed, "correct": score(case, reviewed)},
            "scaffold": {"prompt": scaffold_prompt, "response": scaffolded, "correct": score(case, scaffolded)},
            "compact": {"prompt": compact_prompt, "response": compacted},
            "compact_reuse": {"prompt": reuse_prompt, "response": reused, "correct": score(case, reused)},
        }
        rows.append(row)
        print(
            f"[deep-interaction] {case['id']} initial={row['initial']['correct']} "
            f"review={row['review']['correct']} scaffold={row['scaffold']['correct']} "
            f"reuse={row['compact_reuse']['correct']}",
            flush=True,
        )
    summary = {
        condition: sum(row[condition]["correct"] for row in rows)
        for condition in ("initial", "review", "scaffold", "compact_reuse")
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
    result = {
        "audit": "deep_interaction_v1",
        "device": device,
        "cases": len(CASES),
        "model": interview(args.ckpt, Tokenizer.from_file(args.tokenizer), device, args.max_new),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"[deep-interaction] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
