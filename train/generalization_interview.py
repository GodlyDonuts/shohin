#!/usr/bin/env python3
"""Run a transcript-first, composition-focused capability interview.

This evaluator is deliberately separate from training data and public boards.
Each case uses an unseen surface form and tests whether a checkpoint can execute
the task, self-correct, use a supplied true intermediate, and emit a reusable
state.  A correct reuse answer counts as compact-state evidence only when the
model first emitted the requested `state=` line.
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
        "id": "word_product_rejects",
        "kind": "number",
        "question": (
            "A workshop packs 17 boxes with 23 gears in each box. Eight gears are rejected. "
            "How many gears remain? Return only the integer."
        ),
        "answer": "383",
        "fact": "17 times 23 is 391.",
        "state": "Record the product and rejected count as one compact state before answering.",
    },
    {
        "id": "base_seven_place_value",
        "kind": "number",
        "question": "Convert the base-7 numeral 356 to base 10. Return only the integer.",
        "answer": "188",
        "fact": "Use 3 times 7 squared, plus 5 times 7, plus 6.",
        "state": "Record the three place-value terms as one compact state before answering.",
    },
    {
        "id": "sequential_inventory",
        "kind": "number",
        "question": (
            "A counter starts at 6. Double it, add 7, then triple the result. "
            "What is the final counter value? Return only the integer."
        ),
        "answer": "51",
        "fact": "After doubling and adding 7, the counter is 19.",
        "state": "Record the counter after each operation as one compact state before answering.",
    },
    {
        "id": "negative_sort_unique",
        "kind": "list",
        "question": (
            "Sort [-1, 12, 7, -1, 12, 0, 7] in ascending order and remove duplicates. "
            "Return only the list."
        ),
        "answer": "[-1,0,7,12]",
        "fact": "The distinct values are -1, 12, 7, and 0.",
        "state": "Record the distinct values as one compact state before answering.",
    },
    {
        "id": "string_splice",
        "kind": "string",
        "question": "Insert 'XY' immediately after the first 4 characters of 'notebook'. Return only the string.",
        "answer": "noteXYbook",
        "fact": "The prefix is 'note' and the suffix is 'book'.",
        "state": "Record prefix, inserted text, and suffix as one compact state before answering.",
    },
    {
        "id": "set_constraint",
        "kind": "yesno",
        "question": "Every fep is a lor. No lor is a vek. Can a fep be a vek? Answer yes or no only.",
        "answer": "no",
        "fact": "A fep must be a lor, and no lor may be a vek.",
        "state": "Record the two set constraints as one compact state before answering.",
    },
    {
        "id": "python_count_evens",
        "kind": "code",
        "question": (
            "Write only Python code for count_evens(values). It must return how many integers in values "
            "are even, including negative even integers."
        ),
        "answer": "count_evens",
        "fact": "An integer is even exactly when value modulo 2 equals 0.",
        "state": "Record the predicate compactly, then write only the Python function.",
    },
    {
        "id": "code_review_sum_positive",
        "kind": "code",
        "question": (
            "Write only Python code for sum_positive(values). It must return the sum of exactly the "
            "strictly positive integers in values."
        ),
        "answer": "sum_positive",
        "fact": "Zero is not strictly positive, so only values greater than zero are added.",
        "state": "Record the inclusion rule compactly, then write only the Python function.",
    },
]


def normalized(value):
    return re.sub(r"\s+", "", str(value).strip().lower()).rstrip(".")


def function_matches(candidate, expected):
    try:
        tree = ast.parse(candidate)
    except SyntaxError:
        return False
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == expected]
    if len(functions) != 1 or len(functions[0].args.args) != 1:
        return False
    namespace = {
        "__builtins__": {
            "abs": abs,
            "enumerate": enumerate,
            "len": len,
            "max": max,
            "min": min,
            "range": range,
            "sum": sum,
        }
    }
    try:
        exec(compile(tree, "<response>", "exec"), namespace, namespace)
        func = namespace[expected]
        if expected == "count_evens":
            return func([2, -4, -3, 0, 5, 8]) == 4
        return func([-2, 0, 4, -1, 7]) == 11
    except Exception:
        return False


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
        return any(function_matches(candidate, case["answer"]) for candidate in candidates)
    raise ValueError(f"unknown kind: {case['kind']}")


def has_state_line(response):
    return bool(re.search(r"(?mi)^state=\S.+$", response))


def load_model(path, device, n_loop=0):
    checkpoint = torch.load(path, map_location="cpu")
    cfg = GPTConfig(**checkpoint["cfg"])
    if n_loop:
        cfg.n_loop = n_loop
    model = GPT(cfg).to(device).eval()
    model.load_state_dict(checkpoint["model"])
    return checkpoint, model


def ask(model, tokenizer, device, prompt, max_new):
    return generate(model, tokenizer, prompt, device, max_new=max_new, temp=0.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-new", type=int, default=112)
    parser.add_argument("--n-loop", type=int, default=0,
                        help="test-only latent-depth override (0 preserves checkpoint config)")
    args = parser.parse_args()
    if args.n_loop < 0:
        raise SystemExit("--n-loop must be zero or positive")

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    tokenizer = Tokenizer.from_file(args.tokenizer)
    checkpoint, model = load_model(args.ckpt, device, n_loop=args.n_loop)
    used_n_loop = model.cfg.n_loop
    rows = []
    for case in CASES:
        initial_prompt = f"Question: {case['question']}\nAnswer:"
        initial = ask(model, tokenizer, device, initial_prompt, args.max_new)
        review_prompt = (
            f"Question: {case['question']}\nPrevious answer: {initial}\n\n"
            "Check the previous answer independently. If it is wrong, correct it. "
            "Return only the final answer.\nAnswer:"
        )
        review = ask(model, tokenizer, device, review_prompt, args.max_new)
        fact_prompt = (
            f"Question: {case['question']}\nVerified intermediate fact: {case['fact']}\n"
            "Use that fact. Return only the final answer.\nAnswer:"
        )
        fact = ask(model, tokenizer, device, fact_prompt, args.max_new)
        state_prompt = (
            f"Question: {case['question']}\n{case['state']} "
            "First write exactly one short line beginning with 'state=', then give the final answer.\nAnswer:"
        )
        state = ask(model, tokenizer, device, state_prompt, args.max_new)
        reuse_prompt = (
            f"Question: {case['question']}\nThe prior state was:\n{state}\n\n"
            "Use that state to solve the original question. Return only the final answer.\nAnswer:"
        )
        reuse = ask(model, tokenizer, device, reuse_prompt, args.max_new)
        row = {
            "id": case["id"],
            "kind": case["kind"],
            "question": case["question"],
            "answer": case["answer"],
            "initial": {"prompt": initial_prompt, "response": initial, "correct": score(case, initial)},
            "review": {"prompt": review_prompt, "response": review, "correct": score(case, review)},
            "verified_fact": {"prompt": fact_prompt, "response": fact, "correct": score(case, fact)},
            "compact_state": {"prompt": state_prompt, "response": state, "valid_state_line": has_state_line(state)},
            "state_reuse": {"prompt": reuse_prompt, "response": reuse, "correct": score(case, reuse)},
        }
        row["valid_state_and_reuse"] = row["compact_state"]["valid_state_line"] and row["state_reuse"]["correct"]
        rows.append(row)
        print(
            f"[generalization] {case['id']} initial={row['initial']['correct']} "
            f"review={row['review']['correct']} fact={row['verified_fact']['correct']} "
            f"state={row['compact_state']['valid_state_line']} reuse={row['state_reuse']['correct']}",
            flush=True,
        )

    summary = {
        "initial": sum(row["initial"]["correct"] for row in rows),
        "review": sum(row["review"]["correct"] for row in rows),
        "verified_fact": sum(row["verified_fact"]["correct"] for row in rows),
        "valid_state_line": sum(row["compact_state"]["valid_state_line"] for row in rows),
        "state_reuse": sum(row["state_reuse"]["correct"] for row in rows),
        "valid_state_and_reuse": sum(row["valid_state_and_reuse"] for row in rows),
    }
    result = {
        "audit": "generalization_interview_v1",
        "device": device,
        "checkpoint": args.ckpt,
        "step": checkpoint.get("step"),
        "n_loop": used_n_loop,
        "cases": len(rows),
        "summary": summary,
        "rows": rows,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"[generalization] step={result['step']} summary={summary}", flush=True)


if __name__ == "__main__":
    main()
