#!/usr/bin/env python3
"""Post-hoc SSC diagnostic on the immutable raw-260k confirmation cases.

The controller copies only the requested operation schedule. It never supplies a
correct intermediate, retries, repairs, searches, or uses verifier feedback.
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


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parse_schedule(row):
    question = row["question"]
    family = row["family"]
    if family == "multiply_subtract":
        match = re.fullmatch(r"Compute (\d+) times (\d+), then subtract (\d+)\.", question)
        if not match:
            raise ValueError(f"unparsed multiply question: {question}")
        start, multiplier, subtractor = map(int, match.groups())
        return start, [("multiply", multiplier), ("subtract", subtractor)]
    if family == "sequential_state":
        match = re.fullmatch(r"Start at (\d+), add (\d+), multiply by (\d+), then subtract (\d+)\.", question)
        if not match:
            raise ValueError(f"unparsed state question: {question}")
        start, addend, multiplier, subtractor = map(int, match.groups())
        return start, [("add", addend), ("multiply", multiplier), ("subtract", subtractor)]
    if family == "modular_update":
        match = re.fullmatch(r"Add (\d+) and (\d+), then give the remainder after division by (\d+)\.", question)
        if not match:
            raise ValueError(f"unparsed modular question: {question}")
        start, addend, modulus = map(int, match.groups())
        return start, [("add", addend), ("remainder", modulus)]
    if family == "base_conversion":
        match = re.fullmatch(r"Convert the base-(\d+) numeral (\d+) to base 10\.", question)
        if not match:
            raise ValueError(f"unparsed base question: {question}")
        base = int(match.group(1))
        digits = [int(value) for value in match.group(2)]
        schedule = []
        for digit in digits[1:]:
            schedule.extend((("multiply", base), ("add", digit)))
        return digits[0], schedule
    raise ValueError(f"unknown family: {family}")


def apply_operation(value, operation, operand):
    if operation == "add":
        return value + operand
    if operation == "subtract":
        return value - operand
    if operation == "multiply":
        return value * operand
    if operation == "remainder":
        return value % operand
    raise ValueError(operation)


def transition_prompt(value, operation, operand):
    phrase = {
        "add": f"add {operand}",
        "subtract": f"subtract {operand}",
        "multiply": f"multiply by {operand}",
        "remainder": f"take the remainder after division by {operand}",
    }[operation]
    return (
        f"Current state: {value}\n"
        f"Requested operation: {phrase}\n"
        "Apply exactly this one operation. Return only the next integer.\n"
        "Next state:"
    )


def first_integer(response):
    match = re.search(r"(?<![A-Za-z0-9_])-?\d+", response)
    return int(match.group(0)) if match else None


def load_model(path, device):
    checkpoint = torch.load(path, map_location="cpu")
    model = GPT(GPTConfig(**checkpoint["cfg"])).to(device).eval()
    model.load_state_dict(checkpoint["model"])
    return checkpoint, model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--cases", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-new", type=int, default=32)
    args = parser.parse_args()

    source = json.loads(Path(args.cases).read_text())
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    tokenizer = Tokenizer.from_file(args.tokenizer)
    checkpoint, model = load_model(args.ckpt, device)
    rows = []
    total_calls = 0
    for row in source["rows"]:
        start, schedule = parse_schedule(row)
        model_state = start
        true_state = start
        steps = []
        parse_failed = False
        for index, (operation, operand) in enumerate(schedule):
            expected = apply_operation(true_state, operation, operand)
            prompt = transition_prompt(model_state, operation, operand)
            print(f"[ssc] case={row['id']} step={index} operation={operation}", flush=True)
            response = generate(model, tokenizer, prompt, device, max_new=args.max_new, temp=0.0)
            predicted = first_integer(response)
            total_calls += 1
            correct = predicted == expected
            steps.append({
                "index": index,
                "operation": operation,
                "operand": operand,
                "input_state": model_state,
                "expected_state": expected,
                "prompt": prompt,
                "response": response,
                "predicted_state": predicted,
                "correct": correct,
            })
            true_state = expected
            if predicted is None:
                parse_failed = True
                break
            model_state = predicted
        all_transitions = len(steps) == len(schedule) and all(step["correct"] for step in steps)
        rows.append({
            "id": row["id"],
            "family": row["family"],
            "answer": row["answer"],
            "initial_state": start,
            "schedule": schedule,
            "steps": steps,
            "parse_failed": parse_failed,
            "first_transition_correct": bool(steps) and steps[0]["correct"],
            "all_transitions_correct": all_transitions,
            "final_correct": all_transitions and model_state == row["answer"],
        })

    families = sorted({row["family"] for row in rows})
    summary = {
        "case_count": len(rows),
        "model_calls": total_calls,
        "first_transition_correct": sum(row["first_transition_correct"] for row in rows),
        "all_transitions_correct": sum(row["all_transitions_correct"] for row in rows),
        "final_correct": sum(row["final_correct"] for row in rows),
        "by_family": {
            family: {
                metric: sum(row[metric] for row in rows if row["family"] == family)
                for metric in ("first_transition_correct", "all_transitions_correct", "final_correct")
            }
            for family in families
        },
    }
    result = {
        "audit": "source_scheduled_continuation_diagnostic_v1",
        "interpretation": "posthoc selector_halting_diagnostic_only",
        "controller": "fixed parsed operation schedule; no retries, repair, search, or verifier feedback",
        "source_cases": args.cases,
        "source_cases_sha256": sha256_file(args.cases),
        "checkpoint_step": checkpoint.get("step"),
        "checkpoint_sha256": sha256_file(args.ckpt),
        "tokenizer_sha256": sha256_file(args.tokenizer),
        "device": device,
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
