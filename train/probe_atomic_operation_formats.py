#!/usr/bin/env python3
"""Frozen-format atomic and chained arithmetic probe on immutable SSC cases.

Every model call receives only an operation and its operands. No correct
intermediate, demonstration, retry, repair, search, or verifier feedback is
provided. Atomic calls use gold inputs to measure the primitive; chained calls
carry the model's own first-line result to measure source-free composition.
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
from probe_source_scheduled_continuation import apply_operation, parse_schedule


FORMATS = ("question_answer", "bare_equation", "problem_work")


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def operation_clause(value, operation, operand):
    if operation == "add":
        return f"Compute {value} plus {operand}."
    if operation == "subtract":
        return f"Compute {value} minus {operand}."
    if operation == "multiply":
        return f"Compute {value} times {operand}."
    if operation == "remainder":
        return f"Give the remainder after dividing {value} by {operand}."
    raise ValueError(operation)


def format_prompt(name, value, operation, operand):
    clause = operation_clause(value, operation, operand)
    if name == "question_answer":
        return f"Question: {clause} Return only the final integer.\nAnswer:"
    if name == "problem_work":
        return f"Problem: {clause}\nWork:"
    if name == "bare_equation":
        symbol = {"add": "+", "subtract": "-", "multiply": "*", "remainder": "%"}[operation]
        return f"{value} {symbol} {operand} ="
    raise ValueError(name)


def first_nonempty_line(text):
    return next((line.strip() for line in text.splitlines() if line.strip()), "")


def line_integers(text):
    return [int(value) for value in re.findall(r"(?<![A-Za-z0-9_])-?\d+", first_nonempty_line(text))]


def parse_first_line_final(text):
    values = line_integers(text)
    return values[-1] if values else None


def load_model(path, device):
    checkpoint = torch.load(path, map_location="cpu")
    model = GPT(GPTConfig(**checkpoint["cfg"])).to(device).eval()
    model.load_state_dict(checkpoint["model"])
    return checkpoint, model


def call_model(model, tokenizer, device, prompt, max_new):
    response = generate(model, tokenizer, prompt, device, max_new=max_new, temp=0.0)
    predicted = parse_first_line_final(response)
    return {
        "prompt": prompt,
        "response": response,
        "first_line": first_nonempty_line(response),
        "first_line_integers": line_integers(response),
        "predicted_state": predicted,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--cases", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-new", type=int, default=48)
    args = parser.parse_args()

    source = json.loads(Path(args.cases).read_text())
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    tokenizer = Tokenizer.from_file(args.tokenizer)
    checkpoint, model = load_model(args.ckpt, device)
    rows = []
    calls = 0

    for row in source["rows"]:
        start, schedule = parse_schedule(row)
        true_states = []
        true_state = start
        for operation, operand in schedule:
            expected = apply_operation(true_state, operation, operand)
            true_states.append((true_state, operation, operand, expected))
            true_state = expected

        atomic = {}
        chained = {}
        for format_name in FORMATS:
            atomic_steps = []
            for index, (input_state, operation, operand, expected) in enumerate(true_states):
                prompt = format_prompt(format_name, input_state, operation, operand)
                print(f"[atomic] case={row['id']} format={format_name} step={index}", flush=True)
                record = call_model(model, tokenizer, device, prompt, args.max_new)
                calls += 1
                record.update({
                    "index": index,
                    "operation": operation,
                    "operand": operand,
                    "input_state": input_state,
                    "expected_state": expected,
                    "correct": record["predicted_state"] == expected,
                })
                atomic_steps.append(record)
            atomic[format_name] = atomic_steps

            model_state = start
            chain_steps = []
            for index, (gold_input, operation, operand, expected) in enumerate(true_states):
                if model_state is None:
                    break
                prompt = format_prompt(format_name, model_state, operation, operand)
                print(f"[chain] case={row['id']} format={format_name} step={index}", flush=True)
                record = call_model(model, tokenizer, device, prompt, args.max_new)
                calls += 1
                predicted = record["predicted_state"]
                local_expected = apply_operation(model_state, operation, operand)
                record.update({
                    "index": index,
                    "operation": operation,
                    "operand": operand,
                    "input_state": model_state,
                    "gold_input_state": gold_input,
                    "gold_expected_state": expected,
                    "local_expected_state": local_expected,
                    "local_operation_correct": predicted == local_expected,
                    "gold_state_correct": predicted == expected,
                })
                chain_steps.append(record)
                model_state = predicted
            chained[format_name] = {
                "steps": chain_steps,
                "all_gold_transitions_correct": (
                    len(chain_steps) == len(schedule)
                    and all(step["gold_state_correct"] for step in chain_steps)
                ),
                "final_correct": model_state == row["answer"],
            }

        rows.append({
            "id": row["id"],
            "family": row["family"],
            "answer": row["answer"],
            "initial_state": start,
            "schedule": schedule,
            "atomic": atomic,
            "chained": chained,
        })

    summary = {"case_count": len(rows), "model_calls": calls, "formats": {}}
    total_atomic = sum(len(parse_schedule(row)[1]) for row in source["rows"])
    for format_name in FORMATS:
        atomic_correct = sum(
            step["correct"] for row in rows for step in row["atomic"][format_name]
        )
        summary["formats"][format_name] = {
            "atomic_correct": atomic_correct,
            "atomic_total": total_atomic,
            "chains_all_gold_transitions_correct": sum(
                row["chained"][format_name]["all_gold_transitions_correct"] for row in rows
            ),
            "chains_final_correct": sum(row["chained"][format_name]["final_correct"] for row in rows),
        }

    result = {
        "audit": "atomic_operation_format_matrix_v1",
        "interpretation": "posthoc_format_access_diagnostic_only",
        "formats": list(FORMATS),
        "controller": "fixed formats; no demonstrations, retries, repair, search, or verifier feedback",
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
