#!/usr/bin/env python3
"""Frozen confirmation of raw-pretrain procedural behavior across prompt modes.

Cases are generated before model loading from a fixed seed. The artifact preserves
all prompts/responses and hashes the script, checkpoint, tokenizer, and generated
case manifest. No generated row is used for training.
"""

import argparse
import hashlib
import inspect
import json
import random
import re
from pathlib import Path

import torch
from tokenizers import Tokenizer

from eval_suite import generate
from model import GPT, GPTConfig


SEED = 2026071501
PER_FAMILY = 5


DEMOS = {
    "multiply_subtract": (
        "Problem: Compute 16 times 13, then subtract 11.\n"
        "Work: 16 * 13 = 208; 208 - 11 = 197\nAnswer: 197\n\n"
        "Problem: Compute 21 times 14, then subtract 17.\n"
        "Work: 21 * 14 = 294; 294 - 17 = 277\nAnswer: 277"
    ),
    "base_conversion": (
        "Problem: Convert the base-5 numeral 324 to base 10.\n"
        "Work: 3*25 + 2*5 + 4 = 89\nAnswer: 89\n\n"
        "Problem: Convert the base-6 numeral 241 to base 10.\n"
        "Work: 2*36 + 4*6 + 1 = 97\nAnswer: 97"
    ),
    "sequential_state": (
        "Problem: Start at 8, add 6, multiply by 3, then subtract 5.\n"
        "Work: 8+6=14; 14*3=42; 42-5=37\nAnswer: 37\n\n"
        "Problem: Start at 12, add 7, multiply by 2, then subtract 9.\n"
        "Work: 12+7=19; 19*2=38; 38-9=29\nAnswer: 29"
    ),
    "modular_update": (
        "Problem: Add 19 and 27, then give the remainder after division by 8.\n"
        "Work: 19+27=46; 46 mod 8 = 6\nAnswer: 6\n\n"
        "Problem: Add 34 and 25, then give the remainder after division by 11.\n"
        "Work: 34+25=59; 59 mod 11 = 4\nAnswer: 4"
    ),
}


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_cases():
    rng = random.Random(SEED)
    rows = []
    for index in range(PER_FAMILY):
        a = rng.randint(22, 39)
        b = rng.randint(12, 19)
        c = rng.randint(13, 31)
        product = a * b
        rows.append({
            "id": f"multiply_subtract_{index}",
            "family": "multiply_subtract",
            "question": f"Compute {a} times {b}, then subtract {c}.",
            "expression": f"{a} * {b} - {c} =",
            "answer": product - c,
            "required_intermediates": [product],
        })

        base = rng.randint(5, 9)
        d2, d1, d0 = rng.randint(2, base - 1), rng.randint(0, base - 1), rng.randint(0, base - 1)
        numeral = f"{d2}{d1}{d0}"
        terms = [d2 * base * base, d1 * base, d0]
        rows.append({
            "id": f"base_conversion_{index}",
            "family": "base_conversion",
            "question": f"Convert the base-{base} numeral {numeral} to base 10.",
            "expression": f"int('{numeral}', {base}) =",
            "answer": sum(terms),
            "required_intermediates": terms,
        })

        start = rng.randint(7, 19)
        add = rng.randint(5, 14)
        mul = rng.randint(2, 5)
        sub = rng.randint(6, 23)
        after_add = start + add
        after_mul = after_add * mul
        rows.append({
            "id": f"sequential_state_{index}",
            "family": "sequential_state",
            "question": f"Start at {start}, add {add}, multiply by {mul}, then subtract {sub}.",
            "expression": f"(({start} + {add}) * {mul}) - {sub} =",
            "answer": after_mul - sub,
            "required_intermediates": [after_add, after_mul],
        })

        left = rng.randint(21, 59)
        right = rng.randint(17, 53)
        modulus = rng.randint(7, 16)
        total = left + right
        rows.append({
            "id": f"modular_update_{index}",
            "family": "modular_update",
            "question": f"Add {left} and {right}, then give the remainder after division by {modulus}.",
            "expression": f"({left} + {right}) % {modulus} =",
            "answer": total % modulus,
            "required_intermediates": [total],
        })
    return rows


def prompt_modes(case):
    return {
        "direct_qa": f"Question: {case['question']} Return only the final integer.\nAnswer:",
        "bare_expression": case["expression"],
        "worked_completion": f"{DEMOS[case['family']]}\n\nProblem: {case['question']}\nWork:",
    }


def first_answer_segment(response):
    cut = response
    for marker in ("\n\nProblem:", "\n\nQuestion:", "\n\n###", "\n\n##"):
        cut = cut.split(marker, 1)[0]
    return cut.strip()


def score_response(response, case):
    segment = first_answer_segment(response)
    values = [int(value) for value in re.findall(r"(?<![A-Za-z0-9_])-?\d+", segment)]
    answer = case["answer"]
    return {
        "answer_segment": segment,
        "integers": values,
        "leading_correct": bool(values) and values[0] == answer,
        "final_correct": bool(values) and values[-1] == answer,
        "contains_answer": answer in values,
        "intermediates_present": all(value in values for value in case["required_intermediates"]),
    }


def load_model(path, device):
    checkpoint = torch.load(path, map_location="cpu")
    model = GPT(GPTConfig(**checkpoint["cfg"])).to(device).eval()
    model.load_state_dict(checkpoint["model"])
    return checkpoint, model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-new", type=int, default=64)
    args = parser.parse_args()

    cases = build_cases()
    case_bytes = (json.dumps(cases, sort_keys=True, separators=(",", ":")) + "\n").encode()
    source_bytes = inspect.getsource(inspect.getmodule(main)).encode()
    print(f"[confirmation] cases={len(cases)} cases_sha256={sha256_bytes(case_bytes)}", flush=True)

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    tokenizer = Tokenizer.from_file(args.tokenizer)
    checkpoint, model = load_model(args.ckpt, device)
    rows = []
    for case in cases:
        modes = {}
        for mode, prompt in prompt_modes(case).items():
            print(f"[confirmation] case={case['id']} mode={mode}", flush=True)
            response = generate(model, tokenizer, prompt, device, max_new=args.max_new, temp=0.0)
            modes[mode] = {"prompt": prompt, "response": response, **score_response(response, case)}
        rows.append({**case, "modes": modes})

    mode_names = list(prompt_modes(cases[0]))
    summary = {
        mode: {
            metric: sum(row["modes"][mode][metric] for row in rows)
            for metric in ("leading_correct", "final_correct", "contains_answer", "intermediates_present")
        }
        for mode in mode_names
    }
    by_family = {
        family: {
            mode: sum(
                row["modes"][mode]["final_correct"]
                for row in rows if row["family"] == family
            )
            for mode in mode_names
        }
        for family in sorted(DEMOS)
    }
    result = {
        "audit": "raw_continuation_confirmation_v1",
        "seed": SEED,
        "per_family": PER_FAMILY,
        "case_count": len(cases),
        "script_sha256": sha256_bytes(source_bytes),
        "cases_sha256": sha256_bytes(case_bytes),
        "device": device,
        "checkpoint": args.ckpt,
        "checkpoint_step": checkpoint.get("step"),
        "checkpoint_sha256": sha256_file(args.ckpt),
        "tokenizer_sha256": sha256_file(args.tokenizer),
        "max_new": args.max_new,
        "summary": summary,
        "by_family_final_correct": by_family,
        "rows": rows,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"summary": summary, "by_family": by_family}, indent=2), flush=True)


if __name__ == "__main__":
    main()
