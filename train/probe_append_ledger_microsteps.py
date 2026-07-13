#!/usr/bin/env python3
"""Measure raw-model likelihood for one ADL digit/carry microstep.

Each prompt exposes immutable decimal tapes and a scheduled position. The
model ranks exactly the 20 grammar-valid ``digit,carry`` records; no controller
computes or chooses an answer during scoring. This is a diagnostic for local
arithmetic signal, not a decoding method or a reasoning claim.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tokenizers import Tokenizer

from append_ledger_protocol import canonical_delta, expected_delta, initial_base, transition_prompt
from forced_choice_probe import candidate_score, load_model, rank_candidates


CASES = [
    ("fit_add_no_carry", "add", 8, 42, 13),
    ("fit_add_carry", "add", 8, 73, 19),
    ("fit_sub_no_borrow", "sub", 8, 74, 12),
    ("fit_sub_borrow", "sub", 8, 42, 19),
    ("value_add_carry", "add", 8, 9_999_997, 8),
    ("value_sub_borrow", "sub", 8, 9_000_000, 1_111_111),
    ("width16_add", "add", 16, 8_000_000_000_000_008, 9),
    ("width16_sub", "sub", 16, 9_000_000_000_000_002, 19),
]


def candidate_lines(step=0):
    return [canonical_delta({"step": step, "d": digit, "c": carry})
            for digit in range(10) for carry in range(2)]


def probe_case(model, tokenizer, device, case, style):
    identifier, operation, width, left, right = case
    base = initial_base(operation, left, right, width)
    expected = canonical_delta(expected_delta(base, 0, 0))
    prompt = transition_prompt(base, [], [], 0, style=style)
    scored = []
    for candidate in candidate_lines():
        row = candidate_score(model, tokenizer, device, prompt, candidate)
        row["candidate"] = candidate
        row["correct"] = candidate == expected
        scored.append(row)
    ranked = rank_candidates(scored)
    correct_rank = next(index + 1 for index, row in enumerate(ranked) if row["correct"])
    return {
        "id": identifier,
        "style": style,
        "operation": operation,
        "width": width,
        "left": left,
        "right": right,
        "prompt": prompt,
        "expected": expected,
        "ranked": ranked,
        "correct_rank": correct_rank,
        "correct_top1": correct_rank == 1,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--styles", nargs="+", choices=("core", "heldout"), default=("core", "heldout"))
    args = parser.parse_args()
    out = Path(args.out)
    if out.exists():
        raise SystemExit("refusing to overwrite output: {}".format(out))
    import torch
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    checkpoint, model = load_model(args.ckpt, device)
    tokenizer = Tokenizer.from_file(args.tokenizer)
    rows = []
    for style in args.styles:
        for case in CASES:
            row = probe_case(model, tokenizer, device, case, style)
            rows.append(row)
            print("[adl-microstep] {} {} rank={}/20".format(
                style, row["id"], row["correct_rank"]
            ), flush=True)
    result = {
        "audit": "append_ledger_microstep_likelihood_v1",
        "claim_boundary": (
            "This ranks fixed grammar-valid local records. It neither executes a rollout nor establishes "
            "free generation, reasoning, or context scaling."
        ),
        "checkpoint": args.ckpt,
        "step": checkpoint.get("step"),
        "device": device,
        "cases": len(CASES),
        "styles": list(args.styles),
        "top1": sum(row["correct_top1"] for row in rows),
        "mean_correct_rank": sum(row["correct_rank"] for row in rows) / len(rows),
        "rows": rows,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print("[adl-microstep] step={} top1={}/{} mean-rank={:.3f}".format(
        result["step"], result["top1"], len(rows), result["mean_correct_rank"]
    ), flush=True)


if __name__ == "__main__":
    main()
