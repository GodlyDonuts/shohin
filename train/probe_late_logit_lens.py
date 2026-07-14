#!/usr/bin/env python3
"""Read-only late-layer logit-lens screen for candidate reasoning intermediates.

This is intentionally not a Jacobian lens and does not establish a workspace.
It is a cheap, observational screen: if a correct, non-input intermediate is
already highly ranked in a late residual stream while the final answer fails,
then a reflection-style intervention is more plausible than relearning the
entire operation. If it is absent, direct algorithmic supervision is needed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from tokenizers import Tokenizer

from model import GPT, GPTConfig


CASES = [
    {
        "id": "add_then_multiply_5",
        "prompt": "Question: Evaluate (2 + 3) times 4. Return only the final integer.\nAnswer:",
        "intermediates": ["5", "multiplication"],
        "expected_first_answer": ["2"],
    },
    {
        "id": "subtract_then_multiply_5",
        "prompt": "Question: Evaluate (8 - 3) times 2. Return only the final integer.\nAnswer:",
        "intermediates": ["5", "multiplication"],
        "expected_first_answer": ["1"],
    },
    {
        "id": "add_then_multiply_9",
        "prompt": "Question: Evaluate (3 + 6) times 2. Return only the final integer.\nAnswer:",
        "intermediates": ["9", "multiplication"],
        "expected_first_answer": ["1"],
    },
    {
        "id": "state_after_addition",
        "prompt": "Question: Start with n = 12. Add 7, multiply by 4, then subtract 13. What is n? Return only the integer.\nAnswer:",
        "intermediates": ["1", "9", "multiplication"],
        "expected_first_answer": ["6"],
    },
    {
        "id": "base8_place_value",
        "prompt": "Question: What is the base-10 value of base-8 number 725? Return only the integer.\nAnswer:",
        "intermediates": ["5", "8", "addition"],
        "expected_first_answer": ["4"],
    },
    {
        "id": "precedence_correction",
        "prompt": "Question: A student says 18 divided by 3 plus 2 equals 3. Correct the student. Return only the correct integer.\nAnswer:",
        "intermediates": ["6", "addition"],
        "expected_first_answer": ["8"],
    },
]


def one_token_ids(tokenizer: Tokenizer, terms: list[str]) -> dict[str, list[int]]:
    result = {}
    for term in terms:
        ids = set()
        for variant in (term, " " + term):
            encoded = tokenizer.encode(variant).ids
            if len(encoded) == 1:
                ids.add(encoded[0])
        if not ids:
            raise ValueError("no single-token variant for {!r}".format(term))
        result[term] = sorted(ids)
    return result


def token_label(tokenizer: Tokenizer, token_id: int) -> str:
    return tokenizer.decode([token_id], skip_special_tokens=False)


def rank_for_ids(logits: torch.Tensor, token_ids: list[int]) -> tuple[int, int]:
    best_id = max(token_ids, key=lambda token_id: float(logits[token_id]))
    rank = int((logits > logits[best_id]).sum().item()) + 1
    return rank, best_id


def inspect_case(model: GPT, tokenizer: Tokenizer, device: str, case: dict, topk: int) -> dict:
    requested = case["intermediates"] + case["expected_first_answer"]
    candidate_ids = one_token_ids(tokenizer, requested)
    residuals = []
    hooks = [block.register_forward_hook(lambda _module, _inputs, output: residuals.append(output[0][:, -1].detach()))
             for block in model.blocks]
    try:
        token_ids = tokenizer.encode(case["prompt"]).ids
        x = torch.tensor([token_ids], dtype=torch.long, device=device)
        with torch.no_grad():
            logits, _ = model(x)
    finally:
        for hook in hooks:
            hook.remove()
    layers = []
    for layer, hidden in enumerate(residuals):
        with torch.no_grad():
            layer_logits = model.head(model.norm(hidden.to(device)))[0].float()
        top_ids = torch.topk(layer_logits, k=topk).indices.tolist()
        terms = {}
        for term, ids in candidate_ids.items():
            rank, token_id = rank_for_ids(layer_logits, ids)
            terms[term] = {"rank": rank, "token_id": token_id, "token": token_label(tokenizer, token_id)}
        layers.append({
            "layer": layer,
            "terms": terms,
            "top": [{"token_id": token_id, "token": token_label(tokenizer, token_id)} for token_id in top_ids],
        })
    initial_logits = logits[0, -1].float()
    summary = {}
    for term in requested:
        best = min(layers, key=lambda row: row["terms"][term]["rank"])
        summary[term] = {
            "best_rank": best["terms"][term]["rank"],
            "best_layer": best["layer"],
            "top20_at_any_late_layer": any(
                row["layer"] >= len(layers) // 2 and row["terms"][term]["rank"] <= 20 for row in layers
            ),
        }
    first_top = torch.topk(initial_logits, k=topk).indices.tolist()
    return {
        **case,
        "prompt_tokens": len(token_ids),
        "summary": summary,
        "final_top": [{"token_id": token_id, "token": token_label(tokenizer, token_id)} for token_id in first_top],
        "layers": layers,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--topk", type=int, default=10)
    args = parser.parse_args()
    if args.topk <= 0:
        raise SystemExit("topk must be positive")
    out = Path(args.out)
    if out.exists():
        raise SystemExit("refusing to overwrite output")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(args.ckpt, map_location="cpu")
    model = GPT(GPTConfig(**checkpoint["cfg"])).to(device).eval()
    model.load_state_dict(checkpoint["model"])
    tokenizer = Tokenizer.from_file(args.tokenizer)
    rows = [inspect_case(model, tokenizer, device, case, args.topk) for case in CASES]
    result = {
        "audit": "late_logit_lens_screen_v1",
        "claim_boundary": "Observational late-layer unembedding screen, not a Jacobian lens, causal patch, or workspace claim.",
        "checkpoint": args.ckpt,
        "step": checkpoint.get("step"),
        "device": device,
        "cases": rows,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    for row in rows:
        compact = ", ".join("{}@L{}:r{}".format(term, values["best_layer"], values["best_rank"])
                            for term, values in row["summary"].items())
        print("[late-lens] {} {}".format(row["id"], compact), flush=True)


if __name__ == "__main__":
    main()
