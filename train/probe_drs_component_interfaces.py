#!/usr/bin/env python3
"""Fresh transcript probe for the frozen DRS compiler/executor/serializer split.

The rollout host parses and relays model-emitted canonical states. It never
computes, repairs, or replaces an inference-time state. Exact arithmetic is
used only after generation to score immutable transcripts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import torch
from tokenizers import Tokenizer

from digitwise_protocol import (
    apply_microstep,
    canonical_state,
    final_prompt,
    initial_state,
    microstep_prompt,
    parse_answer,
    parse_state,
    state_answer,
)
from eval_suite import generate
from model import GPT, GPTConfig


EXPECTED_CHECKPOINT_SHA256 = (
    "d79e9df26caecb9801118d1bf68bd7b85381a06b256f23478acffe40a2108459"
)
EXPECTED_TOKENIZER_SHA256 = (
    "87532df5c121753de3b29194e1f9e3de47986d3f5359548fdf93606773a233d4"
)

CASES = (
    {"id": "w5_add_c0", "op": "add", "left": 13_342, "right": 26_053, "width": 5},
    {"id": "w5_add_c1", "op": "add", "left": 93_758, "right": 61_267, "width": 5},
    {"id": "w5_sub", "op": "sub", "left": 76_520, "right": 13_342, "width": 5},
    {
        "id": "w7_add_c0",
        "op": "add",
        "left": 1_203_406,
        "right": 2_310_502,
        "width": 7,
    },
    {
        "id": "w7_add_c1",
        "op": "add",
        "left": 7_654_321,
        "right": 5_678_909,
        "width": 7,
    },
    {
        "id": "w7_sub",
        "op": "sub",
        "left": 8_000_003,
        "right": 2_345_678,
        "width": 7,
    },
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_model(path: Path, device: str) -> tuple[dict, GPT]:
    checkpoint = torch.load(path, map_location="cpu")
    model = GPT(GPTConfig(**checkpoint["cfg"])).to(device).eval()
    model.load_state_dict(checkpoint["model"])
    return checkpoint, model


def ask(model: GPT, tokenizer: Tokenizer, device: str, prompt: str, max_new: int) -> str:
    return generate(
        model,
        tokenizer,
        prompt,
        device,
        max_new=max_new,
        temp=0.0,
        stop="\nQuestion:",
    ).strip()


def ordinary_question(case: dict) -> str:
    verb = "Add" if case["op"] == "add" else "Subtract"
    if case["op"] == "add":
        return f"{verb} {case['left']} and {case['right']}."
    return f"{verb} {case['right']} from {case['left']}."


def parse_direct_answer(text: str) -> int | None:
    match = re.fullmatch(r"(?:answer\s*=\s*)?(-?\d+)", text.strip())
    return int(match.group(1)) if match else None


def compiler_prompt(case: dict, demonstrations: bool, constrained: bool) -> str:
    intro = (
        "Translate one ordinary decimal request into a canonical DWS initial state. "
        "Use the shortest width that holds both operands. Digits in a and b are "
        "least-significant first. Set p=0, c=0, r to width zeroes, and z=0. "
        "Emit only one dws line.\n"
    )
    examples = ""
    if demonstrations:
        examples = (
            "Question: Add 314 and 207.\n"
            "DWS: dws:op=add;w=3;p=0;c=0;a=413;b=702;r=000;z=0\n\n"
            "Question: Subtract 1234 from 5678.\n"
            "DWS: dws:op=sub;w=4;p=0;c=0;a=8765;b=4321;r=0000;z=0\n\n"
        )
    suffix = f"Question: {ordinary_question(case)}\nDWS:"
    if constrained:
        suffix += (
            f" dws:op={case['op']};w={case['width']};p=0;c=0;a="
        )
    return intro + examples + suffix


def serializer_prompt(state: dict, mode: str) -> str:
    line = canonical_state(state)
    if mode == "native":
        return final_prompt(state, style="heldout")
    rule = (
        "The r tape is least-significant first. Reverse all r digits, remove only "
        "leading zeroes, and for addition prepend one extra 1 exactly when c=1. "
        "Emit only answer=<integer>.\n"
    )
    if mode == "rule":
        return rule + f"Machine record: {line}\nResult:"
    if mode == "one_shot":
        return (
            rule
            + "Example record: "
            + "dws:op=add;w=4;p=4;c=1;a=0000;b=0000;r=2849;z=1\n"
            + "Example result: answer=19482\n\n"
            + f"Machine record: {line}\nResult:"
        )
    raise ValueError(f"unknown serializer mode: {mode}")


def gold_terminal(case: dict) -> dict:
    state = initial_state(case["op"], case["left"], case["right"], case["width"])
    for _ in range(case["width"]):
        state = apply_microstep(state)
    return state


def model_rollout(
    model: GPT,
    tokenizer: Tokenizer,
    device: str,
    start: dict,
    max_new: int,
) -> dict:
    state = dict(start)
    rows = []
    for turn in range(int(start["w"]) + 1):
        if state["z"]:
            break
        prompt = microstep_prompt(state, style="heldout")
        response = ask(model, tokenizer, device, prompt, max_new)
        parsed = parse_state(response)
        rows.append(
            {
                "turn": turn,
                "prompt": prompt,
                "response": response,
                "parsed": parsed,
            }
        )
        if parsed is None:
            return {"closed": False, "state": None, "turns": rows}
        state = parsed
    return {"closed": bool(state["z"]), "state": state, "turns": rows}


def annotate_oracle_rollout(start: dict, rollout: dict) -> None:
    """Attach gold comparisons after generation without steering the rollout."""
    expected = dict(start)
    first_mismatch = None
    exact_steps = 0
    for row in rollout["turns"]:
        expected = apply_microstep(expected)
        row["expected"] = canonical_state(expected)
        row["exact"] = row["parsed"] == expected
        exact_steps += int(row["exact"])
        if not row["exact"] and first_mismatch is None:
            first_mismatch = {
                "turn": row["turn"],
                "expected": row["expected"],
                "response": row["response"],
            }
    rollout["transition_exact"] = exact_steps
    rollout["transition_total"] = len(rollout["turns"])
    rollout["first_mismatch"] = first_mismatch


def probe_case(
    model: GPT,
    tokenizer: Tokenizer,
    device: str,
    case: dict,
    max_new: int,
) -> dict:
    initial = initial_state(case["op"], case["left"], case["right"], case["width"])
    terminal = gold_terminal(case)
    expected_initial = canonical_state(initial)
    expected_answer = state_answer(terminal)

    direct_prompt = f"Question: {ordinary_question(case)} Return only the integer.\nAnswer:"
    direct_response = ask(model, tokenizer, device, direct_prompt, max_new)
    direct_answer = parse_direct_answer(direct_response)

    compiler = {}
    for mode, demonstrations, constrained in (
        ("zero_shot", False, False),
        ("two_shot", True, False),
        ("two_shot_constrained", True, True),
    ):
        prompt = compiler_prompt(case, demonstrations, constrained)
        response = ask(model, tokenizer, device, prompt, max_new)
        if constrained:
            response = (
                f"dws:op={case['op']};w={case['width']};p=0;c=0;a=" + response
            )
        parsed = parse_state(response)
        compiler[mode] = {
            "prompt": prompt,
            "response": response,
            "parsed": parsed,
            "exact": response == expected_initial,
        }

    serializer = {}
    for mode in ("native", "rule", "one_shot"):
        prompt = serializer_prompt(terminal, mode)
        response = ask(model, tokenizer, device, prompt, max_new)
        serializer[mode] = {
            "prompt": prompt,
            "response": response,
            "parsed_answer": parse_answer(response),
            "exact": parse_answer(response) == expected_answer,
        }

    oracle_compiled = model_rollout(
        model, tokenizer, device, initial, max_new
    )
    annotate_oracle_rollout(initial, oracle_compiled)
    oracle_compiled["state_exact"] = oracle_compiled["state"] == terminal
    oracle_compiled["serializers"] = {}
    if oracle_compiled["closed"]:
        for mode in ("native", "rule", "one_shot"):
            prompt = serializer_prompt(oracle_compiled["state"], mode)
            response = ask(model, tokenizer, device, prompt, max_new)
            oracle_compiled["serializers"][mode] = {
                "prompt": prompt,
                "response": response,
                "parsed_answer": parse_answer(response),
                "exact": parse_answer(response) == expected_answer,
            }

    compiled_rollouts = {}
    for mode in ("zero_shot", "two_shot", "two_shot_constrained"):
        compiled = compiler[mode]["parsed"]
        if compiled is None:
            compiled_rollouts[mode] = {"admitted": False}
            continue
        rollout = model_rollout(model, tokenizer, device, compiled, max_new)
        rollout["admitted"] = True
        rollout["compiler_exact"] = compiler[mode]["exact"]
        rollout["final_exact"] = False
        if rollout["closed"]:
            prompt = serializer_prompt(rollout["state"], "one_shot")
            response = ask(model, tokenizer, device, prompt, max_new)
            rollout["serializer"] = {
                "prompt": prompt,
                "response": response,
                "parsed_answer": parse_answer(response),
            }
            rollout["final_exact"] = parse_answer(response) == expected_answer
        compiled_rollouts[mode] = rollout

    return {
        "id": case["id"],
        "case": case,
        "expected_initial": expected_initial,
        "expected_terminal": canonical_state(terminal),
        "expected_answer": expected_answer,
        "direct": {
            "prompt": direct_prompt,
            "response": direct_response,
            "parsed_answer": direct_answer,
            "exact": direct_answer == expected_answer,
        },
        "compiler": compiler,
        "gold_terminal_serializer": serializer,
        "oracle_compiled_rollout": oracle_compiled,
        "model_compiled_rollouts": compiled_rollouts,
    }


def summarize(rows: list[dict]) -> dict:
    modes = ("zero_shot", "two_shot", "two_shot_constrained")
    serializer_modes = ("native", "rule", "one_shot")
    widths = sorted({int(row["case"]["width"]) for row in rows})
    return {
        "cases": len(rows),
        "direct_exact": sum(row["direct"]["exact"] for row in rows),
        "compiler_exact": {
            mode: sum(row["compiler"][mode]["exact"] for row in rows)
            for mode in modes
        },
        "compiler_parseable": {
            mode: sum(row["compiler"][mode]["parsed"] is not None for row in rows)
            for mode in modes
        },
        "gold_terminal_serializer_exact": {
            mode: sum(row["gold_terminal_serializer"][mode]["exact"] for row in rows)
            for mode in serializer_modes
        },
        "oracle_compiled_closed": sum(
            row["oracle_compiled_rollout"]["closed"] for row in rows
        ),
        "oracle_compiled_state_exact": sum(
            row["oracle_compiled_rollout"]["state_exact"] for row in rows
        ),
        "oracle_transition_exact": {
            "exact": sum(
                row["oracle_compiled_rollout"]["transition_exact"] for row in rows
            ),
            "total": sum(
                row["oracle_compiled_rollout"]["transition_total"] for row in rows
            ),
            "by_width": {
                str(width): {
                    "exact": sum(
                        row["oracle_compiled_rollout"]["transition_exact"]
                        for row in rows
                        if int(row["case"]["width"]) == width
                    ),
                    "total": sum(
                        row["oracle_compiled_rollout"]["transition_total"]
                        for row in rows
                        if int(row["case"]["width"]) == width
                    ),
                }
                for width in widths
            },
        },
        "oracle_compiled_final_exact": {
            mode: sum(
                row["oracle_compiled_rollout"]["serializers"].get(mode, {}).get(
                    "exact", False
                )
                for row in rows
            )
            for mode in serializer_modes
        },
        "model_compiled_final_exact": {
            mode: sum(
                row["model_compiled_rollouts"][mode].get("final_exact", False)
                for row in rows
            )
            for mode in modes
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-new", type=int, default=96)
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    tokenizer_path = Path(args.tokenizer)
    if sha256_file(checkpoint_path) != EXPECTED_CHECKPOINT_SHA256:
        raise SystemExit("checkpoint hash mismatch")
    if sha256_file(tokenizer_path) != EXPECTED_TOKENIZER_SHA256:
        raise SystemExit("tokenizer hash mismatch")

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    checkpoint, model = load_model(checkpoint_path, device)
    rows = []
    for case in CASES:
        print(f"[drs-interface] case={case['id']}", flush=True)
        rows.append(probe_case(model, tokenizer, device, dict(case), args.max_new))

    result = {
        "audit": "fresh_drs_component_interfaces_v1",
        "claim_boundary": (
            "Fresh transcript diagnostic only. Few-shot prompting, gold construction, "
            "and post-hoc scoring are not training or an end-to-end reasoning claim."
        ),
        "inference_boundary": {
            "host_arithmetic_calls": 0,
            "host_state_repairs": 0,
            "host_state_replacements": 0,
            "host_roles": ["parse_model_state", "relay_model_state", "stop_on_model_z"],
            "gold_arithmetic_used_after_generation_for_scoring": True,
        },
        "script_sha256": sha256_file(Path(__file__)),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "checkpoint_step": checkpoint.get("step"),
        "tokenizer_sha256": EXPECTED_TOKENIZER_SHA256,
        "device": device,
        "decoding": {"temperature": 0.0, "max_new": args.max_new},
        "summary": summarize(rows),
        "rows": rows,
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
