#!/usr/bin/env python3
"""Run a small adaptive, transcript-first interaction with Shohin checkpoints.

This is a descriptive diagnostic, not a promotion board. It separates arithmetic,
serialization, state consumption, and self-review while preserving every prompt and
response. Later turns may quote a bounded prefix of an earlier response from the same
checkpoint; no answer is extracted or repaired by host code.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from tokenizers import Tokenizer

from eval_suite import generate
from model import GPT, GPTConfig


MAX_QUOTED_RESPONSE_CHARS = 400


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize(value: str) -> str:
    return re.sub(r"\s+", "", value.strip())


def bounded_quote(value: str) -> str:
    return value[:MAX_QUOTED_RESPONSE_CHARS].replace("\x00", "")


def score_response(
    kind: str, expected: dict[str, Any], response: str
) -> dict[str, Any]:
    stripped = response.strip()
    first_line = stripped.splitlines()[0].strip() if stripped else ""
    expected_text = str(expected["text"])
    parsed: dict[str, Any] | None = None

    if kind == "integer":
        match = re.match(r"\s*(-?\d+)\b", response)
        if match:
            parsed = {"value": int(match.group(1))}
    elif kind == "assignment":
        field = str(expected["field"])
        match = re.search(
            rf"(?<![A-Za-z0-9_]){re.escape(field)}\s*=\s*(-?\d+)", response
        )
        if match:
            parsed = {field: int(match.group(1))}
    elif kind == "digit_packet":
        digit = re.search(r"(?<![A-Za-z0-9_])digit\s*=\s*(-?\d+)", response)
        carry = re.search(r"(?<![A-Za-z0-9_])carry\s*=\s*(-?\d+)", response)
        if digit and carry:
            parsed = {"digit": int(digit.group(1)), "carry": int(carry.group(1))}
    elif kind == "memo":
        match = re.search(
            r"memo\s*\{\s*r\s*=\s*(-?\d+)\s*;\s*seal\s*=\s*([A-Za-z]+)\s*\}", response
        )
        if match:
            parsed = {"r": int(match.group(1)), "seal": match.group(2)}
    else:
        raise ValueError(f"unsupported score kind: {kind}")

    expected_state = expected["state"]
    return {
        "strict_exact": normalize(stripped) == normalize(expected_text),
        "first_line_exact": normalize(first_line) == normalize(expected_text),
        "semantic_correct": parsed == expected_state,
        "parsed_state": parsed,
    }


def probe_definitions() -> list[dict[str, Any]]:
    return [
        {
            "id": "scalar_plain",
            "mechanism": "compute",
            "kind": "integer",
            "expected": {"text": "85", "state": {"value": 85}},
            "prompt": lambda _: "What is 58 + 27? Reply with only the integer.",
        },
        {
            "id": "scalar_review",
            "mechanism": "self_review",
            "kind": "integer",
            "expected": {"text": "85", "state": {"value": 85}},
            "prompt": lambda rows: (
                "A previous attempt answered the question 'What is 58 + 27?' as follows:\n"
                f"{bounded_quote(rows['scalar_plain']['response'])}\n"
                "Check it independently and reply with only the correct integer."
            ),
        },
        {
            "id": "scalar_serialize_gold",
            "mechanism": "serialize_gold_state",
            "kind": "assignment",
            "expected": {"text": "quill=85", "field": "quill", "state": {"quill": 85}},
            "prompt": lambda _: (
                "The verified value of quill is 85. Reply exactly quill=85."
            ),
        },
        {
            "id": "scalar_serialize_model",
            "mechanism": "serialize_model_state",
            "kind": "assignment",
            "expected": {"text": "quill=85", "field": "quill", "state": {"quill": 85}},
            "prompt": lambda rows: (
                "The model's previous computation of 58 + 27 was:\n"
                f"{bounded_quote(rows['scalar_plain']['response'])}\n"
                "Use that computation and reply exactly quill=<integer>."
            ),
        },
        {
            "id": "digit_sum_plain",
            "mechanism": "compute_local_column",
            "kind": "integer",
            "expected": {"text": "13", "state": {"value": 13}},
            "prompt": lambda _: "Compute 6 + 7 + 0. Reply with only the integer.",
        },
        {
            "id": "digit_packet_model",
            "mechanism": "packetize_model_state",
            "kind": "digit_packet",
            "expected": {
                "text": "digit=3;carry=1",
                "state": {"digit": 3, "carry": 1},
            },
            "prompt": lambda rows: (
                "The model's previous rightmost-column total was:\n"
                f"{bounded_quote(rows['digit_sum_plain']['response'])}\n"
                "Write its ones digit and carry. Reply exactly digit=<integer>;carry=<integer>."
            ),
        },
        {
            "id": "memo_copy_gold",
            "mechanism": "copy_packet",
            "kind": "memo",
            "expected": {
                "text": "memo{r=225;seal=KITE}",
                "state": {"r": 225, "seal": "KITE"},
            },
            "prompt": lambda _: (
                "Copy this trusted state without changing it: memo{r=225;seal=KITE}. "
                "Reply with only the copied state."
            ),
        },
        {
            "id": "memo_fused_gold",
            "mechanism": "consume_gold_packet_two_steps",
            "kind": "assignment",
            "expected": {"text": "r=194", "field": "r", "state": {"r": 194}},
            "prompt": lambda _: (
                "Trusted state: memo{r=225;seal=KITE}. Replace r with r minus 37, then add 6. "
                "Preserve the seal internally and reply exactly r=<integer>."
            ),
        },
        {
            "id": "memo_step_one",
            "mechanism": "consume_gold_packet_one_step",
            "kind": "assignment",
            "expected": {"text": "r=188", "field": "r", "state": {"r": 188}},
            "prompt": lambda _: (
                "Trusted state: memo{r=225;seal=KITE}. Apply only r = r - 37. "
                "Reply exactly r=<integer>."
            ),
        },
        {
            "id": "memo_step_two_model",
            "mechanism": "consume_model_packet_next_step",
            "kind": "assignment",
            "expected": {"text": "r=194", "field": "r", "state": {"r": 194}},
            "prompt": lambda rows: (
                "The previous updated register was:\n"
                f"{bounded_quote(rows['memo_step_one']['response'])}\n"
                "Now apply only r = r + 6. Reply exactly r=<integer>."
            ),
        },
    ]


def parse_model_spec(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise ValueError("--model must be NAME=PATH")
    name, raw_path = spec.split("=", 1)
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", name):
        raise ValueError(f"invalid model name: {name}")
    path = Path(raw_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return name, path


def load_model(path: Path, device: str) -> tuple[dict[str, Any], GPT]:
    checkpoint = torch.load(path, map_location="cpu")
    model = GPT(GPTConfig(**checkpoint["cfg"])).to(device).eval()
    model.load_state_dict(checkpoint["model"], strict=True)
    return checkpoint, model


def run_model(
    name: str,
    path: Path,
    tokenizer: Tokenizer,
    device: str,
    max_new: int,
) -> dict[str, Any]:
    checkpoint, model = load_model(path, device)
    completed: dict[str, dict[str, Any]] = {}
    for definition in probe_definitions():
        prompt_builder: Callable[[dict[str, dict[str, Any]]], str] = definition[
            "prompt"
        ]
        prompt = prompt_builder(completed)
        response = generate(
            model,
            tokenizer,
            prompt,
            device,
            max_new=max_new,
            temp=0.0,
            skip_special_tokens=False,
        )
        row = {key: value for key, value in definition.items() if key not in {"prompt"}}
        row.update(
            {
                "prompt": prompt,
                "response": response,
                "score": score_response(
                    str(definition["kind"]), definition["expected"], response
                ),
            }
        )
        completed[str(definition["id"])] = row
        print(
            f"[adaptive-interaction] model={name} turn={definition['id']} "
            f"semantic={row['score']['semantic_correct']} "
            f"strict={row['score']['strict_exact']}",
            flush=True,
        )

    rows = list(completed.values())
    by_mechanism = {
        row["mechanism"]: bool(row["score"]["semantic_correct"]) for row in rows
    }
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return {
        "name": name,
        "checkpoint": str(path),
        "checkpoint_sha256": sha256_file(path),
        "checkpoint_step": checkpoint.get("step"),
        "summary": {
            "semantic_correct": sum(row["score"]["semantic_correct"] for row in rows),
            "first_line_exact": sum(row["score"]["first_line_exact"] for row in rows),
            "strict_exact": sum(row["score"]["strict_exact"] for row in rows),
            "turns": len(rows),
            "by_mechanism": by_mechanism,
        },
        "rows": rows,
    }


def write_json_no_replace(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as target:
            target.write(encoded)
            target.flush()
            os.fsync(target.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", required=True, help="NAME=PATH")
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument(
        "--device", choices=("auto", "cuda", "mps", "cpu"), default="auto"
    )
    parser.add_argument("--max-new", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260717)
    args = parser.parse_args()
    if args.max_new <= 0:
        raise SystemExit("--max-new must be positive")

    specs = [parse_model_spec(spec) for spec in args.model]
    names = [name for name, _ in specs]
    if len(names) != len(set(names)):
        raise SystemExit("model names must be unique")
    tokenizer_path = Path(args.tokenizer).resolve()
    if not tokenizer_path.is_file():
        raise FileNotFoundError(tokenizer_path)
    device = resolve_device(args.device)
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but unavailable")
    torch.manual_seed(args.seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    payload = {
        "audit": "researcher_adaptive_interaction_v1",
        "claim_boundary": (
            "Adaptive descriptive interaction only. This run may localize compute, "
            "serialization, review, and state-consumption failures, but cannot promote "
            "a checkpoint or architecture."
        ),
        "source_commit": args.source_commit,
        "source_sha256": sha256_file(Path(__file__).resolve()),
        "tokenizer": str(tokenizer_path),
        "tokenizer_sha256": sha256_file(tokenizer_path),
        "device": device,
        "seed": args.seed,
        "generation": {"temperature": 0.0, "max_new_tokens": args.max_new},
        "models": [
            run_model(name, path, tokenizer, device, args.max_new)
            for name, path in specs
        ],
    }
    write_json_no_replace(Path(args.out).resolve(), payload)


if __name__ == "__main__":
    main()
