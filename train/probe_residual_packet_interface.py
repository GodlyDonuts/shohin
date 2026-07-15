#!/usr/bin/env python3
"""Exploratory raw-model probe for the native residual-packet interface.

This is not a claim-bearing evaluation. It preserves the exact pre-SFT prompts,
responses, and token accounting used to diagnose compiler, updater, and halt
behavior before RSP-C1 is allowed to train.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch
from tokenizers import Tokenizer

from eval_scheduled_reasoning_confirmation import greedy_completion
from model import GPT, GPTConfig


SCHEMA = "raw_residual_packet_interface_probe_v1"
PROBES = (
    {
        "id": "compile_add_multiply_subtract",
        "kind": "compiler",
        "prompt": (
            "Problem: Start at 19, add 6, multiply by 3, then subtract 11.\n"
            "Write only a compact packet with the current State and remaining Plan.\n"
            "Packet:"
        ),
    },
    {
        "id": "compile_multiply_subtract_add",
        "kind": "compiler",
        "prompt": (
            "Problem: Begin with 27. Multiply by 4, subtract 13, then add 8.\n"
            "Write only a compact packet with the current State and remaining Plan.\n"
            "Packet:"
        ),
    },
    {
        "id": "update_after_add",
        "kind": "updater",
        "prompt": (
            "Packet:\nState: 19\nPlan: add 6; multiply 3; subtract 11\n"
            "Observed result: 25\nWrite only the next packet.\nPacket:"
        ),
    },
    {
        "id": "update_after_multiply",
        "kind": "updater",
        "prompt": (
            "Packet:\nState: 25\nPlan: multiply 3; subtract 11\n"
            "Observed result: 75\nWrite only the next packet.\nPacket:"
        ),
    },
    {
        "id": "halt_after_subtract",
        "kind": "halt",
        "prompt": (
            "Packet:\nState: 75\nPlan: subtract 11\nObserved result: 64\n"
            "Write only the final answer.\nAnswer:"
        ),
    },
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_immutable_json(path: str | Path, value: dict) -> str:
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("ascii")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    with os.fdopen(descriptor, "wb") as sink:
        sink.write(payload)
        sink.flush()
        os.fsync(sink.fileno())
        os.fchmod(sink.fileno(), 0o444)
    if destination.stat().st_mode & 0o222:
        raise PermissionError("probe output remained writable")
    return hashlib.sha256(payload).hexdigest()


def load_model(path: str, device: str):
    checkpoint = torch.load(path, map_location="cpu")
    model = GPT(GPTConfig(**checkpoint["cfg"])).to(device).eval()
    model.load_state_dict(checkpoint["model"])
    return checkpoint, model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-new", type=int, default=64)
    args = parser.parse_args()
    if args.max_new <= 0:
        raise SystemExit("--max-new must be positive")
    if Path(args.out).exists():
        raise FileExistsError("refusing to overwrite probe output")

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    checkpoint, model = load_model(args.ckpt, device)
    tokenizer = Tokenizer.from_file(args.tokenizer)
    rows = []
    for probe in PROBES:
        completion = greedy_completion(
            model, tokenizer, probe["prompt"], device, args.max_new
        )
        rows.append({**probe, **completion})
        print(f"[packet-probe] {probe['id']}", flush=True)

    result = {
        "schema": SCHEMA,
        "claim_status": "exploratory_pre_sft_interface_diagnostic",
        "checkpoint": args.ckpt,
        "checkpoint_step": checkpoint.get("step"),
        "checkpoint_sha256": sha256_file(args.ckpt),
        "tokenizer": args.tokenizer,
        "tokenizer_sha256": sha256_file(args.tokenizer),
        "device": device,
        "max_new": args.max_new,
        "model_calls": len(rows),
        "prompt_token_count": sum(row["prompt_token_count"] for row in rows),
        "sampled_token_count": sum(row["sampled_token_count"] for row in rows),
        "decoded_token_count": sum(row["decoded_token_count"] for row in rows),
        "rows": rows,
    }
    digest = write_immutable_json(args.out, result)
    print(json.dumps({"out": args.out, "sha256": digest}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
