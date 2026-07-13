#!/usr/bin/env python3
"""Evaluate source-free semantic readback from every held-out packet prefix.

This is intentionally separate from the final-answer LSA evaluator.  It asks
whether intermediate packets can be used directly by the language decoder,
which is the causal mechanism trained by ``causal_prefix_readback_memory``.
Normal packets must beat zeroed packets, shuffled source order, the equal-work
final-readback control, and shuffled-readback-label control before the result
can motivate a final-state/pair evaluation.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
import re
from pathlib import Path

import torch
from tokenizers import Tokenizer

from causal_prefix_readback import prefix_readback_targets, validate_readback_targets
from eval_source_dropping_memory import generate_from_packet
from model import GPT, GPTConfig
from source_dropping_memory import SourceDroppingMemory


MODES = ("normal", "zero", "shuffled")


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def last_integer(response: str):
    values = re.findall(r"(?<![A-Za-z0-9_])-?\d+(?![A-Za-z0-9_])", str(response))
    return int(values[-1]) if values else None


def load_rows(path: str):
    rows = []
    required = ("chunks", "initial", "operations", "keys", "state", "heldout", "eval_regime", "reference", "chunk_count")
    with open(path) as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if any(key not in row for key in required) or not row["heldout"]:
                raise ValueError("invalid held-out readback row at line {}".format(line_number))
            if not isinstance(row["chunks"], list) or not row["chunks"] or not all(isinstance(item, str) for item in row["chunks"]):
                raise ValueError("invalid source chunks at line {}".format(line_number))
            targets = prefix_readback_targets(row["initial"], row["operations"], row["keys"], row["state"])
            validate_readback_targets(targets, int(row["chunk_count"]))
            row["readbacks"] = targets
            rows.append(row)
    if not rows:
        raise ValueError("held-out readback data is empty")
    return rows


def select_rows(rows, per_chunk_regime: int, seed: int):
    if per_chunk_regime <= 0:
        raise ValueError("per-chunk-regime must be positive")
    grouped = collections.defaultdict(list)
    for row in rows:
        grouped[(str(row["eval_regime"]), int(row["chunk_count"]))].append(row)
    selected = []
    for key in sorted(grouped):
        candidates = grouped[key]
        if len(candidates) < per_chunk_regime:
            raise ValueError("{} has {}, need {} rows".format(key, len(candidates), per_chunk_regime))
        selected.extend(sorted(
            candidates,
            key=lambda row: hashlib.sha256((str(seed) + "\0" + row["reference"]).encode()).hexdigest(),
        )[:per_chunk_regime])
    return selected


def shuffled_chunks(chunks, reference: str, seed: int):
    order = list(range(len(chunks)))
    random.Random("{}\0{}".format(seed, reference)).shuffle(order)
    if len(order) > 1 and order == list(range(len(order))):
        order = order[1:] + order[:1]
    return [chunks[index] for index in order], order


def summarize(rows):
    def metric(items):
        if not items:
            raise ValueError("empty readback summary group")
        correct = sum(bool(item["correct"]) for item in items)
        return {"cases": len(items), "correct": correct, "accuracy": correct / len(items)}

    summary = {}
    for mode in sorted({row["mode"] for row in rows}):
        mode_rows = [row for row in rows if row["mode"] == mode]
        summary[mode] = metric(mode_rows)
        summary[mode]["by_regime"] = {
            regime: metric([row for row in mode_rows if row["eval_regime"] == regime])
            for regime in sorted({row["eval_regime"] for row in mode_rows})
        }
        summary[mode]["by_chunks"] = {
            str(chunk_count): metric([row for row in mode_rows if int(row["chunk_count"]) == chunk_count])
            for chunk_count in sorted({int(row["chunk_count"]) for row in mode_rows})
        }
        summary[mode]["by_prefix"] = {
            str(prefix_index): metric([row for row in mode_rows if int(row["prefix_index"]) == prefix_index])
            for prefix_index in sorted({int(row["prefix_index"]) for row in mode_rows})
        }
        summary[mode]["by_key"] = {
            key: metric([row for row in mode_rows if row["key"] == key])
            for key in sorted({row["key"] for row in mode_rows})
        }
    return summary


def check_checkpoint(checkpoint):
    memory = checkpoint.get("source_dropping_memory")
    readback = checkpoint.get("causal_prefix_readback")
    if not isinstance(memory, dict) or memory.get("source_present_at_decode") is not False:
        raise ValueError("checkpoint does not certify source-removed decoding")
    if not isinstance(readback, dict) or readback.get("decoder_readback_at_every_prefix") is not True:
        raise ValueError("checkpoint does not certify causal prefix readback")
    return memory, readback


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--per-chunk-regime", type=int, default=16)
    parser.add_argument("--all-heldout", action="store_true")
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--max-new", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--eos", default="<|endoftext|>")
    args = parser.parse_args()
    if args.max_new <= 0:
        raise SystemExit("max-new must be positive")
    output = Path(args.out)
    if output.exists():
        raise SystemExit("refusing existing output: {}".format(output))
    cases = load_rows(args.data)
    if not args.all_heldout:
        cases = select_rows(cases, args.per_chunk_regime, args.seed)
    if not torch.cuda.is_available():
        raise SystemExit("causal-prefix-readback evaluation requires a CUDA allocation")
    tokenizer = Tokenizer.from_file(args.tokenizer)
    eos_id = tokenizer.token_to_id(args.eos)
    if eos_id is None:
        raise SystemExit("tokenizer EOS token is missing")
    checkpoint = torch.load(args.ckpt, map_location="cpu")
    memory_metadata, readback_metadata = check_checkpoint(checkpoint)
    model = GPT(GPTConfig(**checkpoint["cfg"])).to("cuda").eval()
    model.load_state_dict(checkpoint["model"])
    memory = SourceDroppingMemory(model, int(memory_metadata["slots"]), int(memory_metadata["max_chunks"])).to("cuda").eval()
    missing, unexpected = memory.load_state_dict(checkpoint.get("memory_state", {}), strict=False)
    if unexpected or any(not name.startswith("model.") for name in missing):
        raise SystemExit("checkpoint has incompatible source-memory state")

    rows = []
    modes = list(dict.fromkeys(args.modes))
    for mode in modes:
        for case_index, case in enumerate(cases, 1):
            for target in case["readbacks"]:
                prefix_index = int(target["prefix_index"])
                source_chunks = list(case["chunks"][:prefix_index + 1])
                order = list(range(len(source_chunks)))
                if mode == "shuffled":
                    source_chunks, order = shuffled_chunks(
                        source_chunks, "{}:{}".format(case["reference"], prefix_index), args.seed,
                    )
                chunk_ids = tuple(
                    torch.tensor([tokenizer.encode(chunk).ids], dtype=torch.long, device="cuda")
                    for chunk in source_chunks
                )
                query_ids = torch.tensor([tokenizer.encode(target["query"]).ids], dtype=torch.long, device="cuda")
                packet = memory.encode(chunk_ids)
                if mode == "zero":
                    packet = torch.zeros_like(packet)
                generated = generate_from_packet(memory, packet, query_ids, eos_id, args.max_new)
                response = tokenizer.decode(generated.tolist(), skip_special_tokens=False)
                prediction = last_integer(response)
                expected = int(target["answer"])
                rows.append({
                    "mode": mode,
                    "reference": case["reference"],
                    "eval_regime": case["eval_regime"],
                    "chunk_count": int(case["chunk_count"]),
                    "prefix_index": prefix_index,
                    "key": target["key"],
                    "expected": expected,
                    "prediction": prediction,
                    "correct": prediction == expected,
                    "order": order,
                    "response": response,
                })
            if case_index % 25 == 0 or case_index == len(cases):
                correct = sum(bool(row["correct"]) for row in rows if row["mode"] == mode)
                print("[causal-prefix-eval] mode={} {}/{} rows={} correct={}".format(
                    mode, case_index, len(cases), len([row for row in rows if row["mode"] == mode]), correct,
                ), flush=True)
    result = {
        "audit": "causal_prefix_readback_heldout_v1",
        "checkpoint": args.ckpt,
        "checkpoint_step": checkpoint.get("step"),
        "checkpoint_memory_metadata": memory_metadata,
        "checkpoint_readback_metadata": readback_metadata,
        "data": args.data,
        "data_sha256": sha256_file(args.data),
        "per_chunk_regime": args.per_chunk_regime,
        "all_heldout": bool(args.all_heldout),
        "modes": modes,
        "seed": args.seed,
        "summary": summarize(rows),
        "rows": rows,
        "claim_boundary": (
            "This is a held-out source-free prefix readback test. It is narrow retained-state evidence only "
            "and cannot establish broad reasoning or alter flagship pretraining."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("[causal-prefix-eval] summary=" + json.dumps(result["summary"], sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
