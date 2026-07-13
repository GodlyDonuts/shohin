#!/usr/bin/env python3
"""Evaluate fixed-slot source memory after the source tokens have been dropped.

The normal condition writes the serialized record into the continuous packet.
The zero and shuffled-packet conditions are required ablations: they test
whether success depends on retained source information rather than query-format
priors. This is a narrow held-out memory test, not a general-reasoning score.
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

from model import GPT, GPTConfig
from source_dropping_memory import SourceDroppingMemory


FINAL = re.compile(r"the\s+answer\s+is\s*(-?\d+)\b", re.IGNORECASE)
MODES = ("normal", "zero", "shuffled")


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_rows(path: str):
    rows = []
    required = ("chunks", "query", "response", "answer", "chunk_count", "heldout", "eval_regime", "reference")
    with open(path) as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if any(key not in row for key in required) or not row["heldout"]:
                raise ValueError("invalid held-out source-memory row at line {}".format(line_number))
            if not isinstance(row["chunks"], list) or not row["chunks"] or not all(isinstance(chunk, str) for chunk in row["chunks"]):
                raise ValueError("invalid source chunks at line {}".format(line_number))
            rows.append(row)
    if not rows:
        raise ValueError("held-out source-memory data is empty")
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


def select_counterfactual_pairs(rows, pairs_per_regime: int, seed: int):
    """Select complete, deterministic paired interventions for every regime."""
    if pairs_per_regime <= 0:
        return []
    grouped = collections.defaultdict(list)
    for row in rows:
        pair_id = row.get("counterfactual_id")
        if pair_id:
            grouped[(str(row["eval_regime"]), str(pair_id))].append(row)
    by_regime = collections.defaultdict(list)
    for (regime, pair_id), pair in grouped.items():
        if len(pair) == 2 and {item.get("counterfactual_variant") for item in pair} == {"a", "b"}:
            by_regime[regime].append((pair_id, pair))
    selected = []
    expected_regimes = sorted({str(row["eval_regime"]) for row in rows})
    for regime in expected_regimes:
        candidates = by_regime[regime]
        if len(candidates) < pairs_per_regime:
            raise ValueError(
                "{} has {} complete counterfactual pairs, need {}".format(
                    regime, len(candidates), pairs_per_regime,
                )
            )
        candidates.sort(
            key=lambda item: hashlib.sha256((str(seed) + "\0" + item[0]).encode()).hexdigest(),
        )
        for _, pair in candidates[:pairs_per_regime]:
            selected.extend(sorted(pair, key=lambda item: item["counterfactual_variant"]))
    return selected


def final_answer(response: str):
    matches = FINAL.findall(str(response))
    return int(matches[-1]) if matches else None


def shuffled_chunks(chunks, reference: str, seed: int):
    """Deterministically destroy temporal order without exposing the source later."""
    order = list(range(len(chunks)))
    random.Random("{}\0{}".format(seed, reference)).shuffle(order)
    if len(order) > 1 and order == list(range(len(order))):
        order = order[1:] + order[:1]
    return [chunks[index] for index in order], order


@torch.no_grad()
def generate_from_packet(memory, packet, query_ids, eos_id: int, max_new: int):
    context = memory.answer_context(packet, query_ids)
    generated = []
    for _ in range(max_new):
        logits, _ = memory.model.forward_embeds(context)
        next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        token = int(next_id.item())
        if token == int(eos_id):
            break
        generated.append(token)
        if context.shape[1] >= memory.model.cfg.seq_len:
            break
        context = torch.cat((context, memory.model.tok(next_id)), dim=1)
    return torch.tensor(generated, dtype=torch.long, device=query_ids.device)


def summarize(rows):
    def metric(items):
        correct = sum(bool(row["correct"]) for row in items)
        return {"cases": len(items), "correct": correct, "accuracy": correct / len(items)}

    def counterfactual_metric(items):
        pairs = collections.defaultdict(list)
        for item in items:
            if item.get("counterfactual_id"):
                pairs[item["counterfactual_id"]].append(item)
        complete = [
            pair for pair in pairs.values()
            if len(pair) == 2 and {item.get("counterfactual_variant") for item in pair} == {"a", "b"}
        ]
        correct = sum(
            all(bool(item["correct"]) for item in pair)
            and len({item.get("prediction") for item in pair}) == 2
            for pair in complete
        )
        return {"pairs": len(complete), "correct": correct, "accuracy": correct / len(complete) if complete else None}

    summary = {}
    for mode in sorted({row["mode"] for row in rows}):
        mode_rows = [row for row in rows if row["mode"] == mode]
        summary[mode] = metric(mode_rows)
        summary[mode]["by_regime"] = {
            regime: metric([row for row in mode_rows if row["eval_regime"] == regime])
            for regime in sorted({row["eval_regime"] for row in mode_rows})
        }
        summary[mode]["by_chunks"] = {
            str(count): metric([row for row in mode_rows if int(row["chunk_count"]) == count])
            for count in sorted({int(row["chunk_count"]) for row in mode_rows})
        }
        query_kinds = sorted({row.get("query_kind") for row in mode_rows if row.get("query_kind")})
        if query_kinds:
            summary[mode]["by_query_kind"] = {
                kind: metric([row for row in mode_rows if row.get("query_kind") == kind])
                for kind in query_kinds
            }
        ledger_stages = sorted({int(row["ledger_stage"]) for row in mode_rows if row.get("ledger_stage") is not None})
        if ledger_stages:
            summary[mode]["by_ledger_stage"] = {
                str(stage): metric([row for row in mode_rows if int(row.get("ledger_stage", -1)) == stage])
                for stage in ledger_stages
            }
        counterfactual = counterfactual_metric(mode_rows)
        if counterfactual["pairs"]:
            summary[mode]["counterfactual_pairs"] = counterfactual
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--per-chunk-regime", type=int, default=128)
    parser.add_argument("--all-heldout", action="store_true", help="score every held-out row, preserving all counterfactual pairs")
    parser.add_argument(
        "--counterfactual-pairs-per-regime", type=int, default=0,
        help="add this many complete paired interventions per regime to the balanced screen",
    )
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--max-new", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--eos", default="<|endoftext|>")
    args = parser.parse_args()
    if args.max_new <= 0 or args.counterfactual_pairs_per_regime < 0:
        raise SystemExit("max-new must be positive and counterfactual-pairs-per-regime cannot be negative")
    if args.all_heldout and args.counterfactual_pairs_per_regime:
        raise SystemExit("all-heldout already preserves every counterfactual pair")
    modes = list(dict.fromkeys(args.modes))
    out = Path(args.out)
    if out.exists():
        raise SystemExit("refusing existing output: {}".format(out))

    loaded_cases = load_rows(args.data)
    cases = loaded_cases if args.all_heldout else select_rows(loaded_cases, args.per_chunk_regime, args.seed)
    if args.counterfactual_pairs_per_regime:
        selected = {row["reference"]: row for row in cases}
        for row in select_counterfactual_pairs(loaded_cases, args.counterfactual_pairs_per_regime, args.seed):
            selected[row["reference"]] = row
        cases = [selected[reference] for reference in sorted(selected)]
    if not torch.cuda.is_available():
        raise SystemExit("source-memory evaluation requires a CUDA allocation")
    tokenizer = Tokenizer.from_file(args.tokenizer)
    eos_id = tokenizer.token_to_id(args.eos)
    if eos_id is None:
        raise SystemExit("tokenizer EOS token is missing")
    checkpoint = torch.load(args.ckpt, map_location="cpu")
    metadata = checkpoint.get("source_dropping_memory")
    if not isinstance(metadata, dict) or metadata.get("source_present_at_decode") is not False:
        raise SystemExit("checkpoint does not certify source-removed decoding")
    model = GPT(GPTConfig(**checkpoint["cfg"])).to("cuda").eval()
    model.load_state_dict(checkpoint["model"])
    memory = SourceDroppingMemory(model, int(metadata["slots"]), int(metadata["max_chunks"])).to("cuda").eval()
    missing, unexpected = memory.load_state_dict(checkpoint.get("memory_state", {}), strict=False)
    if unexpected or any(not name.startswith("model.") for name in missing):
        raise SystemExit("source-memory checkpoint has incompatible packet state")

    rows = []
    for mode in modes:
        for index, case in enumerate(cases, 1):
            source_chunks = case["chunks"]
            order = list(range(len(source_chunks)))
            if mode == "shuffled":
                source_chunks, order = shuffled_chunks(source_chunks, case["reference"], args.seed)
            chunk_ids = tuple(
                torch.tensor([tokenizer.encode(chunk).ids], dtype=torch.long, device="cuda")
                for chunk in source_chunks
            )
            query_ids = torch.tensor([tokenizer.encode(case["query"]).ids], dtype=torch.long, device="cuda")
            packet = memory.encode(chunk_ids)
            if mode == "zero":
                packet = torch.zeros_like(packet)
            generated = generate_from_packet(memory, packet, query_ids, eos_id, args.max_new)
            response = tokenizer.decode(generated.tolist(), skip_special_tokens=False)
            prediction = final_answer(response)
            rows.append({
                "mode": mode,
                "eval_regime": case["eval_regime"],
                "chunk_count": int(case["chunk_count"]),
                "query_kind": case.get("ledger_probe_kind") or case.get("query_spec", {}).get("kind"),
                "ledger_stage": case.get("ledger_stage"),
                "counterfactual_id": case.get("counterfactual_id"),
                "counterfactual_variant": case.get("counterfactual_variant"),
                "reference": case["reference"],
                "expected": int(case["answer"]),
                "prediction": prediction,
                "correct": prediction == int(case["answer"]),
                "order": order,
                "response": response,
            })
            if index % 25 == 0 or index == len(cases):
                correct = sum(bool(row["correct"]) for row in rows if row["mode"] == mode)
                print("[source-memory-eval] mode={} {}/{} correct={}".format(mode, index, len(cases), correct), flush=True)
    result = {
        "audit": "source_dropping_memory_heldout_v1",
        "checkpoint": args.ckpt,
        "checkpoint_step": checkpoint.get("step"),
        "checkpoint_memory_metadata": metadata,
        "data": args.data,
        "data_sha256": sha256_file(args.data),
        "per_chunk_regime": args.per_chunk_regime,
        "all_heldout": bool(args.all_heldout),
        "counterfactual_pairs_per_regime": args.counterfactual_pairs_per_regime,
        "modes": modes,
        "seed": args.seed,
        "summary": summarize(rows),
        "rows": rows,
        "claim_boundary": (
            "This is a held-out source-removal memory test. Normal must exceed zero and shuffled "
            "ablations before it supports narrow retained-information evidence; it is not a broad reasoning claim."
        ),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("[source-memory-eval] summary=" + json.dumps(result["summary"], sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
