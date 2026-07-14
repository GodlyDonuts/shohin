#!/usr/bin/env python3
"""Fast held-out causal screen for a recurrent scratch adapter.

This uses teacher-forced answer NLL and exact answer-token sequences.  It is a
mechanics/learning gate before expensive autoregressive evaluation, not a
reasoning score.  Every condition keeps the original source prompt visible.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
from pathlib import Path

import torch
from tokenizers import Tokenizer

from causal_recurrent_scratch import CausalRecurrentScratch
from eval_latent_operator import load_rows, regime_of, select_rows
from latent_rollout_train import make_batch
from model import GPT, GPTConfig


def condition_specs(training_mode, trained_depth):
    if training_mode not in {"recurrent", "reset"}:
        raise ValueError("unknown adapter training mode {}".format(training_mode))
    trained_recurrent = training_mode == "recurrent"
    return (
        ("disabled", "disabled", trained_recurrent, 1),
        ("normal_t1", "normal", True, 1),
        ("normal_trained_depth", "normal", trained_recurrent, trained_depth),
        ("reset_trained_depth", "normal", False, trained_depth),
        ("zero_trained_depth", "zero", trained_recurrent, trained_depth),
        ("shuffled_trained_depth", "shuffled", trained_recurrent, trained_depth),
    )


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_examples(rows, tokenizer, eos_id):
    examples = []
    for index, row in enumerate(rows):
        prompt = tokenizer.encode(row["question"]).ids
        answer = tokenizer.encode(" " + row["response"].strip()).ids
        if not prompt or not answer:
            raise ValueError("empty held-out tokenization at row {}".format(index))
        examples.append({
            "prompt": prompt,
            "answer": answer,
            "depth": int(row["depth"]),
            "family": row["family"],
            "regime": regime_of(row),
            "line": index + 1,
        })
    return examples


def all_shape_batches(examples, batch_size):
    """Keep every held-out row without introducing padding tokens."""
    grouped = collections.defaultdict(list)
    for index, example in enumerate(examples):
        grouped[(len(example["prompt"]), len(example["answer"]))].append(index)
    batches = []
    for shape in sorted(grouped):
        indices = grouped[shape]
        batches.extend(indices[offset:offset + batch_size] for offset in range(0, len(indices), batch_size))
    return batches, {
        "buckets": len(grouped),
        "batches": len(batches),
        "examples": sum(len(batch) for batch in batches),
        "partial_batches": sum(len(batch) < batch_size for batch in batches),
    }


def batch_metrics(logits, targets):
    mask = targets.ne(-1)
    losses = torch.nn.functional.cross_entropy(
        logits.float().reshape(-1, logits.shape[-1]), targets.reshape(-1),
        ignore_index=-1, reduction="none",
    ).reshape_as(targets)
    predictions = logits.argmax(dim=-1)
    token_correct = predictions.eq(targets) & mask
    sequence_correct = (token_correct | ~mask).all(dim=1)
    return {
        "nll_sum": float((losses * mask).sum().item()),
        "tokens": int(mask.sum().item()),
        "token_correct": int(token_correct.sum().item()),
        "sequence_correct": sequence_correct.detach().cpu().tolist(),
    }


def summarize(records):
    output = {}
    for condition in sorted({record["condition"] for record in records}):
        matching = [record for record in records if record["condition"] == condition]
        regimes = {}
        for regime in ["all"] + sorted({record["regime"] for record in matching}):
            selected = matching if regime == "all" else [record for record in matching if record["regime"] == regime]
            nll_sum = sum(record["nll_sum"] for record in selected)
            tokens = sum(record["tokens"] for record in selected)
            token_correct = sum(record["token_correct"] for record in selected)
            sequence_correct = sum(record["sequence_correct"] for record in selected)
            regimes[regime] = {
                "cases": len(selected),
                "tokens": tokens,
                "mean_nll": nll_sum / tokens,
                "token_accuracy": token_correct / tokens,
                "sequence_correct": sequence_correct,
                "sequence_accuracy": sequence_correct / len(selected),
            }
        output[condition] = regimes
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--per-depth", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--eos", default="<|endoftext|>")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("scratch evaluation requires a CUDA allocation")
    if Path(args.out).exists():
        raise SystemExit("refusing existing output: {}".format(args.out))
    checkpoint = torch.load(args.adapter, map_location="cpu")
    metadata = checkpoint.get("causal_recurrent_scratch", {})
    if metadata.get("protocol") != "causal_recurrent_scratch_v1":
        raise SystemExit("adapter metadata is missing or incompatible")
    if metadata.get("base_sha256") != sha256_file(args.base):
        raise SystemExit("adapter does not bind the supplied base checkpoint")

    tokenizer = Tokenizer.from_file(args.tokenizer)
    eos_id = tokenizer.token_to_id(args.eos)
    if eos_id is None:
        raise SystemExit("tokenizer EOS token is missing")
    selected = select_rows(load_rows(args.data), args.per_depth, args.seed)
    examples = load_examples(selected, tokenizer, eos_id)
    batches, batch_report = all_shape_batches(examples, args.batch_size)

    base_checkpoint = torch.load(args.base, map_location="cpu")
    model = GPT(GPTConfig(**base_checkpoint["cfg"])).to("cuda").eval()
    model.load_state_dict(base_checkpoint["model"])
    adapter = CausalRecurrentScratch(
        model,
        layer=int(metadata["layer"]),
        slots=int(metadata["slots"]),
        width=int(metadata["width"]),
        workspace_topk=int(metadata.get("workspace_topk", 0)),
        workspace_temperature=float(metadata.get("workspace_temperature", 0.2)),
    ).to("cuda").eval()
    missing, unexpected = adapter.load_state_dict(checkpoint["adapter_state"], strict=False)
    missing = [name for name in missing if not name.startswith("model.")]
    unexpected = [name for name in unexpected if not name.startswith("model.")]
    if missing or unexpected:
        raise SystemExit("adapter state mismatch missing={} unexpected={}".format(missing, unexpected))

    trained_depth = int(metadata["steps"])
    conditions = condition_specs(metadata.get("mode"), trained_depth)
    records = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for batch_number, indices in enumerate(batches, 1):
            prompts, answers = make_batch(examples, indices, "cuda")
            for condition, state_mode, recurrent, actual_steps in conditions:
                state_override = None
                actual_mode = state_mode
                if state_mode == "shuffled" and len(indices) == 1:
                    donor_index = (indices[0] + 1) % len(examples)
                    donor_ids = torch.tensor(
                        [examples[donor_index]["prompt"]], dtype=torch.long, device="cuda",
                    )
                    state_override = adapter.encode_prompt(donor_ids, actual_steps, recurrent=True)
                    actual_mode = "override"
                logits, _, _, targets = adapter.supervised_loss(
                    prompts, answers, eos_id, actual_steps,
                    state_mode=actual_mode, recurrent=recurrent, state_override=state_override,
                )
                metrics = batch_metrics(logits, targets)
                for local, index in enumerate(indices):
                    # Split aggregate token accounting proportionally by
                    # scoring the single sequence. Exact-shape batches make
                    # this deterministic and keep reports auditable.
                    one = batch_metrics(logits[local:local + 1], targets[local:local + 1])
                    records.append({
                        "condition": condition,
                        "regime": examples[index]["regime"],
                        "depth": examples[index]["depth"],
                        "family": examples[index]["family"],
                        "nll_sum": one["nll_sum"],
                        "tokens": one["tokens"],
                        "token_correct": one["token_correct"],
                        "sequence_correct": bool(one["sequence_correct"][0]),
                    })
            if batch_number % 20 == 0 or batch_number == len(batches):
                print("[scratch-nll] {}/{} batches".format(batch_number, len(batches)), flush=True)

    result = {
        "audit": "causal_recurrent_scratch_nll_v1",
        "base": os.path.realpath(args.base),
        "base_sha256": sha256_file(args.base),
        "adapter": os.path.realpath(args.adapter),
        "adapter_sha256": sha256_file(args.adapter),
        "adapter_metadata": metadata,
        "data": os.path.realpath(args.data),
        "data_sha256": sha256_file(args.data),
        "per_depth": args.per_depth,
        "seed": args.seed,
        "batch_report": batch_report,
        "summary": summarize(records),
        "records": records,
        "claim_boundary": "Teacher-forced causal screen only; autoregressive evidence is still required.",
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("[scratch-nll] summary=" + json.dumps(result["summary"], sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
