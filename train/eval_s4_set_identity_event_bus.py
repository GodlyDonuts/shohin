#!/usr/bin/env python3
"""Evaluate frozen S4 v3 once on its fresh development board."""

from __future__ import annotations

import argparse
import collections
import json
import os

import torch
from tokenizers import Tokenizer

from model import GPT, GPTConfig
from s4_set_identity_event_bus import (
    SetIdentityEventBus,
    decode_set_identity_example,
    roster_recovery_exact,
)
from self_delimiting_event_tape import (
    adapter_hash,
    load_examples,
    make_batches,
    pad_batch,
    sha256_file,
)


FIELDS = (
    "valid", "count_exact", "program_exact", "query_exact", "state_exact",
    "answer_correct", "initial_exact",
)


def summarize(records):
    result = {"rows": len(records)}
    for field in FIELDS:
        correct = sum(bool(record[field]) for record in records)
        result[field] = {"correct": correct, "accuracy": correct / max(1, len(records))}
    result["mean_predicted_event_count"] = sum(
        record["predicted_event_count"] for record in records
    ) / max(1, len(records))
    return result


def grouped(records, key):
    values = collections.defaultdict(list)
    for record in records:
        values[str(record[key])].append(record)
    return {name: summarize(group) for name, group in sorted(values.items())}


def record(example, decoded, initial_exact):
    is_valid = bool(decoded.get("valid"))
    return {
        "id": example.row_id,
        "depth": example.depth,
        "surface_type": example.surface_type,
        "valid": is_valid,
        "predicted_event_count": int(decoded.get("event_count", 0)),
        "failure_reason": decoded.get("failure_reason", "unknown"),
        "count_exact": int(decoded.get("event_count", 0)) == example.depth,
        "program_exact": is_valid and decoded.get("program") == example.program,
        "query_exact": is_valid and decoded.get("query") == example.query_target,
        "state_exact": is_valid and decoded.get("final_state") == example.final_state,
        "answer_correct": is_valid and decoded.get("answer_identity") == example.answer_identity,
        "initial_exact": bool(initial_exact),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--parser", required=True)
    parser.add_argument("--s3-executor", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("S4 v3 evaluation requires CUDA")
    if os.path.exists(args.out):
        raise SystemExit("refusing existing S4 v3 evaluation")
    report = json.load(open(args.report))
    if report.get("schema") != "r12_s4_set_identity_fresh_development_report_v1":
        raise SystemExit("invalid S4 v3 development report")
    if not report.get("all_gates_pass"):
        raise SystemExit("S4 v3 development gates failed")
    if report["artifacts"]["development"]["sha256"] != sha256_file(args.data):
        raise SystemExit("S4 v3 report does not bind development data")
    if report.get("confirmation_access") != 0:
        raise SystemExit("S4 v3 report accessed confirmation")

    bundle = torch.load(args.parser, map_location="cpu")
    metadata = bundle.get("parser", {})
    if metadata.get("protocol") not in {
        "r12_s4_set_identity_event_bus_treatment_v3",
        "r12_s4_set_identity_event_bus_shuffled_v3",
    }:
        raise SystemExit("invalid S4 v3 parser protocol")
    if metadata.get("base_sha256") != sha256_file(args.base):
        raise SystemExit("S4 v3 parser/base mismatch")
    s3 = torch.load(args.s3_executor, map_location="cpu").get("executor", {})
    if s3.get("protocol") != "r12_s3_equivariant_permutation_executor_v1_1":
        raise SystemExit("invalid locked S3 executor")
    if s3.get("base_sha256") != metadata.get("base_sha256"):
        raise SystemExit("S3/base mismatch")

    tokenizer = Tokenizer.from_file(args.tokenizer)
    checkpoint = torch.load(args.base, map_location="cpu")
    cfg = GPTConfig(**checkpoint["cfg"])
    examples = load_examples(
        args.data, tokenizer, "s4_event_tape_development", cfg.seq_len,
    )
    model = GPT(cfg).to("cuda").eval()
    model.load_state_dict(checkpoint["model"])
    event_bus = SetIdentityEventBus(
        model,
        layer=int(metadata["layer"]),
        width=int(metadata["width"]),
        heads=int(metadata["heads"]),
        encoder_layers=int(metadata["encoder_layers"]),
        ff=int(metadata["ff"]),
    ).to("cuda").eval()
    missing, unexpected = event_bus.load_state_dict(bundle["adapter_state"], strict=False)
    if unexpected or any(not name.startswith("model.") for name in missing):
        raise SystemExit("S4 v3 adapter mismatch")
    if adapter_hash(event_bus) != metadata["final_adapter_sha256"]:
        raise SystemExit("S4 v3 adapter hash mismatch")

    records = []
    deranged = []
    batches = make_batches(examples, args.batch_size, seed=0, shuffle=False)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for indices in batches:
            selected, ids, valid, _ = pad_batch(examples, indices, "cuda")
            outputs = event_bus(ids, valid)
            for row, example in enumerate(selected):
                initial = roster_recovery_exact(
                    example,
                    outputs,
                    row,
                    valid[row],
                    cfg.vocab_size,
                )
                decoded = decode_set_identity_example(
                    event_bus,
                    example,
                    outputs,
                    row,
                    valid[row],
                    bundle["kind_lexicon"],
                )
                perturbed = decode_set_identity_example(
                    event_bus,
                    example,
                    outputs,
                    row,
                    valid[row],
                    bundle["kind_lexicon"],
                    roster_permutation=(1, 2, 0),
                )
                records.append(record(example, decoded, initial))
                deranged.append(record(example, perturbed, initial))
    gold_sanity = all(
        len(example.program) == example.depth
        and example.answer_identity in {0, 1, 2}
        and sorted(example.final_state) == [0, 1, 2]
        for example in examples
    )
    result = {
        "schema": "r12_s4_set_identity_event_bus_eval_v3",
        "parser_protocol": metadata["protocol"],
        "parser_sha256": sha256_file(args.parser),
        "parser_adapter_sha256": metadata["final_adapter_sha256"],
        "base_sha256": metadata["base_sha256"],
        "s3_executor_sha256": sha256_file(args.s3_executor),
        "data_sha256": sha256_file(args.data),
        "report_sha256": sha256_file(args.report),
        "overall": summarize(records),
        "by_depth": grouped(records, "depth"),
        "by_surface": grouped(records, "surface_type"),
        "roster_deranged": {
            "overall": summarize(deranged),
            "by_depth": grouped(deranged, "depth"),
        },
        "failure_reasons": dict(collections.Counter(
            item["failure_reason"] for item in records if not item["program_exact"]
        )),
        "failures": [item for item in records if not item["program_exact"]][:64],
        "gold_event_s3_sanity": gold_sanity,
        "parameter_count": metadata["total_parameters"],
        "trainable_parameters": metadata["trainable_parameters"],
        "development_access": 1,
        "confirmation_access": 0,
        "claim_boundary": (
            "Fresh-board known-atom set-identity development. No confirmation, unseen semantics, "
            "planning, learned halt, free-form reasoning, benchmark, or novelty claim."
        ),
    }
    os.makedirs(os.path.dirname(os.path.realpath(args.out)), exist_ok=True)
    with open(args.out, "w") as target:
        json.dump(result, target, indent=2, sort_keys=True)
        target.write("\n")
    print(json.dumps({
        "out": os.path.realpath(args.out),
        "protocol": metadata["protocol"],
        "overall": result["overall"],
        "roster_deranged": result["roster_deranged"]["overall"],
        "failure_reasons": result["failure_reasons"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
