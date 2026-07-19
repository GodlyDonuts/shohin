#!/usr/bin/env python3
"""Evaluate autonomous S4 event parsing and locked exact S3 consumption."""

from __future__ import annotations

import argparse
import collections
import json
import os

import torch
from tokenizers import Tokenizer

from model import GPT, GPTConfig
from self_delimiting_event_tape import (
    SelfDelimitingEventTapeParser,
    adapter_hash,
    decode_example,
    load_examples,
    make_batches,
    pad_batch,
    sha256_file,
)


BOOL_FIELDS = (
    "valid",
    "count_exact",
    "program_exact",
    "query_exact",
    "state_exact",
    "answer_correct",
    "initial_exact",
    "all_kind_lexical_matched",
)


def summarize(records):
    total = len(records)
    result = {"rows": total}
    for field in BOOL_FIELDS:
        correct = sum(bool(record[field]) for record in records)
        result[field] = {"correct": correct, "accuracy": correct / max(1, total)}
    result["mean_predicted_event_count"] = sum(
        record["predicted_event_count"] for record in records
    ) / max(1, total)
    return result


def evaluate_decodes(
    examples, outputs_by_index, lexicon, host_count=False, oracle_intro=False,
    oracle_query=False,
):
    records = []
    for index, example in enumerate(examples):
        outputs, row = outputs_by_index[index]
        decoded = decode_example(
            example,
            outputs,
            row,
            lexicon,
            host_count=host_count,
            oracle_intro=oracle_intro,
            oracle_query=oracle_query,
        )
        valid = bool(decoded.get("valid"))
        program = decoded.get("program") if valid else None
        records.append({
            "id": example.row_id,
            "depth": example.depth,
            "surface_type": example.surface_type,
            "valid": valid,
            "predicted_event_count": int(decoded.get("event_count", 0)),
            "raw_component_counts": list(decoded.get("raw_counts", (0, 0, 0))),
            "intro_run_counts": list(decoded.get("intro_run_counts", (0, 0, 0))),
            "query_run_count": int(decoded.get("query_run_count", 0)),
            "failure_reason": decoded.get("failure_reason", "unknown"),
            "count_exact": int(decoded.get("event_count", 0)) == example.depth,
            "program_exact": program == example.program,
            "query_exact": valid and decoded.get("query") == example.query_target,
            "state_exact": valid and decoded.get("final_state") == example.final_state,
            "answer_correct": valid and decoded.get("answer_identity") == example.answer_identity,
            "initial_exact": valid and decoded.get("intro_ids") == example.initial_ids,
            "all_kind_lexical_matched": valid and all(decoded.get("lexical_matches", ())),
        })
    return records


def grouped(records, key):
    groups = collections.defaultdict(list)
    for record in records:
        groups[str(record[key])].append(record)
    return {name: summarize(values) for name, values in sorted(groups.items())}


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
        raise SystemExit("S4 evaluation requires CUDA")
    if os.path.exists(args.out):
        raise SystemExit("refusing existing S4 evaluation")
    report = json.load(open(args.report))
    if not report.get("all_gates_pass"):
        raise SystemExit("S4 corpus report did not pass")
    if report["artifacts"]["development"]["sha256"] != sha256_file(args.data):
        raise SystemExit("S4 report does not bind development data")
    bundle = torch.load(args.parser, map_location="cpu")
    metadata = bundle.get("parser", {})
    if metadata.get("protocol") not in {
        "r12_s4_self_delimiting_event_parser_treatment_v1",
        "r12_s4_self_delimiting_event_parser_shuffled_control_v1",
    }:
        raise SystemExit("invalid S4 parser protocol")
    if metadata.get("base_sha256") != sha256_file(args.base):
        raise SystemExit("S4 parser/base mismatch")
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
    event_parser = SelfDelimitingEventTapeParser(
        model,
        layer=int(metadata["layer"]),
        width=int(metadata["width"]),
        heads=int(metadata["heads"]),
        encoder_layers=int(metadata["encoder_layers"]),
        ff=int(metadata["ff"]),
    ).to("cuda").eval()
    missing, unexpected = event_parser.load_state_dict(bundle["adapter_state"], strict=False)
    if unexpected or any(not name.startswith("model.") for name in missing):
        raise SystemExit("S4 parser adapter mismatch")
    if adapter_hash(event_parser) != metadata["final_adapter_sha256"]:
        raise SystemExit("S4 parser adapter hash mismatch")
    outputs_by_index = [None] * len(examples)
    batches = make_batches(examples, args.batch_size, seed=0, shuffle=False)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for indices in batches:
            _, ids, valid, _ = pad_batch(examples, indices, "cuda")
            outputs = event_parser(ids, valid)
            cpu_outputs = {name: value.float().cpu() for name, value in outputs.items()}
            for row, index in enumerate(indices):
                outputs_by_index[index] = (cpu_outputs, row)
    strict = evaluate_decodes(examples, outputs_by_index, bundle["kind_lexicon"], False)
    host = evaluate_decodes(examples, outputs_by_index, bundle["kind_lexicon"], True)
    gold_boundaries = evaluate_decodes(
        examples,
        outputs_by_index,
        bundle["kind_lexicon"],
        oracle_intro=True,
        oracle_query=True,
    )
    gold_sanity = all(
        example.final_state and example.answer_identity in {0, 1, 2}
        and len(example.program) == example.depth
        for example in examples
    )
    result = {
        "schema": "r12_s4_self_delimiting_event_tape_eval_v1",
        "parser_protocol": metadata["protocol"],
        "parser_sha256": sha256_file(args.parser),
        "parser_adapter_sha256": metadata["final_adapter_sha256"],
        "base_sha256": metadata["base_sha256"],
        "s3_executor_sha256": sha256_file(args.s3_executor),
        "s3_executor_state_sha256": s3["final_executor_sha256"],
        "execution_protocol": "locked_s3_v1_4_exact_local_action_table",
        "data_sha256": sha256_file(args.data),
        "report_sha256": sha256_file(args.report),
        "strict_autonomous": {
            "overall": summarize(strict),
            "by_depth": grouped(strict, "depth"),
            "by_surface": grouped(strict, "surface_type"),
        },
        "gold_count_control": {
            "overall": summarize(host),
            "by_depth": grouped(host, "depth"),
        },
        "gold_intro_query_control": {
            "overall": summarize(gold_boundaries),
            "by_depth": grouped(gold_boundaries, "depth"),
        },
        "gold_event_s3_sanity": gold_sanity,
        "parameter_count": metadata["total_parameters"],
        "development_access": 1,
        "confirmation_access": 0,
        "failures": [record for record in strict if not record["program_exact"]][:32],
        "failure_reasons": dict(collections.Counter(
            record["failure_reason"] for record in strict if not record["program_exact"]
        )),
        "claim_boundary": (
            "Public autonomous event-count/program parsing with locked exact S3 action. "
            "No confirmation, unseen semantics, planning, broad reasoning, or novelty claim."
        ),
    }
    os.makedirs(os.path.dirname(os.path.realpath(args.out)), exist_ok=True)
    with open(args.out, "w") as target:
        json.dump(result, target, indent=2, sort_keys=True)
        target.write("\n")
    print(json.dumps({
        "out": os.path.realpath(args.out),
        "protocol": metadata["protocol"],
        "strict": result["strict_autonomous"]["overall"],
        "gold_count": result["gold_count_control"]["overall"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
