#!/usr/bin/env python3
"""Evaluate unchanged S4 v5 exactly once on its disjoint confirmation board."""

from __future__ import annotations

import argparse
import collections
import json
import os

import torch
from tokenizers import Tokenizer

from eval_s4_hard_island_soft_interface import grouped, make_record, summarize
from model import GPT, GPTConfig
from s4_hard_island_soft_interface import decode_hard_island_soft_interface
from s4_set_identity_event_bus import roster_recovery_exact
from self_delimiting_event_tape import (
    SelfDelimitingEventTapeParser,
    adapter_hash,
    load_examples,
    make_batches,
    pad_batch,
    sha256_file,
)


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
        raise SystemExit("S4 v5 confirmation requires CUDA")
    if os.path.exists(args.out):
        raise SystemExit("refusing existing S4 v5 confirmation evaluation")
    report = json.load(open(args.report))
    if report.get("schema") != "r12_s4_hard_island_confirmation_report_v1":
        raise SystemExit("invalid S4 v5 confirmation report")
    if not report.get("all_gates_pass") or report.get("board_role") != "confirmation":
        raise SystemExit("S4 v5 confirmation board failed admission")
    if report["artifacts"]["confirmation"]["sha256"] != sha256_file(args.data):
        raise SystemExit("S4 v5 confirmation report does not bind data")

    bundle = torch.load(args.parser, map_location="cpu")
    metadata = bundle.get("parser", {})
    if metadata.get("protocol") != "r12_s4_self_delimiting_event_parser_treatment_v1":
        raise SystemExit("S4 v5 confirmation requires frozen v1 treatment parser")
    if metadata.get("base_sha256") != sha256_file(args.base):
        raise SystemExit("S4 v5 confirmation parser/base mismatch")
    s3 = torch.load(args.s3_executor, map_location="cpu").get("executor", {})
    if s3.get("protocol") != "r12_s3_equivariant_permutation_executor_v1_1":
        raise SystemExit("invalid locked S3 executor")
    if s3.get("base_sha256") != metadata.get("base_sha256"):
        raise SystemExit("S3/base mismatch")

    tokenizer = Tokenizer.from_file(args.tokenizer)
    checkpoint = torch.load(args.base, map_location="cpu")
    cfg = GPTConfig(**checkpoint["cfg"])
    examples = load_examples(args.data, tokenizer, "s4_event_tape_development", cfg.seq_len)
    model = GPT(cfg).to("cuda").eval()
    model.load_state_dict(checkpoint["model"])
    frozen = SelfDelimitingEventTapeParser(
        model,
        layer=int(metadata["layer"]),
        width=int(metadata["width"]),
        heads=int(metadata["heads"]),
        encoder_layers=int(metadata["encoder_layers"]),
        ff=int(metadata["ff"]),
    ).to("cuda").eval()
    missing, unexpected = frozen.load_state_dict(bundle["adapter_state"], strict=False)
    if unexpected or any(not name.startswith("model.") for name in missing):
        raise SystemExit("S4 v5 confirmation adapter mismatch")
    if adapter_hash(frozen) != metadata["final_adapter_sha256"]:
        raise SystemExit("S4 v5 confirmation adapter hash mismatch")

    records, roster_deranged, event_deranged = [], [], []
    for indices in make_batches(examples, args.batch_size, seed=0, shuffle=False):
        selected, ids, valid, _ = pad_batch(examples, indices, "cuda")
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = frozen(ids, valid)
            for row, example in enumerate(selected):
                initial = roster_recovery_exact(
                    example, outputs, row, valid[row], cfg.vocab_size,
                )
                normal = decode_hard_island_soft_interface(
                    frozen, example, outputs, row, valid[row], bundle["kind_lexicon"],
                )
                roster = decode_hard_island_soft_interface(
                    frozen,
                    example,
                    outputs,
                    row,
                    valid[row],
                    bundle["kind_lexicon"],
                    roster_permutation=(1, 2, 0),
                )
                event = decode_hard_island_soft_interface(
                    frozen,
                    example,
                    outputs,
                    row,
                    valid[row],
                    bundle["kind_lexicon"],
                    region_shift=1,
                )
                records.append(make_record(example, normal, initial))
                roster_deranged.append(make_record(example, roster, initial))
                event_deranged.append(make_record(example, event, initial))
    result = {
        "schema": "r12_s4_hard_island_soft_interface_confirmation_eval_v5",
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
        "roster_deranged": {"overall": summarize(roster_deranged)},
        "event_region_deranged": {"overall": summarize(event_deranged)},
        "failure_reasons": dict(collections.Counter(
            item["failure_reason"] for item in records if not item["program_exact"]
        )),
        "failures": [item for item in records if not item["program_exact"]][:64],
        "gold_event_s3_sanity": all(
            len(example.program) == example.depth
            and example.answer_identity in {0, 1, 2}
            and sorted(example.final_state) == [0, 1, 2]
            for example in examples
        ),
        "parameter_count": metadata["total_parameters"],
        "trainable_parameters": 0,
        "development_access": 0,
        "confirmation_access": 1,
    }
    os.makedirs(os.path.dirname(os.path.realpath(args.out)), exist_ok=True)
    with open(args.out, "w") as target:
        json.dump(result, target, indent=2, sort_keys=True)
        target.write("\n")
    print(json.dumps({
        "out": os.path.realpath(args.out),
        "overall": result["overall"],
        "roster_deranged": result["roster_deranged"]["overall"],
        "event_region_deranged": result["event_region_deranged"]["overall"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
