#!/usr/bin/env python3
"""Evaluate a complete pointer compiler on development or sealed confirmation."""

from __future__ import annotations

import argparse
import collections
import json
import os
from pathlib import Path

import torch
from tokenizers import Tokenizer

from model import GPT, GPTConfig
from referential_literal_pointer_compiler import (
    CompletePointerCompiler,
    TARGET_LABELS,
    execute_prediction,
    load_examples,
    make_batches,
    pad_batch,
    predictions_from_outputs,
    semantic_exact,
    sha256_file,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--split", choices=("development", "confirmation"), required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("complete compiler evaluation requires CUDA")
    if Path(args.out).exists():
        raise SystemExit("refusing existing evaluation output")

    report = json.load(open(args.report))
    data_sha256 = sha256_file(args.data)
    if not report.get("all_gates_pass"):
        raise SystemExit("corpus report did not pass")
    if report.get("artifacts", {}).get(args.split, {}).get("sha256") != data_sha256:
        raise SystemExit("corpus report does not bind evaluation bytes")
    bundle = torch.load(args.adapter, map_location="cpu")
    metadata = bundle.get("compiler", {})
    if metadata.get("protocol") not in {
        "r12_referential_literal_pointer_compiler_v1_1_development",
        "r12_referential_literal_pointer_compiler_v1_2_structured_development",
        "r12_referential_literal_pointer_compiler_v1_3_islands_development",
    }:
        raise SystemExit("invalid compiler bundle protocol")
    if metadata.get("confirmation_access") != 0:
        raise SystemExit("compiler metadata already records confirmation access")
    if metadata.get("base_sha256") != sha256_file(args.base):
        raise SystemExit("compiler does not bind supplied base")
    if metadata.get("tokenizer_sha256") != sha256_file(args.tokenizer):
        raise SystemExit("compiler does not bind tokenizer")

    tokenizer = Tokenizer.from_file(args.tokenizer)
    checkpoint = torch.load(args.base, map_location="cpu")
    cfg = GPTConfig(**checkpoint["cfg"])
    examples = load_examples(
        args.data, tokenizer, args.split, cfg.seq_len, keep_evidence=True, limit=args.limit,
    )
    encodings = [tokenizer.encode(example.question) for example in examples]
    model = GPT(cfg).to("cuda").eval()
    model.load_state_dict(checkpoint["model"])
    compiler = CompletePointerCompiler(
        model,
        layer=int(metadata["layer"]),
        width=int(metadata["width"]),
        heads=int(metadata["heads"]),
        decoder_layers=int(metadata["decoder_layers"]),
        ff=int(metadata["ff"]),
        encoder_layers=int(metadata.get("encoder_layers", 0)),
        role_supervision=bool(metadata.get("role_supervision", False)),
        separate_kind_decoder=bool(metadata.get("separate_kind_decoder", False)),
    ).to("cuda").eval()
    missing, unexpected = compiler.load_state_dict(bundle["adapter_state"], strict=False)
    missing = [name for name in missing if not name.startswith("model.")]
    unexpected = [name for name in unexpected if not name.startswith("model.")]
    if missing or unexpected:
        raise SystemExit("adapter mismatch missing={} unexpected={}".format(missing, unexpected))

    records = [None] * len(examples)
    batches = make_batches(examples, args.batch_size, seed=0, shuffle=False)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for batch_number, indices in enumerate(batches, 1):
            selected, ids, valid = pad_batch(examples, indices, "cuda")
            outputs = compiler(ids, valid)
            pointer_predictions, kind_predictions = predictions_from_outputs(outputs)
            for local, global_index in enumerate(indices):
                example = selected[local]
                pointers = {
                    label: int(pointer_predictions[label][local]) for label in TARGET_LABELS
                }
                kinds = tuple(map(int, kind_predictions[local]))
                hits = {
                    label: pointers[label] in set(example.target_positions[label])
                    for label in TARGET_LABELS
                }
                kind_hits = tuple(
                    predicted == target for predicted, target in zip(kinds, example.kind_targets)
                )
                predicted_answer, semantic = execute_prediction(
                    example, encodings[global_index], pointers, kinds,
                )
                records[global_index] = {
                    "id": example.row_id,
                    "group": example.group,
                    "surface_type": example.surface_type,
                    "pointer_predictions": pointers,
                    "pointer_hits": hits,
                    "kind_predictions": list(kinds),
                    "kind_hits": list(kind_hits),
                    "initial_joint": all(hits["intro.entity{}".format(index)] for index in range(3)),
                    "operation0_joint": (
                        kind_hits[0] and hits["op0.kind"] and hits["op0.entity"]
                        and hits["op0.literal"]
                    ),
                    "operation1_joint": (
                        kind_hits[1] and hits["op1.kind"] and hits["op1.entity"]
                        and hits["op1.literal"]
                    ),
                    "full_pointer_exact": all(hits.values()) and all(kind_hits),
                    "semantic_program_exact": semantic_exact(example, semantic),
                    "predicted_answer": predicted_answer,
                    "expected_answer": example.answer,
                    "answer_correct": predicted_answer == example.answer,
                }
            if batch_number % 25 == 0:
                print("[compiler-eval] {}/{} batches".format(batch_number, len(batches)), flush=True)

    def summarize(selected):
        total = len(selected)
        return {
            "rows": total,
            "pointer_accuracy": {
                label: sum(record["pointer_hits"][label] for record in selected) / total
                for label in TARGET_LABELS
            },
            "kind_accuracy": sum(sum(record["kind_hits"]) for record in selected) / (2 * total),
            "initial_joint": sum(record["initial_joint"] for record in selected) / total,
            "operation0_joint": sum(record["operation0_joint"] for record in selected) / total,
            "operation1_joint": sum(record["operation1_joint"] for record in selected) / total,
            "full_pointer_exact": sum(record["full_pointer_exact"] for record in selected) / total,
            "semantic_program_exact": sum(record["semantic_program_exact"] for record in selected) / total,
            "answer_accuracy": sum(record["answer_correct"] for record in selected) / total,
        }

    by_surface = {
        surface: summarize([record for record in records if record["surface_type"] == surface])
        for surface in sorted({record["surface_type"] for record in records})
    }
    groups = collections.defaultdict(dict)
    for record in records:
        groups[record["group"]][record["surface_type"]] = record
    group_summary = {
        "groups": len(groups),
        "all_four_full_pointer_exact": sum(
            all(record["full_pointer_exact"] for record in group.values())
            for group in groups.values()
        ),
        "canonical_paraphrase_both_exact": sum(
            group["canonical"]["full_pointer_exact"] and group["paraphrase"]["full_pointer_exact"]
            for group in groups.values()
        ),
        "all_four_answers_correct": sum(
            all(record["answer_correct"] for record in group.values())
            for group in groups.values()
        ),
    }
    result = {
        "schema": "r12_referential_literal_pointer_compiler_eval_v1_1",
        "split": args.split,
        "base_sha256": sha256_file(args.base),
        "adapter": os.path.realpath(args.adapter),
        "adapter_file_sha256": sha256_file(args.adapter),
        "data_sha256": data_sha256,
        "report_sha256": sha256_file(args.report),
        "tokenizer_sha256": sha256_file(args.tokenizer),
        "confirmation_access": int(args.split == "confirmation") * len(examples),
        "overall": summarize(records),
        "by_surface": by_surface,
        "group_summary": group_summary,
        "records": records,
        "claim_boundary": (
            "Complete pointer-compiler evaluation only. Host dereferences selected source words "
            "and applies the frozen list machine; no executor/halt/native-reasoning claim."
        ),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "out": str(Path(args.out).resolve()),
        "split": args.split,
        "overall": result["overall"],
        "group_summary": group_summary,
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
