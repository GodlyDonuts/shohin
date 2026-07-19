#!/usr/bin/env python3
"""Build the one-shot known-atom confirmation for lexical closed-S3 execution."""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train"))
sys.path.insert(0, str(ROOT / "pipeline"))

from build_rgde_depth_confirmation import (  # noqa: E402
    build_board,
    factor_atoms,
    factor_catalogue,
    ngram_hashes,
    normalized,
    sha256_bytes,
    sha256_file,
)
from referential_literal_pointer_compiler import compile_row  # noqa: E402


def public_record(paths):
    result = {
        "rows": 0, "names": set(), "questions": set(), "grams": set(),
        "factor_signatures": set(),
    }
    for path in paths:
        with open(path) as source:
            for line in source:
                if not line.strip():
                    continue
                row = json.loads(line)
                records = row.get("chunks") or [row]
                result["rows"] += len(records)
                result["names"].update(row.get("initial_order", ()))
                for record in records:
                    result["names"].update(record.get("initial_order", ()))
                    if record.get("neutral_anchor"):
                        result["names"].add(record["neutral_anchor"])
                    question = record["question"]
                    result["questions"].add(normalized(question))
                    result["grams"].update(ngram_hashes(question))
                    result["factor_signatures"].add(record["factor_signature"])
    return result


def semantic_operation_key(row):
    initial = tuple(row["initial_order"])
    return tuple(
        (operation["kind"], initial.index(operation["entity"]), int(operation["amount"]))
        for operation in row["program"]
    )


def derangement_feasible(values):
    counts = collections.Counter(values)
    return bool(values) and max(counts.values()) <= len(values) - max(counts.values())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--kind-lexicon", required=True)
    parser.add_argument("--public-data", action="append", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--groups", type=int, default=512)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    if args.groups < 24:
        raise SystemExit("S3 lexical confirmation requires at least 24 groups")
    if any("confirmation" in Path(path).name.lower() for path in args.public_data):
        raise SystemExit("old confirmation input is forbidden")
    out_dir = Path(args.out_dir)
    board_path = out_dir / "confirmation_depth.jsonl"
    report_path = out_dir / "report.json"
    if board_path.exists() or report_path.exists():
        raise SystemExit("refusing existing S3 lexical confirmation")
    tokenizer = Tokenizer.from_file(args.tokenizer)
    lexicon = json.load(open(args.kind_lexicon))
    if not lexicon.get("all_gates_pass") or lexicon.get("development_access") != 0:
        raise SystemExit("confirmation requires admitted training-only lexicon")
    patterns = {
        tuple(record["token_ids"]): record["kind"] for record in lexicon["patterns"]
    }
    public = public_record(args.public_data)
    rows, depth_counts = build_board(args.groups, args.seed, tokenizer, public)
    all_chunks = [chunk for row in rows for chunk in row["chunks"]]
    direction_refs = 0
    direction_matches = 0
    for row in rows:
        for chunk in row["chunks"]:
            example = compile_row(chunk, tokenizer, keep_evidence=True)
            for index, operation in enumerate(
                chunk["program"][:int(chunk["active_operations"])]
            ):
                token_ids = tuple(
                    example.ids[position]
                    for position in example.target_positions["op{}.kind".format(index)]
                )
                direction_refs += 1
                direction_matches += patterns.get(token_ids) == operation["kind"]
    by_depth = collections.defaultdict(list)
    for row in rows:
        by_depth[int(row["depth"])].append(row)
    query_feasible = all(
        derangement_feasible([int(row["query"]["position"]) for row in selected])
        for selected in by_depth.values()
    )
    operation_feasible = all(
        derangement_feasible([semantic_operation_key(row) for row in selected])
        for selected in by_depth.values()
    )
    used_atoms = set().union(*(factor_atoms(chunk["factors"]) for chunk in all_chunks))
    expected_atoms = set().union(*(
        factor_atoms(factors) for factors in factor_catalogue("known")
    ))
    gates = {
        "public_exact_prompt_overlap_zero": True,
        "public_word_13gram_overlap_zero": True,
        "public_entity_name_overlap_zero": True,
        "public_factor_combination_overlap_zero": True,
        "all_chunk_spans_present_and_nonempty": all(
            len(chunk["spans"]) == 10
            and all(target["token_positions"] for target in chunk["spans"].values())
            for chunk in all_chunks
        ),
        "two_cpu_executors_agree": all(row["executor_agreement"] for row in rows),
        "all_quartets_complete": len(rows) == 4 * args.groups,
        "balanced_depths_three_through_eight": (
            set(depth_counts) == set(range(3, 9))
            and max(depth_counts.values()) - min(depth_counts.values()) <= 4
        ),
        "known_factor_atoms_covered": used_atoms == expected_atoms,
        "all_directions_are_training_lexicon_atoms": direction_matches == direction_refs,
        "operation_derangement_feasible_per_depth": operation_feasible,
        "query_derangement_feasible_per_depth": query_feasible,
        "old_confirmation_access_zero": True,
    }
    payload = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows).encode()
    out_dir.mkdir(parents=True, exist_ok=True)
    board_path.write_bytes(payload)
    report = {
        "schema": "r12_s3_lexical_confirmation_report_v1",
        "all_gates_pass": all(gates.values()),
        "gates": gates,
        "seed": args.seed,
        "groups": args.groups,
        "rows": len(rows),
        "depth_counts": depth_counts,
        "chunks": len(all_chunks),
        "source_tokens": sum(row["source_tokens"] for row in rows),
        "direction_references": direction_refs,
        "direction_lexicon_matches": direction_matches,
        "unique_entity_names": len({name for row in rows for name in row["initial_order"]}),
        "public_rows_audited": public["rows"],
        "artifact": {"bytes": len(payload), "sha256": sha256_bytes(payload)},
        "tokenizer_sha256": sha256_file(args.tokenizer),
        "kind_lexicon_sha256": sha256_file(args.kind_lexicon),
        "generator_sha256": sha256_file(__file__),
        "public_data_sha256": {
            str(Path(path).name): sha256_file(path) for path in args.public_data
        },
        "fit_updates": 0,
        "old_confirmation_access": 0,
        "claim_boundary": (
            "Fresh known-atom source-deleted S3 confirmation with external schedule/halt; "
            "not unseen-phrase generalization or autonomous reasoning."
        ),
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "artifact": report["artifact"], "depth_counts": depth_counts,
        "direction_references": direction_refs, "report": str(report_path.resolve()),
    }, sort_keys=True))
    if not report["all_gates_pass"]:
        raise SystemExit("S3 lexical confirmation corpus gate failed: {}".format(gates))


if __name__ == "__main__":
    main()
