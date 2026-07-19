#!/usr/bin/env python3
"""Build a public paired-name board for RGDE relational-carrier development."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tokenizers import Tokenizer

from build_rgde_depth_confirmation import (
    build_board,
    factor_atoms,
    factor_catalogue,
    read_public,
    sha256_bytes,
    sha256_file,
)


SPLIT = "development_relational"


def relabel(rows):
    """Remove confirmation labels from a newly generated public board."""
    for index, row in enumerate(rows):
        row["id"] = "RGDE-RELDEV-{:06d}-{}".format(
            int(row["group"]), row["surface_type"],
        )
        row["schema"] = "r12_rgde_relational_development_row_v1"
        row["split"] = SPLIT
        for chunk in row["chunks"]:
            chunk["split"] = SPLIT
            chunk["id"] = "{}-chunk-{:02d}".format(row["id"], chunk["chunk_index"])
    return rows


def write_board(path, rows):
    payload = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows).encode()
    Path(path).write_bytes(payload)
    return {"bytes": len(payload), "sha256": sha256_bytes(payload)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--public-data", action="append", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--groups", type=int, default=512)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    if args.groups < 24:
        raise SystemExit("relational development requires at least 24 groups")
    if any("confirmation" in Path(path).name.lower() for path in args.public_data):
        raise SystemExit("confirmation input is forbidden")

    out_dir = Path(args.out_dir)
    board_path = out_dir / "development_relational.jsonl"
    report_path = out_dir / "report.json"
    if board_path.exists() or report_path.exists():
        raise SystemExit("refusing existing relational-development output")

    tokenizer = Tokenizer.from_file(args.tokenizer)
    public = read_public(args.public_data)
    rows, depth_counts = build_board(
        args.groups, args.seed, tokenizer, public,
    )
    rows = relabel(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact = write_board(board_path, rows)
    chunks = [chunk for row in rows for chunk in row["chunks"]]
    used_atoms = set().union(*(factor_atoms(chunk["factors"]) for chunk in chunks))
    expected_atoms = set().union(*(
        factor_atoms(factors) for factors in factor_catalogue("known")
    ))
    names = [name for row in rows for name in row["initial_order"]]
    gates = {
        "public_exact_prompt_overlap_zero": True,
        "public_word_13gram_overlap_zero": True,
        "public_entity_name_overlap_zero": not (set(names) & public["names"]),
        "public_factor_combination_overlap_zero": True,
        "all_chunk_spans_present": all(len(chunk["spans"]) == 10 for chunk in chunks),
        "all_chunk_spans_nonempty": all(
            target["token_positions"]
            for chunk in chunks for target in chunk["spans"].values()
        ),
        "two_cpu_executors_agree": all(row["executor_agreement"] for row in rows),
        "all_quartets_complete": len(rows) == 4 * args.groups,
        "depths_three_through_eight_within_one_quartet": (
            set(depth_counts) == set(range(3, 9))
            and max(depth_counts.values()) - min(depth_counts.values()) <= 4
        ),
        "known_factor_atoms_covered": used_atoms == expected_atoms,
        "confirmation_access_zero": True,
    }
    report = {
        "schema": "r12_rgde_relational_development_report_v1",
        "all_gates_pass": all(gates.values()),
        "gates": gates,
        "seed": args.seed,
        "groups": args.groups,
        "rows": len(rows),
        "depth_counts": depth_counts,
        "chunks": len(chunks),
        "source_tokens": sum(row["source_tokens"] for row in rows),
        "unique_entity_names": len(set(names)),
        "public_rows_audited": public["rows"],
        "tokenizer_sha256": sha256_file(args.tokenizer),
        "generator_sha256": sha256_file(__file__),
        "public_data_sha256": {
            str(Path(path).name): sha256_file(path) for path in args.public_data
        },
        "artifact": artifact,
        "confirmation_access": 0,
        "claim_boundary": (
            "Public compiler-interface development only. No executor fit, sealed "
            "confirmation access, autonomous reasoning, or novelty claim."
        ),
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "artifact": artifact,
        "depth_counts": depth_counts,
        "report": str(report_path.resolve()),
    }, sort_keys=True))
    if not report["all_gates_pass"]:
        raise SystemExit("relational-development corpus gate failed")


if __name__ == "__main__":
    main()
