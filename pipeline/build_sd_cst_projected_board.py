#!/usr/bin/env python3
"""Build the one-shot projected SD-CST fresh train/development/confirmation board."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import subprocess
from typing import Iterable, Mapping, Sequence

from audit_sd_cst_board import _overlap_sets, _program_signature, audit_board
from build_sd_cst_board import (
    CONFIRMATION_SPLIT,
    DEVELOPMENT_SPLIT,
    SURFACE_TYPES,
    TRAIN_SPLIT,
    build_all,
    row_ids_sha256,
)


BOARD_SCHEMA = "r12_sd_cst_projected_fresh_board_report_v1"
PROTOCOL = "r12_sd_cst_projected_fresh_v1"
NAME_RE = re.compile(r"\b[a-z0-9]{4}-[a-z0-9]{8}\b")
NUMBER_RE = re.compile(r"\b[0-9]+\b")
EXPECTED_ROWS = {
    TRAIN_SPLIT: 48_000,
    DEVELOPMENT_SPLIT: 2_304,
    CONFIRMATION_SPLIT: 2_304,
}
PARENT_BOARD_REPORT_SHA256 = (
    "e4ac239ccf1cb519a41a0e6d8abb9e5a8c880598a67c060953379d49aa79b0de"
)
PARENT_BOARD_TRAIN_SHA256 = (
    "694bce3a6077244b06ebeabba66e3255dc47507f664f8f45c2e9c3780581e0bb"
)


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _jsonl(rows: Iterable[Mapping[str, object]]) -> bytes:
    return b"".join(
        (json.dumps(row, sort_keys=True) + "\n").encode("utf-8") for row in rows
    )


def _verify_source_commit(source_commit: str) -> str:
    if re.fullmatch(r"[0-9a-f]{40}", source_commit) is None:
        raise ValueError("source commit must be a full lowercase Git SHA")
    root = Path(__file__).resolve().parents[1]
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if head != source_commit:
        raise ValueError(f"source commit {source_commit} is not current HEAD {head}")
    if subprocess.run(
        ["git", "diff", "--quiet", source_commit, "--"],
        cwd=root,
        check=False,
    ).returncode:
        raise ValueError("tracked worktree differs from frozen source commit")
    return source_commit


def _names(row: Mapping[str, object]) -> tuple[str, ...]:
    targets = row.get("compiler_targets")
    if not isinstance(targets, Mapping):
        raise ValueError("row lacks compiler targets")
    values = targets.get("entity_bindings")
    if not isinstance(values, list) or len(values) != 3:
        raise ValueError("row lacks exactly three bindings")
    ordered = sorted(values, key=lambda item: int(item["entity_role"]))
    return tuple(str(item["entity"]) for item in ordered)


def renderer_signature(row: Mapping[str, object]) -> str:
    """Bind grammar bytes while abstracting names and instance-local ordinals."""
    source = str(row["program_text"]) + "\n<QUERY>\n" + str(row["late_query_text"])
    source = NAME_RE.sub("<NAME>", source)
    source = NUMBER_RE.sub("<N>", source)
    return source


def lexical_inventory(row: Mapping[str, object]) -> tuple[str, ...]:
    source = NAME_RE.sub(
        " ", str(row["program_text"]) + " " + str(row["late_query_text"])
    )
    return tuple(sorted(set(re.findall(r"[A-Za-z]+", source))))


def projected_audit(
    train: Sequence[Mapping[str, object]],
    development: Sequence[Mapping[str, object]],
    confirmation: Sequence[Mapping[str, object]],
    parent_train: Sequence[Mapping[str, object]] | None = None,
) -> dict[str, object]:
    base = audit_board(list(train), list(development), list(confirmation))
    split_rows = {
        TRAIN_SPLIT: train,
        DEVELOPMENT_SPLIT: development,
        CONFIRMATION_SPLIT: confirmation,
    }
    name_sets = {
        split: {name for row in rows for name in _names(row)}
        for split, rows in split_rows.items()
    }
    all_names = set().union(*name_sets.values())
    fixed_width = all(len(name.encode("ascii")) == 13 for name in all_names)
    no_prefix_or_substring = all(
        left == right or (left not in right and right not in left)
        for index, left in enumerate(sorted(all_names))
        for right in sorted(all_names)[index + 1 :]
    )
    names_disjoint = (
        name_sets[TRAIN_SPLIT].isdisjoint(name_sets[DEVELOPMENT_SPLIT])
        and name_sets[TRAIN_SPLIT].isdisjoint(name_sets[CONFIRMATION_SPLIT])
        and name_sets[DEVELOPMENT_SPLIT].isdisjoint(name_sets[CONFIRMATION_SPLIT])
    )
    renderer_groups: dict[str, set[str]] = defaultdict(set)
    lexical_groups: dict[str, set[tuple[str, ...]]] = defaultdict(set)
    for split, rows in split_rows.items():
        for row in rows:
            key = f"{split}:{row['template_id']}"
            renderer_groups[key].add(renderer_signature(row))
            lexical_groups[key].add(lexical_inventory(row))
    renderer_digests = {
        key: sha256_bytes("\n---\n".join(sorted(values)).encode("utf-8"))
        for key, values in sorted(renderer_groups.items())
    }
    lexical_digests = {
        key: sha256_bytes(canonical_json(sorted(values)).encode("utf-8"))
        for key, values in sorted(lexical_groups.items())
    }
    expected_renderer_groups = 5
    split_binding_balance = {}
    for split in (DEVELOPMENT_SPLIT, CONFIRMATION_SPLIT):
        role_counts: Counter[int] = Counter()
        query_counts: Counter[int] = Counter()
        answer_counts: Counter[int] = Counter()
        for row in split_rows[split]:
            targets = row["compiler_targets"]
            query_counts[int(row["late_query_target"]["position"])] += 1
            answer_counts[int(row["oracle"]["answer_role"])] += 1
            for binding in targets["entity_bindings"]:
                role_counts[int(binding["entity_role"])] += 1
        split_binding_balance[split] = {
            "roles": dict(sorted(role_counts.items())),
            "queries": dict(sorted(query_counts.items())),
            "answers": dict(sorted(answer_counts.items())),
        }
    gates = {
        "base_board_all_gates_pass": bool(base.get("all_gates_pass")),
        "exact_row_counts": all(
            len(split_rows[split]) == count for split, count in EXPECTED_ROWS.items()
        ),
        "names_disjoint_across_splits": names_disjoint,
        "all_opaque_names_fixed_13_bytes": fixed_width,
        "no_name_prefix_or_substring_relationships": no_prefix_or_substring,
        "renderer_and_lexical_inventory_hashes_bound": (
            len(renderer_digests) == expected_renderer_groups
            and set(renderer_digests) == set(lexical_digests)
        ),
        "evaluation_query_balance_exact": all(
            set(values["queries"].values()) == {768}
            for values in split_binding_balance.values()
        ),
        "evaluation_answer_balance_exact": all(
            set(values["answers"].values()) == {768}
            for values in split_binding_balance.values()
        ),
    }
    inherited_overlap: dict[str, dict[str, int]] | None = None
    if parent_train is not None:
        parent_sets = _overlap_sets(parent_train)
        inherited_overlap = {}
        for split, rows in split_rows.items():
            fresh_sets = _overlap_sets(rows)
            inherited_overlap[split] = {
                field: len(parent_sets[field] & fresh_sets[field])
                for field in ("prompts", "grams", "names", "sequences")
            }
        gates["zero_inherited_parent_instance_overlap"] = all(
            values[field] == 0
            for values in inherited_overlap.values()
            for field in ("prompts", "names", "sequences")
        )
        gates["zero_inherited_scored_13gram_overlap"] = all(
            inherited_overlap[split]["grams"] == 0
            for split in (DEVELOPMENT_SPLIT, CONFIRMATION_SPLIT)
        )
    return {
        "base_audit": base,
        "projected_gates": gates,
        "all_gates_pass": all(gates.values()),
        "name_counts": {key: len(value) for key, value in name_sets.items()},
        "renderer_grammar_sha256": renderer_digests,
        "lexical_inventory_sha256": lexical_digests,
        "evaluation_balance": split_binding_balance,
        "inherited_parent_train_overlap": inherited_overlap,
    }


def load_parent_training_board(
    parent_board_dir: Path,
) -> tuple[list[dict[str, object]], dict[str, str]]:
    report_path = parent_board_dir / "report.json"
    train_path = parent_board_dir / "train.jsonl"
    if (
        sha256_bytes(report_path.read_bytes()) != PARENT_BOARD_REPORT_SHA256
        or sha256_bytes(train_path.read_bytes()) != PARENT_BOARD_TRAIN_SHA256
    ):
        raise ValueError(
            "inherited parent board hashes differ from the frozen contract"
        )
    report = json.loads(report_path.read_text())
    if (
        report.get("schema") != "r12_sd_cst_board_report_v1_1"
        or report.get("all_gates_pass") is not True
        or report.get("files", {}).get("train.jsonl", {}).get("sha256")
        != PARENT_BOARD_TRAIN_SHA256
    ):
        raise ValueError("inherited parent board receipt differs")
    rows = [
        json.loads(line) for line in train_path.read_text().splitlines() if line.strip()
    ]
    if len(rows) != 48_000 or any(row.get("split") != TRAIN_SPLIT for row in rows):
        raise ValueError("inherited parent train split differs")
    return rows, {
        "report_sha256": PARENT_BOARD_REPORT_SHA256,
        "train_sha256": PARENT_BOARD_TRAIN_SHA256,
    }


def _registration(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    row_content = "".join(
        f"{row['id']}:{sha256_bytes(canonical_json(dict(row)).encode('utf-8'))}\n"
        for row in sorted(rows, key=lambda item: str(item["id"]))
    )
    return {
        "protocol": PROTOCOL,
        "row_count": len(rows),
        "family_count": len(rows) // len(SURFACE_TYPES),
        "family_size": len(SURFACE_TYPES),
        "row_ids_sha256": row_ids_sha256(list(rows)),
        "row_content_sha256": sha256_bytes(row_content.encode("utf-8")),
        "depth_counts": dict(
            sorted(
                Counter(
                    int(row["compiler_targets"]["halt_after"]) for row in rows
                ).items()
            )
        ),
        "variants": list(SURFACE_TYPES),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--parent-board-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise SystemExit(f"refusing existing projected board: {args.out_dir}")
    source_commit = _verify_source_commit(args.source_commit)
    parent_train, parent_board = load_parent_training_board(args.parent_board_dir)
    reserved_sequences = {_program_signature(row) for row in parent_train}
    train, development, confirmation = build_all(
        train_rows=EXPECTED_ROWS[TRAIN_SPLIT],
        development_families=288,
        confirmation_families=288,
        seed=args.seed,
        reserved_sequences=reserved_sequences,
    )
    audit = projected_audit(train, development, confirmation, parent_train)
    if not audit["all_gates_pass"]:
        raise SystemExit("projected fresh board audit failed")
    payloads = {
        "train.jsonl": _jsonl(train),
        "development.jsonl": _jsonl(development),
        "confirmation.sealed.jsonl": _jsonl(confirmation),
    }
    report = {
        "schema": BOARD_SCHEMA,
        "protocol": PROTOCOL,
        "seed": args.seed,
        "source_commit": source_commit,
        "source_custody": "clean_head_verified",
        "inherited_parent_board": parent_board,
        "all_gates_pass": True,
        "audit": audit,
        "development_registration": _registration(development),
        "confirmation_registration": _registration(confirmation),
        "files": {
            name: {"bytes": len(value), "sha256": sha256_bytes(value)}
            for name, value in payloads.items()
        },
        "development_accesses": 0,
        "confirmation_accesses": 0,
        "claim_boundary": (
            "Fresh compiler board only. No neural score or broad reasoning claim."
        ),
    }
    args.out_dir.mkdir(parents=True)
    for name, payload in payloads.items():
        path = args.out_dir / name
        path.write_bytes(payload)
        if name == "confirmation.sealed.jsonl":
            path.chmod(0o600)
    (args.out_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(
        json.dumps(
            {
                "all_gates_pass": True,
                "out_dir": str(args.out_dir.resolve()),
                "rows": {
                    key: len(value)
                    for key, value in {
                        "train": train,
                        "development": development,
                        "confirmation": confirmation,
                    }.items()
                },
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
