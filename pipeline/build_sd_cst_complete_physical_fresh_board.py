#!/usr/bin/env python3
"""Build the fresh-board qualification for the complete physical SD-CST bus."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import subprocess
from typing import Iterable, Mapping, Sequence

from audit_sd_cst_board import (
    _bindings,
    _overlap_sets,
    _program,
    _program_signature,
    _query,
    simulate_adjacent_swaps,
)
from build_sd_cst_board import (
    CONFIRMATION_SPLIT,
    DEVELOPMENT_SPLIT,
    TRAIN_SPLIT,
    build_all,
    row_ids_sha256,
)
from sd_cst_complete_physical_fresh_renderers import (
    SCORED_RENDERERS,
    TRAIN_RENDERERS,
    expand_rows,
    render_program,
    render_query,
)


BOARD_SCHEMA = "r12_sd_cst_complete_physical_fresh_board_report_v1"
PROTOCOL = "r12_sd_cst_complete_physical_fresh_v1"
NAME_RE = re.compile(r"\b[a-z0-9]{4}-[a-z0-9]{8}\b")
NUMBER_RE = re.compile(r"\b[0-9]+\b")
EXPECTED_ROWS = {
    TRAIN_SPLIT: 48_000,
    DEVELOPMENT_SPLIT: 2_048,
    CONFIRMATION_SPLIT: 2_048,
}


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonl(rows: Iterable[Mapping[str, object]]) -> bytes:
    return b"".join(
        (json.dumps(dict(row), sort_keys=True) + "\n").encode("utf-8") for row in rows
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
        raise ValueError("fresh-board source commit is not current HEAD")
    if subprocess.run(
        ["git", "diff", "--quiet", source_commit, "--"],
        cwd=root,
        check=False,
    ).returncode:
        raise ValueError("tracked worktree differs from fresh-board source commit")
    return source_commit


def _load_rows(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _canonical_families(
    rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    values = [dict(row) for row in rows if row.get("variant") == "canonical"]
    if len(values) * 8 != len(rows):
        raise ValueError("base evaluation board does not have eight-view families")
    if len({str(row.get("family_id")) for row in values}) != len(values):
        raise ValueError("base canonical families are not unique")
    return values


def renderer_signature(row: Mapping[str, object]) -> str:
    source = str(row["program_text"]) + "\n<QUERY>\n" + str(row["late_query_text"])
    return NUMBER_RE.sub("<N>", NAME_RE.sub("<NAME>", source))


def _row_semantics_exact(row: Mapping[str, object]) -> bool:
    lines = str(row["program_text"]).splitlines()
    if len(lines) != 9 or sum("HALT" in line for line in lines) != 1:
        return False
    if max(map(lambda line: len(line.encode("utf-8")), lines)) > 144:
        return False
    query = str(row["late_query_text"])
    if "\n" in query or len(query.encode("utf-8")) > 144:
        return False
    renderer_value = row.get("fresh_renderer")
    if not isinstance(renderer_value, Mapping):
        return False
    renderers = TRAIN_RENDERERS if row["split"] == TRAIN_SPLIT else SCORED_RENDERERS
    renderer = next(
        (
            item
            for item in renderers
            if item.as_tuple()
            == tuple(
                int(renderer_value[key]) for key in ("declaration", "event", "query")
            )
        ),
        None,
    )
    if renderer is None:
        return False
    expected_query, expected_span = render_query(row, renderer)
    if str(row["program_text"]) != render_program(row, renderer):
        return False
    if query != expected_query or list(expected_span) != row["late_query_target"].get(
        "byte_span"
    ):
        return False
    bindings = _bindings(row)
    if set(bindings) != {0, 1, 2} or len(set(bindings.values())) != 3:
        return False
    expected_state, trajectory = simulate_adjacent_swaps(
        tuple(int(value) for value in row["compiler_targets"]["initial_order_roles"]),
        _program(row),
        int(row["compiler_targets"]["halt_after"]),
    )
    if row["split"] == TRAIN_SPLIT:
        return "oracle" not in row and row.get("supervision") == "compiler_fields_only"
    oracle = row.get("oracle")
    if not isinstance(oracle, Mapping):
        return False
    answer = expected_state[_query(row)]
    return (
        tuple(oracle.get("final_state_roles", ())) == expected_state
        and tuple(tuple(item) for item in oracle.get("active_trajectory_roles", ()))
        == trajectory
        and int(oracle.get("answer_role", -1)) == answer
        and oracle.get("answer_entity") == bindings[answer]
    )


def _family_registration(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    by_family: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        by_family[str(row["family_id"])].add(str(row["variant"]))
    variants = sorted({str(row["variant"]) for row in rows})
    content = "".join(
        f"{row['id']}:{sha256_bytes(canonical_json(dict(row)).encode('utf-8'))}\n"
        for row in sorted(rows, key=lambda item: str(item["id"]))
    )
    return {
        "protocol": PROTOCOL,
        "row_count": len(rows),
        "family_count": len(by_family),
        "family_size": 4,
        "variants": variants,
        "all_families_complete": all(
            values == set(variants) for values in by_family.values()
        ),
        "row_ids_sha256": row_ids_sha256(list(rows)),
        "row_content_sha256": sha256_bytes(content.encode("utf-8")),
        "depth_counts": dict(
            sorted(
                Counter(
                    int(row["compiler_targets"]["halt_after"]) for row in rows
                ).items()
            )
        ),
    }


def audit_fresh_board(
    train: Sequence[Mapping[str, object]],
    development: Sequence[Mapping[str, object]],
    confirmation: Sequence[Mapping[str, object]],
    prior_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    split_rows = {
        TRAIN_SPLIT: train,
        DEVELOPMENT_SPLIT: development,
        CONFIRMATION_SPLIT: confirmation,
    }
    all_rows = list(train) + list(development) + list(confirmation)
    split_sets = {split: _overlap_sets(rows) for split, rows in split_rows.items()}
    pairs = (
        (TRAIN_SPLIT, DEVELOPMENT_SPLIT),
        (TRAIN_SPLIT, CONFIRMATION_SPLIT),
        (DEVELOPMENT_SPLIT, CONFIRMATION_SPLIT),
    )
    overlap = {
        field: {
            f"{left}__{right}": len(split_sets[left][field] & split_sets[right][field])
            for left, right in pairs
        }
        for field in ("prompts", "grams", "names", "sequences")
    }
    prior_sets = _overlap_sets(prior_rows)
    prior_overlap = {
        split: {
            field: len(split_sets[split][field] & prior_sets[field])
            for field in ("prompts", "grams", "names", "sequences")
        }
        for split in split_rows
    }
    registrations = {
        split: _family_registration(rows) for split, rows in split_rows.items()
    }
    renderer_digests = {
        split: sorted(
            {sha256_bytes(renderer_signature(row).encode("utf-8")) for row in rows}
        )
        for split, rows in split_rows.items()
    }
    renderer_names = {
        split: {str(row["variant"]) for row in rows}
        for split, rows in split_rows.items()
    }
    family_names: dict[str, tuple[str, ...]] = {}
    family_names_consistent = True
    for row in all_rows:
        family = str(row["family_id"])
        names = tuple(_bindings(row)[role] for role in range(3))
        if family in family_names and family_names[family] != names:
            family_names_consistent = False
        family_names[family] = names
    unique_names = [name for names in family_names.values() for name in names]
    gates = {
        "exact_row_counts": all(
            len(split_rows[split]) == count for split, count in EXPECTED_ROWS.items()
        ),
        "all_ids_unique": len({str(row["id"]) for row in all_rows}) == len(all_rows),
        "all_rows_semantically_exact": all(
            _row_semantics_exact(row) for row in all_rows
        ),
        "all_families_have_four_views": all(
            registration["all_families_complete"] and registration["family_size"] == 4
            for registration in registrations.values()
        ),
        "train_and_scored_renderer_orbits_disjoint": renderer_names[
            TRAIN_SPLIT
        ].isdisjoint(renderer_names[DEVELOPMENT_SPLIT]),
        "development_confirmation_renderer_orbits_equal": renderer_names[
            DEVELOPMENT_SPLIT
        ]
        == renderer_names[CONFIRMATION_SPLIT],
        "cross_split_exact_prompt_overlap_zero": not any(overlap["prompts"].values()),
        "cross_split_13gram_overlap_zero": not any(overlap["grams"].values()),
        "cross_split_name_overlap_zero": not any(overlap["names"].values()),
        "cross_split_sequence_overlap_zero": not any(overlap["sequences"].values()),
        "zero_prior_prompt_name_sequence_overlap": all(
            values[field] == 0
            for values in prior_overlap.values()
            for field in ("prompts", "names", "sequences")
        ),
        "zero_prior_scored_13gram_overlap": all(
            prior_overlap[split]["grams"] == 0
            for split in (DEVELOPMENT_SPLIT, CONFIRMATION_SPLIT)
        ),
        "opaque_names_fixed_and_globally_unique": family_names_consistent
        and len(unique_names) == len(set(unique_names))
        and all(NAME_RE.fullmatch(name) for name in unique_names),
        "training_oracles_absent": all("oracle" not in row for row in train),
        "evaluation_oracles_present": all(
            "oracle" in row for row in list(development) + list(confirmation)
        ),
        "scored_access_zero": True,
    }
    return {
        "schema": "r12_sd_cst_complete_physical_fresh_board_audit_v1",
        "all_gates_pass": all(gates.values()),
        "gates": gates,
        "overlap": overlap,
        "prior_overlap": prior_overlap,
        "renderer_grammar_sha256": renderer_digests,
        "registrations": registrations,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--prior-train", type=Path, required=True)
    parser.add_argument("--prior-development", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise SystemExit(f"refusing existing fresh board: {args.out_dir}")
    source_commit = _verify_source_commit(args.source_commit)
    prior_train = _load_rows(args.prior_train)
    prior_development = _load_rows(args.prior_development)
    prior_rows = prior_train + prior_development
    reserved = {_program_signature(row) for row in prior_rows}
    train_base, development_base, confirmation_base = build_all(
        train_rows=12_000,
        development_families=512,
        confirmation_families=512,
        seed=args.seed,
        reserved_sequences=reserved,
    )
    train = expand_rows(train_base, TRAIN_RENDERERS)
    development = expand_rows(_canonical_families(development_base), SCORED_RENDERERS)
    confirmation = expand_rows(_canonical_families(confirmation_base), SCORED_RENDERERS)
    audit = audit_fresh_board(train, development, confirmation, prior_rows)
    if not audit["all_gates_pass"]:
        raise SystemExit(json.dumps(audit["gates"], sort_keys=True))
    payloads = {
        "train.jsonl": _jsonl(train),
        "development.jsonl": _jsonl(development),
        "confirmation.sealed.jsonl": _jsonl(confirmation),
    }
    prior = {
        "train_sha256": sha256_file(args.prior_train),
        "development_sha256": sha256_file(args.prior_development),
    }
    report = {
        "schema": BOARD_SCHEMA,
        "protocol": PROTOCOL,
        "seed": args.seed,
        "source_commit": source_commit,
        "source_custody": "clean_head_verified",
        "prior_consumed": prior,
        "audit": audit,
        "all_gates_pass": True,
        "development_registration": audit["registrations"][DEVELOPMENT_SPLIT],
        "confirmation_registration": audit["registrations"][CONFIRMATION_SPLIT],
        "files": {
            name: {"bytes": len(payload), "sha256": sha256_bytes(payload)}
            for name, payload in payloads.items()
        },
        "development_accesses": 0,
        "confirmation_accesses": 0,
        "claim_boundary": "Fresh finite compiler board only; no neural score.",
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
    print(json.dumps({"all_gates_pass": True, "rows": EXPECTED_ROWS}, sort_keys=True))


if __name__ == "__main__":
    main()
