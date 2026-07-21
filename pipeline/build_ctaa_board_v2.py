#!/usr/bin/env python3
"""Single-use deterministic writer for a future CTAA revision-2 board."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
from typing import Iterable, Mapping

from tokenizers import Tokenizer

from pipeline.ctaa_board_v2 import (
    LONG_PER_CLASS_DEPTH_CELL,
    PROGRAM_CLASSES,
    SCORED_DEPTHS,
    CTAAProgramFamilyV2,
    balanced_renderer_index,
    build_compiler_families,
    build_long_families,
    iter_atomic_exposures,
    iter_closure_exposures,
    make_card_reindex_twin,
    make_equivalent_composite_twin,
    make_order_contrast_twin,
    make_post_stop_poison_twin,
    make_prefix_contrast_twin,
    make_stop_relocation_twin,
    render_family_v2,
)
from pipeline.ctaa_name_pool import audit_name_pools, build_name_pools, sha256_file


SCHEMA = "r12_ctaa_v2_board_v1"


@dataclass(frozen=True)
class BoardSizes:
    atomic_contexts: int = 64
    closure_contexts: int = 64
    compiler_per_depth: int = 4096
    long_per_class_depth_cell: int = LONG_PER_CLASS_DEPTH_CELL
    diagnostics_per_class_depth: int = 144
    name_pool_per_axis: int = 256


PRODUCTION_SIZES = BoardSizes()


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, object]], mode: int) -> int:
    count = 0
    with path.open("x") as handle:
        for row in rows:
            handle.write(canonical_json(dict(row)) + "\n")
            count += 1
    path.chmod(mode)
    return count


def _write_json(path: Path, value: object, mode: int) -> None:
    with path.open("x") as handle:
        handle.write(json.dumps(value, sort_keys=True, indent=2) + "\n")
    path.chmod(mode)


def _surface_record(row, *, scored: bool) -> dict[str, object]:
    record = row.scored_record() if scored else row.compiler_record()
    return {**record, "renderer": row.renderer}


def _intervention_records(
    seed: int,
    parent: CTAAProgramFamilyV2,
    twin,
    name_pools: Mapping[str, tuple[str, ...]],
    renderer_index: int,
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    child_surface = render_family_v2(
        seed,
        twin.child,
        name_pools,
        renderer_index=renderer_index,
        surface_key=parent.family_id,
    )
    scored = _surface_record(child_surface, scored=True)
    program = {
        "family_id": scored["family_id"],
        "program_source": scored["program_source"],
    }
    query = {
        "family_id": scored["family_id"],
        "query_source": scored["query_source"],
    }
    oracle = {
        "parent_family_id": parent.family_id,
        "relation": twin.relation,
        "invariant_terminal": twin.child.terminal_state == parent.terminal_state,
        "invariant_trace": twin.child.execute() == parent.execute(),
        **{
            key: value
            for key, value in scored.items()
            if key not in {"program_source", "query_source"}
        },
    }
    return program, query, oracle


def _build_interventions(
    seed: int,
    families: tuple[CTAAProgramFamilyV2, ...],
    name_pools: Mapping[str, tuple[str, ...]],
    diagnostics_per_class_depth: int,
    per_class_depth_cell: int,
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    CounterLike,
]:
    triple = [family for family in families if family.cell.tag == "hhh"]
    programs: list[dict[str, object]] = []
    queries: list[dict[str, object]] = []
    oracles: list[dict[str, object]] = []
    relation_counts: CounterLike = CounterLike()
    diagnostic_seen: dict[tuple[str, int, str], int] = {}
    for index, family in enumerate(triple):
        renderer_index = balanced_renderer_index(index, per_class_depth_cell)
        try:
            sensitive = make_order_contrast_twin(family)
        except ValueError:
            try:
                sensitive = make_prefix_contrast_twin(family)
            except ValueError:
                sensitive = make_stop_relocation_twin(family)
        primary_twins = (
            make_card_reindex_twin(family),
            make_post_stop_poison_twin(family),
            sensitive,
        )
        for twin in primary_twins:
            program, query, oracle = _intervention_records(
                seed, family, twin, name_pools, renderer_index
            )
            programs.append(program)
            queries.append(query)
            oracles.append(oracle)
            relation_counts[twin.relation] += 1
        stratum = (family.program_class, family.depth)
        diagnostic_makers = {
            "equivalent_composite": lambda: make_equivalent_composite_twin(
                family, seed=seed
            ),
            "prefix_contrast": lambda: make_prefix_contrast_twin(family),
            "stop_relocation": lambda: make_stop_relocation_twin(family),
        }
        for relation, maker in diagnostic_makers.items():
            key = (*stratum, relation)
            if diagnostic_seen.get(key, 0) >= diagnostics_per_class_depth:
                continue
            try:
                twin = maker()
            except ValueError:
                continue
            program, query, oracle = _intervention_records(
                seed, family, twin, name_pools, renderer_index
            )
            programs.append(program)
            queries.append(query)
            oracles.append(oracle)
            relation_counts[twin.relation] += 1
            diagnostic_seen[key] = diagnostic_seen.get(key, 0) + 1
    expected_diagnostics = {
        (program_class, depth, relation)
        for program_class in PROGRAM_CLASSES
        for depth in SCORED_DEPTHS
        for relation in (
            "equivalent_composite",
            "prefix_contrast",
            "stop_relocation",
        )
    }
    if set(diagnostic_seen) != expected_diagnostics or any(
        value != diagnostics_per_class_depth for value in diagnostic_seen.values()
    ):
        raise ValueError("CTAA v2 diagnostic intervention coverage differs")
    return programs, queries, oracles, relation_counts


class CounterLike(dict[str, int]):
    def __missing__(self, key: str) -> int:
        return 0


def _source_audit(
    root: Path,
    tokenizer_path: Path,
    files: Mapping[str, tuple[Path, Path, Path | None]],
    name_pools: Mapping[str, tuple[str, ...]],
) -> dict[str, object]:
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    database = root / "token_ngrams.sqlite"
    connection = sqlite3.connect(database)
    connection.execute(
        "CREATE TABLE grams (digest BLOB, partition TEXT, token_ids TEXT, "
        "PRIMARY KEY (digest, partition)) WITHOUT ROWID"
    )
    all_names = tuple(name for values in name_pools.values() for name in values)
    dynamic_token_ids = {
        token_id
        for name in all_names
        for token_id in tokenizer.encode(name).ids
    }
    exact: dict[str, set[str]] = {partition: set() for partition in files}
    lengths: dict[str, list[int]] = {partition: [] for partition in files}
    kind_lengths: dict[str, dict[str, list[int]]] = {
        partition: {"program": [], "query": []} for partition in files
    }
    cell_kind_lengths: dict[
        str, dict[str, dict[str, list[int]]]
    ] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for partition, (program_path, query_path, oracle_path) in files.items():
        axis_partition = partition.split("_", 1)[0]
        if axis_partition not in {"train", "development", "confirmation"}:
            raise ValueError("CTAA source-audit partition differs")
        query_rows = {
            row["family_id"]: row["query_source"]
            for row in (json.loads(line) for line in query_path.read_text().splitlines())
        }
        cells = (
            {
                row["family_id"]: row["factorial_cell"]
                for row in (json.loads(line) for line in oracle_path.read_text().splitlines())
            }
            if oracle_path is not None
            else {}
        )
        pending: list[tuple[bytes, str, str]] = []
        with program_path.open() as handle:
            for line in handle:
                row = json.loads(line)
                query_source = query_rows[row["family_id"]]
                paired_source = row["program_source"] + "\0" + query_source
                exact[partition].add(hashlib.sha256(paired_source.encode()).hexdigest())
                lexical_axis = (
                    "train"
                    if axis_partition == "train"
                    or cells[row["family_id"]][2] == "i"
                    else axis_partition
                )
                for source_kind, source in (
                    ("program", row["program_source"]),
                    ("query", query_source),
                ):
                    ids = tokenizer.encode(source).ids
                    lengths[partition].append(len(ids))
                    kind_lengths[partition][source_kind].append(len(ids))
                    if axis_partition != "train":
                        cell_key = canonical_json(cells[row["family_id"]])
                        cell_kind_lengths[partition][cell_key][source_kind].append(
                            len(ids)
                        )
                    for start in range(max(0, len(ids) - 12)):
                        gram = ids[start : start + 13]
                        if dynamic_token_ids.isdisjoint(gram):
                            continue
                        encoded = json.dumps(gram, separators=(",", ":"))
                        pending.append(
                            (
                                hashlib.sha256(encoded.encode()).digest()[:16],
                                lexical_axis,
                                encoded,
                            )
                        )
                    if len(pending) >= 50_000:
                        connection.executemany(
                            "INSERT OR IGNORE INTO grams VALUES (?,?,?)",
                            pending,
                        )
                        pending.clear()
        if pending:
            connection.executemany("INSERT OR IGNORE INTO grams VALUES (?,?,?)", pending)
        connection.commit()
    overlap_exact = {
        f"{left}_{right}": len(exact[left] & exact[right])
        for left in files
        for right in files
        if left < right
    }
    shared = connection.execute(
        "SELECT token_ids, COUNT(DISTINCT partition) FROM grams "
        "GROUP BY digest HAVING COUNT(DISTINCT partition) > 1"
    ).fetchall()
    dynamic_leaks = []
    for token_ids, partitions in shared:
        decoded = tokenizer.decode(json.loads(token_ids))
        if any(name in decoded for name in all_names):
            dynamic_leaks.append({"decoded": decoded, "partitions": partitions})
            if len(dynamic_leaks) >= 20:
                break
    connection.close()
    database.unlink()
    cross_partition_length_match = {
        source_kind: Counter(kind_lengths["development"][source_kind])
        == Counter(kind_lengths["confirmation"][source_kind])
        for source_kind in ("program", "query")
    }
    within_partition_cell_length_match: dict[str, dict[str, bool]] = {}
    for partition in ("development", "confirmation"):
        within_partition_cell_length_match[partition] = {}
        for source_kind in ("program", "query"):
            histograms = [
                Counter(by_kind[source_kind])
                for _, by_kind in sorted(cell_kind_lengths[partition].items())
            ]
            within_partition_cell_length_match[partition][source_kind] = (
                len(histograms) == 8
                and all(histogram == histograms[0] for histogram in histograms[1:])
            )
    return {
        "exact_source_overlap": overlap_exact,
        "shared_grammar_13grams": len(shared),
        "dynamic_name_13gram_leaks": dynamic_leaks,
        "token_lengths": {
            partition: {
                "count": len(values),
                "minimum": min(values),
                "maximum": max(values),
                "mean": sum(values) / len(values),
            }
            for partition, values in lengths.items()
        },
        "cross_partition_token_length_histograms_match": (
            cross_partition_length_match
        ),
        "within_partition_factorial_cell_token_length_histograms_match": (
            within_partition_cell_length_match
        ),
        "all_gates_pass": (
            all(value == 0 for value in overlap_exact.values())
            and not dynamic_leaks
            and all(max(values) <= 2048 for values in lengths.values())
            and all(cross_partition_length_match.values())
            and all(
                value
                for partition in within_partition_cell_length_match.values()
                for value in partition.values()
            )
        ),
    }


def build_board(
    seed: int,
    output_dir: Path,
    tokenizer_path: Path,
    *,
    sizes: BoardSizes = PRODUCTION_SIZES,
) -> dict[str, object]:
    if seed < 0:
        raise ValueError("CTAA v2 board seed differs")
    if output_dir.exists():
        raise FileExistsError(f"refusing existing CTAA v2 board: {output_dir}")
    temporary = output_dir.with_name(output_dir.name + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"refusing existing CTAA v2 board temporary: {temporary}")
    temporary.mkdir(parents=True)
    try:
        pools = build_name_pools(
            tokenizer_path,
            per_split=sizes.name_pool_per_axis,
        )
        pool_audit = audit_name_pools(tokenizer_path, pools)
        if not pool_audit["all_gates_pass"]:
            raise ValueError("CTAA v2 name-pool admission failed")
        _write_json(temporary / "name_pools.json", pools, 0o444)

        counts: dict[str, int] = {}
        counts["train_atomic"] = _write_jsonl(
            temporary / "train_atomic.jsonl",
            (asdict(row) for row in iter_atomic_exposures("train", sizes.atomic_contexts)),
            0o444,
        )
        counts["train_closure"] = _write_jsonl(
            temporary / "train_closure.jsonl",
            (asdict(row) for row in iter_closure_exposures(sizes.closure_contexts)),
            0o444,
        )
        compiler_families = build_compiler_families(
            seed,
            per_depth=sizes.compiler_per_depth,
        )
        counts["train_compiler"] = _write_jsonl(
            temporary / "train_compiler.jsonl",
            (
                _surface_record(
                    render_family_v2(seed, family, pools, renderer_index=index),
                    scored=False,
                )
                for index, family in enumerate(compiler_families)
            ),
            0o444,
        )

        intervention_counts: dict[str, dict[str, int]] = {}
        for partition in ("development", "confirmation"):
            program_mode = 0o444 if partition == "development" else 0o600
            sealed_mode = 0o600
            families = build_long_families(
                seed,
                partition,  # type: ignore[arg-type]
                per_class_depth_cell=sizes.long_per_class_depth_cell,
            )
            surfaces = [
                render_family_v2(
                    seed,
                    family,
                    pools,
                    renderer_index=balanced_renderer_index(
                        index,
                        sizes.long_per_class_depth_cell,
                    ),
                )
                for index, family in enumerate(families)
            ]
            program_path = temporary / f"{partition}_program.jsonl"
            query_path = temporary / f"{partition}_query.jsonl"
            oracle_path = temporary / f"{partition}_oracle.jsonl"
            counts[partition] = _write_jsonl(
                program_path,
                (
                    {
                        "family_id": surface.family.family_id,
                        "program_source": surface.program_source,
                    }
                    for surface in surfaces
                ),
                program_mode,
            )
            _write_jsonl(
                query_path,
                (
                    {
                        "family_id": surface.family.family_id,
                        "query_source": surface.query_source,
                    }
                    for surface in surfaces
                ),
                sealed_mode,
            )
            _write_jsonl(
                oracle_path,
                (
                    {
                        key: value
                        for key, value in _surface_record(surface, scored=True).items()
                        if key not in {"program_source", "query_source"}
                    }
                    for surface in surfaces
                ),
                sealed_mode,
            )
            intervention_programs, intervention_queries, intervention_oracles, relation_counts = _build_interventions(
                seed,
                families,
                pools,
                sizes.diagnostics_per_class_depth,
                sizes.long_per_class_depth_cell,
            )
            intervention_program_path = temporary / f"{partition}_intervention_program.jsonl"
            intervention_query_path = temporary / f"{partition}_intervention_query.jsonl"
            intervention_oracle_path = temporary / f"{partition}_intervention_oracle.jsonl"
            counts[f"{partition}_interventions"] = _write_jsonl(
                intervention_program_path,
                intervention_programs,
                program_mode,
            )
            _write_jsonl(intervention_query_path, intervention_queries, sealed_mode)
            _write_jsonl(intervention_oracle_path, intervention_oracles, sealed_mode)
            intervention_counts[partition] = dict(sorted(relation_counts.items()))

        source_audit = _source_audit(
            temporary,
            tokenizer_path,
            {
                "train": (
                    temporary / "train_compiler.jsonl",
                    temporary / "train_compiler.jsonl",
                    None,
                ),
                "development": (
                    temporary / "development_program.jsonl",
                    temporary / "development_query.jsonl",
                    temporary / "development_oracle.jsonl",
                ),
                "confirmation": (
                    temporary / "confirmation_program.jsonl",
                    temporary / "confirmation_query.jsonl",
                    temporary / "confirmation_oracle.jsonl",
                ),
                "development_intervention": (
                    temporary / "development_intervention_program.jsonl",
                    temporary / "development_intervention_query.jsonl",
                    temporary / "development_intervention_oracle.jsonl",
                ),
                "confirmation_intervention": (
                    temporary / "confirmation_intervention_program.jsonl",
                    temporary / "confirmation_intervention_query.jsonl",
                    temporary / "confirmation_intervention_oracle.jsonl",
                ),
            },
            pools,
        )
        if not source_audit["all_gates_pass"]:
            raise ValueError(
                "CTAA v2 source admission failed: " + canonical_json(source_audit)
            )
        access = {
            "schema": "r12_ctaa_v2_access_ledger_v1",
            "development_access": 0,
            "confirmation_access": 0,
        }
        _write_json(temporary / "access_ledger.json", access, 0o600)
        report = {
            "schema": SCHEMA,
            "seed": seed,
            "sizes": asdict(sizes),
            "counts": counts,
            "intervention_relations": intervention_counts,
            "name_pool_audit": pool_audit,
            "source_audit": source_audit,
            "development_access": 0,
            "confirmation_access": 0,
            "all_gates_pass": True,
        }
        _write_json(temporary / "admission_report.json", report, 0o444)
        hashes = {
            path.name: sha256_file(path)
            for path in sorted(temporary.iterdir())
            if path.is_file()
        }
        manifest = {
            "schema": "r12_ctaa_v2_manifest_v1",
            "seed": seed,
            "files": hashes,
        }
        _write_json(temporary / "manifest.json", manifest, 0o444)
        temporary.replace(output_dir)
        return report
    except Exception:
        if temporary.exists():
            for path in temporary.rglob("*"):
                if path.is_file():
                    path.chmod(0o600)
            shutil.rmtree(temporary)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    args = parser.parse_args()
    report = build_board(args.seed, args.output_dir, args.tokenizer)
    print(canonical_json(report))


if __name__ == "__main__":
    main()
