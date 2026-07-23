#!/usr/bin/env python3
"""Write the orbit-matched A4 binding-completion board exactly once.

Every semantic compiler family is rendered under all 24 declaration orders.
Even permutations enter optimization; odd permutations remain sealed until
both neural arms freeze. This writer is separate from the production CTAA
board and does not expose recurrent outcomes.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
from typing import Iterable, Mapping, Sequence

from tokenizers import Tokenizer

from pipeline.ctaa_binding_identification import permutation_parity
from pipeline.ctaa_board_v2 import (
    OPCODE_BINDINGS,
    CTAAProgramFamilyV2,
    build_compiler_families,
    render_family_v2,
)
from pipeline.ctaa_name_pool import audit_name_pools, build_name_pools, sha256_file


SCHEMA = "r12_ctaa_a4_binding_completion_board_v1"
RENDERER_COUNT = 16


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def confirmation_row_id(row: Mapping[str, object]) -> str:
    return hashlib.sha256(
        canonical_json(
            {
                "family_id": row["family_id"],
                "program_source": row["program_source"],
            }
        ).encode("ascii")
    ).hexdigest()


def _write_jsonl(
    path: Path,
    rows: Iterable[Mapping[str, object]],
    mode: int,
) -> int:
    count = 0
    with path.open("x", encoding="ascii") as handle:
        for row in rows:
            handle.write(canonical_json(dict(row)) + "\n")
            count += 1
    path.chmod(mode)
    return count


def orbit_records(
    seed: int,
    family: CTAAProgramFamilyV2,
    name_pools: Mapping[str, tuple[str, ...]],
    *,
    renderer_index: int,
) -> tuple[dict[str, object], ...]:
    """Render one fixed semantic scaffold under the complete S4 orbit."""

    result = []
    for binding_index in range(len(OPCODE_BINDINGS)):
        surface = render_family_v2(
            seed,
            family,
            name_pools,
            renderer_index=renderer_index,
            binding_index=binding_index,
            surface_key=family.family_id,
        )
        record = surface.compiler_record()
        result.append({**record, "renderer": surface.renderer})
    return tuple(result)


def audit_orbits(
    orbits: Sequence[Sequence[Mapping[str, object]]],
    tokenizer: Tokenizer,
) -> dict[str, object]:
    if not orbits:
        raise ValueError("CTAA completion orbit board is empty")
    binding_counts: Counter[tuple[int, ...]] = Counter()
    renderer_counts: Counter[int] = Counter()
    train_rows = 0
    confirmation_rows = 0
    program_sources: dict[int, set[str]] = {0: set(), 1: set()}
    token_histograms: dict[int, Counter[int]] = {
        0: Counter(),
        1: Counter(),
    }
    per_orbit_token_match = True
    for orbit in orbits:
        if len(orbit) != len(OPCODE_BINDINGS):
            raise ValueError("CTAA completion orbit does not span S4")
        family_ids = {str(row["family_id"]) for row in orbit}
        if len(family_ids) != 1:
            raise ValueError("CTAA completion orbit family differs")
        invariant_keys = (
            "query_source",
            "action_cards",
            "initial_state",
            "schedule",
            "query_position",
            "renderer",
        )
        for key in invariant_keys:
            if len({canonical_json(row[key]) for row in orbit}) != 1:
                raise ValueError(f"CTAA completion orbit changes invariant {key}")
        bindings = [tuple(int(item) for item in row["opcode_to_card"]) for row in orbit]
        if set(bindings) != set(OPCODE_BINDINGS):
            raise ValueError("CTAA completion orbit binding support differs")
        parity_counts = Counter(permutation_parity(binding) for binding in bindings)
        if parity_counts != {0: 12, 1: 12}:
            raise ValueError("CTAA completion orbit parity balance differs")
        local_marginals = {
            parity: tuple(
                tuple(
                    sum(
                        binding[opcode] == card
                        for binding in bindings
                        if permutation_parity(binding) == parity
                    )
                    for card in range(4)
                )
                for opcode in range(4)
            )
            for parity in (0, 1)
        }
        if local_marginals != {
            0: ((3, 3, 3, 3),) * 4,
            1: ((3, 3, 3, 3),) * 4,
        }:
            raise ValueError("CTAA completion orbit local marginals differ")
        orbit_lengths: dict[int, Counter[int]] = {0: Counter(), 1: Counter()}
        for row, binding in zip(orbit, bindings, strict=True):
            parity = permutation_parity(binding)
            source = str(row["program_source"])
            if source in program_sources[1 - parity]:
                raise ValueError("CTAA completion source partitions overlap")
            program_sources[parity].add(source)
            length = len(tokenizer.encode(source).ids)
            orbit_lengths[parity][length] += 1
            token_histograms[parity][length] += 1
            binding_counts[binding] += 1
            renderer_counts[int(row["renderer"])] += 1
            train_rows += parity == 0
            confirmation_rows += parity == 1
        per_orbit_token_match &= orbit_lengths[0] == orbit_lengths[1]
    expected_per_binding = len(orbits)
    if binding_counts != {
        binding: expected_per_binding for binding in OPCODE_BINDINGS
    }:
        raise ValueError("CTAA completion global binding balance differs")
    if set(renderer_counts.values()) != {
        len(orbits) // RENDERER_COUNT * len(OPCODE_BINDINGS)
    }:
        raise ValueError("CTAA completion renderer balance differs")
    gates = {
        "complete_s4_orbit_per_family": True,
        "a4_odd_12_12_per_orbit": True,
        "local_3_3_3_3_per_orbit": True,
        "semantic_scaffold_invariant_per_orbit": True,
        "program_sources_disjoint": not program_sources[0].intersection(
            program_sources[1]
        ),
        "token_histograms_match_per_orbit": per_orbit_token_match,
        "token_histograms_match_global": token_histograms[0] == token_histograms[1],
        "renderer_counts_exact": len(renderer_counts) == RENDERER_COUNT,
    }
    return {
        "schema": SCHEMA,
        "claim_boundary": "binding_completion_board_only_no_neural_result",
        "orbits": len(orbits),
        "train_even_rows": train_rows,
        "confirmation_odd_rows": confirmation_rows,
        "per_binding": expected_per_binding,
        "renderer_counts": dict(sorted(renderer_counts.items())),
        "train_token_length_histogram": dict(sorted(token_histograms[0].items())),
        "confirmation_token_length_histogram": dict(
            sorted(token_histograms[1].items())
        ),
        "gates": gates,
        "all_gates_pass": all(gates.values()),
    }


def build_board(
    seed: int,
    output_dir: Path,
    tokenizer_path: Path,
    *,
    orbits_per_depth: int = 256,
    name_pool_per_axis: int = 256,
) -> dict[str, object]:
    if seed < 0:
        raise ValueError("CTAA completion board seed differs")
    if orbits_per_depth < RENDERER_COUNT or orbits_per_depth % RENDERER_COUNT:
        raise ValueError("CTAA completion orbit count must balance renderers")
    if output_dir.exists():
        raise FileExistsError(f"refusing existing CTAA completion board: {output_dir}")
    temporary = output_dir.with_name(output_dir.name + ".tmp")
    if temporary.exists():
        raise FileExistsError(
            f"refusing existing CTAA completion temporary: {temporary}"
        )
    temporary.mkdir(parents=True)
    try:
        pools = build_name_pools(
            tokenizer_path,
            per_split=name_pool_per_axis,
        )
        pool_audit = audit_name_pools(tokenizer_path, pools)
        if not pool_audit["all_gates_pass"]:
            raise ValueError("CTAA completion name-pool admission failed")
        families = build_compiler_families(seed, per_depth=orbits_per_depth)
        orbits = tuple(
            orbit_records(
                seed,
                family,
                pools,
                renderer_index=index % RENDERER_COUNT,
            )
            for index, family in enumerate(families)
        )
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        audit = audit_orbits(orbits, tokenizer)
        if not audit["all_gates_pass"]:
            raise ValueError("CTAA completion orbit audit failed")
        train = (
            row
            for orbit in orbits
            for row in orbit
            if permutation_parity(row["opcode_to_card"]) == 0
        )
        confirmation = (
            row
            for orbit in orbits
            for row in orbit
            if permutation_parity(row["opcode_to_card"]) == 1
        )
        train_path = temporary / "train_even.jsonl"
        confirmation_source_path = temporary / "confirmation_odd_source.jsonl"
        confirmation_oracle_path = temporary / "confirmation_odd_oracle.jsonl"
        train_count = _write_jsonl(train_path, train, 0o444)
        confirmation_rows = tuple(confirmation)
        confirmation_source_count = _write_jsonl(
            confirmation_source_path,
            (
                {
                    "row_id": confirmation_row_id(row),
                    "family_id": row["family_id"],
                    "program_source": row["program_source"],
                }
                for row in confirmation_rows
            ),
            0o400,
        )
        confirmation_oracle_count = _write_jsonl(
            confirmation_oracle_path,
            (
                {
                    "row_id": confirmation_row_id(row),
                    **{
                        key: value
                        for key, value in row.items()
                        if key != "program_source"
                    },
                }
                for row in confirmation_rows
            ),
            0o400,
        )
        manifest = {
            **audit,
            "seed": seed,
            "orbits_per_depth": orbits_per_depth,
            "name_pool_per_axis": name_pool_per_axis,
            "tokenizer_sha256": sha256_file(tokenizer_path),
            "train_even_sha256": sha256_file(train_path),
            "confirmation_odd_source_sha256": sha256_file(
                confirmation_source_path
            ),
            "confirmation_odd_oracle_sha256": sha256_file(
                confirmation_oracle_path
            ),
            "train_even_rows_written": train_count,
            "confirmation_odd_source_rows_written": confirmation_source_count,
            "confirmation_odd_oracle_rows_written": confirmation_oracle_count,
        }
        encoded = json.dumps(manifest, sort_keys=True, indent=2) + "\n"
        manifest_path = temporary / "manifest.json"
        manifest_path.write_text(encoded, encoding="ascii")
        manifest_path.chmod(0o444)
        temporary.replace(output_dir)
        return manifest
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--orbits-per-depth", type=int, default=256)
    parser.add_argument("--name-pool-per-axis", type=int, default=256)
    args = parser.parse_args()
    report = build_board(
        args.seed,
        args.output_dir,
        args.tokenizer,
        orbits_per_depth=args.orbits_per_depth,
        name_pool_per_axis=args.name_pool_per_axis,
    )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
