from __future__ import annotations

from collections import Counter
from pathlib import Path
import json

from tokenizers import Tokenizer

from pipeline.build_ctaa_binding_completion_board import (
    audit_orbits,
    build_board,
    orbit_records,
)
from pipeline.ctaa_binding_identification import permutation_parity
from pipeline.ctaa_board_v2 import (
    OPCODE_BINDINGS,
    build_compiler_families,
)
from pipeline.ctaa_name_pool import build_name_pools


SEED = 711_249
TOKENIZER = Path(__file__).resolve().parents[1] / "artifacts/tokenizer/tokenizer.json"


def test_complete_orbit_holds_semantics_and_surface_nuisance_fixed() -> None:
    pools = build_name_pools(TOKENIZER, per_split=16)
    family = build_compiler_families(SEED, per_depth=1)[0]
    orbit = orbit_records(
        SEED,
        family,
        pools,
        renderer_index=3,
    )
    assert len(orbit) == 24
    assert {tuple(row["opcode_to_card"]) for row in orbit} == set(OPCODE_BINDINGS)
    assert Counter(
        permutation_parity(row["opcode_to_card"]) for row in orbit
    ) == {0: 12, 1: 12}
    for key in (
        "family_id",
        "query_source",
        "action_cards",
        "initial_state",
        "schedule",
        "query_position",
        "renderer",
    ):
        assert len({str(row[key]) for row in orbit}) == 1
    assert len({str(row["program_source"]) for row in orbit}) == 24


def test_orbit_audit_requires_matched_parity_lengths_and_local_marginals() -> None:
    pools = build_name_pools(TOKENIZER, per_split=16)
    families = build_compiler_families(SEED, per_depth=2)
    orbits = tuple(
        orbit_records(
            SEED,
            family,
            pools,
            renderer_index=index,
        )
        for index, family in enumerate(families)
    )
    audit = audit_orbits(orbits, Tokenizer.from_file(str(TOKENIZER)))
    assert audit["orbits"] == 16
    assert audit["train_even_rows"] == 192
    assert audit["confirmation_odd_rows"] == 192
    assert audit["per_binding"] == 16
    assert audit["all_gates_pass"]
    assert all(audit["gates"].values())


def test_writer_separates_confirmation_source_from_oracle(tmp_path: Path) -> None:
    output = tmp_path / "board"
    manifest = build_board(
        SEED,
        output,
        TOKENIZER,
        orbits_per_depth=16,
        name_pool_per_axis=16,
    )
    source_lines = (
        output / "confirmation_odd_source.jsonl"
    ).read_text().splitlines()
    oracle_lines = (
        output / "confirmation_odd_oracle.jsonl"
    ).read_text().splitlines()
    assert len(source_lines) == len(oracle_lines)
    source = json.loads(source_lines[0])
    oracle = json.loads(oracle_lines[0])
    assert set(source) == {"row_id", "family_id", "program_source"}
    assert source["row_id"] == oracle["row_id"]
    source_row_ids = [json.loads(line)["row_id"] for line in source_lines]
    oracle_row_ids = [json.loads(line)["row_id"] for line in oracle_lines]
    assert source_row_ids == oracle_row_ids
    assert len(set(source_row_ids)) == len(source_row_ids)
    assert "program_source" not in oracle
    assert "opcode_to_card" in oracle
    assert manifest["confirmation_odd_source_rows_written"] == len(source_lines)
    assert manifest["confirmation_odd_oracle_rows_written"] == len(oracle_lines)
