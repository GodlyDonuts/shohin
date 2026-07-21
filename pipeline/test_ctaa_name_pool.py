from __future__ import annotations

from pathlib import Path

from pipeline.ctaa_name_pool import audit_name_pools, build_name_pools


TOKENIZER = Path(__file__).resolve().parents[1] / "artifacts/tokenizer/tokenizer.json"


def test_name_pools_are_fixed_width_unique_and_split_disjoint() -> None:
    pools = build_name_pools(TOKENIZER, per_split=16)
    audit = audit_name_pools(TOKENIZER, pools)
    assert audit["all_gates_pass"]
    assert audit["pool_sizes"] == {
        "train": 16,
        "development": 16,
        "confirmation": 16,
    }
    assert all("-" in value for values in pools.values() for value in values)


def test_name_pool_build_is_byte_deterministic() -> None:
    assert build_name_pools(TOKENIZER, per_split=12) == build_name_pools(
        TOKENIZER,
        per_split=12,
    )
