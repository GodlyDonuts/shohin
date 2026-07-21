from __future__ import annotations

import json
from pathlib import Path

from pipeline.build_ctaa_board_v2 import _source_audit


TOKENIZER = Path(__file__).resolve().parents[1] / "artifacts/tokenizer/tokenizer.json"
CELLS = ("iii", "iih", "ihi", "ihh", "hii", "hih", "hhi", "hhh")


def _write_partition(root: Path, partition: str, symbol: str, *, mismatch: bool) -> tuple[Path, Path, Path]:
    program = root / f"{partition}_program.jsonl"
    query = root / f"{partition}_query.jsonl"
    oracle = root / f"{partition}_oracle.jsonl"
    with program.open("w") as program_handle, query.open("w") as query_handle, oracle.open("w") as oracle_handle:
        for index, cell in enumerate(CELLS):
            family_id = f"{partition}-{index}"
            program_handle.write(
                json.dumps({"family_id": family_id, "program_source": f"{symbol} {symbol}"}) + "\n"
            )
            query_source = f"{symbol} {symbol}" if mismatch and index == 0 else symbol
            query_handle.write(
                json.dumps({"family_id": family_id, "query_source": query_source}) + "\n"
            )
            oracle_handle.write(
                json.dumps({"family_id": family_id, "factorial_cell": cell}) + "\n"
            )
    return program, query, oracle


def test_source_audit_requires_exact_length_histogram_balance(tmp_path: Path) -> None:
    development = _write_partition(tmp_path, "development", "A", mismatch=False)
    confirmation = _write_partition(tmp_path, "confirmation", "B", mismatch=False)
    report = _source_audit(
        tmp_path,
        TOKENIZER,
        {"development": development, "confirmation": confirmation},
        {"train": ("unused_train",), "development": ("unused_dev",), "confirmation": ("unused_conf",)},
    )
    assert report["all_gates_pass"]
    assert all(report["cross_partition_token_length_histograms_match"].values())
    assert all(
        value
        for partition in report[
            "within_partition_factorial_cell_token_length_histograms_match"
        ].values()
        for value in partition.values()
    )

    broken_root = tmp_path / "broken"
    broken_root.mkdir()
    development = _write_partition(broken_root, "development", "A", mismatch=False)
    confirmation = _write_partition(broken_root, "confirmation", "B", mismatch=True)
    broken = _source_audit(
        broken_root,
        TOKENIZER,
        {"development": development, "confirmation": confirmation},
        {"train": ("unused_train",), "development": ("unused_dev",), "confirmation": ("unused_conf",)},
    )
    assert not broken["all_gates_pass"]
    assert not broken["cross_partition_token_length_histograms_match"]["query"]
    assert not broken[
        "within_partition_factorial_cell_token_length_histograms_match"
    ]["confirmation"]["query"]
