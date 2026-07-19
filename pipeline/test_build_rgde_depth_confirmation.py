import json
from pathlib import Path

from tokenizers import Tokenizer

from build_rgde_depth_confirmation import build_board


ROOT = Path(__file__).resolve().parents[1]


def empty_public():
    return {
        "rows": 0,
        "names": set(),
        "questions": set(),
        "grams": set(),
        "factor_signatures": set(),
    }


def test_small_depth_board_has_complete_balanced_quartets():
    tokenizer = Tokenizer.from_file(str(ROOT / "artifacts/shohin-tok-32k.json"))
    rows, depths = build_board(6, 918273645, tokenizer, empty_public())
    assert len(rows) == 24
    assert depths == {depth: 4 for depth in range(3, 9)}
    for row in rows:
        assert len(row["program"]) == row["depth"]
        assert sum(chunk["active_operations"] for chunk in row["chunks"]) == row["depth"]
        assert all(len(chunk["spans"]) == 10 for chunk in row["chunks"])
        assert row["answer"] == row["terminal_order"][row["query"]["position"]]
    groups = {}
    for row in rows:
        groups.setdefault(row["group"], {})[row["surface_type"]] = row
    for group in groups.values():
        assert set(group) == {"canonical", "paraphrase", "order_twin", "binding_twin"}
        assert group["canonical"]["answer"] != group["order_twin"]["answer"]
        assert group["canonical"]["answer"] != group["binding_twin"]["answer"]
        assert group["canonical"]["word_bag"] == group["order_twin"]["word_bag"]
        assert group["canonical"]["word_bag"] == group["binding_twin"]["word_bag"]


def test_depth_board_rows_are_json_serializable():
    tokenizer = Tokenizer.from_file(str(ROOT / "artifacts/shohin-tok-32k.json"))
    rows, _ = build_board(6, 564738291, tokenizer, empty_public())
    assert all(json.loads(json.dumps(row))["schema"].endswith("_v1") for row in rows)
