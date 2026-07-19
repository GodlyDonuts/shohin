import json
from pathlib import Path

from tokenizers import Tokenizer

from build_rgde_depth_confirmation import build_board
from build_rgde_relational_development import SPLIT, relabel


ROOT = Path(__file__).resolve().parents[1]


def empty_public():
    return {
        "rows": 0,
        "names": set(),
        "questions": set(),
        "grams": set(),
        "factor_signatures": set(),
    }


def test_relabel_makes_public_development_rows():
    tokenizer = Tokenizer.from_file(str(ROOT / "artifacts/shohin-tok-32k.json"))
    rows, depths = build_board(6, 228177, tokenizer, empty_public())
    rows = relabel(rows)
    assert len(rows) == 24
    assert depths == {depth: 4 for depth in range(3, 9)}
    assert all(row["split"] == SPLIT for row in rows)
    assert all(row["schema"].endswith("development_row_v1") for row in rows)
    assert all(chunk["split"] == SPLIT for row in rows for chunk in row["chunks"])
    assert all(json.loads(json.dumps(row))["id"].startswith("RGDE-RELDEV-") for row in rows)
