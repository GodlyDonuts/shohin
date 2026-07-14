#!/usr/bin/env python3
"""Focused admission checks for the matched reflection auxiliary audit."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from audit_operator_counterfactual_reflection_v1 import audit
from generate_operator_counterfactual_reflection_v1 import build_rows


def write(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def main() -> None:
    reflection, neutral = build_rows(per_family=4, seed=11)
    tokenizer = ROOT.parent / "artifacts" / "shohin-tok-32k.json"
    with tempfile.TemporaryDirectory() as raw:
        tmp = Path(raw)
        r_path, n_path = tmp / "reflection.jsonl", tmp / "neutral.jsonl"
        write(r_path, reflection)
        write(n_path, neutral)
        report = audit(str(r_path), str(n_path), str(tokenizer))
        assert report["matched_pairs"] == len(reflection)
        assert report["supervised_response_token_delta"]["all_pairs_equal"]
        assert sum(report["prompt_token_delta_histogram"].values()) == len(reflection)
        broken = [dict(row) for row in neutral]
        broken[0]["response"] = broken[0]["response"].replace("000000", "000001", 1)
        write(n_path, broken)
        try:
            audit(str(r_path), str(n_path), str(tokenizer))
        except ValueError as exc:
            assert "neutral response" in str(exc)
        else:
            raise AssertionError("non-neutral state must be rejected")
        write(n_path, neutral)
        broken = [dict(row) for row in neutral]
        broken[0]["neutral_states"] = False
        write(n_path, broken)
        try:
            audit(str(r_path), str(n_path), str(tokenizer))
        except ValueError as exc:
            assert "neutral-state labels" in str(exc)
        else:
            raise AssertionError("wrong neutral-state label must be rejected")
    print("operator counterfactual reflection audit checks: passed")


if __name__ == "__main__":
    main()
