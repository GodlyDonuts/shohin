#!/usr/bin/env python3
"""CPU-only contracts for CBC selection and causal metric aggregation."""
import json
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train"))
from eval_counterfactual_bisimulation import select_episodes, summarize_pair


with tempfile.TemporaryDirectory() as directory:
    path = Path(directory) / "episodes.jsonl"
    rows = [
        {"id": "a", "heldout": True, "regime": "r1"},
        {"id": "b", "heldout": True, "regime": "r1"},
        {"id": "c", "heldout": True, "regime": "r2"},
        {"id": "d", "heldout": True, "regime": "r2"},
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    selected = select_episodes(path, 1, 7)
    assert len(selected) == 2 and {row["regime"] for row in selected} == {"r1", "r2"}

world = {
    "compilations": {"a": {"correct": True}, "b": {"correct": True}},
    "compile_equal": True,
    "primary": {"state_closed_loop": True, "final_correct": True, "rows": [
        {"state_correct": True, "inverse_delta_correct": True},
    ]},
    "interchange": {"state_closed_loop": True, "final_correct": True, "rows": []},
}
counts = summarize_pair({
    "normal": world,
    "counterfactual": world,
    "same_world_interchange_success": True,
    "counterfactual_interchange_success": True,
    "cross_world_counterfactual_success": True,
})
assert counts["normal_primary_update_correct"] == 1
assert counts["counterfactual_compile_equal"] == 1
assert counts["cross_world_counterfactual_success"] == 1
print("counterfactual bisimulation evaluator checks: passed")
