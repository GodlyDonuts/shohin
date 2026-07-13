#!/usr/bin/env python3
"""Small deterministic contracts for the DCRD generator."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train"))
sys.path.insert(0, str(ROOT / "pipeline"))

from dual_code_reversible_protocol import codebook_from_record, parse_state
from generate_dual_code_reversible_v1 import counterfactual_episode, episode_from_operands, rows_from_episode


episode = episode_from_operands("smoke", "fit_w4", 4, "add", 95, 8, "heldout-smoke", "heldout")
counterfactual = counterfactual_episode(episode)
assert episode["expected_answer"] != counterfactual["expected_answer"]
a_book = codebook_from_record(episode["codebooks"]["A"])
b_book = codebook_from_record(episode["codebooks"]["B"])
assert a_book.vocabulary == b_book.vocabulary == "heldout"
assert episode["prompt_style"] == "heldout"
assert parse_state(episode["initial_a"], a_book) is not None
rows = rows_from_episode(episode)
assert len(rows) == 4 * episode["width"] + 1
assert {row["kind"] for row in rows} == {"forward_a", "a_to_b", "reverse_b", "b_to_a", "readout"}
assert all(row["question"] == row["completion_prompt"] and row["response"] for row in rows)
assert {row["prompt_style"] for row in rows} == {"heldout"}
assert "dws:" not in "\n".join(row["question"] + "\n" + row["response"] for row in rows)
print("dual-code generator checks: passed")
