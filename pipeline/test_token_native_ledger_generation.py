#!/usr/bin/env python3
"""Smoke contracts for token-native candidate construction."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train"))
sys.path.insert(0, str(ROOT / "pipeline"))
from generate_token_native_ledger_v1 import counterfactual_episode, episode_from_operands, rows_from_episode
from digitwise_factor_protocol import initial_tape
from token_native_ledger_protocol import context_key, parse_delta


episode = episode_from_operands("smoke", "fit_w4", 4, "add", 95, 8, "heldout")
counterpart = counterfactual_episode(episode)
assert episode["expected_answer"] != counterpart["expected_answer"]
rows = rows_from_episode(episode)
assert len(rows) == 5
assert all(parse_delta(row["response"]) is not None for row in rows if row["kind"] == "transition")
assert rows[-1]["kind"] == "final" and "Tape:" not in rows[-1]["completion_prompt"]
assert context_key(initial_tape("add", 95, 8, 4)) in rows[-1]["completion_prompt"]
print("token-native ledger generation checks: passed")
