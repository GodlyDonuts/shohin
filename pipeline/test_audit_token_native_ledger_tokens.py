#!/usr/bin/env python3
"""Contract test for token-native ledger tokenizer accounting."""
from pathlib import Path

from tokenizers import Tokenizer

from audit_token_native_ledger_tokens import summarize


ROOT = Path(__file__).resolve().parents[1]
tokenizer = Tokenizer.from_file(str(ROOT / "artifacts/shohin-tok-32k.json"))
summary = summarize(ROOT / "artifacts/sft/token_native_ledger_v1_train.jsonl", tokenizer)
assert set(summary) == {"transition", "final"}
assert set(summary["transition"]["response_length_counts"]) == {3}
assert summary["transition"]["rows"] > 0
assert summary["final"]["rows"] > 0
print("token-native ledger token accounting checks: passed")
