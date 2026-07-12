#!/usr/bin/env python3
"""Unit test live-evaluation prompt augmentation for shard decontamination."""
import json
import tempfile
from pathlib import Path

from tokenize_shards import direct_eval_grams


def main():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        (root / "eval.jsonl").write_text(
            json.dumps({"question": "one two three four five six seven eight nine ten eleven twelve thirteen"})
            + "\n"
            + json.dumps({"prompt": "short prompt"}) + "\n"
        )
        grams = direct_eval_grams([str(root / "*.jsonl")], 13)
        assert "one two three four five six seven eight nine ten eleven twelve thirteen" in grams
        assert "short prompt" in grams
    print("live evalgram shard gate: passed")


if __name__ == "__main__":
    main()
