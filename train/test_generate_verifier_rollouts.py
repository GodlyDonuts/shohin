#!/usr/bin/env python3
"""Exercise answer-mode parsing without launching a model."""
import json
import tempfile
from pathlib import Path

from generate_verifier_rollouts import read_rows


def main():
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "rows.jsonl"
        path.write_text(
            json.dumps({"question": "What is 2 plus 2?", "answer": "work #### 4"}) + "\n" +
            json.dumps({"question": "State the value", "answer": "  blue  ", "family": "string"}) + "\n"
        )
        gsm = read_rows(path, 0, "gsm8k")
        rg = read_rows(path, 0, "rg")
        assert gsm == [{"question": "What is 2 plus 2?", "gold": "4", "family": None}]
        assert rg[0]["gold"] == "work #### 4"
        assert rg[1] == {"question": "State the value", "gold": "blue", "family": "string"}
    print("verifier rollout answer-mode checks: passed")


if __name__ == "__main__":
    main()
