#!/usr/bin/env python3
"""Focused checks for the hash-bound source-packet token-cost audit."""

import json
import tempfile
from pathlib import Path

from audit_source_packet_tokens import main


def invoke(argv):
    import sys

    old = sys.argv
    try:
        sys.argv = argv
        main()
    finally:
        sys.argv = old


def main_test():
    tokenizer = Path(__file__).parents[1] / "artifacts" / "shohin-tok-32k.json"
    with tempfile.TemporaryDirectory() as directory:
        directory = Path(directory)
        data = directory / "rows.jsonl"
        output = directory / "report.json"
        data.write_text(
            "\n".join([
                json.dumps({
                    "chunks": ["Reference a b c d e f g h i j k l.", "Update one."],
                    "query": "What changed?",
                    "response": "The answer is one.",
                    "protocol": "source_removed_readback_v2_compact_tags",
                    "tag_scheme": "compact_v2",
                }),
                json.dumps({
                    "chunks": ["Reference m n o p q r s t u v w x."],
                    "query": "What remains?",
                    "response": "The answer is two.",
                    "protocol": "source_removed_readback_v2_compact_tags",
                    "tag_scheme": "compact_v2",
                }),
            ]) + "\n"
        )
        invoke([
            "audit_source_packet_tokens.py", "--tokenizer", str(tokenizer),
            "--data", "fixture={}".format(data), "--out", str(output),
        ])
        result = json.loads(output.read_text())
        fixture = result["data"]["fixture"]
        assert fixture["rows"] == 2
        assert fixture["chunk_tokens"]["count"] == 3
        assert fixture["protocols"] == ["source_removed_readback_v2_compact_tags"]
        assert fixture["tag_schemes"] == ["compact_v2"]
    print("source-packet token audit tests passed")


if __name__ == "__main__":
    main_test()
