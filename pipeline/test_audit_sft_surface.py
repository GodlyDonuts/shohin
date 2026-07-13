#!/usr/bin/env python3
"""Regression checks for the tokenizer-only SFT surface audit."""
import json
import subprocess
import sys
import tempfile
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory() as temporary:
        temporary = Path(temporary)
        data, out = temporary / "rows.jsonl", temporary / "report.json"
        data.write_text("\n".join([
            json.dumps({"completion_prompt": "State: x\\nAnswer:", "response": "ok", "kind": "transition", "width": 4, "prompt_style": "core"}),
            json.dumps({"completion_prompt": "State: y\\nAnswer:", "response": "long " * 32, "kind": "final", "width": 6, "prompt_style": "heldout"}),
            json.dumps({"completion_prompt": "", "response": "ignored", "kind": "bad"}),
        ]) + "\n")
        subprocess.run([
            sys.executable, str(root / "pipeline" / "audit_sft_surface.py"),
            "--data", str(data), "--tokenizer", str(root / "artifacts" / "shohin-tok-32k.json"),
            "--out", str(out), "--pack-len", "16",
        ], check=True, stdout=subprocess.DEVNULL)
        report = json.loads(out.read_text())
        assert report["invalid_or_missing_rows"] == 1
        assert report["overall"]["rows"] == 2
        assert report["overall"]["fit_rows"] == 1
        assert report["by_field_value"]["kind=transition"]["rows"] == 1
        assert report["by_field_value"]["prompt_style=heldout"]["over_pack_len_rows"] == 1
    print("SFT surface audit checks: passed")


if __name__ == "__main__":
    main()
