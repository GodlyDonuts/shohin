#!/usr/bin/env python3
"""CPU-only contract checks for the CRA decision artifact."""
import json
import os
import subprocess
import sys
import tempfile


def report(strict=0, normal=0, paraphrase=0, counterfactual=0, zero=0, shuffled=0):
    return {"summary": {
        "strict_causal": strict,
        "normal_correct": normal,
        "paraphrase_correct": paraphrase,
        "counterfactual_correct": counterfactual,
        "zero_recreates_normal": zero,
        "shuffle_recreates_normal": shuffled,
    }}


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    script = os.path.join(root, "pipeline", "assess_counterfactual_residual_algebra.py")
    names = ("raw", "raw_nll", "combined", "combined_nll", "train", "language", "values", "delta", "query", "two_edit")
    with tempfile.TemporaryDirectory() as directory:
        paths = {}
        for name in names:
            path = os.path.join(directory, name + ".json")
            with open(path, "w") as output:
                json.dump(report(), output)
            paths[name] = path
        output = os.path.join(directory, "decision.json")
        command = [sys.executable, script]
        for name in names:
            command.extend(["--" + name.replace("_", "-"), paths[name]])
        command.extend(["--out", output])
        subprocess.check_call(command)
        decision = json.load(open(output))
        assert not decision["combined_gate_pass"]
        assert decision["decision"].startswith("reject_cra")
        assert not decision["combined_checks"]["strict_causal"]
    print("CRA gate aggregation checks: passed")


if __name__ == "__main__":
    main()
