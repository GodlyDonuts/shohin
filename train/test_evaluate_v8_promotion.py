#!/usr/bin/env python3
import json
import subprocess
import sys
import tempfile
from pathlib import Path


def board(counts):
    rows = []
    for (task, metric), (correct, total) in counts.items():
        rows.append(f"{task}  {metric}  ckpt=x  step=180000  {correct}/{total} = {100 * correct / total:.1f}%")
    return "\n".join(rows) + "\n"


REQUIRED = {
    ("gsm8k", "maj@8"): (5, 200),
    ("gsm8k", "pass@1"): (4, 200),
    ("math500", "pass@1"): (3, 200),
    ("humaneval", "pass@1"): (6, 164),
    ("mbpp", "pass@1"): (0, 200),
}


def interview(initial, fact):
    return {"cases": 8, "summary": {"initial": initial, "verified_fact": fact, "valid_state_and_reuse": 0}}


def run(candidate_counts, candidate_initial, candidate_fact):
    root = Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory() as temporary:
        temporary = Path(temporary)
        raw_board = temporary / "raw.log"; raw_board.write_text(board(REQUIRED))
        candidate_board = temporary / "candidate.log"; candidate_board.write_text(board(candidate_counts))
        raw_interview = temporary / "raw.json"; raw_interview.write_text(json.dumps(interview(1, 1)))
        candidate_interview = temporary / "candidate.json"; candidate_interview.write_text(json.dumps(interview(candidate_initial, candidate_fact)))
        out = temporary / "out.json"
        subprocess.run([
            sys.executable, str(root / "evaluate_v8_promotion.py"),
            "--raw-board", str(raw_board), "--candidate-board", str(candidate_board),
            "--raw-interview", str(raw_interview), "--candidate-interview", str(candidate_interview),
            "--out", str(out),
        ], check=True, capture_output=True, text=True)
        return json.loads(out.read_text())


def main():
    accepted = dict(REQUIRED)
    accepted[("gsm8k", "maj@8")] = (7, 200)
    accepted[("math500", "pass@1")] = (5, 200)
    assert run(accepted, 2, 1)["verdict"] == "accept_followup"
    regressed = dict(accepted)
    regressed[("humaneval", "pass@1")] = (5, 164)
    result = run(regressed, 2, 1)
    assert result["verdict"] == "reject"
    assert "code_regression:humaneval" in result["reasons"]
    print("V8 promotion decision gate: passed")


if __name__ == "__main__":
    main()
