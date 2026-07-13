#!/usr/bin/env python3
import json
import subprocess
import sys
import tempfile
from pathlib import Path


REQUIRED = {
    ("gsm8k", "maj@8"): (5, 200),
    ("gsm8k", "pass@1"): (4, 200),
    ("math500", "pass@1"): (3, 200),
    ("humaneval", "pass@1"): (6, 164),
    ("mbpp", "pass@1"): (0, 200),
}


def board(rows):
    return "\n".join(
        f"{task}  {metric}  ckpt=x  step=180000  {correct}/{total} = {100 * correct / total:.1f}%"
        for (task, metric), (correct, total) in rows.items()
    ) + "\n"


def interview(initial, fact):
    return {"cases": 8, "summary": {"initial": initial, "verified_fact": fact, "valid_state_and_reuse": 0}}


def memory(total, length8=1, short=1):
    return {
        "prompt_style": "semantic",
        "self_repair": False,
        "by_split": {
            "value_ood_len4": {"episodes": 80, "closed_loop_correct": short},
            "value_and_length_ood_len8": {"episodes": 80, "closed_loop_correct": length8},
            "value_and_length_ood_len16": {"episodes": 80, "closed_loop_correct": 0},
            "value_and_length_ood_len32": {"episodes": 80, "closed_loop_correct": 0},
            "wide_range_and_length_ood_len8": {"episodes": 80, "closed_loop_correct": total - short - length8},
        },
    }


def trace(trace_correct, both):
    return {
        "cases": 12,
        "summary": {
            "trace_correct": trace_correct,
            "answer_correct": both,
            "correct_trace_and_final": both,
        },
    }


def run(candidate_counts, candidate_initial, candidate_fact, semantic_total, candidate_trace=1):
    root = Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory() as temporary:
        temporary = Path(temporary)
        paths = {
            "raw_board": temporary / "raw.log", "candidate_board": temporary / "candidate.log",
            "raw_interview": temporary / "raw.json", "candidate_interview": temporary / "candidate.json",
            "raw_trace": temporary / "raw_trace.json", "candidate_trace": temporary / "candidate_trace.json",
            "default": temporary / "default.json", "semantic": temporary / "semantic.json",
            "reference": temporary / "reference.json", "out": temporary / "out.json",
        }
        paths["raw_board"].write_text(board(REQUIRED)); paths["candidate_board"].write_text(board(candidate_counts))
        paths["raw_interview"].write_text(json.dumps(interview(1, 1)))
        paths["candidate_interview"].write_text(json.dumps(interview(candidate_initial, candidate_fact)))
        paths["raw_trace"].write_text(json.dumps(trace(0, 0)))
        paths["candidate_trace"].write_text(json.dumps(trace(candidate_trace, candidate_trace)))
        default = memory(30); default["prompt_style"] = "default"
        paths["default"].write_text(json.dumps(default)); paths["semantic"].write_text(json.dumps(memory(semantic_total)))
        paths["reference"].write_text(json.dumps(memory(21)))
        subprocess.run([
            sys.executable, str(root / "evaluate_v9_memory_mix.py"),
            "--raw-board", str(paths["raw_board"]), "--candidate-board", str(paths["candidate_board"]),
            "--raw-interview", str(paths["raw_interview"]), "--candidate-interview", str(paths["candidate_interview"]),
            "--raw-trace", str(paths["raw_trace"]), "--candidate-trace", str(paths["candidate_trace"]),
            "--memory-default", str(paths["default"]), "--memory-semantic", str(paths["semantic"]),
            "--semantic-reference", str(paths["reference"]), "--out", str(paths["out"]),
        ], check=True, capture_output=True, text=True)
        return json.loads(paths["out"].read_text())


def main():
    accepted = dict(REQUIRED)
    accepted[("gsm8k", "maj@8")] = (7, 200)
    accepted[("math500", "pass@1")] = (5, 200)
    assert run(accepted, 2, 1, 21)["verdict"] == "accept_followup"
    rejected = run(accepted, 2, 1, 20)
    assert rejected["verdict"] == "reject"
    assert "semantic_memory_regression_vs_r4" in rejected["reasons"]
    trace_rejected = run(accepted, 2, 1, 21, candidate_trace=0)
    assert trace_rejected["verdict"] == "reject"
    assert "no_visible_trace_and_final_gain" in trace_rejected["reasons"]
    print("V9 broad-memory promotion decision gate: passed")


if __name__ == "__main__":
    main()
