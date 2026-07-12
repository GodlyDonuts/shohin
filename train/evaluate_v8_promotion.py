#!/usr/bin/env python3
"""Apply the precommitted raw-to-V8 transfer rule to frozen evaluator outputs.

This deliberately produces only an ``accept_followup`` or ``reject`` decision.
No SFT experiment may promote itself into a flagship checkpoint or future mix.
"""
import argparse
import json
import re
from pathlib import Path


METRIC_RE = re.compile(
    r"^(gsm8k|math500|humaneval|mbpp)\s+"
    r"(maj@\d+|pass@\d+)\s+"
    r"ckpt=.*?\s+step=.*?\s+"
    r"(\d+)/(\d+)\s+=\s+([0-9.]+)%"
)
REQUIRED = (
    ("gsm8k", "maj@8"),
    ("gsm8k", "pass@1"),
    ("math500", "pass@1"),
    ("humaneval", "pass@1"),
    ("mbpp", "pass@1"),
)


def board_metrics(path):
    metrics = {}
    for line in Path(path).read_text(errors="replace").splitlines():
        match = METRIC_RE.search(line.strip())
        if match:
            metrics[(match.group(1), match.group(2))] = {
                "correct": int(match.group(3)),
                "total": int(match.group(4)),
                "accuracy": float(match.group(5)) / 100,
            }
    return metrics


def interview_summary(path):
    payload = json.loads(Path(path).read_text())
    summary = payload.get("summary", {})
    return {
        "cases": int(payload.get("cases", 0)),
        "initial": int(summary.get("initial", 0)),
        "verified_fact": int(summary.get("verified_fact", 0)),
        "valid_state_and_reuse": int(summary.get("valid_state_and_reuse", 0)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-board", required=True)
    parser.add_argument("--candidate-board", required=True)
    parser.add_argument("--raw-interview", required=True)
    parser.add_argument("--candidate-interview", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    raw = board_metrics(args.raw_board)
    candidate = board_metrics(args.candidate_board)
    missing = [f"{task}:{metric}" for task, metric in REQUIRED
               if (task, metric) not in raw or (task, metric) not in candidate]
    raw_interview = interview_summary(args.raw_interview)
    candidate_interview = interview_summary(args.candidate_interview)
    reasons = []
    if raw_interview["cases"] != 8 or candidate_interview["cases"] != 8:
        reasons.append("missing_or_changed_direct_interview")
    if missing:
        reasons.append("missing_board_metrics:" + ",".join(missing))

    improvements = []
    regressions = []
    if not missing:
        for key in REQUIRED:
            delta = candidate[key]["correct"] - raw[key]["correct"]
            if delta > 0:
                improvements.append({"task": key[0], "metric": key[1], "delta_correct": delta})
            elif delta < 0:
                regressions.append({"task": key[0], "metric": key[1], "delta_correct": delta})
        for key in (("humaneval", "pass@1"), ("mbpp", "pass@1")):
            if candidate[key]["correct"] < raw[key]["correct"]:
                reasons.append(f"code_regression:{key[0]}")
        if len(improvements) < 2:
            reasons.append("fewer_than_two_public_metric_improvements")

    if candidate_interview["initial"] <= raw_interview["initial"]:
        reasons.append("no_direct_initial_gain")
    if candidate_interview["verified_fact"] < raw_interview["verified_fact"]:
        reasons.append("verified_fact_regression")

    result = {
        "audit": "v8_transfer_promotion_v1",
        "rule": {
            "requires": [
                "all five fixed public metrics",
                "at least two public metric improvements",
                "no HumanEval or MBPP regression",
                "eight-case direct interview with initial gain",
                "no verified-fact regression",
            ],
            "explicitly_not_sufficient": [
                "constructed generator holdout",
                "format compliance",
                "state marker emission alone",
            ],
        },
        "raw_board": str(Path(args.raw_board)),
        "candidate_board": str(Path(args.candidate_board)),
        "raw_interview": raw_interview,
        "candidate_interview": candidate_interview,
        "public_improvements": improvements,
        "public_regressions": regressions,
        "verdict": "accept_followup" if not reasons else "reject",
        "reasons": reasons,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
