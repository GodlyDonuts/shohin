#!/usr/bin/env python3
"""Apply the joint broad-capability and semantic-memory gate for V9.

The V9 hypothesis is deliberately stronger than "the model obeys a memory
format": a low-share memory curriculum must preserve ordinary language/code
behavior while retaining at least the semantic working-memory transfer reached
by the r4 scratch control. This program writes a verdict only; it never
promotes weights or schedules further training.
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
SEMANTIC_LENGTH8 = "value_and_length_ood_len8"
SEMANTIC_SHORT = "value_ood_len4"


def board_metrics(path):
    metrics = {}
    for line in Path(path).read_text(errors="replace").splitlines():
        match = METRIC_RE.search(line.strip())
        if match:
            metrics[(match.group(1), match.group(2))] = {
                "correct": int(match.group(3)),
                "total": int(match.group(4)),
            }
    return metrics


def interview(path):
    payload = json.loads(Path(path).read_text())
    summary = payload.get("summary", {})
    return {
        "cases": int(payload.get("cases", 0)),
        "initial": int(summary.get("initial", 0)),
        "verified_fact": int(summary.get("verified_fact", 0)),
        "valid_state_and_reuse": int(summary.get("valid_state_and_reuse", 0)),
    }


def memory(path):
    payload = json.loads(Path(path).read_text())
    splits = payload.get("by_split", {})
    return {
        "episodes": sum(int(row.get("episodes", 0)) for row in splits.values()),
        "closed_loop": sum(int(row.get("closed_loop_correct", 0)) for row in splits.values()),
        "length8": int(splits.get(SEMANTIC_LENGTH8, {}).get("closed_loop_correct", 0)),
        "short": int(splits.get(SEMANTIC_SHORT, {}).get("closed_loop_correct", 0)),
        "prompt_style": payload.get("prompt_style"),
        "self_repair": bool(payload.get("self_repair")),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-board", required=True)
    parser.add_argument("--candidate-board", required=True)
    parser.add_argument("--raw-interview", required=True)
    parser.add_argument("--candidate-interview", required=True)
    parser.add_argument("--memory-default", required=True)
    parser.add_argument("--memory-semantic", required=True)
    parser.add_argument("--semantic-reference", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    raw, candidate = board_metrics(args.raw_board), board_metrics(args.candidate_board)
    raw_interview, candidate_interview = interview(args.raw_interview), interview(args.candidate_interview)
    default, semantic, reference = memory(args.memory_default), memory(args.memory_semantic), memory(args.semantic_reference)
    missing = [f"{task}:{metric}" for task, metric in REQUIRED if key_not_present(raw, candidate, task, metric)]
    reasons, improvements, regressions = [], [], []
    if missing:
        reasons.append("missing_board_metrics:" + ",".join(missing))
    else:
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
    if raw_interview["cases"] != 8 or candidate_interview["cases"] != 8:
        reasons.append("missing_or_changed_direct_interview")
    if candidate_interview["initial"] <= raw_interview["initial"]:
        reasons.append("no_direct_initial_gain")
    if candidate_interview["verified_fact"] < raw_interview["verified_fact"]:
        reasons.append("verified_fact_regression")
    if default["prompt_style"] != "default" or semantic["prompt_style"] != "semantic":
        reasons.append("wrong_memory_prompt_style")
    if default["self_repair"] or semantic["self_repair"]:
        reasons.append("memory_eval_uses_self_repair")
    if semantic["episodes"] != 400 or reference["episodes"] != 400:
        reasons.append("incomplete_semantic_memory_eval")
    if semantic["closed_loop"] < reference["closed_loop"]:
        reasons.append("semantic_memory_regression_vs_r4")
    if semantic["length8"] <= 0 or semantic["short"] <= 0:
        reasons.append("no_semantic_memory_length_transfer")
    result = {
        "audit": "v9_broad_memory_promotion_v1",
        "rule": {
            "requires": [
                "all five fixed public metrics",
                "at least two public metric improvements",
                "no HumanEval or MBPP regression",
                "eight-case direct interview with initial gain",
                "no verified-fact regression",
                "400-case default and semantic memory reports without self-repair",
                "semantic memory at least matches r4 scratch and has nonzero length-8 transfer",
            ],
            "explicitly_not_sufficient": [
                "default-format working-memory score",
                "state marker emission alone",
                "self-repair that a controller selected or corrected",
            ],
        },
        "raw_interview": raw_interview,
        "candidate_interview": candidate_interview,
        "memory_default": default,
        "memory_semantic": semantic,
        "memory_semantic_reference": reference,
        "public_improvements": improvements,
        "public_regressions": regressions,
        "verdict": "accept_followup" if not reasons else "reject",
        "reasons": reasons,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


def key_not_present(raw, candidate, task, metric):
    return (task, metric) not in raw or (task, metric) not in candidate


if __name__ == "__main__":
    main()
