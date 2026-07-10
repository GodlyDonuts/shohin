#!/usr/bin/env python
"""Build a frozen, auditable Shohin SFT mix from verified trace sources.

The distillers write live JSONL files while they run. Training directly from those files makes
reproducibility and handoff hard, so this script snapshots the current verified traces into one
deduplicated mix plus a report. It is deliberately conservative:

  - malformed/partial live-writer lines are skipped;
  - examples must have question + response;
  - responses are capped by characters/words to keep tiny-student CoT concise;
  - questions are deduped by normalized hash, keeping the highest-priority source.

Default output:
  artifacts/sft/sft_mix_core.jsonl
  artifacts/sft/sft_mix_core.report.json
"""
import argparse
import collections
import glob
import hashlib
import json
import os
import pickle
import random
import re
from pathlib import Path


DEFAULT_INPUTS = [
    "artifacts/sft/openmath2.jsonl",
    "artifacts/sft/rgym.jsonl",
    "artifacts/sft/code.jsonl",
    "artifacts/sft/hy3_reasoning_claude.jsonl",
    "artifacts/sft/hy3_reasoning_glm.jsonl",
    "artifacts/sft/hy3_reasoning_nemotron.jsonl",
    "artifacts/sft/hy3_reasoning.jsonl",
    "artifacts/sft/hy3_reasoning_minimax.jsonl",
    "artifacts/sft/hy3_gsm8k.jsonl",
    "artifacts/sft/hy3_arc_challenge.jsonl",
]

SOURCE_PRIORITY = {
    # Execution-verified/code first where applicable; then strongest teachers.
    "reasoning_gym_trace": 98,
    "mbpp_train": 100,
    "mbpp_validation": 100,
    "code": 100,
    "claude": 92,
    "glm": 90,
    "nemotron": 88,
    "hy3": 82,
    "hy3_gsm8k": 82,
    "hy3_arc": 82,
    "minimax": 70,
    "self_correct": 45,
}


WORD = re.compile(r"\w+")


def norm_question(q: str) -> str:
    return " ".join(WORD.findall(str(q).lower()))


def qhash(q: str) -> str:
    return hashlib.sha1(norm_question(q).encode("utf-8", "ignore")).hexdigest()[:16]


def response_stats(resp: str) -> tuple[int, int]:
    return len(resp), len(WORD.findall(resp))


def priority(row: dict, path: str) -> tuple[int, int]:
    src = str(row.get("source") or "").strip()
    base = SOURCE_PRIORITY.get(src, 60)
    # Shorter correct traces are better for this student, but only as a tie-break.
    _, words = response_stats(str(row.get("response") or ""))
    return base, -words


def read_problem_domains(path: str) -> dict[str, str]:
    domains = {}
    if not path or not os.path.exists(path):
        return domains
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            q = row.get("question")
            if q:
                domains[qhash(q)] = row.get("source") or row.get("domain") or "unknown"
    return domains


def read_eval_hashes(patterns: list[str]) -> set[str]:
    hashes = set()
    fields = ("question", "problem", "prompt", "task", "text")
    for pat in patterns:
        for path in sorted(glob.glob(pat)):
            with open(path, errors="replace") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line)
                    except Exception:
                        continue
                    q = ""
                    for field in fields:
                        if row.get(field):
                            q = str(row[field])
                            break
                    if q:
                        hashes.add(qhash(q))
    return hashes


def load_gram_set(path: str):
    if not path or not os.path.exists(path):
        return None, None
    with open(path, "rb") as f:
        d = pickle.load(f)
    return d["grams"], d["n"]


def has_eval_gram(text: str, gram_set, n: int) -> bool:
    if gram_set is None:
        return False
    words = WORD.findall(str(text).lower())
    return any(" ".join(words[i:i + n]) in gram_set for i in range(len(words) - n + 1))


def iter_rows(paths: list[str]):
    for pat in paths:
        for path in sorted(glob.glob(pat)) or [pat]:
            if not os.path.exists(path):
                yield path, None, "missing"
                continue
            with open(path) as f:
                for lineno, line in enumerate(f, 1):
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line)
                    except Exception:
                        yield path, None, f"bad_json:{lineno}"
                        continue
                    yield path, row, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="*", default=DEFAULT_INPUTS)
    ap.add_argument("--out", default="artifacts/sft/sft_mix_core.jsonl")
    ap.add_argument("--report", default=None)
    ap.add_argument("--problem-bank", default="artifacts/problems/combined.jsonl")
    ap.add_argument("--include-self-correct", action="store_true",
                    help="include self_correct.jsonl in the mix; default keeps it ablation-gated")
    ap.add_argument("--max-response-chars", type=int, default=4000)
    ap.add_argument("--max-response-words", type=int, default=750)
    ap.add_argument("--max-openmath", type=int, default=100000)
    ap.add_argument("--eval-glob", nargs="*", default=["artifacts/evals/*.jsonl"],
                    help="eval JSONLs whose exact prompt hashes must never enter the SFT mix")
    ap.add_argument("--decontam-grams", default="artifacts/evals/evalgrams.pkl",
                    help="optional eval n-gram pickle; if present, question+response hits are dropped")
    ap.add_argument("--no-eval-decontam", action="store_true",
                    help="disable eval exact/n-gram filtering; only use for debugging")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    inputs = list(args.inputs)
    if args.include_self_correct:
        inputs.append("artifacts/sft/self_correct.jsonl")

    domains = read_problem_domains(args.problem_bank)
    eval_hashes = set()
    gram_set, gram_n = None, None
    if not args.no_eval_decontam:
        eval_hashes = read_eval_hashes(args.eval_glob)
        gram_set, gram_n = load_gram_set(args.decontam_grams)
    kept_by_hash = {}
    report = {
        "inputs": inputs,
        "out": args.out,
        "limits": {
            "max_response_chars": args.max_response_chars,
            "max_response_words": args.max_response_words,
            "max_openmath": args.max_openmath,
            "include_self_correct": args.include_self_correct,
            "eval_glob": args.eval_glob,
            "decontam_grams": args.decontam_grams if gram_set is not None else None,
            "eval_hash_count": len(eval_hashes),
        },
        "seen_by_file": collections.Counter(),
        "kept_by_file": collections.Counter(),
        "kept_by_source": collections.Counter(),
        "kept_by_domain": collections.Counter(),
        "drops": collections.Counter(),
        "missing": [],
    }
    openmath_seen = 0

    for path, row, err in iter_rows(inputs):
        if err:
            if err == "missing":
                report["missing"].append(path)
            else:
                report["drops"][err.split(":", 1)[0]] += 1
            continue
        report["seen_by_file"][path] += 1
        q = str(row.get("question") or row.get("problem") or row.get("prompt") or "").strip()
        resp = str(row.get("response") or row.get("solution") or row.get("completion") or row.get("output") or "").strip()
        if not q or not resp:
            report["drops"]["missing_question_or_response"] += 1
            continue
        chars, words = response_stats(resp)
        if chars > args.max_response_chars or words > args.max_response_words:
            report["drops"]["too_long"] += 1
            continue
        src = str(row.get("source") or "")
        if path.endswith("openmath2.jsonl"):
            openmath_seen += 1
            if openmath_seen > args.max_openmath:
                report["drops"]["openmath_cap"] += 1
                continue
        h = qhash(q)
        if h in eval_hashes:
            report["drops"]["eval_exact_prompt"] += 1
            continue
        if has_eval_gram(q + "\n" + resp, gram_set, gram_n):
            report["drops"]["eval_ngram"] += 1
            continue
        clean = {
            "question": q,
            "response": resp,
            "source": src or Path(path).stem,
        }
        if row.get("answer") is not None:
            clean["answer"] = str(row.get("answer"))
        prev = kept_by_hash.get(h)
        cand = (priority(clean, path), path, clean)
        if prev is None or cand[0] > prev[0]:
            kept_by_hash[h] = cand
            if prev is not None:
                report["drops"]["dedup_replaced"] += 1
        else:
            report["drops"]["dedup_lower_priority"] += 1

    rows = [v[2] for v in kept_by_hash.values()]
    rng = random.Random(args.seed)
    rng.shuffle(rows)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    for _, path, row in kept_by_hash.values():
        report["kept_by_file"][path] += 1
        report["kept_by_source"][row.get("source", "?")] += 1
        report["kept_by_domain"][domains.get(qhash(row["question"]), "unmapped")] += 1
    report["kept_total"] = len(rows)

    # Convert Counters for JSON.
    for k in ("seen_by_file", "kept_by_file", "kept_by_source", "kept_by_domain", "drops"):
        report[k] = dict(report[k])
    report_path = args.report or args.out.replace(".jsonl", ".report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, sort_keys=True)

    print(f"[sft-mix] wrote {len(rows):,} examples -> {args.out}")
    print(f"[sft-mix] report -> {report_path}")
    print("[sft-mix] kept_by_source:", json.dumps(report["kept_by_source"], sort_keys=True))
    if report["missing"]:
        print("[sft-mix] missing:", ", ".join(report["missing"]))


if __name__ == "__main__":
    main()
