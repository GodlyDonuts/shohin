#!/usr/bin/env python3
"""Select concise, answer-verified OpenMathReasoning rows without leakage.

OpenMathReasoning is useful only if its long synthetic traces are reduced to a
student-sized curriculum. This curator checks the trace's own final answer
against ``expected_answer`` and decontaminates both the problem and trace. Its
``--dry-run`` mode is the mandatory yield measurement before any candidate is
allowed to be written.
"""
import argparse
import hashlib
import json
import os
import re
from pathlib import Path

from audit_training_text_overlap import first_overlap, load_eval_index


WORD = re.compile(r"\w+")


def normalized_problem(text):
    return " ".join(WORD.findall(str(text).lower()))


def clean_trace(trace):
    trace = str(trace or "").strip()
    trace = re.sub(r"^<think>\s*", "", trace)
    trace = re.sub(r"\s*</think>\s*", "\n", trace)
    return trace.strip()


def extract_final(text):
    """Extract the last boxed answer, or the explicit answer line as fallback."""
    text = str(text)
    index = text.rfind(r"\boxed")
    if index >= 0:
        start = text.find("{", index)
        if start >= 0:
            depth = 0
            for end in range(start, len(text)):
                if text[end] == "{":
                    depth += 1
                elif text[end] == "}":
                    depth -= 1
                    if depth == 0:
                        return text[start + 1:end].strip()
    matches = re.findall(r"answer is\s*[:\-]?\s*([^\n.$]+)", text, flags=re.I)
    return matches[-1].strip() if matches else None


def normalize_answer(value):
    value = str(value)
    value = re.sub(r"\\[dt]?frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}", r"(\1)/(\2)", value)
    value = re.sub(r"\\(?:text|mbox)\s*\{[^{}]*\}", "", value)
    value = re.sub(r"\\(?:left|right|displaystyle|mathrm|,|;|:|!|\s)", "", value)
    return value.replace("$", "").replace("\\", "").replace(" ", "").rstrip(".").strip().lower()


def numeric_answer(value):
    value = normalize_answer(value)
    try:
        return float(value)
    except ValueError:
        pass
    match = re.fullmatch(r"\(?(-?\d+(?:\.\d+)?)\)?/\(?(-?\d+(?:\.\d+)?)\)?", value)
    if match:
        try:
            return float(match.group(1)) / float(match.group(2))
        except ZeroDivisionError:
            return None
    return None


def answer_matches(trace, answer):
    predicted = extract_final(trace)
    expected = extract_final(answer) or str(answer)
    if predicted is None:
        return False
    if normalize_answer(predicted) == normalize_answer(expected):
        return True
    p_value, e_value = numeric_answer(predicted), numeric_answer(expected)
    return p_value is not None and e_value is not None and abs(p_value - e_value) < 1e-4


def parsed_rate(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def truthy(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--evals", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--out", help="fresh candidate JSONL; omit with --dry-run")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dataset", default="nvidia/OpenMathReasoning")
    parser.add_argument("--config", default="default")
    parser.add_argument("--split", default="cot")
    parser.add_argument("--max-seen", type=int, default=0, help="0 means stream the full split")
    parser.add_argument("--max-keep", type=int, default=0, help="0 means do not cap accepted rows")
    parser.add_argument("--max-problem-tokens", type=int, default=512)
    parser.add_argument("--max-trace-tokens", type=int, default=512)
    parser.add_argument("--min-pass-rate", type=float)
    parser.add_argument("--exclude-used-in-kaggle", action="store_true",
                        help="exclude the source's AIMO-2-training subset; off by default because project eval decontamination is the relevant leakage control")
    parser.add_argument("--ngram", type=int, default=13)
    args = parser.parse_args()
    if args.max_seen < 0 or args.max_keep < 0 or args.max_problem_tokens <= 0 or args.max_trace_tokens <= 0:
        raise ValueError("limits must be positive, or zero only for max-seen/max-keep")
    if args.dry_run == bool(args.out):
        raise SystemExit("use exactly one of --dry-run or --out")

    report_path = Path(args.report)
    if report_path.exists():
        raise SystemExit(f"refusing to overwrite report: {report_path}")
    out_path = Path(args.out) if args.out else None
    if out_path and out_path.exists():
        raise SystemExit(f"refusing to overwrite candidate: {out_path}")

    from datasets import load_dataset
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(args.tokenizer)
    exact, eval_ngrams, eval_files = load_eval_index(args.evals, args.ngram)
    stats = {
        "seen": 0,
        "kept": 0,
        "missing": 0,
        "used_in_kaggle_seen": 0,
        "used_in_kaggle_excluded": 0,
        "duplicate_problem": 0,
        "long_problem": 0,
        "long_trace": 0,
        "answer_mismatch": 0,
        "low_pass_rate": 0,
        "contam_exact_problem": 0,
        "contam_exact_trace": 0,
        "contam_ngram_problem": 0,
        "contam_ngram_trace": 0,
    }
    seen_problems = set()
    temporary = out_path.with_suffix(out_path.suffix + ".partial") if out_path else None
    if temporary and temporary.exists():
        raise SystemExit(f"stale candidate partial exists: {temporary}")

    destination = temporary.open("w") if temporary else None
    try:
        stream = load_dataset(args.dataset, name=args.config, split=args.split, streaming=True)
        for row in stream:
            if args.max_seen and stats["seen"] >= args.max_seen:
                break
            stats["seen"] += 1
            if stats["seen"] % 10_000 == 0:
                print(f"[openmath-reasoning] seen={stats['seen']:,} kept={stats['kept']:,}", flush=True)
            problem = str(row.get("problem") or "").strip()
            trace = clean_trace(row.get("generated_solution"))
            answer = str(row.get("expected_answer") or "").strip()
            if not problem or not trace or not answer:
                stats["missing"] += 1
                continue
            used_in_kaggle = truthy(row.get("used_in_kaggle"))
            if used_in_kaggle:
                stats["used_in_kaggle_seen"] += 1
                if args.exclude_used_in_kaggle:
                    stats["used_in_kaggle_excluded"] += 1
                    continue
            if args.min_pass_rate is not None:
                rate = parsed_rate(row.get("pass_rate_72b_tir"))
                if rate is None or rate < args.min_pass_rate:
                    stats["low_pass_rate"] += 1
                    continue
            key = hashlib.sha1(normalized_problem(problem).encode()).hexdigest()
            if key in seen_problems:
                stats["duplicate_problem"] += 1
                continue
            problem_hit = first_overlap(problem, exact, eval_ngrams, args.ngram)
            if problem_hit == "exact":
                stats["contam_exact_problem"] += 1
                continue
            if problem_hit == "ngram":
                stats["contam_ngram_problem"] += 1
                continue
            trace_hit = first_overlap(trace, exact, eval_ngrams, args.ngram)
            if trace_hit == "exact":
                stats["contam_exact_trace"] += 1
                continue
            if trace_hit == "ngram":
                stats["contam_ngram_trace"] += 1
                continue
            if len(tokenizer.encode(problem).ids) > args.max_problem_tokens:
                stats["long_problem"] += 1
                continue
            if len(tokenizer.encode(trace).ids) > args.max_trace_tokens:
                stats["long_trace"] += 1
                continue
            if not answer_matches(trace, answer):
                stats["answer_mismatch"] += 1
                continue
            seen_problems.add(key)
            stats["kept"] += 1
            if destination:
                response = f"<think>{trace}</think>\nThe answer is {answer}."
                destination.write(json.dumps({
                    "question": problem,
                    "response": response,
                    "answer": answer,
                    "source": f"openmath_reasoning_{args.split}",
                    "training_group": "math",
                    "verification": "expected_answer_trace_final",
                    "problem_source": row.get("problem_source"),
                    "generation_model": row.get("generation_model"),
                    "used_in_kaggle": used_in_kaggle,
                }, ensure_ascii=False) + "\n")
            if args.max_keep and stats["kept"] >= args.max_keep:
                break
    finally:
        if destination:
            destination.close()
    if temporary:
        os.replace(temporary, out_path)

    report = {
        "schema": "shohin-openmath-reasoning-selection-v1",
        "mode": "dry_run" if args.dry_run else "candidate",
        "dataset": args.dataset,
        "config": args.config,
        "split": args.split,
        "candidate": str(out_path.resolve()) if out_path else None,
        "max_seen": args.max_seen,
        "max_keep": args.max_keep,
        "max_problem_tokens": args.max_problem_tokens,
        "max_trace_tokens": args.max_trace_tokens,
        "min_pass_rate": args.min_pass_rate,
        "exclude_used_in_kaggle": args.exclude_used_in_kaggle,
        "ngram": args.ngram,
        "eval_files": eval_files,
        "eval_exact_prompts": len(exact),
        "eval_ngrams": len(eval_ngrams),
        **stats,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    partial_report = report_path.with_suffix(report_path.suffix + ".partial")
    partial_report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    partial_report.replace(report_path)
    print(json.dumps(report, sort_keys=True), flush=True)
    # See probe_reasoning_source.py: streaming datasets can retain worker
    # threads after all artifacts are closed and atomically committed.
    os._exit(0)


if __name__ == "__main__":
    main()
