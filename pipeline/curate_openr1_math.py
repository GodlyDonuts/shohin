#!/usr/bin/env python3
"""Build a concise, verified, decontaminated OpenR1-Math SFT candidate.

The official default subset provides per-generation correctness metadata. This
curator keeps one completed trace per normalized problem, preferring Math Verify
over LLM-judge verification, trims it to a small-model-friendly token budget,
and rejects any overlap with the project's held-out evaluation prompts.
"""
import argparse
import hashlib
import json
import os
import re
from pathlib import Path

from tokenizers import Tokenizer


WORD = re.compile(r"\w+")


def normalized_question(question):
    return " ".join(WORD.findall(str(question).lower()))


def grams(text, n):
    words = WORD.findall(str(text).lower())
    if len(words) < n:
        yield " ".join(words)
    else:
        for index in range(len(words) - n + 1):
            yield " ".join(words[index:index + n])


def load_eval_grams(evals_dir, n):
    paths = ("gsm8k.jsonl", "math500.jsonl", "gsm8k_platinum.jsonl",
             "humaneval_full.jsonl", "mbpp_full.jsonl")
    fields = ("question", "problem", "prompt", "text")
    result = set()
    for name in paths:
        path = Path(evals_dir) / name
        if not path.exists():
            continue
        for line in path.open(errors="replace"):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            question = next((row[field] for field in fields if row.get(field)), "")
            result.update(grams(question, n))
    return result


def at(values, index):
    return bool(values[index]) if isinstance(values, list) and index < len(values) else False


def choose_generation(row):
    """Return (trace, verification) or None using strict per-trace metadata."""
    generations = row.get("generations")
    if not isinstance(generations, list):
        return None
    complete = row.get("is_reasoning_complete")
    math_verify = row.get("correctness_math_verify")
    llama = row.get("correctness_llama")
    for verification, flags in (("math_verify", math_verify), ("llama_judge", llama)):
        for index, trace in enumerate(generations):
            if not at(flags, index):
                continue
            if isinstance(complete, list) and not at(complete, index):
                continue
            trace = str(trace or "").strip()
            if trace:
                return trace, verification
    return None


def clean_trace(trace):
    trace = str(trace).strip()
    trace = re.sub(r"^<think>\s*", "", trace)
    trace = re.sub(r"\s*</think>\s*", "\n", trace)
    return trace.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--evals", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--report", default=None)
    parser.add_argument("--dataset", default="open-r1/OpenR1-Math-220k")
    parser.add_argument("--config", default="default")
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-keep", type=int, default=90_000)
    parser.add_argument("--max-problem-tokens", type=int, default=512)
    parser.add_argument("--max-trace-tokens", type=int, default=512)
    parser.add_argument("--ngram", type=int, default=13)
    args = parser.parse_args()

    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_suffix(out.suffix + ".partial")
    if temporary.exists():
        raise SystemExit(f"stale partial output exists: {temporary}")

    from datasets import load_dataset

    tokenizer = Tokenizer.from_file(args.tokenizer)
    eval_grams = load_eval_grams(args.evals, args.ngram)
    stats = dict(seen=0, kept=0, duplicate=0, no_verified_trace=0, long_problem=0,
                 long_trace=0, contaminated=0, missing=0, math_verify=0, llama_judge=0)
    seen_questions = set()
    stream = load_dataset(args.dataset, name=args.config, split=args.split, streaming=True)
    with temporary.open("w") as destination:
        for row in stream:
            stats["seen"] += 1
            problem = str(row.get("problem") or "").strip()
            answer = str(row.get("answer") or "").strip()
            if not problem or not answer:
                stats["missing"] += 1
                continue
            key = hashlib.sha1(normalized_question(problem).encode("utf-8")).hexdigest()
            if key in seen_questions:
                stats["duplicate"] += 1
                continue
            if any(gram in eval_grams for gram in grams(problem, args.ngram)):
                stats["contaminated"] += 1
                continue
            if len(tokenizer.encode(problem).ids) > args.max_problem_tokens:
                stats["long_problem"] += 1
                continue
            selected = choose_generation(row)
            if selected is None:
                stats["no_verified_trace"] += 1
                continue
            trace, verification = selected
            trace = clean_trace(trace)
            if len(tokenizer.encode(trace).ids) > args.max_trace_tokens:
                stats["long_trace"] += 1
                continue
            seen_questions.add(key)
            response = f"<think>{trace}</think>\nThe answer is {answer}."
            destination.write(json.dumps({
                "question": problem,
                "response": response,
                "answer": answer,
                "source": "openr1_math_default",
                "training_group": "math",
                "verification": verification,
                "uuid": row.get("uuid"),
            }, ensure_ascii=False) + "\n")
            stats["kept"] += 1
            stats[verification] += 1
            if stats["kept"] % 10_000 == 0:
                print(f"[openr1] kept={stats['kept']:,} seen={stats['seen']:,}", flush=True)
            if stats["kept"] >= args.max_keep:
                break
    os.replace(temporary, out)
    report = dict(dataset=args.dataset, config=args.config, split=args.split,
                  max_problem_tokens=args.max_problem_tokens,
                  max_trace_tokens=args.max_trace_tokens, ngram=args.ngram,
                  eval_grams=len(eval_grams), **stats)
    report_path = Path(args.report) if args.report else out.with_suffix(out.suffix + ".report.json")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
