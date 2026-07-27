#!/usr/bin/env python
"""Fetch eval test sets and build the decontamination gram set (master plan §6.5, §8).

Downloads the graded reasoning suite's TEST splits, saves each as jsonl, and builds
the 13-gram set over all eval questions+answers so every training corpus can be
scanned for contamination. Unknown/renamed dataset ids skip gracefully (logged).

    python fetch_evals.py --out-dir artifacts/evals
"""
import argparse, json, os, sys, pickle
from datasets import load_dataset
from decontam import grams, GRAM_N

# (name, hf_id, config, split, question_field, answer_field)
EVALS = [
    ("gsm8k",          "openai/gsm8k",                  "main", "test", "question", "answer"),
    ("gsm8k_platinum", "madrylab/gsm8k-platinum",       "main", "test", "question", "answer"),
    ("math500",        "HuggingFaceH4/MATH-500",        None,   "test", "problem",  "solution"),
    ("humaneval",      "openai/openai_humaneval",       None,   "test", "prompt",   "canonical_solution"),
    ("mbpp",           "google-research-datasets/mbpp",  "full", "test", "text",     "code"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    allgrams, summary = set(), []
    for name, hfid, cfg, split, qf, af in EVALS:
        try:
            kw = dict(split=split)
            if cfg:
                kw["name"] = cfg
            ds = load_dataset(hfid, **kw)
        except Exception as e:
            print(f"[skip] {name} ({hfid}): {repr(e)[:140]}", file=sys.stderr)
            summary.append((name, "SKIP"))
            continue
        path = os.path.join(a.out_dir, f"{name}.jsonl")
        n = 0
        with open(path, "w") as f:
            for ex in ds:
                q, an = str(ex.get(qf, "")), str(ex.get(af, ""))
                f.write(json.dumps({"question": q, "answer": an}, ensure_ascii=False) + "\n")
                for g in grams(q + " " + an):
                    allgrams.add(g)
                n += 1
        print(f"[eval] {name}: {n} rows -> {path}")
        summary.append((name, n))

    with open(os.path.join(a.out_dir, "evalgrams.pkl"), "wb") as f:
        pickle.dump({"n": GRAM_N, "grams": allgrams}, f)
    print(f"[decontam] {len(allgrams):,} {GRAM_N}-grams -> evalgrams.pkl")
    print("[summary]", summary)


if __name__ == "__main__":
    main()
