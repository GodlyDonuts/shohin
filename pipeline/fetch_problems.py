#!/usr/bin/env python
"""Fetch + normalize reasoning problem banks (TRAIN splits, disjoint from our evals) into a unified
schema for hermes_distill.py. Goal: BREADTH of reasoning — math, science, commonsense, logic — not
just math, so the distilled traces teach general step-by-step reasoning.

Each output row: {"question", "prompt", "gold", "answer_type", "source"}
  - prompt: the full teacher instruction (choices lettered for MC), ending-format enforced.
  - gold:   the verifiable answer (number / boxed expr / MC letter).
  - answer_type: gsm8k | boxed | mc  (consumed by hermes_distill.verify).

Robust: each dataset is wrapped in try/except so one bad source doesn't sink the rest.
  python fetch_problems.py --out-dir ../artifacts/problems --only arc_challenge commonsenseqa
"""
import argparse, json, os, random, re, traceback

MC_TMPL = ("Answer this multiple-choice question with concise, correct step-by-step reasoning "
           "(brief — no rambling), then end with a line exactly \"The answer is X.\" where X is the "
           "LETTER of the correct choice.\n\nQuestion: {q}\nChoices:\n{choices}")
BOXED_TMPL = ("Solve this problem with concise, correct step-by-step reasoning (brief). End with a line "
              "exactly \"The answer is X.\" where X is the final answer.\n\nProblem: {q}")


def mc_row(q, choices, gold_letter, source):
    lines = "\n".join(f"{chr(65+i)}) {c}" for i, c in enumerate(choices))
    return {"question": q.strip(), "prompt": MC_TMPL.format(q=q.strip(), choices=lines),
            "gold": gold_letter, "answer_type": "mc", "source": source}


def boxed_row(q, gold, source):
    return {"question": q.strip(), "prompt": BOXED_TMPL.format(q=q.strip()),
            "gold": gold, "answer_type": "boxed", "source": source}


def extract_boxed(text):
    i = text.rfind(r"\boxed")
    if i >= 0:
        j = text.find("{", i)
        if j >= 0:
            depth = 0
            for k in range(j, len(text)):
                if text[k] == "{":
                    depth += 1
                elif text[k] == "}":
                    depth -= 1
                    if depth == 0:
                        return text[j + 1:k].strip()
    return None


def write(rows, path):
    n = 0
    with open(path, "w") as f:
        for r in rows:
            if r is None:
                continue
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


# ---- per-dataset normalizers (return list of unified rows) ---------------------------------------

def get_arc(cfg, source):
    from datasets import load_dataset
    ds = load_dataset("allenai/ai2_arc", cfg, split="train")
    out = []
    for r in ds:
        texts, labels, ak = r["choices"]["text"], r["choices"]["label"], r["answerKey"]
        if ak not in labels:
            continue
        out.append(mc_row(r["question"], texts, chr(65 + labels.index(ak)), source))
    return out


def get_commonsenseqa():
    from datasets import load_dataset
    ds = load_dataset("tau/commonsense_qa", split="train")
    out = []
    for r in ds:
        texts, labels, ak = r["choices"]["text"], r["choices"]["label"], r["answerKey"]
        if not ak or ak not in labels:
            continue
        out.append(mc_row(r["question"], texts, chr(65 + labels.index(ak)), "csqa"))
    return out


def get_openbookqa():
    from datasets import load_dataset
    ds = load_dataset("allenai/openbookqa", "main", split="train")
    out = []
    for r in ds:
        texts, labels, ak = r["choices"]["text"], r["choices"]["label"], r["answerKey"]
        if ak not in labels:
            continue
        out.append(mc_row(r["question_stem"], texts, chr(65 + labels.index(ak)), "openbookqa"))
    return out


def get_sciq():
    from datasets import load_dataset
    ds = load_dataset("allenai/sciq", split="train")
    rng = random.Random(7)
    out = []
    for r in ds:
        correct = r["correct_answer"]
        opts = [correct, r["distractor1"], r["distractor2"], r["distractor3"]]
        if not all(opts):
            continue
        rng.shuffle(opts)
        out.append(mc_row(r["question"], opts, chr(65 + opts.index(correct)), "sciq"))
    return out


def get_logiqa():
    from datasets import load_dataset
    ds = load_dataset("lucasmccabe/logiqa", split="train")
    out = []
    for r in ds:
        q = (str(r.get("context", "")) + "\n" + str(r.get("query", ""))).strip()
        opts, gi = r.get("options"), r.get("correct_option")
        if not opts or gi is None or gi >= len(opts):
            continue
        out.append(mc_row(q, list(opts), chr(65 + int(gi)), "logiqa"))
    return out


def get_math():
    from datasets import load_dataset
    last = None
    for repo, cfg in [("lighteval/MATH", "all"), ("hendrycks/competition_math", None),
                      ("qwedsacf/competition_math", None), ("EleutherAI/hendrycks_math", None)]:
        try:
            ds = load_dataset(repo, cfg, split="train") if cfg else load_dataset(repo, split="train")
            out = []
            for r in ds:
                sol = r.get("solution") or r.get("answer") or ""
                b = extract_boxed(str(sol))
                if b:
                    out.append(boxed_row(r.get("problem") or r.get("question") or "", b, "math"))
            if out:
                print(f"  (MATH via {repo})")
                return out
        except Exception as e:
            last = e
            continue
    raise last or RuntimeError("no MATH source worked")


SOURCES = {
    "arc_challenge": lambda: get_arc("ARC-Challenge", "arc_challenge"),
    "arc_easy":      lambda: get_arc("ARC-Easy", "arc_easy"),
    "commonsenseqa": get_commonsenseqa,
    "openbookqa":    get_openbookqa,
    "sciq":          get_sciq,
    "logiqa":        get_logiqa,
    "math":          get_math,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--only", nargs="+", default=None, help="subset of: " + ", ".join(SOURCES))
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    names = a.only or list(SOURCES)
    summary = {}
    for name in names:
        if name not in SOURCES:
            print(f"[skip] unknown source {name}")
            continue
        try:
            rows = SOURCES[name]()
            n = write(rows, os.path.join(a.out_dir, f"{name}.jsonl"))
            summary[name] = n
            print(f"[ok] {name}: {n} problems")
        except Exception:
            print(f"[FAIL] {name}:\n{traceback.format_exc().splitlines()[-1]}")
            summary[name] = 0
    print("[summary]", json.dumps(summary))


if __name__ == "__main__":
    main()
