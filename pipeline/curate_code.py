#!/usr/bin/env python
"""Curate an EXECUTION-VERIFIED code SFT set (the SFT mix is math-only otherwise, so HumanEval/MBPP
stay weak). Source: MBPP train+validation splits — DISJOINT from the 500-problem test set we
benchmark on. Every kept example's reference code is run against its own tests; only passing ones
are kept. Decontaminated vs the HumanEval/MBPP TEST questions. Emits {question, response} where the
response is the working function (a code response, not a 'The answer is X' math response).

  python curate_code.py --evals artifacts/evals --out artifacts/sft/code.jsonl
"""
import argparse, json, os, re, subprocess, sys, tempfile
from datasets import load_dataset

WORD = re.compile(r"\w+")


def grams(text, n=13):
    w = WORD.findall(text.lower())
    if len(w) < n:
        yield " ".join(w)
    else:
        for i in range(len(w) - n + 1):
            yield " ".join(w[i:i + n])


def run_ok(program, timeout=8):
    fd, path = tempfile.mkstemp(suffix=".py")
    try:
        os.write(fd, program.encode()); os.close(fd)
        r = subprocess.run([sys.executable, path], capture_output=True, timeout=timeout,
                           text=True, cwd=tempfile.gettempdir())
        return r.returncode == 0
    except Exception:
        return False
    finally:
        try: os.unlink(path)
        except OSError: pass


def build_test_grams(evals_dir, n=13):
    S = set()
    for name in ("mbpp_full.jsonl", "humaneval_full.jsonl"):
        p = os.path.join(evals_dir, name)
        if not os.path.exists(p):
            continue
        for line in open(p):
            r = json.loads(line)
            q = r.get("text") or r.get("prompt") or r.get("question") or ""
            for g in grams(q, n):
                S.add(g)
    print(f"[decontam] {len(S):,} test n-grams", file=sys.stderr)
    return S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--evals", default=None)
    ap.add_argument("--splits", nargs="+", default=["train", "validation"])
    ap.add_argument("--ngram", type=int, default=13)
    a = ap.parse_args()

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    test_grams = build_test_grams(a.evals, a.ngram) if a.evals else set()

    kept = seen = drop_fail = drop_contam = 0
    with open(a.out, "w") as f:
        for split in a.splits:
            ds = load_dataset("google-research-datasets/mbpp", "full", split=split)
            for r in ds:
                seen += 1
                text, code = r.get("text", ""), r.get("code", "")
                tests = r.get("test_list", [])
                setup = r.get("test_setup_code", "") or ""
                if not text or not code or not tests:
                    continue
                if test_grams and any(g in test_grams for g in grams(text, a.ngram)):
                    drop_contam += 1
                    continue
                program = code + "\n" + setup + "\n" + "\n".join(tests) + "\n"
                if not run_ok(program):                       # execution-verify the reference solution
                    drop_fail += 1
                    continue
                f.write(json.dumps({"question": text, "response": code.strip(),
                                    "source": f"mbpp_{split}"}, ensure_ascii=False) + "\n")
                kept += 1
    print(f"[code] kept {kept:,} / {seen:,} seen  (dropped: exec-fail={drop_fail:,} "
          f"contaminated={drop_contam:,}) -> {a.out}")


if __name__ == "__main__":
    main()
