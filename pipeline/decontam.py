#!/usr/bin/env python
"""13-gram decontamination (master plan §6.5).

Build the set of 13-word-grams from every eval set, then scan any training text
and report the hit rate + offending grams. Publish the counts — this is the
claim's armor.

Uses a plain set (exact, correct) for moderate scale; for the full corpus swap in
a real Bloom filter (rbloom / pybloomfiltermmap) with the same gram function.

    python decontam.py build --evals gsm8k_test.txt math500.txt --out evalgrams.pkl
    python decontam.py scan  --grams evalgrams.pkl --text some_training_shard.txt
"""
import argparse, re, pickle, sys

GRAM_N = 13


def grams(text, n=GRAM_N):
    w = re.findall(r"\w+", text.lower())
    for i in range(len(w) - n + 1):
        yield " ".join(w[i:i + n])


def build(evals, out, n=GRAM_N):
    S = set()
    for p in evals:
        with open(p, errors="ignore") as f:
            for line in f:
                S.update(grams(line, n))
    with open(out, "wb") as f:
        pickle.dump({"n": n, "grams": S}, f)
    print(f"[decontam] built {len(S):,} {n}-grams from {len(evals)} eval files -> {out}")


def scan(grams_path, text_path):
    with open(grams_path, "rb") as f:
        d = pickle.load(f)
    S, n = d["grams"], d["n"]
    hits = docs = matched = 0
    with open(text_path, errors="ignore") as f:
        for line in f:
            docs += 1
            hit = False
            for g in grams(line, n):
                if g in S:
                    hit = True
                    matched += 1
            if hit:
                hits += 1
    rate = hits / max(docs, 1)
    print(f"[decontam] {text_path}: {hits}/{docs} docs contaminated "
          f"({rate:.4%}), {matched} gram hits")
    return hits, docs


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--evals", nargs="+", required=True)
    b.add_argument("--out", required=True)
    s = sub.add_parser("scan")
    s.add_argument("--grams", required=True)
    s.add_argument("--text", required=True)
    a = ap.parse_args()
    if a.cmd == "build":
        build(a.evals, a.out)
    else:
        scan(a.grams, a.text)


if __name__ == "__main__":
    main()
