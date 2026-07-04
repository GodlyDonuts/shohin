#!/usr/bin/env python
"""Filter a JSONL corpus against the eval 13-gram set (master plan §6.5).

Drops any doc whose text (question and/or solution) contains a 13-gram present in
the eval sets. This is the OPERATIONAL armor — run on EVERY training corpus before
it's used. Publishes kept/dropped counts.

    python decontam_filter.py --grams artifacts/evals/evalgrams.pkl \\
        --in artifacts/sft/openmath2_concise.jsonl --field problem --also-field solution \\
        --out artifacts/sft/openmath2_concise.clean.jsonl
"""
import argparse, json, pickle, re


def grams(text, n):
    w = re.findall(r"\w+", text.lower())
    for i in range(len(w) - n + 1):
        yield " ".join(w[i:i + n])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grams", required=True)
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--field", default="question")
    ap.add_argument("--also-field", default=None, help="second field to check (e.g. solution)")
    a = ap.parse_args()

    d = pickle.load(open(a.grams, "rb"))
    S, n = d["grams"], d["n"]
    kept = dropped = 0
    with open(a.inp) as fin, open(a.out, "w") as fout:
        for line in fin:
            try:
                rec = json.loads(line)
            except Exception:
                continue
            text = str(rec.get(a.field, ""))
            if a.also_field:
                text += " " + str(rec.get(a.also_field, ""))
            if any(g in S for g in grams(text, n)):
                dropped += 1
                continue
            fout.write(line)
            kept += 1
    total = kept + dropped
    print(f"[decontam-filter] kept {kept}, dropped {dropped} "
          f"({dropped/max(total,1):.3%}) -> {a.out}")


if __name__ == "__main__":
    main()
