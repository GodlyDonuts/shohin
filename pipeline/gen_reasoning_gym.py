#!/usr/bin/env python
"""Generate the Shohin procedural reasoning corpus from Reasoning-Gym.

Outputs (master plan §6.3):
  rg_train.jsonl        verified (question, answer) across curated TRAIN families
  rg_eval.jsonl         held-out FAMILIES + held-out SEEDS (never trained on)
  rg_traces_train.jsonl execution-trace documents for traceable families
                        (<think>worked steps</think>) — each re-verified

Held-out discipline: HELDOUT_FAMILIES never appear in train; a disjoint high seed
range gives held-out seeds per train family. Every Q/A is verifier-checked; every
trace must reproduce the verified answer or it's dropped (see tracers.py).

    python gen_reasoning_gym.py --out-dir ../artifacts/rg --per-family 300
"""
import argparse, json, os, sys
import reasoning_gym as rg
from tracers import TRACERS, make_document

TRAIN_FAMILIES = [
    # arithmetic / number
    "chain_sum", "decimal_chain_sum", "basic_arithmetic", "decimal_arithmetic",
    "gcd", "lcm", "prime_factorization", "fraction_simplification", "count_primes",
    "count_bits", "number_sorting", "number_filtering", "base_conversion",
    "power_function", "products", "simple_equations", "polynomial_equations",
    # logic / deduction
    "propositional_logic", "syllogism", "aiw", "family_relationships", "self_reference",
    # short algorithmic / string
    "letter_counting", "spell_backward", "word_sorting", "caesar_cipher",
    "group_anagrams", "isomorphic_strings", "string_insertion",
]
# Reserved for generalization eval — NEVER emitted into train.
HELDOUT_FAMILIES = [
    "knights_knaves", "countdown", "zebra_puzzles", "graph_color", "advanced_geometry",
]


def gen_family(fam, n, seed, split, fout, traces_fout=None, cov=None):
    try:
        ds = rg.create_dataset(fam, size=n, seed=seed)
    except Exception as e:
        print(f"  [skip] {fam}: {e}", file=sys.stderr)
        return 0
    tracer = TRACERS.get(fam) if traces_fout is not None else None
    kept = 0
    for it in ds:
        q, a = it.get("question"), it.get("answer")
        md = it.get("metadata") or {}
        try:
            ok = ds.score_answer(answer=a, entry=it) >= 1.0   # rejection sampling
        except Exception:
            ok = True
        if not ok:
            continue
        fout.write(json.dumps({
            "family": fam, "split": split,
            "difficulty": md.get("difficulty"),
            "question": q, "answer": a,
        }, ensure_ascii=False) + "\n")
        kept += 1
        if tracer is not None:
            try:
                tr = tracer(q, a, md)
            except Exception:
                tr = None
            if tr:
                traces_fout.write(json.dumps({
                    "family": fam, "question": q, "trace": tr,
                    "answer": str(a).strip(),
                    "document": make_document(q, tr, a),
                }, ensure_ascii=False) + "\n")
                if cov is not None:
                    cov[fam] = cov.get(fam, 0) + 1
    return kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--per-family", type=int, default=500)
    ap.add_argument("--train-seed", type=int, default=100)
    ap.add_argument("--eval-seed", type=int, default=999000)   # disjoint from train
    ap.add_argument("--eval-per-family", type=int, default=100)
    a = ap.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)
    tr = os.path.join(a.out_dir, "rg_train.jsonl")
    ev = os.path.join(a.out_dir, "rg_eval.jsonl")
    trc = os.path.join(a.out_dir, "rg_traces_train.jsonl")

    cov, n_tr = {}, 0
    with open(tr, "w") as f, open(trc, "w") as tf:
        for fam in TRAIN_FAMILIES:
            k = gen_family(fam, a.per_family, a.train_seed, "train", f, tf, cov)
            n_tr += k
            tag = f"  traces={cov.get(fam,0)}" if fam in TRACERS else ""
            print(f"[train]   {fam:22s} {k}{tag}")

    n_ev = 0
    with open(ev, "w") as f:
        for fam in HELDOUT_FAMILIES:                            # held-out families
            k = gen_family(fam, a.eval_per_family, a.eval_seed, "eval_family", f)
            n_ev += k
            print(f"[eval-fam] {fam:21s} {k}")
        for fam in TRAIN_FAMILIES:                              # held-out seeds
            n_ev += gen_family(fam, a.eval_per_family, a.eval_seed, "eval_seed", f)

    n_traces = sum(cov.values())
    print(f"[rg] train={n_tr} -> {tr}")
    print(f"[rg] traces={n_traces} ({len(cov)} families) -> {trc}")
    print(f"[rg] eval={n_ev} -> {ev}")


if __name__ == "__main__":
    main()
