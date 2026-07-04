#!/usr/bin/env python
"""Curate a concise-CoT SFT corpus (master plan §6.4).

Primary source: NVIDIA OpenMathInstruct-2 (CC-BY-4.0, Llama-405B solutions). Keep
only CONCISE solutions (<= --max-sol-tokens under the Shohin tokenizer): the
OpenMathInstruct-2 ablation found concise formats beat verbose by 3.9% while 40%
shorter, and a 135M student cannot represent long traces.

    python sft_curate.py --tokenizer artifacts/shohin-tok-32k.json \\
        --out artifacts/sft/openmath2_concise.jsonl --max-keep 300000 --max-sol-tokens 400
"""
import argparse, json, os, sys
from tokenizers import Tokenizer
from datasets import load_dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dataset", default="nvidia/OpenMathInstruct-2")
    ap.add_argument("--split", default="train")
    ap.add_argument("--max-keep", type=int, default=300000)
    ap.add_argument("--max-sol-tokens", type=int, default=400)
    ap.add_argument("--char-prefilter", type=int, default=2400)  # cheap skip before tokenizing
    a = ap.parse_args()

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    tok = Tokenizer.from_file(a.tokenizer)
    ds = load_dataset(a.dataset, split=a.split, streaming=True)

    kept = seen = 0
    with open(a.out, "w") as f:
        for ex in ds:
            seen += 1
            prob = ex.get("problem") or ex.get("question") or ""
            sol = ex.get("generated_solution") or ex.get("solution") or ""
            ans = ex.get("expected_answer") or ex.get("answer")
            if not prob or not sol or len(sol) > a.char_prefilter:
                continue
            n = len(tok.encode(sol).ids)
            if n > a.max_sol_tokens:
                continue
            f.write(json.dumps({
                "problem": prob, "solution": sol, "answer": ans,
                "source": ex.get("problem_source"), "sol_tokens": n,
            }, ensure_ascii=False) + "\n")
            kept += 1
            if kept % 20000 == 0:
                print(f"  kept {kept}/{seen}", file=sys.stderr)
            if kept >= a.max_keep:
                break
    print(f"[sft] kept {kept} concise / {seen} seen -> {a.out}")


if __name__ == "__main__":
    main()
