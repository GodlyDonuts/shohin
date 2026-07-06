#!/usr/bin/env python
"""Curate a VERIFIED, DECONTAMINATED, concise-CoT SFT corpus (master plan §6.4).

Source: NVIDIA OpenMathInstruct-2 (CC-BY-4.0). Four filters that separate SoTA reasoning data
from mediocre:
  1. CONCISE  — keep solutions <= --max-sol-tokens (a 135M student can't represent long traces;
                the OpenMathInstruct-2 ablation found concise beats verbose by 3.9% at 40% shorter).
  2. VERIFIED — the solution's final \\boxed{} answer must equal expected_answer (train only on
                reasoning that actually reaches the right answer).
  3. DECONTAMINATED — drop any problem sharing a 13-gram (or exact normalized text) with an eval
                question (gsm8k / math500 / humaneval / mbpp). We measured ~0.7% raw contamination.
  4. EVAL-ALIGNED FORMAT — emit {"question", "response"} where response is the CoT ending in an
                explicit "The answer is X." so it matches eval_suite.py's extractor.

    python sft_curate.py --tokenizer artifacts/shohin-tok-32k.json --evals artifacts/evals \\
        --out artifacts/sft/openmath2.jsonl --max-keep 200000 --max-sol-tokens 400
"""
import argparse, json, os, re, sys
from tokenizers import Tokenizer
from datasets import load_dataset

WORD = re.compile(r"\w+")


def grams(text, n=13):
    w = WORD.findall(text.lower())
    if len(w) < n:
        yield " ".join(w)                      # short text -> whole normalized string
    else:
        for i in range(len(w) - n + 1):
            yield " ".join(w[i:i + n])


def extract_boxed(text):
    i = text.rfind(r"\boxed")
    if i < 0:
        return None
    j = text.find("{", i)
    if j < 0:
        return None
    depth = 0
    for k in range(j, len(text)):
        if text[k] == "{":
            depth += 1
        elif text[k] == "}":
            depth -= 1
            if depth == 0:
                return text[j + 1:k].strip()
    return None


def norm_ans(s):
    return re.sub(r"\s+", "", str(s)).replace("$", "").replace("\\!", "") if s is not None else None


def build_eval_grams(evals_dir, n=13):
    """Collect 13-grams from every eval question so we can drop contaminated training problems."""
    S = set()
    fields = ("question", "problem", "prompt", "text")
    for name in ("gsm8k.jsonl", "math500.jsonl", "gsm8k_platinum.jsonl",
                 "humaneval_full.jsonl", "mbpp_full.jsonl"):
        p = os.path.join(evals_dir, name)
        if not os.path.exists(p):
            continue
        for line in open(p):
            try:
                r = json.loads(line)
            except Exception:
                continue
            q = next((r[f] for f in fields if r.get(f)), "")
            for g in grams(q, n):
                S.add(g)
    print(f"[decontam] {len(S):,} eval n-grams from {evals_dir}", file=sys.stderr)
    return S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--evals", default=None, help="dir of eval jsonls for decontamination")
    ap.add_argument("--dataset", default="nvidia/OpenMathInstruct-2")
    ap.add_argument("--split", default="train")
    ap.add_argument("--max-keep", type=int, default=200000)
    ap.add_argument("--max-sol-tokens", type=int, default=400)
    ap.add_argument("--char-prefilter", type=int, default=2400)
    ap.add_argument("--ngram", type=int, default=13)
    a = ap.parse_args()

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    tok = Tokenizer.from_file(a.tokenizer)
    eval_grams = build_eval_grams(a.evals, a.ngram) if a.evals else set()
    ds = load_dataset(a.dataset, split=a.split, streaming=True)

    kept = seen = drop_long = drop_unverified = drop_contam = 0
    with open(a.out, "w") as f:
        for ex in ds:
            seen += 1
            prob = (ex.get("problem") or ex.get("question") or "").strip()
            sol = (ex.get("generated_solution") or ex.get("solution") or "").strip()
            ans = ex.get("expected_answer") or ex.get("answer")
            if not prob or not sol or len(sol) > a.char_prefilter:
                continue
            if len(tok.encode(sol).ids) > a.max_sol_tokens:
                drop_long += 1
                continue
            if norm_ans(extract_boxed(sol)) != norm_ans(ans):     # verify: solution reaches the answer
                drop_unverified += 1
                continue
            if eval_grams and any(g in eval_grams for g in grams(prob, a.ngram)):
                drop_contam += 1
                continue
            response = f"{sol}\nThe answer is {ans}."            # explicit, eval-aligned final answer
            f.write(json.dumps({"question": prob, "response": response,
                                "answer": ans, "source": ex.get("problem_source")},
                               ensure_ascii=False) + "\n")
            kept += 1
            if kept % 20000 == 0:
                print(f"  kept {kept:,}/{seen:,}", file=sys.stderr)
            if kept >= a.max_keep:
                break
    print(f"[sft] kept {kept:,} / {seen:,} seen  (dropped: long={drop_long:,} "
          f"unverified={drop_unverified:,} contaminated={drop_contam:,}) -> {a.out}")


if __name__ == "__main__":
    main()
