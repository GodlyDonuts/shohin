#!/usr/bin/env python
"""Build a SELF-CORRECTION SFT set (ablation-gated): teach the model to notice a slip in its own
arithmetic and fix it. Constructed on arithmetic chains where ground truth is exact, so every trace
is verifiable and every stated final answer is correct.

Design (to teach the *repair move* without teaching a second-guessing tic):
  * ~60% clean correct step-by-step (confidence: most reasoning should NOT be second-guessed);
  * ~40% inject ONE believable slip (off-by-small / wrong-op / transposition) at a random step,
    catch it with VARIED phrasing, recompute correctly, and continue to the true answer;
  * both end with "The answer is <true>." matching eval_suite's extractor.

Kept SEPARATE from the baseline SFT (self_correct.jsonl) so its effect can be measured as an
ablation (baseline vs baseline+self-correction), not assumed.

  python curate_selfcorrect.py --out artifacts/sft/self_correct.jsonl --n 15000 --seed 7
"""
import argparse, json, random

CATCH = ["Wait, that's not right.", "Hmm, let me double-check that.", "Let me recompute that step.",
         "Actually, I made an error there.", "That doesn't look right — let me redo it.",
         "Hold on, let me verify that."]


def apply(op, x, y):
    return x + y if op == "+" else x - y if op == "-" else x * y


def believable_wrong(op, x, y, rng):
    """A plausible slip: off-by-small, or (for *) an adjacent product."""
    true = apply(op, x, y)
    if op == "*":
        cand = [apply(op, x, y + rng.choice([-1, 1])), apply(op, x + rng.choice([-1, 1]), y), true + rng.choice([-10, 10])]
    else:
        cand = [true + rng.choice([-2, -1, 1, 2, 10, -10])]
    w = rng.choice(cand)
    return w if w != true else true + 1


def make(rng):
    n = rng.randint(2, 4)                      # 2-4 terms
    nums = [rng.randint(2, 20) if rng.random() < 0.4 else rng.randint(2, 99) for _ in range(n)]
    ops = [rng.choice(["+", "+", "-", "*"]) for _ in range(n - 1)]
    # left-to-right evaluation (matches how the trace narrates)
    q = str(nums[0])
    for op, v in zip(ops, nums[1:]):
        q += f" {op} {v}"
    question = f"Calculate: {q}"

    acc = nums[0]
    err_step = rng.randint(0, n - 2) if rng.random() < 0.4 else -1   # which step (if any) slips
    lines = [f"Start with {nums[0]}."]
    for i, (op, v) in enumerate(zip(ops, nums[1:])):
        true = apply(op, acc, v)
        verb = {"+": "add", "-": "subtract", "*": "multiply by"}[op]
        if i == err_step:
            wrong = believable_wrong(op, acc, v, rng)
            lines.append(f"{verb.capitalize()} {v}: {acc} {op} {v} = {wrong}.")
            lines.append(f"{rng.choice(CATCH)} {acc} {op} {v} = {true}.")
        else:
            lines.append(f"{verb.capitalize()} {v}: {acc} {op} {v} = {true}.")
        acc = true
    lines.append(f"The answer is {acc}.")
    return {"question": question, "response": " ".join(lines), "answer": str(acc),
            "source": "self_correct", "corrected": err_step >= 0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=15000)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    import os
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    ncorr = 0
    with open(a.out, "w") as f:
        for _ in range(a.n):
            ex = make(rng)
            ncorr += ex["corrected"]
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"[self-correct] wrote {a.n} examples ({ncorr} with a caught-and-fixed slip, "
          f"{a.n - ncorr} clean) -> {a.out}")


if __name__ == "__main__":
    main()
