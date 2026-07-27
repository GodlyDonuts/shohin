#!/usr/bin/env python
"""Shohin reasoning eval suite — pass@1 (greedy) and self-consistency maj@k.

Math tasks (GSM8K, MATH-500) score by verified answer match; self-consistency samples k solutions
at temperature and majority-votes the extracted answer (a large, no-retrain benchmark lift for
reasoning models). Code tasks (HumanEval/MBPP) with execution live in eval_code.py.

  python eval_suite.py --ckpt flagship_out/best_step10000.model.pt \\
      --tokenizer ../artifacts/shohin-tok-32k.json --task gsm8k \\
      --data ../artifacts/evals/gsm8k.jsonl --n 40 --k 8 --temp 0.8
"""
import argparse, json, re, sys
from collections import Counter
import torch
from tokenizers import Tokenizer
from model import GPT, GPTConfig

# --------------------------------------------------------------------------- few-shot primers
GSM8K_SHOTS = [
    ("Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. "
     "How many clips did she sell altogether in April and May?",
     "In April she sold 48 clips. In May she sold 48 / 2 = 24. Altogether 48 + 24 = 72. The answer is 72."),
    ("Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?",
     "Per minute she earns 12 / 60 = 0.2 dollars. For 50 minutes she earned 50 * 0.2 = 10. The answer is 10."),
    ("Betty is saving for a $100 wallet. She has half the money she needs. Her parents give her $15 and her "
     "grandparents twice as much as her parents. How much more money does Betty need?",
     "Betty has 100 / 2 = 50. Grandparents give 2 * 15 = 30. Now she has 50 + 15 + 30 = 95. She needs 100 - 95 = 5. The answer is 5."),
    ("James writes a 3-page letter to 2 different friends twice a week. How many pages does he write a year?",
     "Each time he writes 3 * 2 = 6 pages. Twice a week is 6 * 2 = 12 pages. In a year 12 * 52 = 624. The answer is 624."),
]
MATH_SHOTS = [
    ("What is the value of $3^2 + 4^2$?", "We have $3^2 = 9$ and $4^2 = 16$, so $9 + 16 = 25$. The answer is $\\boxed{25}$."),
    ("If $2x + 3 = 11$, what is $x$?", "Subtract 3: $2x = 8$. Divide by 2: $x = 4$. The answer is $\\boxed{4}$."),
    ("Simplify $\\frac{6}{8}$.", "Divide numerator and denominator by 2 to get $\\frac{3}{4}$. The answer is $\\boxed{\\frac{3}{4}}$."),
    ("How many positive divisors does 12 have?", "The divisors are 1, 2, 3, 4, 6, 12, so there are 6. The answer is $\\boxed{6}$."),
]

# --------------------------------------------------------------------------- answer extraction
def _clean_num(s):
    if s is None:
        return None
    s = s.strip().replace(",", "").replace("$", "").rstrip(".")
    m = re.search(r"-?\d+(?:/\d+)?(?:\.\d+)?", s)
    return m.group(0) if m else None

def extract_gsm8k(text):
    m = re.findall(r"answer is\s*\$?\s*(-?[\d,]+(?:\.\d+)?)", text)
    if m:
        return _clean_num(m[-1])
    nums = re.findall(r"-?[\d,]+(?:\.\d+)?", text)
    return _clean_num(nums[-1]) if nums else None

def extract_boxed(text):
    i = text.rfind(r"\boxed")
    if i >= 0:
        j = text.find("{", i)
        if j >= 0:
            depth, k = 0, j
            for k in range(j, len(text)):
                if text[k] == "{":
                    depth += 1
                elif text[k] == "}":
                    depth -= 1
                    if depth == 0:
                        return text[j + 1:k].strip()
    m = re.findall(r"answer is\s*\$?\s*([^\n.$]+)", text)  # fallback
    return m[-1].strip() if m else None

def gold_gsm8k(row):
    m = re.search(r"####\s*(-?[\d,]+(?:\.\d+)?)", row.get("answer", ""))
    return _clean_num(m.group(1)) if m else None

def gold_math(row):
    for key in ("answer", "solution", "expected_answer"):
        if key in row and row[key]:
            b = extract_boxed(str(row[key]))
            if b is not None:
                return b.replace(" ", "")
    return None

TASKS = {
    "gsm8k":    dict(shots=GSM8K_SHOTS, extract=extract_gsm8k, gold=gold_gsm8k,
                     match=lambda p, g: p is not None and g is not None and _clean_num(p) == _clean_num(g)),
    "math500":  dict(shots=MATH_SHOTS, extract=extract_boxed, gold=gold_math,
                     match=lambda p, g: p is not None and g is not None and str(p).replace(" ", "") == str(g).replace(" ", "")),
}

# --------------------------------------------------------------------------- generation (KV cache)
_COMPLETE_FINAL_ANSWER = re.compile(
    r"\b(?:the\s+)?answer\s+is\b[^\n]{1,192}?(?:\.(?=\s)|\n)",
    flags=re.IGNORECASE,
)


def has_complete_final_answer(text):
    """Return true only after an explicit final-answer line is complete.

    SFT examples deliberately use ``The answer is X.`` after potentially blank
    paragraphs of reasoning. Stopping on a paragraph break loses the answer;
    generating to ``max_new`` after it wastes evaluation time. A period must be
    followed by whitespace so decimal answers such as ``68.04`` are not cut at
    the decimal point.
    """
    return bool(_COMPLETE_FINAL_ANSWER.search(text))


def decode_tokens(tok, tokens, skip_special_tokens):
    """Decode with explicit special-token policy, including minimal test stubs."""
    try:
        return tok.decode(tokens, skip_special_tokens=skip_special_tokens)
    except TypeError:
        return tok.decode(tokens)


def decode_batch(tok, batches, skip_special_tokens):
    if hasattr(tok, "decode_batch"):
        try:
            return tok.decode_batch(batches, skip_special_tokens=skip_special_tokens)
        except TypeError:
            return tok.decode_batch(batches)
    return [decode_tokens(tok, tokens, skip_special_tokens) for tokens in batches]


@torch.no_grad()
def generate_batch(model, tok, prompt, device, n, max_new=256, temp=0.0, top_k=40,
                   stop="\nQuestion:", skip_special_tokens=True):
    """Generate ``n`` independent samples for one prompt in a single decode batch.

    Self-consistency and verifier-data collection need multiple samples of the
    same prompt.  Decoding those samples one at a time leaves a small model
    launch-bound and makes the verifier pipeline miss its wall-time budget.
    Finished rows keep taking no-op decode steps until the batch is done so the
    shared KV-cache shape remains valid; their returned text is frozen at the
    first stop condition.
    """
    if n <= 0:
        return []
    cap = model.cfg.seq_len
    ids = tok.encode(prompt).ids[-cap:]
    eos_id = tok.token_to_id("<|endoftext|>")
    ac = torch.autocast("cuda", dtype=torch.bfloat16, enabled=("cuda" in str(device)))
    with ac:
        logits, cache = model(torch.tensor([ids] * n, device=device), return_cache=True, pos=0)
    pos = len(ids)
    generated = [[] for _ in range(n)]
    finished = [False] * n
    for _ in range(max_new):
        lg = logits[:, -1]
        if temp and temp > 0:
            lg = lg / temp
            if top_k:
                v, _ = torch.topk(lg, min(top_k, lg.size(-1)), dim=-1)
                lg = lg.masked_fill(lg < v[:, [-1]], float("-inf"))
            nxt = torch.multinomial(torch.softmax(lg.float(), -1), 1).squeeze(-1)
        else:
            nxt = lg.argmax(dim=-1)
        for row, token in enumerate(nxt.tolist()):
            if finished[row]:
                continue
            generated[row].append(token)
        # Rust-backed decode_batch removes the Python per-sample/token loop
        # from self-consistency and verifier rollouts. The fallback keeps the
        # helper testable with a minimal tokenizer stub.
        texts = decode_batch(tok, generated, skip_special_tokens)
        for row, txt in enumerate(texts):
            if finished[row]:
                continue
            # SFT targets intentionally contain paragraph breaks before their final
            # answer. Stopping on any blank line truncates that answer and turns a
            # correct completion into an apparent benchmark miss. Once an explicit
            # final-answer line is complete, however, further decoding cannot help
            # the score and makes self-consistency prohibitively expensive.
            if ((stop and stop in txt) or has_complete_final_answer(txt)
                    or token == eos_id or pos >= cap):
                finished[row] = True
        if all(finished) or pos >= cap:
            break
        with ac:
            logits, cache = model(nxt[:, None], cache=cache, pos=pos, return_cache=True)
        pos += 1
    # The completion includes EOS only because it was the sampled stop token.
    # Retaining other special tokens is useful for trace evaluators, but EOS is
    # never model content and would otherwise leak into a special-aware decode.
    return [
        decode_tokens(tok, tokens[:-1] if tokens and tokens[-1] == eos_id else tokens,
                      skip_special_tokens)
        for tokens in generated
    ]


@torch.no_grad()
def generate(model, tok, prompt, device, max_new=256, temp=0.0, top_k=40, stop="\nQuestion:",
             skip_special_tokens=True):
    """Generate one completion; retained as the stable public evaluator API."""
    return generate_batch(
        model, tok, prompt, device, n=1, max_new=max_new, temp=temp,
        top_k=top_k, stop=stop, skip_special_tokens=skip_special_tokens,
    )[0]

def solve(model, tok, prompt, device, task, k, temp, max_new):
    """Return (final_answer, all_extracted) using greedy (k=1) or self-consistency maj@k."""
    ex = task["extract"]
    if k <= 1:
        return ex(generate(model, tok, prompt, device, max_new=max_new, temp=0.0)), None
    answers = [
        str(answer)
        for sample in generate_batch(model, tok, prompt, device, n=k, max_new=max_new, temp=temp)
        if (answer := ex(sample)) is not None
    ]
    if not answers:
        return None, answers
    return Counter(answers).most_common(1)[0][0], answers

# --------------------------------------------------------------------------- driver
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--task", default="gsm8k", choices=list(TASKS))
    ap.add_argument("--data", required=True)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--k", type=int, default=1, help="self-consistency samples (1 = greedy pass@1)")
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--seed", type=int, default=20260712,
                    help="RNG seed for reproducible sampled self-consistency runs")
    a = ap.parse_args()

    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    torch.manual_seed(a.seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(a.seed)
    print(f"[eval] device={device} task={a.task} k={a.k} temp={a.temp} seed={a.seed}", file=sys.stderr)
    ck = torch.load(a.ckpt, map_location="cpu")
    model = GPT(GPTConfig(**ck["cfg"])).to(device).eval()
    model.load_state_dict(ck["model"])
    tok = Tokenizer.from_file(a.tokenizer)
    task = TASKS[a.task]

    shots = "".join(f"Question: {q}\nAnswer: {r}\n\n" for q, r in task["shots"])
    rows = [json.loads(l) for l in open(a.data)][:a.n]
    correct = 0
    for i, r in enumerate(rows):
        q = r.get("question") or r.get("problem") or r.get("prompt") or ""
        prompt = shots + f"Question: {q}\nAnswer:"
        pred, allans = solve(model, tok, prompt, device, task, a.k, a.temp, a.max_new)
        g = task["gold"](r)
        ok = task["match"](pred, g)
        correct += ok
        if i < 4:
            extra = f" votes={allans}" if allans is not None else ""
            print(f"[{i}] gold={g} pred={pred} ok={ok}{extra}", file=sys.stderr)
        elif (i + 1) % 25 == 0:
            print(f"[progress] {i + 1}/{len(rows)} correct={correct}", file=sys.stderr)
    acc = correct / max(len(rows), 1)
    tag = f"maj@{a.k}" if a.k > 1 else "pass@1"
    print(f"{a.task}  {tag}  ckpt={a.ckpt}  step={ck.get('step')}  {correct}/{len(rows)} = {acc:.1%}")


if __name__ == "__main__":
    main()
