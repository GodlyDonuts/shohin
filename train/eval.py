#!/usr/bin/env python
"""Score a Shohin checkpoint on GSM8K (4-shot, base-model, greedy). Extract the final integer,
exact-match vs gold. This is the scoreboard we grow into — a lightly-trained base model will
score low; the number rises with more pretrain tokens + reasoning mid-train + SFT.

  python eval.py --ckpt flagship_out/ckpt_0010000.pt --tokenizer ../artifacts/shohin-tok-32k.json \\
      --data ../artifacts/evals/gsm8k.jsonl --n 100
"""
import argparse, json, re, sys
import torch
from tokenizers import Tokenizer
from model import GPT, GPTConfig

FEWSHOT = [
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


def gold(ans):
    m = re.search(r"####\s*(-?[\d,]+)", ans)
    return m.group(1).replace(",", "") if m else None


def extract(text):
    m = re.findall(r"answer is\s*\$?\s*(-?[\d,]+)", text)
    if m:
        return m[-1].replace(",", "")
    nums = re.findall(r"-?[\d,]+", text)
    return nums[-1].replace(",", "") if nums else None


@torch.no_grad()
def generate(model, tok, prompt, device, max_new=220, stop="Question:"):
    ids = tok.encode(prompt).ids
    gen = []
    for _ in range(max_new):
        x = torch.tensor([ids[-2048:]], device=device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=("cuda" in str(device))):
            logits, _ = model(x)
        nxt = int(logits[0, -1].argmax())
        ids.append(nxt)
        gen.append(nxt)
        txt = tok.decode(gen)
        if stop in txt or "\n\n" in txt:
            break
    return tok.decode(gen)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--n", type=int, default=100)
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(a.ckpt, map_location=device)
    cfg = GPTConfig(**ck["cfg"])
    model = GPT(cfg).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    tok = Tokenizer.from_file(a.tokenizer)

    shots = "".join(f"Question: {q}\nAnswer: {r}\n\n" for q, r in FEWSHOT)
    rows = [json.loads(l) for l in open(a.data)][:a.n]
    correct = 0
    for i, r in enumerate(rows):
        prompt = shots + f"Question: {r['question']}\nAnswer:"
        gen = generate(model, tok, prompt, device)
        pred, g = extract(gen), gold(r["answer"])
        ok = pred is not None and g is not None and pred == g
        correct += ok
        if i < 4:
            print(f"[{i}] gold={g} pred={pred} ok={ok} | {gen[:90]!r}", file=sys.stderr)
    acc = correct / max(len(rows), 1)
    print(f"GSM8K  ckpt={a.ckpt}  step={ck.get('step')}  {correct}/{len(rows)} = {acc:.1%}")


if __name__ == "__main__":
    main()
