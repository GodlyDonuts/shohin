#!/usr/bin/env python
"""Train the Shohin 32k BPE and report fertility vs SmolLM2's tokenizer.

Design (master plan §4):
  - byte-level BPE with byte_fallback  -> no <unk>, full coverage
  - single-digit number splitting      -> measurably better small-model arithmetic
  - reserved <think>/<code>/... tokens -> reasoning + PoT + verifier formats

Fertility (bytes/token) is the *pre-model* tokenizer proxy. True bits-per-byte
needs the 30M proxy model (that's the second half of the tokenizer A/B).

    python train_tokenizer.py --sample tok_sample.txt --out shohin-tok-32k.json
"""
import argparse, sys, random
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
from config import VOCAB_SIZE, SPECIAL_TOKENS


def build():
    tok = Tokenizer(models.BPE(unk_token=None, byte_fallback=True))
    tok.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Digits(individual_digits=True),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True),
    ])
    tok.decoder = decoders.ByteLevel()
    return tok


def reservoir(path, n, cap=2_000_000):
    """Reservoir-sample n lines (bounded scan) for the fertility slice."""
    res = []
    with open(path, errors="ignore") as f:
        for i, l in enumerate(f):
            if i >= cap:
                break
            if len(res) < n:
                res.append(l)
            else:
                j = random.randint(0, i)
                if j < n:
                    res[j] = l
    return res


def fertility(tok, lines):
    nb = sum(len(l.encode("utf-8")) for l in lines)
    nt = sum(len(e.ids) for e in tok.encode_batch(lines))
    return nt, nb, nb / max(nt, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", required=True)
    ap.add_argument("--out", default="shohin-tok-32k.json")
    ap.add_argument("--vocab", type=int, default=VOCAB_SIZE)
    ap.add_argument("--eval-lines", type=int, default=5000)
    a = ap.parse_args()

    tok = build()
    trainer = trainers.BpeTrainer(
        vocab_size=a.vocab,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )
    print(f"[tok] training vocab={a.vocab} on {a.sample}", file=sys.stderr)
    tok.train([a.sample], trainer)
    tok.save(a.out)
    print(f"[tok] saved {a.out}  vocab={tok.get_vocab_size()}")

    ev = reservoir(a.sample, a.eval_lines)
    nt, nb, bpt = fertility(tok, ev)
    print(f"[fertility] shohin-{a.vocab//1000}k: {bpt:.3f} bytes/token "
          f"over {nb/1e6:.1f}MB ({nt} tokens)")

    # sanity: digits split to single tokens; code/LaTeX survive
    for s in ["12345", "3.14159", "def f(x): return x*2", r"\frac{1}{2}+\sqrt{2}"]:
        print("   ", repr(s), "->", tok.encode(s).tokens)

    try:
        ref = Tokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
        rnt, rnb, rbpt = fertility(ref, ev)
        print(f"[fertility] SmolLM2-{ref.get_vocab_size()//1000}k: {rbpt:.3f} bytes/token")
        verdict = "BETTER" if bpt > rbpt else "worse"
        print(f"[compare] shohin vs SmolLM2 compression: {verdict} "
              f"({100*(bpt-rbpt)/rbpt:+.1f}%)")
    except Exception as e:
        print(f"[warn] SmolLM2 compare skipped: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
