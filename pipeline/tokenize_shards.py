#!/usr/bin/env python
"""Stream an HF dataset, QUALITY-FILTER + DECONTAMINATE, tokenize with the Shohin
tokenizer, and write zstd-compressed uint16 shards. Storage-lean: raw never lands.

Quality controls (master plan §6.5 — "highest quality possible"):
  --decontam-grams evalgrams.pkl  drop any doc containing an eval 13-gram
  --min-chars N                   drop trivially short docs
  --lang en --lang-field language keep only that language (where the field exists)

Writes a manifest.json with token counts and per-filter drop counts (audit trail).
vocab 32768 fits uint16. Shards: shard_NNNNN.u16.zst.

    python tokenize_shards.py --tokenizer tok.json --dataset HuggingFaceTB/finemath \\
        --config finemath-4plus --text-col text --lang en \\
        --decontam-grams evals/evalgrams.pkl --out-dir shards/finemath4 \\
        --shard-tokens 100000000 --max-tokens 4000000000
"""
import argparse, os, json, re, pickle
import numpy as np
import zstandard as zstd
from tokenizers import Tokenizer
from datasets import load_dataset


def _grams(text, n):
    w = re.findall(r"\w+", text.lower())
    for i in range(len(w) - n + 1):
        yield " ".join(w[i:i + n])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--split", default="train")
    ap.add_argument("--text-col", default="text")
    ap.add_argument("--text-cols", nargs="+", default=None,
                    help="concat multiple fields (joined by a blank line) instead of --text-col; "
                         "e.g. --text-cols problem generated_solution for OpenMathInstruct-2. "
                         "Decontam/min-chars run on the concatenated text.")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--shard-tokens", type=int, default=100_000_000)
    ap.add_argument("--max-tokens", type=int, default=0, help="0 = unlimited")
    ap.add_argument("--eos", default="<|endoftext|>")
    ap.add_argument("--decontam-grams", default=None)
    ap.add_argument("--min-chars", type=int, default=200)
    ap.add_argument("--lang", default=None)
    ap.add_argument("--lang-field", default="language")
    a = ap.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)
    tok = Tokenizer.from_file(a.tokenizer)
    assert tok.get_vocab_size() <= 65535, "vocab exceeds uint16"
    eos_id = tok.token_to_id(a.eos)

    S = gram_n = None
    if a.decontam_grams:
        d = pickle.load(open(a.decontam_grams, "rb"))
        S, gram_n = d["grams"], d["n"]

    kw = dict(split=a.split, streaming=True)
    if a.config:
        kw["name"] = a.config
    ds = load_dataset(a.dataset, **kw)

    cctx = zstd.ZstdCompressor(level=3)
    buf, shard, tok_total = [], 0, 0
    seen = kept = n_short = n_lang = n_contam = 0

    def flush():
        nonlocal buf, shard
        if not buf:
            return
        arr = np.asarray(buf, dtype=np.uint16)
        p = os.path.join(a.out_dir, f"shard_{shard:05d}.u16.zst")
        with open(p, "wb") as f:
            f.write(cctx.compress(arr.tobytes()))
        print(f"[shard] {p} {len(arr):,} tok {os.path.getsize(p)/1e6:.1f}MB", flush=True)
        shard += 1
        buf = []

    for ex in ds:
        seen += 1
        if a.text_cols:
            parts = [str(ex.get(c) or "") for c in a.text_cols]
            txt = "\n\n".join(p for p in parts if p)
        else:
            txt = ex.get(a.text_col) or ""
        if len(txt) < a.min_chars:
            n_short += 1
            continue
        if a.lang:
            lv = ex.get(a.lang_field)
            if lv is not None and str(lv).lower() != a.lang.lower():
                n_lang += 1
                continue
        if S is not None and any(g in S for g in _grams(txt, gram_n)):
            n_contam += 1
            continue
        ids = tok.encode(txt).ids
        buf.extend(ids)
        if eos_id is not None:
            buf.append(eos_id)
        tok_total += len(ids) + (1 if eos_id is not None else 0)
        kept += 1
        if len(buf) >= a.shard_tokens:
            flush()
        if a.max_tokens and tok_total >= a.max_tokens:
            break
    flush()

    manifest = dict(dataset=a.dataset, config=a.config, tokens=tok_total, shards=shard,
                    seen=seen, kept=kept, dropped_short=n_short,
                    dropped_lang=n_lang, dropped_contam=n_contam)
    with open(os.path.join(a.out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print("[done]", json.dumps(manifest))


if __name__ == "__main__":
    main()
