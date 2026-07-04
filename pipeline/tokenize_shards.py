#!/usr/bin/env python
"""Stream an HF dataset, tokenize with the Shohin tokenizer, write zstd-compressed
uint16 shards. Storage-lean: raw text never lands; only compressed token shards.

vocab 32768 fits in uint16 (< 65535). Shards are `shard_NNNNN.u16.zst`.

    python tokenize_shards.py --tokenizer artifacts/shohin-tok-32k.json \\
        --dataset HuggingFaceTB/finemath --config finemath-4plus --text-col text \\
        --out-dir shards/finemath --shard-tokens 100000000 --max-tokens 2000000000
"""
import argparse, os
import numpy as np
import zstandard as zstd
from tokenizers import Tokenizer
from datasets import load_dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--split", default="train")
    ap.add_argument("--text-col", default="text")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--shard-tokens", type=int, default=100_000_000)
    ap.add_argument("--max-tokens", type=int, default=0, help="0 = unlimited")
    ap.add_argument("--eos", default="<|endoftext|>")
    a = ap.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)
    tok = Tokenizer.from_file(a.tokenizer)
    assert tok.get_vocab_size() <= 65535, "vocab exceeds uint16"
    eos_id = tok.token_to_id(a.eos)

    kw = dict(split=a.split, streaming=True)
    if a.config:
        kw["name"] = a.config
    ds = load_dataset(a.dataset, **kw)

    cctx = zstd.ZstdCompressor(level=3)
    buf, shard_idx, total = [], 0, 0

    def flush():
        nonlocal buf, shard_idx
        if not buf:
            return
        arr = np.asarray(buf, dtype=np.uint16)
        path = os.path.join(a.out_dir, f"shard_{shard_idx:05d}.u16.zst")
        with open(path, "wb") as f:
            f.write(cctx.compress(arr.tobytes()))
        raw = arr.nbytes
        comp = os.path.getsize(path)
        print(f"[shard] {path}  {len(arr):,} tok  {comp/1e6:.1f}MB "
              f"({raw/comp:.2f}x zstd)")
        shard_idx += 1
        buf = []

    for ex in ds:
        txt = ex.get(a.text_col) or ""
        if not txt:
            continue
        ids = tok.encode(txt).ids
        buf.extend(ids)
        if eos_id is not None:
            buf.append(eos_id)
        total += len(ids) + (1 if eos_id is not None else 0)
        if len(buf) >= a.shard_tokens:
            flush()
        if a.max_tokens and total >= a.max_tokens:
            break
    flush()
    print(f"[done] {total:,} tokens -> {shard_idx} shards in {a.out_dir}")


if __name__ == "__main__":
    main()
