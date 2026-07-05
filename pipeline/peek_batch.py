#!/usr/bin/env python
"""Decode the EXACT token window the trainer sees at a given global offset in the seeded
shard order — to inspect the data at a divergence point. Reproduces ShardLoader's ordering
(sorted glob per dir, in dir order, then Random(seed).shuffle) and walks cumulative tokens.

  python peek_batch.py --shard-dirs d1 d2 d3 --tokenizer tok.json --seed 777 \
      --offsets 199833264 210000000 240000000 --window 3000
"""
import argparse, glob, os, random, re
import numpy as np
import zstandard as zstd
from tokenizers import Tokenizer

BYTE_RE = re.compile(r"^<0x[0-9A-Fa-f]{2}>$")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard-dirs", nargs="+", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--offsets", type=int, nargs="+", required=True)
    ap.add_argument("--window", type=int, default=3000)
    a = ap.parse_args()

    tok = Tokenizer.from_file(a.tokenizer)
    paths = []
    for d in a.shard_dirs:
        paths += sorted(glob.glob(os.path.join(d, "*.u16.zst")))
    order = list(range(len(paths)))
    random.Random(a.seed).shuffle(order)
    dctx = zstd.ZstdDecompressor()

    # cumulative token counts in seed order (decompress lengths once)
    print("[peek] computing shard sizes in seed order...", flush=True)
    sizes, cum = [], [0]
    for oi in order:
        with open(paths[oi], "rb") as f:
            n = len(np.frombuffer(dctx.decompress(f.read()), dtype=np.uint16))
        sizes.append(n)
        cum.append(cum[-1] + n)
    print(f"[peek] total tokens {cum[-1]:,} across {len(order)} shards", flush=True)

    for off in a.offsets:
        # find shard covering this global offset
        pos = next((i for i in range(len(order)) if cum[i] <= off < cum[i + 1]), None)
        if pos is None:
            print(f"\n### offset {off:,} beyond corpus"); continue
        oi = order[pos]; inner = off - cum[pos]
        with open(paths[oi], "rb") as f:
            arr = np.frombuffer(dctx.decompress(f.read()), dtype=np.uint16)
        lo = max(0, inner); hi = min(len(arr), inner + a.window)
        seg = arr[lo:hi]
        # stats on this window
        counts = np.bincount(seg, minlength=1); p = counts[counts > 0] / len(seg)
        H = float(-(p * np.log2(p)).sum())
        top = int(counts.argmax())
        vocab = tok.get_vocab(); id2t = {v: k for k, v in vocab.items()}
        bytef = np.mean([bool(BYTE_RE.match(id2t.get(int(t), ""))) for t in seg[:500]])
        print(f"\n### offset {off:,}  seed-order-pos {pos} shard {os.path.relpath(paths[oi])} "
              f"inner {inner:,}/{len(arr):,}")
        print(f"    window stats: H={H:.2f} top_id={top}({id2t.get(top,'?')!r},{counts[top]/len(seg):.1%}) "
              f"uniq={int((counts>0).sum())}/{len(seg)} bytef~{bytef:.2f}")
        print("    ----- decoded -----")
        print("    " + tok.decode([int(x) for x in seg]).replace("\n", "\n    ")[:2400])


if __name__ == "__main__":
    main()
