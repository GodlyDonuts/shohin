#!/usr/bin/env python
"""Hunt for the pathological shard behind the training loss cliff.

Two independent lenses:
  1) GLOBAL OUTLIER SCAN — per-shard stats (entropy, top-token fraction, byte-fallback
     fraction, mean id). A garbage/OOD shard stands out from the corpus median.
  2) TARGETED DECODE — reproduce the loader's seeded shard order, walk cumulative tokens to
     the divergence position, and decode a window of the offending shard so we can SEE it.

  python scan_shards.py --shard-dirs d1 d2 d3 --tokenizer tok.json \
      --seed 777 --diverge-token 204600000
"""
import argparse, glob, math, os, random, re
import numpy as np
import zstandard as zstd
from tokenizers import Tokenizer

BYTE_RE = re.compile(r"^<0x[0-9A-Fa-f]{2}>$")


def shard_paths(shard_dirs):
    paths = []
    for d in shard_dirs:
        paths += sorted(glob.glob(os.path.join(d, "*.u16.zst")))
    return paths


def load_shard(dctx, p):
    with open(p, "rb") as f:
        return np.frombuffer(dctx.decompress(f.read()), dtype=np.uint16)


def stats(arr, byte_ids):
    n = len(arr)
    counts = np.bincount(arr, minlength=1)
    p = counts[counts > 0] / n
    H = float(-(p * np.log2(p)).sum())               # Shannon entropy, bits/token
    top1 = int(counts.argmax()); top1f = counts[top1] / n
    order = np.argsort(counts)[::-1][:5]
    top5f = float(counts[order].sum() / n)
    bf = float(np.isin(arr, byte_ids).mean()) if len(byte_ids) else 0.0
    return dict(n=n, H=round(H, 3), top1=top1, top1f=round(float(top1f), 4),
                top5f=round(top5f, 4), bytef=round(bf, 4), meanid=int(arr.mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard-dirs", nargs="+", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--diverge-token", type=int, default=204_600_000)
    a = ap.parse_args()

    tok = Tokenizer.from_file(a.tokenizer)
    vocab = tok.get_vocab()
    byte_ids = np.array([i for t, i in vocab.items() if BYTE_RE.match(t)], dtype=np.uint16)
    print(f"[tok] vocab={len(vocab)} byte-fallback tokens={len(byte_ids)}", flush=True)

    paths = shard_paths(a.shard_dirs)
    print(f"[scan] {len(paths)} shards", flush=True)
    dctx = zstd.ZstdDecompressor()

    # (1) global per-shard stats
    rows = []
    for i, p in enumerate(paths):
        s = stats(load_shard(dctx, p), byte_ids)
        s["path"] = p
        rows.append(s)
        print(f"[{i:3d}] H={s['H']:6.3f} top1f={s['top1f']:.4f} top5f={s['top5f']:.4f} "
              f"bytef={s['bytef']:.4f} meanid={s['meanid']:6d} n={s['n']:>11,} "
              f"{os.path.relpath(p)}", flush=True)

    def flag(key, hi=True):
        vals = np.array([r[key] for r in rows], float)
        med = np.median(vals); mad = np.median(np.abs(vals - med)) + 1e-9
        z = (vals - med) / (1.4826 * mad)
        idx = np.argsort(z if hi else -z)[::-1][:3]
        print(f"\n== outliers by {key} (median={med:.3f}) ==", flush=True)
        for j in idx:
            print(f"   z={z[j]:+.1f} {key}={rows[j][key]} {os.path.relpath(rows[j]['path'])}",
                  flush=True)

    flag("H", hi=True); flag("H", hi=False)
    flag("bytef", hi=True); flag("top1f", hi=True)

    # (2) targeted decode at the divergence position (seed-order cumulative walk)
    order = list(range(len(paths)))
    random.Random(a.seed).shuffle(order)
    cum = 0
    hit = None
    for pos, oi in enumerate(order):
        n = len(load_shard(dctx, paths[oi]))
        if cum <= a.diverge_token < cum + n:
            hit = (pos, oi, cum, n)
            break
        cum += n
    print(f"\n== targeted decode: seed={a.seed} diverge_token={a.diverge_token:,} ==", flush=True)
    if hit:
        pos, oi, cstart, n = hit
        inner = a.diverge_token - cstart
        arr = load_shard(dctx, paths[oi])
        s = stats(arr, byte_ids)
        print(f"   order-pos {pos} -> shard {os.path.relpath(paths[oi])} "
              f"(inner offset {inner:,} of {n:,})", flush=True)
        print(f"   shard stats: {s}", flush=True)
        for w in range(-1, 2):
            lo = max(0, inner + w * 1024); hi = min(n, lo + 1024)
            txt = tok.decode([int(x) for x in arr[lo:hi]])
            print(f"\n   --- window @ {lo:,}..{hi:,} ---\n{txt[:1500]!r}", flush=True)
    else:
        print("   (position beyond total tokens)", flush=True)


if __name__ == "__main__":
    main()
