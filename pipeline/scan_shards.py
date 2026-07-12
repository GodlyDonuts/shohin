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
import argparse, glob, json, math, os, random, re
from pathlib import Path
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


def robust_outliers(rows, key, high=True, limit=3):
    """Return stable, JSON-safe robust-z outlier records for one statistic."""
    values = np.array([row[key] for row in rows], dtype=float)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    scale = 1.4826 * mad
    if scale <= 1e-12:
        z = np.zeros_like(values)
    else:
        z = (values - median) / scale
    order = np.argsort(z if high else -z)[::-1][:limit]
    report = []
    for index in order:
        report.append({
            "path": rows[int(index)]["path"],
            "value": float(values[int(index)]),
            "robust_z": float(z[int(index)]),
        })
    return {"median": median, "mad": mad, "outliers": report}


def metric_summary(rows, key):
    values = np.array([row[key] for row in rows], dtype=float)
    return {
        "min": float(values.min()),
        "median": float(np.median(values)),
        "max": float(values.max()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard-dirs", nargs="+", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--diverge-token", type=int, default=204_600_000)
    ap.add_argument("--out", help="optional JSON report path outside the shard corpus")
    ap.add_argument("--max-bytef", type=float,
                    help="fail when any shard's byte-fallback fraction exceeds this value")
    ap.add_argument("--max-robust-z", type=float,
                    help="fail when a selected high-side outlier exceeds this robust-z value")
    a = ap.parse_args()

    tok = Tokenizer.from_file(a.tokenizer)
    vocab = tok.get_vocab()
    byte_ids = np.array([i for t, i in vocab.items() if BYTE_RE.match(t)], dtype=np.uint16)
    print(f"[tok] vocab={len(vocab)} byte-fallback tokens={len(byte_ids)}", flush=True)

    paths = shard_paths(a.shard_dirs)
    if not paths:
        raise SystemExit("no .u16.zst shards found")
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

    outlier_sets = {
        "entropy_high": robust_outliers(rows, "H", high=True),
        "entropy_low": robust_outliers(rows, "H", high=False),
        "bytef_high": robust_outliers(rows, "bytef", high=True),
        "top1f_high": robust_outliers(rows, "top1f", high=True),
    }
    for label, report in outlier_sets.items():
        print(f"\n== outliers by {label} (median={report['median']:.3f}) ==", flush=True)
        for item in report["outliers"]:
            print(f"   z={item['robust_z']:+.1f} value={item['value']:.4f} "
                  f"{os.path.relpath(item['path'])}", flush=True)

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

    report = {
        "shard_dirs": [str(Path(directory)) for directory in a.shard_dirs],
        "shards": rows,
        "metrics": {key: metric_summary(rows, key) for key in ("H", "top1f", "top5f", "bytef", "meanid")},
        "outliers": outlier_sets,
        "seed": a.seed,
        "diverge_token": a.diverge_token,
    }
    if a.out:
        out = Path(a.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        temporary = out.with_suffix(out.suffix + ".partial")
        temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        temporary.replace(out)
        print(f"[scan] wrote report {out}", flush=True)

    if a.max_bytef is not None:
        worst = max(row["bytef"] for row in rows)
        if worst > a.max_bytef:
            raise SystemExit(f"byte-fallback gate failed: {worst:.6f} > {a.max_bytef:.6f}")
    if a.max_robust_z is not None:
        failures = []
        for name in ("entropy_high", "bytef_high", "top1f_high"):
            top = outlier_sets[name]["outliers"][0]
            if top["robust_z"] > a.max_robust_z:
                failures.append(f"{name}={top['robust_z']:.2f}")
        if failures:
            raise SystemExit("robust outlier gate failed: " + ", ".join(failures))


if __name__ == "__main__":
    main()
