#!/usr/bin/env python
"""Stream a byte-budgeted, mixed sample (edu/math/code) from HF to a text file
for tokenizer training.

Storage-lean by design: streaming=True, nothing cached to disk beyond the output
sample itself. Safe under the 1 TB Lustre quota.

    python stream_sample.py --out /path/tok_sample.txt --gb 2
"""
import argparse, sys, time
from datasets import load_dataset
from config import SAMPLE_SOURCES


def stream_source(ds, cfg, split, col, budget, fout):
    kwargs = dict(split=split, streaming=True)
    if cfg:
        kwargs["name"] = cfg
    try:
        it = load_dataset(ds, **kwargs)
    except Exception as e:
        print(f"  [skip] {ds} ({cfg}): {e}", file=sys.stderr)
        return 0
    got = n = 0
    for ex in it:
        txt = ex.get(col) or ""
        if not txt:
            continue
        fout.write(txt.replace("\x00", " "))
        fout.write("\n")
        got += len(txt.encode("utf-8", "ignore"))
        n += 1
        if n % 5000 == 0:
            print(f"    {ds}: {got/1e9:.2f}/{budget/1e9:.2f} GB", file=sys.stderr)
        if got >= budget:
            break
    return got


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--gb", type=float, default=2.0, help="total sample size in GB")
    a = ap.parse_args()

    total = int(a.gb * 1e9)
    t0 = time.time()
    grand = 0
    with open(a.out, "w") as f:
        for ds, cfg, split, col, w in SAMPLE_SOURCES:
            bud = int(total * w)
            print(f"[stream] {ds} ({cfg}) target {bud/1e9:.2f} GB", file=sys.stderr)
            g = stream_source(ds, cfg, split, col, bud, f)
            grand += g
            print(f"[done]   {ds}: {g/1e9:.2f} GB", file=sys.stderr)
    print(f"[sample] wrote {a.out}: {grand/1e9:.2f} GB in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
