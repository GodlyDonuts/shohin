#!/usr/bin/env python
"""Strip a training checkpoint down to model weights + config for local eval/testing.
Drops the Muon/AdamW optimizer state (~2/3 of the file, only needed to resume training).

  python strip_ckpt.py <out_dir> ckpt_a.pt ckpt_b.pt ...
"""
import os
import sys
import torch


def main():
    outdir = sys.argv[1]
    os.makedirs(outdir, exist_ok=True)
    for src in sys.argv[2:]:
        ck = torch.load(src, map_location="cpu")
        base = os.path.basename(src).replace(".pt", ".model.pt")
        out = os.path.join(outdir, base)
        torch.save({"model": ck["model"], "cfg": ck["cfg"], "step": ck.get("step")}, out)
        mb = os.path.getsize(out) / 1e6
        print("[strip] {}  {:.0f} MB  step={}".format(out, mb, ck.get("step")), flush=True)


if __name__ == "__main__":
    main()
