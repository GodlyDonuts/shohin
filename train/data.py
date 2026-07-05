"""Async streaming dataloader for zstd uint16 shards — DOMAIN-INTERLEAVED.

Each shard holds ~200M contiguous tokens of ONE domain (finemath / openwebmath / code). Serving
shards whole (the old behavior) means the model trains on 200M tokens of a single domain, over-
specializes, then takes a loss shock at the next domain boundary — which destabilizes a small,
lightly-trained model (this was the root cause of the recurring ~step-6400 divergence). So instead
we keep one continuous read stream PER DOMAIN and assemble every batch round-robin across domains,
guaranteeing each batch is a code+math+web blend. This both removes the domain-shift cliff and is
standard good practice (mixed-domain batches train better).

A background thread decompresses shards and builds batches into a bounded queue so CPU
decompression overlaps GPU compute.
"""
import glob
import os
import queue
import random
import threading
import numpy as np
import torch
import zstandard as zstd


class ShardLoader:
    def __init__(self, shard_dirs, seq_len, batch_size, rank=0, world=1, seed=1337, prefetch=6):
        self.domains = []                       # one list-of-shard-paths per domain dir
        for d in shard_dirs:
            ps = sorted(glob.glob(os.path.join(d, "*.u16.zst")))
            if ps:
                self.domains.append(ps)
        assert self.domains, f"no shards found in {shard_dirs}"
        self.T, self.bs, self.rank, self.world = seq_len, batch_size, rank, world
        self.dctx = zstd.ZstdDecompressor()
        self.rng = random.Random(seed + 991 * rank)   # per-rank offset so DDP ranks draw different data
        self.q = queue.Queue(maxsize=prefetch)
        self._stop = False
        threading.Thread(target=self._worker, daemon=True).start()

    def _domain_stream(self, paths):
        """Yield successive decompressed shard token arrays for one domain, in a shuffled order
        that reshuffles each time the domain is exhausted (so smaller domains upsample cleanly)."""
        order = list(range(len(paths)))
        self.rng.shuffle(order)
        pi = 0
        while True:
            with open(paths[order[pi]], "rb") as f:
                yield np.frombuffer(self.dctx.decompress(f.read()), dtype=np.uint16)
            pi += 1
            if pi >= len(order):
                self.rng.shuffle(order)
                pi = 0

    def _worker(self):
        need = self.T + 1
        nd = len(self.domains)
        streams = [self._domain_stream(p) for p in self.domains]
        bufs = [next(s) for s in streams]       # current token buffer per domain
        offs = [0] * nd
        while not self._stop:
            seqs = []
            for i in range(self.bs):
                di = i % nd                     # round-robin: every batch blends all domains
                while len(bufs[di]) - offs[di] < need:
                    bufs[di] = np.concatenate([bufs[di][offs[di]:], next(streams[di])])
                    offs[di] = 0
                seqs.append(bufs[di][offs[di]:offs[di] + need])
                offs[di] += need
            chunk = np.stack(seqs)              # [bs, T+1]
            x = torch.from_numpy(chunk[:, :-1].astype(np.int64))
            y = torch.from_numpy(chunk[:, 1:].astype(np.int64))
            self.q.put((x, y))

    def next_batch(self, device):
        x, y = self.q.get()
        return (x.pin_memory().to(device, non_blocking=True),
                y.pin_memory().to(device, non_blocking=True))
