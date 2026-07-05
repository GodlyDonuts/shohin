"""Streaming dataloader for zstd-compressed uint16 token shards (from pipeline/tokenize_shards.py).

Each rank reads a disjoint slice of every global batch. v1: decompress-and-concatenate (fine for
smoke + ablations); for the flagship, pre-decompress to a memmap for max throughput.
"""
import glob
import os
import random
import numpy as np
import torch
import zstandard as zstd


class ShardLoader:
    def __init__(self, shard_dirs, seq_len, batch_size, rank=0, world=1, seed=1337):
        self.paths = []
        for d in shard_dirs:
            self.paths += sorted(glob.glob(os.path.join(d, "*.u16.zst")))
        assert self.paths, f"no shards found in {shard_dirs}"
        self.T, self.bs, self.rank, self.world = seq_len, batch_size, rank, world
        self.dctx = zstd.ZstdDecompressor()
        self.rng = random.Random(seed)
        self._order()
        self.buf = np.empty(0, dtype=np.uint16)

    def _order(self):
        self.order = list(range(len(self.paths)))
        self.rng.shuffle(self.order)
        self.pi = 0

    def _load_next(self):
        if self.pi >= len(self.order):
            self._order()
        p = self.paths[self.order[self.pi]]
        self.pi += 1
        with open(p, "rb") as f:
            return np.frombuffer(self.dctx.decompress(f.read()), dtype=np.uint16)

    def next_batch(self, device):
        need = self.world * self.bs * (self.T + 1)
        while len(self.buf) < need:
            self.buf = np.concatenate([self.buf, self._load_next()])
        chunk = self.buf[:need].reshape(self.world, self.bs, self.T + 1)[self.rank]
        self.buf = self.buf[need:]
        x = torch.from_numpy(chunk[:, :-1].astype(np.int64))
        y = torch.from_numpy(chunk[:, 1:].astype(np.int64))
        return x.to(device, non_blocking=True), y.to(device, non_blocking=True)
