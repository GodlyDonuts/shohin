"""Async streaming dataloader for zstd uint16 shards.

A background thread decompresses shards and builds batches into a bounded queue, so CPU
decompression overlaps GPU compute and the GPU doesn't idle. Offset-based buffer avoids
per-batch copies; concatenation happens only when a shard is exhausted (~thousands of batches).
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
        self.paths = []
        for d in shard_dirs:
            self.paths += sorted(glob.glob(os.path.join(d, "*.u16.zst")))
        assert self.paths, f"no shards found in {shard_dirs}"
        self.T, self.bs, self.rank, self.world = seq_len, batch_size, rank, world
        self.dctx = zstd.ZstdDecompressor()
        self.rng = random.Random(seed)
        self.q = queue.Queue(maxsize=prefetch)
        self._stop = False
        threading.Thread(target=self._worker, daemon=True).start()

    def _worker(self):
        order = list(range(len(self.paths)))
        self.rng.shuffle(order)
        pi = 0
        buf = np.empty(0, dtype=np.uint16)
        off = 0
        need = self.world * self.bs * (self.T + 1)
        while not self._stop:
            while len(buf) - off < need:
                if pi >= len(order):
                    self.rng.shuffle(order)
                    pi = 0
                with open(self.paths[order[pi]], "rb") as f:
                    arr = np.frombuffer(self.dctx.decompress(f.read()), dtype=np.uint16)
                pi += 1
                buf = np.concatenate([buf[off:], arr])
                off = 0
            chunk = buf[off:off + need].reshape(self.world, self.bs, self.T + 1)[self.rank]
            off += need
            x = torch.from_numpy(chunk[:, :-1].astype(np.int64))
            y = torch.from_numpy(chunk[:, 1:].astype(np.int64))
            self.q.put((x, y))

    def next_batch(self, device):
        x, y = self.q.get()
        return (x.pin_memory().to(device, non_blocking=True),
                y.pin_memory().to(device, non_blocking=True))
