"""Shohin model — deep-thin GQA transformer (RoPE, RMSNorm, SwiGLU, QK-norm, tied embeddings).

Clean, correct baseline (modded-nanoGPT-class). Speedrun extras (sliding-window attention,
squared-ReLU, value embeddings, weight-shared/looped depth) are ablation-gated add-ons layered
on this later — see MASTER_PLAN.md §3.
"""
import math
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 32768
    n_layer: int = 30
    n_head: int = 9            # query heads
    n_kv_head: int = 3         # GQA key/value heads
    d_model: int = 576
    d_ff: int = 1536           # SwiGLU hidden
    seq_len: int = 2048
    rope_theta: float = 50_000.0
    qk_norm: bool = True
    tie_embeddings: bool = True
    zloss: float = 1e-4
    n_loop: int = 1            # latent recursion: re-run the block stack N times (weight-shared
                               # extra depth). 1 = off (default, byte-identical). >1 = "think longer"
                               # per token without adding params — an ablation-gated reasoning bet.


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        dt = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.to(dt)) * self.w


def build_rope(seq_len, head_dim, theta, device="cpu"):
    inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, inv)                       # [T, hd/2]
    return torch.cos(freqs), torch.sin(freqs)


def apply_rope(x, cos, sin):
    # x: [B, H, T, hd]  (half-split / GPT-NeoX convention)
    d = x.shape[-1]
    x1, x2 = x[..., : d // 2], x[..., d // 2:]
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


class Attention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.nh, self.nkv = cfg.n_head, cfg.n_kv_head
        self.hd = cfg.d_model // cfg.n_head
        self.q = nn.Linear(cfg.d_model, self.nh * self.hd, bias=False)
        self.k = nn.Linear(cfg.d_model, self.nkv * self.hd, bias=False)
        self.v = nn.Linear(cfg.d_model, self.nkv * self.hd, bias=False)
        self.o = nn.Linear(self.nh * self.hd, cfg.d_model, bias=False)
        self.qk_norm = cfg.qk_norm
        if cfg.qk_norm:
            self.qn, self.kn = RMSNorm(self.hd), RMSNorm(self.hd)

    def forward(self, x, cos, sin, past=None):
        B, T, _ = x.shape
        q = self.q(x).view(B, T, self.nh, self.hd).transpose(1, 2)
        k = self.k(x).view(B, T, self.nkv, self.hd).transpose(1, 2)
        v = self.v(x).view(B, T, self.nkv, self.hd).transpose(1, 2)
        if self.qk_norm:
            q, k = self.qn(q), self.kn(k)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if past is not None:                        # inference: prepend cached (already-RoPE'd) K/V
            pk, pv = past
            k = torch.cat([pk, k], dim=2)
            v = torch.cat([pv, v], dim=2)
        new_past = (k, v)
        rep = self.nh // self.nkv
        kk = k.repeat_interleave(rep, dim=1)
        vv = v.repeat_interleave(rep, dim=1)
        # prefill (no cache) uses the causal mask; a single-token decode step attends to all cached keys
        y = F.scaled_dot_product_attention(q, kk, vv, is_causal=(past is None))
        y = y.transpose(1, 2).reshape(B, T, self.nh * self.hd)
        return self.o(y), new_past


class MLP(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.gate = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.up = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.down = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.n1, self.attn = RMSNorm(cfg.d_model), Attention(cfg)
        self.n2, self.mlp = RMSNorm(cfg.d_model), MLP(cfg)

    def forward(self, x, cos, sin, past=None):
        a, new_past = self.attn(self.n1(x), cos, sin, past)
        x = x + a
        x = x + self.mlp(self.n2(x))
        return x, new_past


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.norm = RMSNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.head.weight = self.tok.weight
        cos, sin = build_rope(cfg.seq_len, cfg.d_model // cfg.n_head, cfg.rope_theta)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, cache=None, pos=0, return_cache=False):
        # Training path is unchanged: cache=None, pos=0, return_cache=False -> identical to before.
        # Inference path: pass return_cache=True to get per-layer (K,V); feed it back with pos=len so
        # decoding is O(1) per token instead of re-encoding the whole prompt (KV cache).
        B, T = idx.shape
        x = self.tok(idx)
        cos = self.cos[pos:pos + T].to(x.device)
        sin = self.sin[pos:pos + T].to(x.device)
        new_cache = []
        ci = 0
        for _loop in range(self.cfg.n_loop):   # n_loop=1 -> identical to before; >1 = weight-shared depth
            for b in self.blocks:
                past = cache[ci] if cache is not None else None
                x, np_ = b(x, cos, sin, past)
                if return_cache:
                    new_cache.append(np_)
                ci += 1
        logits = self.head(self.norm(x))
        if return_cache:
            return logits, new_cache
        loss = None
        if targets is not None:
            lf = logits.float()
            loss = F.cross_entropy(lf.view(-1, lf.size(-1)), targets.view(-1), ignore_index=-1)
            if self.cfg.zloss > 0:
                loss = loss + self.cfg.zloss * torch.logsumexp(lf, dim=-1).pow(2).mean()
        return logits, loss

    def num_params(self):
        return sum(p.numel() for p in self.parameters())  # tied weight counted once
