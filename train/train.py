"""Shohin trainer — Muon(+AdamW) · WSD · bf16 · single-node DDP (torchrun) or single-GPU.

Runs on 2xH100 today; drops onto 8xH100 (evc102) unchanged the moment `highgpu` access lands
— just launch with more ranks. Minimal-CPU by design (few dataloader workers).

  torchrun --nproc_per_node=2 train.py --size shohin --shard-dirs <d1> <d2> --steps 200000 ...
  python train.py --size tiny --shard-dirs <d> --steps 50            # 1-GPU smoke
"""
import argparse
import json
import os
import time
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from model import GPT, GPTConfig
from muon import Muon, split_params
from data import ShardLoader

CONFIGS = {
    # smoke: tiny, trains in seconds
    "tiny":   dict(n_layer=4,  n_head=4, n_kv_head=2, d_model=256, d_ff=768,  seq_len=1024),
    # ablation proxy (~30M) — the "Mame" model
    "mame":   dict(n_layer=12, n_head=6, n_kv_head=2, d_model=384, d_ff=1024, seq_len=2048),
    # flagship (~125-135M)
    "shohin": dict(n_layer=30, n_head=9, n_kv_head=3, d_model=576, d_ff=1536, seq_len=2048),
}


def wsd_lr(step, total, warmup, decay_frac=0.2, final=0.1):
    if step < warmup:
        return step / max(1, warmup)
    dstart = total * (1 - decay_frac)
    if step < dstart:
        return 1.0
    r = (step - dstart) / max(1.0, total - dstart)
    return 1.0 + (final - 1.0) * r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="tiny", choices=list(CONFIGS))
    ap.add_argument("--shard-dirs", nargs="+", required=True)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=16)     # per-rank micro-batch (sequences)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--vocab-size", type=int, default=32768)
    ap.add_argument("--lr-muon", type=float, default=0.02)
    ap.add_argument("--lr-adam", type=float, default=3e-3)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--ckpt-every", type=int, default=2000)
    ap.add_argument("--out", default="ckpt")
    ap.add_argument("--compile", action="store_true")
    a = ap.parse_args()

    ddp = "RANK" in os.environ
    if ddp:
        dist.init_process_group("nccl")
        rank, world = dist.get_rank(), dist.get_world_size()
        local = int(os.environ["LOCAL_RANK"])
        device = f"cuda:{local}"
        torch.cuda.set_device(device)
    else:
        rank, world, device = 0, 1, ("cuda" if torch.cuda.is_available() else "cpu")
    master = rank == 0
    torch.manual_seed(1337 + rank)
    torch.set_float32_matmul_precision("high")

    cfg = GPTConfig(vocab_size=a.vocab_size, **CONFIGS[a.size])
    model = GPT(cfg).to(device)
    if master:
        print(f"[model] size={a.size} params={model.num_params()/1e6:.1f}M "
              f"world={world} bs={a.batch_size} accum={a.grad_accum} seq={cfg.seq_len}", flush=True)
    raw = model
    if a.compile:
        model = torch.compile(model)
    if ddp:
        model = DDP(model, device_ids=[local])

    muon_p, adam_p = split_params(raw)
    opt_muon = Muon(muon_p, lr=a.lr_muon)
    opt_adam = torch.optim.AdamW(adam_p, lr=a.lr_adam, betas=(0.9, 0.95), weight_decay=0.0)

    loader = ShardLoader(a.shard_dirs, cfg.seq_len, a.batch_size, rank, world, seed=1337)

    os.makedirs(a.out, exist_ok=True)
    logf = open(os.path.join(a.out, f"log_r{rank}.jsonl"), "a") if master else None
    t0 = time.time()
    tok_per_step = world * a.batch_size * a.grad_accum * cfg.seq_len

    for step in range(a.steps):
        lr_scale = wsd_lr(step, a.steps, a.warmup)
        for g in opt_muon.param_groups:
            g["lr"] = a.lr_muon * lr_scale
        for g in opt_adam.param_groups:
            g["lr"] = a.lr_adam * lr_scale

        opt_muon.zero_grad(set_to_none=True)
        opt_adam.zero_grad(set_to_none=True)
        loss_acc = 0.0
        for micro in range(a.grad_accum):
            x, y = loader.next_batch(device)
            sync = (not ddp) or (micro == a.grad_accum - 1)
            ctx = model.no_sync() if (ddp and not sync) else _null()
            with ctx, torch.autocast("cuda", dtype=torch.bfloat16, enabled=("cuda" in str(device))):
                _, loss = model(x, y)
                loss = loss / a.grad_accum
            loss.backward()
            loss_acc += loss.item()
        torch.nn.utils.clip_grad_norm_(raw.parameters(), a.clip)
        opt_muon.step()
        opt_adam.step()

        if master and step % a.log_every == 0:
            dt = time.time() - t0
            tps = tok_per_step * (step + 1) / dt
            rec = dict(step=step, loss=round(loss_acc, 4), lr=round(a.lr_muon * lr_scale, 5),
                       tok_per_s=int(tps), elapsed=round(dt, 1))
            print(f"step {step:>6} loss {loss_acc:.4f} lr {a.lr_muon*lr_scale:.4f} "
                  f"{int(tps):,} tok/s", flush=True)
            logf.write(json.dumps(rec) + "\n")
            logf.flush()
        if master and a.ckpt_every and step > 0 and step % a.ckpt_every == 0:
            torch.save(dict(model=raw.state_dict(), cfg=cfg.__dict__, step=step),
                       os.path.join(a.out, f"ckpt_{step:07d}.pt"))

    if master:
        torch.save(dict(model=raw.state_dict(), cfg=cfg.__dict__, step=a.steps),
                   os.path.join(a.out, "ckpt_final.pt"))
        print(f"[done] {a.steps} steps in {time.time()-t0:.0f}s", flush=True)
    if ddp:
        dist.destroy_process_group()


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


if __name__ == "__main__":
    main()
