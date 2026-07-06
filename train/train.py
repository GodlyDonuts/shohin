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
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--fresh-opt", action="store_true",
                    help="resume model weights but RESET optimizer momentum (diagnostic: "
                         "isolates loaded-optimizer-state corruption from data-trajectory issues)")
    ap.add_argument("--no-muon", action="store_true",
                    help="disable Muon; put ALL params in AdamW (bisection: tests whether Muon's "
                         "orthogonalized update is the divergence trigger)")
    ap.add_argument("--gnorm-mult", type=float, default=8.0,
                    help="pre-update guard: skip a step whose grad norm exceeds this multiple of "
                         "its running EMA (catches a destabilizing batch before it lands; <=0 disables)")
    ap.add_argument("--data-seed", type=int, default=1337)
    a = ap.parse_args()

    ddp = "RANK" in os.environ
    if ddp:
        local = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local)   # MUST precede NCCL init, else every rank inits on cuda:0
        dist.init_process_group("nccl")     # -> "device busy" on rank>0 (this was breaking 2xH100)
        rank, world = dist.get_rank(), dist.get_world_size()
        device = f"cuda:{local}"
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

    if a.no_muon:                    # pure-AdamW bisection: is Muon's orthogonalized update the trigger?
        muon_p, adam_p = [], [p for p in raw.parameters() if p.requires_grad]
    else:
        muon_p, adam_p = split_params(raw)
    opt_muon = Muon(muon_p, lr=a.lr_muon) if muon_p else None
    opt_adam = torch.optim.AdamW(adam_p, lr=a.lr_adam, betas=(0.9, 0.95), weight_decay=0.0)
    if master and a.no_muon:
        print("[opt] Muon DISABLED — all params on AdamW (bisection run)", flush=True)

    loader = ShardLoader(a.shard_dirs, cfg.seq_len, a.batch_size, rank, world, seed=a.data_seed)

    os.makedirs(a.out, exist_ok=True)
    import glob as _glob
    start_step = 0
    _cks = sorted(_glob.glob(os.path.join(a.out, "ckpt_[0-9]*.pt")))
    if a.resume and _cks:
        ck = torch.load(_cks[-1], map_location=device)
        raw.load_state_dict(ck["model"])
        if not a.fresh_opt:
            if "opt_muon" in ck and opt_muon is not None:
                opt_muon.load_state_dict(ck["opt_muon"])
            if "opt_adam" in ck:
                opt_adam.load_state_dict(ck["opt_adam"])
        start_step = ck["step"] + 1
        if master:
            tag = "  (FRESH optimizer: momentum reset + rewarmup)" if a.fresh_opt else ""
            print(f"[resume] {_cks[-1]} -> start step {start_step}{tag}", flush=True)
    warm0 = start_step if a.fresh_opt else 0
    logf = open(os.path.join(a.out, f"log_r{rank}.jsonl"), "a") if master else None
    t0 = time.time()
    tok_per_step = world * a.batch_size * a.grad_accum * cfg.seq_len
    loss_ema, gnorm_ema, skips = None, None, 0

    for step in range(start_step, a.steps):
        if a.fresh_opt and step - warm0 < a.warmup:
            lr_scale = (step - warm0) / max(1, a.warmup)   # rewarmup the reset optimizer
        else:
            lr_scale = wsd_lr(step, a.steps, a.warmup)
        if opt_muon is not None:
            for g in opt_muon.param_groups:
                g["lr"] = a.lr_muon * lr_scale
        for g in opt_adam.param_groups:
            g["lr"] = a.lr_adam * lr_scale

        if opt_muon is not None:
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
        if ddp:
            # DDP safety: the skip-vs-step decision below reads loss_acc, which is per-rank local.
            # All-reduce it so every rank makes the SAME decision (else ranks desync -> hang).
            # (gnorm is already identical across ranks — gradients are all-reduced in backward.)
            _la = torch.tensor(loss_acc, device=device)
            dist.all_reduce(_la, op=dist.ReduceOp.SUM)
            loss_acc = float(_la) / world
        # loss-spike guard: NEVER apply a destabilizing or non-finite update. A single bad batch
        # (garbage/OOD tokens) must not be able to wreck the model, so we skip such steps ENTIRELY
        # with no capitulation cap — the old `skips < 5` cap forced a bad update every 6th step,
        # which is exactly what destroyed the model at the data cliff. A long run of skips means a
        # genuinely bad data region or real divergence -> break cleanly (best ckpt already saved)
        # so it surfaces in monitoring instead of silently burning GPU.
        # Measure the gradient EVERY step (pre-clip norm) — this is the diagnostic signal that
        # distinguishes a bad-batch grad spike from a normal-norm-but-bad-direction Muon update.
        # Clipping on a skipped step is harmless (grads are zeroed next iter, no opt.step applied).
        gnorm = float(torch.nn.utils.clip_grad_norm_(raw.parameters(), a.clip))
        finite = (loss_acc == loss_acc) and loss_acc not in (float("inf"), float("-inf"))
        lspike = loss_ema is not None and loss_acc > 2.0 * loss_ema
        # pre-update grad-norm guard: skip a step whose gradient is a large outlier vs its EMA,
        # BEFORE it is applied. The loss-spike check only fires one step late (after the damage);
        # this catches a single destabilizing batch at the right moment.
        gspike = (a.gnorm_mult > 0 and gnorm_ema is not None and gnorm > a.gnorm_mult * gnorm_ema)
        if not finite or lspike or gspike:
            skips += 1
            if master and (skips <= 5 or skips % 25 == 0):
                why = "nan" if not finite else ("gnorm" if gspike else "loss")
                gref = gnorm_ema if gnorm_ema is not None else 0.0
                print(f"[skip:{why}] step {step} loss {loss_acc:.3f} gnorm {gnorm:.2f} "
                      f"(ema gnorm {gref:.2f}) skips={skips}", flush=True)
            if skips >= 300:
                if master:
                    print(f"[guard] {skips} consecutive skips at step {step} -> ending run "
                          f"(bad data region or divergence; best ckpt preserved).", flush=True)
                break
        else:
            skips = 0
            if opt_muon is not None:
                opt_muon.step()
            opt_adam.step()
            loss_ema = loss_acc if loss_ema is None else 0.98 * loss_ema + 0.02 * loss_acc
            gnorm_ema = gnorm if gnorm_ema is None else 0.98 * gnorm_ema + 0.02 * gnorm

        if master and step % a.log_every == 0:
            dt = time.time() - t0
            tps = tok_per_step * (step - start_step + 1) / dt
            rec = dict(step=step, loss=round(loss_acc, 4), gnorm=round(gnorm, 3),
                       lr=round(a.lr_muon * lr_scale, 5), tok_per_s=int(tps), elapsed=round(dt, 1))
            print(f"step {step:>6} loss {loss_acc:.4f} gnorm {gnorm:.2f} lr {a.lr_muon*lr_scale:.4f} "
                  f"{int(tps):,} tok/s", flush=True)
            logf.write(json.dumps(rec) + "\n")
            logf.flush()
        if master and a.ckpt_every and step > 0 and step % a.ckpt_every == 0:
            _sd = dict(model=raw.state_dict(), opt_adam=opt_adam.state_dict(),
                       cfg=cfg.__dict__, step=step)
            if opt_muon is not None:
                _sd["opt_muon"] = opt_muon.state_dict()
            torch.save(_sd, os.path.join(a.out, f"ckpt_{step:07d}.pt"))
            for _o in sorted(_glob.glob(os.path.join(a.out, "ckpt_[0-9]*.pt")))[:-3]:
                try:
                    os.remove(_o)
                except OSError:
                    pass

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
