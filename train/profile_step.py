"""Profile an exact Shohin pretraining step on an isolated GPU allocation.

This is deliberately separate from train.py's live path. It loads only model weights from a
preserved checkpoint, warms the compiled graph, times fixed-shape updates, then writes a Chrome
trace and CUDA-kernel summary. It is for choosing kernel/graph work from measurements, never for
producing a training checkpoint.
"""
import argparse
import os
import time

import torch

from data import ShardLoader
from model import GPT, GPTConfig
from muon import Muon, split_params
from train import CONFIGS


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--shard-dirs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", default="shohin", choices=list(CONFIGS))
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--warmup-steps", type=int, default=8)
    ap.add_argument("--timed-steps", type=int, default=12)
    ap.add_argument("--profile-steps", type=int, default=4)
    ap.add_argument("--compile-mode", default="default",
                    choices=("default", "reduce-overhead", "max-autotune"))
    ap.add_argument("--lr-muon", type=float, default=0.005)
    ap.add_argument("--lr-adam", type=float, default=1e-3)
    ap.add_argument("--data-seed", type=int, default=777)
    return ap.parse_args()


def main():
    a = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("profile_step.py requires CUDA")
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(1337)
    torch.set_float32_matmul_precision("high")

    cfg = GPTConfig(**CONFIGS[a.size])
    raw = GPT(cfg).cuda()
    ckpt = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    raw.load_state_dict(ckpt["model"])
    del ckpt
    model = torch.compile(raw, mode=a.compile_mode)

    muon_p, adam_p = split_params(raw)
    opt_muon = Muon(muon_p, lr=a.lr_muon)
    opt_adam = torch.optim.AdamW(adam_p, lr=a.lr_adam, betas=(0.9, 0.95), weight_decay=0.0)
    loader = ShardLoader(a.shard_dirs, cfg.seq_len, a.batch_size, seed=a.data_seed)
    tok_per_step = a.batch_size * a.grad_accum * cfg.seq_len

    def update():
        opt_muon.zero_grad(set_to_none=True)
        opt_adam.zero_grad(set_to_none=True)
        loss_acc = 0.0
        for _ in range(a.grad_accum):
            x, y = loader.next_batch("cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, loss = model(x, y)
                loss = loss / a.grad_accum
            loss.backward()
            loss_acc += loss.item()
        gnorm = float(torch.nn.utils.clip_grad_norm_(raw.parameters(), 1.0))
        opt_muon.step()
        opt_adam.step()
        return loss_acc, gnorm

    for _ in range(a.warmup_steps):
        update()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    last_loss, last_gnorm = 0.0, 0.0
    for _ in range(a.timed_steps):
        last_loss, last_gnorm = update()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    tps = tok_per_step * a.timed_steps / elapsed

    trace_dir = os.path.join(a.out, "trace")
    os.makedirs(trace_dir, exist_ok=True)
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(trace_dir),
    ) as prof:
        for _ in range(a.profile_steps):
            last_loss, last_gnorm = update()
            prof.step()
    torch.cuda.synchronize()

    summary_path = os.path.join(a.out, "summary.txt")
    with open(summary_path, "w") as out:
        out.write(
            f"checkpoint={a.checkpoint}\ncompile_mode={a.compile_mode}\n"
            f"batch_size={a.batch_size}\ngrad_accum={a.grad_accum}\n"
            f"tokens_per_step={tok_per_step}\ntimed_steps={a.timed_steps}\n"
            f"elapsed_s={elapsed:.4f}\ntok_per_s={tps:.1f}\n"
            f"last_loss={last_loss:.6f}\nlast_gnorm={last_gnorm:.6f}\n\n"
        )
        out.write("=== CUDA self time ===\n")
        out.write(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=60))
        out.write("\n\n=== CPU self time ===\n")
        out.write(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=60))
        out.write("\n")
    print(f"[profile] tok/s={tps:.1f} loss={last_loss:.4f} gnorm={last_gnorm:.3f}")
    print(f"[profile] summary={summary_path} trace_dir={trace_dir}")


if __name__ == "__main__":
    main()
