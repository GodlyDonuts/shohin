"""Measure whole-update CUDA-graph replay for Shohin without touching the live trainer.

torch.compile(mode="reduce-overhead") captures individual forwards, which conflicts with this
trainer's eight gradient-accumulation microsteps. This canary captures the complete update instead:
all static microbatches, backward passes, clipping, Muon, and AdamW are one CUDA graph replay.
It never writes checkpoints and is only a speed/stability gate for a future trainer integration.
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
    ap.add_argument("--warmup-steps", type=int, default=4)
    ap.add_argument("--timed-steps", type=int, default=80)
    ap.add_argument("--lr-muon", type=float, default=0.005)
    ap.add_argument("--lr-adam", type=float, default=1e-3)
    ap.add_argument("--data-seed", type=int, default=777)
    return ap.parse_args()


def main():
    a = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("cudagraph_canary.py requires CUDA")
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(1337)
    torch.set_float32_matmul_precision("high")

    cfg = GPTConfig(**CONFIGS[a.size])
    raw = GPT(cfg).cuda()
    ckpt = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    raw.load_state_dict(ckpt["model"])
    muon_p, adam_p = split_params(raw)
    opt_muon = Muon(muon_p, lr=a.lr_muon)
    # Whole-update CUDA graph capture requires AdamW's state updates to be capturable. This is
    # canary-only; the live trainer keeps its established optimizer construction unchanged.
    opt_adam = torch.optim.AdamW(
        adam_p, lr=a.lr_adam, betas=(0.9, 0.95), weight_decay=0.0, capturable=True
    )
    if "opt_muon" in ckpt:
        opt_muon.load_state_dict(ckpt["opt_muon"])
    if "opt_adam" in ckpt:
        opt_adam.load_state_dict(ckpt["opt_adam"])
        # The live checkpoint's param groups were created without graph capture. Loading them
        # replaces constructor settings, so restore this canary-only requirement afterward.
        for group in opt_adam.param_groups:
            group["capturable"] = True
    # `torch.load(..., map_location="cpu")` leaves scalar optimizer state such as AdamW's step
    # counter on CPU. Capturable optimizers require every state tensor on the parameter device.
    for opt in (opt_muon, opt_adam):
        for state in opt.state.values():
            for name, value in list(state.items()):
                if torch.is_tensor(value) and value.device.type != "cuda":
                    state[name] = value.cuda()
    del ckpt

    # Compile gives the baseline's fused elementwise kernels; the explicit graph below removes
    # repeated host launch overhead around the full update rather than nesting Inductor graphs.
    model = torch.compile(raw)
    loader = ShardLoader(a.shard_dirs, cfg.seq_len, a.batch_size, seed=a.data_seed)
    static = [
        (
            torch.empty((a.batch_size, cfg.seq_len), dtype=torch.long, device="cuda"),
            torch.empty((a.batch_size, cfg.seq_len), dtype=torch.long, device="cuda"),
        )
        for _ in range(a.grad_accum)
    ]
    loss_out = torch.zeros((), device="cuda")
    gnorm_out = torch.zeros((), device="cuda")
    tok_per_step = a.batch_size * a.grad_accum * cfg.seq_len

    def stage_inputs():
        for sx, sy in static:
            x, y = loader.q.get()
            sx.copy_(x.pin_memory(), non_blocking=True)
            sy.copy_(y.pin_memory(), non_blocking=True)

    def update():
        opt_muon.zero_grad(set_to_none=True)
        opt_adam.zero_grad(set_to_none=True)
        loss_out.zero_()
        for sx, sy in static:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, loss = model(sx, sy)
                loss = loss / a.grad_accum
            loss_out.add_(loss.detach())
            loss.backward()
        gnorm_out.copy_(torch.nn.utils.clip_grad_norm_(raw.parameters(), 1.0))
        opt_muon.step()
        opt_adam.step()

    # Warm Inductor and allocator state before capture. Inputs are static buffers throughout.
    for _ in range(a.warmup_steps):
        stage_inputs()
        update()
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    stage_inputs()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        update()
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for step in range(a.timed_steps):
        stage_inputs()
        graph.replay()
        if step % 10 == 0 or step == a.timed_steps - 1:
            torch.cuda.synchronize()
            print(f"step {step:>4} loss {float(loss_out):.4f} gnorm {float(gnorm_out):.3f}", flush=True)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    tps = tok_per_step * a.timed_steps / elapsed
    mem_gib = torch.cuda.max_memory_allocated() / (1024 ** 3)

    summary = os.path.join(a.out, "summary.txt")
    with open(summary, "w") as out:
        out.write(
            f"checkpoint={a.checkpoint}\n"
            f"batch_size={a.batch_size}\ngrad_accum={a.grad_accum}\n"
            f"tokens_per_step={tok_per_step}\ntimed_steps={a.timed_steps}\n"
            f"elapsed_s={elapsed:.4f}\ntok_per_s={tps:.1f}\n"
            f"last_loss={float(loss_out):.6f}\nlast_gnorm={float(gnorm_out):.6f}\n"
            f"max_allocated_gib={mem_gib:.2f}\n"
        )
    print(f"[cudagraph] tok/s={tps:.1f} loss={float(loss_out):.4f} gnorm={float(gnorm_out):.3f} mem={mem_gib:.2f}GiB")
    print(f"[cudagraph] summary={summary}")


if __name__ == "__main__":
    main()
