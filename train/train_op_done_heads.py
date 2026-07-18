#!/usr/bin/env python3
"""Train op+done controller heads only (no operand — schedule supplies args)."""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from discrete_controller_heads import (
    OP2ID,
    ControllerHeadConfig,
    DiscreteControllerHeads,
    capture_layer_residual,
)
from eval_typed_controller_host_exec import parse_ops
from model import GPT, GPTConfig

REGISTER_RE = re.compile(r"state=(-?\d+);\s*ops=([^;]+);\s*cursor=(\d+)", re.I)


def load_frozen(ckpt: Path, device):
    blob = torch.load(ckpt, map_location="cpu")
    model = GPT(GPTConfig(**blob["cfg"])).to(device)
    model.load_state_dict(blob["model"])
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    return model


def rows_to_examples(path: Path) -> list[dict]:
    out = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("training_group") != "atomic":
            continue
        m = REGISTER_RE.search(r["completion_prompt"])
        if not m:
            continue
        schedule = parse_ops(m.group(2))
        cursor = int(m.group(3))
        if cursor >= len(schedule):
            continue
        op, _arg = schedule[cursor]
        done = 1 if cursor + 1 >= len(schedule) else 0
        out.append({"prompt": r["completion_prompt"], "op": op, "done": done, "cursor": cursor})
    return out


def balance_by_cursor(rows: list[dict], seed: int = 0) -> list[dict]:
    buckets: dict[str, list] = defaultdict(list)
    for r in rows:
        key = "c0" if r["cursor"] == 0 else ("done" if r["done"] == 1 else "mid")
        buckets[key].append(r)
    target = max(len(buckets.get("c0", [])), 1)
    rng = random.Random(seed)
    out = list(buckets.get("c0", []))
    for key in ("mid", "done"):
        pool = buckets.get(key, [])
        if pool:
            out.extend(rng.choices(pool, k=target))
    rng.shuffle(out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--init", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--heldout", type=Path, required=True)
    ap.add_argument("--tokenizer", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing {args.out}")
    args.out.mkdir(parents=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    gpt = load_frozen(args.init, device)
    heads = DiscreteControllerHeads(ControllerHeadConfig(d_model=gpt.cfg.d_model)).to(device)
    # Freeze operand head — unused
    for p in heads.operand_head.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(
        [p for p in heads.parameters() if p.requires_grad], lr=args.lr
    )
    tok = Tokenizer.from_file(str(args.tokenizer))
    train = balance_by_cursor(rows_to_examples(args.data))
    held = rows_to_examples(args.heldout)
    held_c0 = [r for r in held if r["cursor"] == 0]

    def run_epoch(rows, train_mode: bool):
        heads.train(train_mode)
        total = 0.0
        n_batches = 0
        correct = 0
        for i in range(0, len(rows), args.batch_size):
            batch = rows[i : i + args.batch_size]
            encoded = [tok.encode(r["prompt"]).ids[-gpt.cfg.seq_len :] for r in batch]
            max_len = max(len(e) for e in encoded)
            ids = [[0] * (max_len - len(e)) + e for e in encoded]
            idx = torch.tensor(ids, device=device)
            with torch.no_grad():
                resid = capture_layer_residual(gpt, idx, heads.cfg.read_layer)
            op_id = torch.tensor([OP2ID[r["op"]] for r in batch], device=device)
            done = torch.tensor([r["done"] for r in batch], device=device)
            out = heads.forward(resid)
            loss = F.cross_entropy(out["op_logits"], op_id) + F.cross_entropy(out["done_logits"], done)
            if train_mode:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                total += float(loss.item())
            else:
                with torch.no_grad():
                    pred = heads.decode(resid)
                    for p, r in zip(pred, batch):
                        correct += int(p["op"] == r["op"] and p["done"] == r["done"])
            n_batches += 1
        return total / max(n_batches, 1), correct / max(len(rows), 1)

    history = []
    for ep in range(1, args.epochs + 1):
        tr_loss, _ = run_epoch(train, True)
        _, held_acc = run_epoch(held, False)
        _, c0_acc = run_epoch(held_c0, False) if held_c0 else (0.0, 0.0)
        row = {"epoch": ep, "train_loss": tr_loss, "heldout_op_done_acc": held_acc, "heldout_cursor0_acc": c0_acc}
        history.append(row)
        print(json.dumps(row), flush=True)
        torch.save({"heads": heads.state_dict(), "cfg": heads.cfg.__dict__, "epoch": ep}, args.out / f"heads_ep{ep}.pt")

    decision = {
        "protocol": "R12-SCEB-C-OP-DONE-HEADS-v3",
        "history": history,
        "final_heldout_cursor0_acc": history[-1]["heldout_cursor0_acc"],
        "advance": history[-1]["heldout_cursor0_acc"] >= 0.80,
    }
    (args.out / "decision.json").write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n")
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
