#!/usr/bin/env python3
"""Train discrete controller heads on typed-controller schedules (SCEB-C)."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
from tokenizers import Tokenizer

from discrete_controller_heads import (
    OP2ID,
    ControllerHeadConfig,
    DiscreteControllerHeads,
    capture_layer_residual,
)
from model import GPT, GPTConfig
from eval_typed_controller_host_exec import parse_ops

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
        state = int(m.group(1))
        schedule = parse_ops(m.group(2))
        cursor = int(m.group(3))
        if cursor >= len(schedule):
            continue
        op, arg = schedule[cursor]
        done = 1 if cursor + 1 >= len(schedule) else 0
        operand = arg if op != "horner" else (arg % 1000)
        out.append(
            {
                "prompt": r["completion_prompt"],
                "op": op,
                "operand": int(operand),
                "done": done,
                "state": state,
                "cursor": cursor,
            }
        )
    return out


def balance_by_cursor(rows: list[dict], *, seed: int = 0) -> list[dict]:
    """Upsample cursor=0 and non-done steps so finals do not dominate."""
    import random
    from collections import defaultdict

    buckets: dict[str, list] = defaultdict(list)
    for r in rows:
        key = "c0" if r["cursor"] == 0 else ("done" if r["done"] == 1 else "mid")
        buckets[key].append(r)
    target = max(len(buckets.get("c0", [])), 1)
    rng = random.Random(seed)
    out = list(buckets.get("c0", []))
    for key in ("mid", "done"):
        pool = buckets.get(key, [])
        if not pool:
            continue
        # Match cursor-0 count (with replacement if needed)
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
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-train", type=int, default=30000)
    args = ap.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing {args.out}")
    args.out.mkdir(parents=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    gpt = load_frozen(args.init, device)
    heads = DiscreteControllerHeads(ControllerHeadConfig(d_model=gpt.cfg.d_model)).to(device)
    opt = torch.optim.AdamW(heads.parameters(), lr=args.lr)
    tok = Tokenizer.from_file(str(args.tokenizer))
    train = balance_by_cursor(rows_to_examples(args.data))[: args.max_train]
    held = rows_to_examples(args.heldout)
    held_c0 = [r for r in held if r["cursor"] == 0]

    def run_epoch(rows, train_mode: bool):
        if train_mode:
            heads.train()
        else:
            heads.eval()
        total = 0.0
        n = 0
        correct_op = 0
        for i in range(0, len(rows), args.batch_size):
            batch = rows[i : i + args.batch_size]
            encoded = [tok.encode(r["prompt"]).ids[-gpt.cfg.seq_len :] for r in batch]
            max_len = max(len(e) for e in encoded)
            ids = []
            for e in encoded:
                ids.append([0] * (max_len - len(e)) + e)
            idx = torch.tensor(ids, device=device)
            with torch.no_grad():
                resid = capture_layer_residual(gpt, idx, heads.cfg.read_layer)
            op_id = torch.tensor([OP2ID[r["op"]] for r in batch], device=device)
            operand = torch.tensor([min(r["operand"], heads.cfg.max_operand - 1) for r in batch], device=device)
            done = torch.tensor([r["done"] for r in batch], device=device)
            if train_mode:
                loss = heads.loss(resid, op_id, operand, done)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                total += float(loss.item())
            else:
                with torch.no_grad():
                    pred = heads.decode(resid)
                    for p, r in zip(pred, batch):
                        correct_op += int(
                            p["op"] == r["op"]
                            and p["done"] == r["done"]
                            and p["operand"] == min(r["operand"], heads.cfg.max_operand - 1)
                        )
            n += 1
        return total / max(n, 1), correct_op / max(len(rows), 1)

    history = []
    for ep in range(1, args.epochs + 1):
        tr_loss, _ = run_epoch(train, True)
        _, held_acc = run_epoch(held, False)
        _, held_c0_acc = run_epoch(held_c0, False) if held_c0 else (0.0, 0.0)
        row = {
            "epoch": ep,
            "train_loss": tr_loss,
            "heldout_op_done_operand_acc": held_acc,
            "heldout_cursor0_acc": held_c0_acc,
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        torch.save({"heads": heads.state_dict(), "cfg": heads.cfg.__dict__, "epoch": ep}, args.out / f"heads_ep{ep}.pt")

    decision = {
        "protocol": "R12-SCEB-C-CONTROLLER-HEADS-v2",
        "history": history,
        "final_heldout_op_done_operand_acc": history[-1]["heldout_op_done_operand_acc"],
        "final_heldout_cursor0_acc": history[-1]["heldout_cursor0_acc"],
        "advance": history[-1]["heldout_cursor0_acc"] >= 0.80,
    }
    (args.out / "decision.json").write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n")
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
