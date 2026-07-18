#!/usr/bin/env python3
"""Train Stateful Residual Register on typed-controller state labels (SCEB-B)."""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

import torch
from tokenizers import Tokenizer

from model import GPT, GPTConfig
from stateful_residual_register import (
    SRRConfig,
    StatefulResidualRegister,
    digits_to_int,
    int_to_digits,
    run_with_srr,
    run_with_srr_teacher,
)

REGISTER_RE = re.compile(r"state=(-?\d+);", re.I)


def load_frozen_gpt(ckpt: Path, device: torch.device) -> GPT:
    blob = torch.load(ckpt, map_location="cpu")
    model = GPT(GPTConfig(**blob["cfg"])).to(device)
    model.load_state_dict(blob["model"])
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    return model


def build_rows(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("training_group") not in {"atomic", "rollout", "resume"}:
            continue
        m = REGISTER_RE.search(r["completion_prompt"])
        if not m:
            continue
        rows.append(
            {
                "prompt": r["completion_prompt"],
                "state": int(m.group(1)),
                "final": int(r["final_answer"]),
            }
        )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--init", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--heldout", type=Path, required=True)
    ap.add_argument("--tokenizer", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-train", type=int, default=20000)
    ap.add_argument("--eval-cases", type=int, default=256)
    args = ap.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing existing out: {args.out}")
    args.out.mkdir(parents=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    gpt = load_frozen_gpt(args.init, device)
    srr = StatefulResidualRegister(
        SRRConfig(d_model=gpt.cfg.d_model, n_layer=gpt.cfg.n_layer)
    ).to(device)
    opt = torch.optim.AdamW(srr.parameters(), lr=args.lr)
    tok = Tokenizer.from_file(str(args.tokenizer))

    train_rows = build_rows(args.data)[: args.max_train]
    held_rows = build_rows(args.heldout)
    random.Random(0).shuffle(train_rows)

    def batch_iter(rows, bs):
        for i in range(0, len(rows), bs):
            yield rows[i : i + bs]

    history = []
    for epoch in range(1, args.epochs + 1):
        srr.train()
        total = 0.0
        n = 0
        for batch in batch_iter(train_rows, args.batch_size):
            ids = []
            targets = []
            max_len = 0
            encoded = [tok.encode(r["prompt"]).ids for r in batch]
            max_len = min(max(len(x) for x in encoded), gpt.cfg.seq_len)
            for enc, r in zip(encoded, batch):
                enc = enc[-max_len:]
                pad = [0] * (max_len - len(enc))
                ids.append(pad + enc)
                targets.append(int_to_digits(r["state"], srr.cfg.n_digits))
            idx = torch.tensor(ids, device=device)
            tgt = torch.stack(targets).to(device)
            _, _, aux = run_with_srr_teacher(gpt, srr, idx, tgt)
            opt.zero_grad(set_to_none=True)
            aux.backward()
            opt.step()
            total += float(aux.item())
            n += 1
        # Eval register readout accuracy
        srr.eval()
        correct = 0
        eval_rows = held_rows[: args.eval_cases]
        with torch.no_grad():
            for r in eval_rows:
                enc = tok.encode(r["prompt"]).ids[-gpt.cfg.seq_len :]
                idx = torch.tensor([enc], device=device)
                _, pred = run_with_srr(gpt, srr, idx)
                pred_int = digits_to_int(pred[0].cpu())
                correct += int(pred_int == r["state"])
        acc = correct / max(len(eval_rows), 1)
        row = {"epoch": epoch, "train_aux": total / max(n, 1), "heldout_state_acc": acc}
        history.append(row)
        print(json.dumps(row), flush=True)
        torch.save(
            {
                "srr": srr.state_dict(),
                "srr_cfg": srr.cfg.__dict__,
                "init": str(args.init),
                "epoch": epoch,
            },
            args.out / f"srr_ep{epoch}.pt",
        )

    decision = {
        "protocol": "R12-SCEB-B-SRR",
        "history": history,
        "final_heldout_state_acc": history[-1]["heldout_state_acc"] if history else 0.0,
        "advance_representation": (history[-1]["heldout_state_acc"] >= 0.80) if history else False,
    }
    (args.out / "decision.json").write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n")
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
