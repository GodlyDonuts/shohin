#!/usr/bin/env python3
"""NL SCEB: propose ops from natural-language questions (no schedule in prompt).

This is the non-trivial internalization target. Typed-register boards with a
visible schedule have a host ceiling of 100%; NL does not.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from discrete_controller_heads import (
    OP2ID,
    OPCODES,
    ControllerHeadConfig,
    DiscreteControllerHeads,
    capture_layer_residual,
)
from generate_typed_controller_v1 import apply_op
from model import GPT, GPTConfig

PROTOCOL = "R12-SCEB-NL-v1"


def load_frozen(ckpt, device):
    blob = torch.load(ckpt, map_location="cpu")
    model = GPT(GPTConfig(**blob["cfg"])).to(device)
    model.load_state_dict(blob["model"])
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    return model


def parse_board_case(row: dict) -> dict:
    """Normalize SSC board row to schedule list[(op,arg)]."""
    # Board rows include schedule as list of dicts or tuples
    sched = row["schedule"]
    out = []
    for step in sched:
        if isinstance(step, dict):
            out.append((step["op"], int(step["arg"])))
        elif isinstance(step, (list, tuple)):
            out.append((step[0], int(step[1])))
        else:
            raise ValueError(step)
    return {
        "question": row["question"],
        "answer": int(row["answer"]),
        "initial_state": int(row["initial_state"]),
        "schedule": out,
        "family": row["family"],
    }


def nl_prompt(question: str, state: int, step_i: int) -> str:
    return (
        f"Problem: {question}\n"
        f"Current state: {state}\n"
        f"Step index: {step_i}\n"
        f"Emit next opcode and operand."
    )


def build_examples(board_path: Path) -> list[dict]:
    board = json.loads(board_path.read_text())
    rows = board["rows"]
    ex = []
    for row in rows:
        case = parse_board_case(row)
        state = case["initial_state"]
        for i, (op, arg) in enumerate(case["schedule"]):
            done = 1 if i + 1 == len(case["schedule"]) else 0
            operand = arg if op != "horner" else arg  # board may not use horner packing
            # SSC board uses multiply/add separately for base conversion
            ex.append(
                {
                    "prompt": nl_prompt(case["question"], state, i),
                    "op": op,
                    "operand": int(operand) % 1000,
                    "done": done,
                    "answer": case["answer"],
                    "family": case["family"],
                }
            )
            state = apply_op(state, op, arg) if op != "horner" else apply_op(state, op, arg)
    return ex


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--init", type=Path, required=True)
    ap.add_argument("--board", type=Path, required=True)
    ap.add_argument("--tokenizer", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing {args.out}")
    args.out.mkdir(parents=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    gpt = load_frozen(args.init, device)
    heads = DiscreteControllerHeads(ControllerHeadConfig(d_model=gpt.cfg.d_model)).to(device)
    for p in heads.operand_head.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW([p for p in heads.parameters() if p.requires_grad], lr=args.lr)
    tok = Tokenizer.from_file(str(args.tokenizer))

    examples = build_examples(args.board)
    rng = random.Random(args.seed)
    rng.shuffle(examples)
    # Split by question to avoid leakage
    questions = sorted({e["prompt"].split("\n")[0] for e in examples})
    rng.shuffle(questions)
    n_hold = max(1, len(questions) // 5)
    hold_q = set(questions[:n_hold])
    train = [e for e in examples if e["prompt"].split("\n")[0] not in hold_q]
    held = [e for e in examples if e["prompt"].split("\n")[0] in hold_q]

    def run_epoch(rows, train_mode):
        heads.train(train_mode)
        total = 0.0
        nb = 0
        correct = 0
        for i in range(0, len(rows), args.batch_size):
            batch = rows[i : i + args.batch_size]
            encoded = [tok.encode(r["prompt"]).ids[-gpt.cfg.seq_len :] for r in batch]
            max_len = max(len(e) for e in encoded)
            ids = [[0] * (max_len - len(e)) + e for e in encoded]
            idx = torch.tensor(ids, device=device)
            with torch.no_grad():
                resid = capture_layer_residual(gpt, idx, heads.cfg.read_layer)
            # skip unknown ops
            usable = []
            for r in batch:
                if r["op"] in OP2ID:
                    usable.append(r)
                else:
                    usable.append(None)
            if not any(usable):
                continue
            op_id = torch.tensor([OP2ID.get(r["op"], 0) if r else 0 for r in batch], device=device)
            done = torch.tensor([r["done"] if r else 0 for r in batch], device=device)
            out = heads.forward(resid)
            loss = F.cross_entropy(out["op_logits"], op_id) + F.cross_entropy(out["done_logits"], done)
            if train_mode:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                total += float(loss.item())
            else:
                pred = heads.decode(resid)
                for p, r in zip(pred, batch):
                    if r and p["op"] == r["op"] and p["done"] == r["done"]:
                        correct += 1
            nb += 1
        return total / max(nb, 1), correct / max(len(rows), 1)

    history = []
    for ep in range(1, args.epochs + 1):
        tr, _ = run_epoch(train, True)
        _, ha = run_epoch(held, False)
        row = {"epoch": ep, "train_loss": tr, "heldout_op_done_acc": ha}
        history.append(row)
        print(json.dumps(row), flush=True)
        torch.save({"heads": heads.state_dict(), "cfg": heads.cfg.__dict__, "epoch": ep}, args.out / f"heads_ep{ep}.pt")

    # Closed-loop NL eval on held questions
    hold_questions = {q.replace("Problem: ", "", 1) for q in hold_q}
    board = json.loads(args.board.read_text())
    held_cases = []
    for row in board["rows"]:
        if row["question"] in hold_questions:
            held_cases.append(parse_board_case(row))

    exact = 0
    with torch.no_grad():
        for case in held_cases:
            state = case["initial_state"]
            ok_chain = True
            for i, (op, arg) in enumerate(case["schedule"]):
                prompt = nl_prompt(case["question"], state, i)
                ids = tok.encode(prompt).ids[-gpt.cfg.seq_len :]
                idx = torch.tensor([ids], device=device)
                resid = capture_layer_residual(gpt, idx, heads.cfg.read_layer)
                pred = heads.decode(resid)[0]
                if pred["op"] != op:
                    ok_chain = False
                    break
                state = apply_op(state, op, arg)
            if ok_chain and state == case["answer"]:
                exact += 1

    n = max(len(held_cases), 1)
    decision = {
        "protocol": PROTOCOL,
        "history": history,
        "n_held_cases": len(held_cases),
        "closed_loop_exact": exact / n,
        "closed_loop_correct": exact,
        "final_heldout_op_done_acc": history[-1]["heldout_op_done_acc"],
        "advance": (exact / n) >= 0.20,
        "note": "NL board; schedule not in prompt. Typed SCEB ceiling is separate.",
    }
    (args.out / "decision.json").write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n")
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
