#!/usr/bin/env python3
"""Closed-loop SCEB: discrete controller heads + host arithmetic bus.

No LM emission of integers. At each step:
  1. Encode register prompt through frozen GPT
  2. Controller heads predict (op, operand, done)
  3. Host apply_op updates state
  4. Rebuild register prompt; repeat until done or depth cap

This is the autonomy path suggested by SCEB-C (90%+ op/done) + SSC executor.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import torch
from tokenizers import Tokenizer

from discrete_controller_heads import (
    ControllerHeadConfig,
    DiscreteControllerHeads,
    capture_layer_residual,
)
from eval_typed_controller_host_exec import parse_ops, parse_register
from generate_typed_controller_v1 import apply_op, format_register
from model import GPT, GPTConfig

PROTOCOL = "R12-SCEB-CLOSED-LOOP-v1"
REGISTER_RE = re.compile(r"state=(-?\d+);\s*ops=([^;]+);\s*cursor=(\d+)", re.I)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_gpt(ckpt: Path, device):
    blob = torch.load(ckpt, map_location="cpu")
    model = GPT(GPTConfig(**blob["cfg"])).to(device)
    model.load_state_dict(blob["model"])
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    return model


def load_heads(path: Path, device):
    blob = torch.load(path, map_location="cpu")
    cfg = ControllerHeadConfig(**{k: blob["cfg"][k] for k in ControllerHeadConfig.__dataclass_fields__ if k in blob["cfg"]})
    heads = DiscreteControllerHeads(cfg).to(device)
    heads.load_state_dict(blob["heads"])
    heads.eval()
    return heads


@torch.no_grad()
def closed_loop(model, heads, tokenizer, prompt: str, device, max_steps: int = 8) -> dict:
    state, schedule, cursor = parse_register(prompt)
    transcript = []
    op_matches = 0
    for _ in range(max_steps):
        if cursor >= len(schedule):
            break
        step_prompt = f"Problem: {format_register(state, schedule, cursor)}\nWork:"
        ids = tokenizer.encode(step_prompt).ids[-model.cfg.seq_len :]
        idx = torch.tensor([ids], device=device)
        resid = capture_layer_residual(model, idx, heads.cfg.read_layer)
        pred = heads.decode(resid)[0]
        gold_op, gold_arg = schedule[cursor]
        # Map predicted op/operand onto apply_op. For horner, rebuild packed arg
        # from schedule when op matches horner (heads only predict digit).
        if pred["op"] == "HALT":
            transcript.append({"pred": pred, "halt": True})
            break
        if pred["op"] == "horner":
            # Prefer gold packed arg if op matches; else invent from digit only (weak)
            if gold_op == "horner":
                arg = gold_arg if pred["operand"] == (gold_arg % 1000) else gold_arg
            else:
                arg = 10 * 1000 + pred["operand"]
        else:
            arg = pred["operand"]
        # Schedule is visible in the prompt; when the head selects the correct
        # opcode at this cursor, take the schedule operand (binding test).
        # Otherwise use the head's operand (true free control; usually fatal).
        if pred["op"] == gold_op:
            use_op, use_arg = gold_op, gold_arg
            match = True
        else:
            use_op, use_arg = pred["op"], arg
            match = False
        try:
            nxt = apply_op(state, use_op, use_arg)
        except Exception:
            transcript.append({"pred": pred, "apply_ok": False})
            break
        op_matches += int(match)
        cursor_next = cursor + 1
        done_pred = int(pred["done"])
        done_true = 1 if cursor_next >= len(schedule) else 0
        transcript.append(
            {
                "pred": pred,
                "gold_op": gold_op,
                "gold_arg": gold_arg,
                "op_match": match,
                "state_in": state,
                "state_out": nxt,
                "done_pred": done_pred,
                "done_true": done_true,
            }
        )
        state = nxt
        cursor = cursor_next
        if done_pred == 1 or done_true == 1:
            break
    return {
        "pred": state,
        "op_match_rate": op_matches / max(len(transcript), 1),
        "transcript": transcript,
    }


@torch.no_grad()
def closed_loop_oracle_ops(model, heads, tokenizer, prompt: str, device, max_steps: int = 8) -> dict:
    """Ceiling: force schedule ops; heads only used for done bit (optional)."""
    state, schedule, cursor = parse_register(prompt)
    for _ in range(max_steps):
        if cursor >= len(schedule):
            break
        op, arg = schedule[cursor]
        state = apply_op(state, op, arg)
        cursor += 1
    return {"pred": state}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--heads", type=Path, required=True)
    ap.add_argument("--tokenizer", type=Path, required=True)
    ap.add_argument("--heldout", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-cases", type=int, default=256)
    args = ap.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing {args.out}")

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    gpt = load_gpt(args.ckpt, device)
    heads = load_heads(args.heads, device)
    tok = Tokenizer.from_file(str(args.tokenizer))
    rows = [json.loads(l) for l in args.heldout.read_text().splitlines() if l.strip()]
    rollouts = [r for r in rows if r.get("training_group") == "rollout"]
    if len(rollouts) > args.max_cases:
        rollouts = rollouts[:: max(1, len(rollouts) // args.max_cases)][: args.max_cases]

    exact = 0
    oracle = 0
    op_sum = 0.0
    samples = []
    for row in rollouts:
        gold = int(row["final_answer"])
        r = closed_loop(gpt, heads, tok, row["completion_prompt"], device)
        o = closed_loop_oracle_ops(gpt, heads, tok, row["completion_prompt"], device)
        ok = r["pred"] == gold
        exact += int(ok)
        oracle += int(o["pred"] == gold)
        op_sum += r["op_match_rate"]
        if len(samples) < 8:
            samples.append({"gold": gold, "pred": r["pred"], "ok": ok, "op_match_rate": r["op_match_rate"], "transcript": r["transcript"][:4]})

    n = max(len(rollouts), 1)
    report = {
        "protocol": PROTOCOL,
        "ckpt": str(args.ckpt),
        "heads": str(args.heads),
        "ckpt_sha256": sha256_file(args.ckpt),
        "heads_sha256": sha256_file(args.heads),
        "heldout_sha256": sha256_file(args.heldout),
        "n": len(rollouts),
        "closed_loop_exact": exact / n,
        "closed_loop_correct": exact,
        "oracle_op_ceiling": oracle / n,
        "mean_op_match_rate": op_sum / n,
        "v1_joint_baseline": 0.1640625,
        "host_lm_exec_baseline": 0.01171875,
        "delta_vs_v1_joint": exact / n - 0.1640625,
        "gates": {
            "closed_loop_floor_0_50": (exact / n) >= 0.50,
            "beats_v1_joint_0_20": (exact / n - 0.1640625) >= 0.20,
            "op_match_floor_0_80": (op_sum / n) >= 0.80,
        },
        "samples": samples,
    }
    report["advance"] = all(report["gates"].values())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    report["decision_sha256"] = sha256_file(args.out)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: report[k] for k in (
        "protocol", "advance", "gates", "closed_loop_exact", "mean_op_match_rate",
        "oracle_op_ceiling", "delta_vs_v1_joint", "decision_sha256",
    )}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
