#!/usr/bin/env python3
"""Constrained schedule-op bus: LM chooses among ops parsed from the prompt.

At each cursor the legal opcode set is exactly the remaining schedule ops
(usually one: schedule[cursor]). Logits for all other op-leading tokens are
masked. Host applies the chosen (forced) schedule step. This isolates whether
DONE/HALT is the remaining failure once opcode identity is given by the text.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import torch
from tokenizers import Tokenizer

from eval_typed_controller_host_exec import parse_register
from generate_typed_controller_v1 import apply_op, format_register, step_line
from model import GPT, GPTConfig

PROTOCOL = "R12-SCEB-CONSTRAINED-OP-v1"
STEP_RE = re.compile(
    r"(?P<op>add|subtract|multiply|remainder|horner)\s+"
    r"(?P<a>\d+)(?:\s+(?P<b>\d+))?\s*->\s*(?P<next>-?\d+);\s*"
    r"cursor=(?P<cursor>\d+);\s*done=(?P<done>[01])",
    re.I,
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_model(ckpt: Path, device):
    blob = torch.load(ckpt, map_location="cpu")
    model = GPT(GPTConfig(**blob["cfg"])).to(device)
    model.load_state_dict(blob["model"])
    model.eval()
    return model


@torch.no_grad()
def greedy_step_constrained(model, tokenizer, prompt: str, legal_op: str, device, max_new: int = 48) -> str:
    """Greedy decode; at the first token, mask to legal op name prefixes."""
    eos_id = tokenizer.token_to_id("<|endoftext|>")
    # Candidate first-token ids that decode to a prefix of legal_op
    vocab = tokenizer.get_vocab()
    legal_ids = []
    for tok, idx in vocab.items():
        if legal_op.startswith(tok) or tok.startswith(legal_op[: max(1, min(3, len(legal_op)))]):
            # keep tokens that are prefixes of the op or the op is prefix of token
            if tok and (legal_op.startswith(tok) or tok.startswith(legal_op)):
                legal_ids.append(idx)
    # Also add exact piece encodings
    for piece in (legal_op, legal_op + " ", legal_op[:1], legal_op[:2], legal_op[:3]):
        ids = tokenizer.encode(piece).ids
        if ids:
            legal_ids.append(ids[0])
    legal_ids = list(set(legal_ids))
    if not legal_ids:
        legal_ids = tokenizer.encode(legal_op).ids[:1]

    prompt_ids = tokenizer.encode(prompt).ids[-model.cfg.seq_len :]
    logits, cache = model(torch.tensor([prompt_ids], device=device), return_cache=True, pos=0)
    generated = []
    position = len(prompt_ids)
    first = True
    for _ in range(max_new):
        row = logits[:, -1].clone()
        if first and legal_ids:
            mask = torch.full_like(row, float("-inf"))
            mask[:, legal_ids] = row[:, legal_ids]
            row = mask
            first = False
        token = int(row.argmax(dim=-1).item())
        if eos_id is not None and token == eos_id:
            break
        generated.append(token)
        text = tokenizer.decode(generated)
        if STEP_RE.search(text):
            break
        logits, cache = model(
            torch.tensor([[token]], device=device), cache=cache, pos=position, return_cache=True
        )
        position += 1
    return tokenizer.decode(generated)


def run_case(model, tokenizer, prompt: str, device) -> dict:
    state, schedule, cursor = parse_register(prompt)
    # Arm A: pure host schedule follower (ceiling)
    s = state
    for i in range(cursor, len(schedule)):
        s = apply_op(s, schedule[i][0], schedule[i][1])
    ceiling = s

    # Arm B: constrained LM emit + host override arithmetic using schedule[cursor]
    state_b, cursor_b = state, cursor
    transcript = []
    for _ in range(8):
        if cursor_b >= len(schedule):
            break
        op, arg = schedule[cursor_b]
        step_prompt = f"Problem: {format_register(state_b, schedule, cursor_b)}\nWork:"
        raw = greedy_step_constrained(model, tokenizer, step_prompt, op, device)
        nxt = apply_op(state_b, op, arg)
        cursor_b += 1
        done = 1 if cursor_b >= len(schedule) else 0
        transcript.append({"raw": raw[:160], "host": step_line(op, arg, nxt, cursor_b, done)})
        state_b = nxt
        if done:
            break
    return {"ceiling": ceiling, "constrained_host": state_b, "transcript": transcript}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--tokenizer", type=Path, required=True)
    ap.add_argument("--heldout", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-cases", type=int, default=256)
    args = ap.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing {args.out}")

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model = load_model(args.ckpt, device)
    tok = Tokenizer.from_file(str(args.tokenizer))
    rows = [json.loads(l) for l in args.heldout.read_text().splitlines() if l.strip()]
    rollouts = [r for r in rows if r.get("training_group") == "rollout"]
    if len(rollouts) > args.max_cases:
        rollouts = rollouts[:: max(1, len(rollouts) // args.max_cases)][: args.max_cases]

    exact = 0
    ceil = 0
    samples = []
    for row in rollouts:
        gold = int(row["final_answer"])
        r = run_case(model, tok, row["completion_prompt"], device)
        ok = r["constrained_host"] == gold
        exact += int(ok)
        ceil += int(r["ceiling"] == gold)
        if len(samples) < 5:
            samples.append({"gold": gold, "pred": r["constrained_host"], "ok": ok, "transcript": r["transcript"][:3]})

    n = max(len(rollouts), 1)
    report = {
        "protocol": PROTOCOL,
        "ckpt": str(args.ckpt),
        "ckpt_sha256": sha256_file(args.ckpt),
        "n": len(rollouts),
        "constrained_host_exact": exact / n,
        "schedule_ceiling": ceil / n,
        "v1_joint_baseline": 0.1640625,
        "sceb_v2_baseline": 0.25390625,
        "delta_vs_v1": exact / n - 0.1640625,
        "delta_vs_sceb_v2": exact / n - 0.25390625,
        "gates": {
            "floor_0_50": (exact / n) >= 0.50,
            "beats_sceb_v2": (exact / n) >= 0.25390625 + 0.05,
        },
        "samples": samples,
    }
    report["advance"] = all(report["gates"].values())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    report["decision_sha256"] = sha256_file(args.out)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: report[k] for k in report if k != "samples"}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
