#!/usr/bin/env python3
"""Host-executed typed controller bus (SCEB-A).

Model proposes one typed step; host overrides the integer with apply_op and
feeds the corrected register prompt back. Measures whether op/cursor control
(not value emission) is the remaining bottleneck.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import torch
from tokenizers import Tokenizer

from generate_typed_controller_v1 import apply_op, format_ops, format_register, step_line
from model import GPT, GPTConfig

PROTOCOL = "R12-SCEB-A-HOST-EXEC"
STEP_RE = re.compile(
    r"(?P<op>add|subtract|multiply|remainder|horner)\s+"
    r"(?P<a>\d+)(?:\s+(?P<b>\d+))?\s*->\s*(?P<next>-?\d+);\s*"
    r"cursor=(?P<cursor>\d+);\s*done=(?P<done>[01])",
    re.I,
)
REGISTER_RE = re.compile(
    r"state=(-?\d+);\s*ops=([^;]+);\s*cursor=(\d+)",
    re.I,
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_model(ckpt: Path, device: torch.device):
    blob = torch.load(ckpt, map_location="cpu")
    model = GPT(GPTConfig(**blob["cfg"])).to(device)
    model.load_state_dict(blob["model"])
    model.eval()
    return model


@torch.no_grad()
def greedy_until_step(model, tokenizer, prompt: str, device, max_new: int = 64) -> str:
    """Greedy decode until one full step line is matched."""
    eos_id = tokenizer.token_to_id("<|endoftext|>")
    cap = int(model.cfg.seq_len)
    prompt_ids = tokenizer.encode(prompt).ids[-cap:]
    logits, cache = model(
        torch.tensor([prompt_ids], device=device), return_cache=True, pos=0
    )
    generated: list[int] = []
    position = len(prompt_ids)
    for _ in range(max_new):
        token = int(logits[:, -1].argmax(dim=-1).item())
        if eos_id is not None and token == eos_id:
            break
        generated.append(token)
        text = tokenizer.decode(generated)
        if STEP_RE.search(text):
            break
        if position >= cap:
            break
        logits, cache = model(
            torch.tensor([[token]], device=device),
            cache=cache,
            pos=position,
            return_cache=True,
        )
        position += 1
    return tokenizer.decode(generated)


def parse_ops(ops_str: str) -> list[tuple[str, int]]:
    parts = [p.strip() for p in ops_str.split("|")]
    out: list[tuple[str, int]] = []
    for p in parts:
        toks = p.split()
        if not toks:
            continue
        op = toks[0].lower()
        if op == "horner":
            base, digit = int(toks[1]), int(toks[2])
            out.append(("horner", base * 1000 + digit))
        else:
            out.append((op, int(toks[1])))
    return out


def parse_register(prompt: str) -> tuple[int, list[tuple[str, int]], int]:
    m = REGISTER_RE.search(prompt)
    if not m:
        raise ValueError(f"unparsed register prompt: {prompt[:120]}")
    state = int(m.group(1))
    ops = parse_ops(m.group(2))
    cursor = int(m.group(3))
    return state, ops, cursor


def parse_model_step(text: str) -> dict | None:
    m = STEP_RE.search(text)
    if not m:
        return None
    op = m.group("op").lower()
    if op == "horner":
        arg = int(m.group("a")) * 1000 + int(m.group("b"))
    else:
        arg = int(m.group("a"))
    return {
        "op": op,
        "arg": arg,
        "next_claimed": int(m.group("next")),
        "cursor": int(m.group("cursor")),
        "done": int(m.group("done")),
    }


def host_exec_rollout(model, tokenizer, prompt: str, device, max_steps: int = 8) -> dict:
    state, schedule, cursor = parse_register(prompt)
    gold_final = None
    # Derive gold by replaying full schedule from initial state embedded in first prompt
    s0 = state
    # If cursor>0, rewind is hard; prefer gold from row. Caller passes final_answer.
    transcript = []
    op_matches = 0
    steps = 0
    for _ in range(max_steps):
        if cursor >= len(schedule):
            break
        step_prompt = f"Problem: {format_register(state, schedule, cursor)}\nWork:"
        raw = greedy_until_step(model, tokenizer, step_prompt, device)
        parsed = parse_model_step(raw)
        gold_op, gold_arg = schedule[cursor]
        if parsed is None:
            transcript.append({"raw": raw[:200], "parse_ok": False})
            break
        match = parsed["op"] == gold_op and parsed["arg"] == gold_arg
        op_matches += int(match)
        # Host overrides arithmetic using gold schedule op (controller still must advance).
        # Primary arm: use *model-emitted* op for apply_op (tests op selection).
        try:
            nxt = apply_op(state, parsed["op"], parsed["arg"])
        except Exception:
            transcript.append({"raw": raw[:200], "parse_ok": True, "apply_ok": False})
            break
        cursor_next = cursor + 1
        done = 1 if cursor_next >= len(schedule) else 0
        # If model selected wrong op, still advance cursor on gold schedule to measure
        # pure arithmetic override separately via secondary metrics.
        host_line = step_line(parsed["op"], parsed["arg"], nxt, cursor_next, done)
        transcript.append(
            {
                "raw": raw[:200],
                "host_line": host_line,
                "op_match": match,
                "state_in": state,
                "state_out": nxt,
                "claimed_next": parsed["next_claimed"],
                "arith_match_claimed": parsed["next_claimed"] == nxt,
            }
        )
        state = nxt
        cursor = cursor_next
        steps += 1
        if done:
            break
    # Also compute gold-schedule host path (oracle ops) for ceiling
    return {
        "pred": state,
        "steps": steps,
        "op_match_rate": op_matches / max(steps, 1),
        "op_matches": op_matches,
        "transcript": transcript,
        "initial_state": s0,
        "schedule_len": len(schedule),
    }


def oracle_host_ceiling(prompt: str) -> int:
    state, schedule, cursor = parse_register(prompt)
    for i in range(cursor, len(schedule)):
        op, arg = schedule[i]
        state = apply_op(state, op, arg)
    return state


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
        raise SystemExit(f"refusing existing out: {args.out}")

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model = load_model(args.ckpt, device)
    tok = Tokenizer.from_file(str(args.tokenizer))
    rows = [json.loads(l) for l in args.heldout.read_text().splitlines() if l.strip()]
    rollouts = [r for r in rows if r.get("training_group") == "rollout"]
    if len(rollouts) > args.max_cases:
        rollouts = rollouts[:: max(1, len(rollouts) // args.max_cases)][: args.max_cases]

    exact = 0
    oracle_exact = 0
    op_match_sum = 0.0
    arith_agree = 0
    arith_n = 0
    samples = []
    for row in rollouts:
        prompt = row["completion_prompt"]
        gold = int(row["final_answer"])
        result = host_exec_rollout(model, tok, prompt, device)
        ok = result["pred"] == gold
        exact += int(ok)
        ceil = oracle_host_ceiling(prompt)
        oracle_exact += int(ceil == gold)
        op_match_sum += result["op_match_rate"]
        for step in result["transcript"]:
            if "arith_match_claimed" in step:
                arith_n += 1
                arith_agree += int(step["arith_match_claimed"])
        if len(samples) < 8:
            samples.append(
                {
                    "prompt": prompt,
                    "gold": gold,
                    "pred": result["pred"],
                    "ok": ok,
                    "op_match_rate": result["op_match_rate"],
                    "transcript": result["transcript"][:4],
                }
            )

    n = max(len(rollouts), 1)
    report = {
        "protocol": PROTOCOL,
        "ckpt": str(args.ckpt),
        "ckpt_sha256": sha256_file(args.ckpt),
        "heldout_sha256": sha256_file(args.heldout),
        "n": len(rollouts),
        "host_exec_exact": exact / n,
        "host_exec_correct": exact,
        "oracle_schedule_ceiling": oracle_exact / n,
        "mean_op_match_rate": op_match_sum / n,
        "model_arith_agree_with_host": (arith_agree / arith_n) if arith_n else 0.0,
        "v1_joint_baseline": 0.1640625,
        "delta_vs_v1_joint": exact / n - 0.1640625,
        "gates": {
            "host_exec_floor_0_50": (exact / n) >= 0.50,
            "advantage_vs_v1_0_20": (exact / n - 0.1640625) >= 0.20,
            "op_match_floor_0_80": (op_match_sum / n) >= 0.80,
        },
        "samples": samples,
    }
    report["advance"] = all(report["gates"].values())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    report["decision_sha256"] = sha256_file(args.out)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: report[k] for k in (
        "protocol", "advance", "gates", "host_exec_exact", "mean_op_match_rate",
        "model_arith_agree_with_host", "delta_vs_v1_joint", "decision_sha256",
    )}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
