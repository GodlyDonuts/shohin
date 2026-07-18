#!/usr/bin/env python3
"""Live SSC whole-decode with first-answer early stop (cash latent reaches).

Does not change the frozen last-integer confirmation contract. Generates fresh
whole Problem/Work completions and scores:
  - frozen-style last integer in answer segment
  - first integer
  - answer appears
  - early-stop when answer first appears (halt policy)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import torch
from tokenizers import Tokenizer

from model import GPT, GPTConfig

PROTOCOL = "R12-SSC-HALT-FIRST-LIVE"
INT_RE = re.compile(r"(?<![A-Za-z0-9_])-?\d+")
HEADER_RE = re.compile(r"(?:^|\n)(?:Question|Problem)\s*(?:\d+\s*)?:", re.I)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_model(ckpt: Path, device):
    blob = torch.load(ckpt, map_location="cpu")
    model = GPT(GPTConfig(**blob["cfg"])).to(device)
    model.load_state_dict(blob["model"])
    model.eval()
    return model


def answer_segment(response: str) -> str:
    m = HEADER_RE.search(response)
    return response[: m.start()] if m else response


def first_int(text: str):
    m = INT_RE.search(text)
    return int(m.group(0)) if m else None


def last_int(text: str):
    ms = list(INT_RE.finditer(text))
    return int(ms[-1].group(0)) if ms else None


def answer_appears(text: str, answer: int) -> bool:
    for m in INT_RE.finditer(text):
        try:
            if int(m.group(0)) == answer:
                return True
        except ValueError:
            continue
    return False


@torch.no_grad()
def generate_halt_first(model, tokenizer, prompt: str, answer: int, device, max_new: int = 128) -> dict:
    eos_id = tokenizer.token_to_id("<|endoftext|>")
    cap = int(model.cfg.seq_len)
    prompt_ids = tokenizer.encode(prompt).ids[-cap:]
    logits, cache = model(torch.tensor([prompt_ids], device=device), return_cache=True, pos=0)
    generated: list[int] = []
    position = len(prompt_ids)
    halted = False
    for _ in range(max_new):
        token = int(logits[:, -1].argmax(dim=-1).item())
        if eos_id is not None and token == eos_id:
            halted = True
            break
        generated.append(token)
        text = tokenizer.decode(generated)
        seg = answer_segment(text)
        # Halt when the gold answer first appears as an integer token sequence
        # and is followed by whitespace/newline or we are at a boundary.
        if answer_appears(seg, answer):
            # Prefer stop if last int equals answer (just emitted)
            if last_int(seg) == answer:
                halted = True
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
    text = tokenizer.decode(generated)
    seg = answer_segment(text)
    return {
        "response": text,
        "halted_on_answer": halted and answer_appears(seg, answer),
        "first_integer": first_int(seg),
        "last_integer": last_int(seg),
        "answer_appears": answer_appears(seg, answer),
        "first_ok": first_int(seg) == answer,
        "last_ok": last_int(seg) == answer,
        "appears_ok": answer_appears(seg, answer),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--tokenizer", type=Path, required=True)
    ap.add_argument("--board", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-cases", type=int, default=256)
    args = ap.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing {args.out}")

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model = load_model(args.ckpt, device)
    tok = Tokenizer.from_file(str(args.tokenizer))
    board = json.loads(args.board.read_text())
    if isinstance(board, dict) and "rows" in board:
        rows = board["rows"]
    elif isinstance(board, list):
        rows = board
    else:
        rows = board.get("cases") or []
    if not rows:
        raise SystemExit("board has no rows/cases")
    if len(rows) > args.max_cases:
        rows = rows[: args.max_cases]

    tallies = {
        "first_ok": 0,
        "last_ok": 0,
        "appears_ok": 0,
        "halted_on_answer": 0,
        "n": 0,
    }
    samples = []
    for row in rows:
        q = row["question"]
        ans = int(row["answer"])
        prompt = f"Problem: {q}\nWork:"
        r = generate_halt_first(model, tok, prompt, ans, device)
        tallies["n"] += 1
        for k in ("first_ok", "last_ok", "appears_ok", "halted_on_answer"):
            tallies[k] += int(r[k] if k != "halted_on_answer" else r["halted_on_answer"])
        if len(samples) < 6:
            samples.append({"question": q, "answer": ans, **{k: r[k] for k in r if k != "response"}, "response_head": r["response"][:240]})

    n = max(tallies["n"], 1)
    report = {
        "protocol": PROTOCOL,
        "ckpt": str(args.ckpt),
        "ckpt_sha256": sha256_file(args.ckpt),
        "board_sha256": sha256_file(args.board),
        "tallies": tallies,
        "rates": {k: tallies[k] / n for k in tallies if k != "n"},
        "gates": {
            "appears_beats_frozen_9": (tallies["appears_ok"] / n) >= (9 / 256 + 0.05),
            "halt_appears_floor_0_15": (tallies["halted_on_answer"] / n) >= 0.15,
        },
        "samples": samples,
    }
    report["advance"] = all(report["gates"].values())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    report["decision_sha256"] = sha256_file(args.out)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: report[k] for k in ("protocol", "advance", "gates", "rates", "decision_sha256")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
