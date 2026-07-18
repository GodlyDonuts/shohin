#!/usr/bin/env python3
"""Evaluate typed-controller rollouts for R12-TYPED-CONTROLLER-v1."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import torch
from tokenizers import Tokenizer

from model import GPT, GPTConfig


PROTOCOL = "R12-TYPED-CONTROLLER-v1"
ANSWER_RE = re.compile(r"answer\s*=\s*(-?\d+)", re.I)
STEP_RE = re.compile(
    r"^(?P<op>add|subtract|multiply|remainder|horner)\s+"
    r"(?P<a>\d+)(?:\s+(?P<b>\d+))?\s*->\s*(?P<next>-?\d+);\s*"
    r"cursor=(?P<cursor>\d+);\s*done=(?P<done>[01])\s*$",
    re.I,
)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_model(ckpt: Path, device: torch.device):
    blob = torch.load(ckpt, map_location="cpu")
    cfg = GPTConfig(**blob["cfg"])
    model = GPT(cfg).to(device)
    model.load_state_dict(blob["model"])
    model.eval()
    return model, cfg


@torch.no_grad()
def greedy(model, tokenizer, prompt: str, device, max_new: int = 96) -> str:
    eos_id = tokenizer.token_to_id("<|endoftext|>")
    cap = int(model.cfg.seq_len)
    prompt_ids = tokenizer.encode(prompt).ids[-cap:]
    logits, cache = model(
        torch.tensor([prompt_ids], device=device), return_cache=True, pos=0
    )
    generated = []
    position = len(prompt_ids)
    for _ in range(max_new):
        token = int(logits[:, -1].argmax(dim=-1).item())
        if eos_id is not None and token == eos_id:
            break
        generated.append(token)
        text = tokenizer.decode(generated)
        # Early stop only after a finished answer line (newline or clear end),
        # never mid-digit while the model is still emitting the integer.
        if re.search(r"answer\s*=\s*-?\d+\s*$", text, re.I | re.M):
            # Peek one more token unless we already have a trailing newline.
            if text.endswith("\n") or text.rstrip() != text:
                break
            # Otherwise require that the next token is not a digit.
            # Defer break to after sampling one speculative token below via flag.
            pass
        if position >= cap:
            break
        logits, cache = model(
            torch.tensor([[token]], device=device),
            cache=cache,
            pos=position,
            return_cache=True,
        )
        position += 1
        # If previous text ended with answer=<digits> and this new token is not a digit, stop.
        if re.search(r"answer\s*=\s*-?\d+$", tokenizer.decode(generated[:-1]), re.I):
            piece = tokenizer.decode([token])
            if piece and not piece[0].isdigit():
                # Keep the non-digit delimiter out of the answer parse if it's junk
                if piece[0] in "\n\r ":
                    generated.pop()
                break
    return tokenizer.decode(generated)


def parse_final_state(text: str) -> int | None:
    """Prefer explicit answer=; else last done=1 transition arrow."""
    ans = parse_answer(text)
    if ans is not None:
        # Guard against truncated single-digit answers when a longer next exists.
        arrows = [int(m.group(1)) for m in re.finditer(r"->\s*(-?\d+)", text)]
        if arrows and abs(ans) < 10 and abs(arrows[-1]) >= 10:
            return arrows[-1]
        return ans
    arrows = list(re.finditer(r"->\s*(-?\d+).*done=1", text, re.I))
    if arrows:
        return int(arrows[-1].group(1))
    arrows = [int(m.group(1)) for m in re.finditer(r"->\s*(-?\d+)", text)]
    return arrows[-1] if arrows else None


def parse_answer(text: str) -> int | None:
    matches = list(ANSWER_RE.finditer(text))
    if not matches:
        return None
    return int(matches[-1].group(1))


def has_done(text: str) -> bool:
    return bool(re.search(r"done=1", text))


def evaluate_rows(model, tokenizer, rows, device) -> dict:
    # Fast board: 256 rollouts + 128 atomics is enough for the locked gates.
    rollouts = [r for r in rows if r.get("training_group") == "rollout"]
    atomics = [r for r in rows if r.get("training_group") == "atomic"]
    if len(rollouts) > 256:
        rollouts = rollouts[:: max(1, len(rollouts) // 256)][:256]
    if len(atomics) > 128:
        atomics = atomics[:: max(1, len(atomics) // 128)][:128]

    def score(subset, mode: str) -> dict:
        exact = 0
        done_flags = 0
        eos_like = 0
        transcripts = []
        for row in subset:
            prompt = row["completion_prompt"]
            text = greedy(model, tokenizer, prompt, device)
            pred = parse_final_state(text)
            gold = int(row["final_answer"])
            ok = pred == gold
            exact += int(ok)
            done = has_done(text)
            done_flags += int(done)
            # Heuristic: short completion relative to cap implies stop
            eos_like += int(len(text) < 180)
            if len(transcripts) < 8:
                transcripts.append(
                    {
                        "prompt": prompt,
                        "completion": text[:500],
                        "pred": pred,
                        "gold": gold,
                        "ok": ok,
                        "done": done,
                        "mode": mode,
                    }
                )
        n = max(len(subset), 1)
        return {
            "n": len(subset),
            "exact_accuracy": exact / n if subset else 0.0,
            "exact_correct": exact,
            "done_rate": done_flags / n if subset else 0.0,
            "short_completion_rate": eos_like / n if subset else 0.0,
            "transcripts": transcripts,
        }

    # Atomic: require the first step line's next value equals gold next from response target
    atomic_exact = 0
    for row in atomics:
        text = greedy(model, tokenizer, row["completion_prompt"], device)
        # Gold next is embedded in the supervised response; compare answer if present else first arrow
        gold = None
        m = re.search(r"->\s*(-?\d+)", row["response"])
        if m:
            gold = int(m.group(1))
        pred_m = re.search(r"->\s*(-?\d+)", text)
        pred = int(pred_m.group(1)) if pred_m else None
        if pred is not None and pred == gold:
            atomic_exact += 1
    atomic_n = max(len(atomics), 1)

    return {
        "rollout": score(rollouts, "rollout"),
        "atomic": {
            "n": len(atomics),
            "step_exact_accuracy": atomic_exact / atomic_n if atomics else 0.0,
            "exact_correct": atomic_exact,
        },
    }


def direct_baseline(model, tokenizer, rows, device) -> dict:
    """Natural-language direct QA on the underlying question text if present."""
    # Reconstruct a weak direct prompt from family register is not natural language.
    # Use rollout prompts' state line only as a control that the typed format matters.
    subset = [r for r in rows if r.get("training_group") == "rollout"]
    if len(subset) > 128:
        subset = subset[:: max(1, len(subset) // 128)][:128]
    exact = 0
    for row in subset:
        # Strip to ask for answer only
        prompt = row["completion_prompt"].replace("Work:", "Return only answer=<int>.\nAnswer:")
        text = greedy(model, tokenizer, prompt, device, max_new=32)
        pred = parse_answer(text)
        if pred is None:
            m = re.search(r"-?\d+", text)
            pred = int(m.group()) if m else None
        exact += int(pred == int(row["final_answer"]))
    n = max(len(subset), 1)
    return {"n": len(subset), "exact_accuracy": exact / n, "exact_correct": exact}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--tokenizer", type=Path, required=True)
    ap.add_argument("--heldout", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    if args.out.exists():
        raise SystemExit(f"refusing existing out: {args.out}")

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model, _cfg = load_model(args.ckpt, device)
    tok = Tokenizer.from_file(str(args.tokenizer))
    rows = [json.loads(line) for line in args.heldout.read_text().splitlines() if line.strip()]

    typed = evaluate_rows(model, tok, rows, device)
    direct = direct_baseline(model, tok, rows, device)
    report = {
        "protocol": PROTOCOL,
        "ckpt": str(args.ckpt),
        "ckpt_sha256": _sha256_file(args.ckpt),
        "heldout_sha256": _sha256_file(args.heldout),
        "typed": typed,
        "direct_control": direct,
        "margins": {
            "typed_minus_direct": typed["rollout"]["exact_accuracy"] - direct["exact_accuracy"],
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    report["report_sha256"] = _sha256_file(args.out)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: report[k] for k in ("protocol", "margins", "typed", "direct_control")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
