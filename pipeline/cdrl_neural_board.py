#!/usr/bin/env python3
"""R12-CDRL-NEURAL-v1: matched curriculum board on Heisenberg residuals.

Trains four equal-budget GRU arms (full/core/rand/hard) and writes a locked
decision JSON. No Shohin checkpoint, ACW path, or flagship output is touched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from pipeline.cdrl_conflict_cores import (
    extract_heisenberg_core,
    heisenberg_residual_key,
)


PROTOCOL = "R12-CDRL-NEURAL-v1"
EVENT_TO_ID = {"A": 0, "B": 1, "C": 2, "P": 3}
ID_TO_EVENT = {v: k for k, v in EVENT_TO_ID.items()}
ARMS = ("full", "core", "rand", "hard")
SEEDS = (2026071601, 2026071602, 2026071603)
MODULUS = 5
TRAIN_LEN_BAND = (8, 16)
OOD_LEN_BAND = (20, 28)
ESSENTIAL_BAND = (1, 4)
LABELS_PER_ARM = 12_288
UPDATES = 2_400
BATCH_SIZE = 256
HARD_PROBE_AT = 800
MARGIN = 0.05


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_json(payload: object) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return _sha256_bytes(blob)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass(frozen=True)
class Example:
    history: tuple[str, ...]
    core: tuple[str, ...]
    target: tuple[int, int, int]
    example_id: int


def _sample_history(rng: random.Random, lo: int, hi: int) -> tuple[str, ...]:
    length = rng.randint(lo, hi)
    n_essential = rng.randint(*ESSENTIAL_BAND)
    n_essential = min(n_essential, length)
    essentials = [rng.choice(["A", "B", "C"]) for _ in range(n_essential)]
    seq = ["P"] * length
    positions = sorted(rng.sample(range(length), n_essential))
    for pos, ev in zip(positions, essentials):
        seq[pos] = ev
    return tuple(seq)


def _core_strip_padding(history: Sequence[str]) -> tuple[str, ...]:
    """For this generator, P is residual-neutral and {A,B,C} are essential from 0."""
    return tuple(event for event in history if event != "P")


def build_split(
    *,
    seed: int,
    n: int,
    lo: int,
    hi: int,
    salt: str,
) -> list[Example]:
    mix = int(_sha256_bytes(f"{seed}:{salt}:{n}:{lo}:{hi}".encode())[:16], 16)
    rng = random.Random(mix)
    out: list[Example] = []
    for i in range(n):
        history = _sample_history(rng, lo, hi)
        core = _core_strip_padding(history)
        # Spot-check against the general extractor on a prefix of the stream.
        if i < 32:
            extraction = extract_heisenberg_core(history, modulus=MODULUS)
            if extraction.core != core:
                raise RuntimeError(
                    f"fast core mismatch at {i}: {core=} {extraction.core=}"
                )
        target = heisenberg_residual_key(history, modulus=MODULUS)
        out.append(
            Example(
                history=history,
                core=core,
                target=target,
                example_id=i,
            )
        )
    return out


def encode_events(events: Sequence[str], max_len: int) -> torch.Tensor:
    ids = [EVENT_TO_ID[e] for e in events]
    if len(ids) > max_len:
        ids = ids[-max_len:]
    pad = max_len - len(ids)
    return torch.tensor([3] * pad + ids, dtype=torch.long)  # pad with P on the left


def random_subsequence(
    history: Sequence[str], length: int, *, example_id: int, seed: int
) -> tuple[str, ...]:
    if length <= 0:
        return ()
    if length >= len(history):
        return tuple(history)
    mix = int(_sha256_bytes(f"rand:{seed}:{example_id}:{length}".encode())[:16], 16)
    rng = random.Random(mix)
    idxs = sorted(rng.sample(range(len(history)), length))
    return tuple(history[i] for i in idxs)


class ResidualGRU(nn.Module):
    def __init__(self, *, n_events: int = 4, d_model: int = 32, h: int = 64, modulus: int = 5):
        super().__init__()
        self.modulus = modulus
        self.embed = nn.Embedding(n_events, d_model)
        self.gru = nn.GRU(d_model, h, batch_first=True)
        self.heads = nn.ModuleList([nn.Linear(h, modulus) for _ in range(3)])

    def forward(self, event_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # event_ids: [B, T]
        x = self.embed(event_ids)
        _, h_n = self.gru(x)
        h = h_n.squeeze(0)
        return tuple(head(h) for head in self.heads)  # type: ignore[return-value]


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def sequences_for_arm(
    examples: Sequence[Example], arm: str, *, seed: int
) -> list[tuple[tuple[str, ...], tuple[int, int, int], int]]:
    rows = []
    for ex in examples:
        if arm == "full" or arm == "hard":
            seq = ex.history
        elif arm == "core":
            seq = ex.core
        elif arm == "rand":
            seq = random_subsequence(ex.history, len(ex.core), example_id=ex.example_id, seed=seed)
        else:
            raise ValueError(arm)
        rows.append((seq, ex.target, ex.example_id))
    return rows


@torch.no_grad()
def evaluate(
    model: ResidualGRU,
    examples: Sequence[Example],
    *,
    device: torch.device,
    max_len: int,
) -> dict:
    """Evaluate on full histories (distractors present), regardless of train arm."""
    model.eval()
    exact = 0
    coord = 0
    total = len(examples)
    for ex in examples:
        ids = encode_events(ex.history, max_len).unsqueeze(0).to(device)
        logits = model(ids)
        pred = tuple(int(torch.argmax(logit, dim=-1).item()) for logit in logits)
        if pred == ex.target:
            exact += 1
        coord += sum(int(p == t) for p, t in zip(pred, ex.target))
    return {
        "n": total,
        "exact_accuracy": exact / total,
        "coord_accuracy": coord / (total * 3),
        "exact_correct": exact,
    }


def _ce_loss(logits: Sequence[torch.Tensor], targets: torch.Tensor) -> torch.Tensor:
    loss = logits[0].new_zeros(())
    for i, logit in enumerate(logits):
        loss = loss + F.cross_entropy(logit, targets[:, i])
    return loss


def train_arm(
    *,
    arm: str,
    train_examples: Sequence[Example],
    seed: int,
    device: torch.device,
    max_len: int,
    out_dir: Path,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "model.pt"
    receipt_path = out_dir / "train_receipt.json"
    if model_path.exists() or receipt_path.exists():
        raise FileExistsError(f"refusing overwrite under {out_dir}")

    torch.manual_seed(seed + ARMS.index(arm) * 17)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed + ARMS.index(arm) * 17)

    model = ResidualGRU().to(device)
    n_params = count_parameters(model)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

    rows = sequences_for_arm(train_examples, "full" if arm == "hard" else arm, seed=seed)
    # Pre-encode
    xs = torch.stack([encode_events(seq, max_len) for seq, _, _ in rows])
    ys = torch.tensor([list(target) for _, target, _ in rows], dtype=torch.long)

    hard_weights = torch.ones(len(rows), dtype=torch.double)
    losses_probe: list[float] = []

    model.train()
    t0 = time.time()
    update = 0
    while update < UPDATES:
        if arm == "hard" and update == HARD_PROBE_AT:
            model.eval()
            with torch.no_grad():
                per_loss = []
                for i in range(0, len(rows), BATCH_SIZE):
                    xb = xs[i : i + BATCH_SIZE].to(device)
                    yb = ys[i : i + BATCH_SIZE].to(device)
                    logits = model(xb)
                    # per-example loss
                    part = sum(
                        F.cross_entropy(logit, yb[:, j], reduction="none")
                        for j, logit in enumerate(logits)
                    )
                    per_loss.extend(part.detach().cpu().tolist())
            losses_probe = per_loss
            thresh = sorted(per_loss)[int(0.75 * (len(per_loss) - 1))]
            hard_weights = torch.tensor(
                [4.0 if loss >= thresh else 1.0 for loss in per_loss],
                dtype=torch.double,
            )
            model.train()

        if arm == "hard" and update >= HARD_PROBE_AT:
            # Weighted sampling without replacement in a batch via multinomial
            chosen = torch.multinomial(hard_weights, BATCH_SIZE, replacement=True)
        else:
            start = (update * BATCH_SIZE) % len(rows)
            idx = [(start + k) % len(rows) for k in range(BATCH_SIZE)]
            chosen = torch.tensor(idx, dtype=torch.long)

        xb = xs[chosen].to(device)
        yb = ys[chosen].to(device)
        opt.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = _ce_loss(logits, yb)
        loss.backward()
        opt.step()
        update += 1

    elapsed = time.time() - t0
    payload = {
        "protocol": PROTOCOL,
        "arm": arm,
        "seed": seed,
        "n_params": n_params,
        "labels": len(rows),
        "updates": UPDATES,
        "batch_size": BATCH_SIZE,
        "seconds": elapsed,
        "hard_probe_at": HARD_PROBE_AT if arm == "hard" else None,
        "hard_probe_n": len(losses_probe) if losses_probe else 0,
        "final_loss": float(loss.detach().cpu()),
        "device": str(device),
    }
    torch.save({"model": model.state_dict(), "meta": payload}, model_path)
    payload["model_sha256"] = _sha256_file(model_path)
    receipt_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    payload["receipt_sha256"] = _sha256_file(receipt_path)
    receipt_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def run_seed(seed: int, artifact_root: Path, device: torch.device) -> dict:
    seed_dir = artifact_root / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    eval_path = seed_dir / "eval.json"
    if eval_path.exists():
        raise FileExistsError(eval_path)

    train_examples = build_split(
        seed=seed, n=LABELS_PER_ARM, lo=TRAIN_LEN_BAND[0], hi=TRAIN_LEN_BAND[1], salt="train"
    )
    fit_examples = build_split(
        seed=seed, n=2_048, lo=TRAIN_LEN_BAND[0], hi=TRAIN_LEN_BAND[1], salt="fit_eval"
    )
    ood_examples = build_split(
        seed=seed, n=2_048, lo=OOD_LEN_BAND[0], hi=OOD_LEN_BAND[1], salt="ood_eval"
    )
    max_len = OOD_LEN_BAND[1]

    # Sanity: cores should often be shorter on train
    shorter = sum(1 for ex in train_examples if len(ex.core) < len(ex.history))
    arm_receipts = {}
    arm_metrics = {}
    for arm in ARMS:
        receipt = train_arm(
            arm=arm,
            train_examples=train_examples,
            seed=seed,
            device=device,
            max_len=max_len,
            out_dir=seed_dir / f"arm_{arm}",
        )
        arm_receipts[arm] = receipt
        # reload
        model = ResidualGRU().to(device)
        blob = torch.load(seed_dir / f"arm_{arm}" / "model.pt", map_location=device)
        model.load_state_dict(blob["model"])
        fit = evaluate(model, fit_examples, device=device, max_len=max_len)
        ood = evaluate(model, ood_examples, device=device, max_len=max_len)
        arm_metrics[arm] = {"fit": fit, "ood": ood, "n_params": receipt["n_params"]}

    core_ood = arm_metrics["core"]["ood"]["exact_accuracy"]
    decision = {
        "core_minus_full": core_ood - arm_metrics["full"]["ood"]["exact_accuracy"],
        "core_minus_rand": core_ood - arm_metrics["rand"]["ood"]["exact_accuracy"],
        "core_minus_hard": core_ood - arm_metrics["hard"]["ood"]["exact_accuracy"],
    }
    advance = all(v >= MARGIN for v in decision.values())
    report = {
        "protocol": PROTOCOL,
        "seed": seed,
        "train_core_shorter_fraction": shorter / len(train_examples),
        "arm_metrics": arm_metrics,
        "margins": decision,
        "margin_required": MARGIN,
        "advance_seed": advance,
        "param_counts": {arm: arm_metrics[arm]["n_params"] for arm in ARMS},
    }
    eval_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    report["eval_sha256"] = _sha256_file(eval_path)
    eval_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def decide(seed_reports: Sequence[dict]) -> dict:
    # Median over seeds on each margin, then gate
    def median(xs: list[float]) -> float:
        ys = sorted(xs)
        return ys[len(ys) // 2]

    margins = {
        key: median([r["margins"][key] for r in seed_reports])
        for key in ("core_minus_full", "core_minus_rand", "core_minus_hard")
    }
    advance = all(v >= MARGIN for v in margins.values())
    return {
        "protocol": PROTOCOL,
        "seeds": [r["seed"] for r in seed_reports],
        "median_margins": margins,
        "margin_required": MARGIN,
        "advance": advance,
        "per_seed_advance": {str(r["seed"]): r["advance_seed"] for r in seed_reports},
        "ood_exact_by_seed_arm": {
            str(r["seed"]): {arm: r["arm_metrics"][arm]["ood"]["exact_accuracy"] for arm in ARMS}
            for r in seed_reports
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-root",
        type=Path,
        required=True,
        help="Isolated output root (must not exist as a completed decision)",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(SEEDS),
    )
    args = parser.parse_args(argv)

    artifact_root = args.artifact_root
    artifact_root.mkdir(parents=True, exist_ok=True)
    decision_path = artifact_root / "decision.json"
    if decision_path.exists():
        raise FileExistsError(decision_path)

    prereg = Path(__file__).resolve().parents[1] / "R12_CDRL_NEURAL_OPTIMIZATION_PREREG.md"
    if prereg.is_file():
        (artifact_root / "prereg_sha256.txt").write_text(_sha256_file(prereg) + "\n")

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    reports = []
    for seed in args.seeds:
        print(f"[cdrl-neural] seed={seed} device={device}", flush=True)
        reports.append(run_seed(seed, artifact_root, device))
        print(
            f"[cdrl-neural] seed={seed} advance={reports[-1]['advance_seed']} "
            f"margins={reports[-1]['margins']}",
            flush=True,
        )

    decision = decide(reports)
    decision_path.write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n")
    decision["decision_sha256"] = _sha256_file(decision_path)
    decision_path.write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n")
    print(json.dumps(decision, indent=2, sort_keys=True))
    # A locked reject is a valid completed board, not a job failure.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
