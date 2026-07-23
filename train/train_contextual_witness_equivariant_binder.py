"""Train and audit the contextual witness-equivariant primitive binder."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import random
from typing import Any

import torch
import torch.nn.functional as F

from contextual_relation_primitive_compiler import (
    PRIMITIVE_COUNT,
    relation_primitive_candidates,
)
from contextual_witness_card_data import (
    SHIFT_DENSITIES,
    TRAIN_DENSITIES,
    ContextualCardBatch,
    generate_contextual_card_batch,
)
from contextual_witness_equivariant_binder import (
    BINDER_CLASS_COUNT,
    REJECT_INDEX,
    ContextualWitnessEquivariantBinder,
    ContextualWitnessStatisticsBinder,
)


DEFAULT_SEED = 2026072327
IGNORE_INDEX = -100


@dataclass(frozen=True, slots=True)
class TrainConfig:
    seed: int = DEFAULT_SEED
    steps: int = 2_000
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    width: int = 64
    rounds: int = 2
    architecture: str = "equivariant"
    triad_mode: str = "learned"
    invalid_fraction: float = 0.25
    hard_fraction: float = 0.10
    semantic_weight: float = 0.25
    margin_weight: float = 0.05
    margin: float = 1.0
    eval_rows: int = 512
    device: str = "auto"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _valid_mask(labels: torch.Tensor) -> torch.Tensor:
    return labels.ge(0) & labels.lt(PRIMITIVE_COUNT)


def _semantic_loss(
    probabilities: torch.Tensor,
    batch: ContextualCardBatch,
) -> torch.Tensor:
    candidates = relation_primitive_candidates(
        batch.witness_left,
        batch.witness_right,
        batch.object_mask,
    )
    prediction = torch.einsum(
        "bsp,bswpij->bswij",
        probabilities[..., :PRIMITIVE_COUNT],
        candidates,
    )
    valid = _valid_mask(batch.labels)
    evidence = (
        valid[..., None, None, None]
        & batch.witness_mask[..., None, None]
        & batch.object_mask[:, None, None, :, None]
        & batch.object_mask[:, None, None, None, :]
    ).to(prediction.dtype)
    denominator = evidence.sum().clamp_min(1.0)
    return (
        (prediction - batch.witness_output).square() * evidence
    ).sum() / denominator


def _margin_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    margin: float,
) -> torch.Tensor:
    active = labels.ne(IGNORE_INDEX)
    if not bool(active.any()):
        return logits.sum() * 0.0
    selected = logits[active]
    target = labels[active]
    truth = selected.gather(1, target[:, None]).squeeze(1)
    competitors = selected.masked_fill(
        F.one_hot(target, BINDER_CLASS_COUNT).bool(),
        torch.finfo(selected.dtype).min,
    ).amax(1)
    return F.relu(float(margin) - truth + competitors).mean()


def compute_losses(
    model: ContextualWitnessEquivariantBinder,
    batch: ContextualCardBatch,
    *,
    hard: bool,
    semantic_weight: float,
    margin_weight: float,
    margin: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    binding = model(
        batch.witness_left,
        batch.witness_right,
        batch.witness_output,
        batch.witness_mask,
        batch.argument_mask,
        batch.object_mask,
        hard=hard,
    )
    active = batch.labels.ne(IGNORE_INDEX)
    classification = F.cross_entropy(
        binding.logits[active],
        batch.labels[active],
    )
    semantic = _semantic_loss(binding.probabilities, batch)
    margin_loss = _margin_loss(
        binding.logits,
        batch.labels,
        margin=margin,
    )
    total = (
        classification
        + float(semantic_weight) * semantic
        + float(margin_weight) * margin_loss
    )
    return total, {
        "classification": classification,
        "semantic": semantic,
        "margin": margin_loss,
    }


def _accuracy_receipt(
    model: ContextualWitnessEquivariantBinder,
    batch: ContextualCardBatch,
) -> dict[str, Any]:
    with torch.no_grad():
        binding = model(
            batch.witness_left,
            batch.witness_right,
            batch.witness_output,
            batch.witness_mask,
            batch.argument_mask,
            batch.object_mask,
            hard=True,
        )
    prediction = binding.logits.argmax(-1)
    active = batch.labels.ne(IGNORE_INDEX)
    valid = _valid_mask(batch.labels)
    invalid = batch.labels.eq(REJECT_INDEX)
    per_class: dict[str, dict[str, int | float]] = {}
    for class_index in range(BINDER_CLASS_COUNT):
        selected = batch.labels.eq(class_index)
        count = int(selected.sum().item())
        correct = int((prediction.eq(class_index) & selected).sum().item())
        per_class[str(class_index)] = {
            "correct": correct,
            "count": count,
            "accuracy": correct / count if count else 0.0,
        }
    valid_count = int(valid.sum().item())
    invalid_count = int(invalid.sum().item())
    active_count = int(active.sum().item())
    return {
        "active_correct": int((prediction.eq(batch.labels) & active).sum().item()),
        "active_count": active_count,
        "active_accuracy": float(
            (prediction.eq(batch.labels) & active).sum().item()
        )
        / max(active_count, 1),
        "valid_correct": int((prediction.eq(batch.labels) & valid).sum().item()),
        "valid_count": valid_count,
        "valid_accuracy": float(
            (prediction.eq(batch.labels) & valid).sum().item()
        )
        / max(valid_count, 1),
        "reject_correct": int((prediction.eq(REJECT_INDEX) & invalid).sum().item()),
        "reject_count": invalid_count,
        "reject_accuracy": float(
            (prediction.eq(REJECT_INDEX) & invalid).sum().item()
        )
        / max(invalid_count, 1),
        "per_class": per_class,
    }


def _permute_batch(
    batch: ContextualCardBatch,
    *,
    object_permutation: torch.Tensor,
    slot_permutation: torch.Tensor,
    witness_permutation: torch.Tensor,
) -> ContextualCardBatch:
    def relation(value: torch.Tensor) -> torch.Tensor:
        return value[
            :,
            slot_permutation,
        ][
            :,
            :,
            witness_permutation,
        ][
            :,
            :,
            :,
            object_permutation,
        ][
            :,
            :,
            :,
            :,
            object_permutation,
        ]

    return ContextualCardBatch(
        witness_left=relation(batch.witness_left),
        witness_right=relation(batch.witness_right),
        witness_output=relation(batch.witness_output),
        witness_mask=batch.witness_mask[
            :,
            slot_permutation,
        ][
            :,
            :,
            witness_permutation,
        ],
        argument_mask=batch.argument_mask[
            :,
            slot_permutation,
        ][
            :,
            :,
            witness_permutation,
        ],
        object_mask=batch.object_mask[:, object_permutation],
        labels=batch.labels[:, slot_permutation],
        cardinality=batch.cardinality,
    )


def _equivariance_receipt(
    model: ContextualWitnessEquivariantBinder,
    batch: ContextualCardBatch,
    *,
    generator: torch.Generator,
) -> dict[str, int | bool]:
    device = batch.witness_left.device
    objects = torch.randperm(
        batch.object_mask.shape[1],
        generator=generator,
    ).to(device)
    slots = torch.randperm(
        batch.labels.shape[1],
        generator=generator,
    ).to(device)
    witnesses = torch.randperm(
        batch.witness_mask.shape[2],
        generator=generator,
    ).to(device)
    permuted = _permute_batch(
        batch,
        object_permutation=objects,
        slot_permutation=slots,
        witness_permutation=witnesses,
    )
    with torch.no_grad():
        original = model(
            batch.witness_left,
            batch.witness_right,
            batch.witness_output,
            batch.witness_mask,
            batch.argument_mask,
            batch.object_mask,
            hard=True,
        ).logits.argmax(-1)
        transformed = model(
            permuted.witness_left,
            permuted.witness_right,
            permuted.witness_output,
            permuted.witness_mask,
            permuted.argument_mask,
            permuted.object_mask,
            hard=True,
        ).logits.argmax(-1)
    expected = original[:, slots]
    active = permuted.labels.ne(IGNORE_INDEX)
    exact = transformed.eq(expected) | ~active
    return {
        "correct": int(exact.sum().item()),
        "count": int(exact.numel()),
        "all_exact": bool(exact.all().item()),
    }


def _make_batch(
    config: TrainConfig,
    generator: torch.Generator,
    device: torch.device,
    *,
    eval_shift: bool,
) -> ContextualCardBatch:
    cardinalities = (2, 7, 8) if eval_shift else (3, 4, 5, 6)
    densities = SHIFT_DENSITIES if eval_shift else TRAIN_DENSITIES
    rows = config.eval_rows if eval_shift else config.batch_size
    return generate_contextual_card_batch(
        batch_size=rows,
        generator=generator,
        cardinalities=cardinalities,
        densities=densities,
        invalid_fraction=config.invalid_fraction,
    ).to(device)


def train_binder(
    config: TrainConfig,
    *,
    output_dir: Path,
) -> dict[str, Any]:
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = _resolve_device(config.device)
    if config.architecture == "equivariant":
        model = ContextualWitnessEquivariantBinder(
            width=config.width,
            rounds=config.rounds,
            triad_mode=config.triad_mode,
        ).to(device)
    elif config.architecture == "statistics":
        model = ContextualWitnessStatisticsBinder(
            width=config.width,
        ).to(device)
    else:
        raise ValueError("binder architecture differs")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    train_generator = torch.Generator().manual_seed(config.seed + 1)
    trace: list[dict[str, float | int | bool]] = []
    hard_start = max(
        0,
        config.steps - round(config.steps * config.hard_fraction),
    )
    model.train()
    for step in range(config.steps):
        batch = _make_batch(
            config,
            train_generator,
            device,
            eval_shift=False,
        )
        hard = step >= hard_start
        loss, parts = compute_losses(
            model,
            batch,
            hard=hard,
            semantic_weight=config.semantic_weight,
            margin_weight=config.margin_weight,
            margin=config.margin,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if (
            step == 0
            or step + 1 == config.steps
            or (step + 1) % max(config.steps // 20, 1) == 0
        ):
            trace.append(
                {
                    "step": step + 1,
                    "hard": hard,
                    "total": float(loss.detach().cpu()),
                    **{
                        name: float(value.detach().cpu())
                        for name, value in parts.items()
                    },
                }
            )

    model.eval()
    in_generator = torch.Generator().manual_seed(config.seed + 10_000)
    shift_generator = torch.Generator().manual_seed(config.seed + 20_000)
    in_distribution = generate_contextual_card_batch(
        batch_size=config.eval_rows,
        generator=in_generator,
        cardinalities=(3, 4, 5, 6),
        densities=TRAIN_DENSITIES,
        invalid_fraction=config.invalid_fraction,
    ).to(device)
    shifted = generate_contextual_card_batch(
        batch_size=config.eval_rows,
        generator=shift_generator,
        cardinalities=(2, 7, 8),
        densities=SHIFT_DENSITIES,
        invalid_fraction=config.invalid_fraction,
    ).to(device)
    evaluation = {
        "in_distribution": _accuracy_receipt(model, in_distribution),
        "shifted_cardinality_density": _accuracy_receipt(model, shifted),
        "in_distribution_equivariance": _equivariance_receipt(
            model,
            in_distribution,
            generator=torch.Generator().manual_seed(config.seed + 30_000),
        ),
        "shifted_equivariance": _equivariance_receipt(
            model,
            shifted,
            generator=torch.Generator().manual_seed(config.seed + 40_000),
        ),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "binder.pt"
    report_path = output_dir / "report.json"
    source_paths = {
        "binder": Path(__file__).with_name(
            "contextual_witness_equivariant_binder.py"
        ),
        "data": Path(__file__).with_name("contextual_witness_card_data.py"),
        "trainer": Path(__file__),
    }
    checkpoint = {
        "protocol": "contextual_witness_equivariant_binder_v1",
        "config": asdict(config),
        "parameter_receipt": model.parameter_receipt(),
        "model_state": {
            name: value.detach().cpu()
            for name, value in model.state_dict().items()
        },
        "source_sha256": {
            name: _sha256(path)
            for name, path in source_paths.items()
        },
        "evaluation": evaluation,
    }
    torch.save(checkpoint, checkpoint_path)
    report = {
        "protocol": checkpoint["protocol"],
        "claim_boundary": (
            "This evaluates learned contextual operation-card binding under "
            "object/card/witness permutations and held-out cardinality/density. "
            "It is not by itself evidence of general reasoning."
        ),
        "config": asdict(config),
        "device": str(device),
        "parameter_receipt": checkpoint["parameter_receipt"],
        "trace": trace,
        "evaluation": evaluation,
        "source_sha256": checkpoint["source_sha256"],
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": _sha256(checkpoint_path),
        },
    }
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--steps", type=int, default=2_000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument(
        "--architecture",
        choices=("equivariant", "statistics"),
        default="equivariant",
    )
    parser.add_argument(
        "--triad-mode",
        choices=("learned", "false", "zero"),
        default="learned",
    )
    parser.add_argument("--invalid-fraction", type=float, default=0.25)
    parser.add_argument("--hard-fraction", type=float, default=0.10)
    parser.add_argument("--semantic-weight", type=float, default=0.25)
    parser.add_argument("--margin-weight", type=float, default=0.05)
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--eval-rows", type=int, default=512)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = TrainConfig(
        seed=args.seed,
        steps=args.steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        width=args.width,
        rounds=args.rounds,
        architecture=args.architecture,
        triad_mode=args.triad_mode,
        invalid_fraction=args.invalid_fraction,
        hard_fraction=args.hard_fraction,
        semantic_weight=args.semantic_weight,
        margin_weight=args.margin_weight,
        margin=args.margin,
        eval_rows=args.eval_rows,
        device=args.device,
    )
    report = train_binder(config, output_dir=args.output_dir)
    print(json.dumps(report["evaluation"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
