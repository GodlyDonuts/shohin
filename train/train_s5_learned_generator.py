#!/usr/bin/env python3
"""Fit matched learned and shuffled six-cell S3 unit generators."""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import torch
import torch.nn.functional as F

from s5_learned_generator_executor import (
    GeneratorFactoredS3Executor,
    LearnedUnitGenerator,
    module_state_hash,
    unit_generator_examples,
)
from self_delimiting_event_tape import sha256_file


def fit_arm(initial_state, targets, width, updates, lr, seed, device):
    torch.manual_seed(seed)
    model = LearnedUnitGenerator(width).to(device)
    model.load_state_dict(initial_state, strict=True)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.0,
    )
    locations, directions, _ = unit_generator_examples(device)
    losses = []
    for update in range(updates):
        optimizer.zero_grad(set_to_none=True)
        logits = model(locations, directions)
        loss = F.cross_entropy(logits, targets)
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite S5 loss at update {}".format(update))
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(grad_norm):
            raise RuntimeError("non-finite S5 gradient at update {}".format(update))
        optimizer.step()
        losses.append(float(loss.detach()))
    with torch.inference_mode():
        predictions = model(locations, directions).argmax(-1)
    return model, {
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "fit_correct": int(predictions.eq(targets).sum()),
        "fit_total": int(targets.numel()),
        "predictions": predictions.cpu().tolist(),
        "targets": targets.cpu().tolist(),
    }


def bundle(model, metadata):
    executor = GeneratorFactoredS3Executor(model.width)
    executor.generator.load_state_dict(model.state_dict(), strict=True)
    metadata = dict(metadata)
    metadata["generator_state_sha256"] = module_state_hash(model)
    metadata["executor_state_sha256"] = module_state_hash(executor)
    return {
        "metadata": metadata,
        "generator_state": {
            name: value.detach().cpu() for name, value in model.state_dict().items()
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--updates", type=int, default=500)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=2026071911)
    parser.add_argument("--base-parameters", type=int, default=133689935)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    output = Path(args.out_dir)
    treatment_path = output / "treatment.pt"
    shuffled_path = output / "shuffled.pt"
    report_path = output / "report.json"
    if any(path.exists() for path in (treatment_path, shuffled_path, report_path)):
        raise SystemExit("refusing existing S5 generator output")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    template = LearnedUnitGenerator(args.width).to(device)
    initial_state = {
        name: value.detach().clone() for name, value in template.state_dict().items()
    }
    initial_hash = module_state_hash(template)
    _, _, semantic_targets = unit_generator_examples(device)
    shuffled_targets = (semantic_targets + 1) % 6
    treatment, treatment_fit = fit_arm(
        initial_state, semantic_targets, args.width, args.updates, args.lr,
        args.seed + 1, device,
    )
    shuffled, shuffled_fit = fit_arm(
        initial_state, shuffled_targets, args.width, args.updates, args.lr,
        args.seed + 1, device,
    )
    parameters = treatment.num_params()
    total_parameters = args.base_parameters + parameters
    if total_parameters >= 150_000_000:
        raise SystemExit("S5 system exceeds strict 150M parameter cap")
    common = {
        "schema": "r12_s5_learned_unit_generator_checkpoint_v1",
        "training_contract": (
            "six balanced unit actions only; no amount-two, recurrent-program, "
            "language-source, development, or confirmation supervision"
        ),
        "width": args.width,
        "updates": args.updates,
        "learning_rate": args.lr,
        "seed": args.seed,
        "generator_parameters": parameters,
        "base_parameters": args.base_parameters,
        "total_parameters": total_parameters,
        "initial_state_sha256": initial_hash,
        "amount_two_training_examples": 0,
        "recurrent_training_examples": 0,
        "development_access": 0,
        "confirmation_access": 0,
        "trainer_sha256": sha256_file(__file__),
    }
    output.mkdir(parents=True, exist_ok=True)
    torch.save(bundle(treatment, {
        **common,
        "arm": "semantic_unit_generators",
        "fit": treatment_fit,
    }), treatment_path)
    torch.save(bundle(shuffled, {
        **common,
        "arm": "fixed_deranged_unit_generators",
        "fit": shuffled_fit,
    }), shuffled_path)
    report = {
        "schema": "r12_s5_learned_generator_fit_report_v1",
        "all_fit_gates_pass": (
            treatment_fit["fit_correct"] == 6 and shuffled_fit["fit_correct"] == 6
        ),
        "treatment": treatment_fit,
        "shuffled": shuffled_fit,
        "metadata": common,
        "artifacts": {
            "treatment": {
                "path": os.path.realpath(treatment_path),
                "sha256": sha256_file(treatment_path),
            },
            "shuffled": {
                "path": os.path.realpath(shuffled_path),
                "sha256": sha256_file(shuffled_path),
            },
        },
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))
    if not report["all_fit_gates_pass"]:
        raise SystemExit("S5 generator fit gates failed")


if __name__ == "__main__":
    main()
