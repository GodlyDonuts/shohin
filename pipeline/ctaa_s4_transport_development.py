#!/usr/bin/env python3
"""Retrospective sample-efficiency development probe for CTAA S4 transport.

This toy isolates transition-law learning from language, Shohin features, and
the CTAA executor. Each of six cues denotes one transposition in S4. The
matched-data arms observe only the six transitions from the identity state,
then must compose unseen cue words. A data-rich dense ceiling observes all
24 * 6 one-step transitions.

The result is retrospective development evidence for a hardcoded structured
parameter-tying prior only. It is not a neural Shohin result, an independently
preregistered gate, or evidence of native or general reasoning.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from ctaa_s4_particle_transport import (
    BINDINGS,
    BINDING_TO_INDEX,
    IDENTITY,
    IDENTITY_INDEX,
    PARTICLE_COUNT,
    S4_GENERATORS,
    Dense24TransportControl,
    S4TiedTransport,
    Z24CircularTransportControl,
    compose_permutations,
    one_hot_group_element,
)


SCHEMA = "r12_ctaa_s4_tied_particle_transport_development_v2"
SEEDS = (
    2_026_072_301,
    2_026_072_302,
    2_026_072_303,
    2_026_072_304,
    2_026_072_305,
)
OPTIMIZER_STEPS = 400
LEARNING_RATE = 0.1
EVALUATION_DEPTHS = (2, 3, 4)


def canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("ascii")


def _oracle_compose(
    left: tuple[int, ...],
    right: tuple[int, ...],
) -> tuple[int, ...]:
    if (
        len(left) != 4
        or len(right) != 4
        or sorted(left) != [0, 1, 2, 3]
        or sorted(right) != [0, 1, 2, 3]
    ):
        raise AssertionError("independent S4 target oracle received invalid element")
    return tuple(left[right[index]] for index in range(4))


def _target_index(cues: tuple[int, ...], start: tuple[int, ...] = IDENTITY) -> int:
    state = start
    for cue in cues:
        state = _oracle_compose(state, S4_GENERATORS[cue])
    return BINDING_TO_INDEX[state]


def _identity_only_examples() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    cues = torch.arange(len(S4_GENERATORS), dtype=torch.long)[:, None]
    initial = one_hot_group_element(IDENTITY_INDEX, batch=len(S4_GENERATORS))
    targets = torch.tensor(
        [_target_index((cue,)) for cue in range(len(S4_GENERATORS))],
        dtype=torch.long,
    )
    return initial, cues, targets


def _complete_transition_examples() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    source_indices: list[int] = []
    cue_rows: list[list[int]] = []
    targets: list[int] = []
    for source_index, source in enumerate(BINDINGS):
        for cue in range(len(S4_GENERATORS)):
            source_indices.append(source_index)
            cue_rows.append([cue])
            targets.append(_target_index((cue,), source))
    initial = F.one_hot(
        torch.tensor(source_indices, dtype=torch.long),
        PARTICLE_COUNT,
    ).float()
    return (
        initial,
        torch.tensor(cue_rows, dtype=torch.long),
        torch.tensor(targets, dtype=torch.long),
    )


def _fit(
    constructor: Callable[[], nn.Module],
    seed: int,
    examples: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> tuple[nn.Module, int]:
    torch.manual_seed(seed)
    model = constructor()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=0.0,
    )
    initial, cues, targets = examples
    row_indices = torch.arange(targets.numel())
    for _ in range(OPTIMIZER_STEPS):
        probabilities = model(initial, cues)
        loss = -torch.log(
            probabilities[row_indices, targets].clamp_min(1e-12)
        ).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        correct = int(model(initial, cues).argmax(-1).eq(targets).sum())
    return model, correct


def _evaluate_words(model: nn.Module, depth: int) -> tuple[int, int]:
    words = tuple(itertools.product(range(len(S4_GENERATORS)), repeat=depth))
    cues = torch.tensor(words, dtype=torch.long)
    initial = one_hot_group_element(IDENTITY_INDEX, batch=len(words))
    targets = torch.tensor([_target_index(word) for word in words])
    with torch.no_grad():
        correct = int(model(initial, cues).argmax(-1).eq(targets).sum())
    return correct, len(words)


def _arm_result(
    name: str,
    constructor: Callable[[], nn.Module],
    examples: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    seed: int,
) -> dict[str, object]:
    model, train_correct = _fit(constructor, seed, examples)
    depth_results = {
        str(depth): {
            "correct": correct,
            "total": total,
        }
        for depth in EVALUATION_DEPTHS
        for correct, total in [_evaluate_words(model, depth)]
    }
    return {
        "arm": name,
        "seed": seed,
        "supervised_transitions": int(examples[2].numel()),
        "train_correct": train_correct,
        "train_total": int(examples[2].numel()),
        "depths": depth_results,
    }


def build_report() -> dict[str, object]:
    torch.set_num_threads(1)
    independent_oracle_checks = 0
    for left in BINDINGS:
        for right in BINDINGS:
            if compose_permutations(left, right) != _oracle_compose(left, right):
                raise AssertionError("transport composition disagrees with target oracle")
            independent_oracle_checks += 1
    identity_examples = _identity_only_examples()
    complete_examples = _complete_transition_examples()
    arms = (
        ("s4_tied_identity_only", S4TiedTransport, identity_examples),
        (
            "z24_abelian_identity_only",
            Z24CircularTransportControl,
            identity_examples,
        ),
        ("dense24_identity_only", Dense24TransportControl, identity_examples),
        (
            "dense24_complete_transition_ceiling",
            Dense24TransportControl,
            complete_examples,
        ),
    )
    results = [
        _arm_result(name, constructor, examples, seed)
        for seed in SEEDS
        for name, constructor, examples in arms
    ]

    by_arm = {
        name: [result for result in results if result["arm"] == name]
        for name, _, _ in arms
    }

    def all_train_exact(name: str) -> bool:
        return all(
            result["train_correct"] == result["train_total"]
            for result in by_arm[name]
        )

    def all_depth_exact(name: str) -> bool:
        return all(
            depth_result["correct"] == depth_result["total"]
            for result in by_arm[name]
            for depth_result in result["depths"].values()
        )

    def minimum_advantage(treatment: str, control: str) -> float:
        advantages = []
        for treatment_result, control_result in zip(
            by_arm[treatment],
            by_arm[control],
            strict=True,
        ):
            if treatment_result["seed"] != control_result["seed"]:
                raise AssertionError("S4 transport development seed pairing differs")
            for depth in EVALUATION_DEPTHS:
                treatment_depth = treatment_result["depths"][str(depth)]
                control_depth = control_result["depths"][str(depth)]
                advantages.append(
                    treatment_depth["correct"] / treatment_depth["total"]
                    - control_depth["correct"] / control_depth["total"]
                )
        return min(advantages)

    gates = {
        "all_arms_fit_their_supervision": all(
            all_train_exact(name) for name, _, _ in arms
        ),
        "s4_tied_exact_all_unseen_words": all_depth_exact(
            "s4_tied_identity_only"
        ),
        "dense_complete_ceiling_exact_all_words": all_depth_exact(
            "dense24_complete_transition_ceiling"
        ),
        "s4_beats_matched_dense_by_70pp": (
            minimum_advantage(
                "s4_tied_identity_only",
                "dense24_identity_only",
            )
            >= 0.70
        ),
        "s4_beats_matched_abelian_by_70pp": (
            minimum_advantage(
                "s4_tied_identity_only",
                "z24_abelian_identity_only",
            )
            >= 0.70
        ),
        "matched_transition_budget": (
            identity_examples[2].numel() == len(S4_GENERATORS)
        ),
        "dense_ceiling_has_complete_transition_table": (
            complete_examples[2].numel()
            == len(BINDINGS) * len(S4_GENERATORS)
        ),
        "independent_target_oracle": (
            independent_oracle_checks == len(BINDINGS) ** 2
        ),
    }
    report: dict[str, object] = {
        "schema": SCHEMA,
        "claim_boundary": (
            "retrospective_source_free_hardcoded_prior_development_only_"
            "no_shohin_no_reasoning_no_neural_authorization"
        ),
        "evidence_status": "retrospective_development_not_preregistered",
        "protocol": {
            "seeds": list(SEEDS),
            "optimizer": "AdamW",
            "optimizer_steps": OPTIMIZER_STEPS,
            "learning_rate": LEARNING_RATE,
            "weight_decay": 0.0,
            "matched_supervised_transitions": int(identity_examples[2].numel()),
            "dense_ceiling_supervised_transitions": int(
                complete_examples[2].numel()
            ),
            "evaluation_depths": list(EVALUATION_DEPTHS),
            "independent_target_oracle_checks": independent_oracle_checks,
            "evaluation_words_per_depth": {
                str(depth): len(S4_GENERATORS) ** depth
                for depth in EVALUATION_DEPTHS
            },
        },
        "results": results,
        "gates": gates,
        "decision": (
            "record_retrospective_parameter_tying_signature_only"
            if all(gates.values())
            else "reject_s4_transport_sample_efficiency"
        ),
    }
    report["payload_sha256"] = hashlib.sha256(canonical_json_bytes(report)).hexdigest()
    return report


def write_exclusive(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o444,
    )
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(descriptor)
    directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = build_report()
    write_exclusive(args.out, canonical_json_bytes(report))
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
