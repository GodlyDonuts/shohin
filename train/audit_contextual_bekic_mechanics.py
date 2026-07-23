#!/usr/bin/env python3
"""Audit exact source-deleted contextual Bekić execution.

This script evaluates structured mechanics only. It does not train a neural
compiler and its report must not be described as a general-reasoning result.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import torch

from contextual_bekic_graph_machine import (
    ContextualBekicGraphMachine,
    LateContextualProgramQuery,
    contextual_graph_parameter_receipt,
)
from contextual_relation_primitive_compiler import (
    ContextualRelationPrimitiveCompiler,
)
from contextualize_bekic_program import contextualize_simultaneous_packet
from contrastive_bekic_program_orbits import (
    DEVELOPMENT_CELLS,
    assert_split_disjoint,
    evaluate_simultaneous,
    generate_train_development,
    select_machine_input,
    transplant_constants,
    transplant_program,
)
from independent_bekic_oracle import (
    evaluate_nested_independently,
    evaluate_simultaneous_independently,
)
from tensorize_contextual_bekic import (
    tensorize_contextual_packets,
    tensorize_target_environment,
)


ARMS = ("p", "p_prime", "p_eq")
SOURCE_FILES = (
    "pipeline/contrastive_bekic_program_orbits.py",
    "pipeline/contextualize_bekic_program.py",
    "pipeline/independent_bekic_oracle.py",
    "train/contextual_relation_primitive_compiler.py",
    "train/contextual_bekic_graph_machine.py",
    "train/tensorize_contextual_bekic.py",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _expected(packet: dict[str, object]) -> torch.Tensor:
    environment = evaluate_simultaneous(packet)
    return tensorize_target_environment(
        environment,
        [str(item) for item in packet["program"]["variables"]],
        cardinality=int(packet["cardinality"]),
    )


def _transplants(
    development: list[dict[str, object]],
) -> list[tuple[str, str, dict[str, object]]]:
    output: list[tuple[str, str, dict[str, object]]] = []
    for cell in DEVELOPMENT_CELLS:
        donors = [row for row in development if str(row["axes"]["cell"]) == cell]
        pair = next(
            (
                (donor, recipient)
                for donor in donors
                for recipient in development
                if donor is not recipient
                and donor["axes"]["cardinality"] == recipient["axes"]["cardinality"]
            ),
            None,
        )
        if pair is None:
            raise RuntimeError(f"no legal transplant pair for {cell}")
        for label, forms in (
            ("program_transplant", transplant_program(*pair)),
            ("constant_transplant", transplant_constants(*pair)),
        ):
            output.append((cell, label, forms["simultaneous"]))
    return output


def _cell_metrics(
    labels: list[tuple[str, str, str]],
    exact: torch.Tensor,
    converged: torch.Tensor,
) -> dict[str, dict[str, int | float]]:
    counts: Counter[tuple[str, str]] = Counter()
    correct: Counter[tuple[str, str]] = Counter()
    halted: Counter[tuple[str, str]] = Counter()
    for index, (_split, cell, arm) in enumerate(labels):
        key = (cell, arm)
        counts[key] += 1
        correct[key] += int(exact[index])
        halted[key] += int(converged[index])
    return {
        f"{cell}:{arm}": {
            "total": counts[(cell, arm)],
            "exact": correct[(cell, arm)],
            "exact_rate": correct[(cell, arm)] / counts[(cell, arm)],
            "converged": halted[(cell, arm)],
            "convergence_rate": halted[(cell, arm)] / counts[(cell, arm)],
        }
        for cell, arm in sorted(counts)
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--seed", type=int, default=2026072320)
    parser.add_argument("--train-count", type=int, default=10)
    parser.add_argument("--development-count", type=int, default=15)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing existing report: {args.out}")
    if args.train_count <= 0 or args.development_count < len(DEVELOPMENT_CELLS):
        raise SystemExit("audit counts do not cover every required cell")
    if len(args.source_commit) != 40 or any(
        character not in "0123456789abcdef" for character in args.source_commit
    ):
        raise SystemExit("source commit must be a lowercase full git hash")

    started = time.time()
    train, development = generate_train_development(
        train_count=args.train_count,
        development_count=args.development_count,
        seed=args.seed,
    )
    assert_split_disjoint(train, development)
    source_packets: list[dict[str, object]] = []
    labels: list[tuple[str, str, str]] = []
    independent_nested_agreement: list[bool] = []
    for split, rows in (("train", train), ("development", development)):
        for row in rows:
            cell = str(row["axes"]["cell"])
            for arm in ARMS:
                simultaneous = select_machine_input(
                    row,
                    arm=arm,
                    form="simultaneous",
                )
                nested = select_machine_input(
                    row,
                    arm=arm,
                    form="nested",
                )
                source_packets.append(simultaneous)
                independent_nested_agreement.append(
                    evaluate_nested_independently(nested)
                    == evaluate_simultaneous(simultaneous)
                )
                labels.append((split, cell, arm))
    for cell, label, packet in _transplants(development):
        source_packets.append(packet)
        labels.append(("development", cell, label))

    contextual = [
        contextualize_simultaneous_packet(
            packet,
            seed=args.seed + 1_000_000 + index,
        )
        for index, packet in enumerate(source_packets)
    ]
    device = torch.device(args.device)
    tensors = tensorize_contextual_packets(contextual, device=device)
    compiler = ContextualRelationPrimitiveCompiler().to(device)
    compilation = compiler(
        tensors.witness_left,
        tensors.witness_right,
        tensors.witness_output,
        tensors.witness_mask,
        tensors.argument_mask,
        tensors.object_mask,
        hard=True,
    )
    active_slots = tensors.packet.slot_arity.ge(0)
    query = LateContextualProgramQuery(
        variable=torch.zeros(
            len(source_packets),
            dtype=torch.long,
            device=device,
        ),
        position=torch.zeros(
            len(source_packets),
            dtype=torch.long,
            device=device,
        ),
    )
    machine = ContextualBekicGraphMachine().to(device)
    rollout = machine(
        tensors.packet,
        compilation.discrete_assignment,
        query,
    )
    expected = torch.stack([_expected(packet) for packet in source_packets]).to(device)
    independent_simultaneous_agreement = [
        evaluate_simultaneous_independently(packet)
        == evaluate_simultaneous(packet)
        for packet in source_packets
    ]
    exact = rollout.terminal_variables.eq(expected).flatten(1).all(-1)
    discrete_exact = torch.equal(
        compilation.discrete_assignment,
        compilation.discrete_assignment.round(),
    )
    all_cells = {str(row["axes"]["cell"]) for row in development}
    required_cells = set(DEVELOPMENT_CELLS)
    admitted = bool(
        exact.all()
        and rollout.converged.all()
        and compilation.identifiable[active_slots].all()
        and not compilation.identifiable[~active_slots].any()
        and discrete_exact
        and all_cells == required_cells
        and all(independent_simultaneous_agreement)
        and all(independent_nested_agreement)
    )

    root = Path(__file__).resolve().parents[1]
    report = {
        "schema": "contextual_bekic_mechanics_audit_v1",
        "decision": (
            "admit_structured_mechanics_only"
            if admitted
            else "reject_structured_mechanics"
        ),
        "claim_boundary": (
            "Exact source-deleted tensor mechanics only; no neural compiler, "
            "language grounding, or general-reasoning claim."
        ),
        "source_commit": args.source_commit,
        "source_sha256": {
            relative: _sha256(root / relative) for relative in SOURCE_FILES
        },
        "seed": args.seed,
        "train_orbits": len(train),
        "development_orbits": len(development),
        "development_cells": sorted(all_cells),
        "packets": len(source_packets),
        "active_operation_slots": int(active_slots.sum()),
        "identifiable_operation_slots": int(
            compilation.identifiable[active_slots].sum()
        ),
        "inactive_operation_slots_zero": int(
            compilation.discrete_assignment[~active_slots].count_nonzero()
        )
        == 0,
        "discrete_assignment_exact": discrete_exact,
        "exact_terminal_packets": int(exact.sum()),
        "converged_packets": int(rollout.converged.sum()),
        "independent_simultaneous_oracle_agreement": sum(
            independent_simultaneous_agreement
        ),
        "independent_simultaneous_oracle_packets": len(
            independent_simultaneous_agreement
        ),
        "independent_nested_oracle_agreement": sum(
            independent_nested_agreement
        ),
        "independent_nested_oracle_packets": len(
            independent_nested_agreement
        ),
        "metrics": _cell_metrics(labels, exact, rollout.converged),
        "train_receipt_sha256": _json_sha256([row["receipts"] for row in train]),
        "development_receipt_sha256": _json_sha256(
            [row["receipts"] for row in development]
        ),
        "compiler_parameters": compiler.parameter_receipt(),
        "executor_parameters": contextual_graph_parameter_receipt(machine),
        "elapsed_seconds": time.time() - started,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))
    if not admitted:
        raise SystemExit("contextual Bekić mechanics audit rejected")


if __name__ == "__main__":
    main()
