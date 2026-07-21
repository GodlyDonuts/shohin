#!/usr/bin/env python3
"""Train-only falsifier for the dual-stream ER-TT repair.

This canary deliberately reads only the already-public training split. It fits on
10,000 families and probes 2,000 disjoint training families before any fresh
board, development split, or confirmation split is created or consumed.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import replace
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import re
import subprocess
import sys
from typing import Mapping, Sequence

import torch

from build_er_relation_tensor_board import TRAIN_SPLIT
from er_cst_fresh import (
    _cosine_scale,
    byte_batch,
    canonical_json,
    derived_seed,
    trainable_state,
)
from er_relation_tensor_training import (
    RelationTensorRow,
    evaluate_arm,
    group_families,
    load_board_receipt,
    load_split,
    loss_batch,
)
from pilot_er_dual_stream_relation_adapter import (
    EXPECTED_PARAMETERS,
    initialize_dual_stream_relation,
)
from pilot_er_relation_tensor import FROZEN_SOURCE_PATHS as ER_TT_FROZEN_SOURCE_PATHS
from pilot_sd_cst_byte_addressed import sha256_file
from pilot_sd_cst_renderer_native_program import frozen_state_digest


SCHEMA = "r12_er_dual_stream_train_only_canary_v1"
BOARD_REPORT_SHA256 = (
    "64ea4c0e19ea029102af240d44242c830d7b014e49a59af09a836b2d3efb6010"
)
OPAQUE_PATTERN = re.compile(rb"(?<!\S)[0-9a-z]{6}(?!\S)")
CONTRACT = {
    "fit_families": 10_000,
    "probe_families": 2_000,
    "views_per_family": 4,
    "fit_rows": 40_000,
    "probe_rows": 8_000,
    "epochs": 2,
    "family_batch_size": 8,
    "rows_per_update": 32,
    "updates": 2_500,
    "lr": 2e-4,
    "warmup": 100,
    "weight_decay": 0.01,
    "betas": [0.9, 0.95],
    "gradient_clip": 1.0,
    "outcome_supervision": False,
    "probe_is_family_disjoint": True,
    "development_reads": 0,
    "confirmation_reads": 0,
}
THRESHOLDS = {
    "packet": 0.85,
    "state": 0.85,
    "answer": 0.85,
    "joint": 0.85,
    "relation_rows": 0.90,
    "witness_pointer": 0.90,
    "events": 0.95,
    "halt": 0.95,
    "minimum_cardinality_joint": 0.75,
    "alpha_exact": 1.0,
}
FROZEN_SOURCE_PATHS = tuple(
    sorted(
        set(
            ER_TT_FROZEN_SOURCE_PATHS
            + (
                "R12_ER_DUAL_STREAM_RELATION_REPAIR_PREREG.md",
                "train/er_dual_stream_relation_adapter.py",
                "train/pilot_er_dual_stream_relation_adapter.py",
                "train/pilot_er_dual_stream_train_canary.py",
                "train/test_er_dual_stream_relation_adapter.py",
                "train/test_pilot_er_dual_stream_relation_adapter.py",
                "train/test_pilot_er_dual_stream_train_canary.py",
                "train/jobs/er_dual_stream_train_canary.sbatch",
            )
        )
    )
)


def runtime_manifest() -> dict[str, object]:
    value: dict[str, object] = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": str(torch.__version__),
        "torch_cuda": str(torch.version.cuda),
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        value.update(
            {
                "cuda_device": torch.cuda.get_device_name(),
                "cuda_capability": list(torch.cuda.get_device_capability()),
                "bf16_supported": torch.cuda.is_bf16_supported(),
            }
        )
    value["sha256"] = hashlib.sha256(canonical_json(value).encode()).hexdigest()
    return value


def source_manifest(repo_root: Path, expected_commit: str) -> dict[str, object]:
    def git(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ("git", *args),
            cwd=repo_root,
            check=False,
            capture_output=True,
            text=True,
        )

    resolved = git("rev-parse", "--verify", f"{expected_commit}^{{commit}}")
    if resolved.returncode or resolved.stdout.strip() != expected_commit:
        raise RuntimeError("dual-stream scientific source commit is unavailable")
    if git("merge-base", "--is-ancestor", expected_commit, "HEAD").returncode:
        raise RuntimeError("dual-stream source commit is not an ancestor")
    hashes = {}
    for relative in FROZEN_SOURCE_PATHS:
        if git("cat-file", "-e", f"{expected_commit}:{relative}").returncode:
            raise RuntimeError(f"dual-stream source omits frozen path: {relative}")
        if git("diff", "--quiet", expected_commit, "--", relative).returncode:
            raise RuntimeError(f"dual-stream runtime differs from source: {relative}")
        hashes[relative] = sha256_file(repo_root / relative)
    value = {"commit": expected_commit, "files": hashes}
    value["sha256"] = hashlib.sha256(canonical_json(value).encode()).hexdigest()
    return value


def split_train_families(
    rows: Sequence[RelationTensorRow],
    seed: int,
    *,
    fit_families: int = 10_000,
    probe_families: int = 2_000,
) -> tuple[list[RelationTensorRow], list[RelationTensorRow], dict[str, object]]:
    groups = group_families(rows)
    if len(groups) != fit_families + probe_families:
        raise ValueError("dual-stream train-only family count differs")

    def key(group: Sequence[RelationTensorRow]) -> tuple[str, str]:
        family = group[0].family_id
        digest = hashlib.sha256(f"{seed}:split:{family}".encode()).hexdigest()
        return digest, family

    ordered = sorted(groups, key=key)
    fit_groups = ordered[:fit_families]
    probe_groups = ordered[fit_families:]
    fit_ids = {group[0].family_id for group in fit_groups}
    probe_ids = {group[0].family_id for group in probe_groups}
    if fit_ids & probe_ids or len(fit_ids) != fit_families or len(probe_ids) != probe_families:
        raise ValueError("dual-stream family split is not disjoint")
    fit_rows = [row for group in fit_groups for row in group]
    probe_rows = [row for group in probe_groups for row in group]
    receipt = {
        "fit_families": len(fit_ids),
        "probe_families": len(probe_ids),
        "fit_rows": len(fit_rows),
        "probe_rows": len(probe_rows),
        "family_overlap": len(fit_ids & probe_ids),
        "fit_family_sha256": hashlib.sha256(
            "\n".join(sorted(fit_ids)).encode()
        ).hexdigest(),
        "probe_family_sha256": hashlib.sha256(
            "\n".join(sorted(probe_ids)).encode()
        ).hexdigest(),
    }
    return fit_rows, probe_rows, receipt


def score_train_row(row: RelationTensorRow) -> RelationTensorRow:
    """Derive probe outcomes mechanically without exposing them to fitting."""
    state = tuple(row.initial_order)
    alive = True
    for card, halt in zip(row.event_cards, row.event_halt, strict=True):
        if not alive:
            continue
        if halt:
            alive = False
            continue
        relation = row.relation_rows[card]
        state = tuple(state[index] for index in relation)
    return replace(
        row,
        final_state=state,
        answer_role=state[row.query_position],
    )


def alpha_recode_row(row: RelationTensorRow, salt: str) -> RelationTensorRow:
    """Move every six-byte symbol into one neutral namespace, bijectively."""
    payload = bytes(row.program_bytes)
    tokens = sorted(set(OPAQUE_PATTERN.findall(payload)))
    mapping: dict[bytes, bytes] = {}
    used: set[bytes] = set()
    for token in tokens:
        retry = 0
        while True:
            digest = hashlib.sha256(
                b":".join(
                    (
                        salt.encode(),
                        row.family_id.encode(),
                        token,
                        str(retry).encode(),
                    )
                )
            ).hexdigest()
            candidate = ("z" + digest[:5]).encode()
            if candidate not in used:
                break
            retry += 1
        mapping[token] = candidate
        used.add(candidate)
    recoded = OPAQUE_PATTERN.sub(lambda match: mapping[match.group(0)], payload)
    if len(recoded) != len(payload) or len(mapping) != len(used):
        raise ValueError("dual-stream alpha recode changes width or is not bijective")
    found = set(OPAQUE_PATTERN.findall(recoded))
    if found != used or any(not token.startswith(b"z") for token in found):
        raise ValueError("dual-stream alpha recode is not one neutral namespace")
    return replace(row, program_bytes=tuple(recoded))


def fit_train_only(
    model: torch.nn.Module,
    rows: Sequence[RelationTensorRow],
    *,
    seed: int,
    frozen_digest: str,
    trainable_names: frozenset[str],
) -> dict[str, object]:
    groups = group_families(rows)
    if len(groups) != int(CONTRACT["fit_families"]):
        raise ValueError("dual-stream fit-family count differs")
    model.train()
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(CONTRACT["lr"]),
        betas=tuple(CONTRACT["betas"]),
        weight_decay=float(CONTRACT["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: _cosine_scale(
            step, int(CONTRACT["updates"]), int(CONTRACT["warmup"])
        ),
    )
    rng = random.Random(seed)
    history = []
    update = 0
    for epoch in range(int(CONTRACT["epochs"])):
        order = list(range(len(groups)))
        rng.shuffle(order)
        totals: dict[str, float] = {}
        seen = 0
        for start in range(0, len(order), int(CONTRACT["family_batch_size"])):
            batch = [
                groups[index]
                for index in order[
                    start : start + int(CONTRACT["family_batch_size"])
                ]
            ]
            optimizer.zero_grad(set_to_none=True)
            autocast = (
                torch.autocast("cuda", dtype=torch.bfloat16)
                if next(model.parameters()).is_cuda
                else nullcontext()
            )
            with autocast:
                loss, pieces = loss_batch(
                    model, batch, next(model.parameters()).device
                )
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(
                trainable, float(CONTRACT["gradient_clip"])
            )
            if not bool(torch.isfinite(norm)):
                raise RuntimeError("dual-stream gradient is non-finite")
            optimizer.step()
            scheduler.step()
            update += 1
            row_count = sum(map(len, batch))
            seen += row_count
            for name, value in pieces.items():
                totals[name] = totals.get(name, 0.0) + value * row_count
        history.append(
            {
                "epoch": epoch + 1,
                "updates": update,
                "losses": {
                    name: value / seen for name, value in sorted(totals.items())
                },
            }
        )
    if update != int(CONTRACT["updates"]):
        raise RuntimeError("dual-stream update count differs")
    frozen_after = frozen_state_digest(model, trainable_names)
    if frozen_after != frozen_digest:
        raise RuntimeError("dual-stream excluded parent changed")
    return {
        "seed": seed,
        "updates": update,
        "history": history,
        "frozen_parent_unchanged": True,
        "frozen_digest": frozen_after,
    }


@torch.no_grad()
def alpha_predictions(
    model: torch.nn.Module,
    rows: Sequence[RelationTensorRow],
    *,
    batch_size: int,
) -> dict[str, torch.Tensor]:
    model.eval()
    device = next(model.parameters()).device
    values: dict[str, list[torch.Tensor]] = {}
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        program_ids, program_valid = byte_batch(batch, "program_bytes", device)
        query_ids, query_valid = byte_batch(batch, "query_bytes", device)
        output = model.compile_relation_program(
            program_ids, program_valid, query_ids, query_valid
        )
        hard = output.program.hard()
        masked_query = output.query.logits.masked_fill(
            ~hard.active, torch.finfo(output.query.logits.dtype).min
        )
        packet = {
            "cardinality": hard.cardinality,
            "initial": hard.initial_state.argmax(-1),
            "relations": hard.rule_cards.argmax(-1),
            "rule_active": hard.rule_active.long(),
            "events": hard.event_card,
            "halt": hard.event_halt.long(),
            "query": masked_query.argmax(-1),
            "line_pointer": output.line_pointer_logits.argmax(-1),
            "binding_pointer": output.binding_pointer_logits.argmax(-1),
            "initial_pointer": output.initial_entity_pointer_logits.argmax(-1),
            "witness_pointer": output.witness_pointer_logits.argmax(-1),
            "query_pointer": output.query.pointer_logits.argmax(-1),
        }
        for name, tensor in packet.items():
            values.setdefault(name, []).append(tensor.detach().cpu().to(torch.int16))
    return {name: torch.cat(parts) for name, parts in values.items()}


def alpha_metrics(
    canonical: Mapping[str, torch.Tensor],
    recoded: Mapping[str, torch.Tensor],
) -> dict[str, object]:
    if set(canonical) != set(recoded):
        raise ValueError("dual-stream alpha evidence fields differ")
    result: dict[str, object] = {}
    row_exact: list[torch.Tensor] = []
    for name in sorted(canonical):
        if canonical[name].shape != recoded[name].shape:
            raise ValueError(f"dual-stream alpha field shape differs: {name}")
        exact = canonical[name].eq(recoded[name]).reshape(canonical[name].shape[0], -1).all(-1)
        row_exact.append(exact)
        result[name] = {
            "exact": int(exact.sum()),
            "rows": int(exact.numel()),
            "rate": float(exact.float().mean()),
        }
    complete = torch.stack(row_exact).all(0)
    result["complete"] = {
        "exact": int(complete.sum()),
        "rows": int(complete.numel()),
        "rate": float(complete.float().mean()),
    }
    result["complete_mask"] = complete
    return result


def compute_gates(
    metrics: Mapping[str, object],
    alpha: Mapping[str, object],
    parameters: Mapping[str, int],
    fit: Mapping[str, object],
    split: Mapping[str, object],
) -> dict[str, bool]:
    overall = metrics["overall"]
    cardinality = metrics["by_cardinality"]
    return {
        "family_split_exact_and_disjoint": split
        == {
            **split,
            "fit_families": int(CONTRACT["fit_families"]),
            "probe_families": int(CONTRACT["probe_families"]),
            "fit_rows": int(CONTRACT["fit_rows"]),
            "probe_rows": int(CONTRACT["probe_rows"]),
            "family_overlap": 0,
        },
        "packet_state_answer_joint_at_least_85pct": all(
            float(overall[name]["rate"]) >= float(THRESHOLDS[name])
            for name in ("packet", "state", "answer", "joint")
        ),
        "relation_rows_at_least_90pct": float(overall["relation_rows"]["rate"])
        >= float(THRESHOLDS["relation_rows"]),
        "witness_pointers_at_least_90pct": float(
            overall["witness_pointer"]["rate"]
        )
        >= float(THRESHOLDS["witness_pointer"]),
        "events_and_halt_at_least_95pct": all(
            float(overall[name]["rate"]) >= float(THRESHOLDS[name])
            for name in ("events", "halt")
        ),
        "minimum_cardinality_joint_at_least_75pct": min(
            float(value["joint"]["rate"]) for value in cardinality.values()
        )
        >= float(THRESHOLDS["minimum_cardinality_joint"]),
        "all_hard_fields_and_pointers_alpha_exact": float(
            alpha["complete"]["rate"]
        )
        == float(THRESHOLDS["alpha_exact"]),
        "confirmed_parent_unchanged": fit["frozen_parent_unchanged"] is True,
        "parameter_certificate_exact_and_below_200m": dict(parameters)
        == EXPECTED_PARAMETERS,
        "train_only_and_zero_scored_split_reads": CONTRACT["outcome_supervision"]
        is False
        and int(CONTRACT["development_reads"]) == 0
        and int(CONTRACT["confirmation_reads"]) == 0,
    }


def atomic_torch_save(value: object, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    with temporary.open("rb+") as handle:
        os.fsync(handle.fileno())
    temporary.replace(path)
    path.chmod(0o444)


def atomic_json_save(value: object, path: Path) -> None:
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    path.chmod(0o444)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--joint-checkpoint", type=Path, required=True)
    parser.add_argument("--physical-checkpoint", type=Path, required=True)
    parser.add_argument("--v1-checkpoint", type=Path, required=True)
    parser.add_argument("--v1-2-checkpoint", type=Path, required=True)
    parser.add_argument("--confirmed-checkpoint", type=Path, required=True)
    parser.add_argument("--confirmation-assessment", type=Path, required=True)
    parser.add_argument("--witness-checkpoint", type=Path, required=True)
    parser.add_argument("--witness-confirmation-assessment", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    if args.out_dir.exists():
        raise SystemExit(f"refusing existing dual-stream output: {args.out_dir}")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise SystemExit("dual-stream canary requires bf16 CUDA")
    source = source_manifest(args.repo_root.resolve(), args.source_commit)
    board = load_board_receipt(args.data_dir)
    if board.get("report_sha256") != BOARD_REPORT_SHA256:
        raise SystemExit("dual-stream board identity differs")
    train_rows = load_split(
        args.data_dir,
        board,
        filename="train.jsonl",
        split=TRAIN_SPLIT,
        expected=48_000,
    )
    fit_rows, probe_rows, split_receipt = split_train_families(
        train_rows, derived_seed(args.seed, "dual-stream-train-probe-split")
    )
    if (
        len(fit_rows) != int(CONTRACT["fit_rows"])
        or len(probe_rows) != int(CONTRACT["probe_rows"])
    ):
        raise RuntimeError("dual-stream row split differs")

    args.out_dir.mkdir(parents=True)
    device = torch.device("cuda")
    model, parameters, frozen_digest, parent_receipt = initialize_dual_stream_relation(
        joint_checkpoint=args.joint_checkpoint,
        physical_checkpoint=args.physical_checkpoint,
        v1_checkpoint=args.v1_checkpoint,
        v1_2_checkpoint=args.v1_2_checkpoint,
        confirmed_checkpoint=args.confirmed_checkpoint,
        confirmation_assessment=args.confirmation_assessment,
        witness_checkpoint=args.witness_checkpoint,
        witness_confirmation_assessment=args.witness_confirmation_assessment,
        seed=args.seed,
        device=device,
    )
    trainable_names = frozenset(parent_receipt["trainable_names"])
    fit = fit_train_only(
        model,
        fit_rows,
        seed=derived_seed(args.seed, "dual-stream-fit-order"),
        frozen_digest=frozen_digest,
        trainable_names=trainable_names,
    )
    checkpoint = {
        "schema": SCHEMA,
        "source_manifest": source,
        "seed": args.seed,
        "contract": CONTRACT,
        "parameters": parameters,
        "parent_receipt": parent_receipt,
        "split": split_receipt,
        "fit": fit,
        "compiler_trainable_state": trainable_state(model),
        "development_accesses": 0,
        "confirmation_accesses": 0,
    }
    checkpoint_path = args.out_dir / "compiler.pt"
    atomic_torch_save(checkpoint, checkpoint_path)

    scored_probe = [score_train_row(row) for row in probe_rows]
    metrics = evaluate_arm(
        model,
        scored_probe,
        batch_size=args.batch_size,
        include_raw=False,
        include_invariances=False,
    )
    recoded_probe = [
        alpha_recode_row(row, "dual-stream-neutral-alpha") for row in scored_probe
    ]
    canonical_predictions = alpha_predictions(
        model, scored_probe, batch_size=args.batch_size
    )
    recoded_predictions = alpha_predictions(
        model, recoded_probe, batch_size=args.batch_size
    )
    alpha = alpha_metrics(canonical_predictions, recoded_predictions)
    complete_mask = alpha.pop("complete_mask")
    evidence = {
        "schema": SCHEMA,
        "source_commit": args.source_commit,
        "seed": args.seed,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "canonical_predictions": canonical_predictions,
        "recoded_predictions": recoded_predictions,
        "alpha_complete_mask": complete_mask,
        "development_accesses": 0,
        "confirmation_accesses": 0,
    }
    evidence_path = args.out_dir / "train_probe_evidence.pt"
    atomic_torch_save(evidence, evidence_path)
    gates = compute_gates(metrics, alpha, parameters, fit, split_receipt)
    report = {
        "schema": SCHEMA,
        "source_commit": args.source_commit,
        "source_manifest": source,
        "runtime": runtime_manifest(),
        "seed": args.seed,
        "contract": CONTRACT,
        "thresholds": THRESHOLDS,
        "parameters": parameters,
        "parent_receipt": parent_receipt,
        "split": split_receipt,
        "fit": fit,
        "metrics": metrics,
        "alpha_invariance": alpha,
        "gates": gates,
        "all_gates_pass": all(gates.values()),
        "decision": (
            "authorize_fresh_dual_stream_board"
            if all(gates.values())
            else "reject_dual_stream_before_fresh_board"
        ),
        "artifacts": {
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "evidence_sha256": sha256_file(evidence_path),
        },
        "custody": {
            "train_only_probe_accesses": 1,
            "development_accesses": 0,
            "confirmation_accesses": 0,
        },
        "claim_boundary": (
            "Passing admits only a fresh-board test of dual-stream routing and exact "
            "identity transport. It is not a development, confirmation, or reasoning claim."
        ),
    }
    report_path = args.out_dir / "train_probe_report.json"
    atomic_json_save(report, report_path)
    print(
        json.dumps(
            {
                "decision": report["decision"],
                "checkpoint_sha256": sha256_file(checkpoint_path),
                "evidence_sha256": sha256_file(evidence_path),
                "report_sha256": sha256_file(report_path),
            },
            sort_keys=True,
        )
    )
    model.cpu()
    gc.collect()


if __name__ == "__main__":
    main()
