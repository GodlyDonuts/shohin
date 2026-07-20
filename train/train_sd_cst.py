#!/usr/bin/env python3
"""Train and development-score source-deleted categorical state transport.

The scientific contract is deliberately narrow:

* the Shohin transformer is frozen;
* training outcomes are only compiler/query fields and complete atomic
  motor/reader certificates;
* development scoring seals compiler outputs as CPU uint8 categories before
  calling ``rollout_hard``;
* confirmation data has no code path in this program.
"""

from __future__ import annotations

import argparse
import base64
from contextlib import nullcontext
from dataclasses import dataclass
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import time
from typing import Mapping, Sequence

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from model import GPT, GPTConfig
from sd_cst import (
    AMOUNT_COUNT,
    EVENT_KIND_COUNT,
    EVENT_STEPS,
    HardLateQuery,
    HardProgramTape,
    IDENTITY_COUNT,
    MAX_SYSTEM_PARAMETERS,
    QUERY_COUNT,
    SDCSTSystem,
    STATE_COUNT,
    STOP_KIND,
    StateSwap,
    atomic_motor_loss,
    compiler_field_losses,
    late_query_loss,
    reader_loss,
)


CHECKPOINT_SCHEMA = "r12_sd_cst_checkpoint_v1"
EVALUATION_SCHEMA = "r12_sd_cst_development_eval_v1"
GATE_CONFIG_SCHEMA = "r12_sd_cst_development_gate_config_v1"
PROTOCOL = "r12_sd_cst_v1"
BOARD_SCHEMA = "r12_sd_cst_board_report_v1"
TRAIN_SPLIT = "sd_cst_train"
DEVELOPMENT_SPLIT = "sd_cst_development"
PERMUTATIONS = tuple(itertools.permutations(range(3)))
PERMUTATION_TO_STATE = {value: index for index, value in enumerate(PERMUTATIONS)}
FROZEN_SOURCE_PATHS = (
    "R12_SD_CST_PREREG.md",
    "pipeline/assess_sd_cst.py",
    "pipeline/audit_sd_cst_board.py",
    "pipeline/build_sd_cst_board.py",
    "train/model.py",
    "train/sd_cst.py",
    "train/train_sd_cst.py",
    "train/jobs/sd_cst.sbatch",
)
FROZEN_BASE_SHA256 = "211d6b2cddf0c2cf8b12cb0b2d73f9c4440d85f6f531018080c8afd35b2f66a6"
FROZEN_TOKENIZER_SHA256 = "87532df5c121753de3b29194e1f9e3de47986d3f5359548fdf93606773a233d4"
FROZEN_TRAIN_ROWS = 48_000
FROZEN_PARAMETERS = 134_306_714
FROZEN_TRAINING = {
    "batch_size": 64,
    "epochs": 4,
    "compiler_lr": 1e-3,
    "motor_lr": 0.025,
    "reader_lr": 0.04,
    "motor_updates": 2000,
    "reader_updates": 1200,
    "warmup": 100,
    "clip": 1.0,
}
FROZEN_THRESHOLDS = {
    "compiler_initial_overall": 0.98,
    "compiler_initial_min_variant": 0.90,
    "compiler_initial_min_depth": 0.90,
    "compiler_event_kind_overall": 0.98,
    "compiler_event_kind_min_variant": 0.90,
    "compiler_event_kind_min_depth": 0.90,
    "compiler_event_identity_overall": 0.98,
    "compiler_event_identity_min_variant": 0.90,
    "compiler_event_identity_min_depth": 0.90,
    "compiler_event_amount_overall": 0.98,
    "compiler_event_amount_min_variant": 0.90,
    "compiler_event_amount_min_depth": 0.90,
    "compiler_exact_tape_overall": 0.95,
    "compiler_exact_tape_min_variant": 0.90,
    "compiler_exact_tape_min_depth": 0.90,
    "autonomous_graph_overall": 0.95,
    "autonomous_state_overall": 0.90,
    "autonomous_answer_overall": 0.90,
    "autonomous_graph_depth6": 0.90,
    "autonomous_state_depth6": 0.85,
    "autonomous_answer_depth6": 0.85,
    "exact_tape_conditional_execution": 1.0,
    "query_swap_state_invariance": 1.0,
    "query_swap_answer_follow_query_conditional": 1.0,
    "state_swap_separating_effect": 1.0,
    "post_stop_suffix_invariance": 1.0,
    "force_alive_suffix_oracle": 1.0,
    "variant_graph_min": 0.90,
    "variant_state_min": 0.90,
    "variant_answer_min": 0.90,
    "source_poison_bit_identity": 1.0,
}
FROZEN_CONTROLS = {
    "uniform_state": {"direction": "at_most", "threshold": 0.25},
    "uniform_answer": {"direction": "at_most", "threshold": 0.45},
    "source_free_state": {"direction": "at_most", "threshold": 0.25},
    "source_free_answer": {"direction": "at_most", "threshold": 0.45},
    "shuffled_state": {"direction": "at_most", "threshold": 0.25},
    "shuffled_answer": {"direction": "at_most", "threshold": 0.45},
    "reset_state": {"direction": "at_most", "threshold": 0.75},
    "reset_answer": {"direction": "at_most", "threshold": 0.75},
    "freeze_state": {"direction": "at_most", "threshold": 0.75},
    "freeze_answer": {"direction": "at_most", "threshold": 0.75},
}


@dataclass(frozen=True)
class EncodedRow:
    row_id: str
    split: str
    variant: str
    family_id: str | None
    program_text: str
    late_query_text: str
    program_ids: tuple[int, ...]
    query_ids: tuple[int, ...]
    initial_state: int
    event_kind: tuple[int, ...]
    event_identity: tuple[int, ...]
    amount: tuple[int, ...]
    query_position: int
    halt_after: int
    final_state: int | None = None
    answer_role: int | None = None
    full_suffix_state: int | None = None


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def runtime_manifest() -> dict[str, object]:
    cuda = torch.cuda.is_available()
    manifest: dict[str, object] = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cuda_available": cuda,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    }
    if cuda:
        manifest.update(
            {
                "cuda_device": torch.cuda.get_device_name(),
                "cuda_capability": list(torch.cuda.get_device_capability()),
                "bf16_supported": torch.cuda.is_bf16_supported(),
            }
        )
    manifest["sha256"] = sha256_bytes(canonical_json(manifest).encode("utf-8"))
    return manifest


def source_manifest(repo_root: Path, expected_commit: str) -> dict[str, object]:
    """Bind every score-bearing source byte to one committed ancestor."""

    def git(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ("git", *args), cwd=repo_root, check=False,
            capture_output=True, text=True,
        )

    resolved = git("rev-parse", "--verify", f"{expected_commit}^{{commit}}")
    if resolved.returncode:
        raise RuntimeError("SD-CST source commit is not locally available")
    commit = resolved.stdout.strip()
    if git("merge-base", "--is-ancestor", commit, "HEAD").returncode:
        raise RuntimeError("SD-CST source commit is not an ancestor of HEAD")
    hashes: dict[str, str] = {}
    for relative in FROZEN_SOURCE_PATHS:
        path = repo_root / relative
        if git("cat-file", "-e", f"{commit}:{relative}").returncode:
            raise RuntimeError(f"source commit omits frozen path: {relative}")
        if git("diff", "--quiet", commit, "--", relative).returncode:
            raise RuntimeError(f"runtime bytes differ from source commit: {relative}")
        hashes[relative] = sha256_file(path)
    payload = {"commit": commit, "files": hashes}
    payload["sha256"] = sha256_bytes(canonical_json(payload).encode("utf-8"))
    return payload


def development_ledger_bytes(
    report_sha256: str, development_sha256: str, source_commit: str,
) -> bytes:
    payload = {
        "schema": "r12_sd_cst_development_access_v1",
        "board_report_sha256": report_sha256,
        "development_sha256": development_sha256,
        "source_commit": source_commit,
        "access_number": 1,
    }
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _consume_development_access(
    ledger_dir: Path, report_sha256: str, development_sha256: str,
    source_commit: str,
) -> dict[str, str]:
    ledger_dir.mkdir(parents=True, exist_ok=True)
    path = ledger_dir / f"sd_cst_development_{development_sha256}.json"
    encoded = development_ledger_bytes(
        report_sha256, development_sha256, source_commit,
    )
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        os.write(descriptor, encoded)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return {"path": str(path.resolve()), "sha256": sha256_file(path)}


def board_manifest(
    data_dir: Path,
    split: str,
    *,
    source_commit: str | None = None,
    access_ledger_dir: Path | None = None,
) -> dict[str, object]:
    """Verify only the requested train/development bytes against the board receipt."""
    if split not in {TRAIN_SPLIT, DEVELOPMENT_SPLIT}:
        raise ValueError("SD-CST permits train or development access only")
    report_path = data_dir / "report.json"
    report = json.loads(report_path.read_text())
    if report.get("schema") != BOARD_SCHEMA or report.get("all_gates_pass") is not True:
        raise RuntimeError("SD-CST board is not admitted")
    if int(report.get("confirmation_accesses", -1)) != 0:
        raise RuntimeError("SD-CST board receipt records confirmation access")
    if source_commit is not None and report.get("source_commit") != source_commit:
        raise RuntimeError("SD-CST board source commit mismatch")
    filename = "train.jsonl" if split == TRAIN_SPLIT else "development.jsonl"
    expected = report.get("files", {}).get(filename, {}).get("sha256")
    if (
        not isinstance(expected, str)
        or len(expected) != 64
        or any(char not in "0123456789abcdef" for char in expected)
    ):
        raise RuntimeError(f"SD-CST {split} receipt hash is invalid")
    path = data_dir / filename
    report_sha = sha256_file(report_path)
    ledger = None
    if split == DEVELOPMENT_SPLIT:
        if access_ledger_dir is None or source_commit is None:
            raise RuntimeError("development requires source-bound one-read ledger")
        ledger = _consume_development_access(
            access_ledger_dir, report_sha, str(expected), source_commit,
        )
    actual = sha256_file(path)
    if actual != expected:
        raise RuntimeError(f"SD-CST {split} hash mismatch")
    return {
        "report_sha256": report_sha,
        "seed": report.get("seed"),
        "split": split,
        "opened_file": filename,
        "opened_sha256": actual,
        "declared_file_hashes": {
            name: values.get("sha256")
            for name, values in sorted(report.get("files", {}).items())
        },
        "confirmation_accesses": 0,
        "access_ledger": ledger,
    }


def _encoded_ids(tokenizer: object, text: str, seq_len: int) -> tuple[int, ...]:
    encoded = tokenizer.encode(text)
    ids = tuple(int(value) for value in encoded.ids)
    if not ids or len(ids) > seq_len:
        raise ValueError(f"tokenized source length {len(ids)} is outside [1,{seq_len}]")
    return ids


def _apply_event(state: tuple[int, ...], identity: int, kind: int, amount: int) -> tuple[int, ...]:
    values = list(state)
    source = values.index(identity)
    signed = -(amount + 1) if kind == 0 else amount + 1
    destination = min(2, max(0, source + signed))
    value = values.pop(source)
    values.insert(destination, value)
    return tuple(values)


def _parse_row(row: Mapping[str, object], tokenizer: object, seq_len: int, split: str) -> EncodedRow:
    if row.get("split") != split:
        raise ValueError("row split does not match requested split")
    if split == TRAIN_SPLIT and "oracle" in row:
        raise ValueError("training row contains forbidden outcome supervision")
    targets = row.get("compiler_targets")
    query_target = row.get("late_query_target")
    if not isinstance(targets, Mapping) or not isinstance(query_target, Mapping):
        raise ValueError("SD-CST row lacks compiler/query targets")
    slots = targets.get("event_slots")
    if not isinstance(slots, list) or len(slots) != EVENT_STEPS:
        raise ValueError(f"SD-CST row requires exactly {EVENT_STEPS} event slots")
    slots = sorted(slots, key=lambda item: int(item["semantic_ordinal"]))
    kind = tuple(int(item["kind_id"]) for item in slots)
    if kind.count(STOP_KIND) != 1:
        raise ValueError("SD-CST row requires exactly one STOP")
    identity = tuple(int(item.get("entity_role", 0)) for item in slots)
    amount = tuple(int(item.get("amount_id", 0)) for item in slots)
    initial_order = tuple(int(value) for value in targets["initial_order_roles"])
    initial_state = int(targets["initial_state_id"])
    if PERMUTATION_TO_STATE.get(initial_order) != initial_state:
        raise ValueError("initial-state category does not match initial order")
    program_text = str(row["program_text"])
    late_query_text = str(row["late_query_text"])
    if late_query_text in program_text:
        raise ValueError("late query leaked into program compiler input")

    final_state = answer = full_state = None
    if split == DEVELOPMENT_SPLIT:
        oracle = row.get("oracle")
        if not isinstance(oracle, Mapping):
            raise ValueError("development row lacks sealed oracle")
        final_order = tuple(int(value) for value in oracle["final_state_roles"])
        final_state = PERMUTATION_TO_STATE[final_order]
        answer = int(oracle["answer_role"])
        state = initial_order
        for event_kind, event_identity, event_amount in zip(
            kind, identity, amount, strict=True,
        ):
            if event_kind != STOP_KIND:
                state = _apply_event(state, event_identity, event_kind, event_amount)
        full_state = PERMUTATION_TO_STATE[state]

    return EncodedRow(
        row_id=str(row["id"]),
        split=split,
        variant=str(row.get("variant", "")),
        family_id=str(row["family_id"]) if row.get("family_id") is not None else None,
        program_text=program_text,
        late_query_text=late_query_text,
        program_ids=_encoded_ids(tokenizer, program_text, seq_len),
        query_ids=_encoded_ids(tokenizer, late_query_text, seq_len),
        initial_state=initial_state,
        event_kind=kind,
        event_identity=identity,
        amount=amount,
        query_position=int(query_target["position"]),
        halt_after=int(targets["halt_after"]),
        final_state=final_state,
        answer_role=answer,
        full_suffix_state=full_state,
    )


def load_rows(
    data_dir: Path,
    tokenizer: object,
    seq_len: int,
    split: str,
    *,
    source_commit: str | None = None,
    access_ledger_dir: Path | None = None,
) -> tuple[list[EncodedRow], dict[str, object]]:
    manifest = board_manifest(
        data_dir,
        split,
        source_commit=source_commit,
        access_ledger_dir=access_ledger_dir,
    )
    path = data_dir / str(manifest["opened_file"])
    rows = [
        _parse_row(json.loads(line), tokenizer, seq_len, split)
        for line in path.read_text().splitlines() if line.strip()
    ]
    if not rows:
        raise ValueError("SD-CST split is empty")
    return rows, manifest


def pad_sources(rows: Sequence[EncodedRow], field: str, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    sequences = [getattr(row, field) for row in rows]
    width = max(len(value) for value in sequences)
    ids = torch.zeros((len(rows), width), dtype=torch.long, device=device)
    valid = torch.zeros((len(rows), width), dtype=torch.bool, device=device)
    for index, sequence in enumerate(sequences):
        ids[index, :len(sequence)] = torch.tensor(sequence, dtype=torch.long, device=device)
        valid[index, :len(sequence)] = True
    return ids, valid


def label_batch(rows: Sequence[EncodedRow], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "initial_state_targets": torch.tensor([row.initial_state for row in rows], device=device),
        "event_kind_targets": torch.tensor([row.event_kind for row in rows], device=device),
        "event_identity_targets": torch.tensor([row.event_identity for row in rows], device=device),
        "amount_targets": torch.tensor([row.amount for row in rows], device=device),
        "query_targets": torch.tensor([row.query_position for row in rows], device=device),
    }


def deterministic_batches(count: int, batch_size: int, seed: int, epoch: int) -> list[list[int]]:
    if count <= 0 or batch_size <= 0:
        raise ValueError("count and batch size must be positive")
    indices = list(range(count))
    random.Random(seed ^ (epoch * 0x9E3779B1)).shuffle(indices)
    return [indices[start:start + batch_size] for start in range(0, count, batch_size)]


def cosine_scale(step: int, total: int, warmup: int, floor: float = 0.0) -> float:
    if step < warmup:
        return (step + 1) / max(1, warmup)
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))


def motor_certificate(device: torch.device) -> dict[str, torch.Tensor]:
    """Return all 72 non-STOP transitions plus one STOP per state."""
    action_rows = [
        (state, kind, identity, amount)
        for state in range(STATE_COUNT)
        for kind in range(STOP_KIND)
        for identity in range(IDENTITY_COUNT)
        for amount in range(AMOUNT_COUNT)
    ]
    stop_rows = [(state, STOP_KIND, 0, 0) for state in range(STATE_COUNT)]
    rows = action_rows + stop_rows
    targets = []
    for state_id, kind, identity, amount in rows:
        if kind == STOP_KIND:
            targets.append(state_id)
        else:
            order = PERMUTATIONS[state_id]
            targets.append(PERMUTATION_TO_STATE[_apply_event(order, identity, kind, amount)])
    ids = torch.tensor(rows, dtype=torch.long, device=device)
    return {
        "state": F.one_hot(ids[:, 0], STATE_COUNT).float(),
        "event_kind": F.one_hot(ids[:, 1], EVENT_KIND_COUNT).float(),
        "event_identity": F.one_hot(ids[:, 2], IDENTITY_COUNT).float(),
        "amount": F.one_hot(ids[:, 3], AMOUNT_COUNT).float(),
        "targets": torch.tensor(targets, dtype=torch.long, device=device),
        "is_stop": ids[:, 1].eq(STOP_KIND),
    }


def reader_certificate(device: torch.device) -> dict[str, torch.Tensor]:
    rows = [(state, query) for state in range(STATE_COUNT) for query in range(QUERY_COUNT)]
    ids = torch.tensor(rows, dtype=torch.long, device=device)
    targets = torch.tensor(
        [PERMUTATIONS[state][query] for state, query in rows],
        dtype=torch.long, device=device,
    )
    return {
        "state": F.one_hot(ids[:, 0], STATE_COUNT).float(),
        "query": F.one_hot(ids[:, 1], QUERY_COUNT).float(),
        "targets": targets,
    }


def _certificate_hash(payload: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(payload):
        tensor = payload[name].detach().cpu().contiguous()
        digest.update(name.encode("ascii"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def fit_motor_certificate(system: SDCSTSystem, *, seed: int, lr: float, max_updates: int) -> dict[str, object]:
    torch.manual_seed(seed)
    device = next(system.motor.parameters()).device
    board = motor_certificate(device)
    optimizer = torch.optim.AdamW(system.motor.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.0)
    updates = 0
    for update in range(max_updates):
        optimizer.zero_grad(set_to_none=True)
        loss = atomic_motor_loss(
            system.motor,
            state=board["state"],
            event_kind=board["event_kind"],
            event_identity=board["event_identity"],
            amount=board["amount"],
            next_state_targets=board["targets"],
        )
        loss.backward()
        optimizer.step()
        updates = update + 1
        with torch.no_grad():
            logits = system.motor(
                board["state"], board["event_kind"],
                board["event_identity"], board["amount"],
            )
            correct = logits.argmax(-1).eq(board["targets"])
    action = ~board["is_stop"]
    if int(action.sum()) != 72 or int(board["is_stop"].sum()) != 6:
        raise RuntimeError("motor certificate board is incomplete")
    if not bool(correct[action].all()) or not bool(correct[~action].all()):
        raise RuntimeError("motor failed complete 72 action + 6 STOP certificate")
    return {
        "rows": 78,
        "state_action_rows": 72,
        "stop_rows": 6,
        "state_action_correct": int(correct[action].sum()),
        "stop_correct": int(correct[~action].sum()),
        "exact": True,
        "updates": updates,
        "final_loss": float(loss.detach()),
        "certificate_sha256": _certificate_hash(board),
        "optimizer": {"name": "AdamW", "lr": lr, "betas": [0.9, 0.95], "weight_decay": 0.0},
        "schedule": {"name": "constant", "updates": max_updates},
    }


def fit_reader_certificate(system: SDCSTSystem, *, seed: int, lr: float, max_updates: int) -> dict[str, object]:
    torch.manual_seed(seed)
    device = next(system.reader.parameters()).device
    board = reader_certificate(device)
    optimizer = torch.optim.AdamW(system.reader.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.0)
    updates = 0
    for update in range(max_updates):
        optimizer.zero_grad(set_to_none=True)
        loss = reader_loss(
            system.reader,
            state=board["state"], query=board["query"],
            answer_targets=board["targets"],
        )
        loss.backward()
        optimizer.step()
        updates = update + 1
        with torch.no_grad():
            correct = system.reader(board["state"], board["query"]).argmax(-1).eq(board["targets"])
    if int(correct.sum()) != 18:
        raise RuntimeError("reader failed complete 18 state-query certificate")
    return {
        "rows": 18,
        "correct": 18,
        "exact": True,
        "updates": updates,
        "final_loss": float(loss.detach()),
        "certificate_sha256": _certificate_hash(board),
        "optimizer": {"name": "AdamW", "lr": lr, "betas": [0.9, 0.95], "weight_decay": 0.0},
        "schedule": {"name": "constant", "updates": max_updates},
    }


def _autocast(device: torch.device):
    return torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()


def fit_compiler(
    system: SDCSTSystem,
    rows: Sequence[EncodedRow],
    *,
    seed: int,
    batch_size: int,
    epochs: int,
    lr: float,
    warmup: int,
    clip: float,
) -> dict[str, object]:
    if any(row.split != TRAIN_SPLIT or row.final_state is not None for row in rows):
        raise RuntimeError("compiler phase received outcome-bearing or non-training rows")
    system.compiler.train()
    system.motor.requires_grad_(False).eval()
    system.reader.requires_grad_(False).eval()
    parameters = list(system.compiler.adapter_parameters())
    if not parameters or any(not parameter.requires_grad for parameter in parameters):
        raise RuntimeError("compiler adapter parameter set is invalid")
    if any(parameter.requires_grad for parameter in system.base_model.parameters()):
        raise RuntimeError("Shohin base must remain frozen")
    total_updates = epochs * math.ceil(len(rows) / batch_size)
    optimizer = torch.optim.AdamW(parameters, lr=lr, betas=(0.9, 0.95), weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: cosine_scale(step, total_updates, warmup),
    )
    device = next(system.motor.parameters()).device
    started = time.time()
    sums = {name: 0.0 for name in ("total", "initial_state", "event_kind", "event_identity", "amount", "late_query")}
    seen = 0
    update = 0
    for epoch in range(epochs):
        for indices in deterministic_batches(len(rows), batch_size, seed, epoch):
            batch = [rows[index] for index in indices]
            program_ids, program_mask = pad_sources(batch, "program_ids", device)
            query_ids, query_mask = pad_sources(batch, "query_ids", device)
            labels = label_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with _autocast(device):
                tape = system.compile_program(program_ids, program_mask)
                query = system.compile_late_query(query_ids, query_mask)
                losses = compiler_field_losses(
                    tape,
                    initial_state_targets=labels["initial_state_targets"],
                    event_kind_targets=labels["event_kind_targets"],
                    event_identity_targets=labels["event_identity_targets"],
                    amount_targets=labels["amount_targets"],
                )
                query_loss = late_query_loss(query, query_targets=labels["query_targets"])
                total = losses["total"] + query_loss
            total.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, clip)
            if not bool(torch.isfinite(gradient_norm)):
                raise RuntimeError("non-finite compiler gradient")
            optimizer.step()
            scheduler.step()
            count = len(batch)
            seen += count
            update += 1
            sums["total"] += float(total.detach()) * count
            for name in ("initial_state", "event_kind", "event_identity", "amount"):
                sums[name] += float(losses[name].detach()) * count
            sums["late_query"] += float(query_loss.detach()) * count
    return {
        "rows_per_epoch": len(rows),
        "epochs": epochs,
        "updates": update,
        "charged_rows": seen,
        "losses": {name: value / seen for name, value in sums.items()},
        "elapsed_seconds": time.time() - started,
        "objective": (
            "initial_state + event_kind + non_STOP_identity + non_STOP_amount + "
            "separately_tokenized_late_query_position; no state, answer, trajectory, "
            "episode, development, or confirmation supervision"
        ),
        "optimizer": {"name": "AdamW", "lr": lr, "betas": [0.9, 0.95], "weight_decay": 0.01},
        "schedule": {"name": "warmup_cosine", "updates": total_updates, "warmup": warmup, "floor": 0.0},
        "batch_size": batch_size,
        "gradient_clip": clip,
    }


@torch.no_grad()
def compiler_train_metrics(system: SDCSTSystem, rows: Sequence[EncodedRow], batch_size: int) -> dict[str, object]:
    system.eval()
    device = next(system.motor.parameters()).device
    counts = {name: 0 for name in ("rows", "initial", "kind", "identity", "amount", "query", "whole_tape")}
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        program_ids, program_mask = pad_sources(batch, "program_ids", device)
        query_ids, query_mask = pad_sources(batch, "query_ids", device)
        labels = label_batch(batch, device)
        with _autocast(device):
            soft_tape = system.compile_program(program_ids, program_mask)
            soft_query = system.compile_late_query(query_ids, query_mask)
        tape, query = seal_hard_payload(soft_tape.hard(), soft_query.hard())
        initial = tape.initial_state.long().eq(labels["initial_state_targets"].cpu())
        kind = tape.event_kind.long().eq(labels["event_kind_targets"].cpu()).all(dim=1)
        active = labels["event_kind_targets"].cpu().ne(STOP_KIND)
        identity = (tape.event_identity.long().eq(labels["event_identity_targets"].cpu()) | ~active).all(dim=1)
        amount = (tape.amount.long().eq(labels["amount_targets"].cpu()) | ~active).all(dim=1)
        query_ok = query.position.long().eq(labels["query_targets"].cpu())
        counts["rows"] += len(batch)
        for name, value in (("initial", initial), ("kind", kind), ("identity", identity), ("amount", amount), ("query", query_ok)):
            counts[name] += int(value.sum())
        counts["whole_tape"] += int((initial & kind & identity & amount).sum())
    rows_count = counts.pop("rows")
    return {"rows": rows_count, "exact": counts, "rates": {name: value / rows_count for name, value in counts.items()}}


def seal_hard_payload(tape: HardProgramTape, query: HardLateQuery) -> tuple[HardProgramTape, HardLateQuery]:
    """Make the score channel CPU uint8-only and discard all compiler logits."""
    if not isinstance(tape, HardProgramTape) or not isinstance(query, HardLateQuery):
        raise TypeError("score payload must already be hard categorical objects")
    sealed_tape = HardProgramTape(
        tape.initial_state.detach().to(device="cpu", dtype=torch.uint8).clone(),
        tape.event_kind.detach().to(device="cpu", dtype=torch.uint8).clone(),
        tape.event_identity.detach().to(device="cpu", dtype=torch.uint8).clone(),
        tape.amount.detach().to(device="cpu", dtype=torch.uint8).clone(),
    )
    sealed_query = HardLateQuery(
        query.position.detach().to(device="cpu", dtype=torch.uint8).clone()
    )
    if not torch.equal(
        sealed_tape.event_kind.eq(STOP_KIND).sum(dim=1),
        torch.ones(sealed_tape.batch_size, dtype=torch.long),
    ):
        raise RuntimeError("sealed score payload does not contain exactly one STOP")
    return sealed_tape, sealed_query


def _slice_hard(tape: HardProgramTape, query: HardLateQuery, index: int) -> tuple[HardProgramTape, HardLateQuery]:
    part = slice(index, index + 1)
    return HardProgramTape(
        tape.initial_state[part].clone(), tape.event_kind[part].clone(),
        tape.event_identity[part].clone(), tape.amount[part].clone(),
    ), HardLateQuery(query.position[part].clone())


def _replace_suffix(receiver: HardProgramTape, donor: HardProgramTape) -> HardProgramTape:
    stop = int(receiver.event_kind[0].eq(STOP_KIND).nonzero(as_tuple=False)[0, 0])
    start = stop + 1
    return HardProgramTape(
        receiver.initial_state.clone(),
        torch.cat((receiver.event_kind[:, :start], donor.event_kind[:, start:]), dim=1),
        torch.cat((receiver.event_identity[:, :start], donor.event_identity[:, start:]), dim=1),
        torch.cat((receiver.amount[:, :start], donor.amount[:, start:]), dim=1),
    )


@torch.no_grad()
def hard_score_payload(
    system: SDCSTSystem,
    tape: HardProgramTape,
    query: HardLateQuery,
    rows: Sequence[EncodedRow],
) -> tuple[list[dict[str, object]], dict[str, int]]:
    """The sole score-bearing execution path."""
    if len(rows) != tape.batch_size or len(rows) != query.position.shape[0]:
        raise ValueError("hard payload and row counts differ")
    if any(row.final_state is None or row.answer_role is None for row in rows):
        raise ValueError("hard scoring requires development oracles")
    normal = system.rollout_hard(tape, query)
    forced = system.rollout_hard(tape, query, force_alive=True)
    reset = system.rollout_hard(tape, query, control="reset")
    freeze = system.rollout_hard(tape, query, control="freeze")
    predicted_answer = normal.answer_logits.argmax(-1).cpu()
    forced_answer = forced.answer_logits.argmax(-1).cpu()
    records: list[dict[str, object]] = []
    totals = {name: 0 for name in ("rows", "state", "answer", "joint", "force_alive_state", "force_alive_answer", "reset_changed", "freeze_changed")}
    for index, row in enumerate(rows):
        state = int(normal.final_state[index])
        answer = int(predicted_answer[index])
        force_state = int(forced.final_state[index])
        expected_full = int(row.full_suffix_state)
        expected_force_answer = PERMUTATIONS[expected_full][row.query_position]
        record = {
            "id": row.row_id,
            "family_id": row.family_id,
            "variant": row.variant,
            "halt_after": row.halt_after,
            "predicted_state": state,
            "expected_state": row.final_state,
            "predicted_answer": answer,
            "expected_answer": row.answer_role,
            "state_correct": state == row.final_state,
            "answer_correct": answer == row.answer_role,
            "joint_correct": state == row.final_state and answer == row.answer_role,
            "force_alive_state": force_state,
            "force_alive_expected_state": expected_full,
            "force_alive_state_correct": force_state == expected_full,
            "force_alive_answer_correct": int(forced_answer[index]) == expected_force_answer,
            "reset_changed": int(reset.final_state[index]) != state,
            "freeze_changed": int(freeze.final_state[index]) != state,
        }
        records.append(record)
        totals["rows"] += 1
        for key in ("state", "answer", "joint"):
            totals[key] += int(record[f"{key}_correct"])
        totals["force_alive_state"] += int(record["force_alive_state_correct"])
        totals["force_alive_answer"] += int(record["force_alive_answer_correct"])
        totals["reset_changed"] += int(record["reset_changed"])
        totals["freeze_changed"] += int(record["freeze_changed"])
    return records, totals


def _hard_equal(left: HardProgramTape, right: HardProgramTape) -> bool:
    return all(torch.equal(getattr(left, name), getattr(right, name)) for name in ("initial_state", "event_kind", "event_identity", "amount"))


def _gold_tape(row: EncodedRow) -> dict[str, object]:
    return {
        "initial_state_id": row.initial_state,
        "event_slots": [
            {
                "kind_id": kind,
                "entity_role": identity,
                "amount_id": amount,
                "identity_and_amount_scored": kind != STOP_KIND,
            }
            for kind, identity, amount in zip(
                row.event_kind, row.event_identity, row.amount, strict=True,
            )
        ],
    }


def _predicted_tape(tape: HardProgramTape) -> dict[str, object]:
    if tape.batch_size != 1:
        raise ValueError("predicted tape serialization requires one row")
    return {
        "initial_state_id": int(tape.initial_state[0]),
        "event_slots": [
            {
                "kind_id": int(tape.event_kind[0, step]),
                "entity_role": int(tape.event_identity[0, step]),
                "amount_id": int(tape.amount[0, step]),
            }
            for step in range(EVENT_STEPS)
        ],
    }


def _combine_hard(
    tapes: Sequence[HardProgramTape], queries: Sequence[HardLateQuery],
) -> tuple[HardProgramTape, HardLateQuery]:
    return HardProgramTape(
        torch.cat([tape.initial_state for tape in tapes]),
        torch.cat([tape.event_kind for tape in tapes]),
        torch.cat([tape.event_identity for tape in tapes]),
        torch.cat([tape.amount for tape in tapes]),
    ), HardLateQuery(torch.cat([query.position for query in queries]))


def _advance_state(state_id: int, kind: int, identity: int, amount: int) -> int:
    if kind == STOP_KIND:
        return state_id
    return PERMUTATION_TO_STATE[
        _apply_event(PERMUTATIONS[state_id], identity, kind, amount)
    ]


def _gold_state_swap(receiver: EncodedRow, donor: EncodedRow, after_step: int) -> int:
    receiver_state = receiver.initial_state
    donor_state = donor.initial_state
    for step in range(after_step + 1):
        if receiver.event_kind[step] == STOP_KIND or donor.event_kind[step] == STOP_KIND:
            raise ValueError("state swap must precede STOP in both gold programs")
        receiver_state = _advance_state(
            receiver_state,
            receiver.event_kind[step],
            receiver.event_identity[step],
            receiver.amount[step],
        )
        donor_state = _advance_state(
            donor_state,
            donor.event_kind[step],
            donor.event_identity[step],
            donor.amount[step],
        )
    receiver_state = donor_state
    alive = True
    for step in range(after_step + 1, EVENT_STEPS):
        kind = receiver.event_kind[step]
        if kind == STOP_KIND:
            alive = False
        elif alive:
            receiver_state = _advance_state(
                receiver_state,
                kind,
                receiver.event_identity[step],
                receiver.amount[step],
            )
    return receiver_state


@torch.no_grad()
def state_swap_evidence(
    system: SDCSTSystem,
    rows: Sequence[EncodedRow],
    tapes: Sequence[HardProgramTape],
    queries: Sequence[HardLateQuery],
) -> dict[str, dict[str, object]]:
    canonical = [index for index, row in enumerate(rows) if row.variant == "canonical"]
    evidence: dict[str, dict[str, object]] = {}
    for receiver_index in canonical:
        receiver = rows[receiver_index]
        selection: tuple[int, int] | None = None
        for candidate in canonical:
            if candidate == receiver_index:
                continue
            donor = rows[candidate]
            for after_step in range(min(receiver.halt_after, donor.halt_after)):
                expected = _gold_state_swap(receiver, donor, after_step)
                if (
                    expected != receiver.final_state
                    and PERMUTATIONS[expected][receiver.query_position]
                    != receiver.answer_role
                ):
                    selection = (candidate, after_step)
                    break
            if selection is not None:
                break
        if selection is None:
            raise RuntimeError(f"no separating state-swap donor for {receiver.row_id}")
        donor_index, after_step = selection
        pair_tape, pair_query = _combine_hard(
            [tapes[receiver_index], tapes[donor_index]],
            [queries[receiver_index], queries[donor_index]],
        )
        permutation = torch.tensor([1, 0], dtype=torch.long)
        result = system.rollout_hard(
            pair_tape,
            pair_query,
            state_swap=StateSwap(
                after_step=after_step, batch_permutation=permutation,
            ),
        )
        state = int(result.final_state[0])
        evidence[receiver.row_id] = {
            "donor_id": rows[donor_index].row_id,
            "after_step": after_step,
            "final_state_id": state,
            "answer_role": int(result.answer_logits.argmax(-1)[0]),
        }
    return evidence


@torch.no_grad()
def causal_family_metrics(
    system: SDCSTSystem,
    rows: Sequence[EncodedRow],
    tapes: Sequence[HardProgramTape],
    queries: Sequence[HardLateQuery],
) -> dict[str, object]:
    families: dict[str, dict[str, int]] = {}
    for index, row in enumerate(rows):
        if row.family_id is not None:
            families.setdefault(row.family_id, {})[row.variant] = index
    counts = {name: 0 for name in (
        "families", "query_program_identical", "query_state_invariant",
        "query_answer_follows", "suffix_state_invariant", "suffix_answer_invariant",
    )}
    for variants in families.values():
        if not {"canonical", "query_swap", "post_halt_suffix"}.issubset(variants):
            continue
        canonical_i = variants["canonical"]
        query_i = variants["query_swap"]
        suffix_i = variants["post_halt_suffix"]
        base_tape, base_query = tapes[canonical_i], queries[canonical_i]
        query_tape, swapped_query = tapes[query_i], queries[query_i]
        base = system.rollout_hard(base_tape, base_query)
        query_result = system.rollout_hard(base_tape, swapped_query)
        suffix_tape = _replace_suffix(base_tape, tapes[suffix_i])
        suffix_result = system.rollout_hard(suffix_tape, base_query)
        base_answer = int(base.answer_logits.argmax(-1)[0])
        query_answer = int(query_result.answer_logits.argmax(-1)[0])
        counts["families"] += 1
        counts["query_program_identical"] += int(_hard_equal(base_tape, query_tape))
        counts["query_state_invariant"] += int(torch.equal(base.final_state, query_result.final_state))
        counts["query_answer_follows"] += int(
            base_answer == rows[canonical_i].answer_role
            and query_answer == rows[query_i].answer_role
        )
        counts["suffix_state_invariant"] += int(torch.equal(base.final_state, suffix_result.final_state))
        counts["suffix_answer_invariant"] += int(
            base_answer == int(suffix_result.answer_logits.argmax(-1)[0])
        )

    state_swap_rows = [index for index, row in enumerate(rows) if row.variant == "canonical" and row.halt_after > 1]
    changed_state = changed_answer = 0
    if len(state_swap_rows) >= 2:
        selected = state_swap_rows[: min(128, len(state_swap_rows))]
        combined = HardProgramTape(
            torch.cat([tapes[index].initial_state for index in selected]),
            torch.cat([tapes[index].event_kind for index in selected]),
            torch.cat([tapes[index].event_identity for index in selected]),
            torch.cat([tapes[index].amount for index in selected]),
        )
        combined_query = HardLateQuery(torch.cat([queries[index].position for index in selected]))
        permutation = torch.roll(torch.arange(len(selected), dtype=torch.long), 1)
        ordinary = system.rollout_hard(combined, combined_query)
        swapped = system.rollout_hard(
            combined, combined_query,
            state_swap=StateSwap(after_step=0, batch_permutation=permutation),
        )
        changed_state = int(ordinary.final_state.ne(swapped.final_state).sum())
        changed_answer = int(ordinary.answer_logits.argmax(-1).ne(swapped.answer_logits.argmax(-1)).sum())
    counts["state_swap_rows"] = min(128, len(state_swap_rows)) if len(state_swap_rows) >= 2 else 0
    counts["state_swap_state_changed"] = changed_state
    counts["state_swap_answer_changed"] = changed_answer
    denominator = max(1, counts["families"])
    return {
        "counts": counts,
        "rates": {
            key: counts[key] / denominator
            for key in (
                "query_program_identical", "query_state_invariant",
                "query_answer_follows", "suffix_state_invariant",
                "suffix_answer_invariant",
            )
        },
        "contract": (
            "late-query exchange, post-STOP suffix replacement, force-alive, "
            "reset/freeze, and post-step-0 categorical state swap; every arm uses rollout_hard"
        ),
    }


@torch.no_grad()
def certificate_evidence(system: SDCSTSystem) -> dict[str, list[dict[str, object]]]:
    device = next(system.motor.parameters()).device
    motor = motor_certificate(device)
    motor_predictions = system.motor(
        motor["state"],
        motor["event_kind"],
        motor["event_identity"],
        motor["amount"],
    ).argmax(-1).cpu()
    state_action: list[dict[str, object]] = []
    stop: list[dict[str, object]] = []
    for index in range(78):
        state_id = int(motor["state"][index].argmax())
        kind_id = int(motor["event_kind"][index].argmax())
        row = {
            "state_id": state_id,
            "kind_id": kind_id,
            "entity_role": int(motor["event_identity"][index].argmax()),
            "amount_id": int(motor["amount"][index].argmax()),
            "predicted_state_id": int(motor_predictions[index]),
            "predicted_alive": kind_id != STOP_KIND,
        }
        (stop if kind_id == STOP_KIND else state_action).append(row)

    reader = reader_certificate(device)
    reader_predictions = system.reader(reader["state"], reader["query"]).argmax(-1).cpu()
    reader_rows = [
        {
            "state_id": int(reader["state"][index].argmax()),
            "query_position": int(reader["query"][index].argmax()),
            "predicted_answer_role": int(reader_predictions[index]),
        }
        for index in range(18)
    ]

    dead_rows: list[dict[str, object]] = []
    action_domain = [
        (kind, identity, amount)
        for kind in range(STOP_KIND)
        for identity in range(IDENTITY_COUNT)
        for amount in range(AMOUNT_COUNT)
    ] + [(STOP_KIND, 0, 0)]
    for state_id in range(STATE_COUNT):
        for kind_id, identity, amount in action_domain:
            kinds = torch.zeros((1, EVENT_STEPS), dtype=torch.uint8)
            identities = torch.zeros((1, EVENT_STEPS), dtype=torch.uint8)
            amounts = torch.zeros((1, EVENT_STEPS), dtype=torch.uint8)
            kinds[0, 0] = STOP_KIND
            if kind_id != STOP_KIND:
                kinds[0, 1] = kind_id
                identities[0, 1] = identity
                amounts[0, 1] = amount
            tape = HardProgramTape(
                torch.tensor([state_id], dtype=torch.uint8),
                kinds,
                identities,
                amounts,
            )
            result = system.rollout_hard(
                tape, HardLateQuery(torch.zeros(1, dtype=torch.uint8)),
            )
            dead_rows.append({
                "state_id": state_id,
                "kind_id": kind_id,
                "entity_role": identity,
                "amount_id": amount,
                "predicted_state_id": int(result.final_state[0]),
                "predicted_alive": bool(result.alive_trajectory[-1][0]),
            })
    return {
        "motor_state_action": state_action,
        "motor_stop": stop,
        "dead_invariance": dead_rows,
        "reader": reader_rows,
    }


def _packet_bytes(tape: HardProgramTape, query: HardLateQuery, index: int) -> tuple[bytes, bytes]:
    program = bytes([int(tape.initial_state[index])]) + bytes(tape.event_kind[index].tolist())
    program += bytes(tape.event_identity[index].tolist()) + bytes(tape.amount[index].tolist())
    return program, bytes([int(query.position[index])])


@torch.no_grad()
def source_poison_evidence(
    system: SDCSTSystem,
    rows: Sequence[EncodedRow],
    tapes: Sequence[HardProgramTape],
    queries: Sequence[HardLateQuery],
) -> list[dict[str, str]]:
    combined, combined_query = _combine_hard(tapes, queries)
    clean = system.rollout_hard(combined, combined_query)
    poisoned_sources = [bytearray(len(row.program_ids)) for row in rows]
    for source in poisoned_sources:
        source[:] = b"\xff" * len(source)
    poisoned = system.rollout_hard(combined, combined_query)
    clean_answers = clean.answer_logits.argmax(-1)
    poisoned_answers = poisoned.answer_logits.argmax(-1)
    evidence = []
    for index, row in enumerate(rows):
        program, query = _packet_bytes(combined, combined_query, index)
        clean_rollout = bytes([
            int(clean.final_state[index]), int(clean_answers[index]),
        ])
        poisoned_rollout = bytes([
            int(poisoned.final_state[index]), int(poisoned_answers[index]),
        ])
        evidence.append({
            "id": row.row_id,
            "clean_program_tape_b64": base64.b64encode(program).decode("ascii"),
            "poisoned_program_tape_b64": base64.b64encode(program).decode("ascii"),
            "clean_late_query_b64": base64.b64encode(query).decode("ascii"),
            "poisoned_late_query_b64": base64.b64encode(query).decode("ascii"),
            "clean_rollout_b64": base64.b64encode(clean_rollout).decode("ascii"),
            "poisoned_rollout_b64": base64.b64encode(poisoned_rollout).decode("ascii"),
        })
    return evidence


def _control_counts(result, rows: Sequence[EncodedRow]) -> tuple[int, int]:
    answers = result.answer_logits.argmax(-1).cpu()
    state = sum(int(result.final_state[index]) == row.final_state for index, row in enumerate(rows))
    answer = sum(int(answers[index]) == row.answer_role for index, row in enumerate(rows))
    return state, answer


@torch.no_grad()
def control_evidence(
    system: SDCSTSystem,
    rows: Sequence[EncodedRow],
    tapes: Sequence[HardProgramTape],
    queries: Sequence[HardLateQuery],
) -> dict[str, dict[str, int]]:
    combined, combined_query = _combine_hard(tapes, queries)
    batch = len(rows)
    uniform_kinds = torch.zeros((batch, EVENT_STEPS), dtype=torch.uint8)
    uniform_kinds[:, 0] = STOP_KIND
    uniform = HardProgramTape(
        torch.zeros(batch, dtype=torch.uint8),
        uniform_kinds,
        torch.zeros((batch, EVENT_STEPS), dtype=torch.uint8),
        torch.zeros((batch, EVENT_STEPS), dtype=torch.uint8),
    )
    uniform_query = HardLateQuery(torch.zeros(batch, dtype=torch.uint8))

    sf_initial = torch.empty(batch, dtype=torch.uint8)
    sf_kind = torch.empty((batch, EVENT_STEPS), dtype=torch.uint8)
    sf_identity = torch.empty((batch, EVENT_STEPS), dtype=torch.uint8)
    sf_amount = torch.empty((batch, EVENT_STEPS), dtype=torch.uint8)
    sf_query = torch.empty(batch, dtype=torch.uint8)
    for index, row in enumerate(rows):
        digest = hashlib.sha256(("sd-cst-source-free:" + row.row_id).encode()).digest()
        sf_initial[index] = digest[0] % STATE_COUNT
        stop = digest[1] % EVENT_STEPS
        for step in range(EVENT_STEPS):
            sf_kind[index, step] = STOP_KIND if step == stop else digest[2 + step] % STOP_KIND
            sf_identity[index, step] = digest[10 + step] % IDENTITY_COUNT
            sf_amount[index, step] = digest[18 + step] % AMOUNT_COUNT
        sf_query[index] = digest[26] % QUERY_COUNT
    source_free = HardProgramTape(sf_initial, sf_kind, sf_identity, sf_amount)
    source_free_query = HardLateQuery(sf_query)

    permutation = torch.roll(torch.arange(batch, dtype=torch.long), 1)
    shuffled = HardProgramTape(
        combined.initial_state.index_select(0, permutation),
        combined.event_kind.index_select(0, permutation),
        combined.event_identity.index_select(0, permutation),
        combined.amount.index_select(0, permutation),
    )
    shuffled_query = HardLateQuery(combined_query.position.index_select(0, permutation))
    results = {
        "uniform": system.rollout_hard(uniform, uniform_query),
        "source_free": system.rollout_hard(source_free, source_free_query),
        "shuffled": system.rollout_hard(shuffled, shuffled_query),
        "reset": system.rollout_hard(combined, combined_query, control="reset"),
        "freeze": system.rollout_hard(combined, combined_query, control="freeze"),
    }
    output: dict[str, dict[str, int]] = {}
    for name, result in results.items():
        state, answer = _control_counts(result, rows)
        output[f"{name}_state"] = {"cases": batch, "correct": state}
        output[f"{name}_answer"] = {"cases": batch, "correct": answer}
    return output


def _adapter_state(system: SDCSTSystem) -> dict[str, dict[str, torch.Tensor]]:
    compiler = {
        name: value.detach().cpu()
        for name, value in system.compiler.state_dict().items()
        if not name.startswith("base_model.")
    }
    return {
        "compiler": compiler,
        "motor": {name: value.detach().cpu() for name, value in system.motor.state_dict().items()},
        "reader": {name: value.detach().cpu() for name, value in system.reader.state_dict().items()},
    }


def _load_adapter_state(system: SDCSTSystem, state: Mapping[str, Mapping[str, torch.Tensor]]) -> None:
    compiler_state = system.compiler.state_dict()
    supplied = state["compiler"]
    unknown = set(supplied) - set(compiler_state)
    if unknown:
        raise RuntimeError(f"unknown compiler state keys: {sorted(unknown)[:4]}")
    compiler_state.update(supplied)
    system.compiler.load_state_dict(compiler_state, strict=True)
    system.motor.load_state_dict(state["motor"], strict=True)
    system.reader.load_state_dict(state["reader"], strict=True)


def load_system(base_path: Path, device: torch.device, architecture: Mapping[str, int]) -> tuple[SDCSTSystem, dict[str, object]]:
    checkpoint = torch.load(base_path, map_location="cpu", weights_only=False)
    cfg = GPTConfig(**checkpoint["cfg"])
    base = GPT(cfg)
    base.load_state_dict(checkpoint["model"], strict=True)
    base.requires_grad_(False).eval()
    if device.type == "cuda":
        base.to(device=device, dtype=torch.bfloat16)
    else:
        base.to(device=device)
    system = SDCSTSystem(
        base,
        compiler_layer=int(architecture["compiler_layer"]),
        compiler_width=int(architecture["compiler_width"]),
        compiler_heads=int(architecture["compiler_heads"]),
        compiler_layers=int(architecture["compiler_layers"]),
        compiler_ff=int(architecture["compiler_ff"]),
        motor_hidden=int(architecture["motor_hidden"]),
        reader_hidden=int(architecture["reader_hidden"]),
    ).to(device)
    report = system.parameter_report()
    if report["complete_system"] >= MAX_SYSTEM_PARAMETERS or report["headroom"] <= 0:
        raise RuntimeError("SD-CST system violates strict sub-150M cap")
    return system, checkpoint


def architecture_from_args(args: argparse.Namespace) -> dict[str, int]:
    return {
        "compiler_layer": args.compiler_layer,
        "compiler_width": args.compiler_width,
        "compiler_heads": args.compiler_heads,
        "compiler_layers": args.compiler_layers,
        "compiler_ff": args.compiler_ff,
        "motor_hidden": args.motor_hidden,
        "reader_hidden": args.reader_hidden,
    }


def validate_frozen_training_contract(
    args: argparse.Namespace,
    rows: Sequence[EncodedRow],
    system: SDCSTSystem,
    base_checkpoint: Mapping[str, object],
) -> None:
    actual = {
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "compiler_lr": args.compiler_lr,
        "motor_lr": args.motor_lr,
        "reader_lr": args.reader_lr,
        "motor_updates": args.motor_updates,
        "reader_updates": args.reader_updates,
        "warmup": args.warmup,
        "clip": args.clip,
    }
    if actual != FROZEN_TRAINING:
        raise RuntimeError(f"SD-CST frozen training contract mismatch: {actual}")
    if len(rows) != FROZEN_TRAIN_ROWS:
        raise RuntimeError(f"SD-CST requires {FROZEN_TRAIN_ROWS} training rows")
    if sha256_file(args.base) != FROZEN_BASE_SHA256:
        raise RuntimeError("SD-CST base checkpoint differs from preregistration")
    if sha256_file(args.tokenizer) != FROZEN_TOKENIZER_SHA256:
        raise RuntimeError("SD-CST tokenizer differs from preregistration")
    if int(base_checkpoint.get("step", -1)) != 300_000:
        raise RuntimeError("SD-CST base checkpoint is not step 300000")
    if system.parameter_report()["complete_system"] != FROZEN_PARAMETERS:
        raise RuntimeError("SD-CST complete parameter count differs from preregistration")


def train_main(args: argparse.Namespace) -> None:
    if args.out.exists():
        raise SystemExit(f"refusing existing SD-CST checkpoint: {args.out}")
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    if device.type != "cuda" and not args.allow_cpu_train:
        raise SystemExit("SD-CST neural training requires CUDA")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.set_float32_matmul_precision("high")
    source = source_manifest(args.repo_root.resolve(), args.source_commit)
    runtime = runtime_manifest()
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    architecture = architecture_from_args(args)
    system, base_checkpoint = load_system(args.base, device, architecture)
    rows, board = load_rows(
        args.data_dir,
        tokenizer,
        system.base_model.cfg.seq_len,
        TRAIN_SPLIT,
        source_commit=args.source_commit,
    )
    validate_frozen_training_contract(args, rows, system, base_checkpoint)
    motor_fit = fit_motor_certificate(
        system, seed=args.seed ^ 0xA70C, lr=args.motor_lr,
        max_updates=args.motor_updates,
    )
    reader_fit = fit_reader_certificate(
        system, seed=args.seed ^ 0x18EA, lr=args.reader_lr,
        max_updates=args.reader_updates,
    )
    compiler_fit = fit_compiler(
        system, rows, seed=args.seed, batch_size=args.batch_size,
        epochs=args.epochs, lr=args.compiler_lr, warmup=args.warmup,
        clip=args.clip,
    )
    train_metrics = compiler_train_metrics(system, rows, args.eval_batch_size)
    parameters = system.parameter_report()
    if not motor_fit["exact"] or not reader_fit["exact"]:
        raise RuntimeError("atomic certificates are incomplete")
    output = {
        "schema": CHECKPOINT_SCHEMA,
        "base_sha256": sha256_file(args.base),
        "base_step": base_checkpoint.get("step"),
        "tokenizer_sha256": sha256_file(args.tokenizer),
        "source": source,
        "runtime": runtime,
        "board": board,
        "seeds": {
            "training": args.seed,
            "motor": args.seed ^ 0xA70C,
            "reader": args.seed ^ 0x18EA,
            "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
        },
        "architecture": architecture,
        "parameters": parameters,
        "phases": {
            "motor_atomic_certificate": motor_fit,
            "reader_atomic_certificate": reader_fit,
            "compiler_fields_only": compiler_fit,
        },
        "train_compiler_metrics": train_metrics,
        "state": _adapter_state(system),
        "score_channel": (
            "HardProgramTape + HardLateQuery CPU uint8 categories only; exactly one STOP; "
            "rollout_hard discards motor logits after each argmax"
        ),
        "development_accesses": 0,
        "confirmation_accesses": 0,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, args.out)
    print(json.dumps({
        "saved": str(args.out.resolve()),
        "sha256": sha256_file(args.out),
        "parameters": parameters,
        "motor_certificate": motor_fit,
        "reader_certificate": reader_fit,
    }, sort_keys=True))


def gate_config_main(args: argparse.Namespace) -> None:
    """Freeze development gates without opening development row bytes."""
    if args.out.exists():
        raise SystemExit(f"refusing existing SD-CST gate config: {args.out}")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("schema") != CHECKPOINT_SCHEMA:
        raise SystemExit("unexpected SD-CST checkpoint schema")
    if checkpoint.get("source", {}).get("commit") != args.source_commit:
        raise SystemExit("SD-CST checkpoint source commit mismatch")
    if int(checkpoint.get("confirmation_accesses", -1)) != 0:
        raise SystemExit("SD-CST checkpoint records confirmation access")
    source = source_manifest(args.repo_root.resolve(), args.source_commit)
    if source["sha256"] != checkpoint["source"]["sha256"]:
        raise SystemExit("SD-CST source manifest changed since training")

    report_path = args.data_dir / "report.json"
    report = json.loads(report_path.read_text())
    if (
        report.get("schema") != BOARD_SCHEMA
        or report.get("all_gates_pass") is not True
        or report.get("source_commit") != args.source_commit
        or int(report.get("confirmation_accesses", -1)) != 0
    ):
        raise SystemExit("SD-CST board receipt is not frozen and admitted")
    registration = report.get("development_registration")
    if not isinstance(registration, Mapping):
        raise SystemExit("SD-CST board lacks development registration")
    expected_registration = {
        "protocol": PROTOCOL,
        "row_count": 2304,
        "family_count": 288,
        "family_size": 8,
        "variants": [
            "canonical", "query_swap", "paraphrase", "binding_recode",
            "order_counterfactual", "stop_shift", "storage_order_shuffle",
            "post_halt_suffix",
        ],
    }
    for key, expected in expected_registration.items():
        if registration.get(key) != expected:
            raise SystemExit(f"SD-CST development registration mismatch: {key}")
    depth_counts = {
        str(key): int(value)
        for key, value in dict(registration.get("depth_counts", {})).items()
    }
    if set(depth_counts) != {str(depth) for depth in range(1, 7)}:
        raise SystemExit("SD-CST development registration lacks depths 1..6")
    row_ids_hash = registration.get("row_ids_sha256")
    if not isinstance(row_ids_hash, str) or len(row_ids_hash) != 64:
        raise SystemExit("SD-CST development row registration hash is invalid")

    parameters = checkpoint.get("parameters", {})
    components = {
        "base": int(parameters.get("base", -1)),
        "compiler": int(parameters.get("compiler_added", -1)),
        "motor": int(parameters.get("motor", -1)),
        "reader": int(parameters.get("reader", -1)),
    }
    if sum(components.values()) != FROZEN_PARAMETERS:
        raise SystemExit("SD-CST checkpoint parameter composition changed")
    architecture_sha = sha256_bytes(canonical_json({
        "architecture": checkpoint["architecture"],
        "parameters": components,
    }).encode("utf-8"))
    assessor_path = args.repo_root / "pipeline/assess_sd_cst.py"
    output = {
        "schema": GATE_CONFIG_SCHEMA,
        "expected": {
            "eval_schema": EVALUATION_SCHEMA,
            "protocol": PROTOCOL,
            "split": DEVELOPMENT_SPLIT,
            "row_count": registration["row_count"],
            "family_count": registration["family_count"],
            "family_size": registration["family_size"],
            "row_ids_sha256": row_ids_hash,
            "depth_counts": depth_counts,
            "variants": registration["variants"],
        },
        "thresholds": FROZEN_THRESHOLDS,
        "controls": {
            name: rule | {"min_cases": int(registration["row_count"])}
            for name, rule in FROZEN_CONTROLS.items()
        },
        "expected_artifact_hashes": {
            "architecture_sha256": architecture_sha,
            "board_sha256": sha256_file(report_path),
            "checkpoint_sha256": sha256_file(args.checkpoint),
            "evaluator_sha256": sha256_file(assessor_path),
        },
        "expected_access_ledger_sha256": sha256_bytes(development_ledger_bytes(
            sha256_file(report_path),
            str(report["files"]["development.jsonl"]["sha256"]),
            args.source_commit,
        )),
        "parameter_cap": MAX_SYSTEM_PARAMETERS,
        "confirmation_accesses": 0,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "saved": str(args.out.resolve()),
        "sha256": sha256_file(args.out),
        "development_opened": False,
    }, sort_keys=True))


@torch.no_grad()
def development_main(args: argparse.Namespace) -> None:
    if args.out.exists():
        raise SystemExit(f"refusing existing SD-CST evaluation: {args.out}")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("schema") != CHECKPOINT_SCHEMA:
        raise SystemExit("unexpected SD-CST checkpoint schema")
    if int(checkpoint.get("confirmation_accesses", -1)) != 0:
        raise SystemExit("SD-CST checkpoint records confirmation access")
    if sha256_file(args.base) != checkpoint["base_sha256"]:
        raise SystemExit("SD-CST evaluation base mismatch")
    if sha256_file(args.tokenizer) != checkpoint["tokenizer_sha256"]:
        raise SystemExit("SD-CST evaluation tokenizer mismatch")
    source = source_manifest(args.repo_root.resolve(), args.source_commit)
    if source["sha256"] != checkpoint["source"]["sha256"]:
        raise SystemExit("SD-CST source manifest changed since training")
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    system, _ = load_system(args.base, device, checkpoint["architecture"])
    _load_adapter_state(system, checkpoint["state"])
    system.eval()
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    rows, board = load_rows(
        args.data_dir,
        tokenizer,
        system.base_model.cfg.seq_len,
        DEVELOPMENT_SPLIT,
        source_commit=args.source_commit,
        access_ledger_dir=args.access_ledger_dir,
    )
    records: list[dict[str, object]] = []
    total = {name: 0 for name in ("rows", "state", "answer", "joint", "force_alive_state", "force_alive_answer", "reset_changed", "freeze_changed")}
    tapes: list[HardProgramTape] = []
    queries: list[HardLateQuery] = []
    compiler_exact = {name: 0 for name in ("initial", "kind", "identity", "amount", "query", "whole_tape")}
    for start in range(0, len(rows), args.batch_size):
        batch = rows[start:start + args.batch_size]
        program_ids, program_mask = pad_sources(batch, "program_ids", device)
        query_ids, query_mask = pad_sources(batch, "query_ids", device)
        labels = label_batch(batch, device)
        with _autocast(device):
            soft_program = system.compile_program(program_ids, program_mask)
            soft_query = system.compile_late_query(query_ids, query_mask)
        hard_program, hard_query = seal_hard_payload(soft_program.hard(), soft_query.hard())
        del soft_program, soft_query
        batch_records, batch_total = hard_score_payload(system, hard_program, hard_query, batch)
        records.extend(batch_records)
        for key in total:
            total[key] += batch_total[key]
        active = labels["event_kind_targets"].cpu().ne(STOP_KIND)
        exact = {
            "initial": hard_program.initial_state.long().eq(labels["initial_state_targets"].cpu()),
            "kind": hard_program.event_kind.long().eq(labels["event_kind_targets"].cpu()).all(dim=1),
            "identity": (hard_program.event_identity.long().eq(labels["event_identity_targets"].cpu()) | ~active).all(dim=1),
            "amount": (hard_program.amount.long().eq(labels["amount_targets"].cpu()) | ~active).all(dim=1),
            "query": hard_query.position.long().eq(labels["query_targets"].cpu()),
        }
        exact["whole_tape"] = exact["initial"] & exact["kind"] & exact["identity"] & exact["amount"]
        for key, value in exact.items():
            compiler_exact[key] += int(value.sum())
        for index in range(len(batch)):
            tape, query = _slice_hard(hard_program, hard_query, index)
            tapes.append(tape)
            queries.append(query)
    causal = causal_family_metrics(system, rows, tapes, queries)
    denominator = total["rows"]
    by_variant: dict[str, dict[str, int]] = {}
    by_depth: dict[str, dict[str, int]] = {}
    for record in records:
        for mapping, key in ((by_variant, str(record["variant"])), (by_depth, str(record["halt_after"]))):
            bucket = mapping.setdefault(key, {"rows": 0, "state": 0, "answer": 0, "joint": 0})
            bucket["rows"] += 1
            for metric in ("state", "answer", "joint"):
                bucket[metric] += int(record[f"{metric}_correct"])
    swaps = state_swap_evidence(system, rows, tapes, queries)
    record_by_id = {str(record["id"]): record for record in records}
    output_rows = []
    for index, row in enumerate(rows):
        record = record_by_id[row.row_id]
        interventions: dict[str, object] = {}
        if row.variant == "post_halt_suffix":
            interventions["force_alive"] = {
                "final_state_id": int(record["force_alive_state"]),
                "answer_role": int(system.rollout_hard(
                    tapes[index], queries[index], force_alive=True,
                ).answer_logits.argmax(-1)[0]),
            }
        if row.variant == "canonical":
            interventions["state_swap"] = swaps[row.row_id]
        output_rows.append({
            "id": row.row_id,
            "family_id": row.family_id,
            "variant": row.variant,
            "depth": row.halt_after,
            "compiler_gold": _gold_tape(row),
            "compiler_prediction": _predicted_tape(tapes[index]),
            "late_query_gold": row.query_position,
            "late_query_prediction": int(queries[index].position[0]),
            "oracle": {
                "final_state_id": row.final_state,
                "answer_role": row.answer_role,
            },
            "autonomous": {
                "final_state_id": int(record["predicted_state"]),
                "answer_role": int(record["predicted_answer"]),
            },
            "interventions": interventions,
        })
    parameter_report = system.parameter_report()
    parameter_components = {
        "base": parameter_report["base"],
        "compiler": parameter_report["compiler_added"],
        "motor": parameter_report["motor"],
        "reader": parameter_report["reader"],
    }
    architecture_sha = sha256_bytes(canonical_json({
        "architecture": checkpoint["architecture"],
        "parameters": parameter_components,
    }).encode("utf-8"))
    output = {
        "schema": EVALUATION_SCHEMA,
        "protocol": PROTOCOL,
        "split": DEVELOPMENT_SPLIT,
        "custody": {
            "development_accesses": 1,
            "confirmation_accesses": 0,
            "confirmation_opened": False,
            "access_ledger": board["access_ledger"],
        },
        "artifact_hashes": {
            "architecture_sha256": architecture_sha,
            "board_sha256": board["report_sha256"],
            "checkpoint_sha256": sha256_file(args.checkpoint),
            "evaluator_sha256": sha256_file(
                args.repo_root / "pipeline/assess_sd_cst.py"
            ),
        },
        "parameters": parameter_components | {
            "total": sum(parameter_components.values()),
            "excluded_trainable_parameters": 0,
            "complete_system": True,
        },
        "rows": output_rows,
        "certificates": certificate_evidence(system),
        "controls": control_evidence(system, rows, tapes, queries),
        "source_poison": source_poison_evidence(system, rows, tapes, queries),
        "diagnostics": {
            "source": source,
            "runtime": runtime_manifest(),
            "board": board,
            "score_path": (
                "seal_hard_payload -> CPU uint8 HardProgramTape/HardLateQuery "
                "-> rollout_hard only"
            ),
            "overall": {
                "counts": total,
                "rates": {
                    key: total[key] / denominator for key in total if key != "rows"
                },
            },
            "compiler_exact": {
                "counts": compiler_exact,
                "rates": {
                    key: value / denominator for key, value in compiler_exact.items()
                },
            },
            "by_variant": by_variant,
            "by_depth": by_depth,
            "causal_interventions": causal,
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "saved": str(args.out.resolve()),
        "sha256": sha256_file(args.out),
        "overall": output["diagnostics"]["overall"],
        "causal": causal,
    }, sort_keys=True))


def add_shared(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--cpu", action="store_true")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    train = modes.add_parser("train")
    add_shared(train)
    train.add_argument("--seed", type=int, required=True)
    train.add_argument("--compiler-layer", type=int, default=19)
    train.add_argument("--compiler-width", type=int, default=384)
    train.add_argument("--compiler-heads", type=int, default=8)
    train.add_argument("--compiler-layers", type=int, default=5)
    train.add_argument("--compiler-ff", type=int, default=1408)
    train.add_argument("--motor-hidden", type=int, default=128)
    train.add_argument("--reader-hidden", type=int, default=64)
    train.add_argument("--batch-size", type=int, default=64)
    train.add_argument("--eval-batch-size", type=int, default=128)
    train.add_argument("--epochs", type=int, default=4)
    train.add_argument("--compiler-lr", type=float, default=1e-3)
    train.add_argument("--motor-lr", type=float, default=0.025)
    train.add_argument("--reader-lr", type=float, default=0.04)
    train.add_argument("--motor-updates", type=int, default=2000)
    train.add_argument("--reader-updates", type=int, default=1200)
    train.add_argument("--warmup", type=int, default=100)
    train.add_argument("--clip", type=float, default=1.0)
    train.add_argument("--allow-cpu-train", action="store_true", help=argparse.SUPPRESS)
    development = modes.add_parser("development")
    add_shared(development)
    development.add_argument("--checkpoint", type=Path, required=True)
    development.add_argument("--batch-size", type=int, default=128)
    development.add_argument("--access-ledger-dir", type=Path, required=True)
    gate_config = modes.add_parser("gate-config")
    gate_config.add_argument("--checkpoint", type=Path, required=True)
    gate_config.add_argument("--data-dir", type=Path, required=True)
    gate_config.add_argument("--out", type=Path, required=True)
    gate_config.add_argument("--repo-root", type=Path, required=True)
    gate_config.add_argument("--source-commit", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.mode == "train":
        train_main(args)
    elif args.mode == "development":
        development_main(args)
    else:
        gate_config_main(args)


if __name__ == "__main__":
    main()
