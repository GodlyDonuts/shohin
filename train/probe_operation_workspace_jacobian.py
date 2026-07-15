#!/usr/bin/env python3
"""Frozen Jacobian-aligned operation-selection diagnostic for raw Shohin 260k.

This probe uses token-directed averaged future-logit gradients, not a naive
logit lens and not the full d_model x d_model Jacobian. It reads residuals at
a source anchor before a later operation-report query, then performs one
bounded two-coordinate activation swap with a norm-matched sham.

The model is immutable. The probe generates no tokens, trains no parameters,
and cannot establish reasoning, a global workspace, or consciousness.
"""
from __future__ import annotations

import argparse
import collections
import contextlib
import hashlib
import json
import math
import os
import re
import resource
import time
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import torch
from tokenizers import Tokenizer

from model import GPT, GPTConfig


ROOT = Path(__file__).resolve().parents[1]
AUDIT_NAME = "operation_workspace_jacobian_v1"
EXPECTED_CHECKPOINT_SHA256 = "91d5288f184fc5230516add9851ac1a8815d3369ffd816cd7d0c03d8bafc741d"
EXPECTED_TOKENIZER_SHA256 = "87532df5c121753de3b29194e1f9e3de47986d3f5359548fdf93606773a233d4"
EXPECTED_CHECKPOINT_STEP = 260000
EXPECTED_ARCHITECTURE = {"n_layer": 30, "d_model": 576, "n_loop": 1}

# The canonical source hash replaces only this assignment's value before
# hashing. This lets the module fail closed on edits without a self-hash loop.
SOURCE_FREEZE_SHA256 = "00b50f82a5aa15d6734c5edeee0fa9630b7bb1c8f74efa84b03324bcd557b4a0"
FROZEN_SPEC_SHA256 = "5c807955be3d94cf40f04179bdf31fbefb000765885d409d05448f22d2c9b398"

CANDIDATES = ("add", "multiply", "subtract", "remainder")
CANDIDATE_TOKEN_TEXT = {
    "add": " add",
    "multiply": " multiply",
    "subtract": " subtract",
    "remainder": " remainder",
}
CANDIDATE_TOKEN_IDS = {
    "add": 820,
    "multiply": 4307,
    "subtract": 5498,
    "remainder": 7486,
}
DONOR = {
    "add": "multiply",
    "multiply": "subtract",
    "subtract": "remainder",
    "remainder": "add",
}
SHAM_PAIR = {
    "add": ("subtract", "remainder"),
    "multiply": ("remainder", "add"),
    "subtract": ("add", "multiply"),
    "remainder": ("multiply", "subtract"),
}
PERMUTED_LABEL = {
    "add": "subtract",
    "multiply": "remainder",
    "subtract": "add",
    "remainder": "multiply",
}

SOURCE_LAYERS = (5, 9, 13, 17, 21, 25, 28)
READOUT_LAYERS = (13, 17, 21)
PRIMARY_CAUSAL_LAYER = 17
MAX_RELATIVE_SWAP_L2 = 0.05
MIN_RELATIVE_SWAP_L2 = 1e-4
REALIZED_NORM_MATCH_RTOL = 0.02

DIRECTION_SUFFIX = "\nWhen asked to report it, the operation is"
EVALUATION_SUFFIX = "\nWhich operation comes next?\nThe next operation is"
DIRECTION_TEMPLATES = (
    "A controller records the operation {operation}.\nThe record has been read and retained.",
    "The named state update is {operation}.\nThe update has been placed in memory.",
    "For this process, the chosen operation is {operation}.\nThe choice is now fixed.",
    "A worksheet labels the pending action {operation}.\nThe label has been registered.",
    "The operator card says {operation}.\nThe card has been inspected.",
    "The transition rule selected for later use is {operation}.\nThe rule has been stored.",
)

# (case_id, split, initial state, completed prefix length, ordered operations)
# Every plan contains every candidate exactly once. The current state is
# derived from the completed prefix and is never supplied as a stage index.
CASE_SPECS = (
    ("primary-add-01", "heldout_primary", 7, 1,
     (("multiply", 3), ("add", 8), ("subtract", 5), ("remainder", 11))),
    ("primary-add-02", "heldout_primary", 31, 2,
     (("subtract", 9), ("multiply", 2), ("add", 7), ("remainder", 13))),
    ("primary-add-03", "heldout_primary", 25, 3,
     (("remainder", 7), ("multiply", 5), ("subtract", 3), ("add", 12))),
    ("primary-multiply-01", "heldout_primary", 18, 1,
     (("add", 7), ("multiply", 4), ("subtract", 9), ("remainder", 13))),
    ("primary-multiply-02", "heldout_primary", 40, 2,
     (("subtract", 6), ("add", 5), ("multiply", 3), ("remainder", 17))),
    ("primary-multiply-03", "heldout_primary", 29, 3,
     (("remainder", 8), ("subtract", 2), ("add", 9), ("multiply", 6))),
    ("primary-subtract-01", "heldout_primary", 11, 1,
     (("multiply", 5), ("subtract", 14), ("add", 6), ("remainder", 13))),
    ("primary-subtract-02", "heldout_primary", 27, 2,
     (("add", 8), ("multiply", 3), ("subtract", 10), ("remainder", 19))),
    ("primary-subtract-03", "heldout_primary", 50, 3,
     (("remainder", 17), ("add", 9), ("multiply", 4), ("subtract", 7))),
    ("primary-remainder-01", "heldout_primary", 22, 1,
     (("add", 11), ("remainder", 8), ("multiply", 5), ("subtract", 3))),
    ("primary-remainder-02", "heldout_primary", 35, 2,
     (("multiply", 2), ("subtract", 9), ("remainder", 11), ("add", 6))),
    ("primary-remainder-03", "heldout_primary", 17, 3,
     (("add", 10), ("multiply", 4), ("subtract", 5), ("remainder", 9))),
    ("replication-add-01", "heldout_replication", 9, 1,
     (("multiply", 4), ("add", 13), ("remainder", 11), ("subtract", 2))),
    ("replication-add-02", "heldout_replication", 44, 2,
     (("remainder", 13), ("multiply", 6), ("add", 5), ("subtract", 8))),
    ("replication-add-03", "heldout_replication", 63, 3,
     (("subtract", 12), ("remainder", 10), ("multiply", 7), ("add", 4))),
    ("replication-multiply-01", "heldout_replication", 16, 1,
     (("subtract", 7), ("multiply", 5), ("add", 8), ("remainder", 14))),
    ("replication-multiply-02", "heldout_replication", 28, 2,
     (("remainder", 9), ("add", 6), ("multiply", 5), ("subtract", 3))),
    ("replication-multiply-03", "heldout_replication", 52, 3,
     (("add", 9), ("subtract", 4), ("remainder", 12), ("multiply", 8))),
    ("replication-subtract-01", "heldout_replication", 14, 1,
     (("add", 12), ("subtract", 7), ("multiply", 3), ("remainder", 10))),
    ("replication-subtract-02", "heldout_replication", 33, 2,
     (("multiply", 3), ("remainder", 14), ("subtract", 5), ("add", 11))),
    ("replication-subtract-03", "heldout_replication", 48, 3,
     (("remainder", 13), ("multiply", 5), ("add", 4), ("subtract", 9))),
    ("replication-remainder-01", "heldout_replication", 19, 1,
     (("multiply", 3), ("remainder", 10), ("subtract", 2), ("add", 7))),
    ("replication-remainder-02", "heldout_replication", 37, 2,
     (("subtract", 8), ("add", 6), ("remainder", 9), ("multiply", 4))),
    ("replication-remainder-03", "heldout_replication", 21, 3,
     (("add", 5), ("multiply", 3), ("subtract", 4), ("remainder", 8))),
)

THRESHOLDS = {
    "readout_min_top1_per_12": 8,
    "readout_min_top1_over_permuted": 4,
    "readout_min_mean_margin": 0.10,
    "readout_max_binomial_p": 0.05,
    "absence_max_top1_per_12": 5,
    "absence_max_top1_over_permuted": 1,
    "absence_max_mean_margin": 0.0,
    "output_selection_min_top1_per_12": 8,
    "output_not_selected_max_top1_per_12": 5,
    "causal_min_mean_signal_delta": 0.20,
    "causal_min_mean_signal_minus_sham": 0.10,
    "causal_min_positive_signal_minus_sham_per_12": 10,
    "causal_max_sign_p": 0.05,
}

CLAIM_BOUNDARY = (
    "This frozen diagnostic tests only whether one of four next-operation labels is detectable in "
    "a token-directed future-logit-aligned residual and whether a bounded coordinate swap changes "
    "restricted operation logits. It does not establish reasoning, a global workspace, autonomous "
    "recurrence, semantic transport beyond these labels, or consciousness. Outcome A means not "
    "detected by this preregistered probe, not ontological absence from the model."
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_stream(source) -> str:
    digest = hashlib.sha256()
    for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
        digest.update(chunk)
    return digest.hexdigest()


def sha256_file(path: os.PathLike[str] | str) -> str:
    with open(path, "rb") as source:
        return sha256_stream(source)


def stable_json_sha256(value) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256_bytes(encoded)


def canonical_source_sha256(path: os.PathLike[str] | str = __file__) -> str:
    text = Path(path).read_text()
    pattern = r'^SOURCE_FREEZE_SHA256 = "[^"]+"$'
    canonical, replacements = re.subn(
        pattern,
        'SOURCE_FREEZE_SHA256 = "<CANONICAL_SOURCE_HASH>"',
        text,
        count=1,
        flags=re.MULTILINE,
    )
    if replacements != 1:
        raise RuntimeError("source freeze assignment is missing or ambiguous")
    return sha256_bytes(canonical.encode("utf-8"))


def frozen_spec_payload() -> dict:
    return {
        "audit": AUDIT_NAME,
        "expected_checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "expected_tokenizer_sha256": EXPECTED_TOKENIZER_SHA256,
        "expected_checkpoint_step": EXPECTED_CHECKPOINT_STEP,
        "expected_architecture": EXPECTED_ARCHITECTURE,
        "candidates": list(CANDIDATES),
        "candidate_token_text": CANDIDATE_TOKEN_TEXT,
        "candidate_token_ids": CANDIDATE_TOKEN_IDS,
        "donor": DONOR,
        "sham_pair": {key: list(value) for key, value in SHAM_PAIR.items()},
        "permuted_label": PERMUTED_LABEL,
        "source_layers": list(SOURCE_LAYERS),
        "readout_layers": list(READOUT_LAYERS),
        "primary_causal_layer": PRIMARY_CAUSAL_LAYER,
        "max_relative_swap_l2": MAX_RELATIVE_SWAP_L2,
        "min_relative_swap_l2": MIN_RELATIVE_SWAP_L2,
        "realized_norm_match_rtol": REALIZED_NORM_MATCH_RTOL,
        "direction_suffix": DIRECTION_SUFFIX,
        "evaluation_suffix": EVALUATION_SUFFIX,
        "direction_templates": list(DIRECTION_TEMPLATES),
        "case_specs": [
            [case_id, split, start, completed, [list(operation) for operation in plan]]
            for case_id, split, start, completed, plan in CASE_SPECS
        ],
        "thresholds": THRESHOLDS,
        "claim_boundary": CLAIM_BOUNDARY,
    }


def verify_frozen_contract() -> None:
    observed_source = canonical_source_sha256()
    if observed_source != SOURCE_FREEZE_SHA256:
        raise RuntimeError(
            "probe source is not frozen: expected {}, observed {}".format(
                SOURCE_FREEZE_SHA256, observed_source
            )
        )
    observed_spec = stable_json_sha256(frozen_spec_payload())
    if observed_spec != FROZEN_SPEC_SHA256:
        raise RuntimeError(
            "frozen spec mismatch: expected {}, observed {}".format(
                FROZEN_SPEC_SHA256, observed_spec
            )
        )


def verify_file_hash(path: os.PathLike[str] | str, expected: str, label: str) -> str:
    observed = sha256_file(path)
    if observed != expected:
        raise RuntimeError("{} SHA-256 mismatch: expected {}, observed {}".format(label, expected, observed))
    return observed


def apply_operation(value: int, operation: str, operand: int) -> int:
    if operation == "add":
        return value + operand
    if operation == "multiply":
        return value * operand
    if operation == "subtract":
        return value - operand
    if operation == "remainder":
        if operand <= 0:
            raise ValueError("remainder operand must be positive")
        return value % operand
    raise ValueError("unknown operation: {}".format(operation))


def case_trajectory(start: int, plan: Sequence[Sequence[object]]) -> list[int]:
    values = [int(start)]
    for operation, operand in plan:
        values.append(apply_operation(values[-1], str(operation), int(operand)))
    return values


def validate_case_specs() -> None:
    identifiers = set()
    split_counts = collections.Counter()
    target_counts = collections.Counter()
    rendered_identities = set()
    for case_id, split, start, completed, plan in CASE_SPECS:
        if case_id in identifiers:
            raise ValueError("duplicate case id: {}".format(case_id))
        identifiers.add(case_id)
        if split not in ("heldout_primary", "heldout_replication"):
            raise ValueError("unknown split: {}".format(split))
        if completed not in (1, 2, 3):
            raise ValueError("completed prefix must be 1, 2, or 3")
        operations = tuple(operation for operation, _ in plan)
        if len(plan) != len(CANDIDATES) or set(operations) != set(CANDIDATES):
            raise ValueError("every plan must contain each candidate exactly once")
        if any(not isinstance(operand, int) for _, operand in plan):
            raise ValueError("operands must be integers")
        trajectory = case_trajectory(start, plan)
        if len(set(trajectory)) != len(trajectory):
            raise ValueError("trajectory states must be unique: {}".format(case_id))
        target = operations[completed]
        split_counts[split] += 1
        target_counts[(split, target)] += 1
        identity = (start, completed, tuple(plan))
        if identity in rendered_identities:
            raise ValueError("duplicate case payload")
        rendered_identities.add(identity)
    if split_counts != {"heldout_primary": 12, "heldout_replication": 12}:
        raise ValueError("each held-out board must have 12 cases")
    for split in split_counts:
        for operation in CANDIDATES:
            if target_counts[(split, operation)] != 3:
                raise ValueError("each operation must be correct three times per board")


def operation_clause(operation: str, operand: int) -> str:
    if operation == "add":
        return "add {}".format(operand)
    if operation == "multiply":
        return "multiply by {}".format(operand)
    if operation == "subtract":
        return "subtract {}".format(operand)
    if operation == "remainder":
        return "take the remainder after dividing by {}".format(operand)
    raise ValueError("unknown operation: {}".format(operation))


def render_evaluation_source(start: int, plan, current_state: int) -> str:
    clauses = "; ".join(operation_clause(operation, operand) for operation, operand in plan)
    return (
        "Problem: Start with {}. Follow this plan in order: {}.\n"
        "Current state: {}.\n"
        "The current state was produced after completing a prefix of that plan."
    ).format(start, clauses, current_state)


def encode_anchored(tokenizer: Tokenizer, source: str, suffix: str) -> dict:
    source_ids = tokenizer.encode(source).ids
    full_text = source + suffix
    full_ids = tokenizer.encode(full_text).ids
    if not source_ids or len(full_ids) <= len(source_ids):
        raise ValueError("anchored prompt must have nonempty source and future suffix")
    if full_ids[:len(source_ids)] != source_ids:
        raise ValueError("tokenization crosses the frozen source/suffix boundary")
    return {
        "source": source,
        "suffix": suffix,
        "text": full_text,
        "input_ids": full_ids,
        "anchor_index": len(source_ids) - 1,
        "target_logit_index": len(full_ids) - 1,
        "source_sha256": sha256_bytes(source.encode("utf-8")),
        "prompt_sha256": sha256_bytes(full_text.encode("utf-8")),
        "input_ids_sha256": stable_json_sha256(full_ids),
    }


def validate_tokenizer(tokenizer: Tokenizer) -> None:
    for operation in CANDIDATES:
        observed = tokenizer.encode(CANDIDATE_TOKEN_TEXT[operation]).ids
        expected = [CANDIDATE_TOKEN_IDS[operation]]
        if observed != expected:
            raise ValueError(
                "candidate {} is not the frozen one-token encoding: {} != {}".format(
                    operation, observed, expected
                )
            )
        future = tokenizer.encode(EVALUATION_SUFFIX + CANDIDATE_TOKEN_TEXT[operation]).ids
        if not future or future[-1] != expected[0]:
            raise ValueError("candidate token is not stable after the evaluation suffix")


def build_direction_records(tokenizer: Tokenizer) -> list[dict]:
    records = []
    for operation in CANDIDATES:
        for template_index, template in enumerate(DIRECTION_TEMPLATES):
            source = template.format(operation=operation)
            record = encode_anchored(tokenizer, source, DIRECTION_SUFFIX)
            record.update({
                "id": "direction-{}-{:02d}".format(operation, template_index + 1),
                "operation": operation,
                "token_id": CANDIDATE_TOKEN_IDS[operation],
            })
            records.append(record)
    return records


def build_evaluation_records(tokenizer: Tokenizer) -> list[dict]:
    records = []
    for case_id, split, start, completed, plan in CASE_SPECS:
        trajectory = case_trajectory(start, plan)
        current_state = trajectory[completed]
        correct = plan[completed][0]
        source = render_evaluation_source(start, plan, current_state)
        record = encode_anchored(tokenizer, source, EVALUATION_SUFFIX)
        source_token_counts = {
            operation: record["input_ids"][:record["anchor_index"] + 1].count(token_id)
            for operation, token_id in CANDIDATE_TOKEN_IDS.items()
        }
        if any(count != 1 for count in source_token_counts.values()):
            raise ValueError("candidate lexical exposure is not matched in {}".format(case_id))
        record.update({
            "id": case_id,
            "split": split,
            "start": start,
            "completed_prefix_length": completed,
            "plan": [list(operation) for operation in plan],
            "trajectory": trajectory,
            "current_state": current_state,
            "correct_operation": correct,
            "donor_operation": DONOR[correct],
            "sham_pair": list(SHAM_PAIR[correct]),
            "source_candidate_token_counts": source_token_counts,
        })
        records.append(record)
    return records


def _autocast_context(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def normalized(vector: torch.Tensor) -> torch.Tensor:
    vector = vector.detach().float().cpu()
    norm = vector.norm()
    if not torch.isfinite(norm) or float(norm) <= 1e-12:
        raise RuntimeError("cannot normalize a zero or nonfinite vector")
    return vector / norm


def future_logit_gradients(
    model: GPT,
    input_ids: torch.Tensor,
    *,
    anchor_index: int,
    target_logit_index: int,
    candidate_token_id: int,
    all_candidate_token_ids: Sequence[int],
    layers: Sequence[int],
) -> tuple[dict[int, torch.Tensor], float]:
    """Return future candidate-logit contrast gradients at one source anchor."""
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("input_ids must have shape [1, tokens]")
    if not 0 <= anchor_index < target_logit_index < input_ids.shape[1]:
        raise ValueError("the target logit must be strictly after the source anchor")
    if len(set(all_candidate_token_ids)) != len(all_candidate_token_ids):
        raise ValueError("candidate token ids must be unique")
    if candidate_token_id not in all_candidate_token_ids:
        raise ValueError("candidate token id is outside the candidate set")
    if int(model.cfg.n_loop) != 1:
        raise ValueError("frozen layer indices require n_loop=1")
    ordered_layers = tuple(sorted(set(int(layer) for layer in layers)))
    if not ordered_layers or ordered_layers[0] < 0 or ordered_layers[-1] >= len(model.blocks):
        raise ValueError("invalid source layers")

    captured: dict[int, torch.Tensor] = {}
    handles = []
    first_layer = ordered_layers[0]

    def make_hook(layer: int):
        def hook(_module, _inputs, output):
            hidden, cache = output
            if layer == first_layer:
                hidden = hidden.detach().requires_grad_(True)
            captured[layer] = hidden
            return hidden, cache
        return hook

    for layer in ordered_layers:
        handles.append(model.blocks[layer].register_forward_hook(make_hook(layer)))
    try:
        model.zero_grad(set_to_none=True)
        with torch.enable_grad(), _autocast_context(input_ids.device):
            logits, _ = model(input_ids)
            if set(captured) != set(ordered_layers):
                raise RuntimeError("not every frozen layer was captured")
            others = [token_id for token_id in all_candidate_token_ids if token_id != candidate_token_id]
            contrast = (
                logits[0, target_logit_index, candidate_token_id]
                - logits[0, target_logit_index, others].mean()
            )
            gradients = torch.autograd.grad(
                contrast,
                [captured[layer] for layer in ordered_layers],
                retain_graph=False,
                allow_unused=False,
            )
    finally:
        for handle in handles:
            handle.remove()
    result = {
        layer: gradient[0, anchor_index].detach().float().cpu()
        for layer, gradient in zip(ordered_layers, gradients)
    }
    return result, float(contrast.detach().float().cpu())


def discover_directions(
    model: GPT,
    records: Sequence[Mapping[str, object]],
    device: torch.device,
) -> tuple[dict[int, dict[str, torch.Tensor]], dict]:
    grouped: dict[str, list[Mapping[str, object]]] = collections.defaultdict(list)
    for record in records:
        grouped[str(record["operation"])].append(record)
    all_token_ids = [CANDIDATE_TOKEN_IDS[operation] for operation in CANDIDATES]
    samples = {
        layer: {operation: [] for operation in CANDIDATES}
        for layer in SOURCE_LAYERS
    }
    contrast_values = collections.defaultdict(list)
    for record in records:
        operation = str(record["operation"])
        ids = torch.tensor([record["input_ids"]], dtype=torch.long, device=device)
        gradients, contrast = future_logit_gradients(
            model,
            ids,
            anchor_index=int(record["anchor_index"]),
            target_logit_index=int(record["target_logit_index"]),
            candidate_token_id=CANDIDATE_TOKEN_IDS[operation],
            all_candidate_token_ids=all_token_ids,
            layers=SOURCE_LAYERS,
        )
        contrast_values[operation].append(contrast)
        for layer, gradient in gradients.items():
            samples[layer][operation].append(normalized(gradient))

    directions: dict[int, dict[str, torch.Tensor]] = {}
    diagnostics = {}
    for layer in SOURCE_LAYERS:
        directions[layer] = {}
        diagnostics[str(layer)] = {}
        for operation in CANDIDATES:
            stack = torch.stack(samples[layer][operation])
            direction = normalized(stack.mean(dim=0))
            directions[layer][operation] = direction
            cosines = stack @ direction
            diagnostics[str(layer)][operation] = {
                "contexts": len(stack),
                "mean_unit_gradient_cosine_to_average": float(cosines.mean()),
                "min_unit_gradient_cosine_to_average": float(cosines.min()),
                "mean_clean_future_logit_contrast": sum(contrast_values[operation]) / len(contrast_values[operation]),
            }
    return directions, diagnostics


@torch.inference_mode()
def capture_hidden_and_candidate_logits(
    model: GPT,
    input_ids: torch.Tensor,
    *,
    anchor_index: int,
    target_logit_index: int,
    layers: Sequence[int],
) -> tuple[dict[int, torch.Tensor], torch.Tensor]:
    captured = {}
    handles = []

    def make_hook(layer: int):
        def hook(_module, _inputs, output):
            hidden, _cache = output
            captured[layer] = hidden[0, anchor_index].detach().float().cpu()
        return hook

    for layer in layers:
        handles.append(model.blocks[layer].register_forward_hook(make_hook(int(layer))))
    try:
        with _autocast_context(input_ids.device):
            logits, _ = model(input_ids)
    finally:
        for handle in handles:
            handle.remove()
    if set(captured) != set(layers):
        raise RuntimeError("clean forward did not capture every layer")
    indices = torch.tensor(
        [CANDIDATE_TOKEN_IDS[operation] for operation in CANDIDATES],
        dtype=torch.long,
        device=logits.device,
    )
    candidate_logits = logits[0, target_logit_index].index_select(0, indices).detach().float().cpu()
    return captured, candidate_logits


@torch.inference_mode()
def patched_candidate_logits(
    model: GPT,
    input_ids: torch.Tensor,
    *,
    layer: int,
    anchor_index: int,
    target_logit_index: int,
    delta: torch.Tensor,
) -> tuple[torch.Tensor, dict]:
    if delta.ndim != 1 or delta.shape[0] != model.cfg.d_model:
        raise ValueError("patch delta must have shape [d_model]")

    patch_metrics = {}

    def hook(_module, _inputs, output):
        hidden, cache = output
        clean = hidden[0, anchor_index]
        local_delta = delta.to(device=hidden.device, dtype=hidden.dtype)
        patched = hidden.clone()
        patched[0, anchor_index] = clean + local_delta
        realized = (patched[0, anchor_index] - clean).detach().float()
        clean_norm = float(clean.detach().float().norm().cpu())
        realized_norm = float(realized.norm().cpu())
        patch_metrics.update({
            "clean_hidden_l2": clean_norm,
            "realized_delta_l2": realized_norm,
            "realized_relative_delta_l2": realized_norm / max(clean_norm, 1e-30),
        })
        return patched, cache

    handle = model.blocks[layer].register_forward_hook(hook)
    try:
        with _autocast_context(input_ids.device):
            logits, _ = model(input_ids)
    finally:
        handle.remove()
    indices = torch.tensor(
        [CANDIDATE_TOKEN_IDS[operation] for operation in CANDIDATES],
        dtype=torch.long,
        device=logits.device,
    )
    if not patch_metrics:
        raise RuntimeError("patch hook did not record the realized intervention")
    selected = logits[0, target_logit_index].index_select(0, indices).detach().float().cpu()
    return selected, patch_metrics


def layer_readout_scores(hidden: torch.Tensor, directions: Mapping[str, torch.Tensor]) -> torch.Tensor:
    raw = torch.stack([hidden.float() @ directions[operation].float() for operation in CANDIDATES])
    centered = raw - raw.mean()
    rms = centered.square().mean().sqrt()
    if not torch.isfinite(rms) or float(rms) <= 1e-12:
        return torch.zeros_like(centered)
    return centered / rms


def aggregate_readout_scores(
    hidden_by_layer: Mapping[int, torch.Tensor],
    directions: Mapping[int, Mapping[str, torch.Tensor]],
) -> tuple[torch.Tensor, dict[str, dict[str, float]]]:
    per_layer = {}
    values = []
    for layer in READOUT_LAYERS:
        scores = layer_readout_scores(hidden_by_layer[layer], directions[layer])
        values.append(scores)
        per_layer[str(layer)] = {
            operation: float(scores[index]) for index, operation in enumerate(CANDIDATES)
        }
    return torch.stack(values).mean(dim=0), per_layer


def ordered_rank(scores: torch.Tensor, target: str) -> int:
    values = {operation: float(scores[index]) for index, operation in enumerate(CANDIDATES)}
    ordered = sorted(CANDIDATES, key=lambda operation: (-values[operation], CANDIDATES.index(operation)))
    return ordered.index(target) + 1


def top_candidate(scores: torch.Tensor) -> str:
    return min(CANDIDATES, key=lambda operation: (-float(scores[CANDIDATES.index(operation)]), CANDIDATES.index(operation)))


def target_margin(scores: torch.Tensor, target: str) -> float:
    target_index = CANDIDATES.index(target)
    others = [float(scores[index]) for index in range(len(CANDIDATES)) if index != target_index]
    return float(scores[target_index]) - max(others)


def coordinate_swap_delta(
    hidden: torch.Tensor,
    left_direction: torch.Tensor,
    right_direction: torch.Tensor,
) -> torch.Tensor:
    """Swap two least-squares coordinates while preserving their orthogonal complement."""
    hidden = hidden.detach().float().cpu()
    vectors = torch.stack((left_direction.float().cpu(), right_direction.float().cpu()), dim=1)
    if int(torch.linalg.matrix_rank(vectors)) != 2:
        raise RuntimeError("swap directions are linearly dependent")
    coordinates = torch.linalg.pinv(vectors) @ hidden
    return vectors @ (coordinates[[1, 0]] - coordinates)


def norm_match_and_bound(
    signal_delta: torch.Tensor,
    sham_delta: torch.Tensor,
    hidden: torch.Tensor,
    *,
    max_relative_l2: float = MAX_RELATIVE_SWAP_L2,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    if not 0.0 < max_relative_l2 <= 1.0:
        raise ValueError("max_relative_l2 must be in (0, 1]")
    hidden_norm = float(hidden.float().norm())
    signal_norm = float(signal_delta.float().norm())
    sham_norm = float(sham_delta.float().norm())
    if min(hidden_norm, signal_norm, sham_norm) <= 1e-12:
        raise RuntimeError("cannot norm-match a degenerate activation swap")
    budget = hidden_norm * max_relative_l2
    target_norm = min(signal_norm, sham_norm, budget)
    relative_norm = target_norm / hidden_norm
    if relative_norm < MIN_RELATIVE_SWAP_L2:
        raise RuntimeError("norm-matched swap is below the frozen minimum relative norm")
    signal = signal_delta.float() * (target_norm / signal_norm)
    sham = sham_delta.float() * (target_norm / sham_norm)
    if float(signal.norm()) > budget * (1.0 + 1e-6) or float(sham.norm()) > budget * (1.0 + 1e-6):
        raise RuntimeError("bounded swap exceeded its residual-norm budget")
    return signal, sham, {
        "hidden_l2": hidden_norm,
        "raw_signal_l2": signal_norm,
        "raw_sham_l2": sham_norm,
        "matched_delta_l2": target_norm,
        "relative_delta_l2": relative_norm,
        "max_relative_delta_l2": max_relative_l2,
    }


def candidate_logits_dict(logits: torch.Tensor) -> dict[str, float]:
    return {operation: float(logits[index]) for index, operation in enumerate(CANDIDATES)}


def evaluate_record(
    model: GPT,
    record: Mapping[str, object],
    directions: Mapping[int, Mapping[str, torch.Tensor]],
    device: torch.device,
) -> dict:
    ids = torch.tensor([record["input_ids"]], dtype=torch.long, device=device)
    hidden, baseline_logits = capture_hidden_and_candidate_logits(
        model,
        ids,
        anchor_index=int(record["anchor_index"]),
        target_logit_index=int(record["target_logit_index"]),
        layers=SOURCE_LAYERS,
    )
    aggregate_scores, per_layer_scores = aggregate_readout_scores(hidden, directions)
    correct = str(record["correct_operation"])
    donor = str(record["donor_operation"])
    sham_left, sham_right = (str(value) for value in record["sham_pair"])
    causal_hidden = hidden[PRIMARY_CAUSAL_LAYER]
    raw_signal = coordinate_swap_delta(
        causal_hidden,
        directions[PRIMARY_CAUSAL_LAYER][correct],
        directions[PRIMARY_CAUSAL_LAYER][donor],
    )
    raw_sham = coordinate_swap_delta(
        causal_hidden,
        directions[PRIMARY_CAUSAL_LAYER][sham_left],
        directions[PRIMARY_CAUSAL_LAYER][sham_right],
    )
    signal_delta, sham_delta, bound = norm_match_and_bound(raw_signal, raw_sham, causal_hidden)
    signal_logits, signal_realized = patched_candidate_logits(
        model,
        ids,
        layer=PRIMARY_CAUSAL_LAYER,
        anchor_index=int(record["anchor_index"]),
        target_logit_index=int(record["target_logit_index"]),
        delta=signal_delta,
    )
    sham_logits, sham_realized = patched_candidate_logits(
        model,
        ids,
        layer=PRIMARY_CAUSAL_LAYER,
        anchor_index=int(record["anchor_index"]),
        target_logit_index=int(record["target_logit_index"]),
        delta=sham_delta,
    )
    signal_relative = signal_realized["realized_relative_delta_l2"]
    sham_relative = sham_realized["realized_relative_delta_l2"]
    if max(signal_relative, sham_relative) > MAX_RELATIVE_SWAP_L2 * (1.0 + 1e-6):
        raise RuntimeError("realized activation swap exceeded the frozen L2 bound")
    if min(signal_relative, sham_relative) < MIN_RELATIVE_SWAP_L2:
        raise RuntimeError("realized activation swap fell below the frozen minimum L2")
    norm_ratio = signal_realized["realized_delta_l2"] / sham_realized["realized_delta_l2"]
    if abs(norm_ratio - 1.0) > REALIZED_NORM_MATCH_RTOL:
        raise RuntimeError("realized signal and sham swaps are not norm matched")
    bound.update({
        "realized_signal": signal_realized,
        "realized_sham": sham_realized,
        "realized_signal_to_sham_l2_ratio": norm_ratio,
        "realized_norm_match_rtol": REALIZED_NORM_MATCH_RTOL,
    })
    correct_index = CANDIDATES.index(correct)
    donor_index = CANDIDATES.index(donor)
    baseline_contrast = float(baseline_logits[donor_index] - baseline_logits[correct_index])
    signal_contrast = float(signal_logits[donor_index] - signal_logits[correct_index])
    sham_contrast = float(sham_logits[donor_index] - sham_logits[correct_index])
    signal_effect = signal_contrast - baseline_contrast
    sham_effect = sham_contrast - baseline_contrast
    aggregate_dict = {
        operation: float(aggregate_scores[index]) for index, operation in enumerate(CANDIDATES)
    }
    return {
        "id": record["id"],
        "split": record["split"],
        "source_sha256": record["source_sha256"],
        "prompt_sha256": record["prompt_sha256"],
        "input_ids_sha256": record["input_ids_sha256"],
        "tokens": len(record["input_ids"]),
        "anchor_index": record["anchor_index"],
        "target_logit_index": record["target_logit_index"],
        "start": record["start"],
        "plan": record["plan"],
        "completed_prefix_length": record["completed_prefix_length"],
        "trajectory": record["trajectory"],
        "current_state": record["current_state"],
        "correct_operation": correct,
        "donor_operation": donor,
        "sham_pair": [sham_left, sham_right],
        "source_candidate_token_counts": record["source_candidate_token_counts"],
        "readout": {
            "aggregate_scores": aggregate_dict,
            "per_layer_scores": per_layer_scores,
            "predicted_operation": top_candidate(aggregate_scores),
            "correct_rank": ordered_rank(aggregate_scores, correct),
            "correct_margin_over_best_other": target_margin(aggregate_scores, correct),
            "permuted_control_label": PERMUTED_LABEL[correct],
        },
        "output_selection": {
            "candidate_logits": candidate_logits_dict(baseline_logits),
            "predicted_operation": top_candidate(baseline_logits),
            "correct_rank": ordered_rank(baseline_logits, correct),
            "correct_margin_over_best_other": target_margin(baseline_logits, correct),
        },
        "causal_swap": {
            "layer": PRIMARY_CAUSAL_LAYER,
            "bound": bound,
            "baseline_donor_minus_correct_logit": baseline_contrast,
            "signal_candidate_logits": candidate_logits_dict(signal_logits),
            "sham_candidate_logits": candidate_logits_dict(sham_logits),
            "signal_donor_minus_correct_delta": signal_effect,
            "sham_donor_minus_correct_delta": sham_effect,
            "signal_minus_sham_delta": signal_effect - sham_effect,
        },
    }


def binomial_tail(successes: int, trials: int, probability: float) -> float:
    if not 0 <= successes <= trials or not 0.0 <= probability <= 1.0:
        raise ValueError("invalid binomial-tail arguments")
    return sum(
        math.comb(trials, value)
        * probability ** value
        * (1.0 - probability) ** (trials - value)
        for value in range(successes, trials + 1)
    )


def mean(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        raise ValueError("mean requires values")
    return sum(values) / len(values)


def summarize_split(rows: Sequence[Mapping[str, object]]) -> dict:
    if len(rows) != 12:
        raise ValueError("each independent board must contain exactly 12 rows")
    readout_top1 = sum(row["readout"]["correct_rank"] == 1 for row in rows)
    readout_permuted = sum(
        row["readout"]["predicted_operation"] == row["readout"]["permuted_control_label"]
        for row in rows
    )
    output_top1 = sum(row["output_selection"]["correct_rank"] == 1 for row in rows)
    readout_margin = mean(row["readout"]["correct_margin_over_best_other"] for row in rows)
    signal = [row["causal_swap"]["signal_donor_minus_correct_delta"] for row in rows]
    sham = [row["causal_swap"]["sham_donor_minus_correct_delta"] for row in rows]
    differences = [left - right for left, right in zip(signal, sham)]
    positive = sum(value > 0.0 for value in differences)
    readout_p = binomial_tail(readout_top1, len(rows), 0.25)
    output_p = binomial_tail(output_top1, len(rows), 0.25)
    sign_p = binomial_tail(positive, len(rows), 0.5)
    max_relative = max(row["causal_swap"]["bound"]["relative_delta_l2"] for row in rows)

    readout_gate = (
        readout_top1 >= THRESHOLDS["readout_min_top1_per_12"]
        and readout_top1 - readout_permuted >= THRESHOLDS["readout_min_top1_over_permuted"]
        and readout_margin >= THRESHOLDS["readout_min_mean_margin"]
        and readout_p <= THRESHOLDS["readout_max_binomial_p"]
    )
    absence_gate = (
        readout_top1 <= THRESHOLDS["absence_max_top1_per_12"]
        and readout_top1 - readout_permuted <= THRESHOLDS["absence_max_top1_over_permuted"]
        and readout_margin <= THRESHOLDS["absence_max_mean_margin"]
    )
    output_gate = output_top1 >= THRESHOLDS["output_selection_min_top1_per_12"]
    output_not_selected = output_top1 <= THRESHOLDS["output_not_selected_max_top1_per_12"]
    causal_gate = (
        mean(signal) >= THRESHOLDS["causal_min_mean_signal_delta"]
        and mean(differences) >= THRESHOLDS["causal_min_mean_signal_minus_sham"]
        and positive >= THRESHOLDS["causal_min_positive_signal_minus_sham_per_12"]
        and sign_p <= THRESHOLDS["causal_max_sign_p"]
        and max_relative <= MAX_RELATIVE_SWAP_L2 * (1.0 + 1e-6)
    )
    return {
        "cases": len(rows),
        "readout_top1": readout_top1,
        "readout_permuted_label_top1": readout_permuted,
        "readout_mean_correct_margin": readout_margin,
        "readout_chance_binomial_p": readout_p,
        "output_selection_top1": output_top1,
        "output_selection_chance_binomial_p": output_p,
        "mean_signal_donor_minus_correct_delta": mean(signal),
        "mean_sham_donor_minus_correct_delta": mean(sham),
        "mean_signal_minus_sham_delta": mean(differences),
        "positive_signal_minus_sham": positive,
        "signal_minus_sham_sign_p": sign_p,
        "max_relative_swap_l2": max_relative,
        "readout_gate": readout_gate,
        "absence_gate": absence_gate,
        "output_selection_gate": output_gate,
        "output_not_selected_gate": output_not_selected,
        "causal_swap_gate": causal_gate,
    }


def classify_outcome(board_summaries: Mapping[str, Mapping[str, object]]) -> dict:
    required = ("heldout_primary", "heldout_replication")
    if tuple(sorted(board_summaries)) != tuple(sorted(required)):
        raise ValueError("both frozen held-out boards are required")
    boards = [board_summaries[name] for name in required]
    replicated_readout = all(bool(board["readout_gate"]) for board in boards)
    replicated_absence = all(bool(board["absence_gate"]) for board in boards)
    replicated_output = all(bool(board["output_selection_gate"]) for board in boards)
    replicated_not_selected = all(bool(board["output_not_selected_gate"]) for board in boards)
    replicated_causal = all(bool(board["causal_swap_gate"]) for board in boards)
    any_causal = any(bool(board["causal_swap_gate"]) for board in boards)
    if replicated_causal:
        outcome = "C_BOUNDED_SWAP_CAUSALLY_REDIRECTS_OPERATION"
    elif not any_causal and replicated_readout and replicated_not_selected:
        outcome = "B_JACOBIAN_ALIGNED_OPERATION_PRESENT_NOT_SELECTED"
    elif not any_causal and replicated_absence:
        outcome = "A_OPERATION_NOT_DETECTED_IN_FROZEN_JACOBIAN_ALIGNED_READOUT"
    else:
        outcome = "D_MIXED_OR_INCONCLUSIVE"
    return {
        "outcome": outcome,
        "replicated_readout_gate": replicated_readout,
        "replicated_absence_gate": replicated_absence,
        "replicated_output_selection_gate": replicated_output,
        "replicated_output_not_selected_gate": replicated_not_selected,
        "replicated_causal_swap_gate": replicated_causal,
        "any_board_causal_swap_gate": any_causal,
        "interpretation_boundary": CLAIM_BOUNDARY,
    }


def resolve_device(name: str) -> torch.device:
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        return torch.device("cuda")
    if name == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is unavailable")
        return torch.device("mps")
    if name == "cpu":
        return torch.device("cpu")
    raise ValueError("unknown device: {}".format(name))


def configure_determinism(device: torch.device) -> None:
    torch.manual_seed(20260715)
    torch.use_deterministic_algorithms(True)
    if device.type == "cuda":
        if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
            raise RuntimeError("canonical CUDA execution requires CUBLAS_WORKSPACE_CONFIG=:4096:8")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False


def load_frozen_model(checkpoint_source, device: torch.device) -> tuple[dict, GPT]:
    checkpoint = torch.load(checkpoint_source, map_location="cpu", weights_only=False)
    if checkpoint.get("step") != EXPECTED_CHECKPOINT_STEP:
        raise RuntimeError("checkpoint step is not the frozen raw-260k step")
    config = GPTConfig(**checkpoint["cfg"])
    observed_architecture = {
        "n_layer": int(config.n_layer),
        "d_model": int(config.d_model),
        "n_loop": int(config.n_loop),
    }
    if observed_architecture != EXPECTED_ARCHITECTURE:
        raise RuntimeError("checkpoint architecture does not match the frozen contract")
    model = GPT(config).to(device).eval()
    model.load_state_dict(checkpoint["model"])
    model.requires_grad_(False)
    return checkpoint, model


def source_hashes() -> dict:
    paths = {
        "probe": Path(__file__).resolve(),
        "preregistration": ROOT / "R12_OPERATION_WORKSPACE_JACOBIAN_PREREG.md",
        "tests": ROOT / "train" / "test_probe_operation_workspace_jacobian.py",
        "job_wrapper": ROOT / "train" / "jobs" / "probe_operation_workspace_jacobian.sbatch",
    }
    result = {}
    for label, path in paths.items():
        if not path.is_file():
            raise RuntimeError("missing frozen source file: {}".format(path))
        result[label] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    result["probe"]["canonical_freeze_sha256"] = canonical_source_sha256(paths["probe"])
    return result


def atomic_write_json(path: os.PathLike[str] | str, payload) -> None:
    destination = Path(path)
    if destination.exists():
        raise FileExistsError("refusing to overwrite output: {}".format(destination))
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp.{}".format(os.getpid()))
    try:
        with open(temporary, "x") as sink:
            json.dump(payload, sink, indent=2, sort_keys=True)
            sink.write("\n")
            sink.flush()
            os.fsync(sink.fileno())
        if destination.exists():
            raise FileExistsError("output appeared during write: {}".format(destination))
        os.link(temporary, destination)
        temporary.unlink()
    finally:
        if temporary.exists():
            temporary.unlink()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True, help="immutable raw-260k checkpoint")
    parser.add_argument("--tokenizer", required=True, help="frozen Shohin tokenizer JSON")
    parser.add_argument("--out", required=True, help="fresh result JSON path")
    parser.add_argument("--device", choices=("cuda", "mps", "cpu"), default="cuda")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    started = time.perf_counter()
    verify_frozen_contract()
    validate_case_specs()
    checkpoint_path = Path(args.ckpt).resolve()
    tokenizer_path = Path(args.tokenizer).resolve()
    output_path = Path(args.out).resolve()
    if output_path.exists():
        raise SystemExit("refusing to overwrite output: {}".format(output_path))
    initial_source_records = source_hashes()
    tokenizer_bytes = tokenizer_path.read_bytes()
    observed_tokenizer_sha = sha256_bytes(tokenizer_bytes)
    if observed_tokenizer_sha != EXPECTED_TOKENIZER_SHA256:
        raise RuntimeError("Shohin tokenizer SHA-256 mismatch")
    tokenizer = Tokenizer.from_str(tokenizer_bytes.decode("utf-8"))
    validate_tokenizer(tokenizer)
    direction_records = build_direction_records(tokenizer)
    evaluation_records = build_evaluation_records(tokenizer)
    direction_ids = {record["source_sha256"] for record in direction_records}
    evaluation_ids = {record["source_sha256"] for record in evaluation_records}
    if direction_ids.intersection(evaluation_ids):
        raise RuntimeError("direction and evaluation prompts overlap")
    primary_ids = {
        record["source_sha256"] for record in evaluation_records
        if record["split"] == "heldout_primary"
    }
    replication_ids = {
        record["source_sha256"] for record in evaluation_records
        if record["split"] == "heldout_replication"
    }
    if primary_ids.intersection(replication_ids):
        raise RuntimeError("held-out boards overlap")

    device = resolve_device(args.device)
    configure_determinism(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    with open(checkpoint_path, "rb") as checkpoint_source:
        checkpoint_bytes = os.fstat(checkpoint_source.fileno()).st_size
        observed_checkpoint_sha = sha256_stream(checkpoint_source)
        if observed_checkpoint_sha != EXPECTED_CHECKPOINT_SHA256:
            raise RuntimeError("raw-260k checkpoint SHA-256 mismatch")
        checkpoint_source.seek(0)
        checkpoint, model = load_frozen_model(checkpoint_source, device)
    directions, direction_diagnostics = discover_directions(model, direction_records, device)
    rows = []
    for index, record in enumerate(evaluation_records, 1):
        rows.append(evaluate_record(model, record, directions, device))
        print(
            "[operation-jacobian] evaluated {}/{} {}".format(index, len(evaluation_records), record["id"]),
            flush=True,
        )

    board_summaries = {}
    for split in ("heldout_primary", "heldout_replication"):
        board_summaries[split] = summarize_split([row for row in rows if row["split"] == split])
    decision = classify_outcome(board_summaries)
    canonical_device = device.type == "cuda"
    decision["canonical_execution"] = canonical_device
    if not canonical_device:
        decision["reported_outcome"] = "NONCANONICAL_DEVICE_MECHANICS_ONLY"
    else:
        decision["reported_outcome"] = decision["outcome"]

    elapsed = time.perf_counter() - started
    peak_accelerator = None
    if device.type == "cuda":
        peak_accelerator = int(torch.cuda.max_memory_allocated(device))
    process_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    final_source_records = source_hashes()
    if final_source_records != initial_source_records:
        raise RuntimeError("frozen source files changed during execution")
    result = {
        "audit": AUDIT_NAME,
        "status": "complete",
        "claim_boundary": CLAIM_BOUNDARY,
        "custody": {
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": observed_checkpoint_sha,
            "checkpoint_bytes": checkpoint_bytes,
            "checkpoint_step": checkpoint.get("step"),
            "tokenizer_path": str(tokenizer_path),
            "tokenizer_sha256": observed_tokenizer_sha,
            "tokenizer_bytes": len(tokenizer_bytes),
            "frozen_spec_sha256": FROZEN_SPEC_SHA256,
            "source_freeze_sha256": SOURCE_FREEZE_SHA256,
            "sources": initial_source_records,
        },
        "frozen_protocol": {
            "candidate_token_text": CANDIDATE_TOKEN_TEXT,
            "candidate_token_ids": CANDIDATE_TOKEN_IDS,
            "source_layers": list(SOURCE_LAYERS),
            "readout_layers": list(READOUT_LAYERS),
            "primary_causal_layer": PRIMARY_CAUSAL_LAYER,
            "max_relative_swap_l2": MAX_RELATIVE_SWAP_L2,
            "thresholds": THRESHOLDS,
            "direction_prompt_manifest_sha256": stable_json_sha256([
                {
                    "id": record["id"],
                    "prompt_sha256": record["prompt_sha256"],
                    "input_ids_sha256": record["input_ids_sha256"],
                }
                for record in direction_records
            ]),
            "evaluation_prompt_manifest_sha256": stable_json_sha256([
                {
                    "id": record["id"],
                    "split": record["split"],
                    "prompt_sha256": record["prompt_sha256"],
                    "input_ids_sha256": record["input_ids_sha256"],
                }
                for record in evaluation_records
            ]),
            "direction_contexts": len(direction_records),
            "heldout_cases": len(evaluation_records),
        },
        "direction_diagnostics": direction_diagnostics,
        "cases": rows,
        "board_summaries": board_summaries,
        "decision": decision,
        "resource_ledger": {
            "device": str(device),
            "canonical_device": canonical_device,
            "model_forward_calls": len(direction_records) + 3 * len(evaluation_records),
            "model_backward_calls": len(direction_records),
            "direction_contexts": len(direction_records),
            "clean_evaluation_forwards": len(evaluation_records),
            "signal_patch_forwards": len(evaluation_records),
            "sham_patch_forwards": len(evaluation_records),
            "candidate_logit_reads": len(direction_records) * len(CANDIDATES)
            + 3 * len(evaluation_records) * len(CANDIDATES),
            "generated_tokens": 0,
            "optimizer_steps": 0,
            "trained_parameters": 0,
            "model_parameter_count": model.num_params(),
            "wall_seconds": elapsed,
            "peak_accelerator_allocated_bytes": peak_accelerator,
            "process_ru_maxrss_platform_units": process_rss,
        },
    }
    atomic_write_json(output_path, result)
    print(json.dumps({"decision": decision, "boards": board_summaries}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
