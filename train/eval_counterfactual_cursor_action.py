#!/usr/bin/env python3
"""Score-blind inference for the R12 counterfactual cursor-action canary.

The evaluator is intentionally unable to score its output.  It reads the
independently audited canary, exposes only prompt token IDs and cursor state to
the model, and writes token predictions plus the five preregistered token
logits.  Gold actions and all correctness fields stay exclusively in the
canary and never enter the raw inference artifact or receipt.

The frozen adapter export is::

    {
      "schema": "counterfactual_cursor_action_adapter_v1",
      "arm": "orbit_interchange",
      "adapter_spec": {
        "arm": "orbit_interchange",
        "adapter_type": "centered_three_bit_q_sidecar",
        "parameters": 192,
        "retained_cursor_bits": 3,
        "applies_at_all_tokens": false,
        "layer": "final",
        "head": 0,
        "query_only": true
      },
      "adapter_state": {"projection.weight": "torch.Tensor[64,3]"},
      "bindings": {
        "base_sha256": "<64 lowercase hex>",
        "base_step": 260000,
        "canary_sha256": "<64 lowercase hex>",
        "canary_payload_sha256": "<64 lowercase hex>",
        "audit_sha256": "<64 lowercase hex>",
        "tokenizer_sha256": "<64 lowercase hex>",
        "implementation_commit": "<40 lowercase hex>"
      },
      "training": {"seed": 2026071506, "...": "trainer metadata"},
      "resource_ledger": {"...": "frozen resource accounting"}
    }

The adapter is reconstructed with ``build_adapter(arm, cfg, training["seed"])``
and ``adapter_state`` is loaded strictly.  Inference always runs the complete
model forward path; the final-block training cache is never imported or used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

import torch


RAW_SCHEMA = "counterfactual_cursor_action_raw_inference_v1"
RECEIPT_SCHEMA = "counterfactual_cursor_action_inference_receipt_v1"
ADAPTER_SCHEMA = "counterfactual_cursor_action_adapter_v1"
MANIFEST_SCHEMA = "counterfactual_cursor_action_training_manifest_v1"
CANARY_SCHEMA = "counterfactual_cursor_action_canary_v1"
AUDIT_SCHEMA = "counterfactual_cursor_action_canary_audit_v1"
CANARY_ID = "ccaa-neural-canary-v1"
SHA256_RE = re.compile(r"[0-9a-f]{64}")
ARM_RE = re.compile(r"[a-z0-9][a-z0-9_.-]{0,95}")
COMMIT_RE = re.compile(r"[0-9a-f]{40}")
ARMS = (
    "orbit_interchange",
    "ordinary_loss",
    "relation_sham",
    "source_only",
    "cursor_table",
    "text_cursor_lora",
)
MATCHED_ARMS = ARMS[:4]
FROZEN_BASE_STEP = 260000
FROZEN_UPDATES = 1_152
EXPECTED_PARAMETERS = {
    "orbit_interchange": 192,
    "ordinary_loss": 192,
    "relation_sham": 192,
    "source_only": 192,
    "cursor_table": 512,
    "text_cursor_lora": 640,
}
EXPECTED_RELATION_COEFFICIENTS = {
    "orbit_interchange": 1.0,
    "ordinary_loss": 0.0,
    "relation_sham": 1.0,
    "source_only": 0.0,
    "cursor_table": 0.0,
    "text_cursor_lora": 0.0,
}

ROOT = Path(__file__).resolve().parents[1]
CODE_PATHS = {
    "evaluator": Path(__file__).resolve(),
    "model": ROOT / "train/model.py",
    "cursor_sidecar": ROOT / "train/counterfactual_cursor_action.py",
    "adapter_factory": ROOT / "train/counterfactual_cursor_action_training.py",
}
CODE_IMPLEMENTATION_PATHS = {
    "evaluator": "train/eval_counterfactual_cursor_action.py",
    "model": "train/model.py",
    "cursor_sidecar": "train/counterfactual_cursor_action.py",
    "adapter_factory": "train/counterfactual_cursor_action_training.py",
}
EXPECTED_SEED = 2026071506
EXPECTED_EPOCHS = 4
EXPECTED_EXAMPLES_PER_UPDATE = 60
EXPECTED_TRAINING_EXAMPLES = 69_120
TRAINING_KEYS = {
    "seed", "epochs", "updates", "units_per_epoch", "examples_per_update",
    "optimizer", "learning_rate", "minimum_lr_ratio", "warmup_updates", "betas",
    "epsilon", "weight_decay", "gradient_clip", "cursor_margin", "action_ce_weight",
    "relation_coefficient", "relation_mapping", "relation_pairs_per_update",
    "epoch_history", "elapsed_seconds", "initial_adapter_sha256",
    "final_adapter_sha256", "created_at_utc",
}
RESOURCE_LEDGER_KEYS = {
    "trainable_scalars", "active_trainable_scalars", "inactive_trainable_scalars",
    "base_trainable_scalars", "retained_cursor_bits_selector",
    "retained_phase_bits_selector", "retained_bits_future_one_call", "adapter_dtype",
    "base_autocast_dtype", "source_token_count", "source_token_storage_bytes_int64",
    "padded_cache_token_positions", "pre_final_hidden_cache_bytes",
    "unique_training_cells", "training_examples_with_repetition", "oracle_calls",
    "fixed_training_compute_proxy", "inference_compute_proxy_per_cell",
    "sequential_token_depth", "external_memory", "external_execution",
}

CONDITION_LIBRARY = {
    "canonical": (0, 1, 2, 3, 4),
    "clamped_zero": (0, 0, 0, 0, 0),
    "deranged_cycle": (1, 2, 3, 4, 0),
}

CANARY_TOP_KEYS = {
    "schema",
    "canary_id",
    "contract_sha256",
    "tokenizer_sha256",
    "generator_sha256",
    "implementation_identity",
    "exposure_contract",
    "label_order",
    "label_token_ids",
    "splits",
    "payload_sha256",
}
SPLIT_KEYS = {
    "geometry",
    "sources_sha256",
    "cells_sha256",
    "sources",
    "cells",
    "content_groups",
    "adjacent_pairs",
    "training_units",
}
SOURCE_KEYS = {
    "schema",
    "source_id",
    "split",
    "renderer_id",
    "pack_id",
    "permutation_id",
    "source_text",
    "prompt",
    "prompt_token_ids",
    "operation_order",
    "clause_spans",
}
CELL_KEYS = {
    "schema",
    "cell_id",
    "source_id",
    "cursor",
    "text_prompt",
    "text_prompt_token_ids",
    "target_action",
    "target_index",
    "target_token_id",
}
AUDIT_KEYS = {
    "schema",
    "canary_id",
    "canary_file_sha256",
    "canary_payload_sha256",
    "contract_sha256",
    "tokenizer_sha256",
    "evalgrams_sha256",
    "auditor_sha256",
    "implementation_identity",
    "split_summary",
    "cross_split_13gram_counts",
    "pretraining_corpus_overlap",
    "all_checks_pass",
}
BINDING_KEYS = {
    "canary_file_sha256",
    "canary_payload_sha256",
    "canary_contract_sha256",
    "canary_audit_file_sha256",
    "base_checkpoint_sha256",
    "base_checkpoint_step",
    "adapter_sha256",
    "adapter_implementation_commit",
    "tokenizer_sha256",
    "confirmation_sources_sha256",
    "confirmation_cells_sha256", "training_manifest_sha256",
    "code_sha256",
}
MANIFEST_KEYS = {
    "schema", "arms", "arm_order", "bindings", "all_arms_complete",
    "score_bearing_evaluation_performed",
}
MANIFEST_ARM_KEYS = {
    "arm", "artifact", "artifact_sha256", "initial_adapter_sha256",
    "final_adapter_sha256", "trainable_scalars", "updates",
    "relation_coefficient", "fixed_training_compute_proxy",
}
RAW_FORBIDDEN_KEY_PARTS = ("correct", "accuracy", "score", "target", "gold")
RECEIPT_FORBIDDEN_KEY_PARTS = RAW_FORBIDDEN_KEY_PARTS + (
    "prediction",
    "argmax",
    "logit",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _unique_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _reject_constant(value: str) -> None:
    raise ValueError(f"invalid JSON constant: {value}")


def strict_json_loads(payload: bytes | str) -> Any:
    if isinstance(payload, bytes):
        payload = payload.decode("ascii")
    return json.loads(
        payload,
        object_pairs_hook=_unique_object,
        parse_constant=_reject_constant,
    )


def _absolute(path: str | os.PathLike[str]) -> Path:
    absolute = Path(os.path.abspath(os.fspath(path)))
    # Canonicalize only Apple's fixed system aliases before checking every
    # component.  Arbitrary user-controlled symlinks remain forbidden.
    aliases = {
        Path("/var"): Path("/private/var"),
        Path("/tmp"): Path("/private/tmp"),
        Path("/etc"): Path("/private/etc"),
    }
    for alias, target in aliases.items():
        if (
            alias.is_symlink()
            and alias.resolve() == target
            and (absolute == alias or alias in absolute.parents)
        ):
            return target / absolute.relative_to(alias)
    return absolute


def reject_symlink_components(path: str | os.PathLike[str]) -> Path:
    absolute = _absolute(path)
    current = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        current /= component
        if os.path.lexists(current) and stat.S_ISLNK(current.lstat().st_mode):
            raise ValueError(f"symlink path component is forbidden: {current}")
    return absolute


def read_regular_file(
    path: str | os.PathLike[str], *, require_read_only: bool = True
) -> bytes:
    absolute = reject_symlink_components(path)
    metadata = absolute.lstat()
    require(stat.S_ISREG(metadata.st_mode), f"input is not a regular file: {absolute}")
    if require_read_only:
        require(metadata.st_mode & 0o222 == 0, f"input is writable: {absolute}")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(absolute, flags)
    with os.fdopen(descriptor, "rb") as source:
        return source.read()


def hash_regular_file(
    path: str | os.PathLike[str], *, require_read_only: bool = True
) -> str:
    return sha256_bytes(read_regular_file(path, require_read_only=require_read_only))


def validate_sha256(value: str, label: str) -> str:
    require(isinstance(value, str) and SHA256_RE.fullmatch(value) is not None,
            f"{label} is not a lowercase SHA-256")
    return value


def require_file_hash(
    path: str | os.PathLike[str], expected: str, label: str
) -> str:
    expected = validate_sha256(expected, label)
    actual = hash_regular_file(path)
    require(actual == expected, f"{label} SHA-256 mismatch")
    return actual


def load_strict_json_file(path: str | os.PathLike[str]) -> tuple[dict[str, Any], bytes]:
    payload = read_regular_file(path)
    value = strict_json_loads(payload)
    require(type(value) is dict, f"JSON input is not an object: {path}")
    return value, payload


def _walk_keys(value: Any):
    if isinstance(value, Mapping):
        for key, item in value.items():
            yield key
            yield from _walk_keys(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_keys(item)


def reject_forbidden_keys(value: Any, parts: Sequence[str], label: str) -> None:
    for key in _walk_keys(value):
        lowered = key.lower()
        if any(part in lowered for part in parts):
            raise ValueError(f"{label} contains forbidden key: {key}")


def write_exclusive_read_only_json(
    path: str | os.PathLike[str], value: Mapping[str, Any]
) -> tuple[str, int]:
    destination = reject_symlink_components(path)
    require(not os.path.lexists(destination), f"refusing existing output: {destination}")
    parent = reject_symlink_components(destination.parent)
    require(parent.is_dir(), f"output parent is not a directory: {parent}")
    payload = json.dumps(
        value,
        indent=2,
        sort_keys=True,
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii") + b"\n"
    temporary = parent / f".{destination.name}.{os.getpid()}.tmp"
    require(not os.path.lexists(temporary), f"temporary output already exists: {temporary}")
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(temporary, flags, 0o400)
    try:
        with os.fdopen(descriptor, "wb") as sink:
            sink.write(payload)
            sink.flush()
            os.fsync(sink.fileno())
            os.fchmod(sink.fileno(), 0o444)
        os.link(temporary, destination, follow_symlinks=False)
        os.unlink(temporary)
        directory = os.open(parent, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    metadata = destination.lstat()
    require(stat.S_ISREG(metadata.st_mode), "published output is not a regular file")
    require(metadata.st_mode & 0o222 == 0, "published output remained writable")
    return sha256_bytes(payload), len(payload)


@dataclass(frozen=True)
class BoundCanary:
    document: dict[str, Any]
    audit: dict[str, Any]
    canary_sha256: str
    audit_sha256: str

    @property
    def confirmation(self) -> dict[str, Any]:
        return self.document["splits"]["confirmation"]

    def bindings(
        self,
        *,
        base_sha256: str,
        base_step: int,
        adapter_sha256: str,
        adapter_implementation_commit: str,
        training_manifest_sha256: str,
        code_sha256: Mapping[str, str],
    ) -> dict[str, Any]:
        return {
            "canary_file_sha256": self.canary_sha256,
            "canary_payload_sha256": self.document["payload_sha256"],
            "canary_contract_sha256": self.document["contract_sha256"],
            "canary_audit_file_sha256": self.audit_sha256,
            "base_checkpoint_sha256": base_sha256,
            "base_checkpoint_step": base_step,
            "adapter_sha256": adapter_sha256,
            "adapter_implementation_commit": adapter_implementation_commit,
            "tokenizer_sha256": self.document["tokenizer_sha256"],
            "confirmation_sources_sha256": self.confirmation["sources_sha256"],
            "confirmation_cells_sha256": self.confirmation["cells_sha256"],
            "training_manifest_sha256": training_manifest_sha256,
            "code_sha256": dict(code_sha256),
        }


def load_bound_canary(
    canary_path: str | os.PathLike[str],
    audit_path: str | os.PathLike[str],
    expected_canary_sha256: str,
    expected_audit_sha256: str,
) -> BoundCanary:
    canary, canary_payload = load_strict_json_file(canary_path)
    audit, audit_payload = load_strict_json_file(audit_path)
    canary_sha256 = sha256_bytes(canary_payload)
    audit_sha256 = sha256_bytes(audit_payload)
    require(canary_sha256 == validate_sha256(expected_canary_sha256, "canary"),
            "canary SHA-256 mismatch")
    require(audit_sha256 == validate_sha256(expected_audit_sha256, "audit"),
            "audit SHA-256 mismatch")
    require(set(canary) == CANARY_TOP_KEYS, "canary top-level schema changed")
    require(canary["schema"] == CANARY_SCHEMA, "canary schema mismatch")
    require(canary["canary_id"] == CANARY_ID, "canary ID mismatch")
    expected_payload_hash = sha256_bytes(canonical_json_bytes({
        key: value for key, value in canary.items() if key != "payload_sha256"
    }))
    require(canary["payload_sha256"] == expected_payload_hash,
            "canary payload hash mismatch")
    require(type(canary["splits"]) is dict and "confirmation" in canary["splits"],
            "confirmation split is missing")
    confirmation = canary["splits"]["confirmation"]
    require(type(confirmation) is dict and set(confirmation) == SPLIT_KEYS,
            "confirmation split schema changed")
    require(confirmation["sources_sha256"] == sha256_bytes(
        canonical_json_bytes(confirmation["sources"])),
        "confirmation source hash mismatch")
    require(confirmation["cells_sha256"] == sha256_bytes(
        canonical_json_bytes(confirmation["cells"])),
        "confirmation cell hash mismatch")
    label_token_ids = canary["label_token_ids"]
    require(
        isinstance(label_token_ids, list)
        and len(label_token_ids) == 5
        and all(type(value) is int and value >= 0 for value in label_token_ids)
        and len(set(label_token_ids)) == 5,
        "restricted token contract changed",
    )

    require(set(audit) == AUDIT_KEYS, "audit schema changed")
    require(audit["schema"] == AUDIT_SCHEMA, "audit schema mismatch")
    require(audit["canary_id"] == CANARY_ID, "audit canary ID mismatch")
    require(audit["all_checks_pass"] is True, "independent canary audit did not pass")
    require(audit["canary_file_sha256"] == canary_sha256,
            "audit does not bind the canary file")
    require(audit["canary_payload_sha256"] == canary["payload_sha256"],
            "audit does not bind the canary payload")
    require(audit["contract_sha256"] == canary["contract_sha256"],
            "audit/canary contract binding mismatch")
    require(audit["tokenizer_sha256"] == canary["tokenizer_sha256"],
            "audit/canary tokenizer binding mismatch")
    require(audit["implementation_identity"] == canary["implementation_identity"],
            "audit/canary implementation binding mismatch")
    require(audit["pretraining_corpus_overlap"] == {
        "status": "not_audited_packed_shards_lack_raw_row_boundaries",
        "claim_authorized": False,
        "consequence": "no_pretraining_novelty_or_memorization_exclusion_claim",
    }, "audit pretraining-corpus overlap boundary changed")
    summary = audit["split_summary"].get("confirmation")
    require(type(summary) is dict, "audit lacks confirmation summary")
    require(summary.get("cells") == len(confirmation["cells"]),
            "audit confirmation row count mismatch")
    return BoundCanary(canary, audit, canary_sha256, audit_sha256)


@dataclass(frozen=True)
class InferenceExample:
    cell_id: str
    source_id: str
    semantic_cursor: int
    prompt_token_ids: tuple[int, ...]
    text_prompt_token_ids_by_cursor: tuple[tuple[int, ...], ...]


def _token_tuple(value: Any, label: str) -> tuple[int, ...]:
    require(
        isinstance(value, list)
        and value
        and all(type(token) is int and token >= 0 for token in value),
        f"{label} is not a nonempty token-id list",
    )
    return tuple(value)


def build_inference_examples(bound: BoundCanary) -> list[InferenceExample]:
    """Construct model inputs without reading any gold field value."""
    confirmation = bound.confirmation
    source_tokens: dict[str, tuple[int, ...]] = {}
    for source in confirmation["sources"]:
        require(type(source) is dict and set(source) == SOURCE_KEYS,
                "confirmation source schema changed")
        source_id = source["source_id"]
        require(isinstance(source_id, str) and source_id not in source_tokens,
                "invalid or duplicate source ID")
        source_tokens[source_id] = _token_tuple(
            source["prompt_token_ids"], f"prompt tokens for {source_id}"
        )

    text_tokens: dict[tuple[str, int], tuple[int, ...]] = {}
    cell_identity: list[tuple[str, str, int]] = []
    observed_cells: set[str] = set()
    for cell in confirmation["cells"]:
        require(type(cell) is dict and set(cell) == CELL_KEYS,
                "confirmation cell schema changed")
        cell_id = cell["cell_id"]
        source_id = cell["source_id"]
        cursor = cell["cursor"]
        require(isinstance(cell_id, str) and cell_id not in observed_cells,
                "invalid or duplicate cell ID")
        require(source_id in source_tokens, "cell references an unknown source")
        require(type(cursor) is int and 0 <= cursor < 5, "cell cursor is invalid")
        require(cell_id == f"{source_id}-c{cursor}", "cell identity mismatch")
        text_tokens[(source_id, cursor)] = _token_tuple(
            cell["text_prompt_token_ids"], f"text prompt tokens for {cell_id}"
        )
        observed_cells.add(cell_id)
        cell_identity.append((cell_id, source_id, cursor))

    examples = []
    for cell_id, source_id, cursor in cell_identity:
        variants = tuple(text_tokens[(source_id, index)] for index in range(5))
        examples.append(InferenceExample(
            cell_id=cell_id,
            source_id=source_id,
            semantic_cursor=cursor,
            prompt_token_ids=source_tokens[source_id],
            text_prompt_token_ids_by_cursor=variants,
        ))
    require(len(examples) == len(confirmation["cells"]), "inference row count drift")
    return examples


class InferenceRuntime(Protocol):
    arm_name: str
    adapter_contract: Mapping[str, Any]

    def forward_logits(
        self, examples: Sequence[InferenceExample], effective_cursors: Sequence[int]
    ) -> torch.Tensor:
        """Return CPU or device logits with shape ``[batch, vocab]``."""


class TorchAdapterRuntime:
    def __init__(
        self,
        *,
        base: torch.nn.Module,
        adapter: torch.nn.Module,
        arm: str,
        adapter_spec: Mapping[str, Any],
        device: str,
        base_step: int,
        implementation_commit: str,
    ) -> None:
        self.base = base
        self.adapter = adapter
        self.arm_name = arm
        self.arm = arm
        self.device = device
        self.base_step = base_step
        self.implementation_commit = implementation_commit
        self.adapter_contract = dict(adapter_spec)

    @torch.inference_mode()
    def forward_logits(
        self, examples: Sequence[InferenceExample], effective_cursors: Sequence[int]
    ) -> torch.Tensor:
        require(len(examples) == len(effective_cursors) and bool(examples),
                "runtime batch shape mismatch")
        cursors = [0 if self.arm == "source_only" else value
                   for value in effective_cursors]
        require(all(type(value) is int and 0 <= value < 5 for value in cursors),
                "runtime cursor is invalid")
        if self.arm == "text_cursor_lora":
            token_rows = [
                example.text_prompt_token_ids_by_cursor[cursor]
                for example, cursor in zip(examples, cursors, strict=True)
            ]
        else:
            token_rows = [example.prompt_token_ids for example in examples]
        positions = torch.tensor(
            [len(tokens) - 1 for tokens in token_rows], dtype=torch.long, device=self.device
        )
        width = max(len(tokens) for tokens in token_rows)
        idx = torch.zeros((len(token_rows), width), dtype=torch.long, device=self.device)
        for row_index, tokens in enumerate(token_rows):
            idx[row_index, :len(tokens)] = torch.tensor(
                tokens, dtype=torch.long, device=self.device
            )

        with torch.autocast(
            "cuda", dtype=torch.bfloat16, enabled=self.device.startswith("cuda"),
        ):
            if self.arm != "text_cursor_lora":
                from counterfactual_cursor_action import selector_position_grid

                cursor_tensor = torch.tensor(cursors, dtype=torch.long, device=self.device)
                grid, mask = selector_position_grid(cursor_tensor, positions, width)
                q_delta = self.adapter(grid, mask)
                logits, _ = self.base(
                    idx,
                    q_delta=q_delta,
                    q_delta_layer=-1,
                    q_delta_head=0,
                )
            else:
                logits, _ = self.base(
                    idx,
                    q_adapter=self.adapter,
                    q_delta_layer=-1,
                    q_delta_head=0,
                )
        require(logits.ndim == 3 and logits.shape[:2] == idx.shape,
                "model returned the wrong logit shape")
        batch_indices = torch.arange(len(token_rows), device=self.device)
        return logits[batch_indices, positions].float().detach().cpu()


def _adapter_binding_contract(
    bound: BoundCanary, base_sha256: str, base_step: int,
    implementation_commit: str,
) -> dict[str, Any]:
    return {
        "base_sha256": base_sha256,
        "base_step": base_step,
        "canary_sha256": bound.canary_sha256,
        "canary_payload_sha256": bound.document["payload_sha256"],
        "audit_sha256": bound.audit_sha256,
        "tokenizer_sha256": bound.document["tokenizer_sha256"],
        "implementation_commit": implementation_commit,
    }


@dataclass(frozen=True)
class TrainingManifest:
    document: dict[str, Any]
    sha256: str
    path: Path
    entries: Mapping[str, Mapping[str, Any]]
    artifact_paths: Mapping[str, Path]


def load_training_manifest(
    manifest_path: str | os.PathLike[str],
    *,
    expected_sha256: str,
    bound: BoundCanary,
    base_sha256: str,
) -> TrainingManifest:
    """Validate the completed matched run and every arm artifact by hash."""
    absolute_manifest = reject_symlink_components(manifest_path)
    document, payload = load_strict_json_file(absolute_manifest)
    observed_sha256 = sha256_bytes(payload)
    require(
        observed_sha256 == validate_sha256(expected_sha256, "training manifest"),
        "training manifest SHA-256 mismatch",
    )
    require(type(document) is dict and set(document) == MANIFEST_KEYS,
            "training manifest schema changed")
    require(document["schema"] == MANIFEST_SCHEMA,
            "training manifest schema mismatch")
    require(document["arm_order"] == list(ARMS),
            "training manifest arm order changed")
    require(document["all_arms_complete"] is True,
            "training manifest is incomplete")
    require(document["score_bearing_evaluation_performed"] is False,
            "training manifest claims prior score-bearing evaluation")

    identity = bound.document.get("implementation_identity")
    require(type(identity) is dict and set(identity) == {"git_commit", "file_sha256"},
            "canary implementation identity is missing")
    implementation_commit = identity["git_commit"]
    require(isinstance(implementation_commit, str)
            and COMMIT_RE.fullmatch(implementation_commit) is not None,
            "canary implementation commit is invalid")
    expected_bindings = _adapter_binding_contract(
        bound, base_sha256, FROZEN_BASE_STEP, implementation_commit,
    )
    require(document["bindings"] == expected_bindings,
            "training manifest immutable bindings mismatch")

    arms = document["arms"]
    require(isinstance(arms, list) and len(arms) == len(ARMS),
            "training manifest arm count mismatch")
    entries: dict[str, Mapping[str, Any]] = {}
    artifact_paths: dict[str, Path] = {}
    for expected_arm, entry in zip(ARMS, arms, strict=True):
        require(type(entry) is dict and set(entry) == MANIFEST_ARM_KEYS,
                "training manifest arm schema changed")
        arm = entry["arm"]
        require(arm == expected_arm and arm not in entries,
                "training manifest arm identity mismatch")
        require(entry["artifact"] == f"{arm}/adapter.pt",
                "training manifest artifact path changed")
        require(entry["trainable_scalars"] == EXPECTED_PARAMETERS[arm],
                "training manifest parameter count mismatch")
        require(entry["updates"] == FROZEN_UPDATES,
                "training manifest update count mismatch")
        require(entry["relation_coefficient"] == EXPECTED_RELATION_COEFFICIENTS[arm],
                "training manifest relation coefficient mismatch")
        require(type(entry["fixed_training_compute_proxy"]) is dict,
                "training manifest compute proxy is malformed")
        for key in (
            "artifact_sha256", "initial_adapter_sha256", "final_adapter_sha256",
        ):
            validate_sha256(entry[key], f"training manifest {arm} {key}")
        artifact = reject_symlink_components(
            absolute_manifest.parent / entry["artifact"]
        )
        require(artifact.parent.parent == absolute_manifest.parent,
                "training manifest artifact escaped its output root")
        require(hash_regular_file(artifact) == entry["artifact_sha256"],
                "training manifest arm artifact hash mismatch")
        entries[arm] = entry
        artifact_paths[arm] = artifact

    matched_initial = {entries[arm]["initial_adapter_sha256"] for arm in MATCHED_ARMS}
    require(len(matched_initial) == 1,
            "information-matched arms did not share one adapter initialization")
    matched_compute = [entries[arm]["fixed_training_compute_proxy"] for arm in MATCHED_ARMS]
    require(all(proxy == matched_compute[0] for proxy in matched_compute[1:]),
            "information-matched arm compute ledgers differ")
    return TrainingManifest(
        document=document,
        sha256=observed_sha256,
        path=absolute_manifest,
        entries=entries,
        artifact_paths=artifact_paths,
    )


def hash_adapter_state(state: Mapping[str, torch.Tensor]) -> str:
    """Hash a tensor state independently of the trainer's implementation."""
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("ascii") + b"\0")
        digest.update(str(tensor.dtype).encode("ascii") + b"\0")
        digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii"))
        digest.update(b"\0" + tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def validate_adapter_provenance(
    *,
    arm: str,
    adapter_spec: Mapping[str, Any],
    state_dict: Mapping[str, torch.Tensor],
    training: Mapping[str, Any],
    resource_ledger: Mapping[str, Any],
    initialized_adapter: torch.nn.Module,
    manifest_entry: Mapping[str, Any],
) -> None:
    """Reconcile serialized adapter internals with the frozen run contract."""
    require(set(training) == TRAINING_KEYS, "adapter training ledger schema changed")
    require(set(resource_ledger) == RESOURCE_LEDGER_KEYS,
            "adapter resource ledger schema changed")
    expected_training = {
        "seed": EXPECTED_SEED,
        "epochs": EXPECTED_EPOCHS,
        "updates": FROZEN_UPDATES,
        "units_per_epoch": 288,
        "examples_per_update": EXPECTED_EXAMPLES_PER_UPDATE,
        "optimizer": "AdamW",
        "learning_rate": 0.01,
        "minimum_lr_ratio": 0.1,
        "warmup_updates": 50,
        "betas": [0.9, 0.95],
        "epsilon": 1e-8,
        "weight_decay": 0.0,
        "gradient_clip": 1.0,
        "cursor_margin": 1.0,
        "action_ce_weight": 1.0,
        "relation_coefficient": EXPECTED_RELATION_COEFFICIENTS[arm],
        "relation_mapping": "deranged" if arm == "relation_sham" else "true",
    }
    for key, expected in expected_training.items():
        require(training[key] == expected, f"adapter frozen training field changed: {key}")
    history = training["epoch_history"]
    require(isinstance(history, list) and len(history) == EXPECTED_EPOCHS,
            "adapter epoch history length changed")
    require(all(type(item) is dict and item.get("epoch") == index + 1
                and item.get("updates") == 288
                for index, item in enumerate(history)),
            "adapter epoch history accounting mismatch")
    require(type(training["elapsed_seconds"]) in (int, float)
            and training["elapsed_seconds"] > 0,
            "adapter elapsed time is invalid")
    require(isinstance(training["created_at_utc"], str) and training["created_at_utc"],
            "adapter creation timestamp is invalid")
    initial_hash = hash_adapter_state(initialized_adapter.state_dict())
    final_hash = hash_adapter_state(state_dict)
    require(initial_hash == training["initial_adapter_sha256"]
            == manifest_entry["initial_adapter_sha256"],
            "adapter initial-state provenance mismatch")
    require(final_hash == training["final_adapter_sha256"]
            == manifest_entry["final_adapter_sha256"],
            "adapter final-state provenance mismatch")
    require(final_hash != initial_hash, "adapter state did not change during training")
    require(adapter_spec.get("parameters") == EXPECTED_PARAMETERS[arm],
            "adapter spec parameter count changed")
    require(resource_ledger["trainable_scalars"] == EXPECTED_PARAMETERS[arm],
            "adapter resource parameter count mismatch")
    require(resource_ledger["base_trainable_scalars"] == 0,
            "adapter resource ledger claims trainable base weights")
    require(resource_ledger["unique_training_cells"] == 5_760,
            "adapter unique training cell count changed")
    require(resource_ledger["training_examples_with_repetition"] == EXPECTED_TRAINING_EXAMPLES,
            "adapter repeated training example count changed")
    require(resource_ledger["oracle_calls"] == 0
            and resource_ledger["external_execution"] == 0,
            "adapter resource ledger contains external assistance")
    require(resource_ledger["fixed_training_compute_proxy"]
            == manifest_entry["fixed_training_compute_proxy"],
            "adapter compute ledger differs from training manifest")


def load_model_and_adapter(
    base_path: str | os.PathLike[str],
    adapter_path: str | os.PathLike[str],
    *,
    device: str,
    bound: BoundCanary,
    base_sha256: str,
    adapter_sha256: str,
    training_manifest: TrainingManifest,
) -> TorchAdapterRuntime:
    """Load the full base and one frozen adapter export.

    The adapter is reconstructed through the shared training factory and its
    state is loaded strictly.  The returned runtime always calls ``GPT.forward``
    from token IDs; it never imports or consumes the training prefix cache.
    """
    from counterfactual_cursor_action_training import (
        adapter_state_payload,
        build_adapter,
    )
    from model import GPT, GPTConfig

    adapter_payload = torch.load(adapter_path, map_location="cpu", weights_only=False)
    require(type(adapter_payload) is dict and set(adapter_payload) == {
        "schema", "arm", "adapter_spec", "adapter_state", "bindings",
        "training", "resource_ledger",
    }, "adapter export schema changed")
    require(adapter_payload["schema"] == ADAPTER_SCHEMA, "adapter export schema mismatch")
    arm = adapter_payload["arm"]
    require(arm in ARMS, "adapter arm is invalid")
    require(isinstance(arm, str) and ARM_RE.fullmatch(arm) is not None,
            "adapter arm name is malformed")
    require(_absolute(adapter_path) == training_manifest.artifact_paths[arm],
            "selected adapter path is not the manifest-bound artifact")
    require(training_manifest.entries[arm]["artifact_sha256"] == adapter_sha256,
            "selected adapter hash is not the manifest-bound artifact")
    adapter_spec = adapter_payload["adapter_spec"]
    state_dict = adapter_payload["adapter_state"]
    training = adapter_payload["training"]
    resource_ledger = adapter_payload["resource_ledger"]
    bindings = adapter_payload["bindings"]
    require(type(adapter_spec) is dict and isinstance(state_dict, Mapping),
            "adapter spec/state must be objects")
    require(type(training) is dict and type(resource_ledger) is dict,
            "adapter training/resource ledgers must be objects")
    seed = training.get("seed")
    require(type(seed) is int and seed >= 0, "adapter training seed is missing or invalid")
    require(type(bindings) is dict and set(bindings) == {
        "base_sha256", "base_step", "canary_sha256", "canary_payload_sha256",
        "audit_sha256", "tokenizer_sha256", "implementation_commit",
    }, "adapter binding schema changed")
    implementation_commit = bindings["implementation_commit"]
    require(isinstance(implementation_commit, str)
            and COMMIT_RE.fullmatch(implementation_commit) is not None,
            "adapter implementation commit is invalid")
    base_step = bindings["base_step"]
    require(type(base_step) is int and base_step >= 0, "adapter base step is invalid")
    require(bindings == _adapter_binding_contract(
        bound, base_sha256, base_step, implementation_commit,
    ), "adapter immutable bindings mismatch")
    require(all(isinstance(key, str) and isinstance(value, torch.Tensor)
                for key, value in state_dict.items()),
            "adapter state contains non-tensor entries")
    require(all(bool(torch.isfinite(value).all()) for value in state_dict.values()),
            "adapter state contains non-finite tensors")

    base_payload = torch.load(
        base_path, map_location="cpu", weights_only=False, mmap=True,
    )
    require(type(base_payload) is dict and type(base_payload.get("cfg")) is dict,
            "base checkpoint metadata is invalid")
    require(isinstance(base_payload.get("model"), Mapping),
            "base checkpoint lacks model state")
    require(type(base_payload.get("step")) is int and base_payload["step"] == base_step,
            "base checkpoint step binding mismatch")
    base = GPT(GPTConfig(**base_payload["cfg"]))
    base.load_state_dict(base_payload["model"], strict=True)
    base.requires_grad_(False)
    base = base.to(device).eval()
    adapter, spec = build_adapter(arm, base.cfg, seed)
    expected_spec = adapter_state_payload(adapter, spec)["adapter_spec"]
    require(adapter_spec == expected_spec, "adapter spec does not match shared factory")
    validate_adapter_provenance(
        arm=arm,
        adapter_spec=adapter_spec,
        state_dict=state_dict,
        training=training,
        resource_ledger=resource_ledger,
        initialized_adapter=adapter,
        manifest_entry=training_manifest.entries[arm],
    )
    adapter.load_state_dict(state_dict, strict=True)
    observed_parameters = sum(parameter.numel() for parameter in adapter.parameters())
    require(observed_parameters == adapter_spec["parameters"],
            "adapter parameter count mismatch")
    adapter.requires_grad_(False)
    adapter = adapter.to(device).eval()
    del base_payload, adapter_payload
    return TorchAdapterRuntime(
        base=base,
        adapter=adapter,
        arm=arm,
        adapter_spec=adapter_spec,
        device=device,
        base_step=base_step,
        implementation_commit=implementation_commit,
    )


def parse_conditions(value: str) -> list[dict[str, Any]]:
    names = value.split(",")
    require(bool(names) and all(names), "condition list is empty or malformed")
    require(len(names) == len(set(names)), "condition list contains duplicates")
    require(names[0] == "canonical", "canonical must be the first condition")
    result = []
    for name in names:
        require(name in CONDITION_LIBRARY, f"unknown inference condition: {name}")
        result.append({"name": name, "cursor_map": list(CONDITION_LIBRARY[name])})
    return result


def _finite_logits(logits: torch.Tensor) -> torch.Tensor:
    require(logits.ndim == 1 and logits.numel() > 0, "logit row is empty")
    require(bool(torch.isfinite(logits).all()), "model returned non-finite logits")
    return logits.float().cpu()


def prediction_record(logits: torch.Tensor, restricted_token_ids: Sequence[int]) -> dict[str, Any]:
    logits = _finite_logits(logits)
    require(max(restricted_token_ids) < logits.numel(), "restricted token is outside vocabulary")
    full_max = torch.max(logits)
    full_top_count = int(logits.eq(full_max).sum().item())
    full_argmax = int(torch.argmax(logits).item())
    restricted = logits.index_select(0, torch.tensor(restricted_token_ids, dtype=torch.long))
    restricted_max = torch.max(restricted)
    restricted_top_count = int(restricted.eq(restricted_max).sum().item())
    restricted_argmax = int(torch.argmax(restricted).item())
    full_unique = full_top_count == 1
    restricted_unique = restricted_top_count == 1
    return {
        "full_vocab_argmax_token_id": full_argmax,
        "full_vocab_argmax_logit": float(full_max.item()),
        "full_vocab_top_count": full_top_count,
        "full_vocab_unique_top1": full_unique,
        "full_vocab_prediction_token_id": full_argmax if full_unique else None,
        "restricted_argmax_index": restricted_argmax,
        "restricted_argmax_token_id": int(restricted_token_ids[restricted_argmax]),
        "restricted_top_count": restricted_top_count,
        "restricted_unique_top1": restricted_unique,
        "restricted_prediction_token_id": (
            int(restricted_token_ids[restricted_argmax]) if restricted_unique else None
        ),
        "restricted_logits": [float(value) for value in restricted.tolist()],
    }


def evaluate_examples(
    examples: Sequence[InferenceExample],
    runtime: InferenceRuntime,
    restricted_token_ids: Sequence[int],
    conditions: Sequence[Mapping[str, Any]],
    *,
    batch_size: int,
) -> tuple[list[dict[str, Any]], int]:
    require(batch_size > 0, "batch size must be positive")
    require(bool(examples), "inference example set is empty")
    records = [{"cell_id": example.cell_id, "conditions": {}} for example in examples]
    forward_count = 0
    for condition in conditions:
        name = condition["name"]
        cursor_map = condition["cursor_map"]
        require(name in CONDITION_LIBRARY and tuple(cursor_map) == CONDITION_LIBRARY[name],
                "condition contract drift")
        for start in range(0, len(examples), batch_size):
            batch = examples[start:start + batch_size]
            effective = [cursor_map[item.semantic_cursor] for item in batch]
            logits = runtime.forward_logits(batch, effective)
            forward_count += 1
            require(
                isinstance(logits, torch.Tensor)
                and logits.ndim == 2
                and logits.shape[0] == len(batch),
                "runtime returned the wrong batch logit shape",
            )
            for offset, row_logits in enumerate(logits):
                records[start + offset]["conditions"][name] = prediction_record(
                    row_logits, restricted_token_ids
                )
    expected_names = [condition["name"] for condition in conditions]
    require(all(list(record["conditions"]) == expected_names for record in records),
            "inference condition coverage drift")
    return records, forward_count


def build_raw_artifact(
    *,
    runtime: InferenceRuntime,
    bindings: Mapping[str, Any],
    restricted_token_ids: Sequence[int],
    conditions: Sequence[Mapping[str, Any]],
    records: list[dict[str, Any]],
    forward_count: int,
) -> dict[str, Any]:
    require(set(bindings) == BINDING_KEYS, "raw binding schema changed")
    raw = {
        "schema": RAW_SCHEMA,
        "arm_name": runtime.arm_name,
        "adapter_contract": dict(runtime.adapter_contract),
        "bindings": dict(bindings),
        "condition_contract": [dict(condition) for condition in conditions],
        "restricted_token_ids": list(restricted_token_ids),
        "row_count": len(records),
        "condition_count": len(conditions),
        "forward_count": forward_count,
        "inference_records": records,
    }
    reject_forbidden_keys(raw, RAW_FORBIDDEN_KEY_PARTS, "raw inference artifact")
    return raw


def build_receipt(
    *,
    job_identity: Mapping[str, str],
    arm_name: str,
    bindings: Mapping[str, Any],
    row_count: int,
    condition_count: int,
    forward_count: int,
    raw_sha256: str,
    raw_bytes: int,
) -> dict[str, Any]:
    require(set(job_identity) == {"scheduler", "job_id", "array_task_id", "attempt_id"},
            "job identity schema changed")
    require(all(isinstance(value, str) and value for value in job_identity.values()),
            "job identity contains an empty field")
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "job_identity": dict(job_identity),
        "arm_name": arm_name,
        "bindings": dict(bindings),
        "row_count": row_count,
        "condition_count": condition_count,
        "forward_count": forward_count,
        "raw_result_sha256": validate_sha256(raw_sha256, "raw result"),
        "raw_result_bytes": raw_bytes,
    }
    reject_forbidden_keys(receipt, RECEIPT_FORBIDDEN_KEY_PARTS, "inference receipt")
    return receipt


def code_hashes() -> dict[str, str]:
    return {
        name: hash_regular_file(path, require_read_only=False)
        for name, path in CODE_PATHS.items()
    }


def verify_code_identity(bound: BoundCanary, observed: Mapping[str, str]) -> None:
    """Require every score-bearing live code file to equal the frozen commit ledger."""
    identity = bound.document.get("implementation_identity")
    require(type(identity) is dict and set(identity) == {"git_commit", "file_sha256"},
            "canary implementation identity is missing")
    ledger = identity["file_sha256"]
    require(type(ledger) is dict, "canary implementation ledger is malformed")
    expected = {
        name: ledger.get(relative)
        for name, relative in CODE_IMPLEMENTATION_PATHS.items()
    }
    require(all(isinstance(value, str) and SHA256_RE.fullmatch(value) is not None
                for value in expected.values()),
            "canary implementation ledger lacks evaluator dependencies")
    require(dict(observed) == expected,
            "live score-bearing evaluator code differs from frozen implementation")


def _bound_input_hashes(paths: Mapping[str, str | os.PathLike[str]]) -> dict[str, str]:
    return {name: hash_regular_file(path) for name, path in paths.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--base-sha256", required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--adapter-sha256", required=True)
    parser.add_argument("--training-manifest", type=Path, required=True)
    parser.add_argument("--training-manifest-sha256", required=True)
    parser.add_argument("--canary", type=Path, required=True)
    parser.add_argument("--canary-sha256", required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--audit-sha256", required=True)
    parser.add_argument("--out-raw", type=Path, required=True)
    parser.add_argument("--out-receipt", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--conditions",
        default="canonical,clamped_zero,deranged_cycle",
        help="comma-separated frozen condition names; canonical must be first",
    )
    parser.add_argument("--scheduler", default="slurm")
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--array-task-id", default="none")
    parser.add_argument("--attempt-id", default="1")
    arguments = parser.parse_args()

    require(_absolute(arguments.out_raw) != _absolute(arguments.out_receipt),
            "raw and receipt outputs must be distinct")
    for output in (arguments.out_raw, arguments.out_receipt):
        destination = reject_symlink_components(output)
        require(not os.path.lexists(destination), f"refusing existing output: {destination}")

    expected_input_hashes = {
        "base": validate_sha256(arguments.base_sha256, "base checkpoint"),
        "adapter": validate_sha256(arguments.adapter_sha256, "adapter"),
        "training_manifest": validate_sha256(
            arguments.training_manifest_sha256, "training manifest"
        ),
        "canary": validate_sha256(arguments.canary_sha256, "canary"),
        "audit": validate_sha256(arguments.audit_sha256, "audit"),
    }
    input_paths = {
        "base": arguments.base,
        "adapter": arguments.adapter,
        "training_manifest": arguments.training_manifest,
        "canary": arguments.canary,
        "audit": arguments.audit,
    }
    require(_bound_input_hashes(input_paths) == expected_input_hashes,
            "one or more immutable input hashes mismatch")
    initial_code_hashes = code_hashes()
    bound = load_bound_canary(
        arguments.canary,
        arguments.audit,
        arguments.canary_sha256,
        arguments.audit_sha256,
    )
    verify_code_identity(bound, initial_code_hashes)
    examples = build_inference_examples(bound)
    conditions = parse_conditions(arguments.conditions)
    training_manifest = load_training_manifest(
        arguments.training_manifest,
        expected_sha256=expected_input_hashes["training_manifest"],
        bound=bound,
        base_sha256=expected_input_hashes["base"],
    )
    runtime = load_model_and_adapter(
        arguments.base,
        arguments.adapter,
        device=arguments.device,
        bound=bound,
        base_sha256=expected_input_hashes["base"],
        adapter_sha256=expected_input_hashes["adapter"],
        training_manifest=training_manifest,
    )
    records, forward_count = evaluate_examples(
        examples,
        runtime,
        bound.document["label_token_ids"],
        conditions,
        batch_size=arguments.batch_size,
    )
    require(_bound_input_hashes(input_paths) == expected_input_hashes,
            "an immutable input changed during inference")
    require(code_hashes() == initial_code_hashes, "evaluator code changed during inference")
    bindings = bound.bindings(
        base_sha256=expected_input_hashes["base"],
        base_step=runtime.base_step,
        adapter_sha256=expected_input_hashes["adapter"],
        adapter_implementation_commit=runtime.implementation_commit,
        training_manifest_sha256=training_manifest.sha256,
        code_sha256=initial_code_hashes,
    )
    raw = build_raw_artifact(
        runtime=runtime,
        bindings=bindings,
        restricted_token_ids=bound.document["label_token_ids"],
        conditions=conditions,
        records=records,
        forward_count=forward_count,
    )
    raw_sha256, raw_bytes = write_exclusive_read_only_json(arguments.out_raw, raw)
    receipt = build_receipt(
        job_identity={
            "scheduler": arguments.scheduler,
            "job_id": arguments.job_id,
            "array_task_id": arguments.array_task_id,
            "attempt_id": arguments.attempt_id,
        },
        arm_name=runtime.arm_name,
        bindings=bindings,
        row_count=len(records),
        condition_count=len(conditions),
        forward_count=forward_count,
        raw_sha256=raw_sha256,
        raw_bytes=raw_bytes,
    )
    receipt_sha256, _ = write_exclusive_read_only_json(arguments.out_receipt, receipt)
    print(json.dumps({
        "schema": RECEIPT_SCHEMA,
        "arm_name": runtime.arm_name,
        "row_count": len(records),
        "forward_count": forward_count,
        "raw_result_sha256": raw_sha256,
        "receipt_sha256": receipt_sha256,
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
