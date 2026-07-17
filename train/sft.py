#!/usr/bin/env python
"""Shohin SFT — completion-masked fine-tuning on (question, reasoning) pairs.

Loss is computed ONLY on the answer tokens (prompt tokens are masked with ignore_index=-1), so the
model learns to *produce* step-by-step reasoning, not to model the questions. Examples are packed to
seq_len for efficiency (single-GPU friendly). The prompt format matches eval_suite.py
("Question: ...\\nAnswer: ...") so the learned behavior transfers directly to the benchmarks.

  python sft.py --init flagship_out/best_step10000.model.pt --data ../artifacts/sft/math.jsonl \\
      --tokenizer ../artifacts/shohin-tok-32k.json --epochs 3 --out sft_out
"""

import argparse
import base64
import glob
import hashlib
import io
import json
import math
import os
import platform
import stat
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import tokenizers
from tokenizers import Tokenizer
from model import GPT, GPTConfig
from muon import Muon, split_params
from sft_encoding import encode_supervised_example


DEFAULT_Q_FIELDS = ["question", "problem", "prompt", "instruction"]
DEFAULT_R_FIELDS = ["response", "answer", "solution", "completion", "output"]
CANONICAL_EXACT_BUDGET = {
    "updates": 1560,
    "batch_size": 16,
    "pack_len": 2048,
    "seed": 1337,
    "epochs": 1,
    "lr_muon": 1e-3,
    "lr_adam": 2e-4,
    "warmup": 50,
    "clip": 1.0,
}
EXACT_BUDGET_AUDIT = "sft_exact_budget_v1"
PRODUCTION_ADMISSION_AUDIT = "digitwise_factorial_v4_admission"
PRODUCTION_ADMISSION_SCHEMA = "shohin-factorial-v4-production-admission-v1"
REVIEWED_SOURCE_MANIFEST_SCHEMA = "shohin-factorial-v4-reviewed-source-manifest-v1"
FACTORIAL_SCHEMA = "shohin-digitwise-factorial-v4"
FACTORIAL_TRAINING_GROUP = "digitwise_factorial_v4"
FACTORIAL_ARMS = ("iid", "term", "width", "term_width")
FACTORIAL_ARM_SEEDS = {
    "iid": 202607170101,
    "term": 202607170211,
    "width": 202607170307,
    "term_width": 202607170409,
}
FACTORIAL_ROW_SOURCES = {
    "digitwise_factorial_transition_v4",
    "digitwise_factorial_readout_v4",
    "digitwise_factorial_final_v4",
}
FACTORIAL_PRODUCTION_TARGET = {
    "episodes": 39_985,
    "rows": 439_865,
    "transitions": 199_940,
}
FACTORIAL_REQUIRED_PACKS = 24_960
FROZEN_FACTORIAL_HELDOUT_SHA256 = (
    "89ce11b36ff2f56e83cda72a1f07b1a90f4a3dc3803c69db2779a27219712646"
)
FROZEN_FACTORIAL_HELDOUT_COUNTS = {
    "branches": 3_000,
    "controller_prompts": 19_800,
    "counterfactual_pairs": 1_500,
    "top_level_episodes": 1_500,
    "unique_normalized_prompts": 19_800,
    "unique_signatures": 3_000,
}
FROZEN_FACTORIAL_HELDOUT_REGIMES = {
    "fit_w4": 300,
    "fit_w6": 300,
    "value_ood_w4": 300,
    "value_ood_w6": 300,
    "width_ood_w8": 300,
}
CANONICAL_SOURCE_PATHS = (
    "pipeline/generate_digitwise_factorial_v4.py",
    "pipeline/generate_digitwise_recurrent_v1.py",
    "pipeline/test_generate_digitwise_factorial_v4.py",
    "pipeline/audit_digitwise_factorial_v4.py",
    "pipeline/test_audit_digitwise_factorial_v4.py",
    "train/digitwise_protocol.py",
    "train/sft.py",
    "train/model.py",
    "train/muon.py",
    "train/sft_encoding.py",
    "train/test_sft_exact_budget.py",
    "train/jobs/sft_factorial.sbatch",
)
EXACT_BUDGET_TRUST_BOUNDARY = {
    "remote_attestation": False,
    "input_same_uid_path_replacement": (
        "Inputs and scientific sources are detached once into exact-byte immutable "
        "private-memory snapshots before parsing, loading, or Python execution, so "
        "later same-UID path replacement or same-inode mutation cannot alter consumed "
        "bytes in this process."
    ),
    "output_same_uid_boundary": (
        "Outputs use fsynced private inodes, atomic no-replace links, retained descriptors, "
        "and hash/path reconciliation through final sealing. Modes 0444 and 0555 are "
        "accidental-write seals only: the owning UID can chmod, mutate, or replace artifacts "
        "after descriptor custody ends, so recipients must rehash the closed-world inventory."
    ),
    "scope": (
        "This is process-local exact-byte and filesystem custody, not remote attestation. "
        "It does not provide owner-proof immutability, post-process same-UID protection, "
        "or protection from in-process memory mutation, execution-state compromise, kernel "
        "or filesystem subversion."
    ),
}
EXACT_BUDGET_CONSUMPTION_CUSTODY = {
    "admission": "validated_from_expected_sha256_immutable_private_bytes",
    "corpus": "jsonl_parsed_from_admitted_immutable_private_bytesio",
    "init": "torch_load_from_admitted_immutable_private_bytesio",
    "reviewed_source_manifest": (
        "validated_from_expected_sha256_immutable_private_bytes"
    ),
    "sources": (
        "python_compiled_from_and_all_source_identities_hashed_from_the_same_"
        "bootstrap_immutable_private_bytes"
    ),
    "tokenizer": "tokenizer_from_str_from_admitted_immutable_private_bytes",
}


def build_packed(
    data_paths,
    tok,
    seq_len,
    q_fields,
    r_fields,
    eos_id,
    max_examples=0,
    group_field=None,
    prompt_override_field=None,
    return_stats=False,
):
    """Tokenize (question, response) rows -> packed sequences with a completion-only loss mask.
    Returns (X[int64 N,seq_len], Y[int64 N,seq_len]) where Y is -1 on prompt/pad (ignored)."""
    grouped_buffers = {}  # group -> (token ids, loss-mask)
    n_ex = n_tok = n_ans = 0
    skipped = {"blank_lines": 0, "invalid_fields": 0, "too_long": 0}
    for data_source in data_paths:
        owns_source = not hasattr(data_source, "read")
        source = open(data_source, encoding="utf-8") if owns_source else data_source
        try:
            source.seek(0)
            for raw_line in source:
                line = (
                    raw_line.decode("utf-8", errors="strict")
                    if isinstance(raw_line, bytes)
                    else raw_line
                )
                line = line.strip()
                if not line:
                    skipped["blank_lines"] += 1
                    continue
                r = json.loads(line)
                q = next((r[f] for f in q_fields if r.get(f)), None)
                a = next((r[f] for f in r_fields if r.get(f)), None)
                if not q or not a:
                    skipped["invalid_fields"] += 1
                    continue
                group = (
                    str(r.get(group_field) or "default") if group_field else "default"
                )
                buf_x, buf_m = grouped_buffers.setdefault(group, ([], []))
                prompt_override = (
                    str(r.get(prompt_override_field) or "")
                    if prompt_override_field
                    else ""
                )
                prompt = prompt_override or f"Question: {q}\nAnswer:"
                # Completion-form code must retain the indentation beginning its
                # function body. Standard answer-form SFT keeps its established
                # trimmed response behavior.
                answer = str(a).rstrip() if prompt_override else str(a).strip()
                sep = (
                    "" if prompt_override or prompt.endswith((" ", "\n", "\t")) else " "
                )
                _, fids, mask = encode_supervised_example(
                    tok, prompt, sep + answer, eos_id
                )
                if len(fids) > seq_len:  # skip pathologically long examples
                    skipped["too_long"] += 1
                    continue
                buf_x.extend(fids)
                buf_m.extend(mask)
                n_ex += 1
                n_tok += len(fids)
                n_ans += sum(mask)
                if max_examples and n_ex >= max_examples:
                    break
        finally:
            if owns_source:
                source.close()
    # slice into seq_len+1 windows; target = next token, -1 where the next token is masked
    X, Y, groups = [], [], []
    group_stats = {}
    for group, (buf_x, buf_m) in grouped_buffers.items():
        first_pack = len(X)
        # Exact mode uses the mathematically complete window range. Legacy mode
        # retains its historical strict stop so existing noncanonical jobs do
        # not silently change their pack count at the one-token boundary.
        stop = len(buf_x) - seq_len if return_stats else len(buf_x) - seq_len - 1
        for i in range(0, stop, seq_len):
            xi = buf_x[i : i + seq_len]
            yi = [buf_x[j + 1] if buf_m[j + 1] else -1 for j in range(i, i + seq_len)]
            X.append(xi)
            Y.append(yi)
            groups.append(group)
        packed_count = len(X) - first_pack
        first_unused = packed_count * seq_len + 1
        group_stats[group] = {
            "encoded_tokens": len(buf_x),
            "encoded_supervised_tokens": int(sum(buf_m)),
            "packed_sequences": packed_count,
            "packed_forward_positions": packed_count * seq_len,
            "unpacked_tail_tokens": max(0, len(buf_x) - first_unused),
            "unpacked_tail_supervised_tokens": int(sum(buf_m[first_unused:])),
        }
    X_array = np.array(X, dtype=np.int64)
    Y_array = np.array(Y, dtype=np.int64)
    groups_array = np.array(groups, dtype=object)
    print(
        f"[sft-data] {n_ex:,} examples, {n_tok:,} tokens ({n_ans:,} answer tokens = "
        f"{100 * n_ans / max(n_tok, 1):.0f}% trained), {len(X):,} packed seqs of {seq_len}",
        flush=True,
    )
    if group_field:
        counts = {group: groups.count(group) for group in sorted(set(groups))}
        print(f"[sft-data] packed groups={counts}", flush=True)
    if not return_stats:
        return X_array, Y_array, groups_array
    digest = hashlib.sha256()
    for name, array in (("X", X_array), ("Y", Y_array)):
        canonical = np.asarray(array, dtype="<i8", order="C")
        digest.update(
            json.dumps(
                {"name": name, "shape": canonical.shape, "dtype": "<i8"},
                sort_keys=True,
                separators=(",", ":"),
            ).encode("ascii")
        )
        digest.update(memoryview(canonical).cast("B"))
    stats = {
        "examples": n_ex,
        "encoded_tokens": n_tok,
        "encoded_supervised_tokens": n_ans,
        "packed_sequences": len(X),
        "packed_forward_positions": len(X) * seq_len,
        "packed_supervised_tokens": int(np.count_nonzero(Y_array != -1)),
        "pack_len": seq_len,
        "packing_sha256": digest.hexdigest(),
        "skipped": skipped,
        "groups": {key: group_stats[key] for key in sorted(group_stats)},
    }
    return X_array, Y_array, groups_array, stats


def parse_sample_weights(items):
    weights = {}
    for item in items:
        key, sep, value = item.partition("=")
        if not sep or not key:
            raise ValueError(
                f"invalid --sample-weights item {item!r}; expected group=weight"
            )
        weight = float(value)
        if weight <= 0:
            raise ValueError(f"sample weight must be positive: {item!r}")
        if key in weights:
            raise ValueError(f"duplicate sample-weight group: {key}")
        weights[key] = weight
    return weights


def weighted_epoch_order(rng, groups, batch_size, weights):
    """Sample packed sequences by immutable group labels without duplicating data files."""
    group_to_indices = {
        group: np.flatnonzero(groups == group) for group in sorted(set(groups))
    }
    missing = sorted(set(weights) - set(group_to_indices))
    if missing:
        raise ValueError(f"sample weights name absent groups: {', '.join(missing)}")
    names = list(weights)
    probs = np.array([weights[name] for name in names], dtype=np.float64)
    probs /= probs.sum()
    count = (len(groups) // batch_size) * batch_size
    chosen = rng.choice(len(names), size=count, p=probs)
    order = np.empty(count, dtype=np.int64)
    requested = {}
    for i, name in enumerate(names):
        slots = np.flatnonzero(chosen == i)
        requested[name] = int(len(slots))
        if len(slots):
            order[slots] = rng.choice(
                group_to_indices[name], size=len(slots), replace=True
            )
    return order, requested


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fd_sha256(descriptor):
    digest = hashlib.sha256()
    offset = 0
    while True:
        block = os.pread(descriptor, 1024 * 1024, offset)
        if not block:
            return digest.hexdigest()
        digest.update(block)
        offset += len(block)


def _inode_identity(file_stat):
    return (
        int(file_stat.st_dev),
        int(file_stat.st_ino),
        int(file_stat.st_size),
        int(file_stat.st_mtime_ns),
        stat.S_IMODE(file_stat.st_mode),
    )


def _sha256_bytes(payload):
    return hashlib.sha256(payload).hexdigest()


def _fd_binding(descriptor, logical_path, expected_sha256=""):
    file_stat = os.fstat(descriptor)
    binding = {
        "bytes": int(file_stat.st_size),
        "fd_identity": {
            "device": int(file_stat.st_dev),
            "inode": int(file_stat.st_ino),
            "mtime_ns": int(file_stat.st_mtime_ns),
            "mode": f"{stat.S_IMODE(file_stat.st_mode):04o}",
            "size": int(file_stat.st_size),
        },
        "path": logical_path,
        "sha256": _fd_sha256(descriptor),
        "stable_descriptor": True,
    }
    if expected_sha256:
        binding["expected_sha256"] = expected_sha256
    return binding


@dataclass(frozen=True)
class ImmutableByteSnapshot:
    """One admitted payload detached from every mutable filesystem namespace."""

    logical_path: str
    payload: bytes
    sha256: str
    binding: dict

    def verify_bytes(self):
        actual = _sha256_bytes(self.payload)
        if actual != self.sha256:
            raise RuntimeError(
                f"immutable snapshot bytes changed: {self.logical_path}; "
                f"expected {self.sha256}, got {actual}"
            )
        return actual

    def open_bytes(self):
        self.verify_bytes()
        return io.BytesIO(self.payload)


@dataclass
class StableFile:
    """Output-only descriptor custody for a newly published canonical artifact."""

    path: Path
    handle: object
    identity: tuple
    sha256: str
    binding: dict
    parent_identity: tuple

    def verify_bytes(self):
        file_stat = os.fstat(self.handle.fileno())
        if not stat.S_ISREG(file_stat.st_mode):
            raise RuntimeError(f"stable descriptor is no longer regular: {self.path}")
        if _inode_identity(file_stat) != self.identity:
            raise RuntimeError(f"stable descriptor metadata changed: {self.path}")
        actual = _fd_sha256(self.handle.fileno())
        if actual != self.sha256:
            raise RuntimeError(
                f"stable descriptor bytes changed: {self.path}; "
                f"expected {self.sha256}, got {actual}"
            )
        return actual

    def verify_path_identity(self):
        try:
            parent_stat = os.stat(self.path.parent, follow_symlinks=False)
        except FileNotFoundError as error:
            raise RuntimeError(
                f"published parent disappeared: {self.path.parent}"
            ) from error
        if (
            not stat.S_ISDIR(parent_stat.st_mode)
            or (int(parent_stat.st_dev), int(parent_stat.st_ino))
            != self.parent_identity
        ):
            raise RuntimeError(f"published parent was substituted: {self.path.parent}")
        try:
            path_stat = os.stat(self.path, follow_symlinks=False)
        except FileNotFoundError as error:
            raise RuntimeError(f"published path disappeared: {self.path}") from error
        if (
            not stat.S_ISREG(path_stat.st_mode)
            or _inode_identity(path_stat) != self.identity
        ):
            raise RuntimeError(f"published path was substituted: {self.path}")
        return self.verify_bytes()

    def close(self):
        self.handle.close()


def canonical_json_bytes(value):
    return (
        json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("ascii")


def _publish_no_replace(path, writer):
    """Write through a private inode, fsync, then hard-link the final name once."""
    path = Path(path)
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    directory = os.open(path.parent, directory_flags)
    directory_stat = os.fstat(directory)
    path_parent_stat = os.stat(path.parent, follow_symlinks=False)
    if not stat.S_ISDIR(directory_stat.st_mode) or (
        directory_stat.st_dev,
        directory_stat.st_ino,
    ) != (path_parent_stat.st_dev, path_parent_stat.st_ino):
        os.close(directory)
        raise ValueError(f"artifact parent changed while opening: {path.parent}")
    parent_identity = (int(directory_stat.st_dev), int(directory_stat.st_ino))
    temporary = f".{path.name}.tmp.{os.getpid()}.{time.time_ns()}"
    descriptor = -1
    try:
        descriptor = os.open(
            temporary,
            os.O_RDWR | os.O_CREAT | os.O_EXCL,
            0o600,
            dir_fd=directory,
        )
        with os.fdopen(os.dup(descriptor), "wb") as sink:
            writer(sink)
            sink.flush()
            os.fsync(sink.fileno())
        if os.fstat(descriptor).st_size <= 0:
            raise RuntimeError(f"refusing empty canonical artifact: {path}")
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
        os.link(
            temporary,
            path.name,
            src_dir_fd=directory,
            dst_dir_fd=directory,
            follow_symlinks=False,
        )
        os.fsync(directory)
        binding = _fd_binding(descriptor, path.name)
        binding["parent_fd_identity"] = {
            "device": parent_identity[0],
            "inode": parent_identity[1],
        }
        artifact = StableFile(
            path=path,
            handle=os.fdopen(descriptor, "rb"),
            identity=_inode_identity(os.fstat(descriptor)),
            sha256=binding["sha256"],
            binding=binding,
            parent_identity=parent_identity,
        )
        descriptor = -1
        artifact.verify_path_identity()
        return artifact
    finally:
        try:
            os.unlink(temporary, dir_fd=directory)
        except FileNotFoundError:
            pass
        os.fsync(directory)
        if descriptor >= 0:
            os.close(descriptor)
        os.close(directory)


def publish_canonical_json(path, value):
    payload = canonical_json_bytes(value)
    return _publish_no_replace(path, lambda sink: sink.write(payload))


def publish_exact_bytes(path, payload):
    immutable_payload = bytes(payload)
    return _publish_no_replace(path, lambda sink: sink.write(immutable_payload))


def publish_torch_checkpoint(path, value):
    return _publish_no_replace(path, lambda sink: torch.save(value, sink))


def seal_output_directory(path, artifacts):
    """Seal a closed-world artifact directory after descriptor/path reconciliation."""
    path = Path(path)
    if path.is_symlink() or not path.is_dir():
        raise RuntimeError(f"canonical output is not a regular directory: {path}")
    directory = os.open(
        path,
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        directory_stat = os.fstat(directory)
        directory_identity = (
            int(directory_stat.st_dev),
            int(directory_stat.st_ino),
        )
        if any(
            artifact.parent_identity != directory_identity for artifact in artifacts
        ):
            raise RuntimeError("canonical artifacts do not bind the output directory")
        expected = {artifact.path.name for artifact in artifacts}
        actual = set(os.listdir(directory))
        if actual != expected:
            raise RuntimeError(
                f"canonical output inventory mismatch: expected {sorted(expected)}, "
                f"got {sorted(actual)}"
            )
        for artifact in artifacts:
            artifact.verify_path_identity()
            os.fchmod(artifact.handle.fileno(), 0o444)
            os.fsync(artifact.handle.fileno())
        os.fchmod(directory, 0o555)
        os.fsync(directory)
        path_stat = os.stat(path, follow_symlinks=False)
        if (path_stat.st_dev, path_stat.st_ino) != (
            directory_stat.st_dev,
            directory_stat.st_ino,
        ):
            raise RuntimeError(
                "canonical output directory was substituted during sealing"
            )
        if stat.S_IMODE(path_stat.st_mode) != 0o555:
            raise RuntimeError("canonical output directory did not seal mode 0555")
        for artifact in artifacts:
            artifact.verify_path_identity()
            if stat.S_IMODE(os.fstat(artifact.handle.fileno()).st_mode) != 0o444:
                raise RuntimeError(
                    f"canonical artifact did not seal mode 0444: {artifact.path}"
                )
    finally:
        os.close(directory)


def _validate_sha256(value, label):
    if len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
        raise ValueError(f"{label} must be a lowercase 64-character SHA-256")


def _relative_snapshot_path(path, snapshot_root):
    path = Path(path)
    root = Path(snapshot_root).resolve(strict=True)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"canonical input is not a regular non-symlink file: {path}")
    resolved = path.resolve(strict=True)
    try:
        relative = resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(f"canonical input escapes private snapshot: {path}") from error
    if stat.S_IMODE(resolved.stat().st_mode) & 0o222:
        raise ValueError(f"canonical snapshot input remains writable: {path}")
    return relative.as_posix()


def immutable_snapshot_from_bytes(
    logical_path,
    payload,
    expected_sha256,
    *,
    origin,
    admission_identity=None,
):
    """Bind identity and all future consumption to one immutable bytes object."""
    _validate_sha256(expected_sha256, f"expected hash for {logical_path}")
    immutable_payload = bytes(payload)
    actual = _sha256_bytes(immutable_payload)
    if actual != expected_sha256:
        raise ValueError(
            f"hash mismatch for {logical_path}: expected {expected_sha256}, got {actual}"
        )
    binding = {
        "bytes": len(immutable_payload),
        "custody": "one_time_exact_byte_immutable_private_memory_v1",
        "expected_sha256": expected_sha256,
        "immutable_byte_snapshot": True,
        "live_descriptor_retained": False,
        "origin": origin,
        "path": logical_path,
        "sha256": actual,
    }
    if admission_identity is not None:
        binding["admission_fd_identity"] = admission_identity
    snapshot = ImmutableByteSnapshot(
        logical_path=logical_path,
        payload=immutable_payload,
        sha256=actual,
        binding=binding,
    )
    snapshot.verify_bytes()
    return snapshot


def _capture_exact_descriptor_bytes(descriptor):
    """Read once while deriving the hash from the exact returned byte sequence."""
    chunks = []
    digest = hashlib.sha256()
    while True:
        block = os.read(descriptor, 1024 * 1024)
        if not block:
            break
        chunks.append(block)
        digest.update(block)
    return b"".join(chunks), digest.hexdigest()


def freeze_file_bytes(path, expected_sha256, snapshot_root):
    """Admit one private file, then sever all future dependence on its inode/path."""
    _validate_sha256(expected_sha256, f"expected hash for {path}")
    path = Path(path)
    relative = _relative_snapshot_path(path, snapshot_root)
    resolved = path.resolve(strict=True)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(resolved, flags)
    try:
        file_stat_before = os.fstat(descriptor)
        path_stat_before = os.stat(resolved, follow_symlinks=False)
        parent_stat = os.stat(resolved.parent, follow_symlinks=False)
        if not stat.S_ISREG(file_stat_before.st_mode):
            raise ValueError(f"canonical descriptor is not regular: {path}")
        if not stat.S_ISDIR(parent_stat.st_mode):
            raise ValueError(
                f"canonical input parent is not a directory: {path.parent}"
            )
        if _inode_identity(file_stat_before) != _inode_identity(path_stat_before):
            raise ValueError(f"canonical path changed while opening: {path}")
        if stat.S_IMODE(file_stat_before.st_mode) & 0o222:
            raise ValueError(f"canonical snapshot input remains writable: {path}")
        payload, captured_sha256 = _capture_exact_descriptor_bytes(descriptor)
        file_stat_after = os.fstat(descriptor)
        path_stat_after = os.stat(resolved, follow_symlinks=False)
        if _inode_identity(file_stat_after) != _inode_identity(file_stat_before):
            raise ValueError(f"canonical descriptor changed while capturing: {path}")
        if _inode_identity(path_stat_after) != _inode_identity(file_stat_before):
            raise ValueError(f"canonical path changed while capturing: {path}")
        if len(payload) != int(file_stat_before.st_size):
            raise ValueError(
                f"canonical descriptor size changed while capturing: {path}"
            )
        if captured_sha256 != expected_sha256:
            raise ValueError(
                f"hash mismatch for {path}: expected {expected_sha256}, "
                f"got {captured_sha256}"
            )
        return immutable_snapshot_from_bytes(
            relative,
            payload,
            expected_sha256,
            origin="private_snapshot_file_one_pass_capture_v1",
            admission_identity={
                "device": int(file_stat_before.st_dev),
                "inode": int(file_stat_before.st_ino),
                "mtime_ns": int(file_stat_before.st_mtime_ns),
                "mode": f"{stat.S_IMODE(file_stat_before.st_mode):04o}",
                "parent_device": int(parent_stat.st_dev),
                "parent_inode": int(parent_stat.st_ino),
                "size": int(file_stat_before.st_size),
            },
        )
    finally:
        os.close(descriptor)


def exact_file_binding(path, expected_sha256, snapshot_root):
    return freeze_file_bytes(path, expected_sha256, snapshot_root).binding


def parse_expected_source_hashes(items):
    hashes = {}
    for item in items:
        path, separator, digest = item.partition("=")
        if not separator or not path or not digest:
            raise ValueError(
                f"invalid --expected-source-sha256 item {item!r}; expected path=sha256"
            )
        if path in hashes:
            raise ValueError(f"duplicate expected source path: {path}")
        _validate_sha256(digest, f"expected source hash for {path}")
        hashes[path] = digest
    expected = set(CANONICAL_SOURCE_PATHS)
    actual = set(hashes)
    if actual != expected:
        raise ValueError(
            "canonical expected source set mismatch: "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )
    return {key: hashes[key] for key in CANONICAL_SOURCE_PATHS}


def snapshot_scientific_sources(expected_hashes, bootstrap_source_bytes):
    if not isinstance(bootstrap_source_bytes, Mapping):
        raise ValueError(
            "canonical exact-budget mode requires immutable source-byte bootstrap"
        )
    expected_paths = set(CANONICAL_SOURCE_PATHS)
    actual_paths = set(bootstrap_source_bytes)
    if actual_paths != expected_paths:
        raise ValueError(
            "canonical bootstrap source set mismatch: "
            f"missing={sorted(expected_paths - actual_paths)}, "
            f"extra={sorted(actual_paths - expected_paths)}"
        )
    sources = {}
    for relative in CANONICAL_SOURCE_PATHS:
        payload = bootstrap_source_bytes[relative]
        if not isinstance(payload, bytes):
            raise ValueError(f"bootstrap source is not immutable bytes: {relative}")
        sources[relative] = immutable_snapshot_from_bytes(
            relative,
            payload,
            expected_hashes[relative],
            origin="canonical_launcher_bootstrap_exact_bytes_v1",
        )
    return {key: sources[key] for key in sorted(sources)}


def scientific_source_bundle(source_snapshots):
    """Canonical, exactly recoverable source bytes plus their manifest."""
    if set(source_snapshots) != set(CANONICAL_SOURCE_PATHS):
        raise ValueError("cannot bundle an incomplete canonical source set")
    return {
        "encoding": "base64",
        "schema": "shohin-factorial-v4-scientific-source-bundle-v1",
        "sources": {
            relative: {
                "bytes": len(source_snapshots[relative].payload),
                "payload_base64": base64.b64encode(
                    source_snapshots[relative].payload
                ).decode("ascii"),
                "sha256": source_snapshots[relative].sha256,
            }
            for relative in CANONICAL_SOURCE_PATHS
        },
    }


def _admission_require(condition, code):
    if not condition:
        raise ValueError(f"production admission rejected: {code}")


def load_canonical_snapshot_json(snapshot, label):
    snapshot.verify_bytes()
    try:
        value = json.loads(snapshot.payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    if snapshot.payload != canonical_json_bytes(value):
        raise ValueError(f"{label} must use canonical JSON bytes")
    return value


def derive_factorial_arm_from_corpus(corpus_snapshot):
    """Derive the arm and row identity from admitted data, never from a label."""
    corpus_snapshot.verify_bytes()
    try:
        text = corpus_snapshot.payload.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise ValueError("canonical factorial corpus is not UTF-8") from error
    arms = set()
    sources = set()
    rows = 0
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            raise ValueError(
                f"canonical factorial corpus contains blank line {line_number}"
            )
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"canonical factorial corpus has invalid JSON at line {line_number}"
            ) from error
        if not isinstance(row, dict):
            raise ValueError(
                f"canonical factorial corpus row {line_number} is not an object"
            )
        arm = row.get("arm")
        if arm not in FACTORIAL_ARMS:
            raise ValueError(
                f"canonical factorial corpus row {line_number} has invalid arm"
            )
        expected_term = arm in {"term", "term_width"}
        expected_width = arm in {"width", "term_width"}
        identity_ok = (
            row.get("schema") == FACTORIAL_SCHEMA
            and row.get("seed") == FACTORIAL_ARM_SEEDS[arm]
            and row.get("split") == "train"
            and row.get("term_factor") is expected_term
            and row.get("width_factor") is expected_width
            and row.get("training_group") == FACTORIAL_TRAINING_GROUP
            and row.get("source") in FACTORIAL_ROW_SOURCES
            and isinstance(row.get("question"), str)
            and bool(row.get("question"))
            and isinstance(row.get("response"), str)
            and bool(row.get("response"))
        )
        if not identity_ok:
            raise ValueError(
                f"canonical factorial corpus row {line_number} violates identity contract"
            )
        arms.add(arm)
        sources.add(row["source"])
        rows += 1
    if rows == 0 or len(arms) != 1:
        raise ValueError("canonical factorial corpus does not identify exactly one arm")
    if sources != FACTORIAL_ROW_SOURCES:
        raise ValueError("canonical factorial corpus is missing required row sources")
    arm = next(iter(arms))
    return arm, {
        "arm": arm,
        "rows": rows,
        "schema": FACTORIAL_SCHEMA,
        "sources": sorted(sources),
        "split": "train",
        "training_group": FACTORIAL_TRAINING_GROUP,
    }


def validate_reviewed_source_manifest(
    manifest_snapshot, expected_commit, source_snapshots
):
    manifest = load_canonical_snapshot_json(
        manifest_snapshot, "reviewed source manifest"
    )
    commit = manifest.get("reviewed_clean_commit")
    commit_is_hash = (
        isinstance(commit, str)
        and len(commit) in {40, 64}
        and all(character in "0123456789abcdef" for character in commit)
    )
    _admission_require(
        manifest.get("schema") == REVIEWED_SOURCE_MANIFEST_SCHEMA,
        "reviewed_source_manifest_schema",
    )
    _admission_require(commit_is_hash, "reviewed_source_manifest_commit")
    _admission_require(commit == expected_commit, "reviewed_commit_mismatch")
    _admission_require(
        manifest.get("review_status") == "approved",
        "reviewed_source_manifest_not_approved",
    )
    _admission_require(
        manifest.get("clean_source_tree") is True,
        "reviewed_source_manifest_not_clean",
    )
    expected_sources = {
        relative: source_snapshots[relative].sha256
        for relative in CANONICAL_SOURCE_PATHS
    }
    _admission_require(
        manifest.get("sources") == expected_sources,
        "reviewed_source_manifest_source_set",
    )
    _admission_require(
        manifest.get("remote_attestation") is False,
        "reviewed_source_manifest_remote_attestation_boundary",
    )
    return manifest


def _validate_receipt_source_manifest(name, manifest, paths, reviewed_sources):
    _admission_require(isinstance(manifest, dict), f"{name}_manifest_object")
    sources = manifest.get("sources")
    _admission_require(isinstance(sources, dict), f"{name}_manifest_sources")
    _admission_require(set(sources) == set(paths), f"{name}_manifest_source_set")
    for relative in paths:
        binding = sources.get(relative)
        _admission_require(
            isinstance(binding, dict)
            and binding.get("sha256") == reviewed_sources[relative]
            and isinstance(binding.get("bytes"), int)
            and binding["bytes"] > 0,
            f"{name}_manifest_binding_{relative}",
        )


def validate_production_admission(
    admission_snapshot,
    corpus_snapshot,
    tokenizer_snapshot,
    source_snapshots,
    reviewed_manifest,
):
    receipt = load_canonical_snapshot_json(
        admission_snapshot, "production admission receipt"
    )
    arm, corpus_contract = derive_factorial_arm_from_corpus(corpus_snapshot)
    _admission_require(
        receipt.get("audit") == PRODUCTION_ADMISSION_AUDIT, "audit_schema"
    )
    _admission_require(
        receipt.get("receipt_schema") == PRODUCTION_ADMISSION_SCHEMA,
        "receipt_schema",
    )
    _admission_require(receipt.get("schema") == FACTORIAL_SCHEMA, "data_schema")
    _admission_require(receipt.get("mode") == "production", "not_production_mode")
    _admission_require(
        receipt.get("production_contract") is True, "production_contract"
    )
    _admission_require(
        receipt.get("production_admission") is True, "production_admission_false"
    )
    _admission_require(receipt.get("admission_pass") is True, "admission_pass_false")
    _admission_require(receipt.get("mechanical_pass") is True, "mechanical_pass_false")
    checks = receipt.get("checks")
    _admission_require(
        isinstance(checks, dict) and bool(checks) and all(checks.values()),
        "failed_audit_checks",
    )
    _admission_require(receipt.get("failures") == {}, "audit_failures_present")
    _admission_require(
        receipt.get("admitted_arm") == arm and receipt.get("declared_arm") == arm,
        "wrong_arm",
    )
    paired_arm = {
        "iid": "term",
        "term": "iid",
        "width": "term_width",
        "term_width": "width",
    }[arm]
    expected_factors = {
        "term": arm in {"term", "term_width"},
        "width": arm in {"width", "term_width"},
    }
    _admission_require(receipt.get("test_scale") is None, "test_scale_receipt")
    _admission_require(
        receipt.get("target") == FACTORIAL_PRODUCTION_TARGET,
        "production_target",
    )
    _admission_require(
        receipt.get("seed") == FACTORIAL_ARM_SEEDS[arm]
        and receipt.get("paired_arm") == paired_arm
        and receipt.get("declared_factors") == expected_factors
        and receipt.get("board") == ("wide" if expected_factors["width"] else "narrow"),
        "factorial_design_identity",
    )
    independent = receipt.get("independent_audit", {})
    _admission_require(
        independent.get("auditor_recomputed_solver_and_contract") is True
        and independent.get("generator_implementation_imported") is False
        and independent.get("generator_reports_used_only_as_bound_provenance") is True
        and independent.get("remote_attestation") is False,
        "independent_audit_boundary",
    )
    inputs = receipt.get("inputs")
    required_inputs = {
        "data",
        "episodes",
        "paired_data",
        "paired_episodes",
        "heldout",
        "tokenizer",
    }
    _admission_require(
        isinstance(inputs, dict) and set(inputs) == required_inputs,
        "input_binding_set",
    )
    data_binding = inputs["data"]
    tokenizer_binding = inputs["tokenizer"]
    _admission_require(
        data_binding.get("sha256") == corpus_snapshot.sha256
        and data_binding.get("bytes") == len(corpus_snapshot.payload)
        and receipt.get("data_sha256") == corpus_snapshot.sha256,
        "stale_data",
    )
    _admission_require(
        tokenizer_binding.get("sha256") == tokenizer_snapshot.sha256
        and tokenizer_binding.get("bytes") == len(tokenizer_snapshot.payload),
        "stale_tokenizer",
    )
    for name in required_inputs - {"data", "tokenizer"}:
        binding = inputs[name]
        _admission_require(
            isinstance(binding.get("sha256"), str)
            and len(binding["sha256"]) == 64
            and isinstance(binding.get("bytes"), int)
            and binding["bytes"] > 0,
            f"invalid_{name}_binding",
        )
    heldout = receipt.get("heldout", {})
    heldout_digests = heldout.get("identity_digests", {})
    _admission_require(
        heldout.get("sha256")
        == inputs["heldout"]["sha256"]
        == FROZEN_FACTORIAL_HELDOUT_SHA256,
        "heldout_identity",
    )
    _admission_require(
        heldout.get("frozen_sha256_required") == FROZEN_FACTORIAL_HELDOUT_SHA256
        and {key: heldout.get(key) for key in FROZEN_FACTORIAL_HELDOUT_COUNTS}
        == FROZEN_FACTORIAL_HELDOUT_COUNTS
        and heldout.get("regimes") == FROZEN_FACTORIAL_HELDOUT_REGIMES
        and heldout.get("blank_lines") == 0,
        "heldout_frozen_contract",
    )
    _admission_require(
        set(heldout_digests)
        == {
            "branch_ids_sha256",
            "normalized_prompts_sha256",
            "reserved_signatures_sha256",
        }
        and all(
            isinstance(value, str)
            and len(value) == 64
            and all(character in "0123456789abcdef" for character in value)
            for value in heldout_digests.values()
        ),
        "heldout_identity_digests",
    )
    contract = receipt.get("scientific_contract")
    _admission_require(
        contract
        == {
            "heldout_splits": [
                "fit_w4",
                "fit_w6",
                "value_ood_w4",
                "value_ood_w6",
                "width_ood_w8",
            ],
            "row_sources": sorted(FACTORIAL_ROW_SOURCES),
            "schema": FACTORIAL_SCHEMA,
            "training_group": FACTORIAL_TRAINING_GROUP,
            "training_split": "train",
        },
        "scientific_contract",
    )
    contamination = receipt.get("contamination", {})
    _admission_require(
        contamination.get("train_heldout_reserved_signature_hits") == 0
        and contamination.get("train_heldout_exact_normalized_prompt_hits") == 0
        and contamination.get("train_heldout_literal_13gram_hits") == 0
        and contamination.get("examples") == []
        and contamination.get("gates")
        == {
            "exact_normalized_prompt_clear": True,
            "literal_normalized_word_13gram_clear": True,
            "reserved_signature_clear": True,
        },
        "contamination_not_clear",
    )
    answer_boundary = contamination.get("heldout_answer_boundary", {})
    _admission_require(
        answer_boundary.get("answer_values_retained_for_training") is False
        and answer_boundary.get("training_rows_constructed_by_auditor") is False,
        "heldout_answer_boundary",
    )
    tokenizer_accounting = receipt.get("tokenizer_accounting")
    _admission_require(
        isinstance(tokenizer_accounting, dict)
        and tokenizer_accounting.get("tokenizer_sha256") == tokenizer_snapshot.sha256
        and tokenizer_accounting.get("tokenizer_bytes")
        == len(tokenizer_snapshot.payload),
        "tokenizer_accounting_binding",
    )
    runtime = receipt.get("runtime", {})
    _admission_require(
        isinstance(runtime.get("python"), str)
        and bool(runtime["python"])
        and isinstance(runtime.get("python_implementation"), str)
        and bool(runtime["python_implementation"])
        and runtime.get("tokenizers") == tokenizers.__version__
        and tokenizer_accounting.get("tokenizers_version") == tokenizers.__version__,
        "auditor_runtime",
    )
    admitted_packing = tokenizer_accounting.get("production_build_packed")
    _admission_require(
        isinstance(admitted_packing, dict)
        and isinstance(admitted_packing.get("packing_sha256"), str)
        and len(admitted_packing["packing_sha256"]) == 64
        and admitted_packing.get("examples") == FACTORIAL_PRODUCTION_TARGET["rows"]
        and admitted_packing.get("pack_len") == CANONICAL_EXACT_BUDGET["pack_len"]
        and admitted_packing.get("packed_sequences", 0) >= FACTORIAL_REQUIRED_PACKS
        and admitted_packing.get("skipped")
        == {"blank_lines": 0, "invalid_fields": 0, "too_long": 0},
        "packing_admission",
    )
    _admission_require(
        tokenizer_accounting.get("pack_length") == CANONICAL_EXACT_BUDGET["pack_len"]
        and tokenizer_accounting.get("overall", {}).get("rows_seen")
        == FACTORIAL_PRODUCTION_TARGET["rows"]
        and "EOS is supervised" in tokenizer_accounting.get("encoding_boundary", ""),
        "tokenizer_semantics",
    )
    source_manifests = receipt.get("source_manifests", {})
    reviewed_sources = reviewed_manifest["sources"]
    generator_paths = (
        "pipeline/generate_digitwise_factorial_v4.py",
        "pipeline/generate_digitwise_recurrent_v1.py",
        "pipeline/test_generate_digitwise_factorial_v4.py",
        "train/digitwise_protocol.py",
    )
    auditor_paths = (
        "pipeline/audit_digitwise_factorial_v4.py",
        "pipeline/test_audit_digitwise_factorial_v4.py",
        "train/digitwise_protocol.py",
        "train/sft.py",
        "train/sft_encoding.py",
        "train/test_sft_exact_budget.py",
        "train/jobs/sft_factorial.sbatch",
    )
    _validate_receipt_source_manifest(
        "generator",
        source_manifests.get("generator"),
        generator_paths,
        reviewed_sources,
    )
    _validate_receipt_source_manifest(
        "auditor", source_manifests.get("auditor"), auditor_paths, reviewed_sources
    )
    reports = receipt.get("generator_reports", {})
    _admission_require(set(reports) == {"primary", "counterpart"}, "generator_reports")
    for role in ("primary", "counterpart"):
        evidence = reports[role]
        _admission_require(
            evidence.get("validated") is True
            and isinstance(evidence.get("sha256"), str)
            and len(evidence["sha256"]) == 64
            and isinstance(evidence.get("bytes"), int)
            and evidence["bytes"] > 0
            and isinstance(evidence.get("runtime", {}).get("python"), str)
            and bool(evidence["runtime"]["python"])
            and isinstance(
                evidence.get("runtime", {}).get("python_implementation"), str
            )
            and bool(evidence["runtime"]["python_implementation"])
            and evidence.get("source_manifest", {}).get("sources")
            == source_manifests["generator"]["sources"],
            f"generator_report_{role}",
        )
    for relative, snapshot in source_snapshots.items():
        _admission_require(
            snapshot.sha256 == reviewed_sources[relative],
            f"live_source_mismatch_{relative}",
        )
    _admission_require(
        receipt.get("rows_observed", {}).get("raw")
        == corpus_contract["rows"]
        == receipt.get("rows_observed", {}).get("valid")
        == FACTORIAL_PRODUCTION_TARGET["rows"],
        "row_count_binding",
    )
    return arm, receipt, corpus_contract


def validate_admitted_packing(receipt, packing_stats):
    admitted = receipt["tokenizer_accounting"]["production_build_packed"]
    if packing_stats != admitted:
        raise ValueError(
            "production admission rejected: canonical packing differs from independent audit"
        )
    return admitted["packing_sha256"]


def deterministic_pack_order(pack_count, seed):
    """Cross-runtime order: lexicographic SHA-256(seed_u64 || pack_index_u64)."""
    if pack_count < 0 or seed < 0 or seed >= 2**64:
        raise ValueError(
            "pack count and seed must fit the deterministic ordering contract"
        )
    seed_bytes = int(seed).to_bytes(8, "little", signed=False)
    keyed = []
    for index in range(pack_count):
        key = hashlib.sha256(
            seed_bytes + index.to_bytes(8, "little", signed=False)
        ).digest()
        keyed.append((key, index))
    return np.asarray([index for _, index in sorted(keyed)], dtype=np.int64)


def _int64_sha256(values):
    canonical = np.asarray(values, dtype="<i8", order="C")
    return hashlib.sha256(memoryview(canonical).cast("B")).hexdigest()


def make_exact_budget_plan(supervised_per_pack, updates, batch_size, pack_len, seed):
    supervised_per_pack = np.asarray(supervised_per_pack, dtype=np.int64)
    if supervised_per_pack.ndim != 1:
        raise ValueError("supervised-per-pack counts must be rank one")
    if updates <= 0 or batch_size <= 0 or pack_len <= 0:
        raise ValueError("exact update, batch, and pack budgets must be positive")
    required_packs = updates * batch_size
    available_packs = len(supervised_per_pack)
    if available_packs < required_packs:
        raise ValueError(
            f"exact budget needs {required_packs} packs but corpus provides {available_packs}"
        )
    order = deterministic_pack_order(available_packs, seed)
    consumed = order[:required_packs]
    dropped = order[required_packs:]
    consumed_supervised = int(supervised_per_pack[consumed].sum())
    update_supervised = (
        supervised_per_pack[consumed].reshape(updates, batch_size).sum(axis=1)
    )
    if np.any(update_supervised <= 0):
        raise ValueError(
            "exact budget contains an optimizer update with no supervised tokens"
        )
    dropped_rows = [
        {"index": int(index), "supervised_tokens": int(supervised_per_pack[index])}
        for index in sorted(int(item) for item in dropped)
    ]
    receipt = {
        "available_packs": available_packs,
        "available_supervised_tokens": int(supervised_per_pack.sum()),
        "batch_size": batch_size,
        "consumed_pack_order_sha256": _int64_sha256(consumed),
        "consumed_packs": required_packs,
        "consumed_supervised_tokens": consumed_supervised,
        "dropped_pack_order_sha256": _int64_sha256(dropped),
        "dropped_packs": dropped_rows,
        "dropped_packs_count": len(dropped_rows),
        "dropped_supervised_tokens": int(supervised_per_pack[dropped].sum()),
        "forward_token_positions": required_packs * pack_len,
        "optimizer_updates": updates,
        "pack_len": pack_len,
        "selection_algorithm": "sha256(seed_u64_le||pack_index_u64_le)_lexicographic_v1",
        "seed": seed,
        "supervised_token_equality_enforced": False,
        "supervised_tokens_per_update": [int(value) for value in update_supervised],
        "supervised_tokens_per_update_max": int(update_supervised.max()),
        "supervised_tokens_per_update_min": int(update_supervised.min()),
        "supervised_tokens_per_update_sha256": _int64_sha256(update_supervised),
        "target_thinning": False,
    }
    return consumed, receipt


def validate_exact_packing_stats(packing_stats):
    skipped_rows = sum(packing_stats["skipped"].values())
    if skipped_rows:
        raise ValueError(
            f"canonical packing rejected {skipped_rows} rows: {packing_stats['skipped']}"
        )
    if packing_stats["packed_sequences"] <= 0:
        raise ValueError("canonical packing produced no sequences")


def validate_exact_budget_actual(
    plan, update_supervised_tokens, actual_packs, forward_positions
):
    actual_updates = len(update_supervised_tokens)
    actual_supervised = int(sum(update_supervised_tokens))
    actual_update_sha256 = _int64_sha256(update_supervised_tokens)
    expected = {
        "updates": plan["optimizer_updates"],
        "packs": plan["consumed_packs"],
        "forward_positions": plan["forward_token_positions"],
        "supervised_tokens": plan["consumed_supervised_tokens"],
        "supervised_tokens_per_update_sha256": plan[
            "supervised_tokens_per_update_sha256"
        ],
    }
    actual = {
        "updates": actual_updates,
        "packs": actual_packs,
        "forward_positions": forward_positions,
        "supervised_tokens": actual_supervised,
        "supervised_tokens_per_update_sha256": actual_update_sha256,
    }
    if actual != expected:
        raise RuntimeError(
            f"exact-budget execution mismatch: expected {expected}, got {actual}"
        )
    return {
        "forward_token_positions": forward_positions,
        "optimizer_updates": actual_updates,
        "packs": actual_packs,
        "supervised_tokens": actual_supervised,
        "supervised_tokens_per_update": [
            int(value) for value in update_supervised_tokens
        ],
        "supervised_tokens_per_update_sha256": actual_update_sha256,
    }


def validate_canonical_exact_settings(args, cfg, paths):
    expected = CANONICAL_EXACT_BUDGET
    checks = {
        "--exact-updates": (args.exact_updates, expected["updates"]),
        "--batch-size": (args.batch_size, expected["batch_size"]),
        "--pack-len": (args.pack_len, expected["pack_len"]),
        "--seed": (args.seed, expected["seed"]),
        "--epochs": (args.epochs, expected["epochs"]),
        "--lr-muon": (args.lr_muon, expected["lr_muon"]),
        "--lr-adam": (args.lr_adam, expected["lr_adam"]),
        "--warmup": (args.warmup, expected["warmup"]),
        "--clip": (args.clip, expected["clip"]),
    }
    mismatches = [
        f"{name}={actual!r} (required {wanted!r})"
        for name, (actual, wanted) in checks.items()
        if actual != wanted
    ]
    if mismatches:
        raise ValueError(
            "canonical exact-budget override rejected: " + "; ".join(mismatches)
        )
    forbidden = {
        "--compile": args.compile,
        "--freeze-lexicon": args.freeze_lexicon,
        "--group-field": args.group_field is not None,
        "--max-examples": args.max_examples != 0,
        "--prompt-override-field": args.prompt_override_field is not None,
        "--reference": bool(args.reference),
        "--replay-batch-size": args.replay_batch_size != 4,
        "--replay-max-tokens": args.replay_max_tokens != 128,
        "--replay-prompts": bool(args.replay_prompts),
        "--replay-weight": args.replay_weight != 0.0,
        "--sample-weights": bool(args.sample_weights),
    }
    enabled = [name for name, value in forbidden.items() if value]
    if enabled:
        raise ValueError("canonical exact-budget mode forbids: " + ", ".join(enabled))
    if args.q_fields != DEFAULT_Q_FIELDS or args.r_fields != DEFAULT_R_FIELDS:
        raise ValueError(
            "canonical exact-budget mode requires the default question/response fields"
        )
    if args.eos != "<|endoftext|>":
        raise ValueError("canonical exact-budget mode requires the frozen EOS token")
    if args.arm:
        raise ValueError(
            "canonical exact-budget mode forbids caller arm labels; arm is derived from "
            "the admitted receipt and corpus"
        )
    if len(paths) != 1 or len(args.data) != 1 or any(c in args.data[0] for c in "*?["):
        raise ValueError(
            "canonical exact-budget mode requires exactly one non-glob corpus"
        )
    if cfg.seq_len != expected["pack_len"]:
        raise ValueError(
            f"canonical model context must be {expected['pack_len']}, got {cfg.seq_len}"
        )
    if not args.snapshot_root:
        raise ValueError("canonical exact-budget mode requires --snapshot-root")
    if (
        not args.expected_init_sha256
        or not args.expected_tokenizer_sha256
        or not args.expected_data_sha256
        or not args.expected_production_admission_sha256
        or not args.expected_reviewed_source_manifest_sha256
        or not args.expected_reviewed_commit
        or not args.production_admission
        or not args.reviewed_source_manifest
    ):
        raise ValueError(
            "canonical exact-budget mode requires admission, reviewed source manifest, "
            "reviewed commit, and all expected input SHA-256 values"
        )
    commit = args.expected_reviewed_commit
    if len(commit) not in {40, 64} or any(
        character not in "0123456789abcdef" for character in commit
    ):
        raise ValueError("--expected-reviewed-commit must be a lowercase Git object ID")
    parse_expected_source_hashes(args.expected_source_sha256)


def load_replay_prompts(path, tokenizer, max_tokens):
    """Load prompt-only contexts used for raw-model behavior retention.

    These examples intentionally carry no answer target. They are used only to
    keep the candidate's next-token distribution near the immutable raw model
    on broad natural prompts while SFT teaches verified skills elsewhere.
    """
    if max_tokens < 2:
        raise ValueError("replay max tokens must be at least two")
    prompts, skipped = [], {"invalid": 0, "short": 0}
    with open(path) as source:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            prompt = row.get("prompt")
            if not isinstance(prompt, str) or not prompt.strip():
                skipped["invalid"] += 1
                continue
            ids = tokenizer.encode(prompt).ids[:max_tokens]
            if len(ids) < 2:
                skipped["short"] += 1
                continue
            prompts.append(ids)
    if not prompts:
        raise ValueError("no fitting prompt-only replay rows")
    return prompts, {key: value for key, value in sorted(skipped.items()) if value}


def make_replay_batch(prompts, batch_size, max_tokens, pad_id, rng, device):
    if batch_size <= 0:
        raise ValueError("replay batch size must be positive")
    indices = rng.integers(0, len(prompts), size=batch_size)
    ids = torch.full((batch_size, max_tokens), pad_id, dtype=torch.long, device=device)
    lengths = torch.zeros(batch_size, dtype=torch.long, device=device)
    for row, index in enumerate(indices):
        prompt = prompts[int(index)]
        length = min(len(prompt), max_tokens)
        ids[row, :length] = torch.tensor(
            prompt[:length], dtype=torch.long, device=device
        )
        lengths[row] = length
    return ids, lengths


def replay_kl(student_logits, teacher_logits, lengths):
    """Mean next-token KL on unpadded prompt positions."""
    if student_logits.shape != teacher_logits.shape or student_logits.ndim != 3:
        raise ValueError(
            "student and teacher replay logits must be matching rank-3 tensors"
        )
    positions = torch.arange(
        student_logits.shape[1], device=student_logits.device
    ).unsqueeze(0)
    mask = positions < (lengths.unsqueeze(1) - 1).clamp_min(0)
    if not bool(mask.any()):
        raise ValueError("replay batch has no next-token positions")
    pointwise = F.kl_div(
        F.log_softmax(student_logits.float(), dim=-1),
        F.softmax(teacher_logits.float(), dim=-1),
        reduction="none",
    ).sum(dim=-1)
    return (pointwise * mask).sum() / mask.sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--init", required=True, help="pretrained checkpoint to fine-tune from"
    )
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--q-fields", nargs="+", default=DEFAULT_Q_FIELDS)
    ap.add_argument("--r-fields", nargs="+", default=DEFAULT_R_FIELDS)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument(
        "--lr-muon", type=float, default=2e-3
    )  # gentler than pretrain (fine-tune)
    ap.add_argument("--lr-adam", type=float, default=5e-4)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--max-examples", type=int, default=0)
    ap.add_argument(
        "--seed",
        type=int,
        default=1337,
        help="torch and NumPy ordering seed (default preserves legacy behavior)",
    )
    ap.add_argument(
        "--pack-len",
        type=int,
        default=0,
        help="pack sequence length (0 = model seq_len); shorter = less memory",
    )
    ap.add_argument(
        "--group-field",
        default=None,
        help="optional immutable row field used to keep packed sequences by group",
    )
    ap.add_argument(
        "--prompt-override-field",
        default=None,
        help="optional row field containing an exact completion prompt (for code completion SFT)",
    )
    ap.add_argument(
        "--sample-weights",
        nargs="*",
        default=[],
        metavar="GROUP=WEIGHT",
        help="weighted per-epoch sampling over --group-field values; examples are sampled with replacement",
    )
    ap.add_argument(
        "--reference",
        default="",
        help="immutable raw checkpoint for prompt-only logit retention",
    )
    ap.add_argument(
        "--replay-prompts",
        default="",
        help="JSONL prompt-only replay contexts (no answer targets)",
    )
    ap.add_argument("--replay-weight", type=float, default=0.0)
    ap.add_argument("--replay-batch-size", type=int, default=4)
    ap.add_argument("--replay-max-tokens", type=int, default=128)
    ap.add_argument(
        "--freeze-lexicon",
        action="store_true",
        help="freeze tied token embedding/output geometry during SFT",
    )
    ap.add_argument("--eos", default="<|endoftext|>")
    ap.add_argument("--out", default="sft_out")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--canonical-exact-budget", action="store_true")
    ap.add_argument("--exact-updates", type=int, default=0)
    ap.add_argument("--arm", default="")
    ap.add_argument("--snapshot-root", default="")
    ap.add_argument("--expected-init-sha256", default="")
    ap.add_argument("--expected-tokenizer-sha256", default="")
    ap.add_argument("--expected-data-sha256", default="")
    ap.add_argument("--production-admission", default="")
    ap.add_argument("--expected-production-admission-sha256", default="")
    ap.add_argument("--reviewed-source-manifest", default="")
    ap.add_argument("--expected-reviewed-source-manifest-sha256", default="")
    ap.add_argument("--expected-reviewed-commit", default="")
    ap.add_argument(
        "--expected-source-sha256",
        action="append",
        default=[],
        metavar="PATH=SHA256",
    )
    a = ap.parse_args()

    if bool(a.reference) != bool(a.replay_prompts):
        raise SystemExit("--reference and --replay-prompts must be supplied together")
    if a.replay_prompts and a.replay_weight <= 0:
        raise SystemExit("--replay-weight must be positive with --replay-prompts")
    if a.replay_batch_size <= 0 or a.replay_max_tokens < 2:
        raise SystemExit("replay batch size and max tokens must be valid")
    if a.exact_updates and not a.canonical_exact_budget:
        raise SystemExit(
            "--exact-updates is available only with --canonical-exact-budget"
        )

    device = (
        "cuda"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    torch.manual_seed(a.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(a.seed)
    torch.set_float32_matmul_precision("high")

    paths = []
    for d in a.data:
        paths += sorted(glob.glob(d)) if any(c in d for c in "*?[") else [d]
    exact_input_snapshots = {}
    exact_source_snapshots = {}
    exact_input_bindings = None
    exact_source_bindings = None
    canonical_arm = ""
    canonical_corpus_contract = None
    production_admission_receipt = None
    reviewed_source_manifest = None
    admission_artifact = None
    reviewed_manifest_artifact = None
    source_bundle_artifact = None
    preflight_artifact = None
    checkpoint_artifact = None
    final_artifact = None
    if a.canonical_exact_budget:
        if not a.snapshot_root:
            raise ValueError("canonical exact-budget mode requires --snapshot-root")
        if len(paths) != 1:
            raise ValueError("canonical exact-budget mode requires exactly one corpus")
        expected_source_hashes = parse_expected_source_hashes(a.expected_source_sha256)
        exact_input_snapshots = {
            "corpus": freeze_file_bytes(
                paths[0], a.expected_data_sha256, a.snapshot_root
            ),
            "init": freeze_file_bytes(a.init, a.expected_init_sha256, a.snapshot_root),
            "tokenizer": freeze_file_bytes(
                a.tokenizer, a.expected_tokenizer_sha256, a.snapshot_root
            ),
            "production_admission": freeze_file_bytes(
                a.production_admission,
                a.expected_production_admission_sha256,
                a.snapshot_root,
            ),
            "reviewed_source_manifest": freeze_file_bytes(
                a.reviewed_source_manifest,
                a.expected_reviewed_source_manifest_sha256,
                a.snapshot_root,
            ),
        }
        exact_source_snapshots = snapshot_scientific_sources(
            expected_source_hashes,
            globals().get("_CANONICAL_BOOTSTRAP_SOURCE_BYTES"),
        )
        exact_input_bindings = {
            name: snapshot.binding for name, snapshot in exact_input_snapshots.items()
        }
        exact_source_bindings = {
            name: snapshot.binding for name, snapshot in exact_source_snapshots.items()
        }
        reviewed_source_manifest = validate_reviewed_source_manifest(
            exact_input_snapshots["reviewed_source_manifest"],
            a.expected_reviewed_commit,
            exact_source_snapshots,
        )
        (
            canonical_arm,
            production_admission_receipt,
            canonical_corpus_contract,
        ) = validate_production_admission(
            exact_input_snapshots["production_admission"],
            exact_input_snapshots["corpus"],
            exact_input_snapshots["tokenizer"],
            exact_source_snapshots,
            reviewed_source_manifest,
        )
        ck = torch.load(exact_input_snapshots["init"].open_bytes(), map_location="cpu")
    else:
        ck = torch.load(a.init, map_location="cpu")
    cfg = GPTConfig(**ck["cfg"])
    pack_len = a.pack_len or cfg.seq_len
    if a.canonical_exact_budget:
        if Path(a.out).exists() or Path(a.out).is_symlink():
            raise ValueError(f"canonical output must not already exist: {a.out}")
        validate_canonical_exact_settings(a, cfg, paths)
        if device != "cuda":
            raise ValueError("canonical exact-budget training requires CUDA")
        Path(a.out).mkdir(parents=True, exist_ok=False)
        os.chmod(a.out, 0o700)
    else:
        os.makedirs(a.out, exist_ok=True)

    if a.canonical_exact_budget:
        tokenizer_bytes = exact_input_snapshots["tokenizer"].payload
        tok = Tokenizer.from_str(tokenizer_bytes.decode("utf-8", errors="strict"))
        packed_sources = [exact_input_snapshots["corpus"].open_bytes()]
    else:
        tok = Tokenizer.from_file(a.tokenizer)
        packed_sources = paths
    eos_id = tok.token_to_id(a.eos)
    if eos_id is None:
        raise ValueError(f"tokenizer does not define EOS token {a.eos!r}")
    packed = build_packed(
        packed_sources,
        tok,
        pack_len,
        a.q_fields,
        a.r_fields,
        eos_id,
        a.max_examples,
        group_field=a.group_field,
        prompt_override_field=a.prompt_override_field,
        return_stats=a.canonical_exact_budget,
    )
    if a.canonical_exact_budget:
        X, Y, groups, packing_stats = packed
        validate_exact_packing_stats(packing_stats)
        admitted_packing_sha256 = validate_admitted_packing(
            production_admission_receipt, packing_stats
        )
        for snapshot in exact_input_snapshots.values():
            snapshot.verify_bytes()
        for snapshot in exact_source_snapshots.values():
            snapshot.verify_bytes()
    else:
        X, Y, groups = packed
        packing_stats = None
        admitted_packing_sha256 = ""
    N = len(X)
    if N == 0:
        raise ValueError("no packed sequences; check data and field names")

    weights = parse_sample_weights(a.sample_weights)
    if weights and not a.group_field:
        raise ValueError("--sample-weights requires --group-field")
    exact_order = None
    exact_plan = None
    preflight = None
    preflight_sha256 = ""
    if a.canonical_exact_budget:
        supervised_per_pack = np.count_nonzero(Y != -1, axis=1)
        exact_order, exact_plan = make_exact_budget_plan(
            supervised_per_pack,
            a.exact_updates,
            a.batch_size,
            pack_len,
            a.seed,
        )
        for snapshot in exact_input_snapshots.values():
            snapshot.verify_bytes()
        for snapshot in exact_source_snapshots.values():
            snapshot.verify_bytes()
        admission_artifact = publish_exact_bytes(
            Path(a.out) / "production_admission.json",
            exact_input_snapshots["production_admission"].payload,
        )
        reviewed_manifest_artifact = publish_exact_bytes(
            Path(a.out) / "reviewed_source_manifest.json",
            exact_input_snapshots["reviewed_source_manifest"].payload,
        )
        source_bundle_artifact = publish_canonical_json(
            Path(a.out) / "scientific_sources.json",
            scientific_source_bundle(exact_source_snapshots),
        )
        preflight = {
            "admission": {
                "independent": True,
                "path": admission_artifact.path.name,
                "sha256": admission_artifact.sha256,
            },
            "admitted_corpus_contract": canonical_corpus_contract,
            "arm": canonical_arm,
            "audit": EXACT_BUDGET_AUDIT,
            "budget": exact_plan,
            "consumption_custody": EXACT_BUDGET_CONSUMPTION_CUSTODY,
            "init_step": ck.get("step"),
            "inputs": exact_input_bindings,
            "model_config_sha256": hashlib.sha256(
                canonical_json_bytes(cfg.__dict__)
            ).hexdigest(),
            "optimizer": {
                "adam_betas": [0.9, 0.95],
                "adam_weight_decay": 0.0,
                "clip": a.clip,
                "lr_adam": a.lr_adam,
                "lr_muon": a.lr_muon,
                "muon_momentum": 0.95,
                "muon_nesterov": True,
                "muon_newton_schulz_steps": 5,
                "schedule": "warmup_then_cosine_to_0.1x",
                "warmup_updates": a.warmup,
            },
            "packing": packing_stats,
            "packing_admission_sha256": admitted_packing_sha256,
            "phase": "preflight",
            "reviewed_source_manifest": {
                "clean_source_tree": reviewed_source_manifest["clean_source_tree"],
                "path": reviewed_manifest_artifact.path.name,
                "review_status": reviewed_source_manifest["review_status"],
                "reviewed_clean_commit": reviewed_source_manifest[
                    "reviewed_clean_commit"
                ],
                "sha256": reviewed_manifest_artifact.sha256,
            },
            "runtime": {
                "cuda": torch.version.cuda,
                "device": torch.cuda.get_device_name(0),
                "numpy": np.__version__,
                "python": platform.python_version(),
                "tokenizers": tokenizers.__version__,
                "torch": torch.__version__,
            },
            "semantics": {
                "completion_mask": "corpus_native",
                "supervised_token_equality_enforced": False,
                "target_thinning": False,
                "warning": (
                    "Forward-token positions are equalized; supervised-token counts are "
                    "measured but intentionally not altered."
                ),
            },
            "sources": exact_source_bindings,
            "source_bundle": {
                "path": source_bundle_artifact.path.name,
                "sha256": source_bundle_artifact.sha256,
            },
            "status": "admitted",
            "trust_boundary": EXACT_BUDGET_TRUST_BOUNDARY,
        }
        preflight_artifact = publish_canonical_json(
            Path(a.out) / "exact_budget_preflight.json", preflight
        )
        preflight_sha256 = preflight_artifact.sha256
        print(
            f"[sft-exact] preflight={preflight_sha256} updates={a.exact_updates} "
            f"packs={exact_plan['consumed_packs']} "
            f"dropped={exact_plan['dropped_packs_count']} "
            f"forward_positions={exact_plan['forward_token_positions']} "
            f"supervised_tokens={exact_plan['consumed_supervised_tokens']}",
            flush=True,
        )

    model = GPT(cfg).to(device)
    model.load_state_dict(ck["model"])
    if a.freeze_lexicon:
        # head is tied to tok.weight, so freezing this one parameter preserves
        # both the input lexicon and raw output-token geometry.
        model.tok.weight.requires_grad_(False)
    print(
        f"[sft] init from {a.init} (step {ck.get('step')}), "
        f"params {model.num_params() / 1e6:.1f}M, device {device}",
        flush=True,
    )

    replay_prompts, replay_skipped, reference = [], {}, None
    if a.replay_prompts:
        if a.replay_max_tokens > cfg.seq_len:
            raise ValueError("replay max tokens exceeds model context length")
        replay_prompts, replay_skipped = load_replay_prompts(
            a.replay_prompts, tok, a.replay_max_tokens
        )
        reference_ckpt = torch.load(a.reference, map_location="cpu")
        reference_cfg = GPTConfig(**reference_ckpt["cfg"])
        if reference_cfg != cfg:
            raise ValueError("reference checkpoint config differs from init")
        reference = GPT(reference_cfg).to(device).eval()
        reference.load_state_dict(reference_ckpt["model"])
        for parameter in reference.parameters():
            parameter.requires_grad_(False)
    if not a.canonical_exact_budget:
        metadata = {
            "audit": "behavior_preserving_sft_v1",
            "init": a.init,
            "init_sha256": sha256_file(a.init),
            "data": paths,
            "data_sha256": {path: sha256_file(path) for path in paths},
            "freeze_lexicon": a.freeze_lexicon,
            "reference": a.reference,
            "reference_sha256": sha256_file(a.reference) if a.reference else "",
            "replay_prompts": a.replay_prompts,
            "replay_prompts_sha256": (
                sha256_file(a.replay_prompts) if a.replay_prompts else ""
            ),
            "replay_rows": len(replay_prompts),
            "replay_skipped": replay_skipped,
            "replay_weight": a.replay_weight,
            "replay_batch_size": a.replay_batch_size,
            "replay_max_tokens": a.replay_max_tokens,
            "pack_len": pack_len,
            "packed_sequences": int(N),
            "seed": a.seed,
            "sample_weights": weights,
        }
        with open(os.path.join(a.out, "sft_metadata.json"), "w") as sink:
            json.dump(metadata, sink, indent=2, sort_keys=True)
            sink.write("\n")
    raw = model
    if a.compile:
        model = torch.compile(model)
    muon_p, adam_p = split_params(raw)
    opt_muon = Muon(muon_p, lr=a.lr_muon)
    opt_adam = torch.optim.AdamW(
        adam_p, lr=a.lr_adam, betas=(0.9, 0.95), weight_decay=0.0
    )
    steps_per_epoch = N // a.batch_size
    if steps_per_epoch == 0:
        raise ValueError("packed sequence count is smaller than batch size")
    total_steps = a.epochs * steps_per_epoch
    scheduled_steps = a.exact_updates if a.canonical_exact_budget else total_steps

    def lr_at(step):
        if step < a.warmup:
            return step / max(1, a.warmup)
        r = (step - a.warmup) / max(1, scheduled_steps - a.warmup)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * r))  # cosine decay to 0.1

    rng = np.random.default_rng(a.seed)
    t0, step = time.time(), 0
    actual_update_supervised_tokens = []
    actual_packs = 0
    actual_forward_positions = 0
    checkpoint_path = None
    for ep in range(a.epochs):
        if a.canonical_exact_budget:
            order = exact_order
        elif weights:
            order, requested = weighted_epoch_order(rng, groups, a.batch_size, weights)
            print(f"[sft-data] epoch {ep} weighted samples={requested}", flush=True)
        else:
            order = rng.permutation(N)
        for bi in range(0, len(order) - a.batch_size + 1, a.batch_size):
            idx = order[bi : bi + a.batch_size]
            x = torch.from_numpy(X[idx]).to(device)
            y = torch.from_numpy(Y[idx]).to(device)
            batch_supervised_tokens = (
                int(np.count_nonzero(Y[idx] != -1)) if a.canonical_exact_budget else 0
            )
            sc = lr_at(step)
            for g in opt_muon.param_groups:
                g["lr"] = a.lr_muon * sc
            for g in opt_adam.param_groups:
                g["lr"] = a.lr_adam * sc
            opt_muon.zero_grad(set_to_none=True)
            opt_adam.zero_grad(set_to_none=True)
            with torch.autocast(
                "cuda", dtype=torch.bfloat16, enabled=("cuda" in str(device))
            ):
                _, supervised_loss = model(x, y)
                retention_loss = None
                if reference is not None:
                    replay_x, replay_lengths = make_replay_batch(
                        replay_prompts,
                        a.replay_batch_size,
                        a.replay_max_tokens,
                        eos_id,
                        rng,
                        device,
                    )
                    with torch.no_grad():
                        teacher_logits, _ = reference(replay_x)
                    student_logits, _ = model(replay_x)
                    retention_loss = replay_kl(
                        student_logits, teacher_logits, replay_lengths
                    )
                    loss = supervised_loss + a.replay_weight * retention_loss
                else:
                    loss = supervised_loss
            if a.canonical_exact_budget and not bool(torch.isfinite(loss)):
                raise RuntimeError(f"non-finite loss at optimizer update {step}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(raw.parameters(), a.clip)
            opt_muon.step()
            opt_adam.step()
            if a.canonical_exact_budget:
                actual_update_supervised_tokens.append(batch_supervised_tokens)
                actual_packs += len(idx)
                actual_forward_positions += int(x.numel())
            if step % a.log_every == 0:
                message = (
                    f"epoch {ep} step {step}/{scheduled_steps} loss {loss.item():.4f} "
                    f"supervised {supervised_loss.item():.4f} "
                    f"lr {a.lr_muon * sc:.5f} {time.time() - t0:.0f}s"
                )
                if retention_loss is not None:
                    message += f" replay_kl {retention_loss.item():.4f}"
                print(message, flush=True)
            step += 1
        checkpoint = {
            "cfg": cfg.__dict__,
            "model": raw.state_dict(),
            "step": f"sft_ep{ep + 1}",
        }
        if a.canonical_exact_budget:
            checkpoint["factorial_arm"] = canonical_arm
            checkpoint["production_admission_sha256"] = admission_artifact.sha256
            checkpoint["exact_budget_preflight_sha256"] = preflight_sha256
            checkpoint["exact_budget_updates"] = step
        checkpoint_path = Path(a.out) / f"sft_ep{ep + 1}.pt"
        if a.canonical_exact_budget:
            checkpoint_artifact = publish_torch_checkpoint(checkpoint_path, checkpoint)
        else:
            torch.save(checkpoint, checkpoint_path)
        print(f"[sft] saved epoch {ep + 1}", flush=True)
    if a.canonical_exact_budget:
        actual = validate_exact_budget_actual(
            exact_plan,
            actual_update_supervised_tokens,
            actual_packs,
            actual_forward_positions,
        )
        for snapshot in exact_input_snapshots.values():
            snapshot.verify_bytes()
        for snapshot in exact_source_snapshots.values():
            snapshot.verify_bytes()
        checkpoint_artifact.verify_path_identity()
        closed_world_files = sorted(
            [
                "exact_budget_final.json",
                "exact_budget_preflight.json",
                "production_admission.json",
                "reviewed_source_manifest.json",
                "scientific_sources.json",
                checkpoint_path.name,
            ]
        )
        final_receipt = {
            "actual": actual,
            "admission": preflight["admission"],
            "admitted_corpus_contract": canonical_corpus_contract,
            "arm": canonical_arm,
            "audit": EXACT_BUDGET_AUDIT,
            "checkpoint": {
                "bytes": checkpoint_artifact.binding["bytes"],
                "path": checkpoint_path.name,
                "publication": "fsync_private_inode_no_replace_hardlink_v1",
                "sha256": checkpoint_artifact.sha256,
                "stable_descriptor": True,
            },
            "consumption_custody": preflight["consumption_custody"],
            "completion_requires": {
                "closed_world_files": closed_world_files,
                "directory_mode": "0555",
                "file_mode": "0444",
                "permission_mode_boundary": (
                    "accidental_write_seal_not_owner_proof_immutability"
                ),
                "recipient_verification": "rehash_every_closed_world_artifact",
            },
            "inputs": exact_input_bindings,
            "phase": "final",
            "planned": {
                "consumed_pack_order_sha256": exact_plan["consumed_pack_order_sha256"],
                "forward_token_positions": exact_plan["forward_token_positions"],
                "optimizer_updates": exact_plan["optimizer_updates"],
                "packs": exact_plan["consumed_packs"],
                "supervised_tokens": exact_plan["consumed_supervised_tokens"],
                "supervised_tokens_per_update_sha256": exact_plan[
                    "supervised_tokens_per_update_sha256"
                ],
            },
            "preflight": {
                "path": "exact_budget_preflight.json",
                "sha256": preflight_sha256,
            },
            "reviewed_source_manifest": preflight["reviewed_source_manifest"],
            "semantics": preflight["semantics"],
            "sources": exact_source_bindings,
            "source_bundle": preflight["source_bundle"],
            "status": "complete",
            "trust_boundary": EXACT_BUDGET_TRUST_BOUNDARY,
        }
        final_artifact = publish_canonical_json(
            Path(a.out) / "exact_budget_final.json", final_receipt
        )
        seal_output_directory(
            a.out,
            [
                admission_artifact,
                reviewed_manifest_artifact,
                source_bundle_artifact,
                preflight_artifact,
                checkpoint_artifact,
                final_artifact,
            ],
        )
        final_sha256 = final_artifact.sha256
        print(f"[sft-exact] final={final_sha256} status=complete", flush=True)
        for artifact in (
            admission_artifact,
            reviewed_manifest_artifact,
            source_bundle_artifact,
            preflight_artifact,
            checkpoint_artifact,
            final_artifact,
        ):
            artifact.close()
    print(f"[sft] done {step} steps in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
