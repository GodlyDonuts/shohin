#!/usr/bin/env python3
"""Fit and evaluate a grammar-gated carry motor on a frozen DRS model.

The motor is deliberately narrow: it can change only the token-0 and token-1
logits at the exact canonical ``;c=`` response boundary.  It never receives a
solver value during generation and every base-model parameter stays frozen.
"""

from __future__ import annotations

import argparse
import collections
import contextlib
import hashlib
import io
import json
import math
import os
import random
import re
import stat as stat_module
import subprocess
import tempfile
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer

from digitwise_controller import rollout_episode
from digitwise_protocol import (
    apply_microstep,
    canonical_state,
    initial_state,
    microstep_prompt,
    parse_state,
    state_answer,
)
from eval_suite import decode_tokens, has_complete_final_answer
from model import GPT, GPTConfig
from probe_digitwise_workspace import field_prefix, token_for_digit


EXPECTED_CHECKPOINT_SHA256 = (
    "d79e9df26caecb9801118d1bf68bd7b85381a06b256f23478acffe40a2108459"
)
EXPECTED_EPISODES_SHA256 = (
    "89ce11b36ff2f56e83cda72a1f07b1a90f4a3dc3803c69db2779a27219712646"
)
EXPECTED_TOKENIZER_SHA256 = (
    "87532df5c121753de3b29194e1f9e3de47986d3f5359548fdf93606773a233d4"
)
EXPECTED_CYCLE_SHA256 = (
    "0b927fee009de5e5cf87971ecaf390c716d6d9acb5644cabe3c176f6da9d4e7a"
)
FIT_SEED = 20260717
FIT_WIDTHS = (4, 6)
FIT_STYLES = ("core", "heldout")
FIT_QUOTA = 512
CANONICAL_UPDATES = 2000
CANONICAL_BATCH = 512
CANONICAL_LR = 3e-3
CANONICAL_WEIGHT_DECAY = 1e-4
CANONICAL_PER_REGIME = 100
CANONICAL_MAX_NEW = 96
CANONICAL_EXTRACT_BATCH = 1
RANK = 8
CARRY_SITE = re.compile(r"^dws:op=(add|sub);w=([1-9]\d*);p=(0|[1-9]\d*);c=$")
SCIENTIFIC_SOURCE_PATHS = (
    "R12_CAUSAL_CARRY_MOTOR_PREREG.md",
    "train/causal_carry_motor.py",
    "train/test_causal_carry_motor.py",
    "train/jobs/causal_carry_motor.sbatch",
    "train/digitwise_controller.py",
    "train/digitwise_protocol.py",
    "train/eval_suite.py",
    "train/model.py",
    "train/probe_digitwise_workspace.py",
)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class BoundInput:
    """Immutable private byte snapshot plus path-identity verification."""

    def __init__(self, path):
        self.path = Path(path).resolve()
        source = open(self.path, "rb")
        stat = os.fstat(source.fileno())
        if not stat_module.S_ISREG(stat.st_mode):
            source.close()
            raise ValueError("bound input is not a regular file: {}".format(self.path))
        payload = source.read()
        final_stat = os.fstat(source.fileno())
        source.close()
        if (
            stat.st_dev,
            stat.st_ino,
            stat.st_size,
            stat.st_mtime_ns,
            stat.st_ctime_ns,
        ) != (
            final_stat.st_dev,
            final_stat.st_ino,
            final_stat.st_size,
            final_stat.st_mtime_ns,
            final_stat.st_ctime_ns,
        ):
            raise RuntimeError("input changed while snapshotting: {}".format(self.path))
        self.identity = (
            stat.st_dev,
            stat.st_ino,
            stat.st_size,
            stat.st_mtime_ns,
            stat.st_ctime_ns,
        )
        self.handle = io.BytesIO(payload)
        self.sha256 = self._hash_descriptor()

    def _hash_descriptor(self):
        digest = hashlib.sha256()
        self.handle.seek(0)
        for block in iter(lambda: self.handle.read(1024 * 1024), b""):
            digest.update(block)
        self.handle.seek(0)
        return digest.hexdigest()

    def bytes(self):
        self.handle.seek(0)
        value = self.handle.read()
        self.handle.seek(0)
        return value

    def text(self):
        return self.bytes().decode()

    def verify_path(self):
        stat = os.stat(self.path, follow_symlinks=False)
        observed = (
            stat.st_dev,
            stat.st_ino,
            stat.st_size,
            stat.st_mtime_ns,
            stat.st_ctime_ns,
        )
        if observed != self.identity or self._hash_descriptor() != self.sha256:
            raise RuntimeError(
                "bound input changed or path was substituted: {}".format(self.path)
            )

    def close(self):
        self.handle.close()


def bind_frozen_inputs(checkpoint, episodes, tokenizer, cycle):
    paths = {
        "checkpoint": checkpoint,
        "episodes": episodes,
        "tokenizer": tokenizer,
        "cycle": cycle,
    }
    root = Path(__file__).resolve().parents[1]
    for name in SCIENTIFIC_SOURCE_PATHS:
        paths["source:{}".format(name)] = root / name
    bound = {name: BoundInput(path) for name, path in paths.items()}
    observed = {name: item.sha256 for name, item in bound.items()}
    expected = {
        "checkpoint": EXPECTED_CHECKPOINT_SHA256,
        "episodes": EXPECTED_EPISODES_SHA256,
        "tokenizer": EXPECTED_TOKENIZER_SHA256,
        "cycle": EXPECTED_CYCLE_SHA256,
    }
    if {name: observed[name] for name in expected} != expected:
        for item in bound.values():
            item.close()
        raise ValueError("frozen input hash mismatch: {}".format(observed))
    return bound, observed


def source_manifest_sha256(observed):
    sources = {
        name.removeprefix("source:"): digest
        for name, digest in observed.items()
        if name.startswith("source:")
    }
    return stable_json_sha256(sources)


def verify_reviewed_source_commit(source_commit, source_hashes):
    root = Path(__file__).resolve().parents[1]
    head = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    if source_commit != head:
        raise ValueError("reviewed source commit is not checked out")
    for name, digest in source_hashes.items():
        payload = subprocess.check_output(
            ["git", "-C", str(root), "show", "{}:{}".format(source_commit, name)]
        )
        if hashlib.sha256(payload).hexdigest() != digest:
            raise ValueError(
                "scientific source differs from reviewed commit: {}".format(name)
            )


def validate_source_contract(args, observed, canonical):
    sources = {
        name.removeprefix("source:"): digest
        for name, digest in observed.items()
        if name.startswith("source:")
    }
    manifest = source_manifest_sha256(observed)
    if canonical and (not args.source_commit or not args.source_manifest_sha256):
        raise SystemExit("canonical run requires frozen source commit and manifest")
    if args.source_manifest_sha256 and args.source_manifest_sha256 != manifest:
        raise ValueError("scientific source manifest mismatch")
    if canonical:
        verify_reviewed_source_commit(args.source_commit, sources)
    return {
        "git_commit": args.source_commit,
        "manifest_sha256": manifest,
    }


def verify_bound_inputs(bound):
    for item in bound.values():
        item.verify_path()


def close_bound_inputs(bound):
    for item in bound.values():
        item.close()


def stable_json_sha256(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def tensor_state_sha256(state):
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode() + b"\0")
        digest.update(str(tensor.dtype).encode() + b"\0")
        digest.update(json.dumps(list(tensor.shape)).encode() + b"\0")
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


class CarryMotor(nn.Module):
    """A rank-8 nonlinear motor producing deltas for only carry tokens 0/1."""

    def __init__(self, d_model, rank=RANK):
        super().__init__()
        self.d_model = int(d_model)
        self.rank = int(rank)
        if self.d_model <= 0 or self.rank <= 0:
            raise ValueError("motor dimensions must be positive")
        self.down = nn.Linear(self.d_model, self.rank, bias=True)
        self.up = nn.Linear(self.rank, 2, bias=True)
        nn.init.normal_(self.down.weight, mean=0.0, std=1.0 / math.sqrt(self.d_model))
        nn.init.zeros_(self.down.bias)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, hidden):
        return self.up(F.silu(self.down(hidden.float())))

    def parameter_count(self):
        return sum(parameter.numel() for parameter in self.parameters())


def dws_prompt_state(prompt):
    """Return the state only when the entire prompt is one canonical DWS prompt."""
    candidates = []
    for line in str(prompt).splitlines():
        for marker in ("State: ", "Machine record: "):
            if line.startswith(marker):
                candidates.append(parse_state(line[len(marker) :]))
    if len(candidates) != 1:
        return None
    state = candidates[0]
    if state is None:
        return None
    if prompt not in (
        microstep_prompt(state, style="core"),
        microstep_prompt(state, style="heldout"),
    ):
        return None
    return state


def is_carry_site(prompt, response_prefix):
    """Recognize the next canonical carry field for an actual DWS prompt."""
    state = dws_prompt_state(prompt)
    match = CARRY_SITE.fullmatch(str(response_prefix))
    if state is None or match is None:
        return False
    operation, width, position = match.groups()
    return (
        operation == state["op"]
        and int(width) == int(state["w"])
        and int(position) == int(state["p"]) + 1
        and int(position) <= int(width)
    )


class CarryRouter:
    """Count one grammar opportunity and at most one motor fire per response."""

    def __init__(self, prompt, motor_present):
        self.prompt = str(prompt)
        self.motor_present = bool(motor_present)
        self.site_count = 0
        self.fire_count = 0

    def observe(self, response_prefix):
        site = self.site_count == 0 and is_carry_site(self.prompt, response_prefix)
        if site:
            self.site_count += 1
        active = site and self.motor_present
        if active:
            self.fire_count += 1
        return active


def apply_motor_logits(logits, hidden, motor, zero_id, one_id, active):
    """Return logits with at most token 0/1 changed; gate-off is exact identity."""
    if not active or motor is None:
        return logits
    if (
        zero_id == one_id
        or min(zero_id, one_id) < 0
        or max(zero_id, one_id) >= logits.shape[-1]
    ):
        raise ValueError("invalid carry token ids")
    delta = motor(hidden)
    if delta.shape != logits.shape[:-1] + (2,):
        raise ValueError("motor delta shape mismatch")
    adjusted = logits.clone()
    adjusted[..., zero_id] = adjusted[..., zero_id] + delta[..., 0].to(logits.dtype)
    adjusted[..., one_id] = adjusted[..., one_id] + delta[..., 1].to(logits.dtype)
    return adjusted


def full_vocab_motor_loss(hidden, base01, other_lse, targets, motor):
    """Full-vocabulary CE after the exact deployed logit-dtype arithmetic."""
    delta = motor(hidden).to(base01.dtype)
    adjusted = (base01 + delta).float()
    carry_lse = torch.logsumexp(adjusted, dim=-1)
    total_lse = torch.logaddexp(other_lse.float(), carry_lse)
    target_logits = adjusted.gather(1, targets.long().unsqueeze(1)).squeeze(1)
    return (total_lse - target_logits).mean()


def _episode_states(episode):
    state = parse_state(episode["initial_state"])
    if state is None:
        raise ValueError("invalid initial state in {}".format(episode["id"]))
    for line in episode["expected_states"]:
        next_state = parse_state(line)
        if next_state is None or apply_microstep(state) != next_state:
            raise ValueError("invalid expected state in {}".format(episode["id"]))
        yield state, next_state
        state = next_state


def heldout_prompt_hashes(episodes_text):
    hashes = set()
    for raw in str(episodes_text).splitlines():
        if not raw.strip():
            continue
        outer = json.loads(raw)
        for episode in (outer, outer["counterfactual"]):
            for state, _ in _episode_states(episode):
                for style in FIT_STYLES:
                    prompt = microstep_prompt(state, style=style)
                    hashes.add(hashlib.sha256(prompt.encode()).hexdigest())
    return hashes


def _fit_key(state, next_state, style):
    return (
        state["op"],
        int(state["w"]),
        int(state["p"]),
        str(style),
        int(state["c"]),
        int(next_state["c"]),
    )


def _all_fit_keys():
    keys = set()
    for operation in ("add", "sub"):
        for width in FIT_WIDTHS:
            for position in range(width):
                # A valid nonnegative subtraction cannot borrow beyond its final digit.
                # That one-class terminal is held out rather than creating a position shortcut.
                if operation == "sub" and position == width - 1:
                    continue
                currents = (0,) if position == 0 else (0, 1)
                for style in FIT_STYLES:
                    for current in currents:
                        for target in (0, 1):
                            keys.add(
                                (
                                    operation,
                                    width,
                                    position,
                                    style,
                                    current,
                                    target,
                                )
                            )
    return keys


def generate_fit_rows(tokenizer, episodes_text, seed=FIT_SEED, quota=FIT_QUOTA):
    """Generate a balanced, duplicate-free solver-labelled motor board."""
    if quota <= 0:
        raise ValueError("quota must be positive")
    rng = random.Random(int(seed))
    forbidden_prompts = heldout_prompt_hashes(episodes_text)
    wanted = _all_fit_keys()
    counts = collections.Counter()
    position_counts = collections.Counter()
    rows, prefix_hashes = [], set()
    attempts = 0
    while any(counts[key] < quota for key in wanted):
        attempts += 1
        if attempts > quota * 2000:
            missing = {
                str(key): quota - counts[key] for key in wanted if counts[key] < quota
            }
            raise RuntimeError("could not fill motor strata: {}".format(missing))
        width = rng.choice(FIT_WIDTHS)
        operation = rng.choice(("add", "sub"))
        limit = 10**width
        left, right = rng.randrange(limit), rng.randrange(limit)
        if operation == "sub" and left < right:
            left, right = right, left
        state = initial_state(operation, left, right, width)
        episode_states = []
        while not state["z"]:
            next_state = apply_microstep(state)
            episode_states.append((state, next_state))
            state = next_state
        rng.shuffle(episode_states)
        for state, next_state in episode_states:
            if state["op"] == "sub" and int(state["p"]) == int(state["w"]) - 1:
                continue
            styles = list(FIT_STYLES)
            rng.shuffle(styles)
            for style in styles:
                key = _fit_key(state, next_state, style)
                if counts[key] >= quota:
                    continue
                prompt = microstep_prompt(state, style=style)
                prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
                if prompt_hash in forbidden_prompts:
                    continue
                prefix, target = field_prefix(prompt, next_state, "carry")
                prefix_ids, target_id = token_for_digit(tokenizer, prefix, target)
                prompt_ids = tokenizer.encode(prompt).ids
                if prefix_ids[: len(prompt_ids)] != prompt_ids:
                    raise RuntimeError(
                        "fit prefix does not preserve the prompt token boundary"
                    )
                prefix_hash = hashlib.sha256(
                    b"".join(int(token).to_bytes(4, "little") for token in prefix_ids)
                ).hexdigest()
                if prefix_hash in prefix_hashes:
                    continue
                prefix_hashes.add(prefix_hash)
                counts[key] += 1
                position_counts[key + (int(state["p"]),)] += 1
                rows.append(
                    {
                        "operation": operation,
                        "width": width,
                        "position": int(state["p"]),
                        "style": style,
                        "current_carry": int(state["c"]),
                        "target": int(target),
                        "target_id": int(target_id),
                        "prompt_sha256": prompt_hash,
                        "prefix_sha256": prefix_hash,
                        "prompt_ids": prompt_ids,
                        "prefix_ids": prefix_ids,
                    }
                )
    expected_rows = len(wanted) * quota
    if (
        len(rows) != expected_rows
        or set(counts) != wanted
        or any(counts[key] != quota for key in wanted)
    ):
        raise RuntimeError("fit board balance failure")
    rows.sort(key=lambda row: row["prefix_sha256"])
    return rows, {
        "seed": int(seed),
        "quota": int(quota),
        "rows": len(rows),
        "attempts": attempts,
        "strata": {"|".join(map(str, key)): counts[key] for key in sorted(counts)},
        "position_counts": {
            "|".join(map(str, key)): value
            for key, value in sorted(position_counts.items())
        },
        "prefix_order_sha256": stable_json_sha256(
            [row["prefix_sha256"] for row in rows]
        ),
        "token_length_histogram": dict(
            sorted(collections.Counter(len(row["prefix_ids"]) for row in rows).items())
        ),
        "prompt_length_histogram": dict(
            sorted(collections.Counter(len(row["prompt_ids"]) for row in rows).items())
        ),
        "forbidden_prompt_count": len(forbidden_prompts),
    }


def permuted_control_labels(rows, seed=FIT_SEED + 1):
    """Permute labels inside nuisance strata while preserving exact counts."""
    groups = collections.defaultdict(list)
    for index, row in enumerate(rows):
        nuisance = (
            row["operation"],
            row["width"],
            row["position"],
            row["style"],
            row["current_carry"],
        )
        groups[nuisance].append(index)
    result = [int(row["target"]) for row in rows]
    rng = random.Random(int(seed))
    changed = 0
    for nuisance in sorted(groups):
        indices = groups[nuisance]
        labels = [result[index] for index in indices]
        before = collections.Counter(labels)
        rng.shuffle(labels)
        if labels == [result[index] for index in indices]:
            labels = labels[1:] + labels[:1]
        if collections.Counter(labels) != before:
            raise RuntimeError("control label balance changed")
        for index, label in zip(indices, labels):
            changed += int(result[index] != label)
            result[index] = label
    if changed < len(rows) // 3:
        raise RuntimeError("too few control labels changed: {}".format(changed))
    return result, {
        "seed": int(seed),
        "changed": changed,
        "changed_rate": changed / len(rows),
        "labels_sha256": stable_json_sha256(result),
    }


def load_model(source, device):
    if isinstance(source, BoundInput):
        source.handle.seek(0)
        checkpoint = torch.load(source.handle, map_location="cpu")
        source.handle.seek(0)
    elif isinstance(source, (bytes, bytearray)):
        checkpoint = torch.load(io.BytesIO(source), map_location="cpu")
    else:
        checkpoint = torch.load(source, map_location="cpu")
    model = GPT(GPTConfig(**checkpoint["cfg"])).eval()
    model.load_state_dict(checkpoint["model"])
    if int(model.cfg.n_loop) != 1:
        raise ValueError("carry motor requires n_loop=1")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return checkpoint, model.to(device)


def _autocast(device):
    if str(device).startswith("cuda"):
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


@torch.no_grad()
def extract_frozen_features(
    model,
    rows,
    zero_id,
    one_id,
    device,
    batch_size=CANONICAL_EXTRACT_BATCH,
    store_other_logits=False,
):
    """Extract residuals through the exact cached path used by generation."""
    if batch_size <= 0:
        raise ValueError("batch size must be positive")
    groups = collections.defaultdict(list)
    for index, row in enumerate(rows):
        prompt_ids = row["prompt_ids"]
        prefix_ids = row["prefix_ids"]
        if prefix_ids[: len(prompt_ids)] != prompt_ids or len(prefix_ids) <= len(
            prompt_ids
        ):
            raise ValueError("row does not have a valid prompt/prefix boundary")
        groups[(len(prompt_ids), len(prefix_ids))].append((index, row))
    hidden = torch.empty((len(rows), model.cfg.d_model), dtype=torch.float32)
    base01 = None
    other_lse = torch.empty(len(rows), dtype=torch.float32)
    other_max = torch.empty(len(rows), dtype=torch.float32)
    other_max_token_id = torch.empty(len(rows), dtype=torch.long)
    other_logits = None
    other_token_ids = None
    labels = torch.empty(len(rows), dtype=torch.long)
    for lengths, group in sorted(groups.items()):
        prompt_length, prefix_length = lengths
        for start in range(0, len(group), batch_size):
            batch = group[start : start + batch_size]
            prompts = torch.tensor(
                [row["prompt_ids"] for _, row in batch], dtype=torch.long, device=device
            )
            continuations = torch.tensor(
                [row["prefix_ids"][prompt_length:] for _, row in batch],
                dtype=torch.long,
                device=device,
            )
            captured = {}

            def hook(_module, _inputs, output):
                captured["hidden"] = output[0][:, -1, :].detach()

            handle = model.blocks[-1].register_forward_hook(hook)
            try:
                with _autocast(device):
                    logits, cache = model(prompts, return_cache=True, pos=0)
                    position = prompt_length
                    for column in range(continuations.shape[1]):
                        logits, cache = model(
                            continuations[:, column : column + 1],
                            cache=cache,
                            pos=position,
                            return_cache=True,
                        )
                        position += 1
            finally:
                handle.remove()
            if "hidden" not in captured:
                raise RuntimeError("final residual hook did not fire")
            last = logits[:, -1, :]
            expected_dtype = (
                torch.bfloat16 if str(device).startswith("cuda") else torch.float32
            )
            if last.dtype != expected_dtype:
                raise RuntimeError(
                    "unexpected deployment logit dtype: {} != {}".format(
                        last.dtype, expected_dtype
                    )
                )
            if base01 is None:
                base01 = torch.empty((len(rows), 2), dtype=last.dtype)
            elif base01.dtype != last.dtype:
                raise RuntimeError("deployment logit dtype changed during extraction")
            keep = torch.ones(last.shape[-1], dtype=torch.bool, device=last.device)
            keep[[zero_id, one_id]] = False
            batch_other_ids = torch.arange(last.shape[-1], device=last.device)[keep]
            if other_token_ids is None:
                other_token_ids = batch_other_ids.cpu()
            elif not torch.equal(other_token_ids, batch_other_ids.cpu()):
                raise RuntimeError("other-token ordering changed during extraction")
            positions = [index for index, _ in batch]
            hidden[positions] = captured["hidden"].float().cpu()
            base01[positions] = last[:, [zero_id, one_id]].cpu()
            batch_other = last[:, keep]
            batch_other_float = batch_other.float()
            other_lse[positions] = torch.logsumexp(batch_other_float, dim=-1).cpu()
            max_values, max_indices = batch_other.max(dim=-1)
            other_max[positions] = max_values.float().cpu()
            other_max_token_id[positions] = batch_other_ids[max_indices].cpu()
            if store_other_logits:
                if other_logits is None:
                    other_logits = torch.empty(
                        (len(rows), batch_other.shape[-1]), dtype=batch_other.dtype
                    )
                other_logits[positions] = batch_other.cpu()
            labels[positions] = torch.tensor(
                [int(row["target"]) for _, row in batch], dtype=torch.long
            )
        print(
            "[carry-motor] extracted prompt={} prefix={} rows={}".format(
                prompt_length, prefix_length, len(group)
            ),
            flush=True,
        )
    if base01 is None or other_token_ids is None:
        raise RuntimeError("feature extraction produced no rows")
    return {
        "hidden": hidden,
        "base01": base01,
        "other_lse": other_lse,
        "other_max": other_max,
        "other_max_token_id": other_max_token_id,
        "other_logits": other_logits,
        "other_token_ids": other_token_ids,
        "zero_id": int(zero_id),
        "one_id": int(one_id),
        "deployment_logit_dtype": str(base01.dtype),
        "labels": labels,
    }


def _batch_schedule(size, batch_size, updates, seed):
    if size < batch_size:
        raise ValueError("feature board is smaller than one batch")
    generator = torch.Generator().manual_seed(int(seed))
    schedule, digest = [], hashlib.sha256()
    while len(schedule) < updates:
        permutation = torch.randperm(size, generator=generator)
        for start in range(0, size - batch_size + 1, batch_size):
            batch = permutation[start : start + batch_size]
            digest.update(batch.to(torch.int32).numpy().tobytes())
            schedule.append(batch)
            if len(schedule) == updates:
                return schedule, digest.hexdigest()
    raise AssertionError("unreachable")


def fit_motor(
    features,
    labels,
    initial_state,
    device,
    updates=CANONICAL_UPDATES,
    batch_size=CANONICAL_BATCH,
    lr=CANONICAL_LR,
    weight_decay=CANONICAL_WEIGHT_DECAY,
    seed=FIT_SEED,
):
    motor = CarryMotor(features["hidden"].shape[1], RANK).to(device)
    motor.load_state_dict(initial_state)
    optimizer = torch.optim.AdamW(motor.parameters(), lr=lr, weight_decay=weight_decay)
    schedule, schedule_sha256 = _batch_schedule(len(labels), batch_size, updates, seed)
    x = features["hidden"].to(device)
    base01 = features["base01"].to(device)
    other_lse = features["other_lse"].to(device)
    targets = torch.as_tensor(labels, dtype=torch.long, device=device)
    losses = []
    for update, batch_cpu in enumerate(schedule, 1):
        batch = batch_cpu.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = full_vocab_motor_loss(
            x[batch], base01[batch], other_lse[batch], targets[batch], motor
        )
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite motor loss")
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        if update == 1 or update % 200 == 0 or update == updates:
            print(
                "[carry-motor] update={}/{} loss={:.6f}".format(
                    update, updates, losses[-1]
                ),
                flush=True,
            )
    return motor.cpu(), {
        "updates": updates,
        "batch_size": batch_size,
        "lr": lr,
        "weight_decay": weight_decay,
        "schedule_sha256": schedule_sha256,
        "first_loss": losses[0],
        "final_loss": losses[-1],
        "min_loss": min(losses),
    }


@torch.no_grad()
def feature_metrics(features, labels, motor=None):
    adjusted = features["base01"]
    if motor is not None:
        adjusted = adjusted + motor(features["hidden"]).to(adjusted.dtype)
    zero_id = int(features["zero_id"])
    one_id = int(features["one_id"])
    zero_wins = (adjusted[:, 0] > adjusted[:, 1]) | (
        (adjusted[:, 0] == adjusted[:, 1]) & (zero_id < one_id)
    )
    carry_predictions = torch.where(
        zero_wins,
        torch.zeros(len(adjusted), dtype=torch.long),
        torch.ones(len(adjusted), dtype=torch.long),
    )
    targets = torch.as_tensor(labels, dtype=torch.long)
    target_logits = adjusted.gather(1, targets[:, None]).squeeze(1)
    other_carry_logits = adjusted.gather(1, (1 - targets)[:, None]).squeeze(1)
    target_token_ids = torch.where(targets == 0, zero_id, one_id)
    other_carry_token_ids = torch.where(targets == 0, one_id, zero_id)
    other_max = features["other_max"].to(target_logits.dtype)
    other_max_token_ids = features["other_max_token_id"].long()
    carry_beats_target = (other_carry_logits > target_logits) | (
        (other_carry_logits == target_logits)
        & (other_carry_token_ids < target_token_ids)
    )
    vocab_beats_target = (other_max > target_logits) | (
        (other_max == target_logits) & (other_max_token_ids < target_token_ids)
    )
    global_correct = ~(carry_beats_target | vocab_beats_target)
    result = {
        "carry_pair_correct": int((carry_predictions == targets).sum()),
        "carry_pair_accuracy": float((carry_predictions == targets).float().mean()),
        "global_correct": int(global_correct.sum()),
        "global_accuracy": float(global_correct.float().mean()),
        "rows": len(targets),
        "prediction_ones": int(carry_predictions.sum()),
        "global_rank_available": features.get("other_logits") is not None,
    }
    if features.get("other_logits") is not None:
        rank_sum = rank_max = 0
        rank_above_65 = 0
        other_logits = features["other_logits"]
        other_token_ids = features["other_token_ids"].long()
        for start in range(0, len(targets), 256):
            stop = min(start + 256, len(targets))
            target = target_logits[start:stop]
            target_ids = target_token_ids[start:stop]
            carry_logits = other_carry_logits[start:stop]
            carry_ids = other_carry_token_ids[start:stop]
            ranks = (
                1
                + (
                    (carry_logits > target)
                    | ((carry_logits == target) & (carry_ids < target_ids))
                ).long()
                + (
                    (other_logits[start:stop] > target[:, None])
                    | (
                        (other_logits[start:stop] == target[:, None])
                        & (other_token_ids[None, :] < target_ids[:, None])
                    )
                ).sum(dim=-1)
            )
            rank_sum += int(ranks.sum())
            rank_max = max(rank_max, int(ranks.max()))
            rank_above_65 += int((ranks > 65).sum())
        result.update(
            {
                "mean_target_rank_global": rank_sum / len(targets),
                "max_target_rank_global": rank_max,
                "target_rank_above_65": rank_above_65,
            }
        )
    return result


def _atomic_bytes(path, payload, mode=0o444):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError("refusing to overwrite {}".format(path))
    fd, temporary = tempfile.mkstemp(prefix=".{}-".format(path.name), dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as sink:
            sink.write(payload)
            sink.flush()
            os.fsync(sink.fileno())
        os.chmod(temporary, mode)
        os.link(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def atomic_json(path, value):
    _atomic_bytes(path, (json.dumps(value, indent=2, sort_keys=True) + "\n").encode())


def atomic_torch(path, value):
    path = Path(path)
    if path.exists():
        raise FileExistsError("refusing to overwrite {}".format(path))
    with tempfile.TemporaryDirectory(dir=path.parent) as directory:
        temporary = Path(directory) / "artifact.pt"
        torch.save(value, temporary)
        with open(temporary, "rb") as source:
            _atomic_bytes(path, source.read())


def prepare_output(path, canonical):
    path = Path(path)
    parent = path.parent
    if path.exists() or Path(str(path) + ".json").exists():
        raise FileExistsError("refusing existing output")
    if canonical:
        if parent.exists():
            raise FileExistsError(
                "canonical output directory must be new: {}".format(parent)
            )
        parent.mkdir(parents=False, mode=0o700)
    else:
        parent.mkdir(parents=True, exist_ok=True)
    return path


def seal_output_directory(path):
    parent = Path(path).parent
    for child in parent.iterdir():
        if child.is_file():
            os.chmod(child, 0o444)
    directory_fd = os.open(parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    os.chmod(parent, 0o555)


@torch.no_grad()
def motor_generate(model, motor, tokenizer, prompt, device, max_new=96):
    """Greedy cached generation with one grammar-only carry-motor site."""
    cap = int(model.cfg.seq_len)
    prompt_ids = tokenizer.encode(prompt).ids[-cap:]
    eos_id = tokenizer.token_to_id("<|endoftext|>")
    zero_id = token_for_digit(tokenizer, "x=", "0")[1]
    one_id = token_for_digit(tokenizer, "x=", "1")[1]
    generated = []
    router = CarryRouter(prompt, motor is not None)
    captured = {}

    def hook(_module, _inputs, output):
        captured["hidden"] = output[0][:, -1, :].detach()

    handle = model.blocks[-1].register_forward_hook(hook)
    try:
        with _autocast(device):
            logits, cache = model(
                torch.tensor([prompt_ids], dtype=torch.long, device=device),
                return_cache=True,
                pos=0,
            )
        position = len(prompt_ids)
        for _ in range(max_new):
            response = decode_tokens(tokenizer, generated, skip_special_tokens=False)
            active = router.observe(response)
            selected = apply_motor_logits(
                logits[:, -1, :], captured["hidden"], motor, zero_id, one_id, active
            )
            next_id = int(selected.argmax(dim=-1).item())
            generated.append(next_id)
            text = decode_tokens(tokenizer, generated, skip_special_tokens=False)
            if next_id == eos_id or position >= cap or has_complete_final_answer(text):
                break
            with _autocast(device):
                logits, cache = model(
                    torch.tensor([[next_id]], dtype=torch.long, device=device),
                    cache=cache,
                    pos=position,
                    return_cache=True,
                )
            position += 1
    finally:
        handle.remove()
    if generated and generated[-1] == eos_id:
        generated = generated[:-1]
    return (
        decode_tokens(tokenizer, generated, skip_special_tokens=False),
        router.site_count,
        router.fire_count,
    )


def read_selected_episodes(episodes_text, per_regime):
    selected, counts = [], collections.Counter()
    for raw in str(episodes_text).splitlines():
        if not raw.strip():
            continue
        episode = json.loads(raw)
        if counts[episode["split"]] < per_regime:
            selected.append(episode)
            counts[episode["split"]] += 1
    if not selected or any(value != per_regime for value in counts.values()):
        raise ValueError("development episode selection is incomplete")
    return selected


def development_feature_rows(episodes, tokenizer):
    rows = []
    for outer in episodes:
        for branch in (outer, outer["counterfactual"]):
            for transition, (state, next_state) in enumerate(_episode_states(branch)):
                prompt = microstep_prompt(state, style=branch["prompt_style"])
                prefix, target = field_prefix(prompt, next_state, "carry")
                prompt_ids = tokenizer.encode(prompt).ids
                prefix_ids, target_id = token_for_digit(tokenizer, prefix, target)
                if prefix_ids[: len(prompt_ids)] != prompt_ids:
                    raise RuntimeError("development prefix boundary mismatch")
                rows.append(
                    {
                        "operation": state["op"],
                        "width": int(state["w"]),
                        "position": int(state["p"]),
                        "style": branch["prompt_style"],
                        "current_carry": int(state["c"]),
                        "target": int(target),
                        "target_id": int(target_id),
                        "prompt_ids": prompt_ids,
                        "prefix_ids": prefix_ids,
                        "episode_id": branch["id"],
                        "regime": outer["split"],
                        "transition": transition,
                    }
                )
    return rows


def fit_linear_diagnostic(features, labels, device, seed=FIT_SEED + 2):
    """Train/test diagnostic only; this probe is never inserted into decoding."""
    size = len(labels)
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(size, generator=generator)
    split = int(size * 0.8)
    train_indices, test_indices = order[:split], order[split:]
    classifier = nn.Linear(features["hidden"].shape[1], 2).to(device)
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=1e-2, weight_decay=1e-4)
    x = features["hidden"].to(device)
    y = torch.as_tensor(labels, dtype=torch.long, device=device)
    schedule, schedule_sha256 = _batch_schedule(split, min(512, split), 300, seed)
    train_indices_device = train_indices.to(device)
    for batch in schedule:
        selected = train_indices_device[batch.to(device)]
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(classifier(x[selected]), y[selected])
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        prediction = classifier(x[test_indices.to(device)]).argmax(dim=-1)
        correct = int((prediction == y[test_indices.to(device)]).sum().cpu())
    return {
        "train_rows": split,
        "test_rows": size - split,
        "test_correct": correct,
        "test_accuracy": correct / (size - split),
        "schedule_sha256": schedule_sha256,
        "claim_boundary": "Diagnostic feature extractability only; not an inference arm.",
    }


def evaluate_cycle(cycle_text, episode_map, model, motor, tokenizer, device, max_new):
    document = json.loads(cycle_text)
    first_exact = second_exact = integrated = site_count = fire_count = 0
    examples = []
    for record in document["records"]:
        state = parse_state(record["counterfactual_state"])
        expected_first = parse_state(record["counterfactual_next"])
        if state is None or expected_first is None:
            raise ValueError("invalid state in frozen cycle artifact")
        episode = episode_map[record["episode_id"]]
        prompt = microstep_prompt(state, style=episode["prompt_style"])
        response, sites, fires = motor_generate(
            model, motor, tokenizer, prompt, device, max_new=max_new
        )
        predicted_first = parse_state(response)
        first_ok = predicted_first == expected_first
        first_exact += int(first_ok)
        site_count += sites
        fire_count += fires
        response_second, second_ok = "", False
        if first_ok and not predicted_first["z"]:
            expected_second = apply_microstep(expected_first)
            second_prompt = microstep_prompt(
                predicted_first, style=episode["prompt_style"]
            )
            response_second, sites, fires = motor_generate(
                model, motor, tokenizer, second_prompt, device, max_new=max_new
            )
            second_ok = parse_state(response_second) == expected_second
            second_exact += int(second_ok)
            site_count += sites
            fire_count += fires
        integrated += int(first_ok and second_ok)
        if len(examples) < 10:
            examples.append(
                {
                    "episode_id": record["episode_id"],
                    "prompt": prompt,
                    "first_response": response,
                    "first_exact": first_ok,
                    "second_response": response_second,
                    "second_exact": second_ok,
                }
            )
    return {
        "records": len(document["records"]),
        "first_exact": first_exact,
        "second_exact_after_first": second_exact,
        "integrated_two_call_exact": integrated,
        "site_opportunities": site_count,
        "motor_fires": fire_count,
        "examples": examples,
    }


def _direct_episode(case_id, mode, operation, left, right, width, advance, style):
    state = initial_state(operation, left, right, width)
    for _ in range(advance):
        state = apply_microstep(state)
    initial = canonical_state(state)
    expected_states = []
    while not state["z"]:
        state = apply_microstep(state)
        expected_states.append(canonical_state(state))
    return {
        "id": case_id,
        "mode": mode,
        "split": "fresh_{}_w{}".format(mode, width),
        "prompt_style": style,
        "initial_state": initial,
        "expected_states": expected_states,
        "expected_answer": state_answer(state),
    }


def fresh_direct_cases():
    """Freeze 12 qualitatively distinct researcher-written interactions."""
    specs = (
        ("complete-add-carry", "complete", "add", 9999, 1, 4, 0, "core"),
        ("complete-sub-borrow", "complete", "sub", 1000, 1, 4, 0, "heldout"),
        ("complete-add-mixed", "complete", "add", 590995, 909, 6, 0, "core"),
        ("complete-sub-mixed", "complete", "sub", 500050, 9999, 6, 0, "heldout"),
        ("terminal-add-carry", "terminal", "add", 9999, 1, 4, 3, "core"),
        ("terminal-sub-borrow", "terminal", "sub", 1000, 1, 4, 3, "heldout"),
        ("source-deleted-add", "source_deleted", "add", 8888, 2222, 4, 2, "core"),
        ("source-deleted-sub", "source_deleted", "sub", 7002, 1999, 4, 2, "heldout"),
        ("state-reuse-add", "state_reuse", "add", 99999999, 1, 8, 3, "core"),
        ("state-reuse-sub", "state_reuse", "sub", 10000000, 1, 8, 3, "heldout"),
        ("review-add", "review", "add", 9999999999, 1, 10, 4, "core"),
        (
            "review-sub",
            "review",
            "sub",
            8000000000,
            1999999999,
            10,
            4,
            "heldout",
        ),
    )
    return [_direct_episode(*spec) for spec in specs]


def _review_prompt(state, previous_response):
    return (
        "Review this proposed local decimal rewrite without trusting it.\n"
        "Current state: {}\nPrevious proposal: {}\n"
        "Return exactly the corrected next canonical dws state line.\nAnswer:"
    ).format(canonical_state(state), str(previous_response).strip())


def evaluate_direct_case(case, ask):
    mode = case["mode"]
    if mode in {"complete", "terminal", "source_deleted"}:
        rollout = rollout_episode(case, ask, prompt_style=case["prompt_style"])
        return {"success": rollout["success"], "rollouts": [rollout]}
    if mode == "state_reuse":
        first = rollout_episode(case, ask, prompt_style=case["prompt_style"])
        second = rollout_episode(case, ask, prompt_style=case["prompt_style"])
        return {
            "success": first["success"] and second["success"],
            "exact_reuse": first == second,
            "rollouts": [first, second],
        }
    if mode == "review":
        state = parse_state(case["initial_state"])
        expected = parse_state(case["expected_states"][0])
        first_prompt = microstep_prompt(state, style=case["prompt_style"])
        first_response = ask(first_prompt)
        reviewed_response = ask(_review_prompt(state, first_response))
        reviewed = parse_state(reviewed_response)
        if reviewed != expected:
            return {
                "success": False,
                "first_prompt": first_prompt,
                "first_response": first_response,
                "review_prompt": _review_prompt(state, first_response),
                "review_response": reviewed_response,
                "rollouts": [],
            }
        remainder = dict(case)
        remainder["initial_state"] = canonical_state(expected)
        remainder["expected_states"] = case["expected_states"][1:]
        rollout = rollout_episode(
            remainder, ask, prompt_style=remainder["prompt_style"]
        )
        return {
            "success": rollout["success"],
            "first_prompt": first_prompt,
            "first_response": first_response,
            "review_prompt": _review_prompt(state, first_response),
            "review_response": reviewed_response,
            "rollouts": [rollout],
        }
    raise ValueError("unknown direct interaction mode: {}".format(mode))


def without_motor_fire_accounting(value):
    if isinstance(value, dict):
        return {
            key: without_motor_fire_accounting(item)
            for key, item in value.items()
            if key != "motor_fires"
        }
    if isinstance(value, list):
        return [without_motor_fire_accounting(item) for item in value]
    return value


def validate_canonical_train_args(args, canonical):
    if not canonical:
        return
    if (
        args.quota != FIT_QUOTA
        or args.updates != CANONICAL_UPDATES
        or args.batch_size != CANONICAL_BATCH
        or args.lr != CANONICAL_LR
        or args.weight_decay != CANONICAL_WEIGHT_DECAY
        or args.extract_batch != CANONICAL_EXTRACT_BATCH
    ):
        raise SystemExit("canonical fit budget is immutable")


def validate_canonical_eval_args(args, canonical):
    if not canonical:
        return
    if (
        args.per_regime != CANONICAL_PER_REGIME
        or args.max_new != CANONICAL_MAX_NEW
        or args.extract_batch != CANONICAL_EXTRACT_BATCH
    ):
        raise SystemExit("canonical evaluation board and decode budget are immutable")
    if not args.motor_sha256:
        raise SystemExit("canonical evaluation requires --motor-sha256")


def validate_motor_bundle(bundle, expected_bindings, current_sources, source_contract):
    if bundle.get("audit") != "causal_carry_motor_fit_v2":
        raise ValueError("unsupported carry-motor bundle audit version")
    if bundle.get("deployment_logit_dtype") not in {
        "torch.bfloat16",
        "torch.float32",
    }:
        raise ValueError("invalid carry-motor deployment logit dtype")
    for name, expected in expected_bindings.items():
        if bundle.get(name) != expected:
            raise ValueError("motor bundle input mismatch: {}".format(name))
    if bundle.get("scientific_source_sha256") != current_sources:
        raise ValueError("motor bundle scientific source mismatch")
    if bundle.get("source_contract") != source_contract:
        raise ValueError("motor bundle source contract mismatch")
    if bundle.get("extract_batch") != CANONICAL_EXTRACT_BATCH:
        raise ValueError("motor bundle extraction batch mismatch")
    for arm in ("treatment", "shuffled"):
        expected_hash = bundle.get("{}_state_sha256".format(arm))
        if tensor_state_sha256(bundle[arm]) != expected_hash:
            raise ValueError("{} state hash mismatch".format(arm))


def validate_artifact_receipt(bound_input, expected_sha256):
    if not expected_sha256:
        raise ValueError("artifact receipt SHA-256 is required")
    if bound_input.sha256 != expected_sha256:
        raise ValueError("motor artifact hash mismatch")


def _train(args):
    canonical = not args.allow_non_cuda
    validate_canonical_train_args(args, canonical)
    if not torch.cuda.is_available() and canonical:
        raise SystemExit("canonical motor fit requires CUDA")
    bound, frozen = bind_frozen_inputs(
        args.ckpt, args.episodes, args.tokenizer, args.cycle
    )
    source_contract = validate_source_contract(args, frozen, canonical)
    out = prepare_output(args.out, canonical)
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    try:
        torch.manual_seed(FIT_SEED)
        if device == "cuda":
            torch.cuda.manual_seed_all(FIT_SEED)
        tokenizer = Tokenizer.from_str(bound["tokenizer"].text())
        zero_id = token_for_digit(tokenizer, "x=", "0")[1]
        one_id = token_for_digit(tokenizer, "x=", "1")[1]
        if tokenizer.decode([zero_id]) != "0" or tokenizer.decode([one_id]) != "1":
            raise ValueError("carry tokens are not standalone decimal digits")
        checkpoint, model = load_model(bound["checkpoint"], device)
        episodes_text = bound["episodes"].text()
        rows, board = generate_fit_rows(tokenizer, episodes_text, FIT_SEED, args.quota)
        board["rows_sha256"] = stable_json_sha256(rows)
        control_labels, control = permuted_control_labels(rows)
        features = extract_frozen_features(
            model, rows, zero_id, one_id, device, batch_size=args.extract_batch
        )
        initial = CarryMotor(model.cfg.d_model, RANK)
        initial_state = {
            name: tensor.detach().clone()
            for name, tensor in initial.state_dict().items()
        }
        initial_sha256 = tensor_state_sha256(initial_state)
        treatment, treatment_fit = fit_motor(
            features,
            features["labels"],
            initial_state,
            device,
            args.updates,
            args.batch_size,
            args.lr,
            args.weight_decay,
            FIT_SEED,
        )
        shuffled, shuffled_fit = fit_motor(
            features,
            control_labels,
            initial_state,
            device,
            args.updates,
            args.batch_size,
            args.lr,
            args.weight_decay,
            FIT_SEED,
        )
        if treatment_fit["schedule_sha256"] != shuffled_fit["schedule_sha256"]:
            raise RuntimeError("learned arms did not receive the same batch schedule")
        linear_diagnostic = fit_linear_diagnostic(features, features["labels"], device)
        verify_bound_inputs(bound)
        source_hashes = {
            name.removeprefix("source:"): digest
            for name, digest in frozen.items()
            if name.startswith("source:")
        }
        bundle = {
            "audit": "causal_carry_motor_fit_v2",
            "base_checkpoint_sha256": frozen["checkpoint"],
            "tokenizer_sha256": frozen["tokenizer"],
            "episodes_sha256": frozen["episodes"],
            "cycle_sha256": frozen["cycle"],
            "scientific_source_sha256": source_hashes,
            "source_contract": source_contract,
            "checkpoint_step": checkpoint.get("step"),
            "d_model": int(model.cfg.d_model),
            "rank": RANK,
            "parameter_count": treatment.parameter_count(),
            "extract_batch": args.extract_batch,
            "deployment_logit_dtype": features["deployment_logit_dtype"],
            "zero_id": int(zero_id),
            "one_id": int(one_id),
            "initial_state_sha256": initial_sha256,
            "treatment": treatment.state_dict(),
            "shuffled": shuffled.state_dict(),
            "treatment_state_sha256": tensor_state_sha256(treatment.state_dict()),
            "shuffled_state_sha256": tensor_state_sha256(shuffled.state_dict()),
            "board": board,
            "control": control,
            "treatment_fit": treatment_fit,
            "shuffled_fit": shuffled_fit,
            "linear_diagnostic": linear_diagnostic,
            "fit_feature_metrics": {
                "base": feature_metrics(features, features["labels"]),
                "treatment": feature_metrics(features, features["labels"], treatment),
                "shuffled_on_true_labels": feature_metrics(
                    features, features["labels"], shuffled
                ),
                "shuffled_on_control_labels": feature_metrics(
                    features, control_labels, shuffled
                ),
            },
            "claim_boundary": (
                "A fit establishes no reasoning result. The motor can alter only carry token logits "
                "at a grammar-defined DWS site; heldout autonomous and confirmation gates remain required."
            ),
        }
        atomic_torch(out, bundle)
        report = {
            key: value
            for key, value in bundle.items()
            if key not in {"treatment", "shuffled"}
        }
        report["bundle_sha256"] = sha256_file(out)
        report["bundle"] = str(out)
        atomic_json(str(out) + ".json", report)
        verify_bound_inputs(bound)
        if canonical:
            seal_output_directory(out)
        print(json.dumps(report["fit_feature_metrics"], sort_keys=True), flush=True)
    finally:
        close_bound_inputs(bound)


def _eval(args):
    canonical = not args.allow_non_cuda
    validate_canonical_eval_args(args, canonical)
    if not torch.cuda.is_available() and canonical:
        raise SystemExit("canonical motor evaluation requires CUDA")
    bound, frozen = bind_frozen_inputs(
        args.ckpt, args.episodes, args.tokenizer, args.cycle
    )
    source_contract = validate_source_contract(args, frozen, canonical)
    motor_bound = BoundInput(args.motor)
    try:
        validate_artifact_receipt(motor_bound, args.motor_sha256)
    except (ValueError, SystemExit):
        close_bound_inputs(bound)
        motor_bound.close()
        raise
    out = prepare_output(args.out, canonical)
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    try:
        checkpoint, model = load_model(bound["checkpoint"], device)
        tokenizer = Tokenizer.from_str(bound["tokenizer"].text())
        motor_bound.handle.seek(0)
        bundle = torch.load(motor_bound.handle, map_location="cpu")
        motor_bound.handle.seek(0)
        expected_bindings = {
            "base_checkpoint_sha256": frozen["checkpoint"],
            "tokenizer_sha256": frozen["tokenizer"],
            "episodes_sha256": frozen["episodes"],
            "cycle_sha256": frozen["cycle"],
        }
        current_sources = {
            name.removeprefix("source:"): digest
            for name, digest in frozen.items()
            if name.startswith("source:")
        }
        validate_motor_bundle(
            bundle, expected_bindings, current_sources, source_contract
        )
        treatment = CarryMotor(bundle["d_model"], bundle["rank"]).to(device).eval()
        treatment.load_state_dict(bundle["treatment"])
        shuffled = CarryMotor(bundle["d_model"], bundle["rank"]).to(device).eval()
        shuffled.load_state_dict(bundle["shuffled"])
        dead = CarryMotor(bundle["d_model"], bundle["rank"]).to(device).eval()
        episodes_text = bound["episodes"].text()
        episodes = read_selected_episodes(episodes_text, args.per_regime)
        all_episode_map = {
            json.loads(raw)["id"]: json.loads(raw)
            for raw in episodes_text.splitlines()
            if raw.strip()
        }
        dev_rows = development_feature_rows(episodes, tokenizer)
        zero_id, one_id = int(bundle["zero_id"]), int(bundle["one_id"])
        dev_features = extract_frozen_features(
            model,
            dev_rows,
            zero_id,
            one_id,
            device,
            batch_size=args.extract_batch,
            store_other_logits=True,
        )
        if (
            bundle.get("deployment_logit_dtype")
            != dev_features["deployment_logit_dtype"]
        ):
            raise RuntimeError("fit/evaluation deployment logit dtype mismatch")
        feature_results = {
            "base": feature_metrics(dev_features, dev_features["labels"]),
            "treatment": feature_metrics(
                dev_features, dev_features["labels"], treatment.cpu()
            ),
            "shuffled": feature_metrics(
                dev_features, dev_features["labels"], shuffled.cpu()
            ),
            "dead": feature_metrics(dev_features, dev_features["labels"], dead.cpu()),
        }
        treatment, shuffled, dead = (
            treatment.to(device),
            shuffled.to(device),
            dead.to(device),
        )
        if feature_results["dead"] != feature_results["base"]:
            raise RuntimeError("dead motor does not collapse to base feature metrics")
        arms = {
            "base": None,
            "dead": dead,
            "treatment": treatment,
            "shuffled": shuffled,
        }
        results = {}
        for arm, motor in arms.items():
            by_regime = collections.defaultdict(lambda: collections.Counter())
            transcripts = []
            for index, episode in enumerate(episodes, 1):
                sites = fires = 0

                def ask(prompt):
                    nonlocal sites, fires
                    response, site_count, fire_count = motor_generate(
                        model, motor, tokenizer, prompt, device, max_new=args.max_new
                    )
                    sites += site_count
                    fires += fire_count
                    return response

                normal = rollout_episode(
                    episode, ask, prompt_style=episode["prompt_style"]
                )
                row = by_regime[episode["split"]]
                row["episodes"] += 1
                row["first_transition_correct"] += int(
                    bool(normal["rows"]) and normal["rows"][0]["correct"]
                )
                row["transition_correct"] += sum(
                    int(item["correct"]) for item in normal["rows"]
                )
                row["transition_attempted"] += len(normal["rows"])
                row["state_closed_loop_correct"] += int(normal["state_closed_loop"])
                row["final_answer_correct"] += int(normal["success"])
                row["site_opportunities"] += sites
                row["motor_fires"] += fires
                if len(transcripts) < 15:
                    transcripts.append(
                        {
                            "id": episode["id"],
                            "regime": episode["split"],
                            "rollout": normal,
                        }
                    )
                if index % 50 == 0:
                    print(
                        "[carry-motor] arm={} {}/{}".format(arm, index, len(episodes)),
                        flush=True,
                    )
            results[arm] = {
                "by_regime": {
                    regime: dict(values) for regime, values in sorted(by_regime.items())
                },
                "transcripts": transcripts,
                "cycle": evaluate_cycle(
                    bound["cycle"].text(),
                    all_episode_map,
                    model,
                    motor,
                    tokenizer,
                    device,
                    args.max_new,
                ),
            }
        direct_results = {}
        for arm, motor in arms.items():
            rows = []
            for case in fresh_direct_cases():
                sites = fires = 0

                def ask(prompt):
                    nonlocal sites, fires
                    response, site_count, fire_count = motor_generate(
                        model, motor, tokenizer, prompt, device, max_new=args.max_new
                    )
                    sites += site_count
                    fires += fire_count
                    return response

                evaluation = evaluate_direct_case(case, ask)
                rows.append(
                    {
                        "id": case["id"],
                        "mode": case["mode"],
                        "expected_answer": case["expected_answer"],
                        "success": evaluation["success"],
                        "site_opportunities": sites,
                        "motor_fires": fires,
                        "evaluation": evaluation,
                    }
                )
            direct_results[arm] = {
                "correct": sum(int(row["success"]) for row in rows),
                "rows": rows,
            }
        if without_motor_fire_accounting(
            results["dead"]
        ) != without_motor_fire_accounting(results["base"]):
            raise RuntimeError("dead motor autonomous evaluation diverged from base")
        if without_motor_fire_accounting(
            direct_results["dead"]
        ) != without_motor_fire_accounting(direct_results["base"]):
            raise RuntimeError("dead motor direct evaluation diverged from base")
        preservation_prompts = (
            "Question: What is 17 + 26?\nAnswer:",
            "Question: Sort 8, 2, 5 in increasing order.\nAnswer:",
            "Question: Convert binary 1011 to decimal.\nAnswer:",
            "Question: Write a Python function that returns the larger input.\nAnswer:",
            "Question: If A implies B and A is true, what follows?\nAnswer:",
            "Question: Continue the sequence 2, 4, 8, 16.\nAnswer:",
            "Question: Spell the reverse of lamp.\nAnswer:",
            "Question: Is 97 prime?\nAnswer:",
            "Question: Compute 12 * 7.\nAnswer:",
            "Question: What is the capital of France?\nAnswer:",
            "Question: Simplify x + x.\nAnswer:",
            "Question: Return only the word blue.\nAnswer:",
        )
        preservation = []
        for prompt in preservation_prompts:
            base_response, base_sites, base_fires = motor_generate(
                model, None, tokenizer, prompt, device, max_new=48
            )
            motor_response, motor_sites, motor_fires = motor_generate(
                model, treatment, tokenizer, prompt, device, max_new=48
            )
            preservation.append(
                {
                    "prompt": prompt,
                    "base_response": base_response,
                    "motor_response": motor_response,
                    "exact_identity": base_response == motor_response,
                    "base_sites": base_sites,
                    "base_fires": base_fires,
                    "motor_sites": motor_sites,
                    "motor_fires": motor_fires,
                }
            )
        if not all(
            row["exact_identity"]
            and row["base_sites"] == 0
            and row["motor_sites"] == 0
            and row["motor_fires"] == 0
            for row in preservation
        ):
            raise RuntimeError("non-DWS preservation identity failed")
        verify_bound_inputs(bound)
        motor_bound.verify_path()
        result = {
            "audit": "causal_carry_motor_development_eval_v2",
            "checkpoint_step": checkpoint.get("step"),
            "frozen_sha256": frozen,
            "source_contract": source_contract,
            "motor": str(motor_bound.path),
            "motor_sha256": motor_bound.sha256,
            "per_regime": args.per_regime,
            "max_new": args.max_new,
            "extract_batch": args.extract_batch,
            "teacher_forced_carry": feature_results,
            "results": results,
            "fresh_direct": direct_results,
            "non_dws_preservation": preservation,
            "claim_boundary": (
                "Development evaluation on an already inspected board. It may reject but cannot "
                "confirm the preregistered mechanism or a broad reasoning claim."
            ),
        }
        atomic_json(out, result)
        verify_bound_inputs(bound)
        motor_bound.verify_path()
        if canonical:
            seal_output_directory(out)
        print(
            json.dumps(
                {
                    "teacher_forced_carry": feature_results,
                    "cycle": {
                        arm: value["cycle"]["integrated_two_call_exact"]
                        for arm, value in results.items()
                    },
                    "direct": {
                        arm: value["correct"] for arm, value in direct_results.items()
                    },
                },
                sort_keys=True,
            )
        )
    finally:
        close_bound_inputs(bound)
        motor_bound.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--ckpt", required=True)
    common.add_argument("--tokenizer", required=True)
    common.add_argument("--episodes", required=True)
    common.add_argument("--cycle", required=True)
    common.add_argument("--source-commit", default="")
    common.add_argument("--source-manifest-sha256", default="")
    common.add_argument("--allow-non-cuda", action="store_true")
    train = subparsers.add_parser("train", parents=[common])
    train.add_argument("--out", required=True)
    train.add_argument("--quota", type=int, default=FIT_QUOTA)
    train.add_argument("--extract-batch", type=int, default=CANONICAL_EXTRACT_BATCH)
    train.add_argument("--updates", type=int, default=CANONICAL_UPDATES)
    train.add_argument("--batch-size", type=int, default=CANONICAL_BATCH)
    train.add_argument("--lr", type=float, default=CANONICAL_LR)
    train.add_argument("--weight-decay", type=float, default=CANONICAL_WEIGHT_DECAY)
    train.set_defaults(func=_train)
    evaluate = subparsers.add_parser("eval", parents=[common])
    evaluate.add_argument("--motor", required=True)
    evaluate.add_argument("--motor-sha256", default="")
    evaluate.add_argument("--out", required=True)
    evaluate.add_argument("--per-regime", type=int, default=CANONICAL_PER_REGIME)
    evaluate.add_argument("--max-new", type=int, default=CANONICAL_MAX_NEW)
    evaluate.add_argument("--extract-batch", type=int, default=CANONICAL_EXTRACT_BATCH)
    evaluate.set_defaults(func=_eval)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
