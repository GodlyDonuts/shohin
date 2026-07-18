#!/usr/bin/env python3
"""Fit and evaluate a grammar-gated result-digit motor on a frozen DRS model.

Sibling to Codex's carry motor (``train/causal_carry_motor.py``), which this
module does not import, modify, or share output paths with. Same frozen DRS
backbone class, different grammar site: this motor can change only the ten
decimal-digit token logits at the exact canonical ``;r=`` result-digit write
position (cursor ``p``). It never receives a solver value during generation
and every base-model parameter stays frozen.

This is deliberately leaner than the carry motor's custody apparatus: the
four frozen inputs (checkpoint, tokenizer, episodes, cycle) are hash-checked
directly rather than wrapped in an immutable-snapshot object, and canonical
runs require a plain git source-commit match rather than a sealed manifest
protocol. ``--allow-non-canonical`` skips both for exploratory Newton runs.
"""

from __future__ import annotations

import argparse
import collections
import contextlib
import hashlib
import json
import math
import os
import random
import re
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
    initial_state,
    microstep_prompt,
    parse_state,
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
EXPECTED_MOTOR_SOURCE_MANIFEST_SHA256 = (
    "dd94a00be3984d2f43d6342459ba9749e3164624bb9b96977bd19c11f23845f0"
)
SOURCE_FILES = (
    "R12_CAUSAL_RESULT_DIGIT_MOTOR_PREREG.md",
    "train/causal_result_digit_motor.py",
    "train/test_causal_result_digit_motor.py",
    "train/jobs/run_result_digit_motor_newton.sbatch",
    "train/digitwise_controller.py",
    "train/digitwise_protocol.py",
    "train/eval_suite.py",
    "train/model.py",
    "train/probe_digitwise_workspace.py",
)
FIT_SEED = 20260717
FIT_WIDTHS = (4, 6)
FIT_STYLES = ("core", "heldout")
FIT_QUOTA = 200
CANONICAL_UPDATES = 2000
CANONICAL_BATCH = 512
CANONICAL_LR = 3e-3
CANONICAL_WEIGHT_DECAY = 1e-4
CANONICAL_EXTRACT_BATCH = 1
CANONICAL_PER_REGIME = 50
CANONICAL_MAX_NEW = 96
# Parameter budget: the tied-embedding 300k checkpoint has exactly 125,081,664
# unique parameters. Keep total strictly below 150M.
# SiLU MLP 576→4096→4096→10 = 19,185,674 trainable parameters
# (144,267,338 total, leaving 5,732,661 parameters below the strict ceiling).
MOTOR_HIDDEN = 4096
MOTOR_MID = 4096
DIGIT_COUNT = 10
DIGIT_SITE = re.compile(
    r"^dws:op=(add|sub);w=([1-9]\d*);p=(\d+);c=([01]);a=(\d+);b=(\d+);r=(\d*)$"
)
BASE_PARAM_COUNT = 125_081_664  # Shohin flagship (tied embedding counted once)
MAX_TOTAL_PARAMS = 150_000_000


def unique_model_parameter_count(model):
    """Count tied parameters once, matching PyTorch deployment semantics."""
    return sum(parameter.numel() for parameter in model.parameters())


def parameter_budget(motor_parameters):
    """Return the exact strict-under-cap deployment ledger."""
    motor_parameters = int(motor_parameters)
    total = BASE_PARAM_COUNT + motor_parameters
    if total >= MAX_TOTAL_PARAMS:
        raise ValueError(
            "parameter budget exceeded: base={} motor={} total={} cap<{}".format(
                BASE_PARAM_COUNT, motor_parameters, total, MAX_TOTAL_PARAMS
            )
        )
    return {
        "base_parameters": BASE_PARAM_COUNT,
        "motor_parameters": motor_parameters,
        "total_parameters": total,
        "strict_cap": MAX_TOTAL_PARAMS,
        "remaining_addable_parameters": MAX_TOTAL_PARAMS - total - 1,
    }


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def verify_frozen_inputs(checkpoint, episodes, tokenizer, cycle):
    """Hash-check the four frozen scientific inputs directly, no snapshot theater."""
    observed = {
        "checkpoint": sha256_file(checkpoint),
        "episodes": sha256_file(episodes),
        "tokenizer": sha256_file(tokenizer),
        "cycle": sha256_file(cycle),
    }
    expected = {
        "checkpoint": EXPECTED_CHECKPOINT_SHA256,
        "episodes": EXPECTED_EPISODES_SHA256,
        "tokenizer": EXPECTED_TOKENIZER_SHA256,
        "cycle": EXPECTED_CYCLE_SHA256,
    }
    if observed != expected:
        raise ValueError("frozen input hash mismatch: {}".format(observed))
    return observed


def source_manifest_sha256(root):
    sources = {name: sha256_file(root / name) for name in SOURCE_FILES}
    return stable_json_sha256(sources), sources


def verify_reviewed_source_commit(root, source_commit, source_hashes):
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


def validate_source_contract(args, root, canonical):
    manifest, sources = source_manifest_sha256(root)
    if canonical:
        if not args.source_commit:
            raise SystemExit(
                "canonical run requires --source-commit "
                "(or pass --allow-non-canonical for exploratory runs)"
            )
        verify_reviewed_source_commit(root, args.source_commit, sources)
    return {
        "git_commit": args.source_commit if canonical else None,
        "manifest_sha256": manifest,
    }


class DigitMotor(nn.Module):
    """Wide nonlinear motor producing deltas for only the ten digit tokens.

    Sized to use most of the <150M total budget without touching the frozen
    125.1M backbone: 576→4096→4096→10 ≈ 19.19M params.
    """

    def __init__(self, d_model, hidden=MOTOR_HIDDEN, mid=None):
        super().__init__()
        self.d_model = int(d_model)
        self.hidden = int(hidden)
        self.mid = int(self.hidden if mid is None else mid)
        if self.d_model <= 0 or self.hidden <= 0 or self.mid <= 0:
            raise ValueError("motor dimensions must be positive")
        self.down = nn.Linear(self.d_model, self.hidden, bias=True)
        self.mid_layer = nn.Linear(self.hidden, self.mid, bias=True)
        self.up = nn.Linear(self.mid, DIGIT_COUNT, bias=True)
        nn.init.normal_(self.down.weight, mean=0.0, std=1.0 / math.sqrt(self.d_model))
        nn.init.zeros_(self.down.bias)
        nn.init.normal_(
            self.mid_layer.weight, mean=0.0, std=1.0 / math.sqrt(self.hidden)
        )
        nn.init.zeros_(self.mid_layer.bias)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)
        parameter_budget(self.parameter_count())

    def forward(self, hidden):
        x = F.silu(self.down(hidden.float()))
        x = F.silu(self.mid_layer(x))
        return self.up(x)

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


def is_digit_site(prompt, response_prefix):
    """Recognize the next canonical result-digit field for an actual DWS prompt.

    Grammar only: the carry value ``c`` in the prefix is accepted whether it
    is 0 or 1, since the router never consults a solver. The site requires
    exactly ``state["p"]`` result digits already written, matching the
    ``field_prefix(..., "digit")`` cut point one digit before position ``p``.
    """
    state = dws_prompt_state(prompt)
    match = DIGIT_SITE.fullmatch(str(response_prefix))
    if state is None or match is None:
        return False
    operation, width, position, _carry, a_tape, b_tape, r_written = match.groups()
    width_int = int(width)
    return (
        operation == state["op"]
        and width_int == int(state["w"])
        and int(position) == int(state["p"]) + 1
        and int(position) <= width_int
        and a_tape == state["a"]
        and b_tape == state["b"]
        and len(r_written) == int(state["p"])
    )


class DigitRouter:
    """Count one grammar opportunity and at most one motor fire per response."""

    def __init__(self, prompt, motor_present):
        self.prompt = str(prompt)
        self.motor_present = bool(motor_present)
        self.site_count = 0
        self.fire_count = 0

    def observe(self, response_prefix):
        site = self.site_count == 0 and is_digit_site(self.prompt, response_prefix)
        if site:
            self.site_count += 1
        active = site and self.motor_present
        if active:
            self.fire_count += 1
        return active


def apply_motor_logits(logits, hidden, motor, digit_ids, active):
    """Return logits with at most the ten digit ids changed; gate-off is exact identity."""
    if not active or motor is None:
        return logits
    ids = list(digit_ids)
    if (
        len(ids) != DIGIT_COUNT
        or len(set(ids)) != DIGIT_COUNT
        or min(ids) < 0
        or max(ids) >= logits.shape[-1]
    ):
        raise ValueError("invalid digit token ids")
    delta = motor(hidden)
    if delta.shape != logits.shape[:-1] + (DIGIT_COUNT,):
        raise ValueError("motor delta shape mismatch")
    adjusted = logits.clone()
    id_tensor = torch.as_tensor(ids, dtype=torch.long, device=logits.device)
    adjusted[..., id_tensor] = adjusted[..., id_tensor] + delta.to(logits.dtype)
    return adjusted


def full_vocab_motor_loss(hidden, base10, other_lse, targets, motor):
    """Full-vocabulary CE after the exact deployed logit-dtype arithmetic."""
    delta = motor(hidden).to(base10.dtype)
    adjusted = (base10 + delta).float()
    digit_lse = torch.logsumexp(adjusted, dim=-1)
    total_lse = torch.logaddexp(other_lse.float(), digit_lse)
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


def _all_fit_keys():
    """Balance over (operation, style, target digit) only.

    A full cross with position, width, and current carry (as the carry motor
    does for its two-way target) would multiply a ten-way target into an
    infeasible board size. Position, width, and carry still vary naturally
    from real episode walks and are logged in ``position_counts`` below.
    """
    keys = set()
    for operation in ("add", "sub"):
        for style in FIT_STYLES:
            for target in range(DIGIT_COUNT):
                keys.add((operation, style, target))
    return keys


def generate_fit_rows(tokenizer, episodes_text, seed=FIT_SEED, quota=FIT_QUOTA):
    """Generate a balanced-ish, duplicate-free solver-labelled motor board."""
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
        if attempts > quota * 4000:
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
            styles = list(FIT_STYLES)
            rng.shuffle(styles)
            for style in styles:
                prompt = microstep_prompt(state, style=style)
                prefix, target = field_prefix(prompt, next_state, "digit")
                target = int(target)
                key = (operation, style, target)
                if counts[key] >= quota:
                    continue
                prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
                if prompt_hash in forbidden_prompts:
                    continue
                prefix_ids, target_id = token_for_digit(tokenizer, prefix, str(target))
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
                        "target": target,
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


def load_model(path, device):
    checkpoint = torch.load(path, map_location="cpu")
    model = GPT(GPTConfig(**checkpoint["cfg"])).eval()
    model.load_state_dict(checkpoint["model"])
    actual_parameters = unique_model_parameter_count(model)
    if actual_parameters != BASE_PARAM_COUNT:
        raise ValueError(
            "base checkpoint parameter count changed: {} != {}".format(
                actual_parameters, BASE_PARAM_COUNT
            )
        )
    if int(model.cfg.n_loop) != 1:
        raise ValueError("digit motor requires n_loop=1")
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
    digit_ids,
    device,
    batch_size=CANONICAL_EXTRACT_BATCH,
    store_other_logits=False,
):
    """Extract residuals through the exact cached path used by generation."""
    if batch_size <= 0:
        raise ValueError("batch size must be positive")
    digit_ids = list(digit_ids)
    if len(digit_ids) != DIGIT_COUNT or len(set(digit_ids)) != DIGIT_COUNT:
        raise ValueError("digit ids must be ten distinct token ids")
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
    digit_id_tensor = torch.as_tensor(digit_ids, dtype=torch.long)
    base10 = None
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
            if base10 is None:
                base10 = torch.empty((len(rows), DIGIT_COUNT), dtype=last.dtype)
            elif base10.dtype != last.dtype:
                raise RuntimeError("deployment logit dtype changed during extraction")
            keep = torch.ones(last.shape[-1], dtype=torch.bool, device=last.device)
            keep[digit_id_tensor.to(last.device)] = False
            batch_other_ids = torch.arange(last.shape[-1], device=last.device)[keep]
            if other_token_ids is None:
                other_token_ids = batch_other_ids.cpu()
            elif not torch.equal(other_token_ids, batch_other_ids.cpu()):
                raise RuntimeError("other-token ordering changed during extraction")
            positions = [index for index, _ in batch]
            hidden[positions] = captured["hidden"].float().cpu()
            base10[positions] = last[:, digit_id_tensor.to(last.device)].cpu()
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
            "[digit-motor] extracted prompt={} prefix={} rows={}".format(
                prompt_length, prefix_length, len(group)
            ),
            flush=True,
        )
    if base10 is None or other_token_ids is None:
        raise RuntimeError("feature extraction produced no rows")
    return {
        "hidden": hidden,
        "base10": base10,
        "other_lse": other_lse,
        "other_max": other_max,
        "other_max_token_id": other_max_token_id,
        "other_logits": other_logits,
        "other_token_ids": other_token_ids,
        "digit_ids": digit_id_tensor,
        "deployment_logit_dtype": str(base10.dtype),
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
    initial_state_dict,
    device,
    updates=CANONICAL_UPDATES,
    batch_size=CANONICAL_BATCH,
    lr=CANONICAL_LR,
    weight_decay=CANONICAL_WEIGHT_DECAY,
    seed=FIT_SEED,
):
    motor = DigitMotor(features["hidden"].shape[1], MOTOR_HIDDEN, MOTOR_MID).to(device)
    motor.load_state_dict(initial_state_dict)
    optimizer = torch.optim.AdamW(motor.parameters(), lr=lr, weight_decay=weight_decay)
    schedule, schedule_sha256 = _batch_schedule(len(labels), batch_size, updates, seed)
    x = features["hidden"].to(device)
    base10 = features["base10"].to(device)
    other_lse = features["other_lse"].to(device)
    targets = torch.as_tensor(labels, dtype=torch.long, device=device)
    losses = []
    for update, batch_cpu in enumerate(schedule, 1):
        batch = batch_cpu.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = full_vocab_motor_loss(
            x[batch], base10[batch], other_lse[batch], targets[batch], motor
        )
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite motor loss")
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        if update == 1 or update % 200 == 0 or update == updates:
            print(
                "[digit-motor] update={}/{} loss={:.6f}".format(
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
    adjusted = features["base10"]
    if motor is not None:
        adjusted = adjusted + motor(features["hidden"]).to(adjusted.dtype)
    digit_ids = features["digit_ids"].long()
    targets = torch.as_tensor(labels, dtype=torch.long)
    rows = len(targets)
    max_values = adjusted.max(dim=-1, keepdim=True).values
    is_max = adjusted == max_values
    tie_key = torch.where(
        is_max,
        digit_ids.unsqueeze(0).expand(rows, DIGIT_COUNT),
        torch.full((rows, DIGIT_COUNT), 1 << 62, dtype=torch.long),
    )
    predicted_digit = tie_key.argmin(dim=-1)
    top1_correct = predicted_digit == targets

    target_logits = adjusted.gather(1, targets[:, None]).squeeze(1)
    target_token_ids = digit_ids[targets]
    digit_ids_row = digit_ids.unsqueeze(0).expand(rows, DIGIT_COUNT)
    # The target column itself never contributes: it ties with equal value
    # but its own id is never strictly less than target_token_ids.
    digit_beats_count = (
        (adjusted > target_logits[:, None])
        | (
            (adjusted == target_logits[:, None])
            & (digit_ids_row < target_token_ids[:, None])
        )
    ).sum(dim=-1)
    other_max = features["other_max"].to(target_logits.dtype)
    other_max_token_ids = features["other_max_token_id"].long()
    vocab_beats_target = (other_max > target_logits) | (
        (other_max == target_logits) & (other_max_token_ids < target_token_ids)
    )
    global_correct = (digit_beats_count == 0) & ~vocab_beats_target
    result = {
        "digit_top1_correct": int(top1_correct.sum()),
        "digit_top1_accuracy": float(top1_correct.float().mean()),
        "global_correct": int(global_correct.sum()),
        "global_accuracy": float(global_correct.float().mean()),
        "rows": rows,
        "prediction_histogram": {
            int(digit): int((predicted_digit == digit).sum())
            for digit in range(DIGIT_COUNT)
        },
        "global_rank_available": features.get("other_logits") is not None,
    }
    if features.get("other_logits") is not None:
        rank_sum = rank_max = 0
        rank_above_20 = 0
        other_logits = features["other_logits"]
        other_token_ids = features["other_token_ids"].long()
        for start in range(0, rows, 256):
            stop = min(start + 256, rows)
            target = target_logits[start:stop]
            target_ids = target_token_ids[start:stop]
            vocab_beats = (
                (other_logits[start:stop] > target[:, None])
                | (
                    (other_logits[start:stop] == target[:, None])
                    & (other_token_ids[None, :] < target_ids[:, None])
                )
            ).sum(dim=-1)
            ranks = 1 + digit_beats_count[start:stop] + vocab_beats
            rank_sum += int(ranks.sum())
            rank_max = max(rank_max, int(ranks.max()))
            rank_above_20 += int((ranks > 20).sum())
        result.update(
            {
                "mean_target_rank_global": rank_sum / rows,
                "max_target_rank_global": rank_max,
                "target_rank_above_20": rank_above_20,
            }
        )
    return result


def prepare_output(path):
    path = Path(path)
    report_path = path.with_name("report.json")
    if path.exists() or report_path.exists():
        raise FileExistsError("refusing existing output: {}".format(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    return path, report_path


def prepare_single_output(path):
    """Prepare one output without reserving the sibling training report."""
    path = Path(path)
    if path.exists():
        raise FileExistsError("refusing existing output: {}".format(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path, value):
    path = Path(path)
    if path.exists():
        raise FileExistsError("refusing to overwrite {}".format(path))
    fd, temporary = tempfile.mkstemp(prefix=".{}-".format(path.name), dir=path.parent)
    os.close(fd)
    Path(temporary).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def write_torch(path, value):
    path = Path(path)
    if path.exists():
        raise FileExistsError("refusing to overwrite {}".format(path))
    fd, temporary = tempfile.mkstemp(prefix=".{}-".format(path.name), dir=path.parent)
    os.close(fd)
    torch.save(value, temporary)
    os.replace(temporary, path)


@torch.no_grad()
def motor_generate(
    model, motor, tokenizer, digit_ids, prompt, device, max_new=CANONICAL_MAX_NEW
):
    """Greedy cached generation with one grammar-only digit-motor site."""
    cap = int(model.cfg.seq_len)
    prompt_ids = tokenizer.encode(prompt).ids[-cap:]
    eos_id = tokenizer.token_to_id("<|endoftext|>")
    generated = []
    router = DigitRouter(prompt, motor is not None)
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
                logits[:, -1, :], captured["hidden"], motor, digit_ids, active
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
                prefix, target = field_prefix(prompt, next_state, "digit")
                target = int(target)
                prompt_ids = tokenizer.encode(prompt).ids
                prefix_ids, target_id = token_for_digit(tokenizer, prefix, str(target))
                if prefix_ids[: len(prompt_ids)] != prompt_ids:
                    raise RuntimeError("development prefix boundary mismatch")
                rows.append(
                    {
                        "operation": state["op"],
                        "width": int(state["w"]),
                        "position": int(state["p"]),
                        "style": branch["prompt_style"],
                        "current_carry": int(state["c"]),
                        "target": target,
                        "target_id": int(target_id),
                        "prompt_ids": prompt_ids,
                        "prefix_ids": prefix_ids,
                        "episode_id": branch["id"],
                        "regime": outer["split"],
                        "transition": transition,
                    }
                )
    return rows


def evaluate_autonomous(episodes, model, motor, tokenizer, digit_ids, device, max_new):
    by_regime = collections.defaultdict(lambda: collections.Counter())
    transcripts = []
    for index, episode in enumerate(episodes, 1):
        sites = fires = 0

        def ask(prompt):
            nonlocal sites, fires
            response, site_count, fire_count = motor_generate(
                model, motor, tokenizer, digit_ids, prompt, device, max_new=max_new
            )
            sites += site_count
            fires += fire_count
            return response

        rollout = rollout_episode(episode, ask, prompt_style=episode["prompt_style"])
        row = by_regime[episode["split"]]
        row["episodes"] += 1
        row["first_transition_correct"] += int(
            bool(rollout["rows"]) and rollout["rows"][0]["correct"]
        )
        row["transition_correct"] += sum(
            int(item["correct"]) for item in rollout["rows"]
        )
        row["transition_attempted"] += len(rollout["rows"])
        row["state_closed_loop_correct"] += int(rollout["state_closed_loop"])
        row["final_answer_correct"] += int(rollout["success"])
        row["site_opportunities"] += sites
        row["motor_fires"] += fires
        if len(transcripts) < 15:
            transcripts.append(
                {"id": episode["id"], "regime": episode["split"], "rollout": rollout}
            )
        if index % 50 == 0:
            print(
                "[digit-motor] autonomous {}/{}".format(index, len(episodes)),
                flush=True,
            )
    return {
        "by_regime": {
            regime: dict(values) for regime, values in sorted(by_regime.items())
        },
        "transcripts": transcripts,
    }


def evaluate_cycle_first_transition(
    cycle_text, episode_map, model, motor, tokenizer, digit_ids, device, max_new
):
    """Lean single-step use of the frozen cycle board: first-transition exactness."""
    document = json.loads(cycle_text)
    first_exact = site_count = fire_count = 0
    examples = []
    for record in document["records"]:
        state = parse_state(record["counterfactual_state"])
        expected = parse_state(record["counterfactual_next"])
        if state is None or expected is None:
            raise ValueError("invalid state in frozen cycle artifact")
        episode = episode_map[record["episode_id"]]
        prompt = microstep_prompt(state, style=episode["prompt_style"])
        response, sites, fires = motor_generate(
            model, motor, tokenizer, digit_ids, prompt, device, max_new=max_new
        )
        ok = parse_state(response) == expected
        first_exact += int(ok)
        site_count += sites
        fire_count += fires
        if len(examples) < 10:
            examples.append(
                {
                    "episode_id": record["episode_id"],
                    "prompt": prompt,
                    "response": response,
                    "exact": ok,
                }
            )
    return {
        "records": len(document["records"]),
        "first_exact": first_exact,
        "site_opportunities": site_count,
        "motor_fires": fire_count,
        "examples": examples,
    }


NON_DWS_PRESERVATION_PROMPTS = (
    "Question: What is 17 + 26?\nAnswer:",
    "Question: Sort 8, 2, 5 in increasing order.\nAnswer:",
    "Question: Convert binary 1011 to decimal.\nAnswer:",
    "Question: Is 97 prime?\nAnswer:",
    "Question: Compute 12 * 7.\nAnswer:",
    "Question: What is the capital of France?\nAnswer:",
)


def evaluate_non_dws_preservation(model, motor, tokenizer, digit_ids, device):
    preservation = []
    for prompt in NON_DWS_PRESERVATION_PROMPTS:
        base_response, base_sites, base_fires = motor_generate(
            model, None, tokenizer, digit_ids, prompt, device, max_new=48
        )
        motor_response, motor_sites, motor_fires = motor_generate(
            model, motor, tokenizer, digit_ids, prompt, device, max_new=48
        )
        preservation.append(
            {
                "prompt": prompt,
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
    return preservation


def validate_motor_bundle(
    bundle, expected_bindings, source_contract, allow_evaluator_patch=False
):
    if bundle.get("audit") != "causal_result_digit_motor_fit_v1":
        raise ValueError("unsupported result-digit motor bundle audit version")
    if bundle.get("deployment_logit_dtype") not in {"torch.bfloat16", "torch.float32"}:
        raise ValueError("invalid result-digit motor deployment logit dtype")
    for name, expected in expected_bindings.items():
        if bundle.get(name) != expected:
            raise ValueError("motor bundle input mismatch: {}".format(name))
    motor_source_contract = bundle.get("source_contract")
    if motor_source_contract != source_contract:
        expected_motor_source = {
            "git_commit": None,
            "manifest_sha256": EXPECTED_MOTOR_SOURCE_MANIFEST_SHA256,
        }
        if not allow_evaluator_patch or motor_source_contract != expected_motor_source:
            raise ValueError("motor bundle source contract mismatch")
    if len(bundle.get("digit_ids", [])) != DIGIT_COUNT:
        raise ValueError("motor bundle does not bind ten digit ids")
    for arm in ("treatment", "shuffled"):
        expected_hash = bundle.get("{}_state_sha256".format(arm))
        if tensor_state_sha256(bundle[arm]) != expected_hash:
            raise ValueError("{} state hash mismatch".format(arm))
    return motor_source_contract


def _train(args):
    canonical = not args.allow_non_canonical
    root = Path(__file__).resolve().parents[1]
    if canonical and not torch.cuda.is_available():
        raise SystemExit(
            "canonical digit-motor fit requires CUDA (pass --allow-non-canonical to skip)"
        )
    frozen = verify_frozen_inputs(args.ckpt, args.episodes, args.tokenizer, args.cycle)
    source_contract = validate_source_contract(args, root, canonical)
    out, report_path = prepare_output(args.out)
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    torch.manual_seed(FIT_SEED)
    if device == "cuda":
        torch.cuda.manual_seed_all(FIT_SEED)
    tokenizer = Tokenizer.from_file(args.tokenizer)
    digit_ids = [
        token_for_digit(tokenizer, "x=", str(digit))[1] for digit in range(DIGIT_COUNT)
    ]
    if len(set(digit_ids)) != DIGIT_COUNT:
        raise ValueError("digit tokens are not distinct")
    for digit, token_id in enumerate(digit_ids):
        if tokenizer.decode([token_id]) != str(digit):
            raise ValueError("digit token id does not decode to its digit")
    checkpoint, model = load_model(args.ckpt, device)
    episodes_text = Path(args.episodes).read_text()
    rows, board = generate_fit_rows(tokenizer, episodes_text, FIT_SEED, args.quota)
    board["rows_sha256"] = stable_json_sha256(rows)
    control_labels, control = permuted_control_labels(rows)
    features = extract_frozen_features(
        model, rows, digit_ids, device, batch_size=args.extract_batch
    )
    initial = DigitMotor(model.cfg.d_model, MOTOR_HIDDEN, MOTOR_MID)
    initial_state_dict = {
        name: tensor.detach().clone() for name, tensor in initial.state_dict().items()
    }
    initial_sha256 = tensor_state_sha256(initial_state_dict)
    treatment, treatment_fit = fit_motor(
        features,
        features["labels"],
        initial_state_dict,
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
        initial_state_dict,
        device,
        args.updates,
        args.batch_size,
        args.lr,
        args.weight_decay,
        FIT_SEED,
    )
    if treatment_fit["schedule_sha256"] != shuffled_fit["schedule_sha256"]:
        raise RuntimeError("learned arms did not receive the same batch schedule")
    verify_frozen_inputs(args.ckpt, args.episodes, args.tokenizer, args.cycle)
    bundle = {
        "audit": "causal_result_digit_motor_fit_v1",
        "base_checkpoint_sha256": frozen["checkpoint"],
        "tokenizer_sha256": frozen["tokenizer"],
        "episodes_sha256": frozen["episodes"],
        "cycle_sha256": frozen["cycle"],
        "source_contract": source_contract,
        "checkpoint_step": checkpoint.get("step"),
        "d_model": int(model.cfg.d_model),
        "hidden": MOTOR_HIDDEN,
        "mid": MOTOR_MID,
        "parameter_count": treatment.parameter_count(),
        "parameter_budget": parameter_budget(treatment.parameter_count()),
        "extract_batch": args.extract_batch,
        "deployment_logit_dtype": features["deployment_logit_dtype"],
        "digit_ids": [int(x) for x in digit_ids],
        "initial_state_sha256": initial_sha256,
        "treatment": treatment.state_dict(),
        "shuffled": shuffled.state_dict(),
        "treatment_state_sha256": tensor_state_sha256(treatment.state_dict()),
        "shuffled_state_sha256": tensor_state_sha256(shuffled.state_dict()),
        "board": board,
        "control": control,
        "treatment_fit": treatment_fit,
        "shuffled_fit": shuffled_fit,
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
            "A fit establishes no reasoning result. The motor can alter only the ten "
            "digit-token logits at a grammar-defined DWS result-digit site; heldout "
            "autonomous and confirmation gates remain required."
        ),
    }
    write_torch(out, bundle)
    report = {
        key: value
        for key, value in bundle.items()
        if key not in {"treatment", "shuffled"}
    }
    report["bundle_sha256"] = sha256_file(out)
    report["bundle"] = str(out)
    write_json(report_path, report)
    print(
        "[digit-motor] teacher-forced digit top1: base={:.4f} treatment={:.4f} shuffled={:.4f}".format(
            report["fit_feature_metrics"]["base"]["digit_top1_accuracy"],
            report["fit_feature_metrics"]["treatment"]["digit_top1_accuracy"],
            report["fit_feature_metrics"]["shuffled_on_true_labels"][
                "digit_top1_accuracy"
            ],
        ),
        flush=True,
    )
    print(json.dumps(report["fit_feature_metrics"], sort_keys=True), flush=True)


def _eval(args):
    canonical = not args.allow_non_canonical
    root = Path(__file__).resolve().parents[1]
    if canonical and not torch.cuda.is_available():
        raise SystemExit(
            "canonical digit-motor evaluation requires CUDA (pass --allow-non-canonical to skip)"
        )
    if canonical and args.allow_evaluator_patch:
        raise SystemExit("canonical evaluation cannot authorize an evaluator patch")
    frozen = verify_frozen_inputs(args.ckpt, args.episodes, args.tokenizer, args.cycle)
    source_contract = validate_source_contract(args, root, canonical)
    out = prepare_single_output(args.out)
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    checkpoint, model = load_model(args.ckpt, device)
    tokenizer = Tokenizer.from_file(args.tokenizer)
    bundle = torch.load(args.motor, map_location="cpu")
    expected_bindings = {
        "base_checkpoint_sha256": frozen["checkpoint"],
        "tokenizer_sha256": frozen["tokenizer"],
        "episodes_sha256": frozen["episodes"],
        "cycle_sha256": frozen["cycle"],
    }
    motor_source_contract = validate_motor_bundle(
        bundle,
        expected_bindings,
        source_contract,
        allow_evaluator_patch=args.allow_evaluator_patch,
    )
    digit_ids = [int(x) for x in bundle["digit_ids"]]
    treatment = (
        DigitMotor(
            bundle["d_model"], bundle["hidden"], bundle.get("mid", bundle["hidden"])
        )
        .to(device)
        .eval()
    )
    treatment.load_state_dict(bundle["treatment"])
    shuffled = (
        DigitMotor(
            bundle["d_model"], bundle["hidden"], bundle.get("mid", bundle["hidden"])
        )
        .to(device)
        .eval()
    )
    shuffled.load_state_dict(bundle["shuffled"])
    dead = (
        DigitMotor(
            bundle["d_model"], bundle["hidden"], bundle.get("mid", bundle["hidden"])
        )
        .to(device)
        .eval()
    )
    episodes_text = Path(args.episodes).read_text()
    episodes = read_selected_episodes(episodes_text, args.per_regime)
    episode_map = {
        json.loads(raw)["id"]: json.loads(raw)
        for raw in episodes_text.splitlines()
        if raw.strip()
    }
    dev_rows = development_feature_rows(episodes, tokenizer)
    dev_features = extract_frozen_features(
        model,
        dev_rows,
        digit_ids,
        device,
        batch_size=args.extract_batch,
        store_other_logits=True,
    )
    if bundle.get("deployment_logit_dtype") != dev_features["deployment_logit_dtype"]:
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
    arms = {"base": None, "dead": dead, "treatment": treatment, "shuffled": shuffled}
    autonomous = {}
    cycle_text = Path(args.cycle).read_text()
    cycle_results = {}
    for arm, motor in arms.items():
        autonomous[arm] = evaluate_autonomous(
            episodes, model, motor, tokenizer, digit_ids, device, args.max_new
        )
        cycle_results[arm] = evaluate_cycle_first_transition(
            cycle_text,
            episode_map,
            model,
            motor,
            tokenizer,
            digit_ids,
            device,
            args.max_new,
        )
    preservation = evaluate_non_dws_preservation(
        model, treatment, tokenizer, digit_ids, device
    )
    verify_frozen_inputs(args.ckpt, args.episodes, args.tokenizer, args.cycle)
    result = {
        "audit": "causal_result_digit_motor_development_eval_v1",
        "checkpoint_step": checkpoint.get("step"),
        "frozen_sha256": frozen,
        "motor_source_contract": motor_source_contract,
        "evaluator_source_contract": source_contract,
        "evaluator_patch_authorized": bool(args.allow_evaluator_patch),
        "motor": str(Path(args.motor).resolve()),
        "motor_sha256": sha256_file(args.motor),
        "parameter_budget": parameter_budget(treatment.parameter_count()),
        "per_regime": args.per_regime,
        "max_new": args.max_new,
        "extract_batch": args.extract_batch,
        "teacher_forced_digit": feature_results,
        "autonomous": autonomous,
        "cycle_first_transition": cycle_results,
        "non_dws_preservation": preservation,
        "claim_boundary": (
            "Development evaluation on an already inspected board. It may reject but "
            "cannot confirm the preregistered mechanism or a broad reasoning claim."
        ),
    }
    write_json(out, result)
    exact_state_rate = {
        arm: (
            sum(
                regime["state_closed_loop_correct"]
                for regime in value["by_regime"].values()
            )
            / max(1, sum(regime["episodes"] for regime in value["by_regime"].values()))
        )
        for arm, value in autonomous.items()
    }
    print(
        "[digit-motor] teacher-forced digit top1: base={:.4f} treatment={:.4f} shuffled={:.4f} dead={:.4f}".format(
            feature_results["base"]["digit_top1_accuracy"],
            feature_results["treatment"]["digit_top1_accuracy"],
            feature_results["shuffled"]["digit_top1_accuracy"],
            feature_results["dead"]["digit_top1_accuracy"],
        ),
        flush=True,
    )
    print(
        "[digit-motor] autonomous exact-state rate: {}".format(
            json.dumps(exact_state_rate, sort_keys=True)
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--ckpt", required=True)
    common.add_argument("--tokenizer", required=True)
    common.add_argument("--episodes", required=True)
    common.add_argument("--cycle", required=True)
    common.add_argument("--source-commit", default="")
    common.add_argument(
        "--allow-non-canonical",
        action="store_true",
        help="skip the git source-commit seal and CUDA requirement for exploratory Newton runs",
    )
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
    evaluate.add_argument("--out", required=True)
    evaluate.add_argument("--per-regime", type=int, default=CANONICAL_PER_REGIME)
    evaluate.add_argument("--max-new", type=int, default=CANONICAL_MAX_NEW)
    evaluate.add_argument("--extract-batch", type=int, default=CANONICAL_EXTRACT_BATCH)
    evaluate.add_argument(
        "--allow-evaluator-patch",
        action="store_true",
        help=(
            "exploratory-only: evaluate the exact dd94-bound motor with a separately "
            "hash-recorded evaluator source"
        ),
    )
    evaluate.set_defaults(func=_eval)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
