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
import shutil
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
    final_prompt,
    initial_state,
    microstep_prompt,
    parse_answer,
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
CANONICAL_CHECKPOINT_STEP = "sft_ep1"
EXPECTED_CONFIRMATION_PROMPT_EXCLUSIONS = 33_700
EXPECTED_CONFIRMATION_OPERAND_EXCLUSIONS = 3_000
EXPECTED_CONFIRMATION_EXCLUSION_SHA256 = (
    "df2d7fc97f22b9bd8987141095f95ec2cf0240f4c4bf463f53996f82ef6c1f00"
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
CANONICAL_DEVELOPMENT_EPISODES = 300
CANONICAL_MAX_NEW = 96
CANONICAL_EXTRACT_BATCH = 1
CANONICAL_FEATURE_SHARDS = 8
CANONICAL_DEVICE_NAME = "NVIDIA H100 PCIe"
CANONICAL_CYCLE_CASES = 50
CANONICAL_DATA_ROOT = Path("/lustre/fs1/home/sa305415/shohin")
CANONICAL_PLAN_PARENT = CANONICAL_DATA_ROOT / "artifacts" / "carry_motor"
CANONICAL_CONFIRMATION_PARENT = CANONICAL_PLAN_PARENT / "confirmation_commitments"
CANONICAL_CONFIRMATION_COMMITMENT_AUDIT = (
    "causal_carry_motor_confirmation_commitment_v4"
)
CANONICAL_CONFIRMATION_GENERATOR_SCHEMA = "causal_carry_motor_confirmation_generator_v4"
CANONICAL_CONFIRMATION_EXCLUSION_SCHEMA = (
    "causal_carry_motor_confirmation_exclusions_v1"
)
CANONICAL_CONFIRMATION_GENERATOR_ENTRYPOINT = (
    "train/causal_carry_motor.py:generate_confirmation_board"
)
CANONICAL_CONFIRMATION_GENERATOR_SOURCES = (
    "train/causal_carry_motor.py",
    "train/digitwise_protocol.py",
)
CANONICAL_CONFIRMATION_TIMING = "published_before_canonical_plan_extraction_and_fit"
CANONICAL_CONFIRMATION_CLAIM_BOUNDARY = (
    "This pre-fit commitment freezes generator identity, the exact development/cycle "
    "exclusion identities, and SHA256(secret). It contains no secret, confirmation "
    "board, score, or capability result."
)
CANONICAL_PLAN_AUDIT = "causal_carry_motor_plan_v6"
CANONICAL_SHARD_AUDIT = "causal_carry_motor_feature_shard_v6_canonical"
CANONICAL_FIT_AUDIT = "causal_carry_motor_fit_v8_canonical_sharded"
CANONICAL_EVAL_AUDIT = "causal_carry_motor_development_eval_v7"
CANONICAL_CONFIRMATION_EVAL_AUDIT = "causal_carry_motor_confirmation_eval_v2"
CANONICAL_TEACHER_SCORING_CONTRACT = "h100_bfloat16_batch1_apply_motor_logits_v1"
CANONICAL_DEVELOPMENT_SELECTION_AUDIT = "causal_carry_motor_development_selection_v1"
CANONICAL_DEVELOPMENT_SELECTION_ALGORITHM = "frozen_source_order_prefix_v1"
CANONICAL_DEVELOPMENT_REGIME_SPECS = (
    ("fit_width", (("fit_w4", 50), ("fit_w6", 50))),
    ("value_ood", (("value_ood_w4", 50), ("value_ood_w6", 50))),
    ("width_8", (("width_ood_w8", 100),)),
)
EXPECTED_DEVELOPMENT_SELECTION_SHA256 = (
    "0a68fe542306ae954696c3346cb9c6dcfff14e638e9dd26f0996d46c27e5e80b"
)
CANONICAL_SHARD_CLAIM_BOUNDARY = (
    "This immutable shard is a frozen-feature transport artifact only; "
    "it contains no fitted motor and establishes no capability result."
)
CANONICAL_FIT_CLAIM_BOUNDARY = (
    "A fit establishes no reasoning result. The motor can alter only carry token logits "
    "at a grammar-defined DWS site; heldout autonomous and confirmation gates remain required."
)
LINEAR_DIAGNOSTIC_CLAIM_BOUNDARY = (
    "Diagnostic feature extractability only; not an inference arm."
)
CANONICAL_EVAL_CLAIM_BOUNDARY = (
    "Development evaluation on an already inspected board. It may reject but cannot "
    "confirm the preregistered mechanism or a broad reasoning claim."
)
CANONICAL_CONFIRMATION_EVAL_CLAIM_BOUNDARY = (
    "One post-reveal evaluation of the exact secret-derived confirmation rows. "
    "Its mechanism decision remains conditional on the bound development cycle, "
    "direct, and preservation gates."
)
NON_DWS_PRESERVATION_PROMPTS = (
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
CANONICAL_SHARD_KEYS = frozenset(
    {
        "audit",
        "canonical",
        "plan_sha256",
        "base_checkpoint_sha256",
        "tokenizer_sha256",
        "episodes_sha256",
        "cycle_sha256",
        "confirmation_commitment_sha256",
        "scientific_source_sha256",
        "source_contract",
        "checkpoint_step",
        "board",
        "board_rows_sha256",
        "shard_index",
        "shard_count",
        "global_indices",
        "row_identity_sha256",
        "sentinel_indices",
        "sentinel_row_identity_sha256",
        "extract_batch",
        "features",
        "feature_payload_sha256",
        "sentinel_features",
        "sentinel_payload_sha256",
        "runtime",
        "claim_boundary",
    }
)
CANONICAL_FIT_KEYS = frozenset(
    {
        "audit",
        "canonical",
        "plan_sha256",
        "base_checkpoint_sha256",
        "tokenizer_sha256",
        "episodes_sha256",
        "cycle_sha256",
        "confirmation_commitment_sha256",
        "scientific_source_sha256",
        "source_contract",
        "checkpoint_step",
        "d_model",
        "rank",
        "parameter_count",
        "extract_batch",
        "feature_shard_merge",
        "deployment_logit_dtype",
        "zero_id",
        "one_id",
        "initial_state_sha256",
        "treatment",
        "shuffled",
        "treatment_state_sha256",
        "shuffled_state_sha256",
        "board",
        "control",
        "treatment_fit",
        "shuffled_fit",
        "linear_diagnostic",
        "fit_feature_metrics",
        "claim_boundary",
    }
)
RANK = 8
CARRY_SITE = re.compile(r"^dws:op=(add|sub);w=([1-9]\d*);p=(0|[1-9]\d*);c=$")
SCIENTIFIC_SOURCE_PATHS = (
    "R12_CAUSAL_CARRY_MOTOR_PREREG.md",
    "train/causal_carry_motor.py",
    "train/test_causal_carry_motor.py",
    "train/jobs/causal_carry_motor.sbatch",
    "pipeline/jobs/causal_carry_motor_plan_stokes.sbatch",
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


def _is_canonical_checkpoint_step(value):
    return type(value) is str and value == CANONICAL_CHECKPOINT_STEP


def confirmation_generator_source_contract(observed):
    """Derive the exact reviewed source identity for the secret-bound generator."""
    sources = {}
    for path in CANONICAL_CONFIRMATION_GENERATOR_SOURCES:
        name = "source:{}".format(path)
        digest = observed.get(name)
        if not re.fullmatch(r"[0-9a-f]{64}", str(digest)):
            raise ValueError(
                "confirmation generator source is absent or invalid: {}".format(path)
            )
        sources[path] = digest
    return {
        "schema": CANONICAL_CONFIRMATION_GENERATOR_SCHEMA,
        "entrypoint": CANONICAL_CONFIRMATION_GENERATOR_ENTRYPOINT,
        "sources": sources,
        "manifest_sha256": stable_json_sha256(sources),
    }


def _state_operand_identity(state):
    state = dict(state)
    left = sum(int(digit) * (10**index) for index, digit in enumerate(state["a"]))
    right = sum(int(digit) * (10**index) for index, digit in enumerate(state["b"]))
    identity = {
        "operation": state["op"],
        "width": int(state["w"]),
        "left": left,
        "right": right,
    }
    return {**identity, "sha256": stable_json_sha256(identity)}


def confirmation_exclusion_contract(episodes_text, cycle_text):
    """Derive the sole ordered confirmation exclusion set from frozen inputs."""
    prompt_hashes = set()
    operand_by_hash = {}
    episode_map = {}
    source_rows = [raw for raw in str(episodes_text).splitlines() if raw.strip()]
    for source_index, raw in enumerate(source_rows):
        outer = _load_exact_json(raw, "confirmation episodes[{}]".format(source_index))
        if not isinstance(outer, dict) or not isinstance(
            outer.get("counterfactual"), dict
        ):
            raise ValueError("confirmation episode schema is invalid")
        if outer.get("id") in episode_map:
            raise ValueError("confirmation episode identity is duplicated")
        episode_map[outer.get("id")] = outer
        for branch in (outer, outer["counterfactual"]):
            initial = parse_state(branch.get("initial_state"))
            if initial is None:
                raise ValueError("confirmation episode has an invalid initial state")
            operand = _state_operand_identity(initial)
            prior = operand_by_hash.setdefault(operand["sha256"], operand)
            if prior != operand:
                raise RuntimeError("confirmation operand identity hash collision")
            for state, _next_state in _episode_states(branch):
                for style in FIT_STYLES:
                    prompt = microstep_prompt(state, style=style)
                    prompt_hashes.add(hashlib.sha256(prompt.encode()).hexdigest())

    cycle = cycle_validation_contract(cycle_text, episode_map)
    for case in cycle["cases"]:
        for name in ("first_prompt", "second_prompt"):
            prompt_hashes.add(hashlib.sha256(case[name].encode()).hexdigest())
        state = dws_prompt_state(case["first_prompt"])
        if state is None:
            raise ValueError("confirmation cycle prompt is not canonical")
        operand = _state_operand_identity(state)
        prior = operand_by_hash.setdefault(operand["sha256"], operand)
        if prior != operand:
            raise RuntimeError("confirmation operand identity hash collision")

    identities = [
        {"kind": "prompt", "sha256": digest} for digest in sorted(prompt_hashes)
    ]
    identities.extend(
        {"kind": "operand", **operand_by_hash[digest]}
        for digest in sorted(operand_by_hash)
    )
    contract = {
        "audit": CANONICAL_CONFIRMATION_EXCLUSION_SCHEMA,
        "episodes_sha256": hashlib.sha256(str(episodes_text).encode()).hexdigest(),
        "cycle_sha256": hashlib.sha256(str(cycle_text).encode()).hexdigest(),
        "prompt_count": len(prompt_hashes),
        "operand_count": len(operand_by_hash),
        "identity_count": len(identities),
        "identities": identities,
        "identity_sha256": stable_json_sha256(identities),
    }
    if (
        contract["episodes_sha256"] == EXPECTED_EPISODES_SHA256
        and contract["cycle_sha256"] == EXPECTED_CYCLE_SHA256
        and (
            contract["prompt_count"] != EXPECTED_CONFIRMATION_PROMPT_EXCLUSIONS
            or contract["operand_count"] != EXPECTED_CONFIRMATION_OPERAND_EXCLUSIONS
            or contract["identity_count"]
            != EXPECTED_CONFIRMATION_PROMPT_EXCLUSIONS
            + EXPECTED_CONFIRMATION_OPERAND_EXCLUSIONS
            or contract["identity_sha256"] != EXPECTED_CONFIRMATION_EXCLUSION_SHA256
        )
    ):
        raise RuntimeError("canonical confirmation exclusion identity changed")
    return contract


def _derive_confirmation_board(secret, episodes_text, cycle_text):
    """Derive rows after the public canonical binding checks have succeeded."""
    if type(secret) is not bytes or len(secret) != 32:
        raise ValueError("confirmation secret must contain exactly 256 bits")
    exclusion = confirmation_exclusion_contract(episodes_text, cycle_text)
    forbidden_prompts = {
        identity["sha256"]
        for identity in exclusion["identities"]
        if identity["kind"] == "prompt"
    }
    forbidden_operands = {
        identity["sha256"]
        for identity in exclusion["identities"]
        if identity["kind"] == "operand"
    }
    seed_contract = {
        "secret_sha256": hashlib.sha256(secret).hexdigest(),
        "episodes_sha256": exclusion["episodes_sha256"],
        "cycle_sha256": exclusion["cycle_sha256"],
        "exclusion_identity_sha256": exclusion["identity_sha256"],
    }
    seed = hashlib.sha256(
        b"causal-carry-motor-confirmation-v2\0"
        + json.dumps(seed_contract, sort_keys=True, separators=(",", ":")).encode()
    ).digest()
    rng = random.Random(int.from_bytes(seed, "big"))
    rows = []
    seen_prompts = set()
    seen_operands = set()
    for width in (4, 6, 8, 10):
        limit = 10**width
        for operation in ("add", "sub"):
            for style in FIT_STYLES:
                for target in (0, 1):
                    for replicate in range(8):
                        admitted_positions = tuple(
                            range(width - 1)
                            if operation == "sub" and target == 1
                            else range(width)
                        )
                        desired_position = admitted_positions[
                            replicate % len(admitted_positions)
                        ]
                        for _attempt in range(100_000):
                            left, right = rng.randrange(limit), rng.randrange(limit)
                            if operation == "sub" and left < right:
                                left, right = right, left
                            operand_identity = {
                                "operation": operation,
                                "width": width,
                                "left": left,
                                "right": right,
                            }
                            operand_sha256 = stable_json_sha256(operand_identity)
                            if (
                                operand_sha256 in forbidden_operands
                                or operand_sha256 in seen_operands
                            ):
                                continue
                            initial = initial_state(operation, left, right, width)
                            state = initial
                            candidates = []
                            expected_states = []
                            while not state["z"]:
                                next_state = apply_microstep(state)
                                expected_states.append(canonical_state(next_state))
                                if (
                                    state["p"] == desired_position
                                    and next_state["c"] == target
                                ):
                                    prompt = microstep_prompt(state, style=style)
                                    prompt_sha256 = hashlib.sha256(
                                        prompt.encode()
                                    ).hexdigest()
                                    if (
                                        prompt_sha256 not in forbidden_prompts
                                        and prompt_sha256 not in seen_prompts
                                    ):
                                        candidates.append(
                                            (
                                                dict(state),
                                                dict(next_state),
                                                prompt,
                                                prompt_sha256,
                                            )
                                        )
                                state = next_state
                            if not candidates:
                                continue
                            selected = candidates[0]
                            state, next_state, prompt, prompt_sha256 = selected
                            seen_operands.add(operand_sha256)
                            seen_prompts.add(prompt_sha256)
                            episode_id = "confirmation-{:03d}".format(len(rows))
                            episode = {
                                "id": episode_id,
                                "split": "confirmation_w{}".format(width),
                                "operation": operation,
                                "width": width,
                                "left": left,
                                "right": right,
                                "prompt_style": style,
                                "initial_state": canonical_state(initial),
                                "expected_states": expected_states,
                                "expected_answer": int(
                                    state_answer(parse_state(expected_states[-1]))
                                ),
                            }
                            rows.append(
                                {
                                    "index": len(rows),
                                    "id": episode_id,
                                    "regime": "width_{}".format(width),
                                    "operation": operation,
                                    "width": width,
                                    "style": style,
                                    "target_carry": target,
                                    "position": int(state["p"]),
                                    "operand_sha256": operand_sha256,
                                    "prompt_sha256": prompt_sha256,
                                    "prompt": prompt,
                                    "expected_state": canonical_state(next_state),
                                    "selected_transition": int(state["p"]),
                                    "episode": episode,
                                }
                            )
                            break
                        else:
                            raise RuntimeError(
                                "confirmation generator exhausted a frozen cell"
                            )
    if len(rows) != 256:
        raise RuntimeError("confirmation generator did not produce 256 rows")
    return {
        "audit": CANONICAL_CONFIRMATION_GENERATOR_SCHEMA,
        "secret_sha256": hashlib.sha256(secret).hexdigest(),
        "exclusion_contract": exclusion,
        "rows": rows,
        "rows_sha256": stable_json_sha256(rows),
    }


def generate_confirmation_board(
    secret,
    bound,
    frozen,
    confirmation_commitment,
    plan_path,
    plan_sha256,
    plan_document,
):
    """Derive the sole board after independently binding the canonical plan."""
    if type(secret) is not bytes or len(secret) != 32:
        raise ValueError("confirmation secret must contain exactly 256 bits")
    if not re.fullmatch(r"[0-9a-f]{64}", str(plan_sha256)):
        raise ValueError("confirmation plan receipt is invalid")
    source_contract = confirmation_commitment.get("source_contract")
    if not isinstance(source_contract, dict):
        raise ValueError("confirmation commitment source contract is absent")
    source_commit = source_contract.get("git_commit")
    expected_plan_path = canonical_plan_root(source_commit) / "plan.json"
    raw_plan_path = os.fspath(plan_path)
    plan_path = Path(plan_path)
    if raw_plan_path != str(expected_plan_path) or plan_path != expected_plan_path:
        raise ValueError("confirmation plan path is not the exact canonical path")
    validate_canonical_plan_layout(plan_path, source_commit)
    plan_bound = BoundInput(plan_path)
    try:
        if plan_bound.path != expected_plan_path:
            raise ValueError("confirmation plan path changed before binding")
        validate_artifact_receipt(plan_bound, plan_sha256)
        plan = _load_exact_json(plan_bound.text(), "confirmation canonical plan")
        if (
            not isinstance(plan_document, dict)
            or plan_document != plan
            or stable_json_sha256(plan_document) != stable_json_sha256(plan)
        ):
            raise ValueError(
                "supplied confirmation plan document differs from bound bytes"
            )
        if (
            plan.get("audit") != CANONICAL_PLAN_AUDIT
            or plan.get("canonical") is not True
            or plan.get("source_contract") != source_contract
            or plan.get("plan_path") != str(expected_plan_path)
        ):
            raise ValueError("confirmation plan content is not canonical")
        validate_plan_confirmation_binding(plan, bound, frozen, confirmation_commitment)

        expected_hashes = {
            "checkpoint": EXPECTED_CHECKPOINT_SHA256,
            "episodes": EXPECTED_EPISODES_SHA256,
            "tokenizer": EXPECTED_TOKENIZER_SHA256,
            "cycle": EXPECTED_CYCLE_SHA256,
        }
        if {name: frozen.get(name) for name in expected_hashes} != expected_hashes:
            raise ValueError("confirmation frozen input identities are not canonical")
        expected_inputs = {}
        for name, expected_sha256 in expected_hashes.items():
            item = bound.get(name)
            if (
                item is None
                or item.sha256 != expected_sha256
                or not Path(item.path).is_absolute()
            ):
                raise ValueError(
                    "confirmation frozen input binding mismatch: {}".format(name)
                )
            item.verify_path()
            expected_inputs[name] = {
                "path": str(item.path),
                "sha256": expected_sha256,
            }
        if plan.get("frozen_inputs") != expected_inputs:
            raise ValueError("confirmation frozen inputs differ from canonical plan")

        commitment_item = bound.get("confirmation_commitment")
        expected_commitment = {
            "path": str(commitment_item.path) if commitment_item is not None else "",
            "sha256": frozen.get("confirmation_commitment"),
            "document": confirmation_commitment,
        }
        if (
            commitment_item is None
            or commitment_item.sha256 != frozen.get("confirmation_commitment")
            or plan.get("confirmation_commitment") != expected_commitment
        ):
            raise ValueError("confirmation commitment differs from canonical plan")
        commitment_item.verify_path()

        episodes_payload = bound["episodes"].bytes()
        cycle_payload = bound["cycle"].bytes()
        if (
            hashlib.sha256(episodes_payload).hexdigest() != EXPECTED_EPISODES_SHA256
            or hashlib.sha256(cycle_payload).hexdigest() != EXPECTED_CYCLE_SHA256
        ):
            raise ValueError("confirmation source bytes differ from frozen identities")
        episodes_text = episodes_payload.decode()
        cycle_text = cycle_payload.decode()
        exclusion = confirmation_exclusion_contract(episodes_text, cycle_text)
        secret_sha256 = hashlib.sha256(secret).hexdigest()
        if (
            confirmation_commitment.get("secret_sha256") != secret_sha256
            or confirmation_commitment.get("exclusion_contract") != exclusion
            or plan.get("confirmation_exclusion_contract") != exclusion
        ):
            raise ValueError("confirmation secret or exclusion differs from commitment")

        derived = _derive_confirmation_board(secret, episodes_text, cycle_text)
        if derived["exclusion_contract"] != exclusion:
            raise RuntimeError("confirmation derivation changed its exclusion contract")
        validate_canonical_plan_layout(plan_bound.path, source_commit)
        plan_bound.verify_path()
        return {
            "audit": CANONICAL_CONFIRMATION_GENERATOR_SCHEMA,
            "secret_sha256": secret_sha256,
            "frozen_inputs": expected_inputs,
            "confirmation_commitment": expected_commitment,
            "exclusion_contract": exclusion,
            "plan": {"path": str(plan_bound.path), "sha256": plan_bound.sha256},
            "rows": derived["rows"],
            "rows_sha256": derived["rows_sha256"],
        }
    finally:
        plan_bound.close()


def _load_exact_json(text, label):
    def object_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("{} contains a duplicate JSON key".format(label))
            result[key] = value
        return result

    def reject_constant(value):
        raise ValueError("{} contains non-finite JSON: {}".format(label, value))

    try:
        return json.loads(
            text,
            object_pairs_hook=object_pairs,
            parse_constant=reject_constant,
        )
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("{} is not valid JSON".format(label)) from exc


def tensor_state_sha256(state):
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode() + b"\0")
        digest.update(str(tensor.dtype).encode() + b"\0")
        digest.update(json.dumps(list(tensor.shape)).encode() + b"\0")
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def full_logit_tensor_identity(logits):
    tensor = logits.detach().cpu().contiguous()
    payload = tensor.view(torch.uint8).numpy().tobytes()
    evidence = {
        "dtype": str(tensor.dtype),
        "shape": list(tensor.shape),
        "byte_count": len(payload),
        "bytes_sha256": hashlib.sha256(payload).hexdigest(),
    }
    return {**evidence, "identity_sha256": stable_json_sha256(evidence)}


def feature_payload_sha256(features):
    """Hash a compact feature mapping without dtype-losing serialization."""
    digest = hashlib.sha256()
    for name in sorted(features):
        value = features[name]
        digest.update(name.encode() + b"\0")
        if isinstance(value, torch.Tensor):
            tensor = value.detach().cpu().contiguous()
            digest.update(b"tensor\0")
            digest.update(str(tensor.dtype).encode() + b"\0")
            digest.update(json.dumps(list(tensor.shape)).encode() + b"\0")
            digest.update(tensor.view(torch.uint8).numpy().tobytes())
        else:
            digest.update(b"json\0")
            digest.update(
                json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
            )
        digest.update(b"\0")
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


def feature_shard_indices(size, shard_index, shard_count):
    if shard_count <= 0 or not 0 <= shard_index < shard_count:
        raise ValueError("invalid feature-shard coordinates")
    return list(range(int(shard_index), int(size), int(shard_count)))


def feature_sentinel_indices(rows):
    """Choose one stable cross-node sentinel for every token-length shape."""
    first = {}
    for index, row in enumerate(rows):
        key = (len(row["prompt_ids"]), len(row["prefix_ids"]))
        first.setdefault(key, index)
    if len(first) != 4:
        raise RuntimeError(
            "canonical fit board must have exactly four token-length shapes"
        )
    return [first[key] for key in sorted(first)]


def row_identity_sha256(rows, indices):
    identities = [
        {
            "index": int(index),
            "prefix_sha256": rows[index]["prefix_sha256"],
            "target": int(rows[index]["target"]),
            "target_id": int(rows[index]["target_id"]),
        }
        for index in indices
    ]
    return stable_json_sha256(identities)


def _validate_feature_mapping(
    features,
    expected_rows,
    expected_labels,
    *,
    canonical=False,
    d_model=None,
    vocab_size=None,
    zero_id=None,
    one_id=None,
):
    expected_keys = {
        "hidden",
        "base01",
        "other_lse",
        "other_max",
        "other_max_token_id",
        "other_logits",
        "other_token_ids",
        "zero_id",
        "one_id",
        "deployment_logit_dtype",
        "labels",
    }
    if canonical and set(features) != expected_keys:
        raise ValueError("canonical feature schema mismatch")
    row_keys = (
        "hidden",
        "base01",
        "other_lse",
        "other_max",
        "other_max_token_id",
        "labels",
    )
    for name in row_keys:
        value = features.get(name)
        if not isinstance(value, torch.Tensor) or len(value) != expected_rows:
            raise ValueError("invalid feature tensor: {}".format(name))
    if features.get("other_logits") is not None:
        raise ValueError("fit feature shards must not retain full vocabulary logits")
    if not isinstance(features.get("other_token_ids"), torch.Tensor):
        raise ValueError("feature shard is missing other-token identity")
    if features.get("deployment_logit_dtype") not in {
        "torch.bfloat16",
        "torch.float32",
    }:
        raise ValueError("invalid feature deployment dtype")
    labels = torch.as_tensor(expected_labels, dtype=torch.long)
    if not torch.equal(features["labels"].long(), labels):
        raise ValueError("feature labels do not match frozen rows")
    if not canonical:
        return
    expected_dtypes = {
        "hidden": torch.float32,
        "base01": torch.bfloat16,
        "other_lse": torch.float32,
        "other_max": torch.float32,
        "other_max_token_id": torch.int64,
        "labels": torch.int64,
        "other_token_ids": torch.int64,
    }
    for name, dtype in expected_dtypes.items():
        if features[name].dtype != dtype:
            raise ValueError("canonical feature dtype mismatch: {}".format(name))
    if features["deployment_logit_dtype"] != "torch.bfloat16":
        raise ValueError("canonical feature logits must be bfloat16")
    if d_model is None or vocab_size is None or zero_id is None or one_id is None:
        raise ValueError("canonical feature schema requires model dimensions")
    expected_shapes = {
        "hidden": (expected_rows, int(d_model)),
        "base01": (expected_rows, 2),
        "other_lse": (expected_rows,),
        "other_max": (expected_rows,),
        "other_max_token_id": (expected_rows,),
        "labels": (expected_rows,),
    }
    for name, shape in expected_shapes.items():
        if tuple(features[name].shape) != shape:
            raise ValueError("canonical feature shape mismatch: {}".format(name))
    for name in ("hidden", "base01", "other_lse", "other_max"):
        if not bool(torch.isfinite(features[name].float()).all()):
            raise ValueError(
                "canonical feature contains non-finite values: {}".format(name)
            )
    if (features["zero_id"], features["one_id"]) != (int(zero_id), int(one_id)):
        raise ValueError("canonical carry-token identity mismatch")
    expected_other_ids = torch.arange(int(vocab_size), dtype=torch.long)
    expected_other_ids = expected_other_ids[
        (expected_other_ids != int(zero_id)) & (expected_other_ids != int(one_id))
    ]
    if not torch.equal(features["other_token_ids"], expected_other_ids):
        raise ValueError("canonical other-token vocabulary identity mismatch")
    if not bool(torch.isin(features["other_max_token_id"], expected_other_ids).all()):
        raise ValueError("canonical other-token maxima have invalid token ids")


def validate_canonical_feature_shard(
    shard,
    artifact_path,
    rows,
    expected_bindings,
    source_contract,
    plan,
    plan_sha256,
    expected_shard_index,
):
    if set(shard) != CANONICAL_SHARD_KEYS:
        raise ValueError("canonical feature-shard schema mismatch")
    if (
        shard.get("audit") != CANONICAL_SHARD_AUDIT
        or shard.get("canonical") is not True
    ):
        raise ValueError("feature shard is not canonical")
    if shard.get("plan_sha256") != plan_sha256:
        raise ValueError("feature-shard plan binding mismatch")
    for name, expected in expected_bindings.items():
        if shard.get(name) != expected:
            raise ValueError("feature-shard input mismatch: {}".format(name))
    if shard.get("confirmation_commitment_sha256") != plan.get(
        "confirmation_commitment", {}
    ).get("sha256"):
        raise ValueError("feature-shard confirmation commitment mismatch")
    if shard.get("source_contract") != source_contract:
        raise ValueError("feature-shard source contract mismatch")
    if (
        shard.get("board") != plan["board"]
        or shard.get("board_rows_sha256") != plan["board_rows_sha256"]
    ):
        raise ValueError("feature-shard board mismatch")
    if (
        not _is_canonical_checkpoint_step(plan.get("checkpoint_step"))
        or not _is_canonical_checkpoint_step(shard.get("checkpoint_step"))
        or shard.get("checkpoint_step") != plan["checkpoint_step"]
    ):
        raise ValueError("feature-shard checkpoint-step mismatch")
    if (
        shard.get("shard_count") != CANONICAL_FEATURE_SHARDS
        or shard.get("extract_batch") != CANONICAL_EXTRACT_BATCH
        or shard.get("shard_index") != expected_shard_index
    ):
        raise ValueError("feature-shard coordinate or batch mismatch")
    descriptor = plan["shards"][expected_shard_index]
    path = str(Path(artifact_path).resolve())
    if path != descriptor["artifact"]:
        raise ValueError("feature-shard artifact path mismatch")
    indices = feature_shard_indices(
        len(rows), expected_shard_index, CANONICAL_FEATURE_SHARDS
    )
    expected_descriptor = {
        "shard_index": expected_shard_index,
        "rows": len(indices),
        "global_indices_sha256": stable_json_sha256(indices),
        "row_identity_sha256": row_identity_sha256(rows, indices),
        "artifact": path,
    }
    if descriptor != expected_descriptor or shard.get("global_indices") != indices:
        raise ValueError("feature-shard plan descriptor mismatch")
    if shard.get("row_identity_sha256") != expected_descriptor["row_identity_sha256"]:
        raise ValueError("feature-shard row identity mismatch")
    sentinels = feature_sentinel_indices(rows)
    if shard.get("sentinel_indices") != sentinels or shard.get(
        "sentinel_row_identity_sha256"
    ) != row_identity_sha256(rows, sentinels):
        raise ValueError("feature-shard sentinel identity mismatch")
    features = shard.get("features")
    sentinel_features = shard.get("sentinel_features")
    if not isinstance(features, dict) or not isinstance(sentinel_features, dict):
        raise ValueError("feature-shard payload is missing")
    dimensions = {
        "canonical": True,
        "d_model": plan["d_model"],
        "vocab_size": plan["vocab_size"],
        "zero_id": plan["zero_id"],
        "one_id": plan["one_id"],
    }
    _validate_feature_mapping(
        features,
        len(indices),
        [rows[index]["target"] for index in indices],
        **dimensions,
    )
    _validate_feature_mapping(
        sentinel_features,
        len(sentinels),
        [rows[index]["target"] for index in sentinels],
        **dimensions,
    )
    if shard.get("feature_payload_sha256") != feature_payload_sha256(features):
        raise ValueError("feature-shard tensor hash mismatch")
    if shard.get("sentinel_payload_sha256") != feature_payload_sha256(
        sentinel_features
    ):
        raise ValueError("feature-shard sentinel tensor hash mismatch")
    runtime = shard.get("runtime")
    if runtime != plan["runtime_contract"]["artifact_runtime"]:
        raise ValueError("feature-shard runtime is not canonical H100 CUDA")
    if shard.get("claim_boundary") != CANONICAL_SHARD_CLAIM_BOUNDARY:
        raise ValueError("feature-shard claim boundary is invalid")
    primary_positions = {index: position for position, index in enumerate(indices)}
    row_keys = (
        "hidden",
        "base01",
        "other_lse",
        "other_max",
        "other_max_token_id",
        "labels",
    )
    for sentinel_position, global_index in enumerate(sentinels):
        if global_index not in primary_positions:
            continue
        primary_position = primary_positions[global_index]
        for name in row_keys:
            if not torch.equal(
                features[name][primary_position],
                sentinel_features[name][sentinel_position],
            ):
                raise ValueError("same-process sentinel mismatch: {}".format(name))
    return indices, features, sentinel_features, runtime


def merge_feature_shards(
    shards, rows, expected_bindings, source_contract, plan, plan_sha256
):
    """Validate and merge a closed-world set of immutable feature shards."""
    if len(shards) != CANONICAL_FEATURE_SHARDS:
        raise ValueError("canonical feature-shard set is incomplete")
    size = len(rows)
    sentinel_indices = feature_sentinel_indices(rows)
    board_rows_sha256 = stable_json_sha256(rows)
    seen_shard_indices = set()
    covered = torch.zeros(size, dtype=torch.bool)
    merged = None
    reference_sentinels = None
    reference_runtime = None
    receipts = []
    row_keys = (
        "hidden",
        "base01",
        "other_lse",
        "other_max",
        "other_max_token_id",
        "labels",
    )
    for receipt_sha256, artifact_path, shard in shards:
        try:
            expected_shard_index = int(shard.get("shard_index", -1))
        except (TypeError, ValueError) as exc:
            raise ValueError("invalid feature-shard index") from exc
        validate_canonical_feature_shard(
            shard,
            artifact_path,
            rows,
            expected_bindings,
            source_contract,
            plan,
            plan_sha256,
            expected_shard_index,
        )
        if set(shard) != CANONICAL_SHARD_KEYS:
            raise ValueError("canonical feature-shard schema mismatch")
        if shard.get("audit") != CANONICAL_SHARD_AUDIT:
            raise ValueError("unsupported carry feature-shard version")
        if shard.get("canonical") is not True:
            raise ValueError("feature shard is not canonical")
        if shard.get("plan_sha256") != plan_sha256:
            raise ValueError("feature-shard plan binding mismatch")
        for name, expected in expected_bindings.items():
            if shard.get(name) != expected:
                raise ValueError("feature-shard input mismatch: {}".format(name))
        if shard.get("source_contract") != source_contract:
            raise ValueError("feature-shard source contract mismatch")
        if shard.get("board_rows_sha256") != board_rows_sha256:
            raise ValueError("feature-shard board mismatch")
        if shard.get("board") != plan["board"]:
            raise ValueError("feature-shard board report mismatch")
        if (
            not _is_canonical_checkpoint_step(plan.get("checkpoint_step"))
            or not _is_canonical_checkpoint_step(shard.get("checkpoint_step"))
            or shard.get("checkpoint_step") != plan["checkpoint_step"]
        ):
            raise ValueError("feature-shard checkpoint-step mismatch")
        if shard.get("shard_count") != CANONICAL_FEATURE_SHARDS:
            raise ValueError("feature-shard count mismatch")
        if shard.get("extract_batch") != CANONICAL_EXTRACT_BATCH:
            raise ValueError("feature-shard extraction batch mismatch")
        shard_index = int(shard.get("shard_index", -1))
        if shard_index in seen_shard_indices:
            raise ValueError("duplicate feature-shard index")
        seen_shard_indices.add(shard_index)
        descriptor = plan["shards"][shard_index]
        if str(Path(artifact_path).resolve()) != descriptor["artifact"]:
            raise ValueError("feature-shard artifact path mismatch")
        indices = feature_shard_indices(size, shard_index, CANONICAL_FEATURE_SHARDS)
        if shard.get("global_indices") != indices:
            raise ValueError("feature-shard index assignment mismatch")
        if shard.get("row_identity_sha256") != row_identity_sha256(rows, indices):
            raise ValueError("feature-shard row identity mismatch")
        if descriptor != {
            "shard_index": shard_index,
            "rows": len(indices),
            "global_indices_sha256": stable_json_sha256(indices),
            "row_identity_sha256": row_identity_sha256(rows, indices),
            "artifact": str(Path(artifact_path).resolve()),
        }:
            raise ValueError("feature-shard plan descriptor mismatch")
        if shard.get("sentinel_indices") != sentinel_indices:
            raise ValueError("feature-shard sentinel assignment mismatch")
        if shard.get("sentinel_row_identity_sha256") != row_identity_sha256(
            rows, sentinel_indices
        ):
            raise ValueError("feature-shard sentinel row identity mismatch")
        features = shard.get("features")
        sentinels = shard.get("sentinel_features")
        if not isinstance(features, dict) or not isinstance(sentinels, dict):
            raise ValueError("feature-shard payload is missing")
        validation_dimensions = {
            "canonical": True,
            "d_model": plan["d_model"],
            "vocab_size": plan["vocab_size"],
            "zero_id": plan["zero_id"],
            "one_id": plan["one_id"],
        }
        _validate_feature_mapping(
            features,
            len(indices),
            [rows[index]["target"] for index in indices],
            **validation_dimensions,
        )
        _validate_feature_mapping(
            sentinels,
            len(sentinel_indices),
            [rows[index]["target"] for index in sentinel_indices],
            **validation_dimensions,
        )
        if shard.get("feature_payload_sha256") != feature_payload_sha256(features):
            raise ValueError("feature-shard tensor hash mismatch")
        if shard.get("sentinel_payload_sha256") != feature_payload_sha256(sentinels):
            raise ValueError("feature-shard sentinel tensor hash mismatch")
        runtime = shard.get("runtime")
        if runtime != plan["runtime_contract"]["artifact_runtime"]:
            raise ValueError("feature-shard runtime is not canonical H100 CUDA")
        if reference_runtime is None:
            reference_runtime = runtime
        elif runtime != reference_runtime:
            raise ValueError("feature-shard runtime mismatch")
        if reference_sentinels is None:
            reference_sentinels = sentinels
        else:
            for name in sorted(sentinels):
                left, right = reference_sentinels[name], sentinels[name]
                if isinstance(left, torch.Tensor):
                    if not isinstance(right, torch.Tensor) or not torch.equal(
                        left, right
                    ):
                        raise ValueError(
                            "cross-node sentinel mismatch: {}".format(name)
                        )
                elif left != right:
                    raise ValueError("cross-node sentinel mismatch: {}".format(name))
        primary_positions = {index: position for position, index in enumerate(indices)}
        for sentinel_position, global_index in enumerate(sentinel_indices):
            if global_index not in primary_positions:
                continue
            primary_position = primary_positions[global_index]
            for name in row_keys:
                if not torch.equal(
                    features[name][primary_position],
                    sentinels[name][sentinel_position],
                ):
                    raise ValueError("same-process sentinel mismatch: {}".format(name))
        if bool(covered[indices].any()):
            raise ValueError("feature-shard rows overlap")
        covered[indices] = True
        if merged is None:
            merged = {
                name: torch.empty(
                    (size,) + tuple(features[name].shape[1:]),
                    dtype=features[name].dtype,
                )
                for name in row_keys
            }
            merged.update(
                {
                    "other_logits": None,
                    "other_token_ids": features["other_token_ids"].clone(),
                    "zero_id": int(features["zero_id"]),
                    "one_id": int(features["one_id"]),
                    "deployment_logit_dtype": features["deployment_logit_dtype"],
                }
            )
        for name in row_keys:
            if features[name].dtype != merged[name].dtype:
                raise ValueError("feature-shard dtype mismatch: {}".format(name))
            merged[name][indices] = features[name]
        for name in ("other_token_ids",):
            if not torch.equal(features[name], merged[name]):
                raise ValueError("feature-shard vocabulary identity mismatch")
        for name in ("zero_id", "one_id", "deployment_logit_dtype"):
            if features[name] != merged[name]:
                raise ValueError("feature-shard metadata mismatch: {}".format(name))
        receipts.append(
            {
                "shard_index": shard_index,
                "artifact_sha256": receipt_sha256,
                "artifact": str(Path(artifact_path).resolve()),
                "feature_payload_sha256": shard["feature_payload_sha256"],
                "rows": len(indices),
            }
        )
    if seen_shard_indices != set(range(CANONICAL_FEATURE_SHARDS)):
        raise ValueError("feature-shard index set is incomplete")
    if not bool(covered.all()) or merged is None:
        raise ValueError("feature-shard coverage has gaps")
    _validate_feature_mapping(
        merged,
        size,
        [int(row["target"]) for row in rows],
        canonical=True,
        d_model=plan["d_model"],
        vocab_size=plan["vocab_size"],
        zero_id=plan["zero_id"],
        one_id=plan["one_id"],
    )
    receipts.sort(key=lambda item: item["shard_index"])
    return merged, {
        "shards": receipts,
        "sentinel_indices": sentinel_indices,
        "sentinel_payload_sha256": feature_payload_sha256(reference_sentinels),
        "runtime": reference_runtime,
        "plan_sha256": plan_sha256,
        "merged_feature_payload_sha256": feature_payload_sha256(merged),
        "teacher_metric_feature_payload_sha256": teacher_metric_feature_payload_sha256(
            merged
        ),
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
    adjusted = _singleton_deployment_adjusted_carry_logits(
        features["hidden"], features["base01"], motor, "cpu"
    )
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
    }
    return result


def _teacher_row_identity(row):
    return {
        "index": int(row["index"]),
        "selection_index": int(row["selection_index"]),
        "source_index": int(row["source_index"]),
        "selected_episode_id": row["selected_episode_id"],
        "source_split": row["source_split"],
        "regime": row["regime"],
        "branch": row["branch"],
        "episode_id": row["episode_id"],
        "transition": int(row["transition"]),
        "operation": row["operation"],
        "width": int(row["width"]),
        "position": int(row["position"]),
        "style": row["style"],
        "current_carry": int(row["current_carry"]),
        "prompt": row["prompt"],
        "response_prefix": row["response_prefix"],
        "prompt_token_ids_sha256": stable_json_sha256(row["prompt_ids"]),
        "prefix_token_ids_sha256": stable_json_sha256(row["prefix_ids"]),
    }


def _aggregate_teacher_rows(rows):
    size = len(rows)
    if size <= 0:
        raise ValueError("teacher-forced evidence is empty")
    carry_correct = sum(int(row["carry_pair_correct"]) for row in rows)
    global_correct = sum(int(row["global_correct"]) for row in rows)
    return {
        "carry_pair_correct": carry_correct,
        "carry_pair_accuracy": carry_correct / size,
        "global_correct": global_correct,
        "global_accuracy": global_correct / size,
        "rows": size,
        "prediction_ones": sum(int(row["carry_prediction"] == 1) for row in rows),
        "site_opportunities": sum(int(row["site_opportunities"]) for row in rows),
        "motor_fires": sum(int(row["motor_fires"]) for row in rows),
    }


@torch.no_grad()
def teacher_forced_metric_evidence(
    features, rows, motor, arm, device, scoring_contract
):
    """Retain the complete minimal per-row evidence behind every metric."""
    if arm not in {"base", "dead", "treatment", "shuffled"}:
        raise ValueError("unknown teacher-forced arm")
    if len(rows) != len(features["base01"]):
        raise ValueError("teacher-forced row evidence is incomplete")
    adjusted = _singleton_deployment_adjusted_carry_logits(
        features["hidden"], features["base01"], motor, device
    )
    zero_id = int(features["zero_id"])
    one_id = int(features["one_id"])
    evidence = []
    for index, expected in enumerate(rows):
        target = int(expected["target"])
        target_id = zero_id if target == 0 else one_id
        zero_logit = float(adjusted[index, 0])
        one_logit = float(adjusted[index, 1])
        other_max_logit = float(features["other_max"][index])
        other_max_token_id = int(features["other_max_token_id"][index])
        carry_prediction = (
            0
            if zero_logit > one_logit or (zero_logit == one_logit and zero_id < one_id)
            else 1
        )
        global_prediction_token_id = max(
            (
                (zero_logit, -zero_id, zero_id),
                (one_logit, -one_id, one_id),
                (other_max_logit, -other_max_token_id, other_max_token_id),
            )
        )[2]
        router = CarryRouter(expected["prompt"], arm != "base")
        router.observe(expected["response_prefix"])
        evidence.append(
            {
                "identity": _teacher_row_identity(expected),
                "target": target,
                "target_token_id": target_id,
                "adjusted_zero_logit": zero_logit,
                "adjusted_one_logit": one_logit,
                "other_max_logit": other_max_logit,
                "other_max_token_id": other_max_token_id,
                "carry_prediction": carry_prediction,
                "global_prediction_token_id": global_prediction_token_id,
                "carry_pair_correct": carry_prediction == target,
                "global_correct": global_prediction_token_id == target_id,
                "site_opportunities": router.site_count,
                "motor_fires": router.fire_count,
            }
        )
    return {
        "zero_id": zero_id,
        "one_id": one_id,
        "deployment_logit_dtype": str(adjusted.dtype),
        "scoring_contract": scoring_contract,
        "summary": _aggregate_teacher_rows(evidence),
        "rows": evidence,
    }


def _require_canonical_teacher_scoring_runtime(features, motors, device):
    if (
        torch.device(device).type != "cuda"
        or require_canonical_cuda_runtime() != "cuda"
        or features.get("deployment_logit_dtype") != "torch.bfloat16"
        or features.get("base01", torch.empty(0)).dtype != torch.bfloat16
    ):
        raise RuntimeError("canonical teacher scoring requires the exact H100 path")
    for motor in motors:
        if motor is not None and any(
            parameter.device.type != "cuda" for parameter in motor.parameters()
        ):
            raise RuntimeError("canonical teacher motor is not resident on the H100")


def canonical_teacher_forced_metric_evidence(features, rows, motor, arm, device):
    _require_canonical_teacher_scoring_runtime(features, (motor,), device)
    return teacher_forced_metric_evidence(
        features,
        rows,
        motor,
        arm,
        device,
        CANONICAL_TEACHER_SCORING_CONTRACT,
    )


def _fit_teacher_row_identity(row, index):
    return {
        "index": int(index),
        "operation": row["operation"],
        "width": int(row["width"]),
        "position": int(row["position"]),
        "style": row["style"],
        "current_carry": int(row["current_carry"]),
        "target": int(row["target"]),
        "target_id": int(row["target_id"]),
        "prompt_sha256": row["prompt_sha256"],
        "prefix_sha256": row["prefix_sha256"],
    }


def teacher_metric_feature_payload_sha256(features):
    return feature_payload_sha256(
        {
            "base01": features["base01"],
            "deployment_logit_dtype": features["deployment_logit_dtype"],
            "hidden": features["hidden"],
            "labels": features["labels"],
            "one_id": int(features["one_id"]),
            "other_max": features["other_max"],
            "other_max_token_id": features["other_max_token_id"],
            "zero_id": int(features["zero_id"]),
        }
    )


@torch.no_grad()
def _singleton_deployment_adjusted_carry_logits(hidden, base01, motor, device):
    """Replay autonomous motor arithmetic on one row per forward call."""
    if (
        not isinstance(hidden, torch.Tensor)
        or not isinstance(base01, torch.Tensor)
        or hidden.ndim != 2
        or base01.ndim != 2
        or base01.shape != (len(hidden), 2)
        or len(hidden) <= 0
    ):
        raise ValueError("singleton teacher-scoring tensor contract mismatch")
    adjusted_rows = []
    for index in range(len(hidden)):
        row_hidden = hidden[index : index + 1].to(device)
        row_logits = base01[index : index + 1].to(device)
        adjusted = apply_motor_logits(
            row_logits,
            row_hidden,
            motor,
            0,
            1,
            active=motor is not None,
        )
        adjusted_rows.append(adjusted.detach().cpu())
    return torch.cat(adjusted_rows, dim=0).contiguous()


def _derive_top1_tensor_evidence(
    adjusted, targets, other_max, other_max_token_ids, zero_id, one_id
):
    adjusted = adjusted.detach().cpu().contiguous()
    targets = torch.as_tensor(targets, dtype=torch.long).cpu().contiguous()
    other_max = other_max.detach().cpu().to(adjusted.dtype).contiguous()
    other_max_token_ids = other_max_token_ids.detach().cpu().long().contiguous()
    zero_wins = (adjusted[:, 0] > adjusted[:, 1]) | (
        (adjusted[:, 0] == adjusted[:, 1]) & (int(zero_id) < int(one_id))
    )
    carry_predictions = torch.where(
        zero_wins,
        torch.zeros(len(adjusted), dtype=torch.long),
        torch.ones(len(adjusted), dtype=torch.long),
    )
    best_logits = adjusted[:, 0].clone()
    best_ids = torch.full((len(adjusted),), int(zero_id), dtype=torch.long)
    one_wins = (adjusted[:, 1] > best_logits) | (
        (adjusted[:, 1] == best_logits) & (int(one_id) < best_ids)
    )
    best_logits = torch.where(one_wins, adjusted[:, 1], best_logits)
    best_ids = torch.where(
        one_wins,
        torch.full_like(best_ids, int(one_id)),
        best_ids,
    )
    other_wins = (other_max > best_logits) | (
        (other_max == best_logits) & (other_max_token_ids < best_ids)
    )
    global_predictions = torch.where(other_wins, other_max_token_ids, best_ids)
    target_token_ids = torch.where(targets == 0, int(zero_id), int(one_id))
    return {
        "target_token_ids": target_token_ids,
        "carry_predictions": carry_predictions,
        "global_prediction_token_ids": global_predictions,
        "carry_pair_correct": carry_predictions == targets,
        "global_correct": global_predictions == target_token_ids,
    }


def _tensor_metric_summary(arm):
    rows = len(arm["target_token_ids"])
    carry_correct = int(arm["carry_pair_correct"].sum())
    global_correct = int(arm["global_correct"].sum())
    return {
        "carry_pair_correct": carry_correct,
        "carry_pair_accuracy": carry_correct / rows,
        "global_correct": global_correct,
        "global_accuracy": global_correct / rows,
        "rows": rows,
        "prediction_ones": int(arm["carry_predictions"].sum()),
        "site_opportunities": int(arm["site_opportunities"].sum()),
        "motor_fires": int(arm["motor_fires"].sum()),
    }


@torch.no_grad()
def fit_teacher_forced_evidence(
    features,
    rows,
    control_labels,
    treatment,
    shuffled,
    source_feature_payload_sha256,
    device,
    scoring_contract,
):
    """Retain tensorized per-row fit metrics derived from the merged shards."""
    size = len(rows)
    if size <= 0 or len(features["base01"]) != size or len(control_labels) != size:
        raise ValueError("fit teacher evidence row count mismatch")
    identities = [
        _fit_teacher_row_identity(row, index) for index, row in enumerate(rows)
    ]
    true_targets = features["labels"].detach().cpu().long().contiguous()
    control_targets = torch.as_tensor(control_labels, dtype=torch.long).contiguous()
    zero_id = int(features["zero_id"])
    one_id = int(features["one_id"])
    common = {
        "zero_id": zero_id,
        "one_id": one_id,
        "deployment_logit_dtype": features["deployment_logit_dtype"],
        "scoring_contract": scoring_contract,
        "source_feature_payload_sha256": source_feature_payload_sha256,
        "row_identities": identities,
        "row_identity_sha256": stable_json_sha256(identities),
        "hidden": features["hidden"].detach().cpu().float().contiguous(),
        "true_targets": true_targets,
        "control_targets": control_targets,
        "other_max_logits": features["other_max"].detach().cpu().float().contiguous(),
        "other_max_token_ids": features["other_max_token_id"]
        .detach()
        .cpu()
        .long()
        .contiguous(),
    }
    arm_specs = {
        "base": (None, "true", true_targets),
        "treatment": (treatment, "true", true_targets),
        "shuffled_on_true_labels": (shuffled, "true", true_targets),
        "shuffled_on_control_labels": (shuffled, "control", control_targets),
    }
    arms = {}
    for name, (motor, target_source, targets) in arm_specs.items():
        adjusted = _singleton_deployment_adjusted_carry_logits(
            common["hidden"], features["base01"], motor, device
        )
        derived = _derive_top1_tensor_evidence(
            adjusted,
            targets,
            common["other_max_logits"],
            common["other_max_token_ids"],
            zero_id,
            one_id,
        )
        site_opportunities = torch.ones(size, dtype=torch.long)
        motor_fires = torch.full((size,), int(motor is not None), dtype=torch.long)
        arm = {
            "target_source": target_source,
            "adjusted_carry_logits": adjusted.detach().cpu().contiguous(),
            **derived,
            "site_opportunities": site_opportunities,
            "motor_fires": motor_fires,
        }
        arm["summary"] = _tensor_metric_summary(arm)
        arms[name] = arm
    return {**common, "arms": arms}


def canonical_fit_teacher_forced_evidence(
    features,
    rows,
    control_labels,
    treatment,
    shuffled,
    source_feature_payload_sha256,
    device,
):
    _require_canonical_teacher_scoring_runtime(features, (treatment, shuffled), device)
    return fit_teacher_forced_evidence(
        features,
        rows,
        control_labels,
        treatment,
        shuffled,
        source_feature_payload_sha256,
        device,
        CANONICAL_TEACHER_SCORING_CONTRACT,
    )


def _publication_staging_parent(path, staging_parent):
    path = Path(path)
    staging = path.parent if staging_parent is None else Path(staging_parent)
    if not staging.is_dir() or staging.is_symlink():
        raise ValueError("publication staging parent is not a regular directory")
    if path.parent.stat().st_dev != staging.stat().st_dev:
        raise ValueError("publication staging parent is on another filesystem")
    return staging


def _link_staged_artifact(path, temporary, mode):
    path = Path(path)
    temporary = Path(temporary)
    os.chmod(temporary, mode)
    os.link(temporary, path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _atomic_bytes(path, payload, mode=0o444, staging_parent=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError("refusing to overwrite {}".format(path))
    staging = _publication_staging_parent(path, staging_parent)
    fd, temporary = tempfile.mkstemp(prefix=".{}-".format(path.name), dir=staging)
    try:
        with os.fdopen(fd, "wb") as sink:
            sink.write(payload)
            sink.flush()
            os.fsync(sink.fileno())
        _link_staged_artifact(path, temporary, mode)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def atomic_json(path, value, staging_parent=None):
    _atomic_bytes(
        path,
        (json.dumps(value, indent=2, sort_keys=True) + "\n").encode(),
        staging_parent=staging_parent,
    )


def atomic_torch(path, value, staging_parent=None):
    path = Path(path)
    if path.exists():
        raise FileExistsError("refusing to overwrite {}".format(path))
    staging = _publication_staging_parent(path, staging_parent)
    descriptor, temporary = tempfile.mkstemp(
        prefix=".{}-".format(path.name), dir=staging
    )
    os.close(descriptor)
    try:
        torch.save(value, temporary)
        with open(temporary, "rb") as source:
            os.fsync(source.fileno())
        _link_staged_artifact(path, temporary, 0o444)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


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
    path = Path(path)
    parent = path.parent
    if (
        not parent.is_dir()
        or parent.is_symlink()
        or (parent.stat().st_mode & 0o777) != 0o700
        or not path.is_file()
        or path.is_symlink()
        or path.stat().st_nlink != 1
        or {child.name for child in parent.iterdir()} != {path.name}
    ):
        raise ValueError("planned output directory is not a one-file mode-0700 stage")
    os.chmod(path, 0o444)
    directory_fd = os.open(parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    os.chmod(parent, 0o555)
    directory_fd = os.open(parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def require_empty_planned_directory(path):
    parent = Path(path).parent
    if (
        not parent.is_dir()
        or parent.is_symlink()
        or (parent.stat().st_mode & 0o777) != 0o700
        or any(parent.iterdir())
    ):
        raise FileExistsError("planned output directory is not empty mode-0700")


def require_sealed_artifact(path):
    path = Path(path)
    if (
        not path.parent.is_dir()
        or path.parent.is_symlink()
        or not path.is_file()
        or path.is_symlink()
        or path.stat().st_nlink != 1
        or (path.stat().st_mode & 0o777) != 0o444
        or (path.parent.stat().st_mode & 0o777) != 0o555
        or {child.name for child in path.parent.iterdir()} != {path.name}
    ):
        raise ValueError("planned artifact is not sealed")


def require_recoverable_artifact(path):
    path = Path(path)
    if (
        not path.parent.is_dir()
        or path.parent.is_symlink()
        or not path.is_file()
        or path.is_symlink()
        or path.stat().st_nlink != 1
        or (path.stat().st_mode & 0o777) != 0o444
        or (path.parent.stat().st_mode & 0o777) != 0o700
        or {child.name for child in path.parent.iterdir()} != {path.name}
    ):
        raise ValueError("planned artifact is not a recoverable one-file publication")


def validate_existing_artifact_stage(path):
    """Return True only for an unsealed, crash-recoverable publication."""
    mode = Path(path).parent.stat().st_mode & 0o777
    if mode == 0o555:
        require_sealed_artifact(path)
        return False
    if mode == 0o700:
        require_recoverable_artifact(path)
        return True
    raise ValueError("planned artifact directory has an invalid lifecycle mode")


def _require_lstat_directory(path, mode, label):
    path = Path(path)
    try:
        observed = os.lstat(path)
    except FileNotFoundError as exc:
        raise ValueError("{} is missing".format(label)) from exc
    if (
        not stat_module.S_ISDIR(observed.st_mode)
        or stat_module.S_IMODE(observed.st_mode) != mode
    ):
        raise ValueError("{} is not exact mode-{:04o} directory".format(label, mode))
    return observed


def _require_lstat_file(path, mode, label):
    path = Path(path)
    try:
        observed = os.lstat(path)
    except FileNotFoundError as exc:
        raise ValueError("{} is missing".format(label)) from exc
    if (
        not stat_module.S_ISREG(observed.st_mode)
        or stat_module.S_IMODE(observed.st_mode) != mode
        or observed.st_nlink != 1
    ):
        raise ValueError(
            "{} is not a regular non-symlink mode-{:04o} one-link file".format(
                label, mode
            )
        )
    return observed


def canonical_confirmation_commitment_path(source_commit):
    if not re.fullmatch(r"[0-9a-f]{40}", str(source_commit)):
        raise ValueError(
            "canonical confirmation commitment requires a lowercase 40-hex commit"
        )
    return (
        CANONICAL_CONFIRMATION_PARENT
        / "commitment_{}".format(source_commit)
        / "commitment.json"
    )


def validate_canonical_confirmation_args(args):
    path = getattr(args, "confirmation_commitment", "")
    receipt = getattr(args, "confirmation_commitment_sha256", "")
    if not path or not re.fullmatch(r"[0-9a-f]{64}", str(receipt)):
        raise SystemExit(
            "canonical execution requires the sealed pre-fit confirmation commitment"
        )
    try:
        expected = canonical_confirmation_commitment_path(args.source_commit)
    except (AttributeError, ValueError) as exc:
        raise SystemExit(
            "canonical confirmation commitment identity is invalid"
        ) from exc
    raw_path = os.fspath(path)
    observed = Path(path)
    if not observed.is_absolute() or raw_path != str(expected) or observed != expected:
        raise SystemExit(
            "canonical confirmation commitment path is not the commit-bound path"
        )


def require_canonical_confirmation_commitment(path, source_commit):
    expected = canonical_confirmation_commitment_path(source_commit)
    raw_path = os.fspath(path)
    path = Path(path)
    if raw_path != str(expected) or path != expected:
        raise ValueError("confirmation commitment path differs from canonical custody")
    root = expected.parent
    _require_lstat_directory(root, 0o555, "confirmation commitment root")
    if {child.name for child in root.iterdir()} != {expected.name}:
        raise ValueError("confirmation commitment root is not one-file closed-world")
    _require_lstat_file(expected, 0o444, "confirmation commitment")
    return expected


def validate_confirmation_commitment_document(
    document, observed_sources, source_contract, episodes_text, cycle_text
):
    _require_exact_mapping(
        document,
        {
            "audit",
            "canonical",
            "source_contract",
            "generator_source_contract",
            "exclusion_contract",
            "secret_sha256",
            "timing",
            "claim_boundary",
        },
        "confirmation commitment",
    )
    _reject_nonfinite_tree(document, "confirmation commitment")
    expected_generator = confirmation_generator_source_contract(observed_sources)
    expected_exclusion = confirmation_exclusion_contract(episodes_text, cycle_text)
    if (
        document["audit"] != CANONICAL_CONFIRMATION_COMMITMENT_AUDIT
        or document["canonical"] is not True
        or document["source_contract"] != source_contract
        or document["generator_source_contract"] != expected_generator
        or document["exclusion_contract"] != expected_exclusion
        or expected_exclusion["episodes_sha256"] != observed_sources.get("episodes")
        or expected_exclusion["cycle_sha256"] != observed_sources.get("cycle")
        or not re.fullmatch(r"[0-9a-f]{64}", str(document["secret_sha256"]))
        or document["timing"] != CANONICAL_CONFIRMATION_TIMING
        or document["claim_boundary"] != CANONICAL_CONFIRMATION_CLAIM_BOUNDARY
    ):
        raise ValueError("confirmation commitment content mismatch")
    return document


def bind_confirmation_commitment(args, bound, frozen, source_contract):
    """Bind and validate the secret digest before any planned artifact is consumed."""
    validate_canonical_confirmation_args(args)
    path = require_canonical_confirmation_commitment(
        args.confirmation_commitment, args.source_commit
    )
    item = BoundInput(path)
    try:
        if item.path != path:
            raise ValueError("confirmation commitment path changed before binding")
        validate_artifact_receipt(item, args.confirmation_commitment_sha256)
        document = _load_exact_json(item.text(), "confirmation commitment")
        validate_confirmation_commitment_document(
            document,
            frozen,
            source_contract,
            bound["episodes"].text(),
            bound["cycle"].text(),
        )
        item.verify_path()
    except Exception:
        item.close()
        raise
    bound["confirmation_commitment"] = item
    frozen["confirmation_commitment"] = item.sha256
    return document


def bind_revealed_confirmation_secret(path):
    """Bind the exact 32 raw secret bytes without placing them in an artifact."""
    raw_path = os.fspath(path)
    path = Path(path)
    if not path.is_absolute() or raw_path != str(path):
        raise ValueError("revealed confirmation secret path must be exact and absolute")
    _require_lstat_file(path, 0o400, "revealed confirmation secret")
    item = BoundInput(path)
    try:
        if item.path != path or len(item.bytes()) != 32:
            raise ValueError("revealed confirmation secret must contain 32 raw bytes")
        item.verify_path()
        return item
    except Exception:
        item.close()
        raise


def _planned_directory_state(directory, artifact_name, label):
    directory = Path(directory)
    observed = os.lstat(directory)
    if not stat_module.S_ISDIR(observed.st_mode):
        raise ValueError("{} is not a regular directory".format(label))
    mode = stat_module.S_IMODE(observed.st_mode)
    if mode not in (0o700, 0o555):
        raise ValueError("{} has an invalid lifecycle mode".format(label))
    children = {child.name for child in directory.iterdir()}
    if not children:
        if mode != 0o700:
            raise ValueError("{} sealed directory is empty".format(label))
        return "empty"
    if children != {artifact_name}:
        raise ValueError("{} is not a closed-world one-file directory".format(label))
    _require_lstat_file(directory / artifact_name, 0o444, "{} artifact".format(label))
    return "sealed" if mode == 0o555 else "recoverable"


def validate_canonical_plan_layout(plan_path, source_commit):
    """Validate the canonical plan root and all legal publication states."""
    expected_root = canonical_plan_root(source_commit)
    expected_plan = expected_root / "plan.json"
    raw_plan_path = os.fspath(plan_path)
    plan_path = Path(plan_path)
    if (
        not plan_path.is_absolute()
        or raw_plan_path != str(expected_plan)
        or plan_path != expected_plan
    ):
        raise ValueError("canonical plan path is not the exact commit-bound path")
    _require_lstat_directory(expected_root, 0o555, "canonical plan root")
    _require_lstat_file(expected_plan, 0o444, "canonical plan.json")
    expected_children = {
        "plan.json",
        "fit",
        "development_eval",
        "confirmation_eval",
        *("shard_{:02d}".format(index) for index in range(CANONICAL_FEATURE_SHARDS)),
    }
    if {child.name for child in expected_root.iterdir()} != expected_children:
        raise ValueError("canonical plan root children are not closed-world")
    states = {}
    for index in range(CANONICAL_FEATURE_SHARDS):
        name = "shard_{:02d}".format(index)
        states[name] = _planned_directory_state(
            expected_root / name, "features.pt", "canonical {}".format(name)
        )
    states["fit"] = _planned_directory_state(
        expected_root / "fit", "motor.pt", "canonical fit"
    )
    states["development_eval"] = _planned_directory_state(
        expected_root / "development_eval",
        "evaluation.json",
        "canonical development evaluation",
    )
    states["confirmation_eval"] = _planned_directory_state(
        expected_root / "confirmation_eval",
        "evaluation.json",
        "canonical confirmation evaluation",
    )
    shard_states = [
        states["shard_{:02d}".format(index)]
        for index in range(CANONICAL_FEATURE_SHARDS)
    ]
    if states["fit"] != "empty" and any(state != "sealed" for state in shard_states):
        raise ValueError("canonical fit lifecycle precedes sealed shards")
    if states["development_eval"] != "empty" and states["fit"] != "sealed":
        raise ValueError("canonical evaluation lifecycle precedes sealed fit")
    if (
        states["confirmation_eval"] != "empty"
        and states["development_eval"] != "sealed"
    ):
        raise ValueError(
            "canonical confirmation lifecycle precedes sealed development evaluation"
        )
    return states


def require_canonical_shard_input(path, plan_root, shard_index):
    """Require one exact sealed planned shard before creating BoundInput."""
    plan_root = Path(plan_root)
    expected = plan_root / "shard_{:02d}".format(int(shard_index)) / "features.pt"
    raw_path = os.fspath(path)
    path = Path(path)
    if raw_path != str(expected) or path != expected:
        raise ValueError("canonical shard path differs from frozen plan")
    _require_lstat_directory(plan_root, 0o555, "canonical plan root")
    _require_lstat_directory(path.parent, 0o555, "canonical shard directory")
    if {child.name for child in path.parent.iterdir()} != {path.name}:
        raise ValueError("canonical shard directory is not one-file closed-world")
    _require_lstat_file(path, 0o444, "canonical shard input")
    return path


def _bind_and_load_canonical_shards(descriptors, plan_root):
    """Preflight every shard before a second checked bind/load pass."""
    if (
        not isinstance(descriptors, list)
        or len(descriptors) != CANONICAL_FEATURE_SHARDS
    ):
        raise ValueError("canonical feature-shard descriptor set is incomplete")
    planned_paths = []
    for shard_index, descriptor in enumerate(descriptors):
        if not isinstance(descriptor, dict) or "artifact" not in descriptor:
            raise ValueError("canonical feature-shard descriptor is invalid")
        planned_paths.append(
            require_canonical_shard_input(
                descriptor["artifact"], plan_root, shard_index
            )
        )

    shard_bounds = []
    payloads = []
    try:
        for shard_index, planned_path in enumerate(planned_paths):
            path = require_canonical_shard_input(
                descriptors[shard_index]["artifact"], plan_root, shard_index
            )
            if path != planned_path:
                raise ValueError(
                    "canonical shard path changed after complete preflight"
                )
            shard_bound = BoundInput(path)
            shard_bounds.append(shard_bound)
            if shard_bound.path != path:
                raise ValueError("canonical shard path changed before binding")
            shard_bound.handle.seek(0)
            payload = torch.load(shard_bound.handle, map_location="cpu")
            shard_bound.handle.seek(0)
            payloads.append((shard_bound.sha256, str(shard_bound.path), payload))
        return shard_bounds, payloads
    except Exception:
        for shard_bound in shard_bounds:
            shard_bound.close()
        raise


@torch.no_grad()
def motor_generate(
    model, motor, tokenizer, prompt, device, max_new=96, retain_full_logits=False
):
    """Greedy cached generation with one grammar-only carry-motor site."""
    cap = int(model.cfg.seq_len)
    max_new = int(max_new)
    if cap <= 0 or max_new <= 0:
        raise ValueError("generation limits must be positive")
    prompt_ids = tokenizer.encode(prompt).ids[-cap:]
    eos_id = tokenizer.token_to_id("<|endoftext|>")
    if type(eos_id) is not int:
        raise ValueError("frozen tokenizer is missing the end-of-text token")
    zero_id = token_for_digit(tokenizer, "x=", "0")[1]
    one_id = token_for_digit(tokenizer, "x=", "1")[1]
    generated = []
    boundaries = []
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
        stop_reason = None
        for index in range(max_new):
            response = decode_tokens(tokenizer, generated, skip_special_tokens=False)
            prior_sites = router.site_count
            prior_fires = router.fire_count
            active = router.observe(response)
            selected = apply_motor_logits(
                logits[:, -1, :], captured["hidden"], motor, zero_id, one_id, active
            )
            next_id = int(selected.argmax(dim=-1).item())
            boundaries.append(
                {
                    "index": index,
                    "prefix_token_count": len(generated),
                    "decoded_prefix": response,
                    "next_token_id": next_id,
                    "router_site": router.site_count == prior_sites + 1,
                    "motor_fired": router.fire_count == prior_fires + 1,
                    "full_logits": (
                        full_logit_tensor_identity(selected)
                        if retain_full_logits
                        else None
                    ),
                }
            )
            generated.append(next_id)
            text = decode_tokens(tokenizer, generated, skip_special_tokens=False)
            if next_id == eos_id:
                stop_reason = "eos"
                break
            if position >= cap:
                stop_reason = "sequence_cap"
                break
            if has_complete_final_answer(text):
                stop_reason = "complete_final_answer"
                break
            with _autocast(device):
                logits, cache = model(
                    torch.tensor([[next_id]], dtype=torch.long, device=device),
                    cache=cache,
                    pos=position,
                    return_cache=True,
                )
            position += 1
        if stop_reason is None:
            stop_reason = "max_new"
    finally:
        handle.remove()
    response_ids = generated[:-1] if generated[-1] == eos_id else generated
    response = decode_tokens(tokenizer, response_ids, skip_special_tokens=False)
    generation = {
        "prompt_token_count": len(prompt_ids),
        "prompt_token_ids_sha256": stable_json_sha256(prompt_ids),
        "sequence_cap": cap,
        "max_new": max_new,
        "eos_token_id": eos_id,
        "token_ids": generated,
        "token_ids_sha256": stable_json_sha256(generated),
        "boundaries": boundaries,
        "boundaries_sha256": stable_json_sha256(boundaries),
        "stop_reason": stop_reason,
    }
    return (
        response,
        router.site_count,
        router.fire_count,
        generation,
    )


def _development_selection_specs():
    source_specs = {}
    for regime, sources in CANONICAL_DEVELOPMENT_REGIME_SPECS:
        for source_split, quota in sources:
            if source_split in source_specs:
                raise RuntimeError("duplicate development source split")
            source_specs[source_split] = (regime, int(quota))
    return source_specs


def validate_development_selection_contract(contract, episodes=None):
    """Validate the one frozen ordered 300-episode development selection."""
    _require_exact_mapping(
        contract,
        {
            "audit",
            "algorithm",
            "episodes_sha256",
            "episode_count",
            "per_regime",
            "regime_counts",
            "source_split_counts",
            "identity_sha256",
            "identities",
        },
        "development selection",
    )
    expected_regime_counts = {
        regime: CANONICAL_PER_REGIME
        for regime, _sources in CANONICAL_DEVELOPMENT_REGIME_SPECS
    }
    source_specs = _development_selection_specs()
    expected_source_counts = {
        source_split: quota for source_split, (_regime, quota) in source_specs.items()
    }
    if (
        contract["audit"] != CANONICAL_DEVELOPMENT_SELECTION_AUDIT
        or contract["algorithm"] != CANONICAL_DEVELOPMENT_SELECTION_ALGORITHM
        or contract["episodes_sha256"] != EXPECTED_EPISODES_SHA256
        or contract["episode_count"] != CANONICAL_DEVELOPMENT_EPISODES
        or contract["per_regime"] != CANONICAL_PER_REGIME
        or contract["regime_counts"] != expected_regime_counts
        or contract["source_split_counts"] != expected_source_counts
        or contract["identity_sha256"] != EXPECTED_DEVELOPMENT_SELECTION_SHA256
    ):
        raise ValueError("development selection contract is not canonical")
    identities = contract["identities"]
    if not isinstance(identities, list) or len(identities) != contract["episode_count"]:
        raise ValueError("development selection identity list is incomplete")
    if stable_json_sha256(identities) != contract["identity_sha256"]:
        raise ValueError("development selection identity hash mismatch")
    regime_counts = collections.Counter()
    source_counts = collections.Counter()
    seen_ids = set()
    previous_source_index = -1
    for index, identity in enumerate(identities):
        label = "development selection identities[{}]".format(index)
        _require_exact_mapping(
            identity,
            {"index", "source_index", "id", "source_split", "regime"},
            label,
        )
        source_split = identity["source_split"]
        if (
            type(identity["index"]) is not int
            or identity["index"] != index
            or type(identity["source_index"]) is not int
            or identity["source_index"] <= previous_source_index
            or not isinstance(identity["id"], str)
            or not identity["id"]
            or identity["id"] in seen_ids
            or source_split not in source_specs
            or identity["regime"] != source_specs[source_split][0]
        ):
            raise ValueError("{} is invalid".format(label))
        previous_source_index = identity["source_index"]
        seen_ids.add(identity["id"])
        regime_counts[identity["regime"]] += 1
        source_counts[source_split] += 1
    if (
        dict(regime_counts) != expected_regime_counts
        or dict(source_counts) != expected_source_counts
    ):
        raise ValueError("development selection identity counts mismatch")
    if episodes is not None:
        if not isinstance(episodes, list) or len(episodes) != len(identities):
            raise ValueError("development selected episode list is incomplete")
        for index, (identity, episode) in enumerate(zip(identities, episodes)):
            if (
                not isinstance(episode, dict)
                or episode.get("id") != identity["id"]
                or episode.get("split") != identity["source_split"]
            ):
                raise ValueError(
                    "development selected episode {} differs from identity".format(
                        index
                    )
                )


def canonical_development_selection(episodes_text):
    """Return the only canonical development episodes and their full identity."""
    text = str(episodes_text)
    if hashlib.sha256(text.encode()).hexdigest() != EXPECTED_EPISODES_SHA256:
        raise ValueError("development selection requires the frozen episode bytes")
    source_specs = _development_selection_specs()
    source_counts = collections.Counter()
    selected = []
    identities = []
    for source_index, raw in enumerate(text.splitlines()):
        if not raw.strip():
            continue
        episode = json.loads(raw)
        source_split = episode.get("split") if isinstance(episode, dict) else None
        if source_split not in source_specs:
            continue
        regime, quota = source_specs[source_split]
        if source_counts[source_split] >= quota:
            continue
        identity = {
            "index": len(identities),
            "source_index": source_index,
            "id": episode.get("id"),
            "source_split": source_split,
            "regime": regime,
        }
        identities.append(identity)
        selected.append({"identity": identity, "episode": episode})
        source_counts[source_split] += 1
    regime_counts = collections.Counter(
        entry["identity"]["regime"] for entry in selected
    )
    contract = {
        "audit": CANONICAL_DEVELOPMENT_SELECTION_AUDIT,
        "algorithm": CANONICAL_DEVELOPMENT_SELECTION_ALGORITHM,
        "episodes_sha256": EXPECTED_EPISODES_SHA256,
        "episode_count": len(selected),
        "per_regime": CANONICAL_PER_REGIME,
        "regime_counts": dict(regime_counts),
        "source_split_counts": dict(source_counts),
        "identity_sha256": stable_json_sha256(identities),
        "identities": identities,
    }
    episodes = [entry["episode"] for entry in selected]
    validate_development_selection_contract(contract, episodes)
    return selected, contract


def development_feature_rows(selection, tokenizer):
    rows = []
    for selected in selection:
        outer = selected["episode"]
        identity = selected["identity"]
        regime = identity["regime"]
        for branch_name, branch in (
            ("factual", outer),
            ("counterfactual", outer["counterfactual"]),
        ):
            for transition, (state, next_state) in enumerate(_episode_states(branch)):
                prompt = microstep_prompt(state, style=branch["prompt_style"])
                prefix, target = field_prefix(prompt, next_state, "carry")
                prompt_ids = tokenizer.encode(prompt).ids
                prefix_ids, target_id = token_for_digit(tokenizer, prefix, target)
                if prefix_ids[: len(prompt_ids)] != prompt_ids:
                    raise RuntimeError("development prefix boundary mismatch")
                rows.append(
                    {
                        "index": len(rows),
                        "selection_index": identity["index"],
                        "source_index": identity["source_index"],
                        "selected_episode_id": identity["id"],
                        "source_split": identity["source_split"],
                        "branch": branch_name,
                        "operation": state["op"],
                        "width": int(state["w"]),
                        "position": int(state["p"]),
                        "style": branch["prompt_style"],
                        "current_carry": int(state["c"]),
                        "target": int(target),
                        "target_id": int(target_id),
                        "prompt_ids": prompt_ids,
                        "prefix_ids": prefix_ids,
                        "prompt": prompt,
                        "response_prefix": decode_tokens(
                            tokenizer,
                            prefix_ids[len(prompt_ids) :],
                            skip_special_tokens=False,
                        ),
                        "episode_id": branch["id"],
                        "regime": regime,
                        "transition": transition,
                    }
                )
    return rows


def _confirmation_episode_identity(row):
    return {
        "index": int(row["index"]),
        "source_index": int(row["index"]),
        "id": row["id"],
        "source_split": "confirmation",
        "regime": row["regime"],
    }


def confirmation_feature_rows(board, tokenizer):
    """Build the one teacher-forced carry row for every secret-derived episode."""
    rows = []
    for board_row in board["rows"]:
        state = dws_prompt_state(board_row["prompt"])
        expected = parse_state(board_row["expected_state"])
        if state is None or expected is None or apply_microstep(state) != expected:
            raise ValueError("confirmation row is not an exact frozen transition")
        prefix, target = field_prefix(board_row["prompt"], expected, "carry")
        prompt_ids = tokenizer.encode(board_row["prompt"]).ids
        prefix_ids, target_id = token_for_digit(tokenizer, prefix, target)
        if prefix_ids[: len(prompt_ids)] != prompt_ids:
            raise ValueError("confirmation prefix boundary mismatch")
        if int(target) != board_row["target_carry"]:
            raise ValueError("confirmation target differs from secret-derived row")
        rows.append(
            {
                "index": board_row["index"],
                "selection_index": board_row["index"],
                "source_index": board_row["index"],
                "selected_episode_id": board_row["id"],
                "source_split": "confirmation",
                "branch": "confirmation",
                "operation": board_row["operation"],
                "width": board_row["width"],
                "position": board_row["position"],
                "style": board_row["style"],
                "current_carry": int(state["c"]),
                "target": int(target),
                "target_id": int(target_id),
                "prompt_ids": prompt_ids,
                "prefix_ids": prefix_ids,
                "prompt": board_row["prompt"],
                "response_prefix": decode_tokens(
                    tokenizer,
                    prefix_ids[len(prompt_ids) :],
                    skip_special_tokens=False,
                ),
                "episode_id": board_row["id"],
                "regime": board_row["regime"],
                "transition": board_row["selected_transition"],
            }
        )
    if len(rows) != 256:
        raise ValueError("confirmation feature board must contain exactly 256 rows")
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
        "claim_boundary": LINEAR_DIAGNOSTIC_CLAIM_BOUNDARY,
    }


def cycle_validation_contract(cycle_text, episode_map):
    """Derive immutable identities and expected calls for all 50 cycle cases."""
    document = _load_exact_json(cycle_text, "frozen cycle artifact")
    records = document.get("records")
    if not isinstance(records, list) or len(records) != CANONICAL_CYCLE_CASES:
        raise ValueError("frozen cycle artifact must contain exactly 50 records")
    cases = []
    seen_episode_ids = set()
    for index, record in enumerate(records):
        if not isinstance(record, dict) or not {
            "episode_id",
            "counterfactual_state",
            "counterfactual_next",
        }.issubset(record):
            raise ValueError("frozen cycle record schema is invalid")
        episode_id = record["episode_id"]
        if episode_id not in episode_map or episode_id in seen_episode_ids:
            raise ValueError("frozen cycle record references an unknown episode")
        seen_episode_ids.add(episode_id)
        state = parse_state(record["counterfactual_state"])
        expected_first = parse_state(record["counterfactual_next"])
        if (
            state is None
            or expected_first is None
            or apply_microstep(state) != expected_first
        ):
            raise ValueError("invalid transition in frozen cycle artifact")
        episode = episode_map[episode_id]
        prompt = microstep_prompt(state, style=episode["prompt_style"])
        if expected_first["z"]:
            raise ValueError("frozen cycle case cannot terminate on its first call")
        expected_second = apply_microstep(expected_first)
        cases.append(
            {
                "index": index,
                "episode_id": episode_id,
                "first_prompt": prompt,
                "expected_first": expected_first,
                "second_prompt": microstep_prompt(
                    expected_first, style=episode["prompt_style"]
                ),
                "expected_second": expected_second,
            }
        )
    return {"records": len(records), "cases": cases}


def _raw_call_record(index, kind, prompt, response, sites, fires, generation):
    return {
        "index": int(index),
        "kind": kind,
        "prompt": prompt,
        "response": response,
        "site_opportunities": int(sites),
        "motor_fires": int(fires),
        "generation": generation,
    }


def evaluate_cycle(cycle_text, episode_map, model, motor, tokenizer, device, max_new):
    contract = cycle_validation_contract(cycle_text, episode_map)
    first_exact = second_exact = integrated = site_count = fire_count = 0
    cases = []
    for expected in contract["cases"]:
        calls = []
        response, sites, fires, generation = motor_generate(
            model,
            motor,
            tokenizer,
            expected["first_prompt"],
            device,
            max_new=max_new,
        )
        calls.append(
            _raw_call_record(
                0,
                "first",
                expected["first_prompt"],
                response,
                sites,
                fires,
                generation,
            )
        )
        predicted_first = parse_state(response)
        first_ok = predicted_first == expected["expected_first"]
        first_exact += int(first_ok)
        site_count += sites
        fire_count += fires
        second_ok = False
        if first_ok:
            response_second, sites, fires, generation = motor_generate(
                model,
                motor,
                tokenizer,
                expected["second_prompt"],
                device,
                max_new=max_new,
            )
            calls.append(
                _raw_call_record(
                    1,
                    "second",
                    expected["second_prompt"],
                    response_second,
                    sites,
                    fires,
                    generation,
                )
            )
            second_ok = parse_state(response_second) == expected["expected_second"]
            second_exact += int(second_ok)
            site_count += sites
            fire_count += fires
        integrated += int(first_ok and second_ok)
        cases.append(
            {
                "index": expected["index"],
                "episode_id": expected["episode_id"],
                "first_exact": first_ok,
                "second_exact": second_ok,
                "integrated_two_call_exact": first_ok and second_ok,
                "site_opportunities": sum(call["site_opportunities"] for call in calls),
                "motor_fires": sum(call["motor_fires"] for call in calls),
                "calls": calls,
            }
        )
    return {
        "records": contract["records"],
        "first_exact": first_exact,
        "second_exact_after_first": second_exact,
        "integrated_two_call_exact": integrated,
        "site_opportunities": site_count,
        "motor_fires": fire_count,
        "cases": cases,
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
            if key not in {"motor_fires", "motor_fired", "boundaries_sha256"}
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


def validate_canonical_extract_args(args, canonical):
    if not canonical:
        return
    validate_canonical_confirmation_args(args)
    if (
        args.quota != FIT_QUOTA
        or args.extract_batch != CANONICAL_EXTRACT_BATCH
        or args.shard_count != CANONICAL_FEATURE_SHARDS
        or not 0 <= args.shard_index < args.shard_count
    ):
        raise SystemExit("canonical feature extraction contract is immutable")
    _validate_canonical_plan_reference(args)


def validate_canonical_feature_fit_args(args, canonical):
    if canonical:
        validate_canonical_confirmation_args(args)
    validate_canonical_train_args(args, canonical)
    if canonical:
        _validate_canonical_plan_reference(args)


def validate_canonical_eval_args(args, canonical):
    if not canonical:
        return
    validate_canonical_confirmation_args(args)
    if (
        args.max_new != CANONICAL_MAX_NEW
        or args.extract_batch != CANONICAL_EXTRACT_BATCH
    ):
        raise SystemExit("canonical evaluation board and decode budget are immutable")
    _validate_canonical_plan_reference(args)


def canonical_plan_root(source_commit):
    if not re.fullmatch(r"[0-9a-f]{40}", str(source_commit)):
        raise ValueError("canonical plan root requires a lowercase 40-hex commit")
    return CANONICAL_PLAN_PARENT / "canonical_{}".format(source_commit)


def _validate_canonical_plan_reference(args):
    plan = getattr(args, "plan", "")
    receipt = getattr(args, "plan_sha256", "")
    if not plan or not re.fullmatch(r"[0-9a-f]{64}", receipt):
        raise SystemExit("canonical execution requires a plan and lowercase SHA-256")
    expected = canonical_plan_root(args.source_commit) / "plan.json"
    if (
        not Path(plan).is_absolute()
        or str(plan) != str(expected)
        or Path(plan) != expected
    ):
        raise SystemExit("canonical execution plan path is not the sole committed root")


def require_canonical_cuda_runtime():
    if (
        not torch.cuda.is_available()
        or torch.cuda.device_count() != 1
        or torch.cuda.get_device_name(0) != CANONICAL_DEVICE_NAME
        or not torch.version.cuda
    ):
        raise RuntimeError("canonical execution requires exactly one NVIDIA H100 PCIe")
    probe = torch.empty(1, device="cuda", dtype=torch.bfloat16)
    probe.add_(1)
    torch.cuda.synchronize()
    return "cuda"


def _validate_motor_state(state, d_model, rank, arm):
    expected_shapes = {
        "down.weight": (int(rank), int(d_model)),
        "down.bias": (int(rank),),
        "up.weight": (2, int(rank)),
        "up.bias": (2,),
    }
    if not isinstance(state, dict) or set(state) != set(expected_shapes):
        raise ValueError("{} state schema mismatch".format(arm))
    for name, shape in expected_shapes.items():
        tensor = state[name]
        if (
            not isinstance(tensor, torch.Tensor)
            or tensor.dtype != torch.float32
            or tuple(tensor.shape) != shape
            or not bool(torch.isfinite(tensor).all())
        ):
            raise ValueError("{} state tensor mismatch: {}".format(arm, name))


def _validate_fit_report(report, fit_budget, arm):
    expected_keys = {
        "updates",
        "batch_size",
        "lr",
        "weight_decay",
        "schedule_sha256",
        "first_loss",
        "final_loss",
        "min_loss",
    }
    if not isinstance(report, dict) or set(report) != expected_keys:
        raise ValueError("{} fit report schema mismatch".format(arm))
    for name in ("updates", "batch_size", "lr", "weight_decay"):
        if report[name] != fit_budget[name]:
            raise ValueError("{} fit budget mismatch: {}".format(arm, name))
    if report["schedule_sha256"] != fit_budget["schedule_sha256"]:
        raise ValueError("{} batch schedule mismatch".format(arm))
    for name in ("first_loss", "final_loss", "min_loss"):
        if not isinstance(report[name], (int, float)) or not math.isfinite(
            report[name]
        ):
            raise ValueError("{} fit loss is invalid: {}".format(arm, name))


def _validate_linear_diagnostic(report, expected_rows):
    expected_keys = {
        "train_rows",
        "test_rows",
        "test_correct",
        "test_accuracy",
        "schedule_sha256",
        "claim_boundary",
    }
    if not isinstance(report, dict) or set(report) != expected_keys:
        raise ValueError("linear diagnostic schema mismatch")
    train_rows = int(expected_rows * 0.8)
    test_rows = expected_rows - train_rows
    if (
        type(report["train_rows"]) is not int
        or type(report["test_rows"]) is not int
        or type(report["test_correct"]) is not int
        or report["train_rows"] != train_rows
        or report["test_rows"] != test_rows
        or not 0 <= report["test_correct"] <= test_rows
    ):
        raise ValueError("linear diagnostic counts are invalid")
    accuracy = report["test_accuracy"]
    if (
        not isinstance(accuracy, (int, float))
        or not math.isfinite(accuracy)
        or not math.isclose(
            float(accuracy), report["test_correct"] / test_rows, abs_tol=1e-12
        )
    ):
        raise ValueError("linear diagnostic accuracy is invalid")
    _, expected_schedule_sha256 = _batch_schedule(
        train_rows, min(512, train_rows), 300, FIT_SEED + 2
    )
    if report["schedule_sha256"] != expected_schedule_sha256:
        raise ValueError("linear diagnostic schedule hash is invalid")
    if report["claim_boundary"] != LINEAR_DIAGNOSTIC_CLAIM_BOUNDARY:
        raise ValueError("linear diagnostic claim boundary is invalid")


def _validate_feature_metrics_report(report, expected_rows):
    expected_keys = {
        "carry_pair_correct",
        "carry_pair_accuracy",
        "global_correct",
        "global_accuracy",
        "rows",
        "prediction_ones",
    }
    if not isinstance(report, dict) or set(report) != expected_keys:
        raise ValueError("feature metric schema mismatch")
    if type(report["rows"]) is not int or report["rows"] != expected_rows:
        raise ValueError("feature metric row count mismatch")
    for name in ("carry_pair_correct", "global_correct", "prediction_ones"):
        if type(report[name]) is not int or not 0 <= report[name] <= expected_rows:
            raise ValueError("feature metric count is invalid: {}".format(name))
    if report["global_correct"] > report["carry_pair_correct"]:
        raise ValueError("global feature correctness exceeds carry-pair correctness")
    for count_name, accuracy_name in (
        ("carry_pair_correct", "carry_pair_accuracy"),
        ("global_correct", "global_accuracy"),
    ):
        accuracy = report[accuracy_name]
        if (
            type(accuracy) not in (int, float)
            or not math.isfinite(accuracy)
            or not math.isclose(
                float(accuracy), report[count_name] / expected_rows, abs_tol=1e-7
            )
        ):
            raise ValueError(
                "feature metric accuracy is invalid: {}".format(accuracy_name)
            )


def _validate_teacher_forced_report(report, expected_rows, arm, tokenizer):
    _require_exact_mapping(
        report,
        {
            "zero_id",
            "one_id",
            "deployment_logit_dtype",
            "scoring_contract",
            "summary",
            "rows",
        },
        "teacher-forced arm",
    )
    zero_id = token_for_digit(tokenizer, "x=", "0")[1]
    one_id = token_for_digit(tokenizer, "x=", "1")[1]
    if (
        type(report["zero_id"]) is not int
        or type(report["one_id"]) is not int
        or report["zero_id"] != zero_id
        or report["one_id"] != one_id
    ):
        raise ValueError("teacher-forced carry-token identity mismatch")
    if report["deployment_logit_dtype"] != "torch.bfloat16":
        raise ValueError("teacher-forced deployment logit dtype mismatch")
    if report["scoring_contract"] != CANONICAL_TEACHER_SCORING_CONTRACT:
        raise ValueError("teacher-forced singleton H100 scoring contract mismatch")
    rows = report["rows"]
    if not isinstance(rows, list) or len(rows) != len(expected_rows) or not rows:
        raise ValueError("teacher-forced row evidence is incomplete")
    vocab_size = tokenizer.get_vocab_size()
    for index, (row, expected) in enumerate(zip(rows, expected_rows)):
        label = "teacher-forced {} rows[{}]".format(arm, index)
        _require_exact_mapping(
            row,
            {
                "identity",
                "target",
                "target_token_id",
                "adjusted_zero_logit",
                "adjusted_one_logit",
                "other_max_logit",
                "other_max_token_id",
                "carry_prediction",
                "global_prediction_token_id",
                "carry_pair_correct",
                "global_correct",
                "site_opportunities",
                "motor_fires",
            },
            label,
        )
        if row["identity"] != _teacher_row_identity(expected):
            raise ValueError("{} identity differs from frozen row".format(label))
        target = int(expected["target"])
        target_id = zero_id if target == 0 else one_id
        if (
            type(expected["target"]) is not int
            or expected["target"] not in (0, 1)
            or expected["target_id"] != target_id
            or type(row["target"]) is not int
            or row["target"] != target
            or type(row["target_token_id"]) is not int
            or row["target_token_id"] != target_id
        ):
            raise ValueError("{} target differs from frozen row".format(label))
        raw_float_names = (
            "adjusted_zero_logit",
            "adjusted_one_logit",
            "other_max_logit",
        )
        if any(
            type(row[name]) not in (int, float) or not math.isfinite(row[name])
            for name in raw_float_names
        ):
            raise ValueError("{} raw logit evidence is invalid".format(label))
        other_max_id = row["other_max_token_id"]
        if (
            type(other_max_id) is not int
            or not 0 <= other_max_id < vocab_size
            or other_max_id in {zero_id, one_id}
        ):
            raise ValueError("{} other-token maximum identity is invalid".format(label))
        zero_logit = float(row["adjusted_zero_logit"])
        one_logit = float(row["adjusted_one_logit"])
        carry_prediction = (
            0
            if zero_logit > one_logit or (zero_logit == one_logit and zero_id < one_id)
            else 1
        )
        global_prediction = max(
            (
                (zero_logit, -zero_id, zero_id),
                (one_logit, -one_id, one_id),
                (float(row["other_max_logit"]), -other_max_id, other_max_id),
            )
        )[2]
        for name in ("carry_pair_correct", "global_correct"):
            _require_bool(row[name], "{}.{}".format(label, name))
        if (
            type(row["carry_prediction"]) is not int
            or type(row["global_prediction_token_id"]) is not int
            or row["carry_prediction"] != carry_prediction
            or row["global_prediction_token_id"] != global_prediction
            or row["carry_pair_correct"] is not (carry_prediction == target)
            or row["global_correct"] is not (global_prediction == target_id)
        ):
            raise ValueError(
                "{} prediction differs from complete top-one evidence".format(label)
            )

        prompt_ids = tokenizer.encode(expected["prompt"]).ids
        decoded_prefix = decode_tokens(
            tokenizer,
            expected["prefix_ids"][len(prompt_ids) :],
            skip_special_tokens=False,
        )
        if decoded_prefix != expected["response_prefix"]:
            raise ValueError("{} frozen token boundary is invalid".format(label))
        router = CarryRouter(expected["prompt"], arm != "base")
        router.observe(decoded_prefix)
        if router.site_count != 1:
            raise ValueError(
                "{} is not an exact teacher-forced carry site".format(label)
            )
        if (
            row["site_opportunities"] != router.site_count
            or row["motor_fires"] != router.fire_count
        ):
            raise ValueError(
                "{} router accounting differs from frozen row".format(label)
            )
        _validate_arm_site_counts(
            arm, row["site_opportunities"], row["motor_fires"], label
        )

    summary = report["summary"]
    _require_exact_mapping(
        summary,
        {
            "carry_pair_correct",
            "carry_pair_accuracy",
            "global_correct",
            "global_accuracy",
            "rows",
            "prediction_ones",
            "site_opportunities",
            "motor_fires",
        },
        "teacher-forced {} summary".format(arm),
    )
    compact = {
        key: value
        for key, value in summary.items()
        if key not in {"site_opportunities", "motor_fires"}
    }
    _validate_feature_metrics_report(compact, len(expected_rows))
    _validate_arm_site_counts(
        arm,
        summary["site_opportunities"],
        summary["motor_fires"],
        "teacher-forced {} summary".format(arm),
    )
    recomputed = _aggregate_teacher_rows(rows)
    if summary != recomputed:
        raise ValueError("teacher-forced aggregate differs from raw row evidence")


def _validate_feature_merge(merge, plan, plan_sha256):
    expected_keys = {
        "shards",
        "sentinel_indices",
        "sentinel_payload_sha256",
        "runtime",
        "plan_sha256",
        "merged_feature_payload_sha256",
        "teacher_metric_feature_payload_sha256",
    }
    if not isinstance(merge, dict) or set(merge) != expected_keys:
        raise ValueError("carry-motor feature merge schema mismatch")
    if merge["plan_sha256"] != plan_sha256:
        raise ValueError("carry-motor feature merge is not plan-bound")
    if merge["sentinel_indices"] != plan["sentinel_indices"]:
        raise ValueError("carry-motor sentinel merge mismatch")
    for name in (
        "sentinel_payload_sha256",
        "merged_feature_payload_sha256",
        "teacher_metric_feature_payload_sha256",
    ):
        if not re.fullmatch(r"[0-9a-f]{64}", str(merge[name])):
            raise ValueError(
                "carry-motor feature merge hash is invalid: {}".format(name)
            )
    if merge["runtime"] != plan["runtime_contract"]["artifact_runtime"]:
        raise ValueError("carry-motor feature runtime mismatch")
    receipts = merge["shards"]
    if not isinstance(receipts, list) or len(receipts) != CANONICAL_FEATURE_SHARDS:
        raise ValueError("carry-motor feature receipts are incomplete")
    for index, (receipt, descriptor) in enumerate(zip(receipts, plan["shards"])):
        if not isinstance(receipt, dict) or set(receipt) != {
            "shard_index",
            "artifact_sha256",
            "artifact",
            "feature_payload_sha256",
            "rows",
        }:
            raise ValueError("carry-motor feature receipt schema mismatch")
        if (
            receipt["shard_index"] != index
            or receipt["artifact"] != descriptor["artifact"]
            or receipt["rows"] != descriptor["rows"]
        ):
            raise ValueError("carry-motor feature receipt differs from plan")
        for name in ("artifact_sha256", "feature_payload_sha256"):
            if not re.fullmatch(r"[0-9a-f]{64}", str(receipt[name])):
                raise ValueError("carry-motor feature receipt hash is invalid")


def _require_metric_tensor(value, shape, dtype, label, finite=False):
    if (
        not isinstance(value, torch.Tensor)
        or tuple(value.shape) != tuple(shape)
        or value.dtype != dtype
    ):
        raise ValueError("{} tensor contract mismatch".format(label))
    if finite and not bool(torch.isfinite(value.float()).all()):
        raise ValueError("{} contains non-finite values".format(label))


def _validate_fit_teacher_forced_evidence(
    report,
    plan,
    feature_merge,
    replayed_features,
    treatment_state,
    shuffled_state,
    deployment_device,
):
    expected_keys = {
        "zero_id",
        "one_id",
        "deployment_logit_dtype",
        "scoring_contract",
        "source_feature_payload_sha256",
        "row_identities",
        "row_identity_sha256",
        "hidden",
        "true_targets",
        "control_targets",
        "other_max_logits",
        "other_max_token_ids",
        "arms",
    }
    if not isinstance(report, dict) or set(report) != expected_keys:
        raise ValueError("fit teacher evidence schema mismatch")
    zero_id = plan["zero_id"]
    one_id = plan["one_id"]
    size = int(plan["board"]["rows"])
    if report["zero_id"] != zero_id or report["one_id"] != one_id:
        raise ValueError("fit teacher carry-token identity mismatch")
    if report["deployment_logit_dtype"] != plan["runtime_contract"].get(
        "deployment_logit_dtype"
    ):
        raise ValueError("fit teacher deployment dtype mismatch")
    if report["scoring_contract"] != CANONICAL_TEACHER_SCORING_CONTRACT or report[
        "scoring_contract"
    ] != plan["runtime_contract"].get("teacher_scoring_contract"):
        raise ValueError("fit teacher singleton H100 scoring contract mismatch")
    replayed_metric_sha256 = teacher_metric_feature_payload_sha256(replayed_features)
    if report[
        "source_feature_payload_sha256"
    ] != replayed_metric_sha256 or replayed_metric_sha256 != feature_merge.get(
        "teacher_metric_feature_payload_sha256"
    ):
        raise ValueError("fit teacher evidence is not bound to merged shards")
    identities = report["row_identities"]
    identity_keys = {
        "index",
        "operation",
        "width",
        "position",
        "style",
        "current_carry",
        "target",
        "target_id",
        "prompt_sha256",
        "prefix_sha256",
    }
    if not isinstance(identities, list) or len(identities) != size:
        raise ValueError("fit teacher row identities are incomplete")
    for index, identity in enumerate(identities):
        if not isinstance(identity, dict) or set(identity) != identity_keys:
            raise ValueError("fit teacher row identity schema mismatch")
        if (
            type(identity["index"]) is not int
            or identity["index"] != index
            or identity["operation"] not in {"add", "sub"}
            or type(identity["width"]) is not int
            or identity["width"] not in FIT_WIDTHS
            or type(identity["position"]) is not int
            or not 0 <= identity["position"] < identity["width"]
            or identity["style"] not in FIT_STYLES
            or type(identity["current_carry"]) is not int
            or identity["current_carry"] not in (0, 1)
            or type(identity["target"]) is not int
            or identity["target"] not in (0, 1)
            or identity["target_id"] != (zero_id if identity["target"] == 0 else one_id)
            or not re.fullmatch(r"[0-9a-f]{64}", str(identity["prompt_sha256"]))
            or not re.fullmatch(r"[0-9a-f]{64}", str(identity["prefix_sha256"]))
        ):
            raise ValueError("fit teacher row identity is invalid")
    identity_sha256 = stable_json_sha256(identities)
    if (
        report["row_identity_sha256"] != identity_sha256
        or identity_sha256 != plan["fit_budget"]["teacher_row_identity_sha256"]
    ):
        raise ValueError("fit teacher row identity binding mismatch")

    _require_metric_tensor(report["true_targets"], (size,), torch.int64, "true targets")
    _require_metric_tensor(
        report["control_targets"], (size,), torch.int64, "control targets"
    )
    expected_true = torch.tensor(
        [identity["target"] for identity in identities], dtype=torch.long
    )
    if not torch.equal(report["true_targets"], expected_true):
        raise ValueError("fit teacher true targets differ from row identities")
    if not torch.equal(
        report["true_targets"],
        replayed_features["labels"].detach().cpu().long().contiguous(),
    ):
        raise ValueError("fit teacher targets differ from replayed sealed shards")
    if (
        not bool(
            ((report["control_targets"] == 0) | (report["control_targets"] == 1)).all()
        )
        or stable_json_sha256(report["control_targets"].tolist())
        != plan["fit_budget"]["control"]["labels_sha256"]
    ):
        raise ValueError("fit teacher control targets differ from frozen plan")
    _require_metric_tensor(
        report["hidden"],
        (size, int(plan["d_model"])),
        torch.float32,
        "fit teacher hidden features",
        finite=True,
    )
    if not torch.equal(
        report["hidden"],
        replayed_features["hidden"].detach().cpu().float().contiguous(),
    ):
        raise ValueError(
            "fit teacher hidden features differ from replayed sealed shards"
        )
    _require_metric_tensor(
        report["other_max_logits"],
        (size,),
        torch.float32,
        "other-token maxima",
        finite=True,
    )
    _require_metric_tensor(
        report["other_max_token_ids"],
        (size,),
        torch.int64,
        "other-token maximum identities",
    )
    if not torch.equal(
        report["other_max_logits"],
        replayed_features["other_max"].detach().cpu().float().contiguous(),
    ) or not torch.equal(
        report["other_max_token_ids"],
        replayed_features["other_max_token_id"].detach().cpu().long().contiguous(),
    ):
        raise ValueError("fit teacher non-carry evidence differs from replayed shards")
    other_ids = report["other_max_token_ids"]
    if not bool(
        (
            (other_ids >= 0)
            & (other_ids < int(plan["vocab_size"]))
            & (other_ids != zero_id)
            & (other_ids != one_id)
        ).all()
    ):
        raise ValueError("fit teacher other-token maximum identity is invalid")

    expected_arms = {
        "base": ("true", False),
        "treatment": ("true", True),
        "shuffled_on_true_labels": ("true", True),
        "shuffled_on_control_labels": ("control", True),
    }
    arms = report["arms"]
    if not isinstance(arms, dict) or set(arms) != set(expected_arms):
        raise ValueError("fit teacher arm schema mismatch")
    base_logits = arms["base"].get("adjusted_carry_logits")
    _require_metric_tensor(
        base_logits,
        (size, 2),
        torch.bfloat16,
        "fit teacher retained base logits",
        finite=True,
    )
    replayed_base_logits = replayed_features["base01"].detach().cpu().contiguous()
    if not torch.equal(base_logits, replayed_base_logits):
        raise ValueError("fit teacher base logits differ from replayed sealed shards")
    treatment_motor = (
        CarryMotor(int(plan["d_model"]), int(plan["fit_budget"]["rank"]))
        .to(deployment_device)
        .eval()
    )
    treatment_motor.load_state_dict(treatment_state)
    shuffled_motor = (
        CarryMotor(int(plan["d_model"]), int(plan["fit_budget"]["rank"]))
        .to(deployment_device)
        .eval()
    )
    shuffled_motor.load_state_dict(shuffled_state)
    expected_adjusted = {
        "base": _singleton_deployment_adjusted_carry_logits(
            report["hidden"], base_logits, None, deployment_device
        ),
        "treatment": _singleton_deployment_adjusted_carry_logits(
            report["hidden"], base_logits, treatment_motor, deployment_device
        ),
        "shuffled_on_true_labels": _singleton_deployment_adjusted_carry_logits(
            report["hidden"], base_logits, shuffled_motor, deployment_device
        ),
        "shuffled_on_control_labels": _singleton_deployment_adjusted_carry_logits(
            report["hidden"], base_logits, shuffled_motor, deployment_device
        ),
    }
    arm_keys = {
        "target_source",
        "adjusted_carry_logits",
        "target_token_ids",
        "carry_predictions",
        "global_prediction_token_ids",
        "carry_pair_correct",
        "global_correct",
        "site_opportunities",
        "motor_fires",
        "summary",
    }
    for name, (target_source, motor_present) in expected_arms.items():
        arm = arms[name]
        if not isinstance(arm, dict) or set(arm) != arm_keys:
            raise ValueError("fit teacher {} evidence schema mismatch".format(name))
        if arm["target_source"] != target_source:
            raise ValueError("fit teacher {} target source mismatch".format(name))
        _require_metric_tensor(
            arm["adjusted_carry_logits"],
            (size, 2),
            torch.bfloat16,
            "fit teacher {} adjusted logits".format(name),
            finite=True,
        )
        if not torch.equal(arm["adjusted_carry_logits"], expected_adjusted[name]):
            raise ValueError(
                "fit teacher {} logits differ from fitted motor state".format(name)
            )
        expected_targets = report[
            "true_targets" if target_source == "true" else "control_targets"
        ]
        derived = _derive_top1_tensor_evidence(
            arm["adjusted_carry_logits"],
            expected_targets,
            report["other_max_logits"],
            report["other_max_token_ids"],
            zero_id,
            one_id,
        )
        tensor_contracts = {
            "target_token_ids": torch.int64,
            "carry_predictions": torch.int64,
            "global_prediction_token_ids": torch.int64,
            "carry_pair_correct": torch.bool,
            "global_correct": torch.bool,
            "site_opportunities": torch.int64,
            "motor_fires": torch.int64,
        }
        for field, dtype in tensor_contracts.items():
            _require_metric_tensor(
                arm[field], (size,), dtype, "fit teacher {} {}".format(name, field)
            )
        for field, expected in derived.items():
            if not torch.equal(arm[field], expected):
                raise ValueError(
                    "fit teacher {} {} differs from raw logits".format(name, field)
                )
        expected_sites = torch.ones(size, dtype=torch.long)
        expected_fires = torch.full((size,), int(motor_present), dtype=torch.long)
        if not torch.equal(
            arm["site_opportunities"], expected_sites
        ) or not torch.equal(arm["motor_fires"], expected_fires):
            raise ValueError("fit teacher {} router evidence mismatch".format(name))
        compact = {
            key: value
            for key, value in arm["summary"].items()
            if key not in {"site_opportunities", "motor_fires"}
        }
        _validate_feature_metrics_report(compact, size)
        if arm["summary"] != _tensor_metric_summary(arm):
            raise ValueError("fit teacher {} aggregate differs from rows".format(name))
    retained_source_sha256 = feature_payload_sha256(
        {
            "base01": arms["base"]["adjusted_carry_logits"],
            "deployment_logit_dtype": report["deployment_logit_dtype"],
            "hidden": report["hidden"],
            "labels": report["true_targets"],
            "one_id": one_id,
            "other_max": report["other_max_logits"],
            "other_max_token_id": report["other_max_token_ids"],
            "zero_id": zero_id,
        }
    )
    if retained_source_sha256 != report["source_feature_payload_sha256"]:
        raise ValueError("fit teacher raw evidence differs from merged shard payload")


def _validate_motor_bundle_against_replayed_features(
    bundle,
    expected_bindings,
    current_sources,
    source_contract,
    plan_sha256,
    plan,
    replayed_features,
    replayed_feature_merge,
    deployment_device,
):
    if not isinstance(bundle, dict) or set(bundle) != CANONICAL_FIT_KEYS:
        raise ValueError("canonical carry-motor bundle schema mismatch")
    if bundle.get("audit") != CANONICAL_FIT_AUDIT:
        raise ValueError("unsupported carry-motor bundle audit version")
    if bundle.get("canonical") is not True:
        raise ValueError("carry-motor bundle is not canonical")
    if bundle.get("plan_sha256") != plan_sha256:
        raise ValueError("carry-motor plan binding mismatch")
    _validate_feature_merge(bundle["feature_shard_merge"], plan, plan_sha256)
    _validate_feature_merge(replayed_feature_merge, plan, plan_sha256)
    if bundle["feature_shard_merge"] != replayed_feature_merge:
        raise ValueError("fit feature merge differs from replayed sealed shards")
    if replayed_feature_merge[
        "merged_feature_payload_sha256"
    ] != feature_payload_sha256(replayed_features):
        raise ValueError("replayed merged feature payload hash mismatch")
    fit_budget = plan["fit_budget"]
    if bundle.get("initial_state_sha256") != plan["fit_budget"]["initial_state_sha256"]:
        raise ValueError("carry-motor initial-state plan mismatch")
    if (
        bundle.get("deployment_logit_dtype")
        != plan["runtime_contract"]["deployment_logit_dtype"]
    ):
        raise ValueError("invalid carry-motor deployment logit dtype")
    for name, expected in expected_bindings.items():
        if bundle.get(name) != expected:
            raise ValueError("motor bundle input mismatch: {}".format(name))
    if bundle.get("confirmation_commitment_sha256") != plan.get(
        "confirmation_commitment", {}
    ).get("sha256"):
        raise ValueError("motor bundle confirmation commitment mismatch")
    if bundle.get("scientific_source_sha256") != current_sources:
        raise ValueError("motor bundle scientific source mismatch")
    if bundle.get("source_contract") != source_contract:
        raise ValueError("motor bundle source contract mismatch")
    if (
        not _is_canonical_checkpoint_step(plan.get("checkpoint_step"))
        or not _is_canonical_checkpoint_step(bundle.get("checkpoint_step"))
        or bundle.get("checkpoint_step") != plan["checkpoint_step"]
        or bundle.get("d_model") != plan["d_model"]
        or bundle.get("rank") != fit_budget["rank"]
        or bundle.get("parameter_count")
        != (
            plan["d_model"] * fit_budget["rank"]
            + fit_budget["rank"]
            + 2 * fit_budget["rank"]
            + 2
        )
        or bundle.get("extract_batch") != plan["runtime_contract"]["extract_batch"]
        or bundle.get("zero_id") != plan["zero_id"]
        or bundle.get("one_id") != plan["one_id"]
        or bundle.get("board") != plan["board"]
        or bundle.get("control") != fit_budget["control"]
    ):
        raise ValueError("motor bundle differs from frozen plan")
    if bundle.get("extract_batch") != CANONICAL_EXTRACT_BATCH:
        raise ValueError("motor bundle extraction batch mismatch")
    for arm in ("treatment", "shuffled"):
        _validate_motor_state(bundle.get(arm), plan["d_model"], fit_budget["rank"], arm)
        expected_hash = bundle.get("{}_state_sha256".format(arm))
        if tensor_state_sha256(bundle[arm]) != expected_hash:
            raise ValueError("{} state hash mismatch".format(arm))
        _validate_fit_report(bundle.get("{}_fit".format(arm)), fit_budget, arm)
    if (
        bundle["treatment_fit"]["schedule_sha256"]
        != bundle["shuffled_fit"]["schedule_sha256"]
    ):
        raise ValueError("learned arms used different batch schedules")
    expected_rows = int(plan["board"]["rows"])
    _validate_linear_diagnostic(bundle.get("linear_diagnostic"), expected_rows)
    _validate_fit_teacher_forced_evidence(
        bundle.get("fit_feature_metrics"),
        plan,
        replayed_feature_merge,
        replayed_features,
        bundle["treatment"],
        bundle["shuffled"],
        deployment_device,
    )
    if bundle.get("claim_boundary") != CANONICAL_FIT_CLAIM_BOUNDARY:
        raise ValueError("carry-motor fit claim boundary is invalid")


def validate_motor_bundle(
    bundle,
    expected_bindings,
    current_sources,
    source_contract,
    plan_sha256,
    plan,
    fit_rows,
):
    """Replay all sealed shards, then validate a canonical fit on H100."""
    plan_path = Path(plan.get("plan_path", ""))
    validate_canonical_plan_layout(plan_path, source_contract["git_commit"])
    shard_bounds = []
    try:
        shard_bounds, shard_payloads = _bind_and_load_canonical_shards(
            plan["shards"], plan_path.parent
        )
        replayed_features, replayed_feature_merge = merge_feature_shards(
            shard_payloads,
            fit_rows,
            {
                **expected_bindings,
                "scientific_source_sha256": current_sources,
            },
            source_contract,
            plan,
            plan_sha256,
        )
        for shard_bound in shard_bounds:
            shard_bound.verify_path()
        deployment_device = require_canonical_cuda_runtime()
        _validate_motor_bundle_against_replayed_features(
            bundle,
            expected_bindings,
            current_sources,
            source_contract,
            plan_sha256,
            plan,
            replayed_features,
            replayed_feature_merge,
            deployment_device,
        )
        for shard_bound in shard_bounds:
            shard_bound.verify_path()
        validate_canonical_plan_layout(plan_path, source_contract["git_commit"])
    finally:
        for shard_bound in shard_bounds:
            shard_bound.close()


def _reject_nonfinite_tree(value, path="root"):
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise ValueError("non-string JSON key at {}".format(path))
        for key, item in value.items():
            _reject_nonfinite_tree(item, "{}.{}".format(path, key))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_nonfinite_tree(item, "{}[{}]".format(path, index))
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError("non-finite JSON value at {}".format(path))
    elif not isinstance(value, (str, int, float, bool, type(None))):
        raise ValueError("non-JSON value at {}".format(path))


def _require_exact_mapping(value, expected_keys, label):
    if not isinstance(value, dict) or set(value) != set(expected_keys):
        raise ValueError("{} schema mismatch".format(label))


def _require_count(value, label, upper=None):
    if type(value) is not int or value < 0 or (upper is not None and value > upper):
        raise ValueError("{} count is invalid".format(label))


def _require_bool(value, label):
    if type(value) is not bool:
        raise ValueError("{} must be boolean".format(label))


def _require_string(value, label, *, nonempty=False):
    if not isinstance(value, str) or (nonempty and not value):
        raise ValueError("{} string is invalid".format(label))


def _validate_state_record(value, label, *, allow_none=False):
    if value is None and allow_none:
        return
    _require_exact_mapping(value, {"op", "w", "p", "c", "a", "b", "r", "z"}, label)
    if (
        not isinstance(value["op"], str)
        or not all(type(value[name]) is int for name in ("w", "p", "c", "z"))
        or not all(isinstance(value[name], str) for name in ("a", "b", "r"))
    ):
        raise ValueError("{} field types are invalid".format(label))
    try:
        canonical_state(value)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("{} is not canonical".format(label)) from exc


def _validate_rollout(report, expected_episode, label):
    _require_exact_mapping(
        report,
        {
            "success",
            "state_closed_loop",
            "state",
            "rows",
            "final_response",
            "final_correct",
        },
        label,
    )
    for name in ("success", "state_closed_loop", "final_correct"):
        _require_bool(report[name], "{}.{}".format(label, name))
    _require_string(report["final_response"], "{}.final_response".format(label))
    _validate_state_record(report["state"], "{}.state".format(label))

    initial = parse_state(expected_episode.get("initial_state"))
    expected_lines = expected_episode.get("expected_states")
    style = expected_episode.get("prompt_style")
    if (
        initial is None
        or not isinstance(expected_lines, list)
        or not expected_lines
        or style not in FIT_STYLES
        or type(expected_episode.get("expected_answer")) is not int
    ):
        raise ValueError("{} expected episode contract is invalid".format(label))
    expected_states = [parse_state(line) for line in expected_lines]
    if any(state is None for state in expected_states):
        raise ValueError("{} expected state contract is invalid".format(label))

    rows = report["rows"]
    if not isinstance(rows, list) or not rows or len(rows) > len(expected_states):
        raise ValueError("{} rollout rows are invalid".format(label))
    expected_input = initial
    failed = False
    for index, row in enumerate(rows):
        row_label = "{}.rows[{}]".format(label, index)
        _require_exact_mapping(
            row,
            {
                "index",
                "prompt",
                "response",
                "input_state",
                "predicted_state",
                "expected_state",
                "correct",
            },
            row_label,
        )
        if type(row["index"]) is not int or row["index"] != index:
            raise ValueError("{} index is invalid".format(row_label))
        _require_string(row["prompt"], "{}.prompt".format(row_label), nonempty=True)
        _require_string(row["response"], "{}.response".format(row_label))
        _require_bool(row["correct"], "{}.correct".format(row_label))
        _validate_state_record(row["input_state"], "{}.input_state".format(row_label))
        _validate_state_record(
            row["predicted_state"],
            "{}.predicted_state".format(row_label),
            allow_none=True,
        )
        _validate_state_record(
            row["expected_state"], "{}.expected_state".format(row_label)
        )
        if row["input_state"] != expected_input:
            raise ValueError("{} input-state chain is invalid".format(row_label))
        if row["expected_state"] != expected_states[index]:
            raise ValueError(
                "{} expected state differs from frozen episode".format(row_label)
            )
        if apply_microstep(row["input_state"]) != row["expected_state"]:
            raise ValueError("{} expected transition is invalid".format(row_label))
        if row["prompt"] != microstep_prompt(row["input_state"], style=style):
            raise ValueError("{} prompt differs from frozen protocol".format(row_label))
        predicted = parse_state(row["response"])
        if row["predicted_state"] != predicted:
            raise ValueError(
                "{} parsed response state is inconsistent".format(row_label)
            )
        correct = predicted == row["expected_state"]
        if row["correct"] is not correct:
            raise ValueError("{} correctness flag is inconsistent".format(row_label))
        if correct:
            if failed:
                raise ValueError("{} continues after a failed transition".format(label))
            expected_input = predicted
        else:
            failed = True
            if index != len(rows) - 1:
                raise ValueError("{} continues after a failed transition".format(label))

    if failed:
        if (
            report["success"]
            or report["state_closed_loop"]
            or report["final_correct"]
            or report["final_response"] != ""
            or report["state"] != rows[-1]["input_state"]
        ):
            raise ValueError(
                "{} failed-rollout accounting is inconsistent".format(label)
            )
        return
    if len(rows) != len(expected_states) or report["state_closed_loop"] is not True:
        raise ValueError("{} successful state loop is incomplete".format(label))
    if report["state"] != expected_input or report["state"]["z"] != 1:
        raise ValueError("{} terminal state is inconsistent".format(label))
    if state_answer(report["state"]) != expected_episode["expected_answer"]:
        raise ValueError("{} terminal answer differs from frozen episode".format(label))
    final_correct = parse_answer(report["final_response"]) == state_answer(
        report["state"]
    )
    if (
        report["final_correct"] is not final_correct
        or report["success"] is not final_correct
    ):
        raise ValueError("{} final-answer accounting is inconsistent".format(label))


def _validate_arm_site_counts(arm, sites, fires, label):
    _require_count(sites, "{}.site_opportunities".format(label))
    _require_count(fires, "{}.motor_fires".format(label), sites)
    if arm == "base":
        if fires != 0:
            raise ValueError("{} base arm cannot fire a motor".format(label))
    elif fires != sites:
        raise ValueError("{} motor arm must fire at every reached site".format(label))


def _validate_generation_evidence(
    generation,
    prompt,
    response,
    motor_present,
    tokenizer,
    expected_sequence_cap,
    expected_max_new,
    label,
    require_full_logits=False,
):
    """Replay routing only at the exact decoded token boundaries used online."""
    _require_exact_mapping(
        generation,
        {
            "prompt_token_count",
            "prompt_token_ids_sha256",
            "sequence_cap",
            "max_new",
            "eos_token_id",
            "token_ids",
            "token_ids_sha256",
            "boundaries",
            "boundaries_sha256",
            "stop_reason",
        },
        label,
    )
    cap = generation["sequence_cap"]
    max_new = generation["max_new"]
    if (
        type(cap) is not int
        or cap != expected_sequence_cap
        or cap <= 0
        or type(max_new) is not int
        or max_new != expected_max_new
        or max_new <= 0
    ):
        raise ValueError("{} generation limits differ from frozen decode".format(label))
    prompt_ids = tokenizer.encode(prompt).ids[-cap:]
    if (
        type(generation["prompt_token_count"]) is not int
        or generation["prompt_token_count"] != len(prompt_ids)
        or generation["prompt_token_ids_sha256"] != stable_json_sha256(prompt_ids)
    ):
        raise ValueError("{} prompt token identity mismatch".format(label))
    eos_id = tokenizer.token_to_id("<|endoftext|>")
    if type(eos_id) is not int or generation["eos_token_id"] != eos_id:
        raise ValueError("{} EOS token identity mismatch".format(label))
    token_ids = generation["token_ids"]
    boundaries = generation["boundaries"]
    if (
        not isinstance(token_ids, list)
        or not token_ids
        or len(token_ids) > max_new
        or not isinstance(boundaries, list)
        or len(boundaries) != len(token_ids)
        or generation["token_ids_sha256"] != stable_json_sha256(token_ids)
        or generation["boundaries_sha256"] != stable_json_sha256(boundaries)
    ):
        raise ValueError("{} token-boundary ledger is incomplete".format(label))
    vocab_size = tokenizer.get_vocab_size()
    router = CarryRouter(prompt, motor_present)
    derived_stop = None
    for index, (token_id, boundary) in enumerate(zip(token_ids, boundaries)):
        boundary_label = "{}.boundaries[{}]".format(label, index)
        _require_exact_mapping(
            boundary,
            {
                "index",
                "prefix_token_count",
                "decoded_prefix",
                "next_token_id",
                "router_site",
                "motor_fired",
                "full_logits",
            },
            boundary_label,
        )
        if (
            type(token_id) is not int
            or not 0 <= token_id < vocab_size
            or type(boundary["index"]) is not int
            or boundary["index"] != index
            or type(boundary["prefix_token_count"]) is not int
            or boundary["prefix_token_count"] != index
            or boundary["next_token_id"] != token_id
        ):
            raise ValueError("{} token identity is invalid".format(boundary_label))
        _require_string(
            boundary["decoded_prefix"],
            "{}.decoded_prefix".format(boundary_label),
        )
        _require_bool(boundary["router_site"], "{}.router_site".format(boundary_label))
        _require_bool(boundary["motor_fired"], "{}.motor_fired".format(boundary_label))
        full_logits = boundary["full_logits"]
        if require_full_logits:
            _require_exact_mapping(
                full_logits,
                {
                    "dtype",
                    "shape",
                    "byte_count",
                    "bytes_sha256",
                    "identity_sha256",
                },
                "{}.full_logits".format(boundary_label),
            )
            bytes_per_value = {
                "torch.bfloat16": 2,
                "torch.float32": 4,
            }.get(full_logits["dtype"])
            raw_identity = {
                "dtype": full_logits["dtype"],
                "shape": full_logits["shape"],
                "byte_count": full_logits["byte_count"],
                "bytes_sha256": full_logits["bytes_sha256"],
            }
            if (
                bytes_per_value is None
                or full_logits["shape"] != [1, vocab_size]
                or type(full_logits["byte_count"]) is not int
                or full_logits["byte_count"] != vocab_size * bytes_per_value
                or not re.fullmatch(r"[0-9a-f]{64}", str(full_logits["bytes_sha256"]))
                or full_logits["identity_sha256"] != stable_json_sha256(raw_identity)
            ):
                raise ValueError(
                    "{} full-logit identity is invalid".format(boundary_label)
                )
        elif full_logits is not None:
            raise ValueError(
                "{} has unplanned full-logit evidence".format(boundary_label)
            )
        decoded_prefix = decode_tokens(
            tokenizer, token_ids[:index], skip_special_tokens=False
        )
        if boundary["decoded_prefix"] != decoded_prefix:
            raise ValueError("{} decoded prefix mismatch".format(boundary_label))
        prior_sites = router.site_count
        prior_fires = router.fire_count
        router.observe(decoded_prefix)
        expected_site = router.site_count == prior_sites + 1
        expected_fire = router.fire_count == prior_fires + 1
        if (
            boundary["router_site"] is not expected_site
            or boundary["motor_fired"] is not expected_fire
        ):
            raise ValueError("{} router decision mismatch".format(boundary_label))

        decoded_after = decode_tokens(
            tokenizer, token_ids[: index + 1], skip_special_tokens=False
        )
        position = len(prompt_ids) + index
        candidate_stop = (
            "eos"
            if token_id == eos_id
            else "sequence_cap"
            if position >= cap
            else "complete_final_answer"
            if has_complete_final_answer(decoded_after)
            else None
        )
        if candidate_stop is not None:
            if index != len(token_ids) - 1:
                raise ValueError("{} continues after its stop boundary".format(label))
            derived_stop = candidate_stop
    if derived_stop is None:
        if len(token_ids) != max_new:
            raise ValueError("{} stopped without a valid stop condition".format(label))
        derived_stop = "max_new"
    if generation["stop_reason"] != derived_stop:
        raise ValueError("{} stop reason mismatch".format(label))
    response_ids = token_ids[:-1] if token_ids[-1] == eos_id else token_ids
    decoded_response = decode_tokens(tokenizer, response_ids, skip_special_tokens=False)
    if response != decoded_response:
        raise ValueError("{} response differs from generated token IDs".format(label))
    return router.site_count, router.fire_count


def _validate_raw_call(
    call,
    index,
    kind,
    prompt,
    arm,
    tokenizer,
    sequence_cap,
    max_new,
    label,
    require_full_logits=False,
):
    _require_exact_mapping(
        call,
        {
            "index",
            "kind",
            "prompt",
            "response",
            "site_opportunities",
            "motor_fires",
            "generation",
        },
        label,
    )
    if type(call["index"]) is not int or call["index"] != index:
        raise ValueError("{} index is invalid".format(label))
    if call["kind"] != kind or call["prompt"] != prompt:
        raise ValueError("{} identity differs from frozen call".format(label))
    _require_string(call["prompt"], "{}.prompt".format(label), nonempty=True)
    _require_string(call["response"], "{}.response".format(label))
    _require_count(call["site_opportunities"], "{}.site_opportunities".format(label))
    _require_count(call["motor_fires"], "{}.motor_fires".format(label))
    expected_sites, expected_fires = _validate_generation_evidence(
        call["generation"],
        call["prompt"],
        call["response"],
        arm != "base",
        tokenizer,
        sequence_cap,
        max_new,
        "{}.generation".format(label),
        require_full_logits=require_full_logits,
    )
    if (
        call["site_opportunities"] != expected_sites
        or call["motor_fires"] != expected_fires
    ):
        raise ValueError("{} router counts differ from raw response".format(label))
    _validate_arm_site_counts(
        arm, call["site_opportunities"], call["motor_fires"], label
    )
    return call["response"], expected_sites, expected_fires


def _validate_preservation_trace_identity(base_generation, motor_generation, label):
    if base_generation["token_ids"] != motor_generation["token_ids"]:
        raise ValueError("{} token-ID sequences differ".format(label))
    base_logits = [
        boundary["full_logits"] for boundary in base_generation["boundaries"]
    ]
    motor_logits = [
        boundary["full_logits"] for boundary in motor_generation["boundaries"]
    ]
    if base_logits != motor_logits:
        raise ValueError("{} full-logit boundary identities differ".format(label))
    return True, True


def _episode_evidence_record(identity, calls):
    return {
        "index": identity["index"],
        "source_index": identity["source_index"],
        "id": identity["id"],
        "source_split": identity["source_split"],
        "regime": identity["regime"],
        "calls": calls,
    }


def _derive_episode_accounting_from_evidence(
    evidence,
    episode,
    identity,
    arm,
    tokenizer,
    sequence_cap,
    max_new,
    label,
):
    _require_exact_mapping(
        evidence,
        {"index", "source_index", "id", "source_split", "regime", "calls"},
        label,
    )
    for name in ("index", "source_index", "id", "source_split", "regime"):
        if evidence[name] != identity[name]:
            raise ValueError("{} identity differs from frozen selection".format(label))
    initial = parse_state(episode.get("initial_state"))
    expected_lines = episode.get("expected_states")
    style = episode.get("prompt_style")
    if (
        initial is None
        or not isinstance(expected_lines, list)
        or not expected_lines
        or style not in FIT_STYLES
        or type(episode.get("expected_answer")) is not int
    ):
        raise ValueError("{} frozen episode contract is invalid".format(label))
    expected_states = [parse_state(line) for line in expected_lines]
    if any(state is None for state in expected_states):
        raise ValueError("{} frozen expected states are invalid".format(label))
    calls = evidence["calls"]
    if not isinstance(calls, list) or not calls:
        raise ValueError("{} raw call evidence is empty".format(label))

    state = initial
    attempted = correct = sites = fires = 0
    failed = False
    for transition, expected_state in enumerate(expected_states):
        if transition >= len(calls):
            raise ValueError("{} transition call evidence is incomplete".format(label))
        if apply_microstep(state) != expected_state:
            raise ValueError("{} frozen transition is invalid".format(label))
        prompt = microstep_prompt(state, style=style)
        response, call_sites, call_fires = _validate_raw_call(
            calls[transition],
            transition,
            "transition",
            prompt,
            arm,
            tokenizer,
            sequence_cap,
            max_new,
            "{}.calls[{}]".format(label, transition),
        )
        attempted += 1
        sites += call_sites
        fires += call_fires
        predicted = parse_state(response)
        if predicted != expected_state:
            failed = True
            break
        correct += 1
        state = predicted

    final_correct = False
    if failed:
        if len(calls) != attempted:
            raise ValueError("{} continues after a failed transition".format(label))
    else:
        if attempted != len(expected_states) or len(calls) != attempted + 1:
            raise ValueError("{} final call evidence is incomplete".format(label))
        final_response, call_sites, call_fires = _validate_raw_call(
            calls[-1],
            attempted,
            "final",
            final_prompt(state, style=style),
            arm,
            tokenizer,
            sequence_cap,
            max_new,
            "{}.calls[{}]".format(label, attempted),
        )
        sites += call_sites
        fires += call_fires
        if state["z"] != 1 or state_answer(state) != episode["expected_answer"]:
            raise ValueError(
                "{} terminal state differs from frozen episode".format(label)
            )
        final_correct = parse_answer(final_response) == state_answer(state)

    return {
        "index": identity["index"],
        "id": identity["id"],
        "regime": identity["regime"],
        "transition_capacity": len(expected_states),
        "transition_attempted": attempted,
        "first_transition_correct": correct > 0,
        "transition_correct": correct,
        "state_closed_loop_correct": not failed,
        "final_answer_correct": final_correct,
        "site_opportunities": sites,
        "motor_fires": fires,
    }


def _episode_accounting_record(
    index, episode, rollout, site_opportunities, motor_fires, regime=None
):
    rows = rollout["rows"]
    return {
        "index": int(index),
        "id": episode["id"],
        "regime": episode["split"] if regime is None else regime,
        "transition_capacity": len(episode["expected_states"]),
        "transition_attempted": len(rows),
        "first_transition_correct": bool(rows) and bool(rows[0]["correct"]),
        "transition_correct": sum(int(row["correct"]) for row in rows),
        "state_closed_loop_correct": bool(rollout["state_closed_loop"]),
        "final_answer_correct": bool(rollout["success"]),
        "site_opportunities": int(site_opportunities),
        "motor_fires": int(motor_fires),
    }


def _aggregate_episode_accounting(records):
    fields = (
        "first_transition_correct",
        "transition_correct",
        "transition_attempted",
        "state_closed_loop_correct",
        "final_answer_correct",
        "site_opportunities",
        "motor_fires",
    )
    by_regime = {}
    for record in records:
        regime = record["regime"]
        report = by_regime.setdefault(
            regime,
            {
                "episodes": 0,
                **{field: 0 for field in fields},
            },
        )
        report["episodes"] += 1
        for field in fields:
            report[field] += int(record[field])
    return {regime: by_regime[regime] for regime in sorted(by_regime)}


def _validate_episode_accounting(record, episode, identity, arm, index, label):
    _require_exact_mapping(
        record,
        {
            "index",
            "id",
            "regime",
            "transition_capacity",
            "transition_attempted",
            "first_transition_correct",
            "transition_correct",
            "state_closed_loop_correct",
            "final_answer_correct",
            "site_opportunities",
            "motor_fires",
        },
        label,
    )
    if (
        type(record["index"]) is not int
        or record["index"] != index
        or record["id"] != identity["id"]
        or record["regime"] != identity["regime"]
    ):
        raise ValueError(
            "{} identity or order differs from frozen episode".format(label)
        )
    capacity = len(episode["expected_states"])
    if (
        type(record["transition_capacity"]) is not int
        or record["transition_capacity"] != capacity
    ):
        raise ValueError(
            "{} transition capacity differs from frozen episode".format(label)
        )
    for name in (
        "first_transition_correct",
        "state_closed_loop_correct",
        "final_answer_correct",
    ):
        _require_bool(record[name], "{}.{}".format(label, name))
    for name in (
        "transition_attempted",
        "transition_correct",
        "site_opportunities",
        "motor_fires",
    ):
        _require_count(record[name], "{}.{}".format(label, name))
    attempted = record["transition_attempted"]
    correct = record["transition_correct"]
    first_correct = record["first_transition_correct"]
    state_closed = record["state_closed_loop_correct"]
    final_correct = record["final_answer_correct"]
    if not 1 <= attempted <= capacity or correct > attempted:
        raise ValueError("{} transition counts are invalid".format(label))
    if first_correct is not (correct > 0):
        raise ValueError("{} first-transition accounting is inconsistent".format(label))
    if state_closed:
        if attempted != capacity or correct != capacity:
            raise ValueError("{} closed-loop accounting is incomplete".format(label))
    elif correct + 1 != attempted:
        raise ValueError("{} failed-rollout accounting is inconsistent".format(label))
    if final_correct and not state_closed:
        raise ValueError("{} final-answer accounting is inconsistent".format(label))
    if record["site_opportunities"] > attempted:
        raise ValueError("{} site count exceeds attempted transitions".format(label))
    _validate_arm_site_counts(
        arm, record["site_opportunities"], record["motor_fires"], label
    )


def _validate_regime_metrics(
    report, expected_episodes, transition_capacity, arm, label
):
    _require_exact_mapping(
        report,
        {
            "episodes",
            "first_transition_correct",
            "transition_correct",
            "transition_attempted",
            "state_closed_loop_correct",
            "final_answer_correct",
            "site_opportunities",
            "motor_fires",
        },
        label,
    )
    for name in (
        "episodes",
        "first_transition_correct",
        "transition_correct",
        "transition_attempted",
        "state_closed_loop_correct",
        "final_answer_correct",
    ):
        _require_count(report[name], "{}.{}".format(label, name))
    if report["episodes"] != expected_episodes:
        raise ValueError("{} episode count differs from frozen board".format(label))
    if not expected_episodes <= report["transition_attempted"] <= transition_capacity:
        raise ValueError("{} transition-attempt count is invalid".format(label))
    if not (
        report["first_transition_correct"] <= expected_episodes
        and report["transition_correct"] <= report["transition_attempted"]
        and report["state_closed_loop_correct"]
        <= report["first_transition_correct"]
        <= report["transition_correct"]
        and report["final_answer_correct"] <= report["state_closed_loop_correct"]
    ):
        raise ValueError("{} correctness counts are inconsistent".format(label))
    if report["site_opportunities"] > report["transition_attempted"]:
        raise ValueError("{} site count exceeds attempted transitions".format(label))
    _validate_arm_site_counts(
        arm, report["site_opportunities"], report["motor_fires"], label
    )


def _validate_cycle_metrics(report, contract, arm, tokenizer, sequence_cap, max_new):
    expected_keys = {
        "records",
        "first_exact",
        "second_exact_after_first",
        "integrated_two_call_exact",
        "site_opportunities",
        "motor_fires",
        "cases",
    }
    if not isinstance(report, dict) or set(report) != expected_keys:
        raise ValueError("development cycle schema mismatch")
    if not isinstance(contract, dict) or set(contract) != {"records", "cases"}:
        raise ValueError("development cycle contract schema mismatch")
    expected_records = contract["records"]
    if type(expected_records) is not int or expected_records != CANONICAL_CYCLE_CASES:
        raise ValueError("development cycle contract is invalid")
    if type(report["records"]) is not int or report["records"] != expected_records:
        raise ValueError("development cycle record count mismatch")
    cases = report["cases"]
    expected_cases = contract["cases"]
    if (
        not isinstance(cases, list)
        or len(cases) != expected_records
        or not isinstance(expected_cases, list)
        or len(expected_cases) != expected_records
    ):
        raise ValueError("development cycle case evidence is incomplete")
    derived_first = derived_second = derived_integrated = 0
    derived_sites = derived_fires = 0
    for index, (row, expected) in enumerate(zip(cases, expected_cases)):
        label = "development cycle {} cases[{}]".format(arm, index)
        _require_exact_mapping(
            row,
            {
                "index",
                "episode_id",
                "first_exact",
                "second_exact",
                "integrated_two_call_exact",
                "site_opportunities",
                "motor_fires",
                "calls",
            },
            label,
        )
        if (
            row["index"] != expected["index"]
            or row["index"] != index
            or row["episode_id"] != expected["episode_id"]
        ):
            raise ValueError("{} identity differs from frozen cycle".format(label))
        for name in ("first_exact", "second_exact", "integrated_two_call_exact"):
            _require_bool(row[name], "{}.{}".format(label, name))
        _require_count(row["site_opportunities"], "{}.site_opportunities".format(label))
        _require_count(row["motor_fires"], "{}.motor_fires".format(label))
        calls = row["calls"]
        if not isinstance(calls, list) or not calls:
            raise ValueError("{} raw call evidence is empty".format(label))
        first_response, first_sites, first_fires = _validate_raw_call(
            calls[0],
            0,
            "first",
            expected["first_prompt"],
            arm,
            tokenizer,
            sequence_cap,
            max_new,
            "{}.calls[0]".format(label),
        )
        first_exact = parse_state(first_response) == expected["expected_first"]
        second_exact = False
        case_sites = first_sites
        case_fires = first_fires
        if first_exact:
            if len(calls) != 2:
                raise ValueError("{} two-call evidence is incomplete".format(label))
            second_response, second_sites, second_fires = _validate_raw_call(
                calls[1],
                1,
                "second",
                expected["second_prompt"],
                arm,
                tokenizer,
                sequence_cap,
                max_new,
                "{}.calls[1]".format(label),
            )
            second_exact = parse_state(second_response) == expected["expected_second"]
            case_sites += second_sites
            case_fires += second_fires
        elif len(calls) != 1:
            raise ValueError("{} contains an impossible second call".format(label))
        integrated = first_exact and second_exact
        if (
            row["first_exact"] is not first_exact
            or row["second_exact"] is not second_exact
            or row["integrated_two_call_exact"] is not integrated
            or row["site_opportunities"] != case_sites
            or row["motor_fires"] != case_fires
        ):
            raise ValueError("{} aggregate differs from raw calls".format(label))
        derived_first += int(first_exact)
        derived_second += int(second_exact)
        derived_integrated += int(integrated)
        derived_sites += case_sites
        derived_fires += case_fires
    expected_totals = {
        "first_exact": derived_first,
        "second_exact_after_first": derived_second,
        "integrated_two_call_exact": derived_integrated,
        "site_opportunities": derived_sites,
        "motor_fires": derived_fires,
    }
    for name, expected in expected_totals.items():
        if type(report[name]) is not int or report[name] != expected:
            raise ValueError(
                "development cycle {} total differs from raw calls".format(name)
            )


def _validate_direct_evaluation(evaluation, case, label):
    mode = case["mode"]
    if mode in {"complete", "terminal", "source_deleted"}:
        _require_exact_mapping(evaluation, {"success", "rollouts"}, label)
        _require_bool(evaluation["success"], "{}.success".format(label))
        if (
            not isinstance(evaluation["rollouts"], list)
            or len(evaluation["rollouts"]) != 1
        ):
            raise ValueError("{} rollout count is invalid".format(label))
        _validate_rollout(
            evaluation["rollouts"][0], case, "{}.rollouts[0]".format(label)
        )
        if evaluation["success"] is not evaluation["rollouts"][0]["success"]:
            raise ValueError("{} success flag is inconsistent".format(label))
        return
    if mode == "state_reuse":
        _require_exact_mapping(
            evaluation, {"success", "exact_reuse", "rollouts"}, label
        )
        _require_bool(evaluation["success"], "{}.success".format(label))
        _require_bool(evaluation["exact_reuse"], "{}.exact_reuse".format(label))
        if (
            not isinstance(evaluation["rollouts"], list)
            or len(evaluation["rollouts"]) != 2
        ):
            raise ValueError("{} rollout count is invalid".format(label))
        for index, rollout in enumerate(evaluation["rollouts"]):
            _validate_rollout(rollout, case, "{}.rollouts[{}]".format(label, index))
        exact_reuse = evaluation["rollouts"][0] == evaluation["rollouts"][1]
        success = all(rollout["success"] for rollout in evaluation["rollouts"])
        if (
            evaluation["exact_reuse"] is not exact_reuse
            or evaluation["success"] is not success
        ):
            raise ValueError("{} reuse accounting is inconsistent".format(label))
        return
    if mode != "review":
        raise ValueError("{} has an unknown direct mode".format(label))
    _require_exact_mapping(
        evaluation,
        {
            "success",
            "first_prompt",
            "first_response",
            "review_prompt",
            "review_response",
            "rollouts",
        },
        label,
    )
    _require_bool(evaluation["success"], "{}.success".format(label))
    for name in ("first_prompt", "review_prompt"):
        _require_string(evaluation[name], "{}.{}".format(label, name), nonempty=True)
    for name in ("first_response", "review_response"):
        _require_string(evaluation[name], "{}.{}".format(label, name))
    initial = parse_state(case["initial_state"])
    expected_first = parse_state(case["expected_states"][0])
    if initial is None or expected_first is None:
        raise ValueError("{} frozen review case is invalid".format(label))
    if evaluation["first_prompt"] != microstep_prompt(
        initial, style=case["prompt_style"]
    ):
        raise ValueError(
            "{} first prompt differs from frozen review case".format(label)
        )
    if evaluation["review_prompt"] != _review_prompt(
        initial, evaluation["first_response"]
    ):
        raise ValueError("{} review prompt is inconsistent".format(label))
    reviewed = parse_state(evaluation["review_response"])
    rollouts = evaluation["rollouts"]
    if reviewed != expected_first:
        if rollouts != [] or evaluation["success"] is not False:
            raise ValueError(
                "{} failed review accounting is inconsistent".format(label)
            )
        return
    if not isinstance(rollouts, list) or len(rollouts) != 1:
        raise ValueError("{} reviewed rollout count is invalid".format(label))
    remainder = dict(case)
    remainder["initial_state"] = canonical_state(expected_first)
    remainder["expected_states"] = case["expected_states"][1:]
    _validate_rollout(rollouts[0], remainder, "{}.rollouts[0]".format(label))
    if evaluation["success"] is not rollouts[0]["success"]:
        raise ValueError("{} reviewed success flag is inconsistent".format(label))


def _rollout_raw_call_contract(rollout, style):
    calls = [("transition", row["prompt"], row["response"]) for row in rollout["rows"]]
    if rollout["state_closed_loop"]:
        calls.append(
            (
                "final",
                final_prompt(rollout["state"], style=style),
                rollout["final_response"],
            )
        )
    return calls


def _direct_raw_call_contract(evaluation, case):
    mode = case["mode"]
    if mode in {"complete", "terminal", "source_deleted"}:
        return _rollout_raw_call_contract(
            evaluation["rollouts"][0], case["prompt_style"]
        )
    if mode == "state_reuse":
        calls = []
        for rollout in evaluation["rollouts"]:
            calls.extend(_rollout_raw_call_contract(rollout, case["prompt_style"]))
        return calls
    calls = [
        ("transition", evaluation["first_prompt"], evaluation["first_response"]),
        ("review", evaluation["review_prompt"], evaluation["review_response"]),
    ]
    for rollout in evaluation["rollouts"]:
        calls.extend(_rollout_raw_call_contract(rollout, case["prompt_style"]))
    return calls


def validate_development_eval_result(
    result,
    frozen,
    source_contract,
    checkpoint_step,
    motor_path,
    motor_sha256,
    plan_sha256,
    expected_dev_rows,
    tokenizer,
    sequence_cap,
    expected_development_selection,
    expected_episodes,
    expected_cycle_contract,
    expected_direct_cases,
):
    expected_keys = {
        "audit",
        "checkpoint_step",
        "frozen_sha256",
        "source_contract",
        "motor",
        "motor_sha256",
        "plan_sha256",
        "development_selection",
        "max_new",
        "extract_batch",
        "teacher_forced_carry",
        "results",
        "fresh_direct",
        "non_dws_preservation",
        "claim_boundary",
    }
    if not isinstance(result, dict) or set(result) != expected_keys:
        raise ValueError("development evaluation schema mismatch")
    _reject_nonfinite_tree(result)
    if (
        not _is_canonical_checkpoint_step(checkpoint_step)
        or not _is_canonical_checkpoint_step(result["checkpoint_step"])
        or type(result["max_new"]) is not int
        or type(result["extract_batch"]) is not int
        or result["audit"] != CANONICAL_EVAL_AUDIT
        or result["checkpoint_step"] != checkpoint_step
        or result["frozen_sha256"] != frozen
        or result["source_contract"] != source_contract
        or result["motor"] != str(Path(motor_path).resolve())
        or result["motor_sha256"] != motor_sha256
        or result["plan_sha256"] != plan_sha256
        or result["max_new"] != CANONICAL_MAX_NEW
        or result["extract_batch"] != CANONICAL_EXTRACT_BATCH
        or result["claim_boundary"] != CANONICAL_EVAL_CLAIM_BOUNDARY
    ):
        raise ValueError("development evaluation identity or budget mismatch")
    validate_development_selection_contract(
        expected_development_selection, expected_episodes
    )
    validate_development_selection_contract(
        result["development_selection"], expected_episodes
    )
    if result["development_selection"] != expected_development_selection:
        raise ValueError("development evaluation selection binding mismatch")
    arms = {"base", "dead", "treatment", "shuffled"}
    teacher_forced = result["teacher_forced_carry"]
    if not isinstance(teacher_forced, dict) or set(teacher_forced) != arms:
        raise ValueError("development teacher-forced arm schema mismatch")
    if not isinstance(expected_dev_rows, list) or not expected_dev_rows:
        raise ValueError("development teacher-forced frozen rows are empty")
    for arm in sorted(arms):
        _validate_teacher_forced_report(
            teacher_forced[arm], expected_dev_rows, arm, tokenizer
        )
    if without_motor_fire_accounting(
        teacher_forced["dead"]
    ) != without_motor_fire_accounting(teacher_forced["base"]):
        raise ValueError("development teacher-forced dead/base collapse failed")

    if not isinstance(expected_episodes, list) or not expected_episodes:
        raise ValueError("development frozen episode contract is empty")
    identities = expected_development_selection["identities"]
    regime_counts = collections.Counter()
    transition_capacities = collections.Counter()
    seen_episode_ids = set()
    for identity, episode in zip(identities, expected_episodes):
        episode_id = episode.get("id") if isinstance(episode, dict) else None
        source_split = episode.get("split") if isinstance(episode, dict) else None
        regime = identity["regime"]
        states = episode.get("expected_states") if isinstance(episode, dict) else None
        if (
            not isinstance(episode_id, str)
            or not episode_id
            or episode_id in seen_episode_ids
            or source_split != identity["source_split"]
            or not isinstance(states, list)
            or not states
        ):
            raise ValueError("development frozen episode contract is invalid")
        list(_episode_states(episode))
        seen_episode_ids.add(episode_id)
        regime_counts[regime] += 1
        transition_capacities[regime] += len(states)
    if dict(regime_counts) != expected_development_selection["regime_counts"]:
        raise ValueError("development frozen regime counts are not canonical")
    expected_regimes = sorted(regime_counts)

    results = result["results"]
    if not isinstance(results, dict) or set(results) != arms:
        raise ValueError("development autonomous arm schema mismatch")
    for arm in sorted(arms):
        arm_report = results[arm]
        if not isinstance(arm_report, dict) or set(arm_report) != {
            "by_regime",
            "episode_accounting",
            "episode_evidence",
            "transcripts",
            "cycle",
        }:
            raise ValueError("development autonomous arm report mismatch")
        evidence = arm_report["episode_evidence"]
        if not isinstance(evidence, list) or len(evidence) != len(expected_episodes):
            raise ValueError("development episode evidence is incomplete")
        derived_accounting = []
        for index, (row, episode, identity) in enumerate(
            zip(evidence, expected_episodes, identities)
        ):
            derived = _derive_episode_accounting_from_evidence(
                row,
                episode,
                identity,
                arm,
                tokenizer,
                sequence_cap,
                CANONICAL_MAX_NEW,
                "development {} episode_evidence[{}]".format(arm, index),
            )
            _validate_episode_accounting(
                derived,
                episode,
                identity,
                arm,
                index,
                "development {} derived_accounting[{}]".format(arm, index),
            )
            derived_accounting.append(derived)
        accounting = arm_report["episode_accounting"]
        if not isinstance(accounting, list) or len(accounting) != len(
            expected_episodes
        ):
            raise ValueError("development episode accounting is incomplete")
        for index, (record, episode, identity) in enumerate(
            zip(accounting, expected_episodes, identities)
        ):
            _validate_episode_accounting(
                record,
                episode,
                identity,
                arm,
                index,
                "development {} episode_accounting[{}]".format(arm, index),
            )
        if accounting != derived_accounting:
            raise ValueError("development episode accounting differs from raw evidence")
        recomputed_by_regime = _aggregate_episode_accounting(derived_accounting)
        by_regime = arm_report["by_regime"]
        if not isinstance(by_regime, dict) or set(by_regime) != set(expected_regimes):
            raise ValueError("development regime schema mismatch")
        for regime in expected_regimes:
            _validate_regime_metrics(
                by_regime[regime],
                regime_counts[regime],
                transition_capacities[regime],
                arm,
                "development {} regime {}".format(arm, regime),
            )
        if by_regime != recomputed_by_regime:
            raise ValueError("development regime totals differ from raw evidence")
        transcripts = arm_report["transcripts"]
        expected_transcripts = evidence[: min(15, len(evidence))]
        if not isinstance(transcripts, list) or transcripts != expected_transcripts:
            raise ValueError("development transcript sample is invalid")
        _validate_cycle_metrics(
            arm_report["cycle"],
            expected_cycle_contract,
            arm,
            tokenizer,
            sequence_cap,
            CANONICAL_MAX_NEW,
        )
    if without_motor_fire_accounting(results["dead"]) != without_motor_fire_accounting(
        results["base"]
    ):
        raise ValueError("development autonomous dead/base collapse failed")

    direct = result["fresh_direct"]
    if not isinstance(direct, dict) or set(direct) != arms:
        raise ValueError("development direct arm schema mismatch")
    for arm in sorted(arms):
        report = direct[arm]
        if not isinstance(report, dict) or set(report) != {"correct", "rows"}:
            raise ValueError("development direct report schema mismatch")
        rows = report["rows"]
        if (
            not isinstance(rows, list)
            or len(rows) != len(expected_direct_cases)
            or not rows
        ):
            raise ValueError("development direct rows mismatch")
        correct = 0
        for index, (row, case) in enumerate(zip(rows, expected_direct_cases)):
            label = "development {} direct rows[{}]".format(arm, index)
            _require_exact_mapping(
                row,
                {
                    "id",
                    "mode",
                    "expected_answer",
                    "success",
                    "site_opportunities",
                    "motor_fires",
                    "evaluation",
                    "calls",
                },
                label,
            )
            if (
                row["id"] != case["id"]
                or row["mode"] != case["mode"]
                or type(row["expected_answer"]) is not int
                or row["expected_answer"] != case["expected_answer"]
            ):
                raise ValueError(
                    "{} identity differs from frozen direct case".format(label)
                )
            _require_bool(row["success"], "{}.success".format(label))
            _validate_direct_evaluation(
                row["evaluation"], case, "{}.evaluation".format(label)
            )
            if row["success"] is not row["evaluation"]["success"]:
                raise ValueError("{} success flag is inconsistent".format(label))
            call_contract = _direct_raw_call_contract(row["evaluation"], case)
            calls = row["calls"]
            if not isinstance(calls, list) or len(calls) != len(call_contract):
                raise ValueError("{} raw call evidence is incomplete".format(label))
            derived_sites = derived_fires = 0
            for call_index, (call, expected_call) in enumerate(
                zip(calls, call_contract)
            ):
                kind, prompt, response = expected_call
                observed_response, sites, fires = _validate_raw_call(
                    call,
                    call_index,
                    kind,
                    prompt,
                    arm,
                    tokenizer,
                    sequence_cap,
                    CANONICAL_MAX_NEW,
                    "{}.calls[{}]".format(label, call_index),
                )
                if observed_response != response:
                    raise ValueError(
                        "{} raw response differs from direct evaluation".format(label)
                    )
                derived_sites += sites
                derived_fires += fires
            if (
                row["site_opportunities"] != derived_sites
                or row["motor_fires"] != derived_fires
            ):
                raise ValueError("{} aggregate differs from raw calls".format(label))
            _validate_arm_site_counts(
                arm, row["site_opportunities"], row["motor_fires"], label
            )
            correct += int(row["success"])
        if type(report["correct"]) is not int or report["correct"] != correct:
            raise ValueError("development direct correct count is inconsistent")
    if without_motor_fire_accounting(direct["dead"]) != without_motor_fire_accounting(
        direct["base"]
    ):
        raise ValueError("development direct dead/base collapse failed")

    preservation = result["non_dws_preservation"]
    if not isinstance(preservation, list) or len(preservation) != len(
        NON_DWS_PRESERVATION_PROMPTS
    ):
        raise ValueError("development preservation set mismatch")
    for index, (row, expected_prompt) in enumerate(
        zip(preservation, NON_DWS_PRESERVATION_PROMPTS)
    ):
        label = "development preservation[{}]".format(index)
        _require_exact_mapping(
            row,
            {
                "prompt",
                "base_response",
                "motor_response",
                "token_ids_identical",
                "full_logits_identical",
                "exact_identity",
                "base_sites",
                "base_fires",
                "motor_sites",
                "motor_fires",
                "base_call",
                "motor_call",
            },
            label,
        )
        if row["prompt"] != expected_prompt:
            raise ValueError("{} prompt differs from frozen set".format(label))
        for name in ("base_response", "motor_response"):
            _require_string(row[name], "{}.{}".format(label, name))
        for name in ("token_ids_identical", "full_logits_identical", "exact_identity"):
            _require_bool(row[name], "{}.{}".format(label, name))
        for name in ("base_sites", "base_fires", "motor_sites", "motor_fires"):
            _require_count(row[name], "{}.{}".format(label, name))
        base_response, base_sites, base_fires = _validate_raw_call(
            row["base_call"],
            0,
            "preservation",
            expected_prompt,
            "base",
            tokenizer,
            sequence_cap,
            48,
            "{}.base_call".format(label),
            require_full_logits=True,
        )
        motor_response, motor_sites, motor_fires = _validate_raw_call(
            row["motor_call"],
            0,
            "preservation",
            expected_prompt,
            "treatment",
            tokenizer,
            sequence_cap,
            48,
            "{}.motor_call".format(label),
            require_full_logits=True,
        )
        base_generation = row["base_call"]["generation"]
        motor_generation = row["motor_call"]["generation"]
        token_ids_identical, full_logits_identical = (
            _validate_preservation_trace_identity(
                base_generation, motor_generation, label
            )
        )
        if (
            row["token_ids_identical"] is not token_ids_identical
            or row["full_logits_identical"] is not full_logits_identical
            or row["exact_identity"]
            is not (token_ids_identical and full_logits_identical)
            or row["exact_identity"] is not True
            or row["base_response"] != base_response
            or row["motor_response"] != motor_response
            or base_response != motor_response
            or row["base_sites"] != base_sites
            or row["base_fires"] != base_fires
            or row["motor_sites"] != motor_sites
            or row["motor_fires"] != motor_fires
            or any(
                row[name] != 0
                for name in ("base_sites", "base_fires", "motor_sites", "motor_fires")
            )
        ):
            raise ValueError("{} identity accounting is inconsistent".format(label))


def _validate_confirmation_one_step(report, board, arm, tokenizer, sequence_cap):
    if not isinstance(report, dict) or set(report) != {"correct", "rows"}:
        raise ValueError("confirmation one-step report schema mismatch")
    rows = report["rows"]
    if not isinstance(rows, list) or len(rows) != len(board["rows"]):
        raise ValueError("confirmation one-step evidence is incomplete")
    correct = 0
    for index, (row, expected) in enumerate(zip(rows, board["rows"])):
        label = "confirmation {} one_step[{}]".format(arm, index)
        _require_exact_mapping(
            row,
            {"index", "prompt_sha256", "expected_state", "call"},
            label,
        )
        if (
            row["index"] != index
            or row["prompt_sha256"] != expected["prompt_sha256"]
            or row["expected_state"] != expected["expected_state"]
        ):
            raise ValueError(
                "{} identity differs from secret-derived row".format(label)
            )
        response, _sites, _fires = _validate_raw_call(
            row["call"],
            0,
            "transition",
            expected["prompt"],
            arm,
            tokenizer,
            sequence_cap,
            CANONICAL_MAX_NEW,
            "{}.call".format(label),
        )
        correct += int(parse_state(response) == parse_state(expected["expected_state"]))
    if type(report["correct"]) is not int or report["correct"] != correct:
        raise ValueError("confirmation one-step total differs from raw calls")


def _validate_confirmation_autonomous(report, board, arm, tokenizer, sequence_cap):
    if not isinstance(report, dict) or set(report) != {
        "by_regime",
        "episode_accounting",
        "episode_evidence",
    }:
        raise ValueError("confirmation autonomous report schema mismatch")
    expected_rows = board["rows"]
    evidence = report["episode_evidence"]
    if not isinstance(evidence, list) or len(evidence) != len(expected_rows):
        raise ValueError("confirmation autonomous raw evidence is incomplete")
    derived_accounting = []
    for index, (evidence_row, board_row) in enumerate(zip(evidence, expected_rows)):
        identity = _confirmation_episode_identity(board_row)
        episode = board_row["episode"]
        derived = _derive_episode_accounting_from_evidence(
            evidence_row,
            episode,
            identity,
            arm,
            tokenizer,
            sequence_cap,
            CANONICAL_MAX_NEW,
            "confirmation {} episode_evidence[{}]".format(arm, index),
        )
        _validate_episode_accounting(
            derived,
            episode,
            identity,
            arm,
            index,
            "confirmation {} derived_accounting[{}]".format(arm, index),
        )
        derived_accounting.append(derived)
    if report["episode_accounting"] != derived_accounting:
        raise ValueError("confirmation accounting differs from raw evidence")
    if report["by_regime"] != _aggregate_episode_accounting(derived_accounting):
        raise ValueError("confirmation regime totals differ from raw evidence")


def validate_confirmation_board_binding(board, expected_board):
    if board != expected_board:
        raise ValueError("confirmation board differs from canonical regeneration")
    if stable_json_sha256(board["rows"]) != board["rows_sha256"]:
        raise ValueError("confirmation rows differ from their bound identity")


def validate_confirmation_eval_result(
    result,
    frozen,
    source_contract,
    checkpoint_step,
    motor_path,
    motor_sha256,
    expected_board,
    expected_teacher_rows,
    expected_development_evaluation,
    tokenizer,
    sequence_cap,
):
    """Validate every confirmation identity and derive all retained aggregates."""
    expected_keys = {
        "audit",
        "checkpoint_step",
        "frozen_sha256",
        "source_contract",
        "motor",
        "motor_sha256",
        "confirmation_secret_sha256",
        "frozen_inputs",
        "confirmation_exclusion_contract",
        "confirmation_commitment",
        "confirmation_rows_sha256",
        "confirmation_board",
        "plan",
        "development_evaluation",
        "max_new",
        "extract_batch",
        "teacher_forced_carry",
        "one_step",
        "autonomous",
        "claim_boundary",
    }
    if not isinstance(result, dict) or set(result) != expected_keys:
        raise ValueError("confirmation evaluation schema mismatch")
    _reject_nonfinite_tree(result)
    if (
        result["audit"] != CANONICAL_CONFIRMATION_EVAL_AUDIT
        or not _is_canonical_checkpoint_step(checkpoint_step)
        or not _is_canonical_checkpoint_step(result["checkpoint_step"])
        or result["checkpoint_step"] != checkpoint_step
        or result["frozen_sha256"] != frozen
        or result["source_contract"] != source_contract
        or result["motor"] != str(Path(motor_path).resolve())
        or result["motor_sha256"] != motor_sha256
        or result["confirmation_secret_sha256"] != expected_board["secret_sha256"]
        or result["frozen_inputs"] != expected_board["frozen_inputs"]
        or result["confirmation_exclusion_contract"]
        != expected_board["exclusion_contract"]
        or result["confirmation_commitment"]
        != expected_board["confirmation_commitment"]
        or result["confirmation_rows_sha256"] != expected_board["rows_sha256"]
        or result["plan"] != expected_board["plan"]
        or result["development_evaluation"] != expected_development_evaluation
        or result["max_new"] != CANONICAL_MAX_NEW
        or result["extract_batch"] != CANONICAL_EXTRACT_BATCH
        or result["claim_boundary"] != CANONICAL_CONFIRMATION_EVAL_CLAIM_BOUNDARY
    ):
        raise ValueError("confirmation evaluation identity or binding mismatch")
    validate_confirmation_board_binding(result["confirmation_board"], expected_board)

    arms = {"base", "dead", "treatment", "shuffled"}
    teacher = result["teacher_forced_carry"]
    one_step = result["one_step"]
    autonomous = result["autonomous"]
    if any(
        not isinstance(value, dict) or set(value) != arms
        for value in (teacher, one_step, autonomous)
    ):
        raise ValueError("confirmation evaluation arm schema mismatch")
    for arm in sorted(arms):
        _validate_teacher_forced_report(
            teacher[arm], expected_teacher_rows, arm, tokenizer
        )
        _validate_confirmation_one_step(
            one_step[arm], expected_board, arm, tokenizer, sequence_cap
        )
        _validate_confirmation_autonomous(
            autonomous[arm], expected_board, arm, tokenizer, sequence_cap
        )
    for section in (teacher, one_step, autonomous):
        if without_motor_fire_accounting(
            section["dead"]
        ) != without_motor_fire_accounting(section["base"]):
            raise ValueError("confirmation dead/base collapse failed")


def validate_artifact_receipt(bound_input, expected_sha256):
    if not expected_sha256:
        raise ValueError("artifact receipt SHA-256 is required")
    if bound_input.sha256 != expected_sha256:
        raise ValueError("artifact hash mismatch")


def initial_motor_state(d_model):
    torch.manual_seed(FIT_SEED)
    motor = CarryMotor(int(d_model), RANK)
    state = {
        name: tensor.detach().clone() for name, tensor in motor.state_dict().items()
    }
    return state, tensor_state_sha256(state)


def _plan_shard_descriptors(root, rows):
    descriptors = []
    for shard_index in range(CANONICAL_FEATURE_SHARDS):
        indices = feature_shard_indices(
            len(rows), shard_index, CANONICAL_FEATURE_SHARDS
        )
        descriptors.append(
            {
                "shard_index": shard_index,
                "rows": len(indices),
                "global_indices_sha256": stable_json_sha256(indices),
                "row_identity_sha256": row_identity_sha256(rows, indices),
                "artifact": str(
                    root / "shard_{:02d}".format(shard_index) / "features.pt"
                ),
            }
        )
    return descriptors


def validate_plan_confirmation_binding(plan, bound, frozen, confirmation_commitment):
    expected = {
        "path": str(bound["confirmation_commitment"].path),
        "sha256": frozen["confirmation_commitment"],
        "document": confirmation_commitment,
    }
    if (
        not isinstance(plan, dict)
        or plan.get("confirmation_commitment") != expected
        or plan.get("confirmation_exclusion_contract")
        != confirmation_commitment["exclusion_contract"]
    ):
        raise ValueError("canonical plan confirmation commitment mismatch")


def _expected_plan(
    root,
    bound,
    frozen,
    source_contract,
    confirmation_commitment,
    checkpoint,
    rows,
    board,
    tokenizer,
):
    cfg = checkpoint["cfg"]
    checkpoint_step = checkpoint.get("step")
    if not _is_canonical_checkpoint_step(checkpoint_step):
        raise ValueError("canonical checkpoint-step identity must be sft_ep1")
    if int(cfg.get("n_loop", 1)) != 1:
        raise ValueError("carry motor requires n_loop=1")
    if not torch.version.cuda:
        raise RuntimeError("canonical plan requires a CUDA-enabled PyTorch build")
    d_model = int(cfg["d_model"])
    vocab_size = int(cfg["vocab_size"])
    zero_id = token_for_digit(tokenizer, "x=", "0")[1]
    one_id = token_for_digit(tokenizer, "x=", "1")[1]
    if tokenizer.decode([zero_id]) != "0" or tokenizer.decode([one_id]) != "1":
        raise ValueError("carry tokens are not standalone decimal digits")
    _, initial_sha256 = initial_motor_state(d_model)
    _, control = permuted_control_labels(rows)
    _, schedule_sha256 = _batch_schedule(
        len(rows), CANONICAL_BATCH, CANONICAL_UPDATES, FIT_SEED
    )
    sentinels = feature_sentinel_indices(rows)
    source_hashes = {
        name.removeprefix("source:"): digest
        for name, digest in frozen.items()
        if name.startswith("source:")
    }
    frozen_inputs = {
        name: {"path": str(bound[name].path), "sha256": frozen[name]}
        for name in ("checkpoint", "tokenizer", "episodes", "cycle")
    }
    _, development_selection = canonical_development_selection(bound["episodes"].text())
    board = dict(board)
    board["rows_sha256"] = stable_json_sha256(rows)
    return {
        "audit": CANONICAL_PLAN_AUDIT,
        "canonical": True,
        "source_contract": source_contract,
        "scientific_source_sha256": source_hashes,
        "frozen_inputs": frozen_inputs,
        "confirmation_commitment": {
            "path": str(bound["confirmation_commitment"].path),
            "sha256": frozen["confirmation_commitment"],
            "document": confirmation_commitment,
        },
        "confirmation_exclusion_contract": confirmation_commitment[
            "exclusion_contract"
        ],
        "development_selection": development_selection,
        "checkpoint_step": checkpoint_step,
        "d_model": d_model,
        "vocab_size": vocab_size,
        "zero_id": int(zero_id),
        "one_id": int(one_id),
        "board": board,
        "board_rows_sha256": board["rows_sha256"],
        "extraction_order_sha256": stable_json_sha256(
            [row["prefix_sha256"] for row in rows]
        ),
        "sentinel_indices": sentinels,
        "sentinel_row_identity_sha256": row_identity_sha256(rows, sentinels),
        "shard_count": CANONICAL_FEATURE_SHARDS,
        "shards": _plan_shard_descriptors(root, rows),
        "plan_path": str(root / "plan.json"),
        "fit_artifact": str(root / "fit" / "motor.pt"),
        "development_eval_artifact": str(root / "development_eval" / "evaluation.json"),
        "confirmation_eval_artifact": str(
            root / "confirmation_eval" / "evaluation.json"
        ),
        "runtime_contract": {
            "artifact_runtime": {
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "device": CANONICAL_DEVICE_NAME,
            },
            "deployment_logit_dtype": "torch.bfloat16",
            "extract_batch": CANONICAL_EXTRACT_BATCH,
            "teacher_scoring_contract": CANONICAL_TEACHER_SCORING_CONTRACT,
        },
        "fit_budget": {
            "seed": FIT_SEED,
            "rank": RANK,
            "quota": FIT_QUOTA,
            "updates": CANONICAL_UPDATES,
            "batch_size": CANONICAL_BATCH,
            "lr": CANONICAL_LR,
            "weight_decay": CANONICAL_WEIGHT_DECAY,
            "initial_state_sha256": initial_sha256,
            "schedule_sha256": schedule_sha256,
            "control": control,
            "teacher_row_identity_sha256": stable_json_sha256(
                [
                    _fit_teacher_row_identity(row, index)
                    for index, row in enumerate(rows)
                ]
            ),
        },
        "claim_boundary": (
            "This plan freezes one canonical development fit and its legal artifact paths. "
            "It establishes no feature, motor, evaluation, or reasoning result."
        ),
    }


def _plan(args):
    if args.allow_non_cuda:
        raise SystemExit("canonical plan does not accept --allow-non-cuda")
    validate_canonical_confirmation_args(args)
    bound, frozen = bind_frozen_inputs(
        args.ckpt, args.episodes, args.tokenizer, args.cycle
    )
    temporary = None
    try:
        source_contract = validate_source_contract(args, frozen, canonical=True)
        confirmation_commitment = bind_confirmation_commitment(
            args, bound, frozen, source_contract
        )
        root = Path(args.plan_root)
        expected_root = canonical_plan_root(source_contract["git_commit"])
        if str(args.plan_root) != str(expected_root) or root != expected_root:
            raise ValueError("canonical plan root is not the sole commit-bound path")
        if not root.is_absolute() or root.exists() or root.is_symlink():
            raise FileExistsError("canonical plan root must be a new absolute path")
        tokenizer = Tokenizer.from_str(bound["tokenizer"].text())
        bound["checkpoint"].handle.seek(0)
        checkpoint = torch.load(bound["checkpoint"].handle, map_location="cpu")
        bound["checkpoint"].handle.seek(0)
        rows, board = generate_fit_rows(
            tokenizer, bound["episodes"].text(), FIT_SEED, FIT_QUOTA
        )
        plan = _expected_plan(
            root,
            bound,
            frozen,
            source_contract,
            confirmation_commitment,
            checkpoint,
            rows,
            board,
            tokenizer,
        )
        root.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(
            tempfile.mkdtemp(prefix=".{}.partial-".format(root.name), dir=root.parent)
        )
        for descriptor in plan["shards"]:
            relative = Path(descriptor["artifact"]).relative_to(root)
            (temporary / relative.parent).mkdir(mode=0o700)
        (temporary / "fit").mkdir(mode=0o700)
        (temporary / "development_eval").mkdir(mode=0o700)
        (temporary / "confirmation_eval").mkdir(mode=0o700)
        atomic_json(temporary / "plan.json", plan)
        verify_bound_inputs(bound)
        os.chmod(temporary, 0o555)
        if root.exists():
            raise FileExistsError("canonical plan root appeared during publication")
        os.rename(temporary, root)
        temporary = None
        parent_fd = os.open(root.parent, os.O_RDONLY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
        plan_path = root / "plan.json"
        validate_canonical_plan_layout(plan_path, source_contract["git_commit"])
        print(
            "[carry-motor] plan={} sha256={}".format(plan_path, sha256_file(plan_path)),
            flush=True,
        )
    finally:
        if temporary is not None:
            shutil.rmtree(temporary, ignore_errors=True)
        close_bound_inputs(bound)


def _load_validated_plan(
    args,
    bound,
    frozen,
    source_contract,
    confirmation_commitment,
    checkpoint,
    rows,
    board,
    tokenizer,
):
    expected_root = canonical_plan_root(source_contract["git_commit"])
    plan_path = Path(args.plan)
    validate_canonical_plan_layout(plan_path, source_contract["git_commit"])
    plan_bound = BoundInput(plan_path)
    try:
        if plan_bound.path != expected_root / "plan.json":
            raise ValueError("canonical plan path changed before binding")
        validate_artifact_receipt(plan_bound, args.plan_sha256)
        plan = _load_exact_json(plan_bound.text(), "canonical plan")
        validate_plan_confirmation_binding(plan, bound, frozen, confirmation_commitment)
        expected = _expected_plan(
            expected_root,
            bound,
            frozen,
            source_contract,
            confirmation_commitment,
            checkpoint,
            rows,
            board,
            tokenizer,
        )
        if plan != expected:
            raise ValueError("canonical plan content mismatch")
        if plan_bound.path != Path(expected["plan_path"]):
            raise ValueError("canonical plan path mismatch")
        validate_canonical_plan_layout(plan_bound.path, source_contract["git_commit"])
        plan_bound.verify_path()
        return plan_bound, plan
    except Exception:
        plan_bound.close()
        raise


def _runtime_device():
    return (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )


def _validate_confirmation_preflight(args):
    """Validate commitment semantics on CPU before any canonical CUDA probe."""
    validate_canonical_confirmation_args(args)
    if torch.cuda.is_initialized():
        raise RuntimeError("confirmation semantic preflight must precede CUDA")
    bound, frozen = bind_frozen_inputs(
        args.ckpt, args.episodes, args.tokenizer, args.cycle
    )
    try:
        source_contract = validate_source_contract(args, frozen, canonical=True)
        document = bind_confirmation_commitment(args, bound, frozen, source_contract)
        verify_bound_inputs(bound)
        if torch.cuda.is_initialized():
            raise RuntimeError("confirmation semantic preflight initialized CUDA")
        print(
            "[carry-motor] confirmation semantic preflight sha256={} exclusions={}".format(
                frozen["confirmation_commitment"],
                document["exclusion_contract"]["identity_sha256"],
            ),
            flush=True,
        )
    finally:
        close_bound_inputs(bound)


def _extract_shard(args):
    if args.allow_non_cuda:
        raise SystemExit("canonical extraction does not accept --allow-non-cuda")
    validate_canonical_extract_args(args, canonical=True)
    bound, frozen = bind_frozen_inputs(
        args.ckpt, args.episodes, args.tokenizer, args.cycle
    )
    plan_bound = None
    try:
        source_contract = validate_source_contract(args, frozen, canonical=True)
        confirmation_commitment = bind_confirmation_commitment(
            args, bound, frozen, source_contract
        )
        device = require_canonical_cuda_runtime()
        tokenizer = Tokenizer.from_str(bound["tokenizer"].text())
        zero_id = token_for_digit(tokenizer, "x=", "0")[1]
        one_id = token_for_digit(tokenizer, "x=", "1")[1]
        if tokenizer.decode([zero_id]) != "0" or tokenizer.decode([one_id]) != "1":
            raise ValueError("carry tokens are not standalone decimal digits")
        checkpoint, model = load_model(bound["checkpoint"], device)
        rows, board = generate_fit_rows(
            tokenizer, bound["episodes"].text(), FIT_SEED, FIT_QUOTA
        )
        plan_bound, plan = _load_validated_plan(
            args,
            bound,
            frozen,
            source_contract,
            confirmation_commitment,
            checkpoint,
            rows,
            board,
            tokenizer,
        )
        board_rows_sha256 = stable_json_sha256(rows)
        board["rows_sha256"] = board_rows_sha256
        descriptor = plan["shards"][args.shard_index]
        out = Path(descriptor["artifact"])
        if out.parent.parent != plan_bound.path.parent or out.parent.is_symlink():
            raise ValueError("planned feature-shard directory is invalid")
        if out.exists():
            needs_seal = validate_existing_artifact_stage(out)
            existing = BoundInput(out)
            try:
                existing.handle.seek(0)
                artifact = torch.load(existing.handle, map_location="cpu")
                existing.handle.seek(0)
                expected_bindings = {
                    "base_checkpoint_sha256": frozen["checkpoint"],
                    "tokenizer_sha256": frozen["tokenizer"],
                    "episodes_sha256": frozen["episodes"],
                    "cycle_sha256": frozen["cycle"],
                    "confirmation_commitment_sha256": frozen["confirmation_commitment"],
                    "scientific_source_sha256": plan["scientific_source_sha256"],
                }
                validate_canonical_feature_shard(
                    artifact,
                    out,
                    rows,
                    expected_bindings,
                    source_contract,
                    plan,
                    plan_bound.sha256,
                    args.shard_index,
                )
                verify_bound_inputs(bound)
                plan_bound.verify_path()
                existing.verify_path()
                if needs_seal:
                    seal_output_directory(out)
                validate_canonical_plan_layout(
                    plan_bound.path, source_contract["git_commit"]
                )
                print(
                    "[carry-motor] existing planned shard is valid index={} sha256={}".format(
                        args.shard_index, existing.sha256
                    ),
                    flush=True,
                )
                return
            finally:
                existing.close()
        require_empty_planned_directory(out)
        indices = feature_shard_indices(
            len(rows), args.shard_index, CANONICAL_FEATURE_SHARDS
        )
        sentinels = feature_sentinel_indices(rows)
        features = extract_frozen_features(
            model,
            [rows[index] for index in indices],
            zero_id,
            one_id,
            device,
            batch_size=CANONICAL_EXTRACT_BATCH,
        )
        sentinel_features = extract_frozen_features(
            model,
            [rows[index] for index in sentinels],
            zero_id,
            one_id,
            device,
            batch_size=CANONICAL_EXTRACT_BATCH,
        )
        verify_bound_inputs(bound)
        source_hashes = {
            name.removeprefix("source:"): digest
            for name, digest in frozen.items()
            if name.startswith("source:")
        }
        artifact = {
            "audit": CANONICAL_SHARD_AUDIT,
            "canonical": True,
            "plan_sha256": plan_bound.sha256,
            "base_checkpoint_sha256": frozen["checkpoint"],
            "tokenizer_sha256": frozen["tokenizer"],
            "episodes_sha256": frozen["episodes"],
            "cycle_sha256": frozen["cycle"],
            "confirmation_commitment_sha256": frozen["confirmation_commitment"],
            "scientific_source_sha256": source_hashes,
            "source_contract": source_contract,
            "checkpoint_step": checkpoint.get("step"),
            "board": plan["board"],
            "board_rows_sha256": board_rows_sha256,
            "shard_index": int(args.shard_index),
            "shard_count": CANONICAL_FEATURE_SHARDS,
            "global_indices": indices,
            "row_identity_sha256": row_identity_sha256(rows, indices),
            "sentinel_indices": sentinels,
            "sentinel_row_identity_sha256": row_identity_sha256(rows, sentinels),
            "extract_batch": CANONICAL_EXTRACT_BATCH,
            "features": features,
            "feature_payload_sha256": feature_payload_sha256(features),
            "sentinel_features": sentinel_features,
            "sentinel_payload_sha256": feature_payload_sha256(sentinel_features),
            "runtime": {
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "device": (
                    torch.cuda.get_device_name(0) if device == "cuda" else device
                ),
            },
            "claim_boundary": CANONICAL_SHARD_CLAIM_BOUNDARY,
        }
        expected_bindings = {
            "base_checkpoint_sha256": frozen["checkpoint"],
            "tokenizer_sha256": frozen["tokenizer"],
            "episodes_sha256": frozen["episodes"],
            "cycle_sha256": frozen["cycle"],
            "confirmation_commitment_sha256": frozen["confirmation_commitment"],
            "scientific_source_sha256": source_hashes,
        }
        validate_canonical_feature_shard(
            artifact,
            out,
            rows,
            expected_bindings,
            source_contract,
            plan,
            plan_bound.sha256,
            args.shard_index,
        )
        atomic_torch(out, artifact, staging_parent=plan_bound.path.parent.parent)
        verify_bound_inputs(bound)
        plan_bound.verify_path()
        seal_output_directory(out)
        validate_canonical_plan_layout(plan_bound.path, source_contract["git_commit"])
        print(
            "[carry-motor] sealed feature shard {}/{} rows={} sha256={}".format(
                args.shard_index,
                CANONICAL_FEATURE_SHARDS,
                len(indices),
                sha256_file(out),
            ),
            flush=True,
        )
    finally:
        if plan_bound is not None:
            plan_bound.close()
        close_bound_inputs(bound)


def _fit_from_shards(args):
    if args.allow_non_cuda:
        raise SystemExit("canonical fit does not accept --allow-non-cuda")
    validate_canonical_feature_fit_args(args, canonical=True)
    bound, frozen = bind_frozen_inputs(
        args.ckpt, args.episodes, args.tokenizer, args.cycle
    )
    shard_bounds = []
    plan_bound = None
    try:
        source_contract = validate_source_contract(args, frozen, canonical=True)
        confirmation_commitment = bind_confirmation_commitment(
            args, bound, frozen, source_contract
        )
        device = require_canonical_cuda_runtime()
        torch.manual_seed(FIT_SEED)
        if device == "cuda":
            torch.cuda.manual_seed_all(FIT_SEED)
        tokenizer = Tokenizer.from_str(bound["tokenizer"].text())
        bound["checkpoint"].handle.seek(0)
        checkpoint = torch.load(bound["checkpoint"].handle, map_location="cpu")
        bound["checkpoint"].handle.seek(0)
        cfg = GPTConfig(**checkpoint["cfg"])
        if int(cfg.n_loop) != 1:
            raise ValueError("carry motor requires n_loop=1")
        rows, board = generate_fit_rows(
            tokenizer, bound["episodes"].text(), FIT_SEED, FIT_QUOTA
        )
        plan_bound, plan = _load_validated_plan(
            args,
            bound,
            frozen,
            source_contract,
            confirmation_commitment,
            checkpoint,
            rows,
            board,
            tokenizer,
        )
        board["rows_sha256"] = stable_json_sha256(rows)
        control_labels, control = permuted_control_labels(rows)
        if control != plan["fit_budget"]["control"]:
            raise RuntimeError("shuffled-label control differs from frozen plan")
        out = Path(plan["fit_artifact"])
        if out.parent.parent != plan_bound.path.parent or out.parent.is_symlink():
            raise ValueError("planned fit directory is invalid")
        source_hashes = {
            name.removeprefix("source:"): digest
            for name, digest in frozen.items()
            if name.startswith("source:")
        }
        expected_bindings = {
            "base_checkpoint_sha256": frozen["checkpoint"],
            "tokenizer_sha256": frozen["tokenizer"],
            "episodes_sha256": frozen["episodes"],
            "cycle_sha256": frozen["cycle"],
            "confirmation_commitment_sha256": frozen["confirmation_commitment"],
        }
        if out.exists():
            needs_seal = validate_existing_artifact_stage(out)
            existing = BoundInput(out)
            try:
                existing.handle.seek(0)
                existing_bundle = torch.load(existing.handle, map_location="cpu")
                existing.handle.seek(0)
                validate_motor_bundle(
                    existing_bundle,
                    expected_bindings,
                    source_hashes,
                    source_contract,
                    plan_bound.sha256,
                    plan,
                    rows,
                )
                verify_bound_inputs(bound)
                plan_bound.verify_path()
                existing.verify_path()
                if needs_seal:
                    seal_output_directory(out)
                validate_canonical_plan_layout(
                    plan_bound.path, source_contract["git_commit"]
                )
                print(
                    "[carry-motor] existing planned fit is valid sha256={}".format(
                        existing.sha256
                    ),
                    flush=True,
                )
                return
            finally:
                existing.close()
        require_empty_planned_directory(out)
        shard_bounds, shard_payloads = _bind_and_load_canonical_shards(
            plan["shards"], plan_bound.path.parent
        )
        shard_bindings = {
            **expected_bindings,
            "scientific_source_sha256": source_hashes,
        }
        features, feature_merge = merge_feature_shards(
            shard_payloads,
            rows,
            shard_bindings,
            source_contract,
            plan,
            plan_bound.sha256,
        )
        for shard_bound in shard_bounds:
            shard_bound.verify_path()
        validate_canonical_plan_layout(plan_bound.path, source_contract["git_commit"])
        zero_id = token_for_digit(tokenizer, "x=", "0")[1]
        one_id = token_for_digit(tokenizer, "x=", "1")[1]
        if (features["zero_id"], features["one_id"]) != (zero_id, one_id):
            raise RuntimeError("merged feature carry-token identity mismatch")
        if features["hidden"].shape != (len(rows), int(cfg.d_model)):
            raise RuntimeError("merged hidden feature shape mismatch")
        if features["base01"].shape != (len(rows), 2):
            raise RuntimeError("merged carry-logit feature shape mismatch")
        expected_other_ids = torch.arange(int(cfg.vocab_size), dtype=torch.long)
        expected_other_ids = expected_other_ids[
            (expected_other_ids != zero_id) & (expected_other_ids != one_id)
        ]
        if not torch.equal(features["other_token_ids"], expected_other_ids):
            raise RuntimeError("merged other-token vocabulary identity mismatch")
        initial_state, initial_sha256 = initial_motor_state(cfg.d_model)
        if initial_sha256 != plan["fit_budget"]["initial_state_sha256"]:
            raise RuntimeError("initial motor state differs from frozen plan")
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
        treatment = treatment.to(device).eval()
        shuffled = shuffled.to(device).eval()
        fit_feature_evidence = canonical_fit_teacher_forced_evidence(
            features,
            rows,
            control_labels,
            treatment,
            shuffled,
            feature_merge["teacher_metric_feature_payload_sha256"],
            device,
        )
        treatment = treatment.cpu()
        shuffled = shuffled.cpu()
        verify_bound_inputs(bound)
        for shard_bound in shard_bounds:
            shard_bound.verify_path()
        bundle = {
            "audit": CANONICAL_FIT_AUDIT,
            "canonical": True,
            "plan_sha256": plan_bound.sha256,
            **expected_bindings,
            "scientific_source_sha256": source_hashes,
            "source_contract": source_contract,
            "checkpoint_step": checkpoint.get("step"),
            "d_model": int(cfg.d_model),
            "rank": RANK,
            "parameter_count": treatment.parameter_count(),
            "extract_batch": args.extract_batch,
            "feature_shard_merge": feature_merge,
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
            "fit_feature_metrics": fit_feature_evidence,
            "claim_boundary": CANONICAL_FIT_CLAIM_BOUNDARY,
        }
        validate_motor_bundle(
            bundle,
            expected_bindings,
            source_hashes,
            source_contract,
            plan_bound.sha256,
            plan,
            rows,
        )
        atomic_torch(out, bundle, staging_parent=plan_bound.path.parent.parent)
        verify_bound_inputs(bound)
        plan_bound.verify_path()
        for shard_bound in shard_bounds:
            shard_bound.verify_path()
        seal_output_directory(out)
        validate_canonical_plan_layout(plan_bound.path, source_contract["git_commit"])
        print(
            json.dumps(
                {
                    name: arm["summary"]
                    for name, arm in bundle["fit_feature_metrics"]["arms"].items()
                },
                sort_keys=True,
            ),
            flush=True,
        )
    finally:
        close_bound_inputs(bound)
        for shard_bound in shard_bounds:
            shard_bound.close()
        if plan_bound is not None:
            plan_bound.close()


def _train(args):
    canonical = not args.allow_non_cuda
    validate_canonical_train_args(args, canonical)
    if canonical:
        raise SystemExit(
            "canonical monolithic extraction is disabled; use extract then fit-shards"
        )
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
        treatment = treatment.to(device).eval()
        shuffled = shuffled.to(device).eval()
        fit_feature_evidence = fit_teacher_forced_evidence(
            features,
            rows,
            control_labels,
            treatment,
            shuffled,
            teacher_metric_feature_payload_sha256(features),
            device,
            "noncanonical_singleton_apply_motor_logits_v1",
        )
        treatment = treatment.cpu()
        shuffled = shuffled.cpu()
        verify_bound_inputs(bound)
        source_hashes = {
            name.removeprefix("source:"): digest
            for name, digest in frozen.items()
            if name.startswith("source:")
        }
        bundle = {
            "audit": "causal_carry_motor_fit_development_v1",
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
            "fit_feature_metrics": fit_feature_evidence,
            "claim_boundary": (
                "A fit establishes no reasoning result. The motor can alter only carry token logits "
                "at a grammar-defined DWS site; heldout autonomous and confirmation gates remain required."
            ),
        }
        atomic_torch(out, bundle)
        report = {
            key: value
            for key, value in bundle.items()
            if key not in {"treatment", "shuffled", "fit_feature_metrics"}
        }
        report["fit_feature_metrics"] = {
            "source_feature_payload_sha256": fit_feature_evidence[
                "source_feature_payload_sha256"
            ],
            "row_identity_sha256": fit_feature_evidence["row_identity_sha256"],
            "rows": len(fit_feature_evidence["row_identities"]),
            "summaries": {
                name: arm["summary"]
                for name, arm in fit_feature_evidence["arms"].items()
            },
            "evidence_location": str(out),
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
    if args.allow_non_cuda:
        raise SystemExit("planned evaluation does not accept --allow-non-cuda")
    canonical = True
    validate_canonical_eval_args(args, canonical)
    bound, frozen = bind_frozen_inputs(
        args.ckpt, args.episodes, args.tokenizer, args.cycle
    )
    motor_bound = None
    plan_bound = None
    try:
        source_contract = validate_source_contract(args, frozen, canonical)
        confirmation_commitment = bind_confirmation_commitment(
            args, bound, frozen, source_contract
        )
        device = require_canonical_cuda_runtime()
        checkpoint, model = load_model(bound["checkpoint"], device)
        tokenizer = Tokenizer.from_str(bound["tokenizer"].text())
        fit_rows, fit_board = generate_fit_rows(
            tokenizer, bound["episodes"].text(), FIT_SEED, FIT_QUOTA
        )
        plan_bound, plan = _load_validated_plan(
            args,
            bound,
            frozen,
            source_contract,
            confirmation_commitment,
            checkpoint,
            fit_rows,
            fit_board,
            tokenizer,
        )
        fit_path = Path(plan["fit_artifact"])
        if fit_path != plan_bound.path.parent / "fit" / "motor.pt":
            raise ValueError("planned motor artifact path mismatch")
        require_sealed_artifact(fit_path)
        motor_bound = BoundInput(fit_path)
        if motor_bound.path != fit_path:
            raise ValueError("planned motor artifact path mismatch")
        motor_bound.handle.seek(0)
        bundle = torch.load(motor_bound.handle, map_location="cpu")
        motor_bound.handle.seek(0)
        expected_bindings = {
            "base_checkpoint_sha256": frozen["checkpoint"],
            "tokenizer_sha256": frozen["tokenizer"],
            "episodes_sha256": frozen["episodes"],
            "cycle_sha256": frozen["cycle"],
            "confirmation_commitment_sha256": frozen["confirmation_commitment"],
        }
        current_sources = {
            name.removeprefix("source:"): digest
            for name, digest in frozen.items()
            if name.startswith("source:")
        }
        validate_motor_bundle(
            bundle,
            expected_bindings,
            current_sources,
            source_contract,
            plan_bound.sha256,
            plan,
            fit_rows,
        )
        episodes_text = bound["episodes"].text()
        selection, development_selection = canonical_development_selection(
            episodes_text
        )
        if development_selection != plan["development_selection"]:
            raise ValueError("development selection differs from frozen plan")
        episodes = [entry["episode"] for entry in selection]
        all_episode_map = {
            json.loads(raw)["id"]: json.loads(raw)
            for raw in episodes_text.splitlines()
            if raw.strip()
        }
        dev_rows = development_feature_rows(selection, tokenizer)
        direct_cases = fresh_direct_cases()
        expected_cycle_contract = cycle_validation_contract(
            bound["cycle"].text(), all_episode_map
        )
        out = Path(plan["development_eval_artifact"])
        if out.parent.parent != plan_bound.path.parent or out.parent.is_symlink():
            raise ValueError("planned development evaluation directory is invalid")
        if out.exists():
            needs_seal = validate_existing_artifact_stage(out)
            existing = BoundInput(out)
            try:
                existing_result = _load_exact_json(
                    existing.text(), "development evaluation"
                )
                validate_development_eval_result(
                    existing_result,
                    frozen,
                    source_contract,
                    checkpoint.get("step"),
                    motor_bound.path,
                    motor_bound.sha256,
                    plan_bound.sha256,
                    dev_rows,
                    tokenizer,
                    int(model.cfg.seq_len),
                    development_selection,
                    episodes,
                    expected_cycle_contract,
                    direct_cases,
                )
                verify_bound_inputs(bound)
                motor_bound.verify_path()
                plan_bound.verify_path()
                existing.verify_path()
                if needs_seal:
                    seal_output_directory(out)
                validate_canonical_plan_layout(
                    plan_bound.path, source_contract["git_commit"]
                )
                print(
                    "[carry-motor] existing planned evaluation is valid sha256={}".format(
                        existing.sha256
                    ),
                    flush=True,
                )
                return
            finally:
                existing.close()
        require_empty_planned_directory(out)
        treatment = CarryMotor(bundle["d_model"], bundle["rank"]).to(device).eval()
        treatment.load_state_dict(bundle["treatment"])
        shuffled = CarryMotor(bundle["d_model"], bundle["rank"]).to(device).eval()
        shuffled.load_state_dict(bundle["shuffled"])
        dead = CarryMotor(bundle["d_model"], bundle["rank"]).to(device).eval()
        zero_id, one_id = int(bundle["zero_id"]), int(bundle["one_id"])
        dev_features = extract_frozen_features(
            model,
            dev_rows,
            zero_id,
            one_id,
            device,
            batch_size=args.extract_batch,
            store_other_logits=False,
        )
        if (
            bundle.get("deployment_logit_dtype")
            != dev_features["deployment_logit_dtype"]
        ):
            raise RuntimeError("fit/evaluation deployment logit dtype mismatch")
        feature_results = {
            "base": canonical_teacher_forced_metric_evidence(
                dev_features,
                dev_rows,
                None,
                "base",
                device,
            ),
            "treatment": canonical_teacher_forced_metric_evidence(
                dev_features,
                dev_rows,
                treatment,
                "treatment",
                device,
            ),
            "shuffled": canonical_teacher_forced_metric_evidence(
                dev_features,
                dev_rows,
                shuffled,
                "shuffled",
                device,
            ),
            "dead": canonical_teacher_forced_metric_evidence(
                dev_features,
                dev_rows,
                dead,
                "dead",
                device,
            ),
        }
        if without_motor_fire_accounting(
            feature_results["dead"]
        ) != without_motor_fire_accounting(feature_results["base"]):
            raise RuntimeError("dead motor does not collapse to base feature metrics")
        arms = {
            "base": None,
            "dead": dead,
            "treatment": treatment,
            "shuffled": shuffled,
        }
        results = {}
        for arm, motor in arms.items():
            episode_accounting = []
            episode_evidence = []
            for index, selected in enumerate(selection, 1):
                episode = selected["episode"]
                identity = selected["identity"]
                calls = []

                def ask(prompt):
                    response, site_count, fire_count, generation = motor_generate(
                        model, motor, tokenizer, prompt, device, max_new=args.max_new
                    )
                    calls.append(
                        _raw_call_record(
                            len(calls),
                            "transition"
                            if dws_prompt_state(prompt) is not None
                            else "final",
                            prompt,
                            response,
                            site_count,
                            fire_count,
                            generation,
                        )
                    )
                    return response

                rollout_episode(episode, ask, prompt_style=episode["prompt_style"])
                evidence = _episode_evidence_record(identity, calls)
                accounting = _derive_episode_accounting_from_evidence(
                    evidence,
                    episode,
                    identity,
                    arm,
                    tokenizer,
                    int(model.cfg.seq_len),
                    args.max_new,
                    "development {} generated episode {}".format(arm, index - 1),
                )
                episode_evidence.append(evidence)
                episode_accounting.append(accounting)
                if index % 50 == 0:
                    print(
                        "[carry-motor] arm={} {}/{}".format(arm, index, len(episodes)),
                        flush=True,
                    )
            results[arm] = {
                "by_regime": _aggregate_episode_accounting(episode_accounting),
                "episode_accounting": episode_accounting,
                "episode_evidence": episode_evidence,
                "transcripts": episode_evidence[:15],
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
            for case in direct_cases:
                sites = fires = 0
                calls = []

                def ask(prompt):
                    nonlocal sites, fires
                    response, site_count, fire_count, generation = motor_generate(
                        model, motor, tokenizer, prompt, device, max_new=args.max_new
                    )
                    sites += site_count
                    fires += fire_count
                    kind = (
                        "transition"
                        if dws_prompt_state(prompt) is not None
                        else "review"
                        if str(prompt).startswith("Review this proposed")
                        else "final"
                    )
                    calls.append(
                        _raw_call_record(
                            len(calls),
                            kind,
                            prompt,
                            response,
                            site_count,
                            fire_count,
                            generation,
                        )
                    )
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
                        "calls": calls,
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
        preservation = []
        for prompt in NON_DWS_PRESERVATION_PROMPTS:
            base_response, base_sites, base_fires, base_generation = motor_generate(
                model,
                None,
                tokenizer,
                prompt,
                device,
                max_new=48,
                retain_full_logits=True,
            )
            motor_response, motor_sites, motor_fires, motor_generation = motor_generate(
                model,
                treatment,
                tokenizer,
                prompt,
                device,
                max_new=48,
                retain_full_logits=True,
            )
            token_ids_identical = (
                base_generation["token_ids"] == motor_generation["token_ids"]
            )
            full_logits_identical = [
                boundary["full_logits"] for boundary in base_generation["boundaries"]
            ] == [
                boundary["full_logits"] for boundary in motor_generation["boundaries"]
            ]
            preservation.append(
                {
                    "prompt": prompt,
                    "base_response": base_response,
                    "motor_response": motor_response,
                    "token_ids_identical": token_ids_identical,
                    "full_logits_identical": full_logits_identical,
                    "exact_identity": token_ids_identical and full_logits_identical,
                    "base_sites": base_sites,
                    "base_fires": base_fires,
                    "motor_sites": motor_sites,
                    "motor_fires": motor_fires,
                    "base_call": _raw_call_record(
                        0,
                        "preservation",
                        prompt,
                        base_response,
                        base_sites,
                        base_fires,
                        base_generation,
                    ),
                    "motor_call": _raw_call_record(
                        0,
                        "preservation",
                        prompt,
                        motor_response,
                        motor_sites,
                        motor_fires,
                        motor_generation,
                    ),
                }
            )
        if not all(
            row["exact_identity"]
            and row["base_sites"] == 0
            and row["base_fires"] == 0
            and row["motor_sites"] == 0
            and row["motor_fires"] == 0
            for row in preservation
        ):
            raise RuntimeError("non-DWS preservation identity failed")
        verify_bound_inputs(bound)
        motor_bound.verify_path()
        result = {
            "audit": CANONICAL_EVAL_AUDIT,
            "checkpoint_step": checkpoint.get("step"),
            "frozen_sha256": frozen,
            "source_contract": source_contract,
            "motor": str(motor_bound.path),
            "motor_sha256": motor_bound.sha256,
            "plan_sha256": plan_bound.sha256,
            "development_selection": development_selection,
            "max_new": args.max_new,
            "extract_batch": args.extract_batch,
            "teacher_forced_carry": feature_results,
            "results": results,
            "fresh_direct": direct_results,
            "non_dws_preservation": preservation,
            "claim_boundary": CANONICAL_EVAL_CLAIM_BOUNDARY,
        }
        validate_development_eval_result(
            result,
            frozen,
            source_contract,
            checkpoint.get("step"),
            motor_bound.path,
            motor_bound.sha256,
            plan_bound.sha256,
            dev_rows,
            tokenizer,
            int(model.cfg.seq_len),
            development_selection,
            episodes,
            expected_cycle_contract,
            direct_cases,
        )
        atomic_json(out, result, staging_parent=plan_bound.path.parent.parent)
        verify_bound_inputs(bound)
        motor_bound.verify_path()
        plan_bound.verify_path()
        seal_output_directory(out)
        validate_canonical_plan_layout(plan_bound.path, source_contract["git_commit"])
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
        if motor_bound is not None:
            motor_bound.close()
        if plan_bound is not None:
            plan_bound.close()


def _confirmation_eval(args):
    if args.allow_non_cuda:
        raise SystemExit("confirmation evaluation does not accept --allow-non-cuda")
    validate_canonical_eval_args(args, canonical=True)
    if not args.confirmation_secret_file:
        raise SystemExit("confirmation evaluation requires the revealed secret file")
    bound, frozen = bind_frozen_inputs(
        args.ckpt, args.episodes, args.tokenizer, args.cycle
    )
    secret_bound = motor_bound = plan_bound = development_bound = None
    try:
        source_contract = validate_source_contract(args, frozen, canonical=True)
        confirmation_commitment = bind_confirmation_commitment(
            args, bound, frozen, source_contract
        )
        tokenizer = Tokenizer.from_str(bound["tokenizer"].text())
        bound["checkpoint"].handle.seek(0)
        checkpoint = torch.load(bound["checkpoint"].handle, map_location="cpu")
        bound["checkpoint"].handle.seek(0)
        fit_rows, fit_board = generate_fit_rows(
            tokenizer, bound["episodes"].text(), FIT_SEED, FIT_QUOTA
        )
        plan_bound, plan = _load_validated_plan(
            args,
            bound,
            frozen,
            source_contract,
            confirmation_commitment,
            checkpoint,
            fit_rows,
            fit_board,
            tokenizer,
        )
        secret_bound = bind_revealed_confirmation_secret(args.confirmation_secret_file)
        confirmation_board = generate_confirmation_board(
            secret_bound.bytes(),
            bound,
            frozen,
            confirmation_commitment,
            plan_bound.path,
            plan_bound.sha256,
            plan,
        )
        confirmation_rows = confirmation_feature_rows(confirmation_board, tokenizer)
        verify_bound_inputs(bound)
        secret_bound.verify_path()

        checkpoint_step = checkpoint.get("step")
        del checkpoint
        device = require_canonical_cuda_runtime()
        checkpoint, model = load_model(bound["checkpoint"], device)
        if checkpoint.get("step") != checkpoint_step:
            raise RuntimeError("confirmation checkpoint changed after plan validation")

        fit_path = Path(plan["fit_artifact"])
        require_sealed_artifact(fit_path)
        motor_bound = BoundInput(fit_path)
        motor_bound.handle.seek(0)
        bundle = torch.load(motor_bound.handle, map_location="cpu")
        motor_bound.handle.seek(0)
        expected_bindings = {
            "base_checkpoint_sha256": frozen["checkpoint"],
            "tokenizer_sha256": frozen["tokenizer"],
            "episodes_sha256": frozen["episodes"],
            "cycle_sha256": frozen["cycle"],
            "confirmation_commitment_sha256": frozen["confirmation_commitment"],
        }
        current_sources = {
            name.removeprefix("source:"): digest
            for name, digest in frozen.items()
            if name.startswith("source:")
        }
        validate_motor_bundle(
            bundle,
            expected_bindings,
            current_sources,
            source_contract,
            plan_bound.sha256,
            plan,
            fit_rows,
        )

        development_path = Path(plan["development_eval_artifact"])
        require_sealed_artifact(development_path)
        development_bound = BoundInput(development_path)
        development_result = _load_exact_json(
            development_bound.text(), "development evaluation"
        )
        episodes_text = bound["episodes"].text()
        selection, development_selection = canonical_development_selection(
            episodes_text
        )
        episodes = [entry["episode"] for entry in selection]
        all_episode_map = {
            json.loads(raw)["id"]: json.loads(raw)
            for raw in episodes_text.splitlines()
            if raw.strip()
        }
        validate_development_eval_result(
            development_result,
            frozen,
            source_contract,
            checkpoint_step,
            motor_bound.path,
            motor_bound.sha256,
            plan_bound.sha256,
            development_feature_rows(selection, tokenizer),
            tokenizer,
            int(model.cfg.seq_len),
            development_selection,
            episodes,
            cycle_validation_contract(bound["cycle"].text(), all_episode_map),
            fresh_direct_cases(),
        )
        development_evaluation = {
            "path": str(development_bound.path),
            "sha256": development_bound.sha256,
        }

        out = Path(plan["confirmation_eval_artifact"])
        if out.parent.parent != plan_bound.path.parent or out.parent.is_symlink():
            raise ValueError("planned confirmation evaluation directory is invalid")
        if out.exists():
            needs_seal = validate_existing_artifact_stage(out)
            existing = BoundInput(out)
            try:
                existing_result = _load_exact_json(
                    existing.text(), "confirmation evaluation"
                )
                validate_confirmation_eval_result(
                    existing_result,
                    frozen,
                    source_contract,
                    checkpoint_step,
                    motor_bound.path,
                    motor_bound.sha256,
                    confirmation_board,
                    confirmation_rows,
                    development_evaluation,
                    tokenizer,
                    int(model.cfg.seq_len),
                )
                verify_bound_inputs(bound)
                secret_bound.verify_path()
                motor_bound.verify_path()
                plan_bound.verify_path()
                development_bound.verify_path()
                existing.verify_path()
                if needs_seal:
                    seal_output_directory(out)
                validate_canonical_plan_layout(
                    plan_bound.path, source_contract["git_commit"]
                )
                print(
                    "[carry-motor] existing confirmation evaluation is valid sha256={}".format(
                        existing.sha256
                    ),
                    flush=True,
                )
                return
            finally:
                existing.close()
        require_empty_planned_directory(out)

        treatment = CarryMotor(bundle["d_model"], bundle["rank"]).to(device).eval()
        treatment.load_state_dict(bundle["treatment"])
        shuffled = CarryMotor(bundle["d_model"], bundle["rank"]).to(device).eval()
        shuffled.load_state_dict(bundle["shuffled"])
        dead = CarryMotor(bundle["d_model"], bundle["rank"]).to(device).eval()
        zero_id, one_id = int(bundle["zero_id"]), int(bundle["one_id"])
        features = extract_frozen_features(
            model,
            confirmation_rows,
            zero_id,
            one_id,
            device,
            batch_size=args.extract_batch,
            store_other_logits=False,
        )
        if bundle["deployment_logit_dtype"] != features["deployment_logit_dtype"]:
            raise RuntimeError("fit/confirmation deployment logit dtype mismatch")
        teacher = {
            "base": canonical_teacher_forced_metric_evidence(
                features,
                confirmation_rows,
                None,
                "base",
                device,
            ),
            "treatment": canonical_teacher_forced_metric_evidence(
                features,
                confirmation_rows,
                treatment,
                "treatment",
                device,
            ),
            "shuffled": canonical_teacher_forced_metric_evidence(
                features,
                confirmation_rows,
                shuffled,
                "shuffled",
                device,
            ),
            "dead": canonical_teacher_forced_metric_evidence(
                features,
                confirmation_rows,
                dead,
                "dead",
                device,
            ),
        }
        arms = {
            "base": None,
            "dead": dead,
            "treatment": treatment,
            "shuffled": shuffled,
        }
        one_step = {}
        autonomous = {}
        for arm, motor in arms.items():
            one_step_rows = []
            episode_accounting = []
            episode_evidence = []
            for index, board_row in enumerate(confirmation_board["rows"]):
                response, sites, fires, generation = motor_generate(
                    model,
                    motor,
                    tokenizer,
                    board_row["prompt"],
                    device,
                    max_new=args.max_new,
                )
                one_step_rows.append(
                    {
                        "index": index,
                        "prompt_sha256": board_row["prompt_sha256"],
                        "expected_state": board_row["expected_state"],
                        "call": _raw_call_record(
                            0,
                            "transition",
                            board_row["prompt"],
                            response,
                            sites,
                            fires,
                            generation,
                        ),
                    }
                )
                calls = []

                def ask(prompt):
                    reply, call_sites, call_fires, call_generation = motor_generate(
                        model,
                        motor,
                        tokenizer,
                        prompt,
                        device,
                        max_new=args.max_new,
                    )
                    calls.append(
                        _raw_call_record(
                            len(calls),
                            "transition"
                            if dws_prompt_state(prompt) is not None
                            else "final",
                            prompt,
                            reply,
                            call_sites,
                            call_fires,
                            call_generation,
                        )
                    )
                    return reply

                episode = board_row["episode"]
                rollout_episode(episode, ask, prompt_style=episode["prompt_style"])
                identity = _confirmation_episode_identity(board_row)
                evidence = _episode_evidence_record(identity, calls)
                accounting = _derive_episode_accounting_from_evidence(
                    evidence,
                    episode,
                    identity,
                    arm,
                    tokenizer,
                    int(model.cfg.seq_len),
                    args.max_new,
                    "confirmation {} generated episode {}".format(arm, index),
                )
                episode_evidence.append(evidence)
                episode_accounting.append(accounting)
                if (index + 1) % 64 == 0:
                    print(
                        "[carry-motor] confirmation arm={} {}/256".format(
                            arm, index + 1
                        ),
                        flush=True,
                    )
            one_step[arm] = {
                "correct": sum(
                    int(
                        parse_state(row["call"]["response"])
                        == parse_state(row["expected_state"])
                    )
                    for row in one_step_rows
                ),
                "rows": one_step_rows,
            }
            autonomous[arm] = {
                "by_regime": _aggregate_episode_accounting(episode_accounting),
                "episode_accounting": episode_accounting,
                "episode_evidence": episode_evidence,
            }
        for section in (teacher, one_step, autonomous):
            if without_motor_fire_accounting(
                section["dead"]
            ) != without_motor_fire_accounting(section["base"]):
                raise RuntimeError("confirmation dead motor diverged from base")

        verify_bound_inputs(bound)
        secret_bound.verify_path()
        motor_bound.verify_path()
        plan_bound.verify_path()
        development_bound.verify_path()
        result = {
            "audit": CANONICAL_CONFIRMATION_EVAL_AUDIT,
            "checkpoint_step": checkpoint_step,
            "frozen_sha256": frozen,
            "source_contract": source_contract,
            "motor": str(motor_bound.path),
            "motor_sha256": motor_bound.sha256,
            "confirmation_secret_sha256": confirmation_board["secret_sha256"],
            "frozen_inputs": confirmation_board["frozen_inputs"],
            "confirmation_exclusion_contract": confirmation_board["exclusion_contract"],
            "confirmation_commitment": confirmation_board["confirmation_commitment"],
            "confirmation_rows_sha256": confirmation_board["rows_sha256"],
            "confirmation_board": confirmation_board,
            "plan": confirmation_board["plan"],
            "development_evaluation": development_evaluation,
            "max_new": args.max_new,
            "extract_batch": args.extract_batch,
            "teacher_forced_carry": teacher,
            "one_step": one_step,
            "autonomous": autonomous,
            "claim_boundary": CANONICAL_CONFIRMATION_EVAL_CLAIM_BOUNDARY,
        }
        validate_confirmation_eval_result(
            result,
            frozen,
            source_contract,
            checkpoint_step,
            motor_bound.path,
            motor_bound.sha256,
            confirmation_board,
            confirmation_rows,
            development_evaluation,
            tokenizer,
            int(model.cfg.seq_len),
        )
        atomic_json(out, result, staging_parent=plan_bound.path.parent.parent)
        verify_bound_inputs(bound)
        secret_bound.verify_path()
        motor_bound.verify_path()
        plan_bound.verify_path()
        development_bound.verify_path()
        seal_output_directory(out)
        validate_canonical_plan_layout(plan_bound.path, source_contract["git_commit"])
        print(
            json.dumps(
                {
                    "confirmation_rows_sha256": confirmation_board["rows_sha256"],
                    "teacher_forced": {
                        arm: report["summary"] for arm, report in teacher.items()
                    },
                    "one_step": {
                        arm: report["correct"] for arm, report in one_step.items()
                    },
                    "autonomous": {
                        arm: report["by_regime"] for arm, report in autonomous.items()
                    },
                },
                sort_keys=True,
            ),
            flush=True,
        )
    finally:
        close_bound_inputs(bound)
        for item in (secret_bound, motor_bound, plan_bound, development_bound):
            if item is not None:
                item.close()


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
    common.add_argument("--confirmation-commitment", default="")
    common.add_argument("--confirmation-commitment-sha256", default="")
    common.add_argument("--allow-non-cuda", action="store_true")
    confirmation_preflight = subparsers.add_parser(
        "validate-confirmation", parents=[common]
    )
    confirmation_preflight.set_defaults(func=_validate_confirmation_preflight)
    plan = subparsers.add_parser("plan", parents=[common])
    plan.add_argument("--plan-root", required=True)
    plan.set_defaults(func=_plan)
    extract = subparsers.add_parser("extract", parents=[common])
    extract.add_argument("--plan", required=True)
    extract.add_argument("--plan-sha256", required=True)
    extract.add_argument("--quota", type=int, default=FIT_QUOTA)
    extract.add_argument("--extract-batch", type=int, default=CANONICAL_EXTRACT_BATCH)
    extract.add_argument("--shard-index", type=int, required=True)
    extract.add_argument("--shard-count", type=int, default=CANONICAL_FEATURE_SHARDS)
    extract.set_defaults(func=_extract_shard)
    fit_shards = subparsers.add_parser("fit-shards", parents=[common])
    fit_shards.add_argument("--plan", required=True)
    fit_shards.add_argument("--plan-sha256", required=True)
    fit_shards.add_argument("--quota", type=int, default=FIT_QUOTA)
    fit_shards.add_argument(
        "--extract-batch", type=int, default=CANONICAL_EXTRACT_BATCH
    )
    fit_shards.add_argument("--updates", type=int, default=CANONICAL_UPDATES)
    fit_shards.add_argument("--batch-size", type=int, default=CANONICAL_BATCH)
    fit_shards.add_argument("--lr", type=float, default=CANONICAL_LR)
    fit_shards.add_argument(
        "--weight-decay", type=float, default=CANONICAL_WEIGHT_DECAY
    )
    fit_shards.set_defaults(func=_fit_from_shards)
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
    evaluate.add_argument("--plan", required=True)
    evaluate.add_argument("--plan-sha256", required=True)
    evaluate.add_argument("--max-new", type=int, default=CANONICAL_MAX_NEW)
    evaluate.add_argument("--extract-batch", type=int, default=CANONICAL_EXTRACT_BATCH)
    evaluate.set_defaults(func=_eval)
    confirmation_evaluate = subparsers.add_parser("confirmation-eval", parents=[common])
    confirmation_evaluate.add_argument("--plan", required=True)
    confirmation_evaluate.add_argument("--plan-sha256", required=True)
    confirmation_evaluate.add_argument("--confirmation-secret-file", required=True)
    confirmation_evaluate.add_argument("--max-new", type=int, default=CANONICAL_MAX_NEW)
    confirmation_evaluate.add_argument(
        "--extract-batch", type=int, default=CANONICAL_EXTRACT_BATCH
    )
    confirmation_evaluate.set_defaults(func=_confirmation_eval)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
