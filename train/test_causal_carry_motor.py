#!/usr/bin/env python3
"""CPU contracts for the grammar-gated causal carry motor."""

from __future__ import annotations

import collections
import copy
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train"))
FROZEN_EPISODES_PATH = ROOT / "artifacts/evals/digitwise_recurrent_v2_heldout.jsonl"
FROZEN_CYCLE_PATH = ROOT / "artifacts/evals/drs_causal_cycle_post_drs_r3.json"
FROZEN_TOKENIZER_PATH = ROOT / "artifacts/shohin-tok-32k.json"

from causal_carry_motor import (  # noqa: E402
    CarryMotor,
    CarryRouter,
    BoundInput,
    CANONICAL_BATCH,
    CANONICAL_CHECKPOINT_STEP,
    CANONICAL_CONFIRMATION_CLAIM_BOUNDARY,
    CANONICAL_CONFIRMATION_COMMITMENT_AUDIT,
    CANONICAL_CONFIRMATION_EXCLUSION_SCHEMA,
    CANONICAL_CONFIRMATION_GENERATOR_ENTRYPOINT,
    CANONICAL_CONFIRMATION_GENERATOR_SCHEMA,
    CANONICAL_CONFIRMATION_GENERATOR_SOURCES,
    CANONICAL_CONFIRMATION_TIMING,
    CANONICAL_CYCLE_CASES,
    CANONICAL_DEVELOPMENT_EPISODES,
    CANONICAL_EVAL_AUDIT,
    CANONICAL_EVAL_CLAIM_BOUNDARY,
    CANONICAL_EXTRACT_BATCH,
    CANONICAL_FIT_CLAIM_BOUNDARY,
    CANONICAL_FIT_AUDIT,
    CANONICAL_FEATURE_SHARDS,
    CANONICAL_LR,
    CANONICAL_MAX_NEW,
    CANONICAL_PLAN_AUDIT,
    CANONICAL_SHARD_AUDIT,
    CANONICAL_SHARD_CLAIM_BOUNDARY,
    CANONICAL_TEACHER_SCORING_CONTRACT,
    CANONICAL_UPDATES,
    CANONICAL_WEIGHT_DECAY,
    FIT_QUOTA,
    EXPECTED_CONFIRMATION_EXCLUSION_SHA256,
    EXPECTED_CONFIRMATION_OPERAND_EXCLUSIONS,
    EXPECTED_CONFIRMATION_PROMPT_EXCLUSIONS,
    EXPECTED_CHECKPOINT_SHA256,
    EXPECTED_DEVELOPMENT_SELECTION_SHA256,
    EXPECTED_TOKENIZER_SHA256,
    LINEAR_DIAGNOSTIC_CLAIM_BOUNDARY,
    NON_DWS_PRESERVATION_PROMPTS,
    SCIENTIFIC_SOURCE_PATHS,
    apply_microstep,
    apply_motor_logits,
    atomic_json,
    atomic_torch,
    bind_confirmation_commitment,
    confirmation_exclusion_contract,
    confirmation_feature_rows,
    canonical_development_selection,
    canonical_confirmation_commitment_path,
    canonical_plan_root,
    canonical_state,
    dws_prompt_state,
    evaluate_direct_case,
    extract_frozen_features,
    feature_payload_sha256,
    feature_sentinel_indices,
    feature_shard_indices,
    feature_metrics,
    fit_teacher_forced_evidence,
    fresh_direct_cases,
    full_vocab_motor_loss,
    full_logit_tensor_identity,
    generate_confirmation_board,
    generate_fit_rows,
    initial_motor_state,
    initial_state,
    is_carry_site,
    microstep_prompt,
    merge_feature_shards,
    parse_state,
    permuted_control_labels,
    prepare_output,
    require_empty_planned_directory,
    require_canonical_confirmation_commitment,
    require_canonical_shard_input,
    require_recoverable_artifact,
    require_sealed_artifact,
    require_canonical_cuda_runtime,
    seal_output_directory,
    source_manifest_sha256,
    stable_json_sha256,
    state_answer,
    teacher_forced_metric_evidence,
    teacher_metric_feature_payload_sha256,
    row_identity_sha256,
    rollout_episode,
    tensor_state_sha256,
    validate_artifact_receipt,
    validate_canonical_eval_args,
    validate_canonical_confirmation_args,
    validate_canonical_extract_args,
    validate_canonical_feature_fit_args,
    validate_canonical_train_args,
    validate_canonical_plan_layout,
    validate_confirmation_board_binding,
    validate_development_eval_result,
    validate_development_selection_contract,
    validate_motor_bundle,
    validate_plan_confirmation_binding,
    validate_existing_artifact_stage,
    validate_source_contract,
    _aggregate_episode_accounting,
    _aggregate_teacher_rows,
    _bind_and_load_canonical_shards,
    _derive_episode_accounting_from_evidence,
    _derive_top1_tensor_evidence,
    _eval,
    _confirmation_eval,
    _extract_shard,
    _expected_plan,
    _fit_from_shards,
    _load_validated_plan,
    _validate_confirmation_preflight,
    _plan,
    _validate_preservation_trace_identity,
    _batch_schedule,
    _episode_evidence_record,
    _raw_call_record,
    _singleton_deployment_adjusted_carry_logits,
    _tensor_metric_summary,
    _validate_generation_evidence,
    _validate_motor_bundle_against_replayed_features,
    confirmation_generator_source_contract,
)


def test_router_is_exact_and_response_local():
    state = initial_state("add", 379, 215, 4)
    prompt = microstep_prompt(state, style="core")
    assert is_carry_site(prompt, "dws:op=add;w=4;p=1;c=")
    assert not is_carry_site("Question: add numbers\nAnswer:", "dws:op=add;w=4;p=1;c=")
    assert not is_carry_site(prompt, "dws:op=add;w=04;p=1;c=")
    assert not is_carry_site(prompt, "dws:op=add;w=4;p=999;c=")
    assert not is_carry_site(prompt, "dws:op=sub;w=4;p=1;c=")
    assert not is_carry_site(prompt, "dws:op=add;w=4;p=1;c=0")
    assert not is_carry_site(prompt, "dws:op=add;w=4;p=1;c=\n")


def test_gate_off_is_bit_exact_and_gate_on_changes_only_two_tokens():
    torch.manual_seed(3)
    motor = CarryMotor(6, rank=2)
    with torch.no_grad():
        motor.up.weight.fill_(0.25)
        motor.up.bias.copy_(torch.tensor([1.0, -1.0]))
    logits = torch.randn(3, 11)
    hidden = torch.randn(3, 6)
    off = apply_motor_logits(logits, hidden, motor, 2, 7, False)
    assert torch.equal(off, logits)
    on = apply_motor_logits(logits, hidden, motor, 2, 7, True)
    unchanged = [index for index in range(logits.shape[-1]) if index not in (2, 7)]
    assert torch.equal(on[:, unchanged], logits[:, unchanged])
    assert not torch.equal(on[:, [2, 7]], logits[:, [2, 7]])
    dead = CarryMotor(6, rank=2)
    dead_on = apply_motor_logits(logits, hidden, dead, 2, 7, True)
    assert torch.equal(dead_on, logits)


def test_full_vocab_sufficient_loss_matches_dense_cross_entropy():
    torch.manual_seed(4)
    motor = CarryMotor(5, rank=3)
    hidden = torch.randn(9, 5)
    logits = torch.randn(9, 17)
    targets = torch.randint(0, 2, (9,))
    zero_id, one_id = 3, 12
    keep = torch.ones(logits.shape[-1], dtype=torch.bool)
    keep[[zero_id, one_id]] = False
    compact = full_vocab_motor_loss(
        hidden,
        logits[:, [zero_id, one_id]],
        torch.logsumexp(logits[:, keep], dim=-1),
        targets,
        motor,
    )
    dense = apply_motor_logits(logits, hidden, motor, zero_id, one_id, True)
    dense_targets = torch.where(targets == 0, zero_id, one_id)
    expected = torch.nn.functional.cross_entropy(dense, dense_targets)
    torch.testing.assert_close(compact, expected)


def test_bfloat16_loss_matches_deployed_quantized_dense_logits():
    torch.manual_seed(44)
    motor = CarryMotor(5, rank=3)
    hidden = torch.randn(9, 5)
    logits = torch.randn(9, 17).to(torch.bfloat16)
    targets = torch.randint(0, 2, (9,))
    zero_id, one_id = 3, 12
    keep = torch.ones(logits.shape[-1], dtype=torch.bool)
    keep[[zero_id, one_id]] = False
    compact = full_vocab_motor_loss(
        hidden,
        logits[:, [zero_id, one_id]],
        torch.logsumexp(logits[:, keep].float(), dim=-1),
        targets,
        motor,
    )
    dense = apply_motor_logits(logits, hidden, motor, zero_id, one_id, True)
    dense_targets = torch.where(targets == 0, zero_id, one_id)
    expected = torch.nn.functional.cross_entropy(dense.float(), dense_targets)
    torch.testing.assert_close(compact, expected)


def test_singleton_teacher_scoring_rejects_batch_sensitive_576d_shortcut():
    class BatchSensitiveMotor(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.call_shapes = []

        def forward(self, hidden):
            self.call_shapes.append(tuple(hidden.shape))
            delta = (
                torch.tensor([-1.0, 1.0], device=hidden.device)
                if len(hidden) == 1
                else torch.tensor([1.0, -1.0], device=hidden.device)
            )
            return delta.repeat(len(hidden), 1)

    hidden = torch.zeros((2, 576), dtype=torch.float32)
    base01 = torch.zeros((2, 2), dtype=torch.bfloat16)
    batched_motor = BatchSensitiveMotor()
    batched = base01 + batched_motor(hidden).to(base01.dtype)
    assert batched.argmax(dim=1).tolist() == [0, 0]
    assert batched_motor.call_shapes == [(2, 576)]

    singleton_motor = BatchSensitiveMotor()
    singleton = _singleton_deployment_adjusted_carry_logits(
        hidden, base01, singleton_motor, "cpu"
    )
    assert singleton.argmax(dim=1).tolist() == [1, 1]
    assert singleton_motor.call_shapes == [(1, 576), (1, 576)]


def test_feature_accuracy_is_global_not_two_token_only():
    features = {
        "hidden": torch.zeros(1, 4),
        "base01": torch.tensor([[2.0, 1.0]]),
        "other_lse": torch.tensor([10.0]),
        "other_max": torch.tensor([10.0]),
        "other_max_token_id": torch.tensor([2]),
        "other_logits": torch.tensor([[10.0, 0.0, -1.0]]),
        "other_token_ids": torch.tensor([2, 3, 4]),
        "zero_id": 0,
        "one_id": 1,
    }
    metrics = feature_metrics(features, [0])
    assert metrics["carry_pair_correct"] == 1
    assert metrics["global_correct"] == 0
    assert set(metrics) == {
        "carry_pair_correct",
        "carry_pair_accuracy",
        "global_correct",
        "global_accuracy",
        "rows",
        "prediction_ones",
    }


def test_feature_metrics_match_bfloat16_decode_and_token_id_ties():
    motor = CarryMotor(4, rank=2)
    with torch.no_grad():
        motor.up.bias.copy_(torch.tensor([0.001, 0.0]))
    features = {
        "hidden": torch.zeros(1, 4),
        "base01": torch.tensor([[1.0, 1.0]], dtype=torch.bfloat16),
        "other_lse": torch.tensor([-10.0]),
        "other_max": torch.tensor([1.0]),
        "other_max_token_id": torch.tensor([5]),
        "other_logits": torch.tensor([[1.0, -2.0]], dtype=torch.bfloat16),
        "other_token_ids": torch.tensor([5, 12]),
        "zero_id": 10,
        "one_id": 2,
    }
    metrics = feature_metrics(features, [0], motor)
    assert metrics["carry_pair_correct"] == 0
    assert metrics["global_correct"] == 0
    logits = torch.full((1, 13), -2.0, dtype=torch.bfloat16)
    logits[:, 10] = 1.0
    logits[:, 2] = 1.0
    logits[:, 5] = 1.0
    deployed = apply_motor_logits(logits, features["hidden"], motor, 10, 2, True)
    assert int(deployed.argmax(dim=-1)) == 2


def test_control_permutation_preserves_nuisance_counts():
    rows = []
    for operation in ("add", "sub"):
        for target in (0, 1):
            for index in range(40):
                rows.append(
                    {
                        "operation": operation,
                        "width": 4,
                        "position": 1,
                        "style": "core",
                        "current_carry": index % 2,
                        "target": target,
                    }
                )
    labels, report = permuted_control_labels(rows, seed=77)
    assert report["changed"] >= len(rows) // 3
    before, after = (
        collections.defaultdict(collections.Counter),
        collections.defaultdict(collections.Counter),
    )
    for row, label in zip(rows, labels):
        key = (
            row["operation"],
            row["width"],
            row["position"],
            row["style"],
            row["current_carry"],
        )
        before[key][row["target"]] += 1
        after[key][label] += 1
    assert before == after


def _fit_synthetic(hidden, targets, steps=500):
    torch.manual_seed(9)
    motor = CarryMotor(hidden.shape[1], rank=8)
    base01 = torch.zeros(len(hidden), 2)
    other = torch.full((len(hidden),), -10.0)
    optimizer = torch.optim.AdamW(motor.parameters(), lr=0.03, weight_decay=0.0)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = full_vocab_motor_loss(hidden, base01, other, targets, motor)
        loss.backward()
        optimizer.step()
    return motor


def test_rank8_motor_learns_nonlinear_signal_but_not_absent_signal():
    torch.manual_seed(12)
    hidden = torch.randn(1024, 12)
    targets = ((hidden[:, 0] * hidden[:, 1]) > 0).long()
    learned_motor = _fit_synthetic(hidden[:768], targets[:768])
    with torch.no_grad():
        learned = float(
            (learned_motor(hidden[768:]).argmax(-1) == targets[768:]).float().mean()
        )
    assert learned >= 0.9
    permutation = torch.randperm(768, generator=torch.Generator().manual_seed(88))
    shuffled_motor = _fit_synthetic(hidden[:768], targets[:768][permutation])
    with torch.no_grad():
        shuffled_true = float(
            (shuffled_motor(hidden[768:]).argmax(-1) == targets[768:]).float().mean()
        )
    assert shuffled_true <= 0.65
    absent = torch.zeros_like(hidden[:512])
    balanced = torch.arange(len(absent)).remainder(2)
    absent_motor = _fit_synthetic(absent, balanced, steps=200)
    with torch.no_grad():
        no_signal = float((absent_motor(absent).argmax(-1) == balanced).float().mean())
    assert no_signal <= 0.51


def test_parameter_count_is_frozen():
    assert CarryMotor(576, rank=8).parameter_count() == 4634


def test_real_board_is_position_and_current_carry_balanced():
    tokenizer_path = ROOT / "artifacts" / "shohin-tok-32k.json"
    episodes_path = (
        ROOT / "artifacts" / "evals" / "digitwise_recurrent_v2_heldout.jsonl"
    )
    if not tokenizer_path.exists() or not episodes_path.exists():
        return
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    rows, report = generate_fit_rows(tokenizer, episodes_path.read_text(), quota=1)
    assert report["rows"] == 128
    counts = collections.Counter(
        (
            row["operation"],
            row["width"],
            row["position"],
            row["style"],
            row["current_carry"],
            row["target"],
        )
        for row in rows
    )
    assert set(counts.values()) == {1}
    for row in rows:
        assert not (row["operation"] == "sub" and row["position"] == row["width"] - 1)
        assert row["prefix_ids"][: len(row["prompt_ids"])] == row["prompt_ids"]


class _FakeBlock(torch.nn.Module):
    def forward(self, hidden):
        return hidden, None


class _CacheSensitiveModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.cfg = type("Cfg", (), {"d_model": 4})()
        self.blocks = torch.nn.ModuleList([_FakeBlock()])

    def forward(self, ids, cache=None, pos=0, return_cache=False):
        hidden = torch.nn.functional.one_hot(ids.remainder(4), num_classes=4).float()
        if cache is None and ids.shape[1] > 2:
            hidden = hidden + 100.0
        hidden, _ = self.blocks[-1](hidden)
        logits = torch.cat((hidden, -hidden), dim=-1)
        new_cache = [torch.tensor([pos + ids.shape[1]])]
        return (logits, new_cache) if return_cache else (logits, None)


def test_feature_extraction_uses_incremental_cache_path():
    rows = [
        {
            "prompt_ids": [1, 2],
            "prefix_ids": [1, 2, 3, 0],
            "target": 0,
        }
    ]
    features = extract_frozen_features(
        _CacheSensitiveModel(), rows, zero_id=1, one_id=2, device="cpu", batch_size=1
    )
    torch.testing.assert_close(features["hidden"], torch.tensor([[1.0, 0.0, 0.0, 0.0]]))


def _synthetic_feature_rows(size=32):
    shapes = ((1, 2), (1, 3), (2, 3), (2, 4))
    rows = []
    for index in range(size):
        prompt_length, prefix_length = shapes[index % len(shapes)]
        prompt_ids = [index % 7 + 1] * prompt_length
        prefix_ids = prompt_ids + [8] * (prefix_length - prompt_length)
        rows.append(
            {
                "operation": "add" if index % 2 == 0 else "sub",
                "width": 4,
                "position": index % 4,
                "style": "core" if index % 2 == 0 else "heldout",
                "current_carry": (index // 2) % 2,
                "prompt_sha256": "{:064x}".format(index + 1001),
                "prompt_ids": prompt_ids,
                "prefix_ids": prefix_ids,
                "prefix_sha256": "{:064x}".format(index + 1),
                "target": index % 2,
                "target_id": index % 2,
            }
        )
    return rows


def _synthetic_features(rows, indices, d_model=2):
    values = torch.as_tensor(indices, dtype=torch.float32)
    labels = torch.as_tensor([rows[index]["target"] for index in indices])
    return {
        "hidden": torch.stack(
            tuple(values + 0.5 * offset for offset in range(d_model)), dim=1
        ),
        "base01": torch.stack((values, -values), dim=1).to(torch.bfloat16),
        "other_lse": values + 3,
        "other_max": values + 2,
        "other_max_token_id": torch.full((len(indices),), 5, dtype=torch.long),
        "other_logits": None,
        "other_token_ids": torch.tensor([2, 3, 4, 5], dtype=torch.long),
        "zero_id": 0,
        "one_id": 1,
        "deployment_logit_dtype": "torch.bfloat16",
        "labels": labels,
    }


def _synthetic_plan(rows, bindings, source_contract, d_model=2):
    root = Path(tempfile.gettempdir()).resolve() / "shohin-carry-synthetic-plan"
    board = {"rows": len(rows), "rows_sha256": stable_json_sha256(rows)}
    initial_state, initial_sha256 = initial_motor_state(d_model)
    plan = {
        "audit": CANONICAL_PLAN_AUDIT,
        "canonical": True,
        "source_contract": source_contract,
        "scientific_source_sha256": bindings["scientific_source_sha256"],
        "confirmation_commitment": {
            "path": "/tmp/synthetic-confirmation/commitment.json",
            "sha256": bindings["confirmation_commitment_sha256"],
            "document": {"synthetic": True},
        },
        "confirmation_exclusion_contract": {"synthetic": True},
        "checkpoint_step": CANONICAL_CHECKPOINT_STEP,
        "d_model": d_model,
        "vocab_size": 6,
        "zero_id": 0,
        "one_id": 1,
        "board": board,
        "board_rows_sha256": stable_json_sha256(rows),
        "sentinel_indices": feature_sentinel_indices(rows),
        "shards": [],
        "runtime_contract": {
            "artifact_runtime": {
                "torch": "synthetic-torch",
                "cuda": "synthetic-cuda",
                "device": "NVIDIA H100 PCIe",
            },
            "deployment_logit_dtype": "torch.bfloat16",
            "extract_batch": CANONICAL_EXTRACT_BATCH,
            "teacher_scoring_contract": CANONICAL_TEACHER_SCORING_CONTRACT,
        },
        "fit_budget": {
            "seed": 20260717,
            "rank": 8,
            "quota": FIT_QUOTA,
            "updates": CANONICAL_UPDATES,
            "batch_size": CANONICAL_BATCH,
            "lr": CANONICAL_LR,
            "weight_decay": CANONICAL_WEIGHT_DECAY,
            "initial_state_sha256": initial_sha256,
            "schedule_sha256": "7" * 64,
            "control": {
                "seed": 20260717,
                "changed": len(rows),
                "changed_rate": 1.0,
                "labels_sha256": stable_json_sha256(
                    [1 - int(row["target"]) for row in rows]
                ),
            },
            "teacher_row_identity_sha256": stable_json_sha256(
                [
                    {
                        "index": index,
                        "operation": row["operation"],
                        "width": row["width"],
                        "position": row["position"],
                        "style": row["style"],
                        "current_carry": row["current_carry"],
                        "target": row["target"],
                        "target_id": row["target_id"],
                        "prompt_sha256": row["prompt_sha256"],
                        "prefix_sha256": row["prefix_sha256"],
                    }
                    for index, row in enumerate(rows)
                ]
            ),
        },
    }
    for shard_index in range(CANONICAL_FEATURE_SHARDS):
        indices = feature_shard_indices(
            len(rows), shard_index, CANONICAL_FEATURE_SHARDS
        )
        plan["shards"].append(
            {
                "shard_index": shard_index,
                "rows": len(indices),
                "global_indices_sha256": stable_json_sha256(indices),
                "row_identity_sha256": row_identity_sha256(rows, indices),
                "artifact": str(root / f"shard_{shard_index:02d}" / "features.pt"),
            }
        )
    return plan, "b" * 64, initial_state


def _synthetic_shards(rows):
    bindings = {
        "base_checkpoint_sha256": "base",
        "tokenizer_sha256": "tokenizer",
        "episodes_sha256": "episodes",
        "cycle_sha256": "cycle",
        "confirmation_commitment_sha256": "confirmation",
        "scientific_source_sha256": {"source": "digest"},
    }
    source_contract = {"git_commit": "commit", "manifest_sha256": "manifest"}
    plan, plan_sha256, _ = _synthetic_plan(rows, bindings, source_contract)
    sentinels = feature_sentinel_indices(rows)
    shards = []
    for shard_index in range(CANONICAL_FEATURE_SHARDS):
        indices = feature_shard_indices(
            len(rows), shard_index, CANONICAL_FEATURE_SHARDS
        )
        features = _synthetic_features(rows, indices)
        sentinel_features = _synthetic_features(rows, sentinels)
        artifact = {
            "audit": CANONICAL_SHARD_AUDIT,
            "canonical": True,
            "plan_sha256": plan_sha256,
            **bindings,
            "source_contract": source_contract,
            "checkpoint_step": plan["checkpoint_step"],
            "board": plan["board"],
            "board_rows_sha256": stable_json_sha256(rows),
            "shard_index": shard_index,
            "shard_count": CANONICAL_FEATURE_SHARDS,
            "global_indices": indices,
            "row_identity_sha256": row_identity_sha256(rows, indices),
            "sentinel_indices": sentinels,
            "sentinel_row_identity_sha256": row_identity_sha256(rows, sentinels),
            "features": features,
            "feature_payload_sha256": feature_payload_sha256(features),
            "sentinel_features": sentinel_features,
            "sentinel_payload_sha256": feature_payload_sha256(sentinel_features),
            "extract_batch": CANONICAL_EXTRACT_BATCH,
            "runtime": plan["runtime_contract"]["artifact_runtime"],
            "claim_boundary": CANONICAL_SHARD_CLAIM_BOUNDARY,
        }
        shards.append(
            (
                "{:064x}".format(shard_index + 1),
                plan["shards"][shard_index]["artifact"],
                artifact,
            )
        )
    return shards, bindings, source_contract, plan, plan_sha256


def test_feature_shards_merge_exactly_and_cross_node_sentinels_fail_closed():
    rows = _synthetic_feature_rows()
    shards, bindings, source_contract, plan, plan_sha256 = _synthetic_shards(rows)
    merged, report = merge_feature_shards(
        shards, rows, bindings, source_contract, plan, plan_sha256
    )
    torch.testing.assert_close(
        merged["hidden"][:, 0], torch.arange(len(rows), dtype=torch.float32)
    )
    assert report["sentinel_indices"] == [0, 1, 2, 3]
    assert len(report["shards"]) == CANONICAL_FEATURE_SHARDS
    assert report["merged_feature_payload_sha256"] == feature_payload_sha256(merged)

    corrupted = copy.deepcopy(shards)
    reversed_merged, reversed_report = merge_feature_shards(
        list(reversed(shards)), rows, bindings, source_contract, plan, plan_sha256
    )
    assert feature_payload_sha256(reversed_merged) == feature_payload_sha256(merged)
    assert reversed_report == report

    corrupted[3][2]["sentinel_features"]["hidden"][0, 0] += 1
    corrupted[3][2]["sentinel_payload_sha256"] = feature_payload_sha256(
        corrupted[3][2]["sentinel_features"]
    )
    with pytest.raises(ValueError, match="cross-node sentinel mismatch"):
        merge_feature_shards(
            corrupted, rows, bindings, source_contract, plan, plan_sha256
        )


def test_feature_shards_reject_duplicate_gap_and_payload_mutation():
    rows = _synthetic_feature_rows()
    shards, bindings, source_contract, plan, plan_sha256 = _synthetic_shards(rows)
    duplicate = copy.deepcopy(shards)
    duplicate[-1] = copy.deepcopy(duplicate[0])
    with pytest.raises(ValueError, match="duplicate feature-shard index"):
        merge_feature_shards(
            duplicate, rows, bindings, source_contract, plan, plan_sha256
        )

    corrupted = copy.deepcopy(shards)
    corrupted[0][2]["features"]["hidden"][0, 0] += 1
    with pytest.raises(ValueError, match="tensor hash mismatch"):
        merge_feature_shards(
            corrupted, rows, bindings, source_contract, plan, plan_sha256
        )

    noncanonical = copy.deepcopy(shards)
    noncanonical[0][2]["canonical"] = False
    with pytest.raises(ValueError, match="not canonical"):
        merge_feature_shards(
            noncanonical, rows, bindings, source_contract, plan, plan_sha256
        )

    wrong_runtime = copy.deepcopy(shards)
    wrong_runtime[0][2]["runtime"]["device"] = "NVIDIA A100-SXM4-80GB"
    with pytest.raises(ValueError, match="runtime is not canonical"):
        merge_feature_shards(
            wrong_runtime, rows, bindings, source_contract, plan, plan_sha256
        )

    nonfinite = copy.deepcopy(shards)
    nonfinite[0][2]["features"]["hidden"][0, 0] = float("nan")
    nonfinite[0][2]["feature_payload_sha256"] = feature_payload_sha256(
        nonfinite[0][2]["features"]
    )
    with pytest.raises(ValueError, match="non-finite"):
        merge_feature_shards(
            nonfinite, rows, bindings, source_contract, plan, plan_sha256
        )


def test_bound_input_consumes_immutable_snapshot_and_rejects_substitution():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "input.bin"
        path.write_bytes(b"original")
        bound = BoundInput(path)
        path.write_bytes(b"mutated!")
        assert bound.bytes() == b"original"
        try:
            bound.verify_path()
        except RuntimeError:
            pass
        else:
            raise AssertionError("same-inode mutation was not rejected")
        replacement = Path(directory) / "replacement.bin"
        replacement.write_bytes(b"original")
        os.replace(replacement, path)
        try:
            bound.verify_path()
        except RuntimeError:
            pass
        else:
            raise AssertionError("path substitution was not rejected")
        bound.close()


def test_sealed_output_directory_is_not_replaceable_without_permission_change():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory) / "sealed"
        root.mkdir(mode=0o700)
        artifact = root / "motor.pt"
        artifact.write_bytes(b"artifact")
        seal_output_directory(artifact)
        assert (root.stat().st_mode & 0o777) == 0o555
        assert (artifact.stat().st_mode & 0o777) == 0o444
        try:
            (root / "new").write_bytes(b"x")
        except PermissionError:
            pass
        else:
            raise AssertionError("sealed directory unexpectedly accepted a new file")
        finally:
            os.chmod(root, 0o755)


def test_planned_directory_lifecycle_rejects_writable_existing_artifacts():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory) / "plan"
        root.mkdir(mode=0o755)
        shard = root / "shard_00"
        shard.mkdir(mode=0o700)
        os.chmod(root, 0o555)
        artifact = shard / "features.pt"
        require_empty_planned_directory(artifact)
        artifact.write_bytes(b"features")
        with pytest.raises(FileExistsError, match="not empty"):
            require_empty_planned_directory(artifact)
        with pytest.raises(ValueError, match="not sealed"):
            require_sealed_artifact(artifact)
        seal_output_directory(artifact)
        require_sealed_artifact(artifact)
        os.chmod(shard, 0o700)
        os.chmod(root, 0o755)


def test_crash_recovery_accepts_only_one_complete_artifact():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory) / "plan"
        root.mkdir(mode=0o700)
        artifact = root / "features.pt"
        artifact.write_bytes(b"complete")
        os.chmod(artifact, 0o444)
        require_recoverable_artifact(artifact)
        assert validate_existing_artifact_stage(artifact)
        extra = root / ".orphan-staging"
        extra.write_bytes(b"orphan")
        with pytest.raises(ValueError, match="one-file"):
            require_recoverable_artifact(artifact)
        extra.unlink()
        seal_output_directory(artifact)
        assert not validate_existing_artifact_stage(artifact)
        os.chmod(root, 0o700)


def test_recovery_and_final_seal_reject_external_staging_hard_link():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        planned = root / "planned"
        planned.mkdir(mode=0o700)
        staging = root / ".features.pt-surviving-stage"
        staging.write_bytes(b"complete")
        os.chmod(staging, 0o444)
        artifact = planned / "features.pt"
        os.link(staging, artifact)
        assert artifact.stat().st_nlink == 2
        with pytest.raises(ValueError, match="one-file"):
            require_recoverable_artifact(artifact)
        with pytest.raises(ValueError, match="one-file"):
            validate_existing_artifact_stage(artifact)
        with pytest.raises(ValueError, match="one-file"):
            seal_output_directory(artifact)

        staging.unlink()
        require_recoverable_artifact(artifact)
        seal_output_directory(artifact)
        external = root / ".features.pt-post-seal-link"
        os.link(artifact, external)
        assert artifact.stat().st_nlink == 2
        with pytest.raises(ValueError, match="not sealed"):
            require_sealed_artifact(artifact)
        with pytest.raises(ValueError, match="not sealed"):
            validate_existing_artifact_stage(artifact)
        external.unlink()
        require_sealed_artifact(artifact)
        os.chmod(planned, 0o700)


def _create_empty_canonical_plan_tree(parent, commit):
    root = parent / "canonical_{}".format(commit)
    root.mkdir(mode=0o700)
    for index in range(CANONICAL_FEATURE_SHARDS):
        (root / "shard_{:02d}".format(index)).mkdir(mode=0o700)
    (root / "fit").mkdir(mode=0o700)
    (root / "development_eval").mkdir(mode=0o700)
    (root / "confirmation_eval").mkdir(mode=0o700)
    plan = root / "plan.json"
    plan.write_text("{}\n")
    os.chmod(plan, 0o444)
    os.chmod(root, 0o555)
    return root, plan


@pytest.mark.parametrize(
    "rewrite",
    ("canonical_bool_to_int", "updates_int_to_float", "histogram_count"),
)
def test_expected_plan_json_normalization_survives_real_publication_and_load(
    monkeypatch, rewrite
):
    class RawTorchVersion(str):
        pass

    class PlanInput:
        def __init__(self, path, text=""):
            self.path = Path(path)
            self._text = text

        def text(self):
            return self._text

    raw_version = RawTorchVersion("2.7.1+cu128")
    prompt_histogram = {97: 16, 99: 16, 103: 16, 105: 16}
    token_histogram = {114: 16, 116: 16, 120: 16, 122: 16}
    rows = _synthetic_feature_rows(64)
    board = {
        "seed": 20260717,
        "rows": len(rows),
        "prompt_length_histogram": prompt_histogram,
        "token_length_histogram": token_histogram,
    }
    commit = "c" * 40
    source_contract = {"git_commit": commit, "manifest_sha256": "1" * 64}
    confirmation_commitment = {"exclusion_contract": {"identity_sha256": "2" * 64}}
    bound = {
        "checkpoint": PlanInput("/frozen/checkpoint.pt"),
        "tokenizer": PlanInput("/frozen/tokenizer.json"),
        "episodes": PlanInput("/frozen/episodes.jsonl", "frozen episodes"),
        "cycle": PlanInput("/frozen/cycle.json"),
        "confirmation_commitment": PlanInput("/frozen/commitment.json"),
    }
    frozen = {
        "checkpoint": "3" * 64,
        "tokenizer": "4" * 64,
        "episodes": "5" * 64,
        "cycle": "6" * 64,
        "confirmation_commitment": "7" * 64,
        "source:train/causal_carry_motor.py": "8" * 64,
    }
    tokenizer = Tokenizer.from_file(str(FROZEN_TOKENIZER_PATH))
    checkpoint = {
        "step": CANONICAL_CHECKPOINT_STEP,
        "cfg": {
            "n_loop": 1,
            "d_model": 2,
            "vocab_size": tokenizer.get_vocab_size(),
        },
    }

    monkeypatch.setattr(torch, "__version__", raw_version)
    monkeypatch.setattr(torch.version, "cuda", "12.8")
    monkeypatch.setattr(
        "causal_carry_motor.canonical_development_selection",
        lambda _text: ([], {"audit": "synthetic-development-selection"}),
    )
    monkeypatch.setattr(
        "causal_carry_motor.permuted_control_labels",
        lambda selected: (
            [int(row["target"]) for row in selected],
            {
                "seed": 20260718,
                "changed": 32,
                "changed_rate": 0.5,
                "labels_sha256": "9" * 64,
            },
        ),
    )
    monkeypatch.setattr(
        "causal_carry_motor._batch_schedule",
        lambda *_args: ([], "a" * 64),
    )

    with tempfile.TemporaryDirectory() as directory:
        parent = Path(directory).resolve() / "carry_motor"
        parent.mkdir()
        monkeypatch.setattr("causal_carry_motor.CANONICAL_PLAN_PARENT", parent)
        root = canonical_plan_root(commit)
        expected = _expected_plan(
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
        assert type(raw_version) is RawTorchVersion
        assert type(expected["runtime_contract"]["artifact_runtime"]["torch"]) is str
        assert expected["board"]["prompt_length_histogram"] == {
            str(key): value for key, value in prompt_histogram.items()
        }
        assert expected["board"]["token_length_histogram"] == {
            str(key): value for key, value in token_histogram.items()
        }

        root.mkdir(mode=0o700)
        for index in range(CANONICAL_FEATURE_SHARDS):
            (root / "shard_{:02d}".format(index)).mkdir(mode=0o700)
        (root / "fit").mkdir(mode=0o700)
        (root / "development_eval").mkdir(mode=0o700)
        (root / "confirmation_eval").mkdir(mode=0o700)
        plan_path = root / "plan.json"
        atomic_json(plan_path, expected)
        os.chmod(root, 0o555)
        args = SimpleNamespace(
            plan=str(plan_path),
            plan_sha256=hashlib.sha256(plan_path.read_bytes()).hexdigest(),
        )
        try:
            plan_bound, loaded = _load_validated_plan(
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
            try:
                assert loaded == expected
            finally:
                plan_bound.close()

            rewritten = copy.deepcopy(expected)
            if rewrite == "canonical_bool_to_int":
                rewritten["canonical"] = 1
                assert rewritten == expected
            elif rewrite == "updates_int_to_float":
                rewritten["fit_budget"]["updates"] = float(
                    rewritten["fit_budget"]["updates"]
                )
                assert rewritten == expected
            else:
                rewritten["board"]["prompt_length_histogram"]["97"] += 1
                assert rewritten != expected
            os.chmod(root, 0o700)
            os.chmod(plan_path, 0o644)
            plan_path.write_text(json.dumps(rewritten, indent=2, sort_keys=True) + "\n")
            os.chmod(plan_path, 0o444)
            os.chmod(root, 0o555)
            args.plan_sha256 = hashlib.sha256(plan_path.read_bytes()).hexdigest()
            with pytest.raises(ValueError, match="canonical plan content mismatch"):
                _load_validated_plan(
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
        finally:
            os.chmod(root, 0o700)


_CONFIRMATION_INPUT_CACHE = None


class _FrozenIdentity:
    def __init__(self, path, sha256):
        self.path = Path(path)
        self.sha256 = sha256

    def verify_path(self):
        return None

    def close(self):
        return None


def _confirmation_inputs():
    global _CONFIRMATION_INPUT_CACHE
    if _CONFIRMATION_INPUT_CACHE is None:
        episodes_text = FROZEN_EPISODES_PATH.read_text()
        cycle_text = FROZEN_CYCLE_PATH.read_text()
        _CONFIRMATION_INPUT_CACHE = (
            episodes_text,
            cycle_text,
            confirmation_exclusion_contract(episodes_text, cycle_text),
        )
    return _CONFIRMATION_INPUT_CACHE


def _confirmation_bound_inputs(observed):
    bound = {
        "checkpoint": _FrozenIdentity(
            "/frozen/checkpoint.pt", EXPECTED_CHECKPOINT_SHA256
        ),
        "episodes": BoundInput(FROZEN_EPISODES_PATH),
        "tokenizer": _FrozenIdentity(
            "/frozen/tokenizer.json", EXPECTED_TOKENIZER_SHA256
        ),
        "cycle": BoundInput(FROZEN_CYCLE_PATH),
    }
    frozen = {
        **observed,
        "checkpoint": EXPECTED_CHECKPOINT_SHA256,
        "episodes": bound["episodes"].sha256,
        "tokenizer": EXPECTED_TOKENIZER_SHA256,
        "cycle": bound["cycle"].sha256,
    }
    return bound, frozen


def _confirmation_fixture(parent, commit="a" * 40):
    observed = {
        "source:{}".format(path): hashlib.sha256(path.encode()).hexdigest()
        for path in CANONICAL_CONFIRMATION_GENERATOR_SOURCES
    }
    source_contract = {
        "git_commit": commit,
        "manifest_sha256": "f" * 64,
    }
    document = {
        "audit": CANONICAL_CONFIRMATION_COMMITMENT_AUDIT,
        "canonical": True,
        "source_contract": source_contract,
        "generator_source_contract": confirmation_generator_source_contract(observed),
        "exclusion_contract": _confirmation_inputs()[2],
        "secret_sha256": hashlib.sha256(bytes(range(32))).hexdigest(),
        "timing": CANONICAL_CONFIRMATION_TIMING,
        "claim_boundary": CANONICAL_CONFIRMATION_CLAIM_BOUNDARY,
    }
    root = parent / "commitment_{}".format(commit)
    root.mkdir(mode=0o700)
    path = root / "commitment.json"
    path.write_text(json.dumps(document, sort_keys=True) + "\n")
    os.chmod(path, 0o444)
    os.chmod(root, 0o555)
    receipt = hashlib.sha256(path.read_bytes()).hexdigest()
    args = SimpleNamespace(
        source_commit=commit,
        confirmation_commitment=str(path),
        confirmation_commitment_sha256=receipt,
    )
    return args, observed, source_contract, document, root, path


def _confirmation_plan(bound, frozen, document, plan_path):
    return {
        "audit": CANONICAL_PLAN_AUDIT,
        "canonical": True,
        "source_contract": document["source_contract"],
        "frozen_inputs": {
            name: {"path": str(bound[name].path), "sha256": frozen[name]}
            for name in ("checkpoint", "tokenizer", "episodes", "cycle")
        },
        "confirmation_commitment": {
            "path": str(bound["confirmation_commitment"].path),
            "sha256": frozen["confirmation_commitment"],
            "document": document,
        },
        "confirmation_exclusion_contract": document["exclusion_contract"],
        "plan_path": str(plan_path),
    }


def _publish_confirmation_plan(parent, commit, document):
    root, path = _create_empty_canonical_plan_tree(parent, commit)
    assert document["plan_path"] == str(path)
    os.chmod(root, 0o700)
    os.chmod(path, 0o600)
    path.write_text(json.dumps(document, sort_keys=True) + "\n")
    os.chmod(path, 0o444)
    os.chmod(root, 0o555)
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def test_plan_extract_and_fit_require_confirmation_before_other_inputs():
    missing = {
        "confirmation_commitment": "",
        "confirmation_commitment_sha256": "",
    }
    with pytest.raises(SystemExit, match="pre-fit confirmation commitment"):
        _plan(
            SimpleNamespace(
                allow_non_cuda=False,
                source_commit="a" * 40,
                **missing,
            )
        )
    with pytest.raises(SystemExit, match="pre-fit confirmation commitment"):
        _extract_shard(
            SimpleNamespace(
                allow_non_cuda=False,
                **vars(_canonical_extract_args(**missing)),
            )
        )
    with pytest.raises(SystemExit, match="pre-fit confirmation commitment"):
        _fit_from_shards(
            SimpleNamespace(
                allow_non_cuda=False,
                **vars(_canonical_feature_fit_args(**missing)),
            )
        )


def test_confirmation_semantic_preflight_is_cpu_only_and_fails_closed(monkeypatch):
    commit = "a" * 40
    args = SimpleNamespace(
        ckpt="/frozen/checkpoint",
        episodes="/frozen/episodes",
        tokenizer="/frozen/tokenizer",
        cycle="/frozen/cycle",
        source_commit=commit,
        source_manifest_sha256="b" * 64,
        confirmation_commitment=str(canonical_confirmation_commitment_path(commit)),
        confirmation_commitment_sha256="c" * 64,
    )
    events = []
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    monkeypatch.setattr(
        "causal_carry_motor.bind_frozen_inputs",
        lambda *_args: events.append("inputs") or ({}, {"source:x": "d" * 64}),
    )
    monkeypatch.setattr(
        "causal_carry_motor.validate_source_contract",
        lambda *_args, **_kwargs: (
            events.append("source")
            or {"git_commit": commit, "manifest_sha256": "b" * 64}
        ),
    )

    def bind_semantic(_args, _bound, frozen, _source_contract):
        events.append("semantic")
        frozen["confirmation_commitment"] = "c" * 64
        return {"exclusion_contract": {"identity_sha256": "e" * 64}}

    monkeypatch.setattr(
        "causal_carry_motor.bind_confirmation_commitment", bind_semantic
    )
    _validate_confirmation_preflight(args)
    assert events == ["inputs", "source", "semantic"]

    monkeypatch.setattr(
        "causal_carry_motor.bind_confirmation_commitment",
        lambda *_args: (_ for _ in ()).throw(ValueError("malformed commitment")),
    )
    with pytest.raises(ValueError, match="malformed commitment"):
        _validate_confirmation_preflight(args)


def test_confirmation_commitment_is_exact_sealed_and_rewrite_detected(monkeypatch):
    with tempfile.TemporaryDirectory() as directory:
        parent = Path(directory).resolve() / "confirmation_commitments"
        parent.mkdir()
        monkeypatch.setattr("causal_carry_motor.CANONICAL_CONFIRMATION_PARENT", parent)
        args, observed, source_contract, document, root, path = _confirmation_fixture(
            parent
        )
        validate_canonical_confirmation_args(args)
        assert require_canonical_confirmation_commitment(path, "a" * 40) == path
        assert document["generator_source_contract"]["schema"] == (
            CANONICAL_CONFIRMATION_GENERATOR_SCHEMA
        )
        assert document["generator_source_contract"]["entrypoint"] == (
            CANONICAL_CONFIRMATION_GENERATOR_ENTRYPOINT
        )
        bound, frozen = _confirmation_bound_inputs(observed)
        assert (
            bind_confirmation_commitment(args, bound, frozen, source_contract)
            == document
        )
        assert frozen["confirmation_commitment"] == args.confirmation_commitment_sha256
        plan = {
            "confirmation_commitment": {
                "path": str(bound["confirmation_commitment"].path),
                "sha256": frozen["confirmation_commitment"],
                "document": document,
            },
            "confirmation_exclusion_contract": document["exclusion_contract"],
        }
        validate_plan_confirmation_binding(plan, bound, frozen, document)

        os.chmod(root, 0o700)
        os.chmod(path, 0o644)
        rewritten = copy.deepcopy(document)
        rewritten["secret_sha256"] = "0" * 64
        path.write_text(json.dumps(rewritten, sort_keys=True) + "\n")
        os.chmod(path, 0o444)
        os.chmod(root, 0o555)
        with pytest.raises(RuntimeError, match="changed or path was substituted"):
            bound["confirmation_commitment"].verify_path()
        for item in bound.values():
            item.close()
        args.confirmation_commitment_sha256 = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        rebound, rebound_frozen = _confirmation_bound_inputs(observed)
        rebound_document = bind_confirmation_commitment(
            args, rebound, rebound_frozen, source_contract
        )
        with pytest.raises(ValueError, match="plan confirmation commitment mismatch"):
            validate_plan_confirmation_binding(
                plan, rebound, rebound_frozen, rebound_document
            )
        for item in rebound.values():
            item.close()


def test_confirmation_generator_is_bound_deterministic_and_cell_balanced(monkeypatch):
    secret = bytes(range(32))
    with tempfile.TemporaryDirectory() as directory:
        parent = Path(directory).resolve() / "confirmation_commitments"
        plan_parent = Path(directory).resolve() / "carry_motor"
        parent.mkdir()
        plan_parent.mkdir()
        monkeypatch.setattr("causal_carry_motor.CANONICAL_CONFIRMATION_PARENT", parent)
        monkeypatch.setattr("causal_carry_motor.CANONICAL_PLAN_PARENT", plan_parent)
        args, observed, source_contract, document, _root, _path = _confirmation_fixture(
            parent
        )
        bound, frozen = _confirmation_bound_inputs(observed)
        try:
            assert (
                bind_confirmation_commitment(args, bound, frozen, source_contract)
                == document
            )
            plan = _confirmation_plan(
                bound,
                frozen,
                document,
                canonical_plan_root(source_contract["git_commit"]) / "plan.json",
            )
            plan_path, plan_sha256 = _publish_confirmation_plan(
                plan_parent, source_contract["git_commit"], plan
            )
            board = generate_confirmation_board(
                secret,
                bound,
                frozen,
                document,
                plan_path,
                plan_sha256,
                plan,
            )
            assert board == generate_confirmation_board(
                secret,
                bound,
                frozen,
                document,
                plan_path,
                plan_sha256,
                plan,
            )
            exclusion = document["exclusion_contract"]
            assert board["exclusion_contract"] == exclusion
            assert exclusion["audit"] == CANONICAL_CONFIRMATION_EXCLUSION_SCHEMA
            assert exclusion["prompt_count"] == EXPECTED_CONFIRMATION_PROMPT_EXCLUSIONS
            assert (
                exclusion["operand_count"] == EXPECTED_CONFIRMATION_OPERAND_EXCLUSIONS
            )
            assert (
                exclusion["identity_sha256"] == EXPECTED_CONFIRMATION_EXCLUSION_SHA256
            )
            assert board["secret_sha256"] == hashlib.sha256(secret).hexdigest()
            assert board["plan"] == {
                "path": plan["plan_path"],
                "sha256": plan_sha256,
            }
            assert len(board["rows"]) == 256
            teacher_rows = confirmation_feature_rows(
                board, Tokenizer.from_file(str(FROZEN_TOKENIZER_PATH))
            )
            assert len(teacher_rows) == 256
            assert [row["target"] for row in teacher_rows] == [
                row["target_carry"] for row in board["rows"]
            ]
            assert secret.hex() not in json.dumps(board)
            assert collections.Counter(
                (
                    row["width"],
                    row["operation"],
                    row["style"],
                    row["target_carry"],
                )
                for row in board["rows"]
            ) == {
                (width, operation, style, target): 8
                for width in (4, 6, 8, 10)
                for operation in ("add", "sub")
                for style in ("core", "heldout")
                for target in (0, 1)
            }
            assert all(
                row["position"] < row["width"] - 1
                for row in board["rows"]
                if row["operation"] == "sub" and row["target_carry"] == 1
            )
            with pytest.raises(TypeError):
                generate_confirmation_board(
                    secret,
                    bound,
                    frozen,
                    document,
                    plan_path,
                    plan_sha256,
                    plan,
                    forbidden_prompt_sha256=(),
                )

            with pytest.raises(ValueError, match="secret or exclusion"):
                generate_confirmation_board(
                    bytes(reversed(range(32))),
                    bound,
                    frozen,
                    document,
                    plan_path,
                    plan_sha256,
                    plan,
                )

            with pytest.raises(ValueError, match="exact canonical path"):
                generate_confirmation_board(
                    secret,
                    bound,
                    frozen,
                    document,
                    "/hostile/plan.json",
                    "e" * 64,
                    plan,
                )

            with pytest.raises(ValueError, match="artifact hash mismatch"):
                generate_confirmation_board(
                    secret,
                    bound,
                    frozen,
                    document,
                    plan_path,
                    "e" * 64,
                    plan,
                )

            rewritten_plan = copy.deepcopy(plan)
            rewritten_plan["plan_path"] = "/hostile/plan.json"
            with pytest.raises(ValueError, match="differs from bound bytes"):
                generate_confirmation_board(
                    secret,
                    bound,
                    frozen,
                    document,
                    plan_path,
                    plan_sha256,
                    rewritten_plan,
                )

            rewritten = copy.deepcopy(board)
            rewritten["rows"][0]["prompt_sha256"] = "0" * 64
            rewritten["rows_sha256"] = stable_json_sha256(rewritten["rows"])
            with pytest.raises(ValueError, match="canonical regeneration"):
                validate_confirmation_board_binding(rewritten, board)
        finally:
            for item in bound.values():
                item.close()


def test_confirmation_generator_rejects_altered_episode_1499(monkeypatch):
    secret = bytes(range(32))
    with tempfile.TemporaryDirectory() as directory:
        parent = Path(directory).resolve() / "confirmation_commitments"
        plan_parent = Path(directory).resolve() / "carry_motor"
        parent.mkdir()
        plan_parent.mkdir()
        monkeypatch.setattr("causal_carry_motor.CANONICAL_CONFIRMATION_PARENT", parent)
        monkeypatch.setattr("causal_carry_motor.CANONICAL_PLAN_PARENT", plan_parent)
        args, observed, source_contract, document, _root, _path = _confirmation_fixture(
            parent
        )
        bound, frozen = _confirmation_bound_inputs(observed)
        altered = None
        try:
            assert (
                bind_confirmation_commitment(args, bound, frozen, source_contract)
                == document
            )
            plan = _confirmation_plan(
                bound,
                frozen,
                document,
                canonical_plan_root(source_contract["git_commit"]) / "plan.json",
            )
            plan_path, plan_sha256 = _publish_confirmation_plan(
                plan_parent, source_contract["git_commit"], plan
            )
            lines = FROZEN_EPISODES_PATH.read_text().splitlines()
            row = json.loads(lines[1499])
            row["id"] += "-rewritten"
            lines[1499] = json.dumps(row, sort_keys=True)
            altered_path = Path(directory) / "altered-episodes.jsonl"
            altered_path.write_text("\n".join(lines) + "\n")
            altered = BoundInput(altered_path)
            bound["episodes"].close()
            bound["episodes"] = altered
            with pytest.raises(ValueError, match="binding mismatch: episodes"):
                generate_confirmation_board(
                    secret,
                    bound,
                    frozen,
                    document,
                    plan_path,
                    plan_sha256,
                    plan,
                )
        finally:
            for item in bound.values():
                item.close()


def _rewrite_confirmation_exclusion_self_consistently(document):
    exclusion = document["exclusion_contract"]
    removed = exclusion["identities"].pop(0)
    assert removed["kind"] == "prompt"
    exclusion["prompt_count"] -= 1
    exclusion["identity_count"] -= 1
    exclusion["identity_sha256"] = stable_json_sha256(exclusion["identities"])


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda document: document.update({"secret": "forbidden"}), "schema"),
        (
            lambda document: document["generator_source_contract"]["sources"].update(
                {"train/digitwise_protocol.py": "0" * 64}
            ),
            "content mismatch",
        ),
        (
            lambda document: document.update({"secret_sha256": "bad"}),
            "content mismatch",
        ),
        (
            _rewrite_confirmation_exclusion_self_consistently,
            "content mismatch",
        ),
    ],
)
def test_confirmation_commitment_rejects_malformed_or_mismatched_source(
    monkeypatch, mutation, message
):
    with tempfile.TemporaryDirectory() as directory:
        parent = Path(directory).resolve() / "confirmation_commitments"
        parent.mkdir()
        monkeypatch.setattr("causal_carry_motor.CANONICAL_CONFIRMATION_PARENT", parent)
        args, observed, source_contract, document, root, path = _confirmation_fixture(
            parent
        )
        os.chmod(root, 0o700)
        os.chmod(path, 0o644)
        mutation(document)
        path.write_text(json.dumps(document, sort_keys=True) + "\n")
        os.chmod(path, 0o444)
        os.chmod(root, 0o555)
        args.confirmation_commitment_sha256 = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        bound, frozen = _confirmation_bound_inputs(observed)
        try:
            with pytest.raises(ValueError, match=message):
                bind_confirmation_commitment(args, bound, frozen, source_contract)
        finally:
            for item in bound.values():
                item.close()


def test_canonical_plan_rejects_writable_and_linked_plan_before_binding(monkeypatch):
    with tempfile.TemporaryDirectory() as directory:
        parent = Path(directory) / "carry_motor"
        parent.mkdir()
        commit = "a" * 40
        monkeypatch.setattr("causal_carry_motor.CANONICAL_PLAN_PARENT", parent)
        root, plan = _create_empty_canonical_plan_tree(parent, commit)
        validate_canonical_plan_layout(plan, commit)

        os.chmod(plan, 0o644)
        with pytest.raises(ValueError, match="plan.json.*mode-0444"):
            validate_canonical_plan_layout(plan, commit)
        os.chmod(plan, 0o444)

        external = Path(directory) / "plan-hard-link.json"
        os.link(plan, external)
        with pytest.raises(ValueError, match="plan.json.*one-link"):
            validate_canonical_plan_layout(plan, commit)
        external.unlink()
        validate_canonical_plan_layout(plan, commit)
        os.chmod(root, 0o700)


def test_canonical_fit_rejects_linked_shard_before_bound_input(monkeypatch):
    with tempfile.TemporaryDirectory() as directory:
        parent = Path(directory) / "carry_motor"
        parent.mkdir()
        commit = "b" * 40
        monkeypatch.setattr("causal_carry_motor.CANONICAL_PLAN_PARENT", parent)
        root, plan = _create_empty_canonical_plan_tree(parent, commit)
        shard = root / "shard_00"
        artifact = shard / "features.pt"
        artifact.write_bytes(b"sealed shard")
        os.chmod(artifact, 0o444)
        os.chmod(shard, 0o555)
        validate_canonical_plan_layout(plan, commit)
        require_canonical_shard_input(artifact, root, 0)

        external = Path(directory) / "features-hard-link.pt"
        os.link(artifact, external)
        with pytest.raises(ValueError, match="shard input.*one-link"):
            require_canonical_shard_input(artifact, root, 0)
        external.unlink()
        require_canonical_shard_input(artifact, root, 0)
        os.chmod(root, 0o700)
        os.chmod(shard, 0o700)


def test_all_shards_preflight_before_first_shard_bind_or_load(monkeypatch):
    root = Path("/frozen/canonical")
    descriptors = [
        {"artifact": str(root / f"shard_{index:02d}" / "features.pt")}
        for index in range(CANONICAL_FEATURE_SHARDS)
    ]
    events = []

    def require(path, plan_root, shard_index):
        assert plan_root == root
        events.append(("check", shard_index))
        if shard_index == 1:
            raise ValueError("invalid shard 1")
        return Path(path)

    monkeypatch.setattr("causal_carry_motor.require_canonical_shard_input", require)
    monkeypatch.setattr(
        "causal_carry_motor.BoundInput",
        lambda _path: events.append(("bind", 0)),
    )
    monkeypatch.setattr(
        "causal_carry_motor.torch.load",
        lambda *_args, **_kwargs: events.append(("load", 0)),
    )
    with pytest.raises(ValueError, match="invalid shard 1"):
        _bind_and_load_canonical_shards(descriptors, root)
    assert events == [("check", 0), ("check", 1)]


def test_canonical_bundle_validator_always_replays_all_shards(monkeypatch):
    events = []
    bounds = []

    class FakeBound:
        def __init__(self, index):
            self.index = index

        def verify_path(self):
            events.append(("verify", self.index))

        def close(self):
            events.append(("close", self.index))

    for index in range(CANONICAL_FEATURE_SHARDS):
        bounds.append(FakeBound(index))
    plan = {
        "plan_path": "/canonical/plan.json",
        "shards": [{"artifact": str(index)} for index in range(8)],
    }
    rows = [{"target": 0}]
    features = {"hidden": torch.zeros((1, 1))}
    merge = {"merged": True}

    monkeypatch.setattr(
        "causal_carry_motor.validate_canonical_plan_layout",
        lambda *_args: events.append("layout"),
    )
    monkeypatch.setattr(
        "causal_carry_motor._bind_and_load_canonical_shards",
        lambda *_args: events.append("bind-load") or (bounds, ["payload"] * 8),
    )
    monkeypatch.setattr(
        "causal_carry_motor.merge_feature_shards",
        lambda *_args: events.append("merge") or (features, merge),
    )
    monkeypatch.setattr(
        "causal_carry_motor.require_canonical_cuda_runtime",
        lambda: events.append("h100") or "cuda",
    )

    def validate(*args):
        assert args[-3:] == (features, merge, "cuda")
        events.append("payload-validator")

    monkeypatch.setattr(
        "causal_carry_motor._validate_motor_bundle_against_replayed_features",
        validate,
    )
    validate_motor_bundle({}, {}, {}, {"git_commit": "a" * 40}, "b" * 64, plan, rows)
    assert events[:4] == ["layout", "bind-load", "merge", ("verify", 0)]
    assert events[11:13] == ["h100", "payload-validator"]
    assert events.count("layout") == 2
    assert events[-8:] == [("close", index) for index in range(8)]


def test_canonical_publication_stages_outside_planned_directory():
    with tempfile.TemporaryDirectory() as directory:
        parent = Path(directory)
        output_dir = parent / "planned"
        staging_dir = parent / "staging"
        output_dir.mkdir(mode=0o700)
        staging_dir.mkdir(mode=0o700)
        artifact = output_dir / "motor.pt"
        atomic_torch(
            artifact,
            {"tensor": torch.tensor([1.0])},
            staging_parent=staging_dir,
        )
        assert {child.name for child in output_dir.iterdir()} == {"motor.pt"}
        assert not any(staging_dir.iterdir())
        require_recoverable_artifact(artifact)


def test_initial_motor_state_is_seed_frozen_and_shape_bound():
    torch.manual_seed(1)
    first, first_sha256 = initial_motor_state(13)
    torch.manual_seed(999)
    second, second_sha256 = initial_motor_state(13)
    assert first_sha256 == second_sha256
    assert set(first) == set(second)
    for name in first:
        assert torch.equal(first[name], second[name])
    _, different_sha256 = initial_motor_state(14)
    assert different_sha256 != first_sha256


def _canonical_train_args(**overrides):
    values = {
        "quota": FIT_QUOTA,
        "updates": CANONICAL_UPDATES,
        "batch_size": CANONICAL_BATCH,
        "lr": CANONICAL_LR,
        "weight_decay": CANONICAL_WEIGHT_DECAY,
        "extract_batch": CANONICAL_EXTRACT_BATCH,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _canonical_eval_args(**overrides):
    values = {
        "max_new": CANONICAL_MAX_NEW,
        "extract_batch": CANONICAL_EXTRACT_BATCH,
        "source_commit": "a" * 40,
        "confirmation_commitment": str(
            canonical_confirmation_commitment_path("a" * 40)
        ),
        "confirmation_commitment_sha256": "c" * 64,
        "plan": str(canonical_plan_root("a" * 40) / "plan.json"),
        "plan_sha256": "a" * 64,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _canonical_extract_args(**overrides):
    values = {
        "quota": FIT_QUOTA,
        "extract_batch": CANONICAL_EXTRACT_BATCH,
        "shard_index": 0,
        "shard_count": CANONICAL_FEATURE_SHARDS,
        "source_commit": "a" * 40,
        "confirmation_commitment": str(
            canonical_confirmation_commitment_path("a" * 40)
        ),
        "confirmation_commitment_sha256": "c" * 64,
        "plan": str(canonical_plan_root("a" * 40) / "plan.json"),
        "plan_sha256": "a" * 64,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _canonical_feature_fit_args(**overrides):
    values = vars(_canonical_train_args()).copy()
    values.update(
        {
            "source_commit": "a" * 40,
            "confirmation_commitment": str(
                canonical_confirmation_commitment_path("a" * 40)
            ),
            "confirmation_commitment_sha256": "c" * 64,
            "plan": str(canonical_plan_root("a" * 40) / "plan.json"),
            "plan_sha256": "a" * 64,
        }
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_canonical_budgets_refuse_every_mutable_dimension():
    validate_canonical_train_args(_canonical_train_args(), True)
    for name, value in {
        "quota": FIT_QUOTA - 1,
        "updates": CANONICAL_UPDATES - 1,
        "batch_size": CANONICAL_BATCH // 2,
        "lr": CANONICAL_LR / 2,
        "weight_decay": 0.0,
        "extract_batch": 2,
    }.items():
        with pytest.raises(SystemExit):
            validate_canonical_train_args(_canonical_train_args(**{name: value}), True)
    validate_canonical_eval_args(_canonical_eval_args(), True)
    for name, value in {
        "max_new": CANONICAL_MAX_NEW - 1,
        "extract_batch": 2,
        "plan_sha256": "",
    }.items():
        with pytest.raises(SystemExit):
            validate_canonical_eval_args(_canonical_eval_args(**{name: value}), True)
    validate_canonical_extract_args(_canonical_extract_args(), True)
    for name, value in {
        "quota": FIT_QUOTA - 1,
        "extract_batch": 2,
        "shard_index": CANONICAL_FEATURE_SHARDS,
        "shard_count": CANONICAL_FEATURE_SHARDS - 1,
    }.items():
        with pytest.raises(SystemExit):
            validate_canonical_extract_args(
                _canonical_extract_args(**{name: value}), True
            )
    validate_canonical_feature_fit_args(_canonical_feature_fit_args(), True)
    with pytest.raises(SystemExit):
        validate_canonical_feature_fit_args(_canonical_feature_fit_args(plan=""), True)
    with pytest.raises(SystemExit, match="sole committed root"):
        validate_canonical_feature_fit_args(
            _canonical_feature_fit_args(plan="/tmp/alternate/plan.json"), True
        )
    with pytest.raises(SystemExit):
        validate_canonical_feature_fit_args(
            _canonical_feature_fit_args(plan_sha256="bad"), True
        )


def test_canonical_runtime_rejects_non_h100_even_when_cuda_is_available(monkeypatch):
    class FakeTensor:
        def add_(self, _value):
            return self

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _index: "NVIDIA A100")
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.version, "cuda", "12.4")
    monkeypatch.setattr(torch, "empty", lambda *args, **kwargs: FakeTensor())
    with pytest.raises(RuntimeError, match="H100"):
        require_canonical_cuda_runtime()
    monkeypatch.setattr(
        torch.cuda, "get_device_name", lambda _index: "NVIDIA H100 PCIe"
    )
    assert require_canonical_cuda_runtime() == "cuda"


def test_planned_evaluation_has_no_non_cuda_bypass():
    with pytest.raises(SystemExit, match="does not accept"):
        _eval(SimpleNamespace(allow_non_cuda=True))
    with pytest.raises(SystemExit, match="does not accept"):
        _confirmation_eval(SimpleNamespace(allow_non_cuda=True))


def test_bundle_receipt_and_state_hash_corruption_fail_closed():
    treatment_model = CarryMotor(4, rank=8)
    shuffled_model = CarryMotor(4, rank=8)
    treatment = treatment_model.state_dict()
    shuffled = shuffled_model.state_dict()
    bindings = {
        "base_checkpoint_sha256": "base",
        "tokenizer_sha256": "tokenizer",
        "episodes_sha256": "episodes",
        "cycle_sha256": "cycle",
        "confirmation_commitment_sha256": "confirmation",
    }
    sources = {"train/causal_carry_motor.py": "source"}
    shard_bindings = {**bindings, "scientific_source_sha256": sources}
    source_contract = {"git_commit": "commit", "manifest_sha256": "manifest"}
    rows = _synthetic_feature_rows()
    plan, plan_sha256, _ = _synthetic_plan(
        rows, shard_bindings, source_contract, d_model=4
    )
    features = _synthetic_features(rows, list(range(len(rows))), d_model=4)
    control_labels = [1 - int(row["target"]) for row in rows]
    metric_payload_sha256 = teacher_metric_feature_payload_sha256(features)
    fit_report = {
        "updates": CANONICAL_UPDATES,
        "batch_size": CANONICAL_BATCH,
        "lr": CANONICAL_LR,
        "weight_decay": CANONICAL_WEIGHT_DECAY,
        "schedule_sha256": plan["fit_budget"]["schedule_sha256"],
        "first_loss": 1.0,
        "final_loss": 0.5,
        "min_loss": 0.5,
    }
    feature_merge = {
        "shards": [
            {
                "shard_index": index,
                "artifact_sha256": "{:064x}".format(index + 1),
                "artifact": descriptor["artifact"],
                "feature_payload_sha256": "{:064x}".format(index + 101),
                "rows": descriptor["rows"],
            }
            for index, descriptor in enumerate(plan["shards"])
        ],
        "sentinel_indices": plan["sentinel_indices"],
        "sentinel_payload_sha256": "c" * 64,
        "runtime": plan["runtime_contract"]["artifact_runtime"],
        "plan_sha256": plan_sha256,
        "merged_feature_payload_sha256": feature_payload_sha256(features),
        "teacher_metric_feature_payload_sha256": metric_payload_sha256,
    }
    expected_rows = plan["board"]["rows"]
    fit_feature_evidence = fit_teacher_forced_evidence(
        features,
        rows,
        control_labels,
        treatment_model,
        shuffled_model,
        metric_payload_sha256,
        "cpu",
        CANONICAL_TEACHER_SCORING_CONTRACT,
    )
    diagnostic_test_rows = expected_rows - int(expected_rows * 0.8)
    linear_diagnostic = {
        "train_rows": int(expected_rows * 0.8),
        "test_rows": diagnostic_test_rows,
        "test_correct": 3,
        "test_accuracy": 3 / diagnostic_test_rows,
        "schedule_sha256": _batch_schedule(
            int(expected_rows * 0.8),
            min(512, int(expected_rows * 0.8)),
            300,
            20260719,
        )[1],
        "claim_boundary": LINEAR_DIAGNOSTIC_CLAIM_BOUNDARY,
    }
    bundle = {
        "audit": CANONICAL_FIT_AUDIT,
        "canonical": True,
        "plan_sha256": plan_sha256,
        **bindings,
        "scientific_source_sha256": sources,
        "source_contract": source_contract,
        "checkpoint_step": plan["checkpoint_step"],
        "d_model": plan["d_model"],
        "rank": plan["fit_budget"]["rank"],
        "parameter_count": CarryMotor(4, rank=8).parameter_count(),
        "extract_batch": CANONICAL_EXTRACT_BATCH,
        "feature_shard_merge": feature_merge,
        "deployment_logit_dtype": "torch.bfloat16",
        "zero_id": plan["zero_id"],
        "one_id": plan["one_id"],
        "initial_state_sha256": plan["fit_budget"]["initial_state_sha256"],
        "treatment": treatment,
        "shuffled": shuffled,
        "treatment_state_sha256": tensor_state_sha256(treatment),
        "shuffled_state_sha256": tensor_state_sha256(shuffled),
        "board": plan["board"],
        "control": plan["fit_budget"]["control"],
        "treatment_fit": copy.deepcopy(fit_report),
        "shuffled_fit": copy.deepcopy(fit_report),
        "linear_diagnostic": linear_diagnostic,
        "fit_feature_metrics": fit_feature_evidence,
        "claim_boundary": CANONICAL_FIT_CLAIM_BOUNDARY,
    }

    def validate_bundle(candidate):
        _validate_motor_bundle_against_replayed_features(
            candidate,
            bindings,
            sources,
            source_contract,
            plan_sha256,
            plan,
            features,
            feature_merge,
            "cpu",
        )

    validate_bundle(bundle)
    bad_dtype = copy.deepcopy(bundle)
    bad_dtype["deployment_logit_dtype"] = "torch.float64"
    with pytest.raises(ValueError, match="deployment logit dtype"):
        validate_bundle(bad_dtype)
    for arm in ("treatment", "shuffled"):
        corrupted = copy.deepcopy(bundle)
        first = next(iter(corrupted[arm].values()))
        first.view(-1)[0] += 1
        with pytest.raises(ValueError, match="state hash mismatch"):
            validate_bundle(corrupted)
    development = copy.deepcopy(bundle)
    development["audit"] = "causal_carry_motor_fit_development_v1"
    with pytest.raises(ValueError, match="unsupported carry-motor"):
        validate_bundle(development)
    bad_diagnostic = copy.deepcopy(bundle)
    bad_diagnostic["linear_diagnostic"]["test_accuracy"] = float("nan")
    with pytest.raises(ValueError, match="accuracy"):
        validate_bundle(bad_diagnostic)
    bad_metrics = copy.deepcopy(bundle)
    bad_metrics["fit_feature_metrics"]["arms"]["base"]["summary"]["global_accuracy"] = (
        float("nan")
    )
    with pytest.raises(ValueError, match="accuracy"):
        validate_bundle(bad_metrics)
    deleted_evidence = copy.deepcopy(bundle)
    del deleted_evidence["fit_feature_metrics"]["other_max_token_ids"]
    with pytest.raises(ValueError, match="fit teacher evidence schema mismatch"):
        validate_bundle(deleted_evidence)
    rewritten_logits = copy.deepcopy(bundle)
    evidence = rewritten_logits["fit_feature_metrics"]
    arm = evidence["arms"]["treatment"]
    arm["adjusted_carry_logits"][0] = torch.tensor(
        [120.0, -120.0], dtype=torch.bfloat16
    )
    derived = _derive_top1_tensor_evidence(
        arm["adjusted_carry_logits"],
        evidence["true_targets"],
        evidence["other_max_logits"],
        evidence["other_max_token_ids"],
        evidence["zero_id"],
        evidence["one_id"],
    )
    arm.update(derived)
    arm["summary"] = _tensor_metric_summary(arm)
    with pytest.raises(ValueError, match="logits differ from fitted motor state"):
        validate_bundle(rewritten_logits)
    rewritten_shuffled = copy.deepcopy(bundle)
    evidence = rewritten_shuffled["fit_feature_metrics"]
    for arm_name, target_name in (
        ("shuffled_on_true_labels", "true_targets"),
        ("shuffled_on_control_labels", "control_targets"),
    ):
        arm = evidence["arms"][arm_name]
        arm["adjusted_carry_logits"][0] = torch.tensor(
            [120.0, -120.0], dtype=torch.bfloat16
        )
        arm.update(
            _derive_top1_tensor_evidence(
                arm["adjusted_carry_logits"],
                evidence[target_name],
                evidence["other_max_logits"],
                evidence["other_max_token_ids"],
                evidence["zero_id"],
                evidence["one_id"],
            )
        )
        arm["summary"] = _tensor_metric_summary(arm)
    with pytest.raises(ValueError, match="logits differ from fitted motor state"):
        validate_bundle(rewritten_shuffled)

    rewritten_hidden = copy.deepcopy(bundle)
    forged_features = copy.deepcopy(features)
    forged_features["hidden"][0, 0] += 7.0
    evidence = rewritten_hidden["fit_feature_metrics"]
    evidence["hidden"] = forged_features["hidden"].clone()
    evidence["source_feature_payload_sha256"] = teacher_metric_feature_payload_sha256(
        forged_features
    )
    motors = {
        "base": None,
        "treatment": treatment_model,
        "shuffled_on_true_labels": shuffled_model,
        "shuffled_on_control_labels": shuffled_model,
    }
    for arm_name, motor in motors.items():
        arm = evidence["arms"][arm_name]
        arm["adjusted_carry_logits"] = _singleton_deployment_adjusted_carry_logits(
            evidence["hidden"], forged_features["base01"], motor, "cpu"
        )
        targets = evidence[
            "control_targets" if arm["target_source"] == "control" else "true_targets"
        ]
        arm.update(
            _derive_top1_tensor_evidence(
                arm["adjusted_carry_logits"],
                targets,
                evidence["other_max_logits"],
                evidence["other_max_token_ids"],
                evidence["zero_id"],
                evidence["one_id"],
            )
        )
        arm["summary"] = _tensor_metric_summary(arm)
    rewritten_hidden["feature_shard_merge"]["merged_feature_payload_sha256"] = (
        feature_payload_sha256(forged_features)
    )
    rewritten_hidden["feature_shard_merge"]["teacher_metric_feature_payload_sha256"] = (
        teacher_metric_feature_payload_sha256(forged_features)
    )
    with pytest.raises(ValueError, match="differs from replayed sealed shards"):
        validate_bundle(rewritten_hidden)

    bad_claim = copy.deepcopy(bundle)
    bad_claim["claim_boundary"] = 1
    with pytest.raises(ValueError, match="claim boundary"):
        validate_bundle(bad_claim)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "motor.pt"
        path.write_bytes(b"motor")
        bound = BoundInput(path)
        validate_artifact_receipt(bound, bound.sha256)
        with pytest.raises(ValueError, match="artifact hash mismatch"):
            validate_artifact_receipt(bound, "0" * 64)
        with pytest.raises(ValueError, match="required"):
            validate_artifact_receipt(bound, "")
        bound.close()


def test_atomic_no_replace_and_canonical_directory_reuse_are_refused():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "canonical" / "result.json"
        prepared = prepare_output(path, canonical=True)
        atomic_json(prepared, {"value": 1})
        with pytest.raises(FileExistsError):
            atomic_json(prepared, {"value": 2})
        with pytest.raises(FileExistsError, match="must be new"):
            prepare_output(Path(directory) / "canonical" / "other.json", canonical=True)


def test_router_counts_exactly_one_fire_and_reports_missed_site():
    state = initial_state("add", 379, 215, 4)
    prompt = microstep_prompt(state, style="core")
    router = CarryRouter(prompt, motor_present=True)
    assert not router.observe("")
    assert router.observe("dws:op=add;w=4;p=1;c=")
    assert not router.observe("dws:op=add;w=4;p=1;c=")
    assert router.site_count == 1
    assert router.fire_count == 1
    missed = CarryRouter(prompt, motor_present=True)
    assert not missed.observe("dws:op=add;w=4;p=1;c=0")
    assert missed.site_count == 0
    assert missed.fire_count == 0
    base = CarryRouter(prompt, motor_present=False)
    assert not base.observe("dws:op=add;w=4;p=1;c=")
    assert base.site_count == 1
    assert base.fire_count == 0


def _tokenizer_fixture():
    return Tokenizer.from_file(str(ROOT / "artifacts" / "shohin-tok-32k.json"))


def _generation_fixture(
    tokenizer,
    prompt,
    response,
    motor_present,
    *,
    response_ids=None,
    sequence_cap=4096,
    max_new=CANONICAL_MAX_NEW,
    retain_full_logits=False,
):
    if response_ids is None:
        response_ids = tokenizer.encode(response).ids
    assert tokenizer.decode(response_ids, skip_special_tokens=False) == response
    eos_id = tokenizer.token_to_id("<|endoftext|>")
    token_ids = list(response_ids) + [eos_id]
    assert len(token_ids) <= max_new
    router = CarryRouter(prompt, motor_present)
    boundaries = []
    for index, token_id in enumerate(token_ids):
        prefix = tokenizer.decode(token_ids[:index], skip_special_tokens=False)
        prior_sites, prior_fires = router.site_count, router.fire_count
        router.observe(prefix)
        boundaries.append(
            {
                "index": index,
                "prefix_token_count": index,
                "decoded_prefix": prefix,
                "next_token_id": token_id,
                "router_site": router.site_count == prior_sites + 1,
                "motor_fired": router.fire_count == prior_fires + 1,
                "full_logits": (
                    full_logit_tensor_identity(
                        torch.zeros(
                            (1, tokenizer.get_vocab_size()), dtype=torch.float32
                        )
                    )
                    if retain_full_logits
                    else None
                ),
            }
        )
    prompt_ids = tokenizer.encode(prompt).ids[-sequence_cap:]
    generation = {
        "prompt_token_count": len(prompt_ids),
        "prompt_token_ids_sha256": stable_json_sha256(prompt_ids),
        "sequence_cap": sequence_cap,
        "max_new": max_new,
        "eos_token_id": eos_id,
        "token_ids": token_ids,
        "token_ids_sha256": stable_json_sha256(token_ids),
        "boundaries": boundaries,
        "boundaries_sha256": stable_json_sha256(boundaries),
        "stop_reason": "eos",
    }
    return generation, router.site_count, router.fire_count


class _SkippedEqualsTokenizer:
    pieces = {
        0: "<|endoftext|>",
        1: "dws:op=add;w=2;p=1;c",
        2: "==",
        3: "0",
        4: "prompt",
    }

    class Encoding:
        def __init__(self, ids):
            self.ids = ids

    def encode(self, _text):
        return self.Encoding([4])

    def decode(self, token_ids, skip_special_tokens=False):
        return "".join(
            "" if skip_special_tokens and token_id == 0 else self.pieces[token_id]
            for token_id in token_ids
        )

    def token_to_id(self, token):
        return 0 if token == "<|endoftext|>" else None

    def get_vocab_size(self):
        return len(self.pieces)


class _AliasedEqualsTokenizer:
    class Encoding:
        def __init__(self, ids):
            self.ids = ids

    def encode(self, _text):
        return self.Encoding([1])

    def decode(self, token_ids, skip_special_tokens=False):
        pieces = {0: "<|endoftext|>", 1: "prompt", 41: "=", 818: "=="}
        return "".join(
            "" if skip_special_tokens and token_id == 0 else pieces.get(token_id, "x")
            for token_id in token_ids
        )

    def token_to_id(self, token):
        return 0 if token == "<|endoftext|>" else None

    def get_vocab_size(self):
        return 819


def test_preservation_rejects_distinct_tokenizations_with_same_decoded_text():
    tokenizer = _AliasedEqualsTokenizer()
    prompt = "preservation prompt"
    base, _sites, _fires = _generation_fixture(
        tokenizer,
        prompt,
        "==",
        False,
        response_ids=[818],
        max_new=48,
        retain_full_logits=True,
    )
    treatment, _sites, _fires = _generation_fixture(
        tokenizer,
        prompt,
        "==",
        True,
        response_ids=[41, 41],
        max_new=48,
        retain_full_logits=True,
    )
    for name, generation, motor_present in (
        ("base", base, False),
        ("treatment", treatment, True),
    ):
        _validate_generation_evidence(
            generation,
            prompt,
            "==",
            motor_present,
            tokenizer,
            4096,
            48,
            name,
            require_full_logits=True,
        )
    with pytest.raises(ValueError, match="token-ID sequences differ"):
        _validate_preservation_trace_identity(base, treatment, "aliased equals")

    rewritten = copy.deepcopy(base)
    trace = rewritten["boundaries"][0]["full_logits"]
    trace["bytes_sha256"] = "f" * 64
    trace["identity_sha256"] = stable_json_sha256(
        {
            "dtype": trace["dtype"],
            "shape": trace["shape"],
            "byte_count": trace["byte_count"],
            "bytes_sha256": trace["bytes_sha256"],
        }
    )
    rewritten["boundaries_sha256"] = stable_json_sha256(rewritten["boundaries"])
    with pytest.raises(ValueError, match="full-logit boundary identities differ"):
        _validate_preservation_trace_identity(base, rewritten, "rewritten logits")


def test_router_replay_uses_decoded_token_boundaries_not_characters():
    state = initial_state("add", 19, 1, 2)
    prompt = microstep_prompt(state, style="core")
    tokenizer = _SkippedEqualsTokenizer()
    response = "dws:op=add;w=2;p=1;c==0"
    generation, sites, fires = _generation_fixture(
        tokenizer,
        prompt,
        response,
        True,
        response_ids=[1, 2, 3],
    )
    assert is_carry_site(prompt, "dws:op=add;w=2;p=1;c=")
    assert all(
        boundary["decoded_prefix"] != "dws:op=add;w=2;p=1;c="
        for boundary in generation["boundaries"]
    )
    assert (sites, fires) == (0, 0)
    assert _validate_generation_evidence(
        generation,
        prompt,
        response,
        True,
        tokenizer,
        4096,
        CANONICAL_MAX_NEW,
        "skipped equals",
    ) == (0, 0)


@pytest.mark.parametrize("mutation", ["token", "prefix", "decision", "stop"])
def test_generation_evidence_rejects_token_prefix_decision_and_stop_rewrites(
    mutation,
):
    tokenizer = _tokenizer_fixture()
    state = initial_state("add", 99, 1, 2)
    prompt = microstep_prompt(state, style="core")
    response = canonical_state(apply_microstep(state))
    generation, _sites, _fires = _generation_fixture(tokenizer, prompt, response, True)
    malformed = copy.deepcopy(generation)
    if mutation == "token":
        malformed["token_ids"][0] += 1
        malformed["token_ids_sha256"] = stable_json_sha256(malformed["token_ids"])
    elif mutation == "prefix":
        malformed["boundaries"][1]["decoded_prefix"] += "x"
        malformed["boundaries_sha256"] = stable_json_sha256(malformed["boundaries"])
    elif mutation == "decision":
        site = next(row for row in malformed["boundaries"] if row["router_site"])
        site["router_site"] = False
        malformed["boundaries_sha256"] = stable_json_sha256(malformed["boundaries"])
    else:
        malformed["stop_reason"] = "max_new"
    with pytest.raises(ValueError):
        _validate_generation_evidence(
            malformed,
            prompt,
            response,
            True,
            tokenizer,
            4096,
            CANONICAL_MAX_NEW,
            "mutated generation",
        )


def _oracle_direct_ask(prompt):
    state = dws_prompt_state(prompt)
    if state is None:
        for line in prompt.splitlines():
            for marker in ("Current state: ", "State: ", "Machine record: "):
                if line.startswith(marker):
                    state = parse_state(line[len(marker) :])
                    break
            if state is not None:
                break
    assert state is not None
    if state["z"]:
        return "answer={}".format(state_answer(state))
    return canonical_state(apply_microstep(state))


def test_direct_board_contains_and_executes_all_frozen_interaction_modes():
    cases = fresh_direct_cases()
    assert len(cases) == 12
    assert collections.Counter(case["mode"] for case in cases) == {
        "complete": 4,
        "terminal": 2,
        "source_deleted": 2,
        "state_reuse": 2,
        "review": 2,
    }
    for case in cases:
        result = evaluate_direct_case(case, _oracle_direct_ask)
        assert result["success"], case["id"]


def test_canonical_development_selection_identity_is_frozen():
    text = (ROOT / "artifacts/evals/digitwise_recurrent_v2_heldout.jsonl").read_text()
    selection, contract = canonical_development_selection(text)
    episodes = [entry["episode"] for entry in selection]
    validate_development_selection_contract(contract, episodes)
    assert len(selection) == CANONICAL_DEVELOPMENT_EPISODES
    assert contract["regime_counts"] == {
        "fit_width": 100,
        "value_ood": 100,
        "width_8": 100,
    }
    assert contract["identity_sha256"] == EXPECTED_DEVELOPMENT_SELECTION_SHA256
    assert contract["identities"][0] == {
        "index": 0,
        "source_index": 0,
        "id": "fit_w4-00000",
        "source_split": "fit_w4",
        "regime": "fit_width",
    }
    assert contract["identities"][-1]["id"] == "width_ood_w8-00099"

    mutated = copy.deepcopy(contract)
    mutated["identities"][16]["id"] = "substituted-episode"
    mutated["identity_sha256"] = stable_json_sha256(mutated["identities"])
    with pytest.raises(ValueError, match="not canonical"):
        validate_development_selection_contract(mutated)


_CLOSED_WORLD_DEVELOPMENT_EVAL_CACHE = None


def _closed_world_development_eval_fixture():
    global _CLOSED_WORLD_DEVELOPMENT_EVAL_CACHE
    if _CLOSED_WORLD_DEVELOPMENT_EVAL_CACHE is not None:
        cached_result, cached_context = _CLOSED_WORLD_DEVELOPMENT_EVAL_CACHE
        return copy.deepcopy(cached_result), cached_context
    tokenizer = _tokenizer_fixture()
    sequence_cap = 4096
    heldout = (
        ROOT / "artifacts/evals/digitwise_recurrent_v2_heldout.jsonl"
    ).read_text()
    _selection, development_selection = canonical_development_selection(heldout)
    template = next(case for case in fresh_direct_cases() if case["mode"] == "terminal")
    episodes = []
    for identity in development_selection["identities"]:
        episode = copy.deepcopy(template)
        episode["id"] = identity["id"]
        episode["split"] = identity["source_split"]
        episodes.append(episode)

    arms = ("base", "dead", "treatment", "shuffled")
    by_arm = {}
    cycle_state = initial_state("add", 99, 1, 2)
    cycle_first = apply_microstep(cycle_state)
    cycle_second = apply_microstep(cycle_first)
    cycle_contract = {
        "records": CANONICAL_CYCLE_CASES,
        "cases": [
            {
                "index": index,
                "episode_id": "cycle-episode-{:03d}".format(index),
                "first_prompt": microstep_prompt(cycle_state, style="core"),
                "expected_first": cycle_first,
                "second_prompt": microstep_prompt(cycle_first, style="core"),
                "expected_second": cycle_second,
            }
            for index in range(CANONICAL_CYCLE_CASES)
        ],
    }

    def raw_call(
        index,
        kind,
        prompt,
        response,
        arm,
        max_new=CANONICAL_MAX_NEW,
        retain_full_logits=False,
    ):
        generation, sites, fires = _generation_fixture(
            tokenizer,
            prompt,
            response,
            arm != "base",
            sequence_cap=sequence_cap,
            max_new=max_new,
            retain_full_logits=retain_full_logits,
        )
        return _raw_call_record(index, kind, prompt, response, sites, fires, generation)

    for arm in arms:
        accounting = []
        evidence = []
        for identity, episode in zip(development_selection["identities"], episodes):
            calls = []

            def ask(prompt):
                response = _oracle_direct_ask(prompt)
                calls.append(
                    raw_call(
                        len(calls),
                        "transition"
                        if dws_prompt_state(prompt) is not None
                        else "final",
                        prompt,
                        response,
                        arm,
                    )
                )
                return response

            rollout_episode(episode, ask, prompt_style=episode["prompt_style"])
            row = _episode_evidence_record(identity, calls)
            evidence.append(row)
            accounting.append(
                _derive_episode_accounting_from_evidence(
                    row,
                    episode,
                    identity,
                    arm,
                    tokenizer,
                    sequence_cap,
                    CANONICAL_MAX_NEW,
                    "fixture episode",
                )
            )
        cycle_cases = []
        for expected in cycle_contract["cases"]:
            calls = [
                raw_call(
                    0,
                    "first",
                    expected["first_prompt"],
                    canonical_state(cycle_first),
                    arm,
                ),
                raw_call(
                    1,
                    "second",
                    expected["second_prompt"],
                    canonical_state(cycle_second),
                    arm,
                ),
            ]
            cycle_cases.append(
                {
                    "index": expected["index"],
                    "episode_id": expected["episode_id"],
                    "first_exact": True,
                    "second_exact": True,
                    "integrated_two_call_exact": True,
                    "site_opportunities": sum(
                        call["site_opportunities"] for call in calls
                    ),
                    "motor_fires": sum(call["motor_fires"] for call in calls),
                    "calls": calls,
                }
            )
        cycle_sites = sum(row["site_opportunities"] for row in cycle_cases)
        cycle_fires = sum(row["motor_fires"] for row in cycle_cases)
        by_arm[arm] = {
            "by_regime": _aggregate_episode_accounting(accounting),
            "episode_accounting": accounting,
            "episode_evidence": evidence,
            "transcripts": copy.deepcopy(evidence[:15]),
            "cycle": {
                "records": CANONICAL_CYCLE_CASES,
                "first_exact": CANONICAL_CYCLE_CASES,
                "second_exact_after_first": CANONICAL_CYCLE_CASES,
                "integrated_two_call_exact": CANONICAL_CYCLE_CASES,
                "site_opportunities": cycle_sites,
                "motor_fires": cycle_fires,
                "cases": cycle_cases,
            },
        }

    direct_cases = fresh_direct_cases()
    direct = {}
    for arm in arms:
        rows = []
        for case in direct_cases:
            calls = []

            def ask(prompt):
                response = _oracle_direct_ask(prompt)
                kind = (
                    "transition"
                    if dws_prompt_state(prompt) is not None
                    else "review"
                    if prompt.startswith("Review this proposed")
                    else "final"
                )
                calls.append(raw_call(len(calls), kind, prompt, response, arm))
                return response

            evaluation = evaluate_direct_case(case, ask)
            site_opportunities = sum(call["site_opportunities"] for call in calls)
            motor_fires = sum(call["motor_fires"] for call in calls)
            rows.append(
                {
                    "id": case["id"],
                    "mode": case["mode"],
                    "expected_answer": case["expected_answer"],
                    "success": evaluation["success"],
                    "site_opportunities": site_opportunities,
                    "motor_fires": motor_fires,
                    "evaluation": evaluation,
                    "calls": calls,
                }
            )
        direct[arm] = {
            "correct": sum(int(row["success"]) for row in rows),
            "rows": rows,
        }

    expected_dev_rows = []
    zero_id = tokenizer.token_to_id("0")
    one_id = tokenizer.token_to_id("1")
    for index in range(16):
        target = index % 2
        state = initial_state("add", 99 if target else 11, 1, 2)
        next_state = apply_microstep(state)
        assert next_state["c"] == target
        prompt = microstep_prompt(state, style="core")
        response_prefix = "dws:op=add;w=2;p=1;c="
        prompt_ids = tokenizer.encode(prompt).ids
        prefix_ids = prompt_ids + tokenizer.encode(response_prefix).ids
        expected_dev_rows.append(
            {
                "index": index,
                "selection_index": index,
                "source_index": index,
                "selected_episode_id": "teacher-selected-{:02d}".format(index),
                "source_split": "fit_w4",
                "branch": "factual",
                "operation": "add",
                "width": 2,
                "position": 0,
                "style": "core",
                "current_carry": 0,
                "target": target,
                "target_id": zero_id if target == 0 else one_id,
                "prompt_ids": prompt_ids,
                "prefix_ids": prefix_ids,
                "prompt": prompt,
                "response_prefix": response_prefix,
                "episode_id": "teacher-episode-{:02d}".format(index),
                "regime": "fit_width",
                "transition": 0,
            }
        )
    other_ids = torch.tensor(
        [token_id for token_id in range(6) if token_id not in {zero_id, one_id}],
        dtype=torch.long,
    )
    base01 = torch.empty((len(expected_dev_rows), 2), dtype=torch.bfloat16)
    for index, row in enumerate(expected_dev_rows):
        base01[index] = torch.tensor(
            [3.0, 1.0] if row["target"] == 0 else [1.0, 3.0],
            dtype=torch.bfloat16,
        )
    teacher_features = {
        "hidden": torch.zeros((len(expected_dev_rows), 2)),
        "base01": base01,
        "other_max": torch.zeros(len(expected_dev_rows)),
        "other_max_token_id": torch.full(
            (len(expected_dev_rows),), int(other_ids[0]), dtype=torch.long
        ),
        "other_logits": torch.zeros((len(expected_dev_rows), len(other_ids))),
        "other_token_ids": other_ids,
        "zero_id": zero_id,
        "one_id": one_id,
    }
    dead_motor = CarryMotor(2, rank=8)
    teacher_forced = {
        arm: teacher_forced_metric_evidence(
            teacher_features,
            expected_dev_rows,
            None if arm == "base" else dead_motor,
            arm,
            "cpu",
            CANONICAL_TEACHER_SCORING_CONTRACT,
        )
        for arm in arms
    }
    frozen = {"checkpoint": "frozen"}
    source_contract = {"git_commit": "a" * 40, "manifest_sha256": "b" * 64}
    motor_path = "/tmp/canonical-motor.pt"
    motor_sha256 = "c" * 64
    plan_sha256 = "d" * 64
    preservation = []
    for prompt in NON_DWS_PRESERVATION_PROMPTS:
        base_call = raw_call(
            0,
            "preservation",
            prompt,
            "same response",
            "base",
            max_new=48,
            retain_full_logits=True,
        )
        motor_call = raw_call(
            0,
            "preservation",
            prompt,
            "same response",
            "treatment",
            max_new=48,
            retain_full_logits=True,
        )
        preservation.append(
            {
                "prompt": prompt,
                "base_response": "same response",
                "motor_response": "same response",
                "token_ids_identical": True,
                "full_logits_identical": True,
                "exact_identity": True,
                "base_sites": base_call["site_opportunities"],
                "base_fires": base_call["motor_fires"],
                "motor_sites": motor_call["site_opportunities"],
                "motor_fires": motor_call["motor_fires"],
                "base_call": base_call,
                "motor_call": motor_call,
            }
        )
    result = {
        "audit": CANONICAL_EVAL_AUDIT,
        "checkpoint_step": CANONICAL_CHECKPOINT_STEP,
        "frozen_sha256": frozen,
        "source_contract": source_contract,
        "motor": str(Path(motor_path).resolve()),
        "motor_sha256": motor_sha256,
        "plan_sha256": plan_sha256,
        "development_selection": development_selection,
        "max_new": CANONICAL_MAX_NEW,
        "extract_batch": CANONICAL_EXTRACT_BATCH,
        "teacher_forced_carry": teacher_forced,
        "results": by_arm,
        "fresh_direct": direct,
        "non_dws_preservation": preservation,
        "claim_boundary": CANONICAL_EVAL_CLAIM_BOUNDARY,
    }
    context = {
        "frozen": frozen,
        "source_contract": source_contract,
        "checkpoint_step": CANONICAL_CHECKPOINT_STEP,
        "motor_path": motor_path,
        "motor_sha256": motor_sha256,
        "plan_sha256": plan_sha256,
        "expected_dev_rows": expected_dev_rows,
        "tokenizer": tokenizer,
        "sequence_cap": sequence_cap,
        "expected_development_selection": development_selection,
        "expected_episodes": episodes,
        "expected_cycle_contract": cycle_contract,
        "expected_direct_cases": direct_cases,
    }
    _CLOSED_WORLD_DEVELOPMENT_EVAL_CACHE = (copy.deepcopy(result), context)
    return result, context


def _validate_development_fixture(result, context):
    validate_development_eval_result(result, **context)


def test_development_eval_closed_world_fixture_is_admitted():
    result, context = _closed_world_development_eval_fixture()
    _validate_development_fixture(result, context)


@pytest.mark.parametrize("mutated_step", [200000, "sft_ep2"])
def test_development_publication_binds_canonical_string_checkpoint_step(mutated_step):
    result, context = _closed_world_development_eval_fixture()
    assert type(result["checkpoint_step"]) is str
    assert result["checkpoint_step"] == CANONICAL_CHECKPOINT_STEP
    _validate_development_fixture(result, context)

    rewritten = copy.deepcopy(result)
    rewritten_context = dict(context)
    rewritten["checkpoint_step"] = mutated_step
    rewritten_context["checkpoint_step"] = mutated_step
    with pytest.raises(ValueError, match="identity or budget mismatch"):
        _validate_development_fixture(rewritten, rewritten_context)


def test_development_eval_rejects_empty_and_stripped_nested_rows():
    result, context = _closed_world_development_eval_fixture()
    corruptions = []

    malformed = copy.deepcopy(result)
    malformed["results"]["base"]["by_regime"]["fit_width"] = {"episodes": 100}
    corruptions.append((malformed, "regime fit_width schema mismatch"))

    malformed = copy.deepcopy(result)
    malformed["results"]["base"]["transcripts"] = []
    corruptions.append((malformed, "transcript sample is invalid"))

    malformed = copy.deepcopy(result)
    malformed["results"]["base"]["transcripts"][0] = {}
    corruptions.append((malformed, "transcript sample is invalid"))

    malformed = copy.deepcopy(result)
    malformed["results"]["base"]["episode_evidence"] = []
    corruptions.append((malformed, "episode evidence is incomplete"))

    malformed = copy.deepcopy(result)
    malformed["results"]["base"]["episode_evidence"][0] = {}
    corruptions.append((malformed, r"episode_evidence\[0\] schema mismatch"))

    malformed = copy.deepcopy(result)
    malformed["results"]["base"]["episode_accounting"] = []
    corruptions.append((malformed, "episode accounting is incomplete"))

    malformed = copy.deepcopy(result)
    malformed["results"]["base"]["episode_accounting"][0] = {}
    corruptions.append((malformed, r"episode_accounting\[0\] schema mismatch"))

    malformed = copy.deepcopy(result)
    malformed["results"]["base"]["cycle"]["cases"] = []
    corruptions.append((malformed, "cycle case evidence is incomplete"))

    malformed = copy.deepcopy(result)
    malformed["results"]["base"]["cycle"]["cases"][0] = {}
    corruptions.append((malformed, r"cycle base cases\[0\] schema mismatch"))

    malformed = copy.deepcopy(result)
    malformed["fresh_direct"]["base"]["rows"] = []
    corruptions.append((malformed, "direct rows mismatch"))

    malformed = copy.deepcopy(result)
    direct_row = malformed["fresh_direct"]["base"]["rows"][0]
    malformed["fresh_direct"]["base"]["rows"][0] = {
        "id": direct_row["id"],
        "success": direct_row["success"],
    }
    corruptions.append((malformed, r"direct rows\[0\] schema mismatch"))

    malformed = copy.deepcopy(result)
    malformed["non_dws_preservation"] = []
    corruptions.append((malformed, "preservation set mismatch"))

    malformed = copy.deepcopy(result)
    malformed["non_dws_preservation"][0] = {
        "exact_identity": True,
        "base_sites": 0,
        "motor_sites": 0,
        "motor_fires": 0,
    }
    corruptions.append((malformed, r"preservation\[0\] schema mismatch"))

    for malformed, message in corruptions:
        with pytest.raises(ValueError, match=message):
            _validate_development_fixture(malformed, context)


def test_development_eval_rejects_count_and_dead_base_inconsistency():
    result, context = _closed_world_development_eval_fixture()

    malformed = copy.deepcopy(result)
    malformed["results"]["base"]["by_regime"]["fit_width"]["transition_correct"] += 1
    with pytest.raises(ValueError, match="correctness counts are inconsistent"):
        _validate_development_fixture(malformed, context)

    malformed = copy.deepcopy(result)
    malformed["teacher_forced_carry"]["dead"]["summary"]["prediction_ones"] -= 1
    with pytest.raises(ValueError, match="aggregate differs from raw row evidence"):
        _validate_development_fixture(malformed, context)

    malformed = copy.deepcopy(result)
    dead = malformed["results"]["dead"]
    index = next(
        index
        for index, item in enumerate(dead["episode_accounting"])
        if item["regime"] == "width_8"
    )
    evidence = dead["episode_evidence"][index]
    prompt = evidence["calls"][0]["prompt"]
    generation, sites, fires = _generation_fixture(
        context["tokenizer"],
        prompt,
        "",
        True,
        sequence_cap=context["sequence_cap"],
    )
    evidence["calls"] = [
        _raw_call_record(0, "transition", prompt, "", sites, fires, generation)
    ]
    identity = context["expected_development_selection"]["identities"][index]
    dead["episode_accounting"][index] = _derive_episode_accounting_from_evidence(
        evidence,
        context["expected_episodes"][index],
        identity,
        "dead",
        context["tokenizer"],
        context["sequence_cap"],
        CANONICAL_MAX_NEW,
        "mutated dead fixture",
    )
    dead["by_regime"] = _aggregate_episode_accounting(dead["episode_accounting"])
    with pytest.raises(ValueError, match="autonomous dead/base collapse failed"):
        _validate_development_fixture(malformed, context)

    malformed = copy.deepcopy(result)
    direct_row = malformed["fresh_direct"]["dead"]["rows"][0]
    direct_row["site_opportunities"] -= 1
    direct_row["motor_fires"] -= 1
    with pytest.raises(ValueError, match="aggregate differs from raw calls"):
        _validate_development_fixture(malformed, context)


def test_teacher_forced_target_prediction_and_evidence_deletion_fail():
    result, context = _closed_world_development_eval_fixture()
    malformed = copy.deepcopy(result)
    report = malformed["teacher_forced_carry"]["treatment"]
    row = report["rows"][0]
    row["target"] = 1 - row["target"]
    row["target_token_id"] = report["one_id"] if row["target"] else report["zero_id"]
    row["carry_prediction"] = row["target"]
    row["global_prediction_token_id"] = row["target_token_id"]
    row["carry_pair_correct"] = True
    row["global_correct"] = True
    report["summary"] = _aggregate_teacher_rows(report["rows"])
    with pytest.raises(ValueError, match="target differs from frozen row"):
        _validate_development_fixture(malformed, context)

    malformed = copy.deepcopy(result)
    report = malformed["teacher_forced_carry"]["shuffled"]
    row = report["rows"][1]
    row["carry_prediction"] = 1 - row["carry_prediction"]
    row["global_prediction_token_id"] = report["zero_id"]
    row["carry_pair_correct"] = not row["carry_pair_correct"]
    row["global_correct"] = row["global_prediction_token_id"] == row["target_token_id"]
    report["summary"] = _aggregate_teacher_rows(report["rows"])
    with pytest.raises(
        ValueError, match="prediction differs from complete top-one evidence"
    ):
        _validate_development_fixture(malformed, context)

    malformed = copy.deepcopy(result)
    report = malformed["teacher_forced_carry"]["treatment"]
    del report["rows"][0]["other_max_token_id"]
    with pytest.raises(ValueError, match=r"teacher-forced treatment rows\[0\] schema"):
        _validate_development_fixture(malformed, context)

    malformed = copy.deepcopy(result)
    malformed["teacher_forced_carry"]["treatment"]["scoring_contract"] = (
        "cpu_all_row_head_v1"
    )
    with pytest.raises(ValueError, match="singleton H100 scoring contract"):
        _validate_development_fixture(malformed, context)


def test_direct_router_claim_rewrite_with_recomputed_row_total_fails():
    result, context = _closed_world_development_eval_fixture()
    malformed = copy.deepcopy(result)
    row = malformed["fresh_direct"]["treatment"]["rows"][0]
    call = next(call for call in row["calls"] if call["site_opportunities"] == 1)
    call["site_opportunities"] = 0
    call["motor_fires"] = 0
    row["site_opportunities"] = sum(item["site_opportunities"] for item in row["calls"])
    row["motor_fires"] = sum(item["motor_fires"] for item in row["calls"])
    with pytest.raises(ValueError, match="router counts differ"):
        _validate_development_fixture(malformed, context)


@pytest.mark.parametrize("regime", ["fit_width", "value_ood", "width_8"])
def test_development_eval_rejects_sampled_and_unsampled_aggregate_contradictions(
    regime,
):
    result, context = _closed_world_development_eval_fixture()
    malformed = copy.deepcopy(result)
    malformed["results"]["base"]["by_regime"][regime]["site_opportunities"] -= 1
    with pytest.raises(ValueError, match="totals differ from raw evidence"):
        _validate_development_fixture(malformed, context)


def test_development_eval_binds_sampled_transcript_to_episode_accounting():
    result, context = _closed_world_development_eval_fixture()
    malformed = copy.deepcopy(result)
    transcript = malformed["results"]["base"]["transcripts"][0]
    transcript["calls"][0]["site_opportunities"] = 0
    with pytest.raises(ValueError, match="transcript sample is invalid"):
        _validate_development_fixture(malformed, context)


def test_unsampled_treatment_compact_rewrite_with_recomputed_totals_fails():
    result, context = _closed_world_development_eval_fixture()
    malformed = copy.deepcopy(result)
    treatment = malformed["results"]["treatment"]
    index = 20
    assert index >= len(treatment["transcripts"])
    treatment["episode_accounting"][index].update(
        {
            "transition_attempted": 1,
            "first_transition_correct": False,
            "transition_correct": 0,
            "state_closed_loop_correct": False,
            "final_answer_correct": False,
            "site_opportunities": 0,
            "motor_fires": 0,
        }
    )
    treatment["by_regime"] = _aggregate_episode_accounting(
        treatment["episode_accounting"]
    )
    with pytest.raises(ValueError, match="accounting differs from raw evidence"):
        _validate_development_fixture(malformed, context)


def test_cycle_recomputes_every_call_and_rejects_two_call_one_site_claim():
    result, context = _closed_world_development_eval_fixture()
    cycle = result["results"]["treatment"]["cycle"]
    assert len(cycle["cases"]) == CANONICAL_CYCLE_CASES
    assert all(len(case["calls"]) == 2 for case in cycle["cases"])
    assert all(
        call["site_opportunities"] == 1
        for case in cycle["cases"]
        for call in case["calls"]
    )

    malformed = copy.deepcopy(result)
    cycle = malformed["results"]["treatment"]["cycle"]
    cycle["cases"][0]["site_opportunities"] = 1
    cycle["cases"][0]["motor_fires"] = 1
    cycle["site_opportunities"] -= 1
    cycle["motor_fires"] -= 1
    with pytest.raises(ValueError, match="aggregate differs from raw calls"):
        _validate_development_fixture(malformed, context)


def test_development_report_rejects_selection_identity_mutation():
    result, context = _closed_world_development_eval_fixture()
    malformed = copy.deepcopy(result)
    selection = malformed["development_selection"]
    selection["identities"][17]["id"] = "mutated-selection-id"
    selection["identity_sha256"] = stable_json_sha256(selection["identities"])
    with pytest.raises(ValueError, match="not canonical"):
        _validate_development_fixture(malformed, context)


def test_source_manifest_is_path_and_digest_bound():
    observed = {
        "checkpoint": "ignored",
        "source:a.py": "1",
        "source:b.py": "2",
    }
    first = source_manifest_sha256(observed)
    assert first == source_manifest_sha256(dict(reversed(list(observed.items()))))
    changed = dict(observed)
    changed["source:b.py"] = "3"
    assert source_manifest_sha256(changed) != first


def test_source_contract_and_slurm_allowlist_cover_every_scientific_file(monkeypatch):
    observed = {
        "source:{}".format(path): str(index)
        for index, path in enumerate(SCIENTIFIC_SOURCE_PATHS)
    }
    manifest = source_manifest_sha256(observed)
    args = SimpleNamespace(source_commit="commit", source_manifest_sha256=manifest)
    monkeypatch.setattr(
        "causal_carry_motor.verify_reviewed_source_commit",
        lambda commit, sources: None,
    )
    assert validate_source_contract(args, observed, canonical=True) == {
        "git_commit": "commit",
        "manifest_sha256": manifest,
    }
    with pytest.raises(ValueError, match="manifest mismatch"):
        validate_source_contract(
            SimpleNamespace(source_commit="commit", source_manifest_sha256="bad"),
            observed,
            canonical=True,
        )
    wrapper = (ROOT / "train" / "jobs" / "causal_carry_motor.sbatch").read_text()
    plan_wrapper = (
        ROOT / "pipeline" / "jobs" / "causal_carry_motor_plan_stokes.sbatch"
    ).read_text()
    for path in SCIENTIFIC_SOURCE_PATHS:
        assert path in wrapper
        assert path in plan_wrapper
    assert "carry_motor_%j_%r.out" in wrapper
    assert "carry_motor_%j_%r.err" in wrapper
    assert "#SBATCH --gres=gpu:nvidia_h100_pcie:1" in wrapper
    assert 'scontrol show job -o "$SLURM_JOB_ID"' in wrapper
    assert '" Requeue=0 "' in wrapper
    assert "refusing restarted carry-motor job" in wrapper
    assert "git status --porcelain --untracked-files=all" in wrapper
    assert 'cmp --silent -- "$0"' in wrapper
    assert "stat -c '%a' \"$PLAN_ROOT\"" in wrapper
    assert "stat -c '%h' \"$PLAN\"" in wrapper
    assert "CONFIRMATION_COMMITMENT_SHA256" in wrapper
    assert "validate-confirmation" in wrapper
    assert 'CUDA_VISIBLE_DEVICES= "$PY"' in wrapper
    assert wrapper.index("validate-confirmation") < wrapper.index("nvidia-smi")
    assert wrapper.index("validate-confirmation") < wrapper.index(
        "torch.cuda.is_available"
    )
    assert "for index in {00..07}" in wrapper
    assert "shard=$shard_directory/features.pt" in wrapper
    assert "stat -c '%h' \"$shard\"" in wrapper
    assert "canonical fit shard preflight failed" in wrapper
    assert '"$MODE" == confirmation-eval' in wrapper
    assert "--confirmation-secret-file" in wrapper
    assert "stat -c '%s' \"$CONFIRMATION_SECRET_FILE\"" in wrapper
    assert 'cmp --silent -- "$0"' in plan_wrapper
    assert "stat -c '%a' \"$PLAN_ROOT\"" in plan_wrapper
    assert "stat -c '%h' \"$PLAN_ROOT/plan.json\"" in plan_wrapper
    assert "EXPECTED_CONFIRMATION_COMMITMENT" in plan_wrapper
    assert "--confirmation-commitment-sha256" in plan_wrapper
    assert '"$PLAN_ROOT/confirmation_eval"' in plan_wrapper
