#!/usr/bin/env python3
"""CPU contracts for the grammar-gated causal carry motor."""

from __future__ import annotations

import collections
import copy
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train"))

from causal_carry_motor import (  # noqa: E402
    CarryMotor,
    CarryRouter,
    BoundInput,
    CANONICAL_BATCH,
    CANONICAL_EXTRACT_BATCH,
    CANONICAL_LR,
    CANONICAL_MAX_NEW,
    CANONICAL_PER_REGIME,
    CANONICAL_UPDATES,
    CANONICAL_WEIGHT_DECAY,
    FIT_QUOTA,
    SCIENTIFIC_SOURCE_PATHS,
    apply_microstep,
    apply_motor_logits,
    atomic_json,
    canonical_state,
    dws_prompt_state,
    evaluate_direct_case,
    extract_frozen_features,
    feature_metrics,
    fresh_direct_cases,
    full_vocab_motor_loss,
    generate_fit_rows,
    initial_state,
    is_carry_site,
    microstep_prompt,
    parse_state,
    permuted_control_labels,
    prepare_output,
    seal_output_directory,
    source_manifest_sha256,
    state_answer,
    tensor_state_sha256,
    validate_artifact_receipt,
    validate_canonical_eval_args,
    validate_canonical_train_args,
    validate_motor_bundle,
    validate_source_contract,
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
    assert metrics["mean_target_rank_global"] == 2.0


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
    assert metrics["mean_target_rank_global"] == 3.0
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
        root.mkdir()
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
        "per_regime": CANONICAL_PER_REGIME,
        "max_new": CANONICAL_MAX_NEW,
        "extract_batch": CANONICAL_EXTRACT_BATCH,
        "motor_sha256": "a" * 64,
    }
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
        "per_regime": CANONICAL_PER_REGIME - 1,
        "max_new": CANONICAL_MAX_NEW - 1,
        "extract_batch": 2,
        "motor_sha256": "",
    }.items():
        with pytest.raises(SystemExit):
            validate_canonical_eval_args(_canonical_eval_args(**{name: value}), True)


def test_bundle_receipt_and_state_hash_corruption_fail_closed():
    treatment = CarryMotor(4, rank=2).state_dict()
    shuffled = CarryMotor(4, rank=2).state_dict()
    bindings = {
        "base_checkpoint_sha256": "base",
        "tokenizer_sha256": "tokenizer",
        "episodes_sha256": "episodes",
        "cycle_sha256": "cycle",
    }
    sources = {"train/causal_carry_motor.py": "source"}
    source_contract = {"git_commit": "commit", "manifest_sha256": "manifest"}
    bundle = {
        "audit": "causal_carry_motor_fit_v2",
        **bindings,
        "scientific_source_sha256": sources,
        "source_contract": source_contract,
        "extract_batch": CANONICAL_EXTRACT_BATCH,
        "deployment_logit_dtype": "torch.bfloat16",
        "treatment": treatment,
        "shuffled": shuffled,
        "treatment_state_sha256": tensor_state_sha256(treatment),
        "shuffled_state_sha256": tensor_state_sha256(shuffled),
    }
    validate_motor_bundle(bundle, bindings, sources, source_contract)
    bad_dtype = copy.deepcopy(bundle)
    bad_dtype["deployment_logit_dtype"] = "torch.float64"
    with pytest.raises(ValueError, match="deployment logit dtype"):
        validate_motor_bundle(bad_dtype, bindings, sources, source_contract)
    for arm in ("treatment", "shuffled"):
        corrupted = copy.deepcopy(bundle)
        first = next(iter(corrupted[arm].values()))
        first.view(-1)[0] += 1
        with pytest.raises(ValueError, match="state hash mismatch"):
            validate_motor_bundle(corrupted, bindings, sources, source_contract)
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
    for path in SCIENTIFIC_SOURCE_PATHS:
        assert path in wrapper
    assert "carry_motor_%j_%r.out" in wrapper
    assert "carry_motor_%j_%r.err" in wrapper
    assert "#SBATCH --gres=gpu:nvidia_h100_pcie:1" in wrapper
    assert 'scontrol show job -o "$SLURM_JOB_ID"' in wrapper
    assert '" Requeue=0 "' in wrapper
    assert "refusing restarted carry-motor job" in wrapper
