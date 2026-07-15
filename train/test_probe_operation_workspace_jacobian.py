#!/usr/bin/env python3
"""Strict CPU/static contracts for the frozen operation Jacobian diagnostic."""
from __future__ import annotations

import json
import re
import sys
import tempfile
from pathlib import Path

import torch
from tokenizers import Tokenizer


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train"))

from model import GPT, GPTConfig  # noqa: E402
from probe_operation_workspace_jacobian import (  # noqa: E402
    CANDIDATES,
    CANDIDATE_TOKEN_IDS,
    CANDIDATE_TOKEN_TEXT,
    CASE_SPECS,
    DONOR,
    EXPECTED_CHECKPOINT_SHA256,
    EXPECTED_TOKENIZER_SHA256,
    FROZEN_SPEC_SHA256,
    MAX_RELATIVE_SWAP_L2,
    MIN_RELATIVE_SWAP_L2,
    PERMUTED_LABEL,
    READOUT_LAYERS,
    REALIZED_NORM_MATCH_RTOL,
    SHAM_PAIR,
    SOURCE_FREEZE_SHA256,
    SOURCE_LAYERS,
    aggregate_readout_scores,
    apply_operation,
    atomic_write_json,
    binomial_tail,
    build_direction_records,
    build_evaluation_records,
    build_parser,
    canonical_source_sha256,
    case_trajectory,
    classify_outcome,
    coordinate_swap_delta,
    frozen_spec_payload,
    future_logit_gradients,
    norm_match_and_bound,
    normalized,
    patched_candidate_logits,
    sha256_file,
    stable_json_sha256,
    summarize_split,
    validate_case_specs,
    validate_tokenizer,
    verify_file_hash,
    verify_frozen_contract,
)


def assert_close(left, right, tolerance=1e-6):
    assert abs(left - right) <= tolerance, (left, right)


def static_contract_tests():
    assert EXPECTED_CHECKPOINT_SHA256 == "91d5288f184fc5230516add9851ac1a8815d3369ffd816cd7d0c03d8bafc741d"
    assert EXPECTED_TOKENIZER_SHA256 == "87532df5c121753de3b29194e1f9e3de47986d3f5359548fdf93606773a233d4"
    assert SOURCE_LAYERS == (5, 9, 13, 17, 21, 25, 28)
    assert READOUT_LAYERS == (13, 17, 21)
    assert MAX_RELATIVE_SWAP_L2 == 0.05
    assert MIN_RELATIVE_SWAP_L2 == 1e-4
    assert REALIZED_NORM_MATCH_RTOL == 0.02
    assert len(CASE_SPECS) == 24
    assert stable_json_sha256(frozen_spec_payload()) == FROZEN_SPEC_SHA256
    assert canonical_source_sha256() == SOURCE_FREEZE_SHA256
    verify_frozen_contract()
    validate_case_specs()

    parser = build_parser()
    destinations = {action.dest for action in parser._actions}
    assert destinations == {"help", "ckpt", "tokenizer", "out", "device"}
    assert not any(destination in destinations for destination in ("layers", "alpha", "threshold", "seed"))

    prereg = ROOT / "R12_OPERATION_WORKSPACE_JACOBIAN_PREREG.md"
    prereg_text = prereg.read_text()
    assert "| Probe canonical freeze SHA-256 | `{}` |".format(SOURCE_FREEZE_SHA256) in prereg_text
    assert "| Frozen spec SHA-256 | `{}` |".format(FROZEN_SPEC_SHA256) in prereg_text
    file_match = re.search(r"\| Probe file SHA-256 \| `([0-9a-f]{64})` \|", prereg_text)
    assert file_match and file_match.group(1) == sha256_file(ROOT / "train" / "probe_operation_workspace_jacobian.py")

    wrapper = ROOT / "train" / "jobs" / "probe_operation_workspace_jacobian.sbatch"
    wrapper_text = wrapper.read_text()
    assert "sbatch " not in wrapper_text
    assert "sft.py" not in wrapper_text
    assert "train.py" not in wrapper_text
    assert "flagship_out" in wrapper_text
    assert EXPECTED_CHECKPOINT_SHA256 in wrapper_text
    assert EXPECTED_TOKENIZER_SHA256 in wrapper_text
    assert sha256_file(ROOT / "train" / "probe_operation_workspace_jacobian.py") in wrapper_text
    assert sha256_file(prereg) in wrapper_text


def tokenizer_and_board_tests():
    tokenizer_path = ROOT / "artifacts" / "shohin-tok-32k.json"
    verify_file_hash(tokenizer_path, EXPECTED_TOKENIZER_SHA256, "test tokenizer")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    validate_tokenizer(tokenizer)
    for operation in CANDIDATES:
        assert tokenizer.encode(CANDIDATE_TOKEN_TEXT[operation]).ids == [CANDIDATE_TOKEN_IDS[operation]]

    directions = build_direction_records(tokenizer)
    evaluations = build_evaluation_records(tokenizer)
    assert len(directions) == 24
    assert len(evaluations) == 24
    assert len({row["id"] for row in directions}) == 24
    assert len({row["id"] for row in evaluations}) == 24
    assert {row["source_sha256"] for row in directions}.isdisjoint(
        row["source_sha256"] for row in evaluations
    )
    primary = [row for row in evaluations if row["split"] == "heldout_primary"]
    replication = [row for row in evaluations if row["split"] == "heldout_replication"]
    assert len(primary) == len(replication) == 12
    assert {row["source_sha256"] for row in primary}.isdisjoint(
        row["source_sha256"] for row in replication
    )
    for rows in (primary, replication):
        assert {operation: sum(row["correct_operation"] == operation for row in rows)
                for operation in CANDIDATES} == {operation: 3 for operation in CANDIDATES}
    for row in directions + evaluations:
        assert row["anchor_index"] < row["target_logit_index"] < len(row["input_ids"])
    for row in evaluations:
        assert row["source_candidate_token_counts"] == {operation: 1 for operation in CANDIDATES}
        assert row["donor_operation"] == DONOR[row["correct_operation"]]
        assert tuple(row["sham_pair"]) == SHAM_PAIR[row["correct_operation"]]
        assert row["current_state"] == row["trajectory"][row["completed_prefix_length"]]
        assert row["correct_operation"] == row["plan"][row["completed_prefix_length"]][0]
        assert len(set(row["trajectory"])) == 5

    assert apply_operation(7, "add", 5) == 12
    assert apply_operation(7, "multiply", 5) == 35
    assert apply_operation(7, "subtract", 9) == -2
    assert apply_operation(37, "remainder", 9) == 1
    assert case_trajectory(7, (("add", 3), ("multiply", 2))) == [7, 10, 20]
    try:
        apply_operation(1, "remainder", 0)
    except ValueError:
        pass
    else:
        raise AssertionError("zero remainder operand must fail")


def vector_and_statistics_tests():
    hidden = torch.tensor([3.0, 1.0, 4.0, 2.0])
    basis = torch.eye(4)
    signal_raw = coordinate_swap_delta(hidden, basis[0], basis[1])
    sham_raw = coordinate_swap_delta(hidden, basis[2], basis[3])
    assert torch.allclose(hidden + signal_raw, torch.tensor([1.0, 3.0, 4.0, 2.0]))
    assert torch.allclose(signal_raw[2:], torch.zeros(2))
    signal, sham, bound = norm_match_and_bound(signal_raw, sham_raw, hidden)
    assert_close(float(signal.norm()), float(sham.norm()))
    assert bound["relative_delta_l2"] <= MAX_RELATIVE_SWAP_L2 + 1e-7
    assert_close(float(signal.norm()) / float(hidden.norm()), bound["relative_delta_l2"])

    hidden_by_layer = {layer: torch.tensor([1.0, 0.0, 0.0, 0.0]) for layer in READOUT_LAYERS}
    directions = {layer: {operation: basis[index] for index, operation in enumerate(CANDIDATES)}
                  for layer in READOUT_LAYERS}
    aggregate, per_layer = aggregate_readout_scores(hidden_by_layer, directions)
    assert int(aggregate.argmax()) == 0
    assert set(per_layer) == {str(layer) for layer in READOUT_LAYERS}
    assert_close(binomial_tail(12, 12, 0.5), 1.0 / 4096.0)
    assert_close(binomial_tail(0, 12, 0.25), 1.0)


def synthetic_rows(mode):
    rows = []
    for index in range(12):
        correct = CANDIDATES[index % len(CANDIDATES)]
        if mode in ("present", "causal"):
            readout_rank = 1
            readout_prediction = correct
            readout_margin = 1.0
        else:
            readout_rank = 3
            readout_prediction = PERMUTED_LABEL[correct]
            readout_margin = -1.0
        if mode == "present":
            output_rank = 2
        else:
            output_rank = 3
        signal = 0.5 if mode == "causal" else 0.0
        sham = 0.0
        rows.append({
            "readout": {
                "correct_rank": readout_rank,
                "predicted_operation": readout_prediction,
                "permuted_control_label": PERMUTED_LABEL[correct],
                "correct_margin_over_best_other": readout_margin,
            },
            "output_selection": {"correct_rank": output_rank},
            "causal_swap": {
                "signal_donor_minus_correct_delta": signal,
                "sham_donor_minus_correct_delta": sham,
                "bound": {"relative_delta_l2": 0.01},
            },
        })
    return rows


def decision_tests():
    absent = summarize_split(synthetic_rows("absent"))
    present = summarize_split(synthetic_rows("present"))
    causal = summarize_split(synthetic_rows("causal"))
    assert absent["absence_gate"] and not absent["readout_gate"]
    assert present["readout_gate"] and present["output_not_selected_gate"]
    assert causal["causal_swap_gate"]

    assert classify_outcome({"heldout_primary": absent, "heldout_replication": absent})["outcome"].startswith("A_")
    assert classify_outcome({"heldout_primary": present, "heldout_replication": present})["outcome"].startswith("B_")
    assert classify_outcome({"heldout_primary": causal, "heldout_replication": causal})["outcome"].startswith("C_")
    assert classify_outcome({"heldout_primary": causal, "heldout_replication": absent})["outcome"].startswith("D_")


def tiny_model_autograd_test():
    torch.manual_seed(17)
    model = GPT(GPTConfig(
        vocab_size=8192,
        n_layer=4,
        n_head=4,
        n_kv_head=2,
        d_model=16,
        d_ff=32,
        seq_len=16,
        zloss=0.0,
    )).eval()
    model.requires_grad_(False)
    ids = torch.tensor([[1, 7, 9, 11, 13, 15, 17, 19]], dtype=torch.long)
    candidates = (2, 3, 4, 5)
    gradients, contrast = future_logit_gradients(
        model,
        ids,
        anchor_index=3,
        target_logit_index=7,
        candidate_token_id=2,
        all_candidate_token_ids=candidates,
        layers=(1, 2),
    )
    assert set(gradients) == {1, 2}
    assert all(vector.shape == (16,) and torch.isfinite(vector).all() for vector in gradients.values())
    assert math_is_finite(contrast)
    assert all(parameter.grad is None for parameter in model.parameters())

    delta = normalized(gradients[2]) * 0.001
    logits, realized = patched_candidate_logits(
        model,
        ids,
        layer=2,
        anchor_index=3,
        target_logit_index=7,
        delta=delta,
    )
    assert logits.shape == (4,)
    assert torch.isfinite(logits).all()
    assert 0.0 < realized["realized_relative_delta_l2"] < MAX_RELATIVE_SWAP_L2


def math_is_finite(value):
    return value == value and value not in (float("inf"), float("-inf"))


def file_output_tests():
    with tempfile.TemporaryDirectory() as directory:
        destination = Path(directory) / "result.json"
        atomic_write_json(destination, {"frozen": True})
        assert json.loads(destination.read_text()) == {"frozen": True}
        try:
            atomic_write_json(destination, {"frozen": False})
        except FileExistsError:
            pass
        else:
            raise AssertionError("atomic output must refuse overwrite")
        try:
            verify_file_hash(destination, "0" * 64, "synthetic")
        except RuntimeError:
            pass
        else:
            raise AssertionError("hash mismatch must fail closed")


def main():
    static_contract_tests()
    tokenizer_and_board_tests()
    vector_and_statistics_tests()
    decision_tests()
    tiny_model_autograd_test()
    file_output_tests()
    print("operation workspace Jacobian contracts: passed")


if __name__ == "__main__":
    main()
