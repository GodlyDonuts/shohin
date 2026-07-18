"""Hostile CPU tests for the SCERT execution mechanics."""

from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn

import scert


class FixedMotor(nn.Module):
    def __init__(self, delta):
        super().__init__()
        self.register_buffer("delta", torch.tensor(delta, dtype=torch.float32))
        self.calls = 0

    def forward(self, hidden):
        self.calls += 1
        return self.delta.expand(*hidden.shape[:-1], 2)


def _surface(token_id: int) -> scert.EffectiveLogits:
    base = torch.full((1, scert.VOCAB_SIZE), -1.0)
    base[0, token_id] = 1.0
    return scert.effective_logits(
        base, torch.zeros((1, 576)), FixedMotor((0.0, 0.0)), 1
    )


def _runtime_mapping():
    return scert.initial_runtime_state(19).to_mapping()


def _target_runtime_mapping():
    state = scert.initial_runtime_state(19)
    for token_id in (1, 2, 3):
        state = scert.consume_non_eos(state, _surface(token_id))
    return state.to_mapping()


def _target_switch_case():
    return {
        "case_id": "case-0",
        "P_0": [10, 11],
        "X_nom": [1, 2, 3],
        "X_carry": [1, 4, 3],
        "X_result": [1, 2, 5],
        "carry_edit_index": 1,
        "carry_replacement_id": 4,
        "result_edit_index": 2,
        "result_replacement_id": 5,
        "Y_nom": [20, 21],
        "Y_carry": [22, 23],
        "Y_result": [24, 25],
        "E_1": True,
        "D_1": "COMMIT",
        "Q_1": _target_runtime_mapping(),
    }


def test_parameter_ledger_is_exact_and_under_limit():
    manifest = scert.added_parameter_manifest()
    assert manifest == {
        "parent_unique_parameters": 125081664,
        "motor_unique_parameters": 4634,
        "boundary_head_unique_parameters": 1154,
        "added_unique_parameters": 5788,
        "deployment_unique_parameters": 125087452,
        "strictly_below_150m": True,
    }


def test_pre_motor_eos_order_mutation_is_defeated_by_post_motor_surface():
    base = torch.full((1, scert.VOCAB_SIZE), -5.0)
    base[0, scert.EOS_ID] = 1.0
    base[0, scert.V0_ID] = 0.9
    hidden = torch.zeros((1, 576))
    motor = FixedMotor((0.2, -0.5))
    result = scert.effective_logits(base, hidden, motor, 1)
    assert int(base.argmax(-1).item()) == scert.EOS_ID
    assert int(result.token.item()) == scert.V0_ID
    assert result.event.item() is False
    changed = torch.where(base.ne(result.ell_eff))[1].tolist()
    assert changed == [scert.V0_ID, scert.V1_ID]
    assert motor.calls == 1


def test_runtime_rejects_forged_event_and_undeclared_logit_mutation():
    eos_surface = _surface(scert.EOS_ID)
    forged_event = scert.replace(eos_surface, token=torch.tensor([5]))
    with pytest.raises(scert.ContractError, match="event or token"):
        scert.consume_event(scert.initial_runtime_state(), forged_event, "HALT", 0)
    mutated_logits = eos_surface.ell_eff.clone()
    mutated_logits[0, 10] += 0.25
    undeclared = scert.replace(eos_surface, ell_eff=mutated_logits)
    with pytest.raises(scert.ContractError, match="undeclared coordinate"):
        scert.consume_event(scert.initial_runtime_state(), undeclared, "HALT", 0)


def test_motor_off_computes_same_delta_but_changes_no_logit():
    base = torch.randn(
        (3, scert.VOCAB_SIZE), generator=torch.Generator().manual_seed(3)
    )
    hidden = torch.zeros((3, 576))
    motor = FixedMotor((3.0, -2.0))
    off = scert.effective_logits(base, hidden, motor, 0)
    on = scert.effective_logits(base, hidden, motor, 1)
    assert motor.calls == 2
    assert torch.equal(off.motor_delta, on.motor_delta)
    assert torch.equal(off.ell_eff, base)
    changed = torch.where(off.ell_eff.ne(on.ell_eff))[1].unique().tolist()
    assert changed == [scert.V0_ID, scert.V1_ID]


@pytest.mark.parametrize("v0,v1,eos", ((0, 29, 0), (28, 28, 0)))
def test_effective_surface_rejects_token_aliases(v0, v1, eos):
    with pytest.raises(scert.ContractError, match="identities"):
        scert.effective_logits(
            torch.zeros((1, scert.VOCAB_SIZE)),
            torch.zeros((1, 576)),
            FixedMotor((0.0, 0.0)),
            1,
            v0=v0,
            v1=v1,
            eos_id=eos,
        )


def test_effective_loss_uses_exact_supervised_denominator():
    logits = torch.zeros((1, 4, scert.VOCAB_SIZE))
    surface = scert.effective_logits(
        logits,
        torch.zeros((1, 4, 576)),
        FixedMotor((0.0, 0.0)),
        1,
    )
    labels = torch.tensor([[-100, 3, -100, 4]])
    loss, count = scert.effective_lm_loss(surface, labels)
    expected = torch.log(torch.tensor(float(scert.VOCAB_SIZE))) + 1e-4 * torch.log(
        torch.tensor(float(scert.VOCAB_SIZE))
    ).pow(2)
    assert count == 2
    assert torch.allclose(loss, expected)
    with pytest.raises(scert.ContractError, match="no supervised"):
        scert.effective_lm_loss(surface, torch.full_like(labels, -100))
    with pytest.raises(scert.ContractError, match="z-loss"):
        scert.effective_lm_loss(surface, labels, zloss_weight=0.0)
    with pytest.raises(scert.ContractError, match="post-motor"):
        scert.effective_lm_loss(logits, labels)


def test_stage1_packing_labels_and_masks_are_exact():
    g_l = list(range(70))
    p = [70, 71, 72]
    g_r = [80, 81, 82]
    x = [90, 91, 92, 93]
    row = scert.build_stage1_lane(g_l, p, g_r, x)
    assert row.ids.shape == (2048,)
    assert row.positions.tolist() == list(range(2048))
    assert not row.valid[70:582].any()
    assert row.ids[582:585].tolist() == p
    assert row.ids[1094:1097].tolist() == g_r
    assert row.ids[1097:1101].tolist() == x
    assert row.ids[1101].item() == scert.EOS_ID and row.valid[1101]
    expected_labels = {
        1096: 90,
        1097: 91,
        1098: 92,
        1099: 93,
        1100: 0,
    }
    assert row.supervised_count == 5
    assert {
        index: row.labels[index].item() for index in expected_labels
    } == expected_labels
    assert row.attention[1097, 582]
    assert not row.attention[1097, 70]


def test_empty_target_has_one_eos_loss():
    row = scert.build_stage1_lane(list(range(70)), [5], [6, 7, 8], [])
    assert row.supervised_count == 1
    assert row.labels[1096].item() == 0
    assert row.valid[1097]
    assert row.ids[1097].item() == 0


def test_stage1_update_is_exact_two_by_five_and_flattening_is_ordered():
    template = scert.build_stage1_lane(list(range(70)), [10], [80, 81, 82], [90])
    rows = []
    for marker in range(100, 110):
        ids = template.ids.clone()
        ids[scert.STATE_START] = marker
        rows.append(scert.replace(template, ids=ids))
    update = scert.build_stage1_update((rows[:5], rows[5:]))
    assert update.ids.shape == (2, 5, 2048)
    assert update.attention.shape == (2, 5, 2048, 2048)
    assert update.flat_ids[:, scert.STATE_START].tolist() == list(range(100, 110))
    assert update.supervised_count == 20
    with pytest.raises(scert.ContractError, match="two five-lane"):
        scert.build_stage1_update((rows[:5],))


def test_replay_mask_xor_is_only_valid_x_from_p_edges():
    clean = scert.build_replay_surface(
        list(range(70)), [3, 4], [5, 6, 7], [8, 9, 10], clean=True
    )
    stale = scert.build_replay_surface(
        list(range(70)), [3, 4], [5, 6, 7], [8, 9, 10], clean=False
    )
    xor = scert.expected_reconstruction_mask_xor(clean, stale)
    assert xor.sum().item() == 6
    assert xor[582:585, 70:72].all()
    hostile = scert.PackedSurface(
        stale.ids,
        stale.valid,
        stale.positions + 1,
        stale.attention,
        stale.labels,
        stale.keep_indices,
        stale.supervised_count,
        stale.mode,
    )
    with pytest.raises(scert.ContractError, match="positions"):
        scert.expected_reconstruction_mask_xor(clean, hostile)


def test_runtime_state_rejects_missing_extra_and_tie_mutations():
    value = _runtime_mapping()
    assert scert.RuntimeState.from_mapping(value) == scert.initial_runtime_state(19)
    for mutation in ("missing", "extra", "tie"):
        hostile = copy.deepcopy(value)
        if mutation == "missing":
            hostile.pop("receipt_cursor", None)
            hostile.pop("publication_receipt_cursor")
        elif mutation == "extra":
            hostile["operation"] = "add"
        else:
            hostile["deterministic_tie_state"] = "random"
        with pytest.raises(scert.ContractError):
            scert.RuntimeState.from_mapping(hostile)
    hostile = copy.deepcopy(value)
    hostile["cap_constants"]["max_commits"] = 9
    with pytest.raises(scert.ContractError, match="cap constants"):
        scert.RuntimeState.from_mapping(hostile)
    hostile = copy.deepcopy(value)
    hostile["generation_slot_cursor"] += 1
    with pytest.raises(scert.ContractError, match="generation cursor"):
        scert.RuntimeState.from_mapping(hostile)


def test_runtime_fsm_and_caps_fail_closed():
    state = scert.initial_runtime_state(1)
    state = scert.consume_non_eos(state, _surface(5))
    state = scert.consume_event(state, _surface(scert.EOS_ID), "COMMIT", 1)
    assert state.phase == "ACTIVE" and state.commit_count == 1
    state = scert.consume_non_eos(state, _surface(6))
    state = scert.consume_event(state, _surface(scert.EOS_ID), "HALT", 1)
    assert state.phase == "HALTED"
    with pytest.raises(scert.ContractError):
        scert.consume_non_eos(state, _surface(7))
    empty = scert.consume_event(
        scert.initial_runtime_state(), _surface(scert.EOS_ID), "COMMIT", 0
    )
    assert empty.failure_flag
    capped_state = scert.replace(
        scert.initial_runtime_state(),
        epoch_token_count=512,
        total_token_count=512,
        generation_slot_cursor=scert.GENERATION_START + 512,
        rng_state_and_cursor=(0, 512),
    )
    capped = scert.consume_non_eos(capped_state, _surface(2))
    assert capped.failure_flag
    with pytest.raises(scert.ContractError, match="event transition"):
        scert.consume_non_eos(scert.initial_runtime_state(), _surface(scert.EOS_ID))
    with pytest.raises(scert.ContractError, match="post-motor"):
        scert.consume_event(scert.initial_runtime_state(), _surface(5), "HALT", 0)
    with pytest.raises(scert.ContractError, match="EffectiveLogits"):
        scert.consume_event(scert.initial_runtime_state(), torch.zeros(1), "HALT", 0)


def test_boundary_head_tie_is_halt_and_one_forward():
    head = scert.BoundaryHead()
    with torch.no_grad():
        head.affine.weight.zero_()
        head.affine.bias.zero_()
    logits = head(torch.zeros((1, 576)))
    assert int(scert.argmax_lowest_id(logits).item()) == 0
    assert head.forward_count == 1


def test_stage1_schedule_endpoints_and_bounds():
    first = scert.stage1_schedule(1)
    assert first[0] == pytest.approx(0.02)
    assert first[1] == pytest.approx(0.00002)
    assert first[2] == pytest.approx(0.000004)
    assert scert.stage1_schedule(50) == (1.0, 0.001, 0.0002)
    scale, muon_lr, adam_lr = scert.stage1_schedule(1024)
    assert scale == pytest.approx(0.1)
    assert muon_lr == pytest.approx(0.0001)
    assert adam_lr == pytest.approx(0.00002)
    with pytest.raises(scert.ContractError):
        scert.stage1_schedule(0)


def test_optimizer_identity_and_parent_checkpoint_bindings():
    muon, adam = scert.expected_optimizer_names()
    assert len(muon) == 210 and len(adam) == 126
    assert (
        sum(torch.tensor(shape).prod().item() for shape in muon.values()) == 106168320
    )
    assert sum(torch.tensor(shape).prod().item() for shape in adam.values()) == 18917978
    manifest = scert.validate_parent_and_optimizer_manifest()
    assert manifest["tied_head_deduplicated"] is True
    assert manifest["stage1_unique_trainable"] == 125086298
    assert manifest["stage1_optimizer_binding"] == scert.STAGE1_OPTIMIZER_BINDING
    assert manifest["stage2_optimizer_binding"] == scert.STAGE2_OPTIMIZER_BINDING
    assert manifest["stage1_optimizer_binding"]["adamw"]["fused"] is False
    assert manifest["stage1_optimizer_binding"]["step_order"] == ["muon", "adamw"]


def test_source_manifest_rejects_substitution():
    manifest = scert.build_source_manifest()
    scert.validate_source_manifest(manifest)
    assert {entry["sha256"] for entry in manifest["immutable_inputs"]} == {
        scert.TOKENIZER_SHA256,
        scert.PARENT_SHA256,
    }
    hostile = copy.deepcopy(manifest)
    hostile["entries"][0]["sha256"] = "0" * 64
    with pytest.raises(scert.ContractError, match="source manifest"):
        scert.validate_source_manifest(hostile)
    hostile = copy.deepcopy(manifest)
    hostile["entries"].append(copy.deepcopy(hostile["entries"][0]))
    with pytest.raises(scert.ContractError, match="source manifest"):
        scert.validate_source_manifest(hostile)


def test_runtime_manifest_rejects_substitution():
    manifest = scert.build_runtime_manifest()
    scert.validate_runtime_manifest(manifest)
    hostile = copy.deepcopy(manifest)
    hostile["torch"]["deterministic_algorithms"] = False
    with pytest.raises(scert.ContractError, match="runtime manifest"):
        scert.validate_runtime_manifest(hostile)
    hostile = copy.deepcopy(manifest)
    hostile["modules"][0]["sha256"] = "f" * 64
    with pytest.raises(scert.ContractError, match="runtime manifest"):
        scert.validate_runtime_manifest(hostile)


def test_failure_inclusive_denominator_counts_missing_as_failure():
    ledger = scert.FailureInclusiveLedger(expected=4)
    ledger.record("a", True)
    ledger.record("b", False)
    assert ledger.summary() == {
        "denominator": 4,
        "observed": 2,
        "successes": 1,
        "failures": 3,
        "missing_as_failures": 2,
    }
    with pytest.raises(scert.ContractError, match="duplicate"):
        ledger.record("a", True)
    ledger.record("c", True)
    ledger.record("d", True)
    with pytest.raises(scert.ContractError, match="extra"):
        ledger.record("e", True)


def test_target_switch_exact_controls_and_failure_accounting():
    case = _target_switch_case()
    scert.validate_target_switch_arm_specs(scert.TARGET_SWITCH_ARM_SPECS)
    assert tuple(scert.TARGET_SWITCH_ARM_SPECS) == scert.TARGET_SWITCH_ARMS
    scert.validate_target_switch_case(case)
    nominal = {"event": True, "observed_action": "COMMIT", "output_ids": [20, 21]}
    carry = {"event": True, "observed_action": "COMMIT", "output_ids": [22, 23]}
    score = scert.score_target_switch_pair(case, nominal, carry)
    assert score["paired_target_switch"] is True
    changed = dict(carry, observed_action="HALT")
    score = scert.score_target_switch_pair(case, nominal, changed)
    assert score["boundary_action_changed"] is True
    assert score["paired_target_switch"] is False
    hostile_arms = copy.deepcopy(scert.TARGET_SWITCH_ARM_SPECS)
    hostile_arms["TS-C1M0"]["motor_level"] = 1
    with pytest.raises(scert.ContractError, match="arm"):
        scert.validate_target_switch_arm_specs(hostile_arms)


@pytest.mark.parametrize("mutation", ("length", "multiple", "event", "action", "q"))
def test_target_switch_hostile_mutations_fail(mutation):
    case = _target_switch_case()
    if mutation == "length":
        case["X_carry"].append(8)
    elif mutation == "multiple":
        case["X_carry"][0] = 9
    elif mutation == "event":
        case["E_1"] = False
    elif mutation == "action":
        case["D_1"] = "HALT"
    else:
        case["Q_1"]["operation"] = "add"
    with pytest.raises(scert.ContractError):
        scert.validate_target_switch_case(case)


def test_hidden_hb_denominator_is_immutable():
    scert.validate_hidden_board_receipt_counts(scert.HIDDEN_BOARD_SPECS)
    hostile = copy.deepcopy(scert.HIDDEN_BOARD_SPECS)
    hostile["H_B"]["rows"] = 2687
    with pytest.raises(scert.ContractError, match="specification"):
        scert.validate_hidden_board_receipt_counts(hostile)


def test_observational_denominator_is_separate_and_failure_inclusive():
    ledger = scert.ObservationalAdmissionLedger()
    for stratum in range(18):
        for index in range(24):
            ledger.record(f"{stratum}-{index}", stratum, index < 12)
    summary = ledger.summary()
    assert summary["denominator"] == 432
    assert summary["admitted"] == 216
    assert summary["gate_passed"] is True

    incomplete = scert.ObservationalAdmissionLedger()
    incomplete.record("only", 0, True)
    summary = incomplete.summary()
    assert summary["missing_as_failures"] == 431
    assert summary["gate_passed"] is False


def test_all_13_cpu_reference_gates_pass_without_hidden_content():
    report = scert.run_cpu_reference_gates()
    assert report["all_passed"] is True
    assert len(report["gates"]) == 13
    assert all(report["gates"].values())
    assert report["toy"]["paired_cases"] == 256
    assert report["hidden_content_generated"] is False
    assert report["capability_claim"] is False
    assert report["h100_authorized"] is False


def test_job_wrapper_is_cpu_only_and_has_no_launch_override():
    path = scert.ROOT / "train/jobs/scert_newton.sbatch"
    text = path.read_text()
    assert "#SBATCH --gres" not in text
    assert "#SBATCH --gpus" not in text
    assert 'export CUDA_VISIBLE_DEVICES=""' in text
    assert "Only CPU reference and CPU manifest modes are authorized" in text
    assert "fit" not in {line.strip() for line in text.splitlines()}
