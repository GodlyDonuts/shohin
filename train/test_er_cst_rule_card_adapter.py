from __future__ import annotations

import torch
import torch.nn.functional as F

from er_cst_rule_card_adapter import (
    ER_RECORDS,
    EVENT_SLOTS,
    RULE_CARD_COUNT,
    RULE_COUNT,
    EpisodicRuleCardCompiler,
    HardRuleCardProgram,
    TiedRuleCardMotor,
    er_new_parameter_names,
    freeze_to_er_adaptive,
    rollout_rule_cards,
    rule_card_parameter_report,
    rule_motor_certificate,
)
from pilot_sd_cst_byte_addressed import BASE_PARAMETERS, READER_PARAMETERS
from sd_cst_byte_addressed import BYTE_PAD


def _small_model() -> EpisodicRuleCardCompiler:
    return EpisodicRuleCardCompiler(
        width=32,
        heads=4,
        encoder_layers=1,
        slot_layers=1,
        ff=64,
        slot_ff=64,
        max_bytes=256,
        fingerprint_width=16,
        orbit_width=32,
        orbit_heads=4,
        orbit_layers=1,
        orbit_ff=64,
        native_slot_layers=1,
        native_slot_heads=4,
        native_slot_ff=64,
        record_width=32,
        record_heads=4,
        record_layers=1,
        record_set_layers=1,
        record_ff=64,
        max_line_bytes=32,
        sinkhorn_steps=4,
        occurrence_ff=64,
    )


def _batch(texts: list[bytes]) -> tuple[torch.Tensor, torch.Tensor]:
    width = max(map(len, texts))
    ids = torch.full((len(texts), width), BYTE_PAD, dtype=torch.long)
    valid = torch.zeros_like(ids, dtype=torch.bool)
    for row, text in enumerate(texts):
        values = torch.tensor(list(text), dtype=torch.long)
        ids[row, : len(text)] = values
        valid[row, : len(text)] = True
    return ids, valid


def test_rule_card_compiler_shapes_and_source_only_boundary() -> None:
    torch.manual_seed(11)
    model = _small_model().eval()

    def forbidden_parent_path(*args: object, **kwargs: object) -> torch.Tensor:
        raise AssertionError("global source encoder was called")

    model._encode = forbidden_parent_path  # type: ignore[method-assign]
    model._orbit_encode = forbidden_parent_path  # type: ignore[method-assign]
    program = b"\n".join(f"record {index} zed".encode() for index in range(ER_RECORDS))
    ids, valid = _batch([program])
    query_ids, query_valid = _batch([b"Return rank 2."])
    with torch.no_grad():
        output = model.compile_rule_program(ids, valid, query_ids, query_valid)
    assert output.program.initial_state.shape == (1, RULE_CARD_COUNT)
    assert output.program.rule_cards.shape == (1, RULE_COUNT, RULE_CARD_COUNT)
    assert output.program.event_card.shape == (1, EVENT_SLOTS, RULE_COUNT)
    assert output.program.event_halt.shape == (1, EVENT_SLOTS, 2)
    assert output.line_pointer_logits.shape == (1, ER_RECORDS, ids.shape[1])
    assert output.binding_pointer_logits.shape == (1, 3, ids.shape[1])
    assert output.initial_entity_pointer_logits.shape == (1, 3, ids.shape[1])
    assert output.query.logits.shape == (1, 3)
    assert output.query_pointer_logits.shape == query_ids.shape


def test_er_trainability_is_explicit_and_new_names_are_exact() -> None:
    model = _small_model()
    assert len(er_new_parameter_names(model)) == 13
    declared = set(freeze_to_er_adaptive(model))
    assert declared
    assert er_new_parameter_names(model) < declared
    assert all(
        parameter.requires_grad == (name in declared)
        for name, parameter in model.named_parameters()
    )


def test_er_backward_reaches_every_declared_tensor_only() -> None:
    torch.manual_seed(13)
    model = _small_model().train()
    declared = set(freeze_to_er_adaptive(model))
    program = b"\n".join(f"r{index} a b c".encode() for index in range(ER_RECORDS))
    ids, valid = _batch([program])
    query_ids, query_valid = _batch([b"Return rank 2."])
    output = model.compile_rule_program(ids, valid, query_ids, query_valid)
    loss = (
        output.line_pointer_logits.square().mean()
        + output.binding_pointer_logits.square().mean()
        + output.initial_entity_pointer_logits.square().mean()
        + output.program.initial_state.square().mean()
        + output.program.rule_cards.square().mean()
        + output.program.event_card.square().mean()
        + output.program.event_halt.square().mean()
        + output.query.logits.square().mean()
        + output.query_pointer_logits.square().mean()
    )
    loss.backward()
    missing = [
        name
        for name, parameter in model.named_parameters()
        if name in declared
        and (parameter.grad is None or not bool(parameter.grad.ne(0).any()))
    ]
    leaked = [
        name
        for name, parameter in model.named_parameters()
        if name not in declared and parameter.grad is not None
    ]
    assert missing == []
    assert leaked == []


def test_rule_motor_certificate_can_be_fit_exactly() -> None:
    torch.manual_seed(17)
    motor = TiedRuleCardMotor(hidden=128)
    state, card, target = rule_motor_certificate()
    optimizer = torch.optim.AdamW(
        motor.parameters(), lr=0.03, betas=(0.9, 0.95), weight_decay=0.0
    )
    for _ in range(1_000):
        logits = motor(
            F.one_hot(state, RULE_CARD_COUNT).float(),
            F.one_hot(card, RULE_CARD_COUNT).float(),
        )
        loss = F.cross_entropy(logits, target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if torch.equal(logits.argmax(-1), target):
            break
    assert torch.equal(
        motor(
            F.one_hot(state, RULE_CARD_COUNT).float(),
            F.one_hot(card, RULE_CARD_COUNT).float(),
        ).argmax(-1),
        target,
    )


def test_hard_rollout_uses_selected_cards_and_persistent_halt() -> None:
    class ExactMotor(TiedRuleCardMotor):
        def forward(self, state: torch.Tensor, card: torch.Tensor) -> torch.Tensor:
            state_ids = state.argmax(-1)
            card_ids = card.argmax(-1)
            source_state, source_card, targets = rule_motor_certificate(state.device)
            output = torch.empty_like(state_ids)
            for index, (state_id, card_id) in enumerate(zip(state_ids, card_ids, strict=True)):
                mask = source_state.eq(state_id) & source_card.eq(card_id)
                output[index] = targets[mask].item()
            return 20.0 * F.one_hot(output, RULE_CARD_COUNT).float()

    program = HardRuleCardProgram(
        initial_state=torch.tensor([0], dtype=torch.long),
        rule_cards=torch.tensor([[3, 1, 5]], dtype=torch.long),
        event_card=torch.tensor([[0, 1, 2, 0, 0, 0, 0, 0]], dtype=torch.long),
        event_halt=torch.tensor([[0, 0, 1, 0, 0, 0, 0, 0]], dtype=torch.long),
    )
    result = rollout_rule_cards(program, ExactMotor())
    assert result.alive_trajectory[0].item() is True
    assert result.alive_trajectory[1].item() is True
    assert result.alive_trajectory[2].item() is True
    assert all(not value.item() for value in result.alive_trajectory[3:])
    assert all(
        torch.equal(value, result.state_trajectory[2])
        for value in result.state_trajectory[3:]
    )


def test_default_parameter_certificate_stays_below_200m() -> None:
    model = EpisodicRuleCardCompiler()
    motor = TiedRuleCardMotor()
    freeze_to_er_adaptive(model)
    report = rule_card_parameter_report(
        model,
        motor,
        base_parameters=BASE_PARAMETERS,
        reader_parameters=READER_PARAMETERS,
    )
    assert model.parameter_count() == 67_336_230
    assert report["motor"] == 2_438
    assert report["complete_system"] == 192_421_167
    assert report["headroom_below_200m"] == 7_578_833
    assert report["trainable"] == 11_715_616
    assert report["complete_system"] < 200_000_000
