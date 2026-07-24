from __future__ import annotations

from dataclasses import fields, replace
import inspect

import pytest
import torch

from causal_bind_select_workspace import (
    CausalBindSelectWorkspace,
    CausalWorkspaceConfig,
    CausalWorkspaceGPT,
    WorkspaceContractError,
    WorkspaceControls,
    WorkspaceState,
    freeze_protected_base,
    trainable_workspace_parameters,
)
from model import GPT, GPTConfig


def _small_config() -> CausalWorkspaceConfig:
    return CausalWorkspaceConfig(
        d_model=24,
        slot_width=16,
        num_slots=4,
        num_operators=4,
        operator_rank=4,
        stage_after_block=1,
    )


def _active_workspace() -> CausalBindSelectWorkspace:
    torch.manual_seed(2026072341)
    workspace = CausalBindSelectWorkspace(_small_config())
    with torch.no_grad():
        workspace.read_projection.weight.normal_(std=0.1)
        workspace.read_gate.fill_(2.0)
    return workspace


def test_reference_parameter_receipt_matches_frozen_budget() -> None:
    workspace = CausalBindSelectWorkspace(CausalWorkspaceConfig())
    receipt = workspace.parameter_receipt()
    assert receipt.workspace_parameters == 907_269
    assert receipt.protected_base_parameters == 125_081_664
    assert receipt.complete_system_parameters == 125_988_933
    assert receipt.remaining_under_cap == 74_011_067


def test_zero_initialized_read_path_is_exact_identity() -> None:
    torch.manual_seed(2026072342)
    workspace = CausalBindSelectWorkspace(_small_config())
    world = torch.randn(2, 5, 24)
    query = torch.randn(2, 4, 24)
    state, _ = workspace.compile_world(world)
    sealed = workspace.seal_state(state)
    output, _, _ = workspace.execute_query(query, workspace=sealed)
    assert torch.equal(output, query)


def test_zero_read_gate_can_wake_up_without_breaking_initial_identity() -> None:
    torch.manual_seed(20260723425)
    workspace = CausalBindSelectWorkspace(_small_config())
    world = torch.randn(2, 5, 24)
    query = torch.randn(2, 4, 24)
    state, _ = workspace.compile_world(world)
    state = workspace.seal_state(state)
    output, _, _ = workspace.execute_query(query, workspace=state)
    output.sum().backward()
    assert workspace.read_gate.grad is not None
    assert workspace.read_gate.grad.abs() > 0


def test_world_compilation_is_causal_under_future_perturbation() -> None:
    workspace = _active_workspace()
    torch.manual_seed(2026072343)
    world = torch.randn(2, 7, 24)
    perturbed = world.clone()
    perturbed[:, 5:] = torch.randn_like(perturbed[:, 5:]) * 100
    _, trace = workspace.compile_world(world, return_trace=True)
    _, altered_trace = workspace.compile_world(perturbed, return_trace=True)
    assert torch.equal(trace[:, :5], altered_trace[:, :5])


def test_chunked_world_compilation_matches_one_shot() -> None:
    workspace = _active_workspace()
    torch.manual_seed(2026072344)
    world = torch.randn(2, 7, 24)
    one_shot, _ = workspace.compile_world(world)
    first, _ = workspace.compile_world(world[:, :3])
    chunked, _ = workspace.compile_world(world[:, 3:], state=first)
    assert one_shot.token_position == chunked.token_position == 7
    assert torch.allclose(one_shot.slots, chunked.slots, atol=1e-6, rtol=1e-6)


def test_streamed_query_execution_matches_one_shot() -> None:
    workspace = _active_workspace()
    torch.manual_seed(2026072345)
    world = torch.randn(2, 6, 24)
    query = torch.randn(2, 5, 24)
    state, _ = workspace.compile_world(world)
    sealed = workspace.seal_state(state)
    full_output, full_state, _ = workspace.execute_query(query, workspace=sealed)

    stream = workspace.start_execution(sealed)
    outputs = []
    for index in range(query.shape[1]):
        output, stream, _ = workspace.execute_query(
            query[:, index : index + 1],
            state=stream,
        )
        outputs.append(output)
    streamed_output = torch.cat(outputs, dim=1)
    assert torch.allclose(full_output, streamed_output, atol=1e-6, rtol=1e-6)
    assert torch.allclose(full_state.cursor, stream.cursor, atol=1e-6, rtol=1e-6)
    assert stream.query_tokens == query.shape[1]


def test_sealed_state_is_detached_disjoint_and_query_does_not_mutate_slots() -> None:
    workspace = _active_workspace()
    torch.manual_seed(2026072346)
    world = torch.randn(2, 6, 24)
    query = torch.randn(2, 3, 24)
    state, _ = workspace.compile_world(world)
    sealed = workspace.seal_state(state)
    reference_slots = sealed.slots.clone()
    output, execution, _ = workspace.execute_query(query, workspace=sealed)
    repeated, _, _ = workspace.execute_query(query, workspace=sealed)
    assert torch.equal(output, repeated)
    assert torch.equal(sealed.slots, reference_slots)
    assert sealed.slots.grad_fn is None
    assert sealed.slots.requires_grad is False
    assert sealed.slots.data_ptr() != state.slots.data_ptr()
    assert execution.workspace is sealed


def test_binding_and_operator_controls_are_independently_causal() -> None:
    workspace = _active_workspace()
    torch.manual_seed(2026072347)
    world = torch.randn(2, 8, 24)
    query = torch.randn(2, 5, 24)
    state, _ = workspace.compile_world(world)
    state = workspace.seal_state(state)
    with torch.no_grad():
        treatment, _, treatment_diagnostics = workspace.execute_query(
            query, workspace=state
        )
        binding, _, binding_diagnostics = workspace.execute_query(
            query,
            workspace=state,
            controls=WorkspaceControls(binding_permutation=(1, 2, 3, 0)),
            assessment_only=True,
        )
        operator, _, operator_diagnostics = workspace.execute_query(
            query,
            workspace=state,
            controls=WorkspaceControls(operator_permutation=(3, 0, 1, 2)),
            assessment_only=True,
        )
        zero, _, _ = workspace.execute_query(
            query,
            workspace=state,
            controls=WorkspaceControls(zero_workspace=True),
            assessment_only=True,
        )
    assert not torch.equal(treatment, binding)
    assert not torch.equal(treatment, operator)
    assert not torch.equal(treatment, zero)
    assert not torch.equal(binding, operator)
    binding_index = torch.tensor((1, 2, 3, 0))
    operator_index = torch.tensor((3, 0, 1, 2))
    assert torch.equal(
        binding_diagnostics.bindings,
        treatment_diagnostics.bindings.index_select(-1, binding_index),
    )
    assert torch.equal(
        operator_diagnostics.operator_probabilities,
        treatment_diagnostics.operator_probabilities.index_select(
            -1,
            operator_index,
        ),
    )


def test_hard_binding_and_operator_forward_keep_straight_through_gradients() -> None:
    workspace = _active_workspace()
    torch.manual_seed(20260723471)
    world = torch.randn(3, 7, 24)
    query = torch.randn(3, 4, 24)
    state, _ = workspace.compile_world(world)
    output, _, diagnostics = workspace.execute_query(
        query,
        workspace=state,
        allow_unsealed_for_mechanism_fit=True,
    )

    for distribution in (
        diagnostics.bindings,
        diagnostics.operator_probabilities,
    ):
        assert torch.all((distribution == 0) | (distribution == 1))
        assert torch.equal(
            distribution.sum(dim=-1),
            torch.ones_like(distribution[..., 0]),
        )

    output.square().mean().backward()
    assert workspace.key_projection.weight.grad is not None
    assert workspace.operator_selector.weight.grad is not None
    assert workspace.key_projection.weight.grad.abs().sum() > 0
    assert workspace.operator_selector.weight.grad.abs().sum() > 0


def test_selected_slot_scramble_changes_crafted_output_but_discarded_does_not() -> None:
    workspace = CausalBindSelectWorkspace(_small_config())
    with torch.no_grad():
        workspace.key_projection.weight.zero_()
        workspace.value_projection.weight.zero_()
        workspace.operator_selector.weight.zero_()
        workspace.operator_selector.bias.fill_(-100.0)
        workspace.operator_selector.bias[0] = 100.0
        workspace.operator_down.zero_()
        workspace.operator_up.zero_()
        workspace.read_projection.weight.zero_()
        workspace.read_projection.weight[:16].copy_(torch.eye(16))
        workspace.read_gate.fill_(5.0)

    slots = torch.stack(
        (
            torch.arange(1, 17, dtype=torch.float32),
            torch.arange(101, 117, dtype=torch.float32),
            torch.arange(201, 217, dtype=torch.float32),
            torch.arange(301, 317, dtype=torch.float32),
        )
    ).unsqueeze(0)
    state = WorkspaceState(slots=slots, token_position=145, sealed=True)
    query = torch.zeros(1, 1, 24)

    with torch.no_grad():
        treatment, _, diagnostics = workspace.execute_query(query, workspace=state)
        selected, _, _ = workspace.execute_query(
            query,
            workspace=state,
            controls=WorkspaceControls(scramble_selected_slot=True),
            assessment_only=True,
        )
        discarded, _, _ = workspace.execute_query(
            query,
            workspace=state,
            controls=WorkspaceControls(scramble_discarded_slots=True),
            assessment_only=True,
        )

    assert torch.equal(
        diagnostics.bindings,
        torch.tensor([[[1.0, 0.0, 0.0, 0.0]]]),
    )
    assert not torch.equal(treatment, selected)
    assert torch.equal(treatment, discarded)


def test_interventions_are_assessment_only_and_mutually_exclusive() -> None:
    workspace = _active_workspace()
    hidden = torch.randn(2, 3, 24)
    state, _ = workspace.compile_world(hidden)
    state = workspace.seal_state(state)
    control = WorkspaceControls(scramble_selected_slot=True)

    with pytest.raises(WorkspaceContractError, match="assessment-only"):
        workspace.execute_query(hidden, workspace=state, controls=control)
    with pytest.raises(WorkspaceContractError, match="gradients to be disabled"):
        workspace.execute_query(
            hidden,
            workspace=state,
            controls=control,
            assessment_only=True,
        )
    with pytest.raises(WorkspaceContractError, match="mutually exclusive"):
        with torch.no_grad():
            workspace.execute_query(
                hidden,
                workspace=state,
                controls=WorkspaceControls(
                    scramble_selected_slot=True,
                    scramble_discarded_slots=True,
                ),
                assessment_only=True,
            )
    with pytest.raises(WorkspaceContractError, match="hard selected binding"):
        with torch.no_grad():
            workspace.execute_query(
                hidden,
                workspace=state,
                controls=WorkspaceControls(
                    uniform_binding=True,
                    scramble_selected_slot=True,
                ),
                assessment_only=True,
            )
    with torch.no_grad():
        output, _, _ = workspace.execute_query(
            hidden,
            workspace=state,
            controls=control,
            assessment_only=True,
        )
    assert output.requires_grad is False
    assert (
        "controls"
        not in inspect.signature(CausalWorkspaceGPT.forward_mechanism_fit).parameters
    )
    assert (
        "controls"
        in inspect.signature(CausalWorkspaceGPT.forward_staged_controlled).parameters
    )


def test_explicit_mechanism_fit_path_reaches_compiler_and_executor() -> None:
    workspace = _active_workspace()
    torch.manual_seed(2026072348)
    world = torch.randn(2, 8, 24)
    query = torch.randn(2, 5, 24)
    state, _ = workspace.compile_world(world)
    output, _, _ = workspace.execute_query(
        query,
        workspace=state,
        allow_unsealed_for_mechanism_fit=True,
    )
    output.square().mean().backward()
    assert workspace.key_projection.weight.grad is not None
    assert workspace.operator_selector.weight.grad is not None
    assert workspace.operator_down.grad is not None
    assert workspace.operator_up.grad is not None
    assert workspace.read_projection.weight.grad is not None
    assert workspace.recurrent_update.weight.grad is not None
    assert workspace.initial_slots.grad is not None
    assert workspace.key_projection.weight.grad.abs().sum() > 0
    assert workspace.operator_down.grad.abs().sum() > 0
    assert workspace.recurrent_update.weight.grad.abs().sum() > 0
    assert workspace.initial_slots.grad.abs().sum() > 0


def test_workspace_state_shape_and_control_contracts_fail_closed() -> None:
    workspace = _active_workspace()
    hidden = torch.randn(2, 3, 24)
    bad = WorkspaceState(slots=torch.randn(2, 5, 16), token_position=0)
    with pytest.raises(WorkspaceContractError, match="wrong shape"):
        workspace.execute_query(hidden, workspace=bad)
    state, _ = workspace.compile_world(hidden)
    with pytest.raises(WorkspaceContractError, match="source-deleted sealed"):
        workspace.execute_query(hidden, workspace=state)
    state = workspace.seal_state(state)
    with pytest.raises(WorkspaceContractError, match="not a permutation"):
        workspace.execute_query(
            hidden,
            workspace=state,
            controls=WorkspaceControls(binding_permutation=(0, 0, 1, 2)),
        )


def test_right_padding_is_inert_and_malformed_padding_rejected() -> None:
    workspace = _active_workspace()
    torch.manual_seed(2026072349)
    hidden = torch.randn(2, 5, 24)
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 0, 0]])
    state, _ = workspace.compile_world(hidden, attention_mask=mask)
    short, _ = workspace.compile_world(hidden[:, :3])
    assert state.token_position == short.token_position == 3
    assert torch.allclose(state.slots, short.slots, atol=1e-6, rtol=1e-6)
    malformed = torch.tensor([[1, 0, 1, 0, 0], [1, 0, 1, 0, 0]])
    with pytest.raises(WorkspaceContractError, match="right padded"):
        workspace.compile_world(hidden, attention_mask=malformed)


def test_small_gpt_wrapper_uses_raw_tokens_and_separates_world_from_query() -> None:
    torch.manual_seed(2026072350)
    base = GPT(
        GPTConfig(
            vocab_size=64,
            n_layer=4,
            n_head=4,
            n_kv_head=2,
            d_model=24,
            d_ff=48,
            seq_len=32,
            zloss=0.0,
        )
    )
    wrapper = CausalWorkspaceGPT(base, _small_config())
    world = torch.randint(0, 64, (2, 7))
    query = torch.randint(0, 64, (2, 5))
    targets = torch.randint(0, 64, (2, 5))
    logits, loss, state, diagnostics = wrapper.forward_staged_assessment(
        world,
        query,
        targets=targets,
    )
    assert logits.shape == (2, 5, 64)
    assert loss.ndim == 0
    assert state.token_position == 7
    assert diagnostics.bindings.shape == (2, 5, 4)
    assert diagnostics.operator_probabilities.shape == (2, 5, 4)
    assert state.sealed is True


def test_zero_gate_wrapper_exactly_matches_source_deleted_base_query() -> None:
    torch.manual_seed(202607235005)
    base = GPT(
        GPTConfig(
            vocab_size=64,
            n_layer=4,
            n_head=4,
            n_kv_head=2,
            d_model=24,
            d_ff=48,
            seq_len=32,
            zloss=0.0,
        )
    )
    wrapper = CausalWorkspaceGPT(base, _small_config())
    world = torch.randint(0, 64, (2, 7))
    query = torch.randint(0, 64, (2, 5))
    logits, _ = wrapper.forward_staged(world, query)
    reference, _ = base(query, pos=world.shape[1])
    assert torch.equal(logits, reference)


def test_workspace_state_has_only_allowlisted_execution_fields() -> None:
    assert tuple(field.name for field in fields(WorkspaceState)) == (
        "slots",
        "token_position",
        "sealed",
    )


def test_scored_wrapper_interface_accepts_raw_tokens_and_no_controls() -> None:
    parameters = inspect.signature(CausalWorkspaceGPT.forward_staged).parameters
    assert "world_idx" in parameters
    assert "query_idx" in parameters
    assert "controls" not in parameters
    assert "workspace" not in parameters
    assert "seal_world" not in parameters
    assert "allow_unsealed_for_mechanism_fit" not in parameters


def test_wrapper_logits_are_causal_under_future_query_perturbation() -> None:
    torch.manual_seed(20260723501)
    base = GPT(
        GPTConfig(
            vocab_size=64,
            n_layer=4,
            n_head=4,
            n_kv_head=2,
            d_model=24,
            d_ff=48,
            seq_len=32,
            zloss=0.0,
        )
    )
    wrapper = CausalWorkspaceGPT(base, _small_config())
    with torch.no_grad():
        wrapper.workspace.read_gate.fill_(1.0)
    world = torch.randint(0, 64, (2, 7))
    query = torch.randint(0, 64, (2, 6))
    perturbed = query.clone()
    perturbed[:, 4:] = torch.randint(0, 64, (2, 2))
    logits, _ = wrapper.forward_staged(world, query)
    altered, _ = wrapper.forward_staged(world, perturbed)
    assert torch.equal(logits[:, :4], altered[:, :4])


def test_wrapper_right_padding_is_inert_at_workspace_boundary() -> None:
    torch.manual_seed(202607235015)
    base = GPT(
        GPTConfig(
            vocab_size=64,
            n_layer=4,
            n_head=4,
            n_kv_head=2,
            d_model=24,
            d_ff=48,
            seq_len=32,
            zloss=0.0,
        )
    )
    wrapper = CausalWorkspaceGPT(base, _small_config())
    world = torch.randint(0, 64, (2, 7))
    short = wrapper._compile_world_state(world[:, :5])
    mask = torch.tensor([[1, 1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 0, 0]])
    padded = wrapper._compile_world_state(world, attention_mask=mask)
    assert short.token_position == padded.token_position == 5
    assert torch.allclose(short.slots, padded.slots, atol=1e-6, rtol=1e-6)


def test_full_wrapper_prefix_replay_matches_one_shot_causal_logits() -> None:
    torch.manual_seed(202607235016)
    base = GPT(
        GPTConfig(
            vocab_size=64,
            n_layer=4,
            n_head=4,
            n_kv_head=2,
            d_model=24,
            d_ff=48,
            seq_len=32,
            zloss=0.0,
        )
    )
    wrapper = CausalWorkspaceGPT(base, _small_config())
    with torch.no_grad():
        wrapper.workspace.read_gate.fill_(1.0)
    world = torch.randint(0, 64, (2, 7))
    query = torch.randint(0, 64, (2, 6))
    full, _ = wrapper.forward_staged(world, query)
    for length in range(1, query.shape[1] + 1):
        prefix, _ = wrapper.forward_staged(world, query[:, :length])
        assert torch.allclose(
            prefix[:, -1],
            full[:, length - 1],
            atol=1e-6,
            rtol=1e-6,
        )


def test_active_workspace_makes_raw_world_causally_relevant() -> None:
    torch.manual_seed(202607235017)
    base = GPT(
        GPTConfig(
            vocab_size=64,
            n_layer=4,
            n_head=4,
            n_kv_head=2,
            d_model=24,
            d_ff=48,
            seq_len=32,
            zloss=0.0,
        )
    )
    wrapper = CausalWorkspaceGPT(base, _small_config())
    with torch.no_grad():
        wrapper.workspace.read_gate.fill_(1.0)
    world = torch.randint(0, 64, (2, 7))
    alternate = torch.randint(0, 64, (2, 7))
    query = torch.randint(0, 64, (2, 6))
    logits, _ = wrapper.forward_staged(world, query)
    changed, _ = wrapper.forward_staged(alternate, query)
    assert not torch.equal(logits, changed)


def test_default_staged_path_detaches_compiler_but_fit_path_reaches_it() -> None:
    torch.manual_seed(202607235018)
    base = GPT(
        GPTConfig(
            vocab_size=64,
            n_layer=4,
            n_head=4,
            n_kv_head=2,
            d_model=24,
            d_ff=48,
            seq_len=32,
            zloss=0.0,
        )
    )
    wrapper = CausalWorkspaceGPT(base, _small_config())
    freeze_protected_base(wrapper)
    with torch.no_grad():
        wrapper.workspace.read_gate.fill_(1.0)
    world = torch.randint(0, 64, (2, 7))
    query = torch.randint(0, 64, (2, 6))
    targets = torch.randint(0, 64, (2, 6))

    _, sealed_loss = wrapper.forward_staged(world, query, targets=targets)
    sealed_loss.backward()
    assert wrapper.workspace.initial_slots.grad is None
    assert wrapper.workspace.recurrent_update.weight.grad is None

    wrapper.zero_grad(set_to_none=True)
    _, fit_loss, state, _ = wrapper.forward_mechanism_fit(
        world,
        query,
        targets=targets,
    )
    fit_loss.backward()
    assert state.sealed is False
    assert wrapper.workspace.initial_slots.grad is not None
    assert wrapper.workspace.recurrent_update.weight.grad is not None
    assert wrapper.workspace.initial_slots.grad.abs().sum() > 0
    assert wrapper.workspace.recurrent_update.weight.grad.abs().sum() > 0


def test_public_process_boundary_compiles_without_query_and_executes_without_world() -> (
    None
):
    torch.manual_seed(202607235019)
    base = GPT(
        GPTConfig(
            vocab_size=64,
            n_layer=4,
            n_head=4,
            n_kv_head=2,
            d_model=24,
            d_ff=48,
            seq_len=32,
            zloss=0.0,
        )
    )
    wrapper = CausalWorkspaceGPT(base, _small_config())
    world = torch.randint(0, 64, (2, 7))
    query = torch.randint(0, 64, (2, 5))
    state = wrapper.compile_world_state(world)
    assert state.sealed is True
    assert state.token_position == 7
    logits, diagnostics = wrapper.execute_workspace_state(state, query)
    assert logits.shape == (2, 5, 64)
    assert diagnostics.bindings.shape == (2, 5, 4)

    with torch.no_grad():
        controlled, _ = wrapper.execute_workspace_state(
            state,
            query,
            controls=WorkspaceControls(scramble_selected_slot=True),
        )
    assert controlled.shape == logits.shape


def test_wrapper_construction_does_not_mutate_base_tensors() -> None:
    torch.manual_seed(20260723502)
    base = GPT(
        GPTConfig(
            vocab_size=64,
            n_layer=4,
            n_head=4,
            n_kv_head=2,
            d_model=24,
            d_ff=48,
            seq_len=32,
        )
    )
    before = {
        name: tensor.detach().clone() for name, tensor in base.state_dict().items()
    }
    CausalWorkspaceGPT(base, _small_config())
    for name, tensor in base.state_dict().items():
        assert torch.equal(tensor, before[name])


def test_base_freeze_produces_exact_workspace_optimizer_partition() -> None:
    torch.manual_seed(2026072351)
    base = GPT(
        GPTConfig(
            vocab_size=64,
            n_layer=4,
            n_head=4,
            n_kv_head=2,
            d_model=24,
            d_ff=48,
            seq_len=32,
        )
    )
    wrapper = CausalWorkspaceGPT(base, _small_config())
    freeze_protected_base(wrapper)
    parameters = trainable_workspace_parameters(wrapper)
    assert parameters
    assert all(parameter.requires_grad for parameter in parameters)
    assert not any(parameter.requires_grad for parameter in wrapper.base.parameters())
    assert set(parameters) == set(wrapper.workspace.parameters())


def test_invalid_stage_and_dimension_configurations_fail_closed() -> None:
    base = GPT(
        GPTConfig(
            vocab_size=64,
            n_layer=4,
            n_head=4,
            n_kv_head=2,
            d_model=24,
            d_ff=48,
            seq_len=32,
        )
    )
    with pytest.raises(WorkspaceContractError, match="leave at least one"):
        CausalWorkspaceGPT(base, replace(_small_config(), stage_after_block=3))
    with pytest.raises(WorkspaceContractError, match="d_model"):
        CausalWorkspaceGPT(base, replace(_small_config(), d_model=32))
