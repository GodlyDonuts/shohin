"""Text-only R4 pointer bridge for R9 bidirectional syndrome microcode.

The frozen referential compiler supplies event features and a soft query
covector from token states plus formatting-derived spans.  R9 never receives
structured keys or opcodes at inference.  During training, structured programs
are used only to construct distinct causal targets: forward next-state effects
and backward future-goal pullbacks.  There is intentionally no opcode
cross-entropy in this module.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from bidirectional_syndrome_microcode import (
    BidirectionalSyndromeMicrocode,
    SyndromeMicrocodeRun,
    apply_operator,
    expected_operators,
    homogeneous_state,
    opcode_operator_bank,
    pull_back_goal,
    signed_log_coordinates,
)
from categorical_microcode import OPCODES, QUERIES
from future_effect_algebra import query_operator


def event_feature_dim(hidden: int) -> int:
    return 2 * int(hidden) + 5 + 2 + 2


def assemble_event_feature(output) -> torch.Tensor:
    """Expose text evidence without committing the old static opcode choice."""
    feature = torch.cat((
        output["kind_context"].float(),
        output["target_context"].float(),
        output["kind_logits"].float(),
        output["role_logits"].float(),
        output["slot_presence_scores"].float(),
    ), dim=-1)
    return feature.detach()


def query_goal_from_logits(query_logits: torch.Tensor) -> torch.Tensor:
    if query_logits.ndim != 2 or query_logits.shape[-1] != len(QUERIES):
        raise ValueError("query_logits must have shape [batch,queries]")
    bank = torch.stack([
        query_operator(query, dtype=query_logits.dtype, device=query_logits.device)
        for query in QUERIES
    ])
    return query_logits.float().softmax(dim=-1).to(bank.dtype) @ bank


def oracle_operator_tensor(targets: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    if targets.shape != values.shape or targets.dtype != torch.long:
        raise ValueError("targets and values must share [batch,events] shape")
    bank = opcode_operator_bank(values)
    index = targets[..., None, None, None].expand(-1, -1, 1, 3, 3)
    return bank.gather(2, index).squeeze(2)


def oracle_prefix_suffix(initial_values, query_goals, operators):
    if operators.ndim != 4 or operators.shape[-2:] != (3, 3):
        raise ValueError("operators must have shape [batch,events,3,3]")
    batch, events = operators.shape[:2]
    if initial_values.shape != (batch, 2) or query_goals.shape != (batch, 3):
        raise ValueError("initial values or goals differ from operator batch")
    state = homogeneous_state(initial_values)
    prefix = [state]
    for event in range(events):
        state = apply_operator(operators[:, event], state)
        prefix.append(state)
    goal = query_goals
    suffix = [None] * (events + 1)
    suffix[events] = goal
    for event in range(events - 1, -1, -1):
        goal = pull_back_goal(goal, operators[:, event])
        suffix[event] = goal
    return torch.stack(prefix, dim=1), torch.stack(suffix, dim=1)


def directional_effect_losses(
    run: SyndromeMicrocodeRun,
    values: torch.Tensor,
    initial_values: torch.Tensor,
    target_query_goals: torch.Tensor,
    operation_targets: torch.Tensor,
    answer_targets: torch.Tensor,
):
    """Disjoint weak causal supervision plus an endpoint consistency loss.

    Forward supervision observes only each operator's effect on its actual
    oracle prefix state.  Backward supervision observes only its pullback of
    the actual future goal.  Neither channel receives an opcode label loss.
    """
    oracle = oracle_operator_tensor(operation_targets, values)
    prefix, suffix = oracle_prefix_suffix(initial_values, target_query_goals, oracle)
    forward_prediction = apply_operator(run.forward_operators, prefix[:, :-1])
    forward_target = apply_operator(oracle, prefix[:, :-1])
    backward_prediction = pull_back_goal(suffix[:, 1:], run.backward_operators)
    backward_target = pull_back_goal(suffix[:, 1:], oracle)
    forward_effect = F.smooth_l1_loss(
        signed_log_coordinates(forward_prediction), signed_log_coordinates(forward_target),
    )
    backward_effect = F.smooth_l1_loss(
        signed_log_coordinates(backward_prediction), signed_log_coordinates(backward_target),
    )
    agreement = run.syndrome.square().mean()

    joint_logits = 0.5 * (run.forward_logits + run.backward_logits)
    joint_operators = expected_operators(joint_logits, values)
    state = homogeneous_state(initial_values)
    for event in range(values.shape[1]):
        state = apply_operator(joint_operators[:, event], state)
    predicted_answer = (target_query_goals * state).sum(dim=-1)
    endpoint = F.smooth_l1_loss(
        signed_log_coordinates(predicted_answer),
        signed_log_coordinates(answer_targets.to(predicted_answer.dtype)),
    )
    forward_probabilities = run.forward_logits.softmax(dim=-1)
    backward_probabilities = run.backward_logits.softmax(dim=-1)
    entropy = -0.5 * (
        (forward_probabilities * forward_probabilities.clamp_min(1e-12).log()).sum(dim=-1).mean()
        + (backward_probabilities * backward_probabilities.clamp_min(1e-12).log()).sum(dim=-1).mean()
    )
    total = forward_effect + backward_effect + 0.25 * agreement + 0.5 * endpoint + 0.01 * entropy
    return {
        "total": total,
        "forward_effect": forward_effect,
        "backward_effect": backward_effect,
        "agreement": agreement,
        "endpoint": endpoint,
        "entropy": entropy,
        "predicted_answer": predicted_answer,
    }


@dataclass
class ReferentialSyndromeBatch:
    event_features: torch.Tensor
    values: torch.Tensor
    initial_values: torch.Tensor
    query_logits: torch.Tensor
    query_goals: torch.Tensor
    operation_targets: torch.Tensor
    query_targets: torch.Tensor
    answer_targets: torch.Tensor


class ReferentialSyndromeBridge(nn.Module):
    """Frozen R4 pointer evidence feeding the trainable R9 recurrent cell."""

    def __init__(self, pointer_compiler, pointer_hidden=256, memory_dim=96):
        super().__init__()
        self.pointer_compiler = pointer_compiler
        self.pointer_hidden = int(pointer_hidden)
        self.pointer_compiler.requires_grad_(False)
        self.microcode = BidirectionalSyndromeMicrocode(
            event_dim=event_feature_dim(pointer_hidden), memory_dim=memory_dim,
        )

    def adapter_parameters(self):
        yield from self.microcode.parameters()

    def adapter_num_params(self) -> int:
        return self.microcode.adapter_num_params()

    def encode_examples(self, ids: torch.Tensor, examples) -> ReferentialSyndromeBatch:
        if len(examples) != ids.shape[0] or not examples:
            raise ValueError("examples must match a nonempty token batch")
        depths = {len(example.compiled.operation_targets) for example in examples}
        if len(depths) != 1:
            raise ValueError("referential syndrome batches must have one event depth")
        with torch.no_grad():
            hidden, identity = self.pointer_compiler.encode(ids)
            results = [
                self.pointer_compiler.classify_text(
                    hidden[index], identity[index], example.intro_positions,
                    example.operation_spans, example.query_span,
                )
                for index, example in enumerate(examples)
            ]
            event_features = torch.stack([
                torch.stack([assemble_event_feature(output) for output in result["operations"]])
                for result in results
            ])
            query_logits = torch.stack([
                self.pointer_compiler.compose_query_logits(
                    result["query"]["kind_logits"].unsqueeze(0),
                    result["query"]["role_logits"].unsqueeze(0),
                ).squeeze(0).float()
                for result in results
            ])
        device = event_features.device
        values = torch.tensor(
            [example.compiled.operation_values for example in examples],
            dtype=event_features.dtype, device=device,
        )
        initial = torch.tensor(
            [example.compiled.initial_values for example in examples],
            dtype=event_features.dtype, device=device,
        )
        operation_targets = torch.tensor(
            [example.compiled.operation_targets for example in examples],
            dtype=torch.long, device=device,
        )
        query_targets = torch.tensor(
            [example.compiled.query_target for example in examples],
            dtype=torch.long, device=device,
        )
        answers = torch.tensor(
            [example.compiled.answer for example in examples],
            dtype=event_features.dtype, device=device,
        )
        return ReferentialSyndromeBatch(
            event_features=event_features,
            values=values,
            initial_values=initial,
            query_logits=query_logits,
            query_goals=query_goal_from_logits(query_logits),
            operation_targets=operation_targets,
            query_targets=query_targets,
            answer_targets=answers,
        )
