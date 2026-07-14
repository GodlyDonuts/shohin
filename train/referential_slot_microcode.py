"""Binding-first microcode compiler with text-only dynamic entity slots.

Structured keys supervise mention attention during training and score it during
evaluation. They are never passed to ``classify_text``: inference receives only
token states plus line boundaries. Register roles are recovered by matching a
predicted target mention to two predicted introductory entity slots.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from categorical_microcode import (
    CompiledExample,
    _line_spans,
    compile_example,
    transition_basis_targets,
)
from role_equivariant_microcode import (
    IGNORE_ROLE,
    OPERATION_KINDS,
    QUERY_KINDS,
    RoleEquivariantMicrocodeCompiler,
    factor_operation,
    factor_query,
)


@dataclass(frozen=True)
class ReferentialCompiledExample:
    compiled: CompiledExample
    intro_positions: tuple
    operation_spans: tuple
    query_span: tuple
    intro_slot_targets: tuple
    operation_mention_targets: tuple
    query_mention_target: tuple


def _tokens_in_span(offsets, start, end):
    positions = tuple(
        index for index, (left, right) in enumerate(offsets)
        if right > left and left >= start and right <= end
    )
    if not positions:
        raise ValueError("text span has no tokenizer tokens")
    return positions


def structural_token_spans(question, encoding):
    """Return intro/event/query token spans using formatting, not task labels."""
    spans = [span for span in _line_spans(question) if span[2].strip()]
    event_spans = [span for span in spans if span[2].startswith(("Step ", "Event "))]
    answer_spans = [span for span in spans if span[2].strip() in {"Answer:", "Result:"}]
    if not event_spans or not answer_spans:
        raise ValueError("question lacks event or answer lines")
    first_event_start = event_spans[0][0]
    intro_candidates = [span for span in spans if span[1] <= first_event_start]
    if len(intro_candidates) != 1:
        raise ValueError("question must have exactly one introductory line")
    answer_start = answer_spans[-1][0]
    query_candidates = [
        span for span in spans
        if span[1] <= answer_start and span not in event_spans and span not in intro_candidates
    ]
    if len(query_candidates) != 1:
        raise ValueError("question must have exactly one query line")
    intro = intro_candidates[0]
    query = query_candidates[0]
    return (
        _tokens_in_span(encoding.offsets, intro[0], intro[1]),
        tuple(_tokens_in_span(encoding.offsets, start, end) for start, end, _ in event_spans),
        _tokens_in_span(encoding.offsets, query[0], query[1]),
        intro,
        tuple(event_spans),
        query,
    )


def _literal_token_positions(text, literal, line_span, offsets):
    start, end, _ = line_span
    positions = set()
    variants = {str(literal), str(literal).replace("_", " ")}
    for variant in variants:
        pattern = re.compile(r"(?<!\w){}(?!\w)".format(re.escape(variant)), re.IGNORECASE)
        for match in pattern.finditer(text, start, end):
            for index, (left, right) in enumerate(offsets):
                if right > left and left < match.end() and right > match.start():
                    positions.add(index)
    if not positions:
        raise ValueError("literal {!r} has no tokens in line {!r}".format(literal, line_span[2]))
    return tuple(sorted(positions))


def _operation_target_key(operation):
    return None if operation["kind"] == "swap" else operation["target"]


def _query_target_key(query):
    if query["kind"] == "read":
        return query["key"]
    if query["kind"] == "difference":
        return query["high"]
    if query["kind"] == "sum":
        return None
    raise ValueError("unknown query kind {}".format(query["kind"]))


def compile_referential_example(row, tokenizer):
    """Compile labels and mention supervision without exposing them to inference."""
    question = row["question"]
    encoding = tokenizer.encode(question)
    compiled = compile_example(row, tokenizer)
    intro, operation_spans, query_span, intro_chars, operation_chars, query_chars = (
        structural_token_spans(question, encoding)
    )
    keys = tuple(row["keys"])
    if len(keys) != 2:
        raise ValueError("referential compiler requires exactly two keys")
    intro_targets = tuple(
        _literal_token_positions(question, key, intro_chars, encoding.offsets) for key in keys
    )
    operation_targets = []
    for operation, line_span, opcode in zip(row["operations"], operation_chars, compiled.operation_targets):
        key = _operation_target_key(operation)
        _, expected_role = factor_operation(opcode)
        if key is None:
            if expected_role != IGNORE_ROLE:
                raise ValueError("role-free operation has a role label")
            operation_targets.append(())
            continue
        if keys.index(key) != expected_role:
            raise ValueError("operation mention role disagrees with opcode")
        operation_targets.append(_literal_token_positions(question, key, line_span, encoding.offsets))
    query_key = _query_target_key(row["query"])
    _, expected_query_role = factor_query(compiled.query_target)
    if query_key is None:
        if expected_query_role != IGNORE_ROLE:
            raise ValueError("role-free query has a role label")
        query_target = ()
    else:
        if keys.index(query_key) != expected_query_role:
            raise ValueError("query mention role disagrees with query label")
        query_target = _literal_token_positions(question, query_key, query_chars, encoding.offsets)
    return ReferentialCompiledExample(
        compiled=compiled,
        intro_positions=intro,
        operation_spans=operation_spans,
        query_span=query_span,
        intro_slot_targets=intro_targets,
        operation_mention_targets=tuple(operation_targets),
        query_mention_target=query_target,
    )


def attention_mass_loss(weights, positions, target_positions):
    """Negative log probability assigned to any token in a supervised mention."""
    if not target_positions:
        return weights.sum() * 0.0
    targets = set(map(int, target_positions))
    mask = torch.tensor([int(position) in targets for position in positions], device=weights.device)
    if not bool(mask.any()):
        raise ValueError("attention targets do not intersect structural span")
    return -torch.log(weights[mask].float().sum().clamp_min(1e-8))


class ReferentialSlotMicrocodeCompiler(nn.Module):
    """Frozen-base compiler whose candidate role head performs entity matching."""

    def __init__(self, model, layer=19, hidden=256, role_mode="pointer"):
        super().__init__()
        if role_mode not in {"absolute", "pointer"}:
            raise ValueError("role_mode must be absolute or pointer")
        if model.cfg.n_loop != 1:
            raise ValueError("referential compiler requires n_loop=1")
        if not 0 <= int(layer) < len(model.blocks):
            raise ValueError("invalid compiler layer")
        self.model = model
        self.layer = int(layer)
        self.role_mode = role_mode
        self.model.requires_grad_(False)
        width = model.cfg.d_model
        self.norm = nn.LayerNorm(width)
        self.trunk = nn.Sequential(
            nn.Linear(width, hidden, bias=False), nn.SiLU(),
            nn.Linear(hidden, hidden, bias=False), nn.SiLU(),
        )
        role_width = max(16, hidden // 2)
        self.operation_kind_score = nn.Linear(hidden, 1, bias=False)
        self.query_kind_score = nn.Linear(hidden, 1, bias=False)
        self.operation_target_score = nn.Linear(hidden, 1, bias=False)
        self.query_target_score = nn.Linear(hidden, 1, bias=False)
        self.intro_slot_score = nn.Linear(hidden, 2, bias=False)
        self.operation_kind_head = nn.Linear(hidden, len(OPERATION_KINDS))
        self.query_kind_head = nn.Linear(hidden, len(QUERY_KINDS))
        self.identity_projection = nn.Linear(width, role_width, bias=False)
        self.absolute_operation_role_head = nn.Linear(hidden, 2)
        self.absolute_query_role_head = nn.Linear(hidden, 2)
        self.log_role_scale = nn.Parameter(torch.tensor(2.0))
        self.transition_logits = nn.Parameter(torch.zeros(2, 2, 10, 10, 20))

    def adapter_parameters(self):
        for name, parameter in self.named_parameters():
            if not name.startswith("model."):
                yield parameter

    def adapter_num_params(self):
        return sum(parameter.numel() for parameter in self.adapter_parameters())

    def encode(self, ids):
        if ids.ndim != 2 or ids.dtype != torch.long:
            raise ValueError("ids must be rank-2 torch.long")
        x = self.model.tok(ids)
        identity = x.detach()
        cos = self.model.cos[:ids.shape[1]].to(x.device)
        sin = self.model.sin[:ids.shape[1]].to(x.device)
        for block in self.model.blocks[:self.layer + 1]:
            x, _ = block(x, cos, sin)
        return x.detach(), identity

    @staticmethod
    def _pool(values, positions, scorer):
        index = torch.tensor(positions, dtype=torch.long, device=values.device)
        selected = values.index_select(0, index)
        weights = torch.softmax(scorer(selected).squeeze(-1).float(), dim=0)
        pooled = (weights.to(selected.dtype).unsqueeze(-1) * selected).sum(0)
        return pooled, weights

    def _intro_slots(self, features, identity, positions):
        index = torch.tensor(positions, dtype=torch.long, device=features.device)
        selected_features = features.index_select(0, index)
        selected_identity = self.identity_projection(identity.index_select(0, index))
        weights = torch.softmax(self.intro_slot_score(selected_features).float(), dim=0)
        slots = torch.einsum("ls,ld->sd", weights.to(selected_identity.dtype), selected_identity)
        return F.normalize(slots.float(), dim=-1), weights

    def _role_logits(self, target_identity, target_context, slots, kind):
        if self.role_mode == "pointer":
            target = F.normalize(target_identity.float(), dim=-1)
            scale = self.log_role_scale.float().clamp(0.0, 4.0).exp()
            return scale * torch.mv(slots, target)
        head = self.absolute_operation_role_head if kind == "operation" else self.absolute_query_role_head
        return head(target_context).float()

    def _line_outputs(self, features, identity, positions, slots, kind):
        kind_scorer = self.operation_kind_score if kind == "operation" else self.query_kind_score
        target_scorer = self.operation_target_score if kind == "operation" else self.query_target_score
        kind_head = self.operation_kind_head if kind == "operation" else self.query_kind_head
        kind_context, kind_weights = self._pool(features, positions, kind_scorer)
        target_context, target_weights = self._pool(features, positions, target_scorer)
        index = torch.tensor(positions, dtype=torch.long, device=features.device)
        projected_identity = self.identity_projection(identity.index_select(0, index))
        target_identity = (
            target_weights.to(projected_identity.dtype).unsqueeze(-1) * projected_identity
        ).sum(0)
        return {
            "kind_logits": kind_head(kind_context).float(),
            "role_logits": self._role_logits(target_identity, target_context, slots, kind),
            "kind_context": kind_context,
            "kind_weights": kind_weights,
            "target_weights": target_weights,
            "positions": positions,
        }

    def classify_text(self, hidden, identity, intro_positions, operation_spans, query_span):
        """Classify from text states and structural spans only; no key labels enter."""
        if hidden.ndim != 2 or identity.shape != hidden.shape:
            raise ValueError("single-example hidden and identity tensors are required")
        features = self.trunk(self.norm(hidden))
        slots, intro_weights = self._intro_slots(features, identity, intro_positions)
        operations = [
            self._line_outputs(features, identity, positions, slots, "operation")
            for positions in operation_spans
        ]
        query = self._line_outputs(features, identity, query_span, slots, "query")
        return {
            "slots": slots,
            "intro_weights": intro_weights,
            "intro_positions": intro_positions,
            "operations": operations,
            "query": query,
        }

    @staticmethod
    def compose_operation_logits(kind_logits, role_logits):
        return RoleEquivariantMicrocodeCompiler.compose_operation_logits(kind_logits, role_logits)

    @staticmethod
    def compose_query_logits(kind_logits, role_logits):
        return RoleEquivariantMicrocodeCompiler.compose_query_logits(kind_logits, role_logits)

    def basis_loss(self):
        targets = transition_basis_targets(self.transition_logits.device)
        return F.cross_entropy(self.transition_logits.reshape(-1, 20), targets.reshape(-1))
