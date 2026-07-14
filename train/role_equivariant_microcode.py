"""Role-factorized, permutation-equivariant microcode compiler.

The compiler separates an operation's semantic kind from the register role it
acts on.  With two registers, a single target bit identifies the argument for
add/sub and the destination for move/merge.  Query kind and selected register
are factorized in the same way.  This exposes a precise counterfactual
intervention: swapping register identities must preserve kind logits and swap
role logits.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from categorical_microcode import (
    OPCODES,
    QUERIES,
    transition_basis_targets,
)


OPERATION_KINDS = ("add", "sub", "move", "merge", "swap")
QUERY_KINDS = ("read", "sum", "difference")
OPERATION_KIND_TO_ID = {name: index for index, name in enumerate(OPERATION_KINDS)}
QUERY_KIND_TO_ID = {name: index for index, name in enumerate(QUERY_KINDS)}
IGNORE_ROLE = -100


def factor_operation(opcode):
    name = OPCODES[int(opcode)] if isinstance(opcode, int) else str(opcode)
    kind = name.split("_", 1)[0]
    if kind == "swap":
        return OPERATION_KIND_TO_ID[kind], IGNORE_ROLE
    return OPERATION_KIND_TO_ID[kind], int(name.rsplit("_", 1)[-1])


def compose_operation(kind, role):
    name = OPERATION_KINDS[int(kind)] if isinstance(kind, int) else str(kind)
    if name == "swap":
        return OPCODES.index("swap")
    target = int(role)
    if target not in (0, 1):
        raise ValueError("operation role must be 0 or 1")
    if name in {"add", "sub"}:
        opcode = "{}_{}".format(name, target)
    elif name in {"move", "merge"}:
        opcode = "{}_{}_{}".format(name, 1 - target, target)
    else:
        raise ValueError("unknown operation kind {}".format(name))
    return OPCODES.index(opcode)


def factor_query(query):
    name = QUERIES[int(query)] if isinstance(query, int) else str(query)
    kind = name.split("_", 1)[0]
    if kind == "sum":
        return QUERY_KIND_TO_ID[kind], IGNORE_ROLE
    return QUERY_KIND_TO_ID[kind], int(name.split("_")[1])


def compose_query(kind, role):
    name = QUERY_KINDS[int(kind)] if isinstance(kind, int) else str(kind)
    if name == "sum":
        return QUERIES.index("sum")
    selected = int(role)
    if selected not in (0, 1):
        raise ValueError("query role must be 0 or 1")
    if name == "read":
        query = "read_{}".format(selected)
    elif name == "difference":
        query = "difference_{}_{}".format(selected, 1 - selected)
    else:
        raise ValueError("unknown query kind {}".format(name))
    return QUERIES.index(query)


def permute_opcode(opcode):
    kind, role = factor_operation(opcode)
    return compose_operation(kind, role if role == IGNORE_ROLE else 1 - role)


def permute_query(query):
    kind, role = factor_query(query)
    return compose_query(kind, role if role == IGNORE_ROLE else 1 - role)


def factored_operation_targets(targets, device=None):
    factors = [factor_operation(int(target)) for target in targets]
    return (
        torch.tensor([kind for kind, _ in factors], dtype=torch.long, device=device),
        torch.tensor([role for _, role in factors], dtype=torch.long, device=device),
    )


def factored_query_targets(targets, device=None):
    factors = [factor_query(int(target)) for target in targets]
    return (
        torch.tensor([kind for kind, _ in factors], dtype=torch.long, device=device),
        torch.tensor([role for _, role in factors], dtype=torch.long, device=device),
    )


class RoleEquivariantMicrocodeCompiler(nn.Module):
    """Frozen-base compiler with separate semantic-kind and register-role heads."""

    def __init__(self, model, layer=19, hidden=256):
        super().__init__()
        if model.cfg.n_loop != 1:
            raise ValueError("role compiler requires n_loop=1")
        if not 0 <= int(layer) < len(model.blocks):
            raise ValueError("invalid compiler layer")
        self.model = model
        self.layer = int(layer)
        self.model.requires_grad_(False)
        width = model.cfg.d_model
        self.norm = nn.LayerNorm(width)
        self.trunk = nn.Sequential(
            nn.Linear(width, hidden, bias=False), nn.SiLU(),
            nn.Linear(hidden, hidden, bias=False), nn.SiLU(),
        )
        self.operation_kind_head = nn.Linear(hidden, len(OPERATION_KINDS))
        self.operation_role_head = nn.Linear(hidden, 2)
        self.query_kind_head = nn.Linear(hidden, len(QUERY_KINDS))
        self.query_role_head = nn.Linear(hidden, 2)
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
        cos = self.model.cos[:ids.shape[1]].to(x.device)
        sin = self.model.sin[:ids.shape[1]].to(x.device)
        for block in self.model.blocks[:self.layer + 1]:
            x, _ = block(x, cos, sin)
        return x.detach()

    def position_features(self, hidden, batch_indices, token_positions):
        selected = hidden[batch_indices, token_positions]
        return self.trunk(self.norm(selected))

    def operation_factors(self, features):
        return self.operation_kind_head(features), self.operation_role_head(features)

    def query_factors(self, features):
        return self.query_kind_head(features), self.query_role_head(features)

    @staticmethod
    def compose_operation_logits(kind_logits, role_logits):
        kind_scores = F.log_softmax(kind_logits.float(), dim=-1)
        role_scores = F.log_softmax(role_logits.float(), dim=-1)
        columns = []
        for opcode in range(len(OPCODES)):
            kind, role = factor_operation(opcode)
            score = kind_scores[:, kind]
            if role != IGNORE_ROLE:
                score = score + role_scores[:, role]
            columns.append(score)
        return torch.stack(columns, dim=-1)

    @staticmethod
    def compose_query_logits(kind_logits, role_logits):
        kind_scores = F.log_softmax(kind_logits.float(), dim=-1)
        role_scores = F.log_softmax(role_logits.float(), dim=-1)
        columns = []
        for query in range(len(QUERIES)):
            kind, role = factor_query(query)
            score = kind_scores[:, kind]
            if role != IGNORE_ROLE:
                score = score + role_scores[:, role]
            columns.append(score)
        return torch.stack(columns, dim=-1)

    def classify_positions(self, hidden, batch_indices, token_positions, kind):
        features = self.position_features(hidden, batch_indices, token_positions)
        if kind == "operation":
            return self.compose_operation_logits(*self.operation_factors(features))
        if kind == "query":
            return self.compose_query_logits(*self.query_factors(features))
        raise ValueError("unknown classification kind")

    def basis_loss(self):
        targets = transition_basis_targets(self.transition_logits.device)
        return F.cross_entropy(self.transition_logits.reshape(-1, 20), targets.reshape(-1))
