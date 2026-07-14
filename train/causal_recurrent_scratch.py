"""Frozen-base, source-visible recurrent scratch adapter.

This is an isolated architecture experiment.  A small shared recurrent cell
reads the ordinary prompt residuals at one transformer boundary, updates a
fixed set of continuous scratch slots, and injects a query-conditioned readout
only into answer-predicting positions.  The original prompt remains visible to
the upper transformer blocks.

The adapter is intentionally causal and removable:

* scratch states are computed only from prompt positions;
* the base GPT parameters are frozen;
* a zero readout gate is exactly the frozen GPT;
* normal, zero, shuffled, and externally supplied states share one decoder;
* recurrent depth changes compute, not parameter count.

Those properties make state-necessity and recurrence claims falsifiable.  They
do not by themselves establish reasoning.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from latent_rollout import build_answer_targets


def _check_ids(name: str, ids: torch.Tensor) -> None:
    if ids.ndim != 2 or ids.dtype != torch.long or not ids.shape[1]:
        raise ValueError("{} must be a nonempty rank-2 torch.long tensor".format(name))


class CausalRecurrentScratch(nn.Module):
    """Add a tiny recurrent workspace to a frozen causal language model."""

    def __init__(
        self,
        model,
        layer: int,
        slots: int = 4,
        width: int = 96,
        workspace_topk: int = 0,
        workspace_temperature: float = 0.2,
    ):
        super().__init__()
        if model.cfg.n_loop != 1:
            raise ValueError("causal recurrent scratch requires n_loop=1")
        if not 0 <= int(layer) < len(model.blocks) - 1:
            raise ValueError("layer must leave at least one upper transformer block")
        if slots <= 0 or width <= 0:
            raise ValueError("slots and width must be positive")
        if workspace_topk < 0 or workspace_topk > model.cfg.vocab_size:
            raise ValueError("workspace_topk must be in [0, vocab_size]")
        if workspace_topk and workspace_temperature <= 0:
            raise ValueError("workspace_temperature must be positive")
        self.model = model
        self.layer = int(layer)
        self.slots = int(slots)
        self.width = int(width)
        self.workspace_topk = int(workspace_topk)
        self.workspace_temperature = float(workspace_temperature)
        d_model = model.cfg.d_model

        # The base is an immutable feature extractor and decoder.  Any gain or
        # regression with the adapter enabled is therefore attributable to the
        # new state path rather than broad checkpoint drift.
        self.model.requires_grad_(False)

        self.initial_state = nn.Parameter(torch.zeros(self.slots, self.width))
        nn.init.normal_(self.initial_state, mean=0.0, std=0.02)
        self.source_norm = nn.LayerNorm(d_model)
        self.source_key = nn.Linear(d_model, self.width, bias=False)
        self.source_value = nn.Linear(d_model, self.width, bias=False)
        self.state_norm = nn.LayerNorm(self.width)
        self.state_query = nn.Linear(self.width, self.width, bias=False)
        self.cell = nn.GRUCell(self.width, self.width)

        self.read_query = nn.Linear(d_model, self.width, bias=False)
        self.read_key = nn.Linear(self.width, self.width, bias=False)
        self.read_value = nn.Linear(self.width, d_model, bias=False)
        # tanh(0) is exactly zero.  This makes the initialized wrapper and the
        # explicit disabled control byte-for-byte equivalent in real arithmetic.
        self.readout_gate = nn.Parameter(torch.zeros(()))

    def adapter_parameters(self):
        for name, parameter in self.named_parameters():
            if not name.startswith("model."):
                yield parameter

    def adapter_num_params(self) -> int:
        return sum(parameter.numel() for parameter in self.adapter_parameters())

    def _lower(self, ids: torch.Tensor) -> torch.Tensor:
        _check_ids("ids", ids)
        if ids.shape[1] > self.model.cfg.seq_len:
            raise ValueError("input exceeds configured sequence length")
        x = self.model.tok(ids)
        cos = self.model.cos[:ids.shape[1]].to(x.device)
        sin = self.model.sin[:ids.shape[1]].to(x.device)
        for block in self.model.blocks[:self.layer + 1]:
            x, _ = block(x, cos, sin)
        return x

    def _upper_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        cos = self.model.cos[:hidden.shape[1]].to(hidden.device)
        sin = self.model.sin[:hidden.shape[1]].to(hidden.device)
        for block in self.model.blocks[self.layer + 1:]:
            hidden, _ = block(hidden, cos, sin)
        return self.model.head(self.model.norm(hidden))

    def compute_scratch(
        self,
        source_hidden: torch.Tensor,
        steps: int,
        *,
        recurrent: bool = True,
        return_trajectory: bool = False,
    ):
        """Read prompt residuals into shared recurrent slots.

        ``recurrent=False`` is a compute-shaped reset control: every iteration
        starts from the same learned initial slots.  It executes the same cell
        the same number of times, but cannot accumulate state across steps.
        """
        if source_hidden.ndim != 3 or source_hidden.shape[-1] != self.model.cfg.d_model:
            raise ValueError("source_hidden must have shape [batch, tokens, d_model]")
        if not source_hidden.shape[1]:
            raise ValueError("source_hidden must contain prompt tokens")
        if steps <= 0:
            raise ValueError("scratch steps must be positive")
        batch = source_hidden.shape[0]
        source = self.source_norm(source_hidden)
        keys = self.source_key(source)
        values = self.source_value(source)
        initial = self.initial_state.to(dtype=source.dtype, device=source.device).unsqueeze(0)
        initial = initial.expand(batch, -1, -1)
        state = initial
        trajectory = []
        for _ in range(steps):
            previous = state if recurrent else initial
            queries = self.state_query(self.state_norm(previous))
            attention = torch.matmul(queries, keys.transpose(-2, -1)) / math.sqrt(self.width)
            context = torch.matmul(F.softmax(attention.float(), dim=-1).to(values.dtype), values)
            state = self.cell(
                context.reshape(batch * self.slots, self.width),
                previous.reshape(batch * self.slots, self.width),
            ).reshape(batch, self.slots, self.width)
            trajectory.append(state)
        if not recurrent:
            # Every reset execution has the same input and therefore the same
            # value. Averaging connects all executions to backward while
            # preserving the one-step, non-accumulating function. This makes
            # the reset arm a genuine compute-shaped control.
            state = torch.stack(trajectory, dim=0).mean(dim=0)
        if return_trajectory:
            return state, tuple(trajectory)
        return state

    def read_scratch(self, query_hidden: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        if query_hidden.ndim != 3 or query_hidden.shape[-1] != self.model.cfg.d_model:
            raise ValueError("query_hidden must have shape [batch, tokens, d_model]")
        if state.ndim != 3 or state.shape != (
            query_hidden.shape[0], self.slots, self.width,
        ):
            raise ValueError("state has the wrong batch, slot, or width shape")
        queries = self.read_query(query_hidden)
        keys = self.read_key(state)
        attention = torch.matmul(queries, keys.transpose(-2, -1)) / math.sqrt(self.width)
        weights = F.softmax(attention.float(), dim=-1).to(state.dtype)
        context = torch.matmul(weights, state)
        projected = self.read_value(context)
        if not self.workspace_topk:
            return projected

        # Late-layer unembedding directions are a cheap approximation to the
        # model's verbalizable workspace coordinates.  Restricting the write
        # to a sparse nonnegative mixture tests whether downstream readability,
        # rather than raw vector capacity, is the missing ingredient.  This is
        # not a Jacobian lens: the approximation is explicit in checkpoint
        # metadata and must be compared against controls.
        probe = F.normalize(projected.float(), dim=-1)
        basis = F.normalize(self.model.head.weight.detach().float(), dim=-1)
        scores = torch.matmul(probe, basis.transpose(0, 1)) / self.workspace_temperature
        values, indices = torch.topk(scores, k=self.workspace_topk, dim=-1)
        mixture = F.softmax(values, dim=-1)
        selected = basis[indices]
        delta = (mixture.unsqueeze(-1) * selected).sum(dim=-2)
        residual_rms = query_hidden.detach().float().square().mean(dim=-1, keepdim=True).sqrt()
        delta = delta * residual_rms * math.sqrt(self.model.cfg.d_model)
        return delta.to(context.dtype)

    @torch.no_grad()
    def encode_prompt(self, prompt_ids: torch.Tensor, steps: int, *, recurrent: bool = True) -> torch.Tensor:
        """Return a reusable scratch state computed from an ordinary prompt."""
        _check_ids("prompt_ids", prompt_ids)
        return self.compute_scratch(self._lower(prompt_ids).detach(), steps, recurrent=recurrent)

    def forward_ids(
        self,
        ids: torch.Tensor,
        prompt_len: int,
        steps: int,
        *,
        state_mode: str = "normal",
        recurrent: bool = True,
        state_override: torch.Tensor | None = None,
        return_state: bool = False,
    ):
        """Return logits with a scratch intervention at answer positions.

        ``prompt_len - 1`` is the first answer-predicting position under the
        project's next-token target convention.  Scratch construction slices
        strictly before ``prompt_len``, so teacher-forced answer tokens cannot
        leak into the recurrent state.
        """
        _check_ids("ids", ids)
        if not 1 <= int(prompt_len) <= ids.shape[1]:
            raise ValueError("prompt_len must identify a nonempty prefix")
        if state_mode not in {"normal", "zero", "shuffled", "disabled", "override"}:
            raise ValueError("unknown state_mode {}".format(state_mode))
        if state_mode == "override" and state_override is None:
            raise ValueError("override mode requires state_override")
        hidden = self._lower(ids)
        source_hidden = hidden[:, :prompt_len, :].detach()
        state = None
        if state_mode != "disabled":
            state = self.compute_scratch(source_hidden, steps, recurrent=recurrent)
            if state_mode == "zero":
                state = torch.zeros_like(state)
            elif state_mode == "shuffled":
                if state.shape[0] < 2:
                    raise ValueError("shuffled state requires batch size at least two")
                state = state.roll(1, dims=0)
            elif state_mode == "override":
                if state_override.shape != state.shape:
                    raise ValueError("state_override has the wrong shape")
                state = state_override.to(device=state.device, dtype=state.dtype)
            suffix = hidden[:, prompt_len - 1:, :]
            delta = self.read_scratch(suffix, state)
            hidden = torch.cat((
                hidden[:, :prompt_len - 1, :],
                suffix + torch.tanh(self.readout_gate).to(suffix.dtype) * delta,
            ), dim=1)
        logits = self._upper_logits(hidden)
        if return_state:
            return logits, state
        return logits

    def supervised_loss(
        self,
        prompt_ids: torch.Tensor,
        answer_ids: torch.Tensor,
        eos_id: int,
        steps: int,
        *,
        state_mode: str = "normal",
        recurrent: bool = True,
        state_override: torch.Tensor | None = None,
    ):
        _check_ids("prompt_ids", prompt_ids)
        _check_ids("answer_ids", answer_ids)
        if prompt_ids.shape[0] != answer_ids.shape[0]:
            raise ValueError("prompt and answer batch sizes differ")
        if prompt_ids.shape[1] + answer_ids.shape[1] > self.model.cfg.seq_len:
            raise ValueError("prompt and answer exceed configured sequence length")
        ids = torch.cat((prompt_ids, answer_ids), dim=1)
        targets = build_answer_targets(answer_ids, prompt_ids.shape[1], 0, eos_id)
        logits, state = self.forward_ids(
            ids, prompt_ids.shape[1], steps, state_mode=state_mode,
            recurrent=recurrent, state_override=state_override, return_state=True,
        )
        loss = F.cross_entropy(
            logits.float().reshape(-1, logits.shape[-1]),
            targets.reshape(-1),
            ignore_index=-1,
        )
        return logits, loss, state, targets


@torch.no_grad()
def generate_with_scratch(
    adapter: CausalRecurrentScratch,
    prompt_ids: torch.Tensor,
    steps: int,
    eos_id: int,
    max_new: int,
    *,
    state_mode: str = "normal",
    recurrent: bool = True,
    state_override: torch.Tensor | None = None,
) -> torch.Tensor:
    """Greedily decode while recomputing the isolated scratch intervention."""
    _check_ids("prompt_ids", prompt_ids)
    if prompt_ids.shape[0] != 1:
        raise ValueError("scratch generation currently accepts one prompt")
    if max_new <= 0:
        raise ValueError("max_new must be positive")
    ids = prompt_ids
    generated = []
    prompt_len = prompt_ids.shape[1]
    for _ in range(max_new):
        logits = adapter.forward_ids(
            ids, prompt_len, steps, state_mode=state_mode, recurrent=recurrent,
            state_override=state_override,
        )
        next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        token = int(next_id.item())
        if token == int(eos_id) or ids.shape[1] >= adapter.model.cfg.seq_len:
            break
        generated.append(token)
        ids = torch.cat((ids, next_id), dim=1)
    return torch.tensor(generated, dtype=torch.long, device=prompt_ids.device)
