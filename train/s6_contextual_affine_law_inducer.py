"""Card-conditioned destination model for S6 unseen-law induction."""

from __future__ import annotations

import torch
import torch.nn as nn


MAX_MODULUS = 13
BASE_REASONING_SYSTEM_PARAMETERS = 133_694_869


class ContextualAffineLawInducer(nn.Module):
    """Infer a destination from two law witnesses and a current location."""

    def __init__(
        self,
        width: int = 256,
        layers: int = 6,
        heads: int = 8,
        feedforward: int = 1024,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.width = int(width)
        self.layers = int(layers)
        self.heads = int(heads)
        self.feedforward = int(feedforward)
        self.role_embedding = nn.Embedding(4, self.width)
        self.modulus_embedding = nn.Embedding(MAX_MODULUS + 1, self.width)
        self.input_embedding = nn.Embedding(MAX_MODULUS, self.width)
        self.output_embedding = nn.Embedding(MAX_MODULUS, self.width)
        block = nn.TransformerEncoderLayer(
            d_model=self.width,
            nhead=self.heads,
            dim_feedforward=self.feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            block, num_layers=self.layers, norm=nn.LayerNorm(self.width)
        )
        self.destination = nn.Linear(self.width, MAX_MODULUS)

    def num_params(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def total_system_params(self) -> int:
        return BASE_REASONING_SYSTEM_PARAMETERS + self.num_params()

    def forward(
        self,
        modulus: torch.Tensor,
        card_y0: torch.Tensor,
        card_y1: torch.Tensor,
        current_location: torch.Tensor,
    ) -> torch.Tensor:
        tensors = (modulus, card_y0, card_y1, current_location)
        if any(tensor.ndim != 1 for tensor in tensors):
            raise ValueError("S6 inducer inputs must be rank-one")
        if any(tensor.shape != modulus.shape for tensor in tensors[1:]):
            raise ValueError("S6 inducer inputs must have matching shapes")
        if not bool(((modulus >= 2) & (modulus <= MAX_MODULUS)).all()):
            raise ValueError("S6 modulus outside model range")

        batch = modulus.shape[0]
        roles = torch.arange(4, device=modulus.device).expand(batch, -1)
        modulus_features = self.modulus_embedding(modulus.long()).unsqueeze(1)
        tokens = self.role_embedding(roles)
        tokens = tokens + modulus_features
        tokens[:, 1] = tokens[:, 1] + self.input_embedding.weight[0]
        tokens[:, 1] = tokens[:, 1] + self.output_embedding(card_y0.long())
        tokens[:, 2] = tokens[:, 2] + self.input_embedding.weight[1]
        tokens[:, 2] = tokens[:, 2] + self.output_embedding(card_y1.long())
        tokens[:, 3] = tokens[:, 3] + self.input_embedding(current_location.long())
        encoded = self.encoder(tokens)
        logits = self.destination(encoded[:, 3])
        positions = torch.arange(MAX_MODULUS, device=modulus.device).unsqueeze(0)
        return logits.masked_fill(positions >= modulus.unsqueeze(1), float("-inf"))


class LawIdMemorizer(nn.Module):
    """Favorable matched control that has IDs for train laws but no law card."""

    def __init__(
        self,
        train_law_count: int,
        width: int = 256,
        layers: int = 6,
        heads: int = 8,
        feedforward: int = 1024,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.train_law_count = int(train_law_count)
        self.oov_law_id = self.train_law_count
        self.backbone = ContextualAffineLawInducer(
            width=width,
            layers=layers,
            heads=heads,
            feedforward=feedforward,
            dropout=dropout,
        )
        self.law_embedding = nn.Embedding(self.train_law_count + 1, width)

    def num_params(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def forward(
        self,
        modulus: torch.Tensor,
        law_id: torch.Tensor,
        current_location: torch.Tensor,
    ) -> torch.Tensor:
        if law_id.ndim != 1 or law_id.shape != modulus.shape:
            raise ValueError("S6 law IDs must match modulus shape")
        zeros = torch.zeros_like(current_location)
        batch = modulus.shape[0]
        roles = torch.arange(4, device=modulus.device).expand(batch, -1)
        tokens = self.backbone.role_embedding(roles)
        tokens = tokens + self.backbone.modulus_embedding(modulus.long()).unsqueeze(1)
        law_features = self.law_embedding(law_id.long())
        tokens[:, 0] = tokens[:, 0] + law_features
        tokens[:, 1] = tokens[:, 1] + law_features
        tokens[:, 2] = tokens[:, 2] + law_features
        tokens[:, 3] = tokens[:, 3] + self.backbone.input_embedding(
            current_location.long()
        )
        tokens[:, 1] = tokens[:, 1] + self.backbone.input_embedding(zeros)
        tokens[:, 2] = tokens[:, 2] + self.backbone.input_embedding(
            torch.ones_like(zeros)
        )
        encoded = self.backbone.encoder(tokens)
        logits = self.backbone.destination(encoded[:, 3])
        positions = torch.arange(MAX_MODULUS, device=modulus.device).unsqueeze(0)
        return logits.masked_fill(positions >= modulus.unsqueeze(1), float("-inf"))

