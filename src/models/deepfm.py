from __future__ import annotations

import torch
from torch import Tensor, nn

from .base import BaseRecommender


class DeepFM(BaseRecommender):
    """Small generic DeepFM block for dense feature tensors.

    The production manifest path currently uses MDL. This module exists as a
    standard model slot and can be wired to a dataset-specific feature pipeline when dense
    feature-field tensors are available.
    """

    def __init__(
        self,
        num_fields: int,
        field_dim: int,
        hidden_dims: tuple[int, ...] = (128, 64),
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_fields = num_fields
        self.field_dim = field_dim
        input_dim = num_fields * field_dim
        layers: list[nn.Module] = []
        previous_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(previous_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            previous_dim = hidden_dim
        layers.append(nn.Linear(previous_dim, 1))
        self.deep = nn.Sequential(*layers)
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, field_embeddings: Tensor) -> dict[str, Tensor]:
        if field_embeddings.ndim != 3:
            raise ValueError("field_embeddings must have shape [batch, num_fields, field_dim]")
        if field_embeddings.size(1) != self.num_fields or field_embeddings.size(2) != self.field_dim:
            raise ValueError(
                "field_embeddings must have shape "
                f"[batch, {self.num_fields}, {self.field_dim}], got {tuple(field_embeddings.shape)}"
            )

        flattened = field_embeddings.flatten(start_dim=1)
        summed = field_embeddings.sum(dim=1)
        squared_sum = summed.square()
        sum_squared = field_embeddings.square().sum(dim=1)
        fm_second_order = 0.5 * (squared_sum - sum_squared).sum(dim=1, keepdim=True)
        logits = self.linear(flattened) + fm_second_order + self.deep(flattened)
        return {"logits": logits}

    @torch.no_grad()
    def predict_proba(self, field_embeddings: Tensor) -> Tensor:
        return torch.sigmoid(self.forward(field_embeddings)["logits"])
