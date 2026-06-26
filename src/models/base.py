from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable

from torch import Tensor, nn


class BaseRecommender(nn.Module, ABC):
    @abstractmethod
    def forward(self, *args: object, **kwargs: object) -> dict[str, Tensor] | Tensor:
        raise NotImplementedError


def count_parameters(modules: Iterable[nn.Module]) -> int:
    return sum(parameter.numel() for module in modules for parameter in module.parameters())
