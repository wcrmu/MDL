from __future__ import annotations

import torch
from torch import Tensor, nn


class PerTokenLinear(nn.Module):
    def __init__(self, num_tokens: int, input_dim: int, output_dim: int) -> None:
        super().__init__()
        if num_tokens <= 0:
            raise ValueError("num_tokens must be positive")
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if output_dim <= 0:
            raise ValueError("output_dim must be positive")
        self.num_tokens = num_tokens
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layers = nn.ModuleList(
            nn.Linear(input_dim, output_dim) for _ in range(num_tokens)
        )

    def forward(self, tokens: Tensor) -> Tensor:
        if tokens.ndim != 3:
            raise ValueError("tokens must have shape [batch, num_tokens, input_dim]")
        if tokens.size(1) != self.num_tokens or tokens.size(2) != self.input_dim:
            raise ValueError(
                f"expected tokens with shape [batch, {self.num_tokens}, {self.input_dim}], "
                f"got {tuple(tokens.shape)}"
            )
        outputs = [
            layer(tokens[:, token_index, :]).unsqueeze(1)
            for token_index, layer in enumerate(self.layers)
        ]
        return torch.cat(outputs, dim=1)


def _activation_layer(name: str) -> type[nn.Module]:
    if name == "gelu":
        return nn.GELU
    if name == "relu":
        return nn.ReLU
    raise ValueError("activation must be 'relu' or 'gelu'")


class PerTokenFFN(nn.Module):
    def __init__(
        self,
        num_tokens: int,
        token_dim: int,
        hidden_dim: int,
        dropout: float = 0.0,
        output_relu: bool = False,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        activation_layer = _activation_layer(activation)
        self.networks = nn.ModuleList(
            nn.Sequential(
                nn.Linear(token_dim, hidden_dim),
                activation_layer(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, token_dim),
                nn.ReLU() if output_relu else nn.Identity(),
            )
            for _ in range(num_tokens)
        )

    def forward(self, tokens: Tensor) -> Tensor:
        if tokens.size(1) != len(self.networks):
            raise ValueError(f"expected {len(self.networks)} tokens, got {tokens.size(1)}")
        outputs = [
            network(tokens[:, token_index, :]).unsqueeze(1)
            for token_index, network in enumerate(self.networks)
        ]
        return torch.cat(outputs, dim=1)


def _zero_like_loss(reference: Tensor) -> Tensor:
    return reference.new_zeros(())


class SparseMoEPerTokenFFN(nn.Module):
    def __init__(
        self,
        num_tokens: int,
        token_dim: int,
        hidden_dim: int,
        num_experts: int = 4,
        dropout: float = 0.0,
        output_relu: bool = False,
        activation: str = "gelu",
        use_dtsi: bool = True,
        inference_threshold: float = 0.0,
    ) -> None:
        super().__init__()
        if num_tokens <= 0:
            raise ValueError("num_tokens must be positive")
        if token_dim <= 0:
            raise ValueError("token_dim must be positive")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if num_experts <= 0:
            raise ValueError("num_experts must be positive")
        if inference_threshold < 0:
            raise ValueError("inference_threshold must be non-negative")
        activation_layer = _activation_layer(activation)
        self.num_tokens = num_tokens
        self.token_dim = token_dim
        self.num_experts = num_experts
        self.use_dtsi = use_dtsi
        self.inference_threshold = inference_threshold
        self.experts = nn.ModuleList(
            nn.ModuleList(
                nn.Sequential(
                    nn.Linear(token_dim, hidden_dim),
                    activation_layer(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, token_dim),
                    nn.ReLU() if output_relu else nn.Identity(),
                )
                for _ in range(num_experts)
            )
            for _ in range(num_tokens)
        )
        self.train_routers = nn.ModuleList(
            nn.Linear(token_dim, num_experts) for _ in range(num_tokens)
        )
        self.infer_routers = nn.ModuleList(
            nn.Linear(token_dim, num_experts) for _ in range(num_tokens)
        )
        self._last_regularization_loss: Tensor | None = None
        self._last_active_ratio: Tensor | None = None

    def _router_gates(self, router: nn.Linear, token_values: Tensor) -> Tensor:
        return torch.relu(router(token_values))

    def _dense_mixture(
        self,
        experts: nn.ModuleList,
        token_values: Tensor,
        gates: Tensor,
    ) -> Tensor:
        expert_outputs = torch.stack(
            [expert(token_values) for expert in experts],
            dim=1,
        )
        return (expert_outputs * gates.unsqueeze(-1)).sum(dim=1)

    def _sparse_mixture(
        self,
        experts: nn.ModuleList,
        token_values: Tensor,
        gates: Tensor,
    ) -> Tensor:
        output = token_values.new_zeros(token_values.size(0), self.token_dim)
        for expert_index, expert in enumerate(experts):
            active = gates[:, expert_index] > self.inference_threshold
            if bool(active.any()):
                output[active] = output[active] + gates[active, expert_index].unsqueeze(1) * expert(
                    token_values[active]
                )
        return output

    def forward(self, tokens: Tensor) -> Tensor:
        if tokens.ndim != 3:
            raise ValueError("tokens must have shape [batch, num_tokens, token_dim]")
        if tokens.size(1) != self.num_tokens or tokens.size(2) != self.token_dim:
            raise ValueError(
                f"expected tokens with shape [batch, {self.num_tokens}, {self.token_dim}], "
                f"got {tuple(tokens.shape)}"
            )

        outputs: list[Tensor] = []
        regularization_terms: list[Tensor] = []
        active_ratios: list[Tensor] = []
        for token_index in range(self.num_tokens):
            token_values = tokens[:, token_index, :]
            infer_gates = self._router_gates(self.infer_routers[token_index], token_values)
            regularization_terms.append(infer_gates.sum(dim=1).mean())
            active_ratios.append((infer_gates > self.inference_threshold).to(dtype=tokens.dtype).mean())

            if self.training:
                infer_output = self._dense_mixture(
                    self.experts[token_index],
                    token_values,
                    infer_gates,
                )
                if self.use_dtsi:
                    train_gates = self._router_gates(self.train_routers[token_index], token_values)
                    train_output = self._dense_mixture(
                        self.experts[token_index],
                        token_values,
                        train_gates,
                    )
                    outputs.append(0.5 * (train_output + infer_output))
                else:
                    outputs.append(infer_output)
            else:
                outputs.append(
                    self._sparse_mixture(
                        self.experts[token_index],
                        token_values,
                        infer_gates,
                    )
                )

        self._last_regularization_loss = torch.stack(regularization_terms).sum()
        self._last_active_ratio = torch.stack(active_ratios).mean().detach()
        return torch.stack(outputs, dim=1)

    def regularization_loss(self, reference: Tensor | None = None) -> Tensor:
        if self._last_regularization_loss is not None:
            return self._last_regularization_loss
        if reference is None:
            return torch.tensor(0.0)
        return _zero_like_loss(reference)

    def active_ratio(self, reference: Tensor | None = None) -> Tensor:
        if self._last_active_ratio is not None:
            return self._last_active_ratio
        if reference is None:
            return torch.tensor(0.0)
        return _zero_like_loss(reference).detach()


class ContextTokenizer(nn.Module):
    def __init__(
        self,
        num_tokens: int,
        context_dim: int,
        token_dim: int,
        hidden_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.networks = nn.ModuleList(
            nn.Sequential(
                nn.Linear(context_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, token_dim),
                nn.ReLU(),
            )
            for _ in range(num_tokens)
        )

    def forward(self, context: Tensor) -> Tensor:
        return torch.stack([network(context) for network in self.networks], dim=1)
