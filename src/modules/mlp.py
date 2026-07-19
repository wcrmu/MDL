from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class PerTokenLinear(nn.Module):
    def __init__(self, num_tokens: int, input_dim: int, output_dim: int) -> None:
        super().__init__()
        if num_tokens <= 0 or input_dim <= 0 or output_dim <= 0:
            raise ValueError("num_tokens, input_dim, and output_dim must be positive")
        self.num_tokens = num_tokens
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight = nn.Parameter(
            torch.empty(num_tokens, output_dim, input_dim)
        )
        self.bias = nn.Parameter(torch.empty(num_tokens, output_dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for token_index in range(self.num_tokens):
            nn.init.kaiming_uniform_(self.weight[token_index], a=math.sqrt(5))
            bound = 1.0 / math.sqrt(self.input_dim)
            nn.init.uniform_(self.bias[token_index], -bound, bound)

    def forward(self, tokens: Tensor) -> Tensor:
        expected = (self.num_tokens, self.input_dim)
        if tokens.ndim != 3 or tuple(tokens.shape[1:]) != expected:
            raise ValueError(
                f"expected tokens with shape [batch, {self.num_tokens}, {self.input_dim}], "
                f"got {tuple(tokens.shape)}"
            )
        # [T,B,I] @ [T,I,O] -> [T,B,O], one batched GEMM instead of T launches.
        output = torch.bmm(
            tokens.transpose(0, 1),
            self.weight.transpose(1, 2),
        ).transpose(0, 1)
        return output + self.bias.unsqueeze(0).to(dtype=output.dtype)


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
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        activation_layer = _activation_layer(activation)
        self.dropout = dropout
        self.output_relu = output_relu
        self.activation = activation
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

    def _forward_independent(self, tokens: Tensor) -> Tensor:
        return torch.cat(
            [
                network(tokens[:, token_index, :]).unsqueeze(1)
                for token_index, network in enumerate(self.networks)
            ],
            dim=1,
        )

    def _forward_batched(self, tokens: Tensor) -> Tensor:
        # Keep the historical ModuleList parameter layout for checkpoint
        # compatibility, but execute the independent token MLPs as two batched
        # GEMMs instead of launching two Linear kernels per token.
        input_weight = torch.stack(
            [network[0].weight for network in self.networks],
            dim=0,
        )
        input_bias = torch.stack(
            [network[0].bias for network in self.networks],
            dim=0,
        )
        output_weight = torch.stack(
            [network[3].weight for network in self.networks],
            dim=0,
        )
        output_bias = torch.stack(
            [network[3].bias for network in self.networks],
            dim=0,
        )
        token_major = tokens.transpose(0, 1)
        hidden = torch.bmm(
            token_major,
            input_weight.transpose(1, 2),
        )
        hidden = hidden + input_bias.unsqueeze(1).to(dtype=hidden.dtype)
        hidden = F.gelu(hidden) if self.activation == "gelu" else F.relu(hidden)
        hidden = F.dropout(hidden, p=self.dropout, training=self.training)
        output = torch.bmm(
            hidden,
            output_weight.transpose(1, 2),
        )
        output = output + output_bias.unsqueeze(1).to(dtype=output.dtype)
        if self.output_relu:
            output = F.relu(output)
        return output.transpose(0, 1)

    def forward(self, tokens: Tensor) -> Tensor:
        if tokens.size(1) != len(self.networks):
            raise ValueError(f"expected {len(self.networks)} tokens, got {tokens.size(1)}")
        # Small independent CPU GEMMs parallelize better through the host BLAS;
        # CUDA benefits from amortizing their launch overhead as one batched op.
        if tokens.device.type != "cuda":
            return self._forward_independent(tokens)
        return self._forward_batched(tokens)


class StackedPerTokenFFN(nn.Module):
    """Independent token FFNs executed as two strided batched GEMMs."""

    def __init__(
        self,
        num_tokens: int,
        token_dim: int,
        hidden_dim: int,
        dropout: float = 0.0,
        output_relu: bool = False,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        if num_tokens <= 0 or token_dim <= 0 or hidden_dim <= 0:
            raise ValueError("token and hidden dimensions must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if activation not in {"gelu", "relu"}:
            raise ValueError("activation must be 'relu' or 'gelu'")
        self.num_tokens = num_tokens
        self.token_dim = token_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.output_relu = output_relu
        self.activation = activation
        self.input_weight = nn.Parameter(
            torch.empty(num_tokens, hidden_dim, token_dim)
        )
        self.input_bias = nn.Parameter(torch.empty(num_tokens, hidden_dim))
        self.output_weight = nn.Parameter(
            torch.empty(num_tokens, token_dim, hidden_dim)
        )
        self.output_bias = nn.Parameter(torch.empty(num_tokens, token_dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for token_index in range(self.num_tokens):
            nn.init.kaiming_uniform_(
                self.input_weight[token_index], a=math.sqrt(5)
            )
            input_bound = 1.0 / math.sqrt(self.token_dim)
            nn.init.uniform_(
                self.input_bias[token_index], -input_bound, input_bound
            )
            nn.init.kaiming_uniform_(
                self.output_weight[token_index], a=math.sqrt(5)
            )
            output_bound = 1.0 / math.sqrt(self.hidden_dim)
            nn.init.uniform_(
                self.output_bias[token_index], -output_bound, output_bound
            )

    def forward(self, tokens: Tensor) -> Tensor:
        expected = (self.num_tokens, self.token_dim)
        if tokens.ndim != 3 or tuple(tokens.shape[1:]) != expected:
            raise ValueError(
                f"expected tokens with shape [batch, {self.num_tokens}, "
                f"{self.token_dim}], got {tuple(tokens.shape)}"
            )
        token_major = tokens.transpose(0, 1)
        hidden = torch.bmm(
            token_major, self.input_weight.transpose(1, 2)
        )
        hidden = hidden + self.input_bias.unsqueeze(1).to(dtype=hidden.dtype)
        hidden = F.gelu(hidden) if self.activation == "gelu" else F.relu(hidden)
        hidden = F.dropout(hidden, p=self.dropout, training=self.training)
        output = torch.bmm(
            hidden, self.output_weight.transpose(1, 2)
        )
        output = output + self.output_bias.unsqueeze(1).to(dtype=output.dtype)
        if self.output_relu:
            output = F.relu(output)
        return output.transpose(0, 1)


class SparseMoEPerTokenFFN(nn.Module):
    """Per-token Sparse-MoE with ReLU routing and DTSI execution.

    Training evaluates a dense softmax router so every expert receives
    gradients and, when DTSI is enabled, a second ReLU router matching the
    sparse inference path.  Inference executes only experts whose ReLU gate
    exceeds ``inference_threshold``.
    """

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
        target_active_ratio: float = 0.25,
        regularization_initial: float = 1.0e-8,
        regularization_multiplier: float = 1.2,
        dtsi_training_output: str | None = None,
    ) -> None:
        super().__init__()
        if num_tokens <= 0 or token_dim <= 0 or hidden_dim <= 0 or num_experts <= 0:
            raise ValueError("token, hidden, and expert dimensions must be positive")
        if inference_threshold < 0.0:
            raise ValueError("inference_threshold must be non-negative")
        if not 0.0 < target_active_ratio <= 1.0:
            raise ValueError("target_active_ratio must be in (0, 1]")
        if regularization_initial <= 0.0:
            raise ValueError("regularization_initial must be positive")
        if regularization_multiplier <= 1.0:
            raise ValueError("regularization_multiplier must be greater than 1")
        if dtsi_training_output not in {None, "dense_router", "mean"}:
            raise ValueError(
                "dtsi_training_output must be dense_router, mean, or None"
            )
        if use_dtsi and dtsi_training_output is None:
            raise ValueError(
                "DTSI training output is not specified by RankMixer; choose "
                "dense_router or mean explicitly"
            )
        activation_layer = _activation_layer(activation)
        self.num_tokens = num_tokens
        self.token_dim = token_dim
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.activation = activation
        self.dropout = dropout
        self.output_relu = output_relu
        self.use_dtsi = use_dtsi
        self.dtsi_training_output = dtsi_training_output
        self.inference_threshold = inference_threshold
        self.target_active_ratio = target_active_ratio
        self.regularization_multiplier = regularization_multiplier
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
        self.dense_routers = nn.ModuleList(
            nn.Linear(token_dim, num_experts) for _ in range(num_tokens)
        )
        self.sparse_routers = nn.ModuleList(
            nn.Linear(token_dim, num_experts) for _ in range(num_tokens)
        )
        self.register_buffer(
            "regularization_coefficient",
            torch.tensor(float(regularization_initial)),
        )
        self._last_regularization_loss: Tensor | None = None
        self._last_active_ratio: Tensor | None = None
        self._coefficient_update_pending = False

    def _expert_outputs(self, token_index: int, values: Tensor) -> Tensor:
        return torch.stack(
            [expert(values) for expert in self.experts[token_index]],
            dim=1,
        )

    def _sparse_inference(self, token_index: int, values: Tensor, gates: Tensor) -> Tensor:
        output = values.new_zeros(values.size(0), self.token_dim)
        for expert_index, expert in enumerate(self.experts[token_index]):
            active = gates[:, expert_index] > self.inference_threshold
            if bool(active.any()):
                output[active] = output[active] + gates[active, expert_index].unsqueeze(1) * expert(
                    values[active]
                )
        return output

    def _stacked_expert_weights(self) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Autograd-connected stacked expert GEMM weights, grouped by (token, expert).

        Group index ``g = token_index * num_experts + expert_index``. ``torch.stack``
        keeps every stacked slice a live view into the original ``nn.Linear``
        parameters, so gradients flow back to ``self.experts`` exactly as in the
        per-expert path and checkpoints keep their historical key layout.
        """

        input_weight = torch.stack(
            [expert[0].weight for experts in self.experts for expert in experts],
            dim=0,
        )
        input_bias = torch.stack(
            [expert[0].bias for experts in self.experts for expert in experts],
            dim=0,
        )
        output_weight = torch.stack(
            [expert[3].weight for experts in self.experts for expert in experts],
            dim=0,
        )
        output_bias = torch.stack(
            [expert[3].bias for experts in self.experts for expert in experts],
            dim=0,
        )
        return input_weight, input_bias, output_weight, output_bias

    def _stacked_router_weights(
        self, routers: nn.ModuleList
    ) -> tuple[Tensor, Tensor]:
        weight = torch.stack([router.weight for router in routers], dim=0)
        bias = torch.stack([router.bias for router in routers], dim=0)
        return weight, bias

    def _all_expert_outputs(self, tokens: Tensor) -> Tensor:
        """Evaluate every (token, expert) FFN as two batched GEMMs.

        Returns ``[num_tokens, num_experts, batch, token_dim]`` matching the
        per-token ``_expert_outputs`` stack, but with two ``bmm`` launches
        instead of ``2 * num_tokens * num_experts`` small Linear kernels.
        """

        batch = tokens.size(0)
        token_count = self.num_tokens
        expert_count = self.num_experts
        groups = token_count * expert_count
        input_weight, input_bias, output_weight, output_bias = (
            self._stacked_expert_weights()
        )
        # [T,B,D] -> replicate each token across its experts -> [T*E, B, D].
        token_major = tokens.transpose(0, 1)
        grouped_input = (
            token_major.unsqueeze(1)
            .expand(token_count, expert_count, batch, self.token_dim)
            .reshape(groups, batch, self.token_dim)
        )
        hidden = torch.bmm(grouped_input, input_weight.transpose(1, 2))
        hidden = hidden + input_bias.unsqueeze(1).to(dtype=hidden.dtype)
        hidden = F.gelu(hidden) if self.activation == "gelu" else F.relu(hidden)
        hidden = F.dropout(hidden, p=self.dropout, training=self.training)
        output = torch.bmm(hidden, output_weight.transpose(1, 2))
        output = output + output_bias.unsqueeze(1).to(dtype=output.dtype)
        if self.output_relu:
            output = F.relu(output)
        return output.view(token_count, expert_count, batch, self.token_dim)

    def _forward_batched(self, tokens: Tensor) -> Tensor:
        token_major = tokens.transpose(0, 1)  # [T,B,D]
        sparse_weight, sparse_bias = self._stacked_router_weights(self.sparse_routers)
        sparse_logits = torch.bmm(
            token_major, sparse_weight.transpose(1, 2)
        ) + sparse_bias.unsqueeze(1)
        sparse_gates = torch.relu(sparse_logits)  # [T,B,E]

        # Per-token mean-then-mean reproduces stack([...per token...]).mean().
        gate_l1 = sparse_gates.sum(dim=-1).mean(dim=1).mean()
        active_ratio = (
            (sparse_gates > 0.0).to(tokens.dtype).mean(dim=(1, 2)).mean().detach()
        )

        expert_outputs = self._all_expert_outputs(tokens)  # [T,E,B,D]
        if self.training:
            sparse_gates_te = sparse_gates.permute(0, 2, 1).unsqueeze(-1)  # [T,E,B,1]
            sparse_output = (expert_outputs * sparse_gates_te).sum(dim=1)  # [T,B,D]
            if self.use_dtsi:
                dense_weight, dense_bias = self._stacked_router_weights(
                    self.dense_routers
                )
                dense_logits = torch.bmm(
                    token_major, dense_weight.transpose(1, 2)
                ) + dense_bias.unsqueeze(1)
                dense_gates = torch.softmax(dense_logits, dim=-1)
                dense_gates_te = dense_gates.permute(0, 2, 1).unsqueeze(-1)
                dense_output = (expert_outputs * dense_gates_te).sum(dim=1)
                if self.dtsi_training_output == "dense_router":
                    combined = dense_output
                elif self.dtsi_training_output == "mean":
                    combined = 0.5 * (dense_output + sparse_output)
                else:
                    raise RuntimeError("DTSI training output policy is not configured")
            else:
                combined = sparse_output
        else:
            # Experts below the inference threshold contribute exactly zero,
            # so masking the gates is equivalent to skipping their kernels.
            gate_mask = (sparse_gates > self.inference_threshold).to(sparse_gates.dtype)
            gated = (sparse_gates * gate_mask).permute(0, 2, 1).unsqueeze(-1)
            combined = (expert_outputs * gated).sum(dim=1)  # [T,B,D]

        self._last_regularization_loss = (
            self.regularization_coefficient.detach().clone() * gate_l1
        )
        self._last_active_ratio = active_ratio
        self._coefficient_update_pending = self.training
        return combined.transpose(0, 1)  # [B,T,D]

    def _forward_looped(self, tokens: Tensor) -> Tensor:
        outputs: list[Tensor] = []
        gate_l1_terms: list[Tensor] = []
        active_ratios: list[Tensor] = []
        for token_index in range(self.num_tokens):
            values = tokens[:, token_index, :]
            sparse_logits = self.sparse_routers[token_index](values)
            sparse_gates = torch.relu(sparse_logits)
            # RankMixer Eq. (11): L1 over non-negative inference-router gates.
            gate_l1_terms.append(sparse_gates.sum(dim=-1).mean())
            active_ratios.append(
                (sparse_gates > 0.0).to(tokens.dtype).mean()
            )
            if self.training:
                expert_outputs = self._expert_outputs(token_index, values)
                sparse_output = (expert_outputs * sparse_gates.unsqueeze(-1)).sum(dim=1)
                if self.use_dtsi:
                    dense_gates = torch.softmax(self.dense_routers[token_index](values), dim=-1)
                    dense_output = (expert_outputs * dense_gates.unsqueeze(-1)).sum(dim=1)
                    if self.dtsi_training_output == "dense_router":
                        outputs.append(dense_output)
                    elif self.dtsi_training_output == "mean":
                        outputs.append(0.5 * (dense_output + sparse_output))
                    else:
                        raise RuntimeError("DTSI training output policy is not configured")
                else:
                    outputs.append(sparse_output)
            else:
                outputs.append(self._sparse_inference(token_index, values, sparse_gates))
        gate_l1 = torch.stack(gate_l1_terms).mean()
        active_ratio = torch.stack(active_ratios).mean().detach()
        self._last_regularization_loss = (
            self.regularization_coefficient.detach().clone() * gate_l1
        )
        self._last_active_ratio = active_ratio
        self._coefficient_update_pending = self.training
        return torch.stack(outputs, dim=1)

    def forward(self, tokens: Tensor) -> Tensor:
        if tokens.ndim != 3 or tokens.size(1) != self.num_tokens or tokens.size(2) != self.token_dim:
            raise ValueError(
                f"expected tokens with shape [batch, {self.num_tokens}, {self.token_dim}], "
                f"got {tuple(tokens.shape)}"
            )
        # CUDA amortizes the per-expert launch overhead as batched GEMMs; the
        # small host BLAS GEMMs of the looped path parallelize better on CPU
        # and preserve the exact per-expert dropout RNG stream there.
        if tokens.device.type == "cuda":
            return self._forward_batched(tokens)
        return self._forward_looped(tokens)

    def step_regularization_controller(
        self,
        active_ratio: Tensor | None = None,
    ) -> None:
        if not self._coefficient_update_pending or self._last_active_ratio is None:
            return
        controller_ratio = self._last_active_ratio if active_ratio is None else active_ratio
        direction = torch.sign(
            controller_ratio
            - controller_ratio.new_tensor(self.target_active_ratio)
        )
        with torch.no_grad():
            self.regularization_coefficient.mul_(
                self.regularization_multiplier ** float(direction.item())
            )
        self._coefficient_update_pending = False

    def regularization_loss(self, reference: Tensor | None = None) -> Tensor:
        if self._last_regularization_loss is not None:
            return self._last_regularization_loss
        return torch.tensor(0.0) if reference is None else reference.new_zeros(())

    def active_ratio(self, reference: Tensor | None = None) -> Tensor:
        if self._last_active_ratio is not None:
            return self._last_active_ratio
        value = torch.tensor(0.0) if reference is None else reference.new_zeros(())
        return value.detach()
