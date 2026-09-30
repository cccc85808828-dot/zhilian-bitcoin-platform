from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F
try:
    from torch_scatter import scatter_mean
except ImportError:  # CPU/web deployment can use native PyTorch index_add.
    from .scatter import scatter_mean

from .baselines import (
    FeatureStandardizer,
    FeatureStatistics,
    RoleAwarePropagationLayer,
    StandardHypergraphPropagationLayer,
)
from .ellipticpp import EllipticPPHypergraphSnapshot


def _pool_address_memory(
    address_memory: torch.Tensor,
    relation_index: torch.Tensor,
    num_transactions: int,
) -> torch.Tensor:
    return scatter_mean(
        address_memory[relation_index[0]],
        relation_index[1],
        dim=0,
        dim_size=num_transactions,
    )


@dataclass
class TemporalTransactionComponents:
    """Transaction representation before the final role-history fusion."""

    transaction_states: torch.Tensor
    input_history_message: torch.Tensor
    output_history_message: torch.Tensor
    environment_gate_logits: Optional[torch.Tensor] = None
    attribute_prior_logits: Optional[torch.Tensor] = None
    address_risk_logits: Optional[torch.Tensor] = None
    fused_transaction_states: Optional[torch.Tensor] = None
    direct_attribute_logits: Optional[torch.Tensor] = None


class LowRankCrossLayer(nn.Module):
    """Memory-efficient explicit feature crossing for tabular attributes."""

    def __init__(self, feature_dim: int, rank: int, dropout: float) -> None:
        super().__init__()
        self.down_projection = nn.Linear(feature_dim, rank, bias=False)
        self.up_projection = nn.Linear(rank, feature_dim, bias=True)
        self.normalization = nn.LayerNorm(feature_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        anchor_features: torch.Tensor,
        current_features: torch.Tensor,
    ) -> torch.Tensor:
        interaction = self.up_projection(
            F.gelu(self.down_projection(current_features))
        )
        crossed_features = torch.tanh(anchor_features) * interaction
        return self.normalization(
            current_features + self.dropout(crossed_features)
        )


class CrossFeatureEncoder(nn.Module):
    """Fuse deep attributes with low-rank multiplicative feature interactions."""

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int,
        rank: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if rank < 1:
            raise ValueError("transaction_interaction_rank must be positive.")
        if num_layers < 1:
            raise ValueError("transaction_interaction_layers must be positive.")
        self.cross_layers = nn.ModuleList(
            [
                LowRankCrossLayer(feature_dim, rank, dropout)
                for _ in range(num_layers)
            ]
        )
        self.deep_projection = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim * 2),
            nn.GLU(dim=-1),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )
        self.cross_projection = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.fusion_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid(),
        )
        self.output_normalization = nn.LayerNorm(hidden_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        crossed_features = features
        for layer in self.cross_layers:
            crossed_features = layer(features, crossed_features)
        deep_state = self.deep_projection(features)
        cross_state = self.cross_projection(crossed_features)
        gate = self.fusion_gate(torch.cat([deep_state, cross_state], dim=-1))
        return self.output_normalization(
            deep_state + gate * cross_state
        )


class TemporalMemoryHGNN(nn.Module):
    """Role-aware HGNN with a recurrent state for every global address."""

    def __init__(
        self,
        address_statistics: FeatureStatistics,
        transaction_statistics: FeatureStatistics,
        hidden_dim: int,
        memory_dim: int,
        dropout: float,
        propagation_layers: int,
        feature_residual_enabled: bool = False,
        num_environment_experts: int = 1,
        global_expert_weight_floor: float = 0.0,
        attribute_prior_residual_enabled: bool = False,
        structural_residual_scale: float = 1.0,
        transaction_encoder_type: str = "linear",
        transaction_interaction_rank: int = 32,
        transaction_interaction_layers: int = 2,
        address_risk_auxiliary_enabled: bool = False,
        address_risk_initial_fusion_scale: float = 0.1,
        transaction_cross_initial_scale: float = 0.05,
        attribute_logit_residual_enabled: bool = False,
        attribute_logit_initial_scale: float = 0.1,
        attribute_logit_uncertainty_temperature: float = 0.0,
        temporal_memory_enabled: bool = True,
        role_aware_propagation: bool = True,
    ) -> None:
        super().__init__()
        if propagation_layers < 1:
            raise ValueError("At least one propagation layer is required.")
        self.memory_dim = int(memory_dim)
        self.temporal_memory_enabled = bool(temporal_memory_enabled)
        self.role_aware_propagation = bool(role_aware_propagation)
        self.num_environment_experts = int(num_environment_experts)
        if self.num_environment_experts < 1:
            raise ValueError("num_environment_experts must be at least one.")
        self.global_expert_weight_floor = float(global_expert_weight_floor)
        if not 0.0 <= self.global_expert_weight_floor < 1.0:
            raise ValueError("global_expert_weight_floor must be in [0, 1).")
        if (
            self.num_environment_experts == 1
            and self.global_expert_weight_floor > 0.0
        ):
            raise ValueError(
                "global_expert_weight_floor requires multiple experts."
            )
        self.attribute_prior_residual_enabled = bool(
            attribute_prior_residual_enabled
        )
        self.structural_residual_scale = float(structural_residual_scale)
        if self.structural_residual_scale <= 0.0:
            raise ValueError("structural_residual_scale must be positive.")
        self.address_standardizer = FeatureStandardizer(address_statistics)
        self.transaction_standardizer = FeatureStandardizer(transaction_statistics)
        self.address_encoder = nn.Sequential(
            nn.Linear(int(address_statistics.mean.numel()), hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        transaction_feature_dim = int(transaction_statistics.mean.numel())
        self.transaction_encoder_type = str(transaction_encoder_type).lower()
        if self.transaction_encoder_type in {"linear", "hybrid"}:
            self.transaction_encoder = nn.Sequential(
                nn.Linear(transaction_feature_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
            )
            if self.transaction_encoder_type == "hybrid":
                self.transaction_cross_encoder = CrossFeatureEncoder(
                    feature_dim=transaction_feature_dim,
                    hidden_dim=hidden_dim,
                    rank=int(transaction_interaction_rank),
                    num_layers=int(transaction_interaction_layers),
                    dropout=dropout,
                )
                self.transaction_cross_fusion_scale = nn.Parameter(
                    torch.tensor(float(transaction_cross_initial_scale))
                )
                self.transaction_cross_fusion_norm = nn.LayerNorm(hidden_dim)
        elif self.transaction_encoder_type == "cross":
            self.transaction_encoder = CrossFeatureEncoder(
                feature_dim=transaction_feature_dim,
                hidden_dim=hidden_dim,
                rank=int(transaction_interaction_rank),
                num_layers=int(transaction_interaction_layers),
                dropout=dropout,
            )
        else:
            raise ValueError(
                "transaction_encoder_type must be 'linear', 'hybrid', or 'cross'."
            )
        self.feature_residual_enabled = bool(feature_residual_enabled)
        if self.feature_residual_enabled:
            self.transaction_feature_residual = nn.Sequential(
                nn.Linear(transaction_feature_dim, hidden_dim * 2),
                nn.LayerNorm(hidden_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
            )
            self.transaction_feature_gate = nn.Sequential(
                nn.Linear(hidden_dim * 3, hidden_dim),
                nn.Sigmoid(),
            )
            self.transaction_feature_fusion_norm = nn.LayerNorm(hidden_dim)
        propagation_layer = (
            RoleAwarePropagationLayer
            if self.role_aware_propagation
            else StandardHypergraphPropagationLayer
        )
        self.propagation_layers = nn.ModuleList(
            [propagation_layer(hidden_dim, dropout) for _ in range(propagation_layers)]
        )
        self.memory_cell = nn.GRUCell(hidden_dim, memory_dim)
        self.memory_projection = nn.Linear(memory_dim, hidden_dim)
        self.address_risk_auxiliary_enabled = bool(
            address_risk_auxiliary_enabled
        )
        if self.address_risk_auxiliary_enabled:
            self.address_risk_projection = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim),
            )
            self.address_risk_classifier = nn.Linear(hidden_dim, 1)
            self.input_address_risk_to_transaction = nn.Linear(
                hidden_dim, hidden_dim, bias=False
            )
            self.output_address_risk_to_transaction = nn.Linear(
                hidden_dim, hidden_dim, bias=False
            )
            self.address_risk_fusion_scale = nn.Parameter(
                torch.tensor(float(address_risk_initial_fusion_scale))
            )
        if self.role_aware_propagation:
            self.input_memory_to_transaction = nn.Linear(
                hidden_dim, hidden_dim, bias=False
            )
            self.output_memory_to_transaction = nn.Linear(
                hidden_dim, hidden_dim, bias=False
            )
        else:
            self.role_agnostic_memory_to_transaction = nn.Linear(
                hidden_dim, hidden_dim, bias=False
            )
        self.transaction_memory_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        if self.num_environment_experts == 1:
            self.classifier = self._build_classifier(hidden_dim, dropout)
        else:
            self.environment_experts = nn.ModuleList(
                [
                    self._build_classifier(hidden_dim, dropout)
                    for _ in range(self.num_environment_experts)
                ]
            )
            self.environment_gate = nn.Sequential(
                nn.Linear(hidden_dim * 4, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, self.num_environment_experts),
            )
        if self.attribute_prior_residual_enabled:
            classifiers = (
                [self.classifier]
                if self.num_environment_experts == 1
                else list(self.environment_experts)
            )
            for classifier in classifiers:
                output_layer = classifier[-1]
                nn.init.zeros_(output_layer.weight)
                nn.init.zeros_(output_layer.bias)
        self.attribute_logit_residual_enabled = bool(
            attribute_logit_residual_enabled
        )
        self.attribute_logit_uncertainty_temperature = float(
            attribute_logit_uncertainty_temperature
        )
        if self.attribute_logit_uncertainty_temperature < 0.0:
            raise ValueError(
                "attribute_logit_uncertainty_temperature cannot be negative."
            )
        if self.attribute_logit_residual_enabled:
            self.direct_attribute_classifier = nn.Sequential(
                nn.Linear(transaction_feature_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )
            self.attribute_logit_fusion_scale = nn.Parameter(
                torch.tensor(float(attribute_logit_initial_scale))
            )

    @staticmethod
    def _build_classifier(hidden_dim: int, dropout: float) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def initial_memory(
        self,
        num_addresses: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        return torch.zeros(
            num_addresses,
            self.memory_dim,
            device=device,
            dtype=dtype,
        )

    def encode_current_snapshot(
        self, snapshot: EllipticPPHypergraphSnapshot
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        address_features = self.address_standardizer(snapshot.address_features)
        transaction_features = self.transaction_standardizer(
            snapshot.transaction_features,
            snapshot.transaction_feature_mask,
        )
        address_states = self.address_encoder(address_features)
        transaction_states = self.transaction_encoder(transaction_features)
        if self.transaction_encoder_type == "hybrid":
            cross_state = self.transaction_cross_encoder(transaction_features)
            cross_scale = torch.tanh(self.transaction_cross_fusion_scale)
            transaction_states = self.transaction_cross_fusion_norm(
                transaction_states + cross_scale * cross_state
            )
        raw_transaction_states = (
            self.transaction_feature_residual(transaction_features)
            if self.feature_residual_enabled
            else None
        )
        for layer in self.propagation_layers:
            address_states, transaction_states = layer(
                address_states,
                transaction_states,
                snapshot,
            )
        if raw_transaction_states is not None:
            feature_gate = self.transaction_feature_gate(
                torch.cat(
                    [
                        transaction_states,
                        raw_transaction_states,
                        torch.abs(transaction_states - raw_transaction_states),
                    ],
                    dim=-1,
                )
            )
            transaction_states = self.transaction_feature_fusion_norm(
                transaction_states + feature_gate * raw_transaction_states
            )
        return address_states, transaction_states

    def fuse_components(
        self,
        components: TemporalTransactionComponents,
    ) -> torch.Tensor:
        memory_message = (
            components.input_history_message
            + components.output_history_message
        )
        transaction_states = self.transaction_memory_norm(
            components.transaction_states + self.dropout(F.relu(memory_message))
        )
        return transaction_states

    def expert_logits_from_components(
        self,
        components: TemporalTransactionComponents,
    ) -> torch.Tensor:
        transaction_states = (
            components.fused_transaction_states
            if components.fused_transaction_states is not None
            else self.fuse_components(components)
        )
        if self.num_environment_experts == 1:
            return self.classifier(transaction_states)
        return torch.cat(
            [expert(transaction_states) for expert in self.environment_experts],
            dim=-1,
        )

    def classify_components(
        self,
        components: TemporalTransactionComponents,
    ) -> torch.Tensor:
        expert_logits = self.expert_logits_from_components(components)
        if self.num_environment_experts == 1:
            structural_logits = expert_logits.squeeze(-1)
        else:
            if components.environment_gate_logits is None:
                raise ValueError(
                    "Multi-expert classification requires environment gate logits."
                )
            gate_weights = torch.softmax(
                components.environment_gate_logits,
                dim=-1,
            )
            if self.global_expert_weight_floor > 0.0:
                gate_weights = gate_weights * (
                    1.0 - self.global_expert_weight_floor
                )
                gate_weights = gate_weights.clone()
                gate_weights[0] = (
                    gate_weights[0] + self.global_expert_weight_floor
                )
            structural_logits = (
                expert_logits * gate_weights.unsqueeze(0)
            ).sum(dim=-1)
        combined_logits = self.structural_residual_scale * structural_logits
        if self.attribute_logit_residual_enabled:
            if components.direct_attribute_logits is None:
                raise ValueError(
                    "Attribute-logit residual classification requires logits."
                )
            if components.direct_attribute_logits.shape != combined_logits.shape:
                raise ValueError("direct_attribute_logits has an invalid shape.")
            attribute_scale = torch.tanh(self.attribute_logit_fusion_scale)
            if self.attribute_logit_uncertainty_temperature > 0.0:
                uncertainty_gate = torch.exp(
                    -combined_logits.detach().abs()
                    / self.attribute_logit_uncertainty_temperature
                )
            else:
                uncertainty_gate = 1.0
            combined_logits = (
                combined_logits
                + attribute_scale
                * uncertainty_gate
                * components.direct_attribute_logits
            )
        if self.attribute_prior_residual_enabled:
            if components.attribute_prior_logits is None:
                raise ValueError(
                    "Attribute-prior residual classification requires prior logits."
                )
            if components.attribute_prior_logits.shape != combined_logits.shape:
                raise ValueError("attribute_prior_logits has an invalid shape.")
            combined_logits = components.attribute_prior_logits + combined_logits
        return combined_logits

    def environment_gate_logits(
        self,
        transaction_states: torch.Tensor,
        projected_memory: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        if self.num_environment_experts == 1:
            return None
        descriptor = torch.cat(
            [
                transaction_states.mean(dim=0),
                transaction_states.std(dim=0, unbiased=False),
                projected_memory.mean(dim=0),
                projected_memory.std(dim=0, unbiased=False),
            ],
            dim=-1,
        )
        return self.environment_gate(descriptor)

    def forward_with_components(
        self,
        snapshot: EllipticPPHypergraphSnapshot,
        previous_address_memory: torch.Tensor,
        attribute_prior_logits: Optional[torch.Tensor] = None,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        TemporalTransactionComponents,
    ]:
        if previous_address_memory.shape != (
            snapshot.num_addresses,
            self.memory_dim,
        ):
            raise ValueError("previous_address_memory has an invalid shape.")

        address_states, transaction_states = self.encode_current_snapshot(snapshot)
        direct_attribute_logits = None
        if self.attribute_logit_residual_enabled:
            standardized_transaction_features = self.transaction_standardizer(
                snapshot.transaction_features,
                snapshot.transaction_feature_mask,
            )
            direct_attribute_logits = self.direct_attribute_classifier(
                standardized_transaction_features
            ).squeeze(-1)
        recurrent_memory = (
            previous_address_memory
            if self.temporal_memory_enabled
            else torch.zeros_like(previous_address_memory)
        )
        updated_memory = self.memory_cell(address_states, recurrent_memory)
        projected_memory = self.memory_projection(updated_memory)
        if self.role_aware_propagation:
            input_history = _pool_address_memory(
                projected_memory,
                snapshot.input_index,
                snapshot.num_transactions,
            )
            output_history = _pool_address_memory(
                projected_memory,
                snapshot.output_index,
                snapshot.num_transactions,
            )
        else:
            relation_index = torch.cat(
                [snapshot.input_index, snapshot.output_index], dim=1
            )
            role_agnostic_history = _pool_address_memory(
                projected_memory,
                relation_index,
                snapshot.num_transactions,
            )
        address_risk_logits: Optional[torch.Tensor] = None
        input_risk_message: Optional[torch.Tensor] = None
        output_risk_message: Optional[torch.Tensor] = None
        if self.address_risk_auxiliary_enabled:
            address_risk_states = self.address_risk_projection(projected_memory)
            address_risk_logits = self.address_risk_classifier(
                address_risk_states
            ).squeeze(-1)
            input_risk_message = _pool_address_memory(
                address_risk_states,
                snapshot.input_index,
                snapshot.num_transactions,
            )
            output_risk_message = _pool_address_memory(
                address_risk_states,
                snapshot.output_index,
                snapshot.num_transactions,
            )
        if self.role_aware_propagation:
            input_history_message = self.input_memory_to_transaction(input_history)
            output_history_message = self.output_memory_to_transaction(output_history)
        else:
            shared_history_message = self.role_agnostic_memory_to_transaction(
                role_agnostic_history
            )
            # Keep the two component fields for RCHA while removing role identity.
            input_history_message = 0.5 * shared_history_message
            output_history_message = 0.5 * shared_history_message
        if input_risk_message is not None and output_risk_message is not None:
            risk_scale = torch.tanh(self.address_risk_fusion_scale)
            input_history_message = input_history_message + risk_scale * (
                self.input_address_risk_to_transaction(input_risk_message)
            )
            output_history_message = output_history_message + risk_scale * (
                self.output_address_risk_to_transaction(output_risk_message)
            )
        components = TemporalTransactionComponents(
            transaction_states=transaction_states,
            input_history_message=input_history_message,
            output_history_message=output_history_message,
            environment_gate_logits=self.environment_gate_logits(
                transaction_states,
                projected_memory,
            ),
            attribute_prior_logits=attribute_prior_logits,
            address_risk_logits=address_risk_logits,
            direct_attribute_logits=direct_attribute_logits,
        )
        components.fused_transaction_states = self.fuse_components(components)
        logits = self.classify_components(components)
        return logits, updated_memory, components

    def forward(
        self,
        snapshot: EllipticPPHypergraphSnapshot,
        previous_address_memory: torch.Tensor,
        attribute_prior_logits: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        logits, updated_memory, _ = self.forward_with_components(
            snapshot,
            previous_address_memory,
            attribute_prior_logits,
        )
        return logits, updated_memory
