from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch import nn
from torch.nn import functional as F
try:
    from torch_geometric.nn import (
        GATConv,
        GCNConv,
        HeteroConv,
        HGTConv,
        SAGEConv,
        TransformerConv,
    )
    from torch_geometric.utils import add_self_loops
except ImportError:  # The live TTHGNN path does not instantiate PyG baselines.
    GATConv = GCNConv = HeteroConv = HGTConv = SAGEConv = TransformerConv = None

    def add_self_loops(
        edge_index: torch.Tensor, num_nodes: int
    ) -> tuple[torch.Tensor, None]:
        nodes = torch.arange(num_nodes, dtype=torch.long, device=edge_index.device)
        loops = torch.stack([nodes, nodes])
        return torch.cat([edge_index, loops], dim=1), None
try:
    from torch_scatter import scatter_add, scatter_mean
except ImportError:  # CPU/web deployment can use native PyTorch index_add.
    from .scatter import scatter_add, scatter_mean

from .ellipticpp import EllipticPPHypergraphSnapshot


@dataclass
class FeatureStatistics:
    mean: torch.Tensor
    std: torch.Tensor
    count: torch.Tensor

    def as_dict(self) -> dict:
        return {"mean": self.mean, "std": self.std, "count": self.count}


def fit_feature_statistics(
    tensors: Iterable[torch.Tensor],
    masks: Optional[Iterable[torch.Tensor]] = None,
) -> FeatureStatistics:
    """Fit per-feature statistics without using validation or test snapshots."""

    tensor_list = list(tensors)
    if not tensor_list:
        raise ValueError("At least one feature tensor is required.")
    feature_dim = tensor_list[0].size(1)
    sums = torch.zeros(feature_dim, dtype=torch.float64)
    squared_sums = torch.zeros(feature_dim, dtype=torch.float64)
    counts = torch.zeros(feature_dim, dtype=torch.float64)
    mask_list = list(masks) if masks is not None else None
    if mask_list is not None and len(mask_list) != len(tensor_list):
        raise ValueError("Feature tensors and masks must have equal lengths.")

    for index, values in enumerate(tensor_list):
        values64 = values.to(dtype=torch.float64)
        if values64.ndim != 2 or values64.size(1) != feature_dim:
            raise ValueError("Feature tensors must have a consistent two-dimensional shape.")
        if mask_list is None:
            valid = torch.ones_like(values64, dtype=torch.bool)
        else:
            valid = mask_list[index].to(dtype=torch.bool)
            if valid.shape != values64.shape:
                raise ValueError("A feature mask has an invalid shape.")
        safe_values = torch.where(valid, values64, torch.zeros_like(values64))
        sums += safe_values.sum(dim=0)
        squared_sums += safe_values.square().sum(dim=0)
        counts += valid.sum(dim=0)

    safe_counts = counts.clamp_min(1.0)
    mean = sums / safe_counts
    variance = squared_sums / safe_counts - mean.square()
    std = variance.clamp_min(0.0).sqrt()
    std = torch.where(std < 1e-6, torch.ones_like(std), std)
    return FeatureStatistics(
        mean=mean.to(dtype=torch.float32),
        std=std.to(dtype=torch.float32),
        count=counts.to(dtype=torch.int64),
    )


class FeatureStandardizer(nn.Module):
    def __init__(self, statistics: FeatureStatistics) -> None:
        super().__init__()
        self.register_buffer("mean", statistics.mean.clone())
        self.register_buffer("std", statistics.std.clone())

    def forward(
        self, values: torch.Tensor, valid_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        standardized = (values - self.mean) / self.std
        if valid_mask is not None:
            standardized = torch.where(
                valid_mask,
                standardized,
                torch.zeros_like(standardized),
            )
        return standardized


class TransactionMLP(nn.Module):
    """Transaction-feature-only sanity baseline."""

    def __init__(
        self,
        transaction_statistics: FeatureStatistics,
        hidden_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        feature_dim = int(transaction_statistics.mean.numel())
        self.transaction_standardizer = FeatureStandardizer(transaction_statistics)
        self.network = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, snapshot: EllipticPPHypergraphSnapshot) -> torch.Tensor:
        features = self.transaction_standardizer(
            snapshot.transaction_features,
            snapshot.transaction_feature_mask,
        )
        return self.network(features).squeeze(-1)


def _address_to_transaction_mean(
    address_states: torch.Tensor,
    relation_index: torch.Tensor,
    num_transactions: int,
) -> torch.Tensor:
    return scatter_mean(
        address_states[relation_index[0]],
        relation_index[1],
        dim=0,
        dim_size=num_transactions,
    )


def _transaction_to_address_mean(
    transaction_states: torch.Tensor,
    relation_index: torch.Tensor,
    num_addresses: int,
) -> torch.Tensor:
    return scatter_mean(
        transaction_states[relation_index[1]],
        relation_index[0],
        dim=0,
        dim_size=num_addresses,
    )


class RoleAwarePropagationLayer(nn.Module):
    """One static address-hyperedge-address propagation layer."""

    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.input_address_to_transaction = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.output_address_to_transaction = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.input_transaction_to_address = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.output_transaction_to_address = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.transaction_norm = nn.LayerNorm(hidden_dim)
        self.address_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        address_states: torch.Tensor,
        transaction_states: torch.Tensor,
        snapshot: EllipticPPHypergraphSnapshot,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_address_pool = _address_to_transaction_mean(
            address_states, snapshot.input_index, snapshot.num_transactions
        )
        output_address_pool = _address_to_transaction_mean(
            address_states, snapshot.output_index, snapshot.num_transactions
        )
        transaction_message = (
            self.input_address_to_transaction(input_address_pool)
            + self.output_address_to_transaction(output_address_pool)
        )
        transaction_states = self.transaction_norm(
            transaction_states + self.dropout(F.relu(transaction_message))
        )

        input_transaction_pool = _transaction_to_address_mean(
            transaction_states, snapshot.input_index, snapshot.num_addresses
        )
        output_transaction_pool = _transaction_to_address_mean(
            transaction_states, snapshot.output_index, snapshot.num_addresses
        )
        address_message = (
            self.input_transaction_to_address(input_transaction_pool)
            + self.output_transaction_to_address(output_transaction_pool)
        )
        address_states = self.address_norm(
            address_states + self.dropout(F.relu(address_message))
        )
        return address_states, transaction_states


class StaticRoleAwareHGNN(nn.Module):
    """Static role-aware HGNN without temporal memory or robust loss."""

    def __init__(
        self,
        address_statistics: FeatureStatistics,
        transaction_statistics: FeatureStatistics,
        hidden_dim: int,
        dropout: float,
        propagation_layers: int,
    ) -> None:
        super().__init__()
        if propagation_layers < 1:
            raise ValueError("At least one propagation layer is required.")
        self.address_standardizer = FeatureStandardizer(address_statistics)
        self.transaction_standardizer = FeatureStandardizer(transaction_statistics)
        self.address_encoder = nn.Sequential(
            nn.Linear(int(address_statistics.mean.numel()), hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        self.transaction_encoder = nn.Sequential(
            nn.Linear(int(transaction_statistics.mean.numel()), hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        self.layers = nn.ModuleList(
            [
                RoleAwarePropagationLayer(hidden_dim, dropout)
                for _ in range(propagation_layers)
            ]
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, snapshot: EllipticPPHypergraphSnapshot) -> torch.Tensor:
        address_features = self.address_standardizer(snapshot.address_features)
        transaction_features = self.transaction_standardizer(
            snapshot.transaction_features,
            snapshot.transaction_feature_mask,
        )
        address_states = self.address_encoder(address_features)
        transaction_states = self.transaction_encoder(transaction_features)
        for layer in self.layers:
            address_states, transaction_states = layer(
                address_states, transaction_states, snapshot
            )
        return self.classifier(transaction_states).squeeze(-1)


class StandardHypergraphPropagationLayer(nn.Module):
    """Role-agnostic address-hyperedge-address propagation."""

    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.address_to_transaction = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.transaction_to_address = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.transaction_norm = nn.LayerNorm(hidden_dim)
        self.address_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        address_states: torch.Tensor,
        transaction_states: torch.Tensor,
        snapshot: EllipticPPHypergraphSnapshot,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        relation_index = torch.cat(
            [snapshot.input_index, snapshot.output_index], dim=1
        )
        address_pool = _address_to_transaction_mean(
            address_states, relation_index, snapshot.num_transactions
        )
        transaction_states = self.transaction_norm(
            transaction_states
            + self.dropout(F.relu(self.address_to_transaction(address_pool)))
        )
        transaction_pool = _transaction_to_address_mean(
            transaction_states, relation_index, snapshot.num_addresses
        )
        address_states = self.address_norm(
            address_states
            + self.dropout(F.relu(self.transaction_to_address(transaction_pool)))
        )
        return address_states, transaction_states


class StaticHGNN(nn.Module):
    """Standard static HGNN with a single role-agnostic incidence channel."""

    def __init__(
        self,
        address_statistics: FeatureStatistics,
        transaction_statistics: FeatureStatistics,
        hidden_dim: int,
        dropout: float,
        propagation_layers: int,
    ) -> None:
        super().__init__()
        if propagation_layers < 1:
            raise ValueError("At least one propagation layer is required.")
        self.address_standardizer = FeatureStandardizer(address_statistics)
        self.transaction_standardizer = FeatureStandardizer(transaction_statistics)
        self.address_encoder = nn.Sequential(
            nn.Linear(int(address_statistics.mean.numel()), hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        self.transaction_encoder = nn.Sequential(
            nn.Linear(int(transaction_statistics.mean.numel()), hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        self.layers = nn.ModuleList(
            [
                StandardHypergraphPropagationLayer(hidden_dim, dropout)
                for _ in range(propagation_layers)
            ]
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, snapshot: EllipticPPHypergraphSnapshot) -> torch.Tensor:
        address_states = self.address_encoder(
            self.address_standardizer(snapshot.address_features)
        )
        transaction_states = self.transaction_encoder(
            self.transaction_standardizer(
                snapshot.transaction_features,
                snapshot.transaction_feature_mask,
            )
        )
        for layer in self.layers:
            address_states, transaction_states = layer(
                address_states, transaction_states, snapshot
            )
        return self.classifier(transaction_states).squeeze(-1)


class TransactionGraphBaseline(nn.Module):
    """Shared implementation for GCN, GAT and graph Transformer baselines."""

    SUPPORTED_KINDS = {"gcn", "gat", "graph_transformer"}

    def __init__(
        self,
        transaction_statistics: FeatureStatistics,
        hidden_dim: int,
        dropout: float,
        propagation_layers: int,
        kind: str,
        attention_heads: int = 4,
    ) -> None:
        super().__init__()
        if kind not in self.SUPPORTED_KINDS:
            raise ValueError(f"Unsupported transaction graph baseline: {kind}")
        if propagation_layers < 1:
            raise ValueError("At least one propagation layer is required.")
        if hidden_dim % attention_heads != 0:
            raise ValueError("hidden_dim must be divisible by attention_heads.")
        self.kind = kind
        self.transaction_standardizer = FeatureStandardizer(transaction_statistics)
        self.encoder = nn.Sequential(
            nn.Linear(int(transaction_statistics.mean.numel()), hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        layers: List[nn.Module] = []
        for _ in range(propagation_layers):
            if kind == "gcn":
                layers.append(GCNConv(hidden_dim, hidden_dim))
            elif kind == "gat":
                layers.append(
                    GATConv(
                        hidden_dim,
                        hidden_dim,
                        heads=attention_heads,
                        concat=False,
                        dropout=dropout,
                    )
                )
            else:
                layers.append(
                    TransformerConv(
                        hidden_dim,
                        hidden_dim // attention_heads,
                        heads=attention_heads,
                        concat=True,
                        beta=True,
                        dropout=dropout,
                    )
                )
        self.layers = nn.ModuleList(layers)
        self.norms = nn.ModuleList(
            [nn.LayerNorm(hidden_dim) for _ in range(propagation_layers)]
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        snapshot: EllipticPPHypergraphSnapshot,
        transaction_edge_index: torch.Tensor,
    ) -> torch.Tensor:
        states = self.encoder(
            self.transaction_standardizer(
                snapshot.transaction_features,
                snapshot.transaction_feature_mask,
            )
        )
        for layer, norm in zip(self.layers, self.norms):
            message = layer(states, transaction_edge_index)
            states = norm(states + self.dropout(F.relu(message)))
        return self.classifier(states).squeeze(-1)


class GATResNetBaseline(nn.Module):
    """Three-stage GAT with the residual paths used by GAT-ResNet.

    ``propagation_layers`` counts the hidden GAT stages.  With the unified
    baseline setting of two propagation layers, the model has two hidden GAT
    layers plus the graph-attention output layer described in the paper.
    """

    def __init__(
        self,
        transaction_statistics: FeatureStatistics,
        hidden_dim: int,
        dropout: float,
        propagation_layers: int,
        attention_heads: int = 4,
    ) -> None:
        super().__init__()
        if propagation_layers < 2:
            raise ValueError("GAT-ResNet requires at least two hidden GAT layers.")
        input_dim = int(transaction_statistics.mean.numel())
        self.transaction_standardizer = FeatureStandardizer(transaction_statistics)
        self.input_layer = GATConv(
            input_dim,
            hidden_dim,
            heads=attention_heads,
            concat=False,
            dropout=dropout,
        )
        self.residual_layers = nn.ModuleList(
            [
                GATConv(
                    hidden_dim,
                    hidden_dim,
                    heads=attention_heads,
                    concat=False,
                    dropout=dropout,
                )
                for _ in range(propagation_layers - 1)
            ]
        )
        self.output_layer = GATConv(
            hidden_dim,
            1,
            heads=1,
            concat=False,
            dropout=dropout,
        )
        self.input_skip = nn.Linear(input_dim, 1, bias=False)
        nn.init.xavier_normal_(self.input_skip.weight)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        snapshot: EllipticPPHypergraphSnapshot,
        transaction_edge_index: torch.Tensor,
    ) -> torch.Tensor:
        features = self.transaction_standardizer(
            snapshot.transaction_features,
            snapshot.transaction_feature_mask,
        )
        states = self.dropout(F.elu(self.input_layer(features, transaction_edge_index)))
        for layer in self.residual_layers:
            residual = F.elu(layer(states, transaction_edge_index))
            states = self.dropout(states + residual)
        graph_logits = self.output_layer(states, transaction_edge_index).squeeze(-1)
        feature_logits = self.input_skip(features).squeeze(-1)
        return graph_logits + feature_logits


class HGTBaseline(nn.Module):
    """Heterogeneous graph Transformer over transaction and address nodes."""

    NODE_TYPES = ["address", "transaction"]
    EDGE_TYPES = [
        ("address", "input_to", "transaction"),
        ("transaction", "input_used_by", "address"),
        ("transaction", "output_to", "address"),
        ("address", "output_from", "transaction"),
        ("transaction", "flow", "transaction"),
    ]

    def __init__(
        self,
        address_statistics: FeatureStatistics,
        transaction_statistics: FeatureStatistics,
        hidden_dim: int,
        dropout: float,
        propagation_layers: int,
        attention_heads: int = 4,
    ) -> None:
        super().__init__()
        if hidden_dim % attention_heads != 0:
            raise ValueError("hidden_dim must be divisible by attention_heads.")
        metadata = (self.NODE_TYPES, self.EDGE_TYPES)
        self.address_standardizer = FeatureStandardizer(address_statistics)
        self.transaction_standardizer = FeatureStandardizer(transaction_statistics)
        self.encoders = nn.ModuleDict(
            {
                "address": nn.Linear(
                    int(address_statistics.mean.numel()), hidden_dim
                ),
                "transaction": nn.Linear(
                    int(transaction_statistics.mean.numel()), hidden_dim
                ),
            }
        )
        self.layers = nn.ModuleList(
            [
                HGTConv(
                    {node_type: hidden_dim for node_type in self.NODE_TYPES},
                    hidden_dim,
                    metadata,
                    heads=attention_heads,
                )
                for _ in range(propagation_layers)
            ]
        )
        self.norms = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        node_type: nn.LayerNorm(hidden_dim)
                        for node_type in self.NODE_TYPES
                    }
                )
                for _ in range(propagation_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    @staticmethod
    def _edge_index_dict(
        snapshot: EllipticPPHypergraphSnapshot,
        transaction_edge_index: torch.Tensor,
    ) -> Dict[Tuple[str, str, str], torch.Tensor]:
        input_reverse = torch.stack(
            [snapshot.input_index[1], snapshot.input_index[0]], dim=0
        )
        output_forward = torch.stack(
            [snapshot.output_index[1], snapshot.output_index[0]], dim=0
        )
        return {
            ("address", "input_to", "transaction"): snapshot.input_index,
            ("transaction", "input_used_by", "address"): input_reverse,
            ("transaction", "output_to", "address"): output_forward,
            ("address", "output_from", "transaction"): snapshot.output_index,
            ("transaction", "flow", "transaction"): transaction_edge_index,
        }

    def forward(
        self,
        snapshot: EllipticPPHypergraphSnapshot,
        transaction_edge_index: torch.Tensor,
    ) -> torch.Tensor:
        x_dict = {
            "address": F.relu(
                self.encoders["address"](
                    self.address_standardizer(snapshot.address_features)
                )
            ),
            "transaction": F.relu(
                self.encoders["transaction"](
                    self.transaction_standardizer(
                        snapshot.transaction_features,
                        snapshot.transaction_feature_mask,
                    )
                )
            ),
        }
        edge_index_dict = self._edge_index_dict(snapshot, transaction_edge_index)
        for layer, norms in zip(self.layers, self.norms):
            messages = layer(x_dict, edge_index_dict)
            x_dict = {
                node_type: norms[node_type](
                    x_dict[node_type]
                    + self.dropout(F.relu(messages[node_type]))
                )
                for node_type in self.NODE_TYPES
            }
        return self.classifier(x_dict["transaction"]).squeeze(-1)


class HeteroSAGEBaseline(nn.Module):
    """Relation-specific GraphSAGE over time-safe Elliptic++ snapshots.

    The original HeteroSAGE study uses one SAGEConv per heterogeneous edge
    type.  Here the same construction is applied to the five causal relations
    already used by the unified HGT baseline.  The all-period address graph is
    intentionally excluded because it has no edge timestamps and would expose
    future relations to earlier snapshots.
    """

    NODE_TYPES = HGTBaseline.NODE_TYPES
    EDGE_TYPES = HGTBaseline.EDGE_TYPES

    def __init__(
        self,
        address_statistics: FeatureStatistics,
        transaction_statistics: FeatureStatistics,
        hidden_dim: int,
        dropout: float,
        propagation_layers: int,
    ) -> None:
        super().__init__()
        if propagation_layers < 1:
            raise ValueError("At least one propagation layer is required.")
        self.address_standardizer = FeatureStandardizer(address_statistics)
        self.transaction_standardizer = FeatureStandardizer(transaction_statistics)
        input_dims = {
            "address": int(address_statistics.mean.numel()),
            "transaction": int(transaction_statistics.mean.numel()),
        }
        layers: List[nn.Module] = []
        for layer_index in range(propagation_layers):
            layer_dims = input_dims if layer_index == 0 else {
                node_type: hidden_dim for node_type in self.NODE_TYPES
            }
            layers.append(
                HeteroConv(
                    {
                        edge_type: SAGEConv(
                            (
                                layer_dims[edge_type[0]],
                                layer_dims[edge_type[2]],
                            ),
                            hidden_dim,
                        )
                        for edge_type in self.EDGE_TYPES
                    },
                    aggr="sum",
                )
            )
        self.layers = nn.ModuleList(layers)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        snapshot: EllipticPPHypergraphSnapshot,
        transaction_edge_index: torch.Tensor,
    ) -> torch.Tensor:
        x_dict = {
            "address": self.address_standardizer(snapshot.address_features),
            "transaction": self.transaction_standardizer(
                snapshot.transaction_features,
                snapshot.transaction_feature_mask,
            ),
        }
        edge_index_dict = HGTBaseline._edge_index_dict(
            snapshot, transaction_edge_index
        )
        for layer in self.layers:
            x_dict = layer(x_dict, edge_index_dict)
            x_dict = {
                node_type: self.dropout(F.relu(states))
                for node_type, states in x_dict.items()
            }
        return self.classifier(x_dict["transaction"]).squeeze(-1)


def _normalized_gcn_propagation(
    node_states: torch.Tensor,
    edge_index: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    edge_index, _ = add_self_loops(edge_index, num_nodes=node_states.size(0))
    source, target = edge_index
    degrees = scatter_add(
        torch.ones_like(target, dtype=node_states.dtype),
        target,
        dim=0,
        dim_size=node_states.size(0),
    ).clamp_min(1.0)
    inverse_sqrt_degree = degrees.pow(-0.5)
    normalization = inverse_sqrt_degree[source] * inverse_sqrt_degree[target]
    support = node_states @ weight
    return scatter_add(
        support[source] * normalization.unsqueeze(-1),
        target,
        dim=0,
        dim_size=node_states.size(0),
    )


class EvolveGCNOBaseline(nn.Module):
    """Compact EvolveGCN-O implementation with recurrent GCN weights."""

    def __init__(
        self,
        transaction_statistics: FeatureStatistics,
        hidden_dim: int,
        dropout: float,
        propagation_layers: int,
    ) -> None:
        super().__init__()
        if propagation_layers < 1:
            raise ValueError("At least one propagation layer is required.")
        self.transaction_standardizer = FeatureStandardizer(transaction_statistics)
        self.encoder = nn.Sequential(
            nn.Linear(int(transaction_statistics.mean.numel()), hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        self.initial_gcn_weights = nn.ParameterList(
            [nn.Parameter(torch.empty(hidden_dim, hidden_dim)) for _ in range(propagation_layers)]
        )
        self.weight_updaters = nn.ModuleList(
            [nn.GRUCell(hidden_dim, hidden_dim) for _ in range(propagation_layers)]
        )
        self.norms = nn.ModuleList(
            [nn.LayerNorm(hidden_dim) for _ in range(propagation_layers)]
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for weight in self.initial_gcn_weights:
            nn.init.xavier_uniform_(weight)

    def initial_state(self) -> List[torch.Tensor]:
        return [weight for weight in self.initial_gcn_weights]

    def forward_snapshot(
        self,
        snapshot: EllipticPPHypergraphSnapshot,
        transaction_edge_index: torch.Tensor,
        previous_weights: Optional[Sequence[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        weights = self.initial_state() if previous_weights is None else list(previous_weights)
        states = self.encoder(
            self.transaction_standardizer(
                snapshot.transaction_features,
                snapshot.transaction_feature_mask,
            )
        )
        next_weights: List[torch.Tensor] = []
        for previous_weight, updater, norm in zip(
            weights, self.weight_updaters, self.norms
        ):
            evolved_weight = updater(previous_weight, previous_weight)
            message = _normalized_gcn_propagation(
                states, transaction_edge_index, evolved_weight
            )
            states = norm(states + self.dropout(F.relu(message)))
            next_weights.append(evolved_weight)
        return self.classifier(states).squeeze(-1), next_weights


@dataclass
class BFHGNSequenceOutput:
    """Transaction outputs for one contiguous BF-HGN temporal window."""

    logits: List[torch.Tensor]
    transaction_embeddings: List[torch.Tensor]


class _BidirectionalEvolvedMatrix(nn.Module):
    """Row-wise LSTM evolution of a graph-convolution weight matrix.

    The BF-HGN paper evolves GCN/RGCN weights instead of recurrently carrying
    node states (the node sets change between Elliptic++ snapshots).  Each row
    is therefore treated as one item in the LSTM batch.  Independent forward
    and backward LSTMs implement the two temporal directions in Eqs. (2)-(7).
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.initial_weight = nn.Parameter(torch.empty(hidden_dim, hidden_dim))
        self.forward_lstm = nn.LSTMCell(hidden_dim, hidden_dim)
        self.backward_lstm = nn.LSTMCell(hidden_dim, hidden_dim)
        nn.init.xavier_uniform_(self.initial_weight)

    @staticmethod
    def _evolve(
        initial_weight: torch.Tensor,
        cell: nn.LSTMCell,
        length: int,
    ) -> List[torch.Tensor]:
        weight = initial_weight
        memory = torch.zeros_like(initial_weight)
        sequence: List[torch.Tensor] = []
        for _ in range(length):
            weight, memory = cell(weight, (weight, memory))
            sequence.append(weight)
        return sequence

    def forward(self, length: int) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        forward_weights = self._evolve(
            self.initial_weight, self.forward_lstm, length
        )
        backward_from_end = self._evolve(
            self.initial_weight, self.backward_lstm, length
        )
        return forward_weights, list(reversed(backward_from_end))


def _address_flow_propagation(
    address_states: torch.Tensor,
    snapshot: EllipticPPHypergraphSnapshot,
    weight: torch.Tensor,
) -> torch.Tensor:
    """Time-local Addr-Addr propagation without materialising dense cliques.

    Elliptic++ publishes a global Addr-Addr edge list without edge timestamps.
    Filtering that all-period graph would leak future relations.  This
    equivalent two-hop implementation sends input-address states through the
    current transaction to output addresses and vice versa using only the
    current snapshot's incidence relations.
    """

    support = address_states @ weight
    input_at_transaction = _address_to_transaction_mean(
        support, snapshot.input_index, snapshot.num_transactions
    )
    output_at_transaction = _address_to_transaction_mean(
        support, snapshot.output_index, snapshot.num_transactions
    )
    message_to_inputs = scatter_mean(
        output_at_transaction[snapshot.input_index[1]],
        snapshot.input_index[0],
        dim=0,
        dim_size=snapshot.num_addresses,
    )
    message_to_outputs = scatter_mean(
        input_at_transaction[snapshot.output_index[1]],
        snapshot.output_index[0],
        dim=0,
        dim_size=snapshot.num_addresses,
    )
    return (support + message_to_inputs + message_to_outputs) / 3.0


class _BFHGNBiEvolveGCN(nn.Module):
    """One-layer bidirectional homogeneous feature transformer."""

    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.transaction_weight = _BidirectionalEvolvedMatrix(hidden_dim)
        self.address_weight = _BidirectionalEvolvedMatrix(hidden_dim)
        self.transaction_fusion = nn.Linear(2 * hidden_dim, hidden_dim)
        self.address_fusion = nn.Linear(2 * hidden_dim, hidden_dim)
        self.transaction_norm = nn.LayerNorm(hidden_dim)
        self.address_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        address_states: Sequence[torch.Tensor],
        transaction_states: Sequence[torch.Tensor],
        snapshots: Sequence[EllipticPPHypergraphSnapshot],
        transaction_edges: Sequence[torch.Tensor],
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        length = len(snapshots)
        tx_forward, tx_backward = self.transaction_weight(length)
        addr_forward, addr_backward = self.address_weight(length)
        next_address: List[torch.Tensor] = []
        next_transaction: List[torch.Tensor] = []
        for index, (address, transaction, snapshot, edge_index) in enumerate(
            zip(address_states, transaction_states, snapshots, transaction_edges)
        ):
            transaction_message = self.transaction_fusion(
                torch.cat(
                    [
                        _normalized_gcn_propagation(
                            transaction, edge_index, tx_forward[index]
                        ),
                        _normalized_gcn_propagation(
                            transaction, edge_index, tx_backward[index]
                        ),
                    ],
                    dim=-1,
                )
            )
            address_message = self.address_fusion(
                torch.cat(
                    [
                        _address_flow_propagation(
                            address, snapshot, addr_forward[index]
                        ),
                        _address_flow_propagation(
                            address, snapshot, addr_backward[index]
                        ),
                    ],
                    dim=-1,
                )
            )
            next_transaction.append(
                self.transaction_norm(
                    transaction + self.dropout(F.relu(transaction_message))
                )
            )
            next_address.append(
                self.address_norm(
                    address + self.dropout(F.relu(address_message))
                )
            )
        return next_address, next_transaction


class _BFHGNBiEvolveRGCNLayer(nn.Module):
    """One bidirectional relation-specific heterogeneous propagation layer."""

    RELATIONS = (
        "address_input_to_transaction",
        "address_output_to_transaction",
        "transaction_input_to_address",
        "transaction_output_to_address",
        "transaction_self",
        "address_self",
        "transaction_flow",
        "address_flow",
    )

    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.weights = nn.ModuleDict(
            {
                relation: _BidirectionalEvolvedMatrix(hidden_dim)
                for relation in self.RELATIONS
            }
        )
        self.transaction_fusion = nn.Linear(2 * hidden_dim, hidden_dim)
        self.address_fusion = nn.Linear(2 * hidden_dim, hidden_dim)
        self.transaction_norm = nn.LayerNorm(hidden_dim)
        self.address_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _relation_messages(
        address_states: torch.Tensor,
        transaction_states: torch.Tensor,
        snapshot: EllipticPPHypergraphSnapshot,
        transaction_edge_index: torch.Tensor,
        weights: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        transaction_message = (
            _address_to_transaction_mean(
                address_states @ weights["address_input_to_transaction"],
                snapshot.input_index,
                snapshot.num_transactions,
            )
            + _address_to_transaction_mean(
                address_states @ weights["address_output_to_transaction"],
                snapshot.output_index,
                snapshot.num_transactions,
            )
            + transaction_states @ weights["transaction_self"]
            + _normalized_gcn_propagation(
                transaction_states,
                transaction_edge_index,
                weights["transaction_flow"],
            )
        ) / 4.0
        address_message = (
            _transaction_to_address_mean(
                transaction_states @ weights["transaction_input_to_address"],
                snapshot.input_index,
                snapshot.num_addresses,
            )
            + _transaction_to_address_mean(
                transaction_states @ weights["transaction_output_to_address"],
                snapshot.output_index,
                snapshot.num_addresses,
            )
            + address_states @ weights["address_self"]
            + _address_flow_propagation(
                address_states, snapshot, weights["address_flow"]
            )
        ) / 4.0
        return address_message, transaction_message

    def forward(
        self,
        address_states: Sequence[torch.Tensor],
        transaction_states: Sequence[torch.Tensor],
        snapshots: Sequence[EllipticPPHypergraphSnapshot],
        transaction_edges: Sequence[torch.Tensor],
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        length = len(snapshots)
        evolved = {
            relation: self.weights[relation](length) for relation in self.RELATIONS
        }
        next_address: List[torch.Tensor] = []
        next_transaction: List[torch.Tensor] = []
        for index, (address, transaction, snapshot, edge_index) in enumerate(
            zip(address_states, transaction_states, snapshots, transaction_edges)
        ):
            forward_weights = {
                relation: evolved[relation][0][index] for relation in self.RELATIONS
            }
            backward_weights = {
                relation: evolved[relation][1][index] for relation in self.RELATIONS
            }
            address_forward, transaction_forward = self._relation_messages(
                address, transaction, snapshot, edge_index, forward_weights
            )
            address_backward, transaction_backward = self._relation_messages(
                address, transaction, snapshot, edge_index, backward_weights
            )
            transaction_message = self.transaction_fusion(
                torch.cat([transaction_forward, transaction_backward], dim=-1)
            )
            address_message = self.address_fusion(
                torch.cat([address_forward, address_backward], dim=-1)
            )
            next_transaction.append(
                self.transaction_norm(
                    transaction + self.dropout(F.relu(transaction_message))
                )
            )
            next_address.append(
                self.address_norm(
                    address + self.dropout(F.relu(address_message))
                )
            )
        return next_address, next_transaction


def _same_type_neighbor_mean(
    states: torch.Tensor,
    edge_index: torch.Tensor,
) -> torch.Tensor:
    edge_index, _ = add_self_loops(edge_index, num_nodes=states.size(0))
    return scatter_mean(
        states[edge_index[0]],
        edge_index[1],
        dim=0,
        dim_size=states.size(0),
    )


class BFHGNReimplemented(nn.Module):
    """Paper-level transaction-node reproduction of BF-HGN.

    It contains the selected MFFE route (one Bi-EvolveGCN followed by two
    Bi-EvolveRGCN layers) and the class-balanced classifier constrained by AA
    and AFSR losses.  The public paper does not include source code; this class
    makes every ambiguity explicit and keeps temporal windows causal at
    validation/test time.
    """

    def __init__(
        self,
        address_statistics: FeatureStatistics,
        transaction_statistics: FeatureStatistics,
        hidden_dim: int,
        dropout: float,
        pseudo_anomaly_ratio: float = 0.2,
        loss_mix: float = 0.3,
        affinity_margin: float = 0.7,
        noise_mean: float = 0.015,
        noise_std: float = 0.005,
        rgcn_layers: int = 2,
        supervision_mode: str = "paper_one_class",
    ) -> None:
        super().__init__()
        if rgcn_layers < 1:
            raise ValueError("BF-HGN requires at least one Bi-EvolveRGCN layer.")
        if not 0.0 < pseudo_anomaly_ratio < 1.0:
            raise ValueError("pseudo_anomaly_ratio must lie strictly between 0 and 1.")
        self.pseudo_anomaly_ratio = float(pseudo_anomaly_ratio)
        self.loss_mix = float(loss_mix)
        self.affinity_margin = float(affinity_margin)
        self.noise_mean = float(noise_mean)
        self.noise_std = float(noise_std)
        if supervision_mode not in {"paper_one_class", "supervised"}:
            raise ValueError("Unknown BF-HGN supervision_mode.")
        self.supervision_mode = supervision_mode
        self.address_standardizer = FeatureStandardizer(address_statistics)
        self.transaction_standardizer = FeatureStandardizer(transaction_statistics)
        self.address_encoder = nn.Sequential(
            nn.Linear(int(address_statistics.mean.numel()), hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        self.transaction_encoder = nn.Sequential(
            nn.Linear(int(transaction_statistics.mean.numel()), hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        self.homogeneous_transformer = _BFHGNBiEvolveGCN(hidden_dim, dropout)
        self.heterogeneous_layers = nn.ModuleList(
            [
                _BFHGNBiEvolveRGCNLayer(hidden_dim, dropout)
                for _ in range(rgcn_layers)
            ]
        )
        self.pseudo_anomaly_generator = nn.Linear(hidden_dim, hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward_sequence(
        self,
        snapshots: Sequence[EllipticPPHypergraphSnapshot],
        transaction_edges: Sequence[torch.Tensor],
    ) -> BFHGNSequenceOutput:
        if len(snapshots) != len(transaction_edges) or not snapshots:
            raise ValueError("BF-HGN requires equally sized non-empty snapshot/edge sequences.")
        address_states = [
            self.address_encoder(
                self.address_standardizer(snapshot.address_features)
            )
            for snapshot in snapshots
        ]
        transaction_states = [
            self.transaction_encoder(
                self.transaction_standardizer(
                    snapshot.transaction_features,
                    snapshot.transaction_feature_mask,
                )
            )
            for snapshot in snapshots
        ]
        address_states, transaction_states = self.homogeneous_transformer(
            address_states,
            transaction_states,
            snapshots,
            transaction_edges,
        )
        for layer in self.heterogeneous_layers:
            address_states, transaction_states = layer(
                address_states, transaction_states, snapshots, transaction_edges
            )
        return BFHGNSequenceOutput(
            logits=[self.classifier(states).squeeze(-1) for states in transaction_states],
            transaction_embeddings=transaction_states,
        )

    def class_balanced_objective(
        self,
        output: BFHGNSequenceOutput,
        snapshots: Sequence[EllipticPPHypergraphSnapshot],
        transaction_edges: Sequence[torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Apply the paper classifier under one-class or unified supervision."""

        total_bce = torch.zeros((), device=output.logits[0].device)
        total_aa = torch.zeros_like(total_bce)
        total_afsr = torch.zeros_like(total_bce)
        used_snapshots = 0
        for snapshot_index, (embeddings, snapshot, edge_index) in enumerate(
            zip(output.transaction_embeddings, snapshots, transaction_edges)
        ):
            normal_indices = torch.nonzero(
                snapshot.labels == 0, as_tuple=False
            ).flatten()
            if normal_indices.numel() < 2:
                continue
            pseudo_count = max(
                1, int(round(self.pseudo_anomaly_ratio * normal_indices.numel()))
            )
            pseudo_count = min(pseudo_count, normal_indices.numel() - 1)
            order = torch.randperm(normal_indices.numel(), device=normal_indices.device)
            pseudo_source = normal_indices[order[:pseudo_count]]
            retained_normal = normal_indices[order[pseudo_count:]]
            neighbor_embeddings = _same_type_neighbor_mean(embeddings, edge_index)
            noisy_normal = (
                embeddings[pseudo_source]
                + torch.randn_like(embeddings[pseudo_source]) * self.noise_std
                + self.noise_mean
            )
            pseudo_embeddings = F.relu(
                self.pseudo_anomaly_generator(neighbor_embeddings[pseudo_source])
            )

            normal_logits = self.classifier(embeddings[retained_normal]).squeeze(-1)
            pseudo_logits = self.classifier(pseudo_embeddings).squeeze(-1)
            if self.supervision_mode == "supervised":
                original_indices = torch.nonzero(
                    snapshot.labeled_mask, as_tuple=False
                ).flatten()
                original_logits = output.logits[snapshot_index][original_indices]
                original_targets = snapshot.labels[original_indices].to(
                    dtype=torch.float32
                )
                logits = torch.cat([original_logits, pseudo_logits], dim=0)
                targets = torch.cat(
                    [original_targets, torch.ones_like(pseudo_logits)], dim=0
                )
                positives = targets.sum().clamp_min(1.0)
                negatives = (targets.numel() - targets.sum()).clamp_min(1.0)
                bce = F.binary_cross_entropy_with_logits(
                    logits, targets, pos_weight=negatives / positives
                )
            else:
                logits = torch.cat([normal_logits, pseudo_logits], dim=0)
                targets = torch.cat(
                    [
                        torch.zeros_like(normal_logits),
                        torch.ones_like(pseudo_logits),
                    ],
                    dim=0,
                )
                class_weights = torch.where(
                    targets > 0.5,
                    torch.full_like(targets, 0.65),
                    torch.full_like(targets, 0.35),
                )
                bce = F.binary_cross_entropy_with_logits(
                    logits, targets, weight=class_weights
                )

            normal_distance = torch.linalg.vector_norm(
                embeddings[retained_normal]
                - neighbor_embeddings[retained_normal],
                dim=-1,
            )
            pseudo_distance = torch.linalg.vector_norm(
                pseudo_embeddings - neighbor_embeddings[pseudo_source], dim=-1
            )
            normal_affinity = torch.exp(
                -normal_distance / embeddings.size(1) ** 0.5
            ).mean()
            pseudo_affinity = torch.exp(
                -pseudo_distance / embeddings.size(1) ** 0.5
            ).mean()
            aa = F.relu(
                self.affinity_margin - (normal_affinity - pseudo_affinity)
            )
            afsr = torch.linalg.vector_norm(
                pseudo_embeddings - noisy_normal, dim=-1
            ).mean() / embeddings.size(1) ** 0.5
            total_bce = total_bce + bce
            total_aa = total_aa + aa
            total_afsr = total_afsr + afsr
            used_snapshots += 1

        if used_snapshots == 0:
            raise ValueError("No snapshot contains enough labeled licit nodes for BF-HGN.")
        total_bce = total_bce / used_snapshots
        total_aa = total_aa / used_snapshots
        total_afsr = total_afsr / used_snapshots
        loss = total_bce + self.loss_mix * total_aa + (1.0 - self.loss_mix) * total_afsr
        components = {
            "bce": float(total_bce.detach()),
            "aa": float(total_aa.detach()),
            "afsr": float(total_afsr.detach()),
        }
        return loss, components
