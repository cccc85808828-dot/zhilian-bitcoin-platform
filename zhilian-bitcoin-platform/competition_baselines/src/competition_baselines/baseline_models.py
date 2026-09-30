from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import GATConv, GCNConv, HGTConv
from torch_geometric.utils import add_self_loops
from torch_scatter import scatter_add

from .baseline_data import FeatureStandardizer, FeatureStatistics, GraphSnapshot


def mean_propagation(
    states: torch.Tensor, edge_index: torch.Tensor, add_loops: bool = True
) -> torch.Tensor:
    if add_loops:
        edge_index, _ = add_self_loops(edge_index, num_nodes=states.size(0))
    source, target = edge_index
    degree = scatter_add(
        torch.ones_like(target, dtype=states.dtype),
        target,
        dim=0,
        dim_size=states.size(0),
    ).clamp_min(1.0)
    return scatter_add(
        states[source], target, dim=0, dim_size=states.size(0)
    ) / degree[:, None]


def symmetric_gcn_propagation(
    states: torch.Tensor, edge_index: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    edge_index, _ = add_self_loops(edge_index, num_nodes=states.size(0))
    source, target = edge_index
    degree = scatter_add(
        torch.ones_like(target, dtype=states.dtype),
        target,
        dim=0,
        dim_size=states.size(0),
    ).clamp_min(1.0)
    norm = degree[source].pow(-0.5) * degree[target].pow(-0.5)
    support = states @ weight
    return scatter_add(
        support[source] * norm[:, None],
        target,
        dim=0,
        dim_size=states.size(0),
    )


def detach_state(state):
    if state is None:
        return None
    if torch.is_tensor(state):
        return state.detach()
    return [value.detach() for value in state]


class TransactionEncoder(nn.Module):
    def __init__(
        self, statistics: FeatureStatistics, hidden_dim: int, dropout: float
    ) -> None:
        super().__init__()
        self.standardizer = FeatureStandardizer(statistics)
        self.network = nn.Sequential(
            nn.Linear(int(statistics.mean.numel()), hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, snapshot: GraphSnapshot) -> torch.Tensor:
        return self.network(
            self.standardizer(
                snapshot.transaction_features, snapshot.transaction_feature_mask
            )
        )


class AddressEncoder(nn.Module):
    def __init__(
        self, statistics: FeatureStatistics, hidden_dim: int, dropout: float
    ) -> None:
        super().__init__()
        self.standardizer = FeatureStandardizer(statistics)
        self.network = nn.Sequential(
            nn.Linear(int(statistics.mean.numel()), hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, snapshot: GraphSnapshot) -> torch.Tensor:
        return self.network(
            self.standardizer(snapshot.address_features, snapshot.address_feature_mask)
        )


def aggregate_role_addresses(
    address_states: torch.Tensor, snapshot: GraphSnapshot
) -> torch.Tensor:
    outputs: List[torch.Tensor] = []
    for relation in (snapshot.input_index, snapshot.output_index):
        address_index, transaction_index = relation
        sums = torch.zeros(
            (snapshot.num_transactions, address_states.size(1)),
            dtype=address_states.dtype,
            device=address_states.device,
        )
        counts = torch.zeros(
            snapshot.num_transactions,
            dtype=address_states.dtype,
            device=address_states.device,
        )
        if transaction_index.numel():
            sums.index_add_(0, transaction_index, address_states[address_index])
            counts.index_add_(
                0, transaction_index, torch.ones_like(transaction_index, dtype=states_dtype(address_states))
            )
        outputs.append(sums / counts.clamp_min(1.0)[:, None])
    return torch.cat(outputs, dim=1)


def states_dtype(states: torch.Tensor) -> torch.dtype:
    return states.dtype


@dataclass
class RootModelOutput:
    logits: torch.Tensor
    embeddings: torch.Tensor


class NSGCNLSTMModel(nn.Module):
    def __init__(
        self,
        transaction_statistics: FeatureStatistics,
        hidden_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.transaction_encoder = TransactionEncoder(
            transaction_statistics, hidden_dim, dropout
        )
        self.gcn1 = GCNConv(hidden_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _chronological_sequences(
        states: torch.Tensor, sequence_index: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        valid = sequence_index >= 0
        lengths = valid.sum(dim=1).clamp_min(1)
        sequence_states = states[sequence_index.clamp_min(0)]
        sequence_states = sequence_states * valid.unsqueeze(-1)
        positions = torch.arange(sequence_index.size(1), device=states.device)[None, :]
        reverse_positions = (lengths[:, None] - 1 - positions).clamp_min(0)
        reverse_positions = torch.where(valid, reverse_positions, positions)
        gather_index = reverse_positions.unsqueeze(-1).expand_as(sequence_states)
        return sequence_states.gather(1, gather_index), lengths

    def forward_roots(
        self,
        snapshot: GraphSnapshot,
        edge_index: torch.Tensor,
        roots: torch.Tensor,
        predecessor_sequences: torch.Tensor,
    ) -> RootModelOutput:
        states = self.transaction_encoder(snapshot)
        states = self.dropout(F.relu(self.gcn1(states, edge_index)))
        states = self.dropout(F.relu(self.gcn2(states, edge_index)))
        sequence_index = predecessor_sequences[roots.cpu()].to(
            roots.device, dtype=torch.long
        )
        ordered, lengths = self._chronological_sequences(states, sequence_index)
        sequence_output, _ = self.lstm(ordered)
        final = sequence_output[
            torch.arange(roots.numel(), device=roots.device), lengths - 1
        ]
        fused = torch.cat([states[roots], final], dim=1)
        return RootModelOutput(
            logits=self.classifier(fused).squeeze(-1), embeddings=fused
        )


class EllipticHGTModel(nn.Module):
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
        transaction_statistics: FeatureStatistics,
        address_statistics: FeatureStatistics,
        hidden_dim: int,
        dropout: float,
        heads: int,
        layers: int,
    ) -> None:
        super().__init__()
        self.transaction_encoder = TransactionEncoder(
            transaction_statistics, hidden_dim, dropout
        )
        self.address_encoder = AddressEncoder(address_statistics, hidden_dim, dropout)
        metadata = (self.NODE_TYPES, self.EDGE_TYPES)
        self.layers = nn.ModuleList(
            [
                HGTConv(
                    {node_type: hidden_dim for node_type in self.NODE_TYPES},
                    hidden_dim,
                    metadata,
                    heads=heads,
                )
                for _ in range(layers)
            ]
        )
        self.norms = nn.ModuleList(
            [
                nn.ModuleDict(
                    {node_type: nn.LayerNorm(hidden_dim) for node_type in self.NODE_TYPES}
                )
                for _ in range(layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    @classmethod
    def edge_dictionary(
        cls, snapshot: GraphSnapshot, transaction_edge_index: torch.Tensor
    ) -> Dict[Tuple[str, str, str], torch.Tensor]:
        return {
            ("address", "input_to", "transaction"): snapshot.input_index,
            ("transaction", "input_used_by", "address"): torch.stack(
                [snapshot.input_index[1], snapshot.input_index[0]], dim=0
            ),
            ("transaction", "output_to", "address"): torch.stack(
                [snapshot.output_index[1], snapshot.output_index[0]], dim=0
            ),
            ("address", "output_from", "transaction"): snapshot.output_index,
            ("transaction", "flow", "transaction"): transaction_edge_index,
        }

    def forward_snapshot(
        self, snapshot: GraphSnapshot, edge_index: torch.Tensor
    ) -> RootModelOutput:
        states = {
            "address": self.address_encoder(snapshot),
            "transaction": self.transaction_encoder(snapshot),
        }
        edge_dict = self.edge_dictionary(snapshot, edge_index)
        for layer, norms in zip(self.layers, self.norms):
            messages = layer(states, edge_dict)
            states = {
                node_type: norms[node_type](
                    states[node_type] + self.dropout(F.relu(messages[node_type]))
                )
                for node_type in self.NODE_TYPES
            }
        embeddings = states["transaction"]
        return RootModelOutput(
            logits=self.classifier(embeddings).squeeze(-1), embeddings=embeddings
        )


class TFGATDCPLUModel(nn.Module):
    """Global-local TFGAT with memory-bounded global summary tokens."""

    def __init__(
        self,
        transaction_statistics: FeatureStatistics,
        hidden_dim: int,
        dropout: float,
        heads: int,
        transformer_layers: int,
        global_chunk_size: int,
        global_tokens: int,
    ) -> None:
        super().__init__()
        self.transaction_encoder = TransactionEncoder(
            transaction_statistics, hidden_dim, dropout
        )
        self.local_gat = GATConv(
            hidden_dim,
            hidden_dim // heads,
            heads=heads,
            dropout=dropout,
            add_self_loops=True,
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.global_transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=transformer_layers
        )
        self.global_chunk_size = int(global_chunk_size)
        self.global_tokens = int(global_tokens)
        self.fusion_gate = nn.Linear(hidden_dim * 2, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.dropout = nn.Dropout(dropout)

    def _global_attention(self, states: torch.Tensor) -> torch.Tensor:
        if states.size(0) <= self.global_chunk_size:
            return self.global_transformer(states.unsqueeze(0)).squeeze(0)
        summary = F.adaptive_avg_pool1d(
            states.t().unsqueeze(0), self.global_tokens
        ).squeeze(0).t()
        outputs: List[torch.Tensor] = []
        for start in range(0, states.size(0), self.global_chunk_size):
            chunk = states[start : start + self.global_chunk_size]
            encoded = self.global_transformer(
                torch.cat([summary, chunk], dim=0).unsqueeze(0)
            ).squeeze(0)
            outputs.append(encoded[self.global_tokens :])
        return torch.cat(outputs, dim=0)

    def forward_snapshot(
        self, snapshot: GraphSnapshot, edge_index: torch.Tensor
    ) -> RootModelOutput:
        base = self.transaction_encoder(snapshot)
        local = self.dropout(F.elu(self.local_gat(base, edge_index)))
        global_states = self.dropout(self._global_attention(base))
        gate = torch.sigmoid(self.fusion_gate(torch.cat([global_states, local], dim=1)))
        embeddings = self.norm(base + gate * global_states + (1.0 - gate) * local)
        return RootModelOutput(
            logits=self.classifier(embeddings).squeeze(-1), embeddings=embeddings
        )


class MultiDistanceEncoder(nn.Module):
    def __init__(self, hidden_dim: int, hops: int, dropout: float) -> None:
        super().__init__()
        self.hops = int(hops)
        self.projections = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim, bias=False) for _ in range(hops)]
        )
        self.self_projection = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, states: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        current = states
        messages = self.self_projection(states)
        for projection in self.projections:
            current = mean_propagation(current, edge_index, add_loops=False)
            messages = messages + projection(current)
        return self.norm(states + self.dropout(F.relu(messages)))


@dataclass
class DynamicModelOutput:
    logits: torch.Tensor
    embeddings: torch.Tensor
    state: object
    auxiliary_loss: torch.Tensor


class MDSTGNNModel(nn.Module):
    """Paper-faithful MDST-GNN adapted to disjoint Elliptic transaction nodes.

    Elliptic transaction identities do not persist between snapshots, so the
    historical model is a causal graph-level GRU context rather than a
    per-transaction recurrent state.
    """

    def __init__(
        self,
        transaction_statistics: FeatureStatistics,
        hidden_dim: int,
        dropout: float,
        hops: int,
    ) -> None:
        super().__init__()
        self.transaction_encoder = TransactionEncoder(
            transaction_statistics, hidden_dim, dropout
        )
        self.multi_distance = MultiDistanceEncoder(hidden_dim, hops, dropout)
        self.history_cell = nn.GRUCell(hidden_dim, hidden_dim)
        self.periodic_projection = nn.Linear(4, hidden_dim)
        self.fusion_gate = nn.Linear(hidden_dim * 2, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.discriminator = nn.Bilinear(hidden_dim, hidden_dim, 1)

    @staticmethod
    def periodic(time_id: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        value = float(time_id)
        return torch.tensor(
            [
                math.sin(2 * math.pi * value / 7.0),
                math.cos(2 * math.pi * value / 7.0),
                math.sin(2 * math.pi * value / 49.0),
                math.cos(2 * math.pi * value / 49.0),
            ],
            device=device,
            dtype=dtype,
        )

    def forward_snapshot(
        self,
        snapshot: GraphSnapshot,
        edge_index: torch.Tensor,
        previous_history: Optional[torch.Tensor],
        compute_ssl: bool,
    ) -> DynamicModelOutput:
        base = self.transaction_encoder(snapshot)
        spatial = self.multi_distance(base, edge_index)
        if previous_history is None:
            previous_history = torch.zeros(
                spatial.size(1), device=spatial.device, dtype=spatial.dtype
            )
        periodic = self.periodic(snapshot.time_id, spatial.device, spatial.dtype)
        history = previous_history + self.periodic_projection(periodic)
        history_nodes = history[None, :].expand_as(spatial)
        gate = torch.sigmoid(self.fusion_gate(torch.cat([spatial, history_nodes], dim=1)))
        embeddings = self.norm(gate * spatial + (1.0 - gate) * history_nodes)
        next_history = self.history_cell(spatial.mean(dim=0), previous_history)
        auxiliary = embeddings.sum() * 0.0
        if compute_ssl:
            corrupt = base[torch.randperm(base.size(0), device=base.device)]
            corrupt = self.multi_distance(corrupt, edge_index)
            summary = torch.sigmoid(embeddings.mean(dim=0))[None, :]
            positive = self.discriminator(
                embeddings, summary.expand_as(embeddings)
            ).squeeze(-1)
            negative = self.discriminator(
                corrupt, summary.expand_as(corrupt)
            ).squeeze(-1)
            auxiliary = 0.5 * (
                F.binary_cross_entropy_with_logits(positive, torch.ones_like(positive))
                + F.binary_cross_entropy_with_logits(negative, torch.zeros_like(negative))
            )
        return DynamicModelOutput(
            logits=self.classifier(embeddings).squeeze(-1),
            embeddings=embeddings,
            state=next_history,
            auxiliary_loss=auxiliary,
        )


class EvolveGCNHEncoder(nn.Module):
    """Compact EvolveGCN-H-style encoder conditioned on each current graph."""

    def __init__(self, hidden_dim: int, layers: int, dropout: float) -> None:
        super().__init__()
        self.initial_weights = nn.ParameterList(
            [nn.Parameter(torch.empty(hidden_dim, hidden_dim)) for _ in range(layers)]
        )
        self.row_embeddings = nn.ParameterList(
            [nn.Parameter(torch.empty(hidden_dim, hidden_dim)) for _ in range(layers)]
        )
        self.updaters = nn.ModuleList(
            [nn.GRUCell(hidden_dim, hidden_dim) for _ in range(layers)]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(layers)])
        self.dropout = nn.Dropout(dropout)
        for weight, rows in zip(self.initial_weights, self.row_embeddings):
            nn.init.xavier_uniform_(weight)
            nn.init.xavier_uniform_(rows)

    def initial_state(self) -> List[torch.Tensor]:
        return [weight for weight in self.initial_weights]

    def forward(
        self,
        states: torch.Tensor,
        edge_index: torch.Tensor,
        previous_weights: Optional[Sequence[torch.Tensor]],
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        weights = self.initial_state() if previous_weights is None else list(previous_weights)
        next_weights: List[torch.Tensor] = []
        for previous, row_embedding, updater, norm in zip(
            weights, self.row_embeddings, self.updaters, self.norms
        ):
            graph_summary = states.mean(dim=0, keepdim=True)
            updater_input = torch.tanh(row_embedding + graph_summary)
            evolved = updater(updater_input, previous)
            message = symmetric_gcn_propagation(states, edge_index, evolved)
            states = norm(states + self.dropout(F.relu(message)))
            next_weights.append(evolved)
        return states, next_weights


class FGEGCNModel(nn.Module):
    def __init__(
        self,
        transaction_statistics: FeatureStatistics,
        hidden_dim: int,
        dropout: float,
        layers: int,
    ) -> None:
        super().__init__()
        self.transaction_encoder = TransactionEncoder(
            transaction_statistics, hidden_dim, dropout
        )
        self.temporal_encoder = EvolveGCNHEncoder(hidden_dim, layers, dropout)
        self.feature_branch = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.LayerNorm(hidden_dim)
        )
        self.gate = nn.Linear(hidden_dim * 2, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward_snapshot(
        self,
        snapshot: GraphSnapshot,
        edge_index: torch.Tensor,
        previous_weights: Optional[Sequence[torch.Tensor]],
    ) -> DynamicModelOutput:
        base = self.transaction_encoder(snapshot)
        temporal, next_weights = self.temporal_encoder(base, edge_index, previous_weights)
        feature = self.feature_branch(base)
        gate = torch.sigmoid(self.gate(torch.cat([temporal, feature], dim=1)))
        embeddings = self.norm(feature + gate * (temporal - feature))
        return DynamicModelOutput(
            logits=self.classifier(embeddings).squeeze(-1),
            embeddings=embeddings,
            state=next_weights,
            auxiliary_loss=embeddings.sum() * 0.0,
        )


class GPNModel(nn.Module):
    """One-class dynamic GPN with CBC, AAD and HCBD objectives."""

    def __init__(
        self,
        transaction_statistics: FeatureStatistics,
        hidden_dim: int,
        dropout: float,
        layers: int,
        noise_std: float,
        aad_margin: float,
    ) -> None:
        super().__init__()
        self.transaction_encoder = TransactionEncoder(
            transaction_statistics, hidden_dim, dropout
        )
        self.temporal_encoder = EvolveGCNHEncoder(hidden_dim, layers, dropout)
        self.generator = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.noise_std = float(noise_std)
        self.aad_margin = float(aad_margin)

    def encode_snapshot(
        self,
        snapshot: GraphSnapshot,
        edge_index: torch.Tensor,
        previous_weights: Optional[Sequence[torch.Tensor]],
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        base = self.transaction_encoder(snapshot)
        return self.temporal_encoder(base, edge_index, previous_weights)

    def inference_snapshot(
        self,
        snapshot: GraphSnapshot,
        edge_index: torch.Tensor,
        previous_weights: Optional[Sequence[torch.Tensor]],
    ) -> DynamicModelOutput:
        embeddings, next_weights = self.encode_snapshot(
            snapshot, edge_index, previous_weights
        )
        return DynamicModelOutput(
            logits=self.classifier(embeddings).squeeze(-1),
            embeddings=embeddings,
            state=next_weights,
            auxiliary_loss=embeddings.sum() * 0.0,
        )

    def training_objective(
        self,
        snapshot: GraphSnapshot,
        edge_index: torch.Tensor,
        previous_weights: Optional[Sequence[torch.Tensor]],
        max_normal_samples: int,
        aad_weight: float,
        hcbd_weight: float,
        compactness_weight: float,
    ) -> Tuple[torch.Tensor, List[torch.Tensor], Dict[str, float]]:
        embeddings, next_weights = self.encode_snapshot(
            snapshot, edge_index, previous_weights
        )
        normal_indices = torch.nonzero(snapshot.labels == 0, as_tuple=False).flatten()
        if normal_indices.numel() > max_normal_samples:
            normal_indices = normal_indices[
                torch.randperm(normal_indices.numel(), device=embeddings.device)[
                    :max_normal_samples
                ]
            ]
        normal = embeddings[normal_indices]
        neighbor_all = mean_propagation(embeddings, edge_index)
        neighbor = neighbor_all[normal_indices]
        noise = torch.randn_like(normal) * self.noise_std
        generated_delta = self.generator(torch.cat([normal, neighbor, noise], dim=1))
        pseudo = normal + generated_delta + noise
        normal_logits = self.classifier(normal).squeeze(-1)
        pseudo_logits = self.classifier(pseudo).squeeze(-1)
        classification = 0.5 * (
            F.binary_cross_entropy_with_logits(normal_logits, torch.zeros_like(normal_logits))
            + F.binary_cross_entropy_with_logits(pseudo_logits, torch.ones_like(pseudo_logits))
        )
        normal_affinity = F.cosine_similarity(normal, neighbor, dim=1)
        pseudo_affinity = F.cosine_similarity(pseudo, neighbor.detach(), dim=1)
        aad = F.relu(
            self.aad_margin - (normal_affinity - pseudo_affinity)
        ).mean()
        pseudo_probability = torch.sigmoid(pseudo_logits)
        hcbd = (pseudo_probability - 0.5).abs().mean()
        center = normal.mean(dim=0, keepdim=True)
        compactness = (normal - center).square().mean()
        loss = (
            classification
            + aad_weight * aad
            + hcbd_weight * hcbd
            + compactness_weight * compactness
        )
        return (
            loss,
            next_weights,
            {
                "classification": float(classification.detach()),
                "aad": float(aad.detach()),
                "hcbd": float(hcbd.detach()),
                "compactness": float(compactness.detach()),
                "normal_samples": float(normal_indices.numel()),
            },
        )
