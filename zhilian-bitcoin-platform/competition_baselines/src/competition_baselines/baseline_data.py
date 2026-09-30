from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch_geometric.utils import to_undirected
from tqdm import tqdm


@dataclass
class GraphSnapshot:
    """One leakage-safe Elliptic++ temporal snapshot.

    Both incidence tensors use ``[local_address, local_transaction]`` row
    order.  Their semantic direction is represented by the field name.
    """

    time_id: int
    global_address_ids: torch.Tensor
    address_features: torch.Tensor
    address_feature_mask: torch.Tensor
    transaction_ids: torch.Tensor
    transaction_features: torch.Tensor
    transaction_feature_mask: torch.Tensor
    input_index: torch.Tensor
    output_index: torch.Tensor
    labels: torch.Tensor
    labeled_mask: torch.Tensor

    @property
    def num_addresses(self) -> int:
        return int(self.global_address_ids.numel())

    @property
    def num_transactions(self) -> int:
        return int(self.transaction_ids.numel())

    def to(self, device: torch.device) -> "GraphSnapshot":
        return GraphSnapshot(
            time_id=self.time_id,
            global_address_ids=self.global_address_ids.to(device),
            address_features=self.address_features.to(device),
            address_feature_mask=self.address_feature_mask.to(device),
            transaction_ids=self.transaction_ids.to(device),
            transaction_features=self.transaction_features.to(device),
            transaction_feature_mask=self.transaction_feature_mask.to(device),
            input_index=self.input_index.to(device),
            output_index=self.output_index.to(device),
            labels=self.labels.to(device),
            labeled_mask=self.labeled_mask.to(device),
        )

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "GraphSnapshot":
        snapshot = cls(
            time_id=int(payload["time_id"]),
            global_address_ids=payload["global_address_ids"],
            address_features=payload["address_features"],
            address_feature_mask=payload["address_feature_mask"],
            transaction_ids=payload["transaction_ids"],
            transaction_features=payload["transaction_features"],
            transaction_feature_mask=payload["transaction_feature_mask"],
            input_index=payload["input_index"],
            output_index=payload["output_index"],
            labels=payload["labels"],
            labeled_mask=payload["labeled_mask"],
        )
        snapshot.validate()
        return snapshot

    def validate(self) -> None:
        if self.transaction_features.size(0) != self.num_transactions:
            raise ValueError("Transaction feature rows do not match transaction IDs.")
        if self.address_features.size(0) != self.num_addresses:
            raise ValueError("Address feature rows do not match address IDs.")
        if self.transaction_feature_mask.shape != self.transaction_features.shape:
            raise ValueError("Invalid transaction feature mask.")
        if self.address_feature_mask.shape != (self.num_addresses,):
            raise ValueError("Invalid address feature mask.")
        if not torch.equal(self.labeled_mask, self.labels >= 0):
            raise ValueError("Unknown labels must be excluded by labeled_mask.")
        for relation in (self.input_index, self.output_index):
            if relation.ndim != 2 or relation.size(0) != 2:
                raise ValueError("Role incidence must have shape [2, edges].")


@dataclass
class FeatureStatistics:
    mean: torch.Tensor
    std: torch.Tensor

    def as_dict(self) -> Dict[str, torch.Tensor]:
        return {"mean": self.mean.cpu(), "std": self.std.cpu()}


class FeatureStandardizer(nn.Module):
    def __init__(self, statistics: FeatureStatistics) -> None:
        super().__init__()
        self.register_buffer("mean", statistics.mean.to(torch.float32))
        self.register_buffer("std", statistics.std.to(torch.float32))

    def forward(
        self, values: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        output = (values.to(torch.float32) - self.mean) / self.std
        if mask is not None:
            if mask.ndim == 1:
                mask = mask[:, None].expand_as(output)
            output = torch.where(mask, output, torch.zeros_like(output))
        return torch.nan_to_num(output)


@dataclass
class GraphDataBundle:
    snapshots: Dict[int, GraphSnapshot]
    directed_edges: Dict[int, torch.Tensor]
    undirected_edges: Dict[int, torch.Tensor]
    predecessor_sequences: Dict[int, torch.Tensor]
    address_statistics: FeatureStatistics
    transaction_statistics: FeatureStatistics


def expand_interval(interval: Sequence[int]) -> List[int]:
    if len(interval) != 2:
        raise ValueError("A chronological split must be [first, last].")
    return list(range(int(interval[0]), int(interval[1]) + 1))


def load_snapshots(processed_root: Path) -> Dict[int, GraphSnapshot]:
    metadata_path = processed_root / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    result: Dict[int, GraphSnapshot] = {}
    for item in tqdm(metadata["snapshots"], desc="load Elliptic++ snapshots"):
        payload = torch.load(
            processed_root / item["file"], map_location="cpu", weights_only=False
        )
        snapshot = GraphSnapshot.from_payload(payload)
        result[snapshot.time_id] = snapshot
    if sorted(result) != list(range(1, 50)):
        raise ValueError("The processed dataset must contain time steps 1 through 49.")
    return result


def _fit_statistics(
    values: Iterable[torch.Tensor], masks: Optional[Iterable[torch.Tensor]] = None
) -> FeatureStatistics:
    values_list = list(values)
    mask_list = list(masks) if masks is not None else [None] * len(values_list)
    total = torch.zeros(values_list[0].size(1), dtype=torch.float64)
    total_sq = torch.zeros_like(total)
    count = torch.zeros_like(total)
    for tensor, mask in zip(values_list, mask_list):
        x = tensor.to(torch.float64)
        if mask is None:
            valid = torch.isfinite(x)
        else:
            valid = mask
            if valid.ndim == 1:
                valid = valid[:, None].expand_as(x)
            valid = valid & torch.isfinite(x)
        clean = torch.where(valid, x, torch.zeros_like(x))
        total += clean.sum(0)
        total_sq += clean.square().sum(0)
        count += valid.sum(0)
    safe_count = count.clamp_min(1.0)
    mean = total / safe_count
    variance = (total_sq / safe_count - mean.square()).clamp_min(1e-8)
    return FeatureStatistics(mean.to(torch.float32), variance.sqrt().to(torch.float32))


def fit_training_statistics(
    snapshots: Dict[int, GraphSnapshot], train_times: Sequence[int]
) -> Tuple[FeatureStatistics, FeatureStatistics]:
    address = _fit_statistics(
        (snapshots[t].address_features for t in train_times),
        (snapshots[t].address_feature_mask for t in train_times),
    )
    transaction = _fit_statistics(
        (snapshots[t].transaction_features for t in train_times),
        (snapshots[t].transaction_feature_mask for t in train_times),
    )
    return address, transaction


def load_transaction_edges(
    edge_path: Path, snapshots: Dict[int, GraphSnapshot]
) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
    locations: Dict[int, Tuple[int, int]] = {}
    for time_id, snapshot in snapshots.items():
        locations.update(
            {
                int(tx_id): (time_id, local_index)
                for local_index, tx_id in enumerate(snapshot.transaction_ids.tolist())
            }
        )
    edges_by_time: Dict[int, List[Tuple[int, int]]] = {t: [] for t in snapshots}
    frame = pd.read_csv(edge_path, usecols=["txId1", "txId2"])
    for source_id, target_id in frame.itertuples(index=False, name=None):
        source = locations.get(int(source_id))
        target = locations.get(int(target_id))
        if source is None or target is None:
            raise ValueError("Transaction edge references an unknown transaction.")
        if source[0] != target[0]:
            raise ValueError("Transaction edge crosses Elliptic++ time steps.")
        edges_by_time[source[0]].append((source[1], target[1]))
    directed: Dict[int, torch.Tensor] = {}
    undirected: Dict[int, torch.Tensor] = {}
    for time_id, pairs in edges_by_time.items():
        if pairs:
            edge_index = torch.tensor(pairs, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        directed[time_id] = edge_index
        undirected[time_id] = to_undirected(
            edge_index, num_nodes=snapshots[time_id].num_transactions
        )
    return directed, undirected


def _bounded_bfs_sequences(
    edge_index: torch.Tensor,
    num_nodes: int,
    max_hops: int,
    max_nodes: int,
    reverse: bool,
) -> torch.Tensor:
    source = edge_index[1 if reverse else 0].tolist()
    target = edge_index[0 if reverse else 1].tolist()
    neighbors: List[List[int]] = [[] for _ in range(num_nodes)]
    for left, right in zip(source, target):
        neighbors[int(left)].append(int(right))
    for values in neighbors:
        values.sort()
    output = torch.full((num_nodes, max_nodes), -1, dtype=torch.int32)
    for root in range(num_nodes):
        values = [root]
        frontier = [root]
        visited = {root}
        for _ in range(max_hops):
            next_frontier: List[int] = []
            for node in frontier:
                for neighbor in neighbors[node]:
                    if neighbor in visited:
                        continue
                    visited.add(neighbor)
                    values.append(neighbor)
                    next_frontier.append(neighbor)
                    if len(values) >= max_nodes:
                        break
                if len(values) >= max_nodes:
                    break
            frontier = next_frontier
            if not frontier or len(values) >= max_nodes:
                break
        output[root, : len(values)] = torch.tensor(values, dtype=torch.int32)
    return output


def _sequence_cache_signature(
    edge_path: Path, max_hops: int, max_nodes: int
) -> str:
    stat = edge_path.stat()
    value = f"{edge_path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}|{max_hops}|{max_nodes}|v1"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def load_or_build_sequences(
    cache_path: Path,
    edge_path: Path,
    directed_edges: Dict[int, torch.Tensor],
    snapshots: Dict[int, GraphSnapshot],
    max_hops: int,
    max_nodes: int,
) -> Dict[int, torch.Tensor]:
    signature = _sequence_cache_signature(edge_path, max_hops, max_nodes)
    if cache_path.is_file():
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        if payload.get("signature") == signature:
            return payload["predecessor"]
    predecessor: Dict[int, torch.Tensor] = {}
    for time_id in tqdm(sorted(snapshots), desc="build 3-hop neighborhood sequences"):
        edge_index = directed_edges[time_id]
        num_nodes = snapshots[time_id].num_transactions
        predecessor[time_id] = _bounded_bfs_sequences(
            edge_index, num_nodes, max_hops, max_nodes, reverse=True
        )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "signature": signature,
            "predecessor": predecessor,
        },
        cache_path,
    )
    return predecessor


def load_graph_data(config: Mapping[str, object]) -> GraphDataBundle:
    project_root = Path(config["project_root"])
    data_config = config["data"]
    processed_root = Path(data_config["processed_root"])
    edge_path = Path(data_config["transaction_edges"])
    cache_path = Path(data_config["sequence_cache"])
    if not processed_root.is_absolute():
        processed_root = project_root / processed_root
    if not edge_path.is_absolute():
        edge_path = project_root / edge_path
    if not cache_path.is_absolute():
        cache_path = project_root / cache_path
    snapshots = load_snapshots(processed_root)
    train_times = expand_interval(config["split"]["train"])
    address_stats, transaction_stats = fit_training_statistics(snapshots, train_times)
    directed, undirected = load_transaction_edges(edge_path, snapshots)
    predecessor = load_or_build_sequences(
        cache_path=cache_path,
        edge_path=edge_path,
        directed_edges=directed,
        snapshots=snapshots,
        max_hops=int(data_config.get("subgraph_hops", 3)),
        max_nodes=int(data_config.get("subgraph_nodes", 80)),
    )
    return GraphDataBundle(
        snapshots=snapshots,
        directed_edges=directed,
        undirected_edges=undirected,
        predecessor_sequences=predecessor,
        address_statistics=address_stats,
        transaction_statistics=transaction_stats,
    )
