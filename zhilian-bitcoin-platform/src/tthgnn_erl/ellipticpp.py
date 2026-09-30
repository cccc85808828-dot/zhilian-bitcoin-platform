from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


PathLike = Union[str, Path]
FORMAT_VERSION = 1


@dataclass
class EllipticPPHypergraphSnapshot:
    """A directed transaction hypergraph for one Elliptic++ time step.

    Addresses are nodes and transactions are hyperedges. ``input_index`` and
    ``output_index`` are COO coordinates whose first row contains local address
    indices and second row contains local transaction indices.
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

    @property
    def num_input_relations(self) -> int:
        return int(self.input_index.size(1))

    @property
    def num_output_relations(self) -> int:
        return int(self.output_index.size(1))

    def input_incidence(
        self, device: Optional[torch.device] = None
    ) -> torch.Tensor:
        return self._incidence(self.input_index, device)

    def output_incidence(
        self, device: Optional[torch.device] = None
    ) -> torch.Tensor:
        return self._incidence(self.output_index, device)

    def _incidence(
        self, index: torch.Tensor, device: Optional[torch.device]
    ) -> torch.Tensor:
        target_device = device if device is not None else index.device
        coordinates = index.to(target_device)
        values = torch.ones(coordinates.size(1), device=target_device)
        return torch.sparse_coo_tensor(
            coordinates,
            values,
            size=(self.num_addresses, self.num_transactions),
            device=target_device,
        ).coalesce()

    def to(self, device: torch.device) -> "EllipticPPHypergraphSnapshot":
        return EllipticPPHypergraphSnapshot(
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

    def validate(self) -> None:
        if self.global_address_ids.ndim != 1:
            raise ValueError("global_address_ids must be one-dimensional.")
        if self.transaction_ids.ndim != 1:
            raise ValueError("transaction_ids must be one-dimensional.")
        if self.address_features.size(0) != self.num_addresses:
            raise ValueError("Address feature rows do not match address IDs.")
        if self.transaction_features.size(0) != self.num_transactions:
            raise ValueError("Transaction feature rows do not match transaction IDs.")
        if self.transaction_feature_mask.shape != self.transaction_features.shape:
            raise ValueError("transaction_feature_mask has an invalid shape.")
        if self.address_feature_mask.shape != (self.num_addresses,):
            raise ValueError("address_feature_mask has an invalid shape.")
        if self.labels.shape != (self.num_transactions,):
            raise ValueError("labels has an invalid shape.")
        if self.labeled_mask.shape != (self.num_transactions,):
            raise ValueError("labeled_mask has an invalid shape.")
        if not torch.equal(self.labeled_mask, self.labels >= 0):
            raise ValueError("labeled_mask must exclude exactly the unknown labels.")
        valid_labels = torch.tensor([-1, 0, 1], device=self.labels.device)
        if not torch.isin(self.labels, valid_labels).all():
            raise ValueError("Labels must be encoded as unknown=-1, licit=0, illicit=1.")
        if not torch.isfinite(self.address_features).all():
            raise ValueError("Address features contain non-finite values.")
        if not torch.isfinite(self.transaction_features).all():
            raise ValueError("Transaction features contain non-finite values.")
        for name, index in (
            ("input_index", self.input_index),
            ("output_index", self.output_index),
        ):
            if index.ndim != 2 or index.size(0) != 2:
                raise ValueError(f"{name} must have shape [2, num_relations].")
            if index.numel() > 0:
                if int(index[0].min()) < 0 or int(index[0].max()) >= self.num_addresses:
                    raise ValueError(f"{name} contains an invalid address index.")
                if int(index[1].min()) < 0 or int(index[1].max()) >= self.num_transactions:
                    raise ValueError(f"{name} contains an invalid transaction index.")

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "EllipticPPHypergraphSnapshot":
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
        return snapshot


class EllipticPPSnapshotDataset(Dataset):
    """Lazy loader for preprocessed Elliptic++ temporal hypergraphs."""

    def __init__(
        self,
        root: PathLike,
        times: Optional[Sequence[int]] = None,
        map_location: Union[str, torch.device] = "cpu",
        validate: bool = True,
    ) -> None:
        self.root = Path(root)
        metadata_path = self.root / "metadata.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(f"Processed metadata not found: {metadata_path}")
        with metadata_path.open("r", encoding="utf-8") as stream:
            self.metadata: Dict[str, Any] = json.load(stream)
        if int(self.metadata["format_version"]) != FORMAT_VERSION:
            raise ValueError("Unsupported Elliptic++ processed format version.")

        manifest_by_time = {
            int(item["time_id"]): item for item in self.metadata["snapshots"]
        }
        selected_times = sorted(manifest_by_time) if times is None else [int(t) for t in times]
        missing = sorted(set(selected_times) - set(manifest_by_time))
        if missing:
            raise ValueError(f"Requested time steps are unavailable: {missing}")
        self.manifest = [manifest_by_time[time_id] for time_id in selected_times]
        self.map_location = map_location
        self.validate_snapshots = validate

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, index: int) -> EllipticPPHypergraphSnapshot:
        item = self.manifest[index]
        payload = torch.load(
            self.root / item["file"],
            map_location=self.map_location,
            weights_only=False,
        )
        snapshot = EllipticPPHypergraphSnapshot.from_payload(payload)
        if self.validate_snapshots:
            snapshot.validate()
        return snapshot

    def __iter__(self) -> Iterator[EllipticPPHypergraphSnapshot]:
        for index in range(len(self)):
            yield self[index]


def _require_columns(frame: pd.DataFrame, required: Sequence[str], source: Path) -> None:
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing columns in {source}: {missing}")


def _read_feature_frame(path: Path, id_column: str) -> pd.DataFrame:
    columns = pd.read_csv(path, nrows=0).columns.tolist()
    required = [id_column, "Time step"]
    if columns[:2] != required:
        raise ValueError(f"Unexpected leading columns in {path}: {columns[:2]}")
    dtypes: Dict[str, str] = {
        column: "float32" for column in columns if column not in required
    }
    dtypes[id_column] = "string" if id_column == "address" else "int64"
    dtypes["Time step"] = "int16"
    return pd.read_csv(path, dtype=dtypes)


def _labels_from_raw(raw_classes: np.ndarray) -> np.ndarray:
    labels = np.full(raw_classes.shape, -99, dtype=np.int64)
    labels[raw_classes == 1] = 1
    labels[raw_classes == 2] = 0
    labels[raw_classes == 3] = -1
    if np.any(labels == -99):
        values = np.unique(raw_classes[labels == -99]).tolist()
        raise ValueError(f"Unexpected raw transaction classes: {values}")
    return labels


def preprocess_ellipticpp(
    raw_root: PathLike,
    output_root: PathLike,
    first_time_step: int = 1,
    last_time_step: int = 49,
) -> Dict[str, Any]:
    """Convert official Elliptic++ CSVs into 49 sparse temporal snapshots."""

    raw_root = Path(raw_root).resolve()
    output_root = Path(output_root).resolve()
    if output_root.exists():
        raise FileExistsError(
            f"Processed output already exists; refusing to overwrite: {output_root}"
        )
    building_root = output_root.with_name(output_root.name + ".building")
    if building_root.exists():
        raise FileExistsError(
            f"A previous preprocessing directory already exists: {building_root}"
        )
    building_root.mkdir(parents=True)
    snapshot_root = building_root / "snapshots"
    snapshot_root.mkdir()

    actors_root = raw_root / "Actors Dataset"
    transactions_root = raw_root / "Transactions Dataset"
    paths = {
        "wallet_classes": actors_root / "wallets_classes.csv",
        "wallet_features": actors_root / "wallets_features.csv",
        "input_relations": actors_root / "AddrTx_edgelist.csv",
        "output_relations": actors_root / "TxAddr_edgelist.csv",
        "transaction_classes": transactions_root / "txs_classes.csv",
        "transaction_features": transactions_root / "txs_features.csv",
    }
    missing_files = [str(path) for path in paths.values() if not path.is_file()]
    if missing_files:
        raise FileNotFoundError(f"Missing Elliptic++ raw files: {missing_files}")

    transaction_frame = _read_feature_frame(paths["transaction_features"], "txId")
    transaction_feature_columns = transaction_frame.columns[2:].tolist()
    if transaction_frame["txId"].duplicated().any():
        raise ValueError("txs_features.csv contains duplicate transaction IDs.")
    transaction_ids = transaction_frame["txId"].to_numpy(dtype=np.int64, copy=True)
    transaction_times = transaction_frame["Time step"].to_numpy(dtype=np.int16, copy=True)
    if transaction_times.min() != first_time_step or transaction_times.max() != last_time_step:
        raise ValueError("Transaction time range does not match the requested range.")
    if np.unique(transaction_times).size != last_time_step - first_time_step + 1:
        raise ValueError("Transaction time steps are not continuous.")
    transaction_features = transaction_frame[transaction_feature_columns].to_numpy(
        dtype=np.float32, copy=True
    )
    transaction_feature_mask = np.isfinite(transaction_features)
    nonfinite_transaction_values = int((~transaction_feature_mask).sum())
    transactions_with_nonfinite_features = int(
        (~transaction_feature_mask).any(axis=1).sum()
    )
    np.nan_to_num(
        transaction_features,
        copy=False,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    del transaction_frame

    transaction_class_frame = pd.read_csv(
        paths["transaction_classes"], dtype={"txId": "int64", "class": "int8"}
    )
    _require_columns(transaction_class_frame, ["txId", "class"], paths["transaction_classes"])
    if transaction_class_frame["txId"].duplicated().any():
        raise ValueError("txs_classes.csv contains duplicate transaction IDs.")
    transaction_class_index = pd.Index(transaction_class_frame["txId"])
    class_positions = transaction_class_index.get_indexer(transaction_ids)
    if np.any(class_positions < 0):
        raise ValueError("Some transaction feature rows have no class record.")
    raw_transaction_classes = transaction_class_frame["class"].to_numpy()[class_positions]
    transaction_labels = _labels_from_raw(raw_transaction_classes)

    wallet_class_frame = pd.read_csv(
        paths["wallet_classes"], dtype={"address": "string", "class": "int8"}
    )
    _require_columns(wallet_class_frame, ["address", "class"], paths["wallet_classes"])
    if wallet_class_frame["address"].duplicated().any():
        raise ValueError("wallets_classes.csv contains duplicate addresses.")
    address_index = pd.Index(wallet_class_frame["address"])

    input_frame = pd.read_csv(
        paths["input_relations"], dtype={"input_address": "string", "txId": "int64"}
    )
    output_frame = pd.read_csv(
        paths["output_relations"], dtype={"txId": "int64", "output_address": "string"}
    )
    _require_columns(input_frame, ["input_address", "txId"], paths["input_relations"])
    _require_columns(output_frame, ["txId", "output_address"], paths["output_relations"])
    if input_frame.duplicated(["input_address", "txId"]).any():
        raise ValueError("Duplicate input address-transaction memberships were found.")
    if output_frame.duplicated(["txId", "output_address"]).any():
        raise ValueError("Duplicate output transaction-address memberships were found.")

    transaction_index = pd.Index(transaction_ids)
    input_transaction_positions = transaction_index.get_indexer(input_frame["txId"])
    output_transaction_positions = transaction_index.get_indexer(output_frame["txId"])
    input_address_ids = address_index.get_indexer(input_frame["input_address"]).astype(np.int64)
    output_address_ids = address_index.get_indexer(output_frame["output_address"]).astype(np.int64)
    if np.any(input_transaction_positions < 0) or np.any(output_transaction_positions < 0):
        raise ValueError("A role relation refers to an unknown transaction.")
    if np.any(input_address_ids < 0) or np.any(output_address_ids < 0):
        raise ValueError("A role relation refers to an unknown wallet address.")
    input_times = transaction_times[input_transaction_positions]
    output_times = transaction_times[output_transaction_positions]
    del input_frame, output_frame

    wallet_feature_frame = _read_feature_frame(paths["wallet_features"], "address")
    address_feature_columns = wallet_feature_frame.columns[2:].tolist()
    duplicate_key_mask = wallet_feature_frame.duplicated(["address", "Time step"])
    duplicate_key_rows = int(duplicate_key_mask.sum())
    if duplicate_key_rows:
        exact_unique_count = len(wallet_feature_frame.drop_duplicates())
        key_unique_count = len(
            wallet_feature_frame.drop_duplicates(["address", "Time step"])
        )
        if exact_unique_count != key_unique_count:
            raise ValueError(
                "Conflicting wallet features exist for the same address and time step."
            )
        wallet_feature_frame = wallet_feature_frame.drop_duplicates(
            ["address", "Time step"], keep="first"
        )
    wallet_feature_address_ids = address_index.get_indexer(
        wallet_feature_frame["address"]
    ).astype(np.int64)
    if np.any(wallet_feature_address_ids < 0):
        raise ValueError("A wallet feature row refers to an unknown address.")
    wallet_feature_times = wallet_feature_frame["Time step"].to_numpy(
        dtype=np.int16, copy=True
    )
    wallet_features = wallet_feature_frame[address_feature_columns].to_numpy(
        dtype=np.float32, copy=True
    )
    del wallet_feature_frame

    manifest: List[Dict[str, Any]] = []
    total_active_addresses = 0
    for time_id in range(first_time_step, last_time_step + 1):
        transaction_positions = np.flatnonzero(transaction_times == time_id)
        input_relation_mask = input_times == time_id
        output_relation_mask = output_times == time_id
        input_global_addresses = input_address_ids[input_relation_mask]
        output_global_addresses = output_address_ids[output_relation_mask]
        global_address_ids = np.unique(
            np.concatenate([input_global_addresses, output_global_addresses])
        )

        global_to_local_transaction = np.full(
            transaction_ids.size, -1, dtype=np.int64
        )
        global_to_local_transaction[transaction_positions] = np.arange(
            transaction_positions.size, dtype=np.int64
        )
        input_local_transactions = global_to_local_transaction[
            input_transaction_positions[input_relation_mask]
        ]
        output_local_transactions = global_to_local_transaction[
            output_transaction_positions[output_relation_mask]
        ]
        if np.any(input_local_transactions < 0) or np.any(output_local_transactions < 0):
            raise ValueError(f"Time alignment failed at time step {time_id}.")

        input_local_addresses = np.searchsorted(
            global_address_ids, input_global_addresses
        ).astype(np.int64)
        output_local_addresses = np.searchsorted(
            global_address_ids, output_global_addresses
        ).astype(np.int64)

        feature_mask = wallet_feature_times == time_id
        feature_address_ids = wallet_feature_address_ids[feature_mask]
        feature_values = wallet_features[feature_mask]
        feature_order = np.argsort(feature_address_ids)
        feature_address_ids = feature_address_ids[feature_order]
        feature_values = feature_values[feature_order]
        feature_positions = np.searchsorted(feature_address_ids, global_address_ids)
        features_present = (
            (feature_positions < feature_address_ids.size)
            & (feature_address_ids[np.minimum(feature_positions, feature_address_ids.size - 1)] == global_address_ids)
        )
        if not np.all(features_present):
            missing_count = int((~features_present).sum())
            raise ValueError(
                f"{missing_count} active addresses lack features at time step {time_id}."
            )
        local_address_features = feature_values[feature_positions]
        labels = transaction_labels[transaction_positions]

        input_index = np.vstack(
            [input_local_addresses, input_local_transactions]
        ).astype(np.int64, copy=False)
        output_index = np.vstack(
            [output_local_addresses, output_local_transactions]
        ).astype(np.int64, copy=False)
        payload = {
            "format_version": FORMAT_VERSION,
            "time_id": time_id,
            "global_address_ids": torch.from_numpy(global_address_ids.astype(np.int64)),
            "address_features": torch.from_numpy(
                np.ascontiguousarray(local_address_features, dtype=np.float32)
            ),
            "address_feature_mask": torch.ones(global_address_ids.size, dtype=torch.bool),
            "transaction_ids": torch.from_numpy(
                transaction_ids[transaction_positions].astype(np.int64, copy=True)
            ),
            "transaction_features": torch.from_numpy(
                np.ascontiguousarray(
                    transaction_features[transaction_positions], dtype=np.float32
                )
            ),
            "transaction_feature_mask": torch.from_numpy(
                np.ascontiguousarray(
                    transaction_feature_mask[transaction_positions], dtype=np.bool_
                )
            ),
            "input_index": torch.from_numpy(input_index),
            "output_index": torch.from_numpy(output_index),
            "labels": torch.from_numpy(labels.astype(np.int64, copy=True)),
            "labeled_mask": torch.from_numpy(labels >= 0),
        }
        snapshot = EllipticPPHypergraphSnapshot.from_payload(payload)
        snapshot.validate()
        relative_file = Path("snapshots") / f"time_{time_id:02d}.pt"
        torch.save(payload, building_root / relative_file)

        zero_input = int(
            transaction_positions.size - np.unique(input_local_transactions).size
        )
        zero_output = int(
            transaction_positions.size - np.unique(output_local_transactions).size
        )
        manifest.append(
            {
                "time_id": time_id,
                "file": relative_file.as_posix(),
                "num_addresses": int(global_address_ids.size),
                "num_transactions": int(transaction_positions.size),
                "num_input_relations": int(input_index.shape[1]),
                "num_output_relations": int(output_index.shape[1]),
                "illicit": int((labels == 1).sum()),
                "licit": int((labels == 0).sum()),
                "unknown": int((labels == -1).sum()),
                "transactions_without_input": zero_input,
                "transactions_without_output": zero_output,
            }
        )
        total_active_addresses += int(global_address_ids.size)

    address_map = pd.DataFrame(
        {
            "global_address_id": np.arange(len(wallet_class_frame), dtype=np.int64),
            "address": wallet_class_frame["address"],
            "raw_class": wallet_class_frame["class"],
        }
    )
    address_map.to_csv(
        building_root / "address_index.csv.gz",
        index=False,
        compression="gzip",
    )
    transaction_map = pd.DataFrame(
        {
            "transaction_id": transaction_ids,
            "time_id": transaction_times,
            "raw_class": raw_transaction_classes,
            "target_label": transaction_labels,
        }
    )
    transaction_map.to_csv(
        building_root / "transaction_index.csv.gz",
        index=False,
        compression="gzip",
    )

    metadata: Dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "dataset": "Elliptic++",
        "source": "https://github.com/git-disl/EllipticPlusPlus",
        "raw_root": str(raw_root),
        "construction": {
            "node_type": "wallet_address",
            "hyperedge_type": "transaction",
            "input_role": "input_address_to_transaction",
            "output_role": "transaction_to_output_address",
            "incidence_format": "sparse_coo",
            "unknown_transactions_kept_in_structure": True,
            "unknown_transactions_excluded_from_supervision": True,
            "duplicate_address_time_policy": "drop_exact_duplicates",
            "nonfinite_transaction_feature_policy": "zero_fill_and_store_mask",
            "time_feature_excluded_from_model_features": True,
        },
        "label_mapping": {"raw_1": 1, "raw_2": 0, "raw_3": -1},
        "feature_dimensions": {
            "address": len(address_feature_columns),
            "transaction": len(transaction_feature_columns),
        },
        "address_feature_columns": address_feature_columns,
        "transaction_feature_columns": transaction_feature_columns,
        "totals": {
            "global_addresses": int(len(wallet_class_frame)),
            "address_time_nodes": int(total_active_addresses),
            "duplicate_address_time_rows_removed": duplicate_key_rows,
            "transactions": int(transaction_ids.size),
            "input_relations": int(input_address_ids.size),
            "output_relations": int(output_address_ids.size),
            "illicit_transactions": int((transaction_labels == 1).sum()),
            "licit_transactions": int((transaction_labels == 0).sum()),
            "unknown_transactions": int((transaction_labels == -1).sum()),
            "nonfinite_transaction_feature_values_replaced": nonfinite_transaction_values,
            "transactions_with_nonfinite_features": transactions_with_nonfinite_features,
        },
        "snapshots": manifest,
    }
    with (building_root / "metadata.json").open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, ensure_ascii=False, indent=2)

    building_root.replace(output_root)
    return metadata
