from __future__ import annotations

import argparse
import copy
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import joblib
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
)
from torch.nn import functional as F
from torch_geometric.utils import to_undirected


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tthgnn_erl.baselines import (  # noqa: E402
    BFHGNReimplemented,
    EvolveGCNOBaseline,
    FeatureStatistics,
    GATResNetBaseline,
    HeteroSAGEBaseline,
    HGTBaseline,
    StaticHGNN,
    TransactionGraphBaseline,
    fit_feature_statistics,
)
from tthgnn_erl.ellipticpp import (  # noqa: E402
    EllipticPPHypergraphSnapshot,
    EllipticPPSnapshotDataset,
)


MODEL_NAMES = (
    "lr",
    "xgboost",
    "rf",
    "gcn",
    "gat",
    "hgnn",
    "evolvegcn_o",
    "hgt",
    "graph_transformer",
    "gat_resnet",
    "heterosage",
    "bf_hgn",
)
CLASSICAL_MODELS = {"lr", "xgboost", "rf"}
GRAPH_MODELS = {
    "gcn",
    "gat",
    "evolvegcn_o",
    "hgt",
    "graph_transformer",
    "gat_resnet",
    "heterosage",
    "bf_hgn",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the Elliptic++ main-experiment baselines"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "main_baselines.yaml",
    )
    parser.add_argument("--model", choices=("all",) + MODEL_NAMES, default="all")
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def resolve_project_path(value: Union[str, Path]) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def expand_interval(interval: Sequence[int]) -> List[int]:
    if len(interval) != 2:
        raise ValueError("A chronological split must contain [first, last].")
    return list(range(int(interval[0]), int(interval[1]) + 1))


def preload_snapshots(root: Path) -> Dict[int, EllipticPPHypergraphSnapshot]:
    dataset = EllipticPPSnapshotDataset(root, validate=True)
    return {snapshot.time_id: snapshot for snapshot in dataset}


def load_transaction_edges(
    edge_path: Path,
    snapshots: Dict[int, EllipticPPHypergraphSnapshot],
) -> Dict[int, torch.Tensor]:
    locations: Dict[int, Tuple[int, int]] = {}
    for time_id, snapshot in snapshots.items():
        for local_index, transaction_id in enumerate(snapshot.transaction_ids.tolist()):
            locations[int(transaction_id)] = (time_id, local_index)

    edges_by_time: Dict[int, List[Tuple[int, int]]] = {
        time_id: [] for time_id in snapshots
    }
    frame = pd.read_csv(edge_path, usecols=["txId1", "txId2"])
    for source_id, target_id in frame.itertuples(index=False, name=None):
        source = locations.get(int(source_id))
        target = locations.get(int(target_id))
        if source is None or target is None:
            raise ValueError("A transaction edge references an unknown transaction ID.")
        if source[0] != target[0]:
            raise ValueError("A transaction edge crosses Elliptic++ time steps.")
        edges_by_time[source[0]].append((source[1], target[1]))

    result: Dict[int, torch.Tensor] = {}
    for time_id, edges in edges_by_time.items():
        if edges:
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
            edge_index = to_undirected(
                edge_index, num_nodes=snapshots[time_id].num_transactions
            )
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        result[time_id] = edge_index
    return result


def fit_training_statistics(
    snapshots: Dict[int, EllipticPPHypergraphSnapshot],
    train_times: Sequence[int],
) -> Tuple[FeatureStatistics, FeatureStatistics]:
    address_statistics = fit_feature_statistics(
        [snapshots[time_id].address_features for time_id in train_times]
    )
    transaction_statistics = fit_feature_statistics(
        [snapshots[time_id].transaction_features for time_id in train_times],
        [snapshots[time_id].transaction_feature_mask for time_id in train_times],
    )
    return address_statistics, transaction_statistics


def build_model(
    name: str,
    model_config: Dict[str, Any],
    address_statistics: FeatureStatistics,
    transaction_statistics: FeatureStatistics,
) -> torch.nn.Module:
    common = {
        "hidden_dim": int(model_config["hidden_dim"]),
        "dropout": float(model_config["dropout"]),
        "propagation_layers": int(model_config["propagation_layers"]),
    }
    if name == "hgnn":
        return StaticHGNN(
            address_statistics=address_statistics,
            transaction_statistics=transaction_statistics,
            **common,
        )
    if name in {"gcn", "gat", "graph_transformer"}:
        return TransactionGraphBaseline(
            transaction_statistics=transaction_statistics,
            kind=name,
            attention_heads=int(model_config["attention_heads"]),
            **common,
        )
    if name == "hgt":
        return HGTBaseline(
            address_statistics=address_statistics,
            transaction_statistics=transaction_statistics,
            attention_heads=int(model_config["attention_heads"]),
            **common,
        )
    if name == "gat_resnet":
        return GATResNetBaseline(
            transaction_statistics=transaction_statistics,
            attention_heads=int(model_config["attention_heads"]),
            **common,
        )
    if name == "heterosage":
        return HeteroSAGEBaseline(
            address_statistics=address_statistics,
            transaction_statistics=transaction_statistics,
            **common,
        )
    if name == "evolvegcn_o":
        return EvolveGCNOBaseline(
            transaction_statistics=transaction_statistics,
            **common,
        )
    if name == "bf_hgn":
        bf_config = model_config.get("bf_hgn", {})
        return BFHGNReimplemented(
            address_statistics=address_statistics,
            transaction_statistics=transaction_statistics,
            hidden_dim=int(model_config["hidden_dim"]),
            dropout=float(model_config["dropout"]),
            pseudo_anomaly_ratio=float(bf_config.get("pseudo_anomaly_ratio", 0.2)),
            loss_mix=float(bf_config.get("loss_mix", 0.3)),
            affinity_margin=float(bf_config.get("affinity_margin", 0.7)),
            noise_mean=float(bf_config.get("noise_mean", 0.015)),
            noise_std=float(bf_config.get("noise_std", 0.005)),
            rgcn_layers=int(bf_config.get("rgcn_layers", 2)),
            supervision_mode=str(bf_config.get("supervision_mode", "paper_one_class")),
        )
    raise ValueError(f"Unknown neural baseline: {name}")


def labeled_counts(
    snapshots: Dict[int, EllipticPPHypergraphSnapshot], times: Iterable[int]
) -> Tuple[int, int]:
    positives = 0
    negatives = 0
    for time_id in times:
        labels = snapshots[time_id].labels
        positives += int((labels == 1).sum())
        negatives += int((labels == 0).sum())
    return positives, negatives


def forward_static_model(
    name: str,
    model: torch.nn.Module,
    snapshot: EllipticPPHypergraphSnapshot,
    transaction_edge_index: Optional[torch.Tensor],
) -> torch.Tensor:
    if name == "hgnn":
        return model(snapshot)
    if transaction_edge_index is None:
        raise ValueError(f"{name} requires a transaction graph.")
    return model(snapshot, transaction_edge_index)


def train_static_epoch(
    name: str,
    model: torch.nn.Module,
    snapshots: Dict[int, EllipticPPHypergraphSnapshot],
    transaction_edges: Dict[int, torch.Tensor],
    train_times: Sequence[int],
    optimizer: torch.optim.Optimizer,
    positive_weight: torch.Tensor,
    device: torch.device,
    snapshots_per_step: int,
    gradient_clip_norm: float,
    seed: int,
) -> float:
    model.train()
    order = np.asarray(train_times, dtype=np.int64)
    np.random.default_rng(seed).shuffle(order)
    total_weighted_loss = 0.0
    total_labeled = 0

    for start in range(0, order.size, snapshots_per_step):
        group = order[start : start + snapshots_per_step].tolist()
        group_labeled = sum(
            int(snapshots[int(time_id)].labeled_mask.sum()) for time_id in group
        )
        optimizer.zero_grad(set_to_none=True)
        for time_id_value in group:
            time_id = int(time_id_value)
            snapshot = snapshots[time_id].to(device)
            edge_index = (
                transaction_edges[time_id].to(device)
                if name in GRAPH_MODELS
                else None
            )
            logits = forward_static_model(name, model, snapshot, edge_index)
            mask = snapshot.labeled_mask
            loss_sum = F.binary_cross_entropy_with_logits(
                logits[mask],
                snapshot.labels[mask].to(dtype=torch.float32),
                pos_weight=positive_weight,
                reduction="sum",
            )
            (loss_sum / group_labeled).backward()
            total_weighted_loss += float(loss_sum.detach())
            total_labeled += int(mask.sum())
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
        optimizer.step()
    return total_weighted_loss / max(total_labeled, 1)


def train_evolve_epoch(
    model: EvolveGCNOBaseline,
    snapshots: Dict[int, EllipticPPHypergraphSnapshot],
    transaction_edges: Dict[int, torch.Tensor],
    train_times: Sequence[int],
    optimizer: torch.optim.Optimizer,
    positive_weight: torch.Tensor,
    device: torch.device,
    snapshots_per_step: int,
    gradient_clip_norm: float,
) -> float:
    model.train()
    recurrent_weights = model.initial_state()
    total_weighted_loss = 0.0
    total_labeled = 0
    for start in range(0, len(train_times), snapshots_per_step):
        group = list(train_times[start : start + snapshots_per_step])
        group_labeled = sum(int(snapshots[t].labeled_mask.sum()) for t in group)
        optimizer.zero_grad(set_to_none=True)
        group_objective = torch.zeros((), device=device)
        for time_id in group:
            snapshot = snapshots[time_id].to(device)
            logits, recurrent_weights = model.forward_snapshot(
                snapshot,
                transaction_edges[time_id].to(device),
                recurrent_weights,
            )
            mask = snapshot.labeled_mask
            loss_sum = F.binary_cross_entropy_with_logits(
                logits[mask],
                snapshot.labels[mask].to(dtype=torch.float32),
                pos_weight=positive_weight,
                reduction="sum",
            )
            group_objective = group_objective + loss_sum / group_labeled
            total_weighted_loss += float(loss_sum.detach())
            total_labeled += int(mask.sum())
        group_objective.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
        optimizer.step()
        recurrent_weights = [weight.detach() for weight in recurrent_weights]
    return total_weighted_loss / max(total_labeled, 1)


def train_bf_hgn_epoch(
    model: BFHGNReimplemented,
    snapshots: Dict[int, EllipticPPHypergraphSnapshot],
    transaction_edges: Dict[int, torch.Tensor],
    train_times: Sequence[int],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    window_size: int,
    windows_per_epoch: int,
    gradient_clip_norm: float,
    seed: int,
) -> Tuple[float, Dict[str, float]]:
    """Train BF-HGN on random contiguous windows, as specified in the paper."""

    if window_size < 1 or window_size > len(train_times):
        raise ValueError("BF-HGN window_size is incompatible with the training split.")
    possible_starts = len(train_times) - window_size + 1
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, possible_starts, size=windows_per_epoch)
    totals = {"loss": 0.0, "bce": 0.0, "aa": 0.0, "afsr": 0.0}
    model.train()
    for start_value in starts.tolist():
        window_times = list(
            train_times[int(start_value) : int(start_value) + window_size]
        )
        window_snapshots = [snapshots[t].to(device) for t in window_times]
        window_edges = [transaction_edges[t].to(device) for t in window_times]
        optimizer.zero_grad(set_to_none=True)
        output = model.forward_sequence(window_snapshots, window_edges)
        loss, components = model.class_balanced_objective(
            output, window_snapshots, window_edges
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
        optimizer.step()
        totals["loss"] += float(loss.detach())
        for key in ("bce", "aa", "afsr"):
            totals[key] += components[key]
    denominator = max(windows_per_epoch, 1)
    means = {key: value / denominator for key, value in totals.items()}
    return means["loss"], {key: means[key] for key in ("bce", "aa", "afsr")}


def predict_static(
    name: str,
    model: torch.nn.Module,
    snapshots: Dict[int, EllipticPPHypergraphSnapshot],
    transaction_edges: Dict[int, torch.Tensor],
    times: Sequence[int],
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, float]:
    model.eval()
    labels: List[torch.Tensor] = []
    probabilities: List[torch.Tensor] = []
    total_loss = 0.0
    total_labeled = 0
    with torch.no_grad():
        for time_id in times:
            snapshot = snapshots[time_id].to(device)
            edge_index = (
                transaction_edges[time_id].to(device)
                if name in GRAPH_MODELS
                else None
            )
            logits = forward_static_model(name, model, snapshot, edge_index)
            mask = snapshot.labeled_mask
            selected_labels = snapshot.labels[mask]
            selected_logits = logits[mask]
            total_loss += float(
                F.binary_cross_entropy_with_logits(
                    selected_logits,
                    selected_labels.to(dtype=torch.float32),
                    reduction="sum",
                )
            )
            total_labeled += int(mask.sum())
            labels.append(selected_labels.cpu())
            probabilities.append(torch.sigmoid(selected_logits).cpu())
    return (
        torch.cat(labels).numpy(),
        torch.cat(probabilities).numpy(),
        total_loss / max(total_labeled, 1),
    )


def predict_evolve(
    model: EvolveGCNOBaseline,
    snapshots: Dict[int, EllipticPPHypergraphSnapshot],
    transaction_edges: Dict[int, torch.Tensor],
    replay_times: Sequence[int],
    prediction_times: Sequence[int],
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, float]:
    model.eval()
    prediction_set = set(prediction_times)
    recurrent_weights = model.initial_state()
    labels: List[torch.Tensor] = []
    probabilities: List[torch.Tensor] = []
    total_loss = 0.0
    total_labeled = 0
    with torch.no_grad():
        for time_id in replay_times:
            snapshot = snapshots[time_id].to(device)
            logits, recurrent_weights = model.forward_snapshot(
                snapshot,
                transaction_edges[time_id].to(device),
                recurrent_weights,
            )
            if time_id not in prediction_set:
                continue
            mask = snapshot.labeled_mask
            selected_labels = snapshot.labels[mask]
            selected_logits = logits[mask]
            total_loss += float(
                F.binary_cross_entropy_with_logits(
                    selected_logits,
                    selected_labels.to(dtype=torch.float32),
                    reduction="sum",
                )
            )
            total_labeled += int(mask.sum())
            labels.append(selected_labels.cpu())
            probabilities.append(torch.sigmoid(selected_logits).cpu())
    return (
        torch.cat(labels).numpy(),
        torch.cat(probabilities).numpy(),
        total_loss / max(total_labeled, 1),
    )


def predict_bf_hgn(
    model: BFHGNReimplemented,
    snapshots: Dict[int, EllipticPPHypergraphSnapshot],
    transaction_edges: Dict[int, torch.Tensor],
    replay_times: Sequence[int],
    prediction_times: Sequence[int],
    device: torch.device,
    window_size: int,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Causal rolling prediction: every target sees only itself and its past."""

    ordered_replay = list(replay_times)
    positions = {time_id: index for index, time_id in enumerate(ordered_replay)}
    missing = sorted(set(prediction_times) - set(positions))
    if missing:
        raise ValueError(f"BF-HGN prediction times are absent from replay: {missing}")
    model.eval()
    labels: List[torch.Tensor] = []
    probabilities: List[torch.Tensor] = []
    total_loss = 0.0
    total_labeled = 0
    with torch.no_grad():
        for time_id in prediction_times:
            target_position = positions[time_id]
            first_position = max(0, target_position - window_size + 1)
            window_times = ordered_replay[first_position : target_position + 1]
            window_snapshots = [snapshots[t].to(device) for t in window_times]
            window_edges = [transaction_edges[t].to(device) for t in window_times]
            output = model.forward_sequence(window_snapshots, window_edges)
            snapshot = window_snapshots[-1]
            selected_logits = output.logits[-1][snapshot.labeled_mask]
            selected_labels = snapshot.labels[snapshot.labeled_mask]
            total_loss += float(
                F.binary_cross_entropy_with_logits(
                    selected_logits,
                    selected_labels.to(dtype=torch.float32),
                    reduction="sum",
                )
            )
            total_labeled += int(snapshot.labeled_mask.sum())
            labels.append(selected_labels.cpu())
            probabilities.append(torch.sigmoid(selected_logits).cpu())
    return (
        torch.cat(labels).numpy(),
        torch.cat(probabilities).numpy(),
        total_loss / max(total_labeled, 1),
    )


def best_f1_threshold(labels: np.ndarray, probabilities: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(labels, probabilities)
    if thresholds.size == 0:
        return 0.5
    f1 = 2.0 * precision[:-1] * recall[:-1] / (
        precision[:-1] + recall[:-1] + 1e-12
    )
    best_value = np.nanmax(f1)
    candidates = np.flatnonzero(np.isclose(f1, best_value))
    candidate_thresholds = thresholds[candidates]
    return float(candidate_thresholds[np.argmin(np.abs(candidate_thresholds - 0.5))])


def classification_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> Dict[str, Union[float, int]]:
    predictions = (probabilities >= threshold).astype(np.int64)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="binary",
        pos_label=1,
        zero_division=0,
    )
    return {
        "average_precision": float(average_precision_score(labels, probabilities)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "precision": float(precision),
        "accuracy": float((predictions == labels).mean()),
        "recall": float(recall),
        "f1": float(f1),
        "threshold": float(threshold),
        "positives": int((labels == 1).sum()),
        "negatives": int((labels == 0).sum()),
    }


def standardized_labeled_arrays(
    snapshots: Dict[int, EllipticPPHypergraphSnapshot],
    times: Sequence[int],
    statistics: FeatureStatistics,
) -> Tuple[np.ndarray, np.ndarray]:
    feature_parts: List[torch.Tensor] = []
    label_parts: List[torch.Tensor] = []
    for time_id in times:
        snapshot = snapshots[time_id]
        features = (snapshot.transaction_features - statistics.mean) / statistics.std
        features = torch.where(
            snapshot.transaction_feature_mask, features, torch.zeros_like(features)
        )
        feature_parts.append(features[snapshot.labeled_mask])
        label_parts.append(snapshot.labels[snapshot.labeled_mask])
    return torch.cat(feature_parts).numpy(), torch.cat(label_parts).numpy()


def train_classical_model(
    name: str,
    config: Dict[str, Any],
    snapshots: Dict[int, EllipticPPHypergraphSnapshot],
    train_times: Sequence[int],
    validation_times: Sequence[int],
    test_times: Sequence[int],
    transaction_statistics: FeatureStatistics,
    output_root: Path,
) -> Dict[str, Any]:
    seed = int(config["seed"])
    train_x, train_y = standardized_labeled_arrays(
        snapshots, train_times, transaction_statistics
    )
    validation_x, validation_y = standardized_labeled_arrays(
        snapshots, validation_times, transaction_statistics
    )
    test_x, test_y = standardized_labeled_arrays(
        snapshots, test_times, transaction_statistics
    )
    start_time = time.perf_counter()
    if name == "lr":
        model: Any = LogisticRegression(
            C=float(config["classical"]["lr_c"]),
            class_weight="balanced",
            max_iter=int(config["classical"]["lr_max_iter"]),
            solver="lbfgs",
            random_state=seed,
        )
        model.fit(train_x, train_y)
        best_epoch: Optional[int] = 1
        parameters: Optional[int] = int(model.coef_.size + model.intercept_.size)
    elif name == "rf":
        model = RandomForestClassifier(
            n_estimators=int(config["classical"]["rf_n_estimators"]),
            max_depth=(
                None
                if config["classical"].get("rf_max_depth") is None
                else int(config["classical"]["rf_max_depth"])
            ),
            min_samples_leaf=int(config["classical"]["rf_min_samples_leaf"]),
            max_features=config["classical"]["rf_max_features"],
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=-1,
        )
        model.fit(train_x, train_y)
        best_epoch = 1
        parameters = None
    else:
        try:
            from xgboost import XGBClassifier
        except ImportError as error:
            raise RuntimeError(
                "XGBoost is not installed in the active environment."
            ) from error
        positives = int((train_y == 1).sum())
        negatives = int((train_y == 0).sum())
        model = XGBClassifier(
            objective="binary:logistic",
            eval_metric="aucpr",
            n_estimators=int(config["classical"]["xgb_n_estimators"]),
            max_depth=int(config["classical"]["xgb_max_depth"]),
            learning_rate=float(config["classical"]["xgb_learning_rate"]),
            subsample=float(config["classical"]["xgb_subsample"]),
            colsample_bytree=float(config["classical"]["xgb_colsample_bytree"]),
            reg_lambda=float(config["classical"]["xgb_reg_lambda"]),
            scale_pos_weight=negatives / positives,
            tree_method="hist",
            early_stopping_rounds=int(config["classical"]["xgb_early_stopping"]),
            random_state=seed,
            n_jobs=-1,
        )
        model.fit(
            train_x,
            train_y,
            eval_set=[(validation_x, validation_y)],
            verbose=False,
        )
        best_iteration = getattr(model, "best_iteration", None)
        best_epoch = int(best_iteration) + 1 if best_iteration is not None else None
        parameters = None

    validation_probabilities = model.predict_proba(validation_x)[:, 1]
    threshold = best_f1_threshold(validation_y, validation_probabilities)
    validation_metrics = classification_metrics(
        validation_y, validation_probabilities, threshold
    )
    test_probabilities = model.predict_proba(test_x)[:, 1]
    test_metrics = classification_metrics(test_y, test_probabilities, threshold)
    model_root = output_root / name
    model_root.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_root / "best.joblib")
    result: Dict[str, Any] = {
        "model": name,
        "seed": seed,
        "best_epoch": best_epoch,
        "epochs_ran": best_epoch,
        "parameters": parameters,
        "duration_seconds": time.perf_counter() - start_time,
        "validation": validation_metrics,
        "test": test_metrics,
    }
    with (model_root / "metrics.json").open("w", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    return result


def train_neural_model(
    name: str,
    config: Dict[str, Any],
    snapshots: Dict[int, EllipticPPHypergraphSnapshot],
    transaction_edges: Dict[int, torch.Tensor],
    train_times: Sequence[int],
    validation_times: Sequence[int],
    test_times: Sequence[int],
    address_statistics: FeatureStatistics,
    transaction_statistics: FeatureStatistics,
    device: torch.device,
    output_root: Path,
    max_epochs_override: Optional[int],
) -> Dict[str, Any]:
    seed = int(config["seed"])
    set_seed(seed)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model = build_model(
        name,
        config["model"],
        address_statistics,
        transaction_statistics,
    ).to(device)
    training = config["training"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    positives, negatives = labeled_counts(snapshots, train_times)
    positive_weight = torch.tensor(negatives / positives, device=device)
    bf_training = config["model"].get("bf_hgn", {}) if name == "bf_hgn" else {}
    max_epochs = (
        int(max_epochs_override)
        if max_epochs_override is not None
        else int(bf_training.get("max_epochs", training["max_epochs"]))
    )
    patience = int(bf_training.get("patience", training["patience"]))
    min_delta = float(training["min_delta"])
    history: List[Dict[str, Any]] = []
    best_score = -float("inf")
    best_epoch = 0
    best_state = None
    epochs_without_improvement = 0
    start_time = time.perf_counter()

    for epoch in range(1, max_epochs + 1):
        if name == "bf_hgn":
            bf_config = config["model"].get("bf_hgn", {})
            train_loss, bf_components = train_bf_hgn_epoch(
                model=model,
                snapshots=snapshots,
                transaction_edges=transaction_edges,
                train_times=train_times,
                optimizer=optimizer,
                device=device,
                window_size=int(bf_config.get("window_size", 5)),
                windows_per_epoch=int(bf_config.get("windows_per_epoch", 10)),
                gradient_clip_norm=float(training["gradient_clip_norm"]),
                seed=seed + epoch,
            )
            validation_labels, validation_probabilities, validation_loss = predict_bf_hgn(
                model=model,
                snapshots=snapshots,
                transaction_edges=transaction_edges,
                replay_times=list(train_times) + list(validation_times),
                prediction_times=validation_times,
                device=device,
                window_size=int(bf_config.get("window_size", 5)),
            )
        elif name == "evolvegcn_o":
            train_loss = train_evolve_epoch(
                model=model,
                snapshots=snapshots,
                transaction_edges=transaction_edges,
                train_times=train_times,
                optimizer=optimizer,
                positive_weight=positive_weight,
                device=device,
                snapshots_per_step=int(training["snapshots_per_step"]),
                gradient_clip_norm=float(training["gradient_clip_norm"]),
            )
            validation_labels, validation_probabilities, validation_loss = predict_evolve(
                model=model,
                snapshots=snapshots,
                transaction_edges=transaction_edges,
                replay_times=list(train_times) + list(validation_times),
                prediction_times=validation_times,
                device=device,
            )
        else:
            train_loss = train_static_epoch(
                name=name,
                model=model,
                snapshots=snapshots,
                transaction_edges=transaction_edges,
                train_times=train_times,
                optimizer=optimizer,
                positive_weight=positive_weight,
                device=device,
                snapshots_per_step=int(training["snapshots_per_step"]),
                gradient_clip_norm=float(training["gradient_clip_norm"]),
                seed=seed + epoch,
            )
            validation_labels, validation_probabilities, validation_loss = predict_static(
                name,
                model,
                snapshots,
                transaction_edges,
                validation_times,
                device,
            )
        validation_ap = float(
            average_precision_score(validation_labels, validation_probabilities)
        )
        history.append(
            {
                "epoch": epoch,
                "train_weighted_loss": train_loss,
                "validation_loss": validation_loss,
                "validation_average_precision": validation_ap,
                **({f"train_{key}": value for key, value in bf_components.items()} if name == "bf_hgn" else {}),
            }
        )
        print(
            f"[{name}] epoch={epoch:03d} train_loss={train_loss:.6f} "
            f"val_loss={validation_loss:.6f} val_ap={validation_ap:.6f}",
            flush=True,
        )
        if validation_ap > best_score + min_delta:
            best_score = validation_ap
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    if best_state is None:
        raise RuntimeError(f"No checkpoint was selected for {name}.")
    model.load_state_dict(best_state)
    if name == "bf_hgn":
        bf_config = config["model"].get("bf_hgn", {})
        validation_labels, validation_probabilities, validation_loss = predict_bf_hgn(
            model,
            snapshots,
            transaction_edges,
            list(train_times) + list(validation_times),
            validation_times,
            device,
            int(bf_config.get("window_size", 5)),
        )
    elif name == "evolvegcn_o":
        validation_labels, validation_probabilities, validation_loss = predict_evolve(
            model,
            snapshots,
            transaction_edges,
            list(train_times) + list(validation_times),
            validation_times,
            device,
        )
    else:
        validation_labels, validation_probabilities, validation_loss = predict_static(
            name,
            model,
            snapshots,
            transaction_edges,
            validation_times,
            device,
        )
    threshold = best_f1_threshold(validation_labels, validation_probabilities)
    validation_metrics = classification_metrics(
        validation_labels, validation_probabilities, threshold
    )
    validation_metrics["loss"] = validation_loss
    if name == "bf_hgn":
        test_labels, test_probabilities, test_loss = predict_bf_hgn(
            model,
            snapshots,
            transaction_edges,
            list(train_times) + list(validation_times) + list(test_times),
            test_times,
            device,
            int(bf_config.get("window_size", 5)),
        )
    elif name == "evolvegcn_o":
        test_labels, test_probabilities, test_loss = predict_evolve(
            model,
            snapshots,
            transaction_edges,
            list(train_times) + list(validation_times) + list(test_times),
            test_times,
            device,
        )
    else:
        test_labels, test_probabilities, test_loss = predict_static(
            name,
            model,
            snapshots,
            transaction_edges,
            test_times,
            device,
        )
    test_metrics = classification_metrics(test_labels, test_probabilities, threshold)
    test_metrics["loss"] = test_loss

    model_root = output_root / name
    model_root.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_name": name,
            "model_state": best_state,
            "address_statistics": address_statistics.as_dict(),
            "transaction_statistics": transaction_statistics.as_dict(),
            "best_epoch": best_epoch,
            "validation_threshold": threshold,
            "config": config,
        },
        model_root / "best.pt",
    )
    result: Dict[str, Any] = {
        "model": name,
        "seed": seed,
        "best_epoch": best_epoch,
        "epochs_ran": len(history),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "positive_weight": float(positive_weight),
        "duration_seconds": time.perf_counter() - start_time,
        "peak_gpu_memory_mb": (
            round(torch.cuda.max_memory_allocated(device) / 1024**2, 2)
            if device.type == "cuda"
            else 0.0
        ),
        "validation": validation_metrics,
        "test": test_metrics,
        "history": history,
    }
    if name == "bf_hgn":
        result.update(
            {
                "implementation": "BF-HGN (reimplemented from paper; no official code)",
                "supervision": str(bf_config.get("supervision_mode", "paper_one_class")),
                "temporal_inference": "causal_rolling_window_no_future_snapshots",
            }
        )
    with (model_root / "metrics.json").open("w", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    return result


def main() -> None:
    args = parse_args()
    with args.config.open("r", encoding="utf-8") as stream:
        config: Dict[str, Any] = yaml.safe_load(stream)
    if args.seed is not None:
        config["seed"] = int(args.seed)
    set_seed(int(config["seed"]))
    device = torch.device(str(config["device"]))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA training was requested but CUDA is unavailable.")
    processed_root = resolve_project_path(config["data"]["processed_root"])
    output_root = (
        resolve_project_path(args.output_root)
        if args.output_root is not None
        else resolve_project_path(config["output_root"])
    )
    output_root.mkdir(parents=True, exist_ok=True)
    train_times = expand_interval(config["split"]["train"])
    validation_times = expand_interval(config["split"]["validation"])
    test_times = expand_interval(config["split"]["test"])
    snapshots = preload_snapshots(processed_root)
    address_statistics, transaction_statistics = fit_training_statistics(
        snapshots, train_times
    )
    requested_models = (
        list(config["models"]) if args.model == "all" else [args.model]
    )
    if any(name in GRAPH_MODELS for name in requested_models):
        transaction_edges = load_transaction_edges(
            resolve_project_path(config["data"]["transaction_edges"]), snapshots
        )
    else:
        transaction_edges = {}

    results: List[Dict[str, Any]] = []
    for name in requested_models:
        if name in CLASSICAL_MODELS:
            result = train_classical_model(
                name=name,
                config=config,
                snapshots=snapshots,
                train_times=train_times,
                validation_times=validation_times,
                test_times=test_times,
                transaction_statistics=transaction_statistics,
                output_root=output_root,
            )
        else:
            result = train_neural_model(
                name=name,
                config=config,
                snapshots=snapshots,
                transaction_edges=transaction_edges,
                train_times=train_times,
                validation_times=validation_times,
                test_times=test_times,
                address_statistics=address_statistics,
                transaction_statistics=transaction_statistics,
                device=device,
                output_root=output_root,
                max_epochs_override=args.max_epochs,
            )
        results.append(result)
        print(
            json.dumps(
                {"model": name, "validation": result["validation"], "test": result["test"]},
                ensure_ascii=False,
                indent=2,
            ),
            flush=True,
        )

    comparison_path = output_root / "comparison.json"
    if args.model != "all" and comparison_path.is_file():
        previous = json.loads(comparison_path.read_text(encoding="utf-8"))
        if int(previous.get("seed", config["seed"])) != int(config["seed"]):
            raise ValueError("Existing comparison.json belongs to a different seed.")
        merged = {
            result["model"]: result for result in previous.get("results", [])
        }
        merged.update({result["model"]: result for result in results})
        results = [merged[name] for name in MODEL_NAMES if name in merged]

    comparison = {
        "status": "PASS",
        "seed": int(config["seed"]),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "split": config["split"],
        "results": results,
    }
    with comparison_path.open("w", encoding="utf-8") as stream:
        json.dump(comparison, stream, ensure_ascii=False, indent=2)
    print(json.dumps(comparison, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
