from __future__ import annotations

import argparse
import copy
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
from sklearn.metrics import average_precision_score, precision_recall_curve
from torch import nn
from torch.nn import functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from evaluate_temporal_stability import load_model  # noqa: E402
from train_temporal_memory import (  # noqa: E402
    best_f1_threshold,
    macro_snapshot_f1_threshold,
    metrics,
)
from tthgnn_erl.baselines import (  # noqa: E402
    FeatureStatistics,
    fit_feature_statistics,
)
from tthgnn_erl.ellipticpp import (  # noqa: E402
    EllipticPPHypergraphSnapshot,
    EllipticPPSnapshotDataset,
)


class QuantileInteractionClassifier(nn.Module):
    """Neural attribute classifier with train-only quantile discretization."""

    def __init__(
        self,
        bin_edges: torch.Tensor,
        embedding_dim: int = 16,
        hidden_dim: int = 256,
        dropout: float = 0.25,
        use_bi_interaction: bool = False,
        interaction_mode: str = "residual",
        interaction_scale_initial: float = 0.10,
    ) -> None:
        super().__init__()
        if bin_edges.ndim != 2:
            raise ValueError("bin_edges must have shape [features, bins - 1].")
        self.register_buffer("bin_edges", bin_edges.clone())
        self.feature_dim = int(bin_edges.size(0))
        self.num_bins = int(bin_edges.size(1) + 1)
        self.embedding_dim = int(embedding_dim)
        self.use_bi_interaction = bool(use_bi_interaction)
        self.interaction_mode = str(interaction_mode).lower()
        if self.interaction_mode not in {"concat", "residual"}:
            raise ValueError("interaction_mode must be 'concat' or 'residual'.")
        self.embedding = nn.Embedding(
            self.feature_dim * self.num_bins,
            self.embedding_dim,
        )
        input_dim = self.feature_dim * (self.embedding_dim + 1)
        if self.use_bi_interaction and self.interaction_mode == "concat":
            input_dim += self.embedding_dim
        middle_dim = max(64, int(hidden_dim) // 2)
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, int(hidden_dim)),
            nn.LayerNorm(int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), middle_dim),
            nn.LayerNorm(middle_dim),
            nn.GELU(),
            nn.Dropout(float(dropout) * 0.6),
            nn.Linear(middle_dim, 1),
        )
        if self.use_bi_interaction and self.interaction_mode == "residual":
            interaction_hidden = max(16, self.embedding_dim * 2)
            self.interaction_normalization = nn.LayerNorm(self.embedding_dim)
            self.interaction_head = nn.Sequential(
                nn.Linear(self.embedding_dim, interaction_hidden),
                nn.GELU(),
                nn.Dropout(float(dropout) * 0.5),
                nn.Linear(interaction_hidden, 1),
            )
            self.interaction_scale = nn.Parameter(
                torch.tensor(float(interaction_scale_initial))
            )

    def forward(self, standardized_features: torch.Tensor) -> torch.Tensor:
        feature_bins = torch.stack(
            [
                torch.bucketize(
                    standardized_features[:, feature_index].contiguous(),
                    self.bin_edges[feature_index],
                )
                for feature_index in range(self.feature_dim)
            ],
            dim=1,
        )
        offsets = (
            torch.arange(self.feature_dim, device=standardized_features.device)
            * self.num_bins
        )
        embedding_ids = feature_bins + offsets.unsqueeze(0)
        embedded = self.embedding(embedding_ids)
        model_parts = [standardized_features, embedded.flatten(start_dim=1)]
        bi_interaction = None
        if self.use_bi_interaction:
            summed = embedded.sum(dim=1)
            bi_interaction = 0.5 * (
                summed.square() - embedded.square().sum(dim=1)
            )
            pair_scale = max(
                (self.feature_dim * (self.feature_dim - 1) / 2.0) ** 0.5,
                1.0,
            )
            bi_interaction = bi_interaction / pair_scale
            if self.interaction_mode == "concat":
                model_parts.append(bi_interaction)
        logits = self.classifier(torch.cat(model_parts, dim=1)).squeeze(1)
        if self.use_bi_interaction and self.interaction_mode == "residual":
            interaction_logits = self.interaction_head(
                self.interaction_normalization(bi_interaction)
            ).squeeze(1)
            logits = logits + torch.tanh(self.interaction_scale) * interaction_logits
        return logits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a quantile-interaction attribute branch and fuse it with "
            "a saved temporal HGNN checkpoint."
        )
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/processed/ellipticpp"),
    )
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument(
        "--graph-checkpoint",
        type=Path,
        default=Path("artifacts/rcha/main/seed_{seed}/best.pt"),
    )
    parser.add_argument(
        "--attribute-checkpoint",
        type=Path,
        default=None,
        help=(
            "Optional previously trained quantile-attribute checkpoint. "
            "This branch is graph-independent and can be reused across ablations."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/quantile_fusion/seed_{seed}"),
    )
    parser.add_argument("--num-bins", type=int, default=16)
    parser.add_argument("--embedding-dim", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument(
        "--disable-bi-interaction",
        action="store_true",
        help="Disable the explicit NFM second-order bi-interaction vector.",
    )
    parser.add_argument(
        "--interaction-mode",
        choices=["concat", "residual"],
        default="residual",
        help="Inject the NFM vector by concatenation or a gated residual logit.",
    )
    parser.add_argument("--interaction-scale-initial", type=float, default=0.10)
    parser.add_argument("--learning-rate", type=float, default=8e-4)
    parser.add_argument("--weight-decay", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=768)
    parser.add_argument("--max-epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--validation-precision-floor", type=float, default=0.985)
    parser.add_argument(
        "--selection-objective",
        choices=[
            "precision_constrained_recall",
            "macro_snapshot_f1",
            "pooled_validation_f1",
        ],
        default="precision_constrained_recall",
        help=(
            "Validation-only rule used to select the fusion weight and threshold. "
            "The macro_snapshot_f1 option matches the graph-model ablation protocol; "
            "pooled_validation_f1 matches the main-baseline decision rule."
        ),
    )
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def project_path(path: Path, seed: int) -> Path:
    rendered = Path(str(path).format(seed=seed))
    return rendered if rendered.is_absolute() else PROJECT_ROOT / rendered


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def standardized_features(
    snapshot: EllipticPPHypergraphSnapshot,
    statistics: FeatureStatistics,
) -> torch.Tensor:
    values = (snapshot.transaction_features - statistics.mean) / statistics.std
    return torch.where(
        snapshot.transaction_feature_mask,
        values,
        torch.zeros_like(values),
    )


def fit_bin_edges(
    snapshots: Dict[int, EllipticPPHypergraphSnapshot],
    train_times: Sequence[int],
    statistics: FeatureStatistics,
    num_bins: int,
) -> torch.Tensor:
    if num_bins < 2:
        raise ValueError("num_bins must be at least two.")
    values = torch.cat(
        [standardized_features(snapshots[t], statistics) for t in train_times]
    )
    masks = torch.cat(
        [snapshots[t].transaction_feature_mask for t in train_times]
    )
    quantiles = torch.arange(1, num_bins, dtype=torch.float32) / num_bins
    edges = [
        torch.quantile(values[:, column][masks[:, column]], quantiles)
        for column in range(values.size(1))
    ]
    return torch.stack(edges)


def pack_labeled_transactions(
    snapshots: Dict[int, EllipticPPHypergraphSnapshot],
    times: Sequence[int],
    statistics: FeatureStatistics,
) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    features: List[torch.Tensor] = []
    labels: List[torch.Tensor] = []
    time_ids: List[int] = []
    for time_id in times:
        snapshot = snapshots[time_id]
        mask = snapshot.labeled_mask
        features.append(standardized_features(snapshot, statistics)[mask])
        labels.append(snapshot.labels[mask])
        time_ids.extend([time_id] * int(mask.sum()))
    return torch.cat(features), torch.cat(labels), np.asarray(time_ids)


def batched_logits(
    model: nn.Module,
    features: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        logits = [
            model(batch.to(device)).cpu()
            for batch in features.split(batch_size)
        ]
    return torch.cat(logits).numpy()


def train_attribute_model(
    model: QuantileInteractionClassifier,
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    validation_features: torch.Tensor,
    validation_labels: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> Tuple[Dict[str, torch.Tensor], int, List[Dict[str, float]]]:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )
    positives = int((train_labels == 1).sum())
    negatives = int((train_labels == 0).sum())
    positive_weight = torch.tensor(
        [negatives / max(positives, 1)],
        dtype=torch.float32,
        device=device,
    )
    best_state: Dict[str, torch.Tensor] | None = None
    best_epoch = 0
    best_average_precision = -float("inf")
    stale_epochs = 0
    history: List[Dict[str, float]] = []
    for epoch in range(1, int(args.max_epochs) + 1):
        model.train()
        epoch_loss = 0.0
        permutation = torch.randperm(train_labels.numel())
        for indices in permutation.split(int(args.batch_size)):
            features = train_features[indices].to(device)
            labels = train_labels[indices].to(device, dtype=torch.float32)
            loss = F.binary_cross_entropy_with_logits(
                model(features),
                labels,
                pos_weight=positive_weight,
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            epoch_loss += float(loss.detach()) * int(indices.numel())
        validation_logits = batched_logits(
            model,
            validation_features,
            int(args.batch_size),
            device,
        )
        validation_ap = float(
            average_precision_score(validation_labels.numpy(), validation_logits)
        )
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": epoch_loss / train_labels.numel(),
                "validation_average_precision": validation_ap,
            }
        )
        print(
            f"[quantile_attribute] epoch={epoch:03d} "
            f"loss={history[-1]['train_loss']:.6f} val_ap={validation_ap:.6f}",
            flush=True,
        )
        if validation_ap > best_average_precision + 1e-4:
            best_average_precision = validation_ap
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= int(args.patience):
                break
    if best_state is None:
        raise RuntimeError("No quantile-interaction checkpoint was selected.")
    return best_state, best_epoch, history


def replay_graph_logits(
    checkpoint_path: Path,
    snapshots: Dict[int, EllipticPPHypergraphSnapshot],
    global_address_count: int,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    model, checkpoint, prior_logits_by_time = load_model(checkpoint_path, device)
    memory_bank = model.initial_memory(global_address_count, device)
    validation_logits: List[torch.Tensor] = []
    test_logits: List[torch.Tensor] = []
    validation_labels: List[torch.Tensor] = []
    test_labels: List[torch.Tensor] = []
    with torch.no_grad():
        for time_id in range(1, 50):
            snapshot = snapshots[time_id].to(device)
            global_ids = snapshot.global_address_ids
            attribute_prior_logits = (
                prior_logits_by_time[time_id].to(device)
                if prior_logits_by_time is not None
                else None
            )
            logits, updated_memory = model(
                snapshot,
                memory_bank.index_select(0, global_ids),
                attribute_prior_logits=attribute_prior_logits,
            )
            memory_bank.index_copy_(0, global_ids, updated_memory)
            if 31 <= time_id <= 35:
                mask = snapshot.labeled_mask
                validation_logits.append(logits[mask].cpu())
                validation_labels.append(snapshot.labels[mask].cpu())
            elif 36 <= time_id <= 49:
                mask = snapshot.labeled_mask
                test_logits.append(logits[mask].cpu())
                test_labels.append(snapshot.labels[mask].cpu())
    return (
        torch.cat(validation_logits).numpy(),
        torch.cat(test_logits).numpy(),
        torch.cat(validation_labels).numpy(),
        torch.cat(test_labels).numpy(),
        checkpoint,
    )


def select_fusion(
    validation_labels: np.ndarray,
    validation_times: np.ndarray,
    graph_logits: np.ndarray,
    attribute_logits: np.ndarray,
    precision_floor: float,
    selection_objective: str,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    candidates: List[Dict[str, Any]] = []
    time_ids = sorted(np.unique(validation_times).tolist())
    for attribute_weight in np.linspace(0.0, 0.70, 29):
        fused_logits = (
            (1.0 - attribute_weight) * graph_logits
            + attribute_weight * attribute_logits
        )
        if selection_objective in {
            "precision_constrained_recall",
            "precision_constrained_f1",
        }:
            precision, recall, thresholds = precision_recall_curve(
                validation_labels,
                fused_logits,
            )
            feasible_indices = np.flatnonzero(precision[:-1] >= precision_floor)
            if feasible_indices.size == 0:
                continue
            if selection_objective == "precision_constrained_recall":
                feasible_score = recall[feasible_indices]
            else:
                feasible_precision = precision[:-1][feasible_indices]
                feasible_recall = recall[:-1][feasible_indices]
                feasible_score = (
                    2.0
                    * feasible_precision
                    * feasible_recall
                    / (feasible_precision + feasible_recall + 1e-12)
                )
            best_score = feasible_score.max()
            score_indices = feasible_indices[
                np.flatnonzero(np.isclose(feasible_score, best_score))
            ]
            selected_index = score_indices[np.argmax(thresholds[score_indices])]
            threshold = float(thresholds[selected_index])
            environment_f1 = [
                metrics(
                    validation_labels[validation_times == time_id],
                    fused_logits[validation_times == time_id],
                    threshold,
                )["f1"]
                for time_id in time_ids
            ]
            macro_snapshot_f1 = float(np.mean(environment_f1))
        elif selection_objective == "macro_snapshot_f1":
            threshold, macro_snapshot_f1 = macro_snapshot_f1_threshold(
                {
                    time_id: (
                        validation_labels[validation_times == time_id],
                        fused_logits[validation_times == time_id],
                    )
                    for time_id in time_ids
                },
                time_ids,
            )
        elif selection_objective == "pooled_validation_f1":
            threshold = best_f1_threshold(validation_labels, fused_logits)
            environment_f1 = [
                metrics(
                    validation_labels[validation_times == time_id],
                    fused_logits[validation_times == time_id],
                    threshold,
                )["f1"]
                for time_id in time_ids
            ]
            macro_snapshot_f1 = float(np.mean(environment_f1))
        else:
            raise ValueError(f"Unknown selection objective: {selection_objective}")
        candidate_metrics = metrics(
            validation_labels,
            fused_logits,
            threshold,
        )
        candidates.append(
            {
                "attribute_weight": float(attribute_weight),
                "graph_weight": float(1.0 - attribute_weight),
                "threshold": float(threshold),
                "macro_snapshot_f1": float(macro_snapshot_f1),
                "validation": candidate_metrics,
            }
        )
    if not candidates:
        raise RuntimeError(
            "No fusion threshold satisfies the validation precision floor."
        )
    if selection_objective == "precision_constrained_recall":
        selected = max(
            candidates,
            key=lambda candidate: (
                candidate["validation"]["recall"],
                candidate["validation"]["f1"],
                -candidate["attribute_weight"],
            ),
        )
    elif selection_objective == "precision_constrained_f1":
        selected = max(
            candidates,
            key=lambda candidate: (
                candidate["validation"]["f1"],
                candidate["validation"]["recall"],
                -candidate["attribute_weight"],
            ),
        )
    elif selection_objective == "macro_snapshot_f1":
        selected = max(
            candidates,
            key=lambda candidate: (
                candidate["macro_snapshot_f1"],
                candidate["validation"]["f1"],
                candidate["validation"]["recall"],
                -candidate["attribute_weight"],
            ),
        )
    else:
        selected = max(
            candidates,
            key=lambda candidate: (
                candidate["validation"]["f1"],
                candidate["validation"]["recall"],
                candidate["macro_snapshot_f1"],
                -candidate["attribute_weight"],
            ),
        )
    selected = dict(selected)
    selected["precision_floor_satisfied"] = (
        selected["validation"]["precision"] >= precision_floor
        if selection_objective
        in {"precision_constrained_recall", "precision_constrained_f1"}
        else None
    )
    selected["selection_objective"] = selection_objective
    return selected, candidates


def main() -> None:
    args = parse_args()
    set_seed(int(args.seed))
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    start_time = time.perf_counter()
    dataset = EllipticPPSnapshotDataset(
        project_path(args.data_root, int(args.seed)),
        validate=True,
    )
    snapshots = {snapshot.time_id: snapshot for snapshot in dataset}
    train_times = list(range(1, 31))
    validation_times = list(range(31, 36))
    test_times = list(range(36, 50))
    cached_attribute = None
    if args.attribute_checkpoint is not None:
        attribute_checkpoint_path = project_path(
            args.attribute_checkpoint, int(args.seed)
        )
        cached_attribute = torch.load(
            attribute_checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        if int(cached_attribute["seed"]) != int(args.seed):
            raise ValueError("Attribute checkpoint seed does not match --seed.")
        saved_statistics = cached_attribute["transaction_statistics"]
        statistics = FeatureStatistics(
            mean=saved_statistics["mean"],
            std=saved_statistics["std"],
            count=saved_statistics["count"],
        )
        bin_edges = cached_attribute["bin_edges"]
        attribute_model_config = cached_attribute["model_config"]
        attribute_model_config.setdefault("use_bi_interaction", False)
        attribute_model_config.setdefault(
            "interaction_mode",
            "concat" if attribute_model_config["use_bi_interaction"] else "residual",
        )
        attribute_model_config.setdefault("interaction_scale_initial", 0.10)
    else:
        statistics = fit_feature_statistics(
            [snapshots[t].transaction_features for t in train_times],
            [snapshots[t].transaction_feature_mask for t in train_times],
        )
        bin_edges = fit_bin_edges(
            snapshots,
            train_times,
            statistics,
            int(args.num_bins),
        )
        attribute_model_config = {
            "num_bins": int(args.num_bins),
            "embedding_dim": int(args.embedding_dim),
            "hidden_dim": int(args.hidden_dim),
            "dropout": float(args.dropout),
            "use_bi_interaction": not bool(args.disable_bi_interaction),
            "interaction_mode": str(args.interaction_mode),
            "interaction_scale_initial": float(args.interaction_scale_initial),
        }
    validation_features, validation_labels, validation_time_ids = (
        pack_labeled_transactions(snapshots, validation_times, statistics)
    )
    test_features, test_labels, _ = pack_labeled_transactions(
        snapshots, test_times, statistics
    )
    model = QuantileInteractionClassifier(
        bin_edges=bin_edges,
        embedding_dim=int(attribute_model_config["embedding_dim"]),
        hidden_dim=int(attribute_model_config["hidden_dim"]),
        dropout=float(attribute_model_config["dropout"]),
        use_bi_interaction=bool(
            attribute_model_config.get("use_bi_interaction", False)
        ),
        interaction_mode=str(attribute_model_config.get("interaction_mode", "residual")),
        interaction_scale_initial=float(
            attribute_model_config.get("interaction_scale_initial", 0.10)
        ),
    ).to(device)
    if cached_attribute is None:
        train_features, train_labels, _ = pack_labeled_transactions(
            snapshots, train_times, statistics
        )
        best_state, best_epoch, history = train_attribute_model(
            model,
            train_features,
            train_labels,
            validation_features,
            validation_labels,
            args,
            device,
        )
    else:
        best_state = cached_attribute["model_state"]
        best_epoch = -1
        history = []
    model.load_state_dict(best_state)
    validation_attribute_logits = batched_logits(
        model, validation_features, int(args.batch_size), device
    )
    test_attribute_logits = batched_logits(
        model, test_features, int(args.batch_size), device
    )
    if args.selection_objective == "pooled_validation_f1":
        standalone_threshold = best_f1_threshold(
            validation_labels.numpy(), validation_attribute_logits
        )
        standalone_macro_f1 = float(
            np.mean(
                [
                    metrics(
                        validation_labels.numpy()[validation_time_ids == time_id],
                        validation_attribute_logits[validation_time_ids == time_id],
                        standalone_threshold,
                    )["f1"]
                    for time_id in validation_times
                ]
            )
        )
    else:
        standalone_threshold, standalone_macro_f1 = macro_snapshot_f1_threshold(
            {
                time_id: (
                    validation_labels.numpy()[validation_time_ids == time_id],
                    validation_attribute_logits[validation_time_ids == time_id],
                )
                for time_id in validation_times
            },
            validation_times,
        )
    graph_checkpoint = project_path(args.graph_checkpoint, int(args.seed))
    (
        validation_graph_logits,
        test_graph_logits,
        graph_validation_labels,
        graph_test_labels,
        graph_checkpoint_payload,
    ) = replay_graph_logits(
        graph_checkpoint,
        snapshots,
        int(dataset.metadata["totals"]["global_addresses"]),
        device,
    )
    if not np.array_equal(validation_labels.numpy(), graph_validation_labels):
        raise RuntimeError("Validation labels are misaligned between branches.")
    if not np.array_equal(test_labels.numpy(), graph_test_labels):
        raise RuntimeError("Test labels are misaligned between branches.")
    selected, candidates = select_fusion(
        validation_labels.numpy(),
        validation_time_ids,
        validation_graph_logits,
        validation_attribute_logits,
        float(args.validation_precision_floor),
        str(args.selection_objective),
    )
    attribute_weight = float(selected["attribute_weight"])
    graph_weight = float(selected["graph_weight"])
    threshold = float(selected["threshold"])
    validation_fused_logits = (
        graph_weight * validation_graph_logits
        + attribute_weight * validation_attribute_logits
    )
    test_fused_logits = (
        graph_weight * test_graph_logits
        + attribute_weight * test_attribute_logits
    )
    validation_graph_probabilities = torch.sigmoid(
        torch.from_numpy(validation_graph_logits)
    ).numpy()
    test_graph_probabilities = torch.sigmoid(
        torch.from_numpy(test_graph_logits)
    ).numpy()
    if args.selection_objective == "pooled_validation_f1":
        graph_threshold = best_f1_threshold(
            graph_validation_labels, validation_graph_probabilities
        )
    else:
        graph_threshold = float(graph_checkpoint_payload["validation_threshold"])
    graph_model_name = str(
        graph_checkpoint_payload.get("model_name", "temporal_memory_hgnn")
    )
    results = {
        "status": "PASS",
        "method": f"{graph_model_name}_quantile_interaction_fusion",
        "seed": int(args.seed),
        "protocol": {
            "train": [1, 30],
            "validation": [31, 35],
            "test": [36, 49],
            "bin_edges_fit_on": "all transactions in training snapshots only",
            "fusion_selected_on": "validation labels only",
            "validation_precision_floor": float(
                args.validation_precision_floor
            ),
            "selection_objective": str(args.selection_objective),
            "attribute_checkpoint_reused": (
                str(project_path(args.attribute_checkpoint, int(args.seed)))
                if args.attribute_checkpoint is not None
                else None
            ),
            "uses_xgboost": False,
            "uses_test_labels_for_selection": False,
        },
        "attribute_model": {
            "num_bins": int(attribute_model_config["num_bins"]),
            "embedding_dim": int(attribute_model_config["embedding_dim"]),
            "hidden_dim": int(attribute_model_config["hidden_dim"]),
            "dropout": float(attribute_model_config["dropout"]),
            "use_bi_interaction": bool(
                attribute_model_config.get("use_bi_interaction", False)
            ),
            "interaction_mode": str(
                attribute_model_config.get("interaction_mode", "residual")
            ),
            "interaction_scale_initial": float(
                attribute_model_config.get("interaction_scale_initial", 0.10)
            ),
            "parameters": int(sum(p.numel() for p in model.parameters())),
            "best_epoch": int(best_epoch),
            "validation": {
                **metrics(
                    validation_labels.numpy(),
                    validation_attribute_logits,
                    standalone_threshold,
                ),
                "macro_snapshot_f1": float(standalone_macro_f1),
            },
            "test": metrics(
                test_labels.numpy(),
                test_attribute_logits,
                standalone_threshold,
            ),
        },
        "graph_model": {
            "checkpoint": str(graph_checkpoint),
            "validation": metrics(
                graph_validation_labels,
                validation_graph_probabilities,
                graph_threshold,
            ),
            "test": metrics(
                graph_test_labels,
                test_graph_probabilities,
                graph_threshold,
            ),
        },
        "fusion_selection": selected,
        "fusion_candidates": candidates,
        "fusion": {
            "validation": metrics(
                validation_labels.numpy(), validation_fused_logits, threshold
            ),
            "test": metrics(test_labels.numpy(), test_fused_logits, threshold),
        },
        "duration_seconds": float(time.perf_counter() - start_time),
        "history": history,
    }
    output_root = project_path(args.output_root, int(args.seed))
    output_root.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "seed": int(args.seed),
            "model_state": best_state,
            "bin_edges": bin_edges,
            "transaction_statistics": statistics.as_dict(),
            "model_config": {
                "num_bins": int(attribute_model_config["num_bins"]),
                "embedding_dim": int(attribute_model_config["embedding_dim"]),
                "hidden_dim": int(attribute_model_config["hidden_dim"]),
                "dropout": float(attribute_model_config["dropout"]),
                "use_bi_interaction": bool(
                    attribute_model_config.get("use_bi_interaction", False)
                ),
                "interaction_mode": str(
                    attribute_model_config.get("interaction_mode", "residual")
                ),
                "interaction_scale_initial": float(
                    attribute_model_config.get("interaction_scale_initial", 0.10)
                ),
            },
            "graph_checkpoint": str(graph_checkpoint),
            "fusion": {
                "attribute_weight": attribute_weight,
                "graph_weight": graph_weight,
                "threshold": threshold,
            },
        },
        output_root / "best.pt",
    )
    with (output_root / "metrics.json").open("w", encoding="utf-8") as stream:
        json.dump(results, stream, ensure_ascii=False, indent=2)
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
