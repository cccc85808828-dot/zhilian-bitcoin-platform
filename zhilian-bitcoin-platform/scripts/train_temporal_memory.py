from __future__ import annotations

import argparse
import copy
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
)
from torch.nn import functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tthgnn_erl.baselines import (  # noqa: E402
    FeatureStatistics,
    fit_feature_statistics,
)
from tthgnn_erl.ellipticpp import (  # noqa: E402
    EllipticPPHypergraphSnapshot,
    EllipticPPSnapshotDataset,
)
from tthgnn_erl.temporal import (  # noqa: E402
    TemporalMemoryHGNN,
    TemporalTransactionComponents,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train temporal-memory HGNN")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "ablations" / "full.yaml",
    )
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument(
        "--fixed-epochs",
        type=int,
        default=None,
        help="Refit for exactly this many epochs without validation selection.",
    )
    parser.add_argument(
        "--fixed-threshold",
        type=float,
        default=None,
        help="Frozen out-of-fold threshold used by a fixed-epoch refit.",
    )
    parser.add_argument(
        "--skip-test",
        action="store_true",
        help="Do not read or evaluate the configured outer test interval.",
    )
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--penalty-weight", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--baseline-comparison", type=Path, default=None)
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
    return list(range(int(interval[0]), int(interval[1]) + 1))


def load_snapshots(root: Path) -> Tuple[Dict[int, EllipticPPHypergraphSnapshot], int]:
    dataset = EllipticPPSnapshotDataset(root, validate=True)
    snapshots = {snapshot.time_id: snapshot for snapshot in dataset}
    return snapshots, int(dataset.metadata["totals"]["global_addresses"])


def _distribution_descriptor(
    snapshot: EllipticPPHypergraphSnapshot,
) -> torch.Tensor:
    """Describe an observable snapshot distribution without using labels."""
    transaction_features = snapshot.transaction_features.to(dtype=torch.float32)
    feature_statistics = torch.cat(
        [
            transaction_features.mean(dim=0),
            transaction_features.std(dim=0, unbiased=False),
            torch.quantile(
                transaction_features,
                torch.tensor([0.25, 0.50, 0.75]),
                dim=0,
            ).flatten(),
        ]
    )
    input_degrees = torch.bincount(
        snapshot.input_index[1], minlength=snapshot.num_transactions
    ).to(dtype=torch.float32)
    output_degrees = torch.bincount(
        snapshot.output_index[1], minlength=snapshot.num_transactions
    ).to(dtype=torch.float32)
    structural_values: List[torch.Tensor] = []
    for degrees in (input_degrees, output_degrees, input_degrees + output_degrees):
        structural_values.extend(
            [
                degrees.mean().reshape(1),
                degrees.std(unbiased=False).reshape(1),
                torch.quantile(
                    degrees,
                    torch.tensor([0.25, 0.50, 0.75]),
                ),
            ]
        )
    size_values = torch.log1p(
        torch.tensor(
            [
                snapshot.num_transactions,
                snapshot.num_addresses,
                snapshot.num_input_relations,
                snapshot.num_output_relations,
            ],
            dtype=torch.float32,
        )
    )
    return torch.cat([feature_statistics, *structural_values, size_values])


def build_drift_robust_state(
    snapshots: Dict[int, EllipticPPHypergraphSnapshot],
    train_times: Sequence[int],
    num_environments: int,
    minimum_environment_size: int,
    temperature: float,
    drift_prior_strength: float,
    recency_prior_strength: float,
    risk_ema_decay: float,
) -> Dict[str, Any]:
    """Build contiguous environments with approximately equal cumulative drift."""
    if num_environments < 2:
        raise ValueError("Drift-robust learning requires at least two environments.")
    if minimum_environment_size < 1:
        raise ValueError("minimum_environment_size must be at least one.")
    if num_environments * minimum_environment_size > len(train_times):
        raise ValueError("The requested drift environments are too small for training.")
    if temperature <= 0.0:
        raise ValueError("DERL temperature must be positive.")
    if not 0.0 <= risk_ema_decay < 1.0:
        raise ValueError("risk_ema_decay must be in [0, 1).")

    ordered_times = [int(time_id) for time_id in train_times]
    descriptors = torch.stack(
        [_distribution_descriptor(snapshots[time_id]) for time_id in ordered_times]
    )
    scale = descriptors.std(dim=0, unbiased=False).clamp_min(1e-4)
    normalized = (descriptors - descriptors.mean(dim=0)) / scale
    drift = torch.zeros(len(ordered_times), dtype=torch.float32)
    if len(ordered_times) > 1:
        drift[1:] = (normalized[1:] - normalized[:-1]).square().mean(dim=1).sqrt()

    cumulative = torch.cumsum(drift, dim=0).numpy()
    total_drift = float(cumulative[-1])
    boundaries = [0]
    previous = 0
    for environment_id in range(num_environments - 1):
        lower = previous + minimum_environment_size
        remaining = num_environments - environment_id - 1
        upper = len(ordered_times) - remaining * minimum_environment_size
        if total_drift > 0.0:
            target = total_drift * (environment_id + 1) / num_environments
            candidates = np.arange(lower, upper + 1, dtype=np.int64)
            candidate_drift = cumulative[candidates - 1]
            boundary = int(candidates[np.argmin(np.abs(candidate_drift - target))])
        else:
            boundary = lower
        boundaries.append(boundary)
        previous = boundary
    boundaries.append(len(ordered_times))

    environment_times: List[List[int]] = []
    environment_by_time: Dict[int, int] = {}
    environment_drift: List[float] = []
    for environment_id, (start, end) in enumerate(
        zip(boundaries[:-1], boundaries[1:])
    ):
        times = ordered_times[start:end]
        environment_times.append(times)
        environment_drift.append(float(drift[start:end].sum()))
        for time_id in times:
            environment_by_time[time_id] = environment_id

    drift_tensor = torch.tensor(environment_drift, dtype=torch.float32)
    standardized_drift = (
        (drift_tensor - drift_tensor.mean())
        / drift_tensor.std(unbiased=False).clamp_min(1e-6)
    )
    recency = torch.linspace(-1.0, 1.0, num_environments)
    prior_logits = (
        drift_prior_strength * standardized_drift
        + recency_prior_strength * recency
    )
    prior_weights = torch.softmax(prior_logits, dim=0)
    return {
        "environment_by_time": environment_by_time,
        "environment_times": environment_times,
        "environment_counts": [len(times) for times in environment_times],
        "snapshot_drift": {
            str(time_id): float(value)
            for time_id, value in zip(ordered_times, drift.tolist())
        },
        "environment_drift": environment_drift,
        "prior_weights": prior_weights,
        "weights": prior_weights.clone(),
        "temperature": float(temperature),
        "risk_ema_decay": float(risk_ema_decay),
        "risk_ema": None,
        "last_environment_risks": None,
        "last_robust_loss": 0.0,
    }


def drift_robust_metadata(state: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if state is None:
        return {"enabled": False}
    risks = state.get("last_environment_risks")
    return {
        "enabled": True,
        "environment_times": state["environment_times"],
        "environment_counts": state["environment_counts"],
        "snapshot_drift": state["snapshot_drift"],
        "environment_drift": state["environment_drift"],
        "prior_weights": state["prior_weights"].tolist(),
        "final_weights": state["weights"].tolist(),
        "final_environment_risks": None if risks is None else risks.tolist(),
        "temperature": state["temperature"],
        "risk_ema_decay": state["risk_ema_decay"],
        "last_robust_loss": state["last_robust_loss"],
        "uses_labels_for_environment_construction": False,
    }


def load_global_address_labels(
    path: Path,
    global_address_count: int,
) -> torch.Tensor:
    frame = pd.read_csv(
        path,
        usecols=["global_address_id", "raw_class"],
        dtype={"global_address_id": "int64", "raw_class": "int8"},
    ).sort_values("global_address_id")
    expected_ids = np.arange(global_address_count, dtype=np.int64)
    actual_ids = frame["global_address_id"].to_numpy(dtype=np.int64, copy=False)
    if actual_ids.shape != expected_ids.shape or not np.array_equal(
        actual_ids, expected_ids
    ):
        raise ValueError("Address index does not cover every global address ID.")
    raw_classes = frame["raw_class"].to_numpy(dtype=np.int8, copy=True)
    labels = np.full(global_address_count, -1, dtype=np.int64)
    labels[raw_classes == 1] = 1
    labels[raw_classes == 2] = 0
    if not np.isin(raw_classes, [1, 2, 3]).all():
        raise ValueError("Address index contains an unexpected raw class.")
    return torch.from_numpy(labels)


def fit_statistics(
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


def training_class_counts(
    snapshots: Dict[int, EllipticPPHypergraphSnapshot], train_times: Sequence[int]
) -> Tuple[int, int]:
    positives = sum(int((snapshots[t].labels == 1).sum()) for t in train_times)
    negatives = sum(int((snapshots[t].labels == 0).sum()) for t in train_times)
    return positives, negatives


def _role_channel_statistics(
    values: torch.Tensor,
    minimum_std: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    detached = values.detach()
    mean = detached.mean(dim=0)
    std = detached.std(dim=0, unbiased=False).clamp_min(minimum_std)
    return mean, std


def _cross_environment_style_transfer(
    selected_values: torch.Tensor,
    source_context: torch.Tensor,
    target_context: torch.Tensor,
    beta_alpha: float,
    minimum_std: float,
) -> torch.Tensor:
    source_mean, source_std = _role_channel_statistics(
        source_context,
        minimum_std,
    )
    target_mean, target_std = _role_channel_statistics(
        target_context,
        minimum_std,
    )
    concentration = torch.tensor(
        beta_alpha,
        device=selected_values.device,
        dtype=selected_values.dtype,
    )
    mixing = torch.distributions.Beta(concentration, concentration).sample(
        (selected_values.size(0), 1)
    )
    mixed_mean = mixing * source_mean + (1.0 - mixing) * target_mean
    mixed_std = mixing * source_std + (1.0 - mixing) * target_std
    normalized = (selected_values - source_mean) / source_std
    return normalized * mixed_std + mixed_mean


def rcha_augmentation_loss(
    model: TemporalMemoryHGNN,
    records: Sequence[Dict[str, Any]],
    augmentation: Dict[str, Any],
) -> Tuple[torch.Tensor, int, int, int]:
    if len(records) < 2:
        device = next(model.parameters()).device
        return torch.zeros((), device=device), 0, 0, 0

    method = str(augmentation.get("method", "rcha"))
    adaptive = method == "adaptive_rcha"
    pseudo_labeling = dict(augmentation.get("pseudo_labeling", {}))
    hard_fraction = float(augmentation.get("hard_positive_fraction", 0.5))
    difficulty_gamma = float(augmentation.get("difficulty_gamma", 2.0))
    beta_alpha = float(augmentation.get("beta_alpha", 0.5))
    consistency_weight = float(augmentation.get("consistency_weight", 0.1))
    minimum_std = float(augmentation.get("minimum_std", 1e-4))
    minimum_sample_weight = float(
        augmentation.get("minimum_sample_weight", 0.05)
    )
    if not 0.0 < hard_fraction <= 1.0:
        raise ValueError("hard_positive_fraction must be in (0, 1].")
    if difficulty_gamma < 0.0:
        raise ValueError("difficulty_gamma cannot be negative.")
    if beta_alpha <= 0.0:
        raise ValueError("beta_alpha must be positive.")
    if consistency_weight < 0.0:
        raise ValueError("consistency_weight cannot be negative.")
    if minimum_std <= 0.0:
        raise ValueError("minimum_std must be positive.")
    if minimum_sample_weight < 0.0:
        raise ValueError("minimum_sample_weight cannot be negative.")
    positive_threshold = float(
        pseudo_labeling.get("positive_threshold", 0.98)
    )
    negative_threshold = float(
        pseudo_labeling.get("negative_threshold", 0.02)
    )
    student_positive_threshold = float(
        pseudo_labeling.get("student_positive_threshold", 0.90)
    )
    student_negative_threshold = float(
        pseudo_labeling.get("student_negative_threshold", 0.10)
    )
    maximum_probability_gap = float(
        pseudo_labeling.get("maximum_probability_gap", 0.10)
    )
    max_pseudo_positive_ratio = float(
        pseudo_labeling.get("max_pseudo_positive_ratio", 1.0)
    )
    max_pseudo_negative_ratio = float(
        pseudo_labeling.get("max_pseudo_negative_ratio", 2.0)
    )
    pseudo_label_weight = float(
        pseudo_labeling.get("loss_weight", 0.25)
    )
    pseudo_negative_weight = float(
        pseudo_labeling.get("negative_weight", 1.0)
    )
    attribute_positive_threshold_value = pseudo_labeling.get(
        "attribute_positive_threshold"
    )
    attribute_positive_threshold = (
        float(attribute_positive_threshold_value)
        if attribute_positive_threshold_value is not None
        else None
    )
    maximum_cross_branch_gap_value = pseudo_labeling.get(
        "maximum_cross_branch_gap"
    )
    maximum_cross_branch_gap = (
        float(maximum_cross_branch_gap_value)
        if maximum_cross_branch_gap_value is not None
        else None
    )
    augmentation_stability_gap_value = pseudo_labeling.get(
        "augmentation_stability_gap"
    )
    augmentation_stability_gap = (
        float(augmentation_stability_gap_value)
        if augmentation_stability_gap_value is not None
        else None
    )
    soft_pseudo_targets = bool(pseudo_labeling.get("soft_targets", False))
    offline_pseudo_stability_gap_value = augmentation.get(
        "offline_pseudo_stability_gap"
    )
    offline_pseudo_stability_gap = (
        float(offline_pseudo_stability_gap_value)
        if offline_pseudo_stability_gap_value is not None
        else None
    )
    offline_pseudo_source_loss = bool(
        augmentation.get("offline_pseudo_source_loss", False)
    )
    if (
        offline_pseudo_stability_gap is not None
        and offline_pseudo_stability_gap < 0.0
    ):
        raise ValueError("offline_pseudo_stability_gap cannot be negative.")
    if adaptive:
        if not 0.5 < positive_threshold <= 1.0:
            raise ValueError("pseudo positive_threshold must be in (0.5, 1].")
        if not 0.0 <= negative_threshold < 0.5:
            raise ValueError("pseudo negative_threshold must be in [0, 0.5).")
        if not 0.5 < student_positive_threshold <= 1.0:
            raise ValueError(
                "pseudo student_positive_threshold must be in (0.5, 1]."
            )
        if not 0.0 <= student_negative_threshold < 0.5:
            raise ValueError(
                "pseudo student_negative_threshold must be in [0, 0.5)."
            )
        if maximum_probability_gap < 0.0:
            raise ValueError("maximum_probability_gap cannot be negative.")
        if max_pseudo_positive_ratio < 0.0 or max_pseudo_negative_ratio < 0.0:
            raise ValueError("pseudo-label sampling ratios cannot be negative.")
        if pseudo_label_weight < 0.0 or pseudo_negative_weight < 0.0:
            raise ValueError("pseudo-label loss weights cannot be negative.")
        if attribute_positive_threshold is not None and not (
            0.5 < attribute_positive_threshold <= 1.0
        ):
            raise ValueError(
                "pseudo attribute_positive_threshold must be in (0.5, 1]."
            )
        if maximum_cross_branch_gap is not None and maximum_cross_branch_gap < 0.0:
            raise ValueError("maximum_cross_branch_gap cannot be negative.")
        if augmentation_stability_gap is not None and augmentation_stability_gap < 0.0:
            raise ValueError("augmentation_stability_gap cannot be negative.")

    losses: List[torch.Tensor] = []
    pseudo_losses: List[torch.Tensor] = []
    selected_count = 0
    pseudo_positive_count = 0
    pseudo_negative_count = 0
    for source_index, source in enumerate(records):
        source_labels = source["labels"]
        source_mask = source["labeled_mask"]
        positive_indices = torch.nonzero(
            source_mask & (source_labels == 1),
            as_tuple=False,
        ).flatten()
        source_logits = source["logits"]
        source_probabilities = torch.sigmoid(source_logits).detach()
        if positive_indices.numel() > 0:
            labeled_difficulty = 1.0 - source_probabilities[positive_indices]
            keep = max(
                1,
                int(np.ceil(hard_fraction * int(positive_indices.numel()))),
            )
            hard_positions = torch.topk(
                labeled_difficulty,
                k=keep,
                largest=True,
            ).indices
            selected_indices = positive_indices[hard_positions]
            selected_difficulty = labeled_difficulty[hard_positions]
        else:
            selected_indices = torch.empty(
                0,
                dtype=torch.long,
                device=source_logits.device,
            )
            selected_difficulty = torch.empty(
                0,
                dtype=source_logits.dtype,
                device=source_logits.device,
            )
        selected_targets = torch.ones_like(selected_difficulty)
        selected_pseudo_mask = torch.zeros(
            selected_indices.numel(),
            dtype=torch.bool,
            device=source_logits.device,
        )

        selected_pseudo_positives = torch.empty(
            0,
            dtype=torch.long,
            device=source_logits.device,
        )
        selected_pseudo_negatives = torch.empty_like(selected_pseudo_positives)
        offline_pseudo_targets = source.get("offline_pseudo_targets")
        if offline_pseudo_targets is not None:
            offline_candidate_mask = (
                torch.isfinite(offline_pseudo_targets) & ~source_mask
            )
            offline_candidates = torch.nonzero(
                offline_candidate_mask,
                as_tuple=False,
            ).flatten()
            if offline_candidates.numel() > 0:
                offline_difficulty = 1.0 - source_probabilities[
                    offline_candidates
                ]
                offline_keep = max(
                    1,
                    int(
                        np.ceil(
                            hard_fraction * int(offline_candidates.numel())
                        )
                    ),
                )
                hard_offline_positions = torch.topk(
                    offline_difficulty,
                    k=offline_keep,
                    largest=True,
                ).indices
                offline_indices = offline_candidates[hard_offline_positions]
                offline_targets = offline_pseudo_targets[
                    offline_indices
                ].detach().clamp(0.500001, 1.0)
                selected_indices = torch.cat(
                    [selected_indices, offline_indices]
                )
                selected_difficulty = torch.cat(
                    [
                        selected_difficulty,
                        offline_difficulty[hard_offline_positions],
                    ]
                )
                selected_targets = torch.cat(
                    [selected_targets, offline_targets]
                )
                selected_pseudo_mask = torch.cat(
                    [
                        selected_pseudo_mask,
                        torch.ones(
                            offline_indices.numel(),
                            dtype=torch.bool,
                            device=source_logits.device,
                        ),
                    ]
                )
        teacher_probabilities = source.get("teacher_probabilities")
        attribute_probabilities = source.get("attribute_probabilities")
        if adaptive:
            if teacher_probabilities is None:
                raise ValueError(
                    "adaptive_rcha requires EMA teacher probabilities."
                )
            if (
                attribute_positive_threshold is not None
                and attribute_probabilities is None
            ):
                raise ValueError(
                    "Attribute-agreement RCHA requires attribute probabilities."
                )
            unlabeled = ~source_mask
            agreement_gap = torch.abs(
                teacher_probabilities - source_probabilities
            )
            positive_candidate_mask = (
                unlabeled
                & (teacher_probabilities >= positive_threshold)
                & (source_probabilities >= student_positive_threshold)
                & (agreement_gap <= maximum_probability_gap)
            )
            if attribute_positive_threshold is not None:
                cross_branch_gap = torch.abs(
                    teacher_probabilities - attribute_probabilities
                )
                positive_candidate_mask = (
                    positive_candidate_mask
                    & (attribute_probabilities >= attribute_positive_threshold)
                )
                if maximum_cross_branch_gap is not None:
                    positive_candidate_mask = (
                        positive_candidate_mask
                        & (cross_branch_gap <= maximum_cross_branch_gap)
                    )
            pseudo_positive_candidates = torch.nonzero(
                positive_candidate_mask,
                as_tuple=False,
            ).flatten()
            positive_cap = int(
                np.ceil(
                    max_pseudo_positive_ratio
                    * max(int(positive_indices.numel()), 1)
                )
            )
            positive_cap = min(positive_cap, int(pseudo_positive_candidates.numel()))
            if positive_cap > 0:
                candidate_rank = teacher_probabilities[
                    pseudo_positive_candidates
                ]
                if attribute_probabilities is not None:
                    candidate_rank = torch.minimum(
                        candidate_rank,
                        attribute_probabilities[pseudo_positive_candidates],
                    )
                best = torch.topk(
                    candidate_rank,
                    k=positive_cap,
                    largest=True,
                ).indices
                selected_pseudo_positives = pseudo_positive_candidates[best]
                pseudo_targets = teacher_probabilities[
                    selected_pseudo_positives
                ]
                if attribute_probabilities is not None:
                    pseudo_targets = 0.5 * (
                        pseudo_targets
                        + attribute_probabilities[selected_pseudo_positives]
                    )
                if not soft_pseudo_targets:
                    pseudo_targets = torch.ones_like(pseudo_targets)
                selected_indices = torch.cat(
                    [selected_indices, selected_pseudo_positives]
                )
                selected_difficulty = torch.cat(
                    [
                        selected_difficulty,
                        1.0 - source_probabilities[selected_pseudo_positives],
                    ]
                )
                selected_targets = torch.cat(
                    [selected_targets, pseudo_targets.detach()]
                )
                selected_pseudo_mask = torch.cat(
                    [
                        selected_pseudo_mask,
                        torch.ones(
                            selected_pseudo_positives.numel(),
                            dtype=torch.bool,
                            device=source_logits.device,
                        ),
                    ]
                )
            pseudo_negative_candidates = torch.nonzero(
                unlabeled
                & (teacher_probabilities <= negative_threshold)
                & (source_probabilities <= student_negative_threshold)
                & (agreement_gap <= maximum_probability_gap),
                as_tuple=False,
            ).flatten()
            negative_cap = int(
                np.ceil(
                    max_pseudo_negative_ratio
                    * max(int(positive_indices.numel()), 1)
                )
            )
            negative_cap = min(negative_cap, int(pseudo_negative_candidates.numel()))
            if negative_cap > 0:
                best = torch.topk(
                    teacher_probabilities[pseudo_negative_candidates],
                    k=negative_cap,
                    largest=False,
                ).indices
                selected_pseudo_negatives = pseudo_negative_candidates[best]
            pseudo_terms: List[torch.Tensor] = []
            if selected_pseudo_negatives.numel() > 0:
                pseudo_terms.append(
                    pseudo_negative_weight
                    * F.binary_cross_entropy_with_logits(
                        source_logits[selected_pseudo_negatives],
                        torch.zeros_like(source_logits[selected_pseudo_negatives]),
                    )
                )
            if pseudo_terms:
                pseudo_losses.append(torch.stack(pseudo_terms).mean())
            pseudo_negative_count += int(selected_pseudo_negatives.numel())

        if selected_indices.numel() == 0:
            continue

        target_index = int(torch.randint(len(records) - 1, (1,)).item())
        if target_index >= source_index:
            target_index += 1
        target = records[target_index]
        source_components: TemporalTransactionComponents = source["components"]
        target_components: TemporalTransactionComponents = target["components"]

        augmented_input = _cross_environment_style_transfer(
            source_components.input_history_message[selected_indices],
            source_components.input_history_message,
            target_components.input_history_message,
            beta_alpha,
            minimum_std,
        )
        augmented_output = _cross_environment_style_transfer(
            source_components.output_history_message[selected_indices],
            source_components.output_history_message,
            target_components.output_history_message,
            beta_alpha,
            minimum_std,
        )
        augmented_components = TemporalTransactionComponents(
            transaction_states=source_components.transaction_states[selected_indices],
            input_history_message=augmented_input,
            output_history_message=augmented_output,
            environment_gate_logits=(
                source_components.environment_gate_logits
            ),
            attribute_prior_logits=(
                source_components.attribute_prior_logits[selected_indices]
                if source_components.attribute_prior_logits is not None
                else None
            ),
            direct_attribute_logits=(
                source_components.direct_attribute_logits[selected_indices]
                if source_components.direct_attribute_logits is not None
                else None
            ),
        )
        augmented_logits = model.classify_components(augmented_components)
        accepted = torch.ones_like(selected_pseudo_mask)
        stability_gap = (
            augmentation_stability_gap
            if adaptive
            else offline_pseudo_stability_gap
        )
        if stability_gap is not None and selected_pseudo_mask.any():
            augmented_probabilities = torch.sigmoid(augmented_logits).detach()
            accepted[selected_pseudo_mask] = (
                torch.abs(
                    augmented_probabilities[selected_pseudo_mask]
                    - selected_targets[selected_pseudo_mask]
                )
                <= stability_gap
            )
        if not accepted.any():
            continue
        accepted_pseudo_mask = selected_pseudo_mask & accepted
        if accepted_pseudo_mask.any():
            accepted_pseudo_indices = selected_indices[accepted_pseudo_mask]
            accepted_pseudo_targets = selected_targets[accepted_pseudo_mask]
            if adaptive or offline_pseudo_source_loss:
                per_sample_pseudo_loss = F.binary_cross_entropy_with_logits(
                    source_logits[accepted_pseudo_indices],
                    accepted_pseudo_targets,
                    reduction="none",
                )
                pseudo_confidence = (
                    2.0 * (accepted_pseudo_targets - 0.5)
                ).clamp_min(0.0)
                pseudo_losses.append(
                    (
                        pseudo_confidence * per_sample_pseudo_loss
                    ).sum()
                    / pseudo_confidence.sum().clamp_min(1e-12)
                )
            pseudo_positive_count += int(accepted_pseudo_mask.sum())
        selected_indices = selected_indices[accepted]
        selected_difficulty = selected_difficulty[accepted]
        selected_targets = selected_targets[accepted]
        augmented_logits = augmented_logits[accepted]
        positive_loss = F.binary_cross_entropy_with_logits(
            augmented_logits,
            selected_targets,
            reduction="none",
        )
        original_probability = torch.sigmoid(
            source_logits[selected_indices]
        ).detach()
        consistency_loss = F.binary_cross_entropy_with_logits(
            augmented_logits,
            original_probability,
            reduction="none",
        )
        sample_weights = selected_difficulty.pow(difficulty_gamma).clamp_min(
            minimum_sample_weight
        )
        combined = positive_loss + consistency_weight * consistency_loss
        losses.append(
            (sample_weights * combined).sum() / sample_weights.sum().clamp_min(1e-12)
        )
        selected_count += int(selected_indices.numel())

    if not losses:
        device = next(model.parameters()).device
        base_loss = torch.zeros((), device=device)
    else:
        base_loss = torch.stack(losses).mean()
    if pseudo_losses:
        base_loss = base_loss + pseudo_label_weight * torch.stack(
            pseudo_losses
        ).mean()
    return (
        base_loss,
        selected_count,
        pseudo_positive_count,
        pseudo_negative_count,
    )


def _ccha_bernoulli_js_divergence(
    source_probability: torch.Tensor,
    augmented_probability: torch.Tensor,
) -> torch.Tensor:
    epsilon = torch.finfo(source_probability.dtype).eps
    source_probability = source_probability.clamp(epsilon, 1.0 - epsilon)
    augmented_probability = augmented_probability.clamp(epsilon, 1.0 - epsilon)
    midpoint = 0.5 * (source_probability + augmented_probability)
    source_kl = (
        source_probability * torch.log(source_probability / midpoint)
        + (1.0 - source_probability)
        * torch.log((1.0 - source_probability) / (1.0 - midpoint))
    )
    augmented_kl = (
        augmented_probability * torch.log(augmented_probability / midpoint)
        + (1.0 - augmented_probability)
        * torch.log((1.0 - augmented_probability) / (1.0 - midpoint))
    )
    return 0.5 * (source_kl + augmented_kl)


def _ccha_transaction_role_counts(
    snapshot: EllipticPPHypergraphSnapshot,
) -> Tuple[torch.Tensor, torch.Tensor]:
    input_counts = torch.bincount(
        snapshot.input_index[1], minlength=snapshot.num_transactions
    )
    output_counts = torch.bincount(
        snapshot.output_index[1], minlength=snapshot.num_transactions
    )
    return input_counts, output_counts


def _ccha_mix_transaction_features(
    source_snapshot: EllipticPPHypergraphSnapshot,
    source_index: int,
    donor_snapshot: EllipticPPHypergraphSnapshot,
    donor_index: int,
    interpolation_alpha: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    source_features = source_snapshot.transaction_features[source_index]
    donor_features = donor_snapshot.transaction_features[donor_index]
    source_mask = source_snapshot.transaction_feature_mask[source_index]
    donor_mask = donor_snapshot.transaction_feature_mask[donor_index]
    concentration = torch.tensor(
        interpolation_alpha,
        device=source_features.device,
        dtype=source_features.dtype,
    )
    mixing = torch.distributions.Beta(concentration, concentration).sample()
    mixed_features = torch.zeros_like(source_features)
    shared_mask = source_mask & donor_mask
    source_only_mask = source_mask & ~donor_mask
    donor_only_mask = donor_mask & ~source_mask
    mixed_features[shared_mask] = (
        mixing * source_features[shared_mask]
        + (1.0 - mixing) * donor_features[shared_mask]
    )
    mixed_features[source_only_mask] = source_features[source_only_mask]
    mixed_features[donor_only_mask] = donor_features[donor_only_mask]
    return mixed_features, source_mask | donor_mask


def _ccha_carrier_pool(
    records: Sequence[Dict[str, Any]],
    role: str,
    pool_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    carrier_features: List[torch.Tensor] = []
    carrier_masks: List[torch.Tensor] = []
    for record in records:
        snapshot: EllipticPPHypergraphSnapshot = record["snapshot"]
        relation_index = (
            snapshot.input_index if role == "input" else snapshot.output_index
        )
        if relation_index.numel() == 0:
            continue
        unknown_relation_mask = ~snapshot.labeled_mask.index_select(
            0, relation_index[1]
        )
        local_address_indices = relation_index[0, unknown_relation_mask]
        if local_address_indices.numel() == 0:
            continue
        local_address_indices = torch.unique(local_address_indices)
        carrier_features.append(
            snapshot.address_features.index_select(0, local_address_indices)
        )
        carrier_masks.append(
            snapshot.address_feature_mask.index_select(0, local_address_indices)
        )
    if not carrier_features:
        device = records[0]["logits"].device
        feature_dimension = int(
            records[0]["snapshot"].address_features.size(1)
        )
        return (
            torch.empty((0, feature_dimension), device=device),
            torch.empty((0,), dtype=torch.bool, device=device),
        )
    features = torch.cat(carrier_features, dim=0)
    masks = torch.cat(carrier_masks, dim=0)
    if features.size(0) > pool_size:
        selected = torch.randperm(features.size(0), device=features.device)[
            :pool_size
        ]
        features = features.index_select(0, selected)
        masks = masks.index_select(0, selected)
    return features, masks


def _ccha_match_carrier_attributes(
    source_features: torch.Tensor,
    carrier_features: torch.Tensor,
    descriptor_indices: Sequence[int],
    top_k: int,
) -> torch.Tensor:
    if carrier_features.size(0) == 0:
        raise ValueError("CCHA requires at least one unknown-address carrier.")
    valid_indices = [
        int(index)
        for index in descriptor_indices
        if 0 <= int(index) < carrier_features.size(1)
    ]
    if not valid_indices:
        valid_indices = list(range(min(3, carrier_features.size(1))))
    index_tensor = torch.tensor(
        valid_indices,
        dtype=torch.long,
        device=carrier_features.device,
    )
    carrier_descriptor = torch.log1p(
        carrier_features.index_select(1, index_tensor).clamp_min(0.0)
    )
    source_descriptor = torch.log1p(
        source_features.index_select(0, index_tensor).clamp_min(0.0)
    )
    descriptor_mean = carrier_descriptor.mean(dim=0)
    descriptor_std = carrier_descriptor.std(dim=0, unbiased=False).clamp_min(
        1e-6
    )
    carrier_descriptor = (carrier_descriptor - descriptor_mean) / descriptor_std
    source_descriptor = (source_descriptor - descriptor_mean) / descriptor_std
    distances = (carrier_descriptor - source_descriptor.unsqueeze(0)).pow(2).sum(
        dim=1
    )
    candidate_count = min(max(int(top_k), 1), int(distances.numel()))
    nearest = torch.topk(
        distances, k=candidate_count, largest=False
    ).indices
    chosen = nearest[torch.randint(candidate_count, (1,), device=nearest.device)]
    return chosen.squeeze(0)


def ccha_augmentation_loss(
    model: TemporalMemoryHGNN,
    records: Sequence[Dict[str, Any]],
    augmentation: Dict[str, Any],
) -> Tuple[torch.Tensor, int, int, int]:
    """Construct label-preserving cold-start counterfactual hyperedges.

    Labeled illicit transactions provide the behavior label and transaction
    attributes. Unknown transactions only provide address-attribute carriers;
    their labels and graph edges are never copied into supervision.
    """
    device = next(model.parameters()).device
    if len(records) < 2:
        return torch.zeros((), device=device), 0, 0, 0

    hard_fraction = float(augmentation.get("hard_positive_fraction", 0.5))
    difficulty_gamma = float(augmentation.get("difficulty_gamma", 2.0))
    interpolation_alpha = float(
        augmentation.get("interpolation_alpha", augmentation.get("beta_alpha", 0.5))
    )
    consistency_weight = float(augmentation.get("consistency_weight", 0.1))
    minimum_sample_weight = float(
        augmentation.get("minimum_sample_weight", 0.05)
    )
    amount_feature_index = int(augmentation.get("amount_feature_index", 167))
    amount_bins = int(augmentation.get("amount_bins", 8))
    donor_top_k = int(augmentation.get("donor_top_k", 8))
    carrier_top_k = int(augmentation.get("carrier_top_k", 16))
    carrier_pool_size = int(augmentation.get("carrier_pool_size", 4096))
    maximum_augmented = int(
        augmentation.get("max_augmented_per_environment", 24)
    )
    maximum_relations_per_role = int(
        augmentation.get("max_relations_per_role", 64)
    )
    input_descriptor_indices = augmentation.get(
        "input_descriptor_indices", [0, 5, 8]
    )
    output_descriptor_indices = augmentation.get(
        "output_descriptor_indices", [1, 5, 8]
    )
    if not 0.0 < hard_fraction <= 1.0:
        raise ValueError("hard_positive_fraction must be in (0, 1].")
    if difficulty_gamma < 0.0:
        raise ValueError("difficulty_gamma cannot be negative.")
    if interpolation_alpha <= 0.0:
        raise ValueError("interpolation_alpha must be positive.")
    if consistency_weight < 0.0:
        raise ValueError("consistency_weight cannot be negative.")
    if amount_bins < 2:
        raise ValueError("amount_bins must be at least two.")
    if donor_top_k < 1 or carrier_top_k < 1 or carrier_pool_size < 1:
        raise ValueError("CCHA candidate-pool sizes must be positive.")
    if maximum_augmented < 1 or maximum_relations_per_role < 1:
        raise ValueError("CCHA augmentation limits must be positive.")

    input_carrier_features, input_carrier_masks = _ccha_carrier_pool(
        records, "input", carrier_pool_size
    )
    output_carrier_features, output_carrier_masks = _ccha_carrier_pool(
        records, "output", carrier_pool_size
    )
    if input_carrier_features.size(0) == 0 or output_carrier_features.size(0) == 0:
        return torch.zeros((), device=device), 0, 0, 0

    role_counts: List[Tuple[torch.Tensor, torch.Tensor]] = []
    positive_entries: List[Tuple[int, int]] = []
    amount_values: List[torch.Tensor] = []
    source_candidates: List[Tuple[float, int, int]] = []
    for record_index, record in enumerate(records):
        snapshot: EllipticPPHypergraphSnapshot = record["snapshot"]
        input_counts, output_counts = _ccha_transaction_role_counts(snapshot)
        role_counts.append((input_counts, output_counts))
        positive_indices = torch.nonzero(
            snapshot.labeled_mask & (snapshot.labels == 1), as_tuple=False
        ).flatten()
        if positive_indices.numel() == 0:
            continue
        eligible_mask = (
            (input_counts.index_select(0, positive_indices) > 0)
            & (output_counts.index_select(0, positive_indices) > 0)
            & (
                input_counts.index_select(0, positive_indices)
                <= maximum_relations_per_role
            )
            & (
                output_counts.index_select(0, positive_indices)
                <= maximum_relations_per_role
            )
        )
        positive_indices = positive_indices[eligible_mask]
        if positive_indices.numel() == 0:
            continue
        probabilities = torch.sigmoid(record["logits"][positive_indices]).detach()
        keep_count = max(1, int(np.ceil(hard_fraction * positive_indices.numel())))
        hard_positions = torch.topk(
            1.0 - probabilities,
            k=min(keep_count, int(positive_indices.numel())),
            largest=True,
        ).indices
        for position in hard_positions.tolist():
            transaction_index = int(positive_indices[position])
            source_candidates.append(
                (
                    float(1.0 - probabilities[position]),
                    record_index,
                    transaction_index,
                )
            )
        for transaction_index in positive_indices.tolist():
            positive_entries.append((record_index, int(transaction_index)))
            feature_index = min(
                max(amount_feature_index, 0), snapshot.transaction_features.size(1) - 1
            )
            amount_values.append(
                snapshot.transaction_features[transaction_index, feature_index].detach()
            )
    if not source_candidates or not positive_entries:
        return torch.zeros((), device=device), 0, 0, 0

    source_candidates.sort(key=lambda item: item[0], reverse=True)
    source_candidates = source_candidates[:maximum_augmented]
    all_amounts = torch.stack(amount_values).to(dtype=torch.float32)
    quantile_points = torch.linspace(
        0.0, 1.0, amount_bins + 1, device=all_amounts.device
    )[1:-1]
    amount_boundaries = torch.quantile(all_amounts, quantile_points)

    virtual_address_features: List[torch.Tensor] = []
    virtual_address_masks: List[torch.Tensor] = []
    synthetic_transaction_features: List[torch.Tensor] = []
    synthetic_transaction_masks: List[torch.Tensor] = []
    input_address_indices: List[int] = []
    input_transaction_indices: List[int] = []
    output_address_indices: List[int] = []
    output_transaction_indices: List[int] = []
    source_logits: List[torch.Tensor] = []
    source_difficulties: List[float] = []

    for difficulty, source_record_index, source_index in source_candidates:
        source_record = records[source_record_index]
        source_snapshot: EllipticPPHypergraphSnapshot = source_record["snapshot"]
        source_input_count = int(role_counts[source_record_index][0][source_index])
        source_output_count = int(role_counts[source_record_index][1][source_index])
        feature_index = min(
            max(amount_feature_index, 0), source_snapshot.transaction_features.size(1) - 1
        )
        source_amount = source_snapshot.transaction_features[
            source_index, feature_index
        ].detach().to(dtype=torch.float32)
        source_bin = int(torch.bucketize(source_amount, amount_boundaries))
        donor_candidates: List[Tuple[float, int, int]] = []
        for donor_record_index, donor_index in positive_entries:
            if donor_record_index == source_record_index:
                continue
            donor_snapshot: EllipticPPHypergraphSnapshot = records[
                donor_record_index
            ]["snapshot"]
            donor_input_count = int(role_counts[donor_record_index][0][donor_index])
            donor_output_count = int(role_counts[donor_record_index][1][donor_index])
            donor_amount = donor_snapshot.transaction_features[
                donor_index, feature_index
            ].detach().to(dtype=torch.float32)
            donor_bin = int(torch.bucketize(donor_amount, amount_boundaries))
            structural_distance = (
                abs(donor_input_count - source_input_count)
                / max(source_input_count, 1)
                + abs(donor_output_count - source_output_count)
                / max(source_output_count, 1)
            )
            bin_distance = abs(donor_bin - source_bin) / max(amount_bins - 1, 1)
            donor_candidates.append(
                (structural_distance + 0.25 * bin_distance, donor_record_index, donor_index)
            )
        if not donor_candidates:
            continue
        donor_candidates.sort(key=lambda item: item[0])
        candidate_count = min(donor_top_k, len(donor_candidates))
        donor_choice = int(torch.randint(candidate_count, (1,)).item())
        _, donor_record_index, donor_index = donor_candidates[donor_choice]
        donor_snapshot = records[donor_record_index]["snapshot"]

        mixed_features, mixed_mask = _ccha_mix_transaction_features(
            source_snapshot,
            source_index,
            donor_snapshot,
            donor_index,
            interpolation_alpha,
        )
        synthetic_index = len(synthetic_transaction_features)
        source_input_addresses = source_snapshot.input_index[0][
            source_snapshot.input_index[1] == source_index
        ]
        source_output_addresses = source_snapshot.output_index[0][
            source_snapshot.output_index[1] == source_index
        ]
        for local_address_index in source_input_addresses.tolist():
            carrier_index = _ccha_match_carrier_attributes(
                source_snapshot.address_features[local_address_index],
                input_carrier_features,
                input_descriptor_indices,
                carrier_top_k,
            )
            virtual_index = len(virtual_address_features)
            virtual_address_features.append(input_carrier_features[carrier_index])
            virtual_address_masks.append(input_carrier_masks[carrier_index])
            input_address_indices.append(virtual_index)
            input_transaction_indices.append(synthetic_index)
        for local_address_index in source_output_addresses.tolist():
            carrier_index = _ccha_match_carrier_attributes(
                source_snapshot.address_features[local_address_index],
                output_carrier_features,
                output_descriptor_indices,
                carrier_top_k,
            )
            virtual_index = len(virtual_address_features)
            virtual_address_features.append(output_carrier_features[carrier_index])
            virtual_address_masks.append(output_carrier_masks[carrier_index])
            output_address_indices.append(virtual_index)
            output_transaction_indices.append(synthetic_index)
        synthetic_transaction_features.append(mixed_features)
        synthetic_transaction_masks.append(mixed_mask)
        source_logits.append(source_record["logits"][source_index])
        source_difficulties.append(difficulty)

    selected_count = len(synthetic_transaction_features)
    if selected_count == 0:
        return torch.zeros((), device=device), 0, 0, 0
    synthetic_snapshot = EllipticPPHypergraphSnapshot(
        time_id=-1,
        global_address_ids=torch.arange(
            len(virtual_address_features), dtype=torch.long, device=device
        ),
        address_features=torch.stack(virtual_address_features),
        address_feature_mask=torch.stack(virtual_address_masks),
        transaction_ids=-torch.arange(
            1, selected_count + 1, dtype=torch.long, device=device
        ),
        transaction_features=torch.stack(synthetic_transaction_features),
        transaction_feature_mask=torch.stack(synthetic_transaction_masks),
        input_index=torch.tensor(
            [input_address_indices, input_transaction_indices],
            dtype=torch.long,
            device=device,
        ),
        output_index=torch.tensor(
            [output_address_indices, output_transaction_indices],
            dtype=torch.long,
            device=device,
        ),
        labels=torch.ones(selected_count, dtype=torch.long, device=device),
        labeled_mask=torch.ones(selected_count, dtype=torch.bool, device=device),
    )
    zero_memory = torch.zeros(
        synthetic_snapshot.num_addresses,
        model.memory_dim,
        dtype=synthetic_snapshot.address_features.dtype,
        device=device,
    )
    augmented_logits, _ = model(synthetic_snapshot, zero_memory)
    source_logit_tensor = torch.stack(source_logits)
    augmented_loss = F.binary_cross_entropy_with_logits(
        augmented_logits,
        torch.ones_like(augmented_logits),
        reduction="none",
    )
    consistency_loss = _ccha_bernoulli_js_divergence(
        torch.sigmoid(source_logit_tensor), torch.sigmoid(augmented_logits)
    )
    sample_weights = torch.tensor(
        source_difficulties,
        dtype=augmented_logits.dtype,
        device=device,
    ).pow(difficulty_gamma).clamp_min(minimum_sample_weight)
    combined_loss = augmented_loss + consistency_weight * consistency_loss
    loss = (sample_weights * combined_loss).sum() / sample_weights.sum().clamp_min(
        1e-12
    )
    return loss, selected_count, 0, 0


@torch.no_grad()
def update_ema_teacher(
    teacher: TemporalMemoryHGNN,
    student: TemporalMemoryHGNN,
    decay: float,
) -> None:
    if not 0.0 <= decay < 1.0:
        raise ValueError("EMA teacher decay must be in [0, 1).")
    for teacher_parameter, student_parameter in zip(
        teacher.parameters(),
        student.parameters(),
    ):
        teacher_parameter.mul_(decay).add_(
            student_parameter.detach(),
            alpha=1.0 - decay,
        )
    for teacher_buffer, student_buffer in zip(
        teacher.buffers(),
        student.buffers(),
    ):
        teacher_buffer.copy_(student_buffer)


def hard_pair_ranking_loss(
    records: Sequence[Dict[str, Any]],
    hard_fraction: float,
    margin: float,
    max_samples: int,
) -> torch.Tensor:
    """Separate the lowest-scored positives from the highest-scored negatives."""

    if not 0.0 < hard_fraction <= 1.0:
        raise ValueError("hard_fraction must be in (0, 1].")
    if margin < 0.0:
        raise ValueError("ranking margin cannot be negative.")
    if max_samples < 1:
        raise ValueError("max_hard_samples must be at least one.")
    losses: List[torch.Tensor] = []
    for record in records:
        mask = record["labeled_mask"]
        labels = record["labels"][mask]
        logits = record["logits"][mask]
        positive_logits = logits[labels == 1]
        negative_logits = logits[labels == 0]
        if positive_logits.numel() == 0 or negative_logits.numel() == 0:
            continue
        positive_count = min(
            max_samples,
            max(1, int(np.ceil(hard_fraction * positive_logits.numel()))),
        )
        negative_count = min(
            max_samples,
            max(1, int(np.ceil(hard_fraction * negative_logits.numel()))),
        )
        hard_positives = torch.topk(
            positive_logits,
            k=positive_count,
            largest=False,
        ).values
        hard_negatives = torch.topk(
            negative_logits,
            k=negative_count,
            largest=True,
        ).values
        pairwise_margin = (
            margin
            - hard_positives[:, None]
            + hard_negatives[None, :]
        )
        losses.append(F.softplus(pairwise_margin).mean())
    if not losses:
        device = records[0]["logits"].device if records else torch.device("cpu")
        return torch.zeros((), device=device)
    return torch.stack(losses).mean()


def positive_environment_tail_loss(
    records: Sequence[Dict[str, Any]],
    temperature: float,
) -> torch.Tensor:
    """Smooth maximum of positive-class risks across snapshot environments."""

    if temperature <= 0.0:
        raise ValueError("positive_tail_temperature must be positive.")
    risks: List[torch.Tensor] = []
    for record in records:
        mask = record["labeled_mask"]
        labels = record["labels"][mask]
        positive_logits = record["logits"][mask][labels == 1]
        if positive_logits.numel() > 0:
            risks.append(
                F.binary_cross_entropy_with_logits(
                    positive_logits,
                    torch.ones_like(positive_logits),
                )
            )
    if not risks:
        device = records[0]["logits"].device if records else torch.device("cpu")
        return torch.zeros((), device=device)
    stacked = torch.stack(risks)
    normalizer = temperature * np.log(float(stacked.numel()))
    return temperature * torch.logsumexp(stacked / temperature, dim=0) - normalizer


def recall_aware_cvar_loss(
    records: Sequence[Dict[str, Any]],
    positive_tail_fraction: float,
    negative_tail_fraction: float,
    negative_guard_weight: float,
    environment_temperature: float,
) -> torch.Tensor:
    """Optimize hard positives while guarding against high-scored negatives."""

    if not 0.0 < positive_tail_fraction <= 1.0:
        raise ValueError("positive_tail_fraction must be in (0, 1].")
    if not 0.0 < negative_tail_fraction <= 1.0:
        raise ValueError("negative_tail_fraction must be in (0, 1].")
    if negative_guard_weight < 0.0:
        raise ValueError("negative_guard_weight cannot be negative.")
    if environment_temperature <= 0.0:
        raise ValueError("environment_temperature must be positive.")
    environment_losses: List[torch.Tensor] = []
    for record in records:
        mask = record["labeled_mask"]
        labels = record["labels"][mask]
        logits = record["logits"][mask]
        positive_logits = logits[labels == 1]
        negative_logits = logits[labels == 0]
        if positive_logits.numel() == 0 or negative_logits.numel() == 0:
            continue
        positive_losses = F.softplus(-positive_logits)
        positive_count = max(
            1,
            int(np.ceil(positive_tail_fraction * positive_losses.numel())),
        )
        positive_cvar = torch.topk(
            positive_losses,
            k=positive_count,
            largest=True,
        ).values.mean()
        negative_losses = F.softplus(negative_logits)
        negative_count = max(
            1,
            int(np.ceil(negative_tail_fraction * negative_losses.numel())),
        )
        negative_cvar = torch.topk(
            negative_losses,
            k=negative_count,
            largest=True,
        ).values.mean()
        environment_losses.append(
            positive_cvar + negative_guard_weight * negative_cvar
        )
    if not environment_losses:
        device = records[0]["logits"].device if records else torch.device("cpu")
        return torch.zeros((), device=device)
    stacked = torch.stack(environment_losses)
    normalizer = environment_temperature * np.log(float(stacked.numel()))
    return (
        environment_temperature
        * torch.logsumexp(stacked / environment_temperature, dim=0)
        - normalizer
    )


def cross_environment_prototype_loss(
    records: Sequence[Dict[str, Any]],
    temperature: float,
    positive_margin: float,
    max_samples_per_class: int,
) -> torch.Tensor:
    """Align minority representations with prototypes from other environments."""

    if temperature <= 0.0:
        raise ValueError("prototype temperature must be positive.")
    if positive_margin < 0.0:
        raise ValueError("prototype positive_margin cannot be negative.")
    if max_samples_per_class < 1:
        raise ValueError("prototype max_samples_per_class must be positive.")
    losses: List[torch.Tensor] = []
    for target_index, target_record in enumerate(records):
        target_states = target_record.get("fused_transaction_states")
        if target_states is None:
            continue
        other_positive_states: List[torch.Tensor] = []
        other_negative_states: List[torch.Tensor] = []
        for source_index, source_record in enumerate(records):
            if source_index == target_index:
                continue
            source_states = source_record.get("fused_transaction_states")
            if source_states is None:
                continue
            source_mask = source_record["labeled_mask"]
            source_labels = source_record["labels"][source_mask]
            source_selected_states = source_states[source_mask]
            if (source_labels == 1).any():
                other_positive_states.append(
                    source_selected_states[source_labels == 1]
                )
            if (source_labels == 0).any():
                other_negative_states.append(
                    source_selected_states[source_labels == 0]
                )
        if not other_positive_states or not other_negative_states:
            continue
        positive_prototype = F.normalize(
            torch.cat(other_positive_states, dim=0).mean(dim=0),
            dim=0,
        )
        negative_prototype = F.normalize(
            torch.cat(other_negative_states, dim=0).mean(dim=0),
            dim=0,
        )
        target_mask = target_record["labeled_mask"]
        target_labels = target_record["labels"][target_mask]
        target_selected_states = target_states[target_mask]
        target_selected_logits = target_record["logits"][target_mask]
        positive_indices = torch.nonzero(
            target_labels == 1,
            as_tuple=False,
        ).flatten()
        negative_indices = torch.nonzero(
            target_labels == 0,
            as_tuple=False,
        ).flatten()
        if positive_indices.numel() == 0 or negative_indices.numel() == 0:
            continue
        positive_count = min(max_samples_per_class, positive_indices.numel())
        if positive_indices.numel() > positive_count:
            hard_positive_order = torch.topk(
                target_selected_logits[positive_indices],
                k=positive_count,
                largest=False,
            ).indices
            positive_indices = positive_indices[hard_positive_order]
        negative_count = min(
            max_samples_per_class,
            positive_indices.numel(),
            negative_indices.numel(),
        )
        hard_negative_order = torch.topk(
            target_selected_logits[negative_indices],
            k=negative_count,
            largest=True,
        ).indices
        negative_indices = negative_indices[hard_negative_order]
        selected_indices = torch.cat(
            [positive_indices[:positive_count], negative_indices],
            dim=0,
        )
        selected_labels = target_labels[selected_indices]
        selected_states = F.normalize(
            target_selected_states[selected_indices],
            dim=-1,
        )
        prototype_logits = torch.stack(
            [
                selected_states @ negative_prototype,
                selected_states @ positive_prototype,
            ],
            dim=-1,
        )
        positive_rows = selected_labels == 1
        prototype_logits = prototype_logits.clone()
        prototype_logits[positive_rows, 1] -= positive_margin
        losses.append(
            F.cross_entropy(prototype_logits / temperature, selected_labels)
        )
    if not losses:
        device = records[0]["logits"].device if records else torch.device("cpu")
        return torch.zeros((), device=device)
    return torch.stack(losses).mean()


def temporal_prototype_extrapolation_loss(
    model: TemporalMemoryHGNN,
    records: Sequence[Dict[str, Any]],
    extrapolation_scale: float,
    maximum_shift_ratio: float,
    max_samples_per_class: int,
) -> torch.Tensor:
    """Train the classifier on class prototypes extrapolated toward the future."""
    if extrapolation_scale < 0.0:
        raise ValueError("extrapolation_scale cannot be negative.")
    if maximum_shift_ratio <= 0.0:
        raise ValueError("maximum_shift_ratio must be positive.")
    if max_samples_per_class < 1:
        raise ValueError("max_samples_per_class must be positive.")
    if model.num_environment_experts != 1:
        raise ValueError("Temporal prototype extrapolation requires one classifier.")
    ordered_records = sorted(records, key=lambda record: int(record["time_id"]))
    class_losses: List[torch.Tensor] = []
    for class_id in (0, 1):
        prototype_times: List[float] = []
        prototypes: List[torch.Tensor] = []
        latest_states: Optional[torch.Tensor] = None
        latest_logits: Optional[torch.Tensor] = None
        for record in ordered_records:
            states = record.get("fused_transaction_states")
            if states is None:
                continue
            mask = record["labeled_mask"]
            labels = record["labels"][mask]
            selected_states = states[mask][labels == class_id]
            if selected_states.numel() == 0:
                continue
            prototypes.append(selected_states.mean(dim=0))
            prototype_times.append(float(record["time_id"]))
            latest_states = selected_states
            latest_logits = record["logits"][mask][labels == class_id]
        if len(prototypes) < 2 or latest_states is None or latest_logits is None:
            continue
        time_tensor = torch.tensor(
            prototype_times,
            device=prototypes[0].device,
            dtype=prototypes[0].dtype,
        )
        centered_time = time_tensor - time_tensor.mean()
        prototype_tensor = torch.stack(prototypes)
        slope = (
            centered_time.unsqueeze(1)
            * (prototype_tensor - prototype_tensor.mean(dim=0))
        ).sum(dim=0) / centered_time.square().sum().clamp_min(1e-6)
        detached_shift = extrapolation_scale * slope.detach()
        reference_norm = latest_states.std(dim=0, unbiased=False).norm().clamp_min(
            1e-6
        )
        maximum_norm = maximum_shift_ratio * reference_norm
        shift_scale = torch.clamp(
            maximum_norm / detached_shift.norm().clamp_min(1e-6),
            max=1.0,
        )
        detached_shift = detached_shift * shift_scale
        sample_count = min(max_samples_per_class, latest_states.size(0))
        if latest_states.size(0) > sample_count:
            hard_order = torch.topk(
                latest_logits,
                k=sample_count,
                largest=class_id == 0,
            ).indices
            latest_states = latest_states[hard_order]
        virtual_states = latest_states + detached_shift.unsqueeze(0)
        virtual_logits = model.classifier(virtual_states).squeeze(-1)
        virtual_targets = torch.full_like(virtual_logits, float(class_id))
        class_losses.append(
            F.binary_cross_entropy_with_logits(virtual_logits, virtual_targets)
        )
    if not class_losses:
        device = next(model.parameters()).device
        return torch.zeros((), device=device)
    return torch.stack(class_losses).mean()


def multi_expert_auxiliary_loss(
    model: TemporalMemoryHGNN,
    records: Sequence[Dict[str, Any]],
    environment_by_time: Dict[int, int],
    positive_weight: torch.Tensor,
    specialization_weight: float,
    routing_weight: float,
    global_expert_enabled: bool,
) -> torch.Tensor:
    if model.num_environment_experts <= 1 or not records:
        device = next(model.parameters()).device
        return torch.zeros((), device=device)
    if specialization_weight < 0.0 or routing_weight < 0.0:
        raise ValueError("Multi-expert loss weights cannot be negative.")
    losses: List[torch.Tensor] = []
    for record in records:
        time_id = int(record["time_id"])
        environment_id = int(environment_by_time[time_id])
        components: TemporalTransactionComponents = record["components"]
        gate_logits = components.environment_gate_logits
        if gate_logits is None:
            raise ValueError("Multi-expert records require gate logits.")
        expert_logits = model.expert_logits_from_components(components)
        mask = record["labeled_mask"]
        selected_labels = record["labels"][mask].to(dtype=torch.float32)
        specialization_targets = (
            [0, environment_id]
            if global_expert_enabled and environment_id != 0
            else [environment_id]
        )
        specialization_loss = torch.stack(
            [
                F.binary_cross_entropy_with_logits(
                    expert_logits[mask, expert_id],
                    selected_labels,
                    pos_weight=positive_weight,
                )
                for expert_id in specialization_targets
            ]
        ).mean()
        routing_target = torch.tensor(
            [environment_id],
            device=gate_logits.device,
            dtype=torch.long,
        )
        route_loss = F.cross_entropy(
            gate_logits.unsqueeze(0),
            routing_target,
        )
        losses.append(
            specialization_weight * specialization_loss
            + routing_weight * route_loss
        )
    return torch.stack(losses).mean()


def train_epoch_contiguous_windows(
    model: TemporalMemoryHGNN,
    snapshots: Dict[int, EllipticPPHypergraphSnapshot],
    train_times: Sequence[int],
    global_address_count: int,
    optimizer: torch.optim.Optimizer,
    positive_weight: torch.Tensor,
    num_environments: int,
    risk_estimator: str,
    penalty_weight: float,
    augmentation: Dict[str, Any],
    augmentation_weight: float,
    gradient_clip_norm: float,
    device: torch.device,
) -> Tuple[float, float, float, float, int, int, int, float, float]:
    if bool(augmentation.get("enabled", False)) and augmentation_weight > 0.0:
        raise ValueError(
            "RCHA currently requires environment_definition='snapshot'."
        )
    if num_environments < 2 or num_environments > len(train_times):
        raise ValueError(
            "num_environments must be between two and the number of training snapshots."
        )
    environment_chunks = np.array_split(
        np.asarray(train_times, dtype=np.int64), num_environments
    )
    environment_by_time = {
        int(time_id): environment_id
        for environment_id, chunk in enumerate(environment_chunks)
        for time_id in chunk
    }
    model.train()
    memory_bank = model.initial_memory(global_address_count, device)
    environment_loss_sums = [torch.zeros((), device=device) for _ in environment_chunks]
    environment_labeled = [0 for _ in environment_chunks]
    environment_positive_loss_sums = [
        torch.zeros((), device=device) for _ in environment_chunks
    ]
    environment_negative_loss_sums = [
        torch.zeros((), device=device) for _ in environment_chunks
    ]
    environment_positives = [0 for _ in environment_chunks]
    environment_negatives = [0 for _ in environment_chunks]
    optimizer.zero_grad(set_to_none=True)

    for time_id in train_times:
        snapshot = snapshots[time_id].to(device)
        global_ids = snapshot.global_address_ids
        previous_memory = memory_bank.index_select(0, global_ids)
        logits, updated_memory = model(snapshot, previous_memory)
        mask = snapshot.labeled_mask
        loss_sum = F.binary_cross_entropy_with_logits(
            logits[mask],
            snapshot.labels[mask].to(dtype=torch.float32),
            pos_weight=positive_weight,
            reduction="sum",
        )
        environment_id = environment_by_time[time_id]
        environment_loss_sums[environment_id] = (
            environment_loss_sums[environment_id] + loss_sum
        )
        environment_labeled[environment_id] += int(mask.sum())
        if risk_estimator == "class_balanced":
            selected_labels = snapshot.labels[mask]
            unweighted_losses = F.binary_cross_entropy_with_logits(
                logits[mask],
                selected_labels.to(dtype=torch.float32),
                reduction="none",
            )
            positive_mask = selected_labels == 1
            negative_mask = selected_labels == 0
            environment_positive_loss_sums[environment_id] = (
                environment_positive_loss_sums[environment_id]
                + unweighted_losses[positive_mask].sum()
            )
            environment_negative_loss_sums[environment_id] = (
                environment_negative_loss_sums[environment_id]
                + unweighted_losses[negative_mask].sum()
            )
            environment_positives[environment_id] += int(positive_mask.sum())
            environment_negatives[environment_id] += int(negative_mask.sum())
        with torch.no_grad():
            memory_bank.index_copy_(0, global_ids, updated_memory.detach())

    if risk_estimator == "class_balanced":
        environment_risks = torch.stack(
            [
                0.5
                * (
                    positive_loss_sum / max(positives, 1)
                    + negative_loss_sum / max(negatives, 1)
                )
                for positive_loss_sum, negative_loss_sum, positives, negatives in zip(
                    environment_positive_loss_sums,
                    environment_negative_loss_sums,
                    environment_positives,
                    environment_negatives,
                )
            ]
        )
    else:
        environment_risks = torch.stack(
            [
                loss_sum / max(labeled, 1)
                for loss_sum, labeled in zip(
                    environment_loss_sums, environment_labeled
                )
            ]
        )
    total_labeled = sum(environment_labeled)
    classification_risk = torch.stack(environment_loss_sums).sum() / max(
        total_labeled, 1
    )
    risk_variance = environment_risks.var(unbiased=False)
    objective = classification_risk + penalty_weight * risk_variance
    objective.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
    optimizer.step()
    return (
        float(classification_risk.detach()),
        float(risk_variance.detach()),
        float(objective.detach()),
        0.0,
        0,
        0,
        0,
        0.0,
        0.0,
    )


def train_epoch(
    model: TemporalMemoryHGNN,
    snapshots: Dict[int, EllipticPPHypergraphSnapshot],
    train_times: Sequence[int],
    global_address_count: int,
    optimizer: torch.optim.Optimizer,
    positive_weight: torch.Tensor,
    global_address_labels: Optional[torch.Tensor],
    address_positive_weight: Optional[torch.Tensor],
    address_auxiliary_weight: float,
    snapshots_per_step: int,
    environment_definition: str,
    num_environments: int,
    risk_estimator: str,
    penalty_weight: float,
    drift_robust_state: Optional[Dict[str, Any]],
    augmentation: Dict[str, Any],
    augmentation_weight: float,
    discriminative_learning: Dict[str, Any],
    discriminative_weight: float,
    guidance_scores: Optional[Dict[int, torch.Tensor]],
    guidance_hard_weight: float,
    guidance_gamma: float,
    attribute_prior_logits_by_time: Optional[Dict[int, torch.Tensor]],
    pseudo_attribute_probabilities_by_time: Optional[
        Dict[int, torch.Tensor]
    ],
    offline_pseudo_targets_by_time: Optional[Dict[int, torch.Tensor]],
    offline_pseudo_loss_weight: float,
    residual_l2_weight: float,
    teacher_model: Optional[TemporalMemoryHGNN],
    teacher_decay: float,
    multi_expert_learning: Dict[str, Any],
    multi_expert_weight: float,
    recall_aware_learning: Dict[str, Any],
    recall_aware_weight: float,
    prototype_learning: Dict[str, Any],
    prototype_weight: float,
    gradient_clip_norm: float,
    device: torch.device,
) -> Tuple[
    float, float, float, float, int, int, int, float, float, float, float, float
]:
    if snapshots_per_step < 1:
        raise ValueError("snapshots_per_step must be at least one.")
    if penalty_weight < 0.0:
        raise ValueError("penalty_weight cannot be negative.")
    if risk_estimator not in {"empirical", "class_balanced"}:
        raise ValueError("risk_estimator must be 'empirical' or 'class_balanced'.")
    if residual_l2_weight < 0.0:
        raise ValueError("residual_l2_weight cannot be negative.")
    if address_auxiliary_weight < 0.0:
        raise ValueError("address_auxiliary_weight cannot be negative.")
    if recall_aware_weight < 0.0:
        raise ValueError("recall_aware_weight cannot be negative.")
    if prototype_weight < 0.0:
        raise ValueError("prototype_weight cannot be negative.")
    if offline_pseudo_loss_weight < 0.0:
        raise ValueError("offline_pseudo_loss_weight cannot be negative.")
    if address_auxiliary_weight > 0.0 and (
        global_address_labels is None or address_positive_weight is None
    ):
        raise ValueError("Address auxiliary supervision requires labels and weight.")
    if environment_definition == "contiguous_windows":
        if recall_aware_weight > 0.0:
            raise ValueError(
                "Recall-aware CVaR currently requires "
                "environment_definition='snapshot'."
            )
        if prototype_weight > 0.0:
            raise ValueError(
                "Cross-environment prototype learning currently requires "
                "environment_definition='snapshot'."
            )
        if address_auxiliary_weight > 0.0:
            raise ValueError(
                "Address auxiliary supervision currently requires "
                "environment_definition='snapshot'."
            )
        if attribute_prior_logits_by_time is not None:
            raise ValueError(
                "Attribute-prior residual training currently requires "
                "environment_definition='snapshot'."
            )
        if guidance_scores is not None and guidance_hard_weight > 0.0:
            raise ValueError(
                "OOF guidance currently requires environment_definition='snapshot'."
            )
        contiguous_results = train_epoch_contiguous_windows(
            model=model,
            snapshots=snapshots,
            train_times=train_times,
            global_address_count=global_address_count,
            optimizer=optimizer,
            positive_weight=positive_weight,
            num_environments=num_environments,
            risk_estimator=risk_estimator,
            penalty_weight=penalty_weight,
            augmentation=augmentation,
            augmentation_weight=augmentation_weight,
            gradient_clip_norm=gradient_clip_norm,
            device=device,
        )
        return (*contiguous_results, 0.0, 0.0, 0.0)
    if environment_definition not in {"snapshot", "drift_balanced"}:
        raise ValueError(
            "environment_definition must be 'snapshot', 'drift_balanced', "
            "or 'contiguous_windows'."
        )
    drift_robust_enabled = environment_definition == "drift_balanced"
    if drift_robust_enabled and drift_robust_state is None:
        raise ValueError("drift_balanced environments require DERL state.")
    model.train()
    if teacher_model is not None:
        teacher_model.eval()
    memory_bank = model.initial_memory(global_address_count, device)
    teacher_memory_bank = (
        teacher_model.initial_memory(global_address_count, device)
        if teacher_model is not None
        else None
    )
    total_loss = 0.0
    total_labeled = 0
    total_risk_variance = 0.0
    total_objective = 0.0
    total_augmentation_loss = 0.0
    total_augmented_samples = 0
    total_pseudo_positives = 0
    total_pseudo_negatives = 0
    total_multi_expert_loss = 0.0
    total_residual_penalty = 0.0
    total_address_auxiliary_loss = 0.0
    total_recall_aware_loss = 0.0
    total_prototype_loss = 0.0
    optimizer_steps = 0
    if drift_robust_enabled:
        derl_environment_count = len(drift_robust_state["environment_times"])
        derl_risk_sums = torch.zeros(derl_environment_count, dtype=torch.float64)
        derl_risk_counts = torch.zeros(derl_environment_count, dtype=torch.long)
        derl_weights = drift_robust_state["weights"].to(
            device=device, dtype=torch.float32
        )
        derl_counts = torch.tensor(
            drift_robust_state["environment_counts"],
            device=device,
            dtype=torch.float32,
        )
    else:
        derl_environment_count = 0
        derl_risk_sums = torch.zeros(0, dtype=torch.float64)
        derl_risk_counts = torch.zeros(0, dtype=torch.long)
        derl_weights = torch.zeros(0, device=device)
        derl_counts = torch.ones(0, device=device)

    environment_by_time: Dict[int, int] = {}
    if model.num_environment_experts > 1:
        global_expert_enabled = bool(
            multi_expert_learning.get("global_expert_enabled", False)
        )
        environment_count = (
            model.num_environment_experts - 1
            if global_expert_enabled
            else model.num_environment_experts
        )
        if environment_count < 1:
            raise ValueError("At least one local environment expert is required.")
        environment_chunks = np.array_split(
            np.asarray(train_times, dtype=np.int64),
            environment_count,
        )
        environment_by_time = {
            int(time_id): environment_id + int(global_expert_enabled)
            for environment_id, chunk in enumerate(environment_chunks)
            for time_id in chunk
        }

    for start in range(0, len(train_times), snapshots_per_step):
        group = train_times[start : start + snapshots_per_step]
        group_labeled = sum(int(snapshots[t].labeled_mask.sum()) for t in group)
        optimizer.zero_grad(set_to_none=True)
        group_loss_sum = torch.zeros((), device=device)
        group_offline_pseudo_loss_sum = torch.zeros((), device=device)
        group_offline_pseudo_confidence_sum = torch.zeros((), device=device)
        group_residual_sum = torch.zeros((), device=device)
        group_address_loss_sum = torch.zeros((), device=device)
        group_address_labeled = 0
        environment_risks: List[torch.Tensor] = []
        drift_environment_risks: List[Tuple[int, torch.Tensor]] = []
        augmentation_records: List[Dict[str, Any]] = []
        discriminative_records: List[Dict[str, Any]] = []
        multi_expert_records: List[Dict[str, Any]] = []
        for time_id in group:
            snapshot = snapshots[time_id].to(device)
            global_ids = snapshot.global_address_ids
            previous_memory = memory_bank.index_select(0, global_ids)
            attribute_prior_logits = (
                attribute_prior_logits_by_time[time_id].to(device)
                if attribute_prior_logits_by_time is not None
                else None
            )
            teacher_probabilities: Optional[torch.Tensor] = None
            attribute_probabilities = (
                pseudo_attribute_probabilities_by_time[time_id].to(device)
                if pseudo_attribute_probabilities_by_time is not None
                else None
            )
            offline_pseudo_targets = (
                offline_pseudo_targets_by_time[time_id].to(device)
                if offline_pseudo_targets_by_time is not None
                else None
            )
            if teacher_model is not None:
                if teacher_memory_bank is None:
                    raise RuntimeError("EMA teacher memory was not initialized.")
                with torch.no_grad():
                    teacher_previous_memory = teacher_memory_bank.index_select(
                        0,
                        global_ids,
                    )
                    teacher_logits, teacher_updated_memory = teacher_model(
                        snapshot,
                        teacher_previous_memory,
                        attribute_prior_logits,
                    )
                    teacher_memory_bank.index_copy_(
                        0,
                        global_ids,
                        teacher_updated_memory,
                    )
                    teacher_probabilities = torch.sigmoid(teacher_logits)
            needs_components = bool(augmentation.get("enabled", False)) or (
                model.num_environment_experts > 1
            ) or address_auxiliary_weight > 0.0 or prototype_weight > 0.0
            fused_transaction_states: Optional[torch.Tensor] = None
            if needs_components:
                logits, updated_memory, components = model.forward_with_components(
                    snapshot,
                    previous_memory,
                    attribute_prior_logits,
                )
                component_record = {
                    "time_id": time_id,
                    "snapshot": snapshot,
                    "logits": logits,
                    "labels": snapshot.labels,
                    "labeled_mask": snapshot.labeled_mask,
                    "components": components,
                    "teacher_probabilities": teacher_probabilities,
                    "attribute_probabilities": attribute_probabilities,
                    "offline_pseudo_targets": offline_pseudo_targets,
                }
                fused_transaction_states = components.fused_transaction_states
                if bool(augmentation.get("enabled", False)):
                    augmentation_records.append(component_record)
                if model.num_environment_experts > 1:
                    multi_expert_records.append(component_record)
            else:
                logits, updated_memory = model(
                    snapshot,
                    previous_memory,
                    attribute_prior_logits,
                )
            if address_auxiliary_weight > 0.0:
                if components.address_risk_logits is None:
                    raise RuntimeError(
                        "The model did not return address-risk logits."
                    )
                local_address_labels = global_address_labels.index_select(
                    0, global_ids
                )
                address_mask = local_address_labels >= 0
                selected_address_labels = local_address_labels[address_mask]
                address_losses = F.binary_cross_entropy_with_logits(
                    components.address_risk_logits[address_mask],
                    selected_address_labels.to(dtype=torch.float32),
                    reduction="none",
                )
                address_weights = torch.where(
                    selected_address_labels == 1,
                    address_positive_weight.expand_as(address_losses),
                    torch.ones_like(address_losses),
                )
                group_address_loss_sum = group_address_loss_sum + (
                    address_weights * address_losses
                ).sum()
                group_address_labeled += int(address_mask.sum())
            discriminative_records.append(
                {
                    "time_id": time_id,
                    "logits": logits,
                    "labels": snapshot.labels,
                    "labeled_mask": snapshot.labeled_mask,
                    "fused_transaction_states": fused_transaction_states,
                }
            )
            mask = snapshot.labeled_mask
            selected_labels = snapshot.labels[mask]
            per_sample_losses = F.binary_cross_entropy_with_logits(
                logits[mask],
                selected_labels.to(dtype=torch.float32),
                reduction="none",
            )
            sample_weights = torch.where(
                selected_labels == 1,
                positive_weight.expand_as(per_sample_losses),
                torch.ones_like(per_sample_losses),
            )
            if guidance_scores is not None and guidance_hard_weight > 0.0:
                teacher_scores = guidance_scores[time_id].to(device)[mask]
                guidance_difficulty = torch.where(
                    selected_labels == 1,
                    1.0 - teacher_scores,
                    teacher_scores,
                ).pow(guidance_gamma)
                sample_weights = sample_weights * (
                    1.0 + guidance_hard_weight * guidance_difficulty
                )
            loss_sum = (sample_weights * per_sample_losses).sum()
            if offline_pseudo_targets is not None:
                offline_pseudo_mask = (
                    torch.isfinite(offline_pseudo_targets)
                    & ~snapshot.labeled_mask
                )
                if offline_pseudo_mask.any():
                    offline_targets = offline_pseudo_targets[
                        offline_pseudo_mask
                    ].clamp(0.500001, 1.0)
                    offline_losses = F.binary_cross_entropy_with_logits(
                        logits[offline_pseudo_mask],
                        offline_targets,
                        reduction="none",
                    )
                    offline_confidences = (
                        2.0 * (offline_targets - 0.5)
                    ).clamp_min(0.0)
                    group_offline_pseudo_loss_sum = (
                        group_offline_pseudo_loss_sum
                        + (offline_confidences * offline_losses).sum()
                    )
                    group_offline_pseudo_confidence_sum = (
                        group_offline_pseudo_confidence_sum
                        + offline_confidences.sum()
                    )
            if attribute_prior_logits is not None:
                structural_residual = logits - attribute_prior_logits
                group_residual_sum = group_residual_sum + structural_residual[
                    mask
                ].pow(2).sum()
            labeled = int(mask.sum())
            group_loss_sum = group_loss_sum + loss_sum
            if labeled > 0:
                if risk_estimator == "class_balanced":
                    selected_labels = snapshot.labels[mask]
                    unweighted_losses = F.binary_cross_entropy_with_logits(
                        logits[mask],
                        selected_labels.to(dtype=torch.float32),
                        reduction="none",
                    )
                    positive_losses = unweighted_losses[selected_labels == 1]
                    negative_losses = unweighted_losses[selected_labels == 0]
                    if positive_losses.numel() > 0 and negative_losses.numel() > 0:
                        snapshot_risk = 0.5 * (
                            positive_losses.mean() + negative_losses.mean()
                        )
                        environment_risks.append(snapshot_risk)
                        if drift_robust_enabled:
                            environment_id = drift_robust_state[
                                "environment_by_time"
                            ][int(time_id)]
                            drift_environment_risks.append(
                                (environment_id, snapshot_risk)
                            )
                else:
                    snapshot_risk = loss_sum / labeled
                    environment_risks.append(snapshot_risk)
                    if drift_robust_enabled:
                        environment_id = drift_robust_state[
                            "environment_by_time"
                        ][int(time_id)]
                        drift_environment_risks.append(
                            (environment_id, snapshot_risk)
                        )
            with torch.no_grad():
                memory_bank.index_copy_(0, global_ids, updated_memory.detach())
            total_loss += float(loss_sum.detach())
            total_labeled += labeled
        classification_risk = group_loss_sum / max(group_labeled, 1)
        offline_pseudo_loss = (
            group_offline_pseudo_loss_sum
            / group_offline_pseudo_confidence_sum.clamp_min(1e-12)
            if group_offline_pseudo_confidence_sum.item() > 0.0
            else torch.zeros((), device=device)
        )
        residual_penalty = group_residual_sum / max(group_labeled, 1)
        address_auxiliary_loss = group_address_loss_sum / max(
            group_address_labeled, 1
        )
        if len(environment_risks) > 1:
            stacked_risks = torch.stack(environment_risks)
            risk_variance = stacked_risks.var(unbiased=False)
        else:
            risk_variance = torch.zeros((), device=device)
        if drift_robust_enabled and drift_environment_risks:
            scaled_risks: List[torch.Tensor] = []
            for environment_id, snapshot_risk in drift_environment_risks:
                snapshot_scale = (
                    derl_weights[environment_id]
                    * len(train_times)
                    / derl_counts[environment_id].clamp_min(1.0)
                )
                scaled_risks.append(snapshot_scale * snapshot_risk)
                derl_risk_sums[environment_id] += float(snapshot_risk.detach())
                derl_risk_counts[environment_id] += 1
            drift_robust_loss = torch.stack(scaled_risks).mean()
        else:
            drift_robust_loss = torch.zeros((), device=device)
        if bool(augmentation.get("enabled", False)) and augmentation_weight > 0.0:
            (
                augmentation_loss,
                augmented_samples,
                pseudo_positives,
                pseudo_negatives,
            ) = (
                ccha_augmentation_loss(
                    model,
                    augmentation_records,
                    augmentation,
                )
                if str(augmentation.get("method", "rcha")) == "ccha"
                else rcha_augmentation_loss(
                    model,
                    augmentation_records,
                    augmentation,
                )
            )
        else:
            augmentation_loss = torch.zeros((), device=device)
            augmented_samples = 0
            pseudo_positives = 0
            pseudo_negatives = 0
        if bool(discriminative_learning.get("enabled", False)) and discriminative_weight > 0.0:
            ranking_loss = hard_pair_ranking_loss(
                discriminative_records,
                hard_fraction=float(
                    discriminative_learning.get("hard_fraction", 0.25)
                ),
                margin=float(discriminative_learning.get("margin", 1.0)),
                max_samples=int(
                    discriminative_learning.get("max_hard_samples", 128)
                ),
            )
            positive_tail_loss = positive_environment_tail_loss(
                discriminative_records,
                temperature=float(
                    discriminative_learning.get(
                        "positive_tail_temperature",
                        0.5,
                    )
                ),
            )
            discriminative_loss = (
                float(discriminative_learning.get("ranking_weight", 1.0))
                * ranking_loss
                + float(discriminative_learning.get("positive_tail_weight", 0.25))
                * positive_tail_loss
            )
        else:
            discriminative_loss = torch.zeros((), device=device)
        if (
            bool(multi_expert_learning.get("enabled", False))
            and multi_expert_weight > 0.0
        ):
            multi_expert_loss = multi_expert_auxiliary_loss(
                model,
                multi_expert_records,
                environment_by_time,
                positive_weight,
                specialization_weight=float(
                    multi_expert_learning.get("specialization_weight", 1.0)
                ),
                routing_weight=float(
                    multi_expert_learning.get("routing_weight", 0.25)
                ),
                global_expert_enabled=bool(
                    multi_expert_learning.get("global_expert_enabled", False)
                ),
            )
        else:
            multi_expert_loss = torch.zeros((), device=device)
        if (
            bool(recall_aware_learning.get("enabled", False))
            and recall_aware_weight > 0.0
        ):
            recall_aware_loss = recall_aware_cvar_loss(
                discriminative_records,
                positive_tail_fraction=float(
                    recall_aware_learning.get("positive_tail_fraction", 0.25)
                ),
                negative_tail_fraction=float(
                    recall_aware_learning.get("negative_tail_fraction", 0.05)
                ),
                negative_guard_weight=float(
                    recall_aware_learning.get("negative_guard_weight", 0.10)
                ),
                environment_temperature=float(
                    recall_aware_learning.get("environment_temperature", 0.50)
                ),
            )
        else:
            recall_aware_loss = torch.zeros((), device=device)
        if (
            bool(prototype_learning.get("enabled", False))
            and prototype_weight > 0.0
        ):
            prototype_method = str(
                prototype_learning.get("method", "cross_environment_alignment")
            )
            if prototype_method == "temporal_extrapolation":
                prototype_loss = temporal_prototype_extrapolation_loss(
                    model,
                    discriminative_records,
                    extrapolation_scale=float(
                        prototype_learning.get("extrapolation_scale", 1.0)
                    ),
                    maximum_shift_ratio=float(
                        prototype_learning.get("maximum_shift_ratio", 0.5)
                    ),
                    max_samples_per_class=int(
                        prototype_learning.get("max_samples_per_class", 128)
                    ),
                )
            elif prototype_method == "cross_environment_alignment":
                prototype_loss = cross_environment_prototype_loss(
                    discriminative_records,
                    temperature=float(
                        prototype_learning.get("temperature", 0.20)
                    ),
                    positive_margin=float(
                        prototype_learning.get("positive_margin", 0.10)
                    ),
                    max_samples_per_class=int(
                        prototype_learning.get("max_samples_per_class", 128)
                    ),
                )
            else:
                raise ValueError(
                    "prototype_learning.method must be "
                    "'cross_environment_alignment' or 'temporal_extrapolation'."
                )
        else:
            prototype_loss = torch.zeros((), device=device)
        objective = (
            classification_risk
            + offline_pseudo_loss_weight * offline_pseudo_loss
            + penalty_weight
            * (drift_robust_loss if drift_robust_enabled else risk_variance)
            + augmentation_weight * augmentation_loss
            + discriminative_weight * discriminative_loss
            + multi_expert_weight * multi_expert_loss
            + residual_l2_weight * residual_penalty
            + address_auxiliary_weight * address_auxiliary_loss
            + recall_aware_weight * recall_aware_loss
            + prototype_weight * prototype_loss
        )
        objective.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
        optimizer.step()
        if teacher_model is not None:
            update_ema_teacher(teacher_model, model, teacher_decay)
        total_risk_variance += float(risk_variance.detach())
        total_objective += float(objective.detach())
        total_augmentation_loss += float(augmentation_loss.detach())
        total_augmented_samples += augmented_samples
        total_pseudo_positives += pseudo_positives
        total_pseudo_negatives += pseudo_negatives
        total_multi_expert_loss += float(multi_expert_loss.detach())
        total_residual_penalty += float(residual_penalty.detach())
        total_address_auxiliary_loss += float(
            address_auxiliary_loss.detach()
        )
        total_recall_aware_loss += float(recall_aware_loss.detach())
        total_prototype_loss += float(prototype_loss.detach())
        optimizer_steps += 1

    if drift_robust_enabled:
        if (derl_risk_counts == 0).any():
            missing = torch.nonzero(derl_risk_counts == 0).flatten().tolist()
            raise RuntimeError(f"DERL environments without observed risk: {missing}")
        observed_risks = derl_risk_sums / derl_risk_counts.to(dtype=torch.float64)
        previous_ema = drift_robust_state.get("risk_ema")
        if previous_ema is None:
            risk_ema = observed_risks.to(dtype=torch.float32)
        else:
            decay = float(drift_robust_state["risk_ema_decay"])
            risk_ema = (
                decay * previous_ema.to(dtype=torch.float32)
                + (1.0 - decay) * observed_risks.to(dtype=torch.float32)
            )
        prior_weights = drift_robust_state["prior_weights"].clamp_min(1e-12)
        temperature = float(drift_robust_state["temperature"])
        updated_weights = torch.softmax(
            prior_weights.log() + risk_ema / temperature,
            dim=0,
        )
        drift_robust_state["risk_ema"] = risk_ema
        drift_robust_state["weights"] = updated_weights
        drift_robust_state["last_environment_risks"] = observed_risks.to(
            dtype=torch.float32
        )
        drift_robust_state["last_robust_loss"] = float(
            (updated_weights * observed_risks.to(dtype=torch.float32)).sum()
        )
    return (
        total_loss / max(total_labeled, 1),
        total_risk_variance / max(optimizer_steps, 1),
        total_objective / max(optimizer_steps, 1),
        total_augmentation_loss / max(optimizer_steps, 1),
        total_augmented_samples,
        total_pseudo_positives,
        total_pseudo_negatives,
        total_multi_expert_loss / max(optimizer_steps, 1),
        total_residual_penalty / max(optimizer_steps, 1),
        total_address_auxiliary_loss / max(optimizer_steps, 1),
        total_recall_aware_loss / max(optimizer_steps, 1),
        total_prototype_loss / max(optimizer_steps, 1),
    )


def replay_predictions_by_time(
    model: TemporalMemoryHGNN,
    snapshots: Dict[int, EllipticPPHypergraphSnapshot],
    replay_times: Sequence[int],
    prediction_times: Sequence[int],
    global_address_count: int,
    device: torch.device,
    attribute_prior_logits_by_time: Optional[Dict[int, torch.Tensor]] = None,
) -> Tuple[Dict[int, Tuple[np.ndarray, np.ndarray]], float]:
    model.eval()
    memory_bank = model.initial_memory(global_address_count, device)
    prediction_set = set(prediction_times)
    predictions: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    total_loss = 0.0
    total_labeled = 0

    with torch.no_grad():
        for time_id in replay_times:
            snapshot = snapshots[time_id].to(device)
            global_ids = snapshot.global_address_ids
            previous_memory = memory_bank.index_select(0, global_ids)
            attribute_prior_logits = (
                attribute_prior_logits_by_time[time_id].to(device)
                if attribute_prior_logits_by_time is not None
                else None
            )
            logits, updated_memory = model(
                snapshot,
                previous_memory,
                attribute_prior_logits,
            )
            memory_bank.index_copy_(0, global_ids, updated_memory)
            if time_id not in prediction_set:
                continue
            mask = snapshot.labeled_mask
            selected_labels = snapshot.labels[mask]
            selected_logits = logits[mask]
            predictions[time_id] = (
                selected_labels.cpu().numpy(),
                torch.sigmoid(selected_logits).cpu().numpy(),
            )
            total_loss += float(
                F.binary_cross_entropy_with_logits(
                    selected_logits,
                    selected_labels.to(dtype=torch.float32),
                    reduction="sum",
                )
            )
            total_labeled += int(mask.sum())
    missing = prediction_set.difference(predictions)
    if missing:
        raise ValueError(f"Prediction times were not replayed: {sorted(missing)}")
    return predictions, total_loss / max(total_labeled, 1)


def replay_and_predict(
    model: TemporalMemoryHGNN,
    snapshots: Dict[int, EllipticPPHypergraphSnapshot],
    replay_times: Sequence[int],
    prediction_times: Sequence[int],
    global_address_count: int,
    device: torch.device,
    attribute_prior_logits_by_time: Optional[Dict[int, torch.Tensor]] = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    predictions, loss = replay_predictions_by_time(
        model=model,
        snapshots=snapshots,
        replay_times=replay_times,
        prediction_times=prediction_times,
        global_address_count=global_address_count,
        device=device,
        attribute_prior_logits_by_time=attribute_prior_logits_by_time,
    )
    return (
        np.concatenate([predictions[time_id][0] for time_id in prediction_times]),
        np.concatenate([predictions[time_id][1] for time_id in prediction_times]),
        loss,
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


def macro_snapshot_f1_threshold(
    predictions: Dict[int, Tuple[np.ndarray, np.ndarray]],
    times: Sequence[int],
) -> Tuple[float, float]:
    candidates = np.unique(
        np.concatenate(
            [predictions[time_id][1] for time_id in times]
            + [np.asarray([0.5], dtype=np.float32)]
        )
    )
    f1_by_time: List[np.ndarray] = []
    for time_id in times:
        labels, probabilities = predictions[time_id]
        predicted = probabilities[:, None] >= candidates[None, :]
        positive = labels[:, None] == 1
        true_positives = np.logical_and(predicted, positive).sum(axis=0)
        false_positives = np.logical_and(predicted, ~positive).sum(axis=0)
        false_negatives = np.logical_and(~predicted, positive).sum(axis=0)
        denominator = 2 * true_positives + false_positives + false_negatives
        f1_by_time.append(
            np.divide(
                2 * true_positives,
                denominator,
                out=np.zeros_like(denominator, dtype=np.float64),
                where=denominator > 0,
            )
        )
    macro_f1 = np.stack(f1_by_time).mean(axis=0)
    best_value = float(macro_f1.max())
    best_indices = np.flatnonzero(np.isclose(macro_f1, best_value))
    selected_index = best_indices[
        np.argmin(np.abs(candidates[best_indices] - 0.5))
    ]
    return float(candidates[selected_index]), best_value


def metrics(
    labels: np.ndarray, probabilities: np.ndarray, threshold: float
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
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    train_times = expand_interval(config["split"]["train"])
    validation_times = expand_interval(config["split"]["validation"])
    test_times = expand_interval(config["split"]["test"])
    snapshots, global_address_count = load_snapshots(
        resolve_project_path(config["data"]["processed_root"])
    )
    address_statistics, transaction_statistics = fit_statistics(
        snapshots, train_times
    )
    attribute_prior = dict(config.get("attribute_prior", {}))
    attribute_prior_enabled = bool(attribute_prior.get("enabled", False))
    structural_residual_scale = float(
        attribute_prior.get("structural_residual_scale", 1.0)
    )
    residual_l2_weight = float(attribute_prior.get("residual_l2_weight", 0.0))
    if residual_l2_weight < 0.0:
        raise ValueError("attribute_prior.residual_l2_weight cannot be negative.")
    if not attribute_prior_enabled:
        residual_l2_weight = 0.0
    address_auxiliary = dict(config.get("address_auxiliary", {}))
    address_auxiliary_enabled = bool(address_auxiliary.get("enabled", False))
    address_auxiliary_weight = float(address_auxiliary.get("loss_weight", 0.0))
    if address_auxiliary_weight < 0.0:
        raise ValueError("address_auxiliary.loss_weight cannot be negative.")
    if not address_auxiliary_enabled:
        address_auxiliary_weight = 0.0
    model_config = config["model"]
    model = TemporalMemoryHGNN(
        address_statistics=address_statistics,
        transaction_statistics=transaction_statistics,
        hidden_dim=int(model_config["hidden_dim"]),
        memory_dim=int(model_config["memory_dim"]),
        dropout=float(model_config["dropout"]),
        propagation_layers=int(model_config["propagation_layers"]),
        feature_residual_enabled=bool(
            model_config.get("feature_residual_enabled", False)
        ),
        num_environment_experts=int(
            model_config.get("num_environment_experts", 1)
        ),
        global_expert_weight_floor=float(
            model_config.get("global_expert_weight_floor", 0.0)
        ),
        attribute_prior_residual_enabled=attribute_prior_enabled,
        structural_residual_scale=structural_residual_scale,
        transaction_encoder_type=str(
            model_config.get("transaction_encoder_type", "linear")
        ),
        transaction_interaction_rank=int(
            model_config.get("transaction_interaction_rank", 32)
        ),
        transaction_interaction_layers=int(
            model_config.get("transaction_interaction_layers", 2)
        ),
        address_risk_auxiliary_enabled=address_auxiliary_enabled,
        address_risk_initial_fusion_scale=float(
            address_auxiliary.get("initial_fusion_scale", 0.1)
        ),
        transaction_cross_initial_scale=float(
            model_config.get("transaction_cross_initial_scale", 0.05)
        ),
        attribute_logit_residual_enabled=bool(
            model_config.get("attribute_logit_residual_enabled", False)
        ),
        attribute_logit_initial_scale=float(
            model_config.get("attribute_logit_initial_scale", 0.1)
        ),
        attribute_logit_uncertainty_temperature=float(
            model_config.get("attribute_logit_uncertainty_temperature", 0.0)
        ),
        temporal_memory_enabled=bool(
            model_config.get("temporal_memory_enabled", True)
        ),
        role_aware_propagation=bool(
            model_config.get("role_aware_propagation", True)
        ),
    ).to(device)
    initialization = dict(config.get("initialization", {}))
    initialization_enabled = bool(initialization.get("enabled", False))
    initialization_metadata: Dict[str, Any] = {
        **initialization,
        "enabled": initialization_enabled,
    }
    if initialization_enabled:
        checkpoint_value = initialization.get("checkpoint")
        if not checkpoint_value:
            raise ValueError(
                "initialization.checkpoint is required when initialization is enabled."
            )
        checkpoint_path = resolve_project_path(
            str(checkpoint_value).format(seed=int(config["seed"]))
        )
        initialization_payload = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        checkpoint_seed = initialization_payload.get("seed")
        if checkpoint_seed is not None and int(checkpoint_seed) != int(config["seed"]):
            raise ValueError(
                "Initialization checkpoint seed does not match the training seed."
            )
        model.load_state_dict(initialization_payload["model_state"], strict=True)
        initialization_metadata.update(
            {
                "checkpoint": str(checkpoint_path),
                "method": str(
                    initialization_payload.get(
                        "method", initialization.get("method", "unknown")
                    )
                ),
                "pretraining_epochs": int(
                    initialization_payload.get("epochs", 0)
                ),
                "uses_class_labels": bool(
                    initialization_payload.get("uses_class_labels", True)
                ),
            }
        )
        print(
            "[INIT] loaded "
            f"{initialization_metadata['method']} from {checkpoint_path}",
            flush=True,
        )
    training = config["training"]
    robust_learning = config.get("robust_learning", {})
    robust_enabled = bool(robust_learning.get("enabled", False))
    configured_penalty_weight = (
        float(args.penalty_weight)
        if args.penalty_weight is not None
        else float(robust_learning.get("penalty_weight", 0.0))
    )
    if configured_penalty_weight < 0.0:
        raise ValueError("The robust penalty weight cannot be negative.")
    if not robust_enabled:
        configured_penalty_weight = 0.0
    penalty_warmup_epochs = int(robust_learning.get("warmup_epochs", 0))
    if penalty_warmup_epochs < 0:
        raise ValueError("warmup_epochs cannot be negative.")
    environment_definition = str(
        robust_learning.get("environment_definition", "snapshot")
    )
    num_environments = int(
        robust_learning.get(
            "num_environments",
            int(training["snapshots_per_step"]),
        )
    )
    risk_estimator = str(robust_learning.get("risk_estimator", "empirical"))
    drift_robust_state: Optional[Dict[str, Any]] = None
    if robust_enabled and environment_definition == "drift_balanced":
        drift_robust_state = build_drift_robust_state(
            snapshots=snapshots,
            train_times=train_times,
            num_environments=num_environments,
            minimum_environment_size=int(
                robust_learning.get("minimum_environment_size", 3)
            ),
            temperature=float(robust_learning.get("temperature", 0.5)),
            drift_prior_strength=float(
                robust_learning.get("drift_prior_strength", 1.0)
            ),
            recency_prior_strength=float(
                robust_learning.get("recency_prior_strength", 0.25)
            ),
            risk_ema_decay=float(robust_learning.get("risk_ema_decay", 0.8)),
        )
        print(
            "[DERL] environments="
            f"{drift_robust_state['environment_times']} "
            "prior="
            f"{[round(value, 4) for value in drift_robust_state['prior_weights'].tolist()]}",
            flush=True,
        )
    augmentation = dict(config.get("augmentation", {}))
    augmentation_enabled = bool(augmentation.get("enabled", False))
    augmentation_method = str(augmentation.get("method", "rcha"))
    if augmentation_enabled and augmentation_method not in {
        "rcha",
        "adaptive_rcha",
        "ccha",
    }:
        raise ValueError(
            "augmentation.method must be 'rcha', 'adaptive_rcha', or 'ccha'."
        )
    configured_augmentation_weight = float(
        augmentation.get("loss_weight", 0.0)
    )
    augmentation_warmup_epochs = int(augmentation.get("warmup_epochs", 0))
    augmentation_ramp_epochs = int(augmentation.get("ramp_epochs", 1))
    if configured_augmentation_weight < 0.0:
        raise ValueError("augmentation.loss_weight cannot be negative.")
    if augmentation_warmup_epochs < 0:
        raise ValueError("augmentation.warmup_epochs cannot be negative.")
    if augmentation_ramp_epochs < 1:
        raise ValueError("augmentation.ramp_epochs must be at least one.")
    if not augmentation_enabled:
        configured_augmentation_weight = 0.0
    if augmentation_enabled and environment_definition not in {
        "snapshot",
        "drift_balanced",
    }:
        raise ValueError(
            "Hyperedge augmentation requires snapshot-based or drift-balanced "
            "temporal training."
        )
    teacher_model: Optional[TemporalMemoryHGNN] = None
    teacher_decay = 0.99
    pseudo_attribute_probabilities_by_time: Optional[
        Dict[int, torch.Tensor]
    ] = None
    if augmentation_enabled and augmentation_method == "adaptive_rcha":
        pseudo_labeling = dict(augmentation.get("pseudo_labeling", {}))
        teacher_decay = float(pseudo_labeling.get("teacher_decay", 0.99))
        if not 0.0 <= teacher_decay < 1.0:
            raise ValueError("pseudo_labeling.teacher_decay must be in [0, 1).")
        teacher_model = copy.deepcopy(model).to(device)
        teacher_model.eval()
        for parameter in teacher_model.parameters():
            parameter.requires_grad_(False)
        attribute_scores_path_value = pseudo_labeling.get(
            "attribute_scores_path"
        )
        if attribute_scores_path_value is not None:
            attribute_scores_path = resolve_project_path(
                str(attribute_scores_path_value).format(seed=int(config["seed"]))
            )
            attribute_scores_payload = torch.load(
                attribute_scores_path,
                map_location="cpu",
                weights_only=False,
            )
            if int(attribute_scores_payload["seed"]) != int(config["seed"]):
                raise ValueError(
                    "Quantile-attribute score seed does not match training seed."
                )
            pseudo_attribute_probabilities_by_time = (
                attribute_scores_payload["probabilities_by_time"]
            )
            missing_attribute_times = set(train_times).difference(
                pseudo_attribute_probabilities_by_time
            )
            if missing_attribute_times:
                raise ValueError(
                    "Quantile-attribute scores are missing training time steps: "
                    f"{sorted(missing_attribute_times)}"
                )
            for time_id in train_times:
                if pseudo_attribute_probabilities_by_time[time_id].shape != (
                    snapshots[time_id].num_transactions,
                ):
                    raise ValueError(
                        "Quantile-attribute scores have an invalid shape at "
                        f"time step {time_id}."
                    )
    offline_self_training = dict(config.get("offline_self_training", {}))
    offline_self_training_enabled = bool(
        offline_self_training.get("enabled", False)
    )
    configured_offline_pseudo_loss_weight = float(
        offline_self_training.get("loss_weight", 0.0)
    )
    offline_pseudo_warmup_epochs = int(
        offline_self_training.get("warmup_epochs", 0)
    )
    offline_pseudo_ramp_epochs = int(
        offline_self_training.get("ramp_epochs", 1)
    )
    if configured_offline_pseudo_loss_weight < 0.0:
        raise ValueError("offline_self_training.loss_weight cannot be negative.")
    if offline_pseudo_warmup_epochs < 0:
        raise ValueError("offline_self_training.warmup_epochs cannot be negative.")
    if offline_pseudo_ramp_epochs < 1:
        raise ValueError("offline_self_training.ramp_epochs must be at least one.")
    offline_pseudo_targets_by_time: Optional[Dict[int, torch.Tensor]] = None
    offline_self_training_metadata: Dict[str, Any] = {
        **offline_self_training,
        "enabled": offline_self_training_enabled,
        "loss_weight": configured_offline_pseudo_loss_weight,
        "warmup_epochs": offline_pseudo_warmup_epochs,
        "ramp_epochs": offline_pseudo_ramp_epochs,
    }
    if offline_self_training_enabled:
        payload_path = resolve_project_path(
            str(offline_self_training["labels_path"]).format(
                seed=int(config["seed"])
            )
        )
        payload = torch.load(
            payload_path,
            map_location="cpu",
            weights_only=False,
        )
        if int(payload["seed"]) != int(config["seed"]):
            raise ValueError("Offline pseudo-label seed does not match training seed.")
        if list(payload["train_interval"]) != list(config["split"]["train"]):
            raise ValueError("Offline pseudo-label training interval does not match.")
        if bool(payload.get("uses_test_labels", True)):
            raise ValueError("Offline pseudo-label payload used test labels.")
        if not bool(payload.get("positive_only", False)):
            raise ValueError("Only pseudo-positive offline training is supported.")
        offline_pseudo_targets_by_time = payload["targets_by_time"]
        missing_times = set(train_times).difference(
            offline_pseudo_targets_by_time
        )
        if missing_times:
            raise ValueError(
                "Offline pseudo labels are missing training time steps: "
                f"{sorted(missing_times)}"
            )
        selected_offline_pseudo_positives = 0
        for time_id in train_times:
            targets = offline_pseudo_targets_by_time[time_id]
            snapshot = snapshots[time_id]
            if targets.shape != (snapshot.num_transactions,):
                raise ValueError(
                    f"Offline pseudo labels at time {time_id} have an invalid shape."
                )
            selected = torch.isfinite(targets)
            if (selected & snapshot.labeled_mask.cpu()).any():
                raise ValueError("Offline pseudo labels overlap real labels.")
            if selected.any() and not (
                (targets[selected] > 0.5) & (targets[selected] <= 1.0)
            ).all():
                raise ValueError("Offline pseudo-positive targets must be in (0.5, 1].")
            selected_offline_pseudo_positives += int(selected.sum())
        offline_self_training_metadata.update(
            {
                "labels_path": str(payload_path),
                "method": str(payload.get("method", "unknown")),
                "selected": selected_offline_pseudo_positives,
                "threshold_selection": payload.get("threshold_selection", {}),
                "uses_test_labels": False,
                "positive_only": True,
            }
        )
    else:
        configured_offline_pseudo_loss_weight = 0.0
    discriminative_learning = dict(config.get("discriminative_learning", {}))
    discriminative_enabled = bool(discriminative_learning.get("enabled", False))
    configured_discriminative_weight = float(
        discriminative_learning.get("loss_weight", 0.0)
    )
    discriminative_warmup_epochs = int(
        discriminative_learning.get("warmup_epochs", 0)
    )
    discriminative_ramp_epochs = int(
        discriminative_learning.get("ramp_epochs", 1)
    )
    if configured_discriminative_weight < 0.0:
        raise ValueError("discriminative_learning.loss_weight cannot be negative.")
    if discriminative_warmup_epochs < 0:
        raise ValueError("discriminative_learning.warmup_epochs cannot be negative.")
    if discriminative_ramp_epochs < 1:
        raise ValueError("discriminative_learning.ramp_epochs must be at least one.")
    if not discriminative_enabled:
        configured_discriminative_weight = 0.0
    recall_aware_learning = dict(config.get("recall_aware_learning", {}))
    recall_aware_enabled = bool(recall_aware_learning.get("enabled", False))
    configured_recall_aware_weight = float(
        recall_aware_learning.get("loss_weight", 0.0)
    )
    recall_aware_warmup_epochs = int(
        recall_aware_learning.get("warmup_epochs", 0)
    )
    recall_aware_ramp_epochs = int(
        recall_aware_learning.get("ramp_epochs", 1)
    )
    if configured_recall_aware_weight < 0.0:
        raise ValueError("recall_aware_learning.loss_weight cannot be negative.")
    if recall_aware_warmup_epochs < 0:
        raise ValueError("recall_aware_learning.warmup_epochs cannot be negative.")
    if recall_aware_ramp_epochs < 1:
        raise ValueError("recall_aware_learning.ramp_epochs must be at least one.")
    if not recall_aware_enabled:
        configured_recall_aware_weight = 0.0
    prototype_learning = dict(config.get("prototype_learning", {}))
    prototype_enabled = bool(prototype_learning.get("enabled", False))
    configured_prototype_weight = float(
        prototype_learning.get("loss_weight", 0.0)
    )
    prototype_warmup_epochs = int(prototype_learning.get("warmup_epochs", 0))
    prototype_ramp_epochs = int(prototype_learning.get("ramp_epochs", 1))
    if configured_prototype_weight < 0.0:
        raise ValueError("prototype_learning.loss_weight cannot be negative.")
    if prototype_warmup_epochs < 0:
        raise ValueError("prototype_learning.warmup_epochs cannot be negative.")
    if prototype_ramp_epochs < 1:
        raise ValueError("prototype_learning.ramp_epochs must be at least one.")
    if not prototype_enabled:
        configured_prototype_weight = 0.0
    multi_expert_learning = dict(config.get("multi_expert_learning", {}))
    multi_expert_enabled = bool(multi_expert_learning.get("enabled", False))
    configured_multi_expert_weight = float(
        multi_expert_learning.get("loss_weight", 0.0)
    )
    multi_expert_warmup_epochs = int(
        multi_expert_learning.get("warmup_epochs", 0)
    )
    multi_expert_ramp_epochs = int(
        multi_expert_learning.get("ramp_epochs", 1)
    )
    if configured_multi_expert_weight < 0.0:
        raise ValueError("multi_expert_learning.loss_weight cannot be negative.")
    if multi_expert_warmup_epochs < 0:
        raise ValueError("multi_expert_learning.warmup_epochs cannot be negative.")
    if multi_expert_ramp_epochs < 1:
        raise ValueError("multi_expert_learning.ramp_epochs must be at least one.")
    if model.num_environment_experts > 1 and not multi_expert_enabled:
        raise ValueError(
            "Multiple environment experts require multi_expert_learning.enabled."
        )
    if multi_expert_enabled and model.num_environment_experts <= 1:
        raise ValueError(
            "multi_expert_learning requires model.num_environment_experts > 1."
        )
    if not multi_expert_enabled:
        configured_multi_expert_weight = 0.0
    residual_guidance = dict(config.get("residual_guidance", {}))
    guidance_enabled = bool(residual_guidance.get("enabled", False))
    guidance_hard_weight = float(residual_guidance.get("hard_weight", 0.0))
    guidance_gamma = float(residual_guidance.get("difficulty_gamma", 2.0))
    if guidance_hard_weight < 0.0:
        raise ValueError("residual_guidance.hard_weight cannot be negative.")
    if guidance_gamma < 0.0:
        raise ValueError("residual_guidance.difficulty_gamma cannot be negative.")
    guidance_scores: Optional[Dict[int, torch.Tensor]] = None
    if guidance_enabled and guidance_hard_weight > 0.0:
        guidance_path_value = str(residual_guidance["scores_path"]).format(
            seed=int(config["seed"])
        )
        guidance_payload = torch.load(
            resolve_project_path(guidance_path_value),
            map_location="cpu",
            weights_only=False,
        )
        if int(guidance_payload["seed"]) != int(config["seed"]):
            raise ValueError("OOF guidance seed does not match the training seed.")
        guidance_scores = guidance_payload["scores_by_time"]
        missing_guidance = set(train_times).difference(guidance_scores)
        if missing_guidance:
            raise ValueError(
                f"OOF guidance is missing time steps: {sorted(missing_guidance)}"
            )
    attribute_prior_logits_by_time: Optional[Dict[int, torch.Tensor]] = None
    attribute_prior_metadata: Dict[str, Any] = {
        "enabled": attribute_prior_enabled,
        "residual_l2_weight": residual_l2_weight,
        "structural_residual_scale": structural_residual_scale,
    }
    if attribute_prior_enabled:
        prior_path_value = str(attribute_prior["scores_path"]).format(
            seed=int(config["seed"])
        )
        prior_path = resolve_project_path(prior_path_value)
        prior_payload = torch.load(
            prior_path,
            map_location="cpu",
            weights_only=False,
        )
        if int(prior_payload["seed"]) != int(config["seed"]):
            raise ValueError("Attribute-prior seed does not match the training seed.")
        attribute_prior_logits_by_time = prior_payload["logits_by_time"]
        required_prior_times = set(train_times + validation_times + test_times)
        missing_prior_times = required_prior_times.difference(
            attribute_prior_logits_by_time
        )
        if missing_prior_times:
            raise ValueError(
                "Attribute prior is missing time steps: "
                f"{sorted(missing_prior_times)}"
            )
        for time_id in required_prior_times:
            if attribute_prior_logits_by_time[time_id].shape != (
                snapshots[time_id].num_transactions,
            ):
                raise ValueError(
                    f"Attribute prior at time {time_id} has an invalid shape."
                )
        attribute_prior_metadata.update(
            {
                "scores_path": str(prior_path),
                "method": str(prior_payload.get("method", "unknown")),
                "folds": int(prior_payload.get("folds", 0)),
                "n_estimators": int(prior_payload.get("n_estimators", 0)),
            }
        )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    positives, negatives = training_class_counts(snapshots, train_times)
    positive_weight = torch.tensor(negatives / positives, device=device)
    global_address_labels: Optional[torch.Tensor] = None
    address_positive_weight: Optional[torch.Tensor] = None
    address_auxiliary_metadata: Dict[str, Any] = {
        **address_auxiliary,
        "enabled": address_auxiliary_enabled,
        "loss_weight": address_auxiliary_weight,
    }
    if address_auxiliary_enabled:
        address_label_path = resolve_project_path(
            address_auxiliary.get(
                "labels_path",
                Path(config["data"]["processed_root"]) / "address_index.csv.gz",
            )
        )
        global_address_labels = load_global_address_labels(
            address_label_path,
            global_address_count,
        ).to(device)
        training_address_ids = torch.unique(
            torch.cat(
                [snapshots[time_id].global_address_ids for time_id in train_times]
            )
        ).to(device)
        training_address_labels = global_address_labels.index_select(
            0, training_address_ids
        )
        labeled_training_addresses = training_address_labels >= 0
        address_positives = int(
            (training_address_labels[labeled_training_addresses] == 1).sum()
        )
        address_negatives = int(
            (training_address_labels[labeled_training_addresses] == 0).sum()
        )
        if address_positives == 0 or address_negatives == 0:
            raise ValueError("Training addresses require both labeled classes.")
        raw_address_positive_weight = address_negatives / address_positives
        positive_weight_cap = float(
            address_auxiliary.get("positive_weight_cap", raw_address_positive_weight)
        )
        address_positive_weight = torch.tensor(
            min(raw_address_positive_weight, positive_weight_cap),
            device=device,
        )
        address_auxiliary_metadata.update(
            {
                "labels_path": str(address_label_path),
                "training_positive_addresses": address_positives,
                "training_negative_addresses": address_negatives,
                "positive_weight": float(address_positive_weight),
                "raw_positive_weight": raw_address_positive_weight,
            }
        )
    fixed_refit = args.fixed_epochs is not None
    if fixed_refit:
        if int(args.fixed_epochs) < 1:
            raise ValueError("fixed_epochs must be at least one.")
        if args.fixed_threshold is None:
            raise ValueError("fixed_threshold is required for a fixed refit.")
        if not np.isfinite(float(args.fixed_threshold)):
            raise ValueError("fixed_threshold must be finite.")
        max_epochs = int(args.fixed_epochs)
    else:
        max_epochs = (
            int(args.max_epochs)
            if args.max_epochs is not None
            else int(training["max_epochs"])
        )
    patience = int(training["patience"])
    min_delta = float(training["min_delta"])
    best_score = -float("inf")
    best_epoch = 0
    best_state: Optional[Dict[str, torch.Tensor]] = None
    epochs_without_improvement = 0
    history: List[Dict[str, float]] = []
    start_time = time.perf_counter()

    if attribute_prior_enabled and not fixed_refit:
        (
            initial_validation_labels,
            initial_validation_probabilities,
            initial_validation_loss,
        ) = replay_and_predict(
            model=model,
            snapshots=snapshots,
            replay_times=train_times + validation_times,
            prediction_times=validation_times,
            global_address_count=global_address_count,
            device=device,
            attribute_prior_logits_by_time=attribute_prior_logits_by_time,
        )
        initial_validation_ap = float(
            average_precision_score(
                initial_validation_labels,
                initial_validation_probabilities,
            )
        )
        best_score = initial_validation_ap
        best_epoch = 0
        best_state = copy.deepcopy(model.state_dict())
        attribute_prior_metadata.update(
            {
                "initial_validation_average_precision": initial_validation_ap,
                "initial_validation_loss": initial_validation_loss,
            }
        )
        print(
            "[attribute_prior] epoch=000 "
            f"val_loss={initial_validation_loss:.6f} "
            f"val_ap={initial_validation_ap:.6f}",
            flush=True,
        )

    for epoch in range(1, max_epochs + 1):
        effective_penalty_weight = (
            configured_penalty_weight if epoch > penalty_warmup_epochs else 0.0
        )
        if epoch <= augmentation_warmup_epochs:
            effective_augmentation_weight = 0.0
        else:
            augmentation_progress = min(
                1.0,
                (epoch - augmentation_warmup_epochs) / augmentation_ramp_epochs,
            )
            effective_augmentation_weight = (
                configured_augmentation_weight * augmentation_progress
            )
        if epoch <= offline_pseudo_warmup_epochs:
            effective_offline_pseudo_loss_weight = 0.0
        else:
            offline_pseudo_progress = min(
                1.0,
                (epoch - offline_pseudo_warmup_epochs)
                / offline_pseudo_ramp_epochs,
            )
            effective_offline_pseudo_loss_weight = (
                configured_offline_pseudo_loss_weight
                * offline_pseudo_progress
            )
        if epoch <= discriminative_warmup_epochs:
            effective_discriminative_weight = 0.0
        else:
            discriminative_progress = min(
                1.0,
                (epoch - discriminative_warmup_epochs)
                / discriminative_ramp_epochs,
            )
            effective_discriminative_weight = (
                configured_discriminative_weight * discriminative_progress
            )
        if epoch <= multi_expert_warmup_epochs:
            effective_multi_expert_weight = 0.0
        else:
            multi_expert_progress = min(
                1.0,
                (epoch - multi_expert_warmup_epochs)
                / multi_expert_ramp_epochs,
            )
            effective_multi_expert_weight = (
                configured_multi_expert_weight * multi_expert_progress
            )
        if epoch <= recall_aware_warmup_epochs:
            effective_recall_aware_weight = 0.0
        else:
            recall_aware_progress = min(
                1.0,
                (epoch - recall_aware_warmup_epochs)
                / recall_aware_ramp_epochs,
            )
            effective_recall_aware_weight = (
                configured_recall_aware_weight * recall_aware_progress
            )
        if epoch <= prototype_warmup_epochs:
            effective_prototype_weight = 0.0
        else:
            prototype_progress = min(
                1.0,
                (epoch - prototype_warmup_epochs) / prototype_ramp_epochs,
            )
            effective_prototype_weight = (
                configured_prototype_weight * prototype_progress
            )
        (
            train_loss,
            train_risk_variance,
            train_objective,
            train_augmentation_loss,
            train_augmented_samples,
            train_pseudo_positives,
            train_pseudo_negatives,
            train_multi_expert_loss,
            train_residual_penalty,
            train_address_auxiliary_loss,
            train_recall_aware_loss,
            train_prototype_loss,
        ) = train_epoch(
            model=model,
            snapshots=snapshots,
            train_times=train_times,
            global_address_count=global_address_count,
            optimizer=optimizer,
            positive_weight=positive_weight,
            global_address_labels=global_address_labels,
            address_positive_weight=address_positive_weight,
            address_auxiliary_weight=address_auxiliary_weight,
            snapshots_per_step=int(training["snapshots_per_step"]),
            environment_definition=environment_definition,
            num_environments=num_environments,
            risk_estimator=risk_estimator,
            penalty_weight=effective_penalty_weight,
            drift_robust_state=drift_robust_state,
            augmentation=augmentation,
            augmentation_weight=effective_augmentation_weight,
            discriminative_learning=discriminative_learning,
            discriminative_weight=effective_discriminative_weight,
            guidance_scores=guidance_scores,
            guidance_hard_weight=guidance_hard_weight,
            guidance_gamma=guidance_gamma,
            attribute_prior_logits_by_time=attribute_prior_logits_by_time,
            pseudo_attribute_probabilities_by_time=(
                pseudo_attribute_probabilities_by_time
            ),
            offline_pseudo_targets_by_time=offline_pseudo_targets_by_time,
            offline_pseudo_loss_weight=(
                effective_offline_pseudo_loss_weight
            ),
            residual_l2_weight=residual_l2_weight,
            teacher_model=teacher_model,
            teacher_decay=teacher_decay,
            multi_expert_learning=multi_expert_learning,
            multi_expert_weight=effective_multi_expert_weight,
            recall_aware_learning=recall_aware_learning,
            recall_aware_weight=effective_recall_aware_weight,
            prototype_learning=prototype_learning,
            prototype_weight=effective_prototype_weight,
            gradient_clip_norm=float(training["gradient_clip_norm"]),
            device=device,
        )
        epoch_record = {
                "epoch": float(epoch),
                "train_weighted_loss": train_loss,
                "train_environment_risk_variance": train_risk_variance,
                "train_objective": train_objective,
                "penalty_weight": effective_penalty_weight,
                "train_augmentation_loss": train_augmentation_loss,
                "augmentation_weight": effective_augmentation_weight,
                "offline_pseudo_loss_weight": (
                    effective_offline_pseudo_loss_weight
                ),
                "augmented_samples": float(train_augmented_samples),
                "pseudo_positives": float(train_pseudo_positives),
                "pseudo_negatives": float(train_pseudo_negatives),
                "train_multi_expert_loss": train_multi_expert_loss,
                "multi_expert_weight": effective_multi_expert_weight,
                "train_structural_residual_penalty": train_residual_penalty,
                "structural_residual_penalty_weight": residual_l2_weight,
                "train_address_auxiliary_loss": train_address_auxiliary_loss,
                "address_auxiliary_weight": address_auxiliary_weight,
                "train_recall_aware_loss": train_recall_aware_loss,
                "recall_aware_weight": effective_recall_aware_weight,
                "train_prototype_loss": train_prototype_loss,
                "prototype_weight": effective_prototype_weight,
            }
        if drift_robust_state is not None:
            epoch_record.update(
                {
                    "train_derl_loss": float(
                        drift_robust_state["last_robust_loss"]
                    ),
                    "derl_environment_weights": drift_robust_state[
                        "weights"
                    ].tolist(),
                    "derl_environment_risks": drift_robust_state[
                        "last_environment_risks"
                    ].tolist(),
                }
            )
        if fixed_refit:
            history.append(epoch_record)
            print(
                f"[temporal_memory_hgnn_refit] epoch={epoch:03d} "
                f"train_loss={train_loss:.6f} "
                f"risk_var={train_risk_variance:.6f} "
                f"beta={effective_penalty_weight:g} "
                f"aug_loss={train_augmentation_loss:.6f} "
                f"aug_weight={effective_augmentation_weight:g} "
                f"aug_n={train_augmented_samples}",
                flush=True,
            )
            continue
        validation_labels, validation_probabilities, validation_loss = replay_and_predict(
            model=model,
            snapshots=snapshots,
            replay_times=train_times + validation_times,
            prediction_times=validation_times,
            global_address_count=global_address_count,
            device=device,
            attribute_prior_logits_by_time=attribute_prior_logits_by_time,
        )
        validation_ap = float(
            average_precision_score(validation_labels, validation_probabilities)
        )
        epoch_record.update(
            {
                "validation_loss": validation_loss,
                "validation_average_precision": validation_ap,
            }
        )
        history.append(epoch_record)
        derl_log = (
            f"derl_loss={drift_robust_state['last_robust_loss']:.6f} "
            if drift_robust_state is not None
            else ""
        )
        print(
            f"[temporal_memory_hgnn] epoch={epoch:03d} "
            f"train_loss={train_loss:.6f} risk_var={train_risk_variance:.6f} "
            f"beta={effective_penalty_weight:g} "
            f"aug_loss={train_augmentation_loss:.6f} "
            f"aug_weight={effective_augmentation_weight:g} "
            f"aug_n={train_augmented_samples} "
            f"pseudo_pos={train_pseudo_positives} "
            f"pseudo_neg={train_pseudo_negatives} "
            f"moe_loss={train_multi_expert_loss:.6f} "
            f"moe_weight={effective_multi_expert_weight:g} "
            f"residual_l2={train_residual_penalty:.6f} "
            f"address_loss={train_address_auxiliary_loss:.6f} "
            f"address_weight={address_auxiliary_weight:g} "
            f"recall_loss={train_recall_aware_loss:.6f} "
            f"recall_weight={effective_recall_aware_weight:g} "
            f"prototype_loss={train_prototype_loss:.6f} "
            f"prototype_weight={effective_prototype_weight:g} "
            f"{derl_log}"
            f"val_loss={validation_loss:.6f} "
            f"val_ap={validation_ap:.6f}",
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

    if fixed_refit:
        best_state = copy.deepcopy(model.state_dict())
        best_epoch = max_epochs
        threshold = float(args.fixed_threshold)
        validation_metrics: Optional[Dict[str, Any]] = {
            "evaluated": False,
            "selection_source": "rolling_out_of_fold_predictions",
            "threshold": threshold,
        }
    else:
        if best_state is None:
            raise RuntimeError("No temporal-memory checkpoint was selected.")
        model.load_state_dict(best_state)
        validation_predictions, validation_loss = replay_predictions_by_time(
            model=model,
            snapshots=snapshots,
            replay_times=train_times + validation_times,
            prediction_times=validation_times,
            global_address_count=global_address_count,
            device=device,
            attribute_prior_logits_by_time=attribute_prior_logits_by_time,
        )
        validation_labels = np.concatenate(
            [validation_predictions[time_id][0] for time_id in validation_times]
        )
        validation_probabilities = np.concatenate(
            [validation_predictions[time_id][1] for time_id in validation_times]
        )
        threshold, validation_macro_f1 = macro_snapshot_f1_threshold(
            validation_predictions,
            validation_times,
        )
        validation_metrics = metrics(
            validation_labels, validation_probabilities, threshold
        )
        validation_metrics["loss"] = validation_loss
        validation_metrics["macro_snapshot_f1"] = validation_macro_f1
        validation_metrics["threshold_method"] = "mean_validation_snapshot_f1"
    model.load_state_dict(best_state)
    test_metrics: Optional[Dict[str, Any]] = None
    if not args.skip_test:
        replay_times = sorted(set(train_times + validation_times + test_times))
        test_labels, test_probabilities, test_loss = replay_and_predict(
            model=model,
            snapshots=snapshots,
            replay_times=replay_times,
            prediction_times=test_times,
            global_address_count=global_address_count,
            device=device,
            attribute_prior_logits_by_time=attribute_prior_logits_by_time,
        )
        test_metrics = metrics(test_labels, test_probabilities, threshold)
        test_metrics["loss"] = test_loss

    output_root = (
        resolve_project_path(args.output_root)
        if args.output_root is not None
        else resolve_project_path(config["output_root"])
    )
    output_root.mkdir(parents=True, exist_ok=True)
    if augmentation_enabled and configured_augmentation_weight > 0.0:
        augmentation_suffix = {
            "adaptive_rcha": "adaptive_rcha",
            "rcha": "rcha",
            "ccha": "ccha",
        }[augmentation_method]
        robust_suffix = "_erl" if robust_enabled and configured_penalty_weight > 0.0 else ""
        model_name = f"temporal_memory_hgnn{robust_suffix}_{augmentation_suffix}"
    elif robust_enabled and configured_penalty_weight > 0.0:
        model_name = "temporal_memory_hgnn_erl"
    else:
        model_name = "temporal_memory_hgnn"
    if drift_robust_state is not None:
        model_name = model_name.replace("_erl", "_derl")
    if model.num_environment_experts > 1:
        model_name = f"{model_name}_moe"
    if attribute_prior_enabled:
        model_name = f"{model_name}_attribute_prior_residual"
    if address_auxiliary_enabled:
        model_name = f"{model_name}_address_aux"
    if recall_aware_enabled and configured_recall_aware_weight > 0.0:
        model_name = f"{model_name}_recall_cvar"
    if prototype_enabled and configured_prototype_weight > 0.0:
        model_name = f"{model_name}_prototype"
    if model.transaction_encoder_type != "linear":
        model_name = f"{model_name}_{model.transaction_encoder_type}_encoder"
    if model.attribute_logit_residual_enabled:
        model_name = f"{model_name}_attribute_logit_residual"
    if not model.temporal_memory_enabled:
        model_name = f"{model_name}_no_temporal_memory"
    if not model.role_aware_propagation:
        model_name = f"{model_name}_role_agnostic"
    if (
        offline_self_training_enabled
        and configured_offline_pseudo_loss_weight > 0.0
    ):
        model_name = f"{model_name}_offline_self_training"
    torch.save(
        {
            "model_name": model_name,
            "model_state": best_state,
            "address_statistics": address_statistics.as_dict(),
            "transaction_statistics": transaction_statistics.as_dict(),
            "best_epoch": best_epoch,
            "validation_threshold": threshold,
            "robust_learning": {
                "enabled": robust_enabled,
                "method": (
                    "drift_group_dro"
                    if drift_robust_state is not None
                    else "risk_variance"
                ),
                "penalty_weight": configured_penalty_weight,
                "warmup_epochs": penalty_warmup_epochs,
                "environment_definition": robust_learning.get(
                    "environment_definition", "snapshot"
                ),
                "num_environments": num_environments,
                "risk_estimator": risk_estimator,
                "drift_robust_state": drift_robust_metadata(
                    drift_robust_state
                ),
            },
            "augmentation": {
                **augmentation,
                "enabled": augmentation_enabled,
                "loss_weight": configured_augmentation_weight,
                "warmup_epochs": augmentation_warmup_epochs,
                "ramp_epochs": augmentation_ramp_epochs,
            },
            "offline_self_training": offline_self_training_metadata,
            "initialization": initialization_metadata,
            "attribute_prior": attribute_prior_metadata,
            "address_auxiliary": address_auxiliary_metadata,
            "recall_aware_learning": {
                **recall_aware_learning,
                "enabled": recall_aware_enabled,
                "loss_weight": configured_recall_aware_weight,
                "warmup_epochs": recall_aware_warmup_epochs,
                "ramp_epochs": recall_aware_ramp_epochs,
            },
            "prototype_learning": {
                **prototype_learning,
                "enabled": prototype_enabled,
                "loss_weight": configured_prototype_weight,
                "warmup_epochs": prototype_warmup_epochs,
                "ramp_epochs": prototype_ramp_epochs,
            },
            "config": config,
        },
        output_root / "best.pt",
    )
    result: Dict[str, Any] = {
        "model": model_name,
        "seed": int(config["seed"]),
        "best_epoch": best_epoch,
        "epochs_ran": len(history),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "model_config": dict(model_config),
        "positive_weight": float(positive_weight),
        "snapshots_per_step": int(training["snapshots_per_step"]),
        "robust_learning": {
            "enabled": robust_enabled,
            "method": (
                "drift_group_dro"
                if drift_robust_state is not None
                else "risk_variance"
            ),
            "penalty_weight": configured_penalty_weight,
            "warmup_epochs": penalty_warmup_epochs,
            "environment_definition": robust_learning.get(
                "environment_definition", "snapshot"
            ),
            "num_environments": num_environments,
            "risk_estimator": risk_estimator,
            "drift_robust_state": drift_robust_metadata(drift_robust_state),
        },
        "augmentation": {
            **augmentation,
            "enabled": augmentation_enabled,
            "loss_weight": configured_augmentation_weight,
            "warmup_epochs": augmentation_warmup_epochs,
            "ramp_epochs": augmentation_ramp_epochs,
        },
        "offline_self_training": offline_self_training_metadata,
        "initialization": initialization_metadata,
        "attribute_prior": attribute_prior_metadata,
        "address_auxiliary": address_auxiliary_metadata,
        "recall_aware_learning": {
            **recall_aware_learning,
            "enabled": recall_aware_enabled,
            "loss_weight": configured_recall_aware_weight,
            "warmup_epochs": recall_aware_warmup_epochs,
            "ramp_epochs": recall_aware_ramp_epochs,
        },
        "prototype_learning": {
            **prototype_learning,
            "enabled": prototype_enabled,
            "loss_weight": configured_prototype_weight,
            "warmup_epochs": prototype_warmup_epochs,
            "ramp_epochs": prototype_ramp_epochs,
        },
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
    with (output_root / "metrics.json").open("w", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)

    comparison: Dict[str, Any] = {"temporal_memory": result}
    baseline_path = (
        resolve_project_path(args.baseline_comparison)
        if args.baseline_comparison is not None
        else resolve_project_path(config["baseline_comparison"])
    )
    if baseline_path.is_file():
        with baseline_path.open("r", encoding="utf-8") as stream:
            comparison["baselines"] = json.load(stream)["results"]
    with (output_root / "comparison_with_baselines.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(comparison, stream, ensure_ascii=False, indent=2)

    summary = {
        "status": "PASS",
        "model": result["model"],
        "best_epoch": best_epoch,
        "epochs_ran": len(history),
        "parameters": result["parameters"],
        "duration_seconds": result["duration_seconds"],
        "peak_gpu_memory_mb": result["peak_gpu_memory_mb"],
        "validation": validation_metrics,
        "test": test_metrics,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
