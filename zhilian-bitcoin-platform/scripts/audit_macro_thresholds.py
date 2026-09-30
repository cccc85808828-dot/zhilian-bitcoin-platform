from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import joblib
import numpy as np
import torch
import yaml
from sklearn.metrics import (
    average_precision_score,
    precision_recall_fscore_support,
    roc_auc_score,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import train_main_baselines as baseline_training  # noqa: E402
import train_temporal_memory as temporal_training  # noqa: E402
from tthgnn_erl.baselines import EvolveGCNOBaseline, FeatureStatistics  # noqa: E402
from tthgnn_erl.ellipticpp import EllipticPPHypergraphSnapshot  # noqa: E402
from tthgnn_erl.temporal import TemporalMemoryHGNN  # noqa: E402


MODEL_ORDER = [
    "lr",
    "xgboost",
    "gcn",
    "gat",
    "hgnn",
    "evolvegcn_o",
    "hgt",
    "graph_transformer",
    "temporal_memory_hgnn_erl",
]
METRICS = ["precision", "accuracy", "recall", "f1"]
PredictionByTime = Dict[int, Tuple[np.ndarray, np.ndarray]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Re-evaluate saved main-experiment checkpoints with a threshold that "
            "maximizes mean validation F1 across time steps."
        )
    )
    parser.add_argument(
        "--experiment-root",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "main_experiment",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=(
            PROJECT_ROOT
            / "artifacts"
            / "main_experiment"
            / "macro_threshold_audit"
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def feature_statistics(payload: Mapping[str, torch.Tensor]) -> FeatureStatistics:
    return FeatureStatistics(
        mean=payload["mean"].detach().cpu(),
        std=payload["std"].detach().cpu(),
        count=payload["count"].detach().cpu(),
    )


def classification_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> Dict[str, Any]:
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
        "predicted_positives": int(predictions.sum()),
    }


def concatenate_predictions(
    predictions: PredictionByTime,
    times: Sequence[int],
) -> Tuple[np.ndarray, np.ndarray]:
    labels = [predictions[time_id][0] for time_id in times]
    probabilities = [predictions[time_id][1] for time_id in times]
    return np.concatenate(labels), np.concatenate(probabilities)


def f1_at_thresholds(
    labels: np.ndarray,
    probabilities: np.ndarray,
    thresholds: np.ndarray,
) -> np.ndarray:
    order = np.argsort(-probabilities, kind="mergesort")
    sorted_probabilities = probabilities[order]
    sorted_labels = labels[order].astype(np.int64, copy=False)
    cumulative_true_positives = np.cumsum(sorted_labels, dtype=np.int64)
    predicted_counts = np.searchsorted(
        -sorted_probabilities,
        -thresholds,
        side="right",
    )
    true_positives = np.zeros_like(predicted_counts, dtype=np.float64)
    nonzero = predicted_counts > 0
    true_positives[nonzero] = cumulative_true_positives[
        predicted_counts[nonzero] - 1
    ]
    false_positives = predicted_counts.astype(np.float64) - true_positives
    false_negatives = float(sorted_labels.sum()) - true_positives
    denominator = 2.0 * true_positives + false_positives + false_negatives
    return np.divide(
        2.0 * true_positives,
        denominator,
        out=np.zeros_like(true_positives),
        where=denominator > 0.0,
    )


def macro_f1_threshold(
    predictions: PredictionByTime,
    validation_times: Sequence[int],
) -> Tuple[float, float]:
    candidate_parts = [
        predictions[time_id][1].astype(np.float64, copy=False)
        for time_id in validation_times
    ]
    candidates = np.unique(np.concatenate(candidate_parts + [np.asarray([0.5])]))
    per_time_f1 = np.stack(
        [
            f1_at_thresholds(
                predictions[time_id][0],
                predictions[time_id][1],
                candidates,
            )
            for time_id in validation_times
        ],
        axis=0,
    )
    macro_f1 = per_time_f1.mean(axis=0)
    best_value = float(macro_f1.max())
    best_indices = np.flatnonzero(np.isclose(macro_f1, best_value))
    selected_index = best_indices[
        np.argmin(np.abs(candidates[best_indices] - 0.5))
    ]
    return float(candidates[selected_index]), best_value


def evaluate_predictions(
    predictions: PredictionByTime,
    validation_times: Sequence[int],
    test_times: Sequence[int],
    original_threshold: float,
) -> Dict[str, Any]:
    threshold, validation_macro_f1 = macro_f1_threshold(
        predictions,
        validation_times,
    )
    validation_labels, validation_probabilities = concatenate_predictions(
        predictions,
        validation_times,
    )
    test_labels, test_probabilities = concatenate_predictions(
        predictions,
        test_times,
    )
    per_time_validation = {
        str(time_id): classification_metrics(
            predictions[time_id][0],
            predictions[time_id][1],
            threshold,
        )
        for time_id in validation_times
    }
    per_time_test = {
        str(time_id): classification_metrics(
            predictions[time_id][0],
            predictions[time_id][1],
            threshold,
        )
        for time_id in test_times
    }
    return {
        "threshold_selection": {
            "method": "mean_validation_snapshot_f1",
            "threshold": threshold,
            "validation_macro_f1": validation_macro_f1,
            "original_pooled_validation_f1_threshold": float(original_threshold),
        },
        "validation": classification_metrics(
            validation_labels,
            validation_probabilities,
            threshold,
        ),
        "test": classification_metrics(
            test_labels,
            test_probabilities,
            threshold,
        ),
        "per_time_validation": per_time_validation,
        "per_time_test": per_time_test,
    }


def predict_classical_by_time(
    model: Any,
    snapshots: Mapping[int, EllipticPPHypergraphSnapshot],
    times: Iterable[int],
    statistics: FeatureStatistics,
) -> PredictionByTime:
    result: PredictionByTime = {}
    for time_id in times:
        features, labels = baseline_training.standardized_labeled_arrays(
            snapshots,
            [time_id],
            statistics,
        )
        probabilities = model.predict_proba(features)[:, 1]
        result[time_id] = (labels, probabilities)
    return result


def predict_static_by_time(
    name: str,
    model: torch.nn.Module,
    snapshots: Mapping[int, EllipticPPHypergraphSnapshot],
    transaction_edges: Mapping[int, torch.Tensor],
    times: Iterable[int],
    device: torch.device,
) -> PredictionByTime:
    model.eval()
    result: PredictionByTime = {}
    with torch.no_grad():
        for time_id in times:
            snapshot = snapshots[time_id].to(device)
            edge_index = (
                transaction_edges[time_id].to(device)
                if name in baseline_training.GRAPH_MODELS
                else None
            )
            logits = baseline_training.forward_static_model(
                name,
                model,
                snapshot,
                edge_index,
            )
            mask = snapshot.labeled_mask
            result[time_id] = (
                snapshot.labels[mask].cpu().numpy(),
                torch.sigmoid(logits[mask]).cpu().numpy(),
            )
    return result


def predict_evolve_by_time(
    model: EvolveGCNOBaseline,
    snapshots: Mapping[int, EllipticPPHypergraphSnapshot],
    transaction_edges: Mapping[int, torch.Tensor],
    replay_times: Sequence[int],
    prediction_times: Iterable[int],
    device: torch.device,
) -> PredictionByTime:
    model.eval()
    prediction_set = set(prediction_times)
    recurrent_weights = model.initial_state()
    result: PredictionByTime = {}
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
            result[time_id] = (
                snapshot.labels[mask].cpu().numpy(),
                torch.sigmoid(logits[mask]).cpu().numpy(),
            )
    return result


def predict_temporal_by_time(
    model: TemporalMemoryHGNN,
    snapshots: Mapping[int, EllipticPPHypergraphSnapshot],
    replay_times: Sequence[int],
    prediction_times: Iterable[int],
    global_address_count: int,
    device: torch.device,
) -> PredictionByTime:
    model.eval()
    prediction_set = set(prediction_times)
    memory_bank = model.initial_memory(global_address_count, device)
    result: PredictionByTime = {}
    with torch.no_grad():
        for time_id in replay_times:
            snapshot = snapshots[time_id].to(device)
            global_ids = snapshot.global_address_ids
            previous_memory = memory_bank.index_select(0, global_ids)
            logits, updated_memory = model(snapshot, previous_memory)
            memory_bank.index_copy_(0, global_ids, updated_memory)
            if time_id not in prediction_set:
                continue
            mask = snapshot.labeled_mask
            result[time_id] = (
                snapshot.labels[mask].cpu().numpy(),
                torch.sigmoid(logits[mask]).cpu().numpy(),
            )
    return result


def evaluate_baseline_seed(
    seed: int,
    experiment_root: Path,
    output_root: Path,
    snapshots: Mapping[int, EllipticPPHypergraphSnapshot],
    transaction_edges: Mapping[int, torch.Tensor],
    train_times: Sequence[int],
    validation_times: Sequence[int],
    test_times: Sequence[int],
    device: torch.device,
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    seed_root = experiment_root / "baselines" / f"seed_{seed}"
    for name in MODEL_ORDER[:-1]:
        model_root = seed_root / name
        original_metrics = json.loads(
            (model_root / "metrics.json").read_text(encoding="utf-8")
        )
        original_threshold = float(original_metrics["validation"]["threshold"])
        if name in baseline_training.CLASSICAL_MODELS:
            model = joblib.load(model_root / "best.joblib")
            checkpoint = torch.load(
                seed_root / "gcn" / "best.pt",
                map_location="cpu",
                weights_only=False,
            )
            transaction_statistics = feature_statistics(
                checkpoint["transaction_statistics"]
            )
            predictions = predict_classical_by_time(
                model,
                snapshots,
                validation_times + test_times,
                transaction_statistics,
            )
        else:
            checkpoint = torch.load(
                model_root / "best.pt",
                map_location="cpu",
                weights_only=False,
            )
            address_statistics = feature_statistics(checkpoint["address_statistics"])
            transaction_statistics = feature_statistics(
                checkpoint["transaction_statistics"]
            )
            config = checkpoint["config"]
            model = baseline_training.build_model(
                name,
                config["model"],
                address_statistics,
                transaction_statistics,
            ).to(device)
            model.load_state_dict(checkpoint["model_state"])
            if name == "evolvegcn_o":
                predictions = predict_evolve_by_time(
                    model,
                    snapshots,
                    transaction_edges,
                    train_times + validation_times + test_times,
                    validation_times + test_times,
                    device,
                )
            else:
                predictions = predict_static_by_time(
                    name,
                    model,
                    snapshots,
                    transaction_edges,
                    validation_times + test_times,
                    device,
                )
        result: Dict[str, Any] = {
            "model": name,
            "seed": seed,
            "source_metrics": str(model_root / "metrics.json"),
            **evaluate_predictions(
                predictions,
                validation_times,
                test_times,
                original_threshold,
            ),
        }
        target = output_root / f"seed_{seed}" / name
        target.mkdir(parents=True, exist_ok=True)
        (target / "metrics.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        results.append(result)
        print(
            f"[{seed}] {name}: threshold="
            f"{result['threshold_selection']['threshold']:.6f} "
            f"test_f1={result['test']['f1']:.6f}",
            flush=True,
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return results


def evaluate_main_seed(
    seed: int,
    experiment_root: Path,
    output_root: Path,
    snapshots: Mapping[int, EllipticPPHypergraphSnapshot],
    global_address_count: int,
    train_times: Sequence[int],
    validation_times: Sequence[int],
    test_times: Sequence[int],
    device: torch.device,
) -> Dict[str, Any]:
    model_root = experiment_root / "main_model" / f"seed_{seed}"
    checkpoint = torch.load(
        model_root / "best.pt",
        map_location="cpu",
        weights_only=False,
    )
    config = checkpoint["config"]
    model_config = config["model"]
    model = TemporalMemoryHGNN(
        address_statistics=feature_statistics(checkpoint["address_statistics"]),
        transaction_statistics=feature_statistics(
            checkpoint["transaction_statistics"]
        ),
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
        transaction_encoder_type=str(
            model_config.get("transaction_encoder_type", "linear")
        ),
        transaction_interaction_rank=int(
            model_config.get("transaction_interaction_rank", 32)
        ),
        transaction_interaction_layers=int(
            model_config.get("transaction_interaction_layers", 2)
        ),
        address_risk_auxiliary_enabled=bool(
            config.get("address_auxiliary", {}).get("enabled", False)
        ),
        address_risk_initial_fusion_scale=float(
            config.get("address_auxiliary", {}).get("initial_fusion_scale", 0.1)
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
    model.load_state_dict(checkpoint["model_state"])
    predictions = predict_temporal_by_time(
        model,
        snapshots,
        train_times + validation_times + test_times,
        validation_times + test_times,
        global_address_count,
        device,
    )
    result: Dict[str, Any] = {
        "model": "temporal_memory_hgnn_erl",
        "seed": seed,
        "source_metrics": str(model_root / "metrics.json"),
        **evaluate_predictions(
            predictions,
            validation_times,
            test_times,
            float(checkpoint["validation_threshold"]),
        ),
    }
    target = output_root / f"seed_{seed}" / result["model"]
    target.mkdir(parents=True, exist_ok=True)
    (target / "metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        f"[{seed}] {result['model']}: threshold="
        f"{result['threshold_selection']['threshold']:.6f} "
        f"test_f1={result['test']['f1']:.6f}",
        flush=True,
    )
    return result


def summarize(
    all_results: Sequence[Dict[str, Any]],
    output_root: Path,
) -> Dict[str, Any]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for result in all_results:
        grouped[result["model"]].append(result)
    summaries: List[Dict[str, Any]] = []
    for model_name in MODEL_ORDER:
        model_results = sorted(
            grouped.get(model_name, []),
            key=lambda item: item["seed"],
        )
        if not model_results:
            continue
        summary: Dict[str, Any] = {
            "model": model_name,
            "num_runs": len(model_results),
            "seeds": [item["seed"] for item in model_results],
        }
        thresholds = np.asarray(
            [item["threshold_selection"]["threshold"] for item in model_results],
            dtype=np.float64,
        )
        summary["threshold"] = {
            "mean": float(thresholds.mean()),
            "std": float(thresholds.std(ddof=1)) if thresholds.size > 1 else 0.0,
        }
        for split in ("validation", "test"):
            summary[split] = {}
            for metric_name in METRICS:
                values = np.asarray(
                    [item[split][metric_name] for item in model_results],
                    dtype=np.float64,
                )
                summary[split][metric_name] = {
                    "mean": float(values.mean()),
                    "std": (
                        float(values.std(ddof=1)) if values.size > 1 else 0.0
                    ),
                }
        summaries.append(summary)
    payload = {
        "status": "PASS",
        "threshold_selection": "mean_validation_snapshot_f1",
        "models": summaries,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    fieldnames = ["model", "num_runs", "threshold_mean", "threshold_std"]
    for split in ("validation", "test"):
        for metric_name in METRICS:
            fieldnames.extend(
                [f"{split}_{metric_name}_mean", f"{split}_{metric_name}_std"]
            )
    with (output_root / "summary.csv").open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for summary in summaries:
            row: Dict[str, Any] = {
                "model": summary["model"],
                "num_runs": summary["num_runs"],
                "threshold_mean": summary["threshold"]["mean"],
                "threshold_std": summary["threshold"]["std"],
            }
            for split in ("validation", "test"):
                for metric_name in METRICS:
                    row[f"{split}_{metric_name}_mean"] = summary[split][
                        metric_name
                    ]["mean"]
                    row[f"{split}_{metric_name}_std"] = summary[split][
                        metric_name
                    ]["std"]
            writer.writerow(row)
    return payload


def main() -> None:
    args = parse_args()
    experiment_root = resolve_path(args.experiment_root)
    output_root = resolve_path(args.output_root)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    with (PROJECT_ROOT / "configs" / "main_baselines.yaml").open(
        "r",
        encoding="utf-8",
    ) as stream:
        config: Dict[str, Any] = yaml.safe_load(stream)
    train_times = baseline_training.expand_interval(config["split"]["train"])
    validation_times = baseline_training.expand_interval(
        config["split"]["validation"]
    )
    test_times = baseline_training.expand_interval(config["split"]["test"])
    snapshots, global_address_count = temporal_training.load_snapshots(
        resolve_path(Path(config["data"]["processed_root"]))
    )
    transaction_edges = baseline_training.load_transaction_edges(
        resolve_path(Path(config["data"]["transaction_edges"])),
        snapshots,
    )
    available_seeds = sorted(
        int(path.name.removeprefix("seed_"))
        for path in (experiment_root / "baselines").glob("seed_*")
        if (path / "comparison.json").exists()
    )
    seeds = args.seeds if args.seeds else available_seeds
    if not seeds:
        raise RuntimeError("No completed baseline seeds were found.")
    all_results: List[Dict[str, Any]] = []
    for seed in seeds:
        all_results.extend(
            evaluate_baseline_seed(
                seed,
                experiment_root,
                output_root,
                snapshots,
                transaction_edges,
                train_times,
                validation_times,
                test_times,
                device,
            )
        )
        all_results.append(
            evaluate_main_seed(
                seed,
                experiment_root,
                output_root,
                snapshots,
                global_address_count,
                train_times,
                validation_times,
                test_times,
                device,
            )
        )
    payload = summarize(all_results, output_root)
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
