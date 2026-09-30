from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    precision_recall_fscore_support,
    roc_auc_score,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import audit_macro_thresholds as audit  # noqa: E402
import train_quantile_fusion as quantile_fusion  # noqa: E402
import train_temporal_memory as temporal_training  # noqa: E402


INTERPRETABLE_FEATURES = {
    165: "输入侧交易度",
    166: "输出侧交易度",
    167: "交易总额（BTC）",
    168: "手续费",
    169: "交易大小",
    170: "输入地址数",
    171: "输出地址数",
    172: "输入金额最小值",
    173: "输入金额最大值",
    174: "输入金额均值",
    175: "输入金额中位数",
    176: "输入金额总值",
    177: "输出金额最小值",
    178: "输出金额最大值",
    179: "输出金额均值",
    180: "输出金额中位数",
    181: "输出金额总值",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export real TTHGNN-QIF test predictions for the competition demo."
    )
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-addresses-per-role", type=int, default=24)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("demo/data/demo_payload.json"),
    )
    return parser.parse_args()


def project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def fit_platt_calibrator(
    validation_logits: np.ndarray,
    validation_labels: np.ndarray,
) -> Dict[str, float]:
    """Fit a monotonic probability calibrator using validation evidence only."""
    calibrator = LogisticRegression(
        C=1.0,
        solver="lbfgs",
        max_iter=2000,
        random_state=0,
    )
    calibrator.fit(validation_logits.reshape(-1, 1), validation_labels)
    coefficient = float(calibrator.coef_[0, 0])
    intercept = float(calibrator.intercept_[0])
    if coefficient <= 0.0:
        raise RuntimeError("Platt calibration must preserve risk-score ordering.")
    probabilities = calibrator.predict_proba(
        validation_logits.reshape(-1, 1)
    )[:, 1]
    return {
        "method": "platt_scaling",
        "fit_scope": "validation_only",
        "coefficient": coefficient,
        "intercept": intercept,
        "validation_brier_score": float(
            brier_score_loss(validation_labels, probabilities)
        ),
        "validation_log_loss": float(
            log_loss(validation_labels, probabilities, labels=[0, 1])
        ),
    }


def calibrated_probability(logit: float, calibration: Mapping[str, float]) -> float:
    return sigmoid(
        float(calibration["coefficient"]) * float(logit)
        + float(calibration["intercept"])
    )


def reference_percentile(logit: float, sorted_reference: np.ndarray) -> float:
    """Map a model logit to its relative position in validation evidence."""
    if sorted_reference.ndim != 1 or sorted_reference.size == 0:
        raise ValueError("Risk-index reference must be a non-empty vector.")
    rank = int(np.searchsorted(sorted_reference, float(logit), side="right"))
    return float((rank + 0.5) / (sorted_reference.size + 1.0))


def safe_float(value: float) -> float | None:
    number = float(value)
    return number if math.isfinite(number) else None


def deduplicate(values: Iterable[int]) -> List[int]:
    return list(dict.fromkeys(int(value) for value in values))


def load_address_lookup(data_root: Path) -> Mapping[int, str]:
    frame = pd.read_csv(
        data_root / "address_index.csv.gz",
        usecols=["global_address_id", "address"],
    )
    return dict(
        zip(
            frame["global_address_id"].astype(int),
            frame["address"].astype(str),
        )
    )


def related_global_ids(
    snapshot: Any,
    transaction_position: int,
    relation_index: torch.Tensor,
) -> List[int]:
    if relation_index.numel() == 0:
        return []
    else:
        mask = relation_index[1] == int(transaction_position)
        local_ids = relation_index[0, mask]
        return deduplicate(
            snapshot.global_address_ids.index_select(0, local_ids).tolist()
        )


def related_addresses(
    global_ids: Sequence[int],
    lookup: Mapping[int, str],
    maximum: int,
) -> Dict[str, Any]:
    visible_ids = global_ids[:maximum]
    return {
        "count": len(global_ids),
        "truncated": len(global_ids) > len(visible_ids),
        "items": [
            {
                "global_id": int(global_id),
                "address": lookup.get(int(global_id), f"address-{global_id}"),
            }
            for global_id in visible_ids
        ],
    }


def attribute_profile(
    snapshot: Any,
    transaction_position: int,
    standardized_row: torch.Tensor,
) -> List[Dict[str, Any]]:
    raw_row = snapshot.transaction_features[transaction_position]
    present = snapshot.transaction_feature_mask[transaction_position]
    rows: List[Dict[str, Any]] = []
    for index, chinese_name in INTERPRETABLE_FEATURES.items():
        if not bool(present[index]):
            continue
        rows.append(
            {
                "feature_index": int(index),
                "name": chinese_name,
                "raw_value": safe_float(raw_row[index]),
                "standardized_value": safe_float(standardized_row[index]),
            }
        )
    rows.sort(
        key=lambda row: abs(row["standardized_value"] or 0.0), reverse=True
    )
    return rows[:6]


def classification_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> Dict[str, Any]:
    predicted = (scores >= threshold).astype(np.int64)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predicted,
        average="binary",
        pos_label=1,
        zero_division=0,
    )
    true_positive = int(np.logical_and(labels == 1, predicted == 1).sum())
    false_positive = int(np.logical_and(labels == 0, predicted == 1).sum())
    true_negative = int(np.logical_and(labels == 0, predicted == 0).sum())
    false_negative = int(np.logical_and(labels == 1, predicted == 0).sum())
    return {
        "samples": int(labels.size),
        "illicit_samples": int((labels == 1).sum()),
        "licit_samples": int((labels == 0).sum()),
        "predicted_high_risk": int(predicted.sum()),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float((predicted == labels).mean()),
        "average_precision": float(average_precision_score(labels, scores)),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "confusion_matrix": {
            "true_positive": true_positive,
            "false_positive": false_positive,
            "true_negative": true_negative,
            "false_negative": false_negative,
        },
    }


def load_efficiency() -> Dict[str, Any] | None:
    path = PROJECT_ROOT / "artifacts" / "computational_efficiency" / "summary.json"
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    full = next(
        (row for row in payload.get("summary", []) if row.get("method") == "full"),
        None,
    )
    return {"protocol": payload.get("protocol", {}), "full_model": full}


def load_five_seed_metrics() -> Dict[str, Dict[str, float]]:
    summary_path = (
        PROJECT_ROOT / "artifacts" / "pooled_f1" / "ablation" / "summary.json"
    )
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    full = next(item for item in payload["methods"] if item["id"] == "M0")
    metrics = {
        name: {
            "mean": float(values["mean"]),
            "std": float(values["std"]),
        }
        for name, values in full["metrics"].items()
    }
    efficiency = load_efficiency()
    if efficiency and efficiency.get("full_model"):
        model = efficiency["full_model"]
        metrics["average_precision"] = {
            "mean": float(model["average_precision_mean"]),
            "std": float(model["average_precision_std"]),
        }
    return metrics


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")

    data_root = PROJECT_ROOT / "data" / "processed" / "ellipticpp"
    snapshots, global_address_count = temporal_training.load_snapshots(data_root)
    validation_times = list(range(31, 36))
    test_times = list(range(36, 50))
    seed = int(args.seed)
    fusion_root = PROJECT_ROOT / "artifacts" / "pooled_f1" / "ablation"
    result_root = fusion_root / "full" / f"seed_{seed}"
    checkpoint = torch.load(
        result_root / "best.pt", map_location="cpu", weights_only=False
    )

    statistics = audit.feature_statistics(checkpoint["transaction_statistics"])
    model_config = checkpoint["model_config"]
    attribute_model = quantile_fusion.QuantileInteractionClassifier(
        bin_edges=checkpoint["bin_edges"],
        embedding_dim=int(model_config["embedding_dim"]),
        hidden_dim=int(model_config["hidden_dim"]),
        dropout=float(model_config["dropout"]),
        use_bi_interaction=bool(model_config.get("use_bi_interaction", False)),
        interaction_mode=str(model_config.get("interaction_mode", "residual")),
        interaction_scale_initial=float(
            model_config.get("interaction_scale_initial", 0.10)
        ),
    ).to(device)
    attribute_model.load_state_dict(checkpoint["model_state"])

    validation_features, validation_labels, _ = (
        quantile_fusion.pack_labeled_transactions(
            snapshots, validation_times, statistics
        )
    )
    test_features, test_labels, test_time_ids = (
        quantile_fusion.pack_labeled_transactions(
            snapshots, test_times, statistics
        )
    )
    validation_attribute_logits = quantile_fusion.batched_logits(
        attribute_model, validation_features, 768, device
    )
    attribute_logits = quantile_fusion.batched_logits(
        attribute_model, test_features, 768, device
    )
    graph_checkpoint = (
        PROJECT_ROOT
        / "artifacts"
        / "ablation"
        / "full"
        / "graph"
        / f"seed_{seed}"
        / "best.pt"
    )
    (
        validation_graph_logits,
        graph_logits,
        validation_graph_labels,
        graph_labels,
        _,
    ) = quantile_fusion.replay_graph_logits(
        graph_checkpoint,
        snapshots,
        global_address_count,
        device,
    )
    labels = test_labels.numpy()
    if not np.array_equal(validation_labels.numpy(), validation_graph_labels):
        raise RuntimeError("Validation labels are not aligned between branches.")
    if not np.array_equal(labels, graph_labels):
        raise RuntimeError("Graph and QIF labels are not aligned.")

    attribute_weight = float(checkpoint["fusion"]["attribute_weight"])
    graph_weight = float(checkpoint["fusion"]["graph_weight"])
    threshold_logit = float(checkpoint["fusion"]["threshold"])
    validation_fused_logits = (
        graph_weight * validation_graph_logits
        + attribute_weight * validation_attribute_logits
    )
    fused_logits = graph_weight * graph_logits + attribute_weight * attribute_logits
    calibration = fit_platt_calibrator(
        validation_fused_logits,
        validation_labels.numpy(),
    )
    sorted_validation_logits = np.sort(validation_fused_logits)
    threshold_score = reference_percentile(
        threshold_logit, sorted_validation_logits
    )
    calibrated_threshold_score = calibrated_probability(
        threshold_logit, calibration
    )
    model_threshold_score = sigmoid(threshold_logit)
    predictions = (fused_logits >= threshold_logit).astype(np.int64)

    address_lookup = load_address_lookup(data_root)
    cases: List[Dict[str, Any]] = []
    address_index: Dict[str, Dict[str, Any]] = {}
    offset = 0
    for time_id in test_times:
        snapshot = snapshots[time_id]
        positions = torch.nonzero(snapshot.labeled_mask, as_tuple=False).flatten()
        standardized = quantile_fusion.standardized_features(snapshot, statistics)
        for within_time_index, position_tensor in enumerate(positions):
            prediction_index = offset + within_time_index
            position = int(position_tensor)
            fused_logit = float(fused_logits[prediction_index])
            graph_logit = float(graph_logits[prediction_index])
            attribute_logit = float(attribute_logits[prediction_index])
            risk_index = reference_percentile(
                fused_logit, sorted_validation_logits
            )
            risk_probability = calibrated_probability(fused_logit, calibration)
            model_score = sigmoid(fused_logit)
            label = int(labels[prediction_index])
            predicted = int(predictions[prediction_index])
            input_global_ids = related_global_ids(
                snapshot, position, snapshot.input_index
            )
            output_global_ids = related_global_ids(
                snapshot, position, snapshot.output_index
            )
            transaction_id = int(snapshot.transaction_ids[position])
            for role_bit, global_ids in (
                (1, input_global_ids),
                (2, output_global_ids),
            ):
                for global_id in global_ids:
                    address = address_lookup.get(
                        int(global_id), f"address-{global_id}"
                    )
                    entry = address_index.setdefault(
                        address,
                        {
                            "global_address_id": int(global_id),
                            "links": {},
                        },
                    )
                    entry["links"][transaction_id] = (
                        int(entry["links"].get(transaction_id, 0)) | role_bit
                    )
            if predicted == label:
                outcome = "true_positive" if predicted else "true_negative"
            else:
                outcome = "false_positive" if predicted else "false_negative"
            cases.append(
                {
                    "transaction_id": transaction_id,
                    "time_step": int(time_id),
                    "event_time": None,
                    "dataset_label": label,
                    "predicted_label": predicted,
                    "outcome": outcome,
                    "risk_score": risk_index,
                    "risk_index": risk_index,
                    "calibrated_risk_probability": risk_probability,
                    "model_score": model_score,
                    "graph_score": sigmoid(graph_logit),
                    "qif_score": sigmoid(attribute_logit),
                    "margin_to_threshold": risk_index - threshold_score,
                    "input_addresses": related_addresses(
                        input_global_ids,
                        address_lookup,
                        int(args.max_addresses_per_role),
                    ),
                    "output_addresses": related_addresses(
                        output_global_ids,
                        address_lookup,
                        int(args.max_addresses_per_role),
                    ),
                    "attribute_profile": attribute_profile(
                        snapshot,
                        position,
                        standardized[position],
                    ),
                }
            )
        offset += int(positions.numel())
    if offset != int(labels.size):
        raise RuntimeError("Case export count does not match prediction count.")

    metrics = classification_metrics(labels, fused_logits, threshold_logit)
    time_metrics = []
    for time_id in test_times:
        mask = test_time_ids == time_id
        item = classification_metrics(
            labels[mask], fused_logits[mask], threshold_logit
        )
        item["time_step"] = int(time_id)
        time_metrics.append(item)

    metadata = json.loads((data_root / "metadata.json").read_text(encoding="utf-8"))
    serialized_address_index = {
        address: {
            "global_address_id": int(entry["global_address_id"]),
            "links": [
                [int(transaction_id), int(role_bits)]
                for transaction_id, role_bits in entry["links"].items()
            ],
        }
        for address, entry in address_index.items()
    }
    payload: Dict[str, Any] = {
        "status": "PASS",
        "method": "TTHGNN-QIF",
        "seed": seed,
        "scope_note": (
            "本演示重放模型在Elliptic++测试时间片36–49上的真实预测；"
            "风险分数是模型输出，不是司法或犯罪事实认定。"
        ),
        "protocol": {
            "dataset": "Elliptic++",
            "train": [1, 30],
            "validation": [31, 35],
            "test": [36, 49],
            "threshold_selected_on": "validation_only",
            "unknown_labels_excluded_from_supervision": True,
        },
        "fusion": {
            "graph_weight": graph_weight,
            "qif_weight": attribute_weight,
            "threshold_logit": threshold_logit,
            "threshold_score": threshold_score,
            "calibrated_threshold_score": calibrated_threshold_score,
            "model_threshold_score": model_threshold_score,
        },
        "risk_calibration": {
            **calibration,
            "display_index_method": "validation_empirical_percentile",
            "display_index_reference_size": int(sorted_validation_logits.size),
            "display_index_threshold": threshold_score,
            # The online-chain adapter uses the exact validation reference to
            # map newly inferred logits onto the same 0-100 risk-index scale.
            # Keeping this vector in the server-side payload avoids replaying
            # the validation period whenever the web service starts.
            "display_index_reference_logits": [
                float(value) for value in sorted_validation_logits
            ],
        },
        "selected_seed_metrics": metrics,
        "five_seed_metrics": load_five_seed_metrics(),
        "time_metrics": time_metrics,
        "dataset": {
            "transactions": int(metadata["totals"]["transactions"]),
            "global_addresses": int(metadata["totals"]["global_addresses"]),
            "test_labeled_transactions": int(labels.size),
            "test_time_steps": len(test_times),
        },
        "efficiency": load_efficiency(),
        "address_index": serialized_address_index,
        "cases": cases,
    }
    output_path = project_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "output": str(output_path),
                "cases": len(cases),
                "threshold_score": threshold_score,
                "metrics": metrics,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
