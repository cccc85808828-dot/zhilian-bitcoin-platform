from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from sklearn.metrics import average_precision_score, precision_recall_fscore_support
from torch.nn import functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tthgnn_erl.baselines import FeatureStatistics  # noqa: E402
from tthgnn_erl.ellipticpp import EllipticPPSnapshotDataset  # noqa: E402
from tthgnn_erl.temporal import TemporalMemoryHGNN  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare prediction-risk stability across future time environments"
    )
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--erl", type=Path, required=True)
    parser.add_argument(
        "--data-root", type=Path, default=Path("data/processed/ellipticpp")
    )
    parser.add_argument("--test-start", type=int, default=35)
    parser.add_argument("--test-end", type=int, default=49)
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/erl/stability_comparison.json")
    )
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def statistics_from_dict(values: Dict[str, torch.Tensor]) -> FeatureStatistics:
    return FeatureStatistics(
        mean=values["mean"],
        std=values["std"],
        count=values["count"],
    )


def load_model(checkpoint_path: Path, device: torch.device) -> tuple:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    model_config = config["model"]
    attribute_prior = dict(config.get("attribute_prior", {}))
    attribute_prior_enabled = bool(attribute_prior.get("enabled", False))
    model = TemporalMemoryHGNN(
        address_statistics=statistics_from_dict(checkpoint["address_statistics"]),
        transaction_statistics=statistics_from_dict(
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
        attribute_prior_residual_enabled=attribute_prior_enabled,
        structural_residual_scale=float(
            attribute_prior.get("structural_residual_scale", 1.0)
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
    model.eval()
    prior_logits_by_time = None
    if attribute_prior_enabled:
        prior_path = project_path(
            Path(
                str(attribute_prior["scores_path"]).format(
                    seed=int(config["seed"])
                )
            )
        )
        prior_payload = torch.load(
            prior_path,
            map_location="cpu",
            weights_only=False,
        )
        prior_logits_by_time = prior_payload["logits_by_time"]
    return model, checkpoint, prior_logits_by_time


def evaluate_checkpoint(
    checkpoint_path: Path,
    snapshots: Dict[int, Any],
    global_address_count: int,
    test_start: int,
    test_end: int,
    device: torch.device,
) -> Dict[str, Any]:
    model, checkpoint, prior_logits_by_time = load_model(checkpoint_path, device)
    threshold = float(checkpoint["validation_threshold"])
    memory_bank = model.initial_memory(global_address_count, device)
    per_environment: List[Dict[str, Any]] = []

    with torch.no_grad():
        for time_id in range(1, test_end + 1):
            snapshot = snapshots[time_id].to(device)
            global_ids = snapshot.global_address_ids
            previous_memory = memory_bank.index_select(0, global_ids)
            attribute_prior_logits = (
                prior_logits_by_time[time_id].to(device)
                if prior_logits_by_time is not None
                else None
            )
            logits, updated_memory = model(
                snapshot,
                previous_memory,
                attribute_prior_logits,
            )
            memory_bank.index_copy_(0, global_ids, updated_memory)
            if time_id < test_start:
                continue

            mask = snapshot.labeled_mask
            labels = snapshot.labels[mask].cpu().numpy()
            selected_logits = logits[mask]
            probabilities = torch.sigmoid(selected_logits).cpu().numpy()
            predictions = (probabilities >= threshold).astype(np.int64)
            unweighted_losses = F.binary_cross_entropy_with_logits(
                selected_logits,
                snapshot.labels[mask].to(dtype=torch.float32),
                reduction="none",
            )
            selected_labels_device = snapshot.labels[mask]
            positive_risk = unweighted_losses[selected_labels_device == 1].mean()
            negative_risk = unweighted_losses[selected_labels_device == 0].mean()
            class_balanced_risk = 0.5 * (positive_risk + negative_risk)
            precision, recall, f1, _ = precision_recall_fscore_support(
                labels,
                predictions,
                average="binary",
                pos_label=1,
                zero_division=0,
            )
            per_environment.append(
                {
                    "time_id": time_id,
                    "labeled": int(mask.sum()),
                    "positives": int((labels == 1).sum()),
                    "risk": float(
                        F.binary_cross_entropy_with_logits(
                            selected_logits,
                            snapshot.labels[mask].to(dtype=torch.float32),
                        )
                    ),
                    "class_balanced_risk": float(class_balanced_risk),
                    "average_precision": float(
                        average_precision_score(labels, probabilities)
                    ),
                    "precision": float(precision),
                    "recall": float(recall),
                    "f1": float(f1),
                }
            )

    risks = np.asarray([row["risk"] for row in per_environment])
    class_balanced_risks = np.asarray(
        [row["class_balanced_risk"] for row in per_environment]
    )
    average_precisions = np.asarray(
        [row["average_precision"] for row in per_environment]
    )
    recalls = np.asarray([row["recall"] for row in per_environment])
    f1_scores = np.asarray([row["f1"] for row in per_environment])
    return {
        "model": checkpoint["model_name"],
        "checkpoint": str(checkpoint_path),
        "validation_threshold": threshold,
        "environment_count": len(per_environment),
        "risk_mean": float(risks.mean()),
        "risk_variance": float(risks.var()),
        "risk_std": float(risks.std()),
        "class_balanced_risk_mean": float(class_balanced_risks.mean()),
        "class_balanced_risk_variance": float(class_balanced_risks.var()),
        "class_balanced_risk_std": float(class_balanced_risks.std()),
        "average_precision_mean": float(average_precisions.mean()),
        "average_precision_std": float(average_precisions.std()),
        "recall_mean": float(recalls.mean()),
        "recall_std": float(recalls.std()),
        "f1_mean": float(f1_scores.mean()),
        "f1_std": float(f1_scores.std()),
        "per_environment": per_environment,
    }


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA evaluation was requested but CUDA is unavailable.")
    dataset = EllipticPPSnapshotDataset(project_path(args.data_root), validate=True)
    snapshots = {snapshot.time_id: snapshot for snapshot in dataset}
    global_address_count = int(dataset.metadata["totals"]["global_addresses"])
    results = {
        "split": {"test": [args.test_start, args.test_end]},
        "baseline": evaluate_checkpoint(
            project_path(args.baseline),
            snapshots,
            global_address_count,
            args.test_start,
            args.test_end,
            device,
        ),
        "erl": evaluate_checkpoint(
            project_path(args.erl),
            snapshots,
            global_address_count,
            args.test_start,
            args.test_end,
            device,
        ),
    }
    baseline_variance = results["baseline"]["risk_variance"]
    erl_variance = results["erl"]["risk_variance"]
    baseline_balanced_variance = results["baseline"][
        "class_balanced_risk_variance"
    ]
    erl_balanced_variance = results["erl"]["class_balanced_risk_variance"]
    results["risk_variance_change"] = {
        "absolute": erl_variance - baseline_variance,
        "relative_percent": 100.0 * (erl_variance - baseline_variance) / baseline_variance,
    }
    results["class_balanced_risk_variance_change"] = {
        "absolute": erl_balanced_variance - baseline_balanced_variance,
        "relative_percent": 100.0
        * (erl_balanced_variance - baseline_balanced_variance)
        / baseline_balanced_variance,
    }
    output_path = project_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as stream:
        json.dump(results, stream, ensure_ascii=False, indent=2)
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
