from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import audit_macro_thresholds as audit  # noqa: E402
import train_quantile_fusion as quantile_fusion  # noqa: E402
import train_temporal_memory as temporal_training  # noqa: E402


FEATURE_INDICES = list(range(165, 182))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train the QIF deployment adapter on the 17 transaction fields that "
            "can be reconstructed from a public Bitcoin node, then calibrate its "
            "fusion with the saved temporal-hypergraph branch."
        )
    )
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-bins", type=int, default=16)
    parser.add_argument("--embedding-dim", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--learning-rate", type=float, default=8e-4)
    parser.add_argument("--weight-decay", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=768)
    parser.add_argument("--max-epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=10)
    return parser.parse_args()


def reference_percentile(value: float, reference: np.ndarray) -> float:
    rank = int(np.searchsorted(reference, float(value), side="right"))
    return float((rank + 0.5) / (reference.size + 1.0))


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    seed = int(args.seed)
    quantile_fusion.set_seed(seed)
    snapshots, global_address_count = temporal_training.load_snapshots(
        PROJECT_ROOT / "data" / "processed" / "ellipticpp"
    )
    fusion_checkpoint = torch.load(
        PROJECT_ROOT
        / "artifacts"
        / "pooled_f1"
        / "ablation"
        / "full"
        / f"seed_{seed}"
        / "best.pt",
        map_location="cpu",
        weights_only=False,
    )
    full_statistics = audit.feature_statistics(
        fusion_checkpoint["transaction_statistics"]
    )
    train_features, train_labels, _ = quantile_fusion.pack_labeled_transactions(
        snapshots, range(1, 31), full_statistics
    )
    validation_features, validation_labels, validation_times = (
        quantile_fusion.pack_labeled_transactions(
            snapshots, range(31, 36), full_statistics
        )
    )
    test_features, test_labels, _ = quantile_fusion.pack_labeled_transactions(
        snapshots, range(36, 50), full_statistics
    )
    train_features = train_features[:, FEATURE_INDICES]
    validation_features = validation_features[:, FEATURE_INDICES]
    test_features = test_features[:, FEATURE_INDICES]

    all_train_features = torch.cat(
        [
            quantile_fusion.standardized_features(snapshot, full_statistics)[
                :, FEATURE_INDICES
            ]
            for time_id, snapshot in snapshots.items()
            if 1 <= time_id <= 30
        ]
    )
    all_train_masks = torch.cat(
        [
            snapshot.transaction_feature_mask[:, FEATURE_INDICES]
            for time_id, snapshot in snapshots.items()
            if 1 <= time_id <= 30
        ]
    )
    quantiles = (
        torch.arange(1, int(args.num_bins), dtype=torch.float32)
        / int(args.num_bins)
    )
    bin_edges = torch.stack(
        [
            torch.quantile(
                all_train_features[:, column][all_train_masks[:, column]],
                quantiles,
            )
            for column in range(len(FEATURE_INDICES))
        ]
    )
    attribute_model = quantile_fusion.QuantileInteractionClassifier(
        bin_edges=bin_edges,
        embedding_dim=int(args.embedding_dim),
        hidden_dim=int(args.hidden_dim),
        dropout=float(args.dropout),
        use_bi_interaction=True,
        interaction_mode="residual",
        interaction_scale_initial=0.1,
    ).to(device)
    best_state, best_epoch, history = quantile_fusion.train_attribute_model(
        attribute_model,
        train_features,
        train_labels,
        validation_features,
        validation_labels,
        args,
        device,
    )
    attribute_model.load_state_dict(best_state)
    validation_attribute_logits = quantile_fusion.batched_logits(
        attribute_model, validation_features, int(args.batch_size), device
    )
    test_attribute_logits = quantile_fusion.batched_logits(
        attribute_model, test_features, int(args.batch_size), device
    )

    # The graph checkpoint is unchanged. Only non-reconstructible transaction
    # attributes are masked while address features, incidence structure and
    # temporal memory remain active.
    for snapshot in snapshots.values():
        snapshot.transaction_features[:, :165] = 0.0
        snapshot.transaction_feature_mask[:, :165] = False
    (
        validation_graph_logits,
        test_graph_logits,
        validation_graph_labels,
        test_graph_labels,
        _,
    ) = quantile_fusion.replay_graph_logits(
        PROJECT_ROOT
        / "artifacts"
        / "ablation"
        / "full"
        / "graph"
        / f"seed_{seed}"
        / "best.pt",
        snapshots,
        global_address_count,
        device,
    )
    if not np.array_equal(validation_labels.numpy(), validation_graph_labels):
        raise RuntimeError("Validation branches are not aligned.")
    if not np.array_equal(test_labels.numpy(), test_graph_labels):
        raise RuntimeError("Test branches are not aligned.")
    selected, _ = quantile_fusion.select_fusion(
        validation_labels.numpy(),
        validation_times,
        validation_graph_logits,
        validation_attribute_logits,
        precision_floor=0.0,
        selection_objective="pooled_validation_f1",
    )
    graph_weight = float(selected["graph_weight"])
    attribute_weight = float(selected["attribute_weight"])
    threshold = float(selected["threshold"])
    validation_logits = (
        graph_weight * validation_graph_logits
        + attribute_weight * validation_attribute_logits
    )
    test_logits = (
        graph_weight * test_graph_logits + attribute_weight * test_attribute_logits
    )
    reference = np.sort(validation_logits)
    output_root = (
        PROJECT_ROOT / "artifacts" / "live_observable" / f"seed_{seed}"
    )
    output_root.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "seed": seed,
        "feature_indices": FEATURE_INDICES,
        "model_state": best_state,
        "bin_edges": bin_edges,
        "transaction_statistics": {
            "mean": full_statistics.mean[FEATURE_INDICES],
            "std": full_statistics.std[FEATURE_INDICES],
            "count": full_statistics.count[FEATURE_INDICES],
        },
        "model_config": {
            "embedding_dim": int(args.embedding_dim),
            "hidden_dim": int(args.hidden_dim),
            "dropout": float(args.dropout),
            "use_bi_interaction": True,
            "interaction_mode": "residual",
            "interaction_scale_initial": 0.1,
        },
        "best_epoch": int(best_epoch),
        "training_history": history,
    }
    torch.save(checkpoint, output_root / "best.pt")
    calibration = {
        "status": "PASS",
        "seed": seed,
        "protocol": {
            "purpose": "live_chain_observable_adapter",
            "observable_transaction_feature_indices": FEATURE_INDICES,
            "graph_masked_transaction_feature_range": [0, 164],
            "threshold_selected_on": "validation_only",
            "test_labels_not_used_for_selection": True,
            "graph_model_weights_unchanged": True,
        },
        "fusion": {
            "graph_weight": graph_weight,
            "attribute_weight": attribute_weight,
            "threshold_logit": threshold,
            "threshold_score": reference_percentile(threshold, reference),
        },
        "validation_metrics": temporal_training.metrics(
            validation_labels.numpy(), validation_logits, threshold
        ),
        "test_metrics": temporal_training.metrics(
            test_labels.numpy(), test_logits, threshold
        ),
        "display_index_method": "observable_validation_empirical_percentile",
        "display_index_reference_logits": [float(value) for value in reference],
    }
    (output_root / "calibration.json").write_text(
        json.dumps(calibration, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "checkpoint": str(output_root / "best.pt"),
                "calibration": str(output_root / "calibration.json"),
                "best_epoch": int(best_epoch),
                "fusion": calibration["fusion"],
                "validation_metrics": calibration["validation_metrics"],
                "test_metrics": calibration["test_metrics"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
