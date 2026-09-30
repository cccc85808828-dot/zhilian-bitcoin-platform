from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from sklearn.metrics import average_precision_score, precision_recall_curve


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import audit_macro_thresholds as audit  # noqa: E402
import train_main_baselines as baseline_training  # noqa: E402
import train_quantile_fusion as quantile_fusion  # noqa: E402
import train_temporal_memory as temporal_training  # noqa: E402
from tthgnn_erl.ellipticpp import EllipticPPHypergraphSnapshot  # noqa: E402


DEFAULT_SEEDS = [20260806, 20260807, 20260808, 20260809, 20260810]
BASELINES = ["hgnn", "hgt", "heterosage", "evolvegcn_o", "bf_hgn"]
DISPLAY_NAMES = {
    "full": "本文方法",
    "hgnn": "HGNN",
    "hgt": "HGT",
    "heterosage": "HeteroSAGE",
    "evolvegcn_o": "EvolveGCN-O",
    "bf_hgn": "BF-HGN",
}
COLORS = {
    "full": "#C00000",
    "hgnn": "#0072B2",
    "hgt": "#E69F00",
    "heterosage": "#009E73",
    "evolvegcn_o": "#CC79A7",
    "bf_hgn": "#56B4E9",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot five-seed mean precision-recall curves."
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/main_experiment/figures/pr_curve"),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument(
        "--fusion-root",
        type=Path,
        default=Path("artifacts/pooled_f1/ablation"),
    )
    return parser.parse_args()


def project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_baseline_scores(
    name: str,
    seed: int,
    snapshots: Mapping[int, EllipticPPHypergraphSnapshot],
    transaction_edges: Mapping[int, torch.Tensor],
    train_times: Sequence[int],
    validation_times: Sequence[int],
    test_times: Sequence[int],
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    model_root = (
        PROJECT_ROOT
        / "artifacts"
        / "main_experiment"
        / "baselines"
        / f"seed_{seed}"
        / name
    )
    checkpoint = torch.load(
        model_root / "best.pt",
        map_location="cpu",
        weights_only=False,
    )
    address_statistics = audit.feature_statistics(checkpoint["address_statistics"])
    transaction_statistics = audit.feature_statistics(
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
        predictions = audit.predict_evolve_by_time(
            model,
            snapshots,
            transaction_edges,
            list(train_times) + list(validation_times) + list(test_times),
            test_times,
            device,
        )
        labels, probabilities = audit.concatenate_predictions(
            predictions, test_times
        )
    elif name == "bf_hgn":
        bf_config = config["model"].get("bf_hgn", {})
        labels, probabilities, _ = baseline_training.predict_bf_hgn(
            model=model,
            snapshots=dict(snapshots),
            transaction_edges=dict(transaction_edges),
            replay_times=list(train_times) + list(validation_times) + list(test_times),
            prediction_times=list(test_times),
            device=device,
            window_size=int(bf_config.get("window_size", 5)),
        )
    else:
        predictions = audit.predict_static_by_time(
            name,
            model,
            snapshots,
            transaction_edges,
            test_times,
            device,
        )
        labels, probabilities = audit.concatenate_predictions(
            predictions, test_times
        )

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return labels.astype(np.int64, copy=False), probabilities.astype(
        np.float64, copy=False
    )


def load_full_scores(
    seed: int,
    snapshots: Dict[int, EllipticPPHypergraphSnapshot],
    global_address_count: int,
    test_times: Sequence[int],
    device: torch.device,
    fusion_root: Path,
) -> Tuple[np.ndarray, np.ndarray]:
    result_root = fusion_root / "full" / f"seed_{seed}"
    payload = torch.load(
        result_root / "best.pt",
        map_location="cpu",
        weights_only=False,
    )
    statistics_payload = payload["transaction_statistics"]
    statistics = audit.feature_statistics(statistics_payload)
    model_config = payload["model_config"]
    attribute_model = quantile_fusion.QuantileInteractionClassifier(
        bin_edges=payload["bin_edges"],
        embedding_dim=int(model_config["embedding_dim"]),
        hidden_dim=int(model_config["hidden_dim"]),
        dropout=float(model_config["dropout"]),
        use_bi_interaction=bool(model_config.get("use_bi_interaction", False)),
        interaction_mode=str(model_config.get("interaction_mode", "residual")),
        interaction_scale_initial=float(
            model_config.get("interaction_scale_initial", 0.10)
        ),
    ).to(device)
    attribute_model.load_state_dict(payload["model_state"])
    test_features, test_labels, _ = quantile_fusion.pack_labeled_transactions(
        snapshots, test_times, statistics
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
        _,
        graph_logits,
        _,
        graph_test_labels,
        _,
    ) = quantile_fusion.replay_graph_logits(
        graph_checkpoint,
        snapshots,
        global_address_count,
        device,
    )
    if not np.array_equal(test_labels.numpy(), graph_test_labels):
        raise RuntimeError(f"Full-model test labels are misaligned for seed {seed}.")
    attribute_weight = float(payload["fusion"]["attribute_weight"])
    fused_scores = (
        (1.0 - attribute_weight) * graph_logits
        + attribute_weight * attribute_logits
    )
    del attribute_model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return graph_test_labels.astype(np.int64, copy=False), fused_scores.astype(
        np.float64, copy=False
    )


def interpolate_curve(
    labels: np.ndarray,
    scores: np.ndarray,
    recall_grid: np.ndarray,
) -> Tuple[np.ndarray, float]:
    precision, recall, _ = precision_recall_curve(labels, scores)
    recall_increasing = recall[::-1]
    precision_increasing = precision[::-1]
    unique_recall, inverse = np.unique(recall_increasing, return_inverse=True)
    unique_precision = np.zeros_like(unique_recall)
    for index in range(unique_recall.size):
        unique_precision[index] = precision_increasing[inverse == index].max()
    interpolated = np.interp(
        recall_grid,
        unique_recall,
        unique_precision,
        left=unique_precision[0],
        right=unique_precision[-1],
    )
    return interpolated, float(average_precision_score(labels, scores))


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    output_root = project_path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    with (PROJECT_ROOT / "configs" / "main_baselines.yaml").open(
        "r", encoding="utf-8"
    ) as stream:
        config = yaml.safe_load(stream)
    train_times = baseline_training.expand_interval(config["split"]["train"])
    validation_times = baseline_training.expand_interval(
        config["split"]["validation"]
    )
    test_times = baseline_training.expand_interval(config["split"]["test"])
    snapshots, global_address_count = temporal_training.load_snapshots(
        project_path(Path(config["data"]["processed_root"]))
    )
    transaction_edges = baseline_training.load_transaction_edges(
        project_path(Path(config["data"]["transaction_edges"])), snapshots
    )

    seeds = [int(seed) for seed in args.seeds]
    scores_by_model: Dict[str, List[np.ndarray]] = {
        name: [] for name in ["full"] + BASELINES
    }
    labels_reference: np.ndarray | None = None
    for seed in seeds:
        print(f"[seed={seed}] replay full model", flush=True)
        labels, scores = load_full_scores(
            seed,
            snapshots,
            global_address_count,
            test_times,
            device,
            project_path(args.fusion_root),
        )
        if labels_reference is None:
            labels_reference = labels
        elif not np.array_equal(labels_reference, labels):
            raise RuntimeError("Test labels changed across full-model seeds.")
        scores_by_model["full"].append(scores)
        for name in BASELINES:
            print(f"[seed={seed}] replay {name}", flush=True)
            baseline_labels, baseline_scores = load_baseline_scores(
                name,
                seed,
                snapshots,
                transaction_edges,
                train_times,
                validation_times,
                test_times,
                device,
            )
            if not np.array_equal(labels_reference, baseline_labels):
                raise RuntimeError(
                    f"Test labels are misaligned for {name}, seed {seed}."
                )
            scores_by_model[name].append(baseline_scores)

    if labels_reference is None:
        raise RuntimeError("No test predictions were generated.")
    recall_grid = np.linspace(0.0, 1.0, 501)
    curve_payload: Dict[str, np.ndarray] = {"recall_grid": recall_grid}
    summary: Dict[str, object] = {
        "status": "PASS",
        "dataset": "Elliptic++ transaction-address subset",
        "test_times": [int(test_times[0]), int(test_times[-1])],
        "test_positives": int((labels_reference == 1).sum()),
        "test_negatives": int((labels_reference == 0).sum()),
        "seeds": seeds,
        "models": {},
    }

    mpl.rcParams.update(
        {
            "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "font.size": 10.5,
            "axes.labelsize": 11.5,
            "axes.titlesize": 12.5,
            "legend.fontsize": 9.3,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
        }
    )
    fig, axis = plt.subplots(figsize=(7.2, 5.25), constrained_layout=True)
    for name in ["full"] + BASELINES:
        curves: List[np.ndarray] = []
        average_precisions: List[float] = []
        for scores in scores_by_model[name]:
            curve, average_precision = interpolate_curve(
                labels_reference, scores, recall_grid
            )
            curves.append(curve)
            average_precisions.append(average_precision)
        curve_array = np.stack(curves)
        mean_curve = curve_array.mean(axis=0)
        std_curve = curve_array.std(axis=0, ddof=1)
        ap_array = np.asarray(average_precisions, dtype=np.float64)
        curve_payload[f"{name}_precision_mean"] = mean_curve
        curve_payload[f"{name}_precision_std"] = std_curve
        curve_payload[f"{name}_average_precision"] = ap_array
        summary["models"][name] = {
            "display_name": DISPLAY_NAMES[name],
            "average_precision_mean": float(ap_array.mean()),
            "average_precision_std": float(ap_array.std(ddof=1)),
            "runs": [float(value) for value in ap_array],
        }
        label = (
            f"{DISPLAY_NAMES[name]} "
            f"(AUPRC={ap_array.mean():.3f}±{ap_array.std(ddof=1):.3f})"
        )
        axis.plot(
            recall_grid,
            mean_curve,
            color=COLORS[name],
            linewidth=2.8 if name == "full" else 1.8,
            linestyle="-" if name == "full" else "--",
            label=label,
            zorder=5 if name == "full" else 3,
        )
        axis.fill_between(
            recall_grid,
            np.clip(mean_curve - std_curve, 0.0, 1.0),
            np.clip(mean_curve + std_curve, 0.0, 1.0),
            color=COLORS[name],
            alpha=0.10 if name == "full" else 0.045,
            linewidth=0,
        )

    prevalence = float((labels_reference == 1).mean())
    axis.axhline(
        prevalence,
        color="#6E6E6E",
        linewidth=1.2,
        linestyle=":",
        label=f"随机分类器 ({prevalence:.3f})",
        zorder=1,
    )
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.02)
    axis.set_xlabel("召回率 Re")
    axis.set_ylabel("精确率 P")
    axis.set_title("Elliptic++测试集精确率-召回率曲线（5次独立运行）")
    axis.grid(True, which="major", linestyle=":", linewidth=0.7, alpha=0.55)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.legend(loc="lower left", frameon=True, framealpha=0.94)

    np.savez_compressed(output_root / "pr_curve_data.npz", **curve_payload)
    (output_root / "pr_curve_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    fig.savefig(output_root / "pr_curve_preview.png", dpi=300, bbox_inches="tight")
    axis.set_title("")
    fig.savefig(output_root / "pr_curve.png", dpi=400, bbox_inches="tight")
    fig.savefig(output_root / "pr_curve.pdf", bbox_inches="tight")
    fig.savefig(output_root / "pr_curve.svg", bbox_inches="tight")
    plt.close(fig)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
