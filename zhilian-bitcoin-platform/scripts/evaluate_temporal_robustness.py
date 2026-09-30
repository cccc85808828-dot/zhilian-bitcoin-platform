from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from scipy.stats import ttest_rel, wilcoxon
from sklearn.metrics import (
    average_precision_score,
    precision_recall_fscore_support,
)


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
ABLATIONS = ["no_erl", "no_robust"]
METHOD_ORDER = ["full"] + ABLATIONS + BASELINES
DISPLAY_NAMES = {
    "full": "本文方法",
    "no_erl": "仅去除ERL",
    "no_robust": "去除稳健学习模块",
    "hgnn": "HGNN",
    "hgt": "HGT",
    "heterosage": "HeteroSAGE",
    "evolvegcn_o": "EvolveGCN-O",
    "bf_hgn": "BF-HGN",
}
COLORS = {
    "full": "#C00000",
    "no_erl": "#2F5597",
    "no_robust": "#7F7F7F",
    "hgnn": "#0072B2",
    "hgt": "#E69F00",
    "heterosage": "#009E73",
    "evolvegcn_o": "#CC79A7",
    "bf_hgn": "#56B4E9",
}
BLOCKS = {
    "near": list(range(36, 40)),
    "middle": list(range(40, 45)),
    "far": list(range(45, 50)),
}
SUMMARY_METRICS = [
    "pooled_precision",
    "pooled_accuracy",
    "pooled_recall",
    "pooled_f1",
    "pooled_average_precision",
    "mean_snapshot_f1",
    "std_snapshot_f1",
    "worst_snapshot_f1",
    "mean_snapshot_average_precision",
    "worst_snapshot_average_precision",
    "near_precision",
    "near_accuracy",
    "near_recall",
    "near_f1",
    "near_average_precision",
    "middle_precision",
    "middle_accuracy",
    "middle_recall",
    "middle_f1",
    "middle_average_precision",
    "far_precision",
    "far_accuracy",
    "far_recall",
    "far_f1",
    "far_average_precision",
    "degradation_f1",
    "degradation_average_precision",
    "f1_slope",
]


PredictionByTime = Dict[int, Tuple[np.ndarray, np.ndarray]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate temporal robustness under fixed validation thresholds."
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/temporal_robustness/erl_mechanism_pooled_f1"),
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


def binary_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> Dict[str, float | int]:
    predictions = (scores >= threshold).astype(np.int64)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="binary",
        pos_label=1,
        zero_division=0,
    )
    positives = int((labels == 1).sum())
    negatives = int((labels == 0).sum())
    average_precision = (
        float(average_precision_score(labels, scores))
        if positives > 0 and negatives > 0
        else float("nan")
    )
    return {
        "precision": float(precision),
        "accuracy": float((predictions == labels).mean()),
        "recall": float(recall),
        "f1": float(f1),
        "average_precision": average_precision,
        "threshold": float(threshold),
        "positives": positives,
        "negatives": negatives,
        "predicted_positives": int(predictions.sum()),
    }


def concatenate_by_time(
    predictions: PredictionByTime,
    times: Sequence[int],
) -> Tuple[np.ndarray, np.ndarray]:
    return (
        np.concatenate([predictions[time_id][0] for time_id in times]),
        np.concatenate([predictions[time_id][1] for time_id in times]),
    )


def load_fusion_predictions(
    variant: str,
    seed: int,
    snapshots: Dict[int, EllipticPPHypergraphSnapshot],
    global_address_count: int,
    test_times: Sequence[int],
    device: torch.device,
    fusion_root: Path | None = None,
) -> Tuple[PredictionByTime, float]:
    if fusion_root is None:
        fusion_root = PROJECT_ROOT / "artifacts" / "pooled_f1" / "ablation"
    result_root = fusion_root / variant / f"seed_{seed}"
    payload = torch.load(
        result_root / "best.pt",
        map_location="cpu",
        weights_only=False,
    )
    statistics = audit.feature_statistics(payload["transaction_statistics"])
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
    test_features, test_labels, test_time_ids = (
        quantile_fusion.pack_labeled_transactions(
            snapshots, test_times, statistics
        )
    )
    attribute_logits = quantile_fusion.batched_logits(
        attribute_model, test_features, 768, device
    )
    graph_checkpoint = (
        PROJECT_ROOT
        / "artifacts"
        / "ablation"
        / variant
        / "graph"
        / f"seed_{seed}"
        / "best.pt"
    )
    (
        _,
        graph_logits,
        _,
        graph_labels,
        _,
    ) = quantile_fusion.replay_graph_logits(
        graph_checkpoint,
        snapshots,
        global_address_count,
        device,
    )
    if not np.array_equal(test_labels.numpy(), graph_labels):
        raise RuntimeError(
            f"Fusion labels are misaligned for {variant}, seed {seed}."
        )
    attribute_weight = float(payload["fusion"]["attribute_weight"])
    scores = (
        (1.0 - attribute_weight) * graph_logits
        + attribute_weight * attribute_logits
    )
    threshold = float(payload["fusion"]["threshold"])
    labels = test_labels.numpy()
    predictions = {
        int(time_id): (
            labels[test_time_ids == time_id],
            scores[test_time_ids == time_id],
        )
        for time_id in test_times
    }
    del attribute_model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return predictions, threshold


def load_baseline_predictions(
    name: str,
    seed: int,
    snapshots: Mapping[int, EllipticPPHypergraphSnapshot],
    transaction_edges: Mapping[int, torch.Tensor],
    train_times: Sequence[int],
    validation_times: Sequence[int],
    test_times: Sequence[int],
    device: torch.device,
) -> Tuple[PredictionByTime, float]:
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
    model = baseline_training.build_model(
        name,
        checkpoint["config"]["model"],
        audit.feature_statistics(checkpoint["address_statistics"]),
        audit.feature_statistics(checkpoint["transaction_statistics"]),
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
    elif name == "bf_hgn":
        bf_config = checkpoint["config"]["model"].get("bf_hgn", {})
        labels, scores, _ = baseline_training.predict_bf_hgn(
            model=model,
            snapshots=dict(snapshots),
            transaction_edges=dict(transaction_edges),
            replay_times=list(train_times) + list(validation_times) + list(test_times),
            prediction_times=list(test_times),
            device=device,
            window_size=int(bf_config.get("window_size", 5)),
        )
        predictions = {}
        offset = 0
        for time_id in test_times:
            count = int(snapshots[time_id].labeled_mask.sum())
            predictions[int(time_id)] = (
                labels[offset : offset + count],
                scores[offset : offset + count],
            )
            offset += count
        if offset != labels.size:
            raise RuntimeError(f"BF-HGN split failed for seed {seed}.")
    else:
        predictions = audit.predict_static_by_time(
            name,
            model,
            snapshots,
            transaction_edges,
            test_times,
            device,
        )
    metrics_payload = json.loads(
        (model_root / "metrics.json").read_text(encoding="utf-8")
    )
    threshold = float(metrics_payload["validation"]["threshold"])
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return predictions, threshold


def evaluate_run(
    predictions: PredictionByTime,
    threshold: float,
    test_times: Sequence[int],
) -> Dict[str, Any]:
    per_time = {
        str(time_id): binary_metrics(
            predictions[time_id][0], predictions[time_id][1], threshold
        )
        for time_id in test_times
    }
    pooled_labels, pooled_scores = concatenate_by_time(predictions, test_times)
    pooled = binary_metrics(pooled_labels, pooled_scores, threshold)
    block_metrics: Dict[str, Dict[str, float | int]] = {}
    for block_name, block_times in BLOCKS.items():
        labels, scores = concatenate_by_time(predictions, block_times)
        block_metrics[block_name] = binary_metrics(labels, scores, threshold)
    snapshot_f1 = np.asarray(
        [per_time[str(time_id)]["f1"] for time_id in test_times],
        dtype=np.float64,
    )
    snapshot_ap = np.asarray(
        [per_time[str(time_id)]["average_precision"] for time_id in test_times],
        dtype=np.float64,
    )
    run_summary = {
        "pooled_precision": float(pooled["precision"]),
        "pooled_accuracy": float(pooled["accuracy"]),
        "pooled_recall": float(pooled["recall"]),
        "pooled_f1": float(pooled["f1"]),
        "pooled_average_precision": float(pooled["average_precision"]),
        "mean_snapshot_f1": float(snapshot_f1.mean()),
        "std_snapshot_f1": float(snapshot_f1.std(ddof=1)),
        "worst_snapshot_f1": float(snapshot_f1.min()),
        "mean_snapshot_average_precision": float(np.nanmean(snapshot_ap)),
        "worst_snapshot_average_precision": float(np.nanmin(snapshot_ap)),
        **{
            f"{block_name}_{metric}": float(block_metrics[block_name][metric])
            for block_name in BLOCKS
            for metric in (
                "precision",
                "accuracy",
                "recall",
                "f1",
                "average_precision",
            )
        },
        "degradation_f1": float(
            block_metrics["near"]["f1"] - block_metrics["far"]["f1"]
        ),
        "near_average_precision": float(
            block_metrics["near"]["average_precision"]
        ),
        "middle_average_precision": float(
            block_metrics["middle"]["average_precision"]
        ),
        "far_average_precision": float(
            block_metrics["far"]["average_precision"]
        ),
        "degradation_average_precision": float(
            block_metrics["near"]["average_precision"]
            - block_metrics["far"]["average_precision"]
        ),
        "f1_slope": float(np.polyfit(np.asarray(test_times), snapshot_f1, 1)[0]),
    }
    return {
        "threshold": float(threshold),
        "summary": run_summary,
        "pooled": pooled,
        "blocks": block_metrics,
        "per_time": per_time,
    }


def aggregate(values: Sequence[float]) -> Dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if array.size > 1 else 0.0,
    }


def write_summary_csv(path: Path, methods: Mapping[str, Any]) -> None:
    fieldnames = ["method", "display_name"] + [
        f"{metric}_{stat}"
        for metric in SUMMARY_METRICS
        for stat in ("mean", "std")
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for method in METHOD_ORDER:
            row: Dict[str, Any] = {
                "method": method,
                "display_name": DISPLAY_NAMES[method],
            }
            for metric in SUMMARY_METRICS:
                row[f"{metric}_mean"] = methods[method]["metrics"][metric]["mean"]
                row[f"{metric}_std"] = methods[method]["metrics"][metric]["std"]
            writer.writerow(row)


def write_per_time_csv(path: Path, methods: Mapping[str, Any]) -> None:
    fields = [
        "method",
        "display_name",
        "time",
        "precision_mean",
        "precision_std",
        "accuracy_mean",
        "accuracy_std",
        "recall_mean",
        "recall_std",
        "f1_mean",
        "f1_std",
        "average_precision_mean",
        "average_precision_std",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for method in METHOD_ORDER:
            for time_id, metrics in methods[method]["per_time"].items():
                row: Dict[str, Any] = {
                    "method": method,
                    "display_name": DISPLAY_NAMES[method],
                    "time": int(time_id),
                }
                for metric in (
                    "precision",
                    "accuracy",
                    "recall",
                    "f1",
                    "average_precision",
                ):
                    row[f"{metric}_mean"] = metrics[metric]["mean"]
                    row[f"{metric}_std"] = metrics[metric]["std"]
                writer.writerow(row)


def write_block_csv(path: Path, methods: Mapping[str, Any]) -> None:
    metric_names = (
        "precision",
        "accuracy",
        "recall",
        "f1",
        "average_precision",
    )
    fields = ["method", "display_name", "block", "time_range"] + [
        f"{metric}_{stat}"
        for metric in metric_names
        for stat in ("mean", "std")
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for method in METHOD_ORDER:
            for block, times in BLOCKS.items():
                row: Dict[str, Any] = {
                    "method": method,
                    "display_name": DISPLAY_NAMES[method],
                    "block": block,
                    "time_range": f"{min(times)}-{max(times)}",
                }
                for metric in metric_names:
                    values = methods[method]["metrics"][f"{block}_{metric}"]
                    row[f"{metric}_mean"] = values["mean"]
                    row[f"{metric}_std"] = values["std"]
                writer.writerow(row)


def paired_robust_analysis(
    runs: Mapping[str, Mapping[int, Any]], comparison_method: str
) -> Dict[str, Any]:
    results: Dict[str, Any] = {}
    seeds = sorted(runs["full"])
    lower_is_better = {
        "std_snapshot_f1",
        "degradation_f1",
        "degradation_average_precision",
    }
    for metric in SUMMARY_METRICS:
        full = np.asarray(
            [runs["full"][seed]["summary"][metric] for seed in seeds],
            dtype=np.float64,
        )
        comparison = np.asarray(
            [runs[comparison_method][seed]["summary"][metric] for seed in seeds],
            dtype=np.float64,
        )
        raw_delta = full - comparison
        oriented_delta = -raw_delta if metric in lower_is_better else raw_delta
        try:
            wilcoxon_p = float(wilcoxon(oriented_delta).pvalue)
        except ValueError:
            wilcoxon_p = 1.0
        t_result = ttest_rel(full, comparison)
        results[metric] = {
            f"raw_full_minus_{comparison_method}": aggregate(raw_delta.tolist()),
            "oriented_improvement": aggregate(oriented_delta.tolist()),
            "full_wins": int((oriented_delta > 0.0).sum()),
            "paired_t_pvalue": float(t_result.pvalue),
            "wilcoxon_pvalue": wilcoxon_p,
        }
    return results


def configure_plot_style() -> None:
    mpl.rcParams.update(
        {
            "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "font.size": 10.5,
            "axes.labelsize": 11.5,
            "legend.fontsize": 9.3,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
        }
    )


def plot_time_f1(output_root: Path, methods: Mapping[str, Any]) -> None:
    configure_plot_style()
    times = np.arange(36, 50)
    fig, axis = plt.subplots(figsize=(7.3, 4.8), constrained_layout=True)
    axis.axvspan(35.5, 39.5, color="#5B9BD5", alpha=0.035)
    axis.axvspan(39.5, 44.5, color="#A5A5A5", alpha=0.035)
    axis.axvspan(44.5, 49.5, color="#ED7D31", alpha=0.035)
    plotted = ["full", "no_erl", "hgnn", "hgt", "heterosage", "bf_hgn"]
    for method in plotted:
        means = np.asarray(
            [methods[method]["per_time"][str(t)]["f1"]["mean"] for t in times]
        )
        stds = np.asarray(
            [methods[method]["per_time"][str(t)]["f1"]["std"] for t in times]
        )
        axis.plot(
            times,
            means,
            color=COLORS[method],
            linewidth=2.7 if method == "full" else (2.2 if method == "no_erl" else 1.5),
            linestyle="-" if method in {"full", "no_erl"} else "--",
            marker="o" if method in {"full", "no_erl"} else None,
            markersize=3.8,
            label=DISPLAY_NAMES[method],
            zorder=5 if method == "full" else 3,
        )
        if method in {"full", "no_erl"}:
            axis.fill_between(
                times,
                np.clip(means - stds, 0.0, 1.0),
                np.clip(means + stds, 0.0, 1.0),
                color=COLORS[method],
                alpha=0.10,
                linewidth=0,
            )
    axis.set_xlim(35.6, 49.4)
    axis.set_ylim(0.0, 1.0)
    axis.set_xticks(times)
    axis.set_xlabel("测试时间片")
    axis.set_ylabel("F1")
    axis.grid(True, linestyle=":", linewidth=0.7, alpha=0.55)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.legend(ncol=2, loc="lower left", frameon=True, framealpha=0.94)
    fig.savefig(output_root / "f1_over_time.png", dpi=400, bbox_inches="tight")
    fig.savefig(output_root / "f1_over_time.pdf", bbox_inches="tight")
    fig.savefig(output_root / "f1_over_time.svg", bbox_inches="tight")
    plt.close(fig)


def plot_horizon_f1(output_root: Path, methods: Mapping[str, Any]) -> None:
    configure_plot_style()
    shown = ["full", "no_erl", "hgt", "heterosage", "bf_hgn"]
    blocks = ["near_f1", "middle_f1", "far_f1"]
    block_labels = ["近期（36—39）", "中期（40—44）", "远期（45—49）"]
    positions = np.arange(len(blocks), dtype=np.float64)
    width = 0.16
    fig, axis = plt.subplots(figsize=(7.2, 4.65), constrained_layout=True)
    for index, method in enumerate(shown):
        means = [methods[method]["metrics"][metric]["mean"] for metric in blocks]
        stds = [methods[method]["metrics"][metric]["std"] for metric in blocks]
        axis.bar(
            positions + (index - 2.0) * width,
            means,
            width=width,
            yerr=stds,
            capsize=2.5,
            color=COLORS[method],
            alpha=0.92,
            edgecolor="white",
            linewidth=0.6,
            label=DISPLAY_NAMES[method],
        )
    axis.set_xticks(positions, block_labels)
    axis.set_ylabel("F1")
    axis.set_ylim(0.0, 1.0)
    axis.grid(True, axis="y", linestyle=":", linewidth=0.7, alpha=0.55)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.legend(ncol=2, loc="upper right", frameon=True, framealpha=0.94)
    fig.savefig(output_root / "horizon_f1.png", dpi=400, bbox_inches="tight")
    fig.savefig(output_root / "horizon_f1.pdf", bbox_inches="tight")
    fig.savefig(output_root / "horizon_f1.svg", bbox_inches="tight")
    plt.close(fig)


def plot_time_auprc(output_root: Path, methods: Mapping[str, Any]) -> None:
    configure_plot_style()
    times = np.arange(36, 50)
    fig, axis = plt.subplots(figsize=(7.3, 4.8), constrained_layout=True)
    axis.axvspan(35.5, 39.5, color="#5B9BD5", alpha=0.035)
    axis.axvspan(39.5, 44.5, color="#A5A5A5", alpha=0.035)
    axis.axvspan(44.5, 49.5, color="#ED7D31", alpha=0.035)
    plotted = ["full", "no_erl", "hgnn", "hgt", "heterosage", "bf_hgn"]
    for method in plotted:
        means = np.asarray(
            [
                methods[method]["per_time"][str(t)]["average_precision"]["mean"]
                for t in times
            ]
        )
        stds = np.asarray(
            [
                methods[method]["per_time"][str(t)]["average_precision"]["std"]
                for t in times
            ]
        )
        axis.plot(
            times,
            means,
            color=COLORS[method],
            linewidth=2.7 if method == "full" else (2.2 if method == "no_erl" else 1.5),
            linestyle="-" if method in {"full", "no_erl"} else "--",
            marker="o" if method in {"full", "no_erl"} else None,
            markersize=3.8,
            label=DISPLAY_NAMES[method],
            zorder=5 if method == "full" else 3,
        )
        if method in {"full", "no_erl"}:
            axis.fill_between(
                times,
                np.clip(means - stds, 0.0, 1.0),
                np.clip(means + stds, 0.0, 1.0),
                color=COLORS[method],
                alpha=0.10,
                linewidth=0,
            )
    axis.set_xlim(35.6, 49.4)
    axis.set_ylim(0.0, 1.0)
    axis.set_xticks(times)
    axis.set_xlabel("测试时间片")
    axis.set_ylabel("AUPRC")
    axis.grid(True, linestyle=":", linewidth=0.7, alpha=0.55)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.legend(ncol=2, loc="lower left", frameon=True, framealpha=0.94)
    fig.savefig(output_root / "auprc_over_time.png", dpi=400, bbox_inches="tight")
    fig.savefig(output_root / "auprc_over_time.pdf", bbox_inches="tight")
    fig.savefig(output_root / "auprc_over_time.svg", bbox_inches="tight")
    plt.close(fig)


def plot_horizon_auprc(output_root: Path, methods: Mapping[str, Any]) -> None:
    configure_plot_style()
    shown = ["full", "no_erl", "hgt", "heterosage", "bf_hgn"]
    blocks = [
        "near_average_precision",
        "middle_average_precision",
        "far_average_precision",
    ]
    block_labels = ["近期（36—39）", "中期（40—44）", "远期（45—49）"]
    positions = np.arange(len(blocks), dtype=np.float64)
    width = 0.16
    fig, axis = plt.subplots(figsize=(7.2, 4.65), constrained_layout=True)
    for index, method in enumerate(shown):
        means = [methods[method]["metrics"][metric]["mean"] for metric in blocks]
        stds = [methods[method]["metrics"][metric]["std"] for metric in blocks]
        axis.bar(
            positions + (index - 2.0) * width,
            means,
            width=width,
            yerr=stds,
            capsize=2.5,
            color=COLORS[method],
            alpha=0.92,
            edgecolor="white",
            linewidth=0.6,
            label=DISPLAY_NAMES[method],
        )
    axis.set_xticks(positions, block_labels)
    axis.set_ylabel("AUPRC")
    axis.set_ylim(0.0, 1.0)
    axis.grid(True, axis="y", linestyle=":", linewidth=0.7, alpha=0.55)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.legend(ncol=2, loc="upper right", frameon=True, framealpha=0.94)
    fig.savefig(output_root / "horizon_auprc.png", dpi=400, bbox_inches="tight")
    fig.savefig(output_root / "horizon_auprc.pdf", bbox_inches="tight")
    fig.savefig(output_root / "horizon_auprc.svg", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    output_root = project_path(args.output_root)
    fusion_root = project_path(args.fusion_root)
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
    runs: Dict[str, Dict[int, Any]] = {method: {} for method in METHOD_ORDER}

    for seed in seeds:
        for method, variant in (
            ("full", "full"),
            ("no_erl", "no_erl"),
            ("no_robust", "no_robust"),
        ):
            print(f"[seed={seed}] replay {method}", flush=True)
            predictions, threshold = load_fusion_predictions(
                variant,
                seed,
                snapshots,
                global_address_count,
                test_times,
                device,
                fusion_root,
            )
            runs[method][seed] = evaluate_run(
                predictions, threshold, test_times
            )
        for method in BASELINES:
            print(f"[seed={seed}] replay {method}", flush=True)
            predictions, threshold = load_baseline_predictions(
                method,
                seed,
                snapshots,
                transaction_edges,
                train_times,
                validation_times,
                test_times,
                device,
            )
            runs[method][seed] = evaluate_run(
                predictions, threshold, test_times
            )

    methods: Dict[str, Any] = {}
    for method in METHOD_ORDER:
        method_runs = runs[method]
        metrics = {
            metric: aggregate(
                [method_runs[seed]["summary"][metric] for seed in seeds]
            )
            for metric in SUMMARY_METRICS
        }
        per_time = {
            str(time_id): {
                metric: aggregate(
                    [
                        method_runs[seed]["per_time"][str(time_id)][metric]
                        for seed in seeds
                    ]
                )
                for metric in (
                    "precision",
                    "accuracy",
                    "recall",
                    "f1",
                    "average_precision",
                )
            }
            for time_id in test_times
        }
        methods[method] = {
            "display_name": DISPLAY_NAMES[method],
            "metrics": metrics,
            "per_time": per_time,
            "runs": [
                {
                    "seed": seed,
                    "threshold": method_runs[seed]["threshold"],
                    **method_runs[seed]["summary"],
                }
                for seed in seeds
            ],
        }

    summary = {
        "status": "PASS",
        "protocol": {
            "train": [1, 30],
            "validation": [31, 35],
            "test": [36, 49],
            "threshold_selection": "pooled_validation_f1",
            "test_threshold_refit": False,
            "uses_test_labels_for_selection": False,
            "blocks": BLOCKS,
            "seeds": seeds,
        },
        "methods": methods,
        "paired_full_vs_no_erl": paired_robust_analysis(runs, "no_erl"),
        "paired_full_vs_no_robust": paired_robust_analysis(
            runs, "no_robust"
        ),
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_summary_csv(output_root / "robustness_summary.csv", methods)
    write_block_csv(output_root / "block_metrics.csv", methods)
    write_per_time_csv(output_root / "per_time_metrics.csv", methods)
    plot_time_f1(output_root, methods)
    plot_horizon_f1(output_root, methods)
    plot_time_auprc(output_root, methods)
    plot_horizon_auprc(output_root, methods)

    concise = {
        method: {
            metric: methods[method]["metrics"][metric]
            for metric in (
                "mean_snapshot_f1",
                "std_snapshot_f1",
                "worst_snapshot_f1",
                "near_f1",
                "middle_f1",
                "far_f1",
                "degradation_f1",
                "near_average_precision",
                "middle_average_precision",
                "far_average_precision",
                "degradation_average_precision",
            )
        }
        for method in METHOD_ORDER
    }
    print(json.dumps(concise, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
