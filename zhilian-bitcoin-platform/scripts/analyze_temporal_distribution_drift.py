from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from scipy.stats import spearmanr


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import evaluate_temporal_robustness as robustness  # noqa: E402
import train_main_baselines as baseline_training  # noqa: E402
import train_temporal_memory as temporal_training  # noqa: E402


BLOCKS = {
    "near": list(range(36, 40)),
    "middle": list(range(40, 45)),
    "far": list(range(45, 50)),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quantify label-independent feature drift with PSI."
    )
    parser.add_argument(
        "--robustness-root",
        type=Path,
        default=Path(
            "artifacts/temporal_robustness/erl_mechanism_pooled_f1"
        ),
    )
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--max-reference-rows", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=20260810)
    return parser.parse_args()


def project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def finite_column(values: np.ndarray) -> np.ndarray:
    return values[np.isfinite(values)]


def fit_psi_reference(
    reference: np.ndarray, bins: int
) -> List[Tuple[np.ndarray, np.ndarray]]:
    specifications: List[Tuple[np.ndarray, np.ndarray]] = []
    quantiles = np.linspace(0.0, 1.0, bins + 1)[1:-1]
    for feature_index in range(reference.shape[1]):
        values = finite_column(reference[:, feature_index])
        if values.size == 0:
            edges = np.asarray([-np.inf, np.inf], dtype=np.float64)
        else:
            internal = np.unique(np.quantile(values, quantiles))
            edges = np.concatenate(([-np.inf], internal, [np.inf]))
        counts, _ = np.histogram(values, bins=edges)
        proportions = (counts.astype(np.float64) + 1e-6) / (
            counts.sum() + 1e-6 * counts.size
        )
        specifications.append((edges, proportions))
    return specifications


def score_psi(
    values: np.ndarray,
    specifications: Sequence[Tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    scores = np.zeros(len(specifications), dtype=np.float64)
    for feature_index, (edges, reference_proportions) in enumerate(
        specifications
    ):
        column = finite_column(values[:, feature_index])
        counts, _ = np.histogram(column, bins=edges)
        current_proportions = (counts.astype(np.float64) + 1e-6) / (
            counts.sum() + 1e-6 * counts.size
        )
        scores[feature_index] = np.sum(
            (current_proportions - reference_proportions)
            * np.log(current_proportions / reference_proportions)
        )
    return scores


def safe_spearman(x: Sequence[float], y: Sequence[float]) -> Dict[str, float]:
    result = spearmanr(np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64))
    return {"rho": float(result.statistic), "pvalue": float(result.pvalue)}


def labeled_features(
    snapshots: Dict[int, Any], times: Sequence[int], label: int | None
) -> np.ndarray:
    features: List[torch.Tensor] = []
    for time_id in times:
        snapshot = snapshots[time_id]
        mask = snapshot.labeled_mask
        if label is not None:
            mask = mask & (snapshot.labels == label)
        features.append(snapshot.transaction_features[mask])
    return torch.cat(features, dim=0).cpu().numpy()


def conditional_block_drift(
    snapshots: Dict[int, Any],
    train_times: Sequence[int],
    bins: int,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for name, label in (("all_labeled", None), ("legal", 0), ("illegal", 1)):
        reference = labeled_features(snapshots, train_times, label)
        specifications = fit_psi_reference(reference, bins)
        result[name] = {
            "reference_rows": int(reference.shape[0]),
            "blocks": {},
        }
        for block_name, block_times in BLOCKS.items():
            current = labeled_features(snapshots, block_times, label)
            feature_psi = score_psi(current, specifications)
            result[name]["blocks"][block_name] = {
                "rows": int(current.shape[0]),
                "psi_mean": float(feature_psi.mean()),
                "psi_median": float(np.median(feature_psi)),
                "psi_p90": float(np.quantile(feature_psi, 0.90)),
            }
    return result


def plot_conditional_drift(
    output_root: Path, conditional: Dict[str, Any]
) -> None:
    robustness.configure_plot_style()
    blocks = ["near", "middle", "far"]
    labels = ["近期（36—39）", "中期（40—44）", "远期（45—49）"]
    series = [
        ("all_labeled", "全部标注交易", "#7F7F7F"),
        ("legal", "合法交易", "#5B9BD5"),
        ("illegal", "非法交易", "#C00000"),
    ]
    positions = np.arange(len(blocks), dtype=np.float64)
    width = 0.24
    figure, axis = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
    for index, (key, display_name, color) in enumerate(series):
        values = [
            conditional[key]["blocks"][block]["psi_mean"] for block in blocks
        ]
        axis.bar(
            positions + (index - 1.0) * width,
            values,
            width=width,
            color=color,
            alpha=0.92,
            edgecolor="white",
            linewidth=0.6,
            label=display_name,
        )
    axis.set_xticks(positions, labels)
    axis.set_ylabel("类别条件平均PSI")
    axis.grid(True, axis="y", linestyle=":", linewidth=0.7, alpha=0.55)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.legend(loc="upper left")
    for suffix in ("png", "pdf", "svg"):
        figure.savefig(
            output_root / f"conditional_drift.{suffix}",
            dpi=400 if suffix == "png" else None,
            bbox_inches="tight",
        )
    plt.close(figure)


def plot_evidence(output_root: Path, rows: Sequence[Dict[str, Any]]) -> None:
    robustness.configure_plot_style()
    times = np.asarray([row["time_id"] for row in rows], dtype=np.int64)
    psi_mean = np.asarray([row["psi_mean"] for row in rows])
    positive_rate = np.asarray([row["illegal_rate"] for row in rows])
    full_ap = np.asarray([row["full_average_precision"] for row in rows])
    no_erl_ap = np.asarray([row["no_erl_average_precision"] for row in rows])

    figure, axes = plt.subplots(
        2, 1, figsize=(8.2, 6.8), sharex=True, constrained_layout=True
    )
    axes[0].plot(
        times,
        psi_mean,
        color="#7030A0",
        marker="o",
        linewidth=2.2,
        label="特征PSI（相对训练期）",
    )
    axes[0].set_ylabel("平均PSI")
    axes[0].grid(True, linestyle=":", linewidth=0.7, alpha=0.55)
    rate_axis = axes[0].twinx()
    rate_axis.bar(
        times,
        positive_rate,
        width=0.55,
        color="#A5A5A5",
        alpha=0.30,
        label="非法交易比例",
    )
    rate_axis.set_ylabel("非法交易比例")
    handles_1, labels_1 = axes[0].get_legend_handles_labels()
    handles_2, labels_2 = rate_axis.get_legend_handles_labels()
    axes[0].legend(handles_1 + handles_2, labels_1 + labels_2, loc="upper left")

    axes[1].plot(
        times,
        full_ap,
        color="#C00000",
        marker="o",
        linewidth=2.5,
        label="本文方法",
    )
    axes[1].plot(
        times,
        no_erl_ap,
        color="#2F5597",
        marker="o",
        linewidth=2.1,
        label="仅去除ERL",
    )
    axes[1].set_xlabel("测试时间片")
    axes[1].set_ylabel("AUPRC")
    axes[1].set_xticks(times)
    axes[1].set_ylim(0.0, 1.0)
    axes[1].grid(True, linestyle=":", linewidth=0.7, alpha=0.55)
    axes[1].legend(loc="upper right")
    for axis in axes:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    for suffix in ("png", "pdf", "svg"):
        figure.savefig(
            output_root / f"drift_evidence.{suffix}",
            dpi=400 if suffix == "png" else None,
            bbox_inches="tight",
        )
    plt.close(figure)


def main() -> None:
    args = parse_args()
    output_root = project_path(args.robustness_root)
    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    with (PROJECT_ROOT / "configs" / "main_baselines.yaml").open(
        "r", encoding="utf-8"
    ) as stream:
        config = yaml.safe_load(stream)
    train_times = baseline_training.expand_interval(config["split"]["train"])
    test_times = baseline_training.expand_interval(config["split"]["test"])
    snapshots, _ = temporal_training.load_snapshots(
        project_path(Path(config["data"]["processed_root"]))
    )

    reference = torch.cat(
        [snapshots[time_id].transaction_features for time_id in train_times],
        dim=0,
    ).cpu().numpy()
    if reference.shape[0] > args.max_reference_rows:
        generator = np.random.default_rng(args.seed)
        selected = generator.choice(
            reference.shape[0], args.max_reference_rows, replace=False
        )
        reference = reference[selected]
    specifications = fit_psi_reference(reference, args.bins)

    rows: List[Dict[str, Any]] = []
    for time_id in test_times:
        snapshot = snapshots[time_id]
        feature_psi = score_psi(
            snapshot.transaction_features.cpu().numpy(), specifications
        )
        labels = snapshot.labels[snapshot.labeled_mask].cpu().numpy()
        full = summary["methods"]["full"]["per_time"][str(time_id)]
        no_erl = summary["methods"]["no_erl"]["per_time"][str(time_id)]
        illegal_count = int((labels == 1).sum())
        labeled_count = int(labels.size)
        rows.append(
            {
                "time_id": time_id,
                "transactions": int(snapshot.num_transactions),
                "labeled_transactions": labeled_count,
                "illegal_transactions": illegal_count,
                "illegal_rate": illegal_count / labeled_count,
                "psi_mean": float(feature_psi.mean()),
                "psi_median": float(np.median(feature_psi)),
                "psi_p90": float(np.quantile(feature_psi, 0.90)),
                "psi_feature_share_gt_0_1": float((feature_psi > 0.1).mean()),
                "full_f1": full["f1"]["mean"],
                "full_average_precision": full["average_precision"]["mean"],
                "no_erl_f1": no_erl["f1"]["mean"],
                "no_erl_average_precision": no_erl["average_precision"]["mean"],
                "f1_delta": full["f1"]["mean"] - no_erl["f1"]["mean"],
                "average_precision_delta": full["average_precision"]["mean"]
                - no_erl["average_precision"]["mean"],
            }
        )

    with (output_root / "drift_by_time.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    block_summary: Dict[str, Any] = {}
    for block_name, block_times in BLOCKS.items():
        selected = [row for row in rows if row["time_id"] in block_times]
        illegal = sum(row["illegal_transactions"] for row in selected)
        labeled = sum(row["labeled_transactions"] for row in selected)
        block_summary[block_name] = {
            "times": block_times,
            "psi_mean_across_snapshots": float(
                np.mean([row["psi_mean"] for row in selected])
            ),
            "illegal_rate": illegal / labeled,
        }

    psi = [row["psi_mean"] for row in rows]
    conditional = conditional_block_drift(
        snapshots, train_times, args.bins
    )
    with (output_root / "conditional_drift_by_block.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as stream:
        fieldnames = [
            "class_group",
            "block",
            "time_range",
            "rows",
            "psi_mean",
            "psi_median",
            "psi_p90",
        ]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for class_group, payload in conditional.items():
            for block_name, values in payload["blocks"].items():
                writer.writerow(
                    {
                        "class_group": class_group,
                        "block": block_name,
                        "time_range": f"{min(BLOCKS[block_name])}-{max(BLOCKS[block_name])}",
                        **values,
                    }
                )
    diagnostics = {
        "protocol": {
            "reference_times": train_times,
            "test_times": test_times,
            "psi_bins": args.bins,
            "reference_rows": int(reference.shape[0]),
            "feature_count": int(reference.shape[1]),
            "labels_used_for_psi": False,
        },
        "blocks": block_summary,
        "conditional_feature_drift": conditional,
        "correlations": {
            "psi_vs_full_f1": safe_spearman(
                psi, [row["full_f1"] for row in rows]
            ),
            "psi_vs_full_average_precision": safe_spearman(
                psi, [row["full_average_precision"] for row in rows]
            ),
            "psi_vs_erl_f1_gain": safe_spearman(
                psi, [row["f1_delta"] for row in rows]
            ),
            "psi_vs_erl_average_precision_gain": safe_spearman(
                psi, [row["average_precision_delta"] for row in rows]
            ),
        },
        "per_time": rows,
    }
    (output_root / "drift_diagnostics.json").write_text(
        json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    plot_evidence(output_root, rows)
    plot_conditional_drift(output_root, conditional)
    print(json.dumps(diagnostics["blocks"], ensure_ascii=False, indent=2))
    print(
        json.dumps(
            diagnostics["conditional_feature_drift"],
            ensure_ascii=False,
            indent=2,
        )
    )
    print(json.dumps(diagnostics["correlations"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
