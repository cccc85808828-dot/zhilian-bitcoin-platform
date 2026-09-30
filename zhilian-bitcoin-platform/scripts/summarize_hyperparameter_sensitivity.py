from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import evaluate_temporal_robustness as robustness  # noqa: E402
from run_hyperparameter_sensitivity import (  # noqa: E402
    DEFAULT_SEEDS,
    SPECS,
    reused_variant,
    value_slug,
)


METRICS = ["precision", "accuracy", "recall", "f1", "average_precision"]
DISPLAY_NAMES = {
    "erl_penalty": r"环境风险权重 $\lambda_r$",
    "rcha_weight": r"增强损失权重 $\lambda_a$",
    "representation_dim": r"表示维度 $d$",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize three-parameter sensitivity experiments."
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/hyperparameter_sensitivity"),
    )
    parser.add_argument("--fusion-root", type=Path, default=None)
    parser.add_argument(
        "--default-fusion-root",
        type=Path,
        default=Path("artifacts/pooled_f1/ablation"),
    )
    parser.add_argument("--summary-root", type=Path, default=None)
    return parser.parse_args()


def project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def aggregate(values: List[float]) -> Dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if array.size > 1 else 0.0,
    }


def paths_for(
    output_root: Path,
    fusion_root: Path,
    default_fusion_root: Path,
    parameter: str,
    value: float | int,
    seed: int,
) -> Tuple[Path, Path, str]:
    reused = reused_variant(parameter, value)
    if reused is not None:
        graph = (
            PROJECT_ROOT
            / "artifacts"
            / "ablation"
            / reused
            / "graph"
            / f"seed_{seed}"
            / "metrics.json"
        )
        fusion = (
            default_fusion_root
            / reused
            / f"seed_{seed}"
            / "metrics.json"
        )
        return graph, fusion, f"reused:{reused}"
    graph_root = output_root / parameter / f"value_{value_slug(value)}"
    selected_fusion_root = fusion_root / parameter / f"value_{value_slug(value)}"
    return (
        graph_root / "graph" / f"seed_{seed}" / "metrics.json",
        selected_fusion_root / "fusion" / f"seed_{seed}" / "metrics.json",
        "trained",
    )


def plot_summary(output_root: Path, summary: Dict[str, Any]) -> None:
    robustness.configure_plot_style()
    figure, axes = plt.subplots(
        1, 3, figsize=(12.0, 3.9), constrained_layout=True
    )
    for axis, parameter in zip(axes, SPECS):
        rows = summary["parameters"][parameter]
        x = np.asarray([row["value"] for row in rows], dtype=np.float64)
        for metric, label, color, marker in (
            ("f1", "F1", "#C00000", "o"),
            ("average_precision", "AUPRC", "#2F5597", "s"),
        ):
            means = np.asarray([row[metric]["mean"] for row in rows])
            stds = np.asarray([row[metric]["std"] for row in rows])
            axis.errorbar(
                x,
                means,
                yerr=stds,
                color=color,
                marker=marker,
                linewidth=2.0,
                capsize=3,
                label=label,
            )
        default = float(SPECS[parameter]["default"])
        axis.axvline(default, color="#7F7F7F", linestyle=":", linewidth=1.2)
        axis.set_xlabel(DISPLAY_NAMES[parameter])
        axis.set_ylim(0.62, 0.76)
        axis.grid(True, linestyle=":", linewidth=0.7, alpha=0.55)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    axes[0].set_ylabel("测试集指标")
    axes[0].legend(loc="lower right")
    for suffix in ("png", "pdf", "svg"):
        figure.savefig(
            output_root / f"sensitivity_curves.{suffix}",
            dpi=400 if suffix == "png" else None,
            bbox_inches="tight",
        )
    plt.close(figure)


def main() -> None:
    args = parse_args()
    output_root = project_path(args.output_root)
    fusion_root = project_path(
        args.output_root if args.fusion_root is None else args.fusion_root
    )
    default_fusion_root = project_path(args.default_fusion_root)
    summary_root = project_path(
        args.output_root if args.summary_root is None else args.summary_root
    )
    summary_root.mkdir(parents=True, exist_ok=True)
    summary: Dict[str, Any] = {
        "protocol": {
            "seeds": args.seeds,
            "selection_objective": "pooled_validation_f1",
            "test_labels_used_for_selection": False,
            "one_factor_at_a_time": True,
        },
        "parameters": {},
    }
    csv_rows: List[Dict[str, Any]] = []
    for parameter, spec in SPECS.items():
        parameter_rows = []
        for value in spec["values"]:
            runs = []
            for seed in args.seeds:
                graph_path, fusion_path, source = paths_for(
                    output_root,
                    fusion_root,
                    default_fusion_root,
                    parameter,
                    value,
                    seed,
                )
                graph = json.loads(graph_path.read_text(encoding="utf-8"))
                fusion = json.loads(fusion_path.read_text(encoding="utf-8"))
                test = fusion["fusion"]["test"]
                runs.append(
                    {
                        "seed": seed,
                        "source": source,
                        **{metric: float(test[metric]) for metric in METRICS},
                        "graph_parameters": int(graph["parameters"]),
                        "graph_training_seconds": float(
                            graph["duration_seconds"]
                        ),
                        "graph_peak_gpu_mb": float(graph["peak_gpu_memory_mb"]),
                    }
                )
            row: Dict[str, Any] = {
                "value": value,
                "default": value == spec["default"],
                "runs": runs,
            }
            for metric in METRICS:
                row[metric] = aggregate([run[metric] for run in runs])
            for metric in (
                "graph_parameters",
                "graph_training_seconds",
                "graph_peak_gpu_mb",
            ):
                row[metric] = aggregate(
                    [float(run[metric]) for run in runs]
                )
            parameter_rows.append(row)
            csv_row: Dict[str, Any] = {
                "parameter": parameter,
                "value": value,
                "default": row["default"],
            }
            for metric in METRICS + [
                "graph_parameters",
                "graph_training_seconds",
                "graph_peak_gpu_mb",
            ]:
                csv_row[f"{metric}_mean"] = row[metric]["mean"]
                csv_row[f"{metric}_std"] = row[metric]["std"]
            csv_rows.append(csv_row)
        summary["parameters"][parameter] = parameter_rows

    (summary_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (summary_root / "sensitivity_summary.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    plot_summary(summary_root, summary)
    print(json.dumps(summary["parameters"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
