from __future__ import annotations

import argparse
import csv
import gc
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import evaluate_temporal_robustness as robustness  # noqa: E402
import train_main_baselines as baseline_training  # noqa: E402
import train_temporal_memory as temporal_training  # noqa: E402


SEEDS = [20260806, 20260807, 20260808, 20260809, 20260810]
METHODS = [
    "gcn",
    "gat",
    "hgnn",
    "evolvegcn_o",
    "hgt",
    "graph_transformer",
    "gat_resnet",
    "heterosage",
    "bf_hgn",
    "full",
]
DISPLAY_NAMES = {
    "gcn": "GCN",
    "gat": "GAT",
    "hgnn": "HGNN",
    "evolvegcn_o": "EvolveGCN-O",
    "hgt": "HGT",
    "graph_transformer": "Graph Transformer",
    "gat_resnet": "GAT-ResNet",
    "heterosage": "HeteroSAGE",
    "bf_hgn": "BF-HGN",
    "full": "本文方法",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze training and inference efficiency."
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/computational_efficiency"),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--benchmark-seed", type=int, default=20260810)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--attribute-root",
        type=Path,
        default=Path("artifacts/quantile_fusion"),
    )
    parser.add_argument(
        "--fusion-root",
        type=Path,
        default=Path("artifacts/pooled_f1/ablation"),
    )
    return parser.parse_args()


def project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def aggregate(values: List[float]) -> Dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if array.size > 1 else 0.0,
    }


def read_baseline(seed: int, method: str) -> Dict[str, Any]:
    comparison = json.loads(
        (
            PROJECT_ROOT
            / "artifacts"
            / "main_experiment"
            / "baselines"
            / f"seed_{seed}"
            / "comparison.json"
        ).read_text(encoding="utf-8")
    )
    return next(row for row in comparison["results"] if row["model"] == method)


def read_full(seed: int, attribute_root: Path, fusion_root: Path) -> Dict[str, Any]:
    graph = json.loads(
        (
            PROJECT_ROOT
            / "artifacts"
            / "ablation"
            / "full"
            / "graph"
            / f"seed_{seed}"
            / "metrics.json"
        ).read_text(encoding="utf-8")
    )
    attribute = json.loads(
        (
            attribute_root
            / f"seed_{seed}"
            / "metrics.json"
        ).read_text(encoding="utf-8")
    )
    fusion = json.loads(
        (
            fusion_root
            / "full"
            / f"seed_{seed}"
            / "metrics.json"
        ).read_text(encoding="utf-8")
    )
    return {
        "parameters": int(graph["parameters"])
        + int(fusion["attribute_model"]["parameters"]),
        "training_seconds": float(graph["duration_seconds"])
        + float(attribute["duration_seconds"])
        + float(fusion["duration_seconds"]),
        "peak_gpu_memory_mb": float(graph["peak_gpu_memory_mb"]),
        "test": fusion["fusion"]["test"],
    }


def timed_call(
    function: Callable[[], Any], device: torch.device, repeats: int
) -> Dict[str, Any]:
    function()
    durations: List[float] = []
    peaks: List[float] = []
    for _ in range(repeats):
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        function()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            peaks.append(torch.cuda.max_memory_allocated(device) / 1024**2)
        durations.append(time.perf_counter() - start)
    return {
        "seconds": aggregate(durations),
        "peak_gpu_memory_mb": aggregate(peaks) if peaks else None,
        "raw_seconds": durations,
    }


def plot_tradeoff(output_root: Path, rows: List[Dict[str, Any]]) -> None:
    robustness.configure_plot_style()
    figure, axis = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    colors = plt.cm.tab10(np.linspace(0.0, 0.9, len(rows)))
    for color, row in zip(colors, rows):
        x = row["inference_ms_mean"]
        y = row["f1_mean"]
        size = 55.0 + 75.0 * np.sqrt(row["parameters_mean"] / 1_000_000.0)
        axis.scatter(x, y, s=size, color=color, alpha=0.88, edgecolor="white")
        axis.annotate(
            row["display_name"],
            (x, y),
            xytext=(5, 4),
            textcoords="offset points",
            fontsize=9,
        )
    axis.set_xlabel("端到端测试重放耗时（ms）")
    axis.set_ylabel("F1")
    axis.grid(True, linestyle=":", linewidth=0.7, alpha=0.55)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    for suffix in ("png", "pdf", "svg"):
        figure.savefig(
            output_root / f"efficiency_tradeoff.{suffix}",
            dpi=400 if suffix == "png" else None,
            bbox_inches="tight",
        )
    plt.close(figure)


def main() -> None:
    args = parse_args()
    output_root = project_path(args.output_root)
    attribute_root = project_path(args.attribute_root)
    fusion_root = project_path(args.fusion_root)
    output_root.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")

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
    test_count = sum(int(snapshots[t].labeled_mask.sum()) for t in test_times)

    benchmark_seed = int(args.benchmark_seed)
    inference: Dict[str, Any] = {}
    for method in METHODS:
        print(f"[benchmark] {method}", flush=True)
        if method == "full":
            function = lambda: robustness.load_fusion_predictions(
                "full",
                benchmark_seed,
                snapshots,
                global_address_count,
                test_times,
                device,
                fusion_root,
            )
        else:
            function = lambda method=method: robustness.load_baseline_predictions(
                method,
                benchmark_seed,
                snapshots,
                transaction_edges,
                train_times,
                validation_times,
                test_times,
                device,
            )
        inference[method] = timed_call(function, device, int(args.repeats))

    rows: List[Dict[str, Any]] = []
    details: Dict[str, Any] = {}
    for method in METHODS:
        runs = []
        for seed in SEEDS:
            if method == "full":
                result = read_full(seed, attribute_root, fusion_root)
                training_seconds = float(result["training_seconds"])
            else:
                result = read_baseline(seed, method)
                training_seconds = float(result["duration_seconds"])
            runs.append(
                {
                    "seed": seed,
                    "parameters": float(result["parameters"]),
                    "training_seconds": training_seconds,
                    "peak_gpu_memory_mb": float(result["peak_gpu_memory_mb"]),
                    "f1": float(result["test"]["f1"]),
                    "average_precision": float(
                        result["test"]["average_precision"]
                    ),
                }
            )
        metrics = {
            key: aggregate([run[key] for run in runs])
            for key in (
                "parameters",
                "training_seconds",
                "peak_gpu_memory_mb",
                "f1",
                "average_precision",
            )
        }
        latency = inference[method]["seconds"]
        throughput = test_count / latency["mean"]
        row = {
            "method": method,
            "display_name": DISPLAY_NAMES[method],
            "parameters_mean": metrics["parameters"]["mean"],
            "parameters_std": metrics["parameters"]["std"],
            "training_seconds_mean": metrics["training_seconds"]["mean"],
            "training_seconds_std": metrics["training_seconds"]["std"],
            "training_peak_gpu_mb_mean": metrics["peak_gpu_memory_mb"]["mean"],
            "training_peak_gpu_mb_std": metrics["peak_gpu_memory_mb"]["std"],
            "inference_ms_mean": 1000.0 * latency["mean"],
            "inference_ms_std": 1000.0 * latency["std"],
            "throughput_labeled_transactions_per_second": throughput,
            "inference_peak_gpu_mb_mean": inference[method][
                "peak_gpu_memory_mb"
            ]["mean"],
            "f1_mean": metrics["f1"]["mean"],
            "f1_std": metrics["f1"]["std"],
            "average_precision_mean": metrics["average_precision"]["mean"],
            "average_precision_std": metrics["average_precision"]["std"],
        }
        rows.append(row)
        details[method] = {"training_runs": runs, "inference": inference[method]}

    with (output_root / "efficiency_summary.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "protocol": {
            "training_seeds": SEEDS,
            "benchmark_seed": benchmark_seed,
            "inference_repeats": int(args.repeats),
            "test_labeled_transactions": test_count,
            "dataset_loading_included": False,
            "checkpoint_loading_included": True,
            "causal_history_replay_included_when_required": True,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else None,
        },
        "summary": rows,
        "details": details,
    }
    (output_root / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    plot_tradeoff(output_root, rows)
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
