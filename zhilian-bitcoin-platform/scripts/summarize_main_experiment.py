from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_ORDER = [
    "gcn",
    "gat",
    "hgnn",
    "evolvegcn_o",
    "hgt",
    "graph_transformer",
    "gat_resnet",
    "heterosage",
    "bf_hgn",
    "tthgnn_qif",
]
METRICS = ["precision", "accuracy", "recall", "f1"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize the main experiment")
    parser.add_argument(
        "--root",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "main_experiment",
    )
    parser.add_argument(
        "--pooled-root",
        type=Path,
        default=Path("artifacts/pooled_f1/ablation/full"),
    )
    parser.add_argument("--output-root", type=Path, default=None)
    return parser.parse_args()


def inferred_seed(path: Path) -> int:
    name = path.parent.name
    if not name.startswith("seed_"):
        raise ValueError(f"Cannot infer a seed from {path}")
    return int(name.removeprefix("seed_"))


def metric_value(metrics: Dict[str, Any], metric: str) -> float:
    if metric in metrics:
        return float(metrics[metric])
    if metric != "accuracy":
        raise KeyError(metric)

    positives = int(metrics["positives"])
    negatives = int(metrics["negatives"])
    true_positives = int(round(float(metrics["recall"]) * positives))
    precision = float(metrics["precision"])
    if precision <= 0.0:
        if true_positives != 0:
            raise ValueError("Invalid precision and recall combination.")
        raise ValueError("Accuracy cannot be reconstructed when precision and recall are zero.")
    predicted_positives = int(round(true_positives / precision))
    false_positives = predicted_positives - true_positives
    true_negatives = negatives - false_positives
    return float((true_positives + true_negatives) / (positives + negatives))


def main() -> None:
    args = parse_args()
    root = args.root if args.root.is_absolute() else PROJECT_ROOT / args.root
    pooled_root = (
        args.pooled_root
        if args.pooled_root.is_absolute()
        else PROJECT_ROOT / args.pooled_root
    )
    output_root = (
        root
        if args.output_root is None
        else (
            args.output_root
            if args.output_root.is_absolute()
            else PROJECT_ROOT / args.output_root
        )
    )
    runs: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for path in sorted((root / "baselines").glob("seed_*/comparison.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        seed = int(payload.get("seed", inferred_seed(path)))
        for result in payload["results"]:
            result = dict(result)
            result["seed"] = seed
            runs[result["model"]].append(result)

    for path in sorted(pooled_root.glob("seed_*/metrics.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        runs["tthgnn_qif"].append(
            {
                "model": "tthgnn_qif",
                "seed": int(result.get("seed", inferred_seed(path))),
                "validation": result["fusion"]["validation"],
                "test": result["fusion"]["test"],
            }
        )

    summaries: List[Dict[str, Any]] = []
    for model in MODEL_ORDER:
        model_runs = sorted(runs.get(model, []), key=lambda item: item["seed"])
        if not model_runs:
            continue
        summary: Dict[str, Any] = {
            "model": model,
            "num_runs": len(model_runs),
            "seeds": [int(item["seed"]) for item in model_runs],
        }
        for split in ("validation", "test"):
            split_summary: Dict[str, Dict[str, float]] = {}
            for metric in METRICS:
                values = np.asarray(
                    [metric_value(item[split], metric) for item in model_runs],
                    dtype=np.float64,
                )
                split_summary[metric] = {
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
                }
            summary[split] = split_summary
        summaries.append(summary)

    payload = {"status": "PASS", "models": summaries}
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (output_root / "summary.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        fieldnames = ["model", "num_runs"]
        for split in ("validation", "test"):
            for metric in METRICS:
                fieldnames.extend([f"{split}_{metric}_mean", f"{split}_{metric}_std"])
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for summary in summaries:
            row: Dict[str, Any] = {
                "model": summary["model"],
                "num_runs": summary["num_runs"],
            }
            for split in ("validation", "test"):
                for metric in METRICS:
                    row[f"{split}_{metric}_mean"] = summary[split][metric]["mean"]
                    row[f"{split}_{metric}_std"] = summary[split][metric]["std"]
            writer.writerow(row)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
