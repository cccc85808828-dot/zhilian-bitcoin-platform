from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from scipy.stats import ttest_rel, wilcoxon


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEEDS = [20260806, 20260807, 20260808, 20260809, 20260810]
METRICS = ["precision", "accuracy", "recall", "f1", "average_precision"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize the explicit NFM bi-interaction ablation."
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument(
        "--full-root",
        type=Path,
        default=Path("artifacts/pooled_f1/ablation/full"),
    )
    parser.add_argument(
        "--no-interaction-root",
        type=Path,
        default=Path("artifacts/pooled_f1_no_bi/ablation/full"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/bi_interaction_ablation"),
    )
    return parser.parse_args()


def project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_runs(root: Path, seeds: List[int]) -> Dict[str, List[float]]:
    values = {metric: [] for metric in METRICS}
    for seed in seeds:
        payload = json.loads(
            (root / f"seed_{seed}" / "metrics.json").read_text(encoding="utf-8")
        )
        test = payload["fusion"]["test"]
        for metric in METRICS:
            values[metric].append(float(test[metric]))
    return values


def aggregate(values: List[float]) -> Dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if array.size > 1 else 0.0,
    }


def main() -> None:
    args = parse_args()
    seeds = [int(seed) for seed in args.seeds]
    full = load_runs(project_path(args.full_root), seeds)
    no_interaction = load_runs(project_path(args.no_interaction_root), seeds)
    methods = {
        "Full QIF": {metric: aggregate(full[metric]) for metric in METRICS},
        "w/o Explicit Bi-interaction": {
            metric: aggregate(no_interaction[metric]) for metric in METRICS
        },
    }
    paired: Dict[str, Any] = {}
    for metric in METRICS:
        full_values = np.asarray(full[metric], dtype=np.float64)
        ablated_values = np.asarray(no_interaction[metric], dtype=np.float64)
        deltas = full_values - ablated_values
        paired[metric] = {
            **aggregate(deltas.tolist()),
            "full_wins": int((deltas > 0.0).sum()),
            "runs": int(deltas.size),
            "paired_t_pvalue": float(ttest_rel(full_values, ablated_values).pvalue),
            "wilcoxon_pvalue": float(wilcoxon(deltas).pvalue),
        }
    summary = {
        "status": "PASS",
        "protocol": {
            "train": [1, 30],
            "validation": [31, 35],
            "test": [36, 49],
            "threshold_selection": "pooled_validation_f1",
            "seeds": seeds,
        },
        "methods": methods,
        "full_minus_no_interaction": paired,
    }
    output_root = project_path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (output_root / "summary.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["method"]
            + [f"{metric}_{stat}" for metric in METRICS for stat in ("mean", "std")],
        )
        writer.writeheader()
        for method, metrics in methods.items():
            row: Dict[str, Any] = {"method": method}
            for metric in METRICS:
                row[f"{metric}_mean"] = metrics[metric]["mean"]
                row[f"{metric}_std"] = metrics[metric]["std"]
            writer.writerow(row)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
