from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEEDS = [20260806, 20260807, 20260808, 20260809, 20260810]
METRICS = ["precision", "accuracy", "recall", "f1"]
ROWS: List[Tuple[str, str, str]] = [
    ("M0", "Full", "full:fusion"),
    ("M1", "w/o QIF", "full:graph"),
    ("M2", "w/o RCHA", "no_rcha:fusion"),
    ("M3", "w/o ERL", "no_erl:fusion"),
    ("M4", "w/o Robust Module (RCHA+ERL)", "no_robust:fusion"),
    ("M5", "w/o Memory", "no_memory:fusion"),
    ("M6", "w/o Role", "no_role:fusion"),
]
STRUCTURAL_ROWS: List[Tuple[str, str, str]] = [
    ("S0", "Structural backbone", "full:graph"),
    ("S1", "w/o RCHA", "no_rcha:graph"),
    ("S2", "w/o ERL", "no_erl:graph"),
    ("S3", "w/o Robust Module (RCHA+ERL)", "no_robust:graph"),
    ("S4", "w/o Memory", "no_memory:graph"),
    ("S5", "w/o Role", "no_role:graph"),
]
QIF_ROWS: List[Tuple[str, str, str]] = [
    ("Q0", "Structural backbone", "full:graph"),
    ("Q1", "Full + QIF", "full:fusion"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize six unified ablations.")
    parser.add_argument(
        "--root", type=Path, default=Path("artifacts/pooled_f1/ablation")
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument(
        "--selection-objective",
        choices=["macro_snapshot_f1", "pooled_validation_f1"],
        default="pooled_validation_f1",
    )
    return parser.parse_args()


def project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_test_metrics(
    root: Path,
    source: str,
    seed: int,
    selection_objective: str,
) -> Dict[str, Any]:
    variant, branch = source.split(":", maxsplit=1)
    if selection_objective == "pooled_validation_f1":
        direct_path = root / variant / f"seed_{seed}" / "metrics.json"
        nested_path = root / variant / "fusion" / f"seed_{seed}" / "metrics.json"
        path = direct_path if direct_path.is_file() else nested_path
    else:
        path = root / variant / branch / f"seed_{seed}" / "metrics.json"
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if branch == "fusion":
        values = payload["fusion"]["test"]
        selection = payload["protocol"].get("selection_objective")
        if selection != selection_objective:
            raise ValueError(f"Unexpected threshold rule in {path}: {selection}")
    elif selection_objective == "pooled_validation_f1":
        values = payload["graph_model"]["test"]
        selection = payload["protocol"].get("selection_objective")
        if selection != selection_objective:
            raise ValueError(f"Unexpected graph threshold rule in {path}")
    else:
        values = payload["test"]
        validation = payload.get("validation", {})
        if validation.get("threshold_method") != "mean_validation_snapshot_f1":
            raise ValueError(f"Unexpected graph threshold rule in {path}")
    return {**values, "path": str(path)}


def aggregate(values: List[float]) -> Dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if array.size > 1 else 0.0,
    }


def summarize_rows(
    root: Path,
    rows: List[Tuple[str, str, str]],
    seeds: List[int],
    selection_objective: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[int, Dict[str, Any]]]]:
    runs: Dict[str, Dict[int, Dict[str, Any]]] = {}
    methods: List[Dict[str, Any]] = []
    for row_id, label, source in rows:
        seed_runs = {
            int(seed): load_test_metrics(
                root, source, int(seed), selection_objective
            )
            for seed in seeds
        }
        runs[row_id] = seed_runs
        counts = {
            (values.get("positives"), values.get("negatives"))
            for values in seed_runs.values()
        }
        if len(counts) != 1:
            raise ValueError(f"Test population changed across seeds for {row_id}")
        population = next(iter(counts))
        methods.append(
            {
                "id": row_id,
                "method": label,
                "source": source,
                "num_runs": len(seed_runs),
                "test_population": {
                    "positives": population[0],
                    "negatives": population[1],
                },
                "metrics": {
                    metric: aggregate(
                        [values[metric] for values in seed_runs.values()]
                    )
                    for metric in METRICS
                },
                "runs": [
                    {
                        "seed": seed,
                        **{metric: values[metric] for metric in METRICS},
                    }
                    for seed, values in seed_runs.items()
                ],
            }
        )
    return methods, runs


def write_csv(path: Path, methods: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        fieldnames = ["id", "method"] + [
            f"{metric}_{stat}" for metric in METRICS for stat in ("mean", "std")
        ]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for method in methods:
            row: Dict[str, Any] = {"id": method["id"], "method": method["method"]}
            for metric in METRICS:
                for stat in ("mean", "std"):
                    row[f"{metric}_{stat}"] = method["metrics"][metric][stat]
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    root = project_path(args.root)
    seeds = [int(seed) for seed in args.seeds]
    methods, runs = summarize_rows(root, ROWS, seeds, args.selection_objective)
    structural_methods, structural_runs = summarize_rows(
        root, STRUCTURAL_ROWS, seeds, args.selection_objective
    )
    qif_methods, _ = summarize_rows(
        root, QIF_ROWS, seeds, args.selection_objective
    )

    paired_deltas: List[Dict[str, Any]] = []
    full = runs["M0"]
    for row_id, label, _ in ROWS[1:]:
        delta_by_metric: Dict[str, Any] = {}
        for metric in METRICS:
            deltas = [
                full[int(seed)][metric] - runs[row_id][int(seed)][metric]
                for seed in args.seeds
            ]
            delta_by_metric[metric] = {
                **aggregate(deltas),
                "full_wins": int(sum(delta > 0.0 for delta in deltas)),
            }
        paired_deltas.append(
            {"comparison": f"Full - {label}", "metrics": delta_by_metric}
        )

    summary = {
        "status": "PASS",
        "protocol": {
            "train": [1, 30],
            "validation": [31, 35],
            "test": [36, 49],
            "threshold_selection": args.selection_objective,
            "seeds": [int(seed) for seed in args.seeds],
        },
        "methods": methods,
        "paired_deltas": paired_deltas,
        "structural_ablation": structural_methods,
        "qif_ablation": qif_methods,
    }
    root.mkdir(parents=True, exist_ok=True)
    with (root / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)

    write_csv(root / "summary.csv", methods)
    write_csv(root / "structural_ablation.csv", structural_methods)
    write_csv(root / "qif_ablation.csv", qif_methods)

    for method in methods:
        formatted = " ".join(
            f"{metric}={method['metrics'][metric]['mean']:.4f}±"
            f"{method['metrics'][metric]['std']:.4f}"
            for metric in METRICS
        )
        print(f"{method['id']} {method['method']}: {formatted}")


if __name__ == "__main__":
    main()
