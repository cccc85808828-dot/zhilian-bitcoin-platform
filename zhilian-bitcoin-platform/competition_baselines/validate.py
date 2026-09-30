from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = PROJECT_ROOT.parent / "artifacts" / "competition_baselines_5seed"
SEEDS = [20260806, 20260807, 20260808, 20260809, 20260810]
MODELS = [
    "mdst_gnn",
    "tfgat_dcplu",
    "ellipticpp_hgt",
    "fg_egcn",
    "gpn",
    "nsgcn_lstm",
]


def require(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def main() -> None:
    errors: list[str] = []
    checked_runs = 0
    for seed in SEEDS:
        for model in MODELS:
            run_dir = OUTPUT_ROOT / f"seed_{seed}" / model
            metrics_path = run_dir / "metrics.json"
            checkpoint = run_dir / "best.pt"
            predictions_path = run_dir / "test_predictions.csv"
            per_time_path = run_dir / "per_time_metrics.csv"
            require(metrics_path.is_file(), f"missing {metrics_path}", errors)
            require(checkpoint.is_file(), f"missing {checkpoint}", errors)
            require(predictions_path.is_file(), f"missing {predictions_path}", errors)
            require(per_time_path.is_file(), f"missing {per_time_path}", errors)
            require(
                (run_dir / "training_history.csv").is_file(),
                f"missing {run_dir / 'training_history.csv'}",
                errors,
            )
            if not all(path.is_file() for path in [metrics_path, predictions_path, per_time_path]):
                continue

            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            predictions = pd.read_csv(predictions_path)
            per_time = pd.read_csv(per_time_path)
            require(metrics.get("model") == model, f"model mismatch: {metrics_path}", errors)
            require(metrics.get("seed") == seed, f"seed mismatch: {metrics_path}", errors)
            require(
                metrics["test"].get("count") == len(predictions),
                f"prediction rows: {predictions_path}",
                errors,
            )
            require(
                set(predictions["time"].unique()) == set(range(36, 50)),
                f"prediction times: {predictions_path}",
                errors,
            )
            require(len(per_time) == 14, f"per-time rows: {per_time_path}", errors)
            require(
                set(per_time["time"].tolist()) == set(range(36, 50)),
                f"per-time ids: {per_time_path}",
                errors,
            )
            require(
                predictions[["score", "prediction"]].notna().all().all(),
                f"NaN predictions: {predictions_path}",
                errors,
            )
            checked_runs += 1

    manifest_path = OUTPUT_ROOT / "run_manifest.json"
    summary_path = OUTPUT_ROOT / "summary.json"
    require(manifest_path.is_file(), "missing run_manifest.json", errors)
    require(summary_path.is_file(), "missing summary.json", errors)
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        require(manifest.get("status") == "PASS", "run manifest not PASS", errors)
        require(manifest.get("completed_runs") == 30, "run manifest not 30/30", errors)
        require(not manifest.get("test_used_for_selection"), "test used for selection", errors)
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        require(summary.get("status") == "PASS", "summary not PASS", errors)

    report = {
        "status": "PASS" if not errors and checked_runs == 30 else "FAIL",
        "checked_runs": checked_runs,
        "expected_runs": 30,
        "expected_test_times": [36, 49],
        "errors": errors,
    }
    (OUTPUT_ROOT / "validation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
