from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = PROJECT_ROOT.parent / "artifacts" / "competition_baselines_5seed"
EXPECTED_SEEDS = [20260806, 20260807, 20260808, 20260809, 20260810]
EXPECTED_MODELS = [
    "mdst_gnn",
    "tfgat_dcplu",
    "ellipticpp_hgt",
    "fg_egcn",
    "gpn",
    "nsgcn_lstm",
]
METRICS = [
    "roc_auc",
    "pr_auc",
    "precision",
    "recall",
    "f1",
    "macro_f1",
    "accuracy",
    "brier",
    "top_capture",
    "top_precision",
    "worst_time_f1",
    "mean_time_f1",
    "std_time_f1",
    "near_f1",
    "near_pr_auc",
    "middle_f1",
    "middle_pr_auc",
    "far_f1",
    "far_pr_auc",
]
VALIDATION_METRICS = [
    "roc_auc",
    "pr_auc",
    "precision",
    "recall",
    "f1",
    "macro_f1",
    "accuracy",
    "brier",
    "top_capture",
    "top_precision",
]
RESOURCE_METRICS = ["duration_seconds", "parameters", "peak_gpu_memory_mb", "best_epoch"]


def main() -> None:
    rows = []
    implementation = {}
    for seed in EXPECTED_SEEDS:
        for model in EXPECTED_MODELS:
            path = OUTPUT_ROOT / f"seed_{seed}" / model / "metrics.json"
            if not path.is_file():
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
            implementation[model] = payload.get("implementation")
            row = {"seed": seed, "model": model}
            row.update({metric: payload["test"].get(metric) for metric in METRICS})
            row.update(
                {
                    f"validation_{metric}": payload["validation"].get(metric)
                    for metric in VALIDATION_METRICS
                }
            )
            row.update({metric: payload.get(metric) for metric in RESOURCE_METRICS})
            rows.append(row)
    frame = pd.DataFrame(rows)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUTPUT_ROOT / "all_seed_metrics.csv", index=False)
    summary_rows = []
    for model in EXPECTED_MODELS:
        selected = frame[frame["model"] == model]
        row = {
            "model": model,
            "completed_seeds": int(len(selected)),
            "expected_seeds": len(EXPECTED_SEEDS),
            "implementation": implementation.get(model),
        }
        for metric in METRICS:
            values = selected[metric].dropna().to_numpy(dtype=float)
            row[f"{metric}_mean"] = float(values.mean()) if values.size else None
            row[f"{metric}_std"] = (
                float(values.std(ddof=1)) if values.size > 1 else 0.0 if values.size else None
            )
        for metric in [f"validation_{name}" for name in VALIDATION_METRICS] + RESOURCE_METRICS:
            values = selected[metric].dropna().to_numpy(dtype=float)
            row[f"{metric}_mean"] = float(values.mean()) if values.size else None
            row[f"{metric}_std"] = (
                float(values.std(ddof=1)) if values.size > 1 else 0.0 if values.size else None
            )
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUTPUT_ROOT / "summary.csv", index=False)
    status = "PASS" if all(row["completed_seeds"] == 5 for row in summary_rows) else "INCOMPLETE"
    payload = {"status": status, "seeds": EXPECTED_SEEDS, "models": summary_rows}
    (OUTPUT_ROOT / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [
        "# 六个新增基线五随机种子汇总",
        "",
        f"状态：{status}",
        "",
        "| 模型 | 完成种子 | Test F1 | Test PR-AUC | Test ROC-AUC | 最差时段 F1 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        def show(metric: str) -> str:
            mean = row.get(f"{metric}_mean")
            std = row.get(f"{metric}_std")
            return "—" if mean is None else f"{mean:.4f}±{std:.4f}"

        lines.append(
            f"| {row['model']} | {row['completed_seeds']}/5 | {show('f1')} | "
            f"{show('pr_auc')} | {show('roc_auc')} | {show('worst_time_f1')} |"
        )
    (OUTPUT_ROOT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
