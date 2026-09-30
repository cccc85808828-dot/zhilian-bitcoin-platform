from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from competition_baselines.baseline_data import load_graph_data  # noqa: E402
from competition_baselines.baseline_runner import (  # noqa: E402
    MODEL_NAMES,
    run_model,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the six leakage-safe Elliptic++ competition baselines"
    )
    parser.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "config.yaml"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=("all",) + MODEL_NAMES,
        default=["all"],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    config["project_root"] = str(PROJECT_ROOT)
    requested_models = (
        list(config["models"]) if "all" in args.models else list(args.models)
    )
    seeds = list(config["seeds"]) if args.seeds is None else list(args.seeds)
    max_epochs = 1 if args.smoke else args.max_epochs
    if args.smoke and args.seeds is None:
        seeds = seeds[:1]
    device_name = str(args.device or config["device"])
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    output_root = Path(config["output_root"])
    if not output_root.is_absolute():
        output_root = PROJECT_ROOT / output_root
    output_root.mkdir(parents=True, exist_ok=True)
    protocol = {
        "status": "RUNNING",
        "models": requested_models,
        "seeds": seeds,
        "split": config["split"],
        "device": device_name,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "unknown_labels_in_supervised_loss_or_metrics": False,
        "validation_used_for_threshold_and_early_stopping": True,
        "test_used_for_selection": False,
        "gpn_training_labels": "labeled normal nodes only",
    }
    write_json(output_root / "run_manifest.json", protocol)
    data = load_graph_data(config)
    results = []
    for seed in seeds:
        for name in requested_models:
            print(f"\n=== {name} | seed {seed} ===", flush=True)
            result = run_model(
                name=name,
                config=config,
                data=data,
                output_root=output_root,
                device=device,
                seed=int(seed),
                max_epochs_override=max_epochs,
                force=args.force,
            )
            results.append(
                {
                    "model": name,
                    "seed": int(seed),
                    "f1": result["test"]["f1"],
                    "pr_auc": result["test"]["pr_auc"],
                }
            )
            print(json.dumps(results[-1], ensure_ascii=False), flush=True)
    protocol["status"] = "PASS"
    protocol["completed_runs"] = len(results)
    protocol["results"] = results
    write_json(output_root / "run_manifest.json", protocol)


if __name__ == "__main__":
    main()
