from __future__ import annotations

import argparse
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEEDS = [20260806, 20260808, 20260810]
SPECS: Dict[str, Dict[str, Any]] = {
    "erl_penalty": {
        "values": [0.0, 5.0, 10.0, 15.0, 20.0],
        "default": 10.0,
    },
    "rcha_weight": {
        "values": [0.0, 0.1, 0.2, 0.3, 0.4],
        "default": 0.2,
    },
    "representation_dim": {
        "values": [32, 48, 64, 96, 128],
        "default": 64,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run three-parameter sensitivity experiments."
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/hyperparameter_sensitivity"),
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--attribute-root",
        type=Path,
        default=Path("artifacts/quantile_fusion"),
    )
    parser.add_argument(
        "--fusion-output-root",
        type=Path,
        default=None,
        help="Optional separate root for new fusion metrics while reusing graph runs.",
    )
    return parser.parse_args()


def project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def value_slug(value: float | int) -> str:
    if isinstance(value, int):
        return str(value)
    return f"{value:g}".replace(".", "p")


def reused_variant(parameter: str, value: float | int) -> str | None:
    if value == SPECS[parameter]["default"]:
        return "full"
    if parameter == "erl_penalty" and float(value) == 0.0:
        return "no_erl"
    if parameter == "rcha_weight" and float(value) == 0.0:
        return "no_rcha"
    return None


def apply_value(
    base_config: Dict[str, Any], parameter: str, value: float | int
) -> Dict[str, Any]:
    config = deepcopy(base_config)
    if parameter == "erl_penalty":
        config["robust_learning"]["enabled"] = float(value) > 0.0
        config["robust_learning"]["penalty_weight"] = float(value)
    elif parameter == "rcha_weight":
        config["augmentation"]["enabled"] = float(value) > 0.0
        config["augmentation"]["loss_weight"] = float(value)
    elif parameter == "representation_dim":
        config["model"]["hidden_dim"] = int(value)
        config["model"]["memory_dim"] = int(value)
    else:
        raise KeyError(parameter)
    return config


def run_logged(command: Iterable[str], log_path: Path) -> None:
    rendered = [str(item) for item in command]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as stream:
        completed = subprocess.run(
            rendered,
            cwd=PROJECT_ROOT,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Command failed with code {completed.returncode}; see {log_path}"
        )


def result_paths(
    output_root: Path,
    fusion_output_root: Path,
    parameter: str,
    value: float | int,
    seed: int,
) -> Tuple[Path, Path]:
    graph_root = output_root / parameter / f"value_{value_slug(value)}"
    fusion_root = fusion_output_root / parameter / f"value_{value_slug(value)}"
    return (
        graph_root / "graph" / f"seed_{seed}",
        fusion_root / "fusion" / f"seed_{seed}",
    )


def main() -> None:
    args = parse_args()
    output_root = project_path(args.output_root)
    fusion_output_root = project_path(
        args.output_root if args.fusion_output_root is None else args.fusion_output_root
    )
    attribute_root = project_path(args.attribute_root)
    output_root.mkdir(parents=True, exist_ok=True)
    fusion_output_root.mkdir(parents=True, exist_ok=True)
    with (PROJECT_ROOT / "configs" / "ablations" / "full.yaml").open(
        "r", encoding="utf-8"
    ) as stream:
        base_config = yaml.safe_load(stream)

    total_new = sum(
        reused_variant(parameter, value) is None
        for parameter, spec in SPECS.items()
        for value in spec["values"]
        for _ in args.seeds
    )
    completed_new = 0
    for parameter, spec in SPECS.items():
        for value in spec["values"]:
            reused = reused_variant(parameter, value)
            if reused is not None:
                print(
                    f"[reuse] {parameter}={value:g} from {reused}", flush=True
                )
                continue
            graph_config = apply_value(base_config, parameter, value)
            config_path = (
                output_root
                / "configs"
                / parameter
                / f"value_{value_slug(value)}.yaml"
            )
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config_path.write_text(
                yaml.safe_dump(graph_config, allow_unicode=True, sort_keys=False),
                encoding="utf-8",
            )
            for seed in args.seeds:
                graph_root, fusion_root = result_paths(
                    output_root, fusion_output_root, parameter, value, seed
                )
                graph_metrics = graph_root / "metrics.json"
                if args.force or not graph_metrics.exists():
                    print(
                        f"[{completed_new + 1}/{total_new}] train graph "
                        f"{parameter}={value:g}, seed={seed}",
                        flush=True,
                    )
                    run_logged(
                        [
                            sys.executable,
                            PROJECT_ROOT / "scripts" / "train_temporal_memory.py",
                            "--config",
                            config_path,
                            "--seed",
                            seed,
                            "--output-root",
                            graph_root,
                            "--baseline-comparison",
                            PROJECT_ROOT
                            / "artifacts"
                            / "main_experiment"
                            / "baselines"
                            / f"seed_{seed}"
                            / "comparison.json",
                        ],
                        graph_root / "run.log",
                    )
                else:
                    print(f"[skip] {graph_metrics}", flush=True)

                fusion_metrics = fusion_root / "metrics.json"
                if args.force or not fusion_metrics.exists():
                    print(
                        f"[{completed_new + 1}/{total_new}] select fusion "
                        f"{parameter}={value:g}, seed={seed}",
                        flush=True,
                    )
                    run_logged(
                        [
                            sys.executable,
                            PROJECT_ROOT / "scripts" / "train_quantile_fusion.py",
                            "--seed",
                            seed,
                            "--graph-checkpoint",
                            graph_root / "best.pt",
                            "--attribute-checkpoint",
                            attribute_root
                            / f"seed_{seed}"
                            / "best.pt",
                            "--output-root",
                            fusion_root,
                            "--selection-objective",
                            "pooled_validation_f1",
                        ],
                        fusion_root / "run.log",
                    )
                else:
                    print(f"[skip] {fusion_metrics}", flush=True)
                completed_new += 1
                print(
                    f"[done] {parameter}={value:g}, seed={seed}", flush=True
                )
    print(f"Sensitivity runs completed under {output_root}", flush=True)


if __name__ == "__main__":
    main()
