from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEEDS = [20260806, 20260807, 20260808, 20260809, 20260810]
VARIANTS = [
    "full",
    "no_rcha",
    "no_erl",
    "no_robust",
    "no_memory",
    "no_role",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the final TTHGNN-QIF ablation protocol."
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=VARIANTS)
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/ablation"))
    parser.add_argument(
        "--pooled-output-root",
        type=Path,
        default=Path("artifacts/pooled_f1/ablation"),
        help="Final fusion metrics selected by pooled validation F1.",
    )
    parser.add_argument(
        "--attribute-root",
        type=Path,
        default=Path("artifacts/quantile_fusion"),
        help="One QIF checkpoint per seed, shared by all structural ablations.",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-fusion", action="store_true")
    parser.add_argument("--disable-bi-interaction", action="store_true")
    parser.add_argument(
        "--interaction-mode",
        choices=["concat", "residual"],
        default="residual",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Use two graph and attribute epochs and write under tmp/ablation_smoke.",
    )
    return parser.parse_args()


def project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def run_logged(command: Iterable[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    rendered = [str(value) for value in command]
    print(f"[run] {' '.join(rendered)}", flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            rendered,
            cwd=PROJECT_ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Command failed with code {completed.returncode}; see {log_path}"
        )


def main() -> None:
    args = parse_args()
    output_root = project_path(
        Path("tmp/ablation_smoke") if args.smoke else args.output_root
    )
    pooled_output_root = project_path(
        Path("tmp/ablation_smoke_pooled")
        if args.smoke
        else args.pooled_output_root
    )
    attribute_root = project_path(
        Path("tmp/ablation_smoke_attribute")
        if args.smoke
        else args.attribute_root
    )
    if "full" not in args.variants:
        missing = [
            seed
            for seed in args.seeds
            if not (attribute_root / f"seed_{seed}" / "best.pt").is_file()
        ]
        if missing:
            raise ValueError(
                "The full variant must be included when shared QIF checkpoints "
                f"are missing for seeds: {missing}"
            )

    for variant in args.variants:
        config_path = PROJECT_ROOT / "configs" / "ablations" / f"{variant}.yaml"
        for seed in args.seeds:
            graph_root = output_root / variant / "graph" / f"seed_{seed}"
            graph_metrics = graph_root / "metrics.json"
            if args.force or not graph_metrics.exists():
                command = [
                    sys.executable,
                    str(PROJECT_ROOT / "scripts" / "train_temporal_memory.py"),
                    "--config",
                    str(config_path),
                    "--seed",
                    str(seed),
                    "--output-root",
                    str(graph_root),
                    "--baseline-comparison",
                    str(
                        PROJECT_ROOT
                        / "artifacts"
                        / "main_experiment"
                        / "baselines"
                        / f"seed_{seed}"
                        / "comparison.json"
                    ),
                ]
                if args.smoke:
                    command.extend(["--max-epochs", "2"])
                run_logged(command, graph_root / "run.log")
            else:
                print(f"[skip] {graph_metrics}", flush=True)

            if args.skip_fusion:
                continue

            attribute_checkpoint = attribute_root / f"seed_{seed}" / "best.pt"
            if variant == "full" and (args.force or not attribute_checkpoint.exists()):
                attribute_seed_root = attribute_root / f"seed_{seed}"
                command = [
                    sys.executable,
                    str(PROJECT_ROOT / "scripts" / "train_quantile_fusion.py"),
                    "--seed",
                    str(seed),
                    "--graph-checkpoint",
                    str(graph_root / "best.pt"),
                    "--output-root",
                    str(attribute_seed_root),
                    "--selection-objective",
                    "pooled_validation_f1",
                    "--interaction-mode",
                    str(args.interaction_mode),
                ]
                if args.disable_bi_interaction:
                    command.append("--disable-bi-interaction")
                if args.smoke:
                    command.extend(["--max-epochs", "2"])
                run_logged(command, attribute_seed_root / "run.log")

            fusion_root = pooled_output_root / variant / f"seed_{seed}"
            fusion_metrics = fusion_root / "metrics.json"
            if args.force or not fusion_metrics.exists():
                command = [
                    sys.executable,
                    str(PROJECT_ROOT / "scripts" / "train_quantile_fusion.py"),
                    "--seed",
                    str(seed),
                    "--graph-checkpoint",
                    str(graph_root / "best.pt"),
                    "--attribute-checkpoint",
                    str(attribute_checkpoint),
                    "--output-root",
                    str(fusion_root),
                    "--selection-objective",
                    "pooled_validation_f1",
                ]
                run_logged(command, fusion_root / "run.log")
            else:
                print(f"[skip] {fusion_metrics}", flush=True)

    print(
        "Ablation runs completed under "
        f"{output_root}; final pooled metrics: {pooled_output_root}",
        flush=True,
    )


if __name__ == "__main__":
    main()
