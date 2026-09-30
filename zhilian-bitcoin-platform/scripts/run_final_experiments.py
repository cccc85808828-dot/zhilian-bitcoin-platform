from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEEDS = [20260806, 20260807, 20260808, 20260809, 20260810]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the experiments retained in the final TTHGNN-QIF paper."
    )
    parser.add_argument(
        "--stage",
        choices=["check", "main", "summarize", "analysis", "all"],
        default="main",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def run(command: Iterable[object]) -> None:
    rendered = [str(item) for item in command]
    print(f"[run] {' '.join(rendered)}", flush=True)
    subprocess.run(rendered, cwd=PROJECT_ROOT, check=True)


def run_check() -> None:
    run([sys.executable, PROJECT_ROOT / "scripts" / "verify_ellipticpp.py"])
    run(
        [
            sys.executable,
            "-m",
            "compileall",
            "-q",
            PROJECT_ROOT / "src",
            PROJECT_ROOT / "scripts",
        ]
    )


def run_main(seeds: Sequence[int], force: bool) -> None:
    for seed in seeds:
        output_root = (
            PROJECT_ROOT
            / "artifacts"
            / "main_experiment"
            / "baselines"
            / f"seed_{seed}"
        )
        comparison = output_root / "comparison.json"
        if force or not comparison.is_file():
            run(
                [
                    sys.executable,
                    PROJECT_ROOT / "scripts" / "train_main_baselines.py",
                    "--config",
                    PROJECT_ROOT / "configs" / "main_baselines.yaml",
                    "--seed",
                    seed,
                    "--output-root",
                    output_root,
                ]
            )
        else:
            print(f"[skip] {comparison}", flush=True)

    command: list[object] = [
        sys.executable,
        PROJECT_ROOT / "scripts" / "run_ablation_experiments.py",
        "--seeds",
        *seeds,
    ]
    if force:
        command.append("--force")
    run(command)
    run(
        [
            sys.executable,
            PROJECT_ROOT / "scripts" / "run_ablation_experiments.py",
            "--seeds",
            *seeds,
            "--variants",
            "full",
            "--attribute-root",
            PROJECT_ROOT / "artifacts" / "quantile_no_bi",
            "--pooled-output-root",
            PROJECT_ROOT / "artifacts" / "pooled_f1_no_bi" / "ablation",
            "--disable-bi-interaction",
        ]
    )
    run_summaries()


def run_summaries() -> None:
    run([sys.executable, PROJECT_ROOT / "scripts" / "summarize_main_experiment.py"])
    run(
        [
            sys.executable,
            PROJECT_ROOT / "scripts" / "summarize_ablation_experiments.py",
        ]
    )
    run(
        [
            sys.executable,
            PROJECT_ROOT / "scripts" / "summarize_bi_interaction_ablation.py",
        ]
    )


def run_analysis(seeds: Sequence[int], force: bool) -> None:
    run(
        [
            sys.executable,
            PROJECT_ROOT / "scripts" / "evaluate_temporal_robustness.py",
            "--seeds",
            *seeds,
        ]
    )
    run([sys.executable, PROJECT_ROOT / "scripts" / "analyze_temporal_distribution_drift.py"])

    sensitivity: list[object] = [
        sys.executable,
        PROJECT_ROOT / "scripts" / "run_hyperparameter_sensitivity.py",
        "--seeds",
        *seeds,
    ]
    if force:
        sensitivity.append("--force")
    run(sensitivity)
    run(
        [
            sys.executable,
            PROJECT_ROOT / "scripts" / "summarize_hyperparameter_sensitivity.py",
            "--seeds",
            *seeds,
        ]
    )
    run([sys.executable, PROJECT_ROOT / "scripts" / "analyze_computational_efficiency.py"])
    run(
        [
            sys.executable,
            PROJECT_ROOT / "scripts" / "plot_main_pr_curves.py",
            "--seeds",
            *seeds,
        ]
    )


def main() -> None:
    args = parse_args()
    seeds = [int(seed) for seed in args.seeds]
    if args.stage in {"check", "all"}:
        run_check()
    if args.stage in {"main", "all"}:
        run_main(seeds, bool(args.force))
    elif args.stage == "summarize":
        run_summaries()
    if args.stage in {"analysis", "all"}:
        run_analysis(seeds, bool(args.force))


if __name__ == "__main__":
    main()
