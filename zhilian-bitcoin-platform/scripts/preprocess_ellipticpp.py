from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tthgnn_erl.ellipticpp import preprocess_ellipticpp  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build role-aware temporal hypergraphs from Elliptic++"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "ellipticpp.yaml",
    )
    return parser.parse_args()


def resolve_project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> None:
    args = parse_args()
    with args.config.open("r", encoding="utf-8") as stream:
        config: Dict[str, Any] = yaml.safe_load(stream)
    data_config = config["data"]
    metadata = preprocess_ellipticpp(
        raw_root=resolve_project_path(data_config["raw_root"]),
        output_root=resolve_project_path(data_config["processed_root"]),
        first_time_step=int(data_config["first_time_step"]),
        last_time_step=int(data_config["last_time_step"]),
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "processed_root": str(
                    resolve_project_path(data_config["processed_root"])
                ),
                "totals": metadata["totals"],
                "feature_dimensions": metadata["feature_dimensions"],
                "snapshots": len(metadata["snapshots"]),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

