from __future__ import annotations

import os
import zipfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = PROJECT_ROOT / "TTHGNN-QIF_网站部署包.zip"


def deployment_files() -> list[Path]:
    explicit = [
        Path("scripts/run_competition_demo.py"),
        Path("scripts/bitcoin_live.py"),
        Path("Dockerfile"),
        Path("docker-compose.yml"),
        Path(".dockerignore"),
        Path(".env.example"),
        Path("requirements-web.txt"),
        Path("README.md"),
        Path("部署说明.md"),
        Path("artifacts/pooled_f1/ablation/full/seed_20260810/best.pt"),
        Path("artifacts/ablation/full/graph/seed_20260810/best.pt"),
        Path("artifacts/live_observable/seed_20260810/calibration.json"),
        Path("artifacts/live_observable/seed_20260810/best.pt"),
    ]
    generated = [
        path.relative_to(PROJECT_ROOT)
        for root in (PROJECT_ROOT / "demo", PROJECT_ROOT / "src" / "tthgnn_erl")
        for path in root.rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and path.suffix.lower() not in {".png", ".pyc"}
    ]
    selected = list(dict.fromkeys([*explicit, *generated]))
    missing = [str(path) for path in selected if not (PROJECT_ROOT / path).is_file()]
    if missing:
        raise FileNotFoundError(f"Deployment files are missing: {missing}")
    return selected


def main() -> None:
    temporary = OUTPUT.with_suffix(OUTPUT.suffix + ".building")
    if temporary.exists():
        temporary.unlink()
    files = deployment_files()
    with zipfile.ZipFile(
        temporary,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        for relative in files:
            archive.write(PROJECT_ROOT / relative, relative.as_posix())
    os.replace(temporary, OUTPUT)
    print(f"Created {OUTPUT} with {len(files)} files ({OUTPUT.stat().st_size} bytes).")


if __name__ == "__main__":
    main()
