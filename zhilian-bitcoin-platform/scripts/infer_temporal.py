from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Dict

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from evaluate_temporal_stability import load_model  # noqa: E402
from tthgnn_erl.ellipticpp import EllipticPPSnapshotDataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run frozen-model chronological inductive inference"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--data-root", type=Path, default=Path("data/processed/ellipticpp")
    )
    parser.add_argument("--start", type=int, default=35)
    parser.add_argument("--end", type=int, default=49)
    parser.add_argument(
        "--output-root", type=Path, default=Path("artifacts/temporal_inference/final")
    )
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def parameter_digest(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, parameter in model.state_dict().items():
        digest.update(name.encode("utf-8"))
        digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    if args.start < 1 or args.end < args.start:
        raise ValueError("The inference interval is invalid.")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA inference was requested but CUDA is unavailable.")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    data_root = project_path(args.data_root)
    checkpoint_path = project_path(args.checkpoint)
    output_root = project_path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    dataset = EllipticPPSnapshotDataset(data_root, validate=True)
    snapshots = {snapshot.time_id: snapshot for snapshot in dataset}
    if args.end not in snapshots:
        raise ValueError("The requested inference end time is unavailable.")
    model, checkpoint = load_model(checkpoint_path, device)
    threshold = float(checkpoint["validation_threshold"])
    global_address_count = int(dataset.metadata["totals"]["global_addresses"])
    memory_bank = model.initial_memory(global_address_count, device)
    digest_before = parameter_digest(model)
    summary_by_time: Dict[str, Dict[str, float]] = {}
    total_transactions = 0
    total_alerts = 0
    started = time.perf_counter()

    prediction_path = output_root / "predictions.csv"
    with prediction_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["time_id", "transaction_id", "risk_probability", "predicted_illicit"]
        )
        with torch.no_grad():
            for time_id in range(1, args.end + 1):
                snapshot = snapshots[time_id].to(device)
                global_ids = snapshot.global_address_ids
                previous_memory = memory_bank.index_select(0, global_ids)
                logits, updated_memory = model(snapshot, previous_memory)
                memory_bank.index_copy_(0, global_ids, updated_memory)
                if time_id < args.start:
                    continue

                probabilities = torch.sigmoid(logits).cpu()
                predictions = probabilities >= threshold
                transaction_ids = snapshot.transaction_ids.cpu()
                writer.writerows(
                    (
                        time_id,
                        int(transaction_id),
                        float(probability),
                        int(prediction),
                    )
                    for transaction_id, probability, prediction in zip(
                        transaction_ids, probabilities, predictions
                    )
                )
                transaction_count = snapshot.num_transactions
                alert_count = int(predictions.sum())
                total_transactions += transaction_count
                total_alerts += alert_count
                summary_by_time[str(time_id)] = {
                    "transactions": transaction_count,
                    "predicted_illicit": alert_count,
                    "alert_rate": alert_count / max(transaction_count, 1),
                    "active_addresses": snapshot.num_addresses,
                }

    digest_after = parameter_digest(model)
    if digest_before != digest_after:
        raise RuntimeError("Model parameters changed during inductive inference.")
    summary = {
        "status": "PASS",
        "model": checkpoint["model_name"],
        "checkpoint": str(checkpoint_path),
        "data_root": str(data_root),
        "inference_interval": [args.start, args.end],
        "history_replayed": [1, args.start - 1],
        "validation_threshold": threshold,
        "uses_future_labels": False,
        "updates_model_parameters": False,
        "updates_address_memory": True,
        "parameter_sha256": digest_after,
        "transactions": total_transactions,
        "predicted_illicit": total_alerts,
        "alert_rate": total_alerts / max(total_transactions, 1),
        "duration_seconds": time.perf_counter() - started,
        "peak_gpu_memory_mb": (
            round(torch.cuda.max_memory_allocated(device) / 1024**2, 2)
            if device.type == "cuda"
            else 0.0
        ),
        "predictions": str(prediction_path),
        "by_time": summary_by_time,
    }
    with (output_root / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
