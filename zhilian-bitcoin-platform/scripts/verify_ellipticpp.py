from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tthgnn_erl.ellipticpp import EllipticPPSnapshotDataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify processed Elliptic++ snapshots")
    parser.add_argument(
        "--root",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "ellipticpp",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA verification was requested but CUDA is unavailable.")

    dataset = EllipticPPSnapshotDataset(args.root, validate=True)
    totals = {
        "address_time_nodes": 0,
        "transactions": 0,
        "input_relations": 0,
        "output_relations": 0,
        "illicit_transactions": 0,
        "licit_transactions": 0,
        "unknown_transactions": 0,
        "transactions_with_nonfinite_features": 0,
    }
    largest_snapshot = None
    largest_relations = -1
    for snapshot in dataset:
        totals["address_time_nodes"] += snapshot.num_addresses
        totals["transactions"] += snapshot.num_transactions
        totals["input_relations"] += snapshot.num_input_relations
        totals["output_relations"] += snapshot.num_output_relations
        totals["illicit_transactions"] += int((snapshot.labels == 1).sum())
        totals["licit_transactions"] += int((snapshot.labels == 0).sum())
        totals["unknown_transactions"] += int((snapshot.labels == -1).sum())
        totals["transactions_with_nonfinite_features"] += int(
            (~snapshot.transaction_feature_mask).any(dim=1).sum()
        )
        relations = snapshot.num_input_relations + snapshot.num_output_relations
        if relations > largest_relations:
            largest_snapshot = snapshot
            largest_relations = relations

    expected = dataset.metadata["totals"]
    for key, value in totals.items():
        if int(expected[key]) != value:
            raise RuntimeError(f"Total mismatch for {key}: {value} != {expected[key]}")
    if largest_snapshot is None:
        raise RuntimeError("No snapshots were loaded.")

    device = torch.device(args.device)
    test_snapshot = largest_snapshot.to(device)
    input_incidence = test_snapshot.input_incidence(device)
    output_incidence = test_snapshot.output_incidence(device)
    node_states = torch.randn(test_snapshot.num_addresses, 8, device=device)
    input_degree = torch.sparse.sum(input_incidence, dim=0).to_dense().clamp_min(1.0)
    output_degree = torch.sparse.sum(output_incidence, dim=0).to_dense().clamp_min(1.0)
    input_role_states = torch.sparse.mm(input_incidence.transpose(0, 1), node_states)
    output_role_states = torch.sparse.mm(output_incidence.transpose(0, 1), node_states)
    input_role_states = input_role_states / input_degree.unsqueeze(-1)
    output_role_states = output_role_states / output_degree.unsqueeze(-1)
    if input_role_states.shape != (test_snapshot.num_transactions, 8):
        raise RuntimeError("Input-role sparse propagation returned an invalid shape.")
    if output_role_states.shape != (test_snapshot.num_transactions, 8):
        raise RuntimeError("Output-role sparse propagation returned an invalid shape.")
    if not torch.isfinite(input_role_states).all() or not torch.isfinite(output_role_states).all():
        raise RuntimeError("Sparse role propagation produced non-finite values.")

    print(
        json.dumps(
            {
                "status": "PASS",
                "snapshots": len(dataset),
                "totals": totals,
                "address_feature_dim": int(
                    dataset.metadata["feature_dimensions"]["address"]
                ),
                "transaction_feature_dim": int(
                    dataset.metadata["feature_dimensions"]["transaction"]
                ),
                "gpu_test": {
                    "device": str(device),
                    "gpu": (
                        torch.cuda.get_device_name(device)
                        if device.type == "cuda"
                        else None
                    ),
                    "time_id": test_snapshot.time_id,
                    "addresses": test_snapshot.num_addresses,
                    "transactions": test_snapshot.num_transactions,
                    "input_nnz": int(input_incidence._nnz()),
                    "output_nnz": int(output_incidence._nnz()),
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
