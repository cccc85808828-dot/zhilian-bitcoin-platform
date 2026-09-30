from __future__ import annotations

import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from competition_baselines.baseline_data import (  # noqa: E402
    FeatureStatistics,
    GraphSnapshot,
)
from competition_baselines.baseline_models import (  # noqa: E402
    EllipticHGTModel,
    FGEGCNModel,
    GPNModel,
    MDSTGNNModel,
    NSGCNLSTMModel,
    TFGATDCPLUModel,
)


def synthetic_snapshot() -> GraphSnapshot:
    transaction_features = torch.randn(6, 6)
    address_features = torch.randn(5, 4)
    return GraphSnapshot(
        time_id=1,
        global_address_ids=torch.arange(5),
        address_features=address_features,
        address_feature_mask=torch.ones(5, dtype=torch.bool),
        transaction_ids=torch.arange(100, 106),
        transaction_features=transaction_features,
        transaction_feature_mask=torch.ones_like(transaction_features, dtype=torch.bool),
        input_index=torch.tensor([[0, 1, 2, 3, 4, 0], [0, 1, 2, 3, 4, 5]]),
        output_index=torch.tensor([[1, 2, 3, 4, 0, 2], [0, 1, 2, 3, 4, 5]]),
        labels=torch.tensor([0, 1, 0, 1, -1, 0]),
        labeled_mask=torch.tensor([True, True, True, True, False, True]),
    )


def statistics(features: int) -> FeatureStatistics:
    return FeatureStatistics(torch.zeros(features), torch.ones(features))


def test_all_neural_baselines_forward() -> None:
    snapshot = synthetic_snapshot()
    transaction_stats = statistics(6)
    address_stats = statistics(4)
    edge_index = torch.tensor(
        [[0, 1, 2, 3, 4, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 0, 1, 2, 3, 4]]
    )
    roots = torch.nonzero(snapshot.labeled_mask, as_tuple=False).flatten()
    sequences = torch.tensor(
        [
            [0, 1, 2, -1],
            [1, 2, 3, -1],
            [2, 3, 4, -1],
            [3, 4, 5, -1],
            [4, 5, -1, -1],
            [5, -1, -1, -1],
        ],
        dtype=torch.int32,
    )

    nsgcn = NSGCNLSTMModel(transaction_stats, 8, 0.1)
    assert nsgcn.forward_roots(snapshot, edge_index, roots, sequences).logits.shape == (5,)

    hgt = EllipticHGTModel(transaction_stats, address_stats, 8, 0.1, 2, 1)
    assert hgt.forward_snapshot(snapshot, edge_index).logits.shape == (6,)

    tfgat = TFGATDCPLUModel(transaction_stats, 8, 0.1, 2, 1, 3, 2)
    assert tfgat.forward_snapshot(snapshot, edge_index).logits.shape == (6,)

    mdst = MDSTGNNModel(transaction_stats, 8, 0.1, 2)
    assert mdst.forward_snapshot(snapshot, edge_index, None, True).logits.shape == (6,)

    fg = FGEGCNModel(transaction_stats, 8, 0.1, 2)
    assert fg.forward_snapshot(snapshot, edge_index, None).logits.shape == (6,)

    gpn = GPNModel(transaction_stats, 8, 0.1, 2, 0.05, 0.2)
    loss, state, components = gpn.training_objective(
        snapshot, edge_index, None, 16, 0.2, 0.05, 0.01
    )
    assert loss.ndim == 0
    assert len(state) == 2
    assert components["normal_samples"] == 3.0
