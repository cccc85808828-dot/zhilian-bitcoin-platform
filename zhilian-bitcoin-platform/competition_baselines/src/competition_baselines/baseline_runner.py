from __future__ import annotations

import copy
import json
import random
import time
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score
from torch import nn
from torch.nn import functional as F

from .baseline_data import (
    GraphDataBundle,
    GraphSnapshot,
    expand_interval,
)
from .baseline_models import (
    DynamicModelOutput,
    EllipticHGTModel,
    FGEGCNModel,
    GPNModel,
    MDSTGNNModel,
    NSGCNLSTMModel,
    RootModelOutput,
    TFGATDCPLUModel,
    detach_state,
)
from .metrics import (
    binary_metrics,
    sanitize_metrics,
    select_threshold,
    temporal_metrics,
)


MODEL_NAMES = (
    "mdst_gnn",
    "tfgat_dcplu",
    "ellipticpp_hgt",
    "fg_egcn",
    "gpn",
    "nsgcn_lstm",
)

STATIC_MODELS = {"tfgat_dcplu", "ellipticpp_hgt", "nsgcn_lstm"}
DYNAMIC_MODELS = {"mdst_gnn", "fg_egcn", "gpn"}

IMPLEMENTATION_NOTES = {
    "mdst_gnn": (
        "method-level reimplementation; disjoint transaction identities require "
        "a causal graph-level historical GRU context"
    ),
    "tfgat_dcplu": (
        "method-level reimplementation; global attention uses global summary "
        "tokens plus bounded chunks to fit an 8 GB GPU"
    ),
    "ellipticpp_hgt": (
        "method-level HGT application on transaction/address snapshots; address "
        "labels are excluded to keep the target task transaction-only"
    ),
    "fg_egcn": (
        "method-level reimplementation with an EvolveGCN-H-style current-graph-"
        "conditioned weight updater, residual feature branch and node-wise gate"
    ),
    "gpn": (
        "method-level one-class reimplementation: training exposes labeled normal "
        "nodes only and uses generated pseudo anomalies with CBC/AAD/HCBD losses"
    ),
    "nsgcn_lstm": (
        "method-level transaction-node reimplementation with 3-hop predecessor "
        "subgraphs capped at 80 nodes, two GCN layers and a fund-flow LSTM"
    ),
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _labeled_roots(snapshot: GraphSnapshot) -> torch.Tensor:
    return torch.nonzero(snapshot.labeled_mask, as_tuple=False).flatten()


def weighted_bce(
    logits: torch.Tensor, labels: torch.Tensor, positive_weight: torch.Tensor
) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(
        logits, labels.to(torch.float32), pos_weight=positive_weight
    )


def focal_binary_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    positive_weight: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    labels_float = labels.to(torch.float32)
    bce = F.binary_cross_entropy_with_logits(
        logits, labels_float, reduction="none", pos_weight=positive_weight
    )
    probability = torch.sigmoid(logits)
    correct_probability = torch.where(labels_float > 0.5, probability, 1.0 - probability)
    return ((1.0 - correct_probability).pow(gamma) * bce).mean()


def labeled_counts(
    snapshots: Dict[int, GraphSnapshot], times: Sequence[int]
) -> Tuple[int, int]:
    positive = sum(int((snapshots[t].labels == 1).sum()) for t in times)
    negative = sum(int((snapshots[t].labels == 0).sum()) for t in times)
    return positive, negative


def build_model(
    name: str, config: Mapping[str, object], data: GraphDataBundle
) -> nn.Module:
    common = config["model"]
    hidden_dim = int(common["hidden_dim"])
    dropout = float(common["dropout"])
    if name == "nsgcn_lstm":
        return NSGCNLSTMModel(
            data.transaction_statistics, hidden_dim, dropout
        )
    if name == "ellipticpp_hgt":
        return EllipticHGTModel(
            data.transaction_statistics,
            data.address_statistics,
            hidden_dim,
            dropout,
            int(common["attention_heads"]),
            int(common.get("layers", 2)),
        )
    if name == "tfgat_dcplu":
        tfgat = common["tfgat_dcplu"]
        return TFGATDCPLUModel(
            data.transaction_statistics,
            int(tfgat.get("hidden_dim", hidden_dim)),
            dropout,
            int(tfgat.get("attention_heads", common["attention_heads"])),
            int(tfgat.get("transformer_layers", 2)),
            int(tfgat.get("global_chunk_size", 768)),
            int(tfgat.get("global_tokens", 32)),
        )
    if name == "mdst_gnn":
        return MDSTGNNModel(
            data.transaction_statistics,
            hidden_dim,
            dropout,
            int(common["mdst_gnn"].get("hops", 3)),
        )
    if name == "fg_egcn":
        return FGEGCNModel(
            data.transaction_statistics,
            hidden_dim,
            dropout,
            int(common.get("layers", 2)),
        )
    if name == "gpn":
        gpn = common["gpn"]
        return GPNModel(
            data.transaction_statistics,
            hidden_dim,
            dropout,
            int(common.get("layers", 2)),
            float(gpn.get("noise_std", 0.05)),
            float(gpn.get("aad_margin", 0.2)),
        )
    raise ValueError(f"Unknown model: {name}")


def _static_forward(
    name: str,
    model: nn.Module,
    snapshot: GraphSnapshot,
    edge_index: torch.Tensor,
    roots: torch.Tensor,
    data: GraphDataBundle,
) -> RootModelOutput:
    if name == "nsgcn_lstm":
        return model.forward_roots(
            snapshot, edge_index, roots, data.predecessor_sequences[snapshot.time_id]
        )
    output = model.forward_snapshot(snapshot, edge_index)
    return RootModelOutput(
        logits=output.logits[roots], embeddings=output.embeddings[roots]
    )


def _prediction_payload(
    labels: List[torch.Tensor],
    scores: List[torch.Tensor],
    times: List[torch.Tensor],
    ids: List[torch.Tensor],
) -> Dict[str, np.ndarray]:
    return {
        "labels": torch.cat(labels).numpy(),
        "scores": torch.cat(scores).numpy(),
        "times": torch.cat(times).numpy(),
        "transaction_ids": torch.cat(ids).numpy(),
    }


@torch.no_grad()
def predict_static(
    name: str,
    model: nn.Module,
    data: GraphDataBundle,
    times_to_predict: Sequence[int],
    device: torch.device,
) -> Dict[str, np.ndarray]:
    model.eval()
    labels: List[torch.Tensor] = []
    scores: List[torch.Tensor] = []
    time_parts: List[torch.Tensor] = []
    ids: List[torch.Tensor] = []
    for time_id in times_to_predict:
        snapshot = data.snapshots[time_id].to(device)
        roots = _labeled_roots(snapshot)
        edge_index = data.undirected_edges[time_id].to(device)
        output = _static_forward(name, model, snapshot, edge_index, roots, data)
        labels.append(snapshot.labels[roots].cpu())
        scores.append(torch.sigmoid(output.logits).cpu())
        time_parts.append(torch.full((roots.numel(),), time_id, dtype=torch.int16))
        ids.append(snapshot.transaction_ids[roots].cpu())
    return _prediction_payload(labels, scores, time_parts, ids)


@torch.no_grad()
def predict_dynamic(
    name: str,
    model: nn.Module,
    data: GraphDataBundle,
    replay_times: Sequence[int],
    times_to_predict: Sequence[int],
    device: torch.device,
) -> Dict[str, np.ndarray]:
    model.eval()
    prediction_set = set(times_to_predict)
    state = None
    labels: List[torch.Tensor] = []
    scores: List[torch.Tensor] = []
    time_parts: List[torch.Tensor] = []
    ids: List[torch.Tensor] = []
    for time_id in replay_times:
        snapshot = data.snapshots[time_id].to(device)
        edge_index = data.undirected_edges[time_id].to(device)
        if name == "mdst_gnn":
            output: DynamicModelOutput = model.forward_snapshot(
                snapshot, edge_index, state, compute_ssl=False
            )
        elif name == "fg_egcn":
            output = model.forward_snapshot(snapshot, edge_index, state)
        elif name == "gpn":
            output = model.inference_snapshot(snapshot, edge_index, state)
        else:
            raise ValueError(name)
        state = output.state
        if time_id not in prediction_set:
            continue
        roots = _labeled_roots(snapshot)
        labels.append(snapshot.labels[roots].cpu())
        scores.append(torch.sigmoid(output.logits[roots]).cpu())
        time_parts.append(torch.full((roots.numel(),), time_id, dtype=torch.int16))
        ids.append(snapshot.transaction_ids[roots].cpu())
    return _prediction_payload(labels, scores, time_parts, ids)


def evaluate_predictions(
    validation: Dict[str, np.ndarray],
    test: Dict[str, np.ndarray],
    top_fraction: float,
) -> Tuple[dict, pd.DataFrame]:
    threshold = select_threshold(validation["labels"], validation["scores"])
    validation_metrics = binary_metrics(
        validation["labels"], validation["scores"], threshold, top_fraction
    )
    test_metrics = binary_metrics(
        test["labels"], test["scores"], threshold, top_fraction
    )
    per_time, temporal_summary = temporal_metrics(
        test["labels"], test["scores"], test["times"], threshold, top_fraction
    )
    test_metrics.update(temporal_summary)
    return (
        {
            "threshold_selection": "validation_f1",
            "validation": validation_metrics,
            "test": test_metrics,
        },
        per_time,
    )


def _refresh_tfgat_pseudo_labels(
    model: nn.Module,
    data: GraphDataBundle,
    train_times: Sequence[int],
    device: torch.device,
) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor], dict]:
    model.eval()
    pseudo_labels: Dict[int, torch.Tensor] = {}
    pseudo_confidence: Dict[int, torch.Tensor] = {}
    counts = {"unknown": 0, "pseudo_illicit": 0, "pseudo_licit": 0}
    with torch.no_grad():
        for time_id in train_times:
            snapshot = data.snapshots[time_id].to(device)
            edge_index = data.undirected_edges[time_id].to(device)
            output = model.forward_snapshot(snapshot, edge_index)
            probability = torch.sigmoid(output.logits).cpu()
            unknown = snapshot.labels.cpu() < 0
            labels = (probability >= 0.5).to(torch.float32)
            confidence = (probability - 0.5).abs().mul(2.0).clamp_min(0.05)
            pseudo_labels[time_id] = labels
            pseudo_confidence[time_id] = confidence
            counts["unknown"] += int(unknown.sum())
            counts["pseudo_illicit"] += int((labels[unknown] == 1).sum())
            counts["pseudo_licit"] += int((labels[unknown] == 0).sum())
    return pseudo_labels, pseudo_confidence, counts


def train_static_model(
    name: str,
    config: Mapping[str, object],
    data: GraphDataBundle,
    output_dir: Path,
    device: torch.device,
    seed: int,
    max_epochs_override: Optional[int],
) -> dict:
    set_seed(seed)
    model = build_model(name, config, data).to(device)
    training = config["training"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    train_times = expand_interval(config["split"]["train"])
    validation_times = expand_interval(config["split"]["validation"])
    test_times = expand_interval(config["split"]["test"])
    positives, negatives = labeled_counts(data.snapshots, train_times)
    positive_weight = torch.tensor(negatives / positives, device=device)
    max_epochs = (
        int(max_epochs_override)
        if max_epochs_override is not None
        else int(training["max_epochs"])
    )
    patience = int(training["patience"])
    snapshots_per_step = int(training["snapshots_per_step"])
    pseudo_labels: Dict[int, torch.Tensor] = {}
    pseudo_confidence: Dict[int, torch.Tensor] = {}
    pseudo_audit: List[dict] = []
    tfgat_settings = config["model"].get("tfgat_dcplu", {})
    warmup_epochs = int(tfgat_settings.get("warmup_epochs", 5))
    refresh_interval = int(tfgat_settings.get("refresh_interval", 3))
    pseudo_weight = float(tfgat_settings.get("pseudo_weight", 0.25))
    min_epochs_before_stop = (
        warmup_epochs + refresh_interval if name == "tfgat_dcplu" else 1
    )
    history: List[dict] = []
    best_score = -float("inf")
    best_epoch = 0
    best_state = None
    stale = 0
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for epoch in range(1, max_epochs + 1):
        if (
            name == "tfgat_dcplu"
            and epoch > warmup_epochs
            and ((epoch - warmup_epochs - 1) % refresh_interval == 0)
        ):
            pseudo_labels, pseudo_confidence, audit = _refresh_tfgat_pseudo_labels(
                model, data, train_times, device
            )
            audit["epoch"] = epoch
            pseudo_audit.append(audit)
        model.train()
        shuffled = list(train_times)
        random.Random(seed + epoch).shuffle(shuffled)
        epoch_loss = 0.0
        epoch_labeled = 0
        for start in range(0, len(shuffled), snapshots_per_step):
            group = shuffled[start : start + snapshots_per_step]
            optimizer.zero_grad(set_to_none=True)
            group_loss = torch.zeros((), device=device)
            for time_id in group:
                snapshot = data.snapshots[time_id].to(device)
                roots = _labeled_roots(snapshot)
                edge_index = data.undirected_edges[time_id].to(device)
                full_output = None
                if name == "tfgat_dcplu":
                    full_output = model.forward_snapshot(snapshot, edge_index)
                    output = RootModelOutput(
                        logits=full_output.logits[roots],
                        embeddings=full_output.embeddings[roots],
                    )
                else:
                    output = _static_forward(
                        name, model, snapshot, edge_index, roots, data
                    )
                loss = weighted_bce(output.logits, snapshot.labels[roots], positive_weight)
                if name == "tfgat_dcplu" and time_id in pseudo_labels:
                    unknown = torch.nonzero(snapshot.labels < 0, as_tuple=False).flatten()
                    labels = pseudo_labels[time_id][unknown.cpu()].to(device)
                    confidence = pseudo_confidence[time_id][unknown.cpu()].to(device)
                    pseudo_loss = F.binary_cross_entropy_with_logits(
                        full_output.logits[unknown], labels, reduction="none"
                    )
                    loss = loss + pseudo_weight * (pseudo_loss * confidence).mean()
                group_loss = group_loss + loss / max(len(group), 1)
                epoch_loss += float(loss.detach()) * roots.numel()
                epoch_labeled += int(roots.numel())
            group_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(training["gradient_clip"])
            )
            optimizer.step()
        validation = predict_static(name, model, data, validation_times, device)
        validation_ap = float(
            average_precision_score(validation["labels"], validation["scores"])
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": epoch_loss / max(epoch_labeled, 1),
                "validation_pr_auc": validation_ap,
            }
        )
        print(
            f"[{name} seed={seed}] epoch={epoch:02d} "
            f"loss={history[-1]['train_loss']:.6f} val_pr={validation_ap:.6f}",
            flush=True,
        )
        if validation_ap > best_score + float(training["min_delta"]):
            best_score = validation_ap
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= patience and epoch >= min_epochs_before_stop:
                break
    if best_state is None:
        raise RuntimeError(f"No checkpoint selected for {name}.")
    model.load_state_dict(best_state)
    validation = predict_static(name, model, data, validation_times, device)
    test = predict_static(name, model, data, test_times, device)
    metrics, per_time = evaluate_predictions(
        validation, test, float(config["evaluation"]["top_fraction"])
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": name,
            "state_dict": best_state,
            "seed": seed,
            "best_epoch": best_epoch,
            "validation_threshold": metrics["test"]["threshold"],
            "transaction_statistics": data.transaction_statistics.as_dict(),
            "address_statistics": data.address_statistics.as_dict(),
            "config": dict(config),
        },
        output_dir / "best.pt",
    )
    pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)
    save_predictions(test, metrics["test"]["threshold"], output_dir)
    per_time.to_csv(output_dir / "per_time_metrics.csv", index=False)
    result = {
        "model": name,
        "seed": seed,
        "implementation": IMPLEMENTATION_NOTES[name],
        "best_epoch": best_epoch,
        "epochs_ran": len(history),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "duration_seconds": time.perf_counter() - started,
        "peak_gpu_memory_mb": (
            float(torch.cuda.max_memory_allocated(device) / 1024**2)
            if device.type == "cuda"
            else 0.0
        ),
        "positive_weight": float(positive_weight),
        "pseudo_label_audit": pseudo_audit,
        **metrics,
    }
    write_json(output_dir / "metrics.json", result)
    return result


def train_dynamic_model(
    name: str,
    config: Mapping[str, object],
    data: GraphDataBundle,
    output_dir: Path,
    device: torch.device,
    seed: int,
    max_epochs_override: Optional[int],
) -> dict:
    set_seed(seed)
    model = build_model(name, config, data).to(device)
    training = config["training"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    train_times = expand_interval(config["split"]["train"])
    validation_times = expand_interval(config["split"]["validation"])
    test_times = expand_interval(config["split"]["test"])
    positives, negatives = labeled_counts(data.snapshots, train_times)
    positive_weight = torch.tensor(negatives / positives, device=device)
    max_epochs = (
        int(max_epochs_override)
        if max_epochs_override is not None
        else int(training["max_epochs"])
    )
    patience = int(training["patience"])
    snapshots_per_step = int(training["snapshots_per_step"])
    history: List[dict] = []
    best_score = -float("inf")
    best_epoch = 0
    best_state = None
    stale = 0
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for epoch in range(1, max_epochs + 1):
        model.train()
        state = None
        epoch_loss = 0.0
        component_totals: Dict[str, float] = {}
        for start in range(0, len(train_times), snapshots_per_step):
            group = train_times[start : start + snapshots_per_step]
            optimizer.zero_grad(set_to_none=True)
            group_loss = torch.zeros((), device=device)
            for time_id in group:
                snapshot = data.snapshots[time_id].to(device)
                edge_index = data.undirected_edges[time_id].to(device)
                if name == "mdst_gnn":
                    output = model.forward_snapshot(
                        snapshot, edge_index, state, compute_ssl=True
                    )
                    roots = _labeled_roots(snapshot)
                    loss = weighted_bce(
                        output.logits[roots], snapshot.labels[roots], positive_weight
                    ) + float(config["model"]["mdst_gnn"].get("ssl_weight", 0.1)) * output.auxiliary_loss
                    state = output.state
                elif name == "fg_egcn":
                    output = model.forward_snapshot(snapshot, edge_index, state)
                    roots = _labeled_roots(snapshot)
                    loss = focal_binary_loss(
                        output.logits[roots],
                        snapshot.labels[roots],
                        positive_weight,
                        gamma=float(config["model"]["fg_egcn"].get("focal_gamma", 2.0)),
                    )
                    state = output.state
                elif name == "gpn":
                    gpn = config["model"]["gpn"]
                    loss, state, components = model.training_objective(
                        snapshot,
                        edge_index,
                        state,
                        max_normal_samples=int(gpn.get("max_normal_samples", 1024)),
                        aad_weight=float(gpn.get("aad_weight", 0.2)),
                        hcbd_weight=float(gpn.get("hcbd_weight", 0.05)),
                        compactness_weight=float(gpn.get("compactness_weight", 0.01)),
                    )
                    for key, value in components.items():
                        component_totals[key] = component_totals.get(key, 0.0) + value
                else:
                    raise ValueError(name)
                group_loss = group_loss + loss / max(len(group), 1)
                epoch_loss += float(loss.detach())
            group_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(training["gradient_clip"])
            )
            optimizer.step()
            state = detach_state(state)
        validation = predict_dynamic(
            name,
            model,
            data,
            list(train_times) + list(validation_times),
            validation_times,
            device,
        )
        validation_ap = float(
            average_precision_score(validation["labels"], validation["scores"])
        )
        row = {
            "epoch": epoch,
            "train_loss": epoch_loss / len(train_times),
            "validation_pr_auc": validation_ap,
        }
        if component_totals:
            row.update(
                {
                    f"train_{key}": value / len(train_times)
                    for key, value in component_totals.items()
                }
            )
        history.append(row)
        print(
            f"[{name} seed={seed}] epoch={epoch:02d} "
            f"loss={row['train_loss']:.6f} val_pr={validation_ap:.6f}",
            flush=True,
        )
        if validation_ap > best_score + float(training["min_delta"]):
            best_score = validation_ap
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is None:
        raise RuntimeError(f"No checkpoint selected for {name}.")
    model.load_state_dict(best_state)
    validation = predict_dynamic(
        name,
        model,
        data,
        list(train_times) + list(validation_times),
        validation_times,
        device,
    )
    test = predict_dynamic(
        name,
        model,
        data,
        list(train_times) + list(validation_times) + list(test_times),
        test_times,
        device,
    )
    metrics, per_time = evaluate_predictions(
        validation, test, float(config["evaluation"]["top_fraction"])
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": name,
            "state_dict": best_state,
            "seed": seed,
            "best_epoch": best_epoch,
            "validation_threshold": metrics["test"]["threshold"],
            "transaction_statistics": data.transaction_statistics.as_dict(),
            "config": dict(config),
        },
        output_dir / "best.pt",
    )
    pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)
    save_predictions(test, metrics["test"]["threshold"], output_dir)
    per_time.to_csv(output_dir / "per_time_metrics.csv", index=False)
    result = {
        "model": name,
        "seed": seed,
        "implementation": IMPLEMENTATION_NOTES[name],
        "best_epoch": best_epoch,
        "epochs_ran": len(history),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "duration_seconds": time.perf_counter() - started,
        "peak_gpu_memory_mb": (
            float(torch.cuda.max_memory_allocated(device) / 1024**2)
            if device.type == "cuda"
            else 0.0
        ),
        "positive_weight": (None if name == "gpn" else float(positive_weight)),
        "training_labels": (
            "labeled_normal_only" if name == "gpn" else "labeled_licit_and_illicit"
        ),
        **metrics,
    }
    write_json(output_dir / "metrics.json", result)
    return result


def save_predictions(test: Dict[str, np.ndarray], threshold: float, output_dir: Path) -> None:
    pd.DataFrame(
        {
            "transaction_id": test["transaction_ids"],
            "time": test["times"],
            "label": test["labels"],
            "score": test["scores"],
            "prediction": (test["scores"] >= threshold).astype(np.int8),
        }
    ).to_csv(output_dir / "test_predictions.csv", index=False)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(sanitize_metrics(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def run_model(
    name: str,
    config: Mapping[str, object],
    data: GraphDataBundle,
    output_root: Path,
    device: torch.device,
    seed: int,
    max_epochs_override: Optional[int] = None,
    force: bool = False,
) -> dict:
    if name not in MODEL_NAMES:
        raise ValueError(f"Unsupported model: {name}")
    output_dir = output_root / f"seed_{seed}" / name
    metrics_path = output_dir / "metrics.json"
    if metrics_path.is_file() and not force:
        print(f"[{name} seed={seed}] completed result exists; skipping", flush=True)
        return json.loads(metrics_path.read_text(encoding="utf-8"))
    set_seed(seed)
    if name in STATIC_MODELS:
        return train_static_model(
            name,
            config,
            data,
            output_dir,
            device,
            seed,
            max_epochs_override,
        )
    return train_dynamic_model(
        name,
        config,
        data,
        output_dir,
        device,
        seed,
        max_epochs_override,
    )
