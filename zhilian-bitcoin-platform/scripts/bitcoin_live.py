from __future__ import annotations

import copy
import datetime as dt
import hashlib
import json
import math
import os
import re
import statistics
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from tthgnn_erl.baselines import FeatureStatistics  # noqa: E402
from tthgnn_erl.ellipticpp import EllipticPPHypergraphSnapshot  # noqa: E402
from tthgnn_erl.temporal import TemporalMemoryHGNN  # noqa: E402


TXID_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")
ADDRESS_PATTERN = re.compile(r"^[A-Za-z0-9]{14,90}$")
SATOSHIS_PER_BTC = 100_000_000.0
INTERPRETABLE_FEATURES = {
    165: "输入侧交易度",
    166: "输出侧交易度",
    167: "交易总额（BTC）",
    168: "手续费（BTC）",
    169: "交易大小（字节）",
    170: "输入地址数",
    171: "输出地址数",
    172: "输入金额最小值",
    173: "输入金额最大值",
    174: "输入金额均值",
    175: "输入金额中位数",
    176: "输入金额总值",
    177: "输出金额最小值",
    178: "输出金额最大值",
    179: "输出金额均值",
    180: "输出金额中位数",
    181: "输出金额总值",
}
ATTRIBUTE_DISPLAY_PRIORITY = (
    167,
    168,
    169,
    170,
    171,
    176,
    181,
    178,
    177,
    174,
    179,
    173,
    172,
    175,
    180,
    165,
    166,
)
ATTRIBUTE_UNITS = {
    165: "count",
    166: "count",
    167: "btc",
    168: "btc",
    169: "bytes",
    170: "count",
    171: "count",
    172: "btc",
    173: "btc",
    174: "btc",
    175: "btc",
    176: "btc",
    177: "btc",
    178: "btc",
    179: "btc",
    180: "btc",
    181: "btc",
}


class ChainDataError(RuntimeError):
    def __init__(self, message: str, *, status: int = 502, hint: str = "") -> None:
        super().__init__(message)
        self.status = int(status)
        self.hint = hint


class LiveInferenceError(RuntimeError):
    pass


def _env_int(name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(os.environ.get(name, str(default)))
    except ValueError:
        value = default
    return min(max(value, minimum), maximum)


def _env_float(name: str, default: float, minimum: float, maximum: float) -> float:
    try:
        value = float(os.environ.get(name, str(default)))
    except ValueError:
        value = default
    return min(max(value, minimum), maximum)


def _utc_iso(timestamp: Any) -> str | None:
    if timestamp is None:
        return None
    try:
        value = int(timestamp)
        return dt.datetime.fromtimestamp(value, tz=dt.timezone.utc).isoformat(
            timespec="seconds"
        )
    except (ValueError, TypeError, OSError, OverflowError):
        return None


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def _safe_statistics(values: Sequence[float]) -> Tuple[float, float, float, float, float]:
    if not values:
        return (0.0, 0.0, 0.0, 0.0, 0.0)
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return (0.0, 0.0, 0.0, 0.0, 0.0)
    return (
        float(sum(finite)),
        float(min(finite)),
        float(max(finite)),
        float(statistics.fmean(finite)),
        float(statistics.median(finite)),
    )


def _unique_in_order(values: Iterable[str]) -> List[str]:
    return list(dict.fromkeys(str(value) for value in values if value))


def _script_identity(output: Mapping[str, Any], index: int) -> str:
    address = output.get("scriptpubkey_address")
    if address:
        return str(address)
    script = str(output.get("scriptpubkey") or f"output-{index}")
    digest = hashlib.sha256(script.encode("utf-8")).hexdigest()[:30]
    return f"script:{digest}"


def transaction_parties(
    transaction: Mapping[str, Any], maximum_per_role: int | None = None
) -> Dict[str, Any]:
    inputs: MutableMapping[str, float] = {}
    for index, item in enumerate(transaction.get("vin", [])):
        previous = item.get("prevout") or {}
        address = previous.get("scriptpubkey_address")
        if not address:
            if item.get("is_coinbase"):
                address = "coinbase"
            else:
                address = _script_identity(previous, index)
        inputs[str(address)] = inputs.get(str(address), 0.0) + (
            float(previous.get("value") or 0.0) / SATOSHIS_PER_BTC
        )

    outputs: MutableMapping[str, float] = {}
    for index, item in enumerate(transaction.get("vout", [])):
        address = _script_identity(item, index)
        outputs[address] = outputs.get(address, 0.0) + (
            float(item.get("value") or 0.0) / SATOSHIS_PER_BTC
        )

    def packed(role_values: Mapping[str, float]) -> Dict[str, Any]:
        rows = [
            {"address": address, "amount_btc": float(amount)}
            for address, amount in role_values.items()
        ]
        visible = rows if maximum_per_role is None else rows[:maximum_per_role]
        return {
            "count": len(rows),
            "truncated": len(visible) < len(rows),
            "items": visible,
            "all_addresses": [row["address"] for row in rows],
            "amounts": [row["amount_btc"] for row in rows],
        }

    return {"input": packed(inputs), "output": packed(outputs)}


def transaction_behavior_observations(
    transaction: Mapping[str, Any],
    context_by_txid: Mapping[str, Mapping[str, Any]],
    *,
    parent_context_checked: bool = False,
) -> Dict[str, Any]:
    """Return observable facts used for behavior wording; no risk label is inferred here."""
    inputs = [
        float((item.get("prevout") or {}).get("value") or 0.0) / SATOSHIS_PER_BTC
        for item in transaction.get("vin", [])
        if float((item.get("prevout") or {}).get("value") or 0.0) > 0.0
    ]
    outputs = [
        float(item.get("value") or 0.0) / SATOSHIS_PER_BTC
        for item in transaction.get("vout", [])
        if float(item.get("value") or 0.0) > 0.0
    ]
    status = transaction.get("status") or {}
    current_time = status.get("block_time")
    holding_seconds: List[float] = []
    eligible_inputs = 0
    for item in transaction.get("vin", []):
        parent_txid = str(item.get("txid") or "")
        if not parent_txid or item.get("is_coinbase"):
            continue
        eligible_inputs += 1
        parent = context_by_txid.get(parent_txid) or {}
        parent_time = (parent.get("status") or {}).get("block_time")
        if current_time is None or parent_time is None:
            continue
        elapsed = float(current_time) - float(parent_time)
        if elapsed >= 0.0:
            holding_seconds.append(elapsed)
    parties = transaction_parties(transaction)
    input_addresses = set(parties["input"]["all_addresses"])
    output_addresses = set(parties["output"]["all_addresses"])
    fee_btc = float(transaction.get("fee") or 0.0) / SATOSHIS_PER_BTC
    weight = float(transaction.get("weight") or 0.0)
    size = float(transaction.get("size") or 0.0)
    virtual_size = math.ceil(weight / 4.0) if weight > 0.0 else size
    return {
        "input_amounts_btc": inputs[:200],
        "output_amounts_btc": outputs[:200],
        "input_holding_seconds": sorted(holding_seconds),
        "holding_time_covered_inputs": len(holding_seconds),
        "holding_time_eligible_inputs": eligible_inputs,
        "parent_context_checked": bool(parent_context_checked),
        "overlap_address_count": len(input_addresses & output_addresses),
        "fee_rate_sat_vb": (
            float(transaction.get("fee") or 0.0) / virtual_size
            if virtual_size > 0.0
            else 0.0
        ),
        "fee_share_of_inputs": fee_btc / sum(inputs) if sum(inputs) > 0.0 else 0.0,
    }


class EsploraClient:
    """Small, cache-aware adapter for the official Esplora HTTP API."""

    def __init__(self) -> None:
        self.base_url = os.environ.get(
            "BITCOIN_API_BASE_URL", "https://blockstream.info/api"
        ).rstrip("/")
        self.timeout = _env_float("BITCOIN_API_TIMEOUT", 20.0, 2.0, 60.0)
        self.cache_ttl = _env_int("BITCOIN_CACHE_TTL", 300, 10, 86_400)
        self.max_transactions = _env_int(
            "BITCOIN_MAX_TRANSACTIONS", 75, 1, 500
        )
        self.maximum_pages = _env_int("BITCOIN_MAX_PAGES", 3, 1, 20)
        self.user_agent = os.environ.get(
            "BITCOIN_API_USER_AGENT", "TTHGNN-QIF/1.0"
        )
        self._cache: Dict[str, Tuple[float, Any]] = {}
        self._lock = threading.RLock()

    @property
    def provider_name(self) -> str:
        if "blockstream.info" in self.base_url:
            return "Blockstream Esplora"
        return "Esplora"

    def _get_json(self, path: str, *, cache_ttl: int | None = None) -> Any:
        now = time.monotonic()
        with self._lock:
            cached = self._cache.get(path)
            if cached and cached[0] > now:
                return copy.deepcopy(cached[1])
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            headers={"Accept": "application/json", "User-Agent": self.user_agent},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            if error.code in {400, 404}:
                raise ChainDataError(
                    "未在比特币主网找到该地址或交易",
                    status=404,
                    hint="请核对地址或交易哈希是否完整，并确认其属于比特币主网。",
                ) from error
            if error.code == 429:
                raise ChainDataError(
                    "链上数据服务当前请求较多",
                    status=503,
                    hint="请稍后重试，或部署自有Esplora节点。",
                ) from error
            raise ChainDataError(
                f"链上数据服务返回HTTP {error.code}", status=502
            ) from error
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
            reason = getattr(error, "reason", str(error))
            raise ChainDataError(
                f"暂时无法连接比特币链上数据服务：{reason}",
                status=502,
                hint="请检查网络连接或BITCOIN_API_BASE_URL配置。",
            ) from error
        ttl = self.cache_ttl if cache_ttl is None else int(cache_ttl)
        with self._lock:
            self._cache[path] = (now + ttl, payload)
        return copy.deepcopy(payload)

    def validate_address(self, address: str) -> str:
        clean = str(address).strip()
        if not ADDRESS_PATTERN.fullmatch(clean) or clean.startswith("script:"):
            raise ChainDataError(
                "比特币地址格式无效",
                status=400,
                hint="请输入完整的Base58、Bech32或Bech32m比特币主网地址。",
            )
        return clean

    def validate_txid(self, transaction_id: str) -> str:
        clean = str(transaction_id).strip().lower()
        if not TXID_PATTERN.fullmatch(clean):
            raise ChainDataError("交易哈希格式无效", status=400)
        return clean

    def transaction(self, transaction_id: str) -> Dict[str, Any]:
        txid = self.validate_txid(transaction_id)
        return dict(self._get_json(f"/tx/{txid}"))

    def address_history(self, address: str) -> Dict[str, Any]:
        address = self.validate_address(address)
        encoded = urllib.parse.quote(address, safe="")
        stats = dict(self._get_json(f"/address/{encoded}"))
        mempool = list(self._get_json(f"/address/{encoded}/txs/mempool"))
        transactions: List[Dict[str, Any]] = []
        seen = set()
        for item in mempool:
            txid = str(item.get("txid", ""))
            if txid and txid not in seen:
                seen.add(txid)
                transactions.append(dict(item))

        last_seen = ""
        pages = 0
        while len(transactions) < self.max_transactions and pages < self.maximum_pages:
            suffix = f"/{last_seen}" if last_seen else ""
            page = list(
                self._get_json(f"/address/{encoded}/txs/chain{suffix}")
            )
            pages += 1
            if not page:
                break
            for item in page:
                txid = str(item.get("txid", ""))
                if txid and txid not in seen:
                    seen.add(txid)
                    transactions.append(dict(item))
                    if len(transactions) >= self.max_transactions:
                        break
            last_seen = str(page[-1].get("txid", ""))
            if len(page) < 25 or not last_seen:
                break

        def order_key(item: Mapping[str, Any]) -> Tuple[int, int, str]:
            status = item.get("status") or {}
            confirmed = bool(status.get("confirmed"))
            return (
                int(status.get("block_time") or (2**62 if not confirmed else 0)),
                int(status.get("block_height") or (2**31 - 1)),
                str(item.get("txid", "")),
            )

        transactions.sort(key=order_key)
        chain_count = int((stats.get("chain_stats") or {}).get("tx_count") or 0)
        mempool_count = int((stats.get("mempool_stats") or {}).get("tx_count") or 0)
        total_count = chain_count + mempool_count
        return {
            "address": address,
            "stats": stats,
            "transactions": transactions,
            "total_transaction_count": total_count,
            "loaded_transaction_count": len(transactions),
            "history_complete": len(transactions) >= total_count,
            "pages_loaded": pages,
            "provider": self.provider_name,
            "provider_url": self.base_url,
            "fetched_at": dt.datetime.now(dt.timezone.utc).isoformat(
                timespec="seconds"
            ),
        }


class QuantileInteractionClassifier(nn.Module):
    """Inference-only QIF definition kept local for portable web deployment."""

    def __init__(
        self,
        bin_edges: torch.Tensor,
        embedding_dim: int,
        hidden_dim: int,
        dropout: float,
        use_bi_interaction: bool,
        interaction_mode: str,
        interaction_scale_initial: float,
    ) -> None:
        super().__init__()
        self.register_buffer("bin_edges", bin_edges.clone())
        self.feature_dim = int(bin_edges.size(0))
        self.num_bins = int(bin_edges.size(1) + 1)
        self.embedding_dim = int(embedding_dim)
        self.use_bi_interaction = bool(use_bi_interaction)
        self.interaction_mode = str(interaction_mode).lower()
        self.embedding = nn.Embedding(
            self.feature_dim * self.num_bins, self.embedding_dim
        )
        input_dim = self.feature_dim * (self.embedding_dim + 1)
        if self.use_bi_interaction and self.interaction_mode == "concat":
            input_dim += self.embedding_dim
        middle_dim = max(64, int(hidden_dim) // 2)
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, int(hidden_dim)),
            nn.LayerNorm(int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), middle_dim),
            nn.LayerNorm(middle_dim),
            nn.GELU(),
            nn.Dropout(float(dropout) * 0.6),
            nn.Linear(middle_dim, 1),
        )
        if self.use_bi_interaction and self.interaction_mode == "residual":
            interaction_hidden = max(16, self.embedding_dim * 2)
            self.interaction_normalization = nn.LayerNorm(self.embedding_dim)
            self.interaction_head = nn.Sequential(
                nn.Linear(self.embedding_dim, interaction_hidden),
                nn.GELU(),
                nn.Dropout(float(dropout) * 0.5),
                nn.Linear(interaction_hidden, 1),
            )
            self.interaction_scale = nn.Parameter(
                torch.tensor(float(interaction_scale_initial))
            )

    def forward(self, standardized_features: torch.Tensor) -> torch.Tensor:
        feature_bins = torch.stack(
            [
                torch.bucketize(
                    standardized_features[:, feature_index].contiguous(),
                    self.bin_edges[feature_index],
                )
                for feature_index in range(self.feature_dim)
            ],
            dim=1,
        )
        offsets = (
            torch.arange(self.feature_dim, device=standardized_features.device)
            * self.num_bins
        )
        embedded = self.embedding(feature_bins + offsets.unsqueeze(0))
        model_parts = [standardized_features, embedded.flatten(start_dim=1)]
        bi_interaction = None
        if self.use_bi_interaction:
            summed = embedded.sum(dim=1)
            bi_interaction = 0.5 * (
                summed.square() - embedded.square().sum(dim=1)
            )
            pair_scale = max(
                (self.feature_dim * (self.feature_dim - 1) / 2.0) ** 0.5,
                1.0,
            )
            bi_interaction = bi_interaction / pair_scale
            if self.interaction_mode == "concat":
                model_parts.append(bi_interaction)
        logits = self.classifier(torch.cat(model_parts, dim=1)).squeeze(1)
        if self.use_bi_interaction and self.interaction_mode == "residual":
            interaction_logits = self.interaction_head(
                self.interaction_normalization(bi_interaction)
            ).squeeze(1)
            logits = logits + torch.tanh(self.interaction_scale) * interaction_logits
        return logits


def _feature_statistics(values: Mapping[str, torch.Tensor]) -> FeatureStatistics:
    return FeatureStatistics(
        mean=values["mean"], std=values["std"], count=values["count"]
    )


def _load_graph_model(
    checkpoint_path: Path, device: torch.device
) -> Tuple[TemporalMemoryHGNN, Dict[str, Any]]:
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    config = checkpoint["config"]
    model_config = config["model"]
    attribute_prior = dict(config.get("attribute_prior", {}))
    model = TemporalMemoryHGNN(
        address_statistics=_feature_statistics(checkpoint["address_statistics"]),
        transaction_statistics=_feature_statistics(
            checkpoint["transaction_statistics"]
        ),
        hidden_dim=int(model_config["hidden_dim"]),
        memory_dim=int(model_config["memory_dim"]),
        dropout=float(model_config["dropout"]),
        propagation_layers=int(model_config["propagation_layers"]),
        feature_residual_enabled=bool(
            model_config.get("feature_residual_enabled", False)
        ),
        num_environment_experts=int(
            model_config.get("num_environment_experts", 1)
        ),
        global_expert_weight_floor=float(
            model_config.get("global_expert_weight_floor", 0.0)
        ),
        attribute_prior_residual_enabled=bool(attribute_prior.get("enabled", False)),
        structural_residual_scale=float(
            attribute_prior.get("structural_residual_scale", 1.0)
        ),
        transaction_encoder_type=str(
            model_config.get("transaction_encoder_type", "linear")
        ),
        transaction_interaction_rank=int(
            model_config.get("transaction_interaction_rank", 32)
        ),
        transaction_interaction_layers=int(
            model_config.get("transaction_interaction_layers", 2)
        ),
        address_risk_auxiliary_enabled=bool(
            config.get("address_auxiliary", {}).get("enabled", False)
        ),
        address_risk_initial_fusion_scale=float(
            config.get("address_auxiliary", {}).get("initial_fusion_scale", 0.1)
        ),
        transaction_cross_initial_scale=float(
            model_config.get("transaction_cross_initial_scale", 0.05)
        ),
        attribute_logit_residual_enabled=bool(
            model_config.get("attribute_logit_residual_enabled", False)
        ),
        attribute_logit_initial_scale=float(
            model_config.get("attribute_logit_initial_scale", 0.1)
        ),
        attribute_logit_uncertainty_temperature=float(
            model_config.get("attribute_logit_uncertainty_temperature", 0.0)
        ),
        temporal_memory_enabled=bool(
            model_config.get("temporal_memory_enabled", True)
        ),
        role_aware_propagation=bool(
            model_config.get("role_aware_propagation", True)
        ),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint


class TTHGNNLiveEngine:
    """Build live transaction hypergraphs and run the saved TTHGNN-QIF model."""

    def __init__(self, payload: Mapping[str, Any]) -> None:
        requested_device = os.environ.get("BITCOIN_INFERENCE_DEVICE", "cpu").lower()
        if requested_device == "auto":
            requested_device = "cuda" if torch.cuda.is_available() else "cpu"
        if requested_device.startswith("cuda") and not torch.cuda.is_available():
            requested_device = "cpu"
        self.device = torch.device(requested_device)
        self.seed = _env_int("BITCOIN_MODEL_SEED", 20260810, 1, 2**31 - 1)
        self.maximum_addresses_per_role = _env_int(
            "BITCOIN_MAX_ADDRESSES_PER_TX", 256, 8, 2048
        )
        fusion_path = (
            PROJECT_ROOT
            / "artifacts"
            / "pooled_f1"
            / "ablation"
            / "full"
            / f"seed_{self.seed}"
            / "best.pt"
        )
        graph_path = (
            PROJECT_ROOT
            / "artifacts"
            / "ablation"
            / "full"
            / "graph"
            / f"seed_{self.seed}"
            / "best.pt"
        )
        if not fusion_path.is_file() or not graph_path.is_file():
            raise LiveInferenceError(
                "实时推理检查点不完整，请保留QIF与时序超图模型文件。"
            )

        self.fusion_checkpoint = torch.load(
            fusion_path, map_location="cpu", weights_only=False
        )
        self.full_transaction_statistics = _feature_statistics(
            self.fusion_checkpoint["transaction_statistics"]
        )
        adapter_path = (
            PROJECT_ROOT
            / "artifacts"
            / "live_observable"
            / f"seed_{self.seed}"
            / "best.pt"
        )
        if adapter_path.is_file():
            self.attribute_checkpoint = torch.load(
                adapter_path, map_location="cpu", weights_only=False
            )
            self.attribute_feature_indices = [
                int(value) for value in self.attribute_checkpoint["feature_indices"]
            ]
            self.attribute_statistics = _feature_statistics(
                self.attribute_checkpoint["transaction_statistics"]
            )
            self.attribute_adapter_mode = "observable_17_feature_qif"
        else:
            self.attribute_checkpoint = self.fusion_checkpoint
            self.attribute_feature_indices = list(range(182))
            self.attribute_statistics = self.full_transaction_statistics
            self.attribute_adapter_mode = "full_qif_fallback"
        model_config = self.attribute_checkpoint["model_config"]
        self.attribute_model = QuantileInteractionClassifier(
            bin_edges=self.attribute_checkpoint["bin_edges"],
            embedding_dim=int(model_config["embedding_dim"]),
            hidden_dim=int(model_config["hidden_dim"]),
            dropout=float(model_config["dropout"]),
            use_bi_interaction=bool(model_config.get("use_bi_interaction", False)),
            interaction_mode=str(model_config.get("interaction_mode", "residual")),
            interaction_scale_initial=float(
                model_config.get("interaction_scale_initial", 0.10)
            ),
        ).to(self.device)
        self.attribute_model.load_state_dict(self.attribute_checkpoint["model_state"])
        self.attribute_model.eval()

        self.graph_model, graph_checkpoint = _load_graph_model(graph_path, self.device)
        self.graph_checkpoint = graph_checkpoint
        self.address_mean = graph_checkpoint["address_statistics"]["mean"].to(
            dtype=torch.float32
        )
        fusion = self.fusion_checkpoint["fusion"]
        self.graph_weight = float(fusion["graph_weight"])
        self.attribute_weight = float(fusion["attribute_weight"])
        calibration = dict(payload.get("risk_calibration", {}))
        live_calibration_path = (
            PROJECT_ROOT
            / "artifacts"
            / "live_observable"
            / f"seed_{self.seed}"
            / "calibration.json"
        )
        live_calibration: Dict[str, Any] = {}
        if live_calibration_path.is_file() and adapter_path.is_file():
            live_calibration = json.loads(
                live_calibration_path.read_text(encoding="utf-8")
            )
        if live_calibration:
            live_fusion = live_calibration["fusion"]
            self.graph_weight = float(live_fusion["graph_weight"])
            self.attribute_weight = float(live_fusion["attribute_weight"])
            self.threshold_logit = float(live_fusion["threshold_logit"])
            reference = live_calibration["display_index_reference_logits"]
            self.risk_index_method = str(live_calibration["display_index_method"])
            self.live_validation_metrics = live_calibration.get(
                "validation_metrics"
            )
            self.live_test_metrics = live_calibration.get("test_metrics")
        else:
            self.threshold_logit = float(fusion["threshold"])
            reference = calibration.get("display_index_reference_logits") or []
            self.risk_index_method = "validation_empirical_percentile"
            self.live_validation_metrics = None
            self.live_test_metrics = None
        self.reference_logits = np.asarray(reference, dtype=np.float64)
        self.platt_coefficient = float(calibration.get("coefficient", 1.0))
        self.platt_intercept = float(calibration.get("intercept", 0.0))
        if self.reference_logits.size:
            self.threshold_score = self._risk_index(self.threshold_logit)
        else:
            self.threshold_score = _sigmoid(
                self.platt_coefficient * self.threshold_logit
                + self.platt_intercept
            )
            self.risk_index_method = "platt_probability_fallback"

    def _risk_index(self, logit: float) -> float:
        if self.reference_logits.size:
            rank = int(np.searchsorted(self.reference_logits, float(logit), side="right"))
            return float((rank + 0.5) / (self.reference_logits.size + 1.0))
        return _sigmoid(self.platt_coefficient * float(logit) + self.platt_intercept)

    @staticmethod
    def _tx_order_key(transaction: Mapping[str, Any]) -> Tuple[int, int, str]:
        status = transaction.get("status") or {}
        return (
            int(status.get("block_time") or 2**62),
            int(status.get("block_height") or 2**31 - 1),
            str(transaction.get("txid", "")),
        )

    def _transaction_features(
        self,
        transaction: Mapping[str, Any],
        loaded_child_counts: Mapping[str, int],
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        values = torch.zeros(182, dtype=torch.float32)
        mask = torch.zeros(182, dtype=torch.bool)
        parties = transaction_parties(transaction)
        inputs = [
            float((item.get("prevout") or {}).get("value") or 0.0)
            / SATOSHIS_PER_BTC
            for item in transaction.get("vin", [])
            if item.get("prevout") is not None
        ]
        outputs = [
            float(item.get("value") or 0.0) / SATOSHIS_PER_BTC
            for item in transaction.get("vout", [])
        ]
        previous_txids = {
            str(item.get("txid"))
            for item in transaction.get("vin", [])
            if item.get("txid") and not item.get("is_coinbase")
        }
        input_stats = _safe_statistics(inputs)
        output_stats = _safe_statistics(outputs)
        fee_btc = float(transaction.get("fee") or 0.0) / SATOSHIS_PER_BTC
        output_total = float(sum(outputs))
        raw = {
            165: float(len(previous_txids)),
            166: float(loaded_child_counts.get(str(transaction.get("txid")), 0)),
            167: output_total,
            168: fee_btc,
            169: float(transaction.get("size") or 0.0),
            170: float(parties["input"]["count"]),
            171: float(parties["output"]["count"]),
            172: input_stats[1],
            173: input_stats[2],
            174: input_stats[3],
            175: input_stats[4],
            176: input_stats[0],
            177: output_stats[1],
            178: output_stats[2],
            179: output_stats[3],
            180: output_stats[4],
            181: output_stats[0],
        }
        for index, value in raw.items():
            values[index] = float(value)
            mask[index] = True
        return values, mask, parties

    @staticmethod
    def _empty_address_state() -> Dict[str, Any]:
        return {
            "txids": set(),
            "blocks": [],
            "sent_blocks": [],
            "received_blocks": [],
            "periods": set(),
            "sent": [],
            "received": [],
            "fees": [],
            "fee_shares": [],
            "counterpart_counts": {},
        }

    def _update_address_states(
        self,
        states: MutableMapping[str, Dict[str, Any]],
        transaction: Mapping[str, Any],
        parties: Mapping[str, Any],
    ) -> None:
        txid = str(transaction.get("txid", ""))
        status = transaction.get("status") or {}
        block = status.get("block_height")
        block_value = int(block) if block is not None else None
        event_time = _utc_iso(status.get("block_time"))
        period = event_time[:10] if event_time else "mempool"
        input_map = {
            item["address"]: float(item["amount_btc"])
            for item in parties["input"]["items"]
        }
        output_map = {
            item["address"]: float(item["amount_btc"])
            for item in parties["output"]["items"]
        }
        fee_btc = float(transaction.get("fee") or 0.0) / SATOSHIS_PER_BTC
        all_addresses = set(input_map) | set(output_map)
        for address in all_addresses:
            state = states.setdefault(address, self._empty_address_state())
            state["txids"].add(txid)
            state["periods"].add(period)
            if block_value is not None:
                state["blocks"].append(block_value)
            if address in input_map:
                amount = input_map[address]
                state["sent"].append(amount)
                state["fees"].append(fee_btc)
                if amount > 0.0:
                    state["fee_shares"].append(fee_btc / amount)
                if block_value is not None:
                    state["sent_blocks"].append(block_value)
            if address in output_map:
                state["received"].append(output_map[address])
                if block_value is not None:
                    state["received_blocks"].append(block_value)
            counterpart_counts = state["counterpart_counts"]
            for counterpart in all_addresses - {address}:
                counterpart_counts[counterpart] = counterpart_counts.get(counterpart, 0) + 1

    def _address_features(self, state: Mapping[str, Any]) -> torch.Tensor:
        values = self.address_mean.clone()
        sent = list(state["sent"])
        received = list(state["received"])
        transacted = sent + received
        blocks = sorted(int(value) for value in state["blocks"])
        sent_blocks = sorted(int(value) for value in state["sent_blocks"])
        received_blocks = sorted(int(value) for value in state["received_blocks"])
        values[0] = float(len(sent))
        values[1] = float(len(received))
        values[5] = float(len(state["txids"]))
        values[8] = float(len(state["periods"]))
        if blocks:
            values[2] = float(blocks[0])
            values[3] = float(blocks[-1])
            values[4] = float(blocks[-1] - blocks[0])
        if sent_blocks:
            values[6] = float(sent_blocks[0])
        if received_blocks:
            values[7] = float(received_blocks[0])

        for start, series in (
            (9, transacted),
            (14, sent),
            (19, received),
            (24, list(state["fees"])),
            (29, list(state["fee_shares"])),
        ):
            stats = _safe_statistics(series)
            values[start : start + 5] = torch.tensor(stats)

        def gaps(series: Sequence[int]) -> List[float]:
            return [float(b - a) for a, b in zip(series, series[1:])]

        for start, series in (
            (34, gaps(blocks)),
            (39, gaps(sent_blocks)),
            (44, gaps(received_blocks)),
        ):
            stats = _safe_statistics(series)
            values[start : start + 5] = torch.tensor(stats)

        counterpart_values = list(state["counterpart_counts"].values())
        values[49] = float(sum(int(value) > 1 for value in counterpart_values))
        values[50:55] = torch.tensor(_safe_statistics(counterpart_values))
        return values

    def _attribute_profile(
        self, features: torch.Tensor, mask: torch.Tensor
    ) -> List[Dict[str, Any]]:
        standardized = (features - self.full_transaction_statistics.mean) / (
            self.full_transaction_statistics.std
        )
        rows_by_index: Dict[int, Dict[str, Any]] = {}
        for index in INTERPRETABLE_FEATURES:
            if not bool(mask[index]):
                continue
            raw_value = float(features[index])
            standardized_value = float(standardized[index])
            deviation = (
                abs(standardized_value) if math.isfinite(standardized_value) else 0.0
            )
            rows_by_index[index] = {
                "feature_index": index,
                "name": INTERPRETABLE_FEATURES[index],
                "raw_value": raw_value,
                "standardized_value": standardized_value,
                "unit": ATTRIBUTE_UNITS[index],
                "signal_strength": 100.0 * deviation / (deviation + 2.0),
            }

        # The card describes observable chain facts, while the QIF branch still
        # consumes every available feature.  Prefer non-zero business values so
        # absent coinbase inputs or zero fees cannot crowd out amounts and size.
        ordered = [
            rows_by_index[index]
            for index in ATTRIBUTE_DISPLAY_PRIORITY
            if index in rows_by_index
        ]
        selected = [row for row in ordered if abs(row["raw_value"]) > 1e-12][:8]
        if len(selected) < 8:
            selected_indexes = {row["feature_index"] for row in selected}
            remaining = [
                row for row in ordered if row["feature_index"] not in selected_indexes
            ]
            remaining.sort(
                key=lambda item: abs(item["standardized_value"])
                if math.isfinite(item["standardized_value"])
                else 0.0,
                reverse=True,
            )
            selected.extend(remaining[: 8 - len(selected)])
        return selected

    def analyze_transactions(
        self, transactions: Sequence[Mapping[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        ordered = sorted((dict(item) for item in transactions), key=self._tx_order_key)
        if not ordered:
            return {}
        context_by_txid = {str(item.get("txid")): item for item in ordered}
        loaded_child_counts: Dict[str, int] = {}
        for transaction in ordered:
            for item in transaction.get("vin", []):
                parent = str(item.get("txid", ""))
                if parent:
                    loaded_child_counts[parent] = loaded_child_counts.get(parent, 0) + 1

        prepared: Dict[str, Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]] = {}
        all_addresses: List[str] = []
        for transaction in ordered:
            txid = str(transaction["txid"])
            features, mask, parties = self._transaction_features(
                transaction, loaded_child_counts
            )
            for role in ("input", "output"):
                addresses = parties[role]["all_addresses"]
                if len(addresses) > self.maximum_addresses_per_role:
                    addresses = addresses[: self.maximum_addresses_per_role]
                all_addresses.extend(addresses)
            prepared[txid] = (features, mask, parties)
        all_addresses = _unique_in_order(all_addresses)
        address_to_global = {address: index for index, address in enumerate(all_addresses)}
        memory_bank = self.graph_model.initial_memory(
            len(all_addresses), self.device
        )
        states: Dict[str, Dict[str, Any]] = {}
        results: Dict[str, Dict[str, Any]] = {}

        groups: List[List[Dict[str, Any]]] = []
        group_keys: List[Tuple[Any, Any]] = []
        for transaction in ordered:
            status = transaction.get("status") or {}
            key = (
                status.get("block_height") if status.get("confirmed") else "mempool",
                status.get("block_time") if status.get("confirmed") else None,
            )
            if not group_keys or key != group_keys[-1]:
                group_keys.append(key)
                groups.append([])
            groups[-1].append(transaction)

        with torch.no_grad():
            sequence_order = 0
            for group_index, group in enumerate(groups, start=1):
                active_addresses: List[str] = []
                for transaction in group:
                    txid = str(transaction["txid"])
                    parties = prepared[txid][2]
                    self._update_address_states(states, transaction, parties)
                    for role in ("input", "output"):
                        role_addresses = parties[role]["all_addresses"][
                            : self.maximum_addresses_per_role
                        ]
                        active_addresses.extend(role_addresses)
                active_addresses = _unique_in_order(active_addresses)
                local_by_address = {
                    address: index for index, address in enumerate(active_addresses)
                }
                global_ids = torch.tensor(
                    [address_to_global[address] for address in active_addresses],
                    dtype=torch.long,
                )
                address_features = torch.stack(
                    [self._address_features(states[address]) for address in active_addresses]
                )
                tx_features = torch.stack(
                    [prepared[str(item["txid"])][0] for item in group]
                )
                tx_masks = torch.stack(
                    [prepared[str(item["txid"])][1] for item in group]
                )
                input_relations: List[Tuple[int, int]] = []
                output_relations: List[Tuple[int, int]] = []
                for tx_position, transaction in enumerate(group):
                    parties = prepared[str(transaction["txid"])][2]
                    for address in parties["input"]["all_addresses"][
                        : self.maximum_addresses_per_role
                    ]:
                        input_relations.append((local_by_address[address], tx_position))
                    for address in parties["output"]["all_addresses"][
                        : self.maximum_addresses_per_role
                    ]:
                        output_relations.append((local_by_address[address], tx_position))

                def relation_tensor(rows: Sequence[Tuple[int, int]]) -> torch.Tensor:
                    if not rows:
                        return torch.empty((2, 0), dtype=torch.long)
                    return torch.tensor(rows, dtype=torch.long).t().contiguous()

                snapshot = EllipticPPHypergraphSnapshot(
                    time_id=group_index,
                    global_address_ids=global_ids,
                    address_features=address_features,
                    address_feature_mask=torch.ones(
                        len(active_addresses), dtype=torch.bool
                    ),
                    transaction_ids=torch.arange(len(group), dtype=torch.long),
                    transaction_features=tx_features,
                    transaction_feature_mask=tx_masks,
                    input_index=relation_tensor(input_relations),
                    output_index=relation_tensor(output_relations),
                    labels=torch.full((len(group),), -1, dtype=torch.long),
                    labeled_mask=torch.zeros(len(group), dtype=torch.bool),
                )
                snapshot.validate()
                device_snapshot = snapshot.to(self.device)
                selected_memory = memory_bank.index_select(
                    0, global_ids.to(self.device)
                )
                graph_logits, updated_memory = self.graph_model(
                    device_snapshot, selected_memory
                )
                memory_bank.index_copy_(
                    0, global_ids.to(self.device), updated_memory
                )
                feature_indices = torch.tensor(
                    self.attribute_feature_indices,
                    dtype=torch.long,
                    device=self.device,
                )
                attribute_values = device_snapshot.transaction_features.index_select(
                    1, feature_indices
                )
                attribute_mask = device_snapshot.transaction_feature_mask.index_select(
                    1, feature_indices
                )
                standardized = (
                    attribute_values
                    - self.attribute_statistics.mean.to(self.device)
                ) / self.attribute_statistics.std.to(self.device)
                standardized = torch.where(
                    attribute_mask,
                    standardized,
                    torch.zeros_like(standardized),
                )
                attribute_logits = self.attribute_model(standardized)
                fused_logits = (
                    self.graph_weight * graph_logits
                    + self.attribute_weight * attribute_logits
                )
                for position, transaction in enumerate(group):
                    sequence_order += 1
                    txid = str(transaction["txid"])
                    fused_logit = float(fused_logits[position].cpu())
                    graph_logit = float(graph_logits[position].cpu())
                    attribute_logit = float(attribute_logits[position].cpu())
                    risk_score = self._risk_index(fused_logit)
                    result_parties = prepared[txid][2]
                    for role in ("input", "output"):
                        result_parties[role].pop("all_addresses", None)
                        result_parties[role].pop("amounts", None)
                        visible_items = result_parties[role]["items"][
                            : self.maximum_addresses_per_role
                        ]
                        result_parties[role]["items"] = visible_items
                        result_parties[role]["truncated"] = (
                            len(visible_items) < int(result_parties[role]["count"])
                        )
                    status = transaction.get("status") or {}
                    results[txid] = {
                        "transaction_id": txid,
                        "predicted_label": int(fused_logit >= self.threshold_logit),
                        "risk_score": risk_score,
                        "risk_index": risk_score,
                        "graph_score": _sigmoid(graph_logit),
                        "qif_score": _sigmoid(attribute_logit),
                        "model_score": _sigmoid(fused_logit),
                        "margin_to_threshold": risk_score - self.threshold_score,
                        "decision_threshold_score": self.threshold_score,
                        "event_time": _utc_iso(status.get("block_time")),
                        "sequence_order": sequence_order,
                        "confirmation": {
                            "confirmed": bool(status.get("confirmed")),
                            "block_height": status.get("block_height"),
                            "block_hash": status.get("block_hash"),
                        },
                        "input_addresses": result_parties["input"],
                        "output_addresses": result_parties["output"],
                        "amount_btc": float(
                            sum(
                                float(item.get("value") or 0.0)
                                for item in transaction.get("vout", [])
                            )
                            / SATOSHIS_PER_BTC
                        ),
                        "fee_btc": float(transaction.get("fee") or 0.0)
                        / SATOSHIS_PER_BTC,
                        "size_bytes": int(transaction.get("size") or 0),
                        "attribute_profile": self._attribute_profile(
                            prepared[txid][0], prepared[txid][1]
                        ),
                        "behavior_observations": transaction_behavior_observations(
                            transaction, context_by_txid
                        ),
                        "data_source": "bitcoin_mainnet",
                        "analysis_mode": "live_tthgnn_qif",
                        "risk_index_method": self.risk_index_method,
                    }
        return results


class BitcoinLiveService:
    def __init__(self, payload: Mapping[str, Any]) -> None:
        self.enabled = os.environ.get("BITCOIN_LIVE_ENABLED", "1").lower() not in {
            "0",
            "false",
            "no",
        }
        self.client = EsploraClient()
        self._payload = payload
        self._engine: TTHGNNLiveEngine | None = None
        self._engine_error: str | None = None
        self._profiles: Dict[str, Tuple[float, Dict[str, Any]]] = {}
        self._cases: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()
        self._inference_lock = threading.Lock()

    def _get_engine(self) -> TTHGNNLiveEngine:
        with self._lock:
            if self._engine is not None:
                return self._engine
            if self._engine_error is not None:
                raise LiveInferenceError(self._engine_error)
            try:
                self._engine = TTHGNNLiveEngine(self._payload)
            except Exception as error:  # preserve a stable diagnostic after first load
                self._engine_error = str(error)
                raise LiveInferenceError(self._engine_error) from error
            return self._engine

    def status(self) -> Dict[str, Any]:
        status = {
            "enabled": self.enabled,
            "network": "bitcoin_mainnet",
            "provider": self.client.provider_name,
            "provider_url": self.client.base_url,
            "max_transactions_per_address": self.client.max_transactions,
            "model_loaded": self._engine is not None,
            "model_error": self._engine_error,
        }
        if self._engine is not None:
            status.update(
                {
                    "attribute_adapter": self._engine.attribute_adapter_mode,
                    "risk_index_method": self._engine.risk_index_method,
                    "decision_threshold_score": self._engine.threshold_score,
                }
            )
        return status

    def case(self, transaction_id: str) -> Dict[str, Any] | None:
        with self._lock:
            case = self._cases.get(str(transaction_id).lower())
            return copy.deepcopy(case) if case is not None else None

    def cached_profile(self, address: str) -> Dict[str, Any] | None:
        key = str(address).strip()
        now = time.monotonic()
        with self._lock:
            cached = self._profiles.get(key)
            if cached and cached[0] > now:
                return copy.deepcopy(cached[1])
        return None

    def address_profile(self, address: str) -> Dict[str, Any]:
        if not self.enabled:
            raise ChainDataError("实时链上查询未启用", status=503)
        cached = self.cached_profile(address)
        if cached is not None:
            return cached
        history = self.client.address_history(address)
        transactions = history["transactions"]
        if not transactions:
            stats = history["stats"]
            profile = self._profile_from_cases(history, [], stats)
        else:
            engine = self._get_engine()
            with self._inference_lock:
                analyzed = engine.analyze_transactions(transactions)
            cases = [
                analyzed[str(transaction["txid"])] for transaction in transactions
            ]
            profile = self._profile_from_cases(history, cases, history["stats"])
            with self._lock:
                self._cases.update({case["transaction_id"]: case for case in cases})
        with self._lock:
            self._profiles[history["address"]] = (
                time.monotonic() + self.client.cache_ttl,
                profile,
            )
        return copy.deepcopy(profile)

    def analyze_transaction(self, transaction_id: str) -> Dict[str, Any]:
        txid = self.client.validate_txid(transaction_id)
        cached = self.case(txid)
        cached_observations = (cached or {}).get("behavior_observations") or {}
        if cached is not None and cached_observations.get("parent_context_checked"):
            return cached
        transaction = self.client.transaction(txid)
        if cached is None:
            with self._inference_lock:
                case = self._get_engine().analyze_transactions([transaction])[txid]
        else:
            case = cached
        parent_transactions: Dict[str, Mapping[str, Any]] = {}
        parent_txids = _unique_in_order(
            str(item.get("txid") or "")
            for item in transaction.get("vin", [])
            if item.get("txid") and not item.get("is_coinbase")
        )[:8]
        for parent_txid in parent_txids:
            try:
                parent_transactions[parent_txid] = self.client.transaction(parent_txid)
            except ChainDataError:
                continue
        context = {txid: transaction, **parent_transactions}
        case["behavior_observations"] = transaction_behavior_observations(
            transaction, context, parent_context_checked=True
        )
        with self._lock:
            self._cases[txid] = case
        return copy.deepcopy(case)

    @staticmethod
    def _build_relation_graph(
        focal_address: str,
        raw_transactions: Mapping[str, Mapping[str, Any]],
        cases: Sequence[Mapping[str, Any]],
    ) -> Dict[str, Any]:
        nodes: Dict[str, Dict[str, Any]] = {}
        edges: List[Dict[str, Any]] = []

        def address_node(address: str) -> Dict[str, Any]:
            node_id = f"address:{address}"
            node = nodes.setdefault(
                node_id,
                {
                    "id": node_id,
                    "kind": "focal" if address == focal_address else "address",
                    "address": address,
                    "label": address,
                    "roles": set(),
                    "degree": 0,
                    "maximum_risk_score": 0.0,
                    "high_risk_links": 0,
                },
            )
            return node

        focal = address_node(focal_address)
        focal["roles"].add("focus")

        for case in cases:
            txid = str(case["transaction_id"])
            raw = raw_transactions.get(txid) or {}
            parties = transaction_parties(raw)
            tx_node_id = f"transaction:{txid}"
            risk_score = float(case.get("risk_score", 0.0))
            predicted_label = int(case.get("predicted_label", 0))
            nodes[tx_node_id] = {
                "id": tx_node_id,
                "kind": "transaction",
                "transaction_id": txid,
                "label": txid,
                "risk_score": risk_score,
                "predicted_label": predicted_label,
                "event_time": case.get("event_time"),
                "degree": 0,
            }
            for role, direction in (("input", "in"), ("output", "out")):
                for item in parties[role]["items"]:
                    address = str(item["address"])
                    address_data = address_node(address)
                    address_data["roles"].add(role)
                    address_data["degree"] += 1
                    address_data["maximum_risk_score"] = max(
                        float(address_data["maximum_risk_score"]), risk_score
                    )
                    address_data["high_risk_links"] += predicted_label
                    nodes[tx_node_id]["degree"] += 1
                    source = address_data["id"] if direction == "in" else tx_node_id
                    target = tx_node_id if direction == "in" else address_data["id"]
                    edges.append(
                        {
                            "id": f"edge:{len(edges)}",
                            "source": source,
                            "target": target,
                            "role": role,
                            "amount_btc": float(item.get("amount_btc") or 0.0),
                            "transaction_id": txid,
                            "risk_score": risk_score,
                            "predicted_label": predicted_label,
                        }
                    )

        serializable_nodes: List[Dict[str, Any]] = []
        for node in nodes.values():
            exported = dict(node)
            if isinstance(exported.get("roles"), set):
                exported["roles"] = sorted(exported["roles"])
            serializable_nodes.append(exported)
        address_count = sum(
            node["kind"] in {"address", "focal"} for node in serializable_nodes
        )
        transaction_count = sum(
            node["kind"] == "transaction" for node in serializable_nodes
        )
        return {
            "focal_node_id": f"address:{focal_address}",
            "nodes": serializable_nodes,
            "edges": edges,
            "node_count": len(serializable_nodes),
            "edge_count": len(edges),
            "address_node_count": address_count,
            "transaction_node_count": transaction_count,
        }

    @staticmethod
    def _profile_from_cases(
        history: Mapping[str, Any],
        cases: Sequence[Mapping[str, Any]],
        stats: Mapping[str, Any],
    ) -> Dict[str, Any]:
        address = str(history["address"])
        raw_transactions = {
            str(item.get("txid")): item for item in history.get("transactions", [])
        }
        transactions: List[Dict[str, Any]] = []
        for case in cases:
            roles = []
            raw = raw_transactions.get(str(case["transaction_id"]), {})
            raw_parties = transaction_parties(raw)
            if address in raw_parties["input"]["all_addresses"]:
                roles.append("input")
            if address in raw_parties["output"]["all_addresses"]:
                roles.append("output")
            transactions.append(
                {
                    "transaction_id": case["transaction_id"],
                    "event_time": case.get("event_time"),
                    "sequence_order": case.get("sequence_order"),
                    "risk_score": case["risk_score"],
                    "predicted_label": case["predicted_label"],
                    "roles": roles,
                    "amount_btc": case.get("amount_btc"),
                    "confirmed": bool(
                        (case.get("confirmation") or {}).get("confirmed")
                    ),
                    "block_height": (case.get("confirmation") or {}).get(
                        "block_height"
                    ),
                }
            )
        transactions.sort(key=lambda item: item["risk_score"], reverse=True)
        scores = [float(item["risk_score"]) for item in transactions]
        input_count = sum("input" in item["roles"] for item in transactions)
        output_count = sum("output" in item["roles"] for item in transactions)
        event_times = sorted(
            item["event_time"] for item in transactions if item.get("event_time")
        )
        chain = stats.get("chain_stats") or {}
        mempool = stats.get("mempool_stats") or {}
        funded = float(chain.get("funded_txo_sum") or 0) + float(
            mempool.get("funded_txo_sum") or 0
        )
        spent = float(chain.get("spent_txo_sum") or 0) + float(
            mempool.get("spent_txo_sum") or 0
        )
        relation_graph = BitcoinLiveService._build_relation_graph(
            address, raw_transactions, cases
        )
        return {
            "address": address,
            "source": "bitcoin_mainnet",
            "provider": history["provider"],
            "fetched_at": history["fetched_at"],
            "related_transaction_count": len(transactions),
            "total_transaction_count": int(history["total_transaction_count"]),
            "loaded_transaction_count": int(history["loaded_transaction_count"]),
            "history_complete": bool(history["history_complete"]),
            "input_role_transactions": input_count,
            "output_role_transactions": output_count,
            "high_risk_transactions": sum(
                int(item["predicted_label"] == 1) for item in transactions
            ),
            "maximum_risk_score": max(scores) if scores else 0.0,
            "average_risk_score": sum(scores) / len(scores) if scores else 0.0,
            "first_order": min(
                (int(item["sequence_order"]) for item in transactions), default=0
            ),
            "last_order": max(
                (int(item["sequence_order"]) for item in transactions), default=0
            ),
            "first_event_time": event_times[0] if event_times else None,
            "last_event_time": event_times[-1] if event_times else None,
            "confirmed_transaction_count": int(chain.get("tx_count") or 0),
            "mempool_transaction_count": int(mempool.get("tx_count") or 0),
            "balance_btc": (funded - spent) / SATOSHIS_PER_BTC,
            "total_received_btc": funded / SATOSHIS_PER_BTC,
            "relation_graph": relation_graph,
            "transactions": transactions,
        }
