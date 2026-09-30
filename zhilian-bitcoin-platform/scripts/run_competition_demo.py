from __future__ import annotations

import argparse
import datetime as dt
import json
import mimetypes
import os
import re
import statistics
import sys
import threading
import time
import urllib.parse
import urllib.error
import urllib.request
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List

SCRIPT_ROOT = Path(__file__).resolve().parent
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

BitcoinLiveService: Any = None


class ChainDataError(RuntimeError):
    status = 502
    hint = ""


class LiveInferenceError(RuntimeError):
    pass


TXID_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")


def load_live_dependencies() -> None:
    """Load the torch-backed live stack only when the server is started."""
    global BitcoinLiveService, ChainDataError, LiveInferenceError, TXID_PATTERN
    from bitcoin_live import (
        BitcoinLiveService as LiveService,
        ChainDataError as LiveChainDataError,
        LiveInferenceError as LiveModelError,
        TXID_PATTERN as LiveTxidPattern,
    )

    BitcoinLiveService = LiveService
    ChainDataError = LiveChainDataError
    LiveInferenceError = LiveModelError
    TXID_PATTERN = LiveTxidPattern


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEMO_ROOT = PROJECT_ROOT / "demo"
DATA_PATH = DEMO_ROOT / "data" / "demo_payload.json"


def bounded_env_int(name: str, default: int, minimum: int, maximum: int) -> int:
    """Read a deployment integer without allowing a malformed value to crash startup."""
    try:
        value = int(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        value = default
    return max(minimum, min(value, maximum))


def load_env_file(path: Path | None = None) -> None:
    """Load a local server-side .env without overriding deployed environment values."""
    env_path = path or (PROJECT_ROOT / ".env")
    if not env_path.is_file():
        return
    for raw_line in env_path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ.setdefault(key, value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the TTHGNN-QIF competition demo.")
    parser.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8765")))
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--check", action="store_true")
    return parser.parse_args()


def load_payload() -> Dict[str, Any]:
    if not DATA_PATH.is_file():
        raise FileNotFoundError(
            f"Demo data not found: {DATA_PATH}. Run scripts/export_competition_demo.py first."
        )
    payload = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    required = {"method", "protocol", "selected_seed_metrics", "cases"}
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"Demo payload is missing fields: {sorted(missing)}")
    if not payload["cases"]:
        raise ValueError("Demo payload contains no cases.")
    return payload


def public_summary(payload: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "method": payload["method"],
        "fusion": payload["fusion"],
        "five_seed_metrics": payload["five_seed_metrics"],
        "efficiency": payload.get("efficiency"),
    }


def public_case(case: Dict[str, Any]) -> Dict[str, Any]:
    visible_keys = (
        "transaction_id",
        "predicted_label",
        "risk_score",
        "graph_score",
        "qif_score",
        "margin_to_threshold",
        "input_addresses",
        "output_addresses",
        "attribute_profile",
        "confirmation",
        "amount_btc",
        "fee_btc",
        "size_bytes",
        "data_source",
        "analysis_mode",
        "decision_threshold_score",
        "risk_index_method",
    )
    result = {key: case.get(key) for key in visible_keys if key in case}
    result["event_time"] = case.get("event_time")
    result["sequence_order"] = int(
        case.get("sequence_order", case.get("time_step", 0))
    )
    return result


def _attribute_display_value(item: Dict[str, Any]) -> str:
    value = float(item.get("raw_value", item.get("value", 0.0)) or 0.0)
    unit = item.get("unit")
    if unit == "btc":
        return f"{value:.8f} BTC"
    if unit == "bytes":
        return f"{value:,.0f} 字节"
    if unit == "count":
        return f"{value:,.0f} 项"
    return f"{value:.6g}"


def _format_duration(seconds: float) -> str:
    value = max(float(seconds), 0.0)
    if value < 3600:
        return f"{max(1, round(value / 60))}分钟"
    if value < 86400:
        return f"{value / 3600:.1f}小时"
    return f"{value / 86400:.1f}天"


def build_behavior_interpretation(case: Dict[str, Any]) -> Dict[str, Any]:
    """Translate observable chain facts into cautious, transaction-level behaviors."""
    input_count = int((case.get("input_addresses") or {}).get("count", 0))
    output_count = int((case.get("output_addresses") or {}).get("count", 0))
    amount_btc = float(case.get("amount_btc", 0.0) or 0.0)
    observations = dict(case.get("behavior_observations") or {})
    attributes = list(case.get("attribute_profile") or [])
    by_index = {
        int(item["feature_index"]): item
        for item in attributes
        if item.get("feature_index") is not None
    }

    def raw_attribute(index: int, default: float = 0.0) -> float:
        item = by_index.get(index) or {}
        return float(item.get("raw_value", item.get("value", default)) or default)

    def standardized_attribute(index: int) -> float:
        item = by_index.get(index) or {}
        value = float(item.get("standardized_value", 0.0) or 0.0)
        return value if value == value and abs(value) != float("inf") else 0.0

    if amount_btc <= 0.0:
        amount_btc = raw_attribute(167)
    output_amounts = [
        float(value)
        for value in observations.get("output_amounts_btc", [])
        if float(value) > 0.0
    ]
    input_amounts = [
        float(value)
        for value in observations.get("input_amounts_btc", [])
        if float(value) > 0.0
    ]
    output_total = float(sum(output_amounts)) if output_amounts else raw_attribute(181)
    output_max = max(output_amounts) if output_amounts else raw_attribute(178)
    output_median = (
        float(statistics.median(output_amounts))
        if output_amounts
        else raw_attribute(180)
    )
    amount_deviation = standardized_attribute(167)
    fee_deviation = standardized_attribute(168)
    fee_rate = float(observations.get("fee_rate_sat_vb", 0.0) or 0.0)
    fee_share = float(observations.get("fee_share_of_inputs", 0.0) or 0.0)
    overlap_count = int(observations.get("overlap_address_count", 0) or 0)
    patterns: List[Dict[str, str]] = []

    def add(code: str, label: str, evidence: str, strength: str = "较强") -> None:
        patterns.append(
            {
                "code": code,
                "label": label,
                "evidence": evidence,
                "strength": strength,
            }
        )

    holding_seconds = sorted(
        float(value)
        for value in observations.get("input_holding_seconds", [])
        if float(value) >= 0.0
    )
    same_block = [value for value in holding_seconds if value == 0.0]
    rapid = [value for value in holding_seconds if 0.0 < value <= 86400.0]
    if same_block:
        covered = int(observations.get("holding_time_covered_inputs", len(holding_seconds)))
        add(
            "same_block_relay",
            "同区块接力转移",
            f"已核验的{covered}个输入资金来源中，有{len(same_block)}个在获得资金的同一区块"
            "即被再次支出，呈现极短链路接力",
            "强",
        )
    if rapid:
        covered = int(observations.get("holding_time_covered_inputs", len(holding_seconds)))
        add(
            "rapid_turnover",
            "疑似快进快出",
            f"已核验的{covered}个输入资金来源中，有{len(rapid)}个在获得后24小时内再次转出，"
            f"最短间隔仅{_format_duration(rapid[0])}",
            "强" if rapid[0] <= 21600.0 else "较强",
        )

    largest_share = output_max / output_total if output_total > 0.0 else 1.0
    median_share = output_median / output_total if output_total > 0.0 else 1.0
    repeated_counts: Dict[int, int] = {}
    for value in output_amounts:
        satoshis = int(round(value * 100_000_000))
        repeated_counts[satoshis] = repeated_counts.get(satoshis, 0) + 1
    equal_output_count = max(repeated_counts.values(), default=0)
    micro_output_count = sum(value <= 0.00001 for value in output_amounts)
    collaborative_mix = (
        input_count >= 5 and output_count >= 5 and equal_output_count >= 3
    )
    if collaborative_mix:
        equal_value_sats = max(repeated_counts, key=repeated_counts.get)
        add(
            "collaborative_mix",
            "疑似协同混合形态",
            f"交易同时包含{input_count}个输入和{output_count}个输出，其中"
            f"{equal_output_count}个输出金额同为{equal_value_sats / 100_000_000:.8f} BTC，"
            "呈多方输入与同额输出并存的混合特征",
            "强",
        )
    split_shape = (
        output_count >= 4
        and output_total > 0.0
        and largest_share <= 0.65
        and median_share <= 0.22
    )
    if split_shape:
        large_prefix = "大额" if amount_deviation >= 1.5 else "多笔"
        add(
            "split_transfer",
            f"疑似{large_prefix}拆分转移",
            f"总转出{output_total:.8f} BTC被分向{output_count}个输出，"
            f"单个最大输出仅占{largest_share * 100:.1f}%，输出中位数为"
            f"{output_median:.8f} BTC",
            "强" if output_count >= 6 or amount_deviation >= 2.0 else "较强",
        )
    elif output_count >= 5 and not collaborative_mix:
        add(
            "fan_out",
            "多地址分散转出",
            f"资金由{input_count}个输入分向{output_count}个输出地址，呈明显扇出扩散形态",
            "中等",
        )

    if equal_output_count >= 3 and not collaborative_mix:
        equal_value_sats = max(repeated_counts, key=repeated_counts.get)
        add(
            "equal_value_batch",
            "同额批量输出",
            f"共有{equal_output_count}个输出金额完全相同，单笔均为"
            f"{equal_value_sats / 100_000_000:.8f} BTC，呈规则化批量转出特征",
            "较强",
        )
    if micro_output_count >= 3:
        add(
            "dense_micro_outputs",
            "密集小额输出",
            f"{output_count}个输出中有{micro_output_count}个不超过0.00001000 BTC，"
            "形成密集微额分发形态",
            "中等",
        )

    peeling_shape = (
        2 <= output_count <= 3
        and output_total > 0.0
        and largest_share >= 0.85
        and (output_total - output_max) / output_total <= 0.15
    )
    if peeling_shape:
        add(
            "dominant_output_peeling",
            "单主输出伴随小额剥离",
            f"最大输出占总转出金额{largest_share * 100:.1f}%，其余资金以小额旁路输出，"
            "呈主额延续、少量剥离的链路形态",
            "中等",
        )

    if input_count >= 5 and output_count <= 2:
        input_total = sum(input_amounts)
        input_median = statistics.median(input_amounts) if input_amounts else 0.0
        fragmented = (
            input_count >= 10
            and input_total > 0.0
            and input_median / input_total <= 0.08
        )
        add(
            "fragmented_aggregation" if fragmented else "aggregation",
            "碎片化输入整合" if fragmented else "多源资金归集",
            f"来自{input_count}个输入地址的资金集中汇入{output_count}个输出地址，"
            + (
                f"输入金额中位数仅占输入总额的{input_median / input_total * 100:.1f}%，"
                "呈大量碎片资金整合形态"
                if fragmented
                else "呈多源归集形态"
            ),
        )
    elif input_count >= 3 and output_count >= 3 and not split_shape and not collaborative_mix:
        add(
            "many_to_many",
            "多源多向复杂换手",
            f"本笔交易同时连接{input_count}个输入与{output_count}个输出，"
            "资金来源和去向均较为分散",
            "中等",
        )

    if overlap_count > 0:
        add(
            "address_reentry",
            "输入输出地址回流复用",
            f"有{overlap_count}个地址同时出现在输入侧和输出侧，资金在同笔交易中回到"
            "已参与支出的地址集合",
            "中等",
        )

    if fee_rate >= 100.0 or (fee_rate >= 20.0 and fee_deviation >= 2.0):
        add(
            "high_fee_acceleration",
            "高费率加速确认倾向",
            f"本笔交易费率约为{fee_rate:.1f} sat/vB，且手续费处于模型参考分布高位，"
            "体现优先确认需求",
            "中等",
        )
    if fee_share >= 0.02:
        add(
            "high_fee_share",
            "异常手续费占比",
            f"手续费约占输入总额的{fee_share * 100:.2f}%，明显侵蚀实际转移金额",
            "较强" if fee_share >= 0.05 else "中等",
        )

    if amount_deviation >= 2.0 and not split_shape:
        add(
            "large_value",
            "相对大额转移",
            f"交易总额为{amount_btc:.8f} BTC，位于模型参考分布的明显高位"
            f"（高于参考中心约{amount_deviation:.1f}个标准差）",
        )

    predicted_high = bool(case.get("predicted_label", 0))
    if not patterns:
        if predicted_high:
            positive_attributes = sorted(
                (
                    item
                    for item in attributes
                    if float(item.get("standardized_value", 0.0) or 0.0) > 0.0
                ),
                key=lambda item: float(item.get("standardized_value", 0.0) or 0.0),
                reverse=True,
            )[:2]
            attribute_text = "、".join(
                f"{item.get('name', '交易属性')}处于参考高位"
                for item in positive_attributes
            )
            topology = f"{input_count}个输入连接{output_count}个输出"
            add(
                "compound_anomaly",
                "结构—属性复合异常",
                f"{topology}，且{attribute_text or '多项交易属性的组合关系偏离常见模式'}；"
                "单项事实不单独定性，但组合模式被模型识别为异常",
                "模型综合证据",
            )
        else:
            add(
                "routine_pattern",
                "未见典型异常行为组合",
                f"当前交易为{input_count}个输入连接{output_count}个输出，"
                "当前已核验的时间、金额分布、地址角色和手续费事实未触发典型行为规则",
                "常规",
            )

    labels = "、".join(pattern["label"] for pattern in patterns[:3])
    evidence = "；".join(pattern["evidence"] for pattern in patterns[:2])
    if predicted_high:
        summary = f"本笔交易的主要异常原因是：{labels}。{evidence}。"
        conclusion = (
            f"系统将该交易判定为重点关注，主要依据不是单一风险分数，而是"
            f"“{labels}”等可观测行为与时序结构、交易属性异常方向相互印证；"
            "综合风险指数用于确认这些证据合并后已达到重点研判标准。"
        )
    else:
        summary = f"当前行为解释为：{labels}。{evidence}。"
        conclusion = (
            f"系统将该交易列为常规关注：{labels}。模型仍保留结构与属性证据，"
            "综合风险指数用于确认现有证据组合尚未达到重点研判标准。"
        )
    return {
        "primary_label": patterns[0]["label"],
        "patterns": patterns,
        "summary": summary,
        "conclusion": conclusion,
    }


def build_case_explanation(
    case: Dict[str, Any], default_threshold: float
) -> Dict[str, Any]:
    """Build transaction-specific evidence text from model outputs and chain facts."""
    threshold = float(case.get("decision_threshold_score", default_threshold))
    risk_score = float(case.get("risk_score", 0.0))
    graph_score = float(case.get("graph_score", 0.0))
    qif_score = float(case.get("qif_score", 0.0))
    predicted_high = bool(case.get("predicted_label", risk_score >= threshold))
    input_count = int((case.get("input_addresses") or {}).get("count", 0))
    output_count = int((case.get("output_addresses") or {}).get("count", 0))

    if input_count > 1 and output_count > 1:
        topology = f"{input_count}个输入连接{output_count}个输出，呈多对多资金连接"
    elif input_count > 1:
        topology = f"{input_count}个输入汇入{output_count}个输出，呈资金归集形态"
    elif output_count > 1:
        topology = f"{input_count}个输入分向{output_count}个输出，呈资金分散形态"
    else:
        topology = f"{input_count}个输入连接{output_count}个输出，连接形态相对集中"

    graph_delta = graph_score - threshold
    graph_direction = "高于" if graph_delta >= 0 else "低于"
    graph_summary = (
        f"从关联网络行为看，本笔交易{topology}；该分支进一步结合输入/输出角色、"
        f"地址历史记忆和相邻连接，关联网络证据{graph_direction}研判阈值"
        f"{abs(graph_delta) * 100:.1f}分。"
    )

    attributes = list(case.get("attribute_profile") or [])
    attributes.sort(
        key=lambda item: abs(float(item.get("signal_strength", 0.0) or 0.0)),
        reverse=True,
    )
    top_attributes = [
        {
            "name": str(item.get("name", "链上属性")),
            "value": _attribute_display_value(item),
            "signal_strength": float(item.get("signal_strength", 0.0) or 0.0),
        }
        for item in attributes[:3]
    ]
    attribute_names = "、".join(
        f"{item['name']}（{item['value']}）" for item in top_attributes
    )
    if not attribute_names:
        attribute_names = "当前可观测交易属性"
    qif_delta = qif_score - threshold
    qif_direction = "高于" if qif_delta >= 0 else "低于"
    attribute_summary = (
        f"从单笔交易属性看，较有代表性的事实为{attribute_names}；该分支评估其分位位置及"
        f"二阶组合关系后，单笔属性证据{qif_direction}研判阈值"
        f"{abs(qif_delta) * 100:.1f}分。"
    )

    graph_high = graph_score >= threshold
    qif_high = qif_score >= threshold
    if graph_high and qif_high:
        agreement = "关联网络行为与单笔交易属性两条证据链方向一致，均支持重点关注"
    elif not graph_high and not qif_high:
        agreement = "关联网络行为与单笔交易属性两条证据链方向一致，均未越过重点关注阈值"
    else:
        stronger = "关联网络行为" if graph_score >= qif_score else "单笔交易属性"
        weaker = "单笔交易属性" if stronger == "关联网络行为" else "关联网络行为"
        agreement = f"两条证据链存在分歧，{stronger}分支信号更强，{weaker}分支形成制衡"
    margin = risk_score - threshold
    verdict = "重点关注交易" if predicted_high else "常规关注交易"
    position = "高于" if margin >= 0 else "低于"
    fusion_summary = (
        f"{agreement}。融合校准后的综合风险指数为{risk_score * 100:.1f}分，"
        f"{position}{threshold * 100:.1f}分阈值{abs(margin) * 100:.1f}分，因此判定为"
        f"“{verdict}”。"
    )
    behavior = build_behavior_interpretation(case)
    conclusion = behavior["conclusion"]
    return {
        "verdict": verdict,
        "threshold_score": threshold,
        "margin_score": margin,
        "behavior": behavior,
        "structure": {
            "score": graph_score,
            "topology": topology,
            "summary": graph_summary,
        },
        "attribute": {
            "score": qif_score,
            "top_features": top_attributes,
            "summary": attribute_summary,
        },
        "fusion": {
            "score": risk_score,
            "agreement": agreement,
            "summary": fusion_summary,
        },
        "conclusion": conclusion,
    }


def assistant_config() -> Dict[str, Any]:
    api_key = (
        os.environ.get("DEEPSEEK_API_KEY")
        or os.environ.get("LLM_API_KEY")
    )
    base_url = (
        os.environ.get("DEEPSEEK_BASE_URL")
        or os.environ.get("LLM_BASE_URL")
        or "https://api.deepseek.com"
    ).rstrip("/")
    model = (
        os.environ.get("DEEPSEEK_MODEL")
        or os.environ.get("LLM_MODEL")
        or "deepseek-v4-flash"
    )
    provider = (
        os.environ.get("LLM_PROVIDER")
        or ("DeepSeek" if "deepseek" in base_url.lower() else "兼容大模型")
    )
    return {
        "api_key": api_key,
        "base_url": base_url,
        "model": model,
        "provider": provider,
        "mode": "llm_api" if api_key else "offline_knowledge",
    }


def extract_response_text(payload: Dict[str, Any]) -> str:
    choices = payload.get("choices") or []
    if choices:
        message = choices[0].get("message") or {}
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return content.strip()
    direct = payload.get("output_text")
    if isinstance(direct, str) and direct.strip():
        return direct.strip()
    texts: List[str] = []
    for item in payload.get("output", []):
        if item.get("type") != "message":
            continue
        for content in item.get("content", []):
            if content.get("type") == "output_text" and content.get("text"):
                texts.append(str(content["text"]))
    return "\n".join(texts).strip()


class DemoServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, server_address: tuple[str, int], payload: Dict[str, Any]):
        super().__init__(server_address, DemoHandler)
        self.max_body_bytes = bounded_env_int(
            "DEMO_MAX_BODY_BYTES", 16_384, 1_024, 1_048_576
        )
        self.rate_limit_per_minute = bounded_env_int(
            "DEMO_RATE_LIMIT_PER_MINUTE", 120, 0, 10_000
        )
        self.rate_limit_window_seconds = 60.0
        self._rate_limit_lock = threading.Lock()
        self._rate_limit_buckets: Dict[str, tuple[float, int]] = {}
        self.payload = payload
        self.case_by_id = {
            str(case["transaction_id"]): case for case in payload["cases"]
        }
        self.address_index: Dict[str, Dict[str, Any]] = {}
        exported_address_index = payload.get("address_index")
        if exported_address_index:
            for address, exported in exported_address_index.items():
                transactions = {}
                for transaction_id, role_bits in exported["links"]:
                    case = self.case_by_id.get(str(transaction_id))
                    if case is None:
                        continue
                    roles = []
                    if int(role_bits) & 1:
                        roles.append("input")
                    if int(role_bits) & 2:
                        roles.append("output")
                    transactions[int(transaction_id)] = {
                        "transaction_id": int(transaction_id),
                        "time_step": int(case["time_step"]),
                        "event_time": case.get("event_time"),
                        "risk_score": float(case["risk_score"]),
                        "predicted_label": int(case["predicted_label"]),
                        "roles": roles,
                    }
                if transactions:
                    self.address_index[str(address)] = {
                        "address": str(address),
                        "global_address_id": int(exported["global_address_id"]),
                        "transactions": transactions,
                    }
        else:
            for case in payload["cases"]:
                transaction_id = int(case["transaction_id"])
                for field, role in (
                    ("input_addresses", "input"),
                    ("output_addresses", "output"),
                ):
                    for address_item in case[field]["items"]:
                        address = str(address_item["address"])
                        entry = self.address_index.setdefault(
                            address,
                            {
                                "address": address,
                                "global_address_id": int(address_item["global_id"]),
                                "transactions": {},
                            },
                        )
                        transaction = entry["transactions"].setdefault(
                            transaction_id,
                            {
                                "transaction_id": transaction_id,
                                "time_step": int(case["time_step"]),
                                "event_time": case.get("event_time"),
                                "risk_score": float(case["risk_score"]),
                                "predicted_label": int(case["predicted_label"]),
                                "roles": [],
                            },
                        )
                        if role not in transaction["roles"]:
                            transaction["roles"].append(role)
        self.assistant = assistant_config()
        self.live = BitcoinLiveService(payload)

    def allow_api_request(self, client_ip: str) -> tuple[bool, int]:
        """Apply a small per-client API limit before public deployment.

        The limit is intentionally disabled when set to 0 so local development
        and offline smoke tests can opt out without changing application code.
        """
        if self.rate_limit_per_minute <= 0:
            return True, 0
        now = time.monotonic()
        with self._rate_limit_lock:
            started, count = self._rate_limit_buckets.get(client_ip, (now, 0))
            if now - started >= self.rate_limit_window_seconds:
                started, count = now, 0
            if count >= self.rate_limit_per_minute:
                retry_after = max(
                    1, int(self.rate_limit_window_seconds - (now - started))
                )
                self._rate_limit_buckets[client_ip] = (started, count)
                return False, retry_after
            self._rate_limit_buckets[client_ip] = (started, count + 1)
            if len(self._rate_limit_buckets) > 4096:
                self._rate_limit_buckets = {
                    key: value
                    for key, value in self._rate_limit_buckets.items()
                    if now - value[0] < self.rate_limit_window_seconds
                }
        return True, 0

    def case(self, transaction_id: str) -> Dict[str, Any] | None:
        live_case = self.live.case(transaction_id)
        if live_case is not None:
            return live_case
        return self.case_by_id.get(str(transaction_id))

    def cached_address_profile(self, address: str) -> Dict[str, Any] | None:
        live_profile = self.live.cached_profile(address)
        if live_profile is not None:
            return live_profile
        return self.address_profile(address)

    def address_profile(self, address: str) -> Dict[str, Any] | None:
        entry = self.address_index.get(address)
        if entry is None:
            return None
        transactions = list(entry["transactions"].values())
        transactions.sort(key=lambda item: item["risk_score"], reverse=True)
        scores = [float(item["risk_score"]) for item in transactions]
        input_count = sum("input" in item["roles"] for item in transactions)
        output_count = sum("output" in item["roles"] for item in transactions)
        return {
            "address": entry["address"],
            "global_address_id": entry["global_address_id"],
            "related_transaction_count": len(transactions),
            "input_role_transactions": input_count,
            "output_role_transactions": output_count,
            "high_risk_transactions": sum(
                int(item["predicted_label"] == 1) for item in transactions
            ),
            "maximum_risk_score": max(scores),
            "average_risk_score": sum(scores) / len(scores),
            "first_order": min(item["time_step"] for item in transactions),
            "last_order": max(item["time_step"] for item in transactions),
            "transactions": [
                {
                    "transaction_id": item["transaction_id"],
                    "event_time": item.get("event_time"),
                    "sequence_order": item["time_step"],
                    "risk_score": item["risk_score"],
                    "predicted_label": item["predicted_label"],
                    "roles": item["roles"],
                }
                for item in transactions
            ],
        }


class DemoHandler(BaseHTTPRequestHandler):
    server: DemoServer

    def log_message(self, format: str, *args: object) -> None:
        # Do not write addresses, transaction IDs, or query strings to public logs.
        path = urllib.parse.urlparse(self.path).path
        status = args[1] if len(args) > 1 else "?"
        print(f"[demo] {self.address_string()} {self.command} {path} {status}")

    def end_headers(self) -> None:  # noqa: N802
        """Add browser-side hardening to every response, including errors."""
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header(
            "Permissions-Policy",
            "camera=(), microphone=(), geolocation=(), payment=()",
        )
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data:; connect-src 'self'; base-uri 'self'; "
            "form-action 'self'; frame-ancestors 'none'",
        )
        super().end_headers()

    def send_json(self, payload: Any, status: int = 200) -> None:
        encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(encoded)

    def send_static(self, relative_path: str) -> None:
        allowed = {
            "index.html": DEMO_ROOT / "index.html",
            "relations.html": DEMO_ROOT / "relations.html",
            "analysis.html": DEMO_ROOT / "analysis.html",
            "styles.css": DEMO_ROOT / "styles.css",
            "common.js": DEMO_ROOT / "common.js",
            "address.js": DEMO_ROOT / "address.js",
            "relations.js": DEMO_ROOT / "relations.js",
            "analysis.js": DEMO_ROOT / "analysis.js",
        }
        path = allowed.get(relative_path)
        if path is None or not path.is_file():
            self.send_error(404)
            return
        content = path.read_bytes()
        content_type, _ = mimetypes.guess_type(path.name)
        self.send_response(200)
        self.send_header("Content-Type", f"{content_type or 'application/octet-stream'}; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(content)

    def reject_untrusted_api_request(self, parsed: urllib.parse.ParseResult) -> bool:
        """Reject oversized URLs and excessive API traffic before dispatch."""
        if len(self.path) > 4096:
            self.send_json({"error": "请求地址过长"}, status=414)
            return True
        if not parsed.path.startswith("/api/") or parsed.path == "/api/health":
            return False
        allowed, retry_after = self.server.allow_api_request(self.address_string())
        if allowed:
            return False
        self.send_response(429)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Retry-After", str(retry_after))
        payload = json.dumps(
            {"error": "请求过于频繁，请稍后再试", "retry_after": retry_after},
            ensure_ascii=False,
        ).encode("utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)
        return True

    def filtered_cases(self, query: Dict[str, List[str]]) -> List[Dict[str, Any]]:
        cases = self.server.payload["cases"]
        text = query.get("query", [""])[0].strip().lower()
        prediction = query.get("prediction", ["all"])[0]
        truth = query.get("truth", ["all"])[0]
        outcome = query.get("outcome", ["all"])[0]
        time_step = query.get("time", ["all"])[0]
        sort = query.get("sort", ["risk_desc"])[0]
        try:
            limit = min(max(int(query.get("limit", ["200"])[0]), 1), 1000)
        except ValueError:
            limit = 200

        filtered = []
        for case in cases:
            if text and text not in str(case["transaction_id"]).lower():
                continue
            if prediction != "all" and case["predicted_label"] != int(prediction):
                continue
            if truth != "all" and case["dataset_label"] != int(truth):
                continue
            if outcome != "all" and case["outcome"] != outcome:
                continue
            if time_step != "all" and case["time_step"] != int(time_step):
                continue
            filtered.append(case)
        if sort == "risk_asc":
            filtered.sort(key=lambda item: item["risk_score"])
        elif sort == "boundary":
            threshold = float(self.server.payload["fusion"]["threshold_score"])
            filtered.sort(key=lambda item: abs(item["risk_score"] - threshold))
        elif sort == "time_asc":
            filtered.sort(key=lambda item: (item["time_step"], -item["risk_score"]))
        else:
            filtered.sort(key=lambda item: item["risk_score"], reverse=True)
        compact_keys = (
            "transaction_id",
            "time_step",
            "predicted_label",
            "risk_score",
            "graph_score",
            "qif_score",
        )
        return [
            {
                **{key: case[key] for key in compact_keys if key != "time_step"},
                "event_time": case.get("event_time"),
                "sequence_order": int(case["time_step"]),
            }
            for case in filtered[:limit]
        ]

    def read_json_body(self) -> Dict[str, Any]:
        content_type = self.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
        if content_type != "application/json":
            raise ValueError("请求必须使用 application/json")
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError as error:
            raise ValueError("请求长度无效") from error
        if content_length <= 0 or content_length > self.server.max_body_bytes:
            raise ValueError("请求内容为空或过大")
        try:
            return json.loads(self.rfile.read(content_length).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("请求不是有效的JSON") from error

    def sample_cases(self) -> List[Dict[str, Any]]:
        cases = self.server.payload["cases"]
        threshold = float(self.server.payload["fusion"]["threshold_score"])
        high = max(cases, key=lambda item: item["risk_score"])
        low = min(cases, key=lambda item: item["risk_score"])
        boundary = min(
            cases, key=lambda item: abs(item["risk_score"] - threshold)
        )
        return [
            {
                "label": "高风险示例",
                "transaction_id": high["transaction_id"],
                "risk_score": high["risk_score"],
            },
            {
                "label": "临界示例",
                "transaction_id": boundary["transaction_id"],
                "risk_score": boundary["risk_score"],
            },
            {
                "label": "低风险示例",
                "transaction_id": low["transaction_id"],
                "risk_score": low["risk_score"],
            },
        ]

    def sample_addresses(self) -> List[Dict[str, Any]]:
        # These are public mainnet addresses already present in the project
        # corpus and have compact histories, so a first-time live demo remains
        # fast while still exercising the real Esplora query path.
        return [
            {
                "label": "简洁链路示例",
                "address": "16YzPMDdCtEM1HDU11b2WwNfMutwu9JgeE",
            },
            {
                "label": "重点关注示例",
                "address": "1LT1dSUS9t777NDZCFhFJ5x7VZVFPbjeCT",
            },
            {
                "label": "多笔历史示例",
                "address": "1ArrU5nQ2VhRBLVqJADicZHKHJVYTTkQP5",
            },
        ]

    def offline_assistant(
        self,
        question: str,
        case: Dict[str, Any] | None,
        address_profile: Dict[str, Any] | None,
        history: List[Dict[str, str]] | None = None,
    ) -> str:
        normalized = question.lower()
        explanation = (
            build_case_explanation(
                case, self.server.payload["fusion"]["threshold_score"]
            )
            if case is not None
            else None
        )
        if case is not None and any(
            token in normalized
            for token in ("报告", "解释", "为什么", "结论", "风险", "异常", "证据")
        ):
            chain_evidence = ""
            if case.get("data_source") == "bitcoin_mainnet":
                confirmation = case.get("confirmation") or {}
                status = (
                    f"已在区块 {confirmation.get('block_height')} 确认"
                    if confirmation.get("confirmed")
                    else "当前处于内存池等待确认"
                )
                chain_evidence = (
                    f"链上事实：交易金额{case.get('amount_btc', 0):.8f} BTC，"
                    f"手续费{case.get('fee_btc', 0):.8f} BTC，{status}。"
                )
            return (
                f"研判对象\n交易 {case['transaction_id']}\n\n"
                f"行为原因\n{explanation['behavior']['summary']}\n\n"
                f"结构判断\n{explanation['structure']['summary']}\n\n"
                f"属性判断\n{explanation['attribute']['summary']}\n\n"
                f"综合判定\n{explanation['fusion']['summary']}\n\n"
                f"结论与建议\n{explanation['conclusion']}\n\n{chain_evidence}"
            )
        if any(token in normalized for token in ("原理", "技术", "qif", "超图")):
            method_answer = (
                "智链底层TTHGNN-QIF由两个互补分支组成：关联网络行为研判分支把地址建模为节点、"
                "把交易建模为带输入/输出方向的超边，并用地址历史记忆表示跨时间"
                "行为；单笔交易属性研判分支对金额、手续费、交易大小、输入输出数量"
                "等可观测属性进行分位编码，并学习属性之间的二阶交互。"
                "最后在对数几率空间融合两条证据链，再用统一风险标尺和阈值形成结论。"
            )
            if explanation is not None:
                method_answer += (
                    f"\n\n结合当前交易：{explanation['structure']['summary']}"
                    f"{explanation['attribute']['summary']}{explanation['fusion']['summary']}"
                )
            return method_answer
        if address_profile is not None:
            graph = address_profile.get("relation_graph") or {}
            return (
                f"地址 {address_profile['address']} 共关联"
                f"{address_profile['related_transaction_count']}笔交易，其中模型判为"
                f"重点关注交易有{address_profile['high_risk_transactions']}笔。关联图当前包含"
                f"{graph.get('node_count', 0)}个节点和{graph.get('edge_count', 0)}条连接。"
                "你可以放大、拖拽或搜索地址节点，并选择任意交易查看结构、属性与融合证据。"
            )
        return (
            "我可以解释当前地址的关联交易、生成所选交易的可解释性报告，或回答"
            "智链的关联网络行为研判、单笔交易属性研判及综合判定原理等技术问题。"
        )

    def call_llm(
        self,
        question: str,
        case: Dict[str, Any] | None,
        address_profile: Dict[str, Any] | None,
        history: List[Dict[str, str]] | None = None,
    ) -> str:
        config = self.server.assistant
        compact_profile = None
        if address_profile is not None:
            relation_graph = address_profile.get("relation_graph") or {}
            compact_profile = {
                key: value
                for key, value in address_profile.items()
                if key not in {"transactions", "relation_graph"}
            }
            compact_profile["highest_risk_transactions"] = list(
                address_profile.get("transactions") or []
            )[:10]
            compact_profile["relation_graph_summary"] = {
                key: relation_graph.get(key)
                for key in ("node_count", "edge_count", "address_node_count", "transaction_node_count")
            }
        explanation = (
            build_case_explanation(
                case, self.server.payload["fusion"]["threshold_score"]
            )
            if case is not None
            else None
        )
        evidence = {
            "address_context": compact_profile,
            "transaction_context": public_case(case) if case is not None else None,
            "decision_explanation": explanation,
            "decision_threshold": (
                case.get("decision_threshold_score")
                if case is not None and case.get("decision_threshold_score") is not None
                else self.server.payload["fusion"]["threshold_score"]
            ),
            "risk_index_method": self.server.payload.get("risk_calibration", {}).get(
                "display_index_method"
            ),
            "method": self.server.payload["method"],
        }
        instructions = (
            "你是智链比特币异常交易智能检测平台的研判助手，底层方法为TTHGNN-QIF。只依据提供的模型输出和"
            "地址/交易证据回答，不得编造链上事实或身份。回答使用积极、专业、"
            "面向业务的表述，突出系统已经完成的分析和得到的证据。"
            "risk_score表示基于验证参照分位校准的0到1风险指数，回答时换算为"
            "0到100分，不要将其称为犯罪概率。"
            "研判原理必须讲清楚：关联网络行为研判分支将地址作为节点、交易作为带方向的超边，"
            "结合输入输出角色、连接形态和地址历史记忆形成关系判断；单笔交易属性研判分支对金额、"
            "手续费、交易大小、输入输出数量等公开属性进行分位编码和二阶交互，输出"
            "属性证据；两条证据在对数几率空间融合，并经统一风险标尺和阈值形成结论。"
            "解释当前交易时必须先回答具体呈现了什么行为：例如有时间间隔证据时说明"
            "疑似快进快出或同区块接力，有输出金额分布证据时说明拆分转移、扇出扩散、"
            "同额批量输出、密集小额输出、协同混合形态或单主输出剥离；还可识别多源"
            "归集、碎片化输入整合、多源多向换手、地址回流复用、相对大额、高费率加速"
            "和异常手续费占比。允许一笔交易同时具有多个行为标签。不得仅凭风险分数"
            "发明行为标签；没有对应"
            "事实时不得声称快进快出或拆分。随后再说明连接形态、具体高位属性、两分支"
            "是否一致以及综合结果。风险指数和阈值只能作为证据合并后的补充，不得作为"
            "异常原因本身。用户要求报告时，按研判对象、链上事实、行为原因、结构证据、"
            "属性证据、融合结论和后续建议组织。使用简洁中文。"
        )
        conversation = []
        for message in (history or [])[-6:]:
            role = str(message.get("role", ""))
            content = str(message.get("content", "")).strip()
            if role in {"user", "assistant"} and content:
                conversation.append({"role": role, "content": content[:1200]})
        messages = [{"role": "system", "content": instructions}]
        messages.extend(conversation)
        messages.append(
            {
                "role": "user",
                "content": (
                    f"证据上下文：{json.dumps(evidence, ensure_ascii=False)}\n\n"
                    f"用户问题：{question}"
                ),
            }
        )
        request_payload = {
            "model": config["model"],
            "messages": messages,
            "max_tokens": 1200,
            "temperature": 0.2,
            "stream": False,
        }
        request = urllib.request.Request(
            f"{config['base_url']}/chat/completions",
            data=json.dumps(request_payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {config['api_key']}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=45) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")[:500]
            raise RuntimeError(f"大模型接口返回HTTP {error.code}: {detail}") from error
        except urllib.error.URLError as error:
            raise RuntimeError(f"无法连接大模型接口: {error.reason}") from error
        text = extract_response_text(payload)
        if not text:
            raise RuntimeError("大模型接口未返回可显示文本")
        return text

    def do_GET(self) -> None:  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        if self.reject_untrusted_api_request(parsed):
            return
        if parsed.path == "/api/summary":
            self.send_json(public_summary(self.server.payload))
            return
        if parsed.path == "/api/health":
            self.send_json(
                {
                    "status": "ok",
                    "method": self.server.payload["method"],
                    "transactions": len(self.server.case_by_id),
                    "indexed_addresses": len(self.server.address_index),
                    "assistant_mode": self.server.assistant["mode"],
                    "security": {
                        "rate_limit_per_minute": self.server.rate_limit_per_minute,
                        "max_body_bytes": self.server.max_body_bytes,
                        "access_log_query_redaction": True,
                    },
                    "chain_data": self.server.live.status(),
                }
            )
            return
        if parsed.path == "/api/cases":
            query = urllib.parse.parse_qs(parsed.query)
            self.send_json(self.filtered_cases(query))
            return
        if parsed.path == "/api/samples":
            self.send_json(self.sample_cases())
            return
        if parsed.path == "/api/address-samples":
            self.send_json(self.sample_addresses())
            return
        if parsed.path == "/api/assistant/status":
            config = self.server.assistant
            self.send_json(
                {
                    "mode": config["mode"],
                    "model": config["model"] if config["api_key"] else None,
                    "configured": bool(config["api_key"]),
                }
            )
            return
        if parsed.path == "/api/address":
            query = urllib.parse.parse_qs(parsed.query)
            address = query.get("value", [""])[0].strip()
            try:
                profile = self.server.live.address_profile(address)
            except ChainDataError as error:
                self.send_json(
                    {
                        "error": str(error),
                        "hint": error.hint,
                    },
                    status=error.status,
                )
                return
            except LiveInferenceError as error:
                self.send_json(
                    {
                        "error": "链上交易已读取，但实时模型未能完成研判",
                        "hint": str(error),
                    },
                    status=503,
                )
                return
            self.send_json(profile)
            return
        if parsed.path.startswith("/api/case/"):
            transaction_id = urllib.parse.unquote(parsed.path.split("/")[-1])
            is_live_transaction = bool(TXID_PATTERN.fullmatch(transaction_id))
            case = None if is_live_transaction else self.server.case(transaction_id)
            if is_live_transaction:
                try:
                    case = self.server.live.analyze_transaction(transaction_id)
                except (ChainDataError, LiveInferenceError) as error:
                    status = error.status if isinstance(error, ChainDataError) else 503
                    self.send_json({"error": str(error)}, status=status)
                    return
            if case is None:
                self.send_json({"error": "未找到该交易编号"}, status=404)
            else:
                self.send_json(public_case(case))
            return
        if parsed.path in {"/", "/index.html"}:
            self.send_static("index.html")
            return
        self.send_static(parsed.path.lstrip("/"))

    def do_POST(self) -> None:  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        if self.reject_untrusted_api_request(parsed):
            return
        if parsed.path not in {"/api/analyze", "/api/assistant"}:
            self.send_json({"error": "接口不存在"}, status=404)
            return
        try:
            body = self.read_json_body()
        except ValueError as error:
            self.send_json({"error": str(error)}, status=400)
            return
        if parsed.path == "/api/assistant":
            question = str(body.get("question", "")).strip()
            if not question or len(question) > 2000:
                self.send_json({"error": "问题为空或超过2000字"}, status=400)
                return
            raw_history = body.get("history") or []
            history: List[Dict[str, str]] = []
            if isinstance(raw_history, list):
                for message in raw_history[-6:]:
                    if not isinstance(message, dict):
                        continue
                    role = str(message.get("role", ""))
                    content = str(message.get("content", "")).strip()
                    if role in {"user", "assistant"} and content:
                        history.append({"role": role, "content": content[:1200]})
            transaction_id = str(body.get("transaction_id", "")).strip()
            address = str(body.get("address", "")).strip()
            case = self.server.case(transaction_id) if transaction_id else None
            profile = (
                self.server.cached_address_profile(address) if address else None
            )
            mode = self.server.assistant["mode"]
            try:
                if mode == "llm_api":
                    answer = self.call_llm(question, case, profile, history)
                else:
                    answer = self.offline_assistant(question, case, profile, history)
            except RuntimeError as error:
                self.send_json(
                    {
                        "error": str(error),
                        "fallback": self.offline_assistant(
                            question, case, profile, history
                        ),
                    },
                    status=502,
                )
                return
            self.send_json(
                {
                    "status": "PASS",
                    "mode": mode,
                    "model": (
                        self.server.assistant["model"] if mode == "llm_api" else None
                    ),
                    "answer": answer,
                }
            )
            return

        transaction_id = str(body.get("transaction_id", "")).strip()
        is_live_transaction = bool(TXID_PATTERN.fullmatch(transaction_id))
        if not transaction_id.isdigit() and not is_live_transaction:
            self.send_json(
                {"error": "请输入有效的交易编号或64位比特币交易哈希"}, status=400
            )
            return
        case = None if is_live_transaction else self.server.case(transaction_id)
        if is_live_transaction:
            try:
                case = self.server.live.analyze_transaction(transaction_id)
            except ChainDataError as error:
                self.send_json(
                    {"error": str(error), "hint": error.hint}, status=error.status
                )
                return
            except LiveInferenceError as error:
                self.send_json({"error": str(error)}, status=503)
                return
        if case is None:
            self.send_json(
                {
                    "error": "当前历史交易库中未找到该编号",
                    "hint": "请返回关联交易页面重新选择交易。",
                },
                status=404,
            )
            return
        now = dt.datetime.now(dt.timezone.utc).astimezone()
        result = public_case(case)
        result["decision_explanation"] = build_case_explanation(
            case, self.server.payload["fusion"]["threshold_score"]
        )
        self.send_json(
            {
                "status": "PASS",
                "analysis_id": f"TQ-{transaction_id}-{now.strftime('%H%M%S')}",
                "processed_at": now.isoformat(timespec="seconds"),
                "data_source": (
                    "比特币主网链上数据"
                    if case.get("data_source") == "bitcoin_mainnet"
                    else "交易情报服务"
                ),
                "stages": [
                    {"id": "lookup", "name": "链上交易读取"},
                    {"id": "context", "name": "地址角色与历史上下文构建"},
                    {"id": "hypergraph", "name": "时序超图结构分析"},
                    {"id": "qif", "name": "QIF属性交互分析"},
                    {"id": "fusion", "name": "风险融合与阈值判定"},
                ],
                "result": result,
            }
        )


def main() -> None:
    load_env_file()
    args = parse_args()
    payload = load_payload()
    if args.check:
        print(
            json.dumps(
                {
                    "status": "PASS",
                    "method": payload["method"],
                    "cases": len(payload["cases"]),
                    "assets": {
                        name: (DEMO_ROOT / name).is_file()
                        for name in (
                            "index.html",
                            "relations.html",
                            "analysis.html",
                            "styles.css",
                            "common.js",
                            "address.js",
                            "relations.js",
                            "analysis.js",
                        )
                    },
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    load_live_dependencies()
    server = DemoServer((args.host, int(args.port)), payload)
    url = f"http://{args.host}:{args.port}"
    print(f"TTHGNN-QIF competition demo: {url}")
    print("Press Ctrl+C to stop.")
    if not args.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
