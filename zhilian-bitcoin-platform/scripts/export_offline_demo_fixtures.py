from __future__ import annotations

import argparse
import datetime as dt
import json
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "demo" / "data" / "offline_fixtures.json"

SAMPLES = (
    {
        "label": "近期多笔归集",
        "description": "包含多笔连续收转与一笔多输入归集交易",
        "address": "1DQgLuCsSvR2om2Z6Zk3u5ao6BvC9X1UCj",
    },
    {
        "label": "近期扇出网络",
        "description": "单输入连接多输出的扇出型交易结构",
        "address": "bc1q0n4tyt8vtre84nhy7yy2ya3awlc9a94shxu4sl",
    },
    {
        "label": "近期复杂换手",
        "description": "多输入、多输出共同形成的复杂换手结构",
        "address": "bc1qfj44gx27fgp00gv5f7htv2gwg89pprre9nqq5m",
    },
)


def get_json(base_url: str, path: str) -> Any:
    with urllib.request.urlopen(f"{base_url}{path}", timeout=120) as response:
        return json.loads(response.read().decode("utf-8"))


def post_json(base_url: str, path: str, payload: dict[str, Any]) -> Any:
    request = urllib.request.Request(
        f"{base_url}{path}",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.loads(response.read().decode("utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export fixed Bitcoin mainnet cases for the portable offline demo."
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8765")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_url = args.base_url.rstrip("/")
    health = get_json(base_url, "/api/health")
    summary = get_json(base_url, "/api/summary")

    profiles: dict[str, Any] = {}
    cases: dict[str, Any] = {}
    analyses: dict[str, Any] = {}
    confirmations: dict[tuple[int, str], dict[str, Any]] = {}

    for sample in SAMPLES:
        address = sample["address"]
        profile = get_json(
            base_url,
            f"/api/address?value={urllib.parse.quote(address, safe='')}",
        )
        profiles[address] = profile
        for transaction in profile.get("transactions", []):
            transaction_id = str(transaction["transaction_id"])
            if transaction_id in cases:
                continue
            case = get_json(
                base_url,
                f"/api/case/{urllib.parse.quote(transaction_id, safe='')}",
            )
            analysis = post_json(
                base_url,
                "/api/analyze",
                {"transaction_id": transaction_id},
            )
            cases[transaction_id] = case
            analyses[transaction_id] = analysis
            confirmation = case.get("confirmation") or {}
            block_height = confirmation.get("block_height")
            block_hash = confirmation.get("block_hash")
            if block_height is not None and block_hash:
                key = (int(block_height), str(block_hash))
                confirmations[key] = {
                    "block_height": key[0],
                    "block_hash": key[1],
                    "block_time": case.get("event_time"),
                }

    captured_at = dt.datetime.now(dt.timezone.utc).astimezone().isoformat(
        timespec="seconds"
    )
    sample_rows = []
    for sample in SAMPLES:
        profile = profiles[sample["address"]]
        graph = profile.get("relation_graph") or {}
        sample_rows.append(
            {
                **sample,
                "transaction_count": profile.get("related_transaction_count", 0),
                "high_risk_transactions": profile.get("high_risk_transactions", 0),
                "maximum_risk_score": profile.get("maximum_risk_score", 0.0),
                "node_count": graph.get("node_count", 0),
                "edge_count": graph.get("edge_count", 0),
                "first_event_time": profile.get("first_event_time"),
                "last_event_time": profile.get("last_event_time"),
            }
        )

    output = {
        "metadata": {
            "captured_at": captured_at,
            "network": "bitcoin_mainnet",
            "provider": "Blockstream Esplora",
            "provider_url": "https://blockstream.info/api",
            "data_mode": "offline_snapshot",
            "purpose": "portable offline demonstration",
            "evidence_scope": (
                "Public blockchain transaction structure only; no real-world identity "
                "or criminal attribution is inferred."
            ),
            "model": health.get("method", "TTHGNN-QIF"),
            "decision_threshold_score": (health.get("chain_data") or {}).get(
                "decision_threshold_score"
            ),
            "source_blocks": sorted(
                confirmations.values(), key=lambda item: item["block_height"]
            ),
        },
        "ui": {
            "offline_address_error": "离线演示包仅提供三个预置的近期真实主网地址，请点击示例地址进行查询。",
            "offline_address_hint": "如需查询其他地址，请使用具备外网访问条件的在线研判模式。",
            "assistant_ready": "本地研判知识库在线",
            "assistant_default": "我可以结合当前页面已经载入的地址、交易结构、链上属性和模型证据回答问题，也可以生成所选交易的可解释性研判报告。",
            "assistant_technical": (
                "智链底层TTHGNN-QIF由两个互补分支组成。关联网络行为研判分支将地址表示为节点，"
                "将多输入多输出交易表示为带方向和角色的超边，并以地址历史记忆描述跨时间行为。"
                "单笔交易属性研判分支对金额、手续费、交易大小和输入输出数量等属性进行分位编码，"
                "进一步学习属性之间的二阶交互。两条证据在对数几率空间完成融合，再映射为统一风险指数并形成分类结论。"
            ),
            "assistant_address_template": (
                "当前地址共关联{0}笔交易，其中{1}笔进入重点关注序列。关联网络包含{2}个节点和{3}条有向连接，"
                "已覆盖该地址作为转出方和接收方的历史关系。可以从交易列表中选择任意一笔，继续查看链上事实、行为特征和双分支研判依据。"
            ),
            "assistant_headings": {
                "object": "研判对象",
                "behavior": "行为原因",
                "structure": "关联网络行为研判",
                "attribute": "单笔交易属性研判",
                "fusion": "综合判定",
                "conclusion": "结论与建议",
            },
        },
        "samples": sample_rows,
        "profiles": profiles,
        "cases": cases,
        "analyses": analyses,
        "summary": summary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "output": str(args.output),
                "addresses": len(profiles),
                "transactions": len(cases),
                "blocks": len(confirmations),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
