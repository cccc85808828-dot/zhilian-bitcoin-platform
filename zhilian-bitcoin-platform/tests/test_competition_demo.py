from __future__ import annotations

import importlib.util
import json
import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT_ROOT / "scripts" / "run_competition_demo.py"
SPEC = importlib.util.spec_from_file_location("competition_demo", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
demo = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(demo)


class FakeResponse:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


class CompetitionDemoTests(unittest.TestCase):
    def test_public_case_uses_business_facing_fields(self) -> None:
        case = {
            "transaction_id": 12,
            "time_step": 42,
            "dataset_label": 1,
            "outcome": "true_positive",
            "predicted_label": 1,
            "risk_score": 0.98,
            "graph_score": 0.97,
            "qif_score": 0.92,
            "margin_to_threshold": 0.03,
            "input_addresses": {"count": 1, "items": []},
            "output_addresses": {"count": 2, "items": []},
            "attribute_profile": [],
        }
        visible = demo.public_case(case)
        self.assertEqual(visible["sequence_order"], 42)
        self.assertIsNone(visible["event_time"])
        self.assertNotIn("time_step", visible)
        self.assertNotIn("dataset_label", visible)
        self.assertNotIn("outcome", visible)

    def test_extract_responses_api_text(self) -> None:
        payload = {
            "output": [
                {
                    "type": "message",
                    "content": [
                        {"type": "output_text", "text": "证据约束回答"}
                    ],
                }
            ]
        }
        self.assertEqual(demo.extract_response_text(payload), "证据约束回答")

    def test_extract_chat_completions_text(self) -> None:
        payload = {"choices": [{"message": {"content": "DeepSeek研判回答"}}]}
        self.assertEqual(demo.extract_response_text(payload), "DeepSeek研判回答")

    def test_case_explanation_uses_specific_structure_attributes_and_margin(self) -> None:
        case = {
            "predicted_label": 1,
            "risk_score": 0.93,
            "graph_score": 0.91,
            "qif_score": 0.96,
            "decision_threshold_score": 0.88,
            "input_addresses": {"count": 4, "items": []},
            "output_addresses": {"count": 2, "items": []},
            "attribute_profile": [
                {
                    "name": "手续费（BTC）",
                    "raw_value": 0.0012,
                    "unit": "btc",
                    "signal_strength": 74.0,
                }
            ],
        }
        explanation = demo.build_case_explanation(case, 0.9)
        self.assertIn("多对多", explanation["structure"]["summary"])
        self.assertIn("手续费", explanation["attribute"]["summary"])
        self.assertIn("高于88.0分阈值5.0分", explanation["fusion"]["summary"])
        self.assertEqual(explanation["verdict"], "重点关注交易")

    def test_behavior_explanation_identifies_rapid_large_split_from_facts(self) -> None:
        case = {
            "predicted_label": 1,
            "input_addresses": {"count": 2, "items": []},
            "output_addresses": {"count": 5, "items": []},
            "amount_btc": 3.0,
            "attribute_profile": [
                {
                    "feature_index": 167,
                    "name": "交易总额（BTC）",
                    "raw_value": 3.0,
                    "standardized_value": 2.4,
                }
            ],
            "behavior_observations": {
                "output_amounts_btc": [0.8, 0.7, 0.6, 0.5, 0.4],
                "input_holding_seconds": [1800.0, 172800.0],
                "holding_time_covered_inputs": 2,
            },
        }
        behavior = demo.build_behavior_interpretation(case)
        labels = [item["label"] for item in behavior["patterns"]]
        self.assertIn("疑似快进快出", labels)
        self.assertIn("疑似大额拆分转移", labels)
        self.assertIn("最短间隔仅30分钟", behavior["summary"])
        self.assertIn("单个最大输出仅占26.7%", behavior["summary"])

    def test_behavior_explanation_does_not_invent_rapid_turnover_without_time(self) -> None:
        case = {
            "predicted_label": 1,
            "input_addresses": {"count": 6, "items": []},
            "output_addresses": {"count": 1, "items": []},
            "attribute_profile": [],
        }
        behavior = demo.build_behavior_interpretation(case)
        labels = [item["label"] for item in behavior["patterns"]]
        self.assertIn("多源资金归集", labels)
        self.assertNotIn("疑似快进快出", labels)

    def test_behavior_catalog_combines_mixing_micro_outputs_reentry_and_fee(self) -> None:
        case = {
            "predicted_label": 1,
            "input_addresses": {"count": 6, "items": []},
            "output_addresses": {"count": 7, "items": []},
            "attribute_profile": [
                {
                    "feature_index": 168,
                    "name": "手续费（BTC）",
                    "raw_value": 0.01,
                    "standardized_value": 2.5,
                }
            ],
            "behavior_observations": {
                "output_amounts_btc": [
                    0.5,
                    0.5,
                    0.5,
                    0.000001,
                    0.000001,
                    0.000001,
                    0.2,
                ],
                "overlap_address_count": 1,
                "fee_rate_sat_vb": 120.0,
                "fee_share_of_inputs": 0.03,
            },
        }
        labels = [
            item["label"] for item in demo.build_behavior_interpretation(case)["patterns"]
        ]
        self.assertIn("疑似协同混合形态", labels)
        self.assertIn("密集小额输出", labels)
        self.assertIn("输入输出地址回流复用", labels)
        self.assertIn("高费率加速确认倾向", labels)
        self.assertIn("异常手续费占比", labels)

    def test_behavior_catalog_detects_same_block_relay_and_peeling_shape(self) -> None:
        case = {
            "predicted_label": 1,
            "input_addresses": {"count": 1, "items": []},
            "output_addresses": {"count": 2, "items": []},
            "attribute_profile": [],
            "behavior_observations": {
                "input_holding_seconds": [0.0],
                "holding_time_covered_inputs": 1,
                "output_amounts_btc": [0.95, 0.05],
            },
        }
        labels = [
            item["label"] for item in demo.build_behavior_interpretation(case)["patterns"]
        ]
        self.assertIn("同区块接力转移", labels)
        self.assertIn("单主输出伴随小额剥离", labels)

    def test_assistant_config_uses_offline_mode_without_key(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            config = demo.assistant_config()
        self.assertEqual(config["mode"], "offline_knowledge")
        self.assertIsNone(config["api_key"])

    def test_llm_call_uses_server_side_chat_completions_endpoint(self) -> None:
        handler = object.__new__(demo.DemoHandler)
        handler.server = SimpleNamespace(
            assistant={
                "api_key": "server-side-test-key",
                "base_url": "https://example.invalid/v1",
                "model": "test-model",
            },
            payload={
                "fusion": {"threshold_score": 0.9},
                "method": "TTHGNN-QIF",
            },
        )
        response_payload = {
            "choices": [{"message": {"content": "模拟大模型解释"}}]
        }
        with patch(
            "urllib.request.urlopen",
            return_value=FakeResponse(response_payload),
        ) as mocked:
            answer = handler.call_llm(
                "继续解释技术原理",
                None,
                None,
                [{"role": "user", "content": "先解释结构分支"}],
            )
        self.assertEqual(answer, "模拟大模型解释")
        request = mocked.call_args.args[0]
        self.assertEqual(request.full_url, "https://example.invalid/v1/chat/completions")
        self.assertIn(b"test-model", request.data)
        self.assertIn("先解释结构分支".encode("utf-8"), request.data)
        self.assertNotIn(b"server-side-test-key", request.data)

    def test_load_env_file_does_not_override_deployment_environment(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as temp_dir:
            env_path = Path(temp_dir) / ".env"
            env_path.write_text(
                "DEEPSEEK_MODEL=deepseek-v4-flash\nPORT=9999\n",
                encoding="utf-8",
            )
            with patch.dict(os.environ, {"PORT": "8765"}, clear=True):
                demo.load_env_file(env_path)
                self.assertEqual(os.environ["PORT"], "8765")
                self.assertEqual(os.environ["DEEPSEEK_MODEL"], "deepseek-v4-flash")

    def test_api_rate_limit_is_per_client_and_returns_retry_window(self) -> None:
        server = object.__new__(demo.DemoServer)
        server.rate_limit_per_minute = 2
        server.rate_limit_window_seconds = 60.0
        server._rate_limit_lock = demo.threading.Lock()
        server._rate_limit_buckets = {}

        self.assertEqual(server.allow_api_request("198.51.100.10"), (True, 0))
        self.assertEqual(server.allow_api_request("198.51.100.10"), (True, 0))
        allowed, retry_after = server.allow_api_request("198.51.100.10")
        self.assertFalse(allowed)
        self.assertGreaterEqual(retry_after, 1)
        self.assertEqual(server.allow_api_request("198.51.100.11"), (True, 0))


if __name__ == "__main__":
    unittest.main()
