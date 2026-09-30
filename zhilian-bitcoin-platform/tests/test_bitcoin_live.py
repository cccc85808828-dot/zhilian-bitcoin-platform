from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from bitcoin_live import (  # noqa: E402
    BitcoinLiveService,
    ChainDataError,
    EsploraClient,
    transaction_behavior_observations,
    transaction_parties,
)
from tthgnn_erl.scatter import scatter_add, scatter_mean  # noqa: E402


class BitcoinLiveTests(unittest.TestCase):
    def test_behavior_observations_use_parent_block_time(self) -> None:
        parent = {
            "txid": "a" * 64,
            "status": {"confirmed": True, "block_time": 1_000},
        }
        transaction = {
            "txid": "b" * 64,
            "status": {"confirmed": True, "block_time": 2_800},
            "vin": [
                {
                    "txid": parent["txid"],
                    "vout": 0,
                    "prevout": {"value": 300_000_000},
                }
            ],
            "vout": [
                {"value": 80_000_000},
                {"value": 70_000_000},
                {"value": 60_000_000},
                {"value": 50_000_000},
                {"value": 40_000_000},
            ],
        }
        observed = transaction_behavior_observations(
            transaction, {parent["txid"]: parent}, parent_context_checked=True
        )
        self.assertEqual(observed["input_holding_seconds"], [1800.0])
        self.assertEqual(len(observed["output_amounts_btc"]), 5)
        self.assertTrue(observed["parent_context_checked"])

    def test_transaction_parties_aggregates_repeated_addresses(self) -> None:
        transaction = {
            "vin": [
                {
                    "txid": "a" * 64,
                    "prevout": {
                        "scriptpubkey_address": "1InputAddress",
                        "value": 100_000_000,
                    },
                },
                {
                    "txid": "b" * 64,
                    "prevout": {
                        "scriptpubkey_address": "1InputAddress",
                        "value": 50_000_000,
                    },
                },
            ],
            "vout": [
                {"scriptpubkey_address": "bc1output", "value": 149_000_000}
            ],
        }
        parties = transaction_parties(transaction)
        self.assertEqual(parties["input"]["count"], 1)
        self.assertAlmostEqual(parties["input"]["items"][0]["amount_btc"], 1.5)
        self.assertEqual(parties["output"]["count"], 1)
        self.assertAlmostEqual(parties["output"]["items"][0]["amount_btc"], 1.49)

    def test_address_history_is_paginated_deduplicated_and_chronological(self) -> None:
        client = EsploraClient()
        client.max_transactions = 3
        client.maximum_pages = 2
        responses = {
            "/address/1ValidBitcoinAddress/address-never-used": None,
            "/address/1ValidBitcoinAddress": {
                "chain_stats": {"tx_count": 3},
                "mempool_stats": {"tx_count": 0},
            },
            "/address/1ValidBitcoinAddress/txs/mempool": [],
            "/address/1ValidBitcoinAddress/txs/chain": [
                {
                    "txid": "b" * 64,
                    "status": {"confirmed": True, "block_time": 200},
                },
                {
                    "txid": "a" * 64,
                    "status": {"confirmed": True, "block_time": 100},
                },
                {
                    "txid": "c" * 64,
                    "status": {"confirmed": True, "block_time": 300},
                },
            ],
        }

        def fake_get(path: str, **_: object):
            return responses[path]

        with patch.object(client, "_get_json", side_effect=fake_get):
            history = client.address_history("1ValidBitcoinAddress")
        self.assertTrue(history["history_complete"])
        self.assertEqual(
            [item["txid"] for item in history["transactions"]],
            ["a" * 64, "b" * 64, "c" * 64],
        )

    def test_invalid_address_is_rejected_before_network_request(self) -> None:
        client = EsploraClient()
        with self.assertRaises(ChainDataError):
            client.address_history("https://example.com/not-an-address")

    def test_native_scatter_fallback_matches_expected_means(self) -> None:
        source = torch.tensor([[1.0, 3.0], [5.0, 7.0], [2.0, 4.0]])
        index = torch.tensor([0, 0, 1])
        self.assertTrue(
            torch.equal(
                scatter_add(source, index, dim=0, dim_size=3),
                torch.tensor([[6.0, 10.0], [2.0, 4.0], [0.0, 0.0]]),
            )
        )
        self.assertTrue(
            torch.equal(
                scatter_mean(source, index, dim=0, dim_size=3),
                torch.tensor([[3.0, 5.0], [2.0, 4.0], [0.0, 0.0]]),
            )
        )

    def test_relation_graph_contains_all_transaction_parties_and_directions(self) -> None:
        txid = "c" * 64
        raw_transactions = {
            txid: {
                "txid": txid,
                "vin": [
                    {
                        "txid": "a" * 64,
                        "prevout": {
                            "scriptpubkey_address": "1FocalAddress",
                            "value": 200_000_000,
                        },
                    },
                    {
                        "txid": "b" * 64,
                        "prevout": {
                            "scriptpubkey_address": "1SecondInput",
                            "value": 100_000_000,
                        },
                    },
                ],
                "vout": [
                    {"scriptpubkey_address": "bc1outputone", "value": 150_000_000},
                    {"scriptpubkey_address": "bc1outputtwo", "value": 149_000_000},
                ],
            }
        }
        cases = [
            {
                "transaction_id": txid,
                "risk_score": 0.93,
                "predicted_label": 1,
                "event_time": "2026-08-20T00:00:00+00:00",
            }
        ]
        graph = BitcoinLiveService._build_relation_graph(
            "1FocalAddress", raw_transactions, cases
        )
        self.assertEqual(graph["address_node_count"], 4)
        self.assertEqual(graph["transaction_node_count"], 1)
        self.assertEqual(graph["edge_count"], 4)
        input_edges = [edge for edge in graph["edges"] if edge["role"] == "input"]
        output_edges = [edge for edge in graph["edges"] if edge["role"] == "output"]
        self.assertTrue(all(edge["target"].startswith("transaction:") for edge in input_edges))
        self.assertTrue(all(edge["source"].startswith("transaction:") for edge in output_edges))


if __name__ == "__main__":
    unittest.main()
