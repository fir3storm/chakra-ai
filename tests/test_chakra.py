"""
Unit test suite for chakra package (KimiWin-Py / KimiPy Engine).

Tests configuration loading, MXFP4 dequantization, Safetensors reader,
TrunkStreamer streaming logic, and ExpertLRUCache.
"""

import json
import os
from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np
import torch

from chakra import (
    E2M1_TABLE,
    ExpertLRUCache,
    KimiConfig,
    SafetensorsReader,
    TrunkStreamer,
    dequantize_mxfp4_numpy,
    dequantize_mxfp4_torch,
    load_config,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"


class TestKimiConfig(unittest.TestCase):
    """Test suite for KimiConfig loading and helper methods."""

    def test_default_config(self) -> None:
        cfg = KimiConfig()
        self.assertEqual(cfg.hidden_size, 7168)
        self.assertEqual(cfg.num_hidden_layers, 93)
        self.assertEqual(cfg.vocab_size, 163840)
        self.assertTrue(len(cfg.full_attn_layers) > 0)

    def test_layer_types(self) -> None:
        cfg = KimiConfig(num_hidden_layers=8, first_k_dense_replace=1)
        cfg.full_attn_layers = [4, 8]  # 1-based indices: layer 4 and 8 are MLA

        # 0-based indices:
        self.assertTrue(cfg.is_dense(0))
        self.assertFalse(cfg.is_dense(1))

        self.assertTrue(cfg.is_mla(3))  # Layer 4 (1-based)
        self.assertFalse(cfg.is_kda(3))

        self.assertTrue(cfg.is_kda(0))  # Layer 1 (1-based)
        self.assertFalse(cfg.is_mla(0))

    def test_load_config_file(self) -> None:
        ref_path = FIXTURES_DIR / "ref_k3.json"
        if ref_path.exists():
            cfg = load_config(ref_path)
            self.assertIsInstance(cfg, KimiConfig)
            self.assertGreater(cfg.hidden_size, 0)
            self.assertGreater(cfg.num_hidden_layers, 0)

    def test_invalid_config_file(self) -> None:
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".json") as f:
            json.dump({"hidden_size": -10, "num_hidden_layers": 0}, f)
            temp_path = f.name

        try:
            with self.assertRaises(ValueError):
                load_config(temp_path)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)


class TestMXFP4Dequantization(unittest.TestCase):
    """Test suite for MXFP4 dequantization to NumPy and PyTorch."""

    def test_e2m1_table_values(self) -> None:
        self.assertEqual(len(E2M1_TABLE), 16)
        self.assertEqual(E2M1_TABLE[0], 0.0)
        self.assertEqual(E2M1_TABLE[1], 0.5)
        self.assertEqual(E2M1_TABLE[7], 6.0)
        self.assertEqual(E2M1_TABLE[8], -0.0)
        self.assertEqual(E2M1_TABLE[15], -6.0)

    def test_mxfp4_numpy_dequantization_synthetic(self) -> None:
        # 1 row, 1 packed byte (2 nibbles: low=1 -> 0.5, high=2 -> 1.0)
        packed = np.array([[0x21]], dtype=np.uint8)  # lo=1, hi=2
        # Scale byte = 127 -> 2^(127-127) = 1.0
        scale = np.array([[127]], dtype=np.uint8)

        out = dequantize_mxfp4_numpy(packed, scale, group_size=32)
        self.assertEqual(out.shape, (1, 2))
        np.testing.assert_allclose(out[0, 0], 0.5, rtol=1e-5)
        np.testing.assert_allclose(out[0, 1], 1.0, rtol=1e-5)

    def test_mxfp4_torch_dequantization_synthetic(self) -> None:
        packed = torch.tensor([[0x21]], dtype=torch.uint8)
        scale = torch.tensor([[127]], dtype=torch.uint8)

        out = dequantize_mxfp4_torch(packed, scale, group_size=32)
        self.assertIsInstance(out, torch.Tensor)
        self.assertEqual(out.shape, torch.Size([1, 2]))
        self.assertAlmostEqual(out[0, 0].item(), 0.5, places=5)
        self.assertAlmostEqual(out[0, 1].item(), 1.0, places=5)

    def test_mxfp4_fixture_golden(self) -> None:
        fixture_path = FIXTURES_DIR / "mxfp4.json"
        if not fixture_path.exists():
            self.skipTest("mxfp4.json fixture not found")

        with open(fixture_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        packed = np.array(data["packed"]["data"], dtype=np.uint8).reshape(data["packed"]["shape"])
        # If scale data is present in fixture or synthetic scale:
        scale_data = data.get("scale", {}).get("data")
        if scale_data:
            scale = np.array(scale_data, dtype=np.uint8).reshape(data["scale"]["shape"])
        else:
            scale = np.full((data["rows"], data["scale_cols"]), 127, dtype=np.uint8)

        out = dequantize_mxfp4_numpy(packed, scale, group_size=data.get("group_size", 32))
        self.assertEqual(out.shape, (data["rows"], data["logical_width"]))


class TestSafetensorsReader(unittest.TestCase):
    """Test suite for SafetensorsReader."""

    def setUp(self) -> None:
        # Create a synthetic Safetensors file in temp dir
        self.temp_dir = tempfile.TemporaryDirectory()
        self.st_path = Path(self.temp_dir.name) / "model.safetensors"

        tensor1_data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32).tobytes()
        tensor2_data = np.array([10, 20, 30], dtype=np.int32).tobytes()

        header_dict = {
            "tensor1": {
                "dtype": "F32",
                "shape": [2, 2],
                "data_offsets": [0, len(tensor1_data)],
            },
            "tensor2": {
                "dtype": "I32",
                "shape": [3],
                "data_offsets": [len(tensor1_data), len(tensor1_data) + len(tensor2_data)],
            },
        }

        header_bytes = json.dumps(header_dict).encode("utf-8")
        header_len = len(header_bytes)

        with open(self.st_path, "wb") as f:
            f.write(struct.pack("<Q", header_len))
            f.write(header_bytes)
            f.write(tensor1_data)
            f.write(tensor2_data)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_reader_parse_header(self) -> None:
        with SafetensorsReader(self.st_path) as reader:
            self.assertIn("tensor1", reader.tensors)
            self.assertIn("tensor2", reader.tensors)

            info1 = reader.get_tensor_info("tensor1")
            self.assertEqual(info1.dtype, "F32")
            self.assertEqual(info1.shape, [2, 2])
            self.assertEqual(info1.num_elements, 4)
            self.assertEqual(info1.size_bytes, 16)

    def test_reader_get_tensor_data(self) -> None:
        with SafetensorsReader(self.st_path) as reader:
            raw_bytes = reader.get_tensor_data("tensor1", return_type="bytes")
            self.assertEqual(len(raw_bytes), 16)

            np_arr = reader.get_tensor_data("tensor1", return_type="numpy")
            self.assertEqual(np_arr.shape, (2, 2))
            np.testing.assert_allclose(np_arr, [[1.0, 2.0], [3.0, 4.0]])

            torch_t = reader.get_tensor_data("tensor2", return_type="torch")
            self.assertEqual(torch_t.shape, torch.Size([3]))
            self.assertTrue(torch.equal(torch_t, torch.tensor([10, 20, 30], dtype=torch.int32)))


class TestTrunkStreamer(unittest.TestCase):
    """Test suite for TrunkStreamer layer-by-layer streaming logic."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.st_path = Path(self.temp_dir.name) / "model.safetensors"

        t1_bytes = np.ones((4, 4), dtype=np.float32).tobytes()
        t2_bytes = np.ones((4, 4), dtype=np.float32).tobytes()
        t3_bytes = np.ones((4, 4), dtype=np.float32).tobytes()
        t4_bytes = np.ones((4, 4), dtype=np.float32).tobytes()

        # Global, Layer 0 trunk, Layer 0 routed expert, Layer 0 shared expert
        header_dict = {
            "model.embed_tokens.weight": {
                "dtype": "F32",
                "shape": [4, 4],
                "data_offsets": [0, 64],
            },
            "model.layers.0.input_layernorm.weight": {
                "dtype": "F32",
                "shape": [4, 4],
                "data_offsets": [64, 128],
            },
            "model.layers.0.block_sparse_moe.experts.0.w1": {
                "dtype": "F32",
                "shape": [4, 4],
                "data_offsets": [128, 192],
            },
            "model.layers.0.block_sparse_moe.shared_experts.w1": {
                "dtype": "F32",
                "shape": [4, 4],
                "data_offsets": [192, 256],
            },
        }

        header_bytes = json.dumps(header_dict).encode("utf-8")

        with open(self.st_path, "wb") as f:
            f.write(struct.pack("<Q", len(header_bytes)))
            f.write(header_bytes)
            f.write(t1_bytes + t2_bytes + t3_bytes + t4_bytes)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_categorization(self) -> None:
        with TrunkStreamer(self.st_path) as streamer:
            self.assertIn("model.embed_tokens.weight", streamer.global_tensors)
            self.assertIn(0, streamer.layer_trunks)

            trunk_names = [t.name for t in streamer.get_layer_trunk_info(0)]
            self.assertIn("model.layers.0.input_layernorm.weight", trunk_names)
            self.assertIn("model.layers.0.block_sparse_moe.shared_experts.w1", trunk_names)
            # Routed expert MUST be excluded from trunk streamer
            self.assertNotIn("model.layers.0.block_sparse_moe.experts.0.w1", trunk_names)

    def test_stream_layers(self) -> None:
        with TrunkStreamer(self.st_path) as streamer:
            layers = list(streamer.stream_layers(return_type="numpy"))
            self.assertEqual(len(layers), 1)
            layer_idx, layer_dict = layers[0]
            self.assertEqual(layer_idx, 0)
            self.assertIn("model.layers.0.input_layernorm.weight", layer_dict)


class TestExpertLRUCache(unittest.TestCase):
    """Test suite for ExpertLRUCache."""

    def test_lru_cache_operations(self) -> None:
        cache = ExpertLRUCache(capacity_bytes=1000, max_experts=2)

        def mock_fetch(layer_idx: int, expert_idx: int) -> bytes:
            return f"expert_{layer_idx}_{expert_idx}".encode("utf-8")

        data0 = cache.get(0, 0, fetch_fn=mock_fetch)
        self.assertEqual(data0, b"expert_0_0")
        self.assertTrue(cache.contains(0, 0))

        data1 = cache.get(0, 1, fetch_fn=mock_fetch)
        self.assertEqual(len(cache), 2)

        # Trigger eviction of (0, 0) by fetching (0, 2)
        data2 = cache.get(0, 2, fetch_fn=mock_fetch)
        self.assertEqual(len(cache), 2)
        self.assertFalse(cache.contains(0, 0))
        self.assertTrue(cache.contains(0, 1))
        self.assertTrue(cache.contains(0, 2))

        stats = cache.get_stats()
        self.assertGreater(stats["misses"], 0)
        self.assertEqual(stats["evictions"], 1)


class TestMultiAgent(unittest.TestCase):
    """Test suite for Multi-Agent support in chakra."""

    def test_multi_agent_exports(self) -> None:
        from chakra import MultiAgentOrchestrator, print_agent_step
        self.assertIsNotNone(MultiAgentOrchestrator)
        self.assertIsNotNone(print_agent_step)

    def test_print_agent_step(self) -> None:
        from chakra.ui import print_agent_step
        # Ensure print_agent_step executes without errors for all 4 team roles
        for role in ["Architect", "Coder", "Auditor", "Supervisor"]:
            print_agent_step(role, f"{role} is ready.")

    def test_orchestrator_list_agents(self) -> None:
        from chakra import MultiAgentOrchestrator
        orchestrator = MultiAgentOrchestrator()
        roles = orchestrator.list_agents()
        self.assertIn("Architect", roles)
        self.assertIn("Coder", roles)
        self.assertIn("Auditor", roles)
        self.assertIn("Supervisor", roles)

    def test_orchestrator_team_collaboration(self) -> None:
        from chakra import MultiAgentOrchestrator
        orchestrator = MultiAgentOrchestrator()
        res = orchestrator.run_team_collaboration("Create a python function to multiply two numbers")
        self.assertIn("success", res)
        self.assertIn("code", res)
        self.assertTrue(len(res["code"]) > 0)


if __name__ == "__main__":
    unittest.main()

