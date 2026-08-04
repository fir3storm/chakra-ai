"""
Unit tests for kimipy core modules: config, st_reader, trunk_streamer, and expert_cache.
"""

import json
from pathlib import Path
import struct
import tempfile
import unittest

from kimipy import (
    ExpertLRUCache,
    KimiConfig,
    SafetensorsReader,
    TrunkStreamer,
    load_config,
)


class TestKimiConfig(unittest.TestCase):
    def test_defaults(self):
        cfg = KimiConfig()
        self.assertEqual(cfg.hidden_size, 7168)
        self.assertEqual(cfg.num_hidden_layers, 93)
        self.assertEqual(cfg.num_experts, 896)
        self.assertEqual(cfg.num_experts_per_token, 16)
        self.assertEqual(cfg.num_shared_experts, 2)
        self.assertEqual(len(cfg.full_attn_layers), 24)

    def test_layer_types(self):
        cfg = KimiConfig(num_hidden_layers=13, first_k_dense_replace=1)
        # 1-based full_attn_layers: 4, 8, 12, 13
        cfg.full_attn_layers = [4, 8, 12, 13]
        self.assertTrue(cfg.is_dense(0))
        self.assertFalse(cfg.is_dense(1))

        self.assertTrue(cfg.is_mla(3))  # Layer 3 (0-based) is index 4 (1-based) -> MLA
        self.assertTrue(cfg.is_kda(0))  # Layer 0 -> KDA

    def test_load_config_fixture(self):
        ref_json = Path("tests/fixtures/ref_k3.json")
        if ref_json.exists():
            cfg = load_config(ref_json)
            self.assertEqual(cfg.num_hidden_layers, 13)
            self.assertEqual(cfg.hidden_size, 128)


class TestSafetensorsReader(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.st_file = Path(self.temp_dir.name) / "test_model.safetensors"
        self._create_dummy_safetensors(self.st_file)

    def tearDown(self):
        self.temp_dir.cleanup()

    def _create_dummy_safetensors(self, filepath: Path):
        t1_data = struct.pack("<4f", 1.0, 2.0, 3.0, 4.0)  # 16 bytes
        t2_data = struct.pack("<2f", 5.0, 6.0)  # 8 bytes

        header_dict = {
            "model.embed_tokens.weight": {
                "dtype": "F32",
                "shape": [2, 2],
                "data_offsets": [0, 16],
            },
            "model.layers.0.self_attn.q_proj.weight": {
                "dtype": "F32",
                "shape": [2, 1],
                "data_offsets": [16, 24],
            },
            "model.layers.0.mlp.experts.0.gate_proj.weight": {
                "dtype": "F32",
                "shape": [2, 1],
                "data_offsets": [24, 24],
            },
        }

        header_json = json.dumps(header_dict).encode("utf-8")
        header_len = len(header_json)

        with open(filepath, "wb") as f:
            f.write(struct.pack("<Q", header_len))
            f.write(header_json)
            f.write(t1_data)
            f.write(t2_data)

    def test_read_tensors(self):
        with SafetensorsReader(self.st_file) as reader:
            self.assertIn("model.embed_tokens.weight", reader.tensors)
            info = reader.get_tensor_info("model.embed_tokens.weight")
            self.assertEqual(info.shape, [2, 2])
            self.assertEqual(info.size_bytes, 16)

            data = reader.get_tensor_bytes("model.embed_tokens.weight")
            vals = struct.unpack("<4f", data)
            self.assertEqual(vals, (1.0, 2.0, 3.0, 4.0))


class TestTrunkStreamer(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.st_file = Path(self.temp_dir.name) / "model.safetensors"
        self._create_dummy_model(self.st_file)

    def tearDown(self):
        self.temp_dir.cleanup()

    def _create_dummy_model(self, filepath: Path):
        data = b"\x00" * 32
        header_dict = {
            "model.embed_tokens.weight": {
                "dtype": "F32",
                "shape": [4, 2],
                "data_offsets": [0, 16],
            },
            "model.layers.0.self_attn.q_proj.weight": {
                "dtype": "F32",
                "shape": [2, 2],
                "data_offsets": [16, 32],
            },
            "model.layers.0.mlp.experts.0.gate_proj.weight": {
                "dtype": "F32",
                "shape": [2, 2],
                "data_offsets": [0, 16],
            },
            "model.norm.weight": {
                "dtype": "F32",
                "shape": [4],
                "data_offsets": [16, 32],
            },
        }

        header_json = json.dumps(header_dict).encode("utf-8")
        header_len = len(header_json)

        with open(filepath, "wb") as f:
            f.write(struct.pack("<Q", header_len))
            f.write(header_json)
            f.write(data)

    def test_trunk_categorization(self):
        with TrunkStreamer(self.st_file) as streamer:
            self.assertIn("model.embed_tokens.weight", streamer.global_tensors)
            self.assertIn("model.norm.weight", streamer.global_tensors)

            layer0_trunks = streamer.get_layer_trunk_info(0)
            tensor_names = [t.name for t in layer0_trunks]
            self.assertIn("model.layers.0.self_attn.q_proj.weight", tensor_names)
            # Routed expert should be excluded from trunk
            self.assertNotIn("model.layers.0.mlp.experts.0.gate_proj.weight", tensor_names)

            # Test streaming
            layers = list(streamer.stream_layers())
            self.assertEqual(len(layers), 1)
            l_idx, l_data = layers[0]
            self.assertEqual(l_idx, 0)
            self.assertIn("model.layers.0.self_attn.q_proj.weight", l_data)


class TestExpertLRUCache(unittest.TestCase):
    def test_cache_eviction_and_stats(self):
        cache = ExpertLRUCache(capacity_bytes=100, max_experts=2)

        def mock_fetch(layer_idx, expert_idx):
            return b"X" * 40  # 40 bytes per expert

        # Put 2 experts
        exp0 = cache.get(0, 0, fetch_fn=mock_fetch)
        self.assertEqual(len(exp0), 40)
        cache.get(0, 1, fetch_fn=mock_fetch)

        stats = cache.get_stats()
        self.assertEqual(stats["cached_experts"], 2)
        self.assertEqual(stats["current_bytes"], 80)

        # Access 0,0 to make it most recently used
        cache.get(0, 0, fetch_fn=mock_fetch)

        # Add 3rd expert (0,2) -> should evict (0,1)
        cache.get(0, 2, fetch_fn=mock_fetch)

        self.assertFalse(cache.contains(0, 1))
        self.assertTrue(cache.contains(0, 0))
        self.assertTrue(cache.contains(0, 2))

        stats = cache.get_stats()
        self.assertEqual(stats["evictions"], 1)
        self.assertEqual(stats["cached_experts"], 2)


if __name__ == "__main__":
    unittest.main()
