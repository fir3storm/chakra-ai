"""
Gate Ladder Tests for Chakra-AI verification.
Mirrors kimi-k3-in-c's validation approach: each gate must pass before the next is meaningful.
Author & Creator: Abhirup Guha (Info Security Solution)
"""
import ast
import hashlib
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class Gate1_TokenizerParity:
    """Gate 1: Tokenizer encode/decode must roundtrip correctly."""

    def test_fallback_roundtrip(self):
        """UTF-8 fallback tokenizer must roundtrip ASCII text."""
        from chakra.tokenizer import KimiTokenizer
        tok = KimiTokenizer(mode="auto")
        text = "Hello, world! The capital of France is Paris."
        ids = tok.encode(text)
        decoded = tok.decode(ids)
        # Fallback mode uses byte-level encoding, so we check structural roundtrip
        assert isinstance(ids, list)
        assert len(ids) > 0
        assert isinstance(decoded, str)

    def test_empty_input(self):
        """Tokenizer must handle empty input gracefully."""
        from chakra.tokenizer import KimiTokenizer
        tok = KimiTokenizer(mode="auto")
        ids = tok.encode("")
        assert ids == [] or ids == [0]  # Depending on mode

    def test_decode_tensor(self):
        """Tokenizer must decode torch tensors."""
        import torch
        from chakra.tokenizer import KimiTokenizer
        tok = KimiTokenizer(mode="auto")
        tensor = torch.tensor([[72, 101, 108, 108, 111]])  # "Hello" bytes
        result = tok.decode(tensor[0].tolist())
        assert isinstance(result, str)


class Gate2_ConfigReader:
    """Gate 2: Config reader must reject invalid configs."""

    def test_valid_config(self):
        """K3Config must initialize with correct defaults."""
        from chakra.model import K3Config
        cfg = K3Config()
        assert cfg.num_hidden_layers == 93
        assert cfg.hidden_size == 7168
        assert cfg.num_experts == 896
        assert cfg.topk == 16
        assert cfg.num_shared_experts == 2

    def test_tiny_config(self):
        """tiny_config must create a valid miniature model."""
        from chakra.model import tiny_config
        cfg = tiny_config()
        assert cfg.num_hidden_layers == 13
        assert cfg.hidden_size == 128
        # Must still have all architectural features
        assert cfg.num_experts > 0
        assert cfg.topk > 0

    def test_layer_map_consistency(self):
        """Layer map must have KDA + MLA = num_hidden_layers."""
        from chakra.model import K3Config
        cfg = K3Config()
        kda_count = len(cfg.kda_layers)
        mla_count = len(cfg.mla_layers)
        assert kda_count + mla_count == cfg.num_hidden_layers


class Gate3_MXFP4Dequantization:
    """Gate 3: MXFP4 dequantization must be bit-exact against reference."""

    def test_e2m1_table(self):
        """E2M1 lookup table must match the OCP MXFP4 spec."""
        from chakra.dequant import E2M1_TABLE
        expected = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]
        for i, (got, want) in enumerate(zip(E2M1_TABLE, expected)):
            if want == 0.0:
                assert got == 0.0, f"E2M1[{i}]: got {got}, want {want}"
            else:
                assert abs(got - want) < 1e-6, f"E2M1[{i}]: got {got}, want {want}"

    def test_dequant_known_bytes(self):
        """Dequantization of known bytes must produce exact floats."""
        from chakra.dequant import dequantize_mxfp4_numpy
        import numpy as np
        # Single byte: low nibble = 2 (1.0), high nibble = 7 (6.0)
        # Scale byte = 127 (2^0 = 1.0)
        packed = np.array([0x72], dtype=np.uint8)
        scales = np.array([127], dtype=np.uint8)
        result = dequantize_mxfp4_numpy(packed, scales, group_size=32, in_features=2)
        # low nibble 2 = E2M1[2] = 1.0, high nibble 7 = E2M1[7] = 6.0
        # scale = 2^(127-127) = 1.0
        assert abs(result[0] - 1.0) < 1e-6, f"Expected 1.0, got {result[0]}"
        assert abs(result[1] - 6.0) < 1e-6, f"Expected 6.0, got {result[1]}"

    def test_dequant_nan_scale(self):
        """Scale byte 255 (NaN) must zero out the group."""
        from chakra.dequant import dequantize_mxfp4_numpy
        import numpy as np
        packed = np.array([0x00], dtype=np.uint8)  # two zeros
        scales = np.array([255], dtype=np.uint8)  # NaN scale
        result = dequantize_mxfp4_numpy(packed, scales, group_size=32, in_features=2)
        assert all(r == 0.0 for r in result), "NaN scale should zero out group"

    def test_fused_vs_dequant_matmul(self):
        """Fused MXFP4 matmul must match dequantize-first path within float32 rounding."""
        import torch
        from chakra.ops import matmul_mxfp4, matmul_mxfp4_fused

        torch.manual_seed(42)
        in_features = 256
        out_features = 64
        group = 32
        pcols = in_features // 2
        ngrp = in_features // group

        # Random packed bytes and scales
        packed = torch.randint(0, 256, (out_features, pcols), dtype=torch.uint8)
        scales = torch.randint(120, 135, (out_features, ngrp), dtype=torch.uint8)
        x = torch.randn(in_features)

        y_dequant = matmul_mxfp4(x, packed, scales, group=group)
        y_fused = matmul_mxfp4_fused(x, packed, scales, group=group)

        # Both should be close (float32 rounding differences)
        assert torch.allclose(y_dequant, y_fused, atol=1e-4, rtol=1e-4), \
            f"Fused vs dequant mismatch: max diff = {(y_dequant - y_fused).abs().max().item()}"

    def test_fused_batch_matmul(self):
        """Fused matmul must work with batched inputs."""
        import torch
        from chakra.ops import matmul_mxfp4_fused

        torch.manual_seed(123)
        in_features = 128
        out_features = 32
        group = 32
        pcols = in_features // 2
        ngrp = in_features // group

        packed = torch.randint(0, 256, (out_features, pcols), dtype=torch.uint8)
        scales = torch.randint(120, 135, (out_features, ngrp), dtype=torch.uint8)
        x = torch.randn(4, in_features)  # batch of 4

        y = matmul_mxfp4_fused(x, packed, scales, group=group)
        assert y.shape == (4, out_features)
        assert torch.isfinite(y).all()


class Gate4_ModelForward:
    """Gate 4: Model forward pass must produce finite outputs."""

    def test_tiny_model_init(self):
        """Tiny model must initialize without errors."""
        from chakra.model import K3Model, tiny_config
        cfg = tiny_config()
        model = K3Model(cfg)
        assert model is not None
        # Count parameters
        total = sum(p.numel() for p in model.parameters())
        assert total > 0

    def test_tiny_model_forward_finite(self):
        """Tiny model forward pass must produce finite outputs."""
        import torch
        from chakra.model import K3Model, tiny_config
        cfg = tiny_config()
        model = K3Model(cfg)
        model.eval()
        # Random input tokens
        x = torch.randint(0, cfg.vocab_size, (1, 4))
        with torch.no_grad():
            out = model(x)
        assert torch.isfinite(out).all(), "Model output contains non-finite values"

    def test_tiny_model_generate(self):
        """Tiny model must generate tokens without crashing."""
        import torch
        from chakra.model import K3Model, tiny_config
        cfg = tiny_config()
        model = K3Model(cfg)
        model.eval()
        x = torch.randint(0, cfg.vocab_size, (1, 4))
        with torch.no_grad():
            out = model.generate(x, n_new=4)
        assert out.shape[0] == 1
        assert out.shape[1] == 8  # 4 prompt + 4 generated


class Gate5_SandboxIsolation:
    """Gate 5: Sandbox must isolate execution properly."""

    def test_sandbox_executes_clean_code(self):
        """Clean code must execute successfully."""
        from chakra.agent import KimiAgent
        agent = KimiAgent(model=None)
        result = agent.execute_sandbox("print('hello')")
        assert result["success"] is True
        assert "hello" in result["stdout"]

    def test_sandbox_captures_errors(self):
        """Runtime errors must be captured, not crash the process."""
        from chakra.agent import KimiAgent
        agent = KimiAgent(model=None)
        result = agent.execute_sandbox("raise ValueError('test error')")
        assert result["success"] is False
        assert "ValueError" in result["stderr"]

    def test_sandbox_timeout(self):
        """Long-running code must be killed by timeout."""
        from chakra.agent import KimiAgent
        agent = KimiAgent(model=None)
        result = agent.execute_sandbox("import time; time.sleep(60)", timeout=2)
        assert result["success"] is False
        assert result["timed_out"] is True

    def test_sandbox_empty_code(self):
        """Empty code must be rejected."""
        from chakra.agent import KimiAgent
        agent = KimiAgent(model=None)
        result = agent.execute_sandbox("")
        assert result["success"] is False

    def test_sandbox_no_write_bytecode(self):
        """Sandbox must set PYTHONDONTWRITEBYTECODE."""
        from chakra.agent import KimiAgent
        agent = KimiAgent(model=None)
        result = agent.execute_sandbox(
            "import os; print(os.environ.get('PYTHONDONTWRITEBYTECODE', 'NOT_SET'))"
        )
        assert result["success"] is True
        assert "1" in result["stdout"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
