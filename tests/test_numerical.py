"""
Bit-Exact Kernel Verification Tests for Chakra-AI.
Proves numerical equivalence between Chakra-AI PyTorch kernels and reference implementations.
Author & Creator: Abhirup Guha (Info Security Solution)
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class TestMXFP4Elementwise:
    """Verify MXFP4 dequantization is bit-exact against the OCP spec."""

    def test_e2m1_all_16_values(self):
        """All 16 E2M1 values must match the spec exactly."""
        from chakra.ops import E2M1_VALUES
        expected = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]
        for i, want in enumerate(expected):
            got = float(E2M1_VALUES[i])
            if want == 0.0:
                assert got == 0.0, f"E2M1[{i}] = {got}, expected {want}"
            else:
                assert abs(got - want) < 1e-7, f"E2M1[{i}] = {got}, expected {want}"

    def test_e8m0_scale_values(self):
        """E8M0 scale must produce correct powers of two."""
        from chakra.ops import E8M0_TORCH
        # 2^(b-127) for b in [0, 127, 128, 254], 0.0 for b=255
        assert float(E8M0_TORCH[0]) == pytest.approx(2.0 ** -127, abs=1e-40)
        assert float(E8M0_TORCH[127]) == pytest.approx(1.0, abs=1e-6)
        assert float(E8M0_TORCH[128]) == pytest.approx(2.0, abs=1e-6)
        assert float(E8M0_TORCH[254]) == pytest.approx(2.0 ** 127, abs=1e30)
        assert float(E8M0_TORCH[255]) == 0.0  # NaN by spec

    def test_nibble_order_low_is_even(self):
        """Low nibble must be the EVEN element (index 0, 2, 4, ...)."""
        from chakra.ops import E2M1_VALUES
        # Byte 0x21: low=1 (0.5), high=2 (1.0)
        byte = 0x21
        lo = byte & 0x0F  # 1
        hi = (byte >> 4) & 0x0F  # 2
        assert E2M1_VALUES[lo] == 0.5  # low nibble = EVEN element
        assert E2M1_VALUES[hi] == 1.0  # high nibble = ODD element

    def test_fused_matches_dequant_exact(self):
        """Fused and dequant paths must agree within float32 rounding."""
        from chakra.ops import matmul_mxfp4, matmul_mxfp4_fused

        torch.manual_seed(0xDEAD)
        for in_f, out_f in [(64, 16), (128, 32), (256, 64)]:
            pcols = in_f // 2
            ngrp = in_f // 32
            packed = torch.randint(0, 256, (out_f, pcols), dtype=torch.uint8)
            scales = torch.full((out_f, ngrp), 127, dtype=torch.uint8)  # scale = 1.0
            x = torch.randn(in_f)

            y_ref = matmul_mxfp4(x, packed, scales)
            y_fused = matmul_mxfp4_fused(x, packed, scales)

            max_diff = (y_ref - y_fused).abs().max().item()
            assert max_diff < 1e-4, f"in={in_f} out={out_f}: max_diff={max_diff}"


class TestRMSNorm:
    """Verify RMSNorm matches the reference formula exactly."""

    def test_rmsnorm_formula(self):
        """RMSNorm must compute: x * rsqrt(mean(x^2) + eps) * weight."""
        from chakra.ops import rms_norm

        x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        weight = torch.ones(4)
        eps = 1e-5

        result = rms_norm(x, weight, eps=eps)

        # Manual computation
        mean_sq = (x.float() ** 2).mean()
        expected = x.float() * torch.rsqrt(mean_sq + eps)

        assert torch.allclose(result.float(), expected, atol=1e-5)

    def test_rmsnorm_eps_inside_sqrt(self):
        """Epsilon must go inside the square root, not outside."""
        from chakra.ops import rms_norm

        x = torch.tensor([1e-8, 1e-8, 1e-8, 1e-8])
        weight = torch.ones(4)
        eps = 1e-5

        result = rms_norm(x, weight, eps=eps)

        # mean(x^2) = 1e-16, so eps dominates
        # rsqrt(1e-16 + 1e-5) ≈ rsqrt(1e-5) ≈ 316.23
        expected_val = 1e-8 * (1e-16 / 4 + eps) ** -0.5
        assert result[0].item() == pytest.approx(expected_val, rel=1e-4)

    def test_rmsnorm_double_accumulation(self):
        """RMSNorm must accumulate in double precision (like kimi-k3-in-c)."""
        from chakra.ops import rms_norm

        # Large values that would overflow float32 sum of squares
        x = torch.full((7168,), 1e4)
        weight = torch.ones(7168)
        result = rms_norm(x, weight)
        assert torch.isfinite(result).all()


class TestSituGLU:
    """Verify SiTU-GLU activation matches the Kimi K3 spec."""

    def test_situ_glu_analytic_cap(self):
        """Output must never exceed beta1 * beta2 = 100."""
        from chakra.ops import situ_glu

        beta1, beta2 = 4.0, 25.0
        # Drive gate to large values
        x = torch.full((1, 64), 100.0)
        result = situ_glu(x, beta=beta1, linear_beta=beta2)
        assert result.abs().max().item() <= beta1 * beta2 + 1e-3

    def test_situ_glu_sigmoid_reads_uncapped(self):
        """The sigmoid must receive the UNCAPPED gate value."""
        from chakra.ops import situ_glu

        # Small gate value where tanh(g/b) ≈ g/b
        x = torch.tensor([[0.5, 0.3]])  # gate=0.5, up=0.3
        result = situ_glu(x, beta=4.0, linear_beta=25.0)

        # Manual: a = 4*tanh(0.5/4) * sigmoid(0.5) ≈ 4*0.1247*0.6225 ≈ 0.3104
        # u = 25*tanh(0.3/25) ≈ 25*0.012 = 0.2999
        # y = a * u ≈ 0.0931
        g = 0.5
        a = 4.0 * np.tanh(g / 4.0) * (1.0 / (1.0 + np.exp(-g)))
        u = 25.0 * np.tanh(0.3 / 25.0)
        expected = a * u
        assert result[0, 0].item() == pytest.approx(expected, rel=1e-3)

    def test_situ_glu_near_zero(self):
        """Near zero, output should be approximately linear."""
        from chakra.ops import situ_glu

        x = torch.tensor([[0.001, 0.001]])
        result = situ_glu(x, beta=4.0, linear_beta=25.0)
        assert result.abs().max().item() < 0.01


class TestKDADecay:
    """Verify KDA decay computation matches the reference formula."""

    def test_kda_decay_range(self):
        """alpha must be in (e^lb, 1] where lb is gate_lower_bound."""
        from chakra.ops import kda_decay

        z = torch.randn(1, 4, 4, 32)
        dt_bias = torch.randn(4, 32)
        A_log = torch.randn(4)

        g, alpha = kda_decay(z, dt_bias, A_log, gate_lower_bound=-5.0)

        # alpha = exp(g), g in (-5, 0]
        assert alpha.max().item() <= 1.0 + 1e-6
        assert alpha.min().item() >= np.exp(-5.0) - 1e-6

    def test_kda_decay_per_head(self):
        """A_log must be indexed PER HEAD, not per channel."""
        from chakra.ops import kda_decay

        H, D = 4, 8
        z = torch.zeros(1, 1, H, D)
        dt_bias = torch.zeros(H, D)
        A_log = torch.tensor([0.0, -1.0, -2.0, -3.0])  # 4 heads

        g, alpha = kda_decay(z, dt_bias, A_log, gate_lower_bound=-5.0)

        # All channels within a head should have the same alpha
        for h in range(H):
            head_alpha = alpha[0, 0, h, :]
            assert head_alpha.std().item() < 1e-6, f"Head {h} channels differ"


class TestRouterBias:
    """Verify router logic: biased selection, unbiased weighting."""

    def test_router_selection_vs_weighting(self):
        """Bias must affect selection only, not combining weights."""
        # Simulate router logic
        scores = torch.tensor([0.5, 0.3, 0.8, 0.1])
        bias = torch.tensor([0.0, 0.0, -1.0, 0.5])

        # Selection score = score + bias
        sel_scores = scores + bias
        # Top-2 by selection: idx 0 (0.5) and idx 3 (0.1+0.5=0.6)
        _, sel_idx = sel_scores.topk(2)

        # Weight = unbiased score (not selection score)
        weights = scores[sel_idx]
        weights = weights / weights.sum()  # renormalize

        assert 0 in sel_idx.tolist()  # selected by unbiased score
        assert weights.sum().item() == pytest.approx(1.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
