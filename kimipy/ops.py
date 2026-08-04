# kimipy/ops.py - Core mathematical operations and kernels for Kimi K3
"""
kimipy.ops - MXFP4 dequantization, RMSNorm, SiTU-GLU/SwiGLU, RoPE, KDA recurrence, and MLA attention.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------- MXFP4
# OCP E2M1 lookup table: 16 values (index 0..15)
# Low nibble = even element index, high nibble = odd element index.
E2M1_VALUES = np.array(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=np.float32,
)

E2M1_TORCH = torch.tensor(E2M1_VALUES, dtype=torch.float32)

# E8M0 scale lookup: 2^(b - 127), with 255 -> 0.0 (NaN by spec)
_E8M0_SCALE = np.zeros(256, dtype=np.float32)
for _b in range(255):
    _E8M0_SCALE[_b] = 2.0 ** (_b - 127)
_E8M0_SCALE[255] = 0.0
E8M0_TORCH = torch.tensor(_E8M0_SCALE, dtype=torch.float32)


def dequant_mxfp4(
    packed: torch.Tensor | np.ndarray,
    scales: torch.Tensor | np.ndarray,
    group: int = 32,
) -> torch.Tensor:
    """Dequantizes MXFP4 (OCP E2M1 format with E8M0 group scales) into float32.

    Args:
        packed: Byte array of packed nibbles with shape [rows, pcols].
        scales: Byte array of E8M0 group scale bytes with shape [rows, ngrp].
        group: Elements per scale group (default 32).

    Returns:
        Float32 tensor of shape [rows, pcols * 2].
    """
    is_numpy = isinstance(packed, np.ndarray)
    if is_numpy:
        packed_np = packed
        scales_np = scales
    else:
        packed_np = packed.cpu().numpy()
        scales_np = scales.cpu().numpy()

    rows, pcols = packed_np.shape
    width = pcols * 2

    lo = packed_np & 0x0F
    hi = (packed_np >> 4) & 0x0F

    out = np.empty((rows, width), dtype=np.float32)
    out[:, 0::2] = E2M1_VALUES[lo]
    out[:, 1::2] = E2M1_VALUES[hi]

    # E8M0 exponent scale: 2^(sb - 127). 255 maps to 0.0 (NaN by spec)
    mult = np.where(
        scales_np == 255,
        0.0,
        np.exp2(scales_np.astype(np.int32) - 127).astype(np.float32),
    )

    # Apply group scale across width
    scale_expanded = np.repeat(mult, group, axis=1)[:, :width]
    out *= scale_expanded

    res = torch.from_numpy(out)
    if not is_numpy and isinstance(packed, torch.Tensor):
        res = res.to(packed.device)
    return res


def matmul_mxfp4(
    x: torch.Tensor,
    packed: torch.Tensor,
    scales: torch.Tensor,
    group: int = 32,
) -> torch.Tensor:
    """Matrix multiplication y = x @ W^T with W stored as packed MXFP4.

    Args:
        x: Input tensor [..., in_features]
        packed: Packed MXFP4 weight bytes [out_features, in_features // 2]
        scales: Scale bytes [out_features, ceil(in_features / group)]
        group: Group size (default 32)

    Returns:
        Result tensor [..., out_features]
    """
    W = dequant_mxfp4(packed, scales, group=group).to(dtype=x.dtype, device=x.device)
    return F.linear(x, W)


def matmul_mxfp4_fused(
    x: torch.Tensor,
    packed: torch.Tensor,
    scales: torch.Tensor,
    group: int = 32,
) -> torch.Tensor:
    """Fused MXFP4 matmul: accumulate within groups, apply scale once per group.

    Reads only packed nibbles (17.55 MB per expert) instead of dequantizing
    to float32 (132 MB per expert). 7.5x less memory traffic.

    Equivalent to: y = x @ dequant(packed, scales).T
    but processes group-by-group to minimize peak memory.

    Args:
        x: Input tensor [..., in_features]
        packed: Packed MXFP4 weight bytes [out_features, in_features // 2] uint8
        scales: Scale bytes [out_features, ceil(in_features / group)] uint8
        group: Group size (default 32)

    Returns:
        Result tensor [..., out_features]
    """
    squeeze = False
    if x.dim() == 1:
        x = x.unsqueeze(0)
        squeeze = True

    batch_shape = x.shape[:-1]
    in_features = x.shape[-1]
    out_features = packed.shape[0]
    n_groups = in_features // group
    half_group = group // 2

    # Flatten batch dims
    x_flat = x.reshape(-1, in_features).float()  # [B, in]
    B = x_flat.shape[0]

    # Prepare output
    y = x_flat.new_zeros(B, out_features)

    # E2M1 lookup on device
    e2m1 = E2M1_TORCH.to(x.device)

    # Process group-by-group to minimize peak memory
    for g in range(n_groups):
        # Scale for this group: [out]
        s = E8M0_TORCH.to(x.device)[scales[:, g].long()]

        # Skip zero-scale groups (NaN by spec)
        if (s == 0).all():
            continue

        # Packed bytes for this group: [out, half_group]
        pb = packed[:, g * half_group : (g + 1) * half_group].long()

        # Unpack nibbles: low = even, high = odd
        lo_idx = pb & 0x0F          # [out, half_group]
        hi_idx = (pb >> 4) & 0x0F   # [out, half_group]

        # E2M1 lookup → [out, half_group]
        w_lo = e2m1[lo_idx]  # [out, half_group]
        w_hi = e2m1[hi_idx]  # [out, half_group]

        # Interleave: [out, half_group, 2] → [out, group]
        w = torch.stack([w_lo, w_hi], dim=-1).reshape(out_features, group)

        # Input slice for this group: [B, group]
        x_g = x_flat[:, g * group : (g + 1) * group]

        # Partial dot product: [B, out] += x_g @ w.T * scale
        # Using addmm for efficiency: y += scale * (x_g @ w.T)
        partial = torch.mm(x_g, w.T)  # [B, out]
        partial = partial * s.unsqueeze(0)  # apply group scale
        y += partial

    if squeeze:
        y = y.squeeze(0)

    return y.to(dtype=x.dtype)


# --------------------------------------------------------------------------- RMSNorm
class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    Upcasts inputs to float32 for numerical stability.
    """

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_float = x.float()
        var = x_float.pow(2).mean(-1, keepdim=True)
        rsqrt = torch.rsqrt(var + self.eps)
        normed = x_float * rsqrt
        return (self.weight.float() * normed).to(dtype=x.dtype)


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Functional RMSNorm."""
    x_float = x.float()
    var = x_float.pow(2).mean(-1, keepdim=True)
    normed = x_float * torch.rsqrt(var + eps)
    return (weight.float() * normed).to(dtype=x.dtype)


def l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """L2 normalization along the last dimension."""
    x_float = x.float()
    norm = torch.rsqrt(x_float.pow(2).sum(-1, keepdim=True) + eps)
    return (x_float * norm).to(dtype=x.dtype)


# --------------------------------------------------------------------------- Activations
def situ_glu(
    x: torch.Tensor,
    beta: float = 4.0,
    linear_beta: Optional[float] = 25.0,
) -> torch.Tensor:
    """SiTU-GLU (Sigmoid-Tanh-Gated Linear Unit) activation as defined in Kimi K3.

    Splits the last dimension into gate and up projections.
    The sigmoid receives the UNCAPPED gate value.
    """
    d = x.shape[-1] // 2
    gate = x[..., :d].float()
    up = x[..., d:].float()

    a = beta * torch.tanh(gate / beta) * torch.sigmoid(gate)
    if linear_beta is not None:
        up = linear_beta * torch.tanh(up / linear_beta)

    return (a * up).to(dtype=x.dtype)


def swiglu(x: torch.Tensor) -> torch.Tensor:
    """Standard SwiGLU activation function."""
    d = x.shape[-1] // 2
    gate, up = x[..., :d], x[..., d:]
    return F.silu(gate) * up


# --------------------------------------------------------------------------- RoPE
def apply_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Applies Rotary Position Embedding (RoPE) to tensor x.

    Args:
        x: [B, T, H, D] or [B, T, D]
        cos: Broadcastable cos tensor
        sin: Broadcastable sin tensor
    """
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    rotated = torch.cat([-x2, x1], dim=-1)
    return (x * cos) + (rotated * sin)


def precompute_rope_freqs(
    dim: int,
    max_seq_len: int = 4096,
    theta: float = 10000.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Precomputes cos and sin frequency tensors for RoPE."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_seq_len).float()
    freqs = torch.outer(t, freqs)
    cos = torch.cos(freqs)
    sin = torch.sin(freqs)
    return torch.cat([cos, cos], dim=-1), torch.cat([sin, sin], dim=-1)


# --------------------------------------------------------------------------- ShortConv
class ShortConv(nn.Module):
    """Causal depthwise 1D convolution with fused SiLU activation.

    Used in Kimi Delta Attention (KDA).
    Weight shape: [channels, 1, kernel_size]
    State shape: [B, channels, kernel_size - 1]
    """

    def __init__(self, channels: int, kernel_size: int = 4):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.weight = nn.Parameter(torch.zeros(channels, 1, kernel_size))

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Args:

        x: [B, T, C]
        state: [B, C, kernel_size - 1] or None

        Returns:
            y: [B, T, C]
            new_state: [B, C, kernel_size - 1]
        """
        B, T, C = x.shape
        xt = x.transpose(1, 2)  # [B, C, T]

        if state is not None:
            xp = torch.cat([state, xt], dim=-1)
        else:
            pad = xt.new_zeros(B, C, self.kernel_size - 1)
            xp = torch.cat([pad, xt], dim=-1)

        y = F.conv1d(xp, self.weight, groups=C)
        new_state = xp[..., -(self.kernel_size - 1):] if self.kernel_size > 1 else None
        return F.silu(y.transpose(1, 2)), new_state


# --------------------------------------------------------------------------- KDA Recurrence
def kda_decay(
    z: torch.Tensor,
    dt_bias: torch.Tensor,
    A_log: torch.Tensor,
    gate_lower_bound: float = -5.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Computes KDA log-decay factor g and decay multiplier alpha.

    z: [B, T, H, D]
    dt_bias: [H * D] or [H, D]
    A_log: [H] (per head)
    gate_lower_bound: scalar float (-5.0)

    Returns:
        g: [B, T, H, D]
        alpha: [B, T, H, D] = exp(g)
    """
    H, D = z.shape[-2], z.shape[-1]
    z_biased = z + dt_bias.float().view(H, D)
    a = A_log.float().exp().view(H, 1)  # per head
    g = gate_lower_bound * torch.sigmoid(a * z_biased)
    alpha = torch.exp(g)
    return g, alpha


def kda_step_single(
    S: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Single-token recurrent step for KDA.

    Args:
        S: State matrix [B, H, D_k, D_v]
        q: Pre-scaled query [B, H, D_k]
        k: Normalized key [B, H, D_k]
        v: Value [B, H, D_v]
        alpha: Decay factor [B, H, D_k] (or [B, H, D_k, D_v])
        beta: Gate scalar [B, H, 1]

    Returns:
        o: Output tensor [B, H, D_v]
        new_S: Updated state matrix [B, H, D_k, D_v]
    """
    if alpha.ndim == 3:
        alpha = alpha.unsqueeze(-1)  # [B, H, D_k, 1]

    # 1. Decay rows of state
    S = S * alpha

    # 2. Read state using key: u = S^T k -> [B, H, D_v]
    u = torch.einsum("bhk,bhkv->bhv", k, S)

    # 3. Delta error update
    w = beta * (v - u)  # [B, H, D_v]
    S = S + torch.einsum("bhk,bhv->bhkv", k, w)

    # 4. Read updated state using query: o = S^T q
    o = torch.einsum("bhk,bhkv->bhv", q, S)

    return o, S


# --------------------------------------------------------------------------- MLA Attention
def mla_attention(
    q_nope: torch.Tensor,
    q_rope: torch.Tensor,
    k_nope: torch.Tensor,
    k_rope: torch.Tensor,
    v: torch.Tensor,
    scale: float,
    cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Gated Multi-head Latent Attention (MLA) step with NoPE (No Position Embeddings).

    Args:
        q_nope: [B, T, H, qk_nope]
        q_rope: [B, T, H, qk_rope]
        k_nope: [B, T_kv, H, qk_nope]
        k_rope: [B, T_kv, H, qk_rope] (broadcasted from 1 head or expanded)
        v: [B, T_kv, H, v_head]
        scale: Softmax scale factor (q_head_dim ** -0.5)
        cache: Optional tuple of past (latent_cache, k_rope_cache)

    Returns:
        y: Output tensor [B, T, H * v_head]
        new_cache: Updated KV cache tuple
    """
    B, T, H, _ = q_nope.shape

    qs = torch.cat([q_nope, q_rope], dim=-1).transpose(1, 2).float()  # [B, H, T, qh]
    ks = torch.cat([k_nope, k_rope], dim=-1).transpose(1, 2).float()  # [B, H, T_kv, qh]
    vs = v.transpose(1, 2).float()  # [B, H, T_kv, v_head]

    # Attention scores
    att = torch.matmul(qs, ks.transpose(-1, -2)) * scale  # [B, H, T, T_kv]

    Tq, Tk = att.shape[-2], att.shape[-1]
    if Tq > 1:
        causal = torch.ones(Tq, Tk, dtype=torch.bool, device=qs.device).tril(Tk - Tq)
        att = att.masked_fill(~causal, float("-inf"))

    probs = F.softmax(att, dim=-1)
    y = torch.matmul(probs, vs).transpose(1, 2).reshape(B, T, H * v.shape[-1])

    new_cache = (k_nope, k_rope)  # Placeholder cache reference if updated externally
    return y, new_cache
