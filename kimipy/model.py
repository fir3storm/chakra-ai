# kimipy/model.py - Full Kimi K3 93-layer forward execution loop and state management
"""
kimipy.model - Model runner, full 93-layer forward pass execution loop, state management,
and incremental decoding support for Kimi K3.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from kimipy.ops import (
    RMSNorm,
    ShortConv,
    l2norm,
    matmul_mxfp4_fused,
    situ_glu,
)


# --------------------------------------------------------------------------- Config
@dataclass
class K3Config:
    """Model configuration parameters for Kimi K3.

    Defaults correspond to full Kimi K3 (2.78T parameter MoE architecture).
    """

    hidden_size: int = 7168
    num_hidden_layers: int = 93
    vocab_size: int = 163840
    rms_norm_eps: float = 1e-5
    tie_word_embeddings: bool = False

    # KDA (Kimi Delta Attention)
    kda_num_heads: int = 96
    kda_head_dim: int = 128
    short_conv_kernel_size: int = 4
    gate_lower_bound: float = -5.0

    # Gated MLA (Multi-head Latent Attention)
    num_attention_heads: int = 96
    q_lora_rank: int = 1536
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    mla_use_output_gate: bool = True

    # MoE (Mixture of Experts)
    num_experts: int = 896
    num_experts_per_token: int = 16
    num_shared_experts: int = 2
    routed_expert_hidden_size: int = 3584
    moe_intermediate_size: int = 3072
    routed_scaling_factor: float = 1.0
    moe_renormalize: bool = True
    latent_moe_use_norm: bool = True

    # Dense layer
    first_k_dense_replace: int = 1
    intermediate_size: int = 33792

    # Block Attention Residuals (AttnRes)
    attn_res_block_size: int = 12

    # SiTU-GLU
    situ_beta: float = 4.0
    situ_linear_beta: float = 25.0

    # One-based full attention (MLA) layer indices
    full_attn_layers: List[int] = field(default_factory=list)

    def __post_init__(self):
        if not self.full_attn_layers:
            fa = [i for i in range(4, self.num_hidden_layers + 1) if i % 4 == 0]
            if self.num_hidden_layers not in fa:
                fa.append(self.num_hidden_layers)
            self.full_attn_layers = sorted(fa)

    def is_mla(self, layer_idx: int) -> bool:
        """layer_idx is 0-based; full_attn_layers is 1-based."""
        return (layer_idx + 1) in self.full_attn_layers

    def is_kda(self, layer_idx: int) -> bool:
        return not self.is_mla(layer_idx)

    def is_dense(self, layer_idx: int) -> bool:
        return layer_idx < self.first_k_dense_replace


def tiny_config(**kwargs) -> K3Config:
    """Helper for small test models exposing every mechanism."""
    c = dict(
        hidden_size=128,
        num_hidden_layers=13,
        vocab_size=256,
        rms_norm_eps=1e-5,
        kda_num_heads=4,
        kda_head_dim=16,
        short_conv_kernel_size=4,
        num_attention_heads=4,
        q_lora_rank=64,
        kv_lora_rank=32,
        qk_nope_head_dim=24,
        qk_rope_head_dim=8,
        v_head_dim=16,
        num_experts=8,
        num_experts_per_token=2,
        num_shared_experts=2,
        routed_expert_hidden_size=64,
        moe_intermediate_size=48,
        first_k_dense_replace=1,
        intermediate_size=96,
        attn_res_block_size=3,
        full_attn_layers=[4, 8, 12, 13],
    )
    c.update(kwargs)
    return K3Config(**c)


# --------------------------------------------------------------------------- MXFP4Linear
class MXFP4Linear(nn.Module):
    """Linear layer backed by packed MXFP4 weights with fused matmul.

    Replaces nn.Linear for routed MoE experts when loading real Kimi K3 weights.
    Uses matmul_mxfp4_fused for 7.5x less memory traffic per expert.

    Usage:
        # Convert nn.Linear to MXFP4Linear after loading checkpoint:
        layer = MXFP4Linear.from_packed(packed_bytes, scales, group=32)
        y = layer(x)  # fused matmul on packed nibbles
    """

    def __init__(self, packed: torch.Tensor, scales: torch.Tensor, group: int = 32):
        super().__init__()
        self.register_buffer("packed", packed)
        self.register_buffer("scales", scales)
        self.group = group
        self.out_features = packed.shape[0]
        self.in_features = packed.shape[1] * 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return matmul_mxfp4_fused(x, self.packed, self.scales, group=self.group)

    @staticmethod
    def from_weights(
        weight: torch.Tensor,
        group: int = 32,
    ) -> "MXFP4Linear":
        """Pack float32 weights into MXFP4 format and create a fused linear layer."""
        rows, cols = weight.shape
        pcols = cols // 2
        ngrp = cols // group

        # Quantize to MXFP4 (simple: find best E2M1 value per element)
        from kimipy.ops import E2M1_VALUES
        e2m1 = torch.tensor(E2M1_VALUES, dtype=torch.float32)

        w_abs = weight.abs().clamp(min=1e-10)
        # Find best E2M1 index for each element
        # Scale per group
        group_max = w_abs.reshape(rows, ngrp, group).abs().max(dim=-1).values
        scales = torch.zeros(rows, ngrp, dtype=torch.uint8)
        packed = torch.zeros(rows, pcols, dtype=torch.uint8)

        for r in range(rows):
            for g in range(ngrp):
                gmax = group_max[r, g].item()
                if gmax < 1e-10:
                    scales[r, g] = 255  # NaN scale
                    continue
                # Find scale: largest E2M1 value is 6.0
                scale_val = gmax / 6.0
                scale_byte = max(0, min(254, int(torch.log2(torch.tensor(scale_val)).item()) + 127))
                scales[r, g] = scale_byte
                s = 2.0 ** (scale_byte - 127)

                for k in range(group):
                    col = g * group + k
                    if col >= cols:
                        continue
                    val = weight[r, col].item() / s
                    # Find nearest E2M1 value
                    idx = (e2m1 - val).abs().argmin().item()
                    byte_idx = col // 2
                    if col % 2 == 0:
                        packed[r, byte_idx] = (packed[r, byte_idx] & 0xF0) | (idx & 0x0F)
                    else:
                        packed[r, byte_idx] = (packed[r, byte_idx] & 0x0F) | ((idx & 0x0F) << 4)

        return MXFP4Linear(packed, scales, group=group)


# --------------------------------------------------------------------------- AttnRes
def apply_attn_res(
    prefix_sum: torch.Tensor,
    block_residual: torch.Tensor,
    proj_w: torch.Tensor,
    norm_w: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Applies Block Attention Residual aggregation."""
    v = torch.cat([block_residual, prefix_sum.unsqueeze(1)], dim=1).float()
    k = v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + eps)
    scores = (k * (norm_w.float() * proj_w.float())).sum(-1)
    probs = F.softmax(scores, dim=-1).unsqueeze(1)
    return torch.matmul(probs, v).squeeze(1).to(dtype=prefix_sum.dtype)


# --------------------------------------------------------------------------- Layers
class KimiDeltaAttention(nn.Module):
    """KDA Attention Layer."""

    def __init__(self, c: K3Config):
        super().__init__()
        self.c = c
        H, D, E = c.kda_num_heads, c.kda_head_dim, c.hidden_size
        P = H * D
        self.H, self.D, self.P = H, D, P

        self.q_proj = nn.Linear(E, P, bias=False)
        self.k_proj = nn.Linear(E, P, bias=False)
        self.v_proj = nn.Linear(E, P, bias=False)
        self.q_conv1d = ShortConv(P, c.short_conv_kernel_size)
        self.k_conv1d = ShortConv(P, c.short_conv_kernel_size)
        self.v_conv1d = ShortConv(P, c.short_conv_kernel_size)

        self.f_a_proj = nn.Linear(E, D, bias=False)
        self.f_b_proj = nn.Linear(D, P, bias=False)
        self.A_log = nn.Parameter(torch.zeros(H))
        self.dt_bias = nn.Parameter(torch.zeros(P))
        self.b_proj = nn.Linear(E, H, bias=False)
        self.g_proj = nn.Linear(E, P, bias=False)
        self.o_norm = RMSNorm(D, c.rms_norm_eps)
        self.o_proj = nn.Linear(P, E, bias=False)

    def decay(self, x: torch.Tensor) -> torch.Tensor:
        H, D = self.H, self.D
        z = self.f_b_proj(self.f_a_proj(x)).float() + self.dt_bias.float()
        z = z.view(*z.shape[:-1], H, D)
        a = self.A_log.float().exp().view(H, 1)
        return self.c.gate_lower_bound * torch.sigmoid(a * z)

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        B, T, _ = x.shape
        H, D = self.H, self.D

        S = state[0] if state and state[0] is not None else x.new_zeros(B, H, D, D, dtype=torch.float32)
        cq = state[1] if state else None
        ck = state[2] if state else None
        cv = state[3] if state else None

        q, cq = self.q_conv1d(self.q_proj(x), cq)
        k, ck = self.k_conv1d(self.k_proj(x), ck)
        v, cv = self.v_conv1d(self.v_proj(x), cv)

        q = l2norm(q.view(B, T, H, D))
        k = l2norm(k.view(B, T, H, D))
        v = v.view(B, T, H, D)

        beta = torch.sigmoid(self.b_proj(x).float())
        g = self.decay(x)
        alpha = g.exp()

        q = q.float() * (D ** -0.5)
        k, v = k.float(), v.float()

        o = x.new_zeros(B, T, H, D, dtype=torch.float32)
        for t in range(T):
            S = S * alpha[:, t].unsqueeze(-1)
            u = torch.einsum("bhk,bhkv->bhv", k[:, t], S)
            w = beta[:, t].unsqueeze(-1) * (v[:, t] - u)
            S = S + torch.einsum("bhk,bhv->bhkv", k[:, t], w)
            o[:, t] = torch.einsum("bhk,bhkv->bhv", q[:, t], S)

        y = self.o_norm(o).view(B, T, self.P)
        y = y * torch.sigmoid(self.g_proj(x))
        return self.o_proj(y.to(dtype=x.dtype)), (S, cq, ck, cv)


class GatedMLA(nn.Module):
    """Gated Multi-Head Latent Attention Layer."""

    def __init__(self, c: K3Config):
        super().__init__()
        self.c = c
        E, Hn = c.hidden_size, c.num_attention_heads
        self.H = Hn
        self.qk_nope, self.qk_rope, self.vh = c.qk_nope_head_dim, c.qk_rope_head_dim, c.v_head_dim
        self.qh = self.qk_nope + self.qk_rope
        self.scale = self.qh ** -0.5

        self.q_a_proj = nn.Linear(E, c.q_lora_rank, bias=False)
        self.q_a_layernorm = RMSNorm(c.q_lora_rank, c.rms_norm_eps)
        self.q_b_proj = nn.Linear(c.q_lora_rank, Hn * self.qh, bias=False)
        self.kv_a_proj_with_mqa = nn.Linear(E, c.kv_lora_rank + self.qk_rope, bias=False)
        self.kv_a_layernorm = RMSNorm(c.kv_lora_rank, c.rms_norm_eps)
        self.kv_b_proj = nn.Linear(c.kv_lora_rank, Hn * (self.qk_nope + self.vh), bias=False)
        self.o_proj = nn.Linear(Hn * self.vh, E, bias=False)
        self.g_proj = nn.Linear(E, Hn * self.vh, bias=False) if c.mla_use_output_gate else None

    def forward(
        self,
        x: torch.Tensor,
        cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        B, T, _ = x.shape
        H = self.H

        q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(x))).view(B, T, H, self.qh)
        q_nope, q_rope = q.split([self.qk_nope, self.qk_rope], dim=-1)

        c_kv = self.kv_a_proj_with_mqa(x)
        latent, k_rope = c_kv.split([self.c.kv_lora_rank, self.qk_rope], dim=-1)
        latent = self.kv_a_layernorm(latent)

        if cache is not None and cache[0] is not None:
            latent = torch.cat([cache[0], latent], dim=1)
            k_rope = torch.cat([cache[1], k_rope], dim=1)
        new_cache = (latent, k_rope)

        kv = self.kv_b_proj(latent).view(B, -1, H, self.qk_nope + self.vh)
        k_nope, v = kv.split([self.qk_nope, self.vh], dim=-1)
        k_rope_b = k_rope.unsqueeze(2).expand(-1, -1, H, -1)

        qs = torch.cat([q_nope, q_rope], dim=-1).transpose(1, 2).float()
        ks = torch.cat([k_nope, k_rope_b], dim=-1).transpose(1, 2).float()
        vs = v.transpose(1, 2).float()

        att = torch.matmul(qs, ks.transpose(-1, -2)) * self.scale
        Tq, Tk = att.shape[-2], att.shape[-1]
        causal = torch.ones(Tq, Tk, dtype=torch.bool, device=x.device).tril(Tk - Tq)
        att = att.masked_fill(~causal, float("-inf")).softmax(-1)
        y = torch.matmul(att, vs).transpose(1, 2).reshape(B, T, H * self.vh)

        if self.g_proj is not None:
            y = y * torch.sigmoid(self.g_proj(x)).float()
        return self.o_proj(y.to(dtype=x.dtype)), new_cache


class Expert(nn.Module):
    """Feed-forward expert module."""

    def __init__(self, din: int, dinter: int):
        super().__init__()
        self.w1 = nn.Linear(din, dinter, bias=False)
        self.w3 = nn.Linear(din, dinter, bias=False)
        self.w2 = nn.Linear(dinter, din, bias=False)

    def forward(self, x: torch.Tensor, beta: float, linear_beta: float) -> torch.Tensor:
        return self.w2(situ_glu(torch.cat([self.w1(x), self.w3(x)], -1), beta, linear_beta))


class LatentMoE(nn.Module):
    """Latent Mixture of Experts (MoE) Layer."""

    def __init__(self, c: K3Config):
        super().__init__()
        self.c = c
        E, L, I = c.hidden_size, c.routed_expert_hidden_size, c.moe_intermediate_size  # noqa: E741
        self.gate = nn.Linear(E, c.num_experts, bias=False)
        self.e_score_correction_bias = nn.Parameter(torch.zeros(c.num_experts))
        self.down = nn.Linear(E, L, bias=False)
        self.up = nn.Linear(L, E, bias=False)
        self.norm = RMSNorm(L, c.rms_norm_eps) if c.latent_moe_use_norm else None
        self.experts = nn.ModuleList([Expert(L, I) for _ in range(c.num_experts)])
        self.shared = Expert(E, I * c.num_shared_experts)

    def route(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = F.linear(x.float(), self.gate.weight.float())
        scores = logits.sigmoid()
        biased = scores + self.e_score_correction_bias.float()
        idx = torch.topk(biased, self.c.num_experts_per_token, dim=-1, sorted=False).indices
        w = scores.gather(1, idx)
        if self.c.moe_renormalize:
            w = w / (w.sum(-1, keepdim=True) + 1e-20)
        return idx, w * self.c.routed_scaling_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, E = x.shape
        identity = x
        flat = x.view(-1, E)
        idx, w = self.route(flat)
        z = self.down(flat)
        out = torch.zeros_like(z)
        for n in range(flat.shape[0]):
            for j in range(idx.shape[1]):
                out[n] += w[n, j].to(z.dtype) * self.experts[idx[n, j]](
                    z[n : n + 1], self.c.situ_beta, self.c.situ_linear_beta
                )[0]
        if self.norm is not None:
            out = self.norm(out)
        y = self.up(out).view(B, T, E)
        return y + self.shared(identity, self.c.situ_beta, self.c.situ_linear_beta)


class DenseMLP(nn.Module):
    """Dense MLP for first layer."""

    def __init__(self, c: K3Config):
        super().__init__()
        self.c = c
        self.gate_proj = nn.Linear(c.hidden_size, c.intermediate_size, bias=False)
        self.up_proj = nn.Linear(c.hidden_size, c.intermediate_size, bias=False)
        self.down_proj = nn.Linear(c.intermediate_size, c.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g = torch.cat([self.gate_proj(x), self.up_proj(x)], dim=-1)
        return self.down_proj(situ_glu(g, self.c.situ_beta, self.c.situ_linear_beta))


class DecoderLayer(nn.Module):
    """Single Kimi K3 Decoder Layer."""

    def __init__(self, c: K3Config, idx: int):
        super().__init__()
        self.c, self.idx = c, idx
        self.is_mla = c.is_mla(idx)
        self.self_attn = GatedMLA(c) if self.is_mla else KimiDeltaAttention(c)
        self.mlp = DenseMLP(c) if c.is_dense(idx) else LatentMoE(c)
        self.input_layernorm = RMSNorm(c.hidden_size, c.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(c.hidden_size, c.rms_norm_eps)
        self.self_attention_res_norm = RMSNorm(c.hidden_size, c.rms_norm_eps)
        self.self_attention_res_proj = nn.Linear(c.hidden_size, 1, bias=False)
        self.mlp_res_norm = RMSNorm(c.hidden_size, c.rms_norm_eps)
        self.mlp_res_proj = nn.Linear(c.hidden_size, 1, bias=False)

    def forward(
        self,
        h: torch.Tensor,
        block_residual: torch.Tensor,
        state: Optional[Tuple] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Tuple]]:
        B, T, E = h.shape
        eps = self.c.rms_norm_eps
        prefix_sum = h

        if block_residual is not None and block_residual.shape[1] > 0:
            h = apply_attn_res(
                prefix_sum.view(-1, E),
                block_residual,
                self.self_attention_res_proj.weight.squeeze(0),
                self.self_attention_res_norm.weight,
                eps,
            ).view(B, T, E)

        if self.idx % self.c.attn_res_block_size == 0:
            block_residual = torch.cat(
                [block_residual, prefix_sum.view(-1, E).unsqueeze(1)], dim=1
            )
            prefix_sum = None

        h = self.input_layernorm(h)
        h, state = self.self_attn(h, state)
        prefix_sum = h if prefix_sum is None else prefix_sum + h

        h = apply_attn_res(
            prefix_sum.view(-1, E),
            block_residual,
            self.mlp_res_proj.weight.squeeze(0),
            self.mlp_res_norm.weight,
            eps,
        ).view(B, T, E)

        h = self.post_attention_layernorm(h)
        h = self.mlp(h)
        prefix_sum = h if prefix_sum is None else prefix_sum + h
        return prefix_sum, block_residual, state


# --------------------------------------------------------------------------- Full Model
class K3Model(nn.Module):
    """Full 93-layer Kimi K3 Model Runner."""

    def __init__(self, c: K3Config):
        super().__init__()
        self.c = c
        self.config = c
        self.embed_tokens = nn.Embedding(c.vocab_size, c.hidden_size)
        self.layers = nn.ModuleList([DecoderLayer(c, i) for i in range(c.num_hidden_layers)])
        self.norm = RMSNorm(c.hidden_size, c.rms_norm_eps)
        self.lm_head = nn.Linear(c.hidden_size, c.vocab_size, bias=False)
        self.output_attn_res_norm = RMSNorm(c.hidden_size, c.rms_norm_eps)
        self.output_attn_res_proj = nn.Linear(c.hidden_size, 1, bias=False)

    def forward(
        self,
        ids: torch.Tensor,
        states: Optional[List[Optional[Tuple]]] = None,
    ) -> Tuple[torch.Tensor, List[Optional[Tuple]]]:
        B, T = ids.shape
        h = self.embed_tokens(ids)
        block_residual = h.new_zeros(B * T, 0, self.c.hidden_size)
        states = states or [None] * len(self.layers)

        for i, layer in enumerate(self.layers):
            h, block_residual, states[i] = layer(h, block_residual, states[i])

        h = apply_attn_res(
            h.view(-1, self.c.hidden_size),
            block_residual,
            self.output_attn_res_proj.weight.squeeze(0),
            self.output_attn_res_norm.weight,
            self.c.rms_norm_eps,
        ).view(B, T, self.c.hidden_size)

        logits = self.lm_head(self.norm(h))
        return logits, states

    @torch.no_grad()
    def generate(
        self,
        ids: torch.Tensor,
        n_new: int = 16,
        incremental: bool = True,
    ) -> torch.Tensor:
        """Generates n_new tokens starting from prompt ids.

        Args:
            ids: Input token tensor [B, T]
            n_new: Number of new tokens to generate
            incremental: Use stateful incremental decoding if True
        """
        out = ids
        if not incremental:
            for _ in range(n_new):
                logits, _ = self.forward(out)
                next_id = logits[:, -1].argmax(-1, keepdim=True)
                out = torch.cat([out, next_id], dim=1)
            return out

        # Stateful incremental decode pass
        logits, states = self.forward(out)
        next_id = logits[:, -1].argmax(-1, keepdim=True)
        out = torch.cat([out, next_id], dim=1)

        for _ in range(n_new - 1):
            logits, states = self.forward(next_id, states)
            next_id = logits[:, -1].argmax(-1, keepdim=True)
            out = torch.cat([out, next_id], dim=1)

        return out
