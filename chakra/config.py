"""
Config parser and dataclass for Kimi K3 architecture in chakra.
Supports both nested HuggingFace format (config.json) and flat fixture format (ref_k3.json).
"""

from dataclasses import dataclass, field, fields
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


@dataclass
class KimiConfig:
    """Configuration dataclass for Kimi K3 architecture defaults."""

    hidden_size: int = 7168
    num_hidden_layers: int = 93
    vocab_size: int = 163840
    rms_norm_eps: float = 1e-5
    tie_word_embeddings: bool = False

    # ---- KDA (Linear Attention) ----
    kda_num_heads: int = 96
    kda_head_dim: int = 128
    short_conv_kernel_size: int = 4
    gate_lower_bound: float = -5.0

    # ---- Gated MLA ----
    num_attention_heads: int = 96
    q_lora_rank: int = 1536
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    mla_use_output_gate: bool = True

    # ---- MoE (Stable LatentMoE) ----
    num_experts: int = 896
    num_experts_per_token: int = 16
    num_shared_experts: int = 2
    routed_expert_hidden_size: int = 3584
    moe_intermediate_size: int = 3072
    routed_scaling_factor: float = 1.0
    moe_renormalize: bool = True
    latent_moe_use_norm: bool = True

    # ---- Dense Layer ----
    first_k_dense_replace: int = 1
    intermediate_size: int = 33792

    # ---- AttnRes ----
    attn_res_block_size: int = 12

    # ---- SiTU-GLU ----
    activation_situ_beta: float = 4.0
    activation_situ_linear_beta: float = 25.0

    # ---- Layer Map (1-based indices in config) ----
    full_attn_layers: List[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Initialize layer map if not specified."""
        if not self.full_attn_layers:
            # 3 KDA then 1 MLA repeating, plus a trailing MLA at the end
            fa = [i for i in range(4, self.num_hidden_layers + 1) if i % 4 == 0]
            if self.num_hidden_layers not in fa:
                fa.append(self.num_hidden_layers)
            self.full_attn_layers = sorted(fa)

    def is_mla(self, layer_idx: int) -> bool:
        """Check if layer index (0-based) is a full-attention (MLA) layer."""
        return (layer_idx + 1) in self.full_attn_layers

    def is_kda(self, layer_idx: int) -> bool:
        """Check if layer index (0-based) is a KDA layer."""
        return not self.is_mla(layer_idx)

    def is_dense(self, layer_idx: int) -> bool:
        """Check if layer index (0-based) is a dense MLP layer."""
        return layer_idx < self.first_k_dense_replace

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary representation."""
        return {f.name: getattr(self, f.name) for f in fields(self)}


def _lookup_key(
    sources: List[Dict[str, Any]], primary: str, alias: Optional[str] = None
) -> Optional[Any]:
    """Look up key across nested candidate dictionaries in order."""
    names = [primary]
    if alias:
        names.append(alias)

    for src in sources:
        if not isinstance(src, dict):
            continue
        for name in names:
            if name in src:
                return src[name]
    return None


def load_config(path: Union[str, Path]) -> KimiConfig:
    """
    Load KimiConfig from JSON file.

    Supports both:
    1. Released nested format (`config.json` with `text_config` and `linear_attn_config`)
    2. Flat fixture format (`ref_k3.json`)
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        root_data = json.load(f)

    if not isinstance(root_data, dict):
        raise ValueError(f"Invalid JSON content in {config_path}: expected object")

    # Search candidates in precedence: text_config.linear_attn_config, text_config,
    # config.linear_attn_config, config, root_data
    sources = []
    text_config = root_data.get("text_config")
    if isinstance(text_config, dict):
        if isinstance(text_config.get("linear_attn_config"), dict):
            sources.append(text_config["linear_attn_config"])
        sources.append(text_config)

    config_sub = root_data.get("config")
    if isinstance(config_sub, dict):
        if isinstance(config_sub.get("linear_attn_config"), dict):
            sources.append(config_sub["linear_attn_config"])
        sources.append(config_sub)

    linear_attn_config = root_data.get("linear_attn_config")
    if isinstance(linear_attn_config, dict):
        sources.append(linear_attn_config)

    sources.append(root_data)

    # Clean non-dict entries
    sources = [s for s in sources if isinstance(s, dict)]

    cfg = KimiConfig()

    # Map fields with candidates
    mapping = {
        "hidden_size": ("hidden_size", None),
        "num_hidden_layers": ("num_hidden_layers", None),
        "vocab_size": ("vocab_size", None),
        "rms_norm_eps": ("rms_norm_eps", None),
        "tie_word_embeddings": ("tie_word_embeddings", None),
        "kda_num_heads": ("kda_num_heads", "num_heads"),
        "kda_head_dim": ("kda_head_dim", "head_dim"),
        "short_conv_kernel_size": ("short_conv_kernel_size", None),
        "gate_lower_bound": ("gate_lower_bound", None),
        "num_attention_heads": ("num_attention_heads", None),
        "q_lora_rank": ("q_lora_rank", None),
        "kv_lora_rank": ("kv_lora_rank", None),
        "qk_nope_head_dim": ("qk_nope_head_dim", None),
        "qk_rope_head_dim": ("qk_rope_head_dim", None),
        "v_head_dim": ("v_head_dim", None),
        "mla_use_output_gate": ("mla_use_output_gate", None),
        "num_experts": ("num_experts", None),
        "num_experts_per_token": ("num_experts_per_token", None),
        "num_shared_experts": ("num_shared_experts", None),
        "routed_expert_hidden_size": ("routed_expert_hidden_size", None),
        "moe_intermediate_size": ("moe_intermediate_size", None),
        "routed_scaling_factor": ("routed_scaling_factor", None),
        "moe_renormalize": ("moe_renormalize", None),
        "latent_moe_use_norm": ("latent_moe_use_norm", None),
        "first_k_dense_replace": ("first_k_dense_replace", None),
        "intermediate_size": ("intermediate_size", None),
        "attn_res_block_size": ("attn_res_block_size", None),
        "activation_situ_beta": ("activation_situ_beta", "situ_beta"),
        "activation_situ_linear_beta": (
            "activation_situ_linear_beta",
            "situ_linear_beta",
        ),
        "full_attn_layers": ("full_attn_layers", None),
    }

    for attr_name, (primary, alias) in mapping.items():
        val = _lookup_key(sources, primary, alias)
        if val is not None:
            if attr_name == "full_attn_layers" and isinstance(val, list):
                val = [int(x) for x in val]
            setattr(cfg, attr_name, val)

    # Perform basic post-init and validation
    cfg.__post_init__()

    if cfg.num_hidden_layers <= 0 or cfg.hidden_size <= 0 or cfg.vocab_size <= 0:
        raise ValueError(
            f"Invalid configuration in {config_path}: num_hidden_layers, hidden_size, "
            "and vocab_size must be positive integers."
        )

    return cfg
