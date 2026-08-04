"""
TrunkStreamer module for chakra.
Implements single-layer trunk streaming for 8GB RAM systems on Windows.
Streams non-routed layer trunk parameters layer-by-layer to minimize peak memory consumption.
"""

import gc
from pathlib import Path
import re
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

from .config import KimiConfig
from .st_reader import SafetensorsReader, TensorInfo


class TrunkStreamer:
    """
    Manages single-layer trunk parameter streaming for Kimi K3 on Windows.

    Categorizes model weights into global parameters (embeddings, final norm, lm_head)
    and single-layer trunk parameters (RMSNorms, MLA/KDA attention, shared experts),
    excluding routed MoE expert weights.
    """

    def __init__(
        self,
        model_source: Union[str, Path, SafetensorsReader],
        config: Optional[KimiConfig] = None,
    ) -> None:
        if isinstance(model_source, SafetensorsReader):
            self.reader = model_source
            self._owns_reader = False
        else:
            self.reader = SafetensorsReader(model_source)
            self._owns_reader = True

        self.config = config or KimiConfig()

        self.global_tensors: Dict[str, TensorInfo] = {}
        self.layer_trunks: Dict[int, List[TensorInfo]] = {}

        self._categorize_tensors()

    def _is_routed_expert_tensor(self, tensor_name: str) -> bool:
        """Check if tensor belongs to routed MoE experts (not shared experts)."""
        # Matches patterns like `mlp.experts.0...` or `experts.12...`
        # But avoids `shared_experts`
        if "shared_expert" in tensor_name:
            return False
        return bool(re.search(r"\bexperts?\.\d+", tensor_name))

    def _extract_layer_index(self, tensor_name: str) -> Optional[int]:
        """Extract 0-based layer index from tensor name if present."""
        match = re.search(r"\blayers?\.\s*(\d+)\b", tensor_name)
        if match:
            return int(match.group(1))
        return None

    def _categorize_tensors(self) -> None:
        """Group tensors into global or layer-specific trunk sets."""
        for name, info in self.reader.tensors.items():
            if self._is_routed_expert_tensor(name):
                # Exclude routed experts from trunk streamer
                continue

            layer_idx = self._extract_layer_index(name)
            if layer_idx is not None:
                if layer_idx not in self.layer_trunks:
                    self.layer_trunks[layer_idx] = []
                self.layer_trunks[layer_idx].append(info)
            else:
                self.global_tensors[name] = info

    def get_layer_trunk_info(self, layer_idx: int) -> List[TensorInfo]:
        """Get list of TensorInfo for trunk parameters of layer layer_idx."""
        return self.layer_trunks.get(layer_idx, [])

    def get_layer_trunk_bytes_size(self, layer_idx: int) -> int:
        """Calculate total byte footprint of trunk parameters for layer layer_idx."""
        return sum(info.size_bytes for info in self.get_layer_trunk_info(layer_idx))

    def load_layer_trunk(
        self,
        layer_idx: int,
        return_type: str = "bytes",
    ) -> Dict[str, Any]:
        """
        Load all trunk parameters for a single layer into memory.

        Args:
            layer_idx: 0-based layer index.
            return_type: 'bytes', 'numpy', or 'torch'.
        """
        infos = self.get_layer_trunk_info(layer_idx)
        if not infos:
            raise KeyError(f"No trunk tensors found for layer index {layer_idx}")

        layer_data: Dict[str, Any] = {}
        for info in infos:
            layer_data[info.name] = self.reader.get_tensor_data(
                info.name, return_type=return_type
            )
        return layer_data

    def load_embed_tokens(self, return_type: str = "bytes") -> Dict[str, Any]:
        """Load word embedding tensor(s)."""
        embed_data: Dict[str, Any] = {}
        for name in self.global_tensors:
            if "embed_tokens" in name or "vocab_embed" in name:
                embed_data[name] = self.reader.get_tensor_data(
                    name, return_type=return_type
                )
        return embed_data

    def load_final_norm_and_head(self, return_type: str = "bytes") -> Dict[str, Any]:
        """Load final norm and LM head tensor(s)."""
        head_data: Dict[str, Any] = {}
        for name in self.global_tensors:
            if "norm" in name or "lm_head" in name:
                head_data[name] = self.reader.get_tensor_data(
                    name, return_type=return_type
                )
        return head_data

    def stream_layers(
        self,
        return_type: str = "bytes",
    ) -> Iterator[Tuple[int, Dict[str, Any]]]:
        """
        Generator yielding (layer_idx, trunk_tensors_dict) sequentially.
        Forces garbage collection between layers to maintain tight 8GB RAM footprint.
        """
        sorted_layers = sorted(self.layer_trunks.keys())
        for layer_idx in sorted_layers:
            layer_dict = self.load_layer_trunk(layer_idx, return_type=return_type)
            yield layer_idx, layer_dict
            # Explicit cleanup hint for low memory systems
            del layer_dict
            gc.collect()

    def close(self) -> None:
        """Close underlying reader if owned."""
        if self._owns_reader:
            self.reader.close()

    def __enter__(self) -> "TrunkStreamer":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()


class RingBufferStreamer:
    """Pinned prefix + ring slot trunk streaming for low-memory systems.

    Memory layout:
        pinned_layers[0..N-1]: always resident in RAM, read once
        ring_slot: one layer slot, overwritten as we stream through remaining layers

    At the laptop preset (3 GB trunk budget), ~10 of 93 layers are pinned.
    This avoids re-reading the first N layers from disk on every token.

    Usage:
        with RingBufferStreamer(model_dir, trunk_gb=3.0) as streamer:
            for layer_idx, layer_data in streamer.stream():
                # pinned layers return instantly (no I/O)
                # streamed layers read from disk into ring slot
                model.forward(layer_idx, layer_data)
    """

    def __init__(
        self,
        model_source: Union[str, Path, SafetensorsReader],
        config: Optional[KimiConfig] = None,
        trunk_gb: float = 3.0,
        return_type: str = "torch",
    ) -> None:
        self._base = TrunkStreamer(model_source, config)
        self.config = self._base.config
        self.trunk_gb = trunk_gb
        self.return_type = return_type

        # Estimate layer size from the first layer
        self._layer_size_bytes = self._estimate_layer_size()

        # Calculate how many layers we can pin
        budget_bytes = int(trunk_gb * 1e9)
        if self._layer_size_bytes > 0:
            # Reserve one slot for the ring buffer
            self.n_pinned = max(0, min(
                len(self._base.layer_trunks),
                (budget_bytes - self._layer_size_bytes) // self._layer_size_bytes
            ))
        else:
            self.n_pinned = 0

        self.pinned_cache: Dict[int, Dict[str, Any]] = {}
        self.ring_slot: Optional[Dict[str, Any]] = None

    def _estimate_layer_size(self) -> int:
        """Estimate byte size of one layer's trunk parameters."""
        if not self._base.layer_trunks:
            return 0
        first_layer = min(self._base.layer_trunks.keys())
        return self._base.get_layer_trunk_bytes_size(first_layer)

    def _ensure_pinned(self) -> None:
        """Load pinned layers into RAM if not already cached."""
        if self.pinned_cache:
            return
        for i in range(self.n_pinned):
            if i in self._base.layer_trunks:
                self.pinned_cache[i] = self._base.load_layer_trunk(
                    i, return_type=self.return_type
                )

    def stream(self) -> Iterator[Tuple[int, Dict[str, Any]]]:
        """Yield (layer_idx, trunk_data) with pinned prefix + ring slot."""
        self._ensure_pinned()
        sorted_layers = sorted(self._base.layer_trunks.keys())

        for layer_idx in sorted_layers:
            if layer_idx in self.pinned_cache:
                yield layer_idx, self.pinned_cache[layer_idx]
            else:
                self.ring_slot = self._base.load_layer_trunk(
                    layer_idx, return_type=self.return_type
                )
                yield layer_idx, self.ring_slot
                del self.ring_slot
                self.ring_slot = None
                gc.collect()

    @property
    def n_layers(self) -> int:
        return len(self._base.layer_trunks)

    @property
    def global_tensors(self) -> Dict[str, TensorInfo]:
        return self._base.global_tensors

    def get_stats(self) -> Dict[str, Any]:
        """Return streaming statistics."""
        pinned_bytes = sum(
            self._base.get_layer_trunk_bytes_size(i)
            for i in range(self.n_pinned)
        )
        return {
            "n_pinned": self.n_pinned,
            "n_streamed": self.n_layers - self.n_pinned,
            "pinned_bytes": pinned_bytes,
            "pinned_gb": pinned_bytes / 1e9,
            "layer_size_bytes": self._layer_size_bytes,
            "trunk_budget_gb": self.trunk_gb,
        }

    def close(self) -> None:
        """Release pinned cache and close reader."""
        self.pinned_cache.clear()
        self._base.close()

    def __enter__(self) -> "RingBufferStreamer":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()
