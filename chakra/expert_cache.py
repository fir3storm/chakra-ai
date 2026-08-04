"""
ExpertLRUCache module for chakra.
Manages routed MoE experts in an LRU (Least Recently Used) memory pool on demand,
ensuring memory stays strictly within allocated bounds on Windows hardware with 8GB RAM constraints.
"""

from collections import OrderedDict
import gc
import sys
from typing import Any, Callable, Dict, Iterable, Optional, Tuple, Union

from .st_reader import SafetensorsReader


def _estimate_size_bytes(obj: Any) -> int:
    """Estimate memory size in bytes for arbitrary data object."""
    if isinstance(obj, (bytes, bytearray, memoryview)):
        return len(obj)
    if hasattr(obj, "nbytes"):  # NumPy arrays or PyTorch tensors
        return int(obj.nbytes)
    if hasattr(obj, "element_size") and hasattr(obj, "nelement"):  # PyTorch tensor
        return int(obj.element_size() * obj.nelement())
    if isinstance(obj, dict):
        return sum(_estimate_size_bytes(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return sum(_estimate_size_bytes(v) for v in obj)
    return sys.getsizeof(obj)


class ExpertLRUCache:
    """
    LRU Memory Cache for routed MoE experts in Kimi K3.

    Stores active experts (layer_idx, expert_idx) in an ordered LRU dictionary
    and automatically evicts least-recently-used experts when capacity limits are reached.
    """

    def __init__(
        self,
        capacity_bytes: int = 2 * 1024 * 1024 * 1024,  # Default 2GB expert cache limit
        max_experts: int = 256,
        reader: Optional[SafetensorsReader] = None,
    ) -> None:
        self.capacity_bytes = capacity_bytes
        self.max_experts = max_experts
        self.reader = reader

        self._cache: OrderedDict[Tuple[int, int], Any] = OrderedDict()
        self._expert_sizes: Dict[Tuple[int, int], int] = {}
        self._current_bytes: int = 0

        self._hits: int = 0
        self._misses: int = 0
        self._evictions: int = 0

    def contains(self, layer_idx: int, expert_idx: int) -> bool:
        """Check if (layer_idx, expert_idx) is present in cache."""
        return (layer_idx, expert_idx) in self._cache

    def get(
        self,
        layer_idx: int,
        expert_idx: int,
        fetch_fn: Optional[Callable[[int, int], Any]] = None,
        return_type: str = "bytes",
    ) -> Any:
        """
        Retrieve expert parameters for (layer_idx, expert_idx).

        Fetches from callback or reader on cache miss.
        """
        key = (layer_idx, expert_idx)

        if key in self._cache:
            self._cache.move_to_end(key)
            self._hits += 1
            return self._cache[key]

        self._misses += 1

        # Fetch expert data
        if fetch_fn is not None:
            data = fetch_fn(layer_idx, expert_idx)
        elif self.reader is not None:
            data = self._fetch_from_reader(layer_idx, expert_idx, return_type=return_type)
        else:
            raise KeyError(
                f"Expert (layer {layer_idx}, expert {expert_idx}) not in cache and no fetch mechanism provided."
            )

        if data is not None:
            self.put(layer_idx, expert_idx, data)

        return data

    def put(
        self,
        layer_idx: int,
        expert_idx: int,
        data: Any,
        size_bytes: Optional[int] = None,
    ) -> None:
        """
        Store expert data into LRU cache, evicting oldest experts if limits are exceeded.
        """
        key = (layer_idx, expert_idx)
        if size_bytes is None:
            size_bytes = _estimate_size_bytes(data)

        # If updating existing key
        if key in self._cache:
            self._current_bytes -= self._expert_sizes[key]
            self._cache.move_to_end(key)
        else:
            # Evict until room is available
            while (
                self._cache
                and (
                    self._current_bytes + size_bytes > self.capacity_bytes
                    or len(self._cache) >= self.max_experts
                )
            ):
                evicted_key, _ = self._cache.popitem(last=False)
                evicted_size = self._expert_sizes.pop(evicted_key, 0)
                self._current_bytes -= evicted_size
                self._evictions += 1

        self._cache[key] = data
        self._expert_sizes[key] = size_bytes
        self._current_bytes += size_bytes

    def prefetch_experts(
        self,
        layer_idx: int,
        expert_indices: Iterable[int],
        fetch_fn: Optional[Callable[[int, int], Any]] = None,
        return_type: str = "bytes",
    ) -> Dict[int, Any]:
        """
        Batch fetch and cache active top-k experts for a specific layer.
        """
        results: Dict[int, Any] = {}
        for expert_idx in expert_indices:
            results[expert_idx] = self.get(
                layer_idx, expert_idx, fetch_fn=fetch_fn, return_type=return_type
            )
        return results

    def _fetch_from_reader(
        self,
        layer_idx: int,
        expert_idx: int,
        return_type: str = "bytes",
    ) -> Dict[str, Any]:
        """Load tensors for (layer_idx, expert_idx) directly from SafetensorsReader."""
        if self.reader is None:
            raise RuntimeError("SafetensorsReader is not configured for ExpertLRUCache")

        # Pattern: find tensors matching layer {layer_idx} and expert {expert_idx}
        layer_pattern = f".layers.{layer_idx}."
        expert_pattern = f".experts.{expert_idx}."

        expert_tensors: Dict[str, Any] = {}
        for tensor_name in self.reader.tensors.keys():
            if layer_pattern in tensor_name and expert_pattern in tensor_name:
                expert_tensors[tensor_name] = self.reader.get_tensor_data(
                    tensor_name, return_type=return_type
                )

        if not expert_tensors:
            raise KeyError(
                f"No tensor weights found for layer {layer_idx}, expert {expert_idx} in Safetensors reader."
            )

        return expert_tensors

    def clear(self) -> None:
        """Purge all cached experts and release memory."""
        self._cache.clear()
        self._expert_sizes.clear()
        self._current_bytes = 0
        gc.collect()

    def get_stats(self) -> Dict[str, Union[int, float]]:
        """Return cache hit/miss statistics and current memory usage."""
        total_requests = self._hits + self._misses
        hit_rate = (self._hits / total_requests) if total_requests > 0 else 0.0

        return {
            "hits": self._hits,
            "misses": self._misses,
            "evictions": self._evictions,
            "hit_rate": hit_rate,
            "cached_experts": len(self._cache),
            "current_bytes": self._current_bytes,
            "capacity_bytes": self.capacity_bytes,
        }

    def __len__(self) -> int:
        return len(self._cache)
