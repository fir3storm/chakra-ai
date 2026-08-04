"""
MXFP4 Dequantization module for kimipy.
Supports fast dequantization of 4-bit MXFP4 packed weight matrices
to NumPy arrays and PyTorch tensors.
"""

from typing import Any
import numpy as np

# E2M1 lookup table for MXFP4 (4-bit nibbles -> floating point representation)
E2M1_TABLE = np.array(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=np.float32,
)


def dequantize_mxfp4_numpy(
    packed_weights: np.ndarray,
    scale_factors: np.ndarray,
    group_size: int = 32,
) -> np.ndarray:
    """
    Dequantize MXFP4 packed weights and scale factors to float32 NumPy array.

    Args:
        packed_weights: 2D uint8 NumPy array of shape (rows, packed_cols).
        scale_factors: 2D uint8 NumPy array of shape (rows, scale_cols).
        group_size: Quantization group size (default 32 weights per scale byte).

    Returns:
        Dequantized 2D float32 NumPy array of shape (rows, packed_cols * 2).
    """
    if packed_weights.ndim != 2 or scale_factors.ndim != 2:
        raise ValueError("packed_weights and scale_factors must be 2D arrays.")

    rows, packed_cols = packed_weights.shape
    lo_nibbles = packed_weights & 0x0F
    hi_nibbles = (packed_weights >> 4) & 0x0F

    out = np.empty((rows, packed_cols * 2), dtype=np.float32)
    out[:, 0::2] = E2M1_TABLE[lo_nibbles]
    out[:, 1::2] = E2M1_TABLE[hi_nibbles]

    # Calculate multiplier: if scale == 255 -> 0.0, else 2^(scale - 127)
    scales_int = scale_factors.astype(np.int32)
    mult = np.where(scale_factors == 255, 0.0, np.exp2(scales_int - 127)).astype(np.float32)

    # Repeat mult for group_size elements along axis 1
    expanded_mult = np.repeat(mult, group_size, axis=1)[:, : packed_cols * 2]
    out *= expanded_mult
    return out


def dequantize_mxfp4_torch(
    packed_weights: Any,
    scale_factors: Any,
    group_size: int = 32,
) -> Any:
    """
    Dequantize MXFP4 packed weights and scale factors to PyTorch float32 tensor.
    """
    import torch

    if not isinstance(packed_weights, torch.Tensor):
        packed_weights = torch.from_numpy(np.asarray(packed_weights))
    if not isinstance(scale_factors, torch.Tensor):
        scale_factors = torch.from_numpy(np.asarray(scale_factors))

    np_out = dequantize_mxfp4_numpy(
        packed_weights.cpu().numpy(),
        scale_factors.cpu().numpy(),
        group_size=group_size,
    )
    return torch.from_numpy(np_out)
