"""
Fast Windows-compatible Safetensors header & tensor reader for kimipy.
Uses Python struct, json, binary offsets, and memory mapping without Linux system dependencies.
"""

from dataclasses import dataclass
import json
import mmap
import os
from pathlib import Path
import struct
from typing import Any, Dict, List, Optional, Tuple, Union


@dataclass
class TensorInfo:
    """Information metadata for a single tensor stored in a Safetensors file."""

    name: str
    dtype: str
    shape: List[int]
    data_offsets: Tuple[int, int]  # (start_offset, end_offset) relative to payload start
    file_path: Path
    payload_start_offset: int  # Byte offset where binary payload starts in the file

    @property
    def abs_offsets(self) -> Tuple[int, int]:
        """Absolute byte range in the file (start, end)."""
        return (
            self.payload_start_offset + self.data_offsets[0],
            self.payload_start_offset + self.data_offsets[1],
        )

    @property
    def size_bytes(self) -> int:
        """Total size in bytes of the tensor data."""
        return self.data_offsets[1] - self.data_offsets[0]

    @property
    def num_elements(self) -> int:
        """Total number of elements in the tensor shape."""
        count = 1
        for d in self.shape:
            count *= d
        return count


class SafetensorsReader:
    """
    Fast, Windows-compatible reader for Safetensors files.

    Supports reading single files, directories of sharded files, or index files
    (e.g., model.safetensors.index.json).
    """

    def __init__(
        self,
        path: Union[str, Path],
        mmap_mode: bool = True,
    ) -> None:
        self.path = Path(path)
        self.mmap_mode = mmap_mode
        self.tensors: Dict[str, TensorInfo] = {}
        self._file_handles: Dict[Path, Any] = {}
        self._mmaps: Dict[Path, mmap.mmap] = {}

        self._load()

    def _load(self) -> None:
        """Scan directory, index file, or single safetensors file."""
        if self.path.is_dir():
            index_path = self.path / "model.safetensors.index.json"
            if index_path.exists():
                self._load_from_index(index_path)
            else:
                for st_file in sorted(self.path.glob("*.safetensors")):
                    self._parse_safetensors_file(st_file)
        elif self.path.name.endswith(".index.json"):
            self._load_from_index(self.path)
        elif self.path.is_file():
            self._parse_safetensors_file(self.path)
        else:
            raise FileNotFoundError(f"Path does not exist: {self.path}")

    def _load_from_index(self, index_path: Path) -> None:
        """Load tensor mapping from model.safetensors.index.json."""
        with open(index_path, "r", encoding="utf-8") as f:
            index_data = json.load(f)

        weight_map = index_data.get("weight_map", {})
        unique_files = set(weight_map.values())
        base_dir = index_path.parent

        for rel_file in sorted(unique_files):
            st_file = base_dir / rel_file
            if st_file.exists():
                self._parse_safetensors_file(st_file)

    def _parse_safetensors_file(self, file_path: Path) -> None:
        """Parse 8-byte header length, JSON metadata header, and tensor offsets."""
        with open(file_path, "rb") as f:
            header_len_bytes = f.read(8)
            if len(header_len_bytes) < 8:
                raise ValueError(f"File too short to be a valid Safetensors file: {file_path}")

            header_len = struct.unpack("<Q", header_len_bytes)[0]
            if header_len > 100 * 1024 * 1024:  # Sanity check 100MB header limit
                raise ValueError(f"Safetensors header size unnaturally large ({header_len} bytes)")

            header_bytes = f.read(header_len)
            if len(header_bytes) < header_len:
                raise ValueError(f"Truncated header in file: {file_path}")

            header_json = json.loads(header_bytes.decode("utf-8"))
            payload_start = 8 + header_len

            for name, meta in header_json.items():
                if name == "__metadata__":
                    continue
                if isinstance(meta, dict) and "data_offsets" in meta:
                    info = TensorInfo(
                        name=name,
                        dtype=meta["dtype"],
                        shape=meta["shape"],
                        data_offsets=tuple(meta["data_offsets"]),
                        file_path=file_path.resolve(),
                        payload_start_offset=payload_start,
                    )
                    self.tensors[name] = info

    def _get_file_and_mmap(self, file_path: Path) -> Tuple[Any, Optional[mmap.mmap]]:
        """Get or open file handle and mmap object for given path."""
        file_path = file_path.resolve()
        if file_path not in self._file_handles:
            f = open(file_path, "rb")
            self._file_handles[file_path] = f
            if self.mmap_mode:
                # Windows mmap compatibility: access=mmap.ACCESS_READ
                mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
                self._mmaps[file_path] = mm

        return self._file_handles[file_path], self._mmaps.get(file_path)

    def get_tensor_info(self, tensor_name: str) -> TensorInfo:
        """Get TensorInfo metadata for a tensor by name."""
        if tensor_name not in self.tensors:
            raise KeyError(f"Tensor '{tensor_name}' not found in Safetensors reader.")
        return self.tensors[tensor_name]

    def get_tensor_bytes(self, tensor_name: str) -> bytes:
        """
        Read raw binary bytes for a tensor without parsing data types.
        Works natively on Windows via binary file seek/read or mmap slice.
        """
        info = self.get_tensor_info(tensor_name)
        start_offset, end_offset = info.abs_offsets
        size = info.size_bytes

        f, mm = self._get_file_and_mmap(info.file_path)

        if mm is not None:
            # Slicing mmap object returns a bytes copy on Windows
            return mm[start_offset:end_offset]

        # Standard file seek + read (portable & Windows safe)
        f.seek(start_offset, os.SEEK_SET)
        data = f.read(size)
        if len(data) != size:
            raise IOError(
                f"Failed to read complete tensor data for {tensor_name} from {info.file_path}"
            )
        return data

    def get_tensor_data(
        self,
        tensor_name: str,
        return_type: str = "bytes",
    ) -> Any:
        """
        Extract tensor data.

        Args:
            tensor_name: Name of tensor to read.
            return_type: 'bytes', 'numpy', or 'torch'.
        """
        raw_bytes = self.get_tensor_bytes(tensor_name)
        info = self.get_tensor_info(tensor_name)

        if return_type == "bytes":
            return raw_bytes

        if return_type == "numpy":
            return self._to_numpy(raw_bytes, info)

        if return_type == "torch":
            return self._to_torch(raw_bytes, info)

        raise ValueError(f"Unsupported return_type: {return_type}. Choose 'bytes', 'numpy', or 'torch'.")

    def _to_numpy(self, raw_bytes: bytes, info: TensorInfo) -> Any:
        """Convert raw bytes to NumPy array."""
        try:
            import numpy as np
        except ImportError as err:
            raise ImportError("NumPy is required when return_type='numpy'") from err

        dtype_map = {
            "F64": np.float64,
            "F32": np.float32,
            "F16": np.float16,
            "I64": np.int64,
            "I32": np.int32,
            "I16": np.int16,
            "I8": np.int8,
            "U8": np.uint8,
        }

        # Check bfloat16 support in numpy
        if info.dtype == "BF16":
            if hasattr(np, "bfloat16"):
                np_dtype = np.bfloat16
            else:
                # Fallback to uint16 array if numpy lacks bfloat16
                np_dtype = np.uint16
        else:
            np_dtype = dtype_map.get(info.dtype)

        if np_dtype is None:
            raise ValueError(f"Unsupported Safetensors dtype for NumPy conversion: {info.dtype}")

        arr = np.frombuffer(raw_bytes, dtype=np_dtype)
        if info.shape:
            arr = arr.reshape(info.shape)
        return arr

    def _to_torch(self, raw_bytes: bytes, info: TensorInfo) -> Any:
        """Convert raw bytes to PyTorch tensor."""
        try:
            import torch
        except ImportError as err:
            raise ImportError("PyTorch is required when return_type='torch'") from err

        dtype_map = {
            "F64": torch.float64,
            "F32": torch.float32,
            "F16": torch.float16,
            "BF16": torch.bfloat16,
            "I64": torch.int64,
            "I32": torch.int32,
            "I16": torch.int16,
            "I8": torch.int8,
            "U8": torch.uint8,
        }

        torch_dtype = dtype_map.get(info.dtype)
        if torch_dtype is None:
            raise ValueError(f"Unsupported Safetensors dtype for PyTorch conversion: {info.dtype}")

        if isinstance(raw_bytes, (bytes, memoryview)):
            raw_bytes = bytearray(raw_bytes)
        tensor = torch.frombuffer(raw_bytes, dtype=torch_dtype)
        if info.shape:
            tensor = tensor.reshape(info.shape)
        return tensor

    def close(self) -> None:
        """Close all opened mmap objects and file handles."""
        for mm in self._mmaps.values():
            try:
                mm.close()
            except Exception:
                pass
        self._mmaps.clear()

        for f in self._file_handles.values():
            try:
                f.close()
            except Exception:
                pass
        self._file_handles.clear()

    def __enter__(self) -> "SafetensorsReader":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()


class DirectReader:
    """Direct I/O reader that bypasses the OS page cache for large sequential reads.

    On Linux: uses O_DIRECT flag for unbuffered reads.
    On Windows: uses FILE_FLAG_NO_BUFFERING via ctypes.
    Falls back to regular mmap if direct I/O is unavailable.

    Direct I/O requires aligned reads (typically 4096-byte alignment).
    The reader handles alignment padding transparently.

    Usage:
        with DirectReader("path/to/file") as reader:
            data = reader.read_aligned(offset, length)
    """

    ALIGNMENT = 4096

    def __init__(self, path: Union[str, Path]) -> None:
        self.path = Path(path)
        self._fd = None
        self._use_direct = False
        self._open()

    def _open(self) -> None:
        """Open file with direct I/O if possible, fall back to buffered."""
        import sys

        if sys.platform == "linux":
            try:
                O_DIRECT = getattr(os, "O_DIRECT", 0)
                self._fd = os.open(str(self.path), os.O_RDONLY | O_DIRECT)
                self._use_direct = True
            except (OSError, AttributeError):
                self._fd = os.open(str(self.path), os.O_RDONLY)
                self._use_direct = False

        elif sys.platform == "win32":
            try:
                import ctypes
                from ctypes import wintypes

                kernel32 = ctypes.windll.kernel32
                FILE_FLAG_NO_BUFFERING = 0x20000000
                FILE_FLAG_SEQUENTIAL_SCAN = 0x08000000
                GENERIC_READ = 0x80000000
                FILE_SHARE_READ = 0x00000001
                OPEN_EXISTING = 3

                handle = kernel32.CreateFileW(
                    str(self.path),
                    GENERIC_READ,
                    FILE_SHARE_READ,
                    None,
                    OPEN_EXISTING,
                    FILE_FLAG_NO_BUFFERING | FILE_FLAG_SEQUENTIAL_SCAN,
                    None,
                )
                if handle != -1:
                    self._fd = handle
                    self._use_direct = True
                else:
                    self._fd = os.open(str(self.path), os.O_RDONLY)
                    self._use_direct = False
            except Exception:
                self._fd = os.open(str(self.path), os.O_RDONLY)
                self._use_direct = False
        else:
            self._fd = os.open(str(self.path), os.O_RDONLY)
            self._use_direct = False

    def read_aligned(self, offset: int, length: int) -> bytes:
        """Read bytes with alignment padding for direct I/O.

        Args:
            offset: Byte offset to read from.
            length: Number of bytes to read.

        Returns:
            Requested bytes (alignment padding stripped).
        """
        if not self._use_direct:
            os.lseek(self._fd, offset, os.SEEK_SET)
            return os.read(self._fd, length)

        # Align to block boundary
        aligned_off = offset & ~(self.ALIGNMENT - 1)
        aligned_end = (offset + length + self.ALIGNMENT - 1) & ~(self.ALIGNMENT - 1)
        aligned_len = aligned_end - aligned_off

        # Read aligned chunk
        os.lseek(self._fd, aligned_off, os.SEEK_SET)
        buf = os.read(self._fd, aligned_len)

        # Strip alignment padding
        pad = offset - aligned_off
        return buf[pad:pad + length]

    @property
    def uses_direct_io(self) -> bool:
        """Whether direct I/O is active."""
        return self._use_direct

    def close(self) -> None:
        """Close the file descriptor."""
        if self._fd is not None:
            if isinstance(self._fd, int) and self._fd > 0:
                os.close(self._fd)
            self._fd = None

    def __enter__(self) -> "DirectReader":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()
