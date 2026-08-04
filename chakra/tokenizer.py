# chakra/tokenizer.py
"""
Offline Tiktoken BPE and UTF-8 Byte Fallback Tokenizer for Kimi K3.

Features:
- 100% offline local execution with zero network requirements.
- Tiktoken BPE tokenizer loading from `tiktoken.model` and `tokenizer_config.json`.
- Smart fallback UTF-8 byte tokenizer for synthetic/tiny test runs or raw text mode,
  ensuring inputs and outputs decode into clean, human-readable text strings.
- Chat formatting helpers for system, user, and assistant turns (`format_chat_prompt`).
"""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Sequence, Set

# Default split pattern string for Kimi K3 BPE tokenizer
DEFAULT_PAT_STR = (
    r"""[\p{Han}]+|(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}\p{Han}]?\p{L}+"""
    r"""|\p{N}{1,3}| ?[^\s\p{L}\p{N}\p{Han}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""
)

# Standard special tokens for Kimi K3
DEFAULT_SPECIAL_TOKENS: Dict[str, int] = {
    "<|im_start|>": 163584,
    "<|im_end|>": 163585,
    "<|endoftext|>": 163586,
}


def find_tokenizer_files(model_dir: str | Path | None = None) -> Path | None:
    """
    Locates directory containing tiktoken.model and tokenizer_config.json locally.
    Order of candidate search:
    1. Explicit model_dir argument
    2. K3_TOK_FILES environment variable
    3. Project relative locations (kimi_k3_hf/files, ../kimi_k3_hf/files)
    4. User home directory locations (~/k3/hf, ~/k3model)
    """
    candidates: list[Path] = []

    if model_dir is not None:
        candidates.append(Path(model_dir))

    env_dir = os.environ.get("K3_TOK_FILES")
    if env_dir:
        candidates.append(Path(env_dir))

    root_dir = Path(__file__).resolve().parent.parent
    candidates.extend(
        [
            root_dir / "kimi_k3_hf" / "files",
            root_dir.parent / "kimi_k3_hf" / "files",
            Path.home() / "k3" / "hf",
            Path.home() / "k3model",
        ]
    )

    for cand in candidates:
        if cand.is_dir() and (cand / "tiktoken.model").is_file():
            return cand.resolve()

    return None


def extract_pat_str_from_script(script_path: Path) -> str | None:
    """
    Extracts `pat_str` from `tokenization_kimi.py` using AST without importing or running code.
    """
    import ast

    if not script_path.is_file():
        return None

    try:
        src = script_path.read_text(encoding="utf-8")
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            if not any(
                isinstance(t, ast.Name) and t.id == "pat_str" for t in node.targets
            ):
                continue
            val = node.value
            if (
                isinstance(val, ast.Call)
                and isinstance(val.func, ast.Attribute)
                and val.func.attr == "join"
            ):
                sep = ast.literal_eval(val.func.value)
                parts = ast.literal_eval(val.args[0])
                return sep.join(parts)
            return str(ast.literal_eval(val))
    except Exception:
        pass
    return None


class KimiTokenizer:
    """
    Tokenizer for Kimi K3 models supporting Tiktoken BPE and smart UTF-8 Byte Fallback.
    """

    def __init__(
        self,
        model_dir: str | Path | None = None,
        mode: str = "auto",
    ) -> None:
        """
        Initialize KimiTokenizer.

        Args:
            model_dir: Path to directory containing tiktoken.model & tokenizer_config.json.
            mode: Mode of operation - 'auto', 'bpe', or 'fallback' ('byte').
        """
        self.mode = mode.lower()
        self.model_dir: Path | None = None
        self.is_fallback: bool = True
        self.special_tokens: Dict[str, int] = dict(DEFAULT_SPECIAL_TOKENS)
        self.rev_special_tokens: Dict[int, str] = {
            v: k for k, v in self.special_tokens.items()
        }
        self.chat_template: str | None = None
        self._encoding: Any = None
        self._vocab_size: int = 256

        if self.mode in ("fallback", "byte"):
            self._init_fallback_mode()
            return

        # Locate files for auto/bpe mode
        found_dir = find_tokenizer_files(model_dir)

        if found_dir is not None:
            self.model_dir = found_dir
            try:
                self._load_tiktoken_bpe(found_dir)
                self.is_fallback = False
                return
            except Exception as err:
                if self.mode == "bpe":
                    raise RuntimeError(
                        f"Failed to load Tiktoken BPE tokenizer from '{found_dir}': {err}"
                    ) from err

        if self.mode == "bpe":
            raise FileNotFoundError(
                "Could not locate 'tiktoken.model'. Set K3_TOK_FILES environment variable "
                "or pass explicit model_dir."
            )

        # Fallback to UTF-8 Byte tokenizer mode
        self._init_fallback_mode()

    def _init_fallback_mode(self) -> None:
        """Initializes smart fallback UTF-8 byte tokenizer mode."""
        self.is_fallback = True
        self._encoding = None
        self._vocab_size = 256

    def _load_tiktoken_bpe(self, files_dir: Path) -> None:
        """Loads tiktoken BPE model and tokenizer configuration from local directory."""
        import tiktoken

        model_file = files_dir / "tiktoken.model"
        cfg_file = files_dir / "tokenizer_config.json"
        script_file = files_dir / "tokenization_kimi.py"

        # Load BPE ranks from tiktoken.model (base64 token_bytes -> rank)
        ranks: Dict[bytes, int] = {}
        with open(model_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    tok_b64, rank_str = parts[0], parts[1]
                    ranks[base64.b64decode(tok_b64)] = int(rank_str)

        # Load special tokens from tokenizer_config.json
        if cfg_file.is_file():
            with open(cfg_file, "r", encoding="utf-8") as f:
                cfg_data = json.load(f)

            added = cfg_data.get("added_tokens_decoder") or {}
            for id_str, entry in added.items():
                if isinstance(entry, dict) and "content" in entry:
                    self.special_tokens[entry["content"]] = int(id_str)

            self.chat_template = cfg_data.get("chat_template")

        self.rev_special_tokens = {v: k for k, v in self.special_tokens.items()}

        # Extract regex split pattern string
        pat_str = extract_pat_str_from_script(script_file) or DEFAULT_PAT_STR

        # Construct Tiktoken Encoding
        self._encoding = tiktoken.Encoding(
            name="kimi_k3",
            pat_str=pat_str,
            mergeable_ranks=ranks,
            special_tokens=self.special_tokens,
        )
        self._vocab_size = len(ranks) + len(self.special_tokens)

    @property
    def vocab_size(self) -> int:
        """Returns vocabulary size of tokenizer."""
        return self._vocab_size

    @property
    def bos_token_id(self) -> int | None:
        """Returns BOS token ID if present."""
        return self.special_tokens.get("<|im_start|>")

    @property
    def eos_token_id(self) -> int | None:
        """Returns EOS token ID if present."""
        return self.special_tokens.get("<|im_end|>") or self.special_tokens.get(
            "<|endoftext|>"
        )

    def encode(
        self,
        text: str,
        allowed_special: str | Set[str] | Sequence[str] | None = "all",
    ) -> List[int]:
        """
        Encodes input string into a list of integer token IDs.

        Args:
            text: Input string to encode.
            allowed_special: Allowed special tokens setting ('all', 'none', or set/sequence of tokens).

        Returns:
            List of integer token IDs.
        """
        if not text:
            return []

        if self.is_fallback:
            # Smart UTF-8 byte encoding fallback
            return list(text.encode("utf-8"))

        # Tiktoken BPE encoding
        if allowed_special == "all":
            spec_set: Set[str] | str = set(self.special_tokens.keys())
        elif allowed_special in ("none", None):
            spec_set = set()
        elif isinstance(allowed_special, (set, list, tuple)):
            spec_set = set(allowed_special)
        else:
            spec_set = set()

        return self._encoding.encode(text, allowed_special=spec_set)

    def decode(self, ids: Sequence[int] | Any) -> str:
        """
        Decodes a list of token integer IDs (or array/tensor) back to clean text.

        Args:
            ids: Sequence of token IDs, PyTorch tensor, or NumPy array.

        Returns:
            Human-readable decoded text string.
        """
        if ids is None:
            return ""

        # Convert tensor / numpy / iterables to python list of ints
        if hasattr(ids, "tolist"):
            int_ids = [int(x) for x in ids.tolist()]
        elif isinstance(ids, (list, tuple)):
            int_ids = [int(x) for x in ids]
        else:
            try:
                int_ids = [int(x) for x in ids]
            except Exception:
                return ""

        if not int_ids:
            return ""

        if self.is_fallback:
            # Convert integer IDs (0..255) to bytes and decode as UTF-8 cleanly
            raw_bytes = bytes([i % 256 for i in int_ids])
            return raw_bytes.decode("utf-8", errors="replace")

        # Tiktoken BPE decoding with special token awareness
        out_chunks: list[str] = []
        bpe_buf: list[int] = []

        for token_id in int_ids:
            if token_id in self.rev_special_tokens:
                if bpe_buf:
                    out_chunks.append(self._encoding.decode(bpe_buf))
                    bpe_buf = []
                out_chunks.append(self.rev_special_tokens[token_id])
            else:
                bpe_buf.append(token_id)

        if bpe_buf:
            out_chunks.append(self._encoding.decode(bpe_buf))

        return "".join(out_chunks)

    def format_chat_prompt(
        self,
        messages: List[Dict[str, str]],
        add_generation_prompt: bool = True,
    ) -> str:
        """
        Formats chat history messages into standard Kimi K3 chat template string.

        Args:
            messages: List of message dictionaries containing 'role' and 'content'.
                      Roles typically include 'system', 'user', 'assistant'.
            add_generation_prompt: Whether to append assistant turn generation prefix.

        Returns:
            Formatted chat prompt string.
        """
        # Try jinja2 rendering if chat_template is present
        if self.chat_template:
            try:
                from jinja2 import Template

                tmpl = Template(self.chat_template)
                rendered = tmpl.render(
                    messages=messages,
                    add_generation_prompt=add_generation_prompt,
                    bos_token="<|im_start|>",
                    eos_token="<|im_end|>",
                )
                return rendered
            except Exception:
                pass

        # Standard Kimi K3 Chat Template Fallback
        parts: list[str] = []
        for msg in messages:
            role = msg.get("role", "user").strip()
            content = msg.get("content", "").strip()
            parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")

        if add_generation_prompt:
            parts.append("<|im_start|>assistant\n")

        return "".join(parts)

    def encode_chat(
        self,
        messages: List[Dict[str, str]],
        add_generation_prompt: bool = True,
    ) -> List[int]:
        """
        Convenience helper to format chat messages and encode directly to token IDs.
        """
        prompt = self.format_chat_prompt(
            messages, add_generation_prompt=add_generation_prompt
        )
        return self.encode(prompt)
