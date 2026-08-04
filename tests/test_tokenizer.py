"""
Unit tests for chakra.tokenizer module (KimiTokenizer).

Tests offline Tiktoken BPE loading, UTF-8 byte fallback mode, text/code roundtrips,
and chat template formatting.
"""

import base64
import json
from pathlib import Path
import tempfile
import unittest

import torch

from chakra.tokenizer import KimiTokenizer


class TestKimiTokenizerFallback(unittest.TestCase):
    """Test suite for KimiTokenizer in UTF-8 byte fallback mode."""

    def setUp(self) -> None:
        self.tok = KimiTokenizer(mode="fallback")

    def test_fallback_flag(self) -> None:
        self.assertTrue(self.tok.is_fallback)
        self.assertEqual(self.tok.vocab_size, 256)

    def test_text_roundtrip(self) -> None:
        sample_texts = [
            "Hello, World!",
            "The quick brown fox jumps over the lazy dog.",
            "你好世界！测试 UTF-8 字节 fallback 模式。",
            "Café, naïve, résumé, über.",
            "Emoji test: 🚀 😊 🎉 🔥",
        ]
        for text in sample_texts:
            encoded = self.tok.encode(text)
            self.assertIsInstance(encoded, list)
            self.assertTrue(all(isinstance(x, int) and 0 <= x < 256 for x in encoded))
            decoded = self.tok.decode(encoded)
            self.assertEqual(decoded, text, f"Failed roundtrip for text: {text}")

    def test_code_roundtrip(self) -> None:
        sample_codes = [
            "def add(a: int, b: int) -> int:\n    # Return sum of two numbers\n    return a + b\n",
            "int main(void) {\n    printf(\"Hello Kimi K3\\n\");\n    return 0;\n}\n",
            '{\n  "model": "Kimi-K3",\n  "vocab_size": 163840,\n  "quant": "MXFP4"\n}',
            "SELECT id, name, val FROM table WHERE status = 'ACTIVE' AND price > 99.99;",
            "<html>\n  <head><title>Test</title></head>\n  <body>\n    <h1>Kimi Engine</h1>\n  </body>\n</html>",
        ]
        for code in sample_codes:
            encoded = self.tok.encode(code)
            decoded = self.tok.decode(encoded)
            self.assertEqual(decoded, code, f"Failed roundtrip for code string:\n{code}")

    def test_tensor_numpy_decode(self) -> None:
        text = "PyTorch tensor decoding test"
        encoded = self.tok.encode(text)
        tensor_ids = torch.tensor(encoded, dtype=torch.long)
        decoded = self.tok.decode(tensor_ids)
        self.assertEqual(decoded, text)

    def test_chat_formatting(self) -> None:
        messages = [
            {"role": "system", "content": "You are a helpful AI coding assistant."},
            {"role": "user", "content": "Write a Python hello world script."},
            {"role": "assistant", "content": "print('Hello, World!')"},
        ]
        formatted = self.tok.format_chat_prompt(messages, add_generation_prompt=True)
        self.assertIn("<|im_start|>system\nYou are a helpful AI coding assistant.<|im_end|>", formatted)
        self.assertIn("<|im_start|>user\nWrite a Python hello world script.<|im_end|>", formatted)
        self.assertIn("<|im_start|>assistant\nprint('Hello, World!')<|im_end|>", formatted)
        self.assertTrue(formatted.endswith("<|im_start|>assistant\n"))

        encoded_chat = self.tok.encode_chat(messages, add_generation_prompt=True)
        decoded_chat = self.tok.decode(encoded_chat)
        self.assertEqual(decoded_chat, formatted)


class TestKimiTokenizerBPE(unittest.TestCase):
    """Test suite for KimiTokenizer BPE loading and execution."""

    def test_synthetic_bpe_loading(self) -> None:
        """Create a synthetic tiktoken.model directory and test BPE encoding/decoding."""
        try:
            import tiktoken  # noqa: F401
        except ImportError:
            self.skipTest("tiktoken package is not installed.")

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            model_path = tmp_path / "tiktoken.model"
            cfg_path = tmp_path / "tokenizer_config.json"

            # Create synthetic BPE ranks (include single byte ranks for tiktoken completeness)
            ranks_data = [(bytes([b]), b) for b in range(256)]
            ranks_data.extend(
                [
                    (b"Hello", 256),
                    (b" World", 257),
                    (b"!", 258),
                    (b"def", 259),
                    (b" foo():", 260),
                ]
            )
            with open(model_path, "w", encoding="utf-8") as f:
                for tok_bytes, rank in ranks_data:
                    b64_tok = base64.b64encode(tok_bytes).decode("ascii")
                    f.write(f"{b64_tok} {rank}\n")

            # Create synthetic tokenizer_config.json
            cfg_data = {
                "added_tokens_decoder": {
                    "100": {"content": "<|im_start|>", "special": True},
                    "101": {"content": "<|im_end|>", "special": True},
                }
            }
            with open(cfg_path, "w", encoding="utf-8") as f:
                json.dump(cfg_data, f)

            tok = KimiTokenizer(model_dir=tmp_path, mode="bpe")
            self.assertFalse(tok.is_fallback)

            # Test text encoding/decoding
            encoded = tok.encode("Hello World!")
            self.assertEqual(encoded, [256, 257, 258])
            decoded = tok.decode(encoded)
            self.assertEqual(decoded, "Hello World!")

            # Test code encoding/decoding
            code_str = "def foo():"
            encoded_code = tok.encode(code_str)
            decoded_code = tok.decode(encoded_code)
            self.assertEqual(decoded_code, code_str)

            # Test special tokens
            chat_str = "<|im_start|>user\nHello<|im_end|>"
            encoded_chat = tok.encode(chat_str, allowed_special="all")
            self.assertIn(100, encoded_chat)
            self.assertIn(101, encoded_chat)
            decoded_chat = tok.decode(encoded_chat)
            self.assertIn("<|im_start|>", decoded_chat)
            self.assertIn("<|im_end|>", decoded_chat)

    def test_auto_mode_without_files(self) -> None:
        """In auto mode with missing files, tokenizer must gracefully fall back to byte mode."""
        with tempfile.TemporaryDirectory() as empty_dir:
            tok = KimiTokenizer(model_dir=empty_dir, mode="auto")
            self.assertTrue(tok.is_fallback)
            text = "Fallback text test"
            encoded = tok.encode(text)
            decoded = tok.decode(encoded)
            self.assertEqual(decoded, text)


if __name__ == "__main__":
    unittest.main()
