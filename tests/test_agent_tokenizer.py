"""
Unit tests for KimiTokenizer and KimiAgent in kimipy package.

Covers:
- KimiTokenizer text decoding (Python lists, NumPy arrays, PyTorch tensors, special tokens)
- KimiAgent code block extraction
- KimiAgent isolated sandbox execution
- KimiAgent self-debugging feedback loop
"""

import unittest
import numpy as np
import torch

from kimipy.tokenizer import KimiTokenizer
from kimipy.agent import KimiAgent


class TestKimiTokenizerDecoding(unittest.TestCase):
    """Test suite for KimiTokenizer decoding functionality."""

    def setUp(self) -> None:
        self.tokenizer = KimiTokenizer(mode="fallback")

    def test_decode_int_list(self) -> None:
        text = "Hello, Kimi K3!"
        token_ids = list(text.encode("utf-8"))
        decoded = self.tokenizer.decode(token_ids)
        self.assertEqual(decoded, text)

    def test_decode_numpy_array(self) -> None:
        text = "NumPy Array Decoding Test"
        token_ids = np.array(list(text.encode("utf-8")), dtype=np.int64)
        decoded = self.tokenizer.decode(token_ids)
        self.assertEqual(decoded, text)

    def test_decode_torch_tensor(self) -> None:
        text = "PyTorch Tensor Decoding Test"
        token_ids = torch.tensor(list(text.encode("utf-8")), dtype=torch.long)
        decoded = self.tokenizer.decode(token_ids)
        self.assertEqual(decoded, text)

    def test_decode_empty_input(self) -> None:
        self.assertEqual(self.tokenizer.decode([]), "")
        self.assertEqual(self.tokenizer.decode(None), "")

    def test_format_chat_prompt(self) -> None:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is Python?"},
        ]
        prompt = self.tokenizer.format_chat_prompt(messages)
        self.assertIn("<|im_start|>system", prompt)
        self.assertIn("<|im_start|>user", prompt)
        self.assertIn("<|im_start|>assistant", prompt)

        encoded = self.tokenizer.encode_chat(messages)
        self.assertIsInstance(encoded, list)
        self.assertGreater(len(encoded), 0)


class TestKimiAgentCodeExtraction(unittest.TestCase):
    """Test suite for KimiAgent code block extraction."""

    def setUp(self) -> None:
        self.agent = KimiAgent()

    def test_extract_python_fence(self) -> None:
        response = "Here is the code:\n```python\nprint('Hello World')\n```\nHope that helps!"
        code = self.agent.extract_code(response)
        self.assertEqual(code, "print('Hello World')")

    def test_extract_generic_fence(self) -> None:
        response = "```\nx = 42\nprint(x)\n```"
        code = self.agent.extract_code(response)
        self.assertEqual(code, "x = 42\nprint(x)")

    def test_extract_multiple_blocks(self) -> None:
        response = "```python\na = 1\n```\nAnd another:\n```python\nb = 2\n```"
        blocks = self.agent.extract_code_blocks(response)
        self.assertEqual(len(blocks), 2)
        self.assertEqual(blocks[0], "a = 1")
        self.assertEqual(blocks[1], "b = 2")

    def test_extract_fallback_no_fence(self) -> None:
        raw_code = "import math\nprint(math.sqrt(16))"
        code = self.agent.extract_code(raw_code)
        self.assertEqual(code, raw_code)


class TestKimiAgentSandboxExecution(unittest.TestCase):
    """Test suite for KimiAgent sandbox code execution."""

    def setUp(self) -> None:
        self.agent = KimiAgent()

    def test_sandbox_valid_code(self) -> None:
        code = "res = 2 + 2\nprint(f'Result: {res}')"
        res = self.agent.execute_sandbox(code)
        self.assertTrue(res["success"])
        self.assertEqual(res["returncode"], 0)
        self.assertIn("Result: 4", res["stdout"])
        self.assertEqual(res["stderr"], "")

    def test_sandbox_runtime_error(self) -> None:
        code = "raise ValueError('Custom error message')"
        res = self.agent.execute_sandbox(code)
        self.assertFalse(res["success"])
        self.assertNotEqual(res["returncode"], 0)
        self.assertIn("ValueError: Custom error message", res["stderr"])

    def test_sandbox_syntax_error(self) -> None:
        code = "def invalid_syntax("
        res = self.agent.execute_sandbox(code)
        self.assertFalse(res["success"])
        self.assertIn("SyntaxError", res["stderr"])

    def test_sandbox_empty_code(self) -> None:
        res = self.agent.execute_sandbox("")
        self.assertFalse(res["success"])
        self.assertIn("Empty code snippet", res["stderr"])


class MockFaultyThenSuccessModel:
    """Mock model that returns faulty code on attempt 1 and working code on attempt 2."""

    def __init__(self, tokenizer: KimiTokenizer) -> None:
        self.tokenizer = tokenizer
        self.calls = 0

    def generate(self, input_tensor: torch.Tensor, n_new: int = 128, incremental: bool = True) -> torch.Tensor:
        self.calls += 1
        prompt_ids = input_tensor[0].tolist()

        if self.calls == 1:
            # Faulty code attempt
            code_text = "```python\nraise RuntimeError('Initial attempt failed')\n```"
        else:
            # Fixed code attempt
            code_text = "```python\nprint('Self-debug success!')\n```"

        gen_ids = list(code_text.encode("utf-8"))
        full_ids = prompt_ids + gen_ids
        return torch.tensor([full_ids], dtype=torch.long)


class TestKimiAgentSelfDebuggingLoop(unittest.TestCase):
    """Test suite for KimiAgent self-debugging feedback loop."""

    def test_self_debugging_loop_success_first_try(self) -> None:
        agent = KimiAgent()
        # Mock run_agentic_loop default fallback
        res = agent.run_agentic_loop("print('Success first try')", max_retries=3)
        self.assertTrue(res["success"])
        self.assertEqual(res["iterations"], 1)
        self.assertIn("Success first try", res["stdout"])

    def test_self_debugging_loop_retry_and_succeed(self) -> None:
        tokenizer = KimiTokenizer(mode="fallback")
        mock_model = MockFaultyThenSuccessModel(tokenizer)
        agent = KimiAgent(model=mock_model, tokenizer=tokenizer)

        res = agent.run_agentic_loop("Write code that succeeds after retry", max_retries=3)
        self.assertTrue(res["success"])
        self.assertEqual(res["iterations"], 2)
        self.assertIn("Self-debug success!", res["stdout"])
        self.assertEqual(len(res["history"]), 2)
        # Verify history contains stderr from 1st attempt
        self.assertIn("RuntimeError: Initial attempt failed", res["history"][0]["exec_result"]["stderr"])

    def test_self_debugging_loop_max_retries_exceeded(self) -> None:
        class AlwaysFailModel:
            def generate(self, input_tensor: torch.Tensor, n_new: int = 128, incremental: bool = True) -> torch.Tensor:
                prompt_ids = input_tensor[0].tolist()
                code_text = "```python\nraise Exception('Permanent failure')\n```"
                return torch.tensor([prompt_ids + list(code_text.encode("utf-8"))], dtype=torch.long)

        tokenizer = KimiTokenizer(mode="fallback")
        agent = KimiAgent(model=AlwaysFailModel(), tokenizer=tokenizer)

        res = agent.run_agentic_loop("Failing task", max_retries=3)
        self.assertFalse(res["success"])
        self.assertEqual(res["iterations"], 3)
        self.assertEqual(len(res["history"]), 3)
        self.assertIn("Permanent failure", res["stderr"])


if __name__ == "__main__":
    unittest.main()
